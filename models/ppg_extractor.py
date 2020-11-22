import tensorflow as tf

from tensorflow import keras


class BLSTMlayer(object):
    def __init__(self, hidden, name='blstm'):
        self.hidden = hidden
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.blstm_layer = keras.layers.Bidirectional(
                keras.layers.CuDNNLSTM(units=self.hidden,
                                       return_sequences=True),
                merge_mode='concat'
            )

    def __call__(self, inputs, seq_lens=None):
        with tf.variable_scope(self.name):
            # mask = tf.sequence_mask(seq_lens, dtype=tf.float32) \
            #    if seq_lens is not None else None
            return self.blstm_layer(inputs)


class CNNBLSTMClassifier(object):
    def __init__(self, out_dims, n_cnn, cnn_hidden,
                 cnn_kernel, n_blstm, lstm_hidden,
                 name='cnn_blstm_classifier'):
        self.name = name
        with tf.variable_scope(self.name):
            self.cnn_layers = []
            for i in range(n_cnn):
                conv_layer = keras.layers.Conv1D(filters=cnn_hidden,
                                                 kernel_size=cnn_kernel,
                                                 strides=1, padding='same',
                                                 activation=tf.nn.relu,
                                                 name='conv_layer{}'.format(i))
                self.cnn_layers.append(conv_layer)
            self.blstm_layers = []
            for i in range(n_blstm):
                blstm_layer = BLSTMlayer(lstm_hidden,
                                         name='blstm_layer{}'.format(i))
                self.blstm_layers.append(blstm_layer)
            self.output_projection = keras.layers.Dense(units=out_dims)

    def __call__(self, inputs, labels=None, lengths=None):
        # 1. CNN block
        cnn_outs = inputs
        for layer in self.cnn_layers:
            cnn_outs = layer(cnn_outs)
        # 2. BLSTM layers
        blstm_outs = cnn_outs
        for layer in self.blstm_layers:
            blstm_outs = layer(blstm_outs, seq_lens=lengths)
        # 3. output projection
        logits = self.output_projection(blstm_outs)

        # 4. compute loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(labels, axis=-1),
            logits=logits) if labels is not None else None
        mask = tf.sequence_mask(lengths, dtype=tf.float32) if lengths is not None else 1.0
        cross_entropy = tf.reduce_mean(cross_entropy * mask) if cross_entropy is not None else None
        return {'logits': logits,  # [time, dims]
                'cross_entropy': cross_entropy}