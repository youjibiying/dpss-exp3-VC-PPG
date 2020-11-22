import os
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from utils import load_wav, _preemphasize, melspectrogram, spectrogram, logf0, wav2unnormalized_mfcc, softmax, lf0_normailze
from models import CNNBLSTMClassifier
from config import Hparams

hps = Hparams


def length_validate(features):
    min_len = 1000000
    for feat in features:
        if feat.shape[0] < min_len:
            min_len = feat.shape[0]
    new_feats = (feat[:min_len, :] for feat in features)
    return new_feats


def main():
    parser = argparse.ArgumentParser('PreprocessingParser')
    parser.add_argument('--data_dir', type=str, help='data root directory')
    parser.add_argument('--save_dir', type=str, help='extracted feature save directory')
    parser.add_argument('--dev_rate', type=float, help='dev set rate', default=0.05)
    parser.add_argument('--test_rate', type=float, help='test set rate', default=0.05)
    args = parser.parse_args()
    # args validation
    if args.dev_rate < 0 or args.dev_rate >= 1:
        raise ValueError('dev rate should be in [0, 1)')
    if args.test_rate < 0 or args.test_rate >= 1:
        raise ValueError('dev rate should be in [0, 1)')
    if args.test_rate + args.dev_rate >= 1:
        raise ValueError('dev rate + test rate should not be >= 1.')
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError('Directory {} not found!'.format(args.data_dir))
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    mel_dir = os.path.join(args.save_dir, 'mels')
    os.makedirs(mel_dir, exist_ok=True)
    linear_dir = os.path.join(args.save_dir, 'linears')
    os.makedirs(linear_dir, exist_ok=True)
    f0_dir = os.path.join(args.save_dir, 'f0s')
    os.makedirs(f0_dir, exist_ok=True)
    ppg_dir = os.path.join(args.save_dir, 'ppgs')
    os.makedirs(ppg_dir, exist_ok=True)
    for mode in ['train', 'dev', 'test']:
        if os.path.isfile(os.path.join(args.save_dir, "{}_meta.csv".format(mode))):
            os.remove(os.path.join(args.save_dir, "{}_meta.csv".format(mode)))
    wav_files = []
    for rootdir, subdir, files in os.walk(args.data_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(rootdir, f))
    random.shuffle(wav_files)

    print('Set up PPGs extraction network')
    # Set up network
    ppg_extractor_hps = hps.PPGExtractor.CNNBLSTMClassifier
    mfcc_pl = tf.placeholder(dtype=tf.float32,
                             shape=[None, None, 3 * hps.Audio.n_mfcc],
                             name='mfcc_pl')
    ppg_extractor = CNNBLSTMClassifier(out_dims=hps.Audio.ppg_dim,
                                       n_cnn=ppg_extractor_hps.n_cnn,
                                       cnn_hidden=ppg_extractor_hps.cnn_hidden,
                                       cnn_kernel=ppg_extractor_hps.cnn_kernel,
                                       n_blstm=ppg_extractor_hps.n_blstm,
                                       lstm_hidden=ppg_extractor_hps.lstm_hidden)
    predicted_ppgs = ppg_extractor(inputs=mfcc_pl)['logits']

    # set up a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # load saved model
    saver = tf.train.Saver()
    print('Restoring ppgs extractor from {}'.format(ppg_extractor_hps.ckpt))
    saver.restore(sess, ppg_extractor_hps.ckpt)
    print('Extracting mel-spectrograms, spectrograms and log-f0s...')
    train_set = []
    dev_set = []
    test_set = []
    dev_start_idx = int(len(wav_files) * (1 - args.dev_rate - args.test_rate))
    test_stat_idx = int(len(wav_files) * (1 - args.test_rate))
    for i, wav_f in tqdm(enumerate(wav_files)):
        try:
            wav_arr = load_wav(wav_f)
        except:
            continue
        pre_emphasized_wav = _preemphasize(wav_arr)
        fid = '{}_{}'.format(wav_f.split('/')[-3].split('_')[2], wav_f.split('/')[-1].split('.')[0].split('_')[1])
        # extract mel-spectrograms
        mel_fn = os.path.join(mel_dir, '{}.npy'.format(fid))
        try:
            mel_spec = melspectrogram(pre_emphasized_wav).astype(np.float32).T
        except:
            continue
        # extract spectrograms
        linear_fn = os.path.join(linear_dir, '{}.npy'.format(fid))
        try:
            linear_spec = spectrogram(pre_emphasized_wav).astype(np.float32).T
        except:
            continue
        # extract log-f0s
        f0_fn = os.path.join(f0_dir, '{}.npy'.format(fid))
        log_f0 = logf0(wav_f)
        try:
            log_f0 = lf0_normailze(log_f0)
        except:
            continue
        # extract ppgs
        mfcc_feats = wav2unnormalized_mfcc(wav_arr)
        ppg = sess.run(predicted_ppgs,
                       feed_dict={mfcc_pl: np.expand_dims(mfcc_feats, axis=0)})
        ppg = softmax(np.squeeze(ppg, axis=0))
        ppg_fn = os.path.join(ppg_dir, '{}.npy'.format(fid))

        # save features to respective directory
        mel_spec, linear_spec, log_f0, ppg = length_validate((mel_spec, linear_spec, log_f0, ppg))
        np.save(mel_fn, mel_spec)
        np.save(linear_fn, linear_spec)
        np.save(f0_fn, log_f0)
        np.save(ppg_fn, ppg)

        # write to csv
        if i < dev_start_idx:
            train_set.append(fid)
            with open(os.path.join(args.save_dir, 'train_meta.csv'),
                      'a', encoding='utf-8') as train_f:
                train_f.write('{}|ppgs/{}.npy|mels/{}.npy|linears/{}.npy|f0s/{}.npy\n'.format(fid, fid, fid, fid, fid))
        elif i < test_stat_idx:
            dev_set.append(fid)
            with open(os.path.join(args.save_dir, 'dev_meta.csv'),
                      'a', encoding='utf-8') as dev_f:
                dev_f.write('{}|ppgs/{}.npy|mels/{}.npy|linears/{}.npy|f0s/{}.npy\n'.format(fid, fid, fid, fid, fid))
        else:
            test_set.append(fid)
            with open(os.path.join(args.save_dir, 'test_meta.csv'),
                      'a', encoding='utf-8') as test_f:
                test_f.write('{}|ppgs/{}.npy|mels/{}.npy|linears/{}.npy|f0s/{}.npy\n'.format(fid, fid, fid, fid, fid))
    print('Done extracting features!')
    return


if __name__ == '__main__':
    main()
