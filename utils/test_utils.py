import os
import threading
import matplotlib

import matplotlib.pyplot as plt
from .audio import inv_preemphasize, inv_mel_spectrogram, save_wav


def synthesize_and_save_wavs(save_dir, step, mel_batch, mel_lengths, ids, prefix=''):
    def _synthesize(mel, fid):
        wav_arr = inv_mel_spectrogram(mel.T)
        wav_arr = inv_preemphasize(wav_arr)
        save_wav(wav_arr, os.path.join(save_dir, '{}-{}-{}.wav'.format(prefix, fid, step)))
        return
    threads = []
    for i in range(mel_batch.shape[0]):
        mel = mel_batch[i][:mel_lengths[i], :]
        # np.save(os.path.join(save_dir, '{}-{}.npy'.format(ids[i], step)), mel)
        idx = ids[i].decode('utf-8') if type(ids[i]) is bytes else ids[i]
        t = threading.Thread(target=_synthesize, args=(mel, idx))
        threads.append(t)
        t.start()
    for x in threads:
        x.join()
    print('All wavs are synthesized!')
    return


def draw_melspectrograms(save_dir, step, mel_batch, mel_lengths, ids, prefix=''):
    matplotlib.use('agg')
    for i, mel in enumerate(mel_batch):
        plt.imshow(mel[:mel_lengths[i], :].T, aspect='auto', origin='lower')
        plt.tight_layout()
        idx = ids[i].decode('utf-8') if type(ids[i]) is bytes else ids[i]
        plt.savefig(save_dir + '/{}-{}-{}.pdf'.format(prefix, idx, step))
        plt.close()
    return