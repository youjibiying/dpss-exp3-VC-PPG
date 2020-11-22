import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def masked_mse_loss(inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor):
    """
    :param inputs: [batch, time, dim]
    :param targets: [batch, time, dim]
    :param lengths: [batch]
    :return:
    """
    if lengths is None:
        return nn.MSELoss()(inputs, targets)
    else:
        max_len = max(lengths.cpu().numpy())
        mask = torch.arange(max_len).expand([len(lengths), max_len]).to(device) < lengths.unsqueeze(1)
        mask = mask.to(dtype=torch.float32)
        mse_loss = torch.mean(
            torch.sum(torch.mean((inputs - targets) ** 2, dim=2) * mask,
                      dim=1) / lengths.to(dtype=torch.float32))
        return mse_loss


def softmax(x, axis=-1):
    assert len(x.shape) == 2
    _max = np.max(x)
    probs = np.exp(x - _max) / np.sum(np.exp(x - _max), axis=axis, keepdims=True)
    return probs


def lf0_normailze(lf0, mean_f=None, std_f=None):
    mean = np.load(mean_f) if mean_f is not None else np.mean(lf0[lf0 > 0])
    std = np.load(std_f) if std_f is not None else np.std(lf0[lf0 > 0])
    normalized = np.copy(lf0)
    vuv = np.zeros([len(lf0)], dtype=np.float32)
    voiced_inds = np.where(lf0 > 0.0)[0]
    vuv[voiced_inds] = 1.0
    normalized[normalized > 0] = (lf0[lf0 > 0] - mean) / std
    normalized[0] = 1e-5 if normalized[0] <= 0 else normalized[0]
    normalized[-1] = 1e-5 if normalized[-1] <= 0 else normalized[-1]
    non_zero_ids = np.where(normalized > 0)[0]
    non_zero_vals = normalized[non_zero_ids]
    f = interp1d(non_zero_ids.astype(np.float32), non_zero_vals)
    x_all = np.arange(len(normalized), dtype=np.float32)
    interpolated = f(x_all)
    lf0_feats = np.concatenate([interpolated.reshape([-1, 1]),
                                vuv.reshape([-1, 1])], axis=1)
    return lf0_feats
