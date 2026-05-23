import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


def ranking_contrastive_loss(features, targets, preds, w=1, weights=1, t=0.02, e=0.001):
    """
    Ranking-aware contrastive loss.

    Args:
        features: input feature tensor
        targets:  ground-truth labels
        preds:    predicted labels
        w:        ordinal distance threshold for negative mining
        weights:  sample weights (scalar or tensor)
        t:        temperature
        e:        pushing power scale
    """
    batch_size = targets.shape[0]
    q = torch.nn.functional.normalize(features, dim=1)
    k = torch.nn.functional.normalize(features, dim=1)

    l_k = targets.unsqueeze(0).expand(batch_size, batch_size)
    l_q = targets.unsqueeze(1).expand(batch_size, batch_size)
    p_k = preds.unsqueeze(0).expand(batch_size, batch_size)
    p_q = preds.unsqueeze(1).expand(batch_size, batch_size)

    l_dist = torch.abs(l_q - l_k)
    p_dist = torch.abs(p_q - p_k)

    pos_i = l_dist.eq(0)
    if w == 0:
        neg_i = ~pos_i
    else:
        neg_i = (~pos_i) * p_dist.le(w * 2)

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0

    prod = torch.einsum("nc,kc->nk", [q, k]) / t
    pos = prod * pos_i
    neg = prod * neg_i

    pushing_w = weights * torch.exp(l_dist * e)
    neg_exp_dot = (pushing_w * torch.exp(neg) * neg_i).sum(1).clamp(min=1e-6)
    no_neg_flag = neg_i.sum(1).bool()
    denom = pos_i.sum(1).clamp(min=1)

    if w == 0:
        loss = ((-torch.log(
            torch.div(torch.exp(pos),
                      (torch.exp(pos).sum(1)).unsqueeze(-1) + 1e-6)
        ) * pos_i).sum(1) / denom)
        loss = loss.unsqueeze(-1).mean()
    else:
        loss = ((-torch.log(
            torch.div(torch.exp(pos),
                      (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1) + 1e-6)
        ) * pos_i).sum(1) / denom)
        loss = (weights * (loss * no_neg_flag).unsqueeze(-1)).mean()

    return loss


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
        kernel_window = kernel_window / max(kernel_window)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1)))
        kernel_window = kernel_window / max(kernel_window)
    return kernel_window
