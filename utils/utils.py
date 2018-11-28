import os
import torch

def GERF_loss(GT, pred, args):
    mask = (GT < args.maxdisp) & (GT >= 0)
    # print(mask.size(), GT.size(), pred.size())
    count = len(torch.nonzero(mask))
    # print(count)
    if count == 0:
        count = 1
    return torch.sum(torch.sqrt(torch.pow(GT[mask] - pred[mask], 2) + 4) /2 - 1) / count