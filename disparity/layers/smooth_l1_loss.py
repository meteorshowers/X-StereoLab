# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def l1_loss(input, target, beta=1., sum_last_dim=False):
    n = torch.abs(input - target) 
    loss = n * beta
    if sum_last_dim:
        loss = loss.sum(dim=-1)
    return loss.mean()

def l2_loss(input, target, beta=1., sum_last_dim=False):
    diff = input - target
    n = diff * diff
    loss = n * beta
    if sum_last_dim:
        loss = loss.sum(dim=-1)
    return loss.mean()


def ordinal_loss(input, target):
    N, C = input.shape

    ranges = torch.arange(C, dtype=torch.int32).cuda() 
    mask = ranges[None, :] < target[:, None]

    loss = -(torch.sum(torch.log( input[mask] + 1e-6 )) \
        + torch.sum(torch.log( 1. - input[1 - mask] + 1e-6 )))

    loss = loss / N / C
    return loss

def dorn_decode(cls, reg, alpha, beta):
    dorn_dim = cls.shape[1]

    depth_discretization = torch.sum((cls > 0.5), dim=1, keepdim=True)
    if reg is not None:
        depth_residual = torch.gather(reg, dim=1, index=depth_discretization)
        depth_continuity = depth_discretization.float() + 0.5 + depth_residual
    else:
        depth_continuity = depth_discretization.float()
    depth = alpha * (beta / alpha) ** (depth_continuity / dorn_dim)

    return depth

def dorn_encode(depth, alpha, beta, dorn_dim):
    depth = dorn_dim * torch.log(depth / alpha + 1e-6) / np.log(beta / alpha + 1e-6)
    depth = depth.clamp(0, dorn_dim)
    return depth.int(), depth - depth.int().float() - 0.5

def bce_loss(score, target):
    loss = - (target * torch.log(score + 1e-6) + (1 - target) * torch.log( 1 - score + 1e-6))

    return loss.mean()

