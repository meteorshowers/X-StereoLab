import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from dsgn import _C


class _BuildCostVolume(Function):
    @staticmethod
    def forward(ctx, left, right, shift):
        ctx.save_for_backward(shift)
        assert torch.all(shift >= 0.)
        output = _C.build_cost_volume_forward(
            left, right, shift
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        shift, = ctx.saved_tensors
        grad_left, grad_right = _C.build_cost_volume_backward(
            grad_output,
            shift
        )
        return grad_left, grad_right, None


build_cost_volume = _BuildCostVolume.apply


class BuildCostVolume(nn.Module):
    def __init__(self):
        super(BuildCostVolume, self).__init__()

    def forward(self, left, right, shift):
        return build_cost_volume(
            left, right, shift
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ 
        return tmpstr
