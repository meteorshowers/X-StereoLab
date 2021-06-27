import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class ScaleShift(nn.Module):
    def __init__(self, scale_value, shift_value, exp=False):
        super(ScaleShift, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([scale_value]))
        self.shift = nn.Parameter(torch.FloatTensor([shift_value]))
        self.exp = exp

    def forward(self, input):
        if not self.exp:
            return input * self.scale + self.shift
        else:
            return torch.exp(input / 10.) * self.scale + self.shift
