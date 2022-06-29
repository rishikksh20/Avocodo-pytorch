import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, spectral_norm
from utils import init_weights, get_padding


class CoMBD(torch.nn.Module):

    def __init__(self, filters, kernels, groups, strides, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList()
        init_channel = 1
        for i, (f, k, g, s) in enumerate(zip(filters, kernels, groups, strides)):
            self.convs.append(norm_f(Conv1d(init_channel, f, k, s, padding=get_padding(k, 1), groups=g)))
            init_channel = f
        self.conv_post = norm_f(Conv1d(filters[-1], 1, 3, 1, padding=get_padding(3, 1)))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            print(x.shape)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        print(x.shape)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap
