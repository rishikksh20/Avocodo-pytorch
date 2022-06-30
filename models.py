import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
from modules import PQMF, CoMBD, SubBandDiscriminator

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        print(self.conv_post)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.out_proj_x1 = weight_norm(Conv1d(h.upsample_initial_channel // 4, 1, 7, 1, padding=3))
        self.out_proj_x2 = weight_norm(Conv1d(h.upsample_initial_channel // 8, 1, 7, 1, padding=3))

    def forward(self, x):

        x1 = None
        x2 = None
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            if i == 1:
                x1 = self.out_proj_x1(x)
            elif i == 2:
                x2 = self.out_proj_x2(x)

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x, x2, x1

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiCoMBDiscriminator(torch.nn.Module):

    def __init__(self, kernels, channels, groups, strides):
        super(MultiCoMBDiscriminator, self).__init__()
        self.combd_1 = CoMBD(filters=channels, kernels=kernels[0], groups=groups, strides=strides)
        self.combd_2 = CoMBD(filters=channels, kernels=kernels[1], groups=groups, strides=strides)
        self.combd_3 = CoMBD(filters=channels, kernels=kernels[2], groups=groups, strides=strides)

        self.pqmf_2 = PQMF(N=2, taps=256, cutoff=0.25, beta=10.0)
        self.pqmf_4 = PQMF(N=4, taps=192, cutoff=0.13, beta=10.0)

    def forward(self, x, x_hat, x2_hat, x1_hat):
        y = []
        y_hat = []
        fmap = []
        fmap_hat = []

        p3, p3_fmap = self.combd_3(x)
        y.append(p3)
        fmap.append(p3_fmap)

        p3_hat, p3_fmap_hat = self.combd_3(x_hat)
        y_hat.append(p3_hat)
        fmap_hat.append(p3_fmap_hat)

        x2_ = self.pqmf_2(x)[:, :1, :]  # Select first band
        x1_ = self.pqmf_4(x)[:, :1, :]  # Select first band

        x2_hat_ = self.pqmf_2(x_hat)[:, :1, :]
        x1_hat_ = self.pqmf_4(x_hat)[:, :1, :]

        p2_, p2_fmap_ = self.combd_2(x2_)
        y.append(p2_)
        fmap.append(p2_fmap_)

        p2_hat_, p2_fmap_hat_ = self.combd_2(x2_hat)
        y_hat.append(p2_hat_)
        fmap_hat.append(p2_fmap_hat_)

        p1_, p1_fmap_ = self.combd_1(x1_)
        y.append(p1_)
        fmap.append(p1_fmap_)

        p1_hat_, p1_fmap_hat_ = self.combd_1(x1_hat)
        y_hat.append(p1_hat_)
        fmap_hat.append(p1_fmap_hat_)





        p2, p2_fmap = self.combd_2(x2_)
        y.append(p2)
        fmap.append(p2_fmap)

        p2_hat, p2_fmap_hat = self.combd_2(x2_hat_)
        y_hat.append(p2_hat)
        fmap_hat.append(p2_fmap_hat)

        p1, p1_fmap = self.combd_1(x1_)
        y.append(p1)
        fmap.append(p1_fmap)

        p1_hat, p1_fmap_hat = self.combd_1(x1_hat_)
        y_hat.append(p1_hat)
        fmap_hat.append(p1_fmap_hat)

        return y, y_hat, fmap, fmap_hat

class MultiSubBandDiscriminator(torch.nn.Module):

    def __init__(self, tkernels, fkernel, tchannels, fchannels, tstrides, fstride, tdilations, fdilations, tsubband,
                 n, m, freq_init_ch):

        super(MultiSubBandDiscriminator, self).__init__()

        self.fsbd = SubBandDiscriminator(init_channel=freq_init_ch, channels=fchannels, kernel=fkernel,
                                         strides=fstride, dilations=fdilations)

        self.tsubband1 = tsubband[0]
        self.tsbd1 = SubBandDiscriminator(init_channel=self.tsubband1, channels=tchannels, kernel=tkernels[0],
                                          strides=tstrides[0], dilations=tdilations[0])

        self.tsubband2 = tsubband[1]
        self.tsbd2 = SubBandDiscriminator(init_channel=self.tsubband2, channels=tchannels, kernel=tkernels[1],
                                          strides=tstrides[1], dilations=tdilations[1])

        self.tsubband3 = tsubband[2]
        self.tsbd3 = SubBandDiscriminator(init_channel=self.tsubband3, channels=tchannels, kernel=tkernels[2],
                                          strides=tstrides[2], dilations=tdilations[2])


        self.pqmf_n = PQMF(N=n, taps=256, cutoff=0.03, beta=10.0)
        self.pqmf_m = PQMF(N=m, taps=256, cutoff=0.1, beta=9.0)

    def forward(self, x, x_hat):
        fmap = []
        fmap_hat = []
        y = []
        y_hat = []

        # Time analysis
        xn = self.pqmf_n(x)
        xn_hat = self.pqmf_n(x_hat)

        q3, feat_q3 = self.tsbd3(xn[:, :self.tsubband3, :])
        q3_hat, feat_q3_hat = self.tsbd3(xn_hat[:, :self.tsubband3, :])
        y.append(q3)
        y_hat.append(q3_hat)
        fmap.append(feat_q3)
        fmap_hat.append(feat_q3_hat)

        q2, feat_q2 = self.tsbd2(xn[:, :self.tsubband2, :])
        q2_hat, feat_q2_hat = self.tsbd2(xn_hat[:, :self.tsubband2, :])
        y.append(q2)
        y_hat.append(q2_hat)
        fmap.append(feat_q2)
        fmap_hat.append(feat_q2_hat)

        q1, feat_q1 = self.tsbd1(xn[:, :self.tsubband1, :])
        q1_hat, feat_q1_hat = self.tsbd1(xn_hat[:, :self.tsubband1, :])
        y.append(q1)
        y_hat.append(q1_hat)
        fmap.append(feat_q1)
        fmap_hat.append(feat_q1_hat)

        # Frequency analysis
        xm = self.pqmf_m(x)
        xm_hat = self.pqmf_m(x_hat)

        xm = xm.transpose(-2, -1)
        xm_hat = xm_hat.transpose(-2, -1)

        q4, feat_q4 = self.fsbd(xm)
        q4_hat, feat_q4_hat = self.fsbd(xm_hat)
        y.append(q4)
        y_hat.append(q4_hat)
        fmap.append(feat_q4)
        fmap_hat.append(feat_q4_hat)

        return y, y_hat, fmap, fmap_hat


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
