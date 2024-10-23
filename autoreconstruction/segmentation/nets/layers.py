#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import torch
import torch.nn as nn


__doc__ = """
Basic building block layers for constructing nets implemented in PyTorch.

Nicholas Turner <nturner@cs.princeton.edu>, 2017
Based on a similar module by
Kisuk Lee <kisuklee@mit.edu>, 2016-2017
"""


def pad_size(ks, mode):
    assert mode in ["valid", "same", "full"]

    if mode == "valid":
        return (0, 0, 0)
    elif mode == "same":
        assert all([x % 2 for x in ks])
        return tuple(x // 2 for x in ks)
    elif mode == "full":
        return tuple(x - 1 for x in ks)


class Conv(nn.Module):
    """ Bare-bones 3D convolution module w/ MSRA init """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple,
            bias: bool = True
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(d_in, d_out, ks, st, pd, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.conv(x)


class FactConv(nn.Module):
    """ Factorized 3D convolution using Conv"""

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple,
            bias: bool = True
    ):
        super(FactConv, self).__init__()
        self.factor = None
        if ks[0] > 1:
            self.factor = Conv(
                d_in, d_out, ks=(1, ks[1], ks[2]), st=(1, st[1], st[2]),
                pd=(0, pd[1], pd[2]), bias=False)
            ks = (ks[0], 1, 1)
            st = (st[0], 1, 1)
            pd = (pd[0], 0, 0)

        self.conv = Conv(d_in, d_out, ks, st, pd, bias)

    def forward(
            self,
            x: torch.tensor
    ):
        out = (
            self.conv(x) if self.factor is None else self.conv(self.factor(x)))
        return out


class ConvT(nn.Module):
    """ Bare Bones 3D ConvTranspose module w/ MSRA init """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple = (0, 0, 0),
            bias: bool = True
    ):
        super(ConvT, self).__init__()
        self.conv = nn.ConvTranspose3d(d_in, d_out, ks, st, pd, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.conv(x)


class ResizeConv(nn.Module):
    """ Upsampling followed by a Convolution """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple,
            bias: bool = True,
            mode: str = "nearest"
    ):
        super(ResizeConv, self).__init__()
        self.upsample = Upsample2D(scale_factor=2, mode=mode)
        self.conv = Conv(d_in, d_out, ks, st, pd, bias=bias)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.conv(self.upsample(x))


class Upsample2D(nn.Module):

    def __init__(
            self,
            scale_factor: int,
            mode: str = "nearest"
    ):
        super(Upsample2D, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)

    def forward(
            self,
            x: torch.tensor
    ):
        # upsample in all dimensions, and undo the z upsampling
        return self.upsample(x)[:, :, ::self.scale_factor, :, :]


class FactConvT(nn.Module):
    """Factorized 3d ConvTranspose using ConvT"""

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple = (0, 0, 0),
            bias: bool = True
    ):
        super(FactConvT, self).__init__()
        self.factor = None
        if ks[0] > 1:
            self.factor = ConvT(
                d_in, d_out, ks=(2, ks[1], ks[2]), st=(1, st[1], st[2]),
                pd=(0, pd[1], pd[2]), bias=False)
            ks = (ks[0], 1, 1)
            st = (st[0], 1, 1)
            pd = (pd[0], 0, 0)

        self.conv = ConvT(d_in, d_out, ks, st, pd, bias)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.conv(self.factor(x) if self.factor is not None else x)
