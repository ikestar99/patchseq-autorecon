#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from autoreconstruction.segmentation.nets import layers
from autoreconstruction.segmentation.nets.netcollector import NetCollector


__doc__ = """
Symmetric 3d U-Net for multiclass segmentation implemented in PyTorch

(Optional)
Factorized 3D convolution, Extra residual connections

Nicholas Turner <nturner@cs.princeton.edu>, 2017
Based on an architecture by Kisuk Lee <kisuklee@mit.edu>, 2017

Ike Ogbonna <Ike.Ogbonna@ucsf.edu> deleted redundant RSUNet module
"""


# Global switches
FACTORIZE = False
RESIDUAL = True
BN = True

# Number of feature maps
NFEATURES = [16, 16, 64, 128, 80, 96]

# Filter size
SIZES = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]

# In/out filter & stride size
IO_SIZE = (3, 3, 3)
IO_STRIDE = (1, 1, 1)


class Conv(nn.Module):
    """ Single convolution module """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple = (1, 1, 1),
            activation=F.elu,
            fact: bool = FACTORIZE
    ):
        super(Conv, self).__init__()
        pd = layers.pad_size(ks, "same")

        conv_constr = layers.FactConv if fact else layers.Conv
        self.conv = conv_constr(d_in, d_out, ks, st, pd, bias=True)
        self.activation = activation

    def forward(
            self,
            x: torch.tensor
    ):
        return self.activation(self.conv(x))


class ConvMod(nn.Module):
    """ Convolution "module" """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            activation=F.elu,
            fact: bool = FACTORIZE,
            resid: bool = RESIDUAL,
            bn: bool = BN,
            momentum: float = 0.5
    ):
        super(ConvMod, self).__init__()
        st = (1, 1, 1)
        pd = layers.pad_size(ks, "same")
        conv_constr = layers.FactConv if fact else layers.Conv
        bias = not bn

        self.resid = resid
        self.bn = bn
        self.activation = activation

        first_pd = layers.pad_size((1, ks[1], ks[2]), "same")
        self.conv1 = conv_constr(
            d_in, d_out, (1, ks[1], ks[2]), st, first_pd, bias)
        self.conv2 = conv_constr(d_out, d_out, ks, st, pd, bias)
        self.conv3 = conv_constr(d_out, d_out, ks, st, pd, bias)
        if self.bn:
            self.bn1 = nn.BatchNorm3d(d_out, momentum=momentum)
            self.bn2 = nn.BatchNorm3d(d_out, momentum=momentum)
            self.bn3 = nn.BatchNorm3d(d_out, momentum=momentum)

    def forward(
            self,
            x: torch.tensor
    ):
        out1 = self.activation(
            self.bn1(self.conv1(x)) if self.bn else self.conv1(x))
        out2 = self.activation(
            self.bn2(self.conv2(out1)) if self.bn else self.conv2(out1))

        out3 = (self.conv3(out2) + out1) if self.resid else self.conv3(out2)
        out3 = self.activation(self.bn3(out3) if self.bn else out3)
        return out3


class ConvTMod(nn.Module):
    """ Transposed Convolution "module" """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            up: tuple = (2, 2, 2),
            activation=F.elu,
            fact: bool = FACTORIZE,
            resid: bool = RESIDUAL,
            bn: bool = BN,
            momentum: float = 0.5
    ):
        super(ConvTMod, self).__init__()

        # ConvT constructor
        convt_constr = layers.FactConvT if fact else layers.ConvT
        self.bn = bn
        self.activation = activation
        bias = not bn

        self.convt = convt_constr(d_in, d_out, ks=up, st=up, bias=bias)
        self.convmod = ConvMod(d_out, d_out, ks, fact=fact, resid=resid, bn=bn)
        if bn:
            self.bn1 = nn.BatchNorm3d(d_out, momentum=momentum)

    def forward(
            self,
            x: torch.tensor,
            skip: torch.tensor
    ):
        out = self.convt(x) + skip
        out = self.convmod(self.activation(self.bn1(out) if self.bn else out))
        return out


class OutputModule(nn.Module):
    """ Hidden representation -> Output module """

    def __init__(
            self,
            d_in: int,
            outspec: OrderedDict,
            ks: tuple = IO_SIZE,
            st: tuple = IO_STRIDE
    ):
        super(OutputModule, self).__init__()
        pd = layers.pad_size(ks, mode="same")
        self.output_layers = []
        # self.output_layers = list(outspec.keys())
        for name, d_out in outspec.items():
            setattr(self, name, layers.Conv(
                d_in, d_out, ks, st, pd, bias=True))
            self.output_layers.append(name)

    def forward(
            self,
            x: torch.tensor
    ):
        return [getattr(self, layer)(x) for layer in self.output_layers]


class RSUNetMulti(nn.Module):
    """ Full model for multiclass segmentation"""

    def __init__(
            self,
            d_in: int = 1,  # convention, N input dims / feature dims
            output_spec: OrderedDict = OrderedDict(label=4),
            # output_spec: OrderedDict = OrderedDict(soma_label=1),  RSUNet
            depth: int = 4,
            io_size: tuple = IO_SIZE,
            io_stride: tuple = IO_STRIDE,
            bn: bool = BN
    ):
        super(RSUNetMulti, self).__init__()

        assert depth < len(NFEATURES)
        self.depth = depth

        # Input feature embedding without batchnorm
        fs = NFEATURES[0]
        self.inputconv = Conv(d_in, fs, io_size, st=io_stride)
        d_in = fs

        # modules in up/down paths added with setattr, obscured by U3D methods
        # Contracting pathway
        for d in range(depth):
            fs = NFEATURES[d]
            ks = SIZES[d]
            self.add_conv_mod(d, d_in, fs, ks, bn)
            self.add_max_pool(d+1, fs)
            d_in = fs

        # Bridge
        fs = NFEATURES[depth]
        ks = SIZES[depth]
        self.add_conv_mod(depth, d_in, fs, ks, bn)
        d_in = fs

        # Expanding pathway
        for d in reversed(range(depth)):
            fs = NFEATURES[d]
            ks = SIZES[d]
            self.add_deconv_mod(d, d_in, fs, bn, ks)
            d_in = fs

        # Output feature embedding without batchnorm
        self.embedconv = Conv(d_in, d_in, ks, st=(1, 1, 1))

        # Output by spec
        self.outputdeconv = OutputModule(
            d_in, output_spec, ks=io_size, st=io_stride)

    def add_conv_mod(self, depth, d_in, d_out, ks, bn):
        setattr(self, f"convmod{depth}", ConvMod(d_in, d_out, ks, bn=bn))

    def add_max_pool(self, depth, down=(2, 2, 2)):
        setattr(self, f"maxpool{depth}", nn.MaxPool3d(down))

    def add_deconv_mod(self, depth, d_in, d_out, bn, up=(2, 2, 2)):
        setattr(self, f"deconv{depth}", ConvTMod(d_in, d_out, up, bn=bn))

    def forward(
            self,
            x: torch.tensor
    ):
        # Input feature embedding without batchnorm
        x = self.inputconv(x)

        # Contracting pathway
        skip = []
        for d in range(self.depth):
            cd = getattr(self, f"convmod{d}")(x)
            x = getattr(self, f"maxpool{d + 1}")(cd)
            skip.append(cd)

        # Bridge
        x = getattr(self, f"convmod{self.depth}")(x)

        # Expanding pathway
        for d in reversed(range(self.depth)):
            x = getattr(self, f"deconv{d}")(x, skip[d])

        # Output feature embedding without batchnorm
        return self.outputdeconv(self.embedconv(x))


# NetCollector().add_module(RSUNetMulti(), "RSUNetMulti")
