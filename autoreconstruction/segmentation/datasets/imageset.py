#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:00:00 2024
@author: ike
"""


import torch
import numpy as np

from torch.utils.data import Dataset


class ImageSet(Dataset):
    def __init__(
            self,
            stack: np.ndarray,
            voxel: tuple,
            pad: tuple,
            mask: np.ndarray = None
    ):
        super(ImageSet, self).__init__()
        assert stack.ndim == len(voxel), (
            f"voxel {voxel} must specify bounds of each dim in {stack.shape}")

        indices = np.array(
            [stack.shape[i] // voxel[i] for i in range(stack.ndim)])
        self.indices = np.arange(np.prod(indices)).reshape(indices)
        self.stack = stack
        self.voxel = voxel
        self.pad = pad
        self.mask = mask

    def __len__(
            self
    ):
        return self.indices.size

    def __getitem__(
            self,
            idx: int
    ):
        idx = self._index_to_voxel(idx)
        sample = torch.tensor(self.stack[idx])
        mask = (
            torch.tensor(self.mask[idx]) if self.mask is not None
            else self.mask)
        return sample, mask

    def __setitem__(
            self,
            idx: int,
            value: np.ndarray
    ):
        assert value.shape == self.voxel, (
            f"set shape {value.shape} must be identical to voxel{self.voxel}")

        idx = self._index_to_voxel(idx)
        self.stack[idx] = value

    def _index_to_voxel(
            self,
            idx: int
    ):
        idx = np.nonzero(self.indices == idx)
        idx = tuple(
            slice(d[0], d[0] + self.voxel[i]) for i, d in enumerate(idx))
        return idx
