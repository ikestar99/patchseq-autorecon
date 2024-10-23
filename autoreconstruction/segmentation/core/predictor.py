#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import torch
import numpy as np


from torch.autograd import Variable

from autoreconstruction.segmentation.datasets.dataset import Data


class Predictor:
    """
    A predictor segments an input volume into an output volume
    """
    def __init__(self, net, checkpoint, gpu_device=None):
        self.device = torch.device(
            f"cuda:{min(torch.cuda.device_count() + 1, gpu_device)}"
            if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device).eval()
        if checkpoint is not None:
            self.net.load_state_dict(
                torch.load(checkpoint, map_location=self.device))

    def run(self, input_volume, output_volume, batch_size=20):
        with torch.no_grad():
            for b in range(0, len(input_volume), batch_size):
                batch = [input_volume[i] for i in range(b, b + batch_size)]
                self.run_batch(batch, output_volume)

    def run_batch(self, batch, output_volume):
        bounding_boxes, arrays = self.toTorch(batch)
        inputs = Variable(arrays).float()
        outputs = self.net(inputs)
        data_list = self.toData(outputs, bounding_boxes)
        for data in data_list:
            output_volume.array.blend(data)

    def toArray(self, data):
        torch_data = data.array.astype(float)
        torch_data = torch_data.reshape(1, 1, *torch_data.shape)
        return torch_data

    def toTorch(self, batch):
        bounding_boxes = [data.bounding_box for data in batch]
        arrays = [self.toArray(data) for data in batch]
        arrays = torch.from_numpy(np.concatenate(arrays, axis=0))
        arrays = arrays.to(self.device)
        return bounding_boxes, arrays

    def toData(self, tensor_list, bounding_boxes):
        tensor = torch.cat(tensor_list).data.cpu().numpy()
        batch = [Data(tensor[i][0], bounding_box)
                 for i, bounding_box in enumerate(bounding_boxes)]
        return batch
