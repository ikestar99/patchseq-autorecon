#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import os
import numpy as np
import tifffile as tif

from autoreconstruction.segmentation.nets import RSUNetMulti
from autoreconstruction.segmentation.core.predictor import Predictor
from autoreconstruction.segmentation.datasets import TiffVolume
from autoreconstruction.segmentation.datasets.dataset import Array
from autoreconstruction.segmentation.datasets import (BoundingBox, Vector)


def predict(checkpoint, test_dir, out_dir, bb, num_parts):
    """
    parser.add_argument('--ckpt', '-c', type=str, help='path to checkpoint')
    parser.add_argument('--test_dir', '-v', type=str, help='directory of validation/test data')
    parser.add_argument('--out_dir', '-o', type=str, help='results directory path')
    parser.add_argument('--bb', '-b', nargs='+', type=int, help='bounding box')
    parser.add_argument('--num_parts', '-n', type=int, help='number of parts to divide volume')
    """
    # Initialize the U-Net architecture
    net = RSUNetMulti()

    offset = 0
    for n in range(num_parts):
        bbn = BoundingBox(Vector(bb[0], bb[1], bb[2]), Vector(bb[3], bb[4], bb[5+n]))
        print(n, bbn)
        print('offset', offset) 
        if num_parts==1:
            filename = 'inputs_cropped' + '.tif'
        else:
            filename = 'inputs_cropped' + str(n) + '.tif'
        print(os.path.join(test_dir, filename))    

        with TiffVolume(os.path.join(test_dir, filename), bbn) as inputs:              
            # Predict
            predictor = Predictor(net, checkpoint, gpu_device=0)
            # Output_volume is a list (len3) of Arrays for each of 3 foreground channels (soma, axon, dendrite)
            output_volume = [Array(np.zeros(inputs.bounding_box.numpy_dims, dtype=np.uint8)) for _ in range(3)]
            print('bb0', inputs.bounding_box)
            predictor.run(inputs, output_volume)
                            
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for ch in range(3):
            ch_dir = os.path.join(out_dir,'ch%d'%(ch+1))
            if not os.path.isdir(ch_dir):
                os.mkdir(ch_dir)
            probability_map = output_volume[ch].getArray()
            for i in range(probability_map.shape[0]): # Save as multiple tif files
                tif.imsave(os.path.join(ch_dir,'%03d.tif'%(i+offset)), probability_map[i,:,:])
        offset = offset + bb[5+n]
