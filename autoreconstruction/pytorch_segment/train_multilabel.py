#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import os
import numpy as np
import torch
import torch.optim as optim
import random

from autoreconstruction.pytorch_segment.neurotorch.datasets.specification import JsonSpec
from autoreconstruction.pytorch_segment.neurotorch.core.trainer_multilabel import Trainer
from autoreconstruction.pytorch_segment.neurotorch.nets.RSUNetMulti import RSUNetMulti


def train(ckpt, ckpt_dir, log_dir, json_dir, eps, epochs, num_stacks, augmentation, pia_dir):
    """
    parser.add_argument('--ckpt', '-ck', type=str, help='path to checkpoint')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='directory to save checkpoint')
    parser.add_argument('--log_dir', '-l', type=str, help='directory to save logs')
    parser.add_argument('--json_dir', '-j', type=str, help='directory of json files')
    parser.add_argument('--eps', '-e', type=float, default=1e-1)
    parser.add_argument('--epochs', '-ep', type=int, default=10)
    parser.add_argument('--num_stacks', '-n', type=int, default=3)
    parser.add_argument('--augmentation', '-a', type=int, default=0, help='1-true 0-false')
    parser.add_argument('--pia_dir', '-p', type=str, help='directory of json files')
    """
    inputs_list = [f for f in os.listdir(json_dir) if 'inputs' in f]
    inputs_list.sort()
    labels_list = [f for f in os.listdir(json_dir) if 'labels' in f]
    labels_list.sort()    

    # Initialize network and json specification
    net = RSUNetMulti()
    json_spec = JsonSpec()

    # Define experiment name from arguments
    exp_name = str(json_dir.split('data/')[1]) + '_' + str(eps) + '_' + str(ckpt_dir)

    # Define checkpoints directory
    ckpt_dir = os.path.join('checkpoints', exp_name)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # Define log directory
    log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Add pia training data (optional)
    if pia_dir != 'None':
        print('adding pia training data')
        inputs_pia_list = [f for f in os.listdir(pia_dir) if 'inputs' in f]
        inputs_pia_list.sort()
        labels_pia_list = [f for f in os.listdir(pia_dir) if 'labels' in f]
        labels_pia_list.sort()  

        pia_spec = JsonSpec()
    else:
        inputs_pia_list = []
        labels_pia_list = []
        pia_spec = None
    
    for k in range(epochs):
        # Create random indices
        idx = np.random.permutation(len(inputs_list))
        print(idx)
        
        if pia_dir != 'None':
            idx_pia = np.random.permutation(len(inputs_pia_list))[:int(np.floor(len(inputs_list)/num_stacks))]
            print('idx_pia:', idx_pia)
        else:
            idx_pia = None
            
        for i in range(int(np.floor(len(idx)/num_stacks))):
            ckpt = run_subset(ckpt, ckpt_dir, log_dir, eps, epochs, num_stacks, net, k, i, idx, json_spec, json_dir, inputs_list, labels_list, 
                       augmentation, pia_dir, idx_pia, pia_spec, inputs_pia_list, labels_pia_list)
            
def run_subset(ckpt, ckpt_dir, log_dir, eps, epochs, num_stacks, net, k, i, idx, json_spec, json_dir, inputs_list, labels_list, 
               augmentation, pia_dir, idx_pia, pia_spec, inputs_pia_list, labels_pia_list): 
    i1 = i*num_stacks
    i2 = i*num_stacks + num_stacks
    select = idx[i1:i2]
    print(i, i1, i2, select)
    spec1 = [json_spec.parse(os.path.join(json_dir, inputs_list[j]))[0] for j in select]
    spec2 = [json_spec.parse(os.path.join(json_dir, labels_list[j]))[0] for j in select]

    validation_split = 0.01
    inputs_vol = []
    labels_vol = []
    inputs_vol_val = []
    labels_vol_val = []

    for s in range(len(spec1)):
        volume1 = []
        volume2 = []
        idx_bkg = []
        spec = [spec1[s]]
        inputs = json_spec.create(spec,stack_size=33)
        spec = [spec2[s]]
        labels = json_spec.create(spec,stack_size=33)

        for n in range(len(inputs)):
            if not (labels[n].array == 0).all(): # Patches with nonzero labels
                volume1.append(inputs[n].array.astype(np.uint8))
                volume2.append(labels[n].array.astype(np.uint8))
            elif (inputs[n].array > 110).any(): # Patches with backgroud>threshold
                idx_bkg.append(n)

        # Load random subset (<=num_bkg) of bkg patches
        random.shuffle(idx_bkg)
        num_bkg = 750
        volume1_bkg = [inputs[n].array.astype(np.uint8) for n in idx_bkg[:num_bkg]]
        volume2_bkg = [labels[n].array.astype(np.uint8) for n in idx_bkg[:num_bkg]]

        valid_indexes = np.arange(len(volume1))
        np.random.seed(0)
        random_idx = np.random.permutation(valid_indexes)
        val_idx = random_idx[int(len(valid_indexes)*(1-validation_split)):].copy()
        volume1_val = [volume1[ind] for ind in val_idx] # Create validation inputs volume
        volume2_val = [volume2[ind] for ind in val_idx] # Create validation labels volume

        for ind in sorted(val_idx, reverse=True): # Remove validation data from training volume
            del volume1[ind]
            del volume2[ind]

        inputs_vol = inputs_vol + volume1 + volume1_bkg
        labels_vol = labels_vol + volume2 + volume2_bkg
        inputs_vol_val = inputs_vol_val + volume1_val
        labels_vol_val = labels_vol_val + volume2_val

    # Apply augmentation (optional)
    if augmentation:
        inputs_vol, labels_vol = augment(inputs_vol, labels_vol)

    # Add pia training data (optional)
    if pia_dir != 'None':
        inputs = pia_spec.open(os.path.join(pia_dir, inputs_pia_list[idx_pia[i]]))
        labels = pia_spec.open(os.path.join(pia_dir, labels_pia_list[idx_pia[i]]))
        volume1 = []
        volume2 = []
        for n in range(len(inputs)):
            # Select patches with zero labels and inputs intensity above threshold=60
            if (labels[n].array == 0).all() & (inputs[n].array > 60).any():
                volume1.append(inputs[n].array.astype(np.uint8))
                volume2.append(labels[n].array.astype(np.uint8))

        # Select subset of patches if total number exceeds num_pia
        num_pia = 2250
        if len(volume1)> num_pia:
            select_pia = np.random.permutation(len(volume1))[:num_pia]
            volume1 = [volume1[m] for m in select_pia]
            volume2 = [volume2[m] for m in select_pia] 

        # Add pia data to training data
        inputs_vol = inputs_vol + volume1
        labels_vol = labels_vol + volume2          

    inputs_vol = [inputs_vol, inputs_vol_val]
    labels_vol = [labels_vol, labels_vol_val]

    # Initialize optimizer with updated epsilon parameter
    optimizer = optim.Adam(net.parameters(), eps=eps)

    # Initialize trainer
    if (i==0) & (ckpt == 'None'):
        trainer = Trainer(net, inputs_vol, labels_vol, checkpoint_dir=ckpt_dir, checkpoint_period=10000, 
                             logger_dir=log_dir, max_epochs=1, gpu_device=0, optimizer=optimizer)       
    else:
        trainer = Trainer(net, inputs_vol, labels_vol, checkpoint_dir=ckpt_dir, checkpoint_period=10000, 
                             logger_dir=log_dir, checkpoint=ckpt, max_epochs=1, gpu_device=0, optimizer=optimizer)

    cell_str = str()
    for s1 in select:
        cell_str = cell_str + '_{:03d}'.format(s1)
    print(k, i, cell_str, ckpt)

    # Begin training
    trainer.run_training()

    # Save the last model       
    trainer.save_checkpoint('last{:03d}_{:03d}{}.ckpt'.format(k, i, cell_str))            

    # Save the best model
    os.rename(os.path.join(ckpt_dir, 'best.ckpt'), os.path.join(ckpt_dir, 'best{:03d}_{:03d}{}.ckpt'.format(k, i, cell_str)))

    # Set a new checkpoint
    ckpt = os.path.join(ckpt_dir, 'last{:03d}_{:03d}{}.ckpt'.format(k, i, cell_str)) # use the last model as a new checkpoint
    return ckpt

def augment(inputs_volume, labels_volume):
    for l in range(len(inputs_volume)):
        r = np.random.randint(4) # One of 4 orientations
        if np.random.randint(2): # Flip or not flip
            inputs_volume[l] = np.flip(np.rot90(inputs_volume[l],r,axes=(1,2)),1)
            labels_volume[l] = np.flip(np.rot90(labels_volume[l],r,axes=(1,2)),1)
        else:
            inputs_volume[l] = np.rot90(inputs_volume[l],r,axes=(1,2))
            labels_volume[l] = np.rot90(labels_volume[l],r,axes=(1,2))
    return inputs_volume, labels_volume