#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import os
import numpy as np
import pandas as pd
import natsort
import tifffile as tif

from pathlib import Path
from datetime import date

from autoreconstruction.pytorch_segment.neurotorch.nets.RSUNetMulti import RSUNetMulti
from autoreconstruction.pytorch_segment.neurotorch.core.predictor_multilabel import Predictor
from autoreconstruction.pytorch_segment.neurotorch.datasets.filetypes import TiffVolume
from autoreconstruction.pytorch_segment.neurotorch.datasets.dataset import Array
from autoreconstruction.pytorch_segment.neurotorch.datasets.datatypes import (
    BoundingBox, Vector)


def validate(
        checkpoint,
        specimen_dir,
        chunk_dir,
        raw_single_tif_dir,
        bb,
        ids,
        gpu,
):
    """
    Will run segmentation on input directory of 3d tif volumes. These volumes
    should have dimensions: 64*n x 64*m x 32. Segmentation will create a
    subdirectory in the specimen_dir named Segmentation.
    Ch1 = soma, Ch2 = axon, Ch3 = dendrite

    :param checkpoint: checkpoint file
    :param specimen_dir: specimens directory
    :param chunk_dir: directory within specimens directory that has tif volumes
    :param raw_single_tif_dir: original single tif image directory
    :param bb: bounding box specifying segmentation dimensions, created in
        PreProcessing_ImageStack.py
    :param ids: specimen id
    :param gpu: 0
    """
    error_list = []
    print(f"Using checkpoint file: {checkpoint}")

    # Step 1. make segmentation output directory
    seg_dir = specimen_dir.joinpath("Segmentation")
    seg_dir.mkdir(parents=True, exist_ok=True)
   
    #Step 2. Run segmentation
    predictor = Predictor(RSUNetMulti(), checkpoint, gpu_device=gpu)
    count = [0,0,0]
    bbn = BoundingBox(
        Vector(bb[0], bb[1], bb[2]),
        Vector(bb[3], bb[4], bb[5]))
    for f in natsort.natsorted(chunk_dir.glob("*.tif")):
        with TiffVolume(f, bbn) as inputs:
            # output_volume is [Ch1 array, Ch2 array, Ch3 array]
            output_volume = [
                Array(np.zeros(inputs.bounding_box.numpy_dims, dtype=np.uint8))
                for _ in range(3)]
            print('bb0', inputs.bounding_box)
            predictor.run(inputs, output_volume)
            for ch in range(3):
                ch_dir = os.path.join(seg_dir,'ch%d'%(ch+1))
                if not os.path.isdir(ch_dir):
                    os.mkdir(ch_dir)
                probability_map = output_volume[ch].getArray()
                for i in range(probability_map.shape[0]): # save as multiple tif files
                    #print('Prob Map Shape= ', probability_map.shape[0])
                    count[ch] +=1
                    tif.imsave(os.path.join(ch_dir,'%03d.tif'%(count[ch])), probability_map[i,:,:])

    #Step 3. Remove Duplicate Files if necessary 
    try:
        
        number_of_individual_tiffs = len([f for f in os.listdir(raw_single_tif_dir) if '.tif' in f])
        for ch in range(3):
            ch_dir = os.path.join(seg_dir,'ch%d'%(ch+1))
            number_of_segmented_tiffs =  len([f for f in os.listdir(ch_dir) if '.tif' in f])
            print('Number of individual tiffs = {}'.format(number_of_individual_tiffs))
            print('Number of segmented tiffs = {}'.format(number_of_segmented_tiffs))

            number_of_duplicates = number_of_segmented_tiffs-number_of_individual_tiffs
                    #assigning the number of duplicates to the difference in length between segmented dir and individual tiff dir. 
            if number_of_duplicates == 0: 
                print('no duplicates were made')
                print('num duplicates = {}'.format(number_of_duplicates))

            else:
                print('num duplicates = {}'.format(number_of_duplicates))
                #this means that list_of_segmented_files[-32:-number_of_suplicates] can be erased because of part 7 in preprocessing
                list_of_segmented_files = [x for x in natsort.natsorted(os.listdir(ch_dir)) if '.tif' in x]
                second_index = 32-number_of_duplicates
                duplicate_segmentations = list_of_segmented_files[-32:-(second_index)] 
                print(duplicate_segmentations)       

                for files in duplicate_segmentations:
                    os.remove(os.path.join(ch_dir,files))
    except:
        print('error with removing files')
        error_list.append(str(ids)+' -removing duplicates')

    return error_list


def main(ckpt, specimen_dir, raw_single_tif_dir, specimen_id, gpu, **kwargs ):
    """
    ckpt = ags.fields.InputFile(description='checkpoint file to use for segmentation ')
    specimen_dir = ags.fields.InputDir(description="specimen directory")
    specimen_id = ags.fields.Str(default=None,description="specimen id")
    gpu = ags.fields.Int(default=0, description = "gpu to use")
    raw_single_tif_dir = ags.fields.InputDir(description='raw image directory')
    """
    today = date.today().strftime("%b_%d_%Y")
    #chunk dir
    chunk_dir = os.path.join(specimen_dir,'Chunks_of_32')

    #bboxes
    bbox_path = os.path.join(specimen_dir,'bbox_{}.csv'.format(specimen_id))
    df = pd.read_csv(bbox_path)
    bb = df.bound_boxing.values

    #validate
    validate(ckpt, specimen_dir, chunk_dir, raw_single_tif_dir, bb, specimen_id, [], gpu)
