#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@author: ike
"""


import os
import tarfile
from autoreconstruction.pipeline import PreProcess_ImageStack
from autoreconstruction.pipeline.ImageStack_To_Segmentation import segment_image_stack
from autoreconstruction.pipeline.Segmentation_To_Skeleton import segmentation_to_skeleton
from autoreconstruction.pipeline.Skeleton_To_Swc import skeleton_to_swc


specimen_id = 2112

# Directory for specimen output files. If None, use basedir of raw_single_tif_dir
specimen_dir = f"/autoreconstruction/pipeline/Example_Specimen_{specimen_id}/"

raw_single_tif_dir_tarball = f"{specimen_dir}Example_Input_Stack.tar.gz"

# A directory with individual tif files (z-slices)
raw_single_tif_dir = f"{specimen_dir}Example_Input_Stack/"

# Neural network will expect inverted (black background) images
invert_image_color = "True"
ckpt = "aspiny_model.ckpt"
intensity_threshold = 252

# Extract relevant files from tarball
with tarfile.open(raw_single_tif_dir_tarball, mode="r:gz") as tar:
    tar.extractall(path=specimen_dir)

# Running image stack processing
PreProcess_ImageStack.main(
    specimen_id, raw_single_tif_dir, specimen_dir, invert_image_color)

echo "RUNNING SEGMENTATION"
python ImageStack_To_Segmentation.py --specimen_dir ${specimen_dir} \
--specimen_id ${specimen_id} \
--raw_single_tif_dir ${raw_single_tif_dir} \
--ckpt ${ckpt}

echo "RUNNING SEGMENTATION TO SKELETON"
python Segmentation_To_Skeleton.py --specimen_dir ${specimen_dir} \
--specimen_id ${specimen_id} \
--intensity_threshold ${intensity_threshold}

echo "RUNNING SKELETON TO SWC"
python Skeleton_To_Swc.py --specimen_dir ${specimen_dir} \
--specimen_id ${specimen_id} \

echo "Example Pieline Completed Succesfully "