#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@author: ike
"""


import tarfile
import os.path as op

from autoreconstruction.pipeline import PreProcess_ImageStack
from autoreconstruction.pipeline import ImageStack_To_Segmentation
from autoreconstruction.pipeline import Segmentation_To_Skeleton
from autoreconstruction.pipeline import Skeleton_To_Swc


def __main__(
        specimen_id: int,
        specimen_dir: str,
        raw_single_tif_dir_tarball: str,
        raw_single_tif_dir: str,
        ckpt: str,
        gpu: int,
        intensity_threshold: int,
        invert_image_color: bool,
        extract: bool,
        preprocess: bool,
        segment: bool,
        skeleton: bool,
        swc: bool):

    if extract:
        # extract relevant files from tarball
        with tarfile.open(raw_single_tif_dir_tarball, mode="r:gz") as tar:
            tar.extractall(path=specimen_dir)

    if preprocess:
        # running image stack processing
        PreProcess_ImageStack.main(
            specimen_id, raw_single_tif_dir, specimen_dir, invert_image_color)

    if segment:
        # running image stack segmentation
        ImageStack_To_Segmentation.main(
            ckpt, specimen_dir, raw_single_tif_dir, specimen_id, gpu)

    if skeleton:
        # running segmentation to skeleton
        Segmentation_To_Skeleton.main(specimen_dir, specimen_id, intensity_threshold)

    if swc:
        # running skeleton to swc
        Skeleton_To_Swc.skeleton_to_swc_parallel(
            specimen_id,
            specimen_dir,
            remove_intermediate_files=False,
            max_stack_size=7e9,
            minimum_soma_area_pixels=500,
            soma_connection_threshold=100)

    print("Example Pipeline Completed Successfully")


if __name__ == "__main__":
    specimen_id = 2112
    root = "/Users/ikogbonna/Documents/Code/patchseq-autorecon/"
    # Directory for specimen output files. If None, use basedir of raw_single_tif_dir
    specimen_dir = f"{root}autoreconstruction/pipeline/Example_Specimen_{specimen_id}/"

    raw_single_tif_dir_tarball = f"{specimen_dir}Example_Input_Stack.tar.gz"

    # A directory with individual tif files (z-slices)
    raw_single_tif_dir = f"{specimen_dir}Example_Input_Stack/"

    # Neural network will expect inverted (black background) images
    invert_image_color = "True"
    ckpt = "aspiny_model.ckpt"

    # 50 for spiny 252 for aspiny
    intensity_threshold = 252
    gpu = 0

    __main__(
        specimen_id=specimen_id,
        specimen_dir=specimen_dir,
        raw_single_tif_dir_tarball=raw_single_tif_dir_tarball,
        raw_single_tif_dir=raw_single_tif_dir,
        ckpt=ckpt,
        gpu=gpu,
        intensity_threshold=intensity_threshold,
        invert_image_color=invert_image_color,
        extract=False,  # ran successfully
        preprocess=False,  # ran successfully
        segment=False,  # ran successfully
        skeleton=False,  # ran successfully
        swc=True  # ran successfully on specimen 2112
    )
