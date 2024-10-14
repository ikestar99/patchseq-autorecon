#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@author: ike
"""


import tarfile

from pathlib import Path

from autoreconstruction.pipeline.PreProcess_ImageStack import process_specimen
from autoreconstruction.pipeline import ImageStack_To_Segmentation
from autoreconstruction.pipeline import Segmentation_To_Skeleton
from autoreconstruction.pipeline import Skeleton_To_Swc


def __main__(
        specimen_id: str,
        specimen_dir: Path,
        raw_single_tif_dir_tarball: Path,
        raw_single_tif_dir: Path,
        ckpt: str,
        gpu: int,
        intensity_threshold: int,
        invert_image_color: bool,
        extract: bool,
        preprocess: bool,
        chunk_z: int,
        chunk_ds: int,
        chunk_xy: int,
        segment: bool,
        skeleton: bool,
        swc: bool):

    if extract:
        # extract relevant files from tarball
        with tarfile.open(raw_single_tif_dir_tarball, mode="r:gz") as tar:
            tar.extractall(path=specimen_dir)

    if preprocess:
        # running image stack processing
        errors = process_specimen(
            ids=specimen_id,
            specimen_dir=specimen_dir,
            raw_single_tif_dir=raw_single_tif_dir,
            invert_image_color=invert_image_color,
            dz=chunk_z,
            ds=chunk_ds,
            dc=chunk_xy)
        print(f"Image Preprocessing errors: {errors}")

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


if __name__ == "__main__":
    """
    Inputs to pipeline
    """
    # specimen_id = "2112"
    specimen_id = "2000"

    # absolute path to local repository
    root = Path("/Users/ike/Documents/Code/patchseq-autorecon")

    # directory for specimen related files, relative to root
    specimen_dir = root.joinpath(
        Path(f"autoreconstruction/pipeline/Example_Specimen_{specimen_id}"))

    # path to tarball specimen file, relative to specimen_dir
    raw_single_tif_dir_tarball = specimen_dir.joinpath(
        "Example_Input_Stack.tar.gz")

    # directory with individual tif files, relative to specimen_dir
    raw_single_tif_dir = specimen_dir.joinpath("Example_Input_Stack")

    # UNet expects input with black background, y and x divisible by chunk_ds
    invert_image_color = True
    ckpt = "aspiny_model.ckpt"
    chunk_z = 32  # from training
    chunk_ds = 64  # from training, y and x
    chunk_scalar = 8  # chunk x, y = chunk_ds * chunk_scalar

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
        chunk_z=chunk_z,
        chunk_ds=chunk_ds,
        chunk_xy=chunk_ds * chunk_scalar,
        segment=False,  # ran successfully
        skeleton=False,  # ran successfully
        swc=True  # ran successfully on specimen 2112
    )
