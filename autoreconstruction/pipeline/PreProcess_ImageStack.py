#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import os
import math
import numpy as np
import pandas as pd
import psutil
import natsort
import tifffile as tf

from pathlib import Path


def check_for_size_limit(
        files: list
):
    """
    Will check and see if any of the file sizes in the input directory are
    greater than the available memory.
    """
    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"

        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"


    memory = psutil.virtual_memory()._asdict()['available']
    for f in files:
        size = os.path.getsize(f)
        if size > memory:
            print(
                f"WARNING: File: {f}\nMemory burden: {convert_size(size)}\n",
                f"Memory available: {convert_size(memory)}")


def int_round(
        x: float,
        base: int  # base=64
):
    return base * (x // base)


def process_specimen(
        ids: str,
        specimen_dir: Path,
        raw_single_tif_dir: Path,
        invert_image_color: bool,
        dz: int = 32,
        ds: int = 64,
        dc: int = 512
):
    """
    Worker function for script that will do a number of pre-processing steps.
    Mostly focused on putting the input images into a format (dimensions and
    color inversion) the neural network will be compatible with. The network
    was trained with a patch size of 64x64x32 so we need to get images into
    nxmx32 dimension where n and m are nearest multiple of 64.

    This script expects an input directory of single tif images
    (not 3d tif volumes) named in naturally ascending
    order (i.e. 1.tif, 2.tif, 3.tif...) and will run the following:

    TODO update split functionality such that whole image volume is sectioned
        into 3D chucks in all directions that are segmented separately and then
        recombined later on
    -- Get crop dimensions so input images are compatible with neural_network
    patch size
    -- Crop the images
    -- Stack the slices into chunks of 32 (check for memory limit)
    -- If number of slices is not a multiple of 32, there will be overlap in
    segmentation that is accounted for in ImageStack_To_Segmentation.py
    -- If memory limit is exceeded try splitting images into left and right
    -- Create raw input max intensity projections

    :param ids: specimen id
    :param specimen_dir: root directory for specimen
    :param raw_single_tif_dir: input directory of single tif images
    :param invert_image_color: boolean to invert images or not
    """

    error_list = []

    # Step 1 was removed because it was not useful for consumers outside AIBS
    # Step 2 chooses last file in list of tif files, extracts crop dimensions
    list_of_files = natsort.natsorted(raw_single_tif_dir.glob("*.tif"))
    check_for_size_limit(list_of_files)

    # step 6. make outdir for 3d chunks
    chunk_dir = specimen_dir.joinpath(f"Chunks_of_{dz}")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # raw image dimensions
    y_raw, x_raw = tf.imread(list_of_files[0]).shape

    # dimensions of largest crop divisible by base arg factor
    y_1, x_1 = int_round(y_raw, base=ds), int_round(x_raw, base=ds)

    # starting image coordinates for a center crop
    y_0, x_0 = (y_raw - y_1) // 2, (x_raw - x_1) // 2
    if (y_raw != y_1) or (x_raw != x_1):
        print(f"{ids} coordinates for x-y crop are not divisible by 64")
        error_list.append(f"{ids} Step 2")

    # Step 3. Crop, invert, chunk, and generate max proj of each image
    z_proj = None
    y_proj = []
    x_proj = []

    # mirror end of file list to nearest multiple of stack depth
    mirror = list_of_files + list_of_files[-2::-1][:(-len(list_of_files)) % dz]
    for z_chunk, f_idx in enumerate(range(0, len(mirror), dz)):
        stack = np.asarray([tf.imread(f) for f in mirror[f_idx:f_idx + dz]])
        stack = stack[..., y_0:y_0 + y_1, x_0:x_0 + x_1]
        stack = 255 - stack if invert_image_color else stack

        z_temp = np.max(stack, axis=0)
        z_proj = z_temp[0] if z_proj is None else np.maximum(z_proj, z_temp)
        y_proj += [np.max(stack, axis=1)]
        x_proj += [np.max(stack, axis=2)]

        for y_chunk, y_idx in enumerate(range(0, y_1, dc)):
            for x_chunk, x_idx in enumerate(range(0, x_1, dc)):
                y_end = min(y_idx + dc, y_1)
                x_end = min(x_idx + dc, x_1)
                tf.imwrite(
                    chunk_dir.joinpath(f"z{z_chunk}y{y_chunk}x{x_chunk}.tif"),
                    stack[:, y_idx:y_end, x_idx:x_end])

    tf.imwrite(specimen_dir.joinpath("Single_Tif_Images_Z_Mip.tif"), z_proj)
    tf.imwrite(
        specimen_dir.joinpath("Single_Tif_Images_Y_Mip.tif"),
        np.concatenate(y_proj, axis=0))
    tf.imwrite(
        specimen_dir.joinpath("Single_Tif_Images_X_Mip.tif"),
        np.concatenate(x_proj, axis=0))

    #Step 9. Generate The Bounding Box File (0 0 0 y x chunk_size)
    pd.DataFrame(
        {"bound_boxing": [0, 0, 0, min(x_1, dc), min(y_1, dc), dz]}).to_csv(
        specimen_dir.joinpath(f"bbox_{ids}.csv"))
    return error_list
