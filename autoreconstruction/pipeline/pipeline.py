#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@author: ike
"""


import time

from autoreconstruction.pipeline.PreProcess_ImageStack import main as process_specimen
from autoreconstruction.pipeline.ImageStack_To_Segmentation import main as segment_stacks
from autoreconstruction.pipeline.Segmentation_To_Skeleton import main as postprocess
# from autoreconstruction.pipeline.Skeleton_To_Swc import main as skeleton_to_swc


def sec_to_time(
        count: float
):
    """
    Convert elapsed time in seconds to hh:mm:ss format.

    Args:
        count (float):
            Elapsed time in seconds.

    Returns:
        (str):
            Elapsed time in hh:mm:ss format.
    """
    h, r = divmod(count, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02}"


"""
Inputs to pipeline
"""
MODE = 0  # 0 = on portable macbook pro, 1 = on PC
# pipeline steps to run
PREPROCESS = True  # chunk images for memory, successful
SEGMENT = True  # run neural network segmentation
SKELETONIZE = True  # generate skeleton from segmentation
TO_SWC = False  # generate SWC file from skeleton

# UNet expects input with black background, y and x divisible by chunk_ds
INVERT_COLOR = True


CHECKPOINT_CKPT = (
    "/Users/ikogbonna/Documents/Code/patchseq-autorecon/autoreconstruction/pytorch_segment/spiny_model.ckpt",
    "C:\\Users\\ikest\\code\\patchseq-autorecon\\autoreconstruction\\pytorch_segment\\spiny_model.ckpt")[MODE]
BASE_DIR = (
    f"/Users/ikogbonna/Desktop/Workshop",
    f"C:\\Users\\ikest\\OneDrive\\Desktop\\clearing_example_PC")[MODE]
D = ("/", "\\")[MODE]
SPECIMENS = [
    # "Before 1", "After 1",
    # "Before 2", "After 2",
    # "KevRec",
    # "Allen",
    # "Ike Neuron",
    # "ACC_S5_C2", "ACC_S7_C1"
    "Auto Test"
]


def __main__(specimen):
    specimen_dir = BASE_DIR + D + specimen
    raw_dir = specimen_dir + D + "Example_Input_Stack"
    if PREPROCESS:  # running image stack processing
        sub_time = time.time()
        process_specimen(
            specimen_id=specimen,
            raw_single_tif_dir=raw_dir,
            specimen_dir=specimen_dir,
            invert_image_color=INVERT_COLOR)
        print(f"preprocess time: {sec_to_time(time.time() - sub_time)}")
    if SEGMENT:  # running image stack segmentation
        sub_time = time.time()
        segment_stacks(
            ckpt=CHECKPOINT_CKPT,
            specimen_dir=specimen_dir,
            raw_single_tif_dir=raw_dir,
            specimen_id=specimen,
            gpu=0)
        print(f"segment time: {sec_to_time(time.time() - sub_time)}")
    if SKELETONIZE:  # running segmentation to skeleton
        sub_time = time.time()
        postprocess(
            specimen_dir=specimen_dir,
            specimen_id=specimen,
            intensity_threshold=50)  # for spiny 252 for aspiny
        print(f"skeleton time: {sec_to_time(time.time() - sub_time)}")
    # if TO_SWC:  # running skeleton to swc
    #     sub_time = time.time()
    #     skeleton_to_swc(
    #         specimen_id=specimen,
    #         specimen_dir=specimen_dir,
    #         remove_intermediate_files=False,
    #         max_stack_size=7_000_000_000,
    #         minimum_soma_area_pixels=500,
    #         soma_connection_threshold=100)
    #     print(f"swc time: {sec_to_time(time.time() - sub_time)}")


if __name__ == "__main__":
    global_start = time.time()
    for s in SPECIMENS:
        print(f"\nautoreconning {s}", "\n", "-" * 60)
        t0 = time.time()
        __main__(s)

    print(f"Autorecon completed in: {sec_to_time(time.time() - global_start)}")
