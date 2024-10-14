#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import os
import glob
import numpy as np
import torch
import pandas as pd
import natsort
import tifffile as tif

from datetime import date
from multiprocessing import Pool

from autoreconstruction.pytorch_segment.neurotorch.nets.RSUNet import RSUNet
from autoreconstruction.pytorch_segment.neurotorch.core.predictor import Predictor
from autoreconstruction.pytorch_segment.neurotorch.datasets.filetypes import TiffVolume
from autoreconstruction.pytorch_segment.neurotorch.datasets.dataset import Array
from autoreconstruction.pytorch_segment.neurotorch.datasets.datatypes import (
    BoundingBox, Vector)


def predict(checkpoint, specimen_dir, chunk_dir, bb, ids, error_list, gpu, files_per_chunk):

    # Step 1. Make Segmentation Output DIrectory 
    try:
        chunk_dir_base_path = os.path.basename(chunk_dir)
        
        if 'Left' in chunk_dir_base_path:
            seg_dir = os.path.join(specimen_dir,'Left_Segmentation')
        elif 'Right' in chunk_dir_base_path:
            seg_dir = os.path.join(specimen_dir,'Right_Segmentation')
        else:
            seg_dir = os.path.join(specimen_dir,'Segmentation')
            
        if not os.path.isdir(seg_dir):
            os.mkdir(seg_dir)

    except:
        print('error making segmentation directory')
        error_list.append(str(ids)+ ' -making directory')
   
    # Step 2. Run segmentation 
    try:
        net = RSUNet()
        count=0
        number_of_small_segments = len([ff for ff in os.listdir(chunk_dir) if '.tif' in ff])
        print('I think there are {} chunk tiff files in {}'.format(number_of_small_segments,chunk_dir))
        for n in range(1,number_of_small_segments+1):
            bbn = BoundingBox(Vector(bb[0], bb[1], bb[2]), Vector(bb[3], bb[4], files_per_chunk))

            nth_tiff_stack = os.path.join(chunk_dir,'chunk{}.tif'.format(n))
            with TiffVolume(nth_tiff_stack, bbn) as inputs:                         
                
                # Predict
                predictor = Predictor(net, checkpoint, gpu_device=gpu)
                output_volume = Array(-np.inf*np.ones(inputs.getBoundingBox().numpy_dims, dtype=np.float32))
                print('bb0', inputs.getBoundingBox())
                predictor.run(inputs, output_volume)      
                # Convert to probability map and save
                probability_map = 1/(1+np.exp(-output_volume.getArray()))
                print('probability_map', type(probability_map), probability_map.shape, probability_map.dtype)
                for i in range(probability_map.shape[0]): # Save 8-bit version as multiple tif files
                    #print('Prob Map Shape= ', probability_map.shape[0])
                    count +=1
                    tif.imsave(os.path.join(seg_dir,'%03d.tif'%(count)), np.uint8(255*probability_map[i,:,:]))                
    except:
        print('error with segmentation')
        error_list.append(str(ids)+ ' -segmentation')  

    # Step 3. Remove Duplicate Files if necessary 
    try:
        
        individual_tif_dir = os.path.join(specimen_dir,'Single_Tif_Images')
        number_of_individual_tiffs = len([f for f in os.listdir(individual_tif_dir) if '.tif' in f])
        number_of_segmented_tiffs =  len([f for f in os.listdir(seg_dir) if '.tif' in f])
        print('Number of individual tiffs = {}'.format(number_of_individual_tiffs))
        print('Number of segmented tiffs = {}'.format(number_of_segmented_tiffs))

        number_of_duplicates = number_of_segmented_tiffs-number_of_individual_tiffs
                # assigning the number of duplicates to the difference in length between segmented dir and individual tiff dir. 
        if number_of_duplicates == 0: 
            print('no duplicates were made')
            print('num duplicates = {}'.format(number_of_duplicates))

        else:
            print('num duplicates = {}'.format(number_of_duplicates))
            # this means that list_of_segmented_files[-files_per_chunk:-number_of_suplicates] can be erased because of part 7 in preprocessing
            list_of_segmented_files = [x for x in natsort.natsorted(os.listdir(seg_dir)) if '.tif' in x]
            second_index = files_per_chunk-number_of_duplicates
            duplicate_segmentations = list_of_segmented_files[-files_per_chunk:-(second_index)] 
            print(duplicate_segmentations)       

            for files in duplicate_segmentations:
                os.remove(os.path.join(seg_dir,files))
    except:
        print('error with removing files')
        error_list.append(str(ids)+' -removing duplicates')

    return error_list


def main(ckpt, outdir, csv, num_processes, gpu, chunk):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt', '-c', type=str, help='path to checkpoint')
    # parser.add_argument('--outdir', '-v', type=str, help='directory of validation/test data')
    # parser.add_argument('--csv', type=str, help='input csv with specimen id column header')
    # parser.add_argument('--num_processes', type=int, default = 1, help='number of processes to run parallel')
    # parser.add_argument('--gpu', type=int, default = 0, help='gpu device')
    # parser.add_argument('--chunk', type=int, default = 32, help='files per chunk')

    todays_date = date.today().strftime("%b_%d_%Y")

    # args = parser.parse_args()
    df = pd.read_csv(csv)
    specimens = list(df.specimen_id.values)
    files_per_chunk = chunk
    
    all_error_list = []
    if num_processes == 1:
        for sp_ids in specimens:       
            specimen_dir = os.path.join(outdir,str(sp_ids))
            error_list = []
            error_list_2 = []

            if os.path.exists(os.path.join(specimen_dir,'Chunks_of_%d_Left'%files_per_chunk)):
                # chunk_dirs 
                chunk_dir_left = os.path.join(specimen_dir,'Chunks_of_%d_Left'%files_per_chunk)
                chunk_dir_right = os.path.join(specimen_dir,'Chunks_of_%d_Right'%files_per_chunk)

                # bboxes
                left_bbox_path = os.path.join(outdir,str(sp_ids),'bbox_{}_Left.csv'.format(sp_ids))
                right_bbox_path = os.path.join(outdir,str(sp_ids),'bbox_{}_Right.csv'.format(sp_ids))
                df_l = pd.read_csv(left_bbox_path)
                df_r = pd.read_csv(right_bbox_path)
                bb_l = df_l.bound_boxing.values
                bb_r = df_r.bound_boxing.values

                # Predict
                res_l = predict(ckpt, specimen_dir, chunk_dir_left, bb_l, sp_ids, error_list, gpu, files_per_chunk)
                res_r = predict(ckpt, specimen_dir, chunk_dir_right, bb_r, sp_ids, error_list_2, gpu, files_per_chunk)

                all_error_list.append(res_l)
                all_error_list.append(res_r)
            
            else:
                # chunk dir
                chunk_dir = os.path.join(specimen_dir,'Chunks_of_%d'%files_per_chunk)
                
                # bboxes
                bbox_path = os.path.join(outdir,str(sp_ids),'bbox_{}.csv'.format(sp_ids))
                df = pd.read_csv(bbox_path)
                bb = df.bound_boxing.values

                # predict
                res = predict(ckpt, specimen_dir, chunk_dir, bb, sp_ids, error_list, gpu, files_per_chunk)
                all_error_list.append(res)

    else:
        p = Pool(processes=num_processes)
        parallel_input = [(ckpt, int(i), pd.read_csv(os.path.join(outdir,str(i),'bbox_{}.csv'.format(i))).bound_boxing.values, []) for i in specimens]
        all_error_list = p.starmap(predict, parallel_input)

    with open('{}_gpu{}_segmentation_error_log.txt'.format(todays_date, str(gpu)), 'w') as f:
        for item in all_error_list:
            f.write("%s\n" % item)
