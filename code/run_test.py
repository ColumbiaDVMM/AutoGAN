#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: run_test.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Tue Oct 15 16:39:19 2019
#
#  Usage: python run_test.py -h
#  Description: Evaluate a GAN image detector
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

import numpy as np
import scipy.io as sio
import time
import os
import sys
import subprocess
import shlex
import argparse

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

####################################################################
# Parse command line
####################################################################
parser.add_argument('--dataset', type=str, default='CycleGAN', help='Training dataset select from: CycleGAN and AutoGAN')
parser.add_argument('--feature', default='image', help='Feature used for training, choose from image and fft')

args = parser.parse_args()

gpu_set = ['0']

#Compare image and spectrum
if args.feature == 'image':
    parameter_set = [
            ' --feature=image ',
            ]
elif args.feature == 'fft':
    parameter_set = [
            ' --feature=fft '
            ]
#Compare different frequency band
#parameter_set = [
#        ' --feature=fft '
#        ' --feature=fft --mode=1'
#        ' --feature=fft --mode=2'
#        ' --feature=fft --mode=3'
#        ]
else:
    print('Not a valid feature!')
    exit(-1)



number_gpu = len(gpu_set)

if args.dataset == 'CycleGAN':
    datasets = ['horse', 'zebra', 'summer', 'winter', 'apple', 'orange',  'facades', 'cityscapes', 'satellite', 'ukiyoe', 'vangogh', 'cezanne', 'monet', 'photo']
elif args.dataset == 'AutoGAN':
    datasets = ['horse_auto', 'zebra_auto', 'summer_auto', 'winter_auto', 'apple_auto', 'orange_auto', 'facades_auto', 'cityscapes_auto', 'satellite_auto', 'ukiyoe_auto', 'vangogh_auto', 'cezanne_auto', 'monet_auto', 'photo_auto']
else:
    print('Not a valid dataset!')
    exit(-1)

#leave one out setting 
#datasets = ['horse+zebra --leave_one_out ', 'apple+orange --leave_one_out ',
#            'summer+winter --leave_one_out ', 'cityscapes --leave_one_out ', 
#            'satellite --leave_one_out ', 'facades --leave_one_out ', 
#            'fold6 --leave_one_out ', 'fold7 --leave_one_out ',
#            'fold8 --leave_one_out ', 'fold9 --leave_one_out ']

process_set = []

for dataset in datasets:
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
        command = 'python ./code/GAN_Detection_Test.py --training-set {} --model=resnet --test-set=transposed_conv --data_augment \
                --batch-size=16 --test-batch-size=16 --epochs 10 {}  --gpu-id {} --model-dir ./model_resnet/ '\
                .format(dataset, parameter, gpu_set[idx%number_gpu]) 
        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)
        
        if (idx+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
    
        time.sleep(10)
    
    for sub_process in process_set:
        sub_process.wait()

