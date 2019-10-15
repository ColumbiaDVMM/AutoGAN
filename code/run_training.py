#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: run_training.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Tue Oct 15 16:27:57 2019
#
#  Usage: python run_training 
#  Description: Train a GAN image detector
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

#! /usr/bin/env python2

import numpy as np
import scipy.io as sio
import time
import os
import sys
import subprocess
import shlex
import argparse

####################################################################
# Parse command line
####################################################################
parser.add_argument('--dataset', type=str, default='CycleGAN', help='Training dataset select from: CycleGAN and AutoGAN')
parser.add_argument('--feature', default='image', help='Feature used for training, choose from image and fft')
parser.add_argument('--gpu-id', default='0', help='Feature used for training, choose from image and fft')

args = parser.parse_args()

gpu_set = args.gpu_id.split(',')
# add more gpu if wanted
# gpu_set = ['0','1']

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

process_set = []

index = 0
for idx, parameter in enumerate(parameter_set):
    for dataset in datasets:
        print('Test Parameter: {}'.format(parameter))
        command = 'python ./code/GAN_Detection_Train.py --training-set {} --model=resnet --test-set=transposed_conv --data_augment\
                --batch-size=16 --test-batch-size=16 {} --gpu-id {} --model-dir ./model_resnet/  --log-dir ./resnet_log/ --enable-logging=False --epochs 20 '\
                .format(dataset, parameter, gpu_set[index%number_gpu])# 
    
        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)
         
        if (index+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
        
        index+=1
        time.sleep(60)
    
for sub_process in process_set:
    sub_process.wait()

