"""
Check the correctness of gor on HardNet loss using multiple GPUs
Usage: check_gor_HardNet.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

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
def usage():
    print >> sys.stderr 
    sys.exit(1)

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

gpu_set = ['0']

#Compare image and spectrum
parameter_set = [
        #' --feature=image ',
        ' --feature=fft '
        ]

#Compare different frequency band
#parameter_set = [
#        ' --feature=fft '
#        ' --feature=fft --mode=1'
#        ' --feature=fft --mode=2'
#        ' --feature=fft --mode=3'
#        ]
number_gpu = len(gpu_set)

#datasets = [ 'horse ', 'horse_auto ', 'summer', 'summer_auto ']
#datasets = ['horse', 'zebra', 'summer', 'winter', 'apple', 'orange',  'facades', 'cityscapes', 'satellite', 'ukiyoe', 'vangogh', 'cezanne', 'monet', 'photo']

datasets = ['horse_auto', 'zebra_auto', 'summer_auto', 'winter_auto', 'apple_auto', 'orange_auto', 'facades_auto', 'cityscapes_auto', 'satellite_auto', 'ukiyoe_auto', 'vangogh_auto', 'cezanne_auto', 'monet_auto', 'photo_auto','horse', 'zebra', 'summer', 'winter', 'apple', 'orange',  'facades', 'cityscapes', 'satellite', 'ukiyoe', 'vangogh', 'cezanne', 'monet', 'photo']

#datasets = ['ukiyoe', 'vangogh', 'cezanne']

#datasets = ['horse']
#datasets = ['horse+zebra --leave_one_out ', 'apple+orange --leave_one_out ',
#            'summer+winter --leave_one_out ', 'cityscapes --leave_one_out ', 
#            'satellite --leave_one_out ', 'facades --leave_one_out ', 
#            'fold6 --leave_one_out ', 'fold7 --leave_one_out ',
#            'fold8 --leave_one_out ', 'fold9 --leave_one_out ']
#datasets = ['horse+zebra --leave_one_out ']
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

