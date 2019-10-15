#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: GAN_Detection_Train.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Tue Oct 15 16:41:30 2019
#
#  Usage: python GAN_Detection_Test.py -h
#  Description: Evaluate a GAN image detector
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
from GAN_Detection_Train import GANDataset
import torch.nn as nn
from collections import OrderedDict
import csv

from torchvision import transforms, models

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

# Training settings
parser.add_argument('--dataroot', type=str,
                    default='./datasets/',
                    help='path to dataset')
parser.add_argument('--training-set', default= 'horse',
                    help='The name of the training set. If leave_one_out flag is set, \
                    it is the leave-out set(use all other sets for training).')
parser.add_argument('--test-set', default='transposed_conv', type=str,
                    help='Choose test set from trainsposed_conv, nn, jpeg and resize')
parser.add_argument('--feature', default='image',
                    help='Feature used for training, choose from image and fft')
parser.add_argument('--mode', type=int, default=0, 
                    help='fft frequency band, 0: full, 1: low, 2: mid, 3: high')
parser.add_argument('--leave_one_out', action='store_true', default=False,
                    help='Test leave one out setting, using all other sets for training and test on a leave-out set.')
parser.add_argument('--jpg_level', type=str, default='90',
                    help='Test with different jpg compression effiecients, only effective when use jpg for test set.')
parser.add_argument('--resize_size', type=str, default='200', 
                    help='Test with different resize sizes, only effective when use resize for test set.')

parser.add_argument('--result-dir', default='./final_output/',
                    help='folder to output result in csv')
parser.add_argument('--model-dir', default='./model/',
                    help='folder to output model checkpoints')
parser.add_argument('--model', default='resnet',
                    help='Base classification model')
parser.add_argument('--num-workers', default= 1,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--resume', default='', type=str, 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, 
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32,
                    help='input batch size for testing (default: 32)')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='learning rate (default: 0.01)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--data_augment', action='store_true', default=False,
                    help='Use data augmentation or not')
parser.add_argument('--check_cached', action='store_true', default=True,
                    help='Use cached dataset or not')
parser.add_argument('--seed', type=int, default=-1, 
                    help='random seed (default: -1)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

suffix = '{}'.format(args.training_set)

if args.test_set == 'transposed_conv':
    dataset_names = ['horse', 'zebra', 'summer', 'winter', 'apple', 'orange',
                'facades', 'cityscapes', 'satellite', 
                'ukiyoe', 'vangogh', 'cezanne', 'monet', 'photo', 'celeba_stargan']  
    #dataset_names = ['horse+zebra', 'apple+orange', 'summer+winter', 'facades', 'cityscapes', 'satellite', 'fold6', 'fold7', 'fold8', 'fold9']
elif args.test_set == 'nn':
    dataset_names = ['horse_nn',  'zebra_nn', 'summer_nn', 'winter_nn', 'apple_nn', 'orange_nn', 'horse', 'zebra', 'summer', 'winter', 'apple', 'orange']
elif args.test_set == 'jpg':
    dataset_names = ['horse_jpg_{}'.format(args.jpg_level), 'zebra_jpg_{}'.format(args.jpg_level),
            'summer_jpg_{}'.format(args.jpg_level), 'winter_jpg_{}'.format(args.jpg_level),
            'apple_jpg_{}'.format(args.jpg_level), 'orange_jpg_{}'.format(args.jpg_level),
            'facades_jpg_{}'.format(args.jpg_level), 'cityscapes_jpg_{}'.format(args.jpg_level),
            'satellite_jpg_{}'.format(args.jpg_level), 'ukiyoe_jpg_{}'.format(args.jpg_level),
            'vangogh_jpg_{}'.format(args.jpg_level), 'cezanne_jpg_{}'.format(args.jpg_level),
            'monet_jpg_{}'.format(args.jpg_level), 'photo_jpg_{}'.format(args.jpg_level)]  
elif args.test_set == 'resize':
    dataset_names = ['horse_resize_{}'.format(args.resize_size), 'zebra_resize_{}'.format(args.resize_size),
            'summer_resize_{}'.format(args.resize_size), 'winter_resize_{}'.format(args.resize_size),
            'apple_resize_{}'.format(args.resize_size), 'orange_resize_{}'.format(args.resize_size),
            'facades_resize_{}'.format(args.resize_size), 'cityscapes_resize_{}'.format(args.resize_size),
            'satellite_resize_{}'.format(args.resize_size), 'ukiyoe_resize_{}'.format(args.resize_size),
            'vangogh_resize_{}'.format(args.resize_size), 'cezanne_resize_{}'.format(args.resize_size),
            'monet_resize_{}'.format(args.resize_size), 'photo_resize_{}'.format(args.resize_size)]  
else:
    print('Test set does not support!')
    exit(-1)
    

if args.leave_one_out:
    args.training_set = args.training_set.replace('_auto','')
    dataset_names = [args.training_set]

if args.data_augment:
    suffix = suffix + '_da'
if args.leave_one_out:
    suffix = suffix + '_oo'
if args.feature is not 'image':
    suffix = suffix + '_{}_{}'.format(args.feature, args.mode)

suffix = suffix + '_{}'.format(args.model)

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
    # set random seeds
    if args.seed>-1:
        torch.cuda.manual_seed_all(args.seed)

# set random seeds
if args.seed>-1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

try:
    os.stat('{}/'.format(args.result_dir))
except:
    os.makedirs('{}/'.format(args.result_dir))

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    print(test_dataset_names) 
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             GANDataset(train=args.leave_one_out,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     check_cached=args.check_cached,
                     transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders

def test(test_loader, model, epoch, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, predicts = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (image_pair, label) in pbar:

        if args.cuda:
            image_pair = image_pair.cuda()

        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)

        out = model(image_pair)
        _, pred = torch.max(out,1)
        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)
        predicts.append(pred)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    
    print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts)/float(num_tests)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))
    
    pos_label = labels[labels==1]
    pos_pred = predicts[labels==1]
    TPR = np.sum(pos_label == pos_pred)/float(pos_label.shape[0])
    print('\33[91mTest set: TPR: {:.8f}\n\33[0m'.format(TPR))

    neg_label = labels[labels==0]
    neg_pred = predicts[labels==0]
    TNR = np.sum(neg_label == neg_pred)/float(neg_label.shape[0])
    print('\33[91mTest set: TNR: {:.8f}\n\33[0m'.format(TNR))

    return acc

def main(test_loaders, model):
    print('\nparsed options:\n{}\n'.format(vars(args)))
    acc_list = []
    if args.cuda:
        model.cuda()

    if not args.leave_one_out:
        csv_file = csv.writer(open('{}/{}.csv'.format(args.result_dir, suffix), 'w'), delimiter=',')
        csv_file.writerow(dataset_names) 
    else:
        result_dict = OrderedDict()
        try:
            read_result_dict = load_csv('{}/leave_one_out.csv'.format(args.result_dir),',')
            read_result_dict = read_result_dict.to_dict()
            for key in read_result_dict:
                result_dict[key] = read_result_dict[key][0]
        except Exception as e:
            print(str(e))
        if result_dict is None:
            result_dict = OrderedDict()

    start = args.start_epoch
    end = start + args.epochs
    for test_loader in test_loaders:
        acc = test(test_loader['dataloader'], model, 0, test_loader['name'])*100
        acc_list.append(str(acc))
        if args.leave_one_out:
            result_dict[test_loader['name']] = acc
    
    #write csv file
    if not args.leave_one_out:
        csv_file.writerow(acc_list) 
    else:
        csv_file = csv.writer(open('{}/leave_one_out.csv'.format(args.result_dir), 'w'), delimiter=',')
        name_list = []
        acc_list = []
        for key in result_dict:
            name_list.append(key)
            acc_list.append(result_dict[key])
        csv_file.writerow(name_list)  
        csv_file.writerow(acc_list) 
    
        
if __name__ == '__main__':
    if args.model == 'resnet':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.model == 'pggan':
        model = pggan_dnet.SimpleDiscriminator(3, label_size=1, mbstat_avg='all', 
                resolution=256, fmap_max=128, fmap_base=2048, sigmoid_at_end=False)
    elif args.model == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

    print('{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,args.epochs))
    load_model = torch.load('{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,args.epochs))
    model.load_state_dict(load_model['state_dict'])

    test_loaders = create_loaders()
    main(test_loaders, model)
