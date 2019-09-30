#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: GAN_Detection_Train.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Sun Sep 29 22:20:13 2019
#
#  Usage: python GAN_Detection_Train.py -h
#  Description: Train a GAN image detector
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
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import cycleGAN_dataset
import torch.nn as nn

from torchvision import transforms, models
import pggan_dnet
from skimage.feature import greycomatrix

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

parser.add_argument('--enable-logging',type=bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='./log/',
                    help='folder to output log')
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
parser.add_argument('--lr-decay', default=1e-2, type=float, 
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
parser.add_argument('--data_augment', action='store_true', default=False,
                    help='Use data augmentation or not')
parser.add_argument('--check_cached', action='store_true', default=True,
                    help='Use cached dataset or not')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (default: -1)')
parser.add_argument('--interval', type=int, default=5,
                    help='logging interval, epoch based. (default: 5)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

suffix = '{}'.format(args.training_set)

if args.data_augment:
    suffix = suffix + '_da'
if args.leave_one_out:
    suffix = suffix + '_oo'
if args.feature is not 'image':
    suffix = suffix + '_{}_{}'.format(args.feature, args.mode)

suffix = suffix + '_{}'.format(args.model)

if args.test_set == 'transposed_conv':
    #Use a small set to save the inferring time. Use the best model to test all the subsets in test phase. 
    dataset_names = ['horse', 'winter'] 
    #dataset_names = ['horse', 'zebra', 'summer', 'winter', 'apple', 'orange',
    #            'facades', 'cityscapes', 'satellite', 
    #            'ukiyoe', 'vangogh', 'cezanne', 'monet', 'photo', 'celeba_stargan']  

if args.test_set == 'nn':
    dataset_names = ['horse_nn',  'zebra_nn', 'summer_nn', 'winter_nn', 'apple_nn', 'orange_nn']

elif args.test_set == 'jpg':
    dataset_names = ['horse_jpg_{}'.format(args.jpg_level), 'zebra_jpg_{}'.format(args.jpg_level),
            'summer_jpg_{}'.format(args.jpg_level), 'winter_jpg_{}'.format(args.jpg_level),
            'apple_jpg_{}'.format(args.jpg_level), 'orange_jpg_{}'.format(args.jpg_level)]  

elif args.test_set == 'resize':
    dataset_names = ['horse_resize_{}'.format(args.resize_size), 'zebra_resize_{}'.format(args.resize_size),
            'summer_resize_{}'.format(args.resize_size), 'winter_resize_{}'.format(args.resize_size),
            'apple_resize_{}'.format(args.resize_size), 'orange_resize_{}'.format(args.resize_size)]  

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

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)


class GANDataset(cycleGAN_dataset.cycleGAN_dataset):
    """
    GANDataset to read images.  
    """
    def __init__(self, train=True, transform=None, batch_size = None, *arg, **kw):
        super(GANDataset, self).__init__(train=train, *arg, **kw)
        self.transform = transform
        self.train = train
        self.batch_size = batch_size
    
    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img
        
        img = self.data[index]
        label = self.labels[index]

        if self.train:
            #data augmentation for training
            if args.data_augment:
                if args.model == 'resnet' or args.model == 'densenet' or args.model == 'googlenet':
                    random_x = random.randint(0,32)
                    random_y = random.randint(0,32)
                    im = deepcopy(img.numpy()[random_y:(random_y+224),\
                            random_x:(random_x+224),:])
                elif args.model == 'pggan':
                    im = deepcopy(img.numpy())
            else:
                if args.model == 'resnet' or args.model == 'densenet' or args.model == 'googlenet':
                    im = deepcopy(img.numpy()[16:240,16:240,:])
                elif args.model == 'pggan':
                    im = deepcopy(img.numpy())
        #centre crop for test
        else:
            if args.model == 'resnet' or args.model == 'densenet' or args.model == 'googlenet':
                im = deepcopy(img.numpy()[16:240,16:240,:])
            elif args.model == 'pggan':
                im = deepcopy(img.numpy())

        #use spectrum
        if args.feature == 'fft':
            im = im.astype(np.float32)
            im = im/255.0
            for i in range(3):
                img = im[:,:,i]
                fft_img = np.fft.fft2(img)
                fft_img = np.log(np.abs(fft_img)+1e-3)
                fft_min = np.percentile(fft_img,5)
                fft_max = np.percentile(fft_img,95)
                fft_img = (fft_img - fft_min)/(fft_max - fft_min)
                fft_img = (fft_img-0.5)*2
                fft_img[fft_img<-1] = -1
                fft_img[fft_img>1] = 1
                #set mid and high freq to 0
                if args.mode>0:
                    fft_img = np.fft.fftshift(fft_img)
                    if args.mode == 1:
                        fft_img[:57, :] = 0
                        fft_img[:, :57] = 0
                        fft_img[177:, :] = 0
                        fft_img[:, 177:] = 0
                    #set low and high freq to 0
                    elif args.mode == 2:
                        fft_img[:21, :] = 0
                        fft_img[:, :21] = 0
                        fft_img[203:, :] = 0
                        fft_img[:, 203:] = 0
                        fft_img[57:177, 57:177] = 0
                    #set low and mid freq to 0
                    elif args.mode == 3:
                        fft_img[21:203, 21:203] = 0
                    fft_img = np.fft.fftshift(fft_img)
                im[:,:,i] = fft_img
        else:
            im = im.astype(np.float32)
            im = (im/255 - 0.5)*2
            #img = transform_img(img)
        im = np.transpose(im, (2,0,1))
        return (im, label)

    def __len__(self):
        return self.labels.size(0)

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    train_loader = torch.utils.data.DataLoader(
            GANDataset(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             check_cached=args.check_cached,
                             leave_one_out = args.leave_one_out,
                             transform=transform),
                             batch_size=args.batch_size,
                             shuffle=True, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             GANDataset(train=False,
                     leave_one_out = False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     check_cached=args.check_cached,
                     transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return train_loader, test_loaders

def train(train_loader, model, optimizer, criterion,  epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        image_pair, label = data

        if args.cuda:
            image_pair, label  = image_pair.cuda(), label.cuda()
            image_pair, label = Variable(image_pair), Variable(label)
            out= model(image_pair)

        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    adjust_learning_rate(optimizer)

    if (args.enable_logging):
        logger.log_value('loss', loss.data.item()).step()

    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir,suffix))

    if ((epoch+1)%10)==0:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir, suffix, epoch+1))

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode, caculate test accuracy
    model.eval()

    labels, predicts = [], []
    outputs = []
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
        out = out.data.cpu().numpy().reshape(-1, 2)
        labels.append(ll)
        predicts.append(pred)
        outputs.append(out)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    outputs = np.vstack(outputs).reshape(num_tests,2)

    print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts)/float(num_tests)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))
    
    if (args.enable_logging):
        logger.log_value(logger_test_name+' Acc', acc)
    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        #group['lr'] = args.lr*((1-args.lr_decay)**group['step'])
        group['lr'] = args.lr
        
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loaders, model, logger):
    print('\nparsed options:\n{}\n'.format(vars(args)))

    optimizer1 = create_optimizer(model, args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda()
        criterion.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    start = args.start_epoch
    end = start + args.epochs
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, 0, logger, test_loader['name'])
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, criterion, epoch, logger)
        if ((epoch+1)%5)==0:
            for test_loader in test_loaders:
                test(test_loader['dataloader'], model, epoch+1, logger, test_loader['name'])
        
if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = args.log_dir + suffix
    logger, file_logger = None, None

    pretrain_flag = not args.feature=='comatrix'
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

    if(args.enable_logging):
        from Loggers import Logger
        logger = Logger(LOG_DIR)
    train_loader, test_loaders = create_loaders()
    main(train_loader, test_loaders, model, logger)
