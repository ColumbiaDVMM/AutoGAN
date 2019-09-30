#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: Utils.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Thu Sep 19 13:04:52 2019
#
#  Usage: 
#  Description: A few useful functions 
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

import torch
import torch.nn.init
import torch.nn as nn
import cv2
import numpy as np
import random

image_size = 224

# resize image to size 32x32
cv2_scale = lambda x: cv2.resize(x, dsize=(image_size, image_size),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (image_size, image_size, 1))

np_reshape_color = lambda x: np.reshape(x, (image_size, image_size, 3))

centerCrop = lambda x: x#[8:56,8:56,:]
#centerCrop = lambda x: x[15:47,15:47,:]

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x, dim=1):
        norm = torch.sqrt(torch.sum(x * x, dim=dim) + self.eps)
        x= x / norm.unsqueeze(dim=dim).expand_as(x)
        return x


class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def get_cooccurance_matrix(img):
    cooccurance_matrix = np.zeros((256,256), dtype = np.float32)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]-1):
            cooccurance_matrix[img[i][j],img[i][j+1]] = cooccurance_matrix[img[i][j],img[i][j+1]] + 1
    cooccurance_matrix = (cooccurance_matrix - np.min(cooccurance_matrix))/(np.max(cooccurance_matrix) - np.min(cooccurance_matrix))
    return cooccurance_matrix
