#!/usr/bin/env python

import os
import sys
import pickle as pkl
from pathlib import Path

import numpy
# from scipy.misc import imread, imresize, imsave
from imageio import imread

image_path = '../data_comer/2016/img'
# image_path='./off_image_test/' for test.pkl
# outFile = 'offline-train.pkl'
outFile = "train_2016"
# outFile='offline-test.pkl'

features = {}

channels = 1

sentNum = 0

scpFile = open(image_path[:-3] + "caption.txt")
# scpFile=open('test_caption.txt')
while True:
    line = scpFile.readline().strip()  # remove the '\r\n'
    if not line:
        break
    else:
        key = line.split('\t')[0]
        # image_file = image_path + key + '_' + str(0) + '.bmp'
        image_file = Path(image_path) / (key + '.bmp')
        im = imread(image_file)
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
        for channel in range(channels):
            # image_file = image_path + key + '_' + str(channel) + '.bmp'
            im = imread(image_file)
            mat[channel, :, :] = im
        sentNum = sentNum + 1
        features[key] = mat
        if sentNum / 500 == sentNum * 1.0 / 500:
            print('process sentences ', sentNum)

print('load images done. sentence number ', sentNum)

with open(outFile, "wb") as f:
    pkl.dump(features, f)
print('save file done')

