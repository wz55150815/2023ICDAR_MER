#!/usr/bin/env python

import os
import glob
from tqdm import tqdm
import cv2
import pickle as pkl


image_path = 'train_set_images'
image_out = 'train_image.pkl'
label_path = 'train_set_hyb'
label_out = 'train_label.pkl'

images = glob.glob(os.path.join(image_path, '*.jpg'))
image_dict = {}

for item in tqdm(images):

    img = cv2.imread(item)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_dict[os.path.basename(item).replace('.jpg', '')] = img

with open(image_out, 'wb') as f:
    pkl.dump(image_dict, f)

labels = glob.glob(os.path.join(label_path, '*.txt'))
label_dict = {}

for item in tqdm(labels):
    with open(item) as f:
        lines = f.readlines()
    label_dict[os.path.basename(item).replace('.txt', '')] = lines

with open(label_out, 'wb') as f:
    pkl.dump(label_dict, f)

