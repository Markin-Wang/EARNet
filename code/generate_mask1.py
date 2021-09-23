import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
from skimage.color import rgb2gray
from collections import Counter
import skimage
from PIL import Image

np.set_printoptions(threshold=np.inf)

# background:255 red: 1 blue:2 yellow:3

root = '/Users/wangjun/Documents/study/paper/2020/accv/DeepLabv3.pytorch-master/data/leafvein/test/labels'
#target_root = '/Users/wangjun/Documents/study/paper/2020/accv/DeepLabv3.pytorch-master/data/leafvein/test/labelss'
for k in range(1, 2):
    img = imageio.imread(''.join([root, os.sep, str(k), '.png']))
    img2 = Image.open(''.join([root, os.sep, str(k), '.png']))
    '''
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                mask[i, j] = 0
            elif img[i, j] == 2:
                mask[i, j] = 1
            elif img[i, j] == 3:
                mask[i, j] = 2
            else:
                mask[i, j] = img[i, j]
    imageio.imsave(''.join([target_root, os.sep, str(k), '.png']), mask)
    '''
    print(np.array(img2, dtype=np.uint8))
    print((np.array(img2, dtype=np.uint8) == 0).sum())
