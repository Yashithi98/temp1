import glob as glob
from scipy.misc import imread 
from scipy.misc import imsave
from scipy.misc import toimage
from scipy.misc import imresize
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2

import os

datasetA ='dataset/testB1/*.png'
datasetB  = 'dataset/trainA2/*.JPEG'
destination = 'dataset/testB/'

dataA = glob.glob(datasetA)
dataA = sorted(dataA)
dataB = glob.glob(datasetB)
dataB = sorted(dataB)
count = 0

for i,j in zip(dataA,dataB):
 print(i)
 count+=1
 if (count>5000):
   break;
 img_A = imread(i)
 dim = (64, 64)
 out = cv2.resize(img_A, dim, interpolation = cv2.INTER_AREA)
 toimage(out, cmin=0, cmax=255).save(destination+str(count)+'.png')



