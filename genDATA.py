import glob as glob
from scipy.misc import imread 
from scipy.misc import imsave
from scipy.misc import toimage
from scipy.misc import imresize
from scipy import ndimage

import numpy as np
import scipy as sp
import cv2

import os

datasetA ='dataset/trainA1/*.JPEG'
datasetB  = 'dataset/trainA2/*.JPEG'
destination = 'dataset/trainA/'

#datasetA ='dataset/testA1/*.JPEG'
#datasetB  = 'dataset/testA2/*.JPEG'
#destination = 'dataset/testA/'

dataA = glob.glob(datasetA)

dataA = sorted(dataA)
dataB = glob.glob(datasetB)
dataB = sorted(dataB)

count = 0

for i,j in zip(dataA,dataB):
  
 print("hi")
 count+=1
 if (count>4000):
   break;
   
 img_A = cv2.imread(j,0)
 img_B = cv2.imread(i,0)
 dim = (32, 32)
 img_A = cv2.resize(img_A, dim, interpolation = cv2.INTER_AREA)
 img_B = cv2.resize(img_B, dim, interpolation = cv2.INTER_AREA)
 
 line = img_A
 #out = np.zeros((np.shape(img_B)[0],np.shape(img_B)[1]*2,3))
 out = cv2.hconcat([img_A,img_B]) 
 #out[:,0:np.shape(img_B)[1],0] = img_B
 #out[:,0:np.shape(img_B)[1],1] = img_B
 #out[:,0:np.shape(img_B)[1],2] = img_B
 #out[:,np.shape(img_B)[1]:2*np.shape(img_B)[1],0] = line
 #out[:,np.shape(img_B)[1]:2*np.shape(img_B)[1],1] = line
 #out[:,np.shape(img_B)[1]:2*np.shape(img_B)[1],2] = line
 print(out.shape)
 toimage(out, cmin=0, cmax=63).save(destination+str(count)+'.jpeg')



