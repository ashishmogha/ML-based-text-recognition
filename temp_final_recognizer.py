#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:56:55 2017

@author: shivam
"""
import numpy as np
import os

from skimage.feature import hog
from skimage import color, transform
from skimage.io import imread,imshow
from sklearn import svm

from sklearn.externals import joblib

def getkey(n):
    if n <= 10:
        return chr(n + 47)
    elif n <= 36:
        return chr(n + 54)
    else:
        return chr(n + 60)

classifier = joblib.load(os.getcwd() + "/savedSVMs/svmfnt-30-10-17.pkl")

img = transform.resize(x,(128,128), mode='reflect') 

feature, hog_img = hog(img, orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=True)

fd = feature.reshape(1,-1)
fd1 = feature.reshape(-1,1)

getkey(classifier.predict(fd))