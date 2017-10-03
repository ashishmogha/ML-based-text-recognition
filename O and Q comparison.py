# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 05:54:13 2017

@author: admin
"""
import numpy as np

from skimage.feature import hog
from skimage import color, transform
from skimage.io import imread,imshow
from sklearn import svm
from sklearn.decomposition import PCA

import os

files_test = [f for f in os.listdir(os.curdir) if os.path.isfile(f)]
collection_test = np.array([color.rgb2gray(imread(im)) for im in files_test])

X_O = np.array([hog(collection_test[x], orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,500)]) 

X_Q = np.array([hog(collection_test[x], orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,500)])


y_O = [0]*500
y_Q = [1]*500


X = np.concatenate((X_O,X_Q))
Y = y_O + y_Q

pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)

import pylab as pl
for i in range(0, pca_2d.shape[0],2):
    if Y[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    s=50,marker='+')
    elif Y[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    s=50,marker='o')

        
pl.legend([c1, c2], ['O', 'Q'])
pl.axis('off')
pl.title('English Auto-generated font dataset for O and Q classes')
pl.show()


clf_2d =   svm.SVC(kernel='rbf').fit(pca_2d, Y)
for i in range(0, pca_2d.shape[0],2):
    if Y[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    s=50,marker='+')
    elif Y[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    s=50,marker='o')
pl.legend([c1, c2], ['O', 'Q'])
x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:,0].max() + 1
y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
Z = clf_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
Z = Z.reshape(xx.shape)
pl.contour(xx, yy, Z)
pl.title('Support Vector Machine Decision Surface using kernel : ' + clf_2d.kernel)
pl.axis('off')
pl.show()

image = color.rgb2gray(imread('img027-00909.png'))
image = transform.resize(image,(128,128))
fd,hog_image = hog(image, orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=True)

imshow(image)
imshow(hog_image)
