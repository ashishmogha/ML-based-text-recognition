
import numpy as np

from skimage.feature import hog
from skimage import color, transform
from skimage.io import imread,imshow
from sklearn import svm
from sklearn.decomposition import PCA

import os


collection = {}
dataCwd = os.getcwd() + "/dataset/EnglishFnt"
i = 1
for f in os.listdir(dataCwd):
   if f != ".DS_Store" :
        sampleDir = dataCwd + "/" + f
        files = os.listdir(sampleDir)
        collection[i] = np.array([color.rgb2gray(imread(sampleDir + "/" + im)) for im in files])
        i = i + 1






'''
files = [f for f in os.listdir(os.curdir) if os.path.isfile(f)]
collection = np.array([color.rgb2gray(imread(im)) for im in files])


X_A = np.array([hog(collection[x], orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,500)]) 

X_B = np.array([hog(collection[x], orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,500)])

X_C = np.array([hog(collection[x], orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,500)])

X_D = np.array([hog(collection[x], orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,500)])

y_A = [0]*500
y_B = [1]*500
y_C = [2]*500
y_D = [3]*500

X = np.concatenate((X_A,X_B,X_C,X_D))
Y = y_A + y_B + y_C + y_D

clf = svm.SVC()
clf.fit(X,Y)

image = color.rgb2gray(imread('img014-00838.png'))
image = transform.resize(image,(128,128))
fd,hog_image = hog(image, orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=True)

imshow(image)
imshow(hog_image)

clf.predict(fd)


pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)

import pylab as pl
for i in range(0, pca_2d.shape[0],5):
    if Y[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    s=50,marker='+')
    elif Y[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    s=50,marker='o')
    elif Y[i] == 2:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    s=50,marker='*')
    elif Y[i] == 3:
        c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='y',    s=50,marker='d')
        
pl.legend([c1, c2, c3, c4], ['A', 'B', 'C', 'D'])
pl.axis('off')
pl.title('English Auto-generated font dataset with 3 classes and known outcomes')
pl.show()


clf_2d =   svm.SVC(kernel='sigmoid').fit(pca_2d, Y)
for i in range(0, pca_2d.shape[0],5):
    if Y[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    s=50,marker='+')
    elif Y[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    s=50,marker='o')
    elif Y[i] == 2:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    s=50,marker='*')
    elif Y[i] == 3:
        c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='y',    s=50,marker='d')
pl.legend([c1, c2, c3, c4], ['A', 'B', 'C', 'D'])
x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:,0].max() + 1
y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
Z = clf_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
Z = Z.reshape(xx.shape)
pl.contour(xx, yy, Z)
pl.title('Support Vector Machine Decision Surface using kernel : ' + clf_2d.kernel)
pl.axis('off')
pl.show()

pca_test = pca.transform(fd)
clf_2d.predict(pca_test)
'''