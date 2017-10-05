
import numpy as np

from skimage.feature import hog
from skimage import color, transform
from skimage.io import imread,imshow
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import os


def getkey(n):
    if n <= 10:
        return chr(n + 47)
    elif n <= 36:
        return chr(n + 54)
    else:
        return chr(n + 60)


dataCwd = os.getcwd() + "/dataset/EnglishFnt"
sampleDir = dataCwd + "/" + "Sample001"
files = os.listdir(sampleDir)
collection = np.array([color.rgb2gray(imread(sampleDir + "/" + im)) for im in files])
data_X = np.array([hog(collection[x], orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,150)])

for f in os.listdir(dataCwd):
   if f != ".DS_Store":
       if f != "Sample001":
           sampleDir = dataCwd + "/" + f
           files = os.listdir(sampleDir)
           collection = np.array([color.rgb2gray(imread(sampleDir + "/" + im)) for im in files])
           data_X = np.concatenate((data_X,np.array([hog(collection[x], orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in range(0,150)])))

del collection
del files,f

Y = []
for f in range(1,63):
    Y += [f] * 150

data_Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.4, random_state=0)

clf_rbf = svm.SVC()
clf_rbf.fit(x_train,y_train)
clf_rbf.score(x_test,y_test)

clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(x_train,y_train)
clf_linear.score(x_test,y_test)


joblib.dump(clf_linear, os.path.join(os.getcwd(),"savedSVMs", "svmfnt.pkl"))


image = color.rgb2gray(imread(dataCwd + "/Sample035" + "/img035-00271.png"))
image = transform.resize(image,(128,128))
fd,hog_image = hog(image, orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=True)

imshow(image)
imshow(hog_image)

clf_rbf.predict(fd.reshape(1,-1))
clf_linear.predict(fd.reshape(1,-1))

#newclf = joblib.load(os.path.join(os.getcwd(),"savedSVMs", "svmfnt.pkl"))
#newclf.predict(fd)


'''
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