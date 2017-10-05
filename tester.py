import numpy as np

from skimage.feature import hog
from skimage import color, transform
from skimage.io import imread,imshow
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib

import os


def getkey(n):
    if n <= 10:
        return chr(n + 47)
    elif n <= 36:
        return chr(n + 54)
    else:
        return chr(n + 60)

data_YFnt = {}

for f in range(1,63):
    data_YFnt[f] = [str(getkey(f))] * 20



YFnt = []
for f in range(1,63):
    YFnt += data_YFnt[f]
    

data_YHnd = {}

for f in range(1,63):
    data_YHnd[f] = [str(getkey(f))] * 25



YHnd = []
for f in range(1,63):
    YHnd += data_YHnd[f]
    
    
clfFnt = joblib.load(os.path.join(os.getcwd(),"savedSVMs", "svmfnt.pkl"))
clfHnd = joblib.load(os.path.join(os.getcwd(),"savedSVMs", "svmhnd.pkl"))

clfFnt.score(Fnt,YFnt)
clfHnd.score(Hnd,YHnd)