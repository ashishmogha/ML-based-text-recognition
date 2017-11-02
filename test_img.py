
import numpy as np
import os
from skimage import color, transform
from skimage.io import imread,imshow,imsave
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import hog
from skimage.filters import threshold_otsu

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

def preprocess(img):
    img_gs = color.rgb2gray(imread(img))
    
    denoised_img_gs = denoise_tv_chambolle(img_gs, weight=0.01)

    img_th = threshold_otsu(denoised_img_gs)
    binary = img_gs <= img_th
    
    denoised = denoise_tv_chambolle(binary,weight = 0.8)
    
    return denoised

dataCwd = os.getcwd() + "/dataset/EnglishImg/GoodImg"
sampleDir = dataCwd + "/Sample001"
files = os.listdir(sampleDir)
files = [f for f in files if f != ".DS_Store"]

collection = np.array([preprocess(sampleDir + "/" + im) for im in files])
data_X = np.array([hog(transform.resize(x,(128,128), mode='reflect'), orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in collection])
data_Y = [1] * 105
x = 2
for f in os.listdir(dataCwd):
   if f != ".DS_Store":
       if f != "Sample001":
           sampleDir = dataCwd + "/" + f
           files = os.listdir(sampleDir)
           files = [i for i in files if i != ".DS_Store"]
           collection = np.array([preprocess(sampleDir + "/" + im) for im in files])
           data_X = np.concatenate((data_X,np.array([hog(transform.resize(x,(128,128), mode='reflect'), orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=False) for x in collection])))
           i = 0
           for k in files:
                i = i + 1
           data_Y = data_Y + ([x]*i)
           x = x + 1

data_Y = np.array(data_Y)

x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(x_train,y_train)
clf_linear.score(x_test,y_test)
  
joblib.dump(clf_linear, os.path.join(os.getcwd(),"savedSVMs", "svmimg-31-10-17.pkl"))

  
'''    
for f in files:
    img = sampleDir + "/" + f
    
    preprocessed = preprocess(img)

    hog_img, imghog = hog(transform.resize(preprocessed,(128,128), mode='reflect'), orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=True)
'''    


'''
img = imread(img)
img_gs = color.rgb2gray(img)

denoised_img = denoise_tv_chambolle(img, weight=0.01,multichannel = True)
denoised_img_gs = denoise_tv_chambolle(img_gs, weight=0.01)

img_th = threshold_otsu(denoised_img_gs)
binary = img_gs > img_th

hog_img, imghog = hog(transform.resize(binary,(128,128), mode='reflect'), orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=True)

denoised = denoise_tv_chambolle(binary,weight = 0.8)

hog_img, imghog = hog(transform.resize(denoised,(128,128), mode='reflect'), orientations=9, block_norm='L2-Hys', pixels_per_cell=(12, 12),
                    cells_per_block=(2, 2), visualise=True)
'''