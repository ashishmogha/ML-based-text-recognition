import numpy as np
import os

from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

data_X = np.concatenate((fntdata_X,hnddata_X,imgdata_X))
data_Y = np.concatenate((fntdata_Y,hnddata_Y,imgdata_Y))

x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

clf_linear = svm.SVC(kernel = 'linear', C = 1)
clf_linear.fit(x_train, y_train)
clf_linear.score(x_test,y_test)

joblib.dump(clf_linear, os.path.join(os.getcwd(),"savedSVMs", "svmall-31-10-17.pkl"))

