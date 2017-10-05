import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import svm
import os
import numpy as np

def getkey(n):
    if n <= 10:
        return chr(n + 47)
    elif n <= 36:
        return chr(n + 54)
    else:
        return chr(n + 60)
    
clf_fnt = joblib.load(os.path.join(os.getcwd(),"savedSVMs", "svmfnt.pkl"))
clf_hnd = joblib.load(os.path.join(os.getcwd(),"savedSVMs", "svmhnd.pkl"))

accuracy_fnt = {}

for i in range(0,(3100), 50):
    X = Fntdata[i : i+50]
    Y = Fnttarget[i : i+50]
    accuracy_fnt[getkey(Fnttarget[i])] = clf_fnt.score(X,Y)    
    
accuracy_hnd = {}

for i in range(0,(3410), 55):
    X = Hnddata[i : i+55]
    Y = Hndtarget[i : i+55]
    accuracy_hnd[getkey(Hndtarget[i])] = clf_hnd.score(X,Y)   
    
for x in range(11,63,13):
    objects = [getkey(f) for f in range(x,x+13)]
    y_pos = np.arange(len(objects))
    performance = [accuracy_hnd[f] for f in objects]
    
    plt.bar(y_pos, performance, align='center', alpha=1.0)
    plt.xticks(y_pos, objects)
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.title('Performance scores for %s-%s handwriting' %(getkey(x),getkey(x+12)))
    plt.show()
    
from sklearn.utils import shuffle

Fnt_X, Fnt_Y = shuffle(Fntdata, Fnttarget, random_state=0)   
Hnd_X, Hnd_Y = shuffle(Hnddata, Hndtarget, random_state=0)   
ovr_accuracy = [clf_fnt.score(Fnt_X,Fnt_Y), clf_hnd.score(Hnd_X, Hnd_Y)]

objects = ['Font', 'Handwriting']
y_pos = np.arange(len(objects))

plt.bar(y_pos, ovr_accuracy, align='center', alpha=1.0)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.xlabel('Class')
plt.title('Overall performance for computer fonts and handwritings')
plt.show()