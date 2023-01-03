#from matplotlib import pyplot as plt
import cv2
import numpy as np
import imageio as iio
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
import pandas as pd
import array as arr
import logging
import random
import csv
from libsvm.svmutil import *
#from grid import *

from numpy import asarray
from numpy import savetxt
from sklearn.metrics import *


# 1- renames images in all folders

def change_name():
    folder = "C:/Users/Nagham/Desktop/SMVProject/Tutorial/test/truck"
# ************this code for rename images in a sepcific folder
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"img{str(count)}.jpg"
        # foldername/filename, if .py file is outside folder
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"
        # rename() function will
        # rename all the files
        os.rename(src, dst)
# *********************************************************************************


# convert all the images in train folder into histogram
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
labels = []
TrainData = []
RGBdata = []
df = []

print("Extract train features \n")
logger.info("***** Extract train features *****")
num = 4
folderno = 4
nbins = 16
while(folderno != 10):
    folder = "/var/home/nhamad/SMVProject/Tutorial/train/{}".format(folderno)
    os.chdir(folder)

    for i in os.listdir(folder):
        img = cv2.imread(i)  # read image
        hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to hsv

        blue_color = cv2.calcHist([img], [0], None, [256], [0, 256])
        red_color = cv2.calcHist([img], [1], None, [256], [0, 256])
        green_color = cv2.calcHist([img], [2], None, [256], [0, 256])

        blue_color = blue_color.reshape(1, 256)
        red_color = red_color.reshape(1, 256)
        green_color = green_color.reshape(1, 256)
        totalrgb = np.concatenate(
            (blue_color, red_color, green_color), axis=None)

        RGBdata.append(totalrgb)

        # convert inage to histograms with bin=16
        histo1 = cv2.calcHist([hsvImage], [0], None, [16], [0, 256])
        histo2 = cv2.calcHist([hsvImage], [1], None, [16], [0, 256])
        histo3 = cv2.calcHist([hsvImage], [2], None, [16], [0, 256])
        histo1 = histo1.reshape(1, 16)
        histo2 = histo2.reshape(1, 16)
        histo3 = histo3.reshape(1, 16)

        totalHisto = np.concatenate((histo1, histo2, histo3), axis=None)
        # add 48 features for the first image
        #TrainData.append(np.hstack((histo1, histo2, histo3)))
        TrainData.append(totalHisto)
        labels.append(folderno)
        # if(num==3000):
        #   break
    folderno = folderno+1
    # num=num+1
logger.info("\n***** finish get features of train *****")

print("finish train \n")
print("test \n")
logger.info("\n***** Extract Test features *****")


TrainData = np.array(TrainData)
RGBdata = np.array(RGBdata)
labels = np.array(labels)
df = pd.DataFrame(TrainData)
df['class'] = labels


TestData = []
folderno = 4
nbins = 16
Testlabels = []

num = 4
while(folderno != 10):
    folder = "/var/home/nhamad/SMVProject/Tutorial/test/{}".format(folderno)
    os.chdir(folder)

    for i in os.listdir(folder):
        img = cv2.imread(i)  # read image
        hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to hsv

        # convert inage to histograms with bin=16
        histo1 = cv2.calcHist([hsvImage], [0], None, [16], [0, 256])
        histo2 = cv2.calcHist([hsvImage], [1], None, [16], [0, 256])
        histo3 = cv2.calcHist([hsvImage], [2], None, [16], [0, 256])
        histo1 = histo1.reshape(1, 16)
        histo2 = histo2.reshape(1, 16)
        histo3 = histo3.reshape(1, 16)

        totalHisto = np.concatenate((histo1, histo2, histo3), axis=None)
        # add 48 features for the first image
        #TrainData.append(np.hstack((histo1, histo2, histo3)))
        TestData.append(totalHisto)
        Testlabels.append(folderno)
       # if(num==3000):
        #    break
    folderno = folderno+1
    # num=num+1

TestData = np.array(TestData)
Testlabels = np.array(Testlabels)
df1 = pd.DataFrame(TestData)
df1['class'] = Testlabels

logger.info("\n***** finish get features of test *****")


# ******* here to get the max and minimum values in train features
logger.info(np.min(TrainData))
logger.info("\n")
logger.info(np.max(TrainData))
logger.info("\n")

# TrainData: training features (28770 * 48)
#labels: (28770 * 1)

#TestData : (6000*48)
#Testlabels :(6000*1)

# *****************************save libsvm format************************************

""" folder = "/var/home/nhamad/SMVProject/Tutorial/libsvm"
os.chdir(folder) 
c=0
r=0
classl=4
f = open("TrainData.txt", "a")
j=0

dd=0
for i in TrainData:
    f.write(str(labels[dd]))
    while(c<=47):
        f.write(" ")
        f.write(str(c+1))
        f.write(":")
        f.write(str(TrainData[r,c]))
        f.write(" ")
        c=c+1
    f.write("\n")    
    dd=dd+1    
    r=r+1
    c=0
f.close()


folder = "/var/home/nhamad/SMVProject/Tutorial/libsvm"
os.chdir(folder)
c=0
r=0
classl=4
f = open("TestData.txt", "a")
j=0

dd=0
for i in TestData:
    f.write(str(Testlabels[dd]))
    while(c<=47):
        f.write(" ")
        f.write(str(c+1))
        f.write(":")
        f.write(str(TestData[r,c]))
        f.write(" ")
        c=c+1
    f.write("\n")    
    dd=dd+1    
    r=r+1
    c=0
f.close() """

# scale train data using :     ./svm-scale -l  0 -u 1 -s range1 TrainData > TrainData.scale
#                             

# scale test data using :       ./svm-scale -r range1 TestData > TestData.scale
#                                ./svm-scale -l  0 -u 1 -s range1 TestData > TestData.scale

#    ./svm-train -t 5 TrainData.scale
#     ./svm-predict TestData.scale TrainData.scale.model TestData.predict


# ****************** start knn *************************************

""" knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
  
# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
# fitting the model for grid search
grid_search=grid.fit(TrainData, labels)

print(grid_search.best_params_)

accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) ) """


print("knn start \n")
logger.info("\n***** start KNN *****")

k = 24
print("k={}".format(k))
knn = KNeighborsClassifier(n_neighbors=k)
print("start train \n")
logger.info("\n***** start train KNN *****")

knn.fit(TrainData, labels)
print("start predict \n")
logger.info("\n***** start predict KNN *****")

predicted = knn.predict(TestData)

acc = accuracy_score(Testlabels, predicted)
logger.info("\n***** Accuracy of KNN= {}".format(acc))

print("Accuracy of KNN:", acc)

print(classification_report(Testlabels,  predicted))

print(confusion_matrix(Testlabels,  predicted))

print(f1_score(Testlabels,  predicted, average='macro'))

#**************************************************************************************
# obtained TestData.predict lables
filename = "/var/home/nhamad/SMVProject/Tutorial/libsvm/TestData.predict"
f = open("/var/home/nhamad/SMVProject/Tutorial/libsvm/TestData.predict", "r")

#print(f.read())

with open(filename, 'r') as f:
   # next(f) # discard the first line
    p_label = [int(line) for line in f]

p_label = np.array(p_label)

print(classification_report(Testlabels,  p_label))

print(confusion_matrix(Testlabels,  p_label))

print(f1_score(Testlabels,  p_label,average='macro'))


# *********************************************************
# Training on different SVM Kernels before grid search
# **************************SVM with Linear kernel***************************


#m = svm_train(labels,normalizedTrainData, '-t 0 -h 0')
#p_label, p_acc, p_val = svm_predict(Testlabels, normalizedTestData, m)
#print("\n Accuracy of Linear:",p_acc)

# ****************************SVM with POLY d=2 kernel

#m = svm_train(labels,normalizedTrainData, '-t 1 -d 2 -h 0')
#p_label, p_acc, p_val = svm_predict(Testlabels, normalizedTestData, m)

#print("\n Accuracy of POLY:",p_acc)


# ****************************SVM with RBF kernel

#m = svm_train(labels,normalizedTrainData, '-t 2 -h 0')
#p_label, p_acc, p_val = svm_predict(Testlabels, normalizedTestData, m)
#print("\n Accuracy of RBF:",p_acc)



# ************************************************************************************************************

#m = svm_train(labels,normalizedTrainData, '-t 5 -h 0')
#p_label, p_acc, p_val = svm_predict(Testlabels, normalizedTestData, m)
#print("\n Accuracy of chi:",p_acc)


# y_true: Testlabels
# y_pred : p_label


# ***************************** start grid search
#rate, param = find_parameters('/var/home/nhamad/SMVProject/libsvm/tools/train', '-log2c -5,5,1 -log2g -5,5,1')

