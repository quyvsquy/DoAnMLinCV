import numpy as np
from os import listdir
import os
from os.path import isfile, join, isdir
from sklearn.model_selection import KFold
import sys
import pandas as pd
from numpy.random import shuffle

# pathFolder = "../DataSet/FaceData/processed/"
def timFolderName(pathFolder):
    listFolderNameMSSV = [f for f in listdir(pathFolder) if isdir(join(pathFolder, f))]
    listFolderNameMSSV.sort()
    return listFolderNameMSSV

def tachTestTrain(pathFolder):
    print("Create New DataSet")
    listFolderNameMSSV = timFolderName(pathFolder)
    dataX = []
    # dataY = []
    # dataName = []
    for ia in listFolderNameMSSV:
        for ib in listdir(join(pathFolder, ia)):
            if isfile(join(pathFolder, ia, ib)):
                dataX.append(join(pathFolder, ia, ib))
                # dataName.append(ia)
    shuffle(dataX)
    # for ia in dataX:
    #     dataY.append(int(ia.split('/')[4].split('_')[0]))
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(dataX)

    tempDictXtrain = {}
    tempDictYtrain = {}
    tempDictXtest = {}
    tempDictYtest = {}
    # tempNtrain = []
    # tempNtest = []
    intDem = 0
    for train_index, test_index in kf.split(dataX):
        y_train = []
        y_test = []

        X_train, X_test = np.array(dataX)[train_index], np.array(dataX)[test_index]
        shuffle(X_train)
        shuffle(X_test)

        for ia in X_train:
            y_train.append(int(ia.split('/')[4].split('_')[0]))
        for ia in X_test:
            y_test.append(int(ia.split('/')[4].split('_')[0]))
        # N_train, N_test = np.array(dataName)[train_index], np.array(dataName)[test_index]

        

        # Luu lai
        tempDictXtrain[intDem] = X_train
        tempDictYtrain[intDem] = y_train
        tempDictXtest[intDem] = X_test
        tempDictYtest[intDem] = y_test
        intDem += 1
        
        # tempNtrain.append(N_train.tolist())
        # tempNtest.append(N_test.tolist())
        
        # print("TRAIN:", train_index, "TEST:", test_index)
        # print("TRAIN_len:", len(train_index), "TEST_len:", len(test_index))
    dfXtrain = pd.DataFrame(tempDictXtrain)
    dfYtrain = pd.DataFrame(tempDictYtrain)
    dfXtest  = pd.DataFrame(tempDictXtest )
    dfYtest  = pd.DataFrame(tempDictYtest )
    if not os.path.exists("./tempLuu"):
        os.makedirs("./tempLuu")
    dfXtrain.to_csv('./tempLuu/tempDictXtrain.csv', index=None) 
    dfYtrain.to_csv('./tempLuu/tempDictYtrain.csv', index=None) 
    dfXtest .to_csv('./tempLuu/tempDictXtest.csv', index=None) 
    dfYtest .to_csv('./tempLuu/tempDictYtest.csv', index=None) 
    
# tachTestTrain(sys.argv[1])

def loadDuLieu(path, createNewDataSet = False):
    if createNewDataSet:
        tachTestTrain(path)        
    # print(path)
    # listFolderNameMSSV = [f for f in listdir(path) if isfile(join(path,f))]
    # listFolderNameMSSV.sort()
    # print(listFolderNameMSSV)

    dfXtrain = pd.read_csv('./tempLuu/tempDictXtrain.csv')
    dfYtrain = pd.read_csv('./tempLuu/tempDictYtrain.csv')
    dfXtest  = pd.read_csv('./tempLuu/tempDictXtest.csv')
    dfYtest  = pd.read_csv('./tempLuu/tempDictYtest.csv')

    tempXtrain = []
    tempYtrain = []
    tempXtest = []
    tempYtest = []


    for ia in range(10):
        tempXtrain.append(list(dfXtrain[str(ia)]))
        tempYtrain.append(list(dfYtrain[str(ia)]))
        tempXtest.append(list( dfXtest [str(ia)]))
        tempYtest.append(list( dfYtest [str(ia)]))

    return tempXtrain, tempYtrain, tempXtest, tempYtest

# loadDuLieu(sys.argv[1], True)
# tachTestTrain(sys.argv[1])
# print(b)
# print(tempXtrain)
# print("\n=============================\n")
# print(tempYtrain)
# print("\n=============================\n")
# print(tempXtest)
# print("\n=============================\n")
# print(tempYtest)

# a = np.array(["33dd", "dg", "rs"])
# a = np.char.add(a, "999999")
# print(a)

