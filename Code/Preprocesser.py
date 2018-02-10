import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os.path 

def ComputeMovingAverage(data,column,lag,add_feature = True):
    ma = data[column].rolling(window = lag).mean()
    if add_feature == True:
        data["ma_"+str(lag)+'_'+column] = ma
    else:
        return(ma)

def ComputeMovingStd(data,column,lag,add_feature = True):
    ma = data[column].rolling(window = lag).std()
    if add_feature == True:
        data["ma_"+str(lag)+'_'+column] = ma
    else:
        return(ma)

class Preprocesser():

    def __init__(self,file,time_shift):
        self.file = file
        self.time_shift = time_shift
        self.time_shift_in_s = None
        self.agg = None
        self.data = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.y_train_no_scale = None
        self.y_test_no_scale = None

    def ReadData(self):
        data = pd.read_csv(self.file)
        finds = re.findall(r'\d',self.file)
        str_agg = ""
        for x in finds:
            str_agg += x
        agg = int(str_agg)
        self.agg = agg
        self.time_shift_in_s = self.agg * self.time_shift

        # clean data
        data.index = pd.to_datetime(data["datetime"])
        del data["datetime"]
        self.data = data.dropna()

    def EngineerFeatures(self):

        data = self.data

        ComputeMovingAverage(data,'bidprice',8)
        ComputeMovingAverage(data,'bidprice',64)
        ComputeMovingAverage(data,'bidprice',256)
        ComputeMovingAverage(data,'bidprice',512)

        ComputeMovingAverage(data,'askprice',8)
        ComputeMovingAverage(data,'askprice',64)
        ComputeMovingAverage(data,'askprice',256)
        ComputeMovingAverage(data,'askprice',512)

        ComputeMovingAverage(data,'asksize',8)
        ComputeMovingAverage(data,'asksize',64)
        ComputeMovingAverage(data,'asksize',256)
        ComputeMovingAverage(data,'asksize',512)

        ComputeMovingAverage(data,'bidsize',8)
        ComputeMovingAverage(data,'bidsize',64)
        ComputeMovingAverage(data,'bidsize',256)
        ComputeMovingAverage(data,'bidsize',512)

        ComputeMovingAverage(data,'voi',8)
        ComputeMovingAverage(data,'voi',64)
        ComputeMovingAverage(data,'voi',256)
        ComputeMovingAverage(data,'voi',512)

        ComputeMovingAverage(data,'I',8)
        ComputeMovingAverage(data,'I',64)
        ComputeMovingAverage(data,'I',256)
        ComputeMovingAverage(data,'I',512)

        ComputeMovingStd(data,'bidprice',8)
        ComputeMovingStd(data,'bidprice',64)
        ComputeMovingStd(data,'bidprice',256)
        ComputeMovingStd(data,'bidprice',512)

        ComputeMovingStd(data,'askprice',8)
        ComputeMovingStd(data,'askprice',64)
        ComputeMovingStd(data,'askprice',256)
        ComputeMovingStd(data,'askprice',512)

        ComputeMovingStd(data,'asksize',8)
        ComputeMovingStd(data,'asksize',64)
        ComputeMovingStd(data,'asksize',256)
        ComputeMovingStd(data,'asksize',512)

        ComputeMovingStd(data,'bidsize',8)
        ComputeMovingStd(data,'bidsize',64)
        ComputeMovingStd(data,'bidsize',256)
        ComputeMovingStd(data,'bidsize',512)

        ComputeMovingStd(data,'voi',8)
        ComputeMovingStd(data,'voi',64)
        ComputeMovingStd(data,'voi',256)
        ComputeMovingStd(data,'voi',512)

        ComputeMovingStd(data,'I',8)
        ComputeMovingStd(data,'I',64)
        ComputeMovingStd(data,'I',256)
        ComputeMovingStd(data,'I',512)

        self.data = data.dropna()
    
    def SplitTrainTestData(self):
        data = self.data
        time_shift = self.time_shift

        from sklearn.model_selection import train_test_split
        index_non_na = ((data["midprice"]).shift(-time_shift)).notnull()
        X = data.loc[index_non_na].values
        y = (data["midprice"].shift(-time_shift) -(data["midprice"]))
        y = y.loc[index_non_na].values
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,shuffle = False)

        # We need to remove the testing set for the last entries (corresponding to time_shift)
        X_train = X_train[:len(X_train)-time_shift]
        y_train = y_train[:len(y_train)-time_shift]

        self.X_train = X_train
        self.X_test = X_test

        self.y_train_no_scale = y_train
        self.y_test_no_scale = y_test

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(y_train.reshape(-1,1))

        self.y_train = scaler.transform(y_train.reshape(-1,1))
        self.y_test = scaler.transform(y_test.reshape(-1,1))

    def SaveData(self):
        self.ReadData()
        self.EngineerFeatures()
        self.SplitTrainTestData()
        pd.DataFrame(self.X_train).to_csv("X_train_temp.csv",index = False)
        pd.DataFrame(self.X_test).to_csv("X_test_temp.csv",index = False)
        pd.DataFrame(self.y_train).to_csv("y_train_temp.csv",index = False)
        pd.DataFrame(self.y_test).to_csv("y_test_temp.csv",index = False)
        pd.DataFrame(self.y_test_no_scale).to_csv("y_test_temp_no_scale.csv",index = False)
        pd.DataFrame(self.y_train_no_scale).to_csv("y_train_temp_no_scale.csv",index = False)
        
        path = "storage.csv"
        if os.path.isfile(path):
            storage = pd.read_csv(path)
        else:
            storage = pd.DataFrame()
        NewStorage = pd.DataFrame()
        NewStorage["time_shift"] = [self.time_shift_in_s]
        NewStorage["agg"] = [self.agg]
        storage = storage.append(NewStorage)
        storage.to_csv("storage.csv",index = False)

    def Save(self):
        self.SaveData()


if __name__ == '__main__':
    time_shift = 1000
    file = "../Data/data_eng_1000ms.csv"
    P = Preprocesser(file,time_shift)
    P.Save()
