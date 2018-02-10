from Layer import *
from ErrorComputer import *
import pandas as pd
import numpy as np
import os.path



class MetricSaver():

    def __init__(self,layer,path_data_storage,Position):
        self.layer = layer
        self.position = Position

        self.predictions_train = layer.predictions_train
        self.predictions_test = layer.predictions_test
        self.path_file = layer.code + "_" + path_data_storage
        self.path_data_storage = path_data_storage

        self.n_errors = min(5,len(self.layer.models))
        self.agg = None
        self.time_shift = None
        self.errors_MAE = None
        self.errors_MSE = None
        self.errors_Acc = None
        self.data = None

    def CreateData(self):
        data = pd.read_csv(self.layer.path_data_storage)
        data.to_csv(self.path_file,index = False)
        
    def GetPreviousData(self):
        fname = self.path_file
        if os.path.isfile(fname):
            data = pd.read_csv(fname)
        else:
            self.CreateData()
            data = pd.read_csv(fname)
        self.data = data
        DataStorage = pd.read_csv("storage.csv")
        self.agg = int(DataStorage["agg"].values[self.position])
        self.time_shift = int(DataStorage["time_shift"].values[self.position])

    def AddMSEToData(self):
        indices = self.layer.errors_train.argsort()[:self.n_errors]
        MSE = self.layer.errors_test[indices]
        MSE = list(MSE)
        self.errors_MSE = MSE

    def AddMAEToData(self):
        indices = self.layer.errors_train.argsort()[:self.n_errors]
        MAE = []
        for i in indices:
            mae = self.layer.ErrComp.ComputeMAE(self.layer.y_test,self.layer.predictions_test[i])
            MAE += [mae]
        MAE = list(MAE)
        self.errors_MAE = MAE

    def AddAccToData(self):
        indices = self.layer.errors_train.argsort()[:self.n_errors]
        Acc = []
        for i in indices:
            acc = self.layer.ErrComp.ComputeAccuracy(self.layer.y_test,self.layer.predictions_test[i])
            Acc += [acc]
        Acc = list(Acc)
        self.errors_Acc = Acc

    def AddBaselineToData(self):
        ErrComp = self.layer.ErrComp
        self.baseline_Acc = ErrComp.ComputeBaselineAccuracy(ErrComp.y_test_NS)
        self.baseline_MAE = ErrComp.ComputeBaselineMAE(ErrComp.y_test_NS)
        self.baseline_MSE = ErrComp.ComputeBaselineMSE(ErrComp.y_test_NS)


    def AddMetricsToDataFirstTime(self):
        self.AddMAEToData()
        self.AddMSEToData()
        self.AddAccToData()
        self.AddBaselineToData()
        self.data["agg"] = [int(self.agg)]
        self.data["time_shift"] = [int(self.time_shift)]
        for k in range(self.n_errors):
            self.data["best_"+str(k+1)+"_MSE"] = [self.errors_MSE[k]]
            self.data["best_"+str(k+1)+"_MAE"] = [self.errors_MAE[k]]
            self.data["best_"+str(k+1)+"_ACC"] = [self.errors_Acc[k]]
        self.data["baseline_ACC"] = [self.baseline_Acc]
        self.data["baseline_MAE"] = [self.baseline_MAE]
        self.data["baseline_MSE"] = [self.baseline_MSE]
    
    def AddMetricsToDataAfterFirst(self):
        self.AddMAEToData()
        self.AddMSEToData()
        self.AddAccToData()
        self.AddBaselineToData()
    
        NewData = pd.DataFrame()
        NewData["agg"] = [int(self.agg)]
        NewData["time_shift"] = [int(self.time_shift)]
        for k in range(self.n_errors):
            NewData["best_"+str(k+1)+"_MSE"] = [self.errors_MSE[k]]
            NewData["best_"+str(k+1)+"_MAE"] = [self.errors_MAE[k]]
            NewData["best_"+str(k+1)+"_ACC"] = [self.errors_Acc[k]]
        NewData["baseline_ACC"] = [self.baseline_Acc]
        NewData["baseline_MAE"] = [self.baseline_MAE]
        NewData["baseline_MSE"] = [self.baseline_MSE]
        
        self.data = self.data.append(NewData)

    def AddMetricsToData(self):
        if os.path.isfile(self.path_file):
            self.GetPreviousData()
            self.AddMetricsToDataAfterFirst()
        else:
            self.GetPreviousData()
            self.AddMetricsToDataFirstTime()

    def SaveData(self):
        self.data.to_csv(self.path_file,index = False)

    def Save(self):
        self.AddMetricsToData()
        self.SaveData()
        
