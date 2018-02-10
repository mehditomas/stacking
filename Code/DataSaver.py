from Layer import *
from ErrorComputer import *
import pandas as pd
import numpy as np
import os.path

class DataSaver():

    def __init__(self,layer):
        self.layer = layer
        self.path_save_train = "X_train_temp_" + self.layer.code + ".csv"
        self.path_save_test = "X_test_temp_" + self.layer.code + ".csv"

    def Save(self):
        df_train = pd.DataFrame(self.layer.X_train)
        df_test = pd.DataFrame(self.layer.X_test)
        for k in range(len(self.layer.models)):
            code = "model_"+str(k+1)
            df_train[code] = self.layer.predictions_train[k,:]
            df_test[code] = self.layer.predictions_test[k,:]
        df_train.to_csv(self.path_save_train,index = False)
        df_test.to_csv(self.path_save_test,index = False)