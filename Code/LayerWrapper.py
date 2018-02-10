import pandas as pd
import numpy as np
from Layer import *
from MetricSaver import *
from DataSaver import *


class LayerWrapper():

    def __init__(self,Layers,Position):
        self.Layers = Layers
        self.Position = Position

    def InitializePathK(self,k):
        self.Layers[k-1].path_data_storage = "storage.csv"

    def InitializeCodeLayers(self):
        for k in range(len(self.Layers)):
            self.Layers[k].code = "layer_"+str(k+1)

    def InitializeDataLayerK(self,k):
        if k == 1:
            path_train_X = "X_train_temp.csv"
            path_train_y = "y_train_temp.csv"
            path_test_X = "X_test_temp.csv"
            path_test_y = "y_test_temp.csv"
        else:
            path_train_X = "X_train_temp_layer_" + str(k-1)+".csv"
            path_train_y = "y_train_temp.csv"
            path_test_X = "X_test_temp_layer_" + str(k-1)+ ".csv"
            path_test_y = "y_test_temp.csv"
    
        i = k-1
        self.Layers[i].X_train = pd.read_csv(path_train_X).values
        self.Layers[i].y_train = pd.read_csv(path_train_y).values.ravel()
        self.Layers[i].X_test = pd.read_csv(path_test_X).values
        self.Layers[i].y_test = pd.read_csv(path_test_y).values.ravel()

    def InitializeErrorComputerLayerK(self,k):
        y_train_NS = pd.read_csv('y_train_temp_no_scale.csv').values.ravel()
        y_test_NS = pd.read_csv('y_test_temp_no_scale.csv').values.ravel()
        ErrComp = ErrorComputer(y_train_NS,y_test_NS)
        i = k-1
        self.Layers[i].ErrComp = ErrComp

    def InitializeLayerK(self,k):
        self.InitializeCodeLayers()
        self.InitializeDataLayerK(k)
        self.InitializeErrorComputerLayerK(k)
        self.InitializePathK(k)

    def RunLayerK(self,k):
        self.InitializeLayerK(k)
        i = k-1
        self.Layers[i].RunLayer()

    def RunAllLayers(self):
        for k_ in range(len(self.Layers)):
            print("running layer "+str(k_+1) + "...")
            k = k_ + 1
            self.RunLayerK(k)
            layer = self.Layers[k_]
            TempDataSaver = DataSaver(layer)
            TempMetricSaver = MetricSaver(layer,"storage.csv",self.Position)
            TempDataSaver.Save()
            TempMetricSaver.Save()


if __name__ == '__main__':
    layer_1 = Layer()
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import ExtraTreeRegressor
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from xgboost import XGBRegressor

    ###### Pipeline models

    # standard scaling
    model = Pipeline(steps = [('scaler',StandardScaler()),('reg',Lasso())])
    layer_1.add_model(model)
    model = Pipeline(steps = [('scaler',StandardScaler()),('reg',Ridge())])
    layer_1.add_model(model)

    # layer_2 = Layer()
    # # standard scaling
    # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
    # layer_2.add_model(model)
    # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
    # layer_2.add_model(model)


    myWrapper = LayerWrapper([layer_1])
    myWrapper.RunAllLayers()
