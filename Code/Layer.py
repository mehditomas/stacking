import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from cross_validation import SplitTrainTime
from ErrorComputer import *

class Layer():

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.models = []

        self.errors_train = []
        self.errors_test = []

        self.predictions_train = []
        self.predictions_test = []

        self.ErrComp = None
        self.metric = metrics.mean_squared_error
        self.n_splits = 5
        self.cross_validation = None
        
        self.code = None
        self.data_storage = None
        self.path_data_storage = None

    def add_model(self,model):
        self.models += [model]

    def TrainModelOnSplit(self,model,split):
        X_train,X_test,y_train,y_test = split
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        err = self.ErrComp.ComputeMSE(y_test,y_pred)
        error_train = err
        return y_pred,error_train

    def TrainModelOnAllSplits(self,model):
        self.cross_validation = SplitTrainTime(self.n_splits,self.X_train,self.y_train)
        new_predictions = np.array([])
        average_error = 0.0
        current_split_number = 1
        for split in self.cross_validation:
            print("current split: ",current_split_number)
            y_pred,error_train = self.TrainModelOnSplit(model,split)
            new_predictions = np.append(new_predictions,y_pred)

            # avoid storing biased error (not really out of sample)
            if current_split_number > 1:
                average_error += error_train
            current_split_number += 1
            
        average_error = average_error / self.n_splits
        return new_predictions,average_error

    def TrainAllModels(self):
        for model in self.models:
            new_predictions,average_error = self.TrainModelOnAllSplits(model)
            self.predictions_train += [new_predictions]
            self.errors_train += [average_error]
        self.errors_train = np.array(self.errors_train)
        self.predictions_train = np.array(self.predictions_train)
        print("layer training done...")

    def PredictOnTestSet(self):
        for model in self.models:
            model.fit(self.X_train,self.y_train)
            predictions_test = model.predict(self.X_test)
            self.predictions_test += [predictions_test]
            err = self.ErrComp.ComputeMSE(self.y_test,predictions_test)
            self.errors_test += [err]
        self.errors_test = np.array(self.errors_test)
        self.predictions_test = np.array(self.predictions_test)
        print("layer prediction done...")

    def RunLayer(self):
        self.TrainAllModels()
        self.PredictOnTestSet()



if __name__ == '__main__':

    X_train = pd.read_csv("X_train_temp.csv").values
    X_test = pd.read_csv("X_test_temp.csv").values
    y_test = pd.read_csv("y_test_temp.csv").values.ravel()
    y_train = pd.read_csv("y_train_temp.csv").values.ravel()
    data_storage = pd.read_csv("storage_1000.csv")
    y_train_NS = pd.read_csv('y_train_temp_no_scale.csv').values.ravel()
    y_test_NS = pd.read_csv('y_test_temp_no_scale.csv').values.ravel()

    ErrComp = ErrorComputer(y_train_NS,y_test_NS)

    l_1 = Layer(X_train,y_train,X_test,y_test,"layer_1",data_storage,"storage_1000.csv",ErrComp)

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
    model = Pipeline(steps = [('scaler',StandardScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
    l_1.add_model(model)
    model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
    l_1.add_model(model)

    # # standard scaling + PCA
    # n_components = 5
    # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
    # l_1.add_model(model)
    # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
    # l_1.add_model(model)

    # # standard scaling
    # l_1.add_model(GradientBoostingRegressor(n_estimators = 100))

    # l_1.add_model(Lasso(alpha = 1.0))
    # l_1.add_model(Lasso(alpha = 10.0))
    # l_1.add_model(Lasso(alpha = 100.0))
    # l_1.add_model(Ridge(alpha = 1.0))
    # l_1.add_model(Ridge(alpha = 10.0))
    # l_1.add_model(Ridge(alpha = 100.0))
    # l_1.add_model(AdaBoostRegressor(n_estimators = 100))
    # l_1.add_model(AdaBoostRegressor(n_estimators = 200))
    # l_1.add_model(AdaBoostRegressor(n_estimators = 50))
    # l_1.add_model(KNeighborsRegressor(8))
    # l_1.add_model(KNeighborsRegressor(16))
    # l_1.add_model(KNeighborsRegressor(32))
    # l_1.add_model(KNeighborsRegressor(64))
    # l_1.add_model(KNeighborsRegressor(128))
    # l_1.add_model(XGBRegressor(n_estimators = 50))
    # l_1.add_model(XGBRegressor(n_estimators = 100))
    # l_1.add_model(XGBRegressor(n_estimators = 200))
    # l_1.add_model(GradientBoostingRegressor(n_estimators = 50))
    # l_1.add_model(GradientBoostingRegressor(n_estimators = 100))
    # l_1.add_model(GradientBoostingRegressor(n_estimators = 200))

    l_1.RunLayer()
    l_1.StoreBestErrors(2)
    l_1.WriteNewData()

    data_storage.to_csv("storage_1000.csv",index = False)

    X_train = pd.read_csv("X_train_temp_layer_1.csv").values
    X_test = pd.read_csv("X_test_temp_layer_1.csv").values
    y_test = pd.read_csv("y_test_temp.csv").values.ravel()
    y_train = pd.read_csv("y_train_temp.csv").values.ravel()

    l_2 = Layer(X_train,y_train,X_test,y_test,"layer_2",data_storage,"storage_1000.csv",ErrComp)


    # l_2.add_model(Lasso(alpha = 1.0))
    # l_2.add_model(Lasso(alpha = 10.0))
    # l_2.add_model(Lasso(alpha = 100.0))
    # l_2.add_model(Ridge(alpha = 1.0))
    # l_2.add_model(Ridge(alpha = 10.0))
    # l_2.add_model(Ridge(alpha = 100.0))
    # l_2.add_model(AdaBoostRegressor(n_estimators = 100))
    # l_2.add_model(AdaBoostRegressor(n_estimators = 200))
    # l_2.add_model(AdaBoostRegressor(n_estimators = 50))
    # l_2.add_model(KNeighborsRegressor(2))
    # l_2.add_model(KNeighborsRegressor(4))
    # l_2.add_model(KNeighborsRegressor(8))
    # l_2.add_model(XGBRegressor(n_estimators = 50))
    # l_2.add_model(XGBRegressor(n_estimators = 100))
    # l_2.add_model(XGBRegressor(n_estimators = 200))
    # l_2.add_model(GradientBoostingRegressor(n_estimators = 50))
    l_2.add_model(GradientBoostingRegressor(n_estimators = 100))
    l_2.add_model(GradientBoostingRegressor(n_estimators = 200))



    l_2.RunLayer()
    l_2.StoreBestErrors(2)
    l_2.WriteNewData()


    ###### FINAL LAYER #########

    X_train = pd.read_csv("X_train_temp_layer_2.csv").values
    X_test = pd.read_csv("X_test_temp_layer_2.csv").values
    y_test = pd.read_csv("y_test_temp.csv").values.ravel()
    y_train = pd.read_csv("y_train_temp.csv").values.ravel()

    clf = GradientBoostingRegressor(n_estimators = 100)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    from sklearn.metrics import classification_report

    print(classification_report(y_test > 0,y_pred > 0))
