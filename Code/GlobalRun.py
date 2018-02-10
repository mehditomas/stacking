import os
from Preprocesser import *
from Layer import *
from LayerWrapper import *

if __name__ == '__main__':
    y = [100]
    for k in range(len(y)):
        time_shift = y[k]
        file = "../Data/data_eng_1000ms.csv"
        P = Preprocesser(file,time_shift)
        P.Save()
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.preprocessing import StandardScaler,MinMaxScaler
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA

        layer_1 = Layer()
        layer_2 = Layer()
        layer_3 = Layer()

        # standard scaling
        model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)

        # # standard scaling + PCA
        # n_components = 5
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)


        # # min max scaling
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)

        # # min max scaling + pca
        # n_components = 5
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_1.add_model(model)


        # layer_1.add_model(Lasso(alpha = 1.0))
        # layer_1.add_model(Lasso(alpha = 10.0))
        # layer_1.add_model(Lasso(alpha = 100.0))
        # layer_1.add_model(Ridge(alpha = 1.0))
        # layer_1.add_model(Ridge(alpha = 10.0))
        # layer_1.add_model(Ridge(alpha = 100.0))
        # layer_1.add_model(AdaBoostRegressor(n_estimators = 100))
        # layer_1.add_model(AdaBoostRegressor(n_estimators = 500))
        # layer_1.add_model(AdaBoostRegressor(n_estimators = 50))
        # layer_1.add_model(KNeighborsRegressor(10))
        # layer_1.add_model(KNeighborsRegressor(40))
        # layer_1.add_model(KNeighborsRegressor(80))
        # layer_1.add_model(GradientBoostingRegressor(n_estimators = 50))
        # layer_1.add_model(GradientBoostingRegressor(n_estimators = 100))
        # layer_1.add_model(GradientBoostingRegressor(n_estimators = 500))


        ###### Pipeline models

        # standard scaling
        model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 10))])
        layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)

        # # standard scaling + PCA
        # n_components = 5
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)

        # n_components = 10
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)

        # n_components = 20
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',StandardScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)



        # # min max scaling
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',AdaBoostRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)

        # # min max scaling + pca
        # n_components = 5
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)

        # n_components = 10
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)

        # n_components = 20
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',AdaBoostRegressor(n_estimators = 100))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 10))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 50))])
        # layer_2.add_model(model)
        # model = Pipeline(steps = [('scaler',MinMaxScaler()),('pca',PCA(n_components)),('reg',GradientBoostingRegressor(n_estimators = 100))])
        # layer_2.add_model(model)

        # layer_2.add_model(Lasso(alpha = 1.0))
        # layer_2.add_model(Lasso(alpha = 10.0))
        # layer_2.add_model(Lasso(alpha = 100.0))
        # layer_2.add_model(Ridge(alpha = 1.0))
        # layer_2.add_model(Ridge(alpha = 10.0))
        # layer_2.add_model(Ridge(alpha = 100.0))
        # layer_2.add_model(AdaBoostRegressor(n_estimators = 100))
        # layer_2.add_model(AdaBoostRegressor(n_estimators = 500))
        # layer_2.add_model(AdaBoostRegressor(n_estimators = 50))
        # layer_2.add_model(KNeighborsRegressor(10))
        # layer_2.add_model(KNeighborsRegressor(40))
        # layer_2.add_model(KNeighborsRegressor(80))
        # layer_2.add_model(GradientBoostingRegressor(n_estimators = 50))
        # layer_2.add_model(GradientBoostingRegressor(n_estimators = 100))
        # layer_2.add_model(GradientBoostingRegressor(n_estimators = 500))

        model = GradientBoostingRegressor(n_estimators = 500)
        layer_3.add_model(model)

        myWrapper = LayerWrapper([layer_1,layer_2,layer_3],k)
        myWrapper.RunAllLayers()
