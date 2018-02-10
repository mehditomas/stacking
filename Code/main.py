from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor 
import pandas as pd
import numpy as np


X_train = pd.read_csv("X_train_temp_layer_2.csv").values
X_test = pd.read_csv("X_test_temp_layer_2.csv").values
y_test = pd.read_csv("y_test_temp.csv").values.ravel()
y_train = pd.read_csv("y_train_temp.csv").values.ravel()
y_train_NS = pd.read_csv("y_train_temp_no_scale.csv").values.ravel()
y_test_NS = pd.read_csv("y_test_temp_no_scale.csv").values.ravel()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(y_train_NS.reshape(-1,1))


clf = GradientBoostingRegressor(n_estimators = 100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test_NS > 0,y_pred > 0))

print(np.mean(np.abs(y_pred - y_test_NS)))
print(np.mean(np.abs(np.mean(y_test_NS) - y_test_NS)))

clf = XGBRegressor(n_estimators = 100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test_NS > 0,y_pred > 0))

print(np.mean(np.abs(y_pred - y_test_NS)))
print(np.mean(np.abs(np.mean(y_test_NS) - y_test_NS)))

clf = Ridge()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test_NS > 0,y_pred > 0))

print(np.mean(np.abs(y_pred - y_test_NS)))
print(np.mean(np.abs(np.mean(y_test_NS) - y_test_NS)))


import keras

from keras.layers import Dense
from keras.models import Sequential
from keras import losses

model = Sequential()
layer_1 = Dense(1,input_dim = X_train.shape[1])
model.add(layer_1)


model.compile(loss = losses.mean_absolute_error, optimizer = "adam")

model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 3,verbose = True)

y_pred = model.predict(X_test)

print(classification_report(y_test_NS > 0,scaler.inverse_transform(y_pred) > 0))
