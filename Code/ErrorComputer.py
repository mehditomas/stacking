from sklearn.preprocessing import StandardScaler
import numpy as np


class ErrorComputer():

    def __init__(self,y_train_NS,y_test_NS):
        self.y_train_NS = y_train_NS
        self.y_test_NS = y_test_NS
        self.scaler = StandardScaler()
        self.scaler.fit(y_train_NS.reshape(-1,1))

    def ComputeAccuracy(self,y,predictions):
        predictions_NS = self.scaler.inverse_transform(predictions)
        return np.mean( ((predictions_NS >= 0) * 1) == ((y >= 0) * 1))

    def ComputeMAE(self,y,predictions):
        predictions_NS = self.scaler.inverse_transform(predictions)
        return np.mean(np.abs(predictions_NS  - y))

    def ComputeMSE(self,y,predictions):
        predictions_NS = self.scaler.inverse_transform(predictions)
        return np.mean((predictions_NS  - y )*(predictions_NS  - y ))

    def ComputeBaselineAccuracy(self,y):
        acc_1 = np.mean((y >= 0) * 1)
        acc_2 = np.mean((y <= 0) * 1)
        return max(acc_1,acc_2)

    def ComputeBaselineMAE(self,y):
        mae = np.mean(np.abs(np.mean(y)  - y))
        return mae

    def ComputeBaselineMSE(self,y):
        mean = np.mean(y)
        mse = np.mean((mean  - y )*(mean  - y ))
        return mse
