import numpy as np


def SplitTrainTime(n_splits,X,y):
    N = len(X)
    folds = []
    for k in range(n_splits):
        final_position = max(1000,k*int(N*1.0/(n_splits)))
        if k == n_splits-1:
            increment = N - final_position
        else:
            increment = int(N*1.0/(n_splits))
        train,test,train_target,test_target = X[0:final_position,:],X[final_position:final_position+increment,:],y[0:final_position],y[final_position:final_position+increment]
        folds += [(train,test,train_target,test_target)]
    return folds
