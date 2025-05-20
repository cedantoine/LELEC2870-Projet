import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

X1 = pd.read_csv("X1.csv")
Y1 = pd.read_csv("Y1.csv",header=None,names =['shares'])

X2 = pd.read_csv("X2.csv")

X1_val = X1.values
Y1_val = Y1.values
X2_val = X2.values

def PCA_selection(X,nb):
    # Initiate StandardScaler class
    scaler = StandardScaler(copy=True,with_mean=True,with_std=True)
    # Initialize PCA
    pca = PCA(n_components=nb)
    X_pca = pca.fit_transform(scaler.fit_transform(X))

    return X_pca

X1_pca_15 = PCA_selection(X1_val,15)
X2_pca_15 = PCA_selection(X2_val,15)

knn = KNeighborsRegressor(18)
knn.fit(X1_pca_15,Y1_val)
Y1_pred = knn.predict(X2_pca_15)

import csv
file = open("Y2.csv", "w")
for val in Y1_pred:
        file.write(str(int(np.round(val)))+'\n')
file.close()
