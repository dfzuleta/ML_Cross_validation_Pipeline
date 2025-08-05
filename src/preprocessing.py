# -*- coding: utf-8 -*-
"""
@author: Daniel Zuleta
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def one_hot_encode(X):
    return pd.get_dummies(X)

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test, numeric_columns):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    return X_train_scaled, X_test_scaled, scaler

def scale_full_dataset(X, numeric_columns):
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    return X_scaled, scaler