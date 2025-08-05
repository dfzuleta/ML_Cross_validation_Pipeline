# -*- coding: utf-8 -*-
"""
@author: Daniel Zuleta
"""

from sklearn.linear_model import LogisticRegression

def seleccion_lasso(X_train, y_train):
    log_lasso = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', random_state=42, max_iter=5000)
    log_lasso.fit(X_train, y_train)
    selected = X_train.columns[(log_lasso.coef_ != 0).any(axis=0)]
    excluded = X_train.columns[(log_lasso.coef_ == 0).all(axis=0)]
    return selected, excluded
