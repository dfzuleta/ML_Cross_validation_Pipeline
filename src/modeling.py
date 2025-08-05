# -*- coding: utf-8 -*-
"""
@author: Daniel Zuleta
"""

import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, roc_auc_score

def register_models():
    """
    Registra manualmente los modelos que deseas evaluar.
    Puedes añadir más modelos aquí. Asegúrate de importar correctamente las librerías necesarias.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    # from xgboost import XGBClassifier  # Descomenta si usas XGBoost

    modelos = {
        'Decision Tree': DecisionTreeClassifier(random_state=226),
        'Random Forest': RandomForestClassifier(random_state=226),
        'K Neighbors': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(multi_class='ovr', random_state=42),
        # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # Ejemplo adicional
    }

    return modelos


def register_cv_methods():
    """
    Registra manualmente los métodos de validación cruzada que deseas utilizar.
    Puedes añadir más si lo deseas.
    """
    from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold, LeavePOut

    cv_metodos = {
        'ShuffleSplit': ShuffleSplit,
        'KFold': KFold,
        'StratifiedKFold': StratifiedKFold,
        'LeavePOut': LeavePOut,
    }

    return cv_metodos


def evaluar_modelos(modelos, cv_metodos, X, y, X_train, X_test, y_train, y_test, valores_k):
    """
    Evalúa cada modelo con cada método de validación cruzada.
    """
    resultados_totales = []
    metricas = {
        'accuracy': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
    }

    for nombre_modelo, modelo in modelos.items():
        for nombre_cv, cv_clase in cv_metodos.items():
            k_list = valores_k if nombre_cv != 'LeavePOut' else ['LOOCV']
            resultados_df = pd.DataFrame({'Valores K': k_list,
                                          'Accuracy': [None]*len(k_list),
                                          'F1 Score': [None]*len(k_list),
                                          'Recall': [None]*len(k_list),
                                          'AUC': [None]*len(k_list)})

            if nombre_cv == 'LeavePOut':
                cv = cv_clase(p=1)
                puntajes = cross_validate(modelo, X, y, cv=cv, scoring=metricas)
                for met in resultados_df.columns[1:]:
                    resultados_df.at[0, met] = puntajes[f'test_{met.lower()}'].mean()
            else:
                for i, k in enumerate(valores_k):
                    cv = cv_clase(n_splits=k, shuffle=True, random_state=42)
                    puntajes = cross_validate(modelo, X, y, cv=cv, scoring=metricas)
                    for met in resultados_df.columns[1:]:
                        resultados_df.at[i, met] = puntajes[f'test_{met.lower()}'].mean()

            # Modelo base sin CV
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)
            base_scores = {
                'Accuracy': accuracy_score(y_test, pred),
                'F1 Score': f1_score(y_test, pred, average='macro'),
                'Recall': recall_score(y_test, pred, average='macro'),
                'AUC': roc_auc_score(y_test, modelo.predict_proba(X_test), multi_class='ovr')
            }
            resultados_df.loc[len(resultados_df)] = ['Modelo Base'] + list(base_scores.values())

            resultados_totales.append((f"{nombre_modelo} con {nombre_cv}", resultados_df))

    return resultados_totales
