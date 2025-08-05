# -*- coding: utf-8 -*-
"""
@author: Daniel Zuleta
"""

import pandas as pd
from src import preprocessing, modeling, feature_selection, visualization

# ========================
# 1. Cargar tus datos aquí
# ========================
# X: DataFrame con tus variables independientes
# y: Series o DataFrame con la variable objetivo
# Ejemplo:
# X = pd.read_csv("data/mis_datos.csv")
# y = X.pop("target")

# Asegúrate de definir X e y antes de continuar
# =============================================
# X = ...
# y = ...

# =============================
# 2. Define las columnas numéricas
# =============================
# Reemplaza la siguiente lista con tus columnas numéricas reales
Columnas_numero = [
    # "edad", "ingresos", "horas_trabajo", ...
]

# ================
# 3. Preprocesamiento
# ================
X_encoded = preprocessing.one_hot_encode(X)
X_train, X_test, y_train, y_test = preprocessing.split_data(X_encoded, y)
X_train_scaled, X_test_scaled, _ = preprocessing.scale_features(X_train, X_test, Columnas_numero)
X_encoded_scaled, _ = preprocessing.scale_full_dataset(X_encoded, Columnas_numero)

# =============================
# 4. Definición de modelos y CV
# =============================
modelos = modeling.register_models()
cv_metodos = modeling.register_cv_methods()
valores_k = list(range(2, 10)) + [15, 20]  # Puedes modificar según tu criterio

# ========================
# 5. Evaluación de modelos
# ========================
resultados = modeling.evaluar_modelos(
    modelos, cv_metodos,
    X_encoded_scaled, y,
    X_train_scaled, X_test_scaled, y_train, y_test,
    valores_k
)

# ========================
# 6. Guardar resultados
# ========================
with pd.ExcelWriter("outputs/resultados_modelos.xlsx") as writer:
    for nombre, df in resultados:
        df.to_excel(writer, sheet_name=nombre[:30], index=False)

# ========================
# 7. Visualización de métricas
# ========================
visualization.plot_metric_results(resultados, output_dir="outputs")

# ========================
# 8. Selección de variables con Lasso
# ========================
selected, excluded = feature_selection.seleccion_lasso(X_train_scaled, y_train)
print(f"✅ Variables seleccionadas por Lasso: {list(selected)}")
print(f"🚫 Variables excluidas por Lasso: {list(excluded)}")
