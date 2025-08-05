# -*- coding: utf-8 -*-
"""
@author: Daniel Zuleta
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_metric_results(resultados, output_dir):
    sns.set(style="whitegrid")
    metricas = ['Accuracy', 'F1 Score', 'Recall', 'AUC']

    for metrica in metricas:
        datos = []
        for nombre, df in resultados:
            modelo, cv = nombre.split(" con ")
            promedio = df[df['Valores K'] != 'Modelo Base'][metrica].mean()
            base = df[df['Valores K'] == 'Modelo Base'][metrica].values[0]
            datos.append((modelo, cv, 'Promedio CV', promedio))
            datos.append((modelo, 'Base', 'Modelo Base', base))
        df_plot = pd.DataFrame(datos, columns=['Modelo', 'Validación Cruzada', 'Tipo', metrica])

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=df_plot, x='Modelo', y=metrica, hue='Validación Cruzada', ci=None)
        plt.xticks(rotation=45)
        plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.1))

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=9, color='black', xytext=(0, 8),
                        textcoords='offset points')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{metrica}_comparativa_modelos.png")
        plt.savefig(output_path)
        plt.close()
