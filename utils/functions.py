import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def false_rates(y_train, y_probs):
    # Inicializar listas para almacenar las tasas de falsos positivos y falsos negativos
    fpr_list = []
    fnr_list = []
    thresholds = np.arange(0.0, 1.1, 0.05)

    # Calcular la tasa de falsos positivos y falsos negativos para diferentes umbrales
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    # Encontrar el umbral donde FPR y FNR son iguales o más cercanos entre sí
    min_diff_index = np.argmin(np.abs(np.array(fpr_list) - np.array(fnr_list)))
    equal_threshold = thresholds[min_diff_index]
    equal_fpr = fpr_list[min_diff_index]
    equal_fnr = fnr_list[min_diff_index]

    print(f"El umbral donde FPR y FNR están más cercanos es {equal_threshold:.2f}")
    print(f"En este umbral: FPR = {equal_fpr:.2f}, FNR = {equal_fnr:.2f}")

    return fnr_list, fpr_list, thresholds, equal_threshold


def model_metrics(real, predicted):
    return pd.DataFrame(classification_report(real, predicted, output_dict=True))
    

def model_parameters(real, predicted):
    fig, ax = plt.subplots()
    cf_matrix = confusion_matrix(real, predicted)
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g', ax = ax)
    plt.title("Confusion matrix model ")
    return fig