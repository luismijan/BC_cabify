import pandas as pd
import numpy as np

from xgboost import XGBClassifier
import catboost as ct
from sklearn.ensemble import RandomForestClassifier

import optuna
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import shap 
import joblib
import os
import configparser

import streamlit as st
from streamlit_shap import st_shap

from utils.functions import *

config = configparser.ConfigParser()
config.read('./utils/config.ini')

file = config["modelling"]["file_transform"]
target_var= config["modelling"]["target_var"]
id_var = config["modelling"]["id_var"]



def hyperparams_research(trial):
    param = {   
        'iterations': trial.suggest_int('iterations', 90, 150),
        'learning_rate':trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'max_bin': trial.suggest_int('max_bin', 3, 10),
        'gamma': trial.suggest_float('gamma', 3, 5),
        'base_score': trial.suggest_float('base_score', 0, 1),
        'eval_metric': trial.suggest_categorical('eval_metric', ['error', 'error@0.45','error@0.5','error@.55']),
        'class_weights':{
            0: trial.suggest_float('0', 0,5),
            1: trial.suggest_float('1', 0, 12)
            },
        'lambda':trial.suggest_float('lambda', 5, 15),
        'alpha':trial.suggest_float('alpha', 5, 15),
        'tree_method': trial.suggest_categorical('tree_method', ['exact','approx', 'hist'])
    }
    if param["tree_method"] != "exact":
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    
    gbm = XGBClassifier(**param, random_seed = 42)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    
    return accuracy

if __name__ == '__main__':

    df = pd.read_feather(file)
    print(df.head())
    df[target_var] = np.where(df[target_var] == 'They differ', 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([target_var, id_var], axis = 1),
        df[target_var]
    )

    try:
        modelos = pd.Series(os.listdir('./utils/models'))
        modelos = modelos[modelos.str.contains('joblib')]
        last_version = modelos.apply(lambda x: float(x.split('_v')[1].replace('.joblib', ''))).sort_values(ascending = False).reset_index(drop = True)[0] 
        print(f'./utils/models/xgboost_model_v{int(last_version)}.joblib')
    except Exception as e:
        st.write(e)
        last_version = 0

    if config['modelling']['execution_type'] == 'train':
        options = st.multiselect(
            "Choose the features to model",
            X_train.columns,
            X_train.columns,
        )


        study = optuna.create_study(direction="maximize")
        study.optimize(hyperparams_research, n_trials=100, timeout=600)

        print("Número de pruebas finalizadas:", len(study.trials))
        print("Mejor prueba:")
        trial = study.best_trial
        print("Valor:", trial.value)
        print("Parámetros:")
        for key, value in trial.params.items():
            print(f"{key}: {value}")
        hyperparams = trial.params
        
        model = XGBClassifier(**hyperparams, random_seed = 42)
        model.fit(X_train[options], y_train, eval_set=[(X_test[options], y_test)], verbose=0)

    else:
        model = joblib.load(f'./utils/models/xgboost_model_v{int(last_version)}.joblib')

    
    if st.button("Save model"):
        model_name = f'./utils/models/xgboost_model_v{int(last_version + 1)}.joblib'
        joblib.dump(model, model_name)
        st.write(model_name, 'SAVED')

    odd_train = model.predict_proba(X_train[model.feature_names_in_])[:,1]

    fnr, fpr, thresholds, equal_threshold = false_rates(y_train, odd_train)
    corte = st.slider('punto de corte', 0,100,50)
    pred_train = np.where(odd_train >= corte/100, 1, 0)

    odd_test = model.predict_proba(X_test[model.feature_names_in_])[:,1]
    pred_test = np.where(odd_test >= corte/100, 1, 0)

    explainer = shap.Explainer(model)
    

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.distplot(odd_train, ax = ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pd.DataFrame(X_train.corr()), ax = ax)
    st.pyplot(fig)

    st.subheader('Train')
    fig, ax = plt.subplots(figsize=(10, 5))

    shap_values = explainer(X_train[model.feature_names_in_])    
    st_shap(shap.summary_plot(shap_values, X_train[model.feature_names_in_], feature_names = model.feature_names_in_))
    
    st.subheader('Test')
    fig, ax = plt.subplots(figsize=(10, 5))
    shap_values = explainer(X_test[model.feature_names_in_])
    st_shap(shap.summary_plot(shap_values, X_test[model.feature_names_in_], feature_names = model.feature_names_in_))

    col1, col2 = st.columns(2)

    with col1:
        st.header('TRAIN')
        st.line_chart(pd.DataFrame({'false negative rate':fnr, 'false positive rate':fpr}, index=thresholds))
        st.table(model_metrics(y_train, pred_train))
        st.pyplot(model_parameters(y_train, pred_train))
    with col2:

        st.header('TEST')
        fig, ax  = plt.subplots()
        fnr, fpr, thresholds, equal_threshold = false_rates(y_train, odd_train)
        st.line_chart(pd.DataFrame({'false negative rate':fnr, 'false positive rate':fpr}, index=thresholds))
        st.table(model_metrics(y_test, pred_test))
        st.pyplot(model_parameters(y_test, pred_test))