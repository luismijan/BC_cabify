import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import json

import torch
from torcheval.metrics import FrechetInceptionDistance, PeakSignalNoiseRatio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import streamlit as st

from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

with open('./data/challenge_dataset.json', 'rb') as f:
    data = f.read()
    # .load(f)

aux = str(data)
st.write(aux.split('},'))
st.write('No matter the way the json file is read there are no moro than 3027 rows')


with open('./data/challenge_dataset.json', 'rb') as f:
    data = json.load(f)


journey_id = []
annotator = []
annotation = []
estimated_route = []
real_route = []

for row in range(len(data)):
    journey_id.append(data[row]['journey_id'])
    annotator.append(data[row]['annotator'])
    annotation.append(data[row]['annotation'])
    estimated_route.append(data[row]['estimated_route'])
    real_route.append(data[row]['real_route'])

schema = pa.schema([
    pa.field('journey_id', pa.string()),
    pa.field('annotator', pa.int32()),
    pa.field('annotation', pa.string()),
    pa.field('estimated_route', pa.list_(pa.list_(pa.float64()))),
    pa.field('real_route', pa.list_(pa.list_(pa.float64())))
]
)

df = pa.Table.from_arrays([journey_id,annotator,annotation,estimated_route,real_route], 
                            names = ['journey_id','annotator','annotation','estimated_route','real_route']
                            )

df_pandas = df.to_pandas()
# Convertir las listas anidadas en tuplas para que pandas pueda agruparlas
df_pandas['estimated_route'] = df_pandas['estimated_route'].apply(lambda x: tuple(map(tuple, x)))
df_pandas['real_route'] = df_pandas['real_route'].apply(lambda x: tuple(map(tuple, x)))

# Agrupar por 'estimated_route' y 'real_route' y contar las anotaciones únicas
result = df_pandas.groupby(['estimated_route', 'real_route'], as_index=False)['annotation'].nunique()
dupli_esti = result.query('annotation > 1')['estimated_route']
dupli_real = result.query('annotation > 1')['real_route']

df_duplicated = df_pandas[(df_pandas['estimated_route'].isin(dupli_esti)) & (df_pandas['real_route'].isin(dupli_real))]
df_duplicated = df_duplicated.groupby('journey_id', as_index = False)['annotation'].unique()
df_duplicated['first'] = df_duplicated['annotation'].apply(lambda x: x[0])
df_duplicated['second'] = df_duplicated['annotation'].apply(lambda x: x[1])

st.write('check the distribution')

st.write(df_duplicated[['first', 'second']].value_counts(normalize = False))

id_wrong = df_duplicated[(df_duplicated['first'].isin(['They differ','Both are the same'])) & (df_duplicated['second'].isin(['They differ','Both are the same']))]['journey_id']
id_easy = df_duplicated[(df_duplicated['first'] == "I don't know") | (df_duplicated['second'] == "I don't know")]['journey_id']


df_duplicated2 = df_pandas[df_pandas['journey_id'].isin(id_wrong)][['journey_id', 'annotation', 'annotator']]


st.write('Almost all the duplicated annotations are noted by an annotator with a clare bias')
st.write(df_duplicated2.groupby(['annotator'], as_index= False)['annotation'].value_counts(normalize = False))
st.write('In consecuence the routes classified by those annotators will be taken appart. The valid result for this routes will be the other one')

st.write('The only annotator without a obvious bias is the 2º')
st.write(df_pandas[df_pandas['annotator'] == 2]['annotation'].value_counts())
aux = df_pandas[df_pandas['annotator'] == 2]['journey_id']

df_duplicated = df_pandas[(df_pandas['journey_id'].isin(aux)) & (df_pandas['journey_id'].isin(id_wrong))].groupby(['journey_id', 'annotator'], as_index = False)['annotation'].unique()
st.write(df_duplicated)
st.write('''Paying attention to this case the possibility of have a travel noted more than twice appears
         It means is possible to take the most probable or the most repeated option in the multiclassification cases when the options are other but "I don't know"
         ''')

aux = df_pandas[df_pandas['journey_id'].isin(id_wrong)].groupby(['journey_id'], as_index = False)['annotation'].value_counts(normalize = False).query('count > 1').sort_values(['journey_id', 'count'], ascending = False).drop_duplicates(subset = ['journey_id', 'count'], keep = 'first')
st.write(aux)
df_pandas = df_pandas[~((df_pandas['journey_id'].isin(id_easy)) & (df_pandas['annotation'] == "I don't know"))]
df_pandas = df_pandas[~((df_pandas['journey_id'].isin(id_wrong))) & ~(pd.Series(df_pandas['annotation'] + df_pandas['journey_id']).isin(aux['annotation'] + aux['journey_id']))]
st.write(df_pandas['annotation'].value_counts(normalize = True))
st.write(df_pandas.shape)
st.write(df_pandas[df_pandas['annotation'] != "I don't know"]['annotation'].value_counts(normalize = True))