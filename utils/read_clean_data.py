import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import json

import os
def reading_data(file):

    with open(file, 'rb') as f:
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

        
    df = pd.DataFrame({'journey_id':journey_id,'annotator':annotator,'annotation':annotation,'estimated_route':estimated_route,'real_route':real_route})

    return df

def cleaning_data(df, real, estimate):

    # Convertir las listas anidadas en tuplas para que pandas pueda agruparlas
    df[estimate] = df[estimate].apply(lambda x: tuple(map(tuple, x)))
    df[real] = df[real].apply(lambda x: tuple(map(tuple, x)))

    # Agrupar por estimate y real y contar las anotaciones únicas
    result = df.groupby([estimate, real], as_index=False)['annotation'].nunique()
    dupli_esti = result.query('annotation > 1')[estimate]
    dupli_real = result.query('annotation > 1')[real]

    df_duplicated = df[(df[estimate].isin(dupli_esti)) & (df[real].isin(dupli_real))]
    df_duplicated = df_duplicated.groupby('journey_id', as_index = False)['annotation'].unique()
    df_duplicated['first'] = df_duplicated['annotation'].apply(lambda x: x[0])
    df_duplicated['second'] = df_duplicated['annotation'].apply(lambda x: x[1])

    id_wrong = df_duplicated[(df_duplicated['first'].isin(['They differ','Both are the same'])) & (df_duplicated['second'].isin(['They differ','Both are the same']))]['journey_id']
    id_easy = df_duplicated[(df_duplicated['first'] == "I don't know") | (df_duplicated['second'] == "I don't know")]['journey_id']

    aux = df[df['journey_id'].isin(id_wrong)].groupby(['journey_id'], as_index = False)['annotation'].value_counts(normalize = False).query('count > 1').sort_values(['journey_id', 'count'], ascending = False).drop_duplicates(subset = ['journey_id', 'count'], keep = 'first')
    df = df[~((df['journey_id'].isin(id_easy)) & (df['annotation'] == "I don't know"))]
    df = df[~((df['journey_id'].isin(id_wrong))) & ~(pd.Series(df['annotation'] + df['journey_id']).isin(aux['annotation'] + aux['journey_id']))]
    df = df[df['annotation'] != "I don't know"]
    
    return df