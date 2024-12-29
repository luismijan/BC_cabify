import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import json

import os

from utils.functions import *
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

def cleaning_data(df, id_var, target_var):
    df = pa.Table.from_pandas(df)

    duplicados = pa.TableGroupBy(df, id_var).aggregate([(target_var, 'count')]).to_pandas()
    duplicados = pa.array(duplicados[duplicados['annotation_count'] > 1][id_var].unique())

    filtro = pc.is_in(df.column(id_var), duplicados)
    aux = df.filter(filtro)

    # st.write(aux.shape)
    aux = aux.to_pandas().groupby(id_var, as_index = False)[target_var].unique()

    no_unicos = aux[aux[target_var].apply(lambda x: get_unique_duplicates(x)).isna()]
    unicos = aux[~aux[target_var].apply(lambda x: get_unique_duplicates(x)).isna()]

    no_unicos = no_unicos[~no_unicos[target_var].astype(str).str.contains('''I don't know''')]

    trial = no_unicos.reset_index(drop = True)
    trial[target_var] = trial[target_var].astype(str)
    
    aux = df.filter(filtro).to_pandas()
    unicos = aux[aux[id_var].isin(unicos[id_var].unique())].drop_duplicates(subset = id_var)
    buenos = df.filter(pc.invert(filtro)).to_pandas()
    df_final = pd.concat([buenos, unicos], axis = 0)
    
    df_final = df_final[df_final[target_var] != "I don't know"]
    
    return df_final
