import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import json

import streamlit as st

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

st.write(df.shape)
duplicados = pa.TableGroupBy(df, 'journey_id').aggregate([('annotation', 'count')]).to_pandas()
duplicados = pa.array(duplicados[duplicados['annotation_count'] > 1]['journey_id'].unique())

filtro = pc.is_in(df.column('journey_id'), duplicados)
aux = df.filter(filtro)

st.write(aux.shape)
aux = aux.to_pandas().groupby('journey_id', as_index = False)['annotation'].unique()
st.write(aux)
st.write('In several cases the duplicated journeys ends up with the same tag')
def get_unique_duplicates(x):
    try:
        x = x[1]
        x = None
    except:
        x = x[0]
    return x

no_unicos = aux[aux['annotation'].apply(lambda x: get_unique_duplicates(x)).isna()]
unicos = aux[~aux['annotation'].apply(lambda x: get_unique_duplicates(x)).isna()]

st.write(no_unicos['annotation'].astype(str).value_counts())
st.write(no_unicos['annotation'].astype(str).value_counts(normalize = True))
st.write('''In the 37% of the cases the tag are contradictory. So is possible to drop those cases when the tag is "I don't know" and analyze those which the tag is still duplicated and not unique''')

no_unicos = no_unicos[~no_unicos['annotation'].astype(str).str.contains('''I don't know''')]
st.write(no_unicos['annotation'].astype(str).value_counts())
st.write('Now is possible to find a solution for those cases out or simply drop the cases to ensure the consistency of the data')

trial = no_unicos.reset_index(drop = True)
trial['annotation'] = trial['annotation'].astype(str)
st.write(trial['annotation'].value_counts(normalize = True))

st.write('There are too many external variables which makes impossible to know how to keep those duplicates. Due to the unkwonledge about the quality of the annotations if more secure to drop those cases.')

aux = df.filter(filtro).to_pandas()
unicos = aux[aux['journey_id'].isin(unicos['journey_id'].unique())].drop_duplicates(subset = 'journey_id')
buenos = df.filter(pc.invert(filtro)).to_pandas()
df_final = pd.concat([buenos, unicos], axis = 0).reset_index(drop = True)
st.write(df_final['annotation'].value_counts(normalize=True))

st.write('To ensure there is no problem whit the known journeys is better to drop those cases')
df_final = df_final[df_final['annotation'] != "I don't know"]
st.write(df_final['annotation'].value_counts(normalize=True))
st.write('Finally the result is pretty balance')
