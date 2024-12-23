import pandas as pd
import numpy as np
import pyarrow as pa

import configparser

import logging

from utils.read_clean_data import *
from utils.var_generator import *

config = configparser.ConfigParser()
config.read('./utils/config.ini')

file = config["data"]["file"]
real = config["data"]["real"]
estimate = config["data"]["estimate"]
id_var = config["data"]["id_var"]
target_var = config["data"]["target_var"]
file_transform = config["modelling"]["file_transform"]

if __name__ == "__main__":
    print(file_transform)
    
    # logging.INFO('Starting process...')
    print('Starting process...')
    data = reading_data(file)
    # logging.INFO('Data read')
    print('Data read')
    data_clean = cleaning_data(data, real, estimate)
    # logging.INFO('Data cleaned')
    print('Data cleaned')
    df = generate_var(data_clean, real, estimate, id_var, target_var).get_pandas_dataframe()
    # logging.INFO('Features generated')
    print(df.head())
    print('Features generated')
    df.to_feather(file_transform)
    print('DF saved')

          