import pandas as pd

def load_datasets():
    df = pd.read_csv("train.csv")
    dt = pd.read_csv("test.csv")
    return df, dt

def load_config(): 
    pd.set_option('display.max_rows', None)