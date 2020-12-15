import pandas as pd
import numpy as np

def preprocess(s, i):
    try:
        if i == 0:
            return float(s.split(' ')[i])
        else:
            return float(s.split(' ')[i].split(':')[-1])
    except ValueError:
        return np.nan
    except IndexError:
        return np.nan
    
def make_csv_ga(path, d):
    df = pd.read_csv(path, header=None)
    for i in range(d+1):
        if i == 0:
            df['target'] = df[0].apply(lambda x: preprocess(x, i))
        else:
            df[str(i)] = df[0].apply(lambda x: preprocess(x, i))
    df = df.drop(columns = [0])
    return df

ALL_DATASETS = ["mg_scale", "mpg_scale", "housing_scale", "bodyfat_scale"]

def download_data(dataset_name):
    '''
    Possible options for name: "mg_scale", "mpg_scale", "housing_scale", "bodyfat_scale".
    Returns the given dataset.
    '''
    
    data_info = {'mg_scale': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mg_scale', 6),
             'mpg_scale': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg_scale', 7),
             'housing_scale': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',13),
             'bodyfat_scale': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat_scale',14)}
    
    url = data_info[dataset_name][0]
    d = data_info[dataset_name][1]
    return make_csv_ga(url, d)