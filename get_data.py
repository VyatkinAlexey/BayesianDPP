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

data_info = {'mg_scale.csv': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mg_scale', 6),
             'mpg_scale.csv': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg_scale', 7),
             'housing_scale.csv': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',13),
             'bodyfat_scale.csv': ('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat_scale',14)}

if __name__ == '__main__':
    
    for name in data_info:
        url = data_info[name][0]
        d = data_info[name][1]
        df = make_csv_ga(url, d)
#         df = df.drop(columns = ['Unnamed: 0'])
        df.to_csv(name, index = False)
