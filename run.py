import argparse

import numpy as np

from selection_methods.uniform import select_uniform
from selection_methods.bottom_up import select_bottom_up
from modules.data import download_data, ALL_DATASETS
from modules.optimality import optimality_A



def parse_args():
    parser = argparse.ArgumentParser(description='Script to perform experiment')
    parser.add_argument('-d', '--dataset', type=str,
                        default='mg_scale',
                        choices=ALL_DATASETS,
                        help='name of given dataset')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    dataset_name = args.dataset
    print(f'Experiment for {dataset_name} starts')
    df = download_data(dataset_name)
    print(df.shape)

    # just test of optimality function
    X = df.values
    n, d = X.shape
    A = (1/n) * np.eye(d)
    sigma = 1 # ?
    k_linspace = np.arange(start=d, stop=5*d+1, step=1)
    print('Num samples' + '\t' + 'score')
    for k in k_linspace:
        selected_samples, score = select_bottom_up(sigma=sigma, X=X, A=A, k=k)
        #print(k, f'{optimality_A(sigma, X[selected_samples], A):.3f}', end='\t')
        print(k, '\t', score)
    # TODO add experiment pipeline there