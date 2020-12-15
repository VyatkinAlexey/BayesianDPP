import argparse

import numpy as np

from selection_methods.uniform import select_uniform
from selection_methods.bottom_up import select_bottom_up



def parse_args():
    parser = argparse.ArgumentParser(description='Script to perform experiment')
    parser.add_argument('-d', '--dataset', type=str,
                        default='libsvm',
                        help='name of given dataset')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    dataset_name = args.dataset
    print(f'Experiment for {dataset_name} starts')

    '''
    # just test of optimality function
    n = 10 # num samples
    d = 5 # sample dimension
    X = np.random.randn(n ,d)
    A = np.eye(d)
    sigma = 2
    k = 3

    selected_samples = select_bottom_up(sigma=sigma, X=X, A=A, k=k)
    print(selected_samples)
    '''
    # TODO add experiment pipeline there