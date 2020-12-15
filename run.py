import argparse

import numpy as np

from modules.optimality import optimality_A


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
    # just test of optimality function
    n = 10 # num samples
    d = 5 # sample dimension
    X = np.random.randn(n ,d)
    A = np.eye(d)
    sigma = 2
    current_optimality = optimality_A(sigma, X[3: 5], A)
    print(f'optimalty A for given X, A, sigma: {current_optimality:.4f}')

    # TODO add experiment pipeline there