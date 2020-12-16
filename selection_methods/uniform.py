from typing import Tuple

import numpy as np

from modules.utils import _get_optimalty
from modules.optimality import subset_covariance

def select_uniform(X: np.ndarray,
                   A: np.ndarray,
                   k: int,
                   optimality: str = 'A') -> Tuple[np.ndarray, float]:
    """

    :param X: np.ndarray, matrix of features
    :param A: np.ndarray, prior precision matrix
    :param k: int, number of samples to select
    :param optimality: str, optimality type, one of ["A", "C", "D", "V"]
    :return: indexes of samples subset
    """
    optimalty_func = _get_optimalty(optimality)
    num_samples = X.shape[0]
    assert num_samples >= k, f'number of samples should be greater than k'
    selected_ixs = np.random.choice(num_samples, size=k, replace=False)
    optimality_value = optimalty_func(Sigma=subset_covariance(X[selected_ixs]), A=A, X=X)
    return selected_ixs, optimality_value