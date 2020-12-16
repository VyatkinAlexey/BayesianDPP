from typing import Tuple

import numpy as np
from scipy.linalg import fractional_matrix_power as frac_power
from modules.utils import _get_optimalty
from modules.optimality import subset_covariance
from modules.dpp import dpp
    

def select_proposed(sigma: float,
                     X: np.ndarray,
                     A: np.ndarray,
                     k: int,
                     optimality: str = 'A') -> Tuple[np.ndarray, float]:
    """

    :param sigma: float, variance
    :param X: np.ndarray, matrix of features
    :param A: np.ndarray, prior precision matrix
    :param k: int, number of samples to select
    :param optimality: str, optimality type, one of ["A", "C", "D", "V"]
    :return: indexes of samples subset
    """
    optimalty_func = _get_optimalty(optimality)
    num_samples = X.shape[0]
    assert num_samples >= k, f'number of samples should be greater than k'
    p = k/num_samples * np.ones(num_samples)
    Z = A + (X.T * p) @ X
    B = (np.sqrt(p[:,None]) * X) @ frac_power(Z,-0.5)
    DPP = dpp(B)
    DPP.sample_exact()
    all_n = np.arange(num_samples)
    b = set(all_n[np.random.uniform(size = num_samples) < p])
    selected_ixs = set(DPP.list_of_samples[0]) | b
    selected_ixs = np.array(list(selected_ixs))
    optimality_value = optimalty_func(Sigma=subset_covariance(X[selected_ixs]), A, X)
    return selected_ixs, optimality_value