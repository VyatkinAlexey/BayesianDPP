from typing import Tuple

import numpy as np

from modules.utils import _get_optimalty


def select_bottom_up(sigma: float,
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
    :return: indexes of samples subset, optimality for them
    """
    num_samples = X.shape[0]
    assert num_samples >= k, f'number of samples should be greater than k'
    selected_ixs = []
    full_samples_set = set(list(range(num_samples)))
    optimalty_func = _get_optimalty(optimality)
    global_score = None
    for i in range(k):
        candidate_samples = list(full_samples_set - set(selected_ixs))
        current_optimalty = np.Inf
        optimal_sample = None
        for candidate_sample in candidate_samples:
            candidate_ixs = np.append(selected_ixs, [candidate_sample]).astype(int)
            candidate_optimality = optimalty_func(sigma, X[candidate_ixs], A)
            if candidate_optimality < current_optimalty:
                current_optimalty = candidate_optimality
                optimal_sample = candidate_sample
        if optimal_sample is not None:
            selected_ixs.append(optimal_sample)
        if i == k-1:
            global_score = current_optimalty

    return np.array(selected_ixs), global_score
