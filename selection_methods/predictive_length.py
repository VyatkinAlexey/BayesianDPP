from typing import Tuple

import numpy as np

from modules.utils import _get_optimalty


def select_predictive_length(X: np.ndarray,
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
    probs = np.linalg.norm(X, axis=1)
    probs = probs / np.sum(probs)
    selected_ixs = np.random.choice(num_samples,
                                    size=k,
                                    replace=False,
                                    p=probs)
    optimality_value = optimalty_func(X_s=X[selected_ixs], A=A, X=X)
    return selected_ixs, optimality_value
