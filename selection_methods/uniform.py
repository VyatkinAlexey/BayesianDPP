import numpy as np


def select_uniform(sigma: float,
                   X: np.ndarray,
                   A: np.ndarray,
                   k: int) -> np.ndarray:
    """

    :param sigma: float, variance
    :param X: np.ndarray, matrix of features
    :param A: np.ndarray, prior precision matrix
    :param k: int, number of samples to select
    :return: indexes of samples subset
    """
    num_samples = X.shape[0]
    assert num_samples >= k, f'number of samples should be greater than k'
    selected_ixs = np.random.choice(list(range(num_samples)), size=k, replace=False)

    return selected_ixs