import numpy as np
from numpy import linalg as LA
from modules.utils import _get_optimalty


def upper_estimation(k: int, Sigma: np.ndarray,
                     A: np.ndarray, X: np.ndarray, opt_type: str):
    n, d = X.shape
    covariance_matrix = X.T @ X
    d_nk = (k/n * covariance_matrix @ LA.inv(k/n * covariance_matrix + A)).trace()
    optimalty = _get_optimalty(opt_type)

    OPT = (1 + 8*d_nk/k + 8*(np.log(k/d_nk)/k)**0.5) * \
          optimalty(Sigma=Sigma, A=A, X=X)

    return OPT
