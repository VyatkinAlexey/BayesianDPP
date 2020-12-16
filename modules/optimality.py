# TODO write functions for A, C, D, V optimalities
# proposed name: optimality_{optimality name}

import numpy as np
import numpy.linalg as linalg


def posterior_covariance(sigma: float, X_s: np.ndarray, A: np.ndarray) -> np.ndarray:
    """

    :param sigma: float, variance
    :param X_s: np.ndarray, matrix of feature subset
    :param A: np.ndarray, prior precision matrix
    :return: np.ndarray: Sigma, posterior covariance matrix
    """

    to_invert = X_s.T @ X_s + A
    return sigma * sigma * linalg.inv(to_invert)


def subset_covariance(X_s: np.ndarray) -> np.ndarray:
    return X_s.T @ X_s


def optimality_A(sigma: float, X_s: np.ndarray, A: np.ndarray) -> float:
    """

    :param sigma: float, variance
    :param X_s: np.ndarray, matrix of feature subset
    :param A: np.ndarray, prior precision matrix
    :return: float, optimality value
    """

    Sigma = subset_covariance(X_s)
    to_invert = Sigma + A
    return np.float(np.trace(linalg.inv(to_invert)))


def optimality_C(sigma: float, X_s: np.ndarray, A: np.ndarray, c: np.array) -> float:
    """
    
    :param sigma: float, variance
    :param X_s: np.ndarray, matrix of feature subset
    :param A: np.ndarray, prior precision matrix
    :param c: np.array, real vector
    :return: float, optimality value
    """

    Sigma = subset_covariance(X_s)
    to_invert = Sigma + A
    return np.float(c.T @ linalg.inv(to_invert) @ c)
    

def optimality_D(sigma: float, X_s: np.ndarray, A: np.ndarray) -> float:
    """
    
    :param sigma: float, variance
    :param X_s: np.ndarray, matrix of feature subset
    :param A: np.ndarray, prior precision matrix
    :return: float, optimality value
    """

    Sigma = subset_covariance(X_s)
    to_invert = Sigma + A
    return np.float(linalg.det(linalg.inv(to_invert)) ** (1 / A.shape[0]))


def optimality_V(sigma: float, X_s: np.ndarray, A: np.ndarray, X: np.ndarray) -> float:
    """
    
    :param sigma: float, variance
    :param X_s: np.ndarray, matrix of feature subset
    :param A: np.ndarray, prior precision matrix
    :param X: np.ndarray, matrix of features
    :return: float, optimality value
    """

    Sigma = subset_covariance(X_s)
    to_invert = Sigma + A
    return np.float(np.trace(X @ linalg.inv(to_invert) @ X.T) / X.shape[0])

