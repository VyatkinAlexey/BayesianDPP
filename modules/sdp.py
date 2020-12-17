import numpy as np
from scipy.optimize import LinearConstraint, minimize

def sdp(X, A, k, optimalty_func):
    '''

    Find optimal probabilities to minimize optimality functional
    :param X is the (n,d)-matrix of features
    :param k: int, number of samples to select
    :param optimalty_func: optimality functional
    '''

    n = X.shape[0]
    con1 = LinearConstraint(np.identity(n), np.zeros(n), np.ones(n))
    con2 = LinearConstraint(np.ones(n), k, k)
    p0 = k / n * np.ones(n)

    func = lambda p: optimalty_func(Sigma=(X.T * p) @ X, A=A, X=X)
    res = minimize(func, p0, constraints=(con1, con2), method='trust-krylov')

    return res.x
