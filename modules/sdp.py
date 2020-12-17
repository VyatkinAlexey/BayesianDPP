import numpy as np
import cvxpy as cp

def sdp(X, A, k, optimalty_func):
    '''

    Find optimal probabilities to minimize optimality functional
    :param X is the (n,d)-matrix of features
    :param k: int, number of samples to select
    :param optimalty_func: optimality functional
    '''

    n = X.shape[0]
    p = cp.Variable(n)

    func = lambda p: optimalty_func(Sigma=(X.T * np.array(p.value).astype(float)) @ X, A=A, X=X)
    
    constraints = [0 <= p[i] for i in range(n)]
    constraints += [p[i] <= 1 for i in range(n)]
    constraints += [sum(p) == k]

    prob = cp.Problem(cp.Minimize(func(p)), constraints)
    prob.solve()
    return np.array(p.value).astype(float)
