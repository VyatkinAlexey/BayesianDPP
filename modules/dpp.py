import numpy as np
from dppy.finite_dpps import FiniteDPP

def dpp(B):
    '''
    B is the (n,d)-matrix from the paper
    '''
    U,s,_ = np.linalg.svd(B)
    eig_vals = np.zeros(len(U))
    eig_vals[:len(s)] = np.abs(s)**2
    DPP = FiniteDPP(kernel_type='correlation',
                projection=False,
                **{'K_eig_dec': (eig_vals, U)})
    return DPP