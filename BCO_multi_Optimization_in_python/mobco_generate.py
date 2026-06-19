"""
Population initialization for N-Objective MOBCO
"""

import numpy as np


def mobco_generate(n, L, ub, lb):
    """
    Generate initial random population
    Works the SAME for ANY number of objectives!
    """
    
    ub = np.array(ub)
    lb = np.array(lb)
    
    if ub.ndim == 0:
        ub = np.full(L, ub)
        lb = np.full(L, lb)
    
    M = np.zeros((n, L))
    for i in range(L):
        M[:, i] = np.random.rand(n) * (ub[i] - lb[i]) + lb[i]
    
    acc = np.random.rand(n, L)
    
    return M, acc


if __name__ == "__main__":
    print("Testing Population Generation")
    
    pop, acc = mobco_generate(30, 10, 1, 0)
    print(f" Population generation works! Shape: {pop.shape}")
