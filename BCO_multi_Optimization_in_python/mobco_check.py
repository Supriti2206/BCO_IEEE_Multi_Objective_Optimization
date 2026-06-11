"""
Bounds checking for N-Objective MOBCO
No changes needed - bounds don't depend on number of objectives!
"""

import numpy as np


def mobco_check(pop, n, L, ub, lb, acc, Vt, t):
    """
    Check and correct invalid values
    Works the SAME for ANY number of objectives!
    """
    
    ub = np.array(ub)
    lb = np.array(lb)
    
    if ub.ndim == 0:
        ub = np.full(L, ub)
        lb = np.full(L, lb)
    
    pop1 = pop.copy()
    acc1 = acc.copy()
    t1 = t.copy()
    Vt1 = Vt.copy()
    
    # Check population positions
    for i in range(n):
        for j in range(L):
            if pop[i, j] >= ub[j] or pop[i, j] <= lb[j] or pop[i, j] == 0:
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
                Vt1[i, j] = np.random.randn() * 0.5
    
    # Check acceleration
    for i in range(n):
        for j in range(L):
            if np.isnan(acc[i, j]) or acc[i, j] == 0:
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
                Vt1[i, j] = np.random.randn() * 0.5
    
    # Check velocity
    for i in range(n):
        for j in range(L):
            if np.isnan(Vt[i, j]) or Vt[i, j] == 0:
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
                Vt1[i, j] = np.random.randn() * 0.5
    
    # Check time
    for i in range(n):
        if np.isnan(t1[i]) or t1[i] == 0:
            for j in range(L):
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                Vt1[i, j] = np.random.randn() * 0.5
            t1[i] = np.random.rand()
    
    return pop1, acc1, t1, Vt1


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Bounds Check")
    print("=" * 70)
    
    n, L = 10, 5
    pop = np.random.randn(n, L) * 100
    acc = np.random.randn(n, L)
    Vt = np.random.randn(n, L)
    t = np.random.rand(n)
    
    pop_corr, acc_corr, t_corr, Vt_corr = mobco_check(pop, n, L, 10, -10, acc, Vt, t)
    
    print(f" Bounds check works!")
    print(f"Original min/max: {np.min(pop):.2f}/{np.max(pop):.2f}")
    print(f"Corrected min/max: {np.min(pop_corr):.2f}/{np.max(pop_corr):.2f}")