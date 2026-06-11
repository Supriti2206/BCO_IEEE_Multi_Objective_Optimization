"""
Position Update for N-Objective MOBCO
No changes needed - positions don't depend on number of objectives!
"""

import numpy as np


def mobco_update_position(pop, Vt, t, acc, n, L, eye_flag):
    """
    Update positions of dogs and sheep
    Works the SAME for ANY number of objectives!
    (Because positions are independent of objectives)
    """
    
    pop1 = np.zeros((n, L))
    
    for i in range(n):
        for j in range(L):
            if i < 3:  # Dogs
                pop1[i, j] = Vt[i, j] * t[i] + 0.5 * acc[i, j] * (t[i] ** 2)
            else:  # Sheep
                if eye_flag == 1:
                    pop1[i, j] = Vt[i, j] * t[i] - 0.5 * acc[i, j] * (t[i] ** 2)
                else:
                    pop1[i, j] = Vt[i, j] * t[i] + 0.5 * acc[i, j] * (t[i] ** 2)
    
    return pop1


def mobco_update_position_vectorized(pop, Vt, t, acc, n, L, eye_flag):
    """Vectorized version - faster!"""
    
    t_reshaped = t.reshape(-1, 1)
    displacement = Vt * t_reshaped
    acc_contribution = 0.5 * acc * (t_reshaped ** 2)
    
    pop1 = np.zeros((n, L))
    
    if n >= 3:
        pop1[0:3, :] = displacement[0:3, :] + acc_contribution[0:3, :]
    
    if n > 3:
        if eye_flag == 1:
            pop1[3:, :] = displacement[3:, :] - acc_contribution[3:, :]
        else:
            pop1[3:, :] = displacement[3:, :] + acc_contribution[3:, :]
    
    return pop1


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Position Update")
    print("=" * 70)
    
    n, L = 10, 5
    Vt = np.random.randn(n, L)
    t = np.random.rand(n)
    acc = np.random.randn(n, L)
    pop = np.random.randn(n, L)
    
    pop_new = mobco_update_position(pop, Vt, t, acc, n, L, eye_flag=0)
    print(f"✅ Position update works! Output shape: {pop_new.shape}")