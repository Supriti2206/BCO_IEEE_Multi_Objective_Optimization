"""
Velocity, Time, and Acceleration update for N-Objective MOBCO
Works with ANY number of objectives!
"""

import numpy as np
from mobco_nondominated_sort import dominates


def compare_fitness_multi(fit1, fit2):
    """
    Compare two multi-objective fitness vectors (any dimension!)
    Returns: 1 if fit1 is better, -1 if fit2 is better, 0 if non-dominated
    """
    if dominates(fit1, fit2):
        return 1  # fit1 dominates fit2 (fit1 is better)
    elif dominates(fit2, fit1):
        return -1  # fit2 dominates fit1 (fit2 is better)
    else:
        return 0  # non-dominated


def mobco_update_velocity(Vt, n, L, a, t, pop, fit, eye_flag, 
                          lead_idx=0, left_idx=1, right_idx=2):
    """
    Update velocities based on multi-objective dominance
    Works for ANY number of objectives!
    """
    
    Vt1 = Vt.copy()
    acc1 = np.zeros((n, L))
    t1 = np.zeros(n)
    
    gathered_idx = []
    stalked_idx = []
    
    # Fitness of dogs (can be any dimension!)
    fit_lead = fit[lead_idx]
    fit_left = fit[left_idx]
    fit_right = fit[right_idx]
    
    for i in range(n):
        for j in range(L):
            # Update dogs (first 3 individuals)
            if i < 3:
                val = Vt[i, j]**2 + 2 * a[i, j] * pop[i, j]
                Vt1[i, j] = np.sqrt(max(val, 1e-10))
            
            # Update sheep
            else:
                # Eyeing behavior
                if eye_flag == 1:
                    # Find the WORST dog using Pareto comparison
                    comp_left_right = compare_fitness_multi(fit_left, fit_right)
                    
                    if comp_left_right == 1:  # left is better than right
                        worst_dog = right_idx
                    elif comp_left_right == -1:  # right is better than left
                        worst_dog = left_idx
                    else:
                        # Non-dominated: use sum of objectives as tie-breaker
                        if np.sum(fit_left) > np.sum(fit_right):
                            worst_dog = left_idx
                        else:
                            worst_dog = right_idx
                    
                    val = Vt1[worst_dog, j]**2 + 2 * a[worst_dog, j] * pop[i, j]
                    Vt1[i, j] = np.sqrt(max(val, 1e-10))
                
                else:
                    # Check if sheep is dominated by lead dog
                    if dominates(fit_lead, fit[i]):
                        # GATHERING behavior
                        val = Vt1[lead_idx, j]**2 + 2 * a[lead_idx, j] * pop[i, j]
                        Vt1[i, j] = np.sqrt(max(val, 1e-10))
                        if i not in gathered_idx:
                            gathered_idx.append(i)
                    
                    # Check if dominated by side dogs
                    elif dominates(fit_left, fit[i]) or dominates(fit_right, fit[i]):
                        # STALKING behavior
                        angle1 = np.random.randint(1, 90)
                        angle2 = np.random.randint(91, 180)
                        
                        term1 = (Vt1[right_idx, j] * np.tan(np.radians(angle1)))**2 + 2 * a[right_idx, j] * pop[right_idx, j]
                        term2 = (Vt1[left_idx, j] * np.tan(np.radians(angle2)))**2 + 2 * a[left_idx, j] * pop[left_idx, j]
                        
                        val1 = np.sqrt(max(term1, 1e-10))
                        val2 = np.sqrt(max(term2, 1e-10))
                        Vt1[i, j] = (val1 + val2) / 2
                        
                        if i not in stalked_idx:
                            stalked_idx.append(i)
                    
                    else:
                        # MIXED behavior (average of gathering and stalking)
                        val_lead = Vt1[lead_idx, j]**2 + 2 * a[lead_idx, j] * pop[i, j]
                        v_lead = np.sqrt(max(val_lead, 1e-10))
                        
                        # Use default angles for mixed behavior
                        term_side1 = (Vt1[right_idx, j] * np.tan(np.radians(45)))**2 + 2 * a[right_idx, j] * pop[right_idx, j]
                        term_side2 = (Vt1[left_idx, j] * np.tan(np.radians(135)))**2 + 2 * a[left_idx, j] * pop[left_idx, j]
                        v_side = (np.sqrt(max(term_side1, 1e-10)) + np.sqrt(max(term_side2, 1e-10))) / 2
                        
                        Vt1[i, j] = (v_lead + v_side) / 2
    
    # Update acceleration and time
    for i in range(n):
        t_safe = max(t[i], 1e-10)
        for j in range(L):
            acc1[i, j] = abs(Vt1[i, j] - Vt[i, j]) / t_safe
        
        s_sum = 0
        for j in range(L):
            if acc1[i, j] > 1e-10:
                s_sum += (Vt1[i, j] - Vt[i, j]) / acc1[i, j]
        t1[i] = abs(np.mean(s_sum)) if s_sum != 0 else t[i]
    
    return Vt1, acc1, t1, gathered_idx, stalked_idx


if __name__ == "__main__":
    print("=" * 70)
    print("Testing N-Objective Velocity Update")
    print("=" * 70)
    
    # Test with 5 objectives!
    n, L, n_obj = 10, 5, 5
    
    Vt = np.random.randn(n, L) * 0.5
    a = np.random.randn(n, L) * 0.1
    t = np.random.uniform(0.5, 2.0, n)
    pop = np.random.randn(n, L) * 5
    fit = np.random.rand(n, n_obj)  # 5 objectives!
    
    # Make dogs better
    fit[0, :] = [0.1, 0.2, 0.1, 0.3, 0.2]  # Lead dog
    fit[1, :] = [0.3, 0.1, 0.4, 0.2, 0.1]  # Left dog
    fit[2, :] = [0.2, 0.4, 0.1, 0.1, 0.4]  # Right dog
    
    print(f"Fitness shape: {fit.shape} ({n_obj} objectives!)")
    print(f"Lead dog fitness (5 objectives): {fit[0]}")
    
    Vt1, acc1, t1, gathered, stalked = mobco_update_velocity(
        Vt, n, L, a, t, pop, fit, eye_flag=0
    )
    
    print(f" Velocity update works with {n_obj} objectives!")
    print(f"Gathered: {len(gathered)} sheep, Stalked: {len(stalked)} sheep")