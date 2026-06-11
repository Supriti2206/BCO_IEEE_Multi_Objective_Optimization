"""
Fitness calculation for N-Objective MOBCO
Returns array of ANY size!
"""

import numpy as np


def mobco_fitness(population, n, dim, fobj, n_obj):
    """
    Calculate multi-objective fitness for all individuals
    Returns n x n_obj array (works for ANY n_obj!)
    """
    
    fitness = np.zeros((n, n_obj))
    
    for i in range(n):
        fitness[i, :] = fobj(population[i, :])
    
    return fitness


if __name__ == "__main__":
    print("=" * 70)
    print("Testing N-Objective Fitness")
    print("=" * 70)
    
    from mobco_function_details import dtlz2
    
    n, dim, n_obj = 10, 14, 5  # 5 objectives!
    pop = np.random.rand(n, dim)
    
    fitness = mobco_fitness(pop, n, dim, lambda x: dtlz2(x, n_obj), n_obj)
    
    print(f" Fitness calculation works!")
    print(f"Population: {n} individuals, {dim} dimensions")
    print(f"Fitness shape: {fitness.shape} ({n_obj} objectives!)")
    print(f"First individual's fitness (5 values): {fitness[0]}")