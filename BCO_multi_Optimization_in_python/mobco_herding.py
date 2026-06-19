"""
Herding function for N-Objective Border Collie Optimization
"""

import numpy as np
from mobco_nondominated_sort import fast_non_dominated_sort, crowding_distance


def mobco_herding(pop, Vt, fit, n, L, acc, t):
    """
    Sort population based on Pareto rank and crowding distance
    Works for ANY number of objectives!
    """
    
    # Perform non-dominated sorting
    fronts = fast_non_dominated_sort(fit)
    
    # Calculate crowding distance for each front
    all_distances = np.zeros(n)
    for front in fronts:
        distances = crowding_distance(fit, front)
        for i, idx in enumerate(front):
            all_distances[idx] = distances[i]
    
    # Create sorting key: (rank, -crowding_distance)
    ranks = np.zeros(n)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
    
    # Sort indices by rank first, then by crowding distance (descending)
    sorted_indices = sorted(range(n), key=lambda i: (ranks[i], -all_distances[i]))
    
    # Reorder all arrays
    sorted_pop = pop[sorted_indices, :]
    sorted_Vt = Vt[sorted_indices, :]
    sorted_fit = fit[sorted_indices, :]
    sorted_acc = acc[sorted_indices, :]
    sorted_t = t[sorted_indices]
    
    return sorted_pop, sorted_Vt, sorted_fit, sorted_acc, sorted_t


def get_dogs_and_sheep(pop, Vt, fit, acc, t, n_dogs=3):
    """Separate dogs (best individuals) from sheep"""
    
    sorted_pop, sorted_Vt, sorted_fit, sorted_acc, sorted_t = mobco_herding(
        pop, Vt, fit, len(pop), pop.shape[1], acc, t
    )
    
    dogs_pop = sorted_pop[:n_dogs, :]
    dogs_Vt = sorted_Vt[:n_dogs, :]
    dogs_fit = sorted_fit[:n_dogs, :]
    dogs_acc = sorted_acc[:n_dogs, :]
    dogs_t = sorted_t[:n_dogs]
    
    sheep_pop = sorted_pop[n_dogs:, :]
    sheep_Vt = sorted_Vt[n_dogs:, :]
    sheep_fit = sorted_fit[n_dogs:, :]
    sheep_acc = sorted_acc[n_dogs:, :]
    sheep_t = sorted_t[n_dogs:]
    
    return (dogs_pop, dogs_Vt, dogs_fit, dogs_acc, dogs_t,
            sheep_pop, sheep_Vt, sheep_fit, sheep_acc, sheep_t)


if __name__ == "__main__":
    print("Testing N-Objective Herding")
    
    n, L, n_obj = 20, 5, 5
    
    pop = np.random.randn(n, L)
    Vt = np.random.randn(n, L)
    fit = np.random.rand(n, n_obj)  # 5 objectives!
    acc = np.random.randn(n, L)
    t = np.random.rand(n)
    
    sorted_pop, sorted_Vt, sorted_fit, sorted_acc, sorted_t = mobco_herding(
        pop, Vt, fit, n, L, acc, t
    )
    
    print(f"Original fitness shape: {fit.shape}")
    print(f"Sorted fitness shape: {sorted_fit.shape}")
    print(f"First individual's fitness (should be best): {sorted_fit[0]}")
    
    print("\n N-objective herding works!")
