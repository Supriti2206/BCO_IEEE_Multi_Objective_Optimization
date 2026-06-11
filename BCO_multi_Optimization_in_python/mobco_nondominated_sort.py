"""
Non-dominated sorting and crowding distance for N objectives
"""

import numpy as np

def fast_non_dominated_sort(fitness_values):
    """Perform fast non-dominated sorting"""
    n = len(fitness_values)
    domination_count = np.zeros(n)
    dominated_solutions = [[] for _ in range(n)]
    fronts = []
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(fitness_values[i], fitness_values[j]):
                    dominated_solutions[i].append(j)
                elif dominates(fitness_values[j], fitness_values[i]):
                    domination_count[i] += 1
        
        if domination_count[i] == 0:
            if not fronts:
                fronts.append([i])
            else:
                fronts[0].append(i)
    
    front_idx = 0
    while len(fronts[front_idx]) > 0:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        front_idx += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    
    return fronts


def crowding_distance(fitness_values, front_indices):
    """Calculate crowding distance"""
    n_obj = fitness_values.shape[1]
    n_front = len(front_indices)
    distances = np.zeros(n_front)
    
    if n_front <= 2:
        distances[:] = np.inf
        return distances
    
    for obj in range(n_obj):
        sorted_indices = sorted(front_indices, key=lambda idx: fitness_values[idx, obj])
        distances[0] = np.inf
        distances[-1] = np.inf
        
        f_max = fitness_values[sorted_indices[-1], obj]
        f_min = fitness_values[sorted_indices[0], obj]
        
        if f_max != f_min:
            for i in range(1, n_front - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                dist = (fitness_values[next_idx, obj] - fitness_values[prev_idx, obj]) / (f_max - f_min)
                distances[i] += dist
    
    return distances


def dominates(a, b):
    """Check if a dominates b"""
    return np.all(a <= b) and np.any(a < b)


def calculate_hypervolume_nD(fitness_points, reference_point=None, n_samples=10000):
    """Calculate hypervolume for N objectives"""
    if len(fitness_points) == 0:
        return 0
    
    n_obj = fitness_points.shape[1]
    
    if reference_point is None:
        reference_point = np.max(fitness_points, axis=0) + 0.1
    
    if n_obj == 2:
        sorted_points = fitness_points[np.argsort(fitness_points[:, 0])]
        hv = 0
        for i in range(len(sorted_points)):
            if i == 0:
                hv = abs((sorted_points[i, 0] - reference_point[0]) * (sorted_points[i, 1] - reference_point[1]))
            else:
                hv += abs((sorted_points[i, 0] - sorted_points[i-1, 0]) * (sorted_points[i, 1] - reference_point[1]))
        return abs(hv)
    
    else:
        # Monte Carlo sampling for higher dimensions
        random_points = np.random.rand(n_samples, n_obj)
        for j in range(n_obj):
            random_points[:, j] = random_points[:, j] * reference_point[j]
        
        dominated_count = 0
        for point in random_points:
            is_dominated = False
            for sol in fitness_points:
                if np.all(sol <= point):
                    is_dominated = True
                    break
            if is_dominated:
                dominated_count += 1
        
        total_volume = np.prod(reference_point)
        return (dominated_count / n_samples) * total_volume