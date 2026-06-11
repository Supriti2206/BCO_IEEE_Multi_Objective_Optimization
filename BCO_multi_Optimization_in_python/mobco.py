"""
Multi-Objective Border Collie Optimization (MOBCO)
USER INPUT VERSION - Works for ANY number of objectives!
"""

import numpy as np
import matplotlib.pyplot as plt
from mobco_nondominated_sort import fast_non_dominated_sort, crowding_distance, calculate_hypervolume_nD
from mobco_function_details import mobco_func_details
from mobco_generate import mobco_generate
from mobco_fitness import mobco_fitness
from mobco_herding import mobco_herding
from mobco_velocity import mobco_update_velocity
from mobco_position import mobco_update_position
from mobco_check import mobco_check
from mobco_visualization import plot_pareto_front_user


def get_user_input():
    """Get user input for optimization parameters"""
    print("WELCOME TO MULTI-OBJECTIVE BORDER COLLIE OPTIMIZATION!")    
    print("\n Please enter the following parameters:")
    print("-" * 50)
    
    # Get number of objectives
    while True:
        try:
            n_obj = int(input(" Number of objectives (2, 3, 5, 10, 20, etc.): "))
            if n_obj >= 2:
                break
            else:
                print(" Number of objectives must be at least 2! Please try again.")
        except ValueError:
            print(" Please enter a valid number!")
    
    # Get function name
    print("\n Available functions:")
    print("   For 2 objectives: zdt1, zdt2, zdt3, zdt4, zdt6")
    print("   For ANY objectives: dtlz1, dtlz2, dtlz3, dtlz4")
    
    while True:
        fname = input(f" Function name (dtlz1, dtlz2, etc.): ").strip().lower()
        if fname in ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']:
            if fname.startswith('zdt') and n_obj != 2:
                print(f" Warning: {fname} works best with 2 objectives. Using dtlz2 instead.")
                fname = 'dtlz2'
            break
        else:
            print(" Invalid function name! Try: dtlz1, dtlz2, zdt1, etc.")
    
    # Get population size
    while True:
        try:
            n = int(input(" Population size (30-100, more sheep = better exploration): "))
            if n >= 10:
                break
            else:
                print(" Population must be at least 10!")
        except ValueError:
            print(" Please enter a valid number!")
    
    # Get generations
    while True:
        try:
            gen = int(input(" Number of generations (100-500, more = better results): "))
            if gen >= 10:
                break
            else:
                print(" Generations must be at least 10!")
        except ValueError:
            print(" Please enter a valid number!")
    
    print("\n" + "=" * 50)
    print(" SETUP COMPLETE!")
    print(f"   Objectives: {n_obj}")
    print(f"   Function: {fname}")
    print(f"   Population: {n}")
    print(f"   Generations: {gen}")
    print("=" * 50)
    
    return n_obj, fname, n, gen


def mobco_user(n_obj, fname, n=30, gen=200, archive_size=100, verbose=True, visualize=True):
    """
    MOBCO with user-specified number of objectives!
    """
    
    # Get function details
    fobj, lb, ub, dim = mobco_func_details(fname, n_obj)
    
    if verbose:
        print("\n" + "=" * 70)
        print(f" MOBCO RUNNING WITH {n_obj} OBJECTIVES!")
        print("=" * 70)
        print(f"Function: {fname}")
        print(f"Dimensions: {dim}")
        print(f"Population: {n} (3 dogs + {n-3} sheep)")
        print(f"Generations: {gen}")
        print(f"Search space: [{lb}, {ub}]")
        print("=" * 70)
    
    # Initialize population
    population, acc = mobco_generate(n, dim, ub, lb)
    Vt = np.zeros((n, dim))
    t = np.random.rand(n)
    k_counter = 0
    
    # External archive
    archive = []
    archive_fitness = []
    history = []
    hv_history = []
    
    for g in range(gen):
        # Calculate fitness
        fitness_values = mobco_fitness(population, n, dim, fobj, n_obj)
        
        # Non-dominated sorting
        fronts = fast_non_dominated_sort(fitness_values)
        for front in fronts:
            crowding_distance(fitness_values, front)
        
        # Update archive
        all_solutions = np.vstack([archive, population]) if len(archive) > 0 else population
        all_fitness = np.vstack([archive_fitness, fitness_values]) if len(archive_fitness) > 0 else fitness_values
        
        combined_fronts = fast_non_dominated_sort(all_fitness)
        
        new_archive = []
        new_archive_fitness = []
        for front in combined_fronts:
            for idx in front:
                if len(new_archive) < archive_size:
                    new_archive.append(all_solutions[idx])
                    new_archive_fitness.append(all_fitness[idx])
                else:
                    break
            if len(new_archive) >= archive_size:
                break
        
        archive = np.array(new_archive)
        archive_fitness = np.array(new_archive_fitness)
        
        # Store history
        if g % 20 == 0 or g == gen - 1:
            hv = calculate_hypervolume_nD(archive_fitness) if len(archive_fitness) > 0 else 0
            hv_history.append(hv)
            history.append({
                'gen': g,
                'fitness': archive_fitness.copy(),
                'size': len(archive),
                'hv': hv
            })
        
        # Eyeing condition
        eye = 0
        if g > 0 and len(hv_history) > 1:
            if hv_history[-1] <= hv_history[-2]:
                k_counter += 1
                if k_counter >= 5:
                    eye = 1
                    k_counter = 0
            else:
                k_counter = 0
        
        # Herding
        population, Vt, fitness_values, acc, t = mobco_herding(
            population, Vt, fitness_values, n, dim, acc, t
        )
        
        # Update velocities and positions
        Vt, acc, t, gathered_idx, stalked_idx = mobco_update_velocity(
            Vt, n, dim, acc, t, population, fitness_values, eye, 0, 1, 2
        )
        
        population = mobco_update_position(population, Vt, t, acc, n, dim, eye)
        population, acc, t, Vt = mobco_check(population, n, dim, ub, lb, acc, Vt, t)
        
        if verbose and (g % 50 == 0 or g == gen - 1):
            progress = int((g+1) / gen * 40)
            #bar = "█" * progress + "░" * (40 - progress)
            print(f"\rProgress: {g+1}/{gen} | Archive: {len(archive)} | HV: {hv_history[-1]:.4f}" if hv_history else f"\rProgress: [{bar}] {g+1}/{gen} | Archive: {len(archive)}", end="")
    
    print()  # New line after progress bar
    
    # Visualize results
    if visualize and len(archive_fitness) > 0:
        plot_pareto_front_user(archive_fitness, fname, hv_history, history, n_obj)
    
    if verbose:
        print("\n" + "=" * 70)
        print(f" OPTIMIZATION COMPLETE!")
        print(f"   Found {len(archive)} Pareto-optimal solutions")
        print(f"   Each solution has {n_obj} objective values")
        if hv_history:
            print(f"   Final Hypervolume: {hv_history[-1]:.6f}")
        print("=" * 70)
    
    return archive, archive_fitness, history


def main():
    """Main function with user input"""
    
    print("BORDER COLLIE OPTIMIZATION - MULTI-OBJECTIVE")
    
    # Get user input
    n_obj, fname, n, gen = get_user_input()
    
    # Run optimization
    archive, fitness, history = mobco_user(
        n_obj=n_obj,
        fname=fname,
        n=n,
        gen=gen,
        verbose=True,
        visualize=True
    )
    
    # Show summary
    print("FINAL RESULTS SUMMARY")
    print(f"\n Number of objectives: {n_obj}")
    print(f" Solutions found: {len(archive)}")
    print(f" Best hypervolume: {history[-1]['hv']:.6f}" if history else "")
    
    # Show first few solutions
    if len(fitness) > 0:
        print(f"\n First 5 Pareto-optimal solutions:")
        print("-" * 60)
        for i in range(min(5, len(fitness))):
            obj_str = ", ".join([f"{fitness[i, j]:.4f}" for j in range(min(5, n_obj))])
            if n_obj > 5:
                obj_str += f", ... ({n_obj} total)"
            print(f"   Solution {i+1}: [{obj_str}]")
    
    print("THANK YOU FOR USING MOBCO!")


if __name__ == "__main__":
    main()