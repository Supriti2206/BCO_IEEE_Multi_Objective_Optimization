"""
Visualization for N-Objective MOBCO - FULLY FIXED for ANY number of objectives!
Works for 2, 3, 5, 10, 17, 20, 50, 100+ objectives!
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pareto_front_user(archive_fitness, fname, hv_history, history, n_obj):
    """
    Choose the right plot based on number of objectives
    """
    if n_obj == 2:
        plot_2d_pareto_user(archive_fitness, fname, hv_history, history)
    elif n_obj == 3:
        plot_3d_pareto_user(archive_fitness, fname, hv_history, history)
    else:
        plot_nd_pareto_user(archive_fitness, fname, hv_history, history, n_obj)


def plot_2d_pareto_user(archive_fitness, fname, hv_history, history):
    """
    Plot for 2 objectives
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Final Pareto Front
    ax1 = axes[0, 0]
    if len(archive_fitness) > 0:
        ax1.scatter(archive_fitness[:, 0], archive_fitness[:, 1], 
                   c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Objective 1 (f₁)', fontsize=12)
        ax1.set_ylabel('Objective 2 (f₂)', fontsize=12)
        ax1.set_title(f' Pareto Front - {fname.upper()}\n({len(archive_fitness)} solutions)', fontsize=12)
        ax1.grid(True, alpha=0.3)
    
    # Hypervolume Convergence
    ax2 = axes[0, 1]
    if hv_history:
        gens = [h['gen'] for h in history]
        ax2.plot(gens, hv_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Hypervolume', fontsize=12)
        ax2.set_title(' Hypervolume Convergence', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(gens, hv_history, alpha=0.3)
    
    # Evolution Over Time
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for i, h in enumerate(history):
        if len(h['fitness']) > 0:
            alpha = 0.3 if i < len(history) - 1 else 0.8
            size = 20 if i < len(history) - 1 else 50
            ax3.scatter(h['fitness'][:, 0], h['fitness'][:, 1], 
                       c=[colors[i]], s=size, alpha=alpha, 
                       label=f"Gen {h['gen']}" if i % max(1, len(history)//5) == 0 else "")
    ax3.set_xlabel('Objective 1', fontsize=12)
    ax3.set_ylabel('Objective 2', fontsize=12)
    ax3.set_title(' Pareto Front Evolution', fontsize=12)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Archive Size Growth
    ax4 = axes[1, 1]
    sizes = [h['size'] for h in history]
    gens = [h['gen'] for h in history]
    ax4.bar(gens, sizes, width=max(1, gens[-1]//20), color='green', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Archive Size', fontsize=12)
    ax4.set_title(' Pareto Front Size Growth', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'MOBCO Results - {fname.upper()} (2 Objectives)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'mobco_{fname}_2obj.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n Plot saved: mobco_{fname}_2obj.png")


def plot_3d_pareto_user(archive_fitness, fname, hv_history, history):
    """
    Plot for 3 objectives
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 3D Pareto Front
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    if len(archive_fitness) > 0:
        scatter = ax1.scatter(archive_fitness[:, 0], archive_fitness[:, 1], archive_fitness[:, 2],
                             c=np.arange(len(archive_fitness)), cmap='viridis', s=40, alpha=0.7)
        ax1.set_xlabel('Objective 1', fontsize=10)
        ax1.set_ylabel('Objective 2', fontsize=10)
        ax1.set_zlabel('Objective 3', fontsize=10)
        ax1.set_title(f' 3D Pareto Front\n({len(archive_fitness)} solutions)', fontsize=12)
        plt.colorbar(scatter, ax=ax1, label='Solution Index')
    
    # Hypervolume Convergence
    ax2 = fig.add_subplot(2, 2, 2)
    if hv_history:
        gens = [h['gen'] for h in history]
        ax2.plot(gens, hv_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Hypervolume', fontsize=12)
        ax2.set_title(' Hypervolume Convergence', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(gens, hv_history, alpha=0.3)
    
    # 2D Projections
    ax3 = fig.add_subplot(2, 2, 3)
    if len(archive_fitness) > 0:
        ax3.scatter(archive_fitness[:, 0], archive_fitness[:, 1], c='blue', s=20, alpha=0.6)
        ax3.set_xlabel('Objective 1', fontsize=10)
        ax3.set_ylabel('Objective 2', fontsize=10)
        ax3.set_title('Projection: Obj1 vs Obj2', fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    if len(archive_fitness) > 0:
        ax4.scatter(archive_fitness[:, 0], archive_fitness[:, 2], c='green', s=20, alpha=0.6)
        ax4.set_xlabel('Objective 1', fontsize=10)
        ax4.set_ylabel('Objective 3', fontsize=10)
        ax4.set_title('Projection: Obj1 vs Obj3', fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'MOBCO Results - {fname.upper()} (3 Objectives)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'mobco_{fname}_3obj.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n Plot saved: mobco_{fname}_3obj.png")


def plot_nd_pareto_user(archive_fitness, fname, hv_history, history, n_obj):
    """
    Plot for 4+ objectives - FULLY FIXED for ANY number of objectives!
    Works for 4, 5, 10, 17, 20, 50, 100 objectives!
    """
    
    # 🔧 FIX 1: Only show up to 10 objectives to avoid overcrowding
    show_obj = min(n_obj, 10)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========== PLOT 1: Hypervolume Convergence ==========
    ax1 = axes[0, 0]
    if hv_history:
        gens = [h['gen'] for h in history]
        ax1.plot(gens, hv_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Hypervolume', fontsize=11)
        ax1.set_title(f' Hypervolume Convergence ({n_obj} objectives)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(gens, hv_history, alpha=0.3)
    
    # ========== PLOT 2: Archive Size Growth ==========
    ax2 = axes[0, 1]
    if len(history) > 0:
        sizes = [h['size'] for h in history]
        gens = [h['gen'] for h in history]
        ax2.bar(gens, sizes, width=max(1, gens[-1]//20), color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Archive Size', fontsize=11)
        ax2.set_title(' Pareto Front Size Growth', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # ========== PLOT 3: Parallel Coordinates ==========
    ax3 = axes[0, 2]
    if len(archive_fitness) > 0:
        # Normalize each objective
        normalized = (archive_fitness - np.min(archive_fitness, axis=0)) / (np.max(archive_fitness, axis=0) - np.min(archive_fitness, axis=0) + 1e-10)
        
        # Sample solutions for clarity
        sample_size = min(50, len(normalized))
        indices = np.random.choice(len(normalized), sample_size, replace=False)
        
        # Plot only first 'show_obj' objectives
        for idx in indices:
            ax3.plot(range(show_obj), normalized[idx, :show_obj], 'b-', alpha=0.2, linewidth=0.8)
        
        # Plot mean line
        if show_obj > 0:
            mean_line = np.mean(normalized[:, :show_obj], axis=0)
            ax3.plot(range(show_obj), mean_line, 'r-', linewidth=2, label='Mean')
        
        # 🔧 FIX 2: Same number of ticks AND labels!
        ax3.set_xticks(range(show_obj))
        ax3.set_xticklabels([f'Obj{i+1}' for i in range(show_obj)], rotation=45, ha='right')
        
        ax3.set_xlabel('Objective Index', fontsize=11)
        ax3.set_ylabel('Normalized Value', fontsize=11)
        
        if n_obj > show_obj:
            ax3.set_title(f' Parallel Coordinates\n(First {show_obj} of {n_obj} objectives)', fontsize=11)
        else:
            ax3.set_title(f' Parallel Coordinates ({n_obj} objectives)', fontsize=11)
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # ========== PLOT 4: Objective Statistics ==========
    ax4 = axes[1, 0]
    if len(archive_fitness) > 0:
        means = np.mean(archive_fitness, axis=0)
        stds = np.std(archive_fitness, axis=0)
        mins = np.min(archive_fitness, axis=0)
        
        # 🔧 FIX 3: Use show_obj for x_pos
        x_pos = np.arange(show_obj)
        width = 0.35
        
        ax4.bar(x_pos - width/2, means[:show_obj], width, yerr=stds[:show_obj], 
                capsize=5, label='Mean ± Std', color='blue', alpha=0.7)
        ax4.bar(x_pos + width/2, mins[:show_obj], width, label='Min', color='green', alpha=0.7)
        
        ax4.set_xlabel('Objective', fontsize=11)
        ax4.set_ylabel('Value', fontsize=11)
        
        if n_obj > show_obj:
            ax4.set_title(f' Objective Statistics\n(First {show_obj} of {n_obj} objectives)', fontsize=11)
        else:
            ax4.set_title(f' Objective Statistics', fontsize=11)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Obj{i+1}' for i in range(show_obj)])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # ========== PLOT 5: Correlation Heatmap ==========
    ax5 = axes[1, 1]
    if len(archive_fitness) > 0 and show_obj > 1:
        # 🔧 FIX 4: Use only first 'show_obj' objectives
        corr_matrix = np.corrcoef(archive_fitness[:, :show_obj].T)
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        ax5.set_xticks(range(show_obj))
        ax5.set_yticks(range(show_obj))
        ax5.set_xticklabels([f'Obj{i+1}' for i in range(show_obj)], rotation=45, ha='right')
        ax5.set_yticklabels([f'Obj{i+1}' for i in range(show_obj)])
        
        if n_obj > show_obj:
            ax5.set_title(f'🔗 Objective Correlation\n(First {show_obj} of {n_obj})', fontsize=11)
        else:
            ax5.set_title(f'🔗 Objective Correlation Matrix', fontsize=11)
        
        plt.colorbar(im, ax=ax5, label='Correlation')
    else:
        ax5.text(0.5, 0.5, 'Not enough data\nfor correlation', 
                ha='center', va='center', fontsize=12)
        ax5.axis('off')
    
    # ========== PLOT 6: Summary Information ==========
    ax6 = axes[1, 2]
    if len(archive_fitness) > 0:
        ax6.text(0.5, 0.7, f'{len(archive_fitness)}', 
                fontsize=60, ha='center', va='center', fontweight='bold', color='darkblue')
        ax6.text(0.5, 0.45, 'Pareto-Optimal\nSolutions Found', 
                fontsize=14, ha='center', va='center')
        ax6.text(0.5, 0.25, f'with {n_obj} objectives', 
                fontsize=12, ha='center', va='center', style='italic')
        ax6.text(0.5, 0.08, f'Showing {show_obj} of {n_obj}\nin parallel coordinates', 
                fontsize=9, ha='center', va='center', style='italic', color='gray')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
    
    plt.suptitle(f'🐕 MOBCO Results - {fname.upper()} ({n_obj} Objectives)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'mobco_{fname}_{n_obj}obj.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n Plot saved: mobco_{fname}_{n_obj}obj.png")
    print(f"   (Showing first {show_obj} of {n_obj} objectives in visualizations)")


def plot_comparison_user(all_results):
    """
    Compare results across different numbers of objectives
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hypervolume comparison
    ax1 = axes[0]
    for n_obj, data in all_results.items():
        history = data.get('history', [])
        if history:
            gens = [h['gen'] for h in history]
            hvs = [h['hv'] for h in history]
            ax1.plot(gens, hvs, 'o-', linewidth=2, markersize=4, label=f'{n_obj} objectives')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Hypervolume', fontsize=12)
    ax1.set_title(' Hypervolume Comparison Across Objective Counts', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final archive size comparison
    ax2 = axes[1]
    n_objs = list(all_results.keys())
    sizes = [all_results[n].get('final_size', 0) for n in n_objs]
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_objs)))
    ax2.bar([str(n) for n in n_objs], sizes, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Objectives', fontsize=12)
    ax2.set_ylabel('Pareto Front Size', fontsize=12)
    ax2.set_title('📦 Pareto Front Size vs Number of Objectives', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MOBCO Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mobco_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n Comparison plot saved: mobco_comparison.png")