# 🐕 MOBCO: Multi-Objective Border Collie Optimization

**A nature-inspired multi-objective optimization algorithm based on Border Collie herding behavior.**

MOBCO extends Border Collie Optimization (BCO) to solve multi-objective problems ranging from **2 objectives to thousands of objectives**, producing diverse Pareto-optimal solutions while maintaining convergence and diversity through an external archive.

---

## ✨ Features

* 🐕 Three-dog leadership strategy (Lead, Left, Right)
* 🐑 Population-based exploration and exploitation
* 👁️ Eyeing mechanism to escape local optima
* 📦 External archive for Pareto-optimal solutions
* 📈 Hypervolume-based convergence monitoring
* 📊 Support for low-, many-, and ultra-high-dimensional objectives
* 🎯 Compatible with DTLZ and ZDT benchmark suites

---

## 📂 Supported Benchmarks

| Family | Functions                    |
| ------ | ---------------------------- |
| DTLZ   | DTLZ1, DTLZ2, DTLZ3, DTLZ4   |
| ZDT    | ZDT1, ZDT2, ZDT3, ZDT4, ZDT6 |

Supported objective counts range from **2 to 7000+ objectives**.

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/MOBCO.git
cd MOBCO
python mobco_user_main.py
```

Example configuration:

```text
Function      : DTLZ2
Objectives    : 17
Population    : 70
Generations   : 200
```

---

## 🧠 Algorithm Overview

MOBCO mimics the herding behavior of Border Collie dogs:

1. Generate an initial population.
2. Select the best individuals as Lead, Left, and Right dogs.
3. Herd candidate solutions using:

   * Gathering
   * Stalking
   * Eyeing
4. Update positions and velocities.
5. Store non-dominated solutions in an archive.
6. Repeat until termination.
7. Return the final Pareto front.

---

## 📊 Experimental Results

### DTLZ2 (2 Objectives)

<img width="2082" height="1473" alt="mobco_dtlz2_2obj" src="https://github.com/user-attachments/assets/ba524bb1-8b89-4e89-bd77-16390d4ba3ca" />

Clean Pareto front approximation with stable hypervolume convergence.

### DTLZ2 (3 Objectives)

<img width="2234" height="1475" alt="mobco_dtlz2_3obj" src="https://github.com/user-attachments/assets/3c9751b7-b5f0-4cc7-b645-4fd93b7c1a06" />

3D Pareto front demonstrating convergence in three-objective space.

### DTLZ1 (4 Objectives)

<img width="2685" height="1477" alt="mobco_dtlz1_4obj" src="https://github.com/user-attachments/assets/ee27dca0-825f-4f49-a128-39cf53d9f6f4" />

Hypervolume convergence, objective statistics, correlation analysis, and parallel-coordinate visualization.

### DTLZ2 (5 Objectives)

<img width="2685" height="1477" alt="mobco_dtlz2_5obj" src="https://github.com/user-attachments/assets/14f9e0b4-8ab2-4ec1-8666-93e2ae945b27" />

Many-objective optimization with strong Pareto-front coverage.

### DTLZ2 (17 Objectives)

<img width="2686" height="1483" alt="mobco_dtlz2_17obj" src="https://github.com/user-attachments/assets/5e044046-a015-4114-93d8-d92996af5faf" />

Visualization of the first ten objectives using parallel coordinates and correlation analysis.

### DTLZ2 (30 Objectives)

<img width="2685" height="1483" alt="mobco_dtlz2_30obj" src="https://github.com/user-attachments/assets/41c13f9f-5b96-4930-bcee-b2b00b60bd6e" />

Statistical and convergence analysis for large-scale objective spaces.

### DTLZ2 (100 Objectives)
<img width="2685" height="1483" alt="mobco_dtlz3_100obj" src="https://github.com/user-attachments/assets/8056714f-68f1-4b55-a2dd-4751265bab15" />


### DTLZ2 (500 Objectives)

<img width="2686" height="1483" alt="mobco_dtlz2_500obj" src="https://github.com/user-attachments/assets/d89f628a-b00b-43cb-ba70-f74572fab64e" />

Demonstrates MOBCO's scalability to ultra-high-dimensional optimization.

### DTLZ2 (7000 Objectives)
<img width="2686" height="1483" alt="mobco_dtlz3_7000obj" src="https://github.com/user-attachments/assets/2b4b3f5b-05b2-41db-95b4-86ca1fa6bdf6" />

---

## 📁 Project Structure

```text
mobco_user_main.py          # Main execution script
mobco_function_details.py   # Benchmark functions
mobco_nondominated_sort.py  # Pareto sorting
mobco_herding.py            # Herding behavior
mobco_velocity.py           # Velocity updates
mobco_position.py           # Position updates
mobco_fitness.py            # Objective evaluation
mobco_generate.py           # Population generation
mobco_check.py              # Boundary handling
mobco_visualization.py      # Result visualization
```

---

## 📚 Citation

```bibtex
@article{dutta2020border,
  title={Border Collie Optimization},
  author={Dutta, Tulika and Bhattacharyya, Siddhartha and Dey, Sandip and Platos, Jan},
  journal={IEEE Access},
  volume={8},
  pages={109177--109197},
  year={2020}
}
```

---

## 📄 License

This project is released under the MIT License.

---

⭐ If you find MOBCO useful, consider starring the repository.
