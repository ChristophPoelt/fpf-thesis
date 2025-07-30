# FPF Thesis Reository

This repository contains the complete codebase, analysis scripts, visualizations and supporting material for the bachelor's thesis:

**"Gradient Descent Optimizability of Final Productive Fitness Landscapes in Evolutionary Algorithms"**
by Christoph Pölt (LMU Munich, 2025)

---

## 📁 Repository Structure

- `diagrams/` – Contains all generated visualizations
  - `DGM/` – Histograms showing the distribution of DGM outcomes and convergence steps
  - `best-fitness/` – Line plots showing the average best individual fitness across generations
  - `gradient-magnitude-heatmap/` – Heatmaps showing gradient magnitude
  - `landscape-heatmap/` – Heatmas showing the fpf landscapes
  - `scatterplot/` – Scatterplots of the fpf landscapes

- `evolutionary-algorithm-2D/` – Java implementation of the evolutionary algorithm for 2D benchmark functions
- `evolutionary-algorithm-10D/` – Java implementation of the evolutionary algorithm for 10D benchmark functions
- `evolutionary-algorithm-HPI/` – Java implementation of the evolutionary algorithm for 2D benchmark functions with High Potential Individuals integrated into the evolution process

- `fpf-landscape-analysis-2D/` – Python scripts to analyze and visualize results from 2D/HPI experiments
- `fpf-landscape-analysis-10D/` – Python scripts for evaluating 10D experiments
