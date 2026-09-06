# Viral Target Optimizer

A lightweight mathematical framework for computational screening and molecular parameter optimization against viral protein targets.

## Overview

This project provides a robust, production-ready structure for matching virtual screening drug candidates with target specifications derived from virology literature. It utilizes Projected Gradient Descent (PGD) driven by a Mean Squared Error (MSE) loss function to optimize critical chemoinformatics descriptors in a normalized canonical coordinate space:

- **LogP** (Partition Coefficient / Lipophilicity)
- **MW** (Molecular Weight)

The system operates as a **comparative benchmarking suite**, executing and evaluating three distinct numerical optimization approaches. The architecture strictly isolates raw input parameters, execution logic, and computational logs, making it easily extendable to advanced chemoinformatics packages like RDKit.

## Project Structure
```txt
viral-target-optimizer/
├── data/
│ ├── target_virus_proteins.json # Input: 10 target profiles extracted from literature
│ └── optimization_results.json # Output: Calculated optimal descriptors and benchmarking logs
├── src/
│ ├── init.py
│ ├── loader.py # Data I/O handling (JSON parsing and logging)
│ └── optimizer.py # Mathematical optimization routines and multi-method pipelines
├── requirements.txt # Environment dependencies
├── main.py # Application entry point
└── README_RU.md # Project documentation on Russian
|__ README_EN.md # Project documentation on English
```

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation & Usage Summary

In short, run `pip install -r requirements.txt`, and execute `python main.py` to process targets and generate telemetry logs.

## Optimization Methods & Mathematics

The framework models optimal drug parameter selection via constrained optimization (\(\min_{x \in S} f(x)\)) using three core algorithms:

### 1. Canonical L-Smooth Adapted PGD
Normalizes descriptor space and uses analytical Lipschitz constant \(L = 2.0\) for rapid convergence. This method guarantees convergence in a single iteration through theoretically optimal step sizing.

### 2. Nesterov Accelerated Gradient (NAG) PGD
Incorporates momentum (\(\mu = 0.9\)) for improved trajectory and speedup on complex functions. This approach leverages the concept of "look-ahead" gradient evaluation to accelerate convergence on ill-conditioned problems.

### 3. Heuristic Static PGD
Acts as a baseline control using fixed step size without momentum. This method provides a performance reference point for evaluating the effectiveness of the adaptive and accelerated methods.

### Key Mathematical Features

- **Canonical Feature Scaling**: Standardizing raw descriptor space into a normalized system where \(x = [\text{LogP}, \text{MW} / 100]\) completely eliminates poor condition numbers (\(\kappa = 10000\)) caused by coordinate scale disparities.

- **Projected Gradient Descent (PGD)**: The core engine optimizes constrained parameters within chemical boundaries (LogP: -2.0 to 6.0; MW: 150.0 to 600.0 Da) using an exact Euclidean box projection operator implemented via deterministic clipping.

- **Lipschitz Step-Size Adaptation**: Analytical evaluation of the Hessian matrix (\(\nabla^2 f(x)\)) yields a strict Lipschitz constant of \(L = 2.0\) for the scaled gradient.

- **Benchmarking Suite**: The runtime environment systematically compares all three optimization approaches, measuring convergence speed, accuracy, and computational efficiency.

## Future Roadmap: RDKit Integration

Upcoming phases target direct coupling with the RDKit ecosystem for real chemical library screening, including:

- **SMILES target filtering**: Parsing and validating chemical structures from SMILES strings
- **Dynamic property extraction**: Computing MolLogP, MolWt, and other descriptors on-the-fly
- **Structural drug-likeness validation**: Applying Lipinski's Rule of Five and other filters
- **Real chemical library screening**: Screening thousands of compounds against viral targets

## Sources

- **Book**: *Principles of Virology*, 4th Edition (2 Vol set) by S. Jane Flint, Lynn W. Enquist, Vincent R. Racaniello, Glenn F. Rall, Anna-Marie Skalka.
- **Mathematical Framework**: Conditional Gradient Methods & Projected Gradient Descent, Advanced Numerical Optimization Syllabus by D. Merkulov (HSE University / MIPT). URL: https://github.com/MerkulovDaniil/hse26/blob/main/lectures/12.md