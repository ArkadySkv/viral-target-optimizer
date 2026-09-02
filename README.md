# Viral Target Optimizer

A lightweight mathematical framework for computational screening and molecular parameter optimization against viral protein targets.

## Overview

This project provides a robust, production-ready structure for matching virtual screening drug candidates with target specifications derived from virology literature. It utilizes a **Random Search algorithm** driven by a **Mean Squared Error (MSE)** loss function to optimize critical chemoinformatics descriptors:
*   **LogP** (Partition Coefficient / Lipophilicity)
*   **MW** (Molecular Weight)

The architecture strictly isolates raw input parameters, execution logic, and computational logs, making it easily extendable to advanced chemoinformatics packages like `RDKit`.

## Project Structure

```text
viral-target-optimizer/
├── data/
│   ├── target_virus_proteins.json  # Input: Target profiles extracted from literature
│   └── optimization_results.json # Output: Calculated optimal descriptors and MSE loss
├── src/
│   ├── __init__.py
│   ├── loader.py                 # Data I/O handling (JSON parsing and logging)
│   └── optimizer.py              # Optimization routines and mathematical loss criteria
├── requirements.txt              # Environment dependencies
├── main.py                       # Application entry point
└── README.md                     # Project documentation
```

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   Active internet connection (only for initial environment setup)

### Installation & Environment Setup

1. Clone or navigate to the project directory:
   ```bash
   cd viral-target-optimizer
   ```

2. Create a isolated Python virtual environment:
   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   *   **Linux/macOS:**
       ```bash
       source .venv/bin/activate
       ```
   *   **Windows (Git Bash):**
       ```bash
       source .venv/Scripts/activate
       ```
   *   **Windows (CMD):**
       ```cmd
       .venv\Scripts\activate.bat
       ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your target dataset is generated and placed at `data/target_virus_proteins.json`.
2. Run the main processing pipeline:
   ```bash
   python main.py
   ```

The script will automatically parse the targets, execute the stochastic optimization loop, print a structured telemetry table directly into the terminal, and dump the comprehensive metrics log to `data/optimization_results.json`.

## Algorithms and Methods

*   **Random Search Screening:** The optimization engine explores a designated chemical space boundary (LogP: -2.0 to 6.0; MW: 150.0 to 600.0 Da) simulating brute-force molecular filtering.
*   **Feature Normalization:** Due to scale disparities between MW and LogP, the loss function normalizes molecular weight variations to ensure balanced convergence criteria during spatial coordinate calculations.

## Sources

Book **Principles_of_Virology_4th_Edition_2_Vol_set_by_S._Jane_Flint_Lynn_W._Enquist_Vincent_R._Racaniello_Glenn_F._Rall_Anna-Marie_Skalka**