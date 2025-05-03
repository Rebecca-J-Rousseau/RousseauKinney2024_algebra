# Deterministic and stochastic simulations for non-equilibrium systems

This repository contains supporting code for:

- Rousseau RJ, Kinney JB. Algebraic and diagrammatic methods for the rule-based modeling of multiparticle complexes. *PRX Life* **3**, 023004 (2025). https://doi.org/10.1103/PRXLife.3.023004

  - Preprint found at arXiv:2409.01529 [physics.bio-ph] https://doi.org/10.48550/arXiv.2409.01529 (2024)

The contents of the repository pertain specifically to Figs. 7 and 9 in that paper, and demonstrate how to apply the algebraic formalism developed in the paper to deterministic and stochastic simulations for a number of non-equilibrium systems.

## Layout
---
The repository is organized into three directories: 

## `simulationcode`
This directory contains all Jupyter notebooks used to simulate each example system:

1. Deterministic simulations
    * Monomer
    * Homodimer
    * Homopolymer
2. Stochastic simulations
    * Monomer
    * Homodimer
    * Homopolymer
    * Heterodimer
    * Occlusive binding
    * Cooperative binding
    * Linear heteropolymer
    * Branched homopolymer

All stochastic simulations use a Gillespie algorithm detailed in Rousseau and Kinney (2024) and coded in `gillespie.py`.

Each .ipynb notebook outputs .csv files containing the vectors of times, averages, and standard deviations for various complexes, and these outputs are saved into the `simulationdata` directory.

## `simulationdata`
This directory contains all output .csv files from the notebooks in the `simulationcode` directory.

## `figures`
This directory contains the Jupyter notebook `paperfigs_Fig7_Fig9.ipynb`, which imports all simulation results from `simulationdata` and plots figures demonstrating the deterministic and averaged stochastic evolution of complex counts in the specified example systems. All .png figures are saved into this directory.
