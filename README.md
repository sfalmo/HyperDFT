# Hyper-density functional theory of soft matter

This repository contains code, data sets and models corresponding to the following publication:

**Hyper-density functional theory of soft matter**  
Florian Sammüller, Silas Robitschko, Sophie Hermann, and Matthias Schmidt

### Setup

A recent version of [Julia](https://julialang.org/downloads/) must be installed on your system.
Launch the Julia REPL and enter the package manager by typing `]`.
Set up the project as follows:

```julia
activate .
instantiate
```

### Instructions

We consider the hard rod fluid ("HR") and the square well fluid with a range of 1.2 ("SW1.2") in one spatial dimension and take the observable of interest to be the largest cluster size of a given microstate.

Neural direct correlation functionals (see also [NeuralDFT](https://github.com/sfalmo/NeuralDFT) and [NeuralDFT-Tutorial](https://github.com/sfalmo/NeuralDFT-Tutorial)) can be loaded from the files `model_<particles>.bson`.
Simulation data is provided in the directories `data_<particles>_L<system length>` (raw) and in the files `data_<particles>_L<system length>.jld2` (preprocessed).
The trained hyper-direct correlation functionals are saved in the files `model_cluster_<particles>_L<system length>.bson`.

Code to generate and process the reference simulation data as well as to train the neural hyper-direct correlation functional is given in `main.jl`.
Utilities are provided in `simulation.jl`, `dft.jl` and `neural.jl`.
Plots of the manuscript can be reproduced with `plots.ipynb` (start a Jupyter server to run this notebook).
