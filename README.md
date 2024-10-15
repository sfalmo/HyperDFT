[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13318633.svg)](https://doi.org/10.5281/zenodo.13318633)


# Hyperdensity functional theory of soft matter

This repository contains code, data sets and models corresponding to the following publication:

**Hyperdensity functional theory of soft matter**  
Florian Sammüller, Silas Robitschko, Sophie Hermann, and Matthias Schmidt, [Phys. Rev. Lett. **133**, 098201](https://doi.org/10.1103/PhysRevLett.133.098201) (2024); [arXiv:2403.07845](https://arxiv.org/abs/2403.07845).

For an introductory account and detailed derivations, see also:

**Why hyperdensity functionals describe any equilibrium observable**  
Florian Sammüller and Matthias Schmidt; [arXiv:2410.10534](https://arxiv.org/abs/2410.10534).


### Setup

A recent version of [Julia](https://julialang.org/downloads/) must be installed on your system.
Launch the Julia REPL and enter the package manager by typing `]`.
Set up the project as follows:

```julia
activate .
instantiate
```

### Instructions

We consider the hard rod fluid ("HR"), the square well fluid with a range of 1.2 ("SW1.2") in one spatial dimension and the hard sphere fluid ("HS") in planar three dimensional geometry.
To test the hyper-DFT framework, the non-trivial observable of interest is chosen to be the largest cluster size of a given microstate (see also `simulation.jl` for an algorithm to detect particle clusters).

Neural direct correlation functionals (see also [NeuralDFT](https://github.com/sfalmo/NeuralDFT) and [NeuralDFT-Tutorial](https://github.com/sfalmo/NeuralDFT-Tutorial)) can be loaded from the files `model_<particles>.bson`.
Simulation data is provided in the directories `data_<particles>_L<system length>` (raw) and in the files `data_<particles>_L<system length>.jld2` (preprocessed).
The trained hyper-direct correlation functionals for the considered cluster observable are saved in the files `model_cluster_<particles>_L<system length>.bson`.

Code to generate and process the reference simulation data as well as to train the neural hyper-direct correlation functional is given in `main.jl` (the data for the 3D HS fluid has been generated with [MBD](https://gitlab.uni-bayreuth.de/bt306964/mbd)).
Utilities are provided in `simulation.jl`, `dft.jl` and `neural.jl`.
Plots of the manuscript can be reproduced with `plots.ipynb` (start a Jupyter server to run this notebook).
