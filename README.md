# CUQIpy-CIL

CUQIpy-CIL is a plugin for the [CUQIpy](https://github.com/CUQI-DTU/CUQIpy) software package.

It adds a thin wrapper around Computed Tomography (CT) forward models from the Core Imaging Library (CIL).

## Installation
First install [CIL](https://github.com/TomographicImaging/CIL). Then install CUQIpy-CIL with pip:
```bash
pip install cuqipy-cil
```
If CUQIpy is not installed, it will be installed automatically.

## Quickstart
```python
import numpy as np
import matplotlib.pyplot as plt
import cuqi
import cuqipy_cil

# Load a CT forward model and data from testproblem library
A, y_data, info = cuqipy_cil.testproblem.ParallelBeam2DProblem.get_components(
    im_size=(128, 128),
    det_count=128,
    angles=np.linspace(0, np.pi, 180),
    phantom="shepp-logan"
)

# Set up Bayesian model
x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), cov=1) # x ~ N(0, 1)
y = cuqi.distribution.Gaussian(A@x, cov=0.05**2)              # y ~ N(Ax, 0.05^2)

# Set up Bayesian Problem
BP = cuqi.problem.BayesianProblem(y, x).set_data(y=y_data)

# Sample from the posterior
samples = BP.sample_posterior(200)

# Analyze the samples
info.exactSolution.plot(); plt.title("Exact solution")
y_data.plot(); plt.title("Data")
samples.plot_mean(); plt.title("Posterior mean")
samples.plot_std(); plt.title("Posterior standard deviation")
```

For more examples, see the [demos](demos) folder.

