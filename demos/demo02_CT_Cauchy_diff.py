# %%
import numpy as np
from cuqi.distribution import CMRF
from cuqipy_cil.testproblem import ParallelBeam2D

# Computed Tomography
TP = ParallelBeam2D(
    im_size=(256, 256),
    det_count=256,
    angles=np.linspace(0, np.pi, 180),
    phantom="shepp-logan",
)

# Cauchy difference prior
TP.prior = CMRF(
    location=0,
    scale=0.01,
    geometry=TP.model.domain_geometry
)

# Sample posterior with automatic sampler choice
samples = TP.sample_posterior(200)

# Plot sample mean and ci
samples.plot_ci()
