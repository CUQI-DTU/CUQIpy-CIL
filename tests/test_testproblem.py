import cuqipy_cil
import pytest
import cuqi
import numpy as np
import matplotlib.pyplot as plt

def test_testproblem_simple():
    # Create simple testproblem
    TP = cuqipy_cil.testproblem.ParallelBeam2D()

    # First check dimension of model matches dimension of image + data
    assert TP.model.domain_dim == TP.exactSolution.geometry.par_dim

    # Check that the output is the correct shape
    assert TP.model.range_dim == TP.exactData.geometry.par_dim

    # Check basic run posterior sampling
    samples = TP.UQ(Ns=20)

    # Check that the output is the correct shape
    assert samples.shape == (TP.model.domain_dim, 20)

    # Check basic MAP estimate
    MAP = TP.MAP()

    # Check that the output is the correct shape
    assert MAP.shape == (TP.model.domain_dim,)

@pytest.mark.parametrize("phantom",
    [
        (cuqi.data.shepp_logan(size=128)),
        (cuqi.data.p_power(size=128)),
        (cuqi.data.grains(size=128)),
        ("shepp_logan"),
        ("p_power"),
        ("grains"),
    ])
def test_testproblem_phantom(phantom):
    # Create simple testproblem
    TP = cuqipy_cil.testproblem.ParallelBeam2D(im_size=(128,128), phantom=phantom)

    if isinstance(phantom, str): # If phantom is a string, it should be a valid phantom name
        phantom = cuqi.data.__dict__[phantom](size=TP.model.domain_geometry.fun_shape[0])

    assert np.allclose(TP.exactSolution, cuqi.data.imresize(phantom, TP.model.domain_geometry.fun_shape))

def test_testproblem_set_prior():
    """ Test if one can set a prior after creating a testproblem """
    TP = cuqipy_cil.testproblem.ParallelBeam2D(
        im_size=(256, 256),
        det_count=256,
        angles=np.linspace(0, np.pi, 180),
        phantom="shepp-logan",
    )

    # Cauchy difference prior
    TP.prior = cuqi.distribution.CMRF(
        location=0,
        scale=0.01,
        geometry=TP.model.domain_geometry,
    )

    # Sample posterior with automatic sampler choice
    samples = TP.sample_posterior(20)

    # Plot sample mean and ci
    samples.plot_ci()

def test_testproblem_from_readme():

    # Load a CT forward model and data from testproblem library
    A, y_data, info = cuqipy_cil.testproblem.ParallelBeam2D(
        im_size=(128, 128),
        det_count=128,
        angles=np.linspace(0, np.pi, 180),
        phantom="shepp-logan"
    ).get_components()

    # Set up Bayesian model
    x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), cov=1) # x ~ N(0, 1)
    y = cuqi.distribution.Gaussian(A@x, cov=0.05**2)              # y ~ N(Ax, 0.05^2)

    # Set up Bayesian Problem
    BP = cuqi.problem.BayesianProblem(y, x).set_data(y=y_data)

    # Sample from the posterior
    samples = BP.sample_posterior(20)

    # Analyze the samples
    info.exactSolution.plot(); plt.title("Exact solution")
    y_data.plot(); plt.title("Data")
    samples.plot_mean(); plt.title("Posterior mean")
    samples.plot_std(); plt.title("Posterior standard deviation")
