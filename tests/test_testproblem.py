import cuqipy_cil
import pytest
import cuqi
import numpy as np
import sys

@pytest.mark.skipif(sys.platform == "linux", reason="Currently fails on miniconda linux image due to some issue with tigre!")
def test_testproblem_simple():
    # Create simple testproblem
    TP = cuqipy_cil.testproblem.ParallelBeam2DProblem()

    # First check dimension of model matches dimension of image + data
    assert TP.model.domain_dim == TP.exactSolution.geometry.par_dim

    # Check that the output is the correct shape
    assert TP.model.range_dim == TP.exactData.geometry.par_dim

    # Check basic run posterior sampling
    samples = TP.UQ(Ns=50)

    # Check that the output is the correct shape
    assert samples.shape == (TP.model.domain_dim, 50)

    # Check basic MAP estimate
    MAP = TP.MAP()

    # Check that the output is the correct shape
    assert MAP.shape == (TP.model.domain_dim,)

@pytest.mark.skipif(sys.platform == "linux", reason="Currently fails on miniconda linux image due to some issue with tigre!")
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
    TP = cuqipy_cil.testproblem.ParallelBeam2DProblem(im_size=(128,128), phantom=phantom)

    if isinstance(phantom, str): # If phantom is a string, it should be a valid phantom name
        phantom = cuqi.data.__dict__[phantom](size=TP.model.domain_geometry.fun_shape[0])

    assert np.allclose(TP.exactSolution, cuqi.data.imresize(phantom, TP.model.domain_geometry.fun_shape))

@pytest.mark.skipif(sys.platform == "linux", reason="Currently fails on miniconda linux image due to some issue with tigre!")
def test_testproblem_set_prior():
    """ Test if one can set a prior after creating a testproblem """
    TP = cuqipy_cil.testproblem.ParallelBeam2DProblem(
        im_size=(256, 256),
        det_count=256,
        angles=np.linspace(0, np.pi, 180),
        phantom="shepp-logan",
    )

    # Cauchy difference prior
    TP.prior = cuqi.distribution.Cauchy_diff(
        location=np.zeros(TP.model.domain_dim),
        scale=0.01,
        physical_dim=2,
    )

    # Sample posterior with automatic sampler choice
    samples = TP.sample_posterior(20)

    # Plot sample mean and ci
    samples.plot_ci()
