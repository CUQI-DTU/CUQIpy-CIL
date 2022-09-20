import cuqipy_cil
import pytest
import numpy as np
import sys

@pytest.mark.parametrize("model",
    [
        (cuqipy_cil.model.ParallelBeam2DModel()),
        (cuqipy_cil.model.FanBeam2DModel()),
        (cuqipy_cil.model.ShiftedFanBeam2DModel())
    ])
def test_model_simple(model: cuqipy_cil.model.CILModel):
    # Test that the model at least computes a forward projection.

    # Simple test input
    x = np.zeros(model.domain_geometry.fun_shape)

    # Compute forward projection
    y = model.forward(x)

    # Check that the output is the correct shape
    assert y.shape == (model.range_dim,)

    # Compute backprojection
    x2 = model.adjoint(y)

    # Check that the output is the correct shape
    assert x2.shape == (model.domain_dim,)


