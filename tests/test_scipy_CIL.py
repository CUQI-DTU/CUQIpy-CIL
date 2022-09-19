from scipy.fftpack import dst, idst
import numpy as np

from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageGeometry, AcquisitionGeometry, DataContainer

def test_dst():
    x = np.random.rand(10)
    y = dst(x)
