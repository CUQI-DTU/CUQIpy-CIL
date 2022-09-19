from scipy.fftpack import dst, idst
import numpy as np

def test_dst():
    x = np.random.rand(10)
    y = dst(x)
