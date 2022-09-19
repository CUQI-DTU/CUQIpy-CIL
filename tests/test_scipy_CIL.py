from scipy.fftpack import dst, idst
import numpy as np

from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageGeometry, AcquisitionGeometry, DataContainer

def test_dst():
    x = np.random.rand(10)
    y = dst(x)

def test_CIL():
    im_size = (45,45),
    det_count = 50
    angles = np.linspace(0,np.pi,60),
    domain = im_size
    det_spacing = 1

    acquisition_geometry = AcquisitionGeometry.create_Parallel2D()
    acquisition_geometry.set_angles(angles, angle_unit="radian")
    acquisition_geometry.set_panel(det_count, pixel_size=det_spacing)
