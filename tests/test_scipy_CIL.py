from scipy.fftpack import dst, idst # Import should work
import numpy as np

from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageGeometry, AcquisitionGeometry

def test_CIL():
    im_size = (45,45)
    det_count = 50
    angles = np.linspace(0, np.pi, 60)
    domain = im_size
    det_spacing = 1

    acquisition_geometry = AcquisitionGeometry.create_Parallel2D()
    acquisition_geometry.set_angles(angles, angle_unit="radian")
    acquisition_geometry.set_panel(det_count, pixel_size=det_spacing)

    # Setup image geometry
    image_geometry = ImageGeometry(
        voxel_num_x=im_size[0],
        voxel_num_y=im_size[1],
        voxel_size_x=domain[0] / im_size[0],
        voxel_size_y=domain[1] / im_size[1],
    )

    # Projection operator
    projection_operator = ProjectionOperator(image_geometry, acquisition_geometry)
        
    # Allocate data containers for efficiency
    acquisition_data = acquisition_geometry.allocate()
    image_data = image_geometry.allocate()

    # Forward projection
    image_data.fill(np.zeros(im_size))
    projection_operator.direct(image_data, out=acquisition_data)
