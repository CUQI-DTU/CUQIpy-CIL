""" This module controls global configuration settings for cuqipy_cil. """
import subprocess as _subprocess

# Try detecting GPU
try:
    _subprocess.check_output('nvidia-smi')
    _detected_device = "gpu"
except:
    _detected_device = "cpu"

PROJECTION_BACKEND = "astra"
""" Projection backend to use. Currently supported: "tigre", "astra" """

PROJECTION_BACKEND_DEVICE = _detected_device
""" Device to use for projection backend. Currently supported: "cpu", "gpu". Only relevant for astra backend. """
