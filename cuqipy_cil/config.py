""" This module controls global configuration settings for the package. """
import subprocess as _subprocess

# Try detecting GPU
try:
    _subprocess.check_output('nvidia-smi')
    _detected_device = "gpu"
except:
    _detected_device = "cpu"

# Try detecting projector backend
_detected_backend = None

try:
    import astra
    _detected_backend = "astra"
except ImportError:
    pass

# Try defaulting to tigre (only if GPU is available)
if _detected_device == "gpu":
    try:
        import tigre
        _detected_backend = "tigre"
    except ImportError:
        pass

if _detected_backend is None:
    raise ImportError("No projector backend found. Please install either astra or tigre.")

PROJECTION_BACKEND = _detected_backend
""" Projection backend to use. Currently supported: "tigre", "astra". Defaults to tigre if possible, otherwise astra. """

PROJECTION_BACKEND_DEVICE = _detected_device
""" Device to use for projection backend. Currently supported: "cpu", "gpu". Defaults to GPU if present, otherwise cpu. Only relevant for astra backend. """
