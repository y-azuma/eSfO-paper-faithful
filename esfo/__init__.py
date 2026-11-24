"""Auditable, paper-faithful eSfO reproduction."""

from .config import FrontendConfig
from .data import EventSequence, RelativeEventSequence
from .frequency import (
    FFTFrequencyInitialization, FrequencyInitializationError,
    WindowedMeanX, initialize_frequency_fft_raw,
)
from .frontend import FrontendResult, run_frontend, run_frontend_raw
from .geometry import (
    GeometryInitializationError, OrbitGeometryInitialization,
    R0Initialization, RegisteredTrajectory, fit_orbit_geometry,
    initialize_r0, orbit_centers, orbit_poses,
)

__all__ = [
    "EventSequence", "FFTFrequencyInitialization", "FrequencyInitializationError",
    "FrontendConfig", "FrontendResult", "GeometryInitializationError",
    "OrbitGeometryInitialization", "R0Initialization", "RegisteredTrajectory",
    "RelativeEventSequence", "WindowedMeanX", "fit_orbit_geometry",
    "initialize_frequency_fft_raw", "initialize_r0", "orbit_centers",
    "orbit_poses", "run_frontend", "run_frontend_raw", "__version__",
]
__version__ = "0.1.0"

