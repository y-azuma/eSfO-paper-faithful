"""Auditable, paper-faithful eSfO reproduction."""

from .config import FrontendConfig
from .data import EventSequence, RelativeEventSequence
from .frontend import FrontendResult, run_frontend, run_frontend_raw

__all__ = [
    "EventSequence", "FrontendConfig", "FrontendResult",
    "RelativeEventSequence", "run_frontend", "run_frontend_raw",
    "__version__",
]
__version__ = "0.1.0"

