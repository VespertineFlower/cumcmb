"""Q3 adapter for the shared official simulator client."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.official_client import CommunicationError, RequestRejected, RetryPolicy
from shared.official_client import SimulatorClient as _SharedSimulatorClient
from .models import Point


class SimulatorClient(_SharedSimulatorClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = Point(0.0, 0.0)
