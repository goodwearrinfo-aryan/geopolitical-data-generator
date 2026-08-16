"""Engine package for the Geopolitical Data Generator.

Contains distributed execution, serialization, calibration, and agent-based
micro-foundations modules.
"""

from __future__ import annotations

__all__ = ["composition", "agents"]

from . import composition
from . import agents
