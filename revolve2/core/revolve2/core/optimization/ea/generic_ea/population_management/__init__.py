"""Functions for combining populations in EA algorithms."""

from ._generational import generational
from ._steady_state import steady_state
from ._NSGA2 import NSGA2

__all__ = [
    "generational",
    "steady_state",
    "NSGA2"
]
