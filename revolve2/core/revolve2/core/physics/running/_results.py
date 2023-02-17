from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pyrr import Quaternion, Vector3
from decimal import *

@dataclass
class ActorState:
    """State of an actor."""

    position: Vector3
    orientation: Quaternion

    # angles of each hinge joint
    hinge_angles: List[float] | None = None
    #velocities of each hinge joint
    hinge_vels: List[float] | None = None

    # torques of each hinge joint
    hinge_torques: List[float] | None = None

    # remaining power
    remaining_power: Decimal | None = None

    # Number of joints
    njnts: int | None = None

@dataclass
class EnvironmentState:
    """State of an environment."""

    time_seconds: float
    actor_states: List[ActorState]


@dataclass
class EnvironmentResults:
    """Result of running an environment."""

    environment_states: List[EnvironmentState]


@dataclass
class BatchResults:
    """Result of running a batch."""

    environment_results: List[EnvironmentResults]
