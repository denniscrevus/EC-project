from revolve2.core.physics.running import ActorState
import math
from decimal import *

def calculate_fitness(begin_state: ActorState, end_state: ActorState) -> (float, Decimal):
    # TODO simulation can continue slightly passed the defined sim time.

    # distance traveled on the xy plane
    distance = float (math.sqrt((begin_state.position[0] - end_state.position[0]) ** 2 + ((begin_state.position[1] - end_state.position[1]) ** 2)))

    # take remaining power of end state
    remaining_power = end_state.remaining_power

    return (distance, remaining_power)



