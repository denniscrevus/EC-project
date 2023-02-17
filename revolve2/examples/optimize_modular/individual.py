from genotype import Genotype
from decimal import *


class Individual:
    def __init__(self, genotype: Genotype, objectives: (Decimal, Decimal)):
        self.genotype = genotype
        self.objectives = objectives
