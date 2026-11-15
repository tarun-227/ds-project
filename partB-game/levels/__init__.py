"""Level registry. Order here = order of play; later levels unlock as earlier ones are passed."""
from levels.level1_overfitting import OverfittingLevel
from levels.level2_bias_variance import BiasVarianceLevel

LEVELS = [OverfittingLevel, BiasVarianceLevel]
