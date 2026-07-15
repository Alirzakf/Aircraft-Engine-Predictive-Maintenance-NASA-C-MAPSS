"""
Central configuration for the finalized modeling pipeline.

The temporal feature parameters and RUL cap were selected through
controlled ablation experiments in Notebook 06.
"""

ROLLING_WINDOW = 10
EMA_SPAN = 10

RUL_CAP = 125
NEAR_CONSTANT_VARIANCE_THRESHOLD = 1e-10