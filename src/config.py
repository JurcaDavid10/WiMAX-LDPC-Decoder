from pathlib import Path

# Expansion factor
Z = 96

# WiMAX rate 5/6 code dimensions
NB_ROWS = 4
NB_COLS = 24
N = 2304
M = 384
K = 1920
CODE_RATE = K / N
BSC_CROSSOVER_PROB = 0.005
BSC_LLR_MAGNITUDE = 5
LLR_SATURATION_LIMIT = 7

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BMAT_PATH = PROJECT_ROOT / "data" / "wimax.rate_5_6.bmat"