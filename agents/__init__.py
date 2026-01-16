"""Baseline agents and opponent wrappers for Pişti RL."""

from agents.baselines import (
    RandomValidAgent,
    GreedyCaptureAgent,
    PistiHunterAgent,
)
from agents.opponents import (
    FrozenCheckpointOpponent,
    OpponentPool,
    SelfPlayOpponent,
)
from agents.probabilistic_agent import (
    ProbabilisticOptimalAgent,
    BeliefTracker,
    ActionEvaluator,
)

# RL agents (optional imports)
try:
    from agents.nfsp_agent import NFSPAgent
    NFSP_AVAILABLE = True
except ImportError:
    NFSP_AVAILABLE = False

try:
    from agents.deep_cfr_agent import DeepCFRAgent
    DEEP_CFR_AVAILABLE = True
except ImportError:
    DEEP_CFR_AVAILABLE = False

try:
    from agents.r2d2_agent import R2D2Agent
    R2D2_AVAILABLE = True
except ImportError:
    R2D2_AVAILABLE = False

__all__ = [
    "RandomValidAgent",
    "GreedyCaptureAgent",
    "PistiHunterAgent",
    "FrozenCheckpointOpponent",
    "OpponentPool",
    "SelfPlayOpponent",
    "ProbabilisticOptimalAgent",
    "BeliefTracker",
    "ActionEvaluator",
]

if NFSP_AVAILABLE:
    __all__.append("NFSPAgent")
if DEEP_CFR_AVAILABLE:
    __all__.append("DeepCFRAgent")
if R2D2_AVAILABLE:
    __all__.append("R2D2Agent")
