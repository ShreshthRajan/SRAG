"""
SRAG-V Package
Learning Verification Through Self-Play - 4-Player Architecture
"""

from .orchestrator import SRAGVOrchestrator
from .models import (
    BasePlayer,
    PlayerConfig,
    ProblemGenerator,
    SolutionGenerator,
    VerificationGenerator,
    MetaVerifier
)

__version__ = "1.0.0"
__all__ = [
    "SRAGVOrchestrator",
    "BasePlayer",
    "PlayerConfig",
    "ProblemGenerator", 
    "SolutionGenerator",
    "VerificationGenerator",
    "MetaVerifier"
]