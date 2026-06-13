"""
SRAG-V Training Components.
"""

from .grpo_trainer import GRPOTrainer, GRPOConfig
from .map_elites import MAPElitesArchive, MAPElitesConfig
from .reward_system import RoleConditionedRewardSystem
from .self_play_trainer import SelfPlayTrainer, SelfPlayConfig

__all__ = [
    'GRPOTrainer', 'GRPOConfig',
    'MAPElitesArchive', 'MAPElitesConfig', 
    'RoleConditionedRewardSystem',
    'SelfPlayTrainer', 'SelfPlayConfig'
]