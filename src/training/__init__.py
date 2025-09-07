"""
训练模块
包含强化学习训练器、回调函数、课程学习和域随机化功能
"""

from .trainer import RLTrainer
from .callback import TrainingCallback, CurriculumCallback, DomainRandomizationCallback
from .curriculum import CurriculumManager
from .domain_randomization import DomainRandomizer

__all__ = [
    'RLTrainer',
    'TrainingCallback',
    'CurriculumCallback',
    'DomainRandomizationCallback',
    'CurriculumManager',
    'DomainRandomizer'
]