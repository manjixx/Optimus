import numpy as np
from typing import Dict, Any, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RewardCalculator:
    """奖励计算类，负责计算环境奖励"""

    def __init__(self, config: Dict[str, Any]):
        """初始化奖励计算器

        Args:
            config: 环境配置
        """
        self.config = config
        self.reward_config = config['reward']

        logger.info("Reward calculator initialized")

    def calculate_reward(self, action: int, is_valid: bool,
                         traffic_history: List[float],
                         release_calendar: np.ndarray) -> float:
        """计算奖励

        Args:
            action: 执行的动作
            is_valid: 动作是否有效
            traffic_history: 流量历史
            release_calendar: 发布日历

        Returns:
            奖励值
        """
        reward = 0.0

        # 基础奖励：鼓励发布但不过度发布
        if action == 1 and is_valid:
            # 有效发布奖励（与版本价值相关）
            reward += self._calculate_release_reward()
        elif action == 1 and not is_valid:
            # 无效发布惩罚
            reward += self.reward_config['illegal_action_penalty']
            logger.debug(f"Invalid action penalty: {reward}")

        # 流量平滑奖励（鼓励流量稳定）
        if len(traffic_history) >= 2:
            traffic_reward = self._calculate_traffic_reward(traffic_history)
            reward += traffic_reward

        return reward

    def _calculate_release_reward(self) -> float:
        """计算发布奖励

        Returns:
            发布奖励值
        """
        # 简化实现：固定小奖励
        return 10.0

    def _calculate_traffic_reward(self, traffic_history: List[float]) -> float:
        """计算流量奖励

        Args:
            traffic_history: 流量历史

        Returns:
            流量奖励值
        """
        # 使用最近一段时间的数据
        recent_traffic = traffic_history[-7:] if len(traffic_history) >= 7 else traffic_history

        if len(recent_traffic) < 2:
            return 0.0

        # 计算流量变化率
        changes = np.diff(recent_traffic)
        relative_changes = np.abs(changes / recent_traffic[:-1])

        # 惩罚大的流量变化
        penalty = -np.mean(relative_changes) * 100

        return penalty

    def calculate_robust_reward(self, scenario_variances: List[float]) -> float:
        """计算稳健性奖励（用于多场景）

        Args:
            scenario_variances: 各场景下的流量方差

        Returns:
            稳健性奖励值
        """
        if not scenario_variances:
            return 0.0

        avg_variance = np.mean(scenario_variances)
        worst_variance = np.max(scenario_variances)

        robust_reward = -(
                self.reward_config['avg_variance_weight'] * avg_variance +
                self.reward_config['worst_variance_weight'] * worst_variance
        )

        return robust_reward