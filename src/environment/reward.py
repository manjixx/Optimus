import numpy as np
from typing import Dict, Any, List
from utils.logger import get_logger

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

        # 添加奖励缩放参数
        self.reward_scale = self.reward_config.get('reward_scale', 0.01)
        self.max_reward = self.reward_config.get('max_reward', 100)
        self.min_reward = self.reward_config.get('min_reward', -100)

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

        # 缩放和裁剪奖励以防止数值不稳定
        reward = np.clip(reward, self.min_reward, self.max_reward)
        reward *= self.reward_scale

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
        recent_traffic = np.array(traffic_history[-7:]) if len(traffic_history) >= 7 else np.array(traffic_history)

        if len(recent_traffic) < 2:
            return 0.0

        # 添加小值以防止除零错误
        epsilon = 1e-5
        recent_traffic = np.where(recent_traffic == 0, epsilon, recent_traffic)

        # 计算流量变化率
        changes = np.diff(recent_traffic)
        relative_changes = np.abs(changes / recent_traffic[:-1])

        # 使用arctan平滑变化率，防止极端值
        smoothed_changes = np.arctan(relative_changes)

        # 惩罚大的流量变化（使用较小的缩放因子）
        penalty = -np.mean(smoothed_changes) * 10

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

        # 添加小值以防止NaN
        scenario_variances = np.array(scenario_variances)
        scenario_variances = np.where(scenario_variances == 0, 1e-5, scenario_variances)

        avg_variance = np.mean(scenario_variances)
        worst_variance = np.max(scenario_variances)

        robust_reward = -(
                self.reward_config['avg_variance_weight'] * avg_variance +
                self.reward_config['worst_variance_weight'] * worst_variance
        )

        # 缩放稳健性奖励
        robust_reward = np.clip(robust_reward, self.min_reward, self.max_reward)
        robust_reward *= self.reward_scale

        return robust_reward