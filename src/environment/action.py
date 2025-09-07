import gym
from gym import spaces
import numpy as np
from typing import Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ActionSpace:
    """动作空间类，负责管理动作空间定义"""

    def __init__(self, config: Dict[str, Any]):
        """初始化动作空间

        Args:
            config: 环境配置
        """
        self.config = config
        self.action_config = config['action']

        # 动作定义
        self.actions = {
            0: "NO_OP",  # 不操作
            1: "RELEASE"  # 发布版本
        }

        logger.info("Action space initialized")

    def get_gym_space(self) -> spaces.Space:
        """获取Gym动作空间

        Returns:
            Gym动作空间
        """
        space_type = self.action_config['space_type']
        num_actions = self.action_config['num_actions']

        if space_type == "discrete":
            return spaces.Discrete(num_actions)
        elif space_type == "multi_discrete":
            # 多离散动作空间示例
            return spaces.MultiDiscrete([num_actions])
        elif space_type == "box":
            # 连续动作空间示例
            return spaces.Box(low=0, high=1, shape=(num_actions,), dtype=np.float32)
        else:
            logger.warning(f"Unknown space type: {space_type}, using discrete")
            return spaces.Discrete(num_actions)

    def get_action_name(self, action: int) -> str:
        """获取动作名称

        Args:
            action: 动作值

        Returns:
            动作名称
        """
        return self.actions.get(action, f"UNKNOWN_{action}")

    def decode_action(self, action) -> int:
        """解码动作（处理多离散或连续动作）

        Args:
            action: 原始动作

        Returns:
            解码后的动作值
        """
        space_type = self.action_config['space_type']

        if space_type == "discrete":
            return int(action)
        elif space_type == "multi_discrete":
            return int(action[0])  # 取第一个维度
        elif space_type == "box":
            # 将连续值转换为离散动作
            return 1 if action[0] > 0.5 else 0
        else:
            logger.warning(f"Unknown space type: {space_type}, treating as discrete")
            return int(action)