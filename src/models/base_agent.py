import abc
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from utils.logger import get_logger
from utils.serialization import save_model, load_model

logger = get_logger(__name__)


class BaseAgent(abc.ABC):
    """智能体基类，定义所有智能体的通用接口"""

    def __init__(self, env: VecEnv, config: Dict[str, Any]):
        """初始化智能体

        Args:
            env: 向量化环境
            config: 配置字典
        """
        self.env = env
        self.config = config
        self.model: Optional[BaseAlgorithm] = None
        self.is_trained = False

        logger.info(f"Initialized {self.__class__.__name__}")

    @abc.abstractmethod
    def _create_model(self) -> BaseAlgorithm:
        """创建模型实例

        Returns:
            模型实例
        """
        pass

    @abc.abstractmethod
    def train(self, total_timesteps: int, callback=None, **kwargs) -> None:
        """训练模型

        Args:
            total_timesteps: 总训练步数
            callback: 训练回调函数
            **kwargs: 其他训练参数
        """
        pass

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """预测动作

        Args:
            observation: 观测值
            deterministic: 是否使用确定性策略

        Returns:
            预测的动作
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")

        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        """保存模型

        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("No model to save.")

        save_model(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """加载模型

        Args:
            path: 模型路径
        """
        self.model = load_model(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数

        Returns:
            模型参数字典
        """
        if self.model is None:
            return {}

        return {
            'policy': self.model.policy,
            'learning_rate': getattr(self.model, 'learning_rate', None),
            'gamma': getattr(self.model, 'gamma', None),
            'n_steps': getattr(self.model, 'n_steps', None),
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置模型参数

        Args:
            params: 参数字典
        """
        if self.model is None:
            logger.warning("Cannot set parameters: model not initialized")
            return

        for key, value in params.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
                logger.info(f"Set parameter {key} = {value}")

    def evaluate(self, eval_env: VecEnv, n_episodes: int = 10) -> Dict[str, float]:
        """评估模型性能

        Args:
            eval_env: 评估环境
            n_episodes: 评估回合数

        Returns:
            评估指标字典
        """
        from stable_baselines3.common.evaluation import evaluate_policy

        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")

        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=n_episodes, deterministic=True
        )

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'n_episodes': n_episodes
        }