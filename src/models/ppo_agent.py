from typing import Dict, Any, Optional, Callable
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from .base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class PPOAgent(BaseAgent):
    """PPO算法智能体实现"""

    def __init__(self, env: VecEnv, config: Dict[str, Any]):
        """初始化PPO智能体

        Args:
            env: 向量化环境
            config: 配置字典
        """
        super().__init__(env, config)
        self.ppo_config = config['models']['ppo']
        self._create_model()

    def _create_model(self) -> PPO:
        """创建PPO模型实例

        Returns:
            PPO模型实例
        """
        policy_kwargs = self.ppo_config.get('policy_kwargs', {})

        # policy_kwargs = self.ppo_config.get('policy_kwargs', {}).copy()  # 创建副本以避免修改原始配置

        # 处理激活函数字符串
        activation_fn_str = policy_kwargs.get('activation_fn')
        if activation_fn_str and isinstance(activation_fn_str, str):
            # 将字符串映射到相应的激活函数类
            activation_map = {
                'tanh': torch.nn.Tanh,
                'relu': torch.nn.ReLU,
                'elu': torch.nn.ELU,
                # 添加其他激活函数...
            }

            if activation_fn_str in activation_map:
                policy_kwargs['activation_fn'] = activation_map[activation_fn_str]
            else:
                logger.warning(f"Unknown activation function: {activation_fn_str}, using default")
                # 移除无效的激活函数参数，让PPO使用默认值
                policy_kwargs.pop('activation_fn', None)

        # 处理学习率调度
        learning_rate = self.ppo_config['learning_rate']['initial']
        learning_rate_schedule = self.ppo_config['learning_rate']['schedule']

        if learning_rate_schedule != 'constant':
            # 创建学习率调度函数
            learning_rate = self._create_learning_rate_schedule(
                learning_rate, learning_rate_schedule
            )

        # 创建模型
        self.model = PPO(
            policy=self.ppo_config['policy'],
            env=self.env,
            learning_rate=learning_rate,
            n_steps=self.ppo_config['n_steps'],
            batch_size=self.ppo_config['batch_size'],
            n_epochs=self.ppo_config['n_epochs'],
            gamma=self.ppo_config['gamma'],
            gae_lambda=self.ppo_config['gae_lambda'],
            clip_range=self.ppo_config['clip_range'],
            clip_range_vf=self.ppo_config.get('clip_range_vf'),
            ent_coef=self.ppo_config['ent_coef'],
            vf_coef=self.ppo_config['vf_coef'],
            max_grad_norm=self.ppo_config['max_grad_norm'],
            use_sde=self.ppo_config.get('use_sde', False),
            sde_sample_freq=self.ppo_config.get('sde_sample_freq', -1),
            target_kl=self.ppo_config.get('target_kl'),
            tensorboard_log="./logs/ppo/",
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="auto"
        )

        logger.info("PPO model created")
        return self.model

    def _create_learning_rate_schedule(self, initial_lr: float, schedule_type: str) -> Callable:
        """创建学习率调度函数

        Args:
            initial_lr: 初始学习率
            schedule_type: 调度类型

        Returns:
            学习率调度函数
        """
        if schedule_type == "linear":
            def linear_schedule(progress_remaining: float) -> float:
                """线性衰减学习率"""
                return initial_lr * progress_remaining

            return linear_schedule

        elif schedule_type == "cosine":
            def cosine_schedule(progress_remaining: float) -> float:
                """余弦衰减学习率"""
                return initial_lr * 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))

            return cosine_schedule

        else:
            # 默认返回常数学习率
            logger.warning(f"Unknown schedule type: {schedule_type}, using constant")
            return initial_lr

    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None, **kwargs) -> None:
        """训练PPO模型

        Args:
            total_timesteps: 总训练步数
            callback: 训练回调函数
            **kwargs: 其他训练参数
        """
        if self.model is None:
            self._create_model()

        logger.info(f"Starting PPO training for {total_timesteps} timesteps")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                **kwargs
            )
            self.is_trained = True
            logger.info("PPO training completed")

        except Exception as e:
            logger.error(f"PPO training failed: {e}")
            raise

    def get_entropy(self) -> float:
        """获取策略熵

        Returns:
            策略熵值
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        # 获取当前策略的熵
        policy: BasePolicy = self.model.policy
        if hasattr(policy, 'get_entropy'):
            return policy.get_entropy()

        # 如果没有直接的方法，尝试计算近似熵
        try:
            # 这是一个近似计算，实际实现可能需要根据具体策略调整
            observations = self.env.reset()
            actions, values, log_probs = policy(observations)
            entropy = -log_probs.mean().item()
            return entropy
        except Exception as e:
            logger.warning(f"Could not compute entropy: {e}")
            return 0.0

    def get_kl_divergence(self) -> float:
        """获取KL散度

        Returns:
            KL散度值
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        # 尝试获取KL散度
        try:
            # 这是一个近似计算，实际实现可能需要根据具体策略调整
            policy: BasePolicy = self.model.policy
            observations = self.env.reset()
            _, _, log_probs = policy(observations)

            # 假设旧策略的log_probs已经存储（实际PPO实现中会有）
            if hasattr(policy, 'old_log_probs'):
                old_log_probs = policy.old_log_probs
                kl = (old_log_probs - log_probs).mean().exp().item()
                return kl

            return 0.0
        except Exception as e:
            logger.warning(f"Could not compute KL divergence: {e}")
            return 0.0