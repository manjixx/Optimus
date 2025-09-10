import os
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import yaml
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from environment import MobileReleaseEnv
from models import PPOAgent
from utils.logger import get_logger
from utils.serialization import save_config, save_results
from .callback import TrainingCallback, CurriculumCallback, DomainRandomizationCallback
from .curriculum import CurriculumManager
from .domain_randomization import DomainRandomizer

logger = get_logger(__name__)


class RLTrainer:
    """强化学习训练器，管理整个训练流程"""

    def __init__(self, config_path: str = "config/train.yaml"):
        """初始化训练器

        Args:
            config_path: 训练配置文件路径
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.training_config = self.config['training']

        # 初始化组件
        self.env = None
        self.eval_env = None
        self.agent = None
        self.callbacks = None

        # 训练状态
        self.start_time = None
        self.total_timesteps = 0
        self.best_mean_reward = -np.inf
        self.training_history = []

        # 初始化课程学习和域随机化
        self.curriculum_manager = None
        self.domain_randomizer = None

        if self.training_config['curriculum']['enabled']:
            self.curriculum_manager = CurriculumManager(self.training_config['curriculum'])

        if self.training_config['domain_randomization']['enabled']:
            self.domain_randomizer = DomainRandomizer(self.training_config['domain_randomization'])

        logger.info("RL Trainer initialized")

    def _load_config(self, config_path: str) -> dict:
        try:
            # 首先尝试直接使用传入的路径
            config_path_obj = Path(config_path)

            # 如果路径是绝对路径且存在，直接使用
            if config_path_obj.is_absolute() and config_path_obj.exists():
                abs_config_path = config_path_obj
            else:
                # 获取项目根目录的几种可能方式
                possible_roots = [
                    # 方式1: 从当前文件位置计算
                    Path(__file__).parent.parent.parent,  # src/training -> 项目根目录
                    # 方式2: 从工作目录计算
                    Path.cwd(),
                    # 方式3: 从环境变量获取
                    Path(os.environ.get('PROJECT_ROOT', '')),
                ]

                # 尝试每种可能的根目录
                for project_root in possible_roots:
                    if not project_root:  # 跳过空路径
                        continue

                    abs_config_path = (project_root / config_path).resolve()
                    print(f"Trying config path: {abs_config_path}")

                    if abs_config_path.exists():
                        print(f"Found config at: {abs_config_path}")
                        break
                else:
                    # 如果所有尝试都失败，抛出错误
                    raise FileNotFoundError(f"Config file not found: {config_path}. "
                                            f"Tried paths: {[str(project_root / config_path) for project_root in possible_roots if project_root]}")

            with open(abs_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _create_env(self, eval_mode: bool = False) -> Any:
        """创建环境

        Args:
            eval_mode: 是否为评估模式

        Returns:
            环境实例
        """
        # 设置环境参数
        env_config = {
            'config_path': "config/base.yaml",
            'eval_mode': eval_mode
        }

        # 应用课程学习设置
        if self.curriculum_manager and not eval_mode:
            current_level = self.curriculum_manager.get_current_level()
            env_config.update(current_level.get('env_params', {}))

        # 应用域随机化
        if self.domain_randomizer and not eval_mode:
            randomized_params = self.domain_randomizer.get_randomized_parameters()
            env_config.update(randomized_params)

        # 创建环境函数
        def make_env():
            env = MobileReleaseEnv(**env_config)
            return Monitor(env)  # 包装环境以监控统计信息

        # 创建向量化环境 - 暂时使用单个环境避免多进程问题
        num_envs = 1  # 强制使用单个环境

        if num_envs == 1:
            return DummyVecEnv([make_env])
        else:
            return SubprocVecEnv([make_env for _ in range(num_envs)])

    def _create_agent(self) -> PPOAgent:
        """创建智能体

        Returns:
            PPO智能体实例
        """
        if self.env is None:
            raise ValueError("Environment must be created before agent")

        return PPOAgent(self.env, self.config)

    def _create_callbacks(self) -> CallbackList:
        """创建回调函数列表

        Returns:
            回调函数列表
        """
        callbacks = []

        # 基础训练回调
        callbacks.append(TrainingCallback(
            checkpoint_interval=self.training_config['save']['checkpoint_interval'],
            best_model_metric=self.training_config['save']['best_model_metric'],
            keep_last_n=self.training_config['save']['keep_last_n_checkpoints']
        ))

        # 课程学习回调
        if self.curriculum_manager:
            callbacks.append(CurriculumCallback(self.curriculum_manager))

        # 域随机化回调
        if self.domain_randomizer:
            callbacks.append(DomainRandomizationCallback(
                self.domain_randomizer,
                update_frequency=self.training_config['domain_randomization']['update_frequency']
            ))

        return CallbackList(callbacks)

    def setup(self) -> None:
        """设置训练环境"""
        logger.info("Setting up training environment")

        # 创建训练环境
        self.env = self._create_env(eval_mode=False)

        # 创建评估环境
        self.eval_env = self._create_env(eval_mode=True)

        # 创建智能体
        self.agent = self._create_agent()

        # 创建回调函数
        self.callbacks = self._create_callbacks()

        # 设置随机种子
        seed = self.training_config.get('seed')
        if seed is not None:
            np.random.seed(seed)
            # 不再调用 env.seed()，因为 Gymnasium 使用不同的方式设置种子

        logger.info("Training setup completed")

    def train(self) -> Dict[str, Any]:
        """执行训练

        Returns:
            训练结果字典
        """
        if self.agent is None or self.env is None:
            raise ValueError("Trainer not setup. Call setup() first.")

        logger.info("Starting training")
        self.start_time = time.time()

        total_timesteps = self.training_config['total_timesteps']
        eval_frequency = self.training_config['evaluation']['eval_frequency']
        n_eval_episodes = self.training_config['evaluation']['n_eval_episodes']

        # 训练循环
        timesteps_so_far = 0

        while timesteps_so_far < total_timesteps:
            # 执行一步训练
            timesteps_before = self.agent.model.num_timesteps
            self.agent.model.learn(
                total_timesteps=min(eval_frequency, total_timesteps - timesteps_so_far),
                callback=self.callbacks,
                reset_num_timesteps=False
            )
            timesteps_after = self.agent.model.num_timesteps
            timesteps_so_far = timesteps_after

            # 执行评估
            eval_results = self.evaluate()
            self.training_history.append({
                'timesteps': timesteps_so_far,
                'eval_results': eval_results,
                'timestamp': time.time()
            })

            # 更新最佳模型
            current_reward = eval_results.get('mean_reward', -np.inf)
            if current_reward > self.best_mean_reward:
                self.best_mean_reward = current_reward
                self._save_best_model()

            # 记录进度
            progress = timesteps_so_far / total_timesteps * 100
            elapsed_time = time.time() - self.start_time
            estimated_total = elapsed_time / (timesteps_so_far / total_timesteps) if timesteps_so_far > 0 else 0
            remaining = estimated_total - elapsed_time

            logger.info(
                f"Progress: {progress:.1f}% ({timesteps_so_far}/{total_timesteps}) | "
                f"Elapsed: {elapsed_time:.1f}s | Remaining: {remaining:.1f}s | "
                f"Mean Reward: {current_reward:.2f} | Best: {self.best_mean_reward:.2f}"
            )

        # 训练完成
        training_time = time.time() - self.start_time
        logger.info(f"Training completed in {training_time:.1f} seconds")

        # 保存最终结果
        final_results = {
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'best_mean_reward': self.best_mean_reward,
            'training_history': self.training_history,
            'final_evaluation': self.evaluate()
        }

        self._save_results(final_results)

        return final_results

    def evaluate(self) -> Dict[str, Any]:
        """评估当前策略

        Returns:
            评估结果字典
        """
        if self.agent is None or self.eval_env is None:
            raise ValueError("Trainer not setup. Call setup() first.")

        n_eval_episodes = self.training_config['evaluation']['n_eval_episodes']
        deterministic = self.training_config['evaluation']['deterministic_eval']

        # 执行评估 - 注意：当 return_episode_rewards=True 时，返回的是两个列表
        episode_rewards, episode_lengths = evaluate_policy(
            self.agent.model,
            self.eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True
        )

        # 计算均值和标准差
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        # 收集额外指标
        traffic_variances = []

        # 我们需要重新运行评估来收集流量数据，因为 evaluate_policy 不提供这些信息
        for _ in range(n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            traffic_values = []

            while not done:
                action, _ = self.agent.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.eval_env.step(action)

                if 'daily_traffic' in info[0]:
                    traffic_values.append(info[0]['daily_traffic'])

            if traffic_values:
                traffic_variances.append(np.var(traffic_values))

        # 计算额外指标
        metrics = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'mean_traffic_variance': float(np.mean(traffic_variances)) if traffic_variances else 0.0,
            'worst_case_variance': float(np.max(traffic_variances)) if traffic_variances else 0.0,
            'n_episodes': n_eval_episodes
        }

        return metrics

    def _save_best_model(self) -> None:
        """保存最佳模型"""
        if self.agent is None:
            return

        best_model_path = os.path.join("models", "best", "ppo_mobile_release")
        self.agent.save(best_model_path)
        logger.info(f"Best model saved to {best_model_path}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存训练结果

        Args:
            results: 训练结果字典
        """
        # 创建结果目录
        os.makedirs("results/training", exist_ok=True)

        # 生成结果文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results/training/training_results_{timestamp}.json"

        # 保存结果
        save_results(results, results_file)
        logger.info(f"Training results saved to {results_file}")

        # 保存配置
        config_file = f"results/training/training_config_{timestamp}.yaml"
        save_config(self.config, config_file)

    def close(self) -> None:
        """关闭训练器，释放资源"""
        if self.env:
            self.env.close()
        if self.eval_env:
            self.eval_env.close()
        logger.info("Trainer closed")