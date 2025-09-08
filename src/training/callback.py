import os
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from utils.logger import get_logger
from utils.serialization import save_model

logger = get_logger(__name__)


class TrainingCallback(BaseCallback):
    """训练回调函数，处理检查点保存和早期停止"""

    def __init__(
            self,
            checkpoint_interval: int = 10000,
            best_model_metric: str = "mean_reward",
            keep_last_n: int = 5,
            verbose: int = 0
    ):
        """初始化训练回调

        Args:
            checkpoint_interval: 检查点保存间隔
            best_model_metric: 最佳模型指标
            keep_last_n: 保留的检查点数量
            verbose: 详细程度
        """
        super(TrainingCallback, self).__init__(verbose)
        self.checkpoint_interval = checkpoint_interval
        self.best_model_metric = best_model_metric
        self.keep_last_n = keep_last_n
        self.best_metric_value = -np.inf
        self.checkpoint_counter = 0
        self.checkpoint_paths = []

    def _on_step(self) -> bool:
        """每一步调用的方法

        Returns:
            是否继续训练
        """
        # 检查是否达到检查点保存间隔
        if self.n_calls % self.checkpoint_interval == 0:
            self._save_checkpoint()

        return True

    def _on_rollout_end(self) -> None:
        """回合结束时调用的方法"""
        # 可以在这里添加额外的日志记录或处理
        pass

    def _on_training_end(self) -> None:
        """训练结束时调用的方法"""
        # 清理旧的检查点
        self._cleanup_old_checkpoints()

    def _save_checkpoint(self) -> None:
        """保存检查点"""
        if self.model is None:
            return

        # 创建检查点目录
        os.makedirs("models/checkpoints", exist_ok=True)

        # 生成检查点文件名
        checkpoint_path = os.path.join(
            "models", "checkpoints",
            f"ppo_checkpoint_{self.checkpoint_counter:04d}_{self.num_timesteps}"
        )

        # 保存模型
        save_model(self.model, checkpoint_path)
        self.checkpoint_paths.append(checkpoint_path)
        self.checkpoint_counter += 1

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # 清理旧的检查点
        if len(self.checkpoint_paths) > self.keep_last_n:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """清理旧的检查点"""
        if len(self.checkpoint_paths) <= self.keep_last_n:
            return

        # 删除最旧的检查点
        while len(self.checkpoint_paths) > self.keep_last_n:
            old_checkpoint = self.checkpoint_paths.pop(0)
            try:
                os.remove(old_checkpoint + ".zip")
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except OSError:
                logger.warning(f"Failed to remove checkpoint: {old_checkpoint}")


class CurriculumCallback(BaseCallback):
    """课程学习回调函数，管理难度级别转换"""

    def __init__(self, curriculum_manager, verbose: int = 0):
        """初始化课程学习回调

        Args:
            curriculum_manager: 课程学习管理器
            verbose: 详细程度
        """
        super(CurriculumCallback, self).__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.last_eval_results = None

    def _on_step(self) -> bool:
        """每一步调用的方法

        Returns:
            是否继续训练
        """
        return True

    def on_rollout_end(self) -> None:
        """回合结束时调用的方法"""
        # 检查是否需要提升难度级别
        if self.last_eval_results and self.curriculum_manager:
            should_advance = self.curriculum_manager.check_progression(self.last_eval_results)

            if should_advance:
                new_level = self.curriculum_manager.advance_level()
                logger.info(f"Advanced to curriculum level: {new_level['name']}")

                # 更新环境参数（如果需要）
                if self.training_env and hasattr(self.training_env, 'set_parameters'):
                    self.training_env.set_parameters(new_level.get('env_params', {}))

    def on_evaluation(self, eval_results: Dict[str, Any]) -> None:
        """评估完成后调用的方法

        Args:
            eval_results: 评估结果
        """
        self.last_eval_results = eval_results


class DomainRandomizationCallback(BaseCallback):
    """域随机化回调函数，管理环境参数随机化"""

    def __init__(self, domain_randomizer, update_frequency: int = 1000, verbose: int = 0):
        """初始化域随机化回调

        Args:
            domain_randomizer: 域随机化器
            update_frequency: 参数更新频率
            verbose: 详细程度
        """
        super(DomainRandomizationCallback, self).__init__(verbose)
        self.domain_randomizer = domain_randomizer
        self.update_frequency = update_frequency

    def _on_step(self) -> bool:
        """每一步调用的方法

        Returns:
            是否继续训练
        """
        # 检查是否达到参数更新频率
        if self.n_calls % self.update_frequency == 0:
            self._update_environment_parameters()

        return True

    def _update_environment_parameters(self) -> None:
        """更新环境参数"""
        if self.domain_randomizer and self.training_env:
            # 生成新的随机参数
            randomized_params = self.domain_randomizer.randomize_parameters()

            # 更新环境参数
            for i in range(self.training_env.num_envs):
                env = self.training_env.envs[i]
                if hasattr(env, 'set_parameters'):
                    env.set_parameters(randomized_params)

            logger.debug("Updated environment parameters with domain randomization")