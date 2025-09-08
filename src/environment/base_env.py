# import gym
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
import yaml
import os
from pathlib import Path
from utils.logger import get_logger
from .state import StateRepresentation
from .action import ActionSpace
from .reward import RewardCalculator
from .validator import ActionValidator

logger = get_logger(__name__)


class BaseMobileReleaseEnv(gym.Env):
    """手机发布环境基类"""

    metadata = {'render.modes': ['human']}

    def __init__(self, config_path: str = "config/base.yaml", **kwargs):
        """初始化环境

        Args:
            config_path: 配置文件路径
            **kwargs: 其他参数（用于兼容性）
        """
        super(BaseMobileReleaseEnv, self).__init__()

        # 加载配置
        self.config = self._load_config(config_path)
        env_config = self.config['environment']  # 从统一配置中获取环境配置

        # 初始化组件
        self.state_representation = StateRepresentation(env_config)
        self.action_space_obj = ActionSpace(env_config)
        self.reward_calculator = RewardCalculator(env_config)
        self.validator = ActionValidator(env_config)

        # 设置Gym接口
        self.action_space = self.action_space_obj.get_gym_space()
        self.observation_space = self.state_representation.get_observation_space()

        # 环境状态变量
        self.current_day = env_config['time']['current_day']
        self.episode_length = env_config['time']['episode_length']
        self.done = False
        self.state = None
        self.release_calendar = None
        self.traffic_history = None
        self.version_info = None

        logger.info("Environment initialized")

    def _load_config(self, config_path: str) -> dict:
        try:
            config_path_obj = Path(config_path)

            # 如果路径是绝对路径且存在，直接使用
            if config_path_obj.is_absolute() and config_path_obj.exists():
                abs_config_path = config_path_obj
            else:
                # 获取项目根目录的几种可能方式
                possible_roots = [
                    Path(__file__).parent.parent.parent,  # src/environment -> 项目根目录
                    Path.cwd(),
                    Path(os.environ.get('PROJECT_ROOT', '')),
                ]

                # 尝试每种可能的根目录
                for project_root in possible_roots:
                    if not project_root:  # 跳过空路径
                        continue

                    abs_config_path = (project_root / config_path).resolve()

                    if abs_config_path.exists():
                        break
                else:
                    # 如果所有尝试都失败，抛出错误
                    raise FileNotFoundError(f"Base config file not found: {config_path}")

            with open(abs_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"加载基础配置文件失败: {str(e)}")
            raise

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """重置环境状态

        Args:
            seed: 随机种子（可选）
            options: 重置选项（可选）

        Returns:
            初始状态观测值和信息字典
        """
        # 设置随机种子（如果提供）
        if seed is not None:
            self._seed(seed)

        self.current_day = 0
        self.done = False

        # 初始化发布日历（全0表示没有发布）
        self.release_calendar = np.zeros(self.episode_length, dtype=np.int32)

        # 初始化流量历史
        self.traffic_history = self._initialize_traffic_history()

        # 初始化版本信息（从配置或外部加载）
        self.version_info = self._initialize_version_info()

        # 生成初始状态
        self.state = self.state_representation.create_state(
            current_day=self.current_day,
            days_remaining=self.episode_length - self.current_day,
            release_calendar=self.release_calendar,
            version_info=self.version_info,
            traffic_history=self.traffic_history
        )

        logger.info("Environment reset")
        return self.state, {}

    def _seed(self, seed=None):
        """设置随机种子

        Args:
            seed: 随机种子

        Returns:
            使用的随机种子
        """
        np.random.seed(seed)
        return [seed]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一个时间步

        Args:
            action: 要执行的动作

        Returns:
            tuple: (观测值, 奖励, 是否终止, 是否截断, 信息字典)
        """
        if self.done:
            logger.warning("Episode is already done, please reset the environment")
            return self.state, 0.0, True, False, {}

        # 验证动作
        is_valid, validation_msg = self.validator.validate_action(
            action, self.current_day, self.release_calendar, self.version_info
        )

        # 执行动作
        if action == 1 and is_valid:  # 发布动作且有效
            self.release_calendar[self.current_day] = 1
            logger.info(f"Day {self.current_day}: Version released")
        elif action == 1 and not is_valid:  # 发布动作但无效
            logger.warning(f"Day {self.current_day}: Invalid release attempt - {validation_msg}")

        # 计算流量影响
        daily_traffic = self._simulate_daily_traffic()
        self.traffic_history.append(daily_traffic)

        # 更新天数
        self.current_day += 1

        # 检查是否结束
        terminated = False
        truncated = False
        if self.current_day >= self.episode_length:
            terminated = True
            logger.info("Episode completed")

        # 更新状态
        self.state = self.state_representation.create_state(
            current_day=self.current_day,
            days_remaining=self.episode_length - self.current_day,
            release_calendar=self.release_calendar,
            version_info=self.version_info,
            traffic_history=self.traffic_history
        )

        # 计算奖励
        reward = self.reward_calculator.calculate_reward(
            action, is_valid, self.traffic_history, self.release_calendar
        )

        # 信息字典
        info = {
            'day': self.current_day,
            'action': action,
            'is_valid': is_valid,
            'validation_msg': validation_msg,
            'daily_traffic': daily_traffic,
            'total_releases': np.sum(self.release_calendar)
        }

        return self.state, reward, terminated, truncated, info

    def render(self, mode: str = 'human') -> Optional[str]:
        """渲染当前环境状态

        Args:
            mode: 渲染模式

        Returns:
            渲染结果（可选）
        """
        if mode == 'human':
            print(f"Day: {self.current_day}/{self.episode_length}")
            print(f"Releases: {np.sum(self.release_calendar)}")
            print(f"Release Calendar: {self.release_calendar}")
            if self.traffic_history:
                print(f"Recent Traffic: {self.traffic_history[-5:]}")
            return None
        else:
            super(BaseMobileReleaseEnv, self).render(mode=mode)

    def close(self) -> None:
        """关闭环境，释放资源"""
        logger.info("Environment closed")

    def _initialize_traffic_history(self) -> list:
        """初始化流量历史数据

        Returns:
            初始流量历史列表
        """
        # 这里可以加载真实历史数据或使用模拟数据
        # 简化实现：使用随机初始值
        base_traffic = 1000  # 基础流量值
        initial_days = 7  # 初始历史天数

        return [base_traffic + np.random.normal(0, 100) for _ in range(initial_days)]

    def _initialize_version_info(self) -> Dict[str, Any]:
        """初始化版本信息

        Returns:
            版本信息字典
        """
        # 这里可以加载真实版本数据或使用模拟数据
        return {
            'user_count': 1000000,
            'package_size': 500,
            'pilot_ratio': 0.1,
            'traffic_pattern_mean': 1.2,
            'cycle_days': 7
        }

    def _simulate_daily_traffic(self) -> float:
        """模拟每日流量

        Returns:
            当日流量值
        """
        # 基础流量（基于历史平均值）
        base_traffic = np.mean(self.traffic_history[-7:]) if len(self.traffic_history) >= 7 else 1000

        # 日期影响（周末/节假日）
        day_factor = 1.0
        if self.current_day % 7 in [5, 6]:  # 周末
            day_factor *= self.config['environment']['traffic_simulation']['weekend_factor']

        # 版本发布影响
        release_impact = 0.0
        if self.release_calendar[self.current_day] == 1:  # 今日有发布
            release_impact = self._calculate_release_impact()

        # 随机波动
        variation = np.random.normal(
            0,
            self.config['environment']['traffic_simulation']['base_variation'] * base_traffic
        )

        # 计算最终流量
        daily_traffic = base_traffic * day_factor + release_impact + variation

        return max(daily_traffic, 0)  # 确保非负

    def _calculate_release_impact(self) -> float:
        """计算版本发布对流量的影响

        Returns:
            流量影响值
        """
        # 基础影响
        base_impact = (
                self.version_info['user_count'] *
                self.version_info['package_size'] *
                self.version_info['traffic_pattern_mean']
        )

        # 不确定性因子
        uncertainty = self.config['environment']['version_impact']['uncertainty_factor']
        impact_factor = np.random.normal(1.0, uncertainty)

        return base_impact * impact_factor

    def get_metrics(self) -> Dict[str, Any]:
        """获取环境性能指标

        Returns:
            性能指标字典
        """
        if not self.done:
            logger.warning("Episode not completed, metrics may be incomplete")

        return {
            'total_traffic': np.sum(self.traffic_history),
            'traffic_variance': np.var(self.traffic_history),
            'max_traffic': np.max(self.traffic_history),
            'min_traffic': np.min(self.traffic_history),
            'release_count': np.sum(self.release_calendar),
            'invalid_attempts': self.validator.invalid_attempts
        }