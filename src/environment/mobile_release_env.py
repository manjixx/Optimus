import numpy as np
from typing import Dict, Any, List, Tuple
from .base_env import BaseMobileReleaseEnv
from ..data_processing.data_loader import DataLoader
from ..data_processing.scenario_generator import ScenarioGenerator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MobileReleaseEnv(BaseMobileReleaseEnv):
    """手机发布环境实现类，增强版本支持多场景和真实数据"""

    def __init__(self, config_path: str = "config/base.yaml"):
        """初始化手机发布环境

        Args:
            config_path: 配置文件路径（现在包含数据和环境配置）
        """
        super(MobileReleaseEnv, self).__init__(config_path)

        # 初始化数据加载器和场景生成器
        self.data_loader = DataLoader(config_path)  # 使用同一个配置文件
        self.scenario_generator = ScenarioGenerator(config_path)  # 使用同一个配置文件

        # 场景相关变量
        self.scenarios = None
        self.current_scenario = None
        self.scenario_traffic_baselines = None

        logger.info("Mobile release environment initialized")

    def reset(self) -> np.ndarray:
        """重置环境状态，包含场景初始化

        Returns:
            初始状态观测值
        """
        # 调用父类重置
        state = super(MobileReleaseEnv, self).reset()

        # 加载或生成场景
        if self.scenarios is None:
            self._load_scenarios()

        # 随机选择一个场景
        self.current_scenario = np.random.choice(len(self.scenarios))
        scenario_data = self.scenarios[self.current_scenario]

        # 设置场景特定的流量基线
        self.scenario_traffic_baselines = scenario_data['traffic_baseline']

        logger.info(f"Environment reset with scenario {self.current_scenario}")
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一个时间步，考虑多场景影响

        Args:
            action: 要执行的动作

        Returns:
            tuple: (观测值, 奖励, 是否结束, 信息字典)
        """
        # 调用父类步骤
        state, reward, done, info = super(MobileReleaseEnv, self).step(action)

        # 在多场景下计算稳健性奖励
        if done:
            robust_reward = self._calculate_robust_reward()
            reward += robust_reward
            info['robust_reward'] = robust_reward

        return state, reward, done, info

    def _load_scenarios(self) -> None:
        """加载或生成场景数据"""
        logger.info("Loading scenarios")

        try:
            # 尝试从文件加载场景
            scenarios = self.scenario_generator.load_scenarios()
            if scenarios:
                self.scenarios = scenarios
                logger.info(f"Loaded {len(scenarios)} scenarios from file")
                return
        except Exception as e:
            logger.warning(f"Failed to load scenarios from file: {e}")

        # 生成新场景
        logger.info("Generating new scenarios")
        scenario_data = self.scenario_generator.run_scenario_generation()
        self.scenarios = scenario_data['traffic_scenarios']

        # 为每个场景提取流量基线
        for i, scenario in enumerate(self.scenarios):
            self.scenarios[i] = {
                'traffic_baseline': scenario['scenario_traffic'].values,
                'impact_factors': scenario.get('impact_factors', {})
            }

        logger.info(f"Generated {len(self.scenarios)} scenarios")

    def _simulate_daily_traffic(self) -> float:
        """模拟每日流量，考虑场景基线

        Returns:
            当日流量值
        """
        # 使用场景特定的流量基线
        if self.scenario_traffic_baselines is not None and self.current_day < len(self.scenario_traffic_baselines):
            base_traffic = self.scenario_traffic_baselines[self.current_day]
        else:
            # 回退到父类实现
            base_traffic = np.mean(self.traffic_history[-7:]) if len(self.traffic_history) >= 7 else 1000

        # 日期影响（周末/节假日）
        day_factor = 1.0
        if self.current_day % 7 in [5, 6]:  # 周末
            day_factor *= self.config['environment']['traffic_simulation']['weekend_factor']

        # 版本发布影响
        release_impact = 0.0
        if self.release_calendar[self.current_day] == 1:  # 今日有发布
            release_impact = self._calculate_release_impact()

        # 随机波动（基于场景不确定性）
        scenario_uncertainty = self.scenarios[self.current_scenario].get('uncertainty', 0.1)
        variation = np.random.normal(0, scenario_uncertainty * base_traffic)

        # 计算最终流量
        daily_traffic = base_traffic * day_factor + release_impact + variation

        return max(daily_traffic, 0)  # 确保非负

    def _calculate_robust_reward(self) -> float:
        """计算稳健性奖励，基于多场景表现

        Returns:
            稳健性奖励值
        """
        if not self.scenarios:
            return 0.0

        # 在当前策略下模拟所有场景
        scenario_variances = []

        for scenario in self.scenarios:
            # 模拟该场景下的流量
            scenario_traffic = self._simulate_scenario_traffic(scenario)

            # 计算方差
            variance = np.var(scenario_traffic)
            scenario_variances.append(variance)

        # 计算平均方差和最坏情况方差
        avg_variance = np.mean(scenario_variances)
        worst_variance = np.max(scenario_variances)

        # 计算稳健性奖励（负值，因为方差越小越好）
        robust_reward = -(
                self.config['environment']['reward']['avg_variance_weight'] * avg_variance +
                self.config['environment']['reward']['worst_variance_weight'] * worst_variance
        )

        return robust_reward

    def _simulate_scenario_traffic(self, scenario: Dict[str, Any]) -> List[float]:
        """模拟特定场景下的流量

        Args:
            scenario: 场景数据

        Returns:
            场景流量列表
        """
        # 简化实现：基于场景基线和发布日历计算流量
        traffic_baseline = scenario['traffic_baseline']
        scenario_traffic = []

        for day in range(self.episode_length):
            daily_traffic = traffic_baseline[day] if day < len(traffic_baseline) else traffic_baseline[-1]

            # 添加发布影响
            if self.release_calendar[day] == 1:
                impact_factors = scenario.get('impact_factors', {})
                impact_factor = impact_factors.get('default', 1.0)
                daily_traffic += self._calculate_release_impact() * impact_factor

            scenario_traffic.append(daily_traffic)

        return scenario_traffic

    def set_version_info(self, version_info: Dict[str, Any]) -> None:
        """设置版本信息

        Args:
            version_info: 版本信息字典
        """
        self.version_info = version_info
        logger.info("Version info updated")

    def set_holidays(self, holidays: List[int]) -> None:
        """设置节假日

        Args:
            holidays: 节假日列表（日期索引）
        """
        self.validator.set_holidays(holidays)
        logger.info(f"Set {len(holidays)} holidays")