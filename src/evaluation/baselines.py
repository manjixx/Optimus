import numpy as np
from typing import Dict, Any, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseBaselinePolicy:
    """基准策略基类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化基准策略

        Args:
            config: 策略配置
        """
        self.config = config
        self.name = config.get('name', 'unknown')
        self.description = config.get('description', '')

    def decide(self, observation) -> int:
        """决策函数

        Args:
            observation: 观测值

        Returns:
            动作
        """
        raise NotImplementedError("Subclasses must implement decide method")


class RuleBasedPolicy(BaseBaselinePolicy):
    """基于规则的策略（模拟人工专家）"""

    def __init__(self, config: Dict[str, Any]):
        """初始化基于规则的策略

        Args:
            config: 策略配置
        """
        super().__init__(config)
        self.rules = config.get('rules', [])

    def decide(self, observation) -> int:
        """基于规则进行决策

        Args:
            observation: 观测值

        Returns:
            动作
        """
        # 提取状态信息（假设状态是标准化后的向量）
        # 这里需要根据实际状态设计进行解析
        current_day = int(observation[0] * 15 + 15)  # 假设状态已归一化到[-1,1]
        is_weekend = observation[5] > 0  # 假设第5个特征是周末标识
        traffic_trend = observation[10]  # 假设第10个特征是流量趋势

        # 应用规则
        for rule in self.rules:
            if self._evaluate_rule(rule, current_day, is_weekend, traffic_trend):
                return rule.get('action', 0)

        # 默认不发布
        return 0

    def _evaluate_rule(self, rule: Dict[str, Any], current_day: int,
                       is_weekend: bool, traffic_trend: float) -> bool:
        """评估规则条件

        Args:
            rule: 规则字典
            current_day: 当前天数
            is_weekend: 是否是周末
            traffic_trend: 流量趋势

        Returns:
            是否满足规则条件
        """
        conditions = rule.get('conditions', {})

        # 检查所有条件
        for condition_type, condition_value in conditions.items():
            if condition_type == 'max_day' and current_day > condition_value:
                return False
            elif condition_type == 'min_day' and current_day < condition_value:
                return False
            elif condition_type == 'avoid_weekend' and is_weekend:
                return False
            elif condition_type == 'traffic_trend_threshold' and traffic_trend < condition_value:
                return False

        return True


class OptimizationPolicy(BaseBaselinePolicy):
    """优化策略（模拟遗传算法等优化方法）"""

    def __init__(self, config: Dict[str, Any]):
        """初始化优化策略

        Args:
            config: 策略配置
        """
        super().__init__(config)
        self.optimization_type = config.get('optimization_type', 'genetic')
        self.plan = self._generate_plan()

    def _generate_plan(self) -> List[int]:
        """生成发布计划

        Returns:
            发布计划列表
        """
        # 简化实现：基于配置生成固定计划
        plan_length = 31  # 一个月
        release_days = self.config.get('release_days', [])

        plan = [0] * plan_length
        for day in release_days:
            if day < plan_length:
                plan[day] = 1

        return plan

    def decide(self, observation) -> int:
        """基于优化计划进行决策

        Args:
            observation: 观测值

        Returns:
            动作
        """
        # 提取当前天数
        current_day = int(observation[0] * 15 + 15)  # 假设状态已归一化到[-1,1]

        # 检查计划
        if current_day < len(self.plan):
            return self.plan[current_day]

        return 0


class RandomPolicy(BaseBaselinePolicy):
    """随机策略"""

    def __init__(self, config: Dict[str, Any]):
        """初始化随机策略

        Args:
            config: 策略配置
        """
        super().__init__(config)
        self.release_probability = config.get('release_probability', 0.1)

    def decide(self, observation) -> int:
        """随机决策

        Args:
            observation: 观测值

        Returns:
            动作
        """
        # 简单随机决策
        if np.random.random() < self.release_probability:
            return 1

        return 0