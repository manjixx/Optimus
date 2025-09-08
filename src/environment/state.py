import numpy as np
from typing import Dict, Any, List
from gymnasium import spaces
from utils.logger import get_logger

logger = get_logger(__name__)


class StateRepresentation:
    """状态表示类，负责创建和规范化状态向量"""

    def __init__(self, config: Dict[str, Any]):
        """初始化状态表示

        Args:
            config: 环境配置
        """
        self.config = config
        self.state_config = config['state']
        self.normalization = self.state_config.get('normalization', True)

        # 定义状态维度
        self.state_dim = self._calculate_state_dimension()

        logger.info(f"State representation initialized with dimension {self.state_dim}")

    def _calculate_state_dimension(self) -> int:
        """计算状态向量维度

        Returns:
            状态向量维度
        """
        dim = 0

        # 当前天数和剩余天数
        if 'current_day' in self.state_config['features']:
            dim += 1
        if 'days_remaining' in self.state_config['features']:
            dim += 1

        # 发布日历
        if 'release_calendar' in self.state_config['features']:
            dim += self.config['time']['episode_length']

        # 版本信息
        if 'version_info' in self.state_config['features']:
            dim += 5  # user_count, package_size, pilot_ratio, traffic_pattern_mean, cycle_days

        # 流量统计特征
        if 'traffic_stats' in self.state_config['features']:
            dim += 4  # mean, std, 25%, 75%

        # 流量趋势
        if 'traffic_trend' in self.state_config['features']:
            dim += 1  # 最近7天变化率

        return dim

    def get_observation_space(self) -> spaces.Box:
        """获取观测空间

        Returns:
            Gym观测空间
        """
        # 使用-1到1的范围，适用于归一化后的状态
        return spaces.Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)

    def create_state(self, current_day: int, days_remaining: int,
                     release_calendar: np.ndarray, version_info: Dict[str, Any],
                     traffic_history: List[float]) -> np.ndarray:
        """创建状态向量

        Args:
            current_day: 当前天数
            days_remaining: 剩余天数
            release_calendar: 发布日历
            version_info: 版本信息
            traffic_history: 流量历史

        Returns:
            状态向量
        """
        state_components = []

        # 添加各个状态组件
        if 'current_day' in self.state_config['features']:
            state_components.append(self._normalize_value(current_day, 0, self.config['time']['episode_length']))

        if 'days_remaining' in self.state_config['features']:
            state_components.append(self._normalize_value(days_remaining, 0, self.config['time']['episode_length']))

        if 'release_calendar' in self.state_config['features']:
            # 发布日历已经是0/1值，不需要特殊归一化
            state_components.extend(release_calendar.astype(np.float32))

        if 'version_info' in self.state_config['features']:
            version_vec = [
                self._normalize_value(version_info['user_count'], 0, 10000000),
                self._normalize_value(version_info['package_size'], 0, 1000),
                version_info['pilot_ratio'],  # 已经在0-1范围内
                self._normalize_value(version_info['traffic_pattern_mean'], 0.5, 2.0),
                self._normalize_value(version_info['cycle_days'], 1, 30)
            ]
            state_components.extend(version_vec)

        if 'traffic_stats' in self.state_config['features'] and traffic_history:
            # 计算流量统计特征
            traffic_array = np.array(traffic_history[-30:])  # 使用最近30天数据
            if len(traffic_array) > 0:
                mean = np.mean(traffic_array)
                std = np.std(traffic_array)
                q25 = np.percentile(traffic_array, 25)
                q75 = np.percentile(traffic_array, 75)

                traffic_stats = [
                    self._normalize_value(mean, 0, 5000),
                    self._normalize_value(std, 0, 1000),
                    self._normalize_value(q25, 0, 5000),
                    self._normalize_value(q75, 0, 5000)
                ]
                state_components.extend(traffic_stats)
            else:
                # 填充默认值
                state_components.extend([0.0, 0.0, 0.0, 0.0])

        if 'traffic_trend' in self.state_config['features'] and len(traffic_history) >= 14:
            # 计算流量趋势（最近7天与前7天的变化率）
            recent_mean = np.mean(traffic_history[-7:])
            previous_mean = np.mean(traffic_history[-14:-7])

            if previous_mean > 0:
                trend = (recent_mean - previous_mean) / previous_mean
                trend = np.clip(trend, -1.0, 1.0)  # 限制在-1到1之间
            else:
                trend = 0.0

            state_components.append(trend)
        else:
            state_components.append(0.0)  # 默认无趋势

        # 确保状态向量长度正确
        if len(state_components) != self.state_dim:
            logger.warning(f"State dimension mismatch: expected {self.state_dim}, got {len(state_components)}")
            # 填充或截断以匹配预期维度
            if len(state_components) < self.state_dim:
                state_components.extend([0.0] * (self.state_dim - len(state_components)))
            else:
                state_components = state_components[:self.state_dim]

        return np.array(state_components, dtype=np.float32)

    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """归一化值到[-1, 1]范围

        Args:
            value: 需要归一化的值
            min_val: 最小值
            max_val: 最大值

        Returns:
            归一化后的值
        """
        if not self.normalization:
            return value

        # 线性映射到[-1, 1]
        if max_val - min_val == 0:
            return 0.0

        normalized = 2 * (value - min_val) / (max_val - min_val) - 1
        return np.clip(normalized, -1.0, 1.0)