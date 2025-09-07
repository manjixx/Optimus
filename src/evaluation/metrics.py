import numpy as np
from typing import Dict, Any, List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(traffic_data: List[float], release_decisions: List[int],
                      total_reward: float) -> Dict[str, Any]:
    """计算评估指标

    Args:
        traffic_data: 流量数据列表
        release_decisions: 发布决策列表
        total_reward: 总奖励

    Returns:
        指标字典
    """
    if not traffic_data:
        return {}

    # 基础流量指标
    traffic_array = np.array(traffic_data)
    traffic_mean = float(np.mean(traffic_array))
    traffic_std = float(np.std(traffic_array))
    traffic_variance = float(np.var(traffic_array))
    traffic_min = float(np.min(traffic_array))
    traffic_max = float(np.max(traffic_array))

    # 流量平稳性指标
    traffic_changes = np.diff(traffic_array)
    max_change = float(np.max(np.abs(traffic_changes))) if len(traffic_changes) > 0 else 0.0
    mean_absolute_change = float(np.mean(np.abs(traffic_changes))) if len(traffic_changes) > 0 else 0.0

    # 发布决策指标
    release_array = np.array(release_decisions)
    release_count = int(np.sum(release_array)) if len(release_array) > 0 else 0
    release_days = np.where(release_array == 1)[0]

    # 发布间隔指标
    release_intervals = np.diff(release_days) if len(release_days) > 1 else []
    mean_release_interval = float(np.mean(release_intervals)) if len(release_intervals) > 0 else 0.0
    min_release_interval = float(np.min(release_intervals)) if len(release_intervals) > 0 else 0.0
    max_release_interval = float(np.max(release_intervals)) if len(release_intervals) > 0 else 0.0

    # 构建指标字典
    metrics = {
        'total_reward': float(total_reward),
        'traffic_mean': traffic_mean,
        'traffic_std': traffic_std,
        'traffic_variance': traffic_variance,
        'traffic_min': traffic_min,
        'traffic_max': traffic_max,
        'traffic_range': traffic_max - traffic_min,
        'max_daily_change': max_change,
        'mean_absolute_change': mean_absolute_change,
        'release_count': release_count,
        'release_density': release_count / len(release_array) if len(release_array) > 0 else 0.0,
        'mean_release_interval': mean_release_interval,
        'min_release_interval': min_release_interval,
        'max_release_interval': max_release_interval,
        'traffic_stability': 1.0 / (1.0 + traffic_variance)  # 稳定性指标（方差越小越稳定）
    }

    return metrics


def calculate_robustness_metrics(metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算稳健性指标

    Args:
        metrics_history: 历史指标列表

    Returns:
        稳健性指标字典
    """
    if not metrics_history:
        return {}

    # 提取关键指标
    rewards = [m.get('total_reward', 0) for m in metrics_history]
    variances = [m.get('traffic_variance', 0) for m in metrics_history]
    stabilities = [m.get('traffic_stability', 0) for m in metrics_history]

    # 计算稳健性指标
    reward_std = float(np.std(rewards)) if rewards else 0.0
    reward_cv = reward_std / np.mean(rewards) if np.mean(rewards) != 0 else 0.0  # 变异系数

    worst_case_variance = float(np.max(variances)) if variances else 0.0
    worst_case_reward = float(np.min(rewards)) if rewards else 0.0

    # 计算分位数指标
    if rewards:
        reward_5th = float(np.percentile(rewards, 5))  # 5%分位数（近似最坏情况）
        reward_95th = float(np.percentile(rewards, 95))  # 95%分位数（近似最好情况）
        reward_iqr = reward_95th - reward_5th  # 四分位距
    else:
        reward_5th = reward_95th = reward_iqr = 0.0

    robustness_metrics = {
        'reward_std': reward_std,
        'reward_cv': reward_cv,  # 变异系数越小越稳健
        'worst_case_variance': worst_case_variance,
        'worst_case_reward': worst_case_reward,
        'reward_5th_percentile': reward_5th,
        'reward_95th_percentile': reward_95th,
        'reward_iqr': reward_iqr,
        'reward_consistency': 1.0 / (1.0 + reward_cv),  # 一致性指标
        'safety_margin': reward_5th / np.mean(rewards) if np.mean(rewards) != 0 else 0.0  # 安全边际
    }

    return robustness_metrics


def calculate_scenario_weights(scenario_results: Dict[str, Dict[str, Any]],
                               config_weights: Dict[str, float]) -> Dict[str, float]:
    """计算场景权重

    Args:
        scenario_results: 场景结果字典
        config_weights: 配置中的权重

    Returns:
        场景权重字典
    """
    # 如果配置中已指定权重，则使用配置权重
    if config_weights:
        return config_weights

    # 否则基于性能自动计算权重
    weights = {}
    total_performance = 0.0

    for scenario_name, results in scenario_results.items():
        # 使用平均奖励作为性能指标
        performance = results.get('mean_reward', 0)
        weights[scenario_name] = performance
        total_performance += performance

    # 归一化权重
    if total_performance > 0:
        for scenario_name in weights:
            weights[scenario_name] /= total_performance

    return weights


def normalize_metrics(metrics: Dict[str, Any],
                      min_values: Dict[str, Any],
                      max_values: Dict[str, Any]) -> Dict[str, Any]:
    """归一化指标到[0, 1]范围

    Args:
        metrics: 原始指标字典
        min_values: 最小值字典
        max_values: 最大值字典

    Returns:
        归一化后的指标字典
    """
    normalized = {}

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            min_val = min_values.get(key, 0)
            max_val = max_values.get(key, 1)

            if max_val - min_val == 0:
                normalized[key] = 0.0
            else:
                normalized[key] = (value - min_val) / (max_val - min_val)
        else:
            normalized[key] = value

    return normalized


def calculate_composite_score(metrics: Dict[str, Any],
                              weights: Dict[str, float]) -> float:
    """计算综合评分

    Args:
        metrics: 指标字典
        weights: 权重字典

    Returns:
        综合评分
    """
    score = 0.0
    total_weight = 0.0

    for metric, weight in weights.items():
        if metric in metrics and isinstance(metrics[metric], (int, float)):
            score += metrics[metric] * weight
            total_weight += weight

    if total_weight > 0:
        return score / total_weight

    return 0.0