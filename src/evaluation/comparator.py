import numpy as np
from typing import Dict, Any, List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compare_strategies(agent_results: Dict[str, Any],
                       baseline_results: Dict[str, Any]) -> Dict[str, Any]:
    """比较智能体与基准策略的性能

    Args:
        agent_results: 智能体结果
        baseline_results: 基准策略结果

    Returns:
        比较结果字典
    """
    comparison = {
        'agent': agent_results,
        'baselines': baseline_results,
        'comparison_metrics': {},
        'rankings': {}
    }

    # 提取关键指标进行比较
    key_metrics = [
        'mean_reward', 'traffic_variance', 'worst_case_variance',
        'traffic_stability', 'reward_consistency', 'safety_margin'
    ]

    # 计算每个指标的相对性能
    for metric in key_metrics:
        if metric in agent_results:
            agent_value = agent_results[metric]
            baseline_values = {}

            for baseline_name, results in baseline_results.items():
                if isinstance(results, dict) and metric in results:
                    baseline_values[baseline_name] = results[metric]

            comparison['comparison_metrics'][metric] = {
                'agent': agent_value,
                'baselines': baseline_values,
                'improvement': calculate_improvement(agent_value, baseline_values)
            }

    # 计算排名
    comparison['rankings'] = calculate_rankings(comparison['comparison_metrics'])

    # 计算综合评分
    comparison['overall_assessment'] = overall_assessment(comparison)

    return comparison


def calculate_improvement(agent_value: float, baseline_values: Dict[str, float]) -> Dict[str, float]:
    """计算相对于基准策略的改进

    Args:
        agent_value: 智能体指标值
        baseline_values: 基准策略指标值字典

    Returns:
        改进百分比字典
    """
    improvements = {}

    for baseline_name, baseline_value in baseline_values.items():
        if baseline_value != 0:
            improvement = ((agent_value - baseline_value) / abs(baseline_value)) * 100
            improvements[baseline_name] = improvement
        else:
            improvements[baseline_name] = float('inf') if agent_value > 0 else float('-inf')

    return improvements


def calculate_rankings(comparison_metrics: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
    """计算策略排名

    Args:
        comparison_metrics: 比较指标字典

    Returns:
        排名字典
    """
    rankings = {}

    for metric, data in comparison_metrics.items():
        # 收集所有策略的值
        values = {'Agent': data['agent']}
        values.update(data['baselines'])

        # 确定排序方向（有些指标是越小越好）
        if metric in ['traffic_variance', 'worst_case_variance', 'reward_std', 'reward_cv']:
            # 这些指标越小越好
            sorted_items = sorted(values.items(), key=lambda x: x[1])
        else:
            # 这些指标越大越好
            sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)

        # 记录排名
        rankings[metric] = [(name, value) for name, value in sorted_items]

    return rankings


def overall_assessment(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """进行总体评估

    Args:
        comparison: 比较结果字典

    Returns:
        总体评估字典
    """
    # 定义指标权重（可根据需求调整）
    metric_weights = {
        'mean_reward': 0.3,
        'traffic_variance': -0.2,  # 负权重表示越小越好
        'worst_case_variance': -0.3,
        'traffic_stability': 0.1,
        'reward_consistency': 0.1,
        'safety_margin': 0.1
    }

    # 计算加权分数
    agent_score = 0.0
    baseline_scores = {}

    for metric, weight in metric_weights.items():
        if metric in comparison['comparison_metrics']:
            metric_data = comparison['comparison_metrics'][metric]

            # 归一化指标值
            all_values = [metric_data['agent']]
            all_values.extend(metric_data['baselines'].values())

            min_val = min(all_values)
            max_val = max(all_values)

            if max_val - min_val > 0:
                # 对于负权重指标，需要反转归一化
                if weight < 0:
                    normalized_agent = 1 - (metric_data['agent'] - min_val) / (max_val - min_val)
                else:
                    normalized_agent = (metric_data['agent'] - min_val) / (max_val - min_val)

                agent_score += normalized_agent * abs(weight)

                # 计算基准策略的分数
                for baseline_name, baseline_value in metric_data['baselines'].items():
                    if baseline_name not in baseline_scores:
                        baseline_scores[baseline_name] = 0.0

                    if weight < 0:
                        normalized_baseline = 1 - (baseline_value - min_val) / (max_val - min_val)
                    else:
                        normalized_baseline = (baseline_value - min_val) / (max_val - min_val)

                    baseline_scores[baseline_name] += normalized_baseline * abs(weight)

    # 创建总体评估
    assessment = {
        'agent_score': agent_score,
        'baseline_scores': baseline_scores,
        'relative_performance': {}
    }

    # 计算相对性能
    for baseline_name, score in baseline_scores.items():
        if score > 0:
            improvement = ((agent_score - score) / score) * 100
            assessment['relative_performance'][baseline_name] = improvement

    return assessment