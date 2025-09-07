import numpy as np
from typing import Dict, Any, List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def analyze_robustness(scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """分析策略在不同场景下的稳健性

    Args:
        scenario_results: 场景结果字典

    Returns:
        稳健性分析结果字典
    """
    analysis = {
        'scenario_performance': {},
        'consistency_metrics': {},
        'sensitivity_analysis': {},
        'overall_robustness': 0.0
    }

    # 提取关键指标
    key_metrics = [
        'mean_reward', 'traffic_variance', 'worst_case_variance',
        'traffic_stability', 'reward_consistency', 'safety_margin'
    ]

    # 分析每个场景的性能
    for scenario_name, results in scenario_results.items():
        analysis['scenario_performance'][scenario_name] = {
            metric: results.get(metric, 0) for metric in key_metrics
        }

    # 计算一致性指标
    analysis['consistency_metrics'] = calculate_consistency_metrics(analysis['scenario_performance'])

    # 执行敏感性分析
    analysis['sensitivity_analysis'] = perform_sensitivity_analysis(analysis['scenario_performance'])

    # 计算总体稳健性评分
    analysis['overall_robustness'] = calculate_overall_robustness(
        analysis['consistency_metrics'],
        analysis['sensitivity_analysis']
    )

    return analysis


def calculate_consistency_metrics(scenario_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """计算一致性指标

    Args:
        scenario_performance: 场景性能字典

    Returns:
        一致性指标字典
    """
    consistency_metrics = {}

    # 收集所有场景的指标值
    metric_values = {}
    for metrics in scenario_performance.values():
        for metric, value in metrics.items():
            if metric not in metric_values:
                metric_values[metric] = []
            metric_values[metric].append(value)

    # 计算每个指标的一致性
    for metric, values in metric_values.items():
        if len(values) > 1:
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val != 0 else 0.0  # 变异系数

            consistency_metrics[metric] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv),  # 变异系数越小越一致
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'consistency_score': 1.0 / (1.0 + cv)  # 一致性评分
            }

    return consistency_metrics


def perform_sensitivity_analysis(scenario_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """执行敏感性分析

    Args:
        scenario_performance: 场景性能字典

    Returns:
        敏感性分析结果字典
    """
    sensitivity_analysis = {}

    # 假设场景有难度级别（从配置中获取）
    # 这里简化处理，假设场景名称包含难度信息
    scenario_difficulty = {
        'normal': 1.0,
        'extreme': 3.0,
        'holiday': 2.0
    }

    # 分析每个指标对场景难度的敏感性
    for metric in next(iter(scenario_performance.values())).keys():
        difficulties = []
        values = []

        for scenario_name, metrics in scenario_performance.items():
            difficulty = scenario_difficulty.get(scenario_name, 1.0)
            value = metrics.get(metric, 0)

            difficulties.append(difficulty)
            values.append(value)

        if len(difficulties) > 1:
            # 计算相关系数
            correlation = np.corrcoef(difficulties, values)[0, 1] if len(difficulties) > 1 else 0.0

            # 计算敏感性系数（回归斜率）
            if np.var(difficulties) > 0:
                sensitivity = np.cov(difficulties, values)[0, 1] / np.var(difficulties)
            else:
                sensitivity = 0.0

            sensitivity_analysis[metric] = {
                'correlation': float(correlation),
                'sensitivity': float(sensitivity),
                'sensitivity_score': 1.0 / (1.0 + abs(sensitivity))  # 敏感性评分（越低越敏感）
            }

    return sensitivity_analysis


def calculate_overall_robustness(consistency_metrics: Dict[str, Any],
                                 sensitivity_analysis: Dict[str, Any]) -> float:
    """计算总体稳健性评分

    Args:
        consistency_metrics: 一致性指标
        sensitivity_analysis: 敏感性分析结果

    Returns:
        总体稳健性评分
    """
    # 定义指标权重（可根据需求调整）
    metric_weights = {
        'mean_reward': 0.3,
        'traffic_variance': 0.2,
        'worst_case_variance': 0.3,
        'traffic_stability': 0.1,
        'reward_consistency': 0.1
    }

    total_score = 0.0
    total_weight = 0.0

    # 基于一致性计算分数
    for metric, weight in metric_weights.items():
        if metric in consistency_metrics:
            consistency_score = consistency_metrics[metric].get('consistency_score', 0.5)
            total_score += consistency_score * weight
            total_weight += weight

    # 基于敏感性调整分数
    for metric, weight in metric_weights.items():
        if metric in sensitivity_analysis:
            sensitivity_score = sensitivity_analysis[metric].get('sensitivity_score', 0.5)
            # 敏感性评分越低，对总体稳健性的负面影响越大
            total_score *= sensitivity_score

    # 归一化到0-100范围
    if total_weight > 0:
        normalized_score = (total_score / total_weight) * 100
        return max(0, min(100, normalized_score))

    return 50.0  # 默认分数


def identify_weak_scenarios(scenario_results: Dict[str, Dict[str, Any]],
                            threshold: float = 0.7) -> List[Tuple[str, float]]:
    """识别性能较弱的场景

    Args:
        scenario_results: 场景结果字典
        threshold: 相对性能阈值（低于此值视为弱场景）

    Returns:
        弱场景列表（场景名称，相对性能）
    """
    weak_scenarios = []

    # 计算平均性能作为基准
    all_rewards = [results.get('mean_reward', 0) for results in scenario_results.values()]
    avg_reward = np.mean(all_rewards) if all_rewards else 0

    if avg_reward == 0:
        return weak_scenarios

    # 识别性能低于阈值的场景
    for scenario_name, results in scenario_results.items():
        scenario_reward = results.get('mean_reward', 0)
        relative_performance = scenario_reward / avg_reward

        if relative_performance < threshold:
            weak_scenarios.append((scenario_name, relative_performance))

    # 按性能排序
    weak_scenarios.sort(key=lambda x: x[1])

    return weak_scenarios


def generate_robustness_report(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """生成稳健性报告

    Args:
        analysis: 稳健性分析结果

    Returns:
        稳健性报告字典
    """
    report = {
        'summary': {
            'overall_robustness': analysis.get('overall_robustness', 0),
            'performance_consistency': calculate_average_consistency(analysis.get('consistency_metrics', {})),
            'scenario_sensitivity': calculate_average_sensitivity(analysis.get('sensitivity_analysis', {}))
        },
        'strengths': identify_strengths(analysis),
        'weaknesses': identify_weaknesses(analysis),
        'recommendations': generate_recommendations(analysis)
    }

    return report


def calculate_average_consistency(consistency_metrics: Dict[str, Any]) -> float:
    """计算平均一致性

    Args:
        consistency_metrics: 一致性指标

    Returns:
        平均一致性评分
    """
    if not consistency_metrics:
        return 0.0

    scores = [metrics.get('consistency_score', 0.5) for metrics in consistency_metrics.values()]
    return float(np.mean(scores))


def calculate_average_sensitivity(sensitivity_analysis: Dict[str, Any]) -> float:
    """计算平均敏感性

    Args:
        sensitivity_analysis: 敏感性分析结果

    Returns:
        平均敏感性评分
    """
    if not sensitivity_analysis:
        return 0.0

    scores = [metrics.get('sensitivity_score', 0.5) for metrics in sensitivity_analysis.values()]
    return float(np.mean(scores))


def identify_strengths(analysis: Dict[str, Any]) -> List[str]:
    """识别策略的优势

    Args:
        analysis: 稳健性分析结果

    Returns:
        优势描述列表
    """
    strengths = []
    consistency_metrics = analysis.get('consistency_metrics', {})
    sensitivity_analysis = analysis.get('sensitivity_analysis', {})

    # 检查一致性高的指标
    for metric, metrics_data in consistency_metrics.items():
        if metrics_data.get('consistency_score', 0) > 0.8:
            strengths.append(f"在{metric}指标上表现一致")

    # 检查敏感性低的指标
    for metric, metrics_data in sensitivity_analysis.items():
        if metrics_data.get('sensitivity_score', 0) > 0.8:
            strengths.append(f"{metric}指标对场景变化不敏感")

    # 添加总体稳健性
    if analysis.get('overall_robustness', 0) > 70:
        strengths.append("总体稳健性良好")

    return strengths


def identify_weaknesses(analysis: Dict[str, Any]) -> List[str]:
    """识别策略的劣势

    Args:
        analysis: 稳健性分析结果

    Returns:
        劣势描述列表
    """
    weaknesses = []
    consistency_metrics = analysis.get('consistency_metrics', {})
    sensitivity_analysis = analysis.get('sensitivity_analysis', {})

    # 检查一致性低的指标
    for metric, metrics_data in consistency_metrics.items():
        if metrics_data.get('consistency_score', 0) < 0.5:
            weaknesses.append(f"在{metric}指标上表现不一致")

    # 检查敏感性高的指标
    for metric, metrics_data in sensitivity_analysis.items():
        if metrics_data.get('sensitivity_score', 0) < 0.5:
            weaknesses.append(f"{metric}指标对场景变化过于敏感")

    # 添加总体稳健性
    if analysis.get('overall_robustness', 0) < 50:
        weaknesses.append("总体稳健性不足")

    return weaknesses


def generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """生成改进建议

    Args:
        analysis: 稳健性分析结果

    Returns:
        改进建议列表
    """
    recommendations = []
    weaknesses = identify_weaknesses(analysis)

    for weakness in weaknesses:
        if "表现不一致" in weakness:
            metric = weakness.replace("在", "").replace("指标上表现不一致", "")
            recommendations.append(f"针对{metric}指标进行专门训练，提高一致性")

        if "过于敏感" in weakness:
            metric = weakness.replace("指标对场景变化过于敏感", "")
            recommendations.append(f"增加在{metric}相关场景下的训练，降低敏感性")

    if "总体稳健性不足" in weaknesses:
        recommendations.append("增加在多样化场景下的训练，提高泛化能力")
        recommendations.append("考虑使用课程学习，从简单场景逐步过渡到复杂场景")

    return recommendations