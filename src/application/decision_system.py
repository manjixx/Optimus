import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import yaml

from environment import MobileReleaseEnv
from models import PPOAgent
from evaluation import StrategyEvaluator
from ..utils.logger import get_logger
from ..utils.serialization import load_model, save_results
from .visualization import VisualizationEngine
from .explainer import ModelExplainer

logger = get_logger(__name__)


class DecisionSupportSystem:
    """决策支持系统，提供版本发布推荐和策略分析"""

    def __init__(self, config_path: str = "config/application.yaml"):
        """初始化决策支持系统

        Args:
            config_path: 应用配置文件路径
        """
        self.config = self._load_config(config_path)
        self.app_config = self.config['application']
        self.decision_config = self.app_config['decision_system']

        # 初始化组件
        self.env = None
        self.agent = None
        self.visualizer = None
        self.explainer = None

        # 加载模型
        self._load_model()

        # 初始化可视化引擎和解释器
        if self.app_config['visualization']['enabled']:
            self.visualizer = VisualizationEngine(self.app_config['visualization'])

        if self.app_config['explainability']['enabled']:
            self.explainer = ModelExplainer(self.app_config['explainability'])

        logger.info("Decision support system initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _load_model(self) -> None:
        """加载训练好的模型"""
        model_path = self.decision_config['model_path']

        try:
            # 创建环境
            self.env = MobileReleaseEnv(config_path="config/base.yaml")

            # 加载智能体
            self.agent = PPOAgent(self.env, self.config)
            self.agent.load(model_path)

            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_recommendation(self, current_state: Optional[Dict[str, Any]] = None,
                                scenario: Optional[str] = None) -> Dict[str, Any]:
        """生成发布推荐

        Args:
            current_state: 当前状态信息
            scenario: 场景名称

        Returns:
            推荐结果字典
        """
        if scenario is None:
            scenario = self.decision_config['default_scenario']

        logger.info(f"Generating recommendation for scenario: {scenario}")

        # 设置环境状态
        if current_state is not None:
            self._set_environment_state(current_state)

        # 设置场景
        self._set_scenario(scenario)

        # 生成推荐
        recommendations = self._generate_release_plan()

        # 评估推荐
        evaluation = self._evaluate_recommendation(recommendations)

        # 生成解释
        explanation = None
        if self.explainer:
            explanation = self.explainer.explain_decision(
                self.agent.model, self.env, recommendations
            )

        # 生成可视化
        visualizations = None
        if self.visualizer:
            visualizations = self.visualizer.generate_visualizations(
                recommendations, evaluation, explanation
            )

        # 构建结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario,
            'recommendations': recommendations,
            'evaluation': evaluation,
            'explanation': explanation,
            'visualizations': visualizations,
            'confidence': self._calculate_confidence(evaluation)
        }

        # 保存结果
        self._save_recommendation(result)

        return result

    def _set_environment_state(self, state_info: Dict[str, Any]) -> None:
        """设置环境状态

        Args:
            state_info: 状态信息字典
        """
        # 这里需要根据实际状态设计实现状态设置逻辑
        # 简化实现：重置环境并设置相关参数
        self.env.reset()

        # 设置版本信息
        if 'version_info' in state_info:
            self.env.set_version_info(state_info['version_info'])

        # 设置节假日
        if 'holidays' in state_info:
            self.env.set_holidays(state_info['holidays'])

        logger.debug("Environment state updated")

    def _set_scenario(self, scenario_name: str) -> None:
        """设置场景

        Args:
            scenario_name: 场景名称
        """
        # 加载场景配置
        scenario_config = self._load_scenario_config(scenario_name)

        # 应用场景配置到环境
        if scenario_config and hasattr(self.env, 'set_parameters'):
            self.env.set_parameters(scenario_config.get('env_params', {}))

        logger.debug(f"Scenario set to: {scenario_name}")

    def _load_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """加载场景配置

        Args:
            scenario_name: 场景名称

        Returns:
            场景配置字典
        """
        scenario_path = f"config/scenarios/{scenario_name}_scenario.yaml"

        try:
            with open(scenario_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Scenario config not found: {scenario_path}")
            return {}

    def _generate_release_plan(self) -> List[Dict[str, Any]]:
        """生成发布计划

        Returns:
            发布计划列表
        """
        plans = []
        max_recommendations = self.decision_config['max_recommendations']

        # 生成多个推荐方案
        for i in range(max_recommendations):
            # 使用不同随机种子生成多样化推荐
            if hasattr(self.env, 'seed'):
                self.env.seed(42 + i)  # 不同的种子

            # 重置环境
            state = self.env.reset()

            # 模拟完整回合
            done = False
            release_days = []
            traffic_values = []

            while not done:
                action, _ = self.agent.predict(state, deterministic=True)
                state, reward, done, info = self.env.step(action)

                if action == 1:  # 发布动作
                    release_days.append(self.env.current_day)

                if 'daily_traffic' in info[0]:
                    traffic_values.append(info[0]['daily_traffic'])

            # 计算方案指标
            plan_metrics = self._calculate_plan_metrics(release_days, traffic_values)

            # 构建推荐方案
            plan = {
                'id': i + 1,
                'release_days': release_days,
                'release_count': len(release_days),
                'metrics': plan_metrics,
                'description': self._generate_plan_description(release_days, plan_metrics)
            }

            plans.append(plan)

        # 按综合评分排序
        plans.sort(key=lambda x: x['metrics'].get('composite_score', 0), reverse=True)

        return plans

    def _calculate_plan_metrics(self, release_days: List[int],
                                traffic_values: List[float]) -> Dict[str, Any]:
        """计算发布计划指标

        Args:
            release_days: 发布日期列表
            traffic_values: 流量值列表

        Returns:
            计划指标字典
        """
        from ..evaluation.metrics import calculate_metrics

        # 计算基础指标
        metrics = calculate_metrics(traffic_values, [1] * len(release_days), 0)

        # 计算额外指标
        if release_days:
            release_intervals = np.diff(sorted(release_days))
            metrics.update({
                'mean_release_interval': float(np.mean(release_intervals)) if len(release_intervals) > 0 else 0,
                'min_release_interval': float(np.min(release_intervals)) if len(release_intervals) > 0 else 0,
                'max_release_interval': float(np.max(release_intervals)) if len(release_intervals) > 0 else 0,
                'weekend_releases': sum(1 for day in release_days if day % 7 in [5, 6]),
                'composite_score': self._calculate_composite_score(metrics)
            })

        return metrics

    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """计算综合评分

        Args:
            metrics: 指标字典

        Returns:
            综合评分
        """
        # 定义指标权重
        weights = {
            'traffic_stability': 0.4,
            'mean_reward': 0.3,
            'safety_margin': 0.2,
            'weekend_releases': -0.1  # 周末发布有惩罚
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += abs(weight)

        if total_weight > 0:
            # 归一化到0-100范围
            normalized_score = (score / total_weight) * 100
            return max(0, min(100, normalized_score))

        return 0.0

    def _generate_plan_description(self, release_days: List[int],
                                   metrics: Dict[str, Any]) -> str:
        """生成计划描述

        Args:
            release_days: 发布日期列表
            metrics: 计划指标

        Returns:
            计划描述
        """
        if not release_days:
            return "无发布计划"

        # 计算日期信息（假设从当月1日开始）
        start_date = datetime.now().replace(day=1)
        release_dates = [start_date + timedelta(days=day) for day in release_days]

        # 构建描述
        desc = f"发布{len(release_days)}次: "
        desc += ", ".join([date.strftime("%m月%d日") for date in release_dates[:3]])
        if len(release_days) > 3:
            desc += f" 等{len(release_days)}天"

        desc += f" | 综合评分: {metrics.get('composite_score', 0):.1f}/100"
        desc += f" | 流量稳定性: {metrics.get('traffic_stability', 0):.3f}"

        if metrics.get('weekend_releases', 0) > 0:
            desc += f" | 注意: 包含{metrics['weekend_releases']}次周末发布"

        return desc

    def _evaluate_recommendation(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估推荐方案

        Args:
            recommendations: 推荐方案列表

        Returns:
            评估结果字典
        """
        if not recommendations:
            return {}

        # 使用最佳方案进行评估
        best_plan = recommendations[0]

        # 在多场景下评估稳健性
        evaluator = StrategyEvaluator()
        scenario_results = {}

        for scenario in self.app_config['decision_system'].get('eval_scenarios', ['normal', 'extreme']):
            try:
                # 设置场景
                self._set_scenario(scenario)

                # 评估方案
                result = self.evaluate_plan(best_plan['release_days'])
                scenario_results[scenario] = result
            except Exception as e:
                logger.error(f"Failed to evaluate scenario {scenario}: {e}")
                scenario_results[scenario] = {'error': str(e)}

        # 计算稳健性指标
        from ..evaluation.robustness_analyzer import analyze_robustness
        robustness = analyze_robustness(scenario_results)

        return {
            'best_plan_metrics': best_plan['metrics'],
            'scenario_results': scenario_results,
            'robustness_analysis': robustness,
            'overall_robustness': robustness.get('overall_robustness', 0)
        }

    def evaluate_plan(self, release_days: List[int]) -> Dict[str, Any]:
        """评估特定发布计划

        Args:
            release_days: 发布日期列表

        Returns:
            评估结果字典
        """
        # 重置环境
        state = self.env.reset()
        done = False
        traffic_values = []
        actual_releases = []

        # 模拟执行计划
        while not done:
            # 决定是否发布（根据计划）
            should_release = self.env.current_day in release_days
            action = 1 if should_release else 0

            # 执行动作
            state, reward, done, info = self.env.step([action])

            # 记录信息
            if 'daily_traffic' in info[0]:
                traffic_values.append(info[0]['daily_traffic'])

            actual_releases.append(action)

        # 计算指标
        from ..evaluation.metrics import calculate_metrics
        metrics = calculate_metrics(traffic_values, actual_releases, 0)

        # 计算计划执行情况
        planned_count = len(release_days)
        actual_count = sum(actual_releases)
        execution_accuracy = actual_count / planned_count if planned_count > 0 else 1.0

        metrics.update({
            'planned_releases': planned_count,
            'actual_releases': actual_count,
            'execution_accuracy': execution_accuracy,
            'missed_releases': planned_count - actual_count
        })

        return metrics

    def _calculate_confidence(self, evaluation: Dict[str, Any]) -> float:
        """计算推荐置信度

        Args:
            evaluation: 评估结果

        Returns:
            置信度分数 (0-1)
        """
        robustness = evaluation.get('overall_robustness', 0) / 100  # 转换为0-1范围
        best_plan_score = evaluation.get('best_plan_metrics', {}).get('composite_score', 0) / 100

        # 综合置信度
        confidence = (robustness * 0.6) + (best_plan_score * 0.4)

        # 应用阈值
        threshold = self.decision_config['confidence_threshold']
        if confidence < threshold:
            logger.warning(f"Low confidence: {confidence:.3f} < {threshold}")

        return confidence

    def _save_recommendation(self, recommendation: Dict[str, Any]) -> None:
        """保存推荐结果

        Args:
            recommendation: 推荐结果字典
        """
        # 创建输出目录
        output_dir = "results/recommendations"
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = recommendation.get('scenario', 'unknown')
        filename = f"recommendation_{scenario}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # 保存结果
        save_results(recommendation, filepath)
        logger.info(f"Recommendation saved to {filepath}")

    def generate_report(self, recommendation: Dict[str, Any],
                        format: str = "pdf") -> str:
        """生成详细报告

        Args:
            recommendation: 推荐结果字典
            format: 报告格式

        Returns:
            报告文件路径
        """
        if not self.visualizer:
            logger.warning("Visualization engine not available, cannot generate report")
            return None

        # 生成报告
        report_path = self.visualizer.generate_report(recommendation, format)

        logger.info(f"Report generated: {report_path}")
        return report_path

    def close(self) -> None:
        """关闭系统，释放资源"""
        if self.env:
            self.env.close()
        logger.info("Decision support system closed")