import os
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from ..environment import MobileReleaseEnv
from ..models import BaseAgent
from ..utils.logger import get_logger
from ..utils.serialization import save_results, load_model
from .metrics import calculate_metrics, calculate_robustness_metrics
from .comparator import compare_strategies
from .robustness_analyzer import analyze_robustness

logger = get_logger(__name__)


class StrategyEvaluator:
    """策略评估器，用于评估不同策略在各种场景下的性能"""

    def __init__(self, config_path: str = "config/eval.yaml"):
        """初始化策略评估器

        Args:
            config_path: 评估配置文件路径
        """
        self.config = self._load_config(config_path)
        self.eval_config = self.config['evaluation']

        # 初始化状态
        self.results = {}
        self.scenario_results = {}
        self.baseline_results = {}

        logger.info("Strategy evaluator initialized")

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

    def _create_eval_env(self, scenario_config: Optional[Dict[str, Any]] = None) -> Any:
        """创建评估环境

        Args:
            scenario_config: 场景配置

        Returns:
            评估环境实例
        """
        # 基础环境配置
        env_config = {
            'config_path': "config/base.yaml",
            'eval_mode': True
        }

        # 应用场景特定配置
        if scenario_config:
            env_config.update(scenario_config.get('env_params', {}))

        # 创建环境函数
        def make_env():
            env = MobileReleaseEnv(**env_config)
            return Monitor(env)

        # 创建向量化环境（单环境用于评估）
        return DummyVecEnv([make_env])

    def evaluate_policy(self, policy, env, n_episodes: int = 10,
                        deterministic: bool = True) -> Dict[str, Any]:
        """评估策略在给定环境下的性能

        Args:
            policy: 要评估的策略
            env: 评估环境
            n_episodes: 评估回合数
            deterministic: 是否使用确定性策略

        Returns:
            评估结果字典
        """
        logger.info(f"Evaluating policy for {n_episodes} episodes")

        # 存储每回合的结果
        episode_rewards = []
        episode_lengths = []
        traffic_histories = []
        release_calendars = []
        metrics_history = []

        for episode in range(n_episodes):
            logger.debug(f"Starting episode {episode + 1}/{n_episodes}")

            obs = env.reset()
            done = [False]
            episode_reward = 0
            episode_length = 0
            traffic_values = []
            release_decisions = []

            while not done[0]:
                # 获取动作
                if hasattr(policy, 'predict'):
                    action, _ = policy.predict(obs, deterministic=deterministic)
                else:
                    # 对于非RL策略，使用其决策函数
                    action = policy.decide(obs)

                # 执行动作
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1

                # 记录信息
                if 'daily_traffic' in info[0]:
                    traffic_values.append(info[0]['daily_traffic'])

                if 'action' in info[0]:
                    release_decisions.append(info[0]['action'])

            # 记录回合结果
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            traffic_histories.append(traffic_values)
            release_calendars.append(release_decisions)

            # 计算回合指标
            episode_metrics = calculate_metrics(
                traffic_values,
                release_decisions,
                episode_reward
            )
            metrics_history.append(episode_metrics)

        # 计算总体指标
        overall_metrics = calculate_metrics(
            [item for sublist in traffic_histories for item in sublist],
            [item for sublist in release_calendars for item in sublist],
            np.sum(episode_rewards)
        )

        # 添加统计信息
        overall_metrics.update({
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'n_episodes': n_episodes
        })

        # 计算稳健性指标
        robustness_metrics = calculate_robustness_metrics(metrics_history)
        overall_metrics.update(robustness_metrics)

        logger.info(f"Evaluation completed. Mean reward: {overall_metrics['mean_reward']:.2f}")

        return overall_metrics

    def evaluate_agent(self, agent_path: str, scenario_name: str = "normal",
                       n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """评估智能体在特定场景下的性能

        Args:
            agent_path: 智能体路径
            scenario_name: 场景名称
            n_episodes: 评估回合数

        Returns:
            评估结果字典
        """
        if n_episodes is None:
            n_episodes = self.eval_config['n_episodes']

        # 获取场景配置
        scenario_config = self._get_scenario_config(scenario_name)

        # 创建评估环境
        env = self._create_eval_env(scenario_config)

        # 加载智能体
        try:
            agent = load_model(agent_path)
            logger.info(f"Loaded agent from {agent_path}")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            env.close()
            raise

        # 评估智能体
        results = self.evaluate_policy(
            agent, env, n_episodes, self.eval_config['deterministic']
        )

        # 添加场景信息
        results['scenario'] = scenario_name
        results['agent'] = os.path.basename(agent_path)

        # 保存结果
        self._save_scenario_results(scenario_name, results)

        # 清理资源
        env.close()

        return results

    def evaluate_baseline(self, baseline_name: str, scenario_name: str = "normal",
                          n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """评估基准策略在特定场景下的性能

        Args:
            baseline_name: 基准策略名称
            scenario_name: 场景名称
            n_episodes: 评估回合数

        Returns:
            评估结果字典
        """
        if n_episodes is None:
            n_episodes = self.eval_config['n_episodes']

        # 获取场景配置
        scenario_config = self._get_scenario_config(scenario_name)

        # 创建评估环境
        env = self._create_eval_env(scenario_config)

        # 创建基准策略
        baseline = self._create_baseline_policy(baseline_name)

        # 评估基准策略
        results = self.evaluate_policy(
            baseline, env, n_episodes, True  # 基准策略通常是确定性的
        )

        # 添加场景和策略信息
        results['scenario'] = scenario_name
        results['baseline'] = baseline_name

        # 保存结果
        self._save_baseline_results(baseline_name, scenario_name, results)

        # 清理资源
        env.close()

        return results

    def evaluate_all_scenarios(self, agent_path: str,
                               n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """评估智能体在所有场景下的性能

        Args:
            agent_path: 智能体路径
            n_episodes: 每个场景的评估回合数

        Returns:
            综合评估结果字典
        """
        if n_episodes is None:
            n_episodes = self.eval_config['n_episodes']

        scenario_results = {}
        weighted_metrics = {}

        # 评估每个场景
        for scenario in self.eval_config['scenarios']:
            scenario_name = scenario['name']
            scenario_weight = scenario.get('weight', 1.0)

            logger.info(f"Evaluating scenario: {scenario_name} (weight: {scenario_weight})")

            # 评估智能体在该场景下的性能
            results = self.evaluate_agent(agent_path, scenario_name, n_episodes)
            scenario_results[scenario_name] = results

            # 计算加权指标
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    if metric not in weighted_metrics:
                        weighted_metrics[metric] = 0.0
                    weighted_metrics[metric] += value * scenario_weight

        # 计算综合稳健性指标
        robustness_analysis = analyze_robustness(scenario_results)

        # 组合结果
        combined_results = {
            'scenario_results': scenario_results,
            'weighted_metrics': weighted_metrics,
            'robustness_analysis': robustness_analysis,
            'overall_score': self._calculate_overall_score(weighted_metrics)
        }

        # 保存结果
        self.results = combined_results
        self._save_results(combined_results, "all_scenarios")

        return combined_results

    def compare_with_baselines(self, agent_path: str, scenario_name: str = "normal",
                               n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """将智能体与所有基准策略进行比较

        Args:
            agent_path: 智能体路径
            scenario_name: 场景名称
            n_episodes: 评估回合数

        Returns:
            比较结果字典
        """
        if n_episodes is None:
            n_episodes = self.eval_config['n_episodes']

        # 评估智能体
        agent_results = self.evaluate_agent(agent_path, scenario_name, n_episodes)

        # 评估所有基准策略
        baseline_results = {}
        for baseline in self.eval_config['baselines']:
            baseline_name = baseline['name']
            logger.info(f"Evaluating baseline: {baseline_name}")

            try:
                results = self.evaluate_baseline(baseline_name, scenario_name, n_episodes)
                baseline_results[baseline_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate baseline {baseline_name}: {e}")
                baseline_results[baseline_name] = {'error': str(e)}

        # 比较策略
        comparison_results = compare_strategies(agent_results, baseline_results)

        # 保存结果
        self._save_comparison_results(comparison_results, scenario_name)

        return comparison_results

    def _get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """获取场景配置

        Args:
            scenario_name: 场景名称

        Returns:
            场景配置字典
        """
        for scenario in self.eval_config['scenarios']:
            if scenario['name'] == scenario_name:
                # 加载场景配置文件
                config_path = scenario['config']
                try:
                    with open(config_path, 'r') as f:
                        return yaml.safe_load(f)
                except FileNotFoundError:
                    logger.warning(f"Scenario config not found: {config_path}")
                    return {}

        logger.warning(f"Scenario {scenario_name} not found in config")
        return {}

    def _create_baseline_policy(self, baseline_name: str) -> Any:
        """创建基准策略

        Args:
            baseline_name: 基准策略名称

        Returns:
            基准策略实例
        """
        # 查找基准配置
        baseline_config = None
        for baseline in self.eval_config['baselines']:
            if baseline['name'] == baseline_name:
                baseline_config = baseline
                break

        if baseline_config is None:
            raise ValueError(f"Baseline {baseline_name} not found in config")

        # 根据类型创建策略
        policy_type = baseline_config['type']

        if policy_type == "rule_based":
            from .baselines import RuleBasedPolicy
            return RuleBasedPolicy(baseline_config)

        elif policy_type == "optimization":
            from .baselines import OptimizationPolicy
            return OptimizationPolicy(baseline_config)

        elif policy_type == "random":
            from .baselines import RandomPolicy
            return RandomPolicy(baseline_config)

        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    def _calculate_overall_score(self, weighted_metrics: Dict[str, float]) -> float:
        """计算总体评分

        Args:
            weighted_metrics: 加权指标字典

        Returns:
            总体评分
        """
        # 定义指标权重（可根据需求调整）
        metric_weights = {
            'mean_reward': 0.3,
            'mean_traffic_variance': -0.2,  # 方差越小越好，所以用负权重
            'worst_case_variance': -0.3,  # 最坏情况方差越小越好
            'reward_consistency': 0.1,  # 奖励一致性
            'safety_margin': 0.1  # 安全边际
        }

        # 计算加权总分
        total_score = 0.0
        total_weight = 0.0

        for metric, weight in metric_weights.items():
            if metric in weighted_metrics:
                total_score += weighted_metrics[metric] * weight
                total_weight += abs(weight)

        # 归一化到0-100范围
        if total_weight > 0:
            normalized_score = (total_score / total_weight) * 100
            return max(0, min(100, normalized_score))

        return 0.0

    def _save_scenario_results(self, scenario_name: str, results: Dict[str, Any]) -> None:
        """保存场景结果

        Args:
            scenario_name: 场景名称
            results: 结果字典
        """
        if scenario_name not in self.scenario_results:
            self.scenario_results[scenario_name] = []

        self.scenario_results[scenario_name].append(results)

    def _save_baseline_results(self, baseline_name: str, scenario_name: str,
                               results: Dict[str, Any]) -> None:
        """保存基准策略结果

        Args:
            baseline_name: 基准策略名称
            scenario_name: 场景名称
            results: 结果字典
        """
        if baseline_name not in self.baseline_results:
            self.baseline_results[baseline_name] = {}

        if scenario_name not in self.baseline_results[baseline_name]:
            self.baseline_results[baseline_name][scenario_name] = []

        self.baseline_results[baseline_name][scenario_name].append(results)

    def _save_results(self, results: Dict[str, Any], prefix: str = "") -> None:
        """保存评估结果

        Args:
            results: 结果字典
            prefix: 文件名前缀
        """
        # 创建输出目录
        output_dir = self.eval_config['output']['save_path']
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_results_{timestamp}.{self.eval_config['output']['format']}"
        filepath = os.path.join(output_dir, filename)

        # 保存结果
        save_results(results, filepath)
        logger.info(f"Results saved to {filepath}")

    def _save_comparison_results(self, results: Dict[str, Any], scenario_name: str) -> None:
        """保存比较结果

        Args:
            results: 比较结果字典
            scenario_name: 场景名称
        """
        # 创建输出目录
        output_dir = os.path.join(self.eval_config['output']['save_path'], "comparisons")
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{scenario_name}_{timestamp}.{self.eval_config['output']['format']}"
        filepath = os.path.join(output_dir, filename)

        # 保存结果
        save_results(results, filepath)
        logger.info(f"Comparison results saved to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要

        Returns:
            评估摘要字典
        """
        summary = {
            'total_evaluations': len(self.scenario_results) + len(self.baseline_results),
            'scenarios_evaluated': list(self.scenario_results.keys()),
            'baselines_evaluated': list(self.baseline_results.keys()),
            'overall_results': self.results
        }

        return summary