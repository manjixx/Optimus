#!/usr/bin/env python3
"""
评估脚本 - 用于评估训练好的手机发布版本编排模型
"""

import argparse
import os
import sys
import json

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import StrategyEvaluator
from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估手机发布版本编排强化学习模型")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型文件路径"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/eval.yaml",
        help="评估配置文件路径"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["normal", "extreme", "holiday", "all"],
        help="评估场景"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="与基准策略进行比较"
    )

    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="评估回合数"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="结果输出文件路径"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别"
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志
    setup_logger(level=args.log_level)

    # 创建评估器
    evaluator = StrategyEvaluator(config_path=args.config)

    try:
        results = {}

        # 执行评估
        if args.scenario == "all":
            # 评估所有场景
            results = evaluator.evaluate_all_scenarios(args.model, args.n_episodes)
            print(f"\n所有场景评估完成!")
            print(f"总体稳健性评分: {results.get('overall_score', 0):.2f}/100")

            # 显示各场景性能
            print("\n各场景性能:")
            for scenario, scenario_results in results.get('scenario_results', {}).items():
                print(f"  {scenario}: 平均奖励 = {scenario_results.get('mean_reward', 0):.2f}")

        else:
            # 评估特定场景
            if args.compare:
                # 与基准策略比较
                results = evaluator.compare_with_baselines(args.model, args.scenario, args.n_episodes)
                print(f"\n场景 '{args.scenario}' 的比较评估完成!")

                # 显示比较结果
                agent_reward = results['agent'].get('mean_reward', 0)
                print(f"智能体平均奖励: {agent_reward:.2f}")

                for baseline, baseline_results in results['baselines'].items():
                    if isinstance(baseline_results, dict):
                        baseline_reward = baseline_results.get('mean_reward', 0)
                        improvement = ((
                                                   agent_reward - baseline_reward) / baseline_reward * 100) if baseline_reward != 0 else float(
                            'inf')
                        print(f"  {baseline}: 平均奖励 = {baseline_reward:.2f} (改进: {improvement:+.1f}%)")

            else:
                # 仅评估智能体
                results = evaluator.evaluate_agent(args.model, args.scenario, args.n_episodes)
                print(f"\n场景 '{args.scenario}' 评估完成!")
                print(f"平均奖励: {results.get('mean_reward', 0):.2f}")
                print(f"流量方差: {results.get('traffic_variance', 0):.2f}")
                print(f"最坏情况方差: {results.get('worst_case_variance', 0):.2f}")

        # 保存结果（如果指定了输出文件）
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"结果已保存到: {args.output}")

        # 显示评估摘要
        summary = evaluator.get_summary()
        print(f"\n评估摘要:")
        print(f"  总评估次数: {summary['total_evaluations']}")
        print(f"  评估的场景: {', '.join(summary['scenarios_evaluated'])}")

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n评估完成!")


if __name__ == "__main__":
    main()