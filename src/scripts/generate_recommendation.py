#!/usr/bin/env python3
"""
推荐生成脚本 - 用于生成版本发布推荐
"""

import argparse
import json
import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.application import DecisionSupportSystem
from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成版本发布推荐")

    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "extreme", "holiday"],
        help="评估场景"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/application.yaml",
        help="应用配置文件路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="结果输出文件路径"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="生成详细报告"
    )

    parser.add_argument(
        "--report-format",
        type=str,
        default="pdf",
        choices=["pdf", "html", "markdown"],
        help="报告格式"
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

    # 创建决策支持系统
    dss = DecisionSupportSystem(config_path=args.config)

    try:
        # 生成推荐
        print(f"生成 {args.scenario} 场景的推荐...")
        recommendation = dss.generate_recommendation(scenario=args.scenario)

        # 显示推荐结果
        print(f"\n推荐结果:")
        print(f"场景: {recommendation['scenario']}")
        print(f"置信度: {recommendation['confidence']:.3f}")
        print(f"生成时间: {recommendation['timestamp']}")

        # 显示推荐方案
        print(f"\n推荐方案:")
        for i, plan in enumerate(recommendation['recommendations'][:3]):  # 只显示前3个
            print(f"{i + 1}. {plan['description']}")

        # 显示稳健性分析
        robustness = recommendation['evaluation'].get('overall_robustness', 0)
        print(f"\n稳健性分析: {robustness:.1f}/100")

        # 保存结果（如果指定了输出文件）
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(recommendation, f, indent=4)
            print(f"结果已保存到: {args.output}")

        # 生成报告（如果请求）
        if args.report:
            report_path = dss.generate_report(recommendation, args.report_format)
            if report_path:
                print(f"报告已生成: {report_path}")

    except Exception as e:
        print(f"推荐生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        dss.close()

    print("\n推荐生成完成!")


if __name__ == "__main__":
    main()