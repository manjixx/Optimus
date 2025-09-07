#!/usr/bin/env python3
"""
训练脚本 - 用于训练手机发布版本编排强化学习模型
"""

import argparse
import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training import RLTrainer
from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练手机发布版本编排强化学习模型")

    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="训练配置文件路径"
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="从检查点恢复训练"
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

    # 创建训练器
    trainer = RLTrainer(config_path=args.config)

    try:
        # 设置训练环境
        trainer.setup()

        # 恢复训练（如果指定了检查点）
        if args.resume:
            # 这里需要实现恢复训练的逻辑
            print(f"恢复训练功能尚未实现: {args.resume}")

        # 执行训练
        results = trainer.train()

        # 输出训练结果
        print(f"\n训练完成!")
        print(f"总步数: {results['total_timesteps']}")
        print(f"训练时间: {results['training_time']:.1f} 秒")
        print(f"最佳平均奖励: {results['best_mean_reward']:.2f}")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        trainer.close()


if __name__ == "__main__":
    main()