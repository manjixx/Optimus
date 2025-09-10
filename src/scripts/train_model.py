#!/usr/bin/env python3
"""
训练脚本 - 用于训练手机发布版本编排强化学习模型
"""

import argparse
import os
import sys
import warnings
import time

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 忽略一些不必要的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Gym has been unmaintained")

import numpy as np
import torch

from training import RLTrainer
from utils.logger import setup_logger


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
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式（更详细的日志和检查）"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存模型和结果（用于测试运行）"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        help="覆盖配置文件中的总时间步数"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="选择训练设备 (auto/cpu/cuda)"
    )

    return parser.parse_args()


def setup_directories():
    """设置必要的目录结构"""
    directories = [
        "logs",
        "models/best",
        "models/checkpoints",
        "results/training",
        "config/backups"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"确保目录存在: {directory}")


def backup_config(config_path):
    """备份配置文件"""
    import shutil
    from datetime import datetime

    if os.path.exists(config_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"config/backups/{os.path.basename(config_path)}.{timestamp}.bak"
        shutil.copy2(config_path, backup_path)
        print(f"配置文件已备份到: {backup_path}")


def setup_device(device_preference):
    """设置训练设备"""
    if device_preference == "auto":
        # 自动检测GPU
        if torch.cuda.is_available():
            device = "cuda"
            print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("未检测到GPU，使用CPU")
    else:
        device = device_preference
        if device == "cuda" and not torch.cuda.is_available():
            print("警告: 选择了CUDA但未检测到GPU，回退到CPU")
            device = "cpu"

    print(f"使用设备: {device}")
    return device


def patch_sklearn_warnings():
    """修补sklearn中的警告"""
    # 这些警告通常是由于除以零或NaN值引起的
    # 我们将在其他地方处理这些根本问题
    warnings.filterwarnings("ignore",
                            message="invalid value encountered in divide",
                            category=RuntimeWarning,
                            module="sklearn")
    warnings.filterwarnings("ignore",
                            message="divide by zero encountered",
                            category=RuntimeWarning,
                            module="sklearn")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置必要的目录
    setup_directories()

    # 备份配置文件
    backup_config(args.config)

    # 修补sklearn警告
    patch_sklearn_warnings()

    # 设置日志
    log_file = "logs/training.log"
    setup_logger(level=args.log_level, log_file=log_file)

    # 设置设备
    device = setup_device(args.device)

    # 创建训练器
    trainer = RLTrainer(config_path=args.config)

    # 如果指定了调试模式，设置更详细的配置
    if args.debug:
        print("调试模式已启用")
        # 这里可以设置一些调试特定的配置

    # 如果指定了时间步数，覆盖配置
    if args.timesteps:
        trainer.config['training']['total_timesteps'] = args.timesteps
        print(f"覆盖总时间步数为: {args.timesteps}")

    # 设置设备配置
    trainer.config['training']['device'] = device

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

        # 如果不保存，则删除模型和结果
        if args.no_save:
            print("测试运行完成，不保存模型和结果")
            # 删除可能已保存的文件
            import shutil
            if os.path.exists("models/best"):
                shutil.rmtree("models/best")
            if os.path.exists("results/training"):
                # 只删除本次运行的结果，保留其他结果
                latest_files = sorted([f for f in os.listdir("results/training")
                                       if f.startswith("training_results")],
                                      reverse=True)[:2]
                for f in latest_files:
                    os.remove(f"results/training/{f}")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 尝试保存当前进度
        try:
            if not args.no_save:
                trainer._save_results({
                    'total_timesteps': trainer.agent.model.num_timesteps if trainer.agent else 0,
                    'training_time': time.time() - trainer.start_time if trainer.start_time else 0,
                    'best_mean_reward': trainer.best_mean_reward,
                    'training_history': trainer.training_history,
                    'status': 'interrupted'
                })
                print("已保存中断时的进度")
        except:
            print("保存进度失败")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 记录错误到日志文件
        error_log = "logs/error.log"
        with open(error_log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Training Error:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "=" * 50 + "\n")

        print(f"错误详情已记录到: {error_log}")
    finally:
        # 清理资源
        trainer.close()


if __name__ == "__main__":
    main()