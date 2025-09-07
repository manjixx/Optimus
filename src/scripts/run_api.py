#!/usr/bin/env python3
"""
API启动脚本 - 用于启动决策支持API服务
"""

import argparse
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.application import APIHandler
from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动决策支持API服务")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API服务主机"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API服务端口"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/application.yaml",
        help="应用配置文件路径"
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

    # 创建API处理器
    api_handler = APIHandler(config_path=args.config)

    try:
        # 运行API服务
        print(f"启动API服务在 {args.host}:{args.port}")
        print("按 Ctrl+C 停止服务")

        # 修改配置（如果命令行参数与配置文件不同）
        if args.host != api_handler.api_config['host'] or args.port != api_handler.api_config['port']:
            api_handler.api_config['host'] = args.host
            api_handler.api_config['port'] = args.port

        api_handler.run()

    except KeyboardInterrupt:
        print("\nAPI服务被用户中断")
    except Exception as e:
        print(f"API服务启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        api_handler.close()

    print("API服务已停止")


if __name__ == "__main__":
    main()