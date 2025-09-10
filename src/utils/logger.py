import logging
import sys
from typing import Optional
import os
from pathlib import Path


def setup_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """设置根日志记录器，支持控制台和文件输出

    Args:
        level: 控制台日志级别
        log_file: 日志文件路径，如果为None则不输出到文件
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 设置根日志级别为DEBUG，让处理器决定实际级别

    # 移除现有的处理器以避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建控制台处理器 - 只显示关键信息
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)  # 控制台只显示指定级别及以上的日志
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 如果提供了日志文件路径，创建文件处理器 - 记录所有信息
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有DEBUG及以上的日志
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """获取命名日志记录器

    Args:
        name: 日志记录器名称
        level: 如果提供，设置此日志记录器的级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger