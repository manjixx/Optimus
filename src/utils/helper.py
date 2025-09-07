import numpy as np
from typing import Dict, Any, List, Union
from .logger import get_logger

logger = get_logger(__name__)


def setup_logger(level: str = "INFO") -> None:
    """设置日志级别

    Args:
        level: 日志级别
    """
    import logging
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info(f"Logger setup with level: {level}")


def normalize_data(data: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    """归一化数据到[0, 1]范围

    Args:
        data: 输入数据
        min_val: 最小值（如果为None，则使用数据的最小值）
        max_val: 最大值（如果为None，则使用数据的最大值）

    Returns:
        归一化后的数据
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)

    if max_val - min_val == 0:
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


def standardize_data(data: np.ndarray, mean: float = None, std: float = None) -> np.ndarray:
    """标准化数据（均值为0，标准差为1）

    Args:
        data: 输入数据
        mean: 均值（如果为None，则使用数据的均值）
        std: 标准差（如果为None，则使用数据的标准差）

    Returns:
        标准化后的数据
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)

    if std == 0:
        return np.zeros_like(data)

    return (data - mean) / std


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """计算置信区间

    Args:
        data: 输入数据
        confidence: 置信水平

    Returns:
        tuple: (均值, 下限, 上限)
    """
    n = len(data)
    if n == 0:
        return 0, 0, 0

    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(n)

    from scipy import stats
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, mean - h, mean + h


def softmax(x: np.ndarray) -> np.ndarray:
    """计算softmax

    Args:
        x: 输入向量

    Returns:
        softmax结果
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def smooth_data(data: List[float], alpha: float = 0.3) -> List[float]:
    """使用指数平滑平滑数据

    Args:
        data: 输入数据列表
        alpha: 平滑系数（0-1）

    Returns:
        平滑后的数据
    """
    if not data:
        return []

    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i - 1])

    return smoothed


def format_timedelta(seconds: float) -> str:
    """格式化时间差

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"