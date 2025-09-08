import pickle
import json
import yaml
import torch
from typing import Any, Dict
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from utils.logger import get_logger

logger = get_logger(__name__)


def save_model(model: Any, path: str) -> None:
    """保存模型

    Args:
        model: 要保存的模型
        path: 保存路径
    """
    # 根据模型类型选择保存方法
    if hasattr(model, 'save'):
        # Stable Baselines3 模型
        model.save(path)
    elif isinstance(model, torch.nn.Module):
        # PyTorch 模型
        torch.save(model.state_dict(), path)
    else:
        # 通用序列化
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    logger.info(f"Model saved to {path}")


def load_model(path: str) -> Any:
    """加载模型

    Args:
        path: 模型路径

    Returns:
        加载的模型
    """
    # 尝试不同的加载方法
    try:
        # 尝试加载 Stable Baselines3 模型
        from stable_baselines3 import PPO, A2C, DQN

        # 检查文件扩展名
        if path.endswith('.zip'):
            # 使用SB3的加载方法
            model = None
            for model_class in [PPO, A2C, DQN]:
                try:
                    model = model_class.load(path)
                    logger.info(f"Loaded {model_class.__name__} model from {path}")
                    break
                except:
                    continue

            if model is not None:
                return model
    except ImportError:
        logger.warning("Stable Baselines3 not available for model loading")

    try:
        # 尝试加载 PyTorch 模型
        if path.endswith('.pth') or path.endswith('.pt'):
            # 返回状态字典，让调用者决定如何加载
            return torch.load(path)
    except:
        logger.warning("Could not load as PyTorch model")

    try:
        # 尝试通用序列化
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {path} using pickle")
        return model
    except:
        logger.error(f"Failed to load model from {path}")
        raise


def save_config(config: Dict[str, Any], path: str) -> None:
    """保存配置

    Args:
        config: 配置字典
        path: 保存路径
    """
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Config saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """加载配置

    Args:
        path: 配置路径

    Returns:
        配置字典
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {path}")
    return config


def save_results(results: Dict[str, Any], path: str) -> None:
    """保存结果

    Args:
        results: 结果字典
        path: 保存路径
    """
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {path}")


def load_results(path: str) -> Dict[str, Any]:
    """加载结果

    Args:
        path: 结果路径

    Returns:
        结果字典
    """
    with open(path, 'r') as f:
        results = json.load(f)

    logger.info(f"Results loaded from {path}")
    return results