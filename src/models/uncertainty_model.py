import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class UncertaintyModel(nn.Module):
    """不确定性模型，用于估计预测的不确定性"""

    def __init__(self, config: Dict[str, Any], input_dim: int, output_dim: int):
        """初始化不确定性模型

        Args:
            config: 配置字典
            input_dim: 输入维度
            output_dim: 输出维度
        """
        super(UncertaintyModel, self).__init__()

        self.config = config
        self.uncertainty_config = config['models']['uncertainty']
        self.type = self.uncertainty_config['type']

        # 根据类型创建不同的不确定性模型
        if self.type == "ensemble":
            self.models = self._create_ensemble(input_dim, output_dim)
        elif self.type == "bayesian":
            self.model = self._create_bayesian_network(input_dim, output_dim)
        elif self.type == "dropout":
            self.model = self._create_dropout_network(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown uncertainty type: {self.type}")

        logger.info(f"Created {self.type} uncertainty model")

    def _create_ensemble(self, input_dim: int, output_dim: int) -> nn.ModuleList:
        """创建集成模型

        Args:
            input_dim: 输入维度
            output_dim: 输出维度

        Returns:
            集成模型列表
        """
        num_models = self.uncertainty_config['num_models']
        models = nn.ModuleList()

        for i in range(num_models):
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
            models.append(model)

        return models

    def _create_bayesian_network(self, input_dim: int, output_dim: int) -> nn.Module:
        """创建贝叶斯神经网络

        Args:
            input_dim: 输入维度
            output_dim: 输出维度

        Returns:
            贝叶斯神经网络
        """
        # 简化实现：使用MC Dropout作为贝叶斯近似
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.uncertainty_config['dropout_rate']),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.uncertainty_config['dropout_rate']),
            nn.Linear(32, output_dim)
        )

    def _create_dropout_network(self, input_dim: int, output_dim: int) -> nn.Module:
        """创建Dropout网络

        Args:
            input_dim: 输入维度
            output_dim: 输出维度

        Returns:
            Dropout网络
        """
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.uncertainty_config['dropout_rate']),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.uncertainty_config['dropout_rate']),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量
            num_samples: 采样次数（用于不确定性估计）

        Returns:
            预测输出
        """
        if self.type == "ensemble":
            return self._forward_ensemble(x, num_samples)
        elif self.type == "bayesian" or self.type == "dropout":
            return self._forward_dropout(x, num_samples)
        else:
            raise ValueError(f"Unknown uncertainty type: {self.type}")

    def _forward_ensemble(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """集成模型前向传播

        Args:
            x: 输入张量
            num_samples: 采样次数

        Returns:
            预测输出
        """
        # 从集成中随机选择模型进行预测
        predictions = []
        for _ in range(num_samples):
            model_idx = np.random.randint(0, len(self.models))
            prediction = self.models[model_idx](x)
            predictions.append(prediction)

        return torch.stack(predictions)

    def _forward_dropout(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Dropout模型前向传播

        Args:
            x: 输入张量
            num_samples: 采样次数

        Returns:
            预测输出
        """
        # 启用dropout进行多次采样
        self.train()  # 确保dropout处于启用状态

        predictions = []
        for _ in range(num_samples):
            prediction = self.model(x)
            predictions.append(prediction)

        return torch.stack(predictions)

    def estimate_uncertainty(self, x: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """估计预测的不确定性

        Args:
            x: 输入张量
            num_samples: 采样次数

        Returns:
            不确定性估计字典
        """
        with torch.no_grad():
            samples = self.forward(x, num_samples)

            # 计算均值和方差
            mean = samples.mean(dim=0)
            variance = samples.var(dim=0)

            # 计算其他不确定性指标
            std = torch.sqrt(variance)
            confidence_interval = 1.96 * std  # 95%置信区间

            return {
                'mean': mean,
                'variance': variance,
                'std': std,
                'confidence_interval': confidence_interval,
                'samples': samples
            }

    def save(self, path: str) -> None:
        """保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'type': self.type
        }, path)
        logger.info(f"Uncertainty model saved to {path}")

    def load(self, path: str) -> None:
        """加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Uncertainty model loaded from {path}")