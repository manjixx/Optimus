import torch
import torch.nn as nn
from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger(__name__)


class VersionImpactModel(nn.Module):
    """版本影响模型，预测版本发布对流量的影响"""

    def __init__(self, config: Dict[str, Any], input_dim: int):
        """初始化版本影响模型

        Args:
            config: 配置字典
            input_dim: 输入维度（版本特征数量）
        """
        super(VersionImpactModel, self).__init__()

        self.config = config
        self.impact_config = config['models']['version_impact']
        self.type = self.impact_config['type']

        # 创建模型
        if self.type == "linear":
            self.model = self._create_linear_model(input_dim)
        elif self.type == "neural_network":
            self.model = self._create_neural_network(input_dim)
        else:
            raise ValueError(f"Unknown version impact model type: {self.type}")

        logger.info(f"Created {self.type} version impact model")

    def _create_linear_model(self, input_dim: int) -> nn.Module:
        """创建线性模型

        Args:
            input_dim: 输入维度

        Returns:
            线性模型
        """
        return nn.Linear(input_dim, 1)  # 预测单个影响值

    def _create_neural_network(self, input_dim: int) -> nn.Module:
        """创建神经网络模型

        Args:
            input_dim: 输入维度

        Returns:
            神经网络模型
        """
        layers = []
        hidden_layers = self.impact_config['hidden_layers']
        activation = self.impact_config['activation']
        use_batch_norm = self.impact_config['use_batch_norm']
        dropout_rate = self.impact_config['dropout_rate']

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_layers[0]))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))

        layers.append(self._get_activation(activation))

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # 隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))

            layers.append(self._get_activation(activation))

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_layers[-1], 1))

        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数

        Args:
            activation: 激活函数名称

        Returns:
            激活函数模块
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        else:
            logger.warning(f"Unknown activation: {activation}, using ReLU")
            return nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量（版本特征）

        Returns:
            预测的影响值
        """
        return self.model(x)

    def predict_impact(self, version_features: Dict[str, Any]) -> float:
        """预测版本影响

        Args:
            version_features: 版本特征字典

        Returns:
            预测的影响值
        """
        # 将特征字典转换为张量
        feature_tensor = self._features_to_tensor(version_features)

        with torch.no_grad():
            impact = self.forward(feature_tensor)
            return impact.item()

    def _features_to_tensor(self, features: Dict[str, Any]) -> torch.Tensor:
        """将特征字典转换为张量

        Args:
            features: 特征字典

        Returns:
            特征张量
        """
        # 定义特征顺序（需要与训练时一致）
        feature_order = [
            'user_count',
            'package_size',
            'pilot_ratio',
            'traffic_pattern_mean',
            'cycle_days'
        ]

        # 提取特征值
        feature_values = []
        for feature_name in feature_order:
            value = features.get(feature_name, 0)
            feature_values.append(value)

        # 转换为张量
        return torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)

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
        logger.info(f"Version impact model saved to {path}")

    def load(self, path: str) -> None:
        """加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Version impact model loaded from {path}")