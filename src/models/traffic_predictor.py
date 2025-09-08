import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class TrafficPredictor(nn.Module):
    """流量预测模型（可选），预测未来流量"""

    def __init__(self, config: Dict[str, Any], input_size: int = 1):
        """初始化流量预测模型

        Args:
            config: 配置字典
            input_size: 输入特征维度
        """
        super(TrafficPredictor, self).__init__()

        self.config = config
        self.predictor_config = config['models']['traffic_predictor']
        self.type = self.predictor_config['type']

        # 模型参数
        self.input_size = input_size
        self.hidden_size = self.predictor_config['hidden_size']
        self.num_layers = self.predictor_config['num_layers']
        self.dropout_rate = self.predictor_config['dropout_rate']
        self.lookback_window = self.predictor_config['lookback_window']
        self.forecast_horizon = self.predictor_config['forecast_horizon']

        # 创建模型
        if self.type == "lstm":
            self.model = self._create_lstm_model()
        elif self.type == "transformer":
            self.model = self._create_transformer_model()
        else:
            raise ValueError(f"Unknown traffic predictor type: {self.type}")

        logger.info(f"Created {self.type} traffic predictor (optional)")

    def _create_lstm_model(self) -> nn.Module:
        """创建LSTM模型

        Returns:
            LSTM模型
        """
        return nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True
        )

    def _create_transformer_model(self) -> nn.Module:
        """创建Transformer模型

        Returns:
            Transformer模型
        """
        # 简化实现：使用TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_size,
            nhead=4,  # 头数
            dim_feedforward=self.hidden_size,
            dropout=self.dropout_rate,
            batch_first=True
        )

        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 [batch_size, lookback_window, input_size]

        Returns:
            预测输出 [batch_size, forecast_horizon, 1]
        """
        if self.type == "lstm":
            return self._forward_lstm(x)
        elif self.type == "transformer":
            return self._forward_transformer(x)
        else:
            raise ValueError(f"Unknown traffic predictor type: {self.type}")

    def _forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM前向传播

        Args:
            x: 输入张量

        Returns:
            预测输出
        """
        # LSTM前向传播
        lstm_out, _ = self.model(x)

        # 只取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 使用全连接层生成预测
        predictions = []
        for _ in range(self.forecast_horizon):
            pred = nn.Linear(self.hidden_size, 1)(last_output)
            predictions.append(pred.unsqueeze(1))

            # 更新输入（使用预测值作为下一个输入）
            # 在实际应用中，可能需要更复杂的解码策略
            if self.input_size == 1:
                last_output = pred.detach()  # 简单实现

        return torch.cat(predictions, dim=1)

    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer前向传播

        Args:
            x: 输入张量

        Returns:
            预测输出
        """
        # Transformer前向传播
        transformer_out = self.model(x)

        # 使用最后一个时间步的输出进行预测
        last_output = transformer_out[:, -1, :]

        # 使用全连接层生成预测
        predictions = []
        for _ in range(self.forecast_horizon):
            pred = nn.Linear(self.input_size, 1)(last_output)
            predictions.append(pred.unsqueeze(1))

        return torch.cat(predictions, dim=1)

    def predict(self, historical_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测未来流量

        Args:
            historical_data: 历史流量数据 [lookback_window]

        Returns:
            tuple: (预测值, 置信区间)
        """
        # 准备输入数据
        input_tensor = torch.tensor(historical_data, dtype=torch.float32)
        input_tensor = input_tensor.view(1, self.lookback_window, self.input_size)

        with torch.no_grad():
            # 进行预测
            predictions = self.forward(input_tensor)

            # 转换为numpy数组
            pred_np = predictions.squeeze().numpy()

            # 简化实现：返回预测值和固定置信区间
            confidence_interval = np.ones_like(pred_np) * 0.1 * pred_np  # 10%的置信区间

            return pred_np, confidence_interval

    def save(self, path: str) -> None:
        """保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'type': self.type,
            'input_size': self.input_size
        }, path)
        logger.info(f"Traffic predictor saved to {path}")

    def load(self, path: str) -> None:
        """加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.input_size = checkpoint['input_size']
        logger.info(f"Traffic predictor loaded from {path}")