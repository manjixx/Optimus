import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .data_loader import DataLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """数据预处理器类，负责数据清洗和预处理"""

    def __init__(self, config_path: str = "config/base.yaml"):
        """初始化数据预处理器

        Args:
            config_path: 配置文件路径
        """
        self.data_loader = DataLoader(config_path)
        self.config = self.data_loader.config
        self.processed_path = self.config['data']['processed_path']

    def preprocess_traffic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理流量数据

        Args:
            df: 原始流量数据

        Returns:
            处理后的流量数据
        """
        logger.info("Preprocessing traffic data")

        # 处理缺失值
        df = self._handle_missing_values(df, self.config['data']['traffic']['value_column'])

        # 去除异常值
        df = self._remove_outliers(df, self.config['data']['traffic']['value_column'])

        # 确保时间序列连续性
        df = self._ensure_time_continuity(df)

        logger.info("Traffic data preprocessing completed")
        return df

    def preprocess_release_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理版本发布数据

        Args:
            df: 原始版本发布数据

        Returns:
            处理后的版本发布数据
        """
        logger.info("Preprocessing release data")

        # 处理缺失值
        required_columns = [
            self.config['data']['releases']['date_column'],
            self.config['data']['releases']['version_column'],
            self.config['data']['releases']['user_count_column'],
            self.config['data']['releases']['package_size_column']
        ]

        for col in required_columns:
            if col in df.columns:
                df = self._handle_missing_values(df, col, method='drop')

        # 确保日期格式正确
        df[self.config['data']['releases']['date_column']] = pd.to_datetime(
            df[self.config['data']['releases']['date_column']]
        )

        # 排序
        df = df.sort_values(by=self.config['data']['releases']['date_column'])

        logger.info("Release data preprocessing completed")
        return df

    def _handle_missing_values(self, df: pd.DataFrame, column: str, method: str = 'interpolate') -> pd.DataFrame:
        """处理缺失值

        Args:
            df: 包含缺失值的数据
            column: 需要处理的列名
            method: 处理方法 ('interpolate', 'fill', 'drop')

        Returns:
            处理后的数据
        """
        if method == 'interpolate':
            df[column] = df[column].interpolate(method='linear')
        elif method == 'fill':
            # 使用前后值的平均值填充
            df[column] = df[column].fillna(df[column].rolling(window=3, min_periods=1).mean())
        elif method == 'drop':
            df = df.dropna(subset=[column])

        return df

    def _remove_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """去除异常值

        Args:
            df: 包含异常值的数据
            column: 需要处理的列名
            threshold: 异常值检测阈值

        Returns:
            处理后的数据
        """
        # 使用Z-score方法检测异常值
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < threshold]

        return df

    def _ensure_time_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保时间序列连续性

        Args:
            df: 时间序列数据

        Returns:
            连续的时间序列数据
        """
        # 创建完整的时间索引
        freq = self.config['data']['traffic']['frequency']
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )

        # 重新索引以填充缺失的时间点
        df = df.reindex(full_range)

        return df

    def save_processed_data(self, data_dict: Dict[str, Any]):
        """保存处理后的数据

        Args:
            data_dict: 包含处理后数据的字典
        """
        import os
        import json

        # 创建目录
        os.makedirs(os.path.join(self.processed_path, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_path, "scenarios"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_path, "normalized"), exist_ok=True)

        # 保存处理后的数据
        for key, value in data_dict.items():
            if isinstance(value, pd.DataFrame):
                file_path = os.path.join(self.processed_path, "features", f"{key}.csv")
                value.to_csv(file_path)
                logger.info(f"Saved {key} data to {file_path}")
            elif isinstance(value, dict):
                file_path = os.path.join(self.processed_path, "features", f"{key}.json")
                with open(file_path, 'w') as f:
                    json.dump(value, f, indent=4)
                logger.info(f"Saved {key} data to {file_path}")

    def run_preprocessing(self) -> Dict[str, Any]:
        """运行完整的数据预处理流程

        Returns:
            包含所有处理后数据的字典
        """
        logger.info("Starting data preprocessing pipeline")

        # 加载原始数据
        raw_data = self.data_loader.load_all_data()

        # 预处理数据
        processed_data = {
            'traffic': self.preprocess_traffic_data(raw_data['traffic']),
            'releases': self.preprocess_release_data(raw_data['releases']),
            'rules': raw_data['rules'],
            'holidays': raw_data['holidays'],
            'events': raw_data['events']
        }

        # 保存处理后的数据
        self.save_processed_data(processed_data)

        logger.info("Data preprocessing pipeline completed")
        return processed_data