import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """特征工程类，负责从原始数据中提取特征"""

    def __init__(self, config_path: str = "config/base.yaml"):
        """初始化特征工程器

        Args:
            config_path: 配置文件路径
        """
        self.preprocessor = DataPreprocessor(config_path)
        self.config = self.preprocessor.config

    def extract_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取流量特征

        Args:
            df: 预处理后的流量数据

        Returns:
            包含流量特征的DataFrame
        """
        logger.info("Extracting traffic features")

        # 重采样为日级数据
        daily_traffic = df.resample('D').agg({
            self.config['data']['traffic']['value_column']: ['mean', 'std', 'min', 'max']
        })

        # 扁平化列名
        daily_traffic.columns = ['_'.join(col).strip() for col in daily_traffic.columns.values]
        daily_traffic = daily_traffic.rename(columns={
            f"{self.config['data']['traffic']['value_column']}_mean": "daily_mean",
            f"{self.config['data']['traffic']['value_column']}_std": "daily_std",
            f"{self.config['data']['traffic']['value_column']}_min": "daily_min",
            f"{self.config['data']['traffic']['value_column']}_max": "daily_max"
        })

        # 添加滑动窗口统计特征
        for window in self.config['data']['features']['window_sizes']:
            daily_traffic[f'rolling_mean_{window}d'] = daily_traffic['daily_mean'].rolling(window=window).mean()
            daily_traffic[f'rolling_std_{window}d'] = daily_traffic['daily_mean'].rolling(window=window).std()
            daily_traffic[f'rolling_min_{window}d'] = daily_traffic['daily_mean'].rolling(window=window).min()
            daily_traffic[f'rolling_max_{window}d'] = daily_traffic['daily_mean'].rolling(window=window).max()

            # 添加分位数特征
            for q in self.config['data']['features']['quantiles']:
                daily_traffic[f'rolling_q{int(q * 100)}_{window}d'] = daily_traffic['daily_mean'].rolling(
                    window=window).quantile(q)

        # 添加趋势特征
        for period in self.config['data']['features']['trend_periods']:
            daily_traffic[f'trend_{period}d'] = daily_traffic['daily_mean'].pct_change(periods=period)

        # 添加时间特征
        daily_traffic['day_of_week'] = daily_traffic.index.dayofweek
        daily_traffic['day_of_month'] = daily_traffic.index.day
        daily_traffic['month'] = daily_traffic.index.month
        daily_traffic['is_weekend'] = daily_traffic['day_of_week'].isin([5, 6]).astype(int)

        logger.info(f"Traffic features extracted. Shape: {daily_traffic.shape}")
        return daily_traffic

    def extract_release_features(self, df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
        """提取版本发布特征

        Args:
            df: 预处理后的版本发布数据
            holidays: 节假日数据

        Returns:
            包含版本发布特征的DataFrame
        """
        logger.info("Extracting release features")

        # 创建日期范围
        date_range = pd.date_range(
            start=self.config['data']['time_range']['start_date'],
            end=self.config['data']['time_range']['end_date']
        )

        # 创建基础日历
        calendar = pd.DataFrame(index=date_range)
        calendar['date'] = calendar.index
        calendar['is_holiday'] = calendar['date'].isin(holidays['date']).astype(int)
        calendar['day_of_week'] = calendar.index.dayofweek
        calendar['is_weekend'] = calendar['day_of_week'].isin([5, 6]).astype(int)

        # 标记发布日
        calendar['is_release_day'] = calendar['date'].isin(df[self.config['data']['releases']['date_column']]).astype(
            int)

        # 添加发布相关信息
        release_info = df.set_index(self.config['data']['releases']['date_column'])
        calendar = calendar.merge(
            release_info,
            left_on='date',
            right_index=True,
            how='left'
        )

        # 计算距离上次发布的天数
        release_dates = df[self.config['data']['releases']['date_column']].sort_values()
        calendar['days_since_last_release'] = self._calculate_days_since_last_release(
            calendar.index, release_dates
        )

        logger.info(f"Release features extracted. Shape: {calendar.shape}")
        return calendar

    def _calculate_days_since_last_release(self, dates: pd.DatetimeIndex, release_dates: pd.Series) -> pd.Series:
        """计算距离上次发布的天数

        Args:
            dates: 日期序列
            release_dates: 发布日期序列

        Returns:
            距离上次发布的天数序列
        """
        result = []
        last_release = None

        for date in dates:
            # 找到最近的一次发布
            past_releases = release_dates[release_dates <= date]
            if not past_releases.empty:
                last_release = past_releases.max()
                days_since = (date - last_release).days
            else:
                days_since = -1  # 表示之前没有发布

            result.append(days_since)

        return pd.Series(result, index=dates)

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化特征

        Args:
            df: 包含特征的DataFrame

        Returns:
            标准化后的特征DataFrame
        """
        logger.info("Normalizing features")

        # 分离数值型和类别型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # 标准化数值型特征
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        df_numeric = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )

        # 合并回原始DataFrame
        result = pd.concat([df_numeric, df[categorical_cols]], axis=1)

        logger.info("Features normalized")
        return result

    def run_feature_engineering(self) -> Dict[str, Any]:
        """运行完整的特征工程流程

        Returns:
            包含所有特征的字典
        """
        logger.info("Starting feature engineering pipeline")

        # 加载预处理后的数据
        processed_data = self.preprocessor.run_preprocessing()

        # 提取特征
        traffic_features = self.extract_traffic_features(processed_data['traffic'])
        release_features = self.extract_release_features(
            processed_data['releases'],
            processed_data['holidays']
        )

        # 合并特征
        merged_features = release_features.merge(
            traffic_features,
            left_index=True,
            right_index=True,
            how='left'
        )

        # 标准化特征
        normalized_features = self.normalize_features(merged_features)

        # 保存特征
        feature_data = {
            'traffic_features': traffic_features,
            'release_features': release_features,
            'merged_features': merged_features,
            'normalized_features': normalized_features,
            'rules': processed_data['rules']
        }

        self.preprocessor.save_processed_data(feature_data)

        logger.info("Feature engineering pipeline completed")
        return feature_data