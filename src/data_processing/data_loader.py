import pandas as pd
import numpy as np
import yaml
import os
from typing import Dict, Any, Tuple
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """数据加载器类，负责从不同源加载数据"""

    def __init__(self, config_path: str = "config/base.yaml"):
        """初始化数据加载器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.raw_path = self.config['data']['raw_path']
        self.processed_path = self.config['data']['processed_path']
        self.external_path = self.config['data']['external_path']

    def _load_config(self, config_path: str) -> dict:
        try:
            config_path_obj = Path(config_path)

            # 如果路径是绝对路径且存在，直接使用
            if config_path_obj.is_absolute() and config_path_obj.exists():
                abs_config_path = config_path_obj
            else:
                # 获取项目根目录的几种可能方式
                possible_roots = [
                    Path(__file__).parent.parent.parent,  # src/data_processing -> 项目根目录
                    Path.cwd(),
                    Path(os.environ.get('PROJECT_ROOT', '')),
                ]

                # 尝试每种可能的根目录
                for project_root in possible_roots:
                    if not project_root:  # 跳过空路径
                        continue

                    abs_config_path = (project_root / config_path).resolve()

                    if abs_config_path.exists():
                        break
                else:
                    # 如果所有尝试都失败，抛出错误
                    raise FileNotFoundError(f"Data loader config file not found: {config_path}")

            # 确保使用 UTF-8 编码
            with open(abs_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"加载数据加载器配置文件失败: {str(e)}")
            raise

    def load_traffic_data(self) -> pd.DataFrame:
        """加载流量数据

        Returns:
            流量数据DataFrame
        """
        file_path = os.path.join(self.raw_path, "traffic", self.config['data']['files']['traffic'])
        logger.info(f"Loading traffic data from {file_path}")

        try:
            df = pd.read_csv(
                file_path,
                parse_dates=[self.config['data']['traffic']['time_column']],
                index_col=self.config['data']['traffic']['time_column']
            )

            # 确保数据按时间排序
            df = df.sort_index()

            logger.info(f"Traffic data loaded successfully. Shape: {df.shape}")
            return df

        except FileNotFoundError:
            logger.error(f"Traffic data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading traffic data: {e}")
            raise

    def load_release_data(self) -> pd.DataFrame:
        """加载版本发布数据

        Returns:
            版本发布数据DataFrame
        """
        file_path = os.path.join(self.raw_path, "releases", self.config['data']['files']['releases'])
        logger.info(f"Loading release data from {file_path}")

        try:
            df = pd.read_csv(
                file_path,
                parse_dates=[self.config['data']['releases']['date_column']]
            )

            logger.info(f"Release data loaded successfully. Shape: {df.shape}")
            return df

        except FileNotFoundError:
            logger.error(f"Release data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading release data: {e}")
            raise

    def load_business_rules(self) -> Dict[str, Any]:
        """加载业务规则

        Returns:
            业务规则字典
        """
        file_path = os.path.join(self.raw_path, "rules", self.config['data']['files']['rules'])
        logger.info(f"Loading business rules from {file_path}")

        try:
            import json
            with open(file_path, 'r') as f:
                rules = json.load(f)

            logger.info("Business rules loaded successfully")
            return rules

        except FileNotFoundError:
            logger.error(f"Business rules file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading business rules: {e}")
            raise

    def load_holidays(self) -> pd.DataFrame:
        """加载节假日数据

        Returns:
            节假日数据DataFrame
        """
        file_path = os.path.join(self.external_path, "holidays", self.config['data']['files']['holidays'])
        logger.info(f"Loading holidays data from {file_path}")

        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            logger.info(f"Holidays data loaded successfully. Shape: {df.shape}")
            return df

        except FileNotFoundError:
            logger.error(f"Holidays data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading holidays data: {e}")
            raise

    def load_special_events(self) -> pd.DataFrame:
        """加载特殊事件数据

        Returns:
            特殊事件数据DataFrame
        """
        file_path = os.path.join(self.external_path, "events", self.config['data']['files']['events'])
        logger.info(f"Loading special events data from {file_path}")

        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            logger.info(f"Special events data loaded successfully. Shape: {df.shape}")
            return df

        except FileNotFoundError:
            logger.warning(f"Special events data file not found: {file_path}")
            return pd.DataFrame(columns=['date', 'event_type', 'impact_factor'])
        except Exception as e:
            logger.error(f"Error loading special events data: {e}")
            raise

    def load_all_data(self) -> Dict[str, Any]:
        """加载所有数据

        Returns:
            包含所有数据的字典
        """
        data_dict = {
            'traffic': self.load_traffic_data(),
            'releases': self.load_release_data(),
            'rules': self.load_business_rules(),
            'holidays': self.load_holidays(),
            'events': self.load_special_events()
        }

        return data_dict