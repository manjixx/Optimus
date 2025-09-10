import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from utils.logger import get_logger
import yaml
import os

logger = get_logger(__name__)


class ScenarioGenerator:
    """场景生成器类，负责生成不确定性场景"""

    def __init__(self, config_path: str = "config/base.yaml"):
        """初始化场景生成器

        Args:
            config_path: 配置文件路径（现在包含数据和环境配置）
        """
        self.feature_engineer = FeatureEngineer(config_path)
        self.config = self.feature_engineer.config

    def load_scenarios(self) -> List[Dict[str, Any]]:
        """从文件加载场景数据

        Returns:
            场景列表，如果文件不存在则返回空列表
        """
        scenarios_path = os.path.join(self.config['data']['processed_path'], "scenarios", "traffic_scenarios.csv")

        try:
            if os.path.exists(scenarios_path):
                scenarios_df = pd.read_csv(scenarios_path)
                scenarios = []

                # 按场景ID分组
                for scenario_id, group in scenarios_df.groupby('scenario_id'):
                    scenario = {
                        'scenario_id': scenario_id,
                        'scenario_name': group['scenario_name'].iloc[0],
                        'scenario_traffic': group['scenario_traffic'].values,
                        'traffic_baseline': group['scenario_traffic'].values
                    }
                    scenarios.append(scenario)

                logger.info(f"Loaded {len(scenarios)} scenarios from file")
                return scenarios
        except Exception as e:
            logger.warning(f"Failed to load scenarios from file: {e}")

        return []

    def generate_traffic_scenarios(self, traffic_features: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成流量场景

        Args:
            traffic_features: 流量特征数据

        Returns:
            流量场景列表
        """
        logger.info("Generating traffic scenarios")

        n_scenarios = self.config['data']['scenarios']['count']
        scenarios = []

        # 获取基础流量数据
        if 'daily_mean' in traffic_features.columns:
            base_traffic = traffic_features['daily_mean'].values
        else:
            # 如果没有daily_mean列，使用第一个数值列
            numeric_cols = traffic_features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                base_traffic = traffic_features[numeric_cols[0]].values
            else:
                # 如果没有数值列，创建默认流量
                base_traffic = np.ones(30) * 1000  # 默认30天，每天1000

        # 生成多个场景
        for i in range(n_scenarios):
            # 使用配置中的不确定性水平
            uncertainty = self.config['data']['scenarios'].get('uncertainty', 0.2)
            perturbation = np.random.normal(1.0, uncertainty, len(base_traffic))

            # 应用扰动
            scenario_traffic = base_traffic * perturbation

            # 确保非负值
            scenario_traffic = np.maximum(scenario_traffic, 0)

            # 创建场景字典
            scenario = {
                'scenario_id': i,
                'scenario_name': f'scenario_{i}',
                'scenario_traffic': scenario_traffic,
                'traffic_baseline': scenario_traffic,
                'uncertainty': uncertainty
            }

            scenarios.append(scenario)

        logger.info(f"Generated {len(scenarios)} traffic scenarios")
        return scenarios

    def generate_release_impact_scenarios(self, release_features: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成发布影响场景

        Args:
            release_features: 发布特征数据

        Returns:
            发布影响场景列表
        """
        logger.info("Generating release impact scenarios")

        n_scenarios = self.config['data']['scenarios']['count']
        scenarios = []

        # 生成多个场景
        for i in range(n_scenarios):
            # 随机生成影响因子
            impact_factor = np.random.uniform(0.8, 1.2)  # 影响因子在0.8到1.2之间

            scenario = {
                'scenario_id': i,
                'impact_factor': impact_factor,
                'description': f'Release impact scenario {i}'
            }

            scenarios.append(scenario)

        logger.info(f"Generated {len(scenarios)} release impact scenarios")
        return scenarios

    def load_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """加载特定场景配置

        Args:
            scenario_name: 场景名称

        Returns:
            场景配置字典
        """
        scenario_path = f"config/scenarios/{scenario_name}_scenario.yaml"
        try:
            with open(scenario_path, 'r') as f:
                scenario_config = yaml.safe_load(f)
            return scenario_config['scenario']
        except FileNotFoundError:
            logger.warning(f"Scenario config not found: {scenario_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading scenario config: {e}")
            return {}

    def generate_scenarios_with_config(self, scenario_name: str, n_scenarios: int = None) -> List[Dict[str, Any]]:
        """根据特定配置生成场景

        Args:
            scenario_name: 场景名称
            n_scenarios: 要生成的场景数量

        Returns:
            场景列表
        """
        if n_scenarios is None:
            n_scenarios = self.config['data']['scenarios']['count']

        logger.info(f"Generating {n_scenarios} {scenario_name} scenarios")

        # 加载场景配置
        scenario_config = self.load_scenario_config(scenario_name)

        scenarios = []
        feature_data = self.feature_engineer.run_feature_engineering()
        traffic_data = feature_data['traffic_features']

        # 获取基础流量
        if 'daily_mean' in traffic_data.columns:
            base_traffic = traffic_data['daily_mean'].values
        else:
            numeric_cols = traffic_data.select_dtypes(include=[np.number]).columns
            base_traffic = traffic_data[numeric_cols[0]].values if len(numeric_cols) > 0 else np.ones(30) * 1000

        for i in range(n_scenarios):
            # 使用场景特定的不确定性水平
            uncertainty = scenario_config.get('uncertainty', {}).get('level', 0.1)
            perturbation = np.random.normal(1.0, uncertainty, len(base_traffic))

            # 应用扰动
            scenario_traffic = base_traffic * perturbation

            # 应用场景特定的趋势和季节性
            trend = scenario_config.get('traffic', {}).get('trend', 0.0)
            seasonal_amplitude = scenario_config.get('traffic', {}).get('seasonality', {}).get('weekly_amplitude', 0.0)

            for j in range(len(scenario_traffic)):
                # 添加趋势
                scenario_traffic[j] *= (1 + trend * j / len(scenario_traffic))

                # 添加季节性
                day_of_week = j % 7
                seasonal_factor = 1 + seasonal_amplitude * np.sin(2 * np.pi * day_of_week / 7)
                scenario_traffic[j] *= seasonal_factor

            # 创建场景DataFrame
            scenario_df = traffic_data.copy()
            scenario_df['scenario_traffic'] = scenario_traffic
            scenario_df['scenario_id'] = i
            scenario_df['scenario_name'] = scenario_name

            scenarios.append({
                'traffic_baseline': scenario_traffic,
                'scenario_config': scenario_config,
                'scenario_df': scenario_df
            })

        logger.info(f"Generated {len(scenarios)} {scenario_name} scenarios")
        return scenarios

    def save_scenarios(self, traffic_scenarios: List[Dict[str, Any]], impact_scenarios: List[Dict[str, Any]]):
        """保存生成的场景

        Args:
            traffic_scenarios: 流量场景列表
            impact_scenarios: 影响场景列表
        """
        # 创建目录
        scenarios_path = os.path.join(self.config['data']['processed_path'], "scenarios")
        os.makedirs(scenarios_path, exist_ok=True)

        # 保存流量场景
        traffic_scenarios_list = []
        for scenario in traffic_scenarios:
            if 'scenario_df' in scenario:
                traffic_scenarios_list.append(scenario['scenario_df'])
            else:
                # 创建简单的DataFrame
                df = pd.DataFrame({
                    'scenario_id': scenario.get('scenario_id', 0),
                    'scenario_name': scenario.get('scenario_name', 'unknown'),
                    'traffic_baseline': scenario.get('traffic_baseline', [])
                })
                traffic_scenarios_list.append(df)

        if traffic_scenarios_list:
            traffic_scenarios_df = pd.concat(traffic_scenarios_list, ignore_index=True)
            traffic_scenarios_path = os.path.join(scenarios_path, "traffic_scenarios.csv")
            traffic_scenarios_df.to_csv(traffic_scenarios_path, index=False)
            logger.info(f"Saved traffic scenarios to {traffic_scenarios_path}")

        # 保存影响场景
        impact_scenarios_path = os.path.join(scenarios_path, "impact_scenarios.json")
        with open(impact_scenarios_path, 'w') as f:
            import json
            json.dump(impact_scenarios, f, indent=4)
        logger.info(f"Saved impact scenarios to {impact_scenarios_path}")

    def run_scenario_generation(self) -> Dict[str, Any]:
        """运行完整的场景生成流程

        Returns:
            包含所有场景的字典
        """
        logger.info("Starting scenario generation pipeline")

        # 运行特征工程获取处理后的数据
        feature_data = self.feature_engineer.run_feature_engineering()

        # 生成场景
        traffic_scenarios = self.generate_traffic_scenarios(feature_data['traffic_features'])
        impact_scenarios = self.generate_release_impact_scenarios(feature_data['release_features'])

        # 保存场景
        self.save_scenarios(traffic_scenarios, impact_scenarios)

        scenario_data = {
            'traffic_scenarios': traffic_scenarios,
            'impact_scenarios': impact_scenarios,
            'features': feature_data
        }

        logger.info("Scenario generation pipeline completed")
        return scenario_data