import numpy as np
from typing import Dict, Any, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DomainRandomizer:
    """域随机化器，管理环境参数的随机化"""

    def __init__(self, config: Dict[str, Any]):
        """初始化域随机化器

        Args:
            config: 域随机化配置
        """
        self.config = config
        self.parameters_config = config['parameters']
        self.current_parameters = self._initialize_parameters()

        logger.info("Domain randomizer initialized")

    def _initialize_parameters(self) -> Dict[str, Any]:
        """初始化参数

        Returns:
            参数字典
        """
        parameters = {}

        for param_name, param_config in self.parameters_config.items():
            if param_config['distribution'] == 'uniform':
                # 均匀分布
                value = np.random.uniform(
                    param_config['min'],
                    param_config['max']
                )
            elif param_config['distribution'] == 'normal':
                # 正态分布
                mean = (param_config['min'] + param_config['max']) / 2
                std = (param_config['max'] - param_config['min']) / 6  # 99.7%在min-max范围内
                value = np.random.normal(mean, std)
                # 裁剪到有效范围
                value = np.clip(value, param_config['min'], param_config['max'])
            else:
                # 默认使用均匀分布
                value = np.random.uniform(
                    param_config['min'],
                    param_config['max']
                )

            parameters[param_name] = value

        return parameters

    def randomize_parameters(self) -> Dict[str, Any]:
        """随机化参数

        Returns:
            随机化后的参数字典
        """
        self.current_parameters = self._initialize_parameters()
        return self.current_parameters

    def get_randomized_parameters(self) -> Dict[str, Any]:
        """获取当前随机化参数

        Returns:
            当前参数字典
        """
        return self.current_parameters

    def set_parameter(self, param_name: str, value: Any) -> None:
        """设置特定参数的值

        Args:
            param_name: 参数名称
            value: 参数值
        """
        if param_name in self.current_parameters:
            # 确保值在有效范围内
            if param_name in self.parameters_config:
                min_val = self.parameters_config[param_name]['min']
                max_val = self.parameters_config[param_name]['max']
                clipped_value = np.clip(value, min_val, max_val)
                self.current_parameters[param_name] = clipped_value
            else:
                self.current_parameters[param_name] = value
        else:
            logger.warning(f"Parameter {param_name} not found in domain randomizer")

    def get_parameter(self, param_name: str) -> Any:
        """获取特定参数的值

        Args:
            param_name: 参数名称

        Returns:
            参数值
        """
        return self.current_parameters.get(param_name)