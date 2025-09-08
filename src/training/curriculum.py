from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class CurriculumManager:
    """课程学习管理器，控制训练难度级别"""

    def __init__(self, config: Dict[str, Any]):
        """初始化课程学习管理器

        Args:
            config: 课程学习配置
        """
        self.config = config
        self.levels = config['levels']
        self.current_level_index = 0
        self.current_level = self.levels[self.current_level_index]
        self.progression_criteria = config['progression_criteria']

        # 跟踪每个级别的表现
        self.level_performance = {
            level['name']: {'episodes': 0, 'best_metric': -float('inf')}
            for level in self.levels
        }

        logger.info(f"Curriculum manager initialized with {len(self.levels)} levels")

    def get_current_level(self) -> Dict[str, Any]:
        """获取当前难度级别

        Returns:
            当前级别配置
        """
        return self.current_level

    def check_progression(self, eval_results: Dict[str, Any]) -> bool:
        """检查是否满足升级条件

        Args:
            eval_results: 评估结果

        Returns:
            是否满足升级条件
        """
        # 检查是否已经是最高级别
        if self.current_level_index >= len(self.levels) - 1:
            return False

        # 获取当前级别的性能指标
        metric_name = self.progression_criteria['metric']
        metric_value = eval_results.get(metric_name, -float('inf'))

        # 更新当前级别的性能记录
        level_name = self.current_level['name']
        self.level_performance[level_name]['episodes'] += eval_results.get('n_episodes', 0)
        self.level_performance[level_name]['best_metric'] = max(
            self.level_performance[level_name]['best_metric'], metric_value
        )

        # 检查是否满足升级条件
        min_episodes = self.progression_criteria.get('min_episodes', 0)
        threshold = self.progression_criteria.get('threshold', 0)

        episodes_completed = self.level_performance[level_name]['episodes']
        meets_threshold = metric_value >= threshold
        meets_min_episodes = episodes_completed >= min_episodes

        return meets_threshold and meets_min_episodes

    def advance_level(self) -> Dict[str, Any]:
        """提升到下一个难度级别

        Returns:
            新的级别配置
        """
        # 检查是否已经是最高级别
        if self.current_level_index >= len(self.levels) - 1:
            logger.warning("Already at the highest curriculum level")
            return self.current_level

        # 提升级别
        self.current_level_index += 1
        self.current_level = self.levels[self.current_level_index]

        logger.info(
            f"Advanced to curriculum level {self.current_level_index + 1}/"
            f"{len(self.levels)}: {self.current_level['name']}"
        )

        return self.current_level

    def reset_level(self) -> Dict[str, Any]:
        """重置到初始难度级别

        Returns:
            初始级别配置
        """
        self.current_level_index = 0
        self.current_level = self.levels[self.current_level_index]

        logger.info(f"Reset to initial curriculum level: {self.current_level['name']}")

        return self.current_level

    def get_progress(self) -> Dict[str, Any]:
        """获取课程学习进度

        Returns:
            进度信息字典
        """
        return {
            'current_level': self.current_level['name'],
            'current_level_index': self.current_level_index,
            'total_levels': len(self.levels),
            'level_performance': self.level_performance,
            'progress_percentage': (self.current_level_index + 1) / len(self.levels) * 100
        }