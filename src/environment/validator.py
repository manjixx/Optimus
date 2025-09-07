from typing import Dict, Any, Tuple, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ActionValidator:
    """规则验证器类，负责验证动作是否符合业务规则"""

    def __init__(self, config: Dict[str, Any]):
        """初始化规则验证器

        Args:
            config: 环境配置
        """
        self.config = config
        self.validation_config = config['validation']

        # 初始化规则数据
        self.holidays = []
        self.weekends = [5, 6]  # 周六和周日

        # 统计信息
        self.invalid_attempts = 0

        logger.info("Action validator initialized")

    def validate_action(self, action: int, current_day: int,
                        release_calendar: np.ndarray,
                        version_info: Dict[str, Any]) -> Tuple[bool, str]:
        """验证动作是否符合规则

        Args:
            action: 要验证的动作
            current_day: 当前天数
            release_calendar: 发布日历
            version_info: 版本信息

        Returns:
            tuple: (是否有效, 验证消息)
        """
        # 无操作总是有效的
        if action == 0:
            return True, "No operation"

        # 检查是否已经在今天发布过
        if release_calendar[current_day] == 1:
            self.invalid_attempts += 1
            return False, "Already released today"

        # 检查是否是周末
        if not self.validation_config['allow_weekends'] and current_day % 7 in self.weekends:
            self.invalid_attempts += 1
            return False, "Weekend release not allowed"

        # 检查是否是节假日
        if not self.validation_config['allow_holidays'] and current_day in self.holidays:
            self.invalid_attempts += 1
            return False, "Holiday release not allowed"

        # 检查发布间隔
        min_days = self.validation_config['min_days_between_releases']
        if self._has_recent_release(release_calendar, current_day, min_days):
            self.invalid_attempts += 1
            return False, f"Minimum {min_days} days between releases required"

        return True, "Valid release"

    def _has_recent_release(self, release_calendar: np.ndarray,
                            current_day: int, min_days: int) -> bool:
        """检查最近是否有发布

        Args:
            release_calendar: 发布日历
            current_day: 当前天数
            min_days: 最小间隔天数

        Returns:
            最近是否有发布
        """
        if min_days <= 0:
            return False

        start_day = max(0, current_day - min_days)
        return np.any(release_calendar[start_day:current_day])

    def set_holidays(self, holidays: List[int]) -> None:
        """设置节假日

        Args:
            holidays: 节假日列表（日期索引）
        """
        self.holidays = holidays
        logger.info(f"Set {len(holidays)} holidays")

    def add_holiday(self, day: int) -> None:
        """添加节假日

        Args:
            day: 日期索引
        """
        if day not in self.holidays:
            self.holidays.append(day)
            logger.info(f"Added holiday on day {day}")

    def remove_holiday(self, day: int) -> None:
        """移除节假日

        Args:
            day: 日期索引
        """
        if day in self.holidays:
            self.holidays.remove(day)
            logger.info(f"Removed holiday on day {day}")