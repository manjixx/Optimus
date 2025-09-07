import numpy as np
from typing import Dict, Any, List, Optional
import shap
import lime
import lime.lime_tabular

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """模型解释器，提供决策可解释性分析"""

    def __init__(self, config: Dict[str, Any]):
        """初始化模型解释器

        Args:
            config: 可解释性配置
        """
        self.config = config
        self.method = config.get('method', 'shap')
        self.num_samples = config.get('num_samples', 100)
        self.top_features = config.get('top_features', 10)

        logger.info(f"Model explainer initialized with method: {self.method}")

    def explain_decision(self, model, env, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解释模型决策

        Args:
            model: 模型
            env: 环境
            recommendations: 推荐方案列表

        Returns:
            解释结果字典
        """
        if self.method == 'shap':
            return self._explain_with_shap(model, env)
        elif self.method == 'lime':
            return self._explain_with_lime(model, env)
        elif self.method == 'integrated_gradients':
            return self._explain_with_integrated_gradients(model, env)
        else:
            logger.warning(f"Unknown explanation method: {self.method}")
            return {}

    def _explain_with_shap(self, model, env) -> Dict[str, Any]:
        """使用SHAP解释模型决策

        Args:
            model: 模型
            env: 环境

        Returns:
            SHAP解释结果
        """
        try:
            # 收集样本数据
            observations = self._collect_observations(env, self.num_samples)

            # 创建解释器
            explainer = shap.Explainer(model.predict, observations)

            # 计算SHAP值
            shap_values = explainer(observations)

            # 提取特征重要性
            feature_importance = self._extract_feature_importance(shap_values, env)

            return {
                'method': 'shap',
                'feature_importance': feature_importance,
                'shap_values': shap_values.values,
                'base_values': shap_values.base_values
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}

    def _explain_with_lime(self, model, env) -> Dict[str, Any]:
        """使用LIME解释模型决策

        Args:
            model: 模型
            env: 环境

        Returns:
            LIME解释结果
        """
        try:
            # 收集样本数据
            observations = self._collect_observations(env, self.num_samples)

            # 创建解释器
            explainer = lime.lime_tabular.LimeTabularExplainer(
                observations,
                mode="regression",
                feature_names=self._get_feature_names(env)
            )

            # 选择要解释的样本
            sample_idx = np.random.randint(0, len(observations))
            sample = observations[sample_idx]

            # 解释预测
            explanation = explainer.explain_instance(
                sample,
                model.predict,
                num_features=self.top_features
            )

            # 提取特征重要性
            feature_importance = {}
            for feature, weight in explanation.as_list():
                feature_importance[feature] = weight

            return {
                'method': 'lime',
                'feature_importance': feature_importance,
                'local_explanation': explanation.as_list(),
                'explained_sample': sample_idx
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}

    def _explain_with_integrated_gradients(self, model, env) -> Dict[str, Any]:
        """使用积分梯度解释模型决策

        Args:
            model: 模型
            env: 环境

        Returns:
            积分梯度解释结果
        """
        # 积分梯度实现需要访问模型内部结构
        # 这里简化实现，返回空结果
        logger.warning("Integrated gradients not fully implemented")

        return {
            'method': 'integrated_gradients',
            'feature_importance': {},
            'message': 'Integrated gradients explanation not fully implemented'
        }

    def _collect_observations(self, env, num_samples: int) -> np.ndarray:
        """收集观测样本

        Args:
            env: 环境
            num_samples: 样本数量

        Returns:
            观测样本数组
        """
        observations = []

        # 重置环境
        state = env.reset()
        observations.append(state.flatten())

        # 收集更多样本
        for _ in range(num_samples - 1):
            # 随机动作
            action = env.action_space.sample()
            state, _, done, _ = env.step([action])

            observations.append(state.flatten())

            if done:
                state = env.reset()

        return np.array(observations)

    def _get_feature_names(self, env) -> List[str]:
        """获取特征名称

        Args:
            env: 环境

        Returns:
            特征名称列表
        """
        # 这里需要根据实际状态设计返回特征名称
        # 简化实现：返回通用特征名称
        return [
            'current_day', 'days_remaining', 'release_calendar',
            'user_count', 'package_size', 'pilot_ratio',
            'traffic_pattern_mean', 'cycle_days',
            'traffic_mean', 'traffic_std', 'traffic_q25', 'traffic_q75',
            'traffic_trend'
        ]

    def _extract_feature_importance(self, shap_values, env) -> Dict[str, float]:
        """从SHAP值中提取特征重要性

        Args:
            shap_values: SHAP值
            env: 环境

        Returns:
            特征重要性字典
        """
        feature_names = self._get_feature_names(env)

        # 计算平均绝对SHAP值
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

        # 创建特征重要性字典
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(mean_abs_shap):
                importance[name] = float(mean_abs_shap[i])

        # 归一化到0-1范围
        max_importance = max(importance.values()) if importance else 1.0
        if max_importance > 0:
            for name in importance:
                importance[name] /= max_importance

        # 只保留最重要的特征
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:self.top_features])

        return top_features

    def generate_explanation_report(self, explanation: Dict[str, Any]) -> str:
        """生成解释报告

        Args:
            explanation: 解释结果

        Returns:
            报告内容
        """
        if not explanation or 'error' in explanation:
            return "无法生成解释报告"

        report = f"模型决策解释报告\n"
        report += f"==================\n\n"
        report += f"解释方法: {explanation.get('method', '未知')}\n\n"

        if 'feature_importance' in explanation:
            report += "特征重要性排名:\n"
            report += "----------------\n"

            features = explanation['feature_importance']
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)

            for i, (feature, importance) in enumerate(sorted_features, 1):
                report += f"{i}. {feature}: {importance:.3f}\n"

        return report