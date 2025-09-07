import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from ..utils.serialization import save_results

logger = get_logger(__name__)


class VisualizationEngine:
    """可视化引擎，生成各种图表和报告"""

    def __init__(self, config: Dict[str, Any]):
        """初始化可视化引擎

        Args:
            config: 可视化配置
        """
        self.config = config

        # 设置绘图样式
        self._setup_plot_style()

        logger.info("Visualization engine initialized")

    def _setup_plot_style(self) -> None:
        """设置绘图样式"""
        style = self.config.get('style', 'seaborn-whitegrid')
        palette = self.config.get('color_palette', 'Set2')

        plt.style.use(style)
        sns.set_palette(palette)

        # 设置中文字体支持（如果需要）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            logger.warning("Chinese font support not available")

    def generate_visualizations(self, recommendations: List[Dict[str, Any]],
                                evaluation: Dict[str, Any],
                                explanation: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """生成可视化图表

        Args:
            recommendations: 推荐方案列表
            evaluation: 评估结果
            explanation: 解释结果

        Returns:
            可视化文件路径字典
        """
        visualizations = {}

        # 创建输出目录
        output_dir = "results/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成流量预测图
        traffic_plot = self._plot_traffic_predictions(recommendations, evaluation)
        if traffic_plot:
            traffic_path = os.path.join(output_dir, f"traffic_{timestamp}.{self.config['output_format']}")
            traffic_plot.savefig(traffic_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close(traffic_plot)
            visualizations['traffic'] = traffic_path

        # 生成发布日历图
        calendar_plot = self._plot_release_calendar(recommendations)
        if calendar_plot:
            calendar_path = os.path.join(output_dir, f"calendar_{timestamp}.{self.config['output_format']}")
            calendar_plot.savefig(calendar_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close(calendar_plot)
            visualizations['calendar'] = calendar_path

        # 生成稳健性分析图
        robustness_plot = self._plot_robustness_analysis(evaluation)
        if robustness_plot:
            robustness_path = os.path.join(output_dir, f"robustness_{timestamp}.{self.config['output_format']}")
            robustness_plot.savefig(robustness_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close(robustness_plot)
            visualizations['robustness'] = robustness_path

        # 生成特征重要性图（如果有解释）
        if explanation and 'feature_importance' in explanation:
            importance_plot = self._plot_feature_importance(explanation['feature_importance'])
            if importance_plot:
                importance_path = os.path.join(output_dir, f"importance_{timestamp}.{self.config['output_format']}")
                importance_plot.savefig(importance_path, dpi=self.config['dpi'], bbox_inches='tight')
                plt.close(importance_plot)
                visualizations['importance'] = importance_path

        # 生成交互式图表（Plotly）
        if self.config['output_format'] == 'html':
            interactive_plot = self._create_interactive_dashboard(recommendations, evaluation, explanation)
            if interactive_plot:
                interactive_path = os.path.join(output_dir, f"dashboard_{timestamp}.html")
                interactive_plot.write_html(interactive_path)
                visualizations['interactive'] = interactive_path

        logger.info(f"Generated {len(visualizations)} visualizations")
        return visualizations

    def _plot_traffic_predictions(self, recommendations: List[Dict[str, Any]],
                                  evaluation: Dict[str, Any]) -> Optional[plt.Figure]:
        """生成流量预测图

        Args:
            recommendations: 推荐方案列表
            evaluation: 评估结果

        Returns:
             matplotlib图对象
        """
        if not recommendations or 'scenario_results' not in evaluation:
            return None

        # 获取最佳方案的流量数据
        best_plan = recommendations[0]
        scenario_results = evaluation['scenario_results']

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=self.config['figure_size'])
        fig.suptitle('流量预测与稳定性分析', fontsize=16)

        # 绘制各场景流量对比
        days = range(31)  # 假设一个月31天
        for scenario, results in scenario_results.items():
            if 'traffic_mean' in results:
                axes[0].plot(days, [results['traffic_mean']] * len(days),
                             label=scenario, alpha=0.7, linewidth=2)

        axes[0].set_title('各场景流量预测')
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('流量')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 绘制流量稳定性指标
        stability_data = []
        labels = []
        for scenario, results in scenario_results.items():
            if 'traffic_stability' in results:
                stability_data.append(results['traffic_stability'])
                labels.append(scenario)

        if stability_data:
            axes[1].bar(labels, stability_data, alpha=0.7)
            axes[1].set_title('各场景流量稳定性')
            axes[1].set_ylabel('稳定性指标')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def _plot_release_calendar(self, recommendations: List[Dict[str, Any]]) -> Optional[plt.Figure]:
        """生成发布日历图

        Args:
            recommendations: 推荐方案列表

        Returns:
            matplotlib图对象
        """
        if not recommendations:
            return None

        # 创建图表
        fig, ax = plt.subplots(figsize=self.config['figure_size'])

        # 绘制每个推荐方案的发布日历
        for i, plan in enumerate(recommendations[:3]):  # 只显示前3个方案
            release_days = plan.get('release_days', [])
            scores = plan.get('metrics', {}).get('composite_score', 0)

            # 创建日历数据
            calendar_data = np.zeros(31)  # 31天的月份
            for day in release_days:
                if day < len(calendar_data):
                    calendar_data[day] = 1

            # 绘制日历
            ax.step(range(len(calendar_data)), calendar_data + i * 0.1,
                    where='mid', label=f'方案 {i + 1} (评分: {scores:.1f})', linewidth=2)

        ax.set_title('发布日历推荐方案')
        ax.set_xlabel('日期')
        ax.set_yticks([0.05, 0.15, 0.25])
        ax.set_yticklabels(['方案 1', '方案 2', '方案 3'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_robustness_analysis(self, evaluation: Dict[str, Any]) -> Optional[plt.Figure]:
        """生成稳健性分析图

        Args:
            evaluation: 评估结果

        Returns:
            matplotlib图对象
        """
        if 'robustness_analysis' not in evaluation:
            return None

        robustness = evaluation['robustness_analysis']

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=self.config['figure_size'])
        fig.suptitle('稳健性分析', fontsize=16)

        # 绘制各场景性能对比
        scenario_results = evaluation.get('scenario_results', {})
        scenarios = list(scenario_results.keys())
        rewards = [results.get('mean_reward', 0) for results in scenario_results.values()]

        axes[0].bar(scenarios, rewards, alpha=0.7)
        axes[0].set_title('各场景平均奖励')
        axes[0].set_ylabel('平均奖励')
        axes[0].tick_params(axis='x', rotation=45)

        # 绘制稳健性指标
        if 'consistency_metrics' in robustness:
            consistency_data = []
            metric_labels = []

            for metric, data in robustness['consistency_metrics'].items():
                if 'consistency_score' in data:
                    consistency_data.append(data['consistency_score'])
                    metric_labels.append(metric)

            if consistency_data:
                axes[1].bar(metric_labels, consistency_data, alpha=0.7)
                axes[1].set_title('指标一致性')
                axes[1].set_ylabel('一致性评分')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def _plot_feature_importance(self, importance_data: Dict[str, float]) -> Optional[plt.Figure]:
        """生成特征重要性图

        Args:
            importance_data: 特征重要性数据

        Returns:
            matplotlib图对象
        """
        if not importance_data:
            return None

        # 提取特征和重要性值
        features = list(importance_data.keys())
        importance_values = list(importance_data.values())

        # 排序
        sorted_indices = np.argsort(importance_values)[::-1]
        features = [features[i] for i in sorted_indices]
        importance_values = [importance_values[i] for i in sorted_indices]

        # 创建图表
        fig, ax = plt.subplots(figsize=self.config['figure_size'])

        # 绘制水平条形图
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance_values, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # 最重要的特征在顶部
        ax.set_xlabel('重要性')
        ax.set_title('特征重要性分析')

        plt.tight_layout()
        return fig

    def _create_interactive_dashboard(self, recommendations: List[Dict[str, Any]],
                                      evaluation: Dict[str, Any],
                                      explanation: Optional[Dict[str, Any]] = None) -> Optional[go.Figure]:
        """创建交互式仪表板

        Args:
            recommendations: 推荐方案列表
            evaluation: 评估结果
            explanation: 解释结果

        Returns:
            Plotly图对象
        """
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('流量预测', '发布日历', '稳健性分析', '特征重要性'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )

            # 添加流量预测图
            scenario_results = evaluation.get('scenario_results', {})
            days = list(range(31))

            for scenario, results in scenario_results.items():
                if 'traffic_mean' in results:
                    fig.add_trace(
                        go.Scatter(
                            x=days,
                            y=[results['traffic_mean']] * len(days),
                            name=scenario,
                            mode='lines'
                        ),
                        row=1, col=1
                    )

            # 添加发布日历图
            for i, plan in enumerate(recommendations[:3]):
                release_days = plan.get('release_days', [])
                scores = plan.get('metrics', {}).get('composite_score', 0)

                # 创建日历数据
                calendar_data = np.zeros(31)
                for day in release_days:
                    if day < len(calendar_data):
                        calendar_data[day] = i + 1  # 用不同值区分方案

                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=calendar_data,
                        name=f'方案 {i + 1} (评分: {scores:.1f})',
                        mode='lines+markers'
                    ),
                    row=1, col=2
                )

            # 添加稳健性分析图
            if 'robustness_analysis' in evaluation:
                robustness = evaluation['robustness_analysis']

                if 'consistency_metrics' in robustness:
                    metrics = list(robustness['consistency_metrics'].keys())
                    scores = [data.get('consistency_score', 0) for data in
                              robustness['consistency_metrics'].values()]

                    fig.add_trace(
                        go.Bar(
                            x=metrics,
                            y=scores,
                            name='一致性评分'
                        ),
                        row=2, col=1
                    )

            # 添加特征重要性图
            if explanation and 'feature_importance' in explanation:
                importance_data = explanation['feature_importance']
                features = list(importance_data.keys())
                importance_values = list(importance_data.values())

                # 排序
                sorted_indices = np.argsort(importance_values)[::-1]
                features = [features[i] for i in sorted_indices]
                importance_values = [importance_values[i] for i in sorted_indices]

                fig.add_trace(
                    go.Bar(
                        x=importance_values,
                        y=features,
                        name='特征重要性',
                        orientation='h'
                    ),
                    row=2, col=2
                )

            # 更新布局
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="发布决策分析仪表板"
            )

            return fig

        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
            return None

    def generate_report(self, recommendation: Dict[str, Any], format: str = "pdf") -> str:
        """生成详细报告

        Args:
            recommendation: 推荐结果字典
            format: 报告格式

        Returns:
            报告文件路径
        """
        # 这里实现报告生成逻辑
        # 可以使用Jinja2模板引擎生成HTML报告，然后转换为PDF
        # 或者直接使用ReportLab生成PDF报告

        logger.info(f"Generating {format} report")

        # 创建输出目录
        output_dir = "results/reports"
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = recommendation.get('scenario', 'unknown')
        filename = f"report_{scenario}_{timestamp}.{format}"
        filepath = os.path.join(output_dir, filename)

        # 简化实现：保存推荐结果为JSON
        save_results(recommendation, filepath.replace(f".{format}", ".json"))

        # TODO: 实现完整的报告生成功能
        logger.warning("Full report generation not implemented yet")

        return filepath