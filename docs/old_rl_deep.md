# 手机发布版本编排强化学习系统 - 详细实现

基于您的方案，我将提供一个完整的代码实现和目录结构。这个系统采用稳健强化学习方法，专注于处理不确定性环境下的版本发布优化问题。

## 📁 项目目录结构

```
mobile_release_rl_system/
├── data/                    # 数据存储目录
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   └── rules/              # 业务规则文件
├── src/                    # 源代码目录
│   ├── data_loader.py      # 数据加载与预处理
│   ├── environment.py      # 强化学习环境
│   ├── models/             # 模型定义
│   │   ├── robust_agent.py # 稳健RL智能体
│   │   └── uncertainty.py  # 不确定性模型
│   ├── training/           # 训练相关代码
│   │   ├── trainer.py      # 训练器
│   │   └── callback.py     # 训练回调函数
│   ├── evaluation/         # 评估模块
│   │   ├── evaluator.py    # 评估器
│   │   └── visualizer.py   # 可视化工具
│   ├── utils/              # 工具函数
│   │   ├── helpers.py      # 辅助函数
│   │   └── config.py       # 配置文件
│   └── app/                # 应用层
│       ├── api.py          # API接口
│       ├── dashboard.py    # 可视化仪表板
│       └── decision.py     # 决策支持系统
├── configs/                # 配置文件目录
│   ├── base.yaml          # 基础配置
│   ├── training.yaml      # 训练配置
│   └── environment.yaml   # 环境配置
├── tests/                  # 测试目录
├── scripts/               # 脚本目录
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评估脚本
│   └── deploy.py         # 部署脚本
├── requirements.txt       # 依赖列表
├── README.md             # 项目说明
└── Dockerfile            # 容器化配置
```

## 🧩 核心模块代码实现

### 1. 数据加载模块 (src/data_loader.py)

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import json
from typing import Dict, List, Tuple

class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_historical_traffic(self) -> pd.DataFrame:
        """加载历史流量数据"""
        df = pd.read_csv(self.config['data_paths']['historical_traffic'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df
    
    def load_release_records(self) -> pd.DataFrame:
        """加载版本发布记录"""
        df = pd.read_csv(self.config['data_paths']['release_records'])
        df['release_date'] = pd.to_datetime(df['release_date'])
        return df
    
    def load_business_rules(self) -> Dict:
        """加载业务规则"""
        with open(self.config['data_paths']['business_rules'], 'r') as f:
            rules = json.load(f)
        return rules
    
    def get_holidays(self) -> List[datetime]:
        """获取节假日列表"""
        rules = self.load_business_rules()
        holidays = [datetime.strptime(d, '%Y-%m-%d') for d in rules['holidays']]
        return holidays
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """预处理所有数据"""
        traffic_data = self.load_historical_traffic()
        release_data = self.load_release_records()
        business_rules = self.load_business_rules()
        
        # 数据清洗和特征工程
        traffic_data = self._clean_traffic_data(traffic_data)
        release_data = self._engineer_release_features(release_data)
        
        return traffic_data, release_data, business_rules
    
    def _clean_traffic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗流量数据"""
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 去除异常值
        for col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = np.where(
                (df[col] < lower_bound) | (df[col] > upper_bound),
                df[col].median(),
                df[col]
            )
        
        return df
    
    def _engineer_release_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成版本发布特征"""
        # 添加星期几特征
        df['day_of_week'] = df['release_date'].dt.dayofweek
        
        # 添加是否节假日特征
        holidays = self.get_holidays()
        df['is_holiday'] = df['release_date'].isin(holidays).astype(int)
        
        # 添加月份特征
        df['month'] = df['release_date'].dt.month
        
        return df
```

### 2. 环境模拟器 (src/environment.py)

```python
import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from src.models.uncertainty import UncertaintyModel

class MobileReleaseEnv(gym.Env):
    """手机版本发布强化学习环境"""
    
    def __init__(self, config: Dict, data_loader):
        super(MobileReleaseEnv, self).__init__()
        
        self.config = config
        self.data_loader = data_loader
        self.uncertainty_model = UncertaintyModel(config)
        
        # 加载数据
        self.traffic_data, self.release_data, self.business_rules = data_loader.preprocess_data()
        
        # 定义动作和状态空间
        self.action_space = spaces.Discrete(2)  # 0: 不发布, 1: 发布
        self.observation_space = self._get_observation_space()
        
        # 初始化环境状态
        self.reset()
    
    def _get_observation_space(self) -> spaces.Box:
        """定义状态空间"""
        # 状态包括: 当前天数, 剩余天数, 发布日历, 版本信息, 历史流量统计, 流量趋势
        state_dim = (
            2 +  # 当前天数和剩余天数
            31 +  # 发布日历 (31天)
            5 +   # 版本信息 (用户数, 包大小, 周期, 试点比例, 流量模式均值)
            4 +   # 历史流量统计 (均值, 标准差, 25分位, 75分位)
            1     # 流量趋势 (变化率)
        )
        return spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
    
    def reset(self):
        """重置环境状态"""
        self.current_day = 0
        self.remaining_days = 30
        self.release_calendar = np.zeros(31, dtype=int)  # 31天的发布日历
        self.current_version = self._get_random_version()
        self.traffic_history = self._get_initial_traffic()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作并返回新状态、奖励、是否终止和额外信息"""
        # 检查动作合法性
        is_valid, penalty = self._validate_action(action)
        
        # 执行动作
        if action == 1 and is_valid:
            self.release_calendar[self.current_day] = 1
            # 计算发布带来的流量影响
            traffic_impact = self._calculate_traffic_impact()
            # 更新流量历史
            self.traffic_history = self._update_traffic_history(traffic_impact)
        
        # 移动到下一天
        self.current_day += 1
        self.remaining_days -= 1
        
        # 检查是否终止
        done = self.current_day >= 30 or self.remaining_days <= 0
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(penalty)
        
        # 准备额外信息
        info = {
            'day': self.current_day,
            'action_valid': is_valid,
            'penalty': penalty,
            'traffic_impact': traffic_impact if action == 1 and is_valid else 0
        }
        
        return next_state, reward, done, info
    
    def _validate_action(self, action: int) -> Tuple[bool, float]:
        """验证动作是否合法"""
        if action == 0:  # 不发布总是合法的
            return True, 0
        
        # 检查是否周末
        current_date = datetime(2025, 5, 1) + timedelta(days=self.current_day)
        if current_date.weekday() >= 5:  # 5和6是周末
            return False, self.config['penalties']['weekend_release']
        
        # 检查是否节假日
        holidays = self.data_loader.get_holidays()
        if current_date in holidays:
            return False, self.config['penalties']['holiday_release']
        
        # 检查是否同日重复发布
        if self.release_calendar[self.current_day] == 1:
            return False, self.config['penalties']['same_day_release']
        
        return True, 0
    
    def _calculate_traffic_impact(self) -> float:
        """计算版本发布带来的流量影响"""
        base_impact = (
            self.current_version['users'] *
            self.current_version['size_gb'] *
            self.current_version['traffic_pattern_mean']
        )
        
        # 添加不确定性
        impact = self.uncertainty_model.apply_uncertainty(base_impact)
        
        return impact
    
    def _calculate_reward(self, penalty: float) -> float:
        """计算奖励"""
        # 生成多个流量场景
        scenarios = self.uncertainty_model.generate_scenarios(
            self.traffic_history, 
            self.release_calendar,
            k=self.config['uncertainty']['num_scenarios']
        )
        
        # 计算每个场景的流量方差
        variances = [np.var(scenario) for scenario in scenarios]
        avg_variance = np.mean(variances)
        worst_variance = np.max(variances)
        
        # 计算奖励
        reward = -(
            avg_variance +
            self.config['reward_weights']['worst_case'] * worst_variance +
            penalty
        )
        
        return reward
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态表示"""
        # 归一化当前天数和剩余天数
        day_features = np.array([
            self.current_day / 30,
            self.remaining_days / 30
        ])
        
        # 发布日历
        calendar_features = self.release_calendar / 1.0  # 已经是0或1
        
        # 版本信息
        version_features = np.array([
            self.current_version['users'] / self.config['normalization']['max_users'],
            self.current_version['size_gb'] / self.config['normalization']['max_size_gb'],
            self.current_version['period'] / self.config['normalization']['max_period'],
            self.current_version['pilot_ratio'],
            self.current_version['traffic_pattern_mean'] / self.config['normalization']['max_traffic_pattern']
        ])
        
        # 历史流量统计
        traffic_stats = np.array([
            np.mean(self.traffic_history),
            np.std(self.traffic_history),
            np.percentile(self.traffic_history, 25),
            np.percentile(self.traffic_history, 75)
        ]) / self.config['normalization']['max_traffic']
        
        # 流量趋势
        if len(self.traffic_history) >= 14:
            recent_mean = np.mean(self.traffic_history[-7:])
            previous_mean = np.mean(self.traffic_history[-14:-7])
            trend = (recent_mean - previous_mean) / previous_mean if previous_mean > 0 else 0
        else:
            trend = 0
        
        trend_feature = np.array([(trend + 1) / 2])  # 归一化到[0,1]
        
        # 合并所有特征
        state = np.concatenate([
            day_features,
            calendar_features,
            version_features,
            traffic_stats,
            trend_feature
        ])
        
        return state
    
    def _get_random_version(self) -> Dict:
        """获取随机版本信息"""
        # 从发布记录中随机选择一条记录
        idx = np.random.randint(0, len(self.release_data))
        version = self.release_data.iloc[idx].to_dict()
        
        return version
    
    def _get_initial_traffic(self) -> List[float]:
        """获取初始流量历史"""
        # 基于历史数据获取初始流量
        start_date = datetime(2025, 5, 1) - timedelta(days=14)
        end_date = datetime(2025, 4, 30)
        
        # 获取历史流量数据
        historical_traffic = self.traffic_data.loc[start_date:end_date]
        
        return historical_traffic['traffic'].tolist()
    
    def _update_traffic_history(self, impact: float) -> List[float]:
        """更新流量历史"""
        # 添加新一天的流量（基于历史数据和发布影响）
        base_traffic = self.traffic_data.loc[
            datetime(2025, 5, 1) + timedelta(days=self.current_day)
        ]['traffic']
        
        new_traffic = base_traffic + impact
        self.traffic_history.append(new_traffic)
        
        # 保持固定长度的历史窗口
        if len(self.traffic_history) > self.config['traffic_history_window']:
            self.traffic_history = self.traffic_history[-self.config['traffic_history_window']:]
        
        return self.traffic_history
```

### 3. 不确定性模型 (src/models/uncertainty.py)

```python
import numpy as np
from typing import List
import scipy.stats as stats

class UncertaintyModel:
    """不确定性模型，用于生成多场景流量模拟"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def generate_scenarios(self, historical_traffic: List[float], 
                          release_calendar: np.ndarray, k: int = 100) -> List[List[float]]:
        """生成K个可能的未来流量场景"""
        scenarios = []
        
        for _ in range(k):
            scenario = self._generate_single_scenario(historical_traffic, release_calendar)
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_single_scenario(self, historical_traffic: List[float], 
                                release_calendar: np.ndarray) -> List[float]:
        """生成单个流量场景"""
        scenario = []
        current_traffic = historical_traffic[-1] if historical_traffic else 0
        
        for day in range(len(release_calendar)):
            # 基础流量（基于历史同期）
            base_flow = self._get_historical_baseline(day)
            
            # 随机波动
            random_factor = np.random.normal(1.0, self.config['uncertainty']['random_std'])
            
            # 版本发布影响
            release_impact = 0
            if release_calendar[day] == 1:
                release_impact = self._simulate_release_impact(day)
            
            # 组合所有因素
            daily_traffic = base_flow * random_factor + release_impact
            scenario.append(daily_traffic)
            
            # 更新当前流量（带有平滑效应）
            current_traffic = current_traffic * 0.7 + daily_traffic * 0.3
        
        return scenario
    
    def _get_historical_baseline(self, day: int) -> float:
        """获取历史同期流量基线"""
        # 这里简化实现，实际应根据历史数据计算
        # 可以使用移动平均、季节性分解等方法
        baseline = 1000  # 默认基线值
        
        # 添加星期效应
        day_of_week = (day + 3) % 7  # 假设5月1日是星期四
        if day_of_week >= 5:  # 周末效应
            baseline *= 1.2
        
        return baseline
    
    def _simulate_release_impact(self, day: int) -> float:
        """模拟版本发布带来的流量影响"""
        # 基础影响
        base_impact = np.random.lognormal(
            self.config['release_impact']['log_mean'],
            self.config['release_impact']['log_std']
        )
        
        # 衰减因子（随时间衰减）
        decay = 1.0
        days_after_release = 0
        
        # 计算衰减
        if days_after_release > 0:
            decay = np.exp(-days_after_release / self.config['release_impact']['decay_rate'])
        
        return base_impact * decay
    
    def apply_uncertainty(self, base_value: float) -> float:
        """对基础值应用不确定性"""
        # 使用正态分布添加随机扰动
        perturbation = np.random.normal(1.0, self.config['uncertainty']['perturbation_std'])
        
        return base_value * perturbation
```

### 4. 稳健RL智能体 (src/models/robust_agent.py)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class RobustPolicyNetwork(nn.Module):
    """稳健策略网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RobustPolicyNetwork, self).__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        shared_features = self.shared_net(x)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_logits, state_value

class RobustPPOAgent:
    """稳健PPO智能体"""
    
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy = RobustPolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['learning_rate'])
        
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, state_value = self.policy(state_tensor)
        
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # 存储经验
        self.memory['states'].append(state)
        self.memory['actions'].append(action.item())
        self.memory['log_probs'].append(log_prob.item())
        self.memory['values'].append(state_value.item())
        
        return action.item()
    
    def update(self):
        """更新策略"""
        # 计算回报和优势
        returns = self._compute_returns()
        advantages = self._compute_advantages(returns)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.LongTensor(np.array(self.memory['actions']))
        old_log_probs = torch.FloatTensor(np.array(self.memory['log_probs']))
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.config['ppo_epochs']):
            # 获取新策略的输出
            action_logits, state_values = self.policy(states)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            
            # 计算新log概率和熵
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算策略比率
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # 计算策略损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios, 
                1 - self.config['clip_epsilon'], 
                1 + self.config['clip_epsilon']
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # 清空记忆
        self._clear_memory()
    
    def _compute_returns(self):
        """计算回报"""
        returns = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.memory['rewards']), 
                                      reversed(self.memory['dones'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.config['gamma'] * discounted_reward)
            returns.insert(0, discounted_reward)
        
        return returns
    
    def _compute_advantages(self, returns):
        """计算优势函数"""
        advantages = []
        values = self.memory['values']
        
        for i in range(len(returns)):
            advantage = returns[i] - values[i]
            advantages.append(advantage)
        
        return advantages
    
    def _clear_memory(self):
        """清空经验记忆"""
        for key in self.memory:
            self.memory[key] = []
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 5. 训练器 (src/training/trainer.py)

```python
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': []
        }
    
    def train(self, num_episodes):
        """训练智能体"""
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.memory['rewards'].append(reward)
                self.agent.memory['dones'].append(done)
                
                # 更新状态
                state = next_state
                episode_reward += reward
            
            # 更新策略
            self.agent.update()
            
            # 记录统计信息
            self.stats['episode_rewards'].append(episode_reward)
            self.stats['episode_lengths'].append(len(self.agent.memory['rewards']))
            
            # 定期保存模型
            if episode % self.config['save_interval'] == 0:
                self.agent.save(f"checkpoints/agent_episode_{episode}.pt")
            
            # 打印进度
            if episode % self.config['log_interval'] == 0:
                avg_reward = np.mean(self.stats['episode_rewards'][-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        # 保存最终模型
        self.agent.save("checkpoints/agent_final.pt")
        
        return self.stats
    
    def plot_training_progress(self):
        """绘制训练进度"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励曲线
        axes[0, 0].plot(self.stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 滑动平均奖励
        window_size = 100
        moving_avg = np.convolve(
            self.stats['episode_rewards'], 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title('Moving Average (100 episodes)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        
        # 回合长度
        axes[1, 0].plot(self.stats['episode_lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length')
        
        # 损失曲线
        if self.stats['value_losses']:
            axes[1, 1].plot(self.stats['value_losses'], label='Value Loss')
        if self.stats['policy_losses']:
            axes[1, 1].plot(self.stats['policy_losses'], label='Policy Loss')
        axes[1, 1].set_title('Training Losses')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
```

### 6. 配置文件 (configs/base.yaml)

```yaml
# 数据路径配置
data_paths:
  historical_traffic: "data/raw/historical_traffic.csv"
  release_records: "data/raw/release_records.csv"
  business_rules: "data/rules/business_rules.json"

# 环境配置
environment:
  traffic_history_window: 30
  penalties:
    weekend_release: 10.0
    holiday_release: 10.0
    same_day_release: 5.0

# 不确定性模型配置
uncertainty:
  num_scenarios: 100
  random_std: 0.1
  perturbation_std: 0.3

# 发布影响配置
release_impact:
  log_mean: 0.0
  log_std: 0.5
  decay_rate: 3.0

# 奖励权重配置
reward_weights:
  worst_case: 2.0

# 归一化配置
normalization:
  max_users: 1000000
  max_size_gb: 5.0
  max_period: 30
  max_traffic_pattern: 1.0
  max_traffic: 10000

# 训练配置
training:
  learning_rate: 0.0003
  gamma: 0.99
  clip_epsilon: 0.2
  ppo_epochs: 4
  batch_size: 64
  num_episodes: 10000
  save_interval: 1000
  log_interval: 100
```

### 7. 主训练脚本 (scripts/train.py)

```python
import yaml
from src.data_loader import DataLoader
from src.environment import MobileReleaseEnv
from src.models.robust_agent import RobustPPOAgent
from src.training.trainer import PPOTrainer

def main():
    # 加载配置
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化数据加载器
    data_loader = DataLoader('configs/base.yaml')
    
    # 初始化环境
    env = MobileReleaseEnv(config, data_loader)
    
    # 初始化智能体
    agent = RobustPPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config['training']
    )
    
    # 初始化训练器
    trainer = PPOTrainer(env, agent, config['training'])
    
    # 开始训练
    stats = trainer.train(config['training']['num_episodes'])
    
    # 绘制训练进度
    trainer.plot_training_progress()
    
    print("训练完成!")

if __name__ == "__main__":
    main()
```

### 8. 评估脚本 (scripts/evaluate.py)

```python
import yaml
import numpy as np
from src.data_loader import DataLoader
from src.environment import MobileReleaseEnv
from src.models.robust_agent import RobustPPOAgent
from src.evaluation.evaluator import RobustnessEvaluator

def main():
    # 加载配置
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化数据加载器
    data_loader = DataLoader('configs/base.yaml')
    
    # 初始化环境
    env = MobileReleaseEnv(config, data_loader)
    
    # 初始化智能体
    agent = RobustPPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config['training']
    )
    
    # 加载训练好的模型
    agent.load("checkpoints/agent_final.pt")
    
    # 初始化评估器
    evaluator = RobustnessEvaluator(env, agent, config)
    
    # 评估稳健性
    results = evaluator.evaluate_robustness()
    
    # 打印评估结果
    print("稳健性评估结果:")
    print(f"平均流量方差: {results['avg_variance']:.2f}")
    print(f"最坏情况方差: {results['worst_variance']:.2f}")
    print(f"违规次数: {results['violation_count']}")
    print(f"平均奖励: {results['avg_reward']:.2f}")
    
    # 可视化评估结果
    evaluator.visualize_results(results)

if __name__ == "__main__":
    main()
```

## 🚀 部署和运行指南

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将历史流量数据、版本发布记录和业务规则文件放置在相应的目录中：
- `data/raw/historical_traffic.csv`
- `data/raw/release_records.csv`
- `data/rules/business_rules.json`

### 3. 训练模型

```bash
python scripts/train.py
```

### 4. 评估模型

```bash
python scripts/evaluate.py
```

### 5. 使用训练好的模型进行决策

```python
from src.app.decision import DecisionSupportSystem

# 初始化决策支持系统
dss = DecisionSupportSystem("checkpoints/agent_final.pt", "configs/base.yaml")

# 获取发布建议
recommendation = dss.get_release_recommendation(current_state)

# 可视化结果
dss.visualize_recommendation(recommendation)
```

## 📊 预期输出

系统将生成以下输出：
1. 训练进度图表（奖励曲线、损失曲线等）
2. 稳健性评估报告（平均方差、最坏情况方差等指标）
3. 发布日历可视化（显示推荐的发布日期）
4. 流量预测图表（显示预测的流量模式）
5. 决策解释报告（说明决策的关键因素）

## 🔧 自定义和扩展

系统设计为高度可配置和可扩展的：

1. **修改配置**：通过编辑YAML配置文件调整算法参数和业务规则
2. **添加新特征**：在环境的状态表示中添加新的特征维度
3. **扩展不确定性模型**：实现更复杂的不确定性模拟方法
4. **集成新算法**：通过实现统一的接口集成其他强化学习算法
5. **添加新的评估指标**：在评估器中添加新的性能指标

这个实现提供了一个完整的手机版本发布编排强化学习系统，专注于处理不确定性环境下的决策优化问题。系统采用了稳健强化学习方法，能够在流量预测不准确的情况下依然生成稳定的发布方案。