好的 ✅ 我会把 **“手机发布版本编排强化学习系统（稳健性优化与不确定性建模技术方案）”** 重新设计成详细的代码实现方案，并且拆分为 **3 批次** 输出。
第一批次我先给你：**代码目录结构 + 数据层 & 环境模拟器实现**。
后续第二批次会给你 **不确定性建模 + 奖励函数 + RL 智能体**，第三批次会给你 **训练流程 + 应用层接口 + 可视化 & 评估模块**。

---

# 📂 第一批次输出

## 1. 代码目录设计

```
rl_release_planner/
├── data_layer/                # 数据层
│   ├── data_loader.py         # 历史流量、版本、规则数据加载
│   ├── holiday_rules.py       # 节假日与规则约束
│   └── preprocess.py          # 特征提取与归一化
│
├── env/                       # 环境模拟器
│   ├── release_env.py         # 主环境类，状态转移 & 奖励
│   ├── traffic_simulator.py   # 流量基线与扰动模拟
│   ├── version_effect.py      # 版本发布对流量影响建模
│   └── utils.py               # 通用工具函数
│
├── agent/                     # 强化学习智能体
│   ├── ppo_agent.py           # PPO 实现（稳定训练）
│   ├── memory.py              # 经验回放缓冲区
│   └── model.py               # 策略网络 & 值函数网络
│
├── evaluation/                # 评估与对比
│   ├── evaluator.py           # 策略评估，多场景模拟
│   ├── baselines.py           # 与 GA / 人工策略对比
│   └── visualization.py       # 曲线、发布日历可视化
│
├── app/                       # 应用层接口
│   ├── api.py                 # REST API / gRPC 接口
│   ├── dashboard.py           # 决策可视化界面
│   └── planner.py             # 决策支持系统
│
├── configs/                   # 配置文件
│   ├── env_config.yaml        # 环境参数 (β, K, 最大天数等)
│   └── agent_config.yaml      # 训练超参数
│
├── tests/                     # 单元测试
│   ├── test_env.py
│   ├── test_agent.py
│   └── test_evaluator.py
│
├── main_train.py              # 主训练入口
├── main_eval.py               # 评估入口
└── requirements.txt           # 依赖库
```

---

## 2. 数据层实现

### `data_loader.py`

```python
import pandas as pd

class DataLoader:
    def __init__(self, traffic_path, version_path, rules_path):
        self.traffic_path = traffic_path
        self.version_path = version_path
        self.rules_path = rules_path

    def load_traffic(self):
        df = pd.read_csv(self.traffic_path, parse_dates=["date"])
        return df

    def load_versions(self):
        df = pd.read_csv(self.version_path, parse_dates=["release_date"])
        return df

    def load_rules(self):
        df = pd.read_csv(self.rules_path, parse_dates=["date"])
        return df
```

---

### `holiday_rules.py`

```python
import datetime

class HolidayRules:
    def __init__(self, holidays, weekends=True):
        self.holidays = set(holidays)
        self.weekends = weekends

    def is_valid_release(self, date):
        if self.weekends and date.weekday() >= 5:
            return False
        if date in self.holidays:
            return False
        return True
```

---

### `preprocess.py`

```python
import numpy as np

def normalize_feature(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val + 1e-8)

def extract_features(traffic_df, window=7):
    """
    输入: 历史流量数据
    输出: 特征 (均值, 标准差, 分位数, 趋势)
    """
    values = traffic_df["traffic"].values
    mean = np.mean(values[-window:])
    std = np.std(values[-window:])
    q25 = np.percentile(values[-window:], 25)
    q75 = np.percentile(values[-window:], 75)

    trend = (np.mean(values[-window:]) - np.mean(values[-2*window:-window])) / (np.mean(values[-2*window:-window]) + 1e-8)

    return np.array([mean, std, q25, q75, trend])
```

---

## 3. 环境模拟器实现

### `traffic_simulator.py`

```python
import numpy as np
import random

class TrafficSimulator:
    def __init__(self, base_traffic_series):
        self.base = base_traffic_series

    def sample_future(self, horizon=31, scenarios=100):
        """
        生成多场景流量轨迹
        """
        scenarios_list = []
        for _ in range(scenarios):
            noise = np.random.normal(1.0, 0.1, size=horizon)  # 随机扰动
            scenario = self.base[:horizon] * noise
            scenarios_list.append(scenario)
        return scenarios_list
```

---

### `version_effect.py`

```python
import numpy as np

class VersionEffect:
    def __init__(self, duration=5, decay=0.8):
        self.duration = duration
        self.decay = decay

    def apply(self, traffic_series, day, version_info):
        """
        版本发布对流量的影响
        """
        base_effect = version_info["users"] * version_info["pkg_size"] * version_info["mean_traffic"]
        disturbance = np.random.normal(1.0, 0.3)

        for i in range(self.duration):
            if day + i < len(traffic_series):
                traffic_series[day + i] += base_effect * (self.decay ** i) * disturbance

        return traffic_series
```

---

### `release_env.py`

```python
import numpy as np
from env.traffic_simulator import TrafficSimulator
from env.version_effect import VersionEffect

class ReleaseEnv:
    def __init__(self, traffic_baseline, rules, versions, horizon=31, scenarios=100):
        self.traffic_baseline = traffic_baseline
        self.rules = rules
        self.versions = versions
        self.horizon = horizon
        self.scenarios = scenarios
        self.reset()

    def reset(self):
        self.day = 0
        self.calendar = np.zeros(self.horizon, dtype=int)
        self.traffic = np.copy(self.traffic_baseline)
        return self._get_state()

    def _get_state(self):
        return {
            "day": self.day,
            "calendar": self.calendar.copy(),
            "traffic": self.traffic[:self.day+1]
        }

    def step(self, action, version_info):
        reward = 0
        done = False

        if action == 1:  # 发布版本
            if not self.rules.is_valid_release(self.day):
                reward -= 100  # 非法惩罚
            else:
                ve = VersionEffect()
                self.traffic = ve.apply(self.traffic, self.day, version_info)
                self.calendar[self.day] = 1
                reward += 1  # 基础奖励

        self.day += 1
        if self.day >= self.horizon:
            done = True

        return self._get_state(), reward, done
```

---

📌 **到这里：第一批次** 已经包含了 **数据层（数据加载、规则校验、特征处理）** + **环境模拟器（流量模拟、版本影响、环境交互）** 的完整实现。

要不要我现在继续输出 **第二批次（不确定性建模 + 奖励函数 + RL 智能体实现）**？

下面给出**第二批次代码**（不确定性建模 + 奖励函数 + RL 智能体实现）。
本批次默认可以直接接入你第一批次中的“环境状态与规则校验接口”，也内置了一个最小可运行的示例环境 `ReleaseEnv` 便于联调与单测。

---

# 代码目录

```
rl_core/
  __init__.py
  uncertainty/
    __init__.py
    scenarios.py          # 不确定性/多场景采样与极端扰动
  reward/
    __init__.py
    reward.py             # 95分位/方差型稳健奖励 + 违规惩罚
  agent/
    __init__.py
    networks.py           # 策略/价值网络
    buffers.py            # PPO经验缓冲
    ppo.py                # PPO-Clip 智能体（含鲁棒训练循环）
  envs/
    __init__.py
    release_env.py        # 最小可运行示例环境（调用不确定性与奖励）
  utils/
    __init__.py
    seeding.py            # 随机种子统一控制
    schedules.py          # 学习率/熵系数等调度
scripts/
  train_ppo.py            # 训练脚本（演示如何拼装各模块）
```

---

## rl\_core/uncertainty/scenarios.py

```python
# rl_core/uncertainty/scenarios.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class ScenarioConfig:
    num_scenarios: int = 100                # K，多场景数量
    horizon_days: int = 31                  # 固定月长示例
    base_jitter_std: float = 0.12           # 基线抖动强度（相对值）
    extreme_prob: float = 0.08              # 极端事件出现概率
    extreme_scale: Tuple[float, float] = (1.3, 1.8)  # 极端放大区间
    decay_days: int = 5                     # 发布影响的衰减天数
    robust_buffer_pct: float = 0.10         # 鲁棒缓冲比例（带宽上行留白）
    seed: Optional[int] = None

class ScenarioSampler:
    """
    多场景不确定性采样器：
      - 基线：从历史同期统计抽样+高斯扰动
      - 极端：以一定概率叠加放大事件
      - 发布影响：依据 “用户数×包大小×模式均值×扰动” 注入并按天衰减
      - 鲁棒缓冲：输出时统一乘以 (1 + buffer_pct)
    """
    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def sample_baseline(self, hist_daily_mean: np.ndarray) -> np.ndarray:
        """
        hist_daily_mean: shape (31,) 历史同期均值或分位数作为基线
        """
        assert hist_daily_mean.shape[0] == self.cfg.horizon_days
        # 对每条场景基线加乘性扰动：B_t * (1 + eps), eps ~ N(0, std)
        eps = self.rng.normal(0.0, self.cfg.base_jitter_std,
                              size=(self.cfg.num_scenarios, self.cfg.horizon_days))
        base = hist_daily_mean[None, :] * (1.0 + eps)
        base = np.clip(base, a_min=0.0, a_max=None)
        return base

    def maybe_extreme(self, base: np.ndarray) -> np.ndarray:
        """
        以概率对单个随机日触发极端放大，模拟节假日突发/热点
        """
        S, T = base.shape
        hit = self.rng.uniform(size=S) < self.cfg.extreme_prob
        # 每个命中场景随机选择1~2天触发
        for i in np.where(hit)[0]:
            days = self.rng.choice(T, size=self.rng.integers(1, 3), replace=False)
            scale = self.rng.uniform(*self.cfg.extreme_scale)
            base[i, days] *= scale
        return base

    def inject_release_impact(
        self,
        scenarios: np.ndarray,
        action_day: int,
        version_profile: Dict
    ) -> np.ndarray:
        """
        将一次“发布动作”的流量影响注入到所有场景
        version_profile:
          {
            "users": int,
            "pkg_mb": float,
            "pilot_ratio": float,   # 本次放量比例（如 0.01 首批≤1%）
            "shape_mean": float     # 历史影响强度基准
          }
        """
        S, T = scenarios.shape
        if not (0 <= action_day < T):
            return scenarios  # 非法天数直接忽略（环境会另行惩罚）

        users = version_profile["users"]
        pkg = version_profile["pkg_mb"]
        pilot = version_profile["pilot_ratio"]
        shape_mean = version_profile.get("shape_mean", 1.0)

        # 基础影响量（单位可近似为“相对带宽”）：用户×包×平均形状×比例
        base_influence = users * pkg * shape_mean * pilot

        # 对每个场景注入乘性扰动：~ lognormal(μ=0, σ=0.3)
        noise = self.rng.lognormal(mean=0.0, sigma=0.3, size=S)
        amp = base_influence * noise  # 每个场景有不同幅度

        # 影响在接下来 d=0..decay_days 的贡献，指数/线性衰减
        for d in range(self.cfg.decay_days):
            day = action_day + d
            if day >= T:
                break
            decay = np.exp(-0.6 * d)   # 指数衰减
            scenarios[:, day] += amp * decay

        return scenarios

    def apply_robust_buffer(self, scenarios: np.ndarray) -> np.ndarray:
        return scenarios * (1.0 + self.cfg.robust_buffer_pct)

    def roll(
        self,
        hist_daily_mean: np.ndarray,
        plan_actions: Dict[int, Dict]
    ) -> np.ndarray:
        """
        生成完整月度场景结果（含所有已决策的发布动作影响）
        plan_actions: { action_day -> version_profile }
        返回：scenarios, shape=(K, horizon_days)
        """
        s = self.sample_baseline(hist_daily_mean)
        s = self.maybe_extreme(s)
        for day, prof in sorted(plan_actions.items()):
            s = self.inject_release_impact(s, day, prof)
        s = self.apply_robust_buffer(s)
        return s
```

---

## rl\_core/reward/reward.py

```python
# rl_core/reward/reward.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class RewardConfig:
    mode: str = "var_worst"     # "p95_cost" | "var_worst"
    beta_worst: float = 0.7     # 最坏场景权重
    illegal_penalty: float = 2.5e6
    smooth_lambda: float = 0.0  # 邻日平滑奖励系数（>0鼓励平滑）
    upgrade_alpha: float = 0.0  # 升级率正向奖励系数（>0鼓励更快升级）

class RobustReward:
    """
    两种常用稳健目标：
      - p95_cost：以95分位作为计费近似，越小越好（取负）
      - var_worst：平均方差 + 最坏场景方差 * beta_worst
    并支持：
      - 规则违规惩罚
      - 邻日平滑奖励（差分平方项）
      - 升级率正向奖励（由环境提供）
    """
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    @staticmethod
    def _p95(x: np.ndarray) -> float:
        return float(np.percentile(x, 95))

    @staticmethod
    def _variance(x: np.ndarray) -> float:
        return float(np.var(x))

    @staticmethod
    def _smooth_penalty(curve: np.ndarray) -> float:
        # 邻日差分平方和
        diffs = np.diff(curve)
        return float(np.mean(diffs * diffs))

    def __call__(
        self,
        scenarios: np.ndarray,          # (K, T)
        illegal: bool = False,
        upgrade_rate: Optional[float] = None
    ) -> float:
        K, T = scenarios.shape
        rewards = []

        if self.cfg.mode == "p95_cost":
            # 对每个场景取P95，再求平均（也可直接对聚合曲线取P95）
            p95_each = np.apply_along_axis(self._p95, 1, scenarios)
            base = - float(np.mean(p95_each))  # 费用越高惩罚越大
            rewards.append(base)

        elif self.cfg.mode == "var_worst":
            # 对每个场景流量曲线计算方差
            var_each = np.var(scenarios, axis=1)
            base = - float(np.mean(var_each))
            worst = - float(np.max(var_each)) * self.cfg.beta_worst
            rewards.extend([base, worst])

        # 平滑奖励：对场景平均曲线求邻日差分惩罚（负号）
        if self.cfg.smooth_lambda > 0:
            mean_curve = scenarios.mean(axis=0)
            rewards.append(- self.cfg.smooth_lambda * self._smooth_penalty(mean_curve))

        # 升级率奖励（由环境/策略侧给入）
        if self.cfg.upgrade_alpha > 0 and upgrade_rate is not None:
            rewards.append(+ self.cfg.upgrade_alpha * float(upgrade_rate))

        # 非法动作惩罚
        if illegal:
            rewards.append(- self.cfg.illegal_penalty)

        return float(sum(rewards))
```

---

## rl\_core/agent/networks.py

```python
# rl_core/agent/networks.py
from __future__ import annotations
import torch
import torch.nn as nn

def mlp(sizes, activation=nn.Tanh, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else out_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

class CategoricalActor(nn.Module):
    """
    离散动作策略：适用于 {0:不发布, 1:发布} 或多臂选择
    """
    def __init__(self, obs_dim:int, act_dim:int, hidden=(128,128)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim])

    def _distribution(self, obs):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None if act is None else pi.log_prob(act)
        return pi, logp_a

class Critic(nn.Module):
    def __init__(self, obs_dim:int, hidden=(128,128)):
        super().__init__()
        self.v = mlp([obs_dim, *hidden, 1])

    def forward(self, obs):
        return self.v(obs).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, hidden=(128,128)):
        super().__init__()
        self.pi = CategoricalActor(obs_dim, act_dim, hidden)
        self.v  = Critic(obs_dim, hidden)

    def step(self, obs: torch.Tensor):
        with torch.no_grad():
            pi, _ = self.pi(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs: torch.Tensor):
        return self.step(obs)[0]
```

---

## rl\_core/agent/buffers.py

```python
# rl_core/agent/buffers.py
from __future__ import annotations
import numpy as np
import torch

class GAEBuffer:
    """
    PPO经验缓冲 + GAE(λ)
    """
    def __init__(self, obs_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int64)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        用于每个 episode 结束后计算GAE和回报
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std() + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v) for k, v in data.items()}

def discount_cumsum(x, discount):
    y = np.zeros_like(x, dtype=np.float32)
    c = 0.0
    for i in reversed(range(len(x))):
        c = x[i] + discount * c
        y[i] = c
    return y
```

---

## rl\_core/agent/ppo.py

```python
# rl_core/agent/ppo.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .networks import ActorCritic
from .buffers import GAEBuffer

@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int = 2
    steps_per_epoch: int = 4096
    epochs: int = 50
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.015
    entropy_coef: float = 0.0
    device: str = "cpu"
    hidden: tuple = (128,128)

class PPOAgent:
    """
    标准 PPO-Clip 实现，支持熵正则与KL早停
    """
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.ac = ActorCritic(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(cfg.device)
        self.buf = GAEBuffer(cfg.obs_dim, cfg.steps_per_epoch, cfg.gamma, cfg.lam)
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=cfg.pi_lr)
        self.vf_optimizer = optim.Adam(self.ac.v.parameters(), lr=cfg.vf_lr)

    def update(self, data):
        obs, act, ret, adv, logp_old = data["obs"], data["act"], data["ret"], data["adv"], data["logp"]

        # 策略更新
        for i in range(self.cfg.train_pi_iters):
            pi, logp = self.ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # 熵正则（可选）
            entropy = pi.entropy().mean()
            loss_pi -= self.cfg.entropy_coef * entropy

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.ac.pi.parameters(), 0.5)
            self.pi_optimizer.step()

            kl = (logp_old - logp).mean().item()
            if kl > 1.5 * self.cfg.target_kl:
                break

        # 价值函数更新
        for _ in range(self.cfg.train_v_iters):
            v = self.ac.v(obs)
            loss_v = ((v - ret)**2).mean()
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.ac.v.parameters(), 0.5)
            self.vf_optimizer.step()

    def train_one_epoch(self, env) -> Dict[str, Any]:
        o, ep_ret, ep_len = env.reset(), 0, 0
        for t in range(self.cfg.steps_per_epoch):
            obs_t = torch.as_tensor(o, dtype=torch.float32)
            a, v, logp = self.ac.step(obs_t)
            next_o, r, d, info = env.step(int(a))
            self.buf.store(o, a, r, v, logp)
            o = next_o
            ep_ret += r; ep_len += 1

            timeout = (t == self.cfg.steps_per_epoch - 1)
            terminal = d or timeout
            if terminal:
                last_val = 0 if d else self.ac.v(torch.as_tensor(o, dtype=torch.float32)).item()
                self.buf.finish_path(last_val)
                o, ep_ret, ep_len = env.reset(), 0, 0

        data = self.buf.get()
        self.update(data)
        return {"steps": self.cfg.steps_per_epoch}
```

---

## rl\_core/envs/release\_env.py  （示例环境，方便把不确定性与奖励串起来）

```python
# rl_core/envs/release_env.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from rl_core.uncertainty.scenarios import ScenarioSampler, ScenarioConfig
from rl_core.reward.reward import RobustReward, RewardConfig

@dataclass
class EnvConfig:
    horizon_days: int = 31
    max_actions: int = 8                 # 最多允许 N 次发布
    illegal_days: Tuple[int,...] = ()    # 周末/节假日（示例：用上游日历替换）
    first_batch_cap: float = 0.01        # 首批≤1%
    hist_level: float = 1.0              # 历史基线缩放
    seed: int = 42

class ReleaseEnv:
    """
    最小可运行环境：
      - 状态: [day_idx, remaining, mean, std, q25, q75, rolling_trend, 已发布计数]
      - 动作: 0=不发布, 1=发布
      - 规则: 避开非法日; 首批≤1%; 同日仅一次发布
      - 奖励: 调用 RobustReward (方差型/95分位) + 违规惩罚
      - 场景: ScenarioSampler 生成K条场景曲线（含极端/缓冲/发布影响）
    """
    def __init__(self,
                 env_cfg: EnvConfig,
                 sc_cfg: ScenarioConfig,
                 rw_cfg: RewardConfig):
        self.env_cfg = env_cfg
        self.sc_cfg = sc_cfg
        self.rw_cfg = rw_cfg

        self.day = 0
        self.done = False
        self.rng = np.random.default_rng(env_cfg.seed)

        self.hist_daily_mean = self._mock_hist_baseline()
        self.sampler = ScenarioSampler(sc_cfg)
        self.rewarder = RobustReward(rw_cfg)
        self.plan: Dict[int, Dict] = {}   # {day: version_profile}
        self.pilot_ratio_used = False

    def _mock_hist_baseline(self) -> np.ndarray:
        """示例：构造一个“工作日高/周末低”的历史均值曲线"""
        T = self.env_cfg.horizon_days
        base = np.array([1.2 if (i % 7) not in (5,6) else 0.8 for i in range(T)], dtype=np.float32)
        base *= self.env_cfg.hist_level * 1000.0  # 缩放到“相对带宽”量级
        return base

    # ---------- Gym-like API ----------
    def reset(self):
        self.day = 0
        self.done = False
        self.plan.clear()
        self.pilot_ratio_used = False
        return self._obs()

    def step(self, action:int):
        if self.done:
            raise RuntimeError("Episode finished, call reset().")

        illegal = False
        upgrade_rate = 0.0

        # 规则：非法日不允许发布
        if action == 1:
            if self.day in self.env_cfg.illegal_days:
                illegal = True
            if self.day in self.plan:
                illegal = True
            # 首批≤1%，只在首次发布时检查
            pilot = 0.005 if not self.pilot_ratio_used else 0.05  # 举例：后续放量变大
            if (not self.pilot_ratio_used) and (pilot > self.env_cfg.first_batch_cap):
                illegal = True

            if not illegal:
                self.plan[self.day] = dict(
                    users=5_000_000,
                    pkg_mb=80.0,
                    pilot_ratio=0.005 if not self.pilot_ratio_used else 0.05,
                    shape_mean=1.0
                )
                self.pilot_ratio_used = True

        # 基于当前计划生成 K 条场景
        scenarios = self.sampler.roll(self.hist_daily_mean, self.plan)

        # 计算奖励（此处为了演示按“日步”就评一次月度稳健指标）
        reward = self.rewarder(scenarios, illegal=illegal, upgrade_rate=upgrade_rate)

        # 推进一天
        self.day += 1
        if self.day >= self.env_cfg.horizon_days or len(self.plan) >= self.env_cfg.max_actions:
            self.done = True

        return self._obs(), reward, self.done, {"illegal": illegal}

    def _obs(self):
        # 观测：简化为若干统计量（可与你第一批次状态拼接）
        recent7 = self.hist_daily_mean[max(0, self.day-7): self.day] or self.hist_daily_mean[:1]
        recent7 = np.array(recent7, dtype=np.float32)
        mean = float(self.hist_daily_mean[self.day-1]) if self.day>0 else float(self.hist_daily_mean[0])
        obs = np.array([
            self.day,                                  # day_idx
            self.env_cfg.horizon_days - self.day,      # remaining days
            float(self.hist_daily_mean.mean()),        # hist mean
            float(self.hist_daily_mean.std()),         # hist std
            float(np.percentile(self.hist_daily_mean, 25)),
            float(np.percentile(self.hist_daily_mean, 75)),
            float(recent7.mean()) if recent7.size>0 else mean,
            float(len(self.plan))                      # 已发布计数
        ], dtype=np.float32)
        return obs
```

---

## rl\_core/utils/seeding.py

```python
# rl_core/utils/seeding.py
import numpy as np
import random
import torch

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## rl\_core/utils/schedules.py

```python
# rl_core/utils/schedules.py
def linear_decay(init: float, end: float, cur_step: int, total_steps: int) -> float:
    frac = max(0.0, min(1.0, 1.0 - cur_step / float(total_steps)))
    return end + (init - end) * frac
```

---

## scripts/train\_ppo.py  （演示：如何把三部分拼起来训练）

```python
# scripts/train_ppo.py
import torch
from rl_core.agent.ppo import PPOAgent, PPOConfig
from rl_core.envs.release_env import ReleaseEnv, EnvConfig
from rl_core.uncertainty.scenarios import ScenarioConfig
from rl_core.reward.reward import RewardConfig
from rl_core.utils.seeding import set_global_seed

def main():
    seed = 2025
    set_global_seed(seed)

    env_cfg = EnvConfig(
        horizon_days=31,
        max_actions=8,
        illegal_days=(5,6,12,13,19,20,26,27),   # 模拟周末
        first_batch_cap=0.01,
        hist_level=1.0,
        seed=seed
    )
    sc_cfg = ScenarioConfig(
        num_scenarios=100,
        horizon_days=31,
        base_jitter_std=0.12,
        extreme_prob=0.08,
        extreme_scale=(1.3, 1.8),
        decay_days=5,
        robust_buffer_pct=0.10,
        seed=seed
    )
    # 奖励可切换：mode="p95_cost" 或 "var_worst"
    rw_cfg = RewardConfig(
        mode="var_worst",
        beta_worst=0.7,
        illegal_penalty=2.5e6,
        smooth_lambda=1e-4,   # 鼓励平滑（可调）
        upgrade_alpha=0.0
    )

    env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)
    obs_dim = env.reset().shape[0]

    agent = PPOAgent(PPOConfig(
        obs_dim=obs_dim,
        act_dim=2,
        steps_per_epoch=4096,
        epochs=10,                 # 演示
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        target_kl=0.015,
        entropy_coef=0.01,
        device="cpu",
        hidden=(128,128)
    ))

    for ep in range(agent.cfg.epochs):
        stats = agent.train_one_epoch(env)
        print(f"[Epoch {ep+1}] steps={stats['steps']}")

if __name__ == "__main__":
    main()
```

---

### 说明与对齐点

* **不确定性建模**：`ScenarioSampler` 实现了“历史基线+高斯扰动+极端事件+发布影响+鲁棒缓冲”全链路，并支持多场景 `K`。
* **奖励函数**：`RobustReward` 同时支持 `p95_cost` 与 `var_worst` 两套稳健目标；包含**非法惩罚**、**平滑奖励**与（可选的）**升级率奖励**。
* **RL 智能体（PPO）**：标准 PPO-Clip，支持 KL 早停与熵正则；`train_one_epoch` 即可与任何 Gym-like 环境联训。
* **规则内嵌**：示例环境 `ReleaseEnv` 内置了“避开周末/节假日、同日一次、首批≤1%”等校验，并将结果传给奖励函数进行惩罚。
* **可替换性**：你可用生产环境的真实 `Env` 替换 `ReleaseEnv`，只需保持 `reset()/step()` 与状态/奖励调用接口一致即可。

如果你需要把本批次与**第一批次环境**全面对接，我可以直接将 `ReleaseEnv` 改造成你那边的状态/动作定义，并加上**版本分批比例动作空间**与**可配置约束器**。

下面给出 **第三批次完整代码实现**：训练流程（训练脚本与调度逻辑）、应用层接口（一个轻量 Flask API + 导出日历脚本）以及可视化与评估模块（多场景评估、与GA对比、绘图与报告导出）。代码与前两批次保持接口一致，可直接与之前的 `rl_core` 模块对接。

> 说明：
>
> * 假设前两批次代码已存在于 `rl_core/` 下（uncertainty, reward, agent, envs 等）。
> * 以下文件路径以项目根 `rl_release_planner/` 为准。
> * 为便于运行，我使用标准 Python 库与 matplotlib、pandas、Flask、numpy、torch 等依赖（与你的 `requirements.txt` 一致）。

---

## 一、训练流程（`main_train.py`）

主训练脚本，包含训练循环、模型保存、周期性评估与早停钩子。

```python
# main_train.py
import os
import time
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from rl_core.utils.seeding import set_global_seed
from rl_core.envs.release_env import ReleaseEnv, EnvConfig
from rl_core.uncertainty.scenarios import ScenarioConfig
from rl_core.reward.reward import RewardConfig
from rl_core.agent.ppo import PPOAgent, PPOConfig

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def save_model(agent: PPOAgent, path: Path):
    torch.save(agent.ac.state_dict(), path)

def load_model(agent: PPOAgent, path: Path):
    agent.ac.load_state_dict(torch.load(path))

def main_train(seed=2025, epochs=50, eval_interval=5):
    set_global_seed(seed)

    env_cfg = EnvConfig(horizon_days=31, max_actions=8,
                        illegal_days=(), first_batch_cap=0.01, hist_level=1.0, seed=seed)
    sc_cfg = ScenarioConfig(num_scenarios=100, horizon_days=31,
                            base_jitter_std=0.12, extreme_prob=0.08,
                            extreme_scale=(1.3,1.8), decay_days=5, robust_buffer_pct=0.10, seed=seed)
    rw_cfg = RewardConfig(mode="var_worst", beta_worst=0.7,
                          illegal_penalty=2.5e6, smooth_lambda=1e-4, upgrade_alpha=0.0)

    env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)
    obs_dim = env.reset().shape[0]

    agent_cfg = PPOConfig(obs_dim=obs_dim, act_dim=2, steps_per_epoch=4096,
                          epochs=epochs, gamma=0.99, lam=0.95, clip_ratio=0.2,
                          pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
                          target_kl=0.015, entropy_coef=0.01, device="cpu", hidden=(128,128))
    agent = PPOAgent(agent_cfg)

    best_reward = -1e9
    best_path = MODEL_DIR / f"best_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

    for ep in range(epochs):
        t0 = time.time()
        stats = agent.train_one_epoch(env)
        t1 = time.time()
        print(f"[Epoch {ep+1}/{epochs}] trained steps={stats['steps']} time={(t1-t0):.1f}s")

        # 定期评估
        if (ep + 1) % eval_interval == 0 or ep == epochs - 1:
            from evaluation.evaluator import Evaluator
            evaluator = Evaluator(env_cfg=env_cfg, sc_cfg=sc_cfg, rw_cfg=rw_cfg)
            metrics = evaluator.evaluate_policy(agent, n_episodes=20)
            avg_reward = metrics["avg_reward"]
            print(f"--> Eval at epoch {ep+1}: avg_reward={avg_reward:.3f} metrics={metrics}")

            # 如果表现更好则保存
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_model(agent, best_path)
                print(f"Saved new best model to {best_path}")

    # 最终模型保存
    final_path = MODEL_DIR / f"final_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    save_model(agent, final_path)
    print(f"Training completed. Final model saved to {final_path}")

if __name__ == "__main__":
    main_train()
```

---

## 二、评估模块（`evaluation/evaluator.py` 与 `evaluation/baselines.py`）

### `evaluation/evaluator.py`

多场景评估器：将 RL 策略在多个随机化场景中运行，输出稳健性指标（均值方差、最坏方差、P95、超阈次数等），并导出 CSV 报表。

```python
# evaluation/evaluator.py
import numpy as np
import pandas as pd
from typing import Dict, Any
from rl_core.envs.release_env import ReleaseEnv, EnvConfig
from rl_core.uncertainty.scenarios import ScenarioConfig
from rl_core.reward.reward import RewardConfig
from tqdm import trange

class Evaluator:
    def __init__(self, env_cfg: EnvConfig, sc_cfg: ScenarioConfig, rw_cfg: RewardConfig):
        self.env_cfg = env_cfg
        self.sc_cfg = sc_cfg
        self.rw_cfg = rw_cfg

    def evaluate_policy(self, agent, n_episodes: int = 50) -> Dict[str, Any]:
        """
        评估 agent 在固定随机种子/随机化下的稳健性能：
          - avg_reward: 评估所得奖励均值（episode-level）
          - avg_p95: 平均P95成本
          - worst_p95: 最大P95成本（最坏场景）
          - avg_var: 场景方差均值
          - worst_var: 场景方差最差
        """
        env = ReleaseEnv(self.env_cfg, self.sc_cfg, self.rw_cfg)
        p95_list, var_list, rewards = [], [], []

        for _ in trange(n_episodes, desc="Eval episodes"):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                # 获取 action from agent (actor net)
                obs_t = np.array(obs, dtype=np.float32)
                a = agent.ac.act(torch.as_tensor(obs_t).unsqueeze(0)).item() if hasattr(agent.ac, "act") else agent.ac.step(torch.as_tensor(obs_t))[0]
                obs, r, done, info = env.step(int(a))
                ep_reward += r

            # After episode end, evaluate final plan via ScenarioSampler roll
            sampler = env.sampler
            scenarios = sampler.roll(env.hist_daily_mean, env.plan)
            p95_each = np.percentile(scenarios, 95, axis=1)
            p95_mean = float(np.mean(p95_each))
            p95_max = float(np.max(p95_each))
            var_each = np.var(scenarios, axis=1)
            var_mean = float(np.mean(var_each))
            var_max = float(np.max(var_each))

            p95_list.append((p95_mean, p95_max))
            var_list.append((var_mean, var_max))
            rewards.append(ep_reward)

        metrics = {
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_p95_mean": float(np.mean([x[0] for x in p95_list])),
            "avg_p95_max": float(np.mean([x[1] for x in p95_list])),
            "avg_var_mean": float(np.mean([x[0] for x in var_list])),
            "avg_var_max": float(np.mean([x[1] for x in var_list])),
            "raw_rewards": rewards
        }
        return metrics

    def evaluate_and_export(self, agent, n_episodes=20, out_csv="evaluation_report.csv"):
        metrics = self.evaluate_policy(agent, n_episodes)
        df = pd.DataFrame([metrics])
        df.to_csv(out_csv, index=False)
        return metrics
```

### `evaluation/baselines.py`

提供一个简单 GA 基线（轻量版）用于对比。注意：这只是示例基线 — 真实GA可复用你先前的 `GeneticOptimizer`。

```python
# evaluation/baselines.py
import numpy as np
from typing import Dict

def uniform_baseline(env):
    """
    简单基线：均匀分配批次到合法工作日上（忽略极端优化）
    返回一个 'plan' 字典 {day: profile_dict}
    """
    horizon = env.env_cfg.horizon_days
    # 找到所有合法日
    legal_days = [d for d in range(horizon) if d not in env.env_cfg.illegal_days]
    max_actions = env.env_cfg.max_actions
    chosen = legal_days[:max_actions]
    plan = {}
    for i, day in enumerate(chosen):
        plan[day] = dict(users=5_000_000, pkg_mb=80.0, pilot_ratio=0.01 if i==0 else 0.1, shape_mean=1.0)
    return plan

def evaluate_baseline(env_cfg, sc_cfg, rw_cfg, baseline_fn=uniform_baseline):
    env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)
    # use baseline plan directly
    plan = baseline_fn(env)
    sampler = env.sampler
    scenarios = sampler.roll(env.hist_daily_mean, plan)
    res = {
        "p95_mean": float(np.mean(np.percentile(scenarios, 95, axis=1))),
        "p95_max": float(np.max(np.percentile(scenarios, 95, axis=1))),
        "var_mean": float(np.mean(np.var(scenarios, axis=1))),
        "var_max": float(np.max(np.var(scenarios, axis=1))),
    }
    return res
```

---

## 三、可视化模块（`evaluation/visualization.py`）

绘制训练曲线、场景流量曲线、发布日历导出图表等。

```python
# evaluation/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any

def plot_scenarios(scenarios: np.ndarray, title="Scenario flows", out_png=None):
    """
    scenarios: (K, T)
    """
    K, T = scenarios.shape
    fig, ax = plt.subplots(figsize=(10,5))
    for i in range(min(K, 50)):   # 最多画50条以免过密
        ax.plot(range(T), scenarios[i], alpha=0.15)
    mean_curve = scenarios.mean(axis=0)
    ax.plot(range(T), mean_curve, color='black', linewidth=2, label='Mean curve')
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Relative Traffic")
    ax.legend()
    if out_png:
        fig.savefig(out_png, dpi=150)
    plt.show()

def plot_calendar(plan: Dict[int, Dict], horizon=31, out_png=None):
    """
    简单绘制发布日历：x轴为天，y为是否发布/批次量
    plan: {day: profile}
    """
    arr = np.zeros(horizon)
    for d, prof in plan.items():
        arr[d] = prof.get("pilot_ratio", 0.0) + prof.get("pkg_mb",0.0) * 0.0 + 1.0
    fig, ax = plt.subplots(figsize=(12,3))
    ax.bar(range(horizon), arr, color='tab:blue')
    ax.set_title("Release Calendar (bars indicate release days)")
    ax.set_xlabel("Day of month")
    ax.set_ylabel("Release flag / proxy size")
    if out_png:
        fig.savefig(out_png, dpi=150)
    plt.show()

def plot_eval_metrics(metrics: Dict[str, Any], out_png=None):
    keys = [k for k in metrics.keys() if k.startswith("avg_") or k.endswith("_p95") or k.endswith("_var")]
    vals = [metrics[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(keys, vals)
    ax.set_title("Evaluation metrics summary")
    ax.set_xticklabels(keys, rotation=45, ha='right')
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 四、应用层接口（`app/api.py` 与 `app/export_calendar.py`）

### `app/api.py`（Flask 快速 API，提供模型预测、导出计划）

```python
# app/api.py
from flask import Flask, request, jsonify, send_file
import torch
import json
from pathlib import Path
from rl_core.envs.release_env import EnvConfig
from rl_core.uncertainty.scenarios import ScenarioConfig
from rl_core.reward.reward import RewardConfig
from rl_core.agent.ppo import PPOAgent, PPOConfig
from evaluation.evaluator import Evaluator
import numpy as np

MODEL_DIR = Path("models")
LATEST_MODEL = sorted(MODEL_DIR.glob("*.pth"))[-1] if any(MODEL_DIR.glob("*.pth")) else None

app = Flask(__name__)

def _load_agent(model_path: str):
    # 创建 agent skeleton，并加载参数
    # 需要与训练时保持 obs_dim, act_dim 一致
    env_cfg = EnvConfig()
    sc_cfg = ScenarioConfig()
    rw_cfg = RewardConfig()
    dummy_env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)
    obs_dim = dummy_env.reset().shape[0]
    agent_cfg = PPOConfig(obs_dim=obs_dim, act_dim=2, steps_per_epoch=4096, epochs=1)
    agent = PPOAgent(agent_cfg)
    agent.ac.load_state_dict(torch.load(model_path, map_location="cpu"))
    return agent

@app.route("/predict_plan", methods=["POST"])
def predict_plan():
    """
    请求示例:
      { "model": "best_agent_xxx.pth", "context": { ... } }
    返回:
      { "plan": {day: profile}, "metrics": {...} }
    """
    data = request.json
    model_name = data.get("model", None) or str(LATEST_MODEL)
    if model_name is None:
        return jsonify({"error": "No model available"}), 400
    model_path = Path(model_name)
    if not model_path.exists():
        model_path = MODEL_DIR / model_name
    if not model_path.exists():
        return jsonify({"error": f"Model {model_name} not found"}), 404

    agent = _load_agent(str(model_path))
    # create env and run greedy policy to produce plan
    env_cfg = EnvConfig()
    sc_cfg = ScenarioConfig()
    rw_cfg = RewardConfig()
    env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)

    obs = env.reset()
    done = False
    while not done:
        obs_t = np.array(obs, dtype=np.float32)
        a = agent.ac.act(torch.as_tensor(obs_t).unsqueeze(0)).item()
        obs, r, done, info = env.step(int(a))

    plan = env.plan
    # Evaluate final plan for reporting
    sampler = env.sampler
    scenarios = sampler.roll(env.hist_daily_mean, plan)
    p95_each = np.percentile(scenarios, 95, axis=1)
    metrics = {
        "p95_mean": float(np.mean(p95_each)),
        "p95_max": float(np.max(p95_each)),
        "var_mean": float(np.mean(np.var(scenarios, axis=1))),
        "var_max": float(np.max(np.var(scenarios, axis=1)))
    }
    return jsonify({"plan": {int(k): v for k,v in plan.items()}, "metrics": metrics})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"alive": True, "model_loaded": bool(LATEST_MODEL)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

### `app/export_calendar.py`（导出 CSV）

```python
# app/export_calendar.py
import csv
from datetime import datetime
from typing import Dict

def export_plan_to_csv(plan: Dict[int, Dict], filename="release_calendar.csv"):
    """
    plan: {day_idx: profile}
    输出列: day, users, pkg_mb, pilot_ratio, shape_mean
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "users", "pkg_mb", "pilot_ratio", "shape_mean"])
        for day, prof in sorted(plan.items()):
            writer.writerow([int(day), prof.get("users"), prof.get("pkg_mb"),
                             prof.get("pilot_ratio"), prof.get("shape_mean")])
    return filename
```

---

## 五、示例评估 & 可视化脚本（`main_eval.py`）

```python
# main_eval.py
from pathlib import Path
import torch
from rl_core.envs.release_env import EnvConfig
from rl_core.uncertainty.scenarios import ScenarioConfig
from rl_core.reward.reward import RewardConfig
from rl_core.agent.ppo import PPOAgent, PPOConfig
from evaluation.evaluator import Evaluator
from evaluation.visualization import plot_eval_metrics, plot_scenarios, plot_calendar
import numpy as np

def load_agent_from_path(path):
    # assume obs_dim matches
    env_cfg = EnvConfig()
    sc_cfg = ScenarioConfig()
    rw_cfg = RewardConfig()
    env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)
    obs_dim = env.reset().shape[0]
    agent_cfg = PPOConfig(obs_dim=obs_dim)
    agent = PPOAgent(agent_cfg)
    agent.ac.load_state_dict(torch.load(path, map_location="cpu"))
    return agent

def main_eval(model_path=None):
    model_dir = Path("models")
    if model_path is None:
        models = sorted(model_dir.glob("*.pth"))
        if not models:
            raise FileNotFoundError("No models found")
        model_path = models[-1]
    print("Loading model:", model_path)
    agent = load_agent_from_path(str(model_path))

    env_cfg = EnvConfig()
    sc_cfg = ScenarioConfig()
    rw_cfg = RewardConfig()

    evaluator = Evaluator(env_cfg, sc_cfg, rw_cfg)
    metrics = evaluator.evaluate_policy(agent, n_episodes=30)
    print("Evaluation metrics:", metrics)

    # one example rollout to visualize
    env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)
    obs = env.reset()
    done = False
    while not done:
        obs_t = np.array(obs, dtype=np.float32)
        a = agent.ac.act(torch.as_tensor(obs_t).unsqueeze(0)).item()
        obs, r, done, info = env.step(int(a))

    scenarios = env.sampler.roll(env.hist_daily_mean, env.plan)
    plot_scenarios(scenarios, title="Example rollout scenarios", out_png="examples_scenarios.png")
    plot_calendar(env.plan, horizon=env_cfg.horizon_days, out_png="examples_calendar.png")
    plot_eval_metrics(metrics, out_png="examples_metrics.png")
    print("Saved example visualizations to disk")

if __name__ == "__main__":
    main_eval()
```

---

## 六、单元测试（`tests/test_training_eval.py` 简单示例）

确保训练 loop / eval 能跑通（运行速度可调整）。

```python
# tests/test_training_eval.py
import pytest
from rl_core.envs.release_env import EnvConfig
from rl_core.uncertainty.scenarios import ScenarioConfig
from rl_core.reward.reward import RewardConfig
from rl_core.agent.ppo import PPOAgent, PPOConfig
from rl_core.utils.seeding import set_global_seed

def test_env_and_agent_quick():
    set_global_seed(123)
    env_cfg = EnvConfig(horizon_days=10, max_actions=4, illegal_days=(5,6), first_batch_cap=0.01, hist_level=1.0, seed=123)
    sc_cfg = ScenarioConfig(num_scenarios=10, horizon_days=10, seed=123)
    rw_cfg = RewardConfig(mode="var_worst", beta_worst=0.5)
    from rl_core.envs.release_env import ReleaseEnv
    env = ReleaseEnv(env_cfg, sc_cfg, rw_cfg)
    obs_dim = env.reset().shape[0]
    agent_cfg = PPOConfig(obs_dim=obs_dim, act_dim=2, steps_per_epoch=128, epochs=1)
    agent = PPOAgent(agent_cfg)
    # run a brief training epoch to check integration
    stats = agent.train_one_epoch(env)
    assert "steps" in stats
```

---

## 七、运行与部署建议

1. 开发/调试阶段：在 CPU 上小规模跑 `main_train.py`（epochs=少量，steps\_per\_epoch小）以确保逻辑链路无误。
2. 训练阶段：在 GPU 上跑 `main_train.py`，并把 `steps_per_epoch` / `epochs` 扩大到生产级别。
3. 模型部署：将训练好的 `.pth` 模型放到 `models/` 下，运行 `app/api.py` 提供预测接口。
4. 评估阶段：定期使用 `main_eval.py` 进行回归评估，与 GA 基线比较。
5. 监控与线上校正：把真实发布数据回写为新的历史样本用于后续在线学习或周期重训。

---

## 八、结束语

以上代码实现了**第三批次**所需的训练流程、评估/对比、可视化与一个轻量的服务接口。它与前两批次（环境、规则、不确定性、奖励、PPO agent）可以无缝衔接。

如果你需要：

* 我可以把整个项目打包为可运行的 GitHub 仓库（包含 `dockerfile`, CI 测试脚本）；
* 或者我可以将 GA 的完整实现（你之前的 `GeneticOptimizer`）并入 `evaluation/baselines.py` 作为更强的 baseline；
* 也可以把 Flask API 扩展为带认证、并发限制的生产服务。

要不要我现在把这些文件合并成一个可下载的 ZIP 包，或把某些文件展开成更详细的实现（例如：GA 完整实现、前端 dashboard 示例）？
