import random
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import torch.nn.functional as F

HOLIDAYS = [
    datetime(2025, 5, 1),
    datetime(2025, 5, 2),
    datetime(2025, 5, 3),
    datetime(2025, 5, 4),
    datetime(2025, 5, 5),
    datetime(2025,5,31),
    datetime(2025, 6, 1),
    datetime(2025, 6, 2),
]  # 节假日示例

START_DATE = datetime(2025, 5, 1)

class Version:
    def __init__(self, vid, pilot_batch, batch_size, users, size_gb, start_time, end_time, period, traffic_pattern):
        self.vid = vid
        self.pilot_batch = pilot_batch
        self.batch_size = batch_size
        self.users = users
        self.size_gb = size_gb
        self.start_time = start_time
        self.end_time = end_time
        self.period = period
        self.traffic_pattern = traffic_pattern  # k线

    def calculate_traffic(self, release_dates, proportions):
        traffic = np.zeros(31+15)  # 存储每日流量（5月1日=索引0）
        # 计算差距
        for i, date in enumerate(release_dates):
            # 处理跨月流量影响
            start_offset = max((START_DATE - date).days, 0)
            for j in range(start_offset, 15):
                current_day = date + timedelta(days=j)
                delta_days = (current_day - START_DATE).days
                # 计算相对于5月1日的天数差
                if delta_days >= 0:
                    # 动态扩展流量数组
                    if delta_days >= len(traffic):
                        padding_length = delta_days - len(traffic) + 1
                        traffic = np.pad(traffic, (0, padding_length), mode='constant')
                    traffic[delta_days] += (
                            self.users * proportions[i]
                            * self.size_gb
                            * self.traffic_pattern[j]
                    )
        return traffic

    @classmethod
    def from_csv_row(cls, row):
        """从CSV行创建Version对象"""
        return cls(
            vid=int(row['vid']),
            batch_size=int(row['batch_size']),
            users=int(float(row['users'])),
            size_gb=float(row['size_gb']),
            start_time=datetime.strptime(row['start_time'], "%Y-%m-%d"),
            end_time=datetime.strptime(row['end_time'], "%Y-%m-%d"),
            period=int(row['period']),
            traffic_pattern=list(map(float, row['traffic_pattern'].split(',')))
        )

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # 共享的特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化权重
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        shared_features = self.shared_net(x)
        action_probs = F.softmax(self.actor(shared_features), dim=-1)
        state_values = self.critic(shared_features)
        return action_probs, state_values

class PPOOptimizer:
    def __init__(self, versions, state_dim=100, action_dim=10, lr=3e-4, 
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, 
                 ppo_epochs=4, batch_size=64):
        self.versions = versions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.history = []
        
    def get_state_representation(self, individual):
        """将当前解表示为状态向量"""
        total_traffic = self.calculate_total_traffic(individual)
        
        # 标准化流量数据
        if len(total_traffic) > self.state_dim:
            total_traffic = total_traffic[:self.state_dim]
        else:
            total_traffic = np.pad(total_traffic, (0, self.state_dim - len(total_traffic)), 'constant')
        
        # 添加统计特征
        mean = np.mean(total_traffic)
        std = np.std(total_traffic)
        max_val = np.max(total_traffic)
        min_val = np.min(total_traffic)
        
        # 组合状态向量
        state = np.concatenate([
            total_traffic / (max_val + 1e-8),  # 归一化流量
            [mean, std, max_val, min_val]  # 统计特征
        ])
        
        # 确保状态向量长度一致
        if len(state) > self.state_dim:
            state = state[:self.state_dim]
        else:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
            
        return state
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, state_value = self.policy(state)
            
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = []
        advantages = []
        last_advantage = 0
        
        # 计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # 终止状态的值为0
            else:
                next_value = values[t+1] * (1 - dones[t])
                
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantage = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
            last_advantage = advantage
            advantages.insert(0, advantage)
            
            returns.insert(0, advantage + values[t])
            
        return np.array(returns), np.array(advantages)
    
    def update_policy(self):
        if len(self.states) < self.batch_size:
            return
            
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(self.compute_returns_and_advantages(
            self.rewards, self.values, self.dones)[0])).to(self.device)
        advantages = torch.FloatTensor(np.array(self.compute_returns_and_advantages(
            self.rewards, self.values, self.dones)[1])).to(self.device)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 计算新策略
                action_probs, state_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 策略比率
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 策略损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(state_values.squeeze(), batch_returns)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 更新策略
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def fitness(self, individual):
        """适应度函数，考虑多种分布特性"""
        # 计算总流量分布
        total_traffic = self.calculate_total_traffic(individual)

        # 构建评估指标体系
        mean = np.mean(total_traffic)
        deviations = total_traffic - mean

        # 核心改进：多维度评估分布质量
        metrics = {
            'std': np.std(total_traffic),  # 标准差,每个数据点与均值的差的平方
            'mad': np.mean(np.abs(deviations)),  # 平均绝对偏差
            'max_dev': np.max(np.abs(deviations)),  # 最大绝对偏差
            'overflow': np.sum(np.where(deviations > 0, deviations ** 2, 0)),  # 正向偏离惩罚
            'underflow': np.sum(np.where(deviations < 0, deviations ** 2, 0)),  # 负向偏离惩罚
            'volatility': np.mean(np.abs(total_traffic[1:] - total_traffic[:-1]))  # 相邻时段变化率约束
        }

        # 组合权重（可调整）
        return (
                0.5 * metrics['mad'] +
                0.5 * metrics['volatility']
        )
    
    def generate_valid_proportions(self, batch_size):
        """生成满足业务约束的发布比例"""
        while True:
            # 初始比例<=1%
            p1 = min(random.uniform(0, 0.011), 0.01)
            remaining = 1 - p1
            max_other = remaining / (batch_size - 1)
            # 剩余比例 均值 ± 10%
            other_props = [random.uniform(max_other * 0.9, max_other * 1.1) for _ in range(batch_size - 1)]
            # 归一化处理
            total_other = sum(other_props)
            if total_other == 0:
                continue
            scale = remaining / total_other
            other_props = [p * scale for p in other_props]
            # 最终校验
            if all(p <= max_other * 1.1 for p in other_props) and abs(sum([p1] + other_props) - 1) < 1e-6:
                return [p1] + other_props

    def generate_valid_dates(self, version, batch_size):
        """生成合法发布日期（避开周末和节假日）"""
        adjusted_start, adjusted_end = self.get_adjusted_dates(version)
        legal_dates = []
        current_end = adjusted_end

        # 生成初始日期范围内的合法日期
        current_date = adjusted_start
        while current_date <= current_end:
            if current_date.weekday() < 5 and current_date not in HOLIDAYS:
                legal_dates.append(current_date)
            current_date += timedelta(days=1)

        # 逐步延长结束日期直到合法日期足够
        extension = 5  # 每次延长7天，然后逐步延长1天看是否满足诉求
        count = 0
        while count < 1 and len(legal_dates) < batch_size:
            new_end = current_end + timedelta(days=extension)
            # 检查新增的日期段
            current_date = current_end + timedelta(days=1)
            while current_date <= new_end:
                if current_date.weekday() < 5 and current_date not in HOLIDAYS:
                    legal_dates.append(current_date)
                current_date += timedelta(days=1)
            current_end = new_end
            count += count
        # 随机选择并排序
        # selected = random.sample(legal_dates, k=batch_size)
        selected = legal_dates[:batch_size]  # 顺序选取前N个元素
        return sorted(selected)
    
    def get_adjusted_dates(self, version):
        """获取调整后的日期范围（供变异使用）"""
        # 调整开始日期（最多推迟3天）
        start_candidates = [version.start_time + timedelta(days=i) for i in range(4)]
        valid_starts = [d for d in start_candidates if d.weekday() < 5 and d not in HOLIDAYS]
        adjusted_start = valid_starts[0] if valid_starts else version.start_time

        # 调整结束日期（最多推迟5天）
        end_candidates = [version.end_time + timedelta(days=i) for i in range(8)]
        valid_ends = [d for d in end_candidates if d.weekday() < 5 and d not in HOLIDAYS]
        adjusted_end = valid_ends[-1] if valid_ends else version.end_time
        return adjusted_start, adjusted_end
    
    def calculate_total_traffic(self, individual):
        """计算个体5月往后的总流量"""
        end_date = max([max(dates) for idx, (_, dates, _) in enumerate(individual)])
        traffic_len = (end_date - START_DATE).days + 15
        total_traffic = np.zeros(traffic_len)
        for i, (batch_size, dates, props) in enumerate(individual):
            traffic = self.versions[i].calculate_traffic(dates, props)
            if len(traffic) < len(total_traffic):
                traffic = np.pad(traffic, (0, len(total_traffic) - len(traffic)), mode='constant')
            elif len(total_traffic) < len(traffic):
                total_traffic = np.pad(total_traffic,(0,len(traffic) - len(total_traffic) ), mode='constant')
            total_traffic += traffic
        return total_traffic
    
    def initialize_individual(self):
        """初始化个体，生成满足约束的个体"""
        individual = []
        for version in self.versions:
            batch_size = version.batch_size
            release_dates = self.generate_valid_dates(version, batch_size)
            proportions = self.generate_valid_proportions(batch_size)
            individual.append((batch_size, release_dates, proportions))
        return individual
    
    def mutate_individual(self, individual):
        """对个体进行变异"""
        mutated = []
        for idx in range(len(individual)):
            version = self.versions[idx]
            batch_size, dates, props = individual[idx]
            
            mutation_type = random.choices(
                ['adjust_dates', 'adjust_batch', 'adjust_props'],
                weights=[0.2, 0.4, 0.4], k=1)[0]

            if mutation_type == 'adjust_dates':
                new_dates = self.generate_valid_dates(version, batch_size)
                new_props = self.generate_valid_proportions(batch_size)
                mutated.append((batch_size, new_dates, new_props))

            elif mutation_type == 'adjust_batch':
                new_batch = self.adjust_batch_size(batch_size)
                new_dates = self.generate_valid_dates(version, new_batch)
                new_props = self.generate_valid_proportions(new_batch)
                mutated.append((new_batch, new_dates, new_props))

            elif mutation_type == 'adjust_props':
                new_props = self.generate_valid_proportions(batch_size)
                mutated.append((batch_size, dates.copy(), new_props))
        return mutated
    
    def adjust_batch_size(self, current_size):
        adjustment = random.randint(-2, 2)
        new_size = current_size + adjustment
        return max(5, min(new_size, 10))
    
    def optimize(self, episodes=1000):
        best_fitness = float('inf')
        best_individual = self.initialize_individual()
        
        for episode in range(episodes):
            # 生成当前个体
            if episode == 0:
                current_individual = best_individual
            else:
                current_individual = self.mutate_individual(best_individual)
            
            # 获取状态表示
            state = self.get_state_representation(current_individual)
            
            # 选择动作
            action, log_prob, value = self.select_action(state)
            
            # 执行动作（这里动作主要用于选择变异类型）
            # 在实际应用中，动作可以更精细地控制变异过程
            mutated_individual = self.mutate_individual(current_individual)
            
            # 计算奖励（负的适应度，因为我们要最小化适应度）
            fitness_value = self.fitness(mutated_individual)
            reward = -fitness_value
            
            # 存储转换
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.values.append(value)
            self.dones.append(0)  # 非终止状态
            
            # 更新策略
            self.update_policy()
            
            # 更新最佳个体
            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_individual = mutated_individual
                
            # 记录历史
            if episode % 10 == 0:
                self.history.append(self.calculate_total_traffic(best_individual))
                print(f"Episode {episode + 1}: Fitness={fitness_value:.2f}")
        
        # 保存结果
        self.visualize_optimization_process()
        self.save_to_csv(best_individual)
        print(f"优化方案已保存到 optimized_schedule.csv")
        return best_individual
    
    def visualize_optimization_process(self):
        """可视化整个优化过程"""
        fig, ax = plt.subplots(figsize=(16, 9))
        cmap = plt.get_cmap('viridis')
        for idx, traffic in enumerate(self.history):
            alpha = 0.2 + 0.6 * (idx / len(self.history))
            color = cmap(idx / len(self.history))
            ax.plot(traffic, color=color, alpha=alpha, lw=0.8)  # 修改x轴数据为日期

        ax.plot(self.history[0], color='black', linestyle='--', lw=1.5, label='Initial')
        ax.plot(self.history[-1], color='red', lw=2.5, label='Optimized')

        final_mean = np.mean(self.history[-1][0:31])
        ax.axhline(final_mean, color='blue', linestyle=':', lw=2, label='Final Mean')
        ax.fill_between(range(31),
                        final_mean * 0.95,
                        final_mean * 1.05,
                        color='gray', alpha=0.1,
                        label='±10% Range')

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(0, len(self.history)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Episode Progress')

        ax.set_title("Optimization Process Visualization")
        ax.set_xlabel("Day of Month")
        ax.set_ylabel("Traffic Volume (GB)")
        ax.legend(loc='upper right')
        ax.grid(True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'ppo_optimization_{timestamp}_result.svg', dpi=300)
        plt.show()

    def save_to_csv(self, individual, filename="optimized_schedule_detail.csv"):
        """增强版CSV保存功能"""
        # 生成日期范围（5月1日-31日）
        end_date = max([max(dates) for idx, (_, dates, _) in enumerate(individual)])
        date_len = (end_date - START_DATE).days + 15
        date_range = [datetime(2025, 5, 1) + timedelta(days=i) for i in range(date_len)]
        date_headers = [d.strftime("%m/%d") for d in date_range]

        with open(filename, 'w', newline='',encoding='utf-8-sig') as csvfile:
            # 定义CSV结构
            writer = csv.writer(csvfile)

            # 第一部分：总体统计
            writer.writerow(["【方案摘要】"])
            writer.writerow([
                "生成时间", datetime.now().strftime("%Y-%m-%d %H:%M"),
                "版本数量", len(individual),
                "总流量(GB)", np.sum(self.calculate_total_traffic(individual)),
                "流量标准差", np.std(self.calculate_total_traffic(individual))
            ])

            # 第二部分：每日总流量
            writer.writerow([])
            writer.writerow(["【每日总流量(GB)】"] + date_headers)
            total_traffic = self.calculate_total_traffic(individual)
            writer.writerow(["Total"] + [f"{v:.2f}" for v in total_traffic])

            # 第三部分：各版本详细信息
            writer.writerow([])
            writer.writerow(["【版本详情】"])
            headers = [
                          "VersionID", "Batch", "ReleaseDate", "Proportion",
                          "Users", "SizeGB", "StartDate", "EndDate"
                      ] + date_headers
            writer.writerow(headers)

            for vid, (batch_size, dates, props) in enumerate(individual):
                version = self.versions[vid]
                traffic_matrix = np.zeros((batch_size, date_len))

                for batch_idx in range(batch_size):
                    release_date = dates[batch_idx]
                    batch_props = props[batch_idx]

                    # 计算日期偏移量
                    day_offset = (release_date - START_DATE).days
                    traffic = version.calculate_traffic([release_date], [batch_props])
                    aligned_traffic = np.zeros(date_len)

                    # 新流量对齐逻辑
                    if day_offset < 0:
                        # 处理发布日期早于5月1日的情况
                        valid_length = min(len(traffic), date_len)
                        aligned_traffic[0:valid_length] = traffic[0:valid_length]
                    else:
                        valid_length = min(len(traffic) - day_offset, date_len - day_offset)  # 新增约束条件
                        valid_length = max(valid_length, 0)  # 保证非负
                        if valid_length > 0:
                            aligned_traffic[day_offset:day_offset + valid_length] = traffic[
                                                                                    day_offset:day_offset + valid_length]  # 精确切片
                        print(f"{release_date}_{day_offset}_{valid_length}_{traffic}")

                    # 写入行数据
                    row = [
                              vid + 1,
                              batch_idx + 1,
                              release_date.strftime("%Y-%m-%d"),
                              f"{batch_props:.6f}",
                              f"{version.users:,}",
                              f"{version.size_gb:.1f}GB",
                              version.start_time.strftime("%Y-%m-%d"),
                              version.end_time.strftime("%Y-%m-%d")
                          ] + [f"{v:.2f}" for v in aligned_traffic]  # 使用对齐后的流量数据

                    writer.writerow(row)
                    traffic_matrix[batch_idx] = aligned_traffic  # 存储对齐后的数据

                    # 版本汇总行
                writer.writerow([
                                    f"Version {vid + 1} Total", "", "", "", "", "", "", ""
                                ] + [f"{np.sum(traffic_matrix[:, d]):.2f}" for d in range(date_len)])

            # 第四部分：统计指标
            writer.writerow([])
            writer.writerow(["【统计指标】"])
            writer.writerow([
                "指标", "值", "出现日期", "说明"
            ])
            stats = {
                "最高流量日": (np.max(total_traffic), np.argmax(total_traffic)),
                "最低流量日": (np.min(total_traffic), np.argmin(total_traffic)),
                "平均流量": (np.mean(total_traffic), -1),
                "超过均值天数": (np.sum(total_traffic > np.mean(total_traffic)), -1)
            }
            for k, (v, d) in stats.items():
                writer.writerow([
                    k,
                    f"{v:.2f}GB",
                    f"5月{d + 1}" if d != -1 else "N/A",
                    self._get_stat_description(k)  # 添加说明文本
                ])

                # 新增版本日期范围校验
            for vid, (batch_size, dates, props) in enumerate(individual):
                version = self.versions[vid]
                adjusted_start, adjusted_end = self.get_adjusted_dates(version)
                for d in dates:
                    if not (adjusted_start <= d <= adjusted_end):
                        print(f"警告：版本{vid + 1}发布日期{d}超出调整后的日期范围[{adjusted_start}-{adjusted_end}]")

    def _get_stat_description(self, key):
        """生成统计指标的解读说明"""
        desc = {
            "最高流量日": "建议检查该日期是否存在多个版本叠加发布",
            "最低流量日": "可能需要调整该日期附近的发布计划",
            "平均流量": "理想状态应保持每日流量在此值±10%范围内",
            "超过均值天数": "值越小表示流量分布越均衡"
        }
        return desc.get(key, "")

    @classmethod
    def load_from_csv(cls, filename):
        versions = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换CSV行数据为对象
                versions.append(Version.from_csv_row(row))
        return cls(versions)

def generate_test_versions(num_versions=5, include_cross_month=True, seed=None):
    """
    生成多版本测试数据生成器

    参数：
    num_versions: 生成版本数量
    include_cross_month: 是否包含跨月版本
    seed: 随机种子

    返回：Version对象列表
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    versions = []

    # 基础参数范围
    param_ranges = {
        'users': (8e5, 1.2e8),  # 用户数量范围
        'size_gb': (0.1, 5.0),  # 包大小范围
        'period': (7, 14),  # 发布周期范围
        'batch_size': (5, 11)  # 批次数范围
    }

    # 流量模式模板
    traffic_templates = [
        lambda: [
                    random.uniform(0.15, 0.25),  # 第1位为0.20±0.05
                    random.uniform(0.40, 0.50),  # 第二位为0.45 ±0.05
                    random.uniform(0.155, 0.165)  # 第三位为0.16 ±0.005
                ] + [0.16 * (0.6 ** i) for i in range(12)],  # 剩下依次衰减
        # 新的生成逻辑2：前几位数值较大，后续数值逐渐衰减，但衰减速度较慢
        lambda: [
                    random.uniform(0.15, 0.25),  # 第1位为0.20±0.05
                    random.uniform(0.40, 0.50),  # 第二位为0.45 ±0.05
                    random.uniform(0.155, 0.165)  # 第三位为0.16 ±0.005
                ] + [0.16 * (0.8 ** i) for i in range(12)],  # 剩下依次衰减
        # 新的生成逻辑3：前几位数值较大，后续数值逐渐衰减，但衰减速度较快
        lambda: [
                    random.uniform(0.15, 0.25),  # 第1位为0.20±0.05
                    random.uniform(0.40, 0.50),  # 第二位为0.45 ±0.05
                    random.uniform(0.155, 0.165)  # 第三位为0.16 ±0.005
                ] + [0.16 * (0.4 ** i) for i in range(12)]  # 剩下依次衰减
    ]

    for vid in range(num_versions):
        # 随机生成基础参数
        params = {
            'vid': vid,
            'batch_size': random.randint(*param_ranges['batch_size']),
            'users': int(random.uniform(*param_ranges['users'])),
            'size_gb': round(random.uniform(*param_ranges['size_gb']), 1),
            'period': random.randint(*param_ranges['period']),
            'pilot_batch':random.choices(
                population=[None, 1, 2],
                weights=[0.1, 0.45, 0.45],
                k=1
            )[0]
        }

        # 生成流量模式（标准化）
        template = random.choice(traffic_templates)
        traffic_pattern = template()
        traffic_pattern = [p / sum(traffic_pattern) for p in traffic_pattern]

        # 生成时间范围
        if include_cross_month and vid < 1:  # 前5个版本作为跨月版本
            start = datetime(2025, 4, 15) + timedelta(days=random.randint(0, 15))
            end = start + timedelta(days=random.randint(7, 14))
        else:
            start = datetime(2025, 5, 1) + timedelta(days=random.randint(0, 31))
            end = start + timedelta(days=random.randint(7, params['period'] + 5))

        # 调整合法开始日期
        while start.weekday() >= 5 or start in HOLIDAYS:
            start += timedelta(days=1)

            # 新增月份校验
        if start.month > START_DATE.month:
            continue

        versions.append(
            Version(
                vid=params['vid'],
                batch_size=params['batch_size'],
                users=params['users'],
                size_gb=params['size_gb'],
                start_time=start,
                end_time=end,
                period=params['period'],
                traffic_pattern=traffic_pattern,
                pilot_batch=params['pilot_batch']  # 新增字段注入
            )
        )

    versions.sort(key=lambda x: x.start_time)
    return versions

def save_versions_to_csv(versions, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'VID', 'Users', 'Size_GB', 'Batch_Size',
            'Start_Date', 'End_Date', 'Traffic_Pattern'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 新增排序逻辑：按 Start_Date 升序排列
        sorted_versions = sorted(versions, key=lambda x: x.start_time)

        for v in sorted_versions:  # 修改遍历对象为排序后的列表
            traffic_str = ",".join([f"{round(p, 3):.3f}" for p in v.traffic_pattern])

            row = {
                'VID': v.vid,
                'Users': v.users,
                'Size_GB': v.size_gb,
                'Batch_Size': v.batch_size,
                'Start_Date': v.start_time.strftime('%Y-%m-%d'),
                'End_Date': v.end_time.strftime('%Y-%m-%d'),
                'Traffic_Pattern': traffic_str
            }
            writer.writerow(row)

# 测试数据（包含跨月版本）
if __name__ == "__main__":
    # 生成可重复的测试数据
    test_versions = generate_test_versions(
        num_versions=2,
        include_cross_month=False,
        seed=42  # 固定种子保证可重复性
    )

    # 调用示例
    save_versions_to_csv(test_versions, 'test_versions.csv')
    print("\n文件已保存至 test_versions.csv")

    start_time = datetime.now()
    optimizer = PPOOptimizer(
        test_versions,
        state_dim=100,
        action_dim=10
    )
    best = optimizer.optimize(episodes=1000)
    end_time = datetime.now()
    print(f"total time: f{end_time - start_time}")
    
    # 输出方案细节
    print("\nOptimized Plan:")
    for vid, (batch_size, dates, props) in enumerate(best):
        print(f"\nVersion {vid}:")
        print(f"Batch size: {batch_size}")
        print(f"Release Dates: {[d.strftime('%m/%d') for d in dates]}")
        print(f"Proportions: {[round(p, 4) for p in props]}")
    print("\nNote: The traffic visualization shows the combined effect of all versions")