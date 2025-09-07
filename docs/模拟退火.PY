import random
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import csv
from collections import deque
import heapq

HOLIDAYS = [
    datetime(2025, 5, 1), datetime(2025, 5, 2), datetime(2025, 5, 3),
    datetime(2025, 5, 4), datetime(2025, 5, 5), datetime(2025, 5, 31),
    datetime(2025, 6, 1), datetime(2025, 6, 2),
]

START_DATE = datetime(2025, 5, 1)

class Version:
    # 保持原有Version类不变
    def __init__(self, vid, pilot_batch, batch_size, users, size_gb, start_time, end_time, period, traffic_pattern):
        self.vid = vid
        self.pilot_batch = pilot_batch
        self.batch_size = batch_size
        self.users = users
        self.size_gb = size_gb
        self.start_time = start_time
        self.end_time = end_time
        self.period = period
        self.traffic_pattern = traffic_pattern

    def calculate_traffic(self, release_dates, proportions):
        # 保持原有流量计算方法不变
        traffic = np.zeros(31 + 15)
        for i, date in enumerate(release_dates):
            start_offset = max((START_DATE - date).days, 0)
            for j in range(start_offset, 15):
                current_day = date + timedelta(days=j)
                delta_days = (current_day - START_DATE).days
                if delta_days >= 0:
                    if delta_days >= len(traffic):
                        padding_length = delta_days - len(traffic) + 1
                        traffic = np.pad(traffic, (0, padding_length), mode='constant')
                    traffic[delta_days] += (
                        self.users * proportions[i] * self.size_gb * self.traffic_pattern[j]
                    )
        return traffic

class LightweightOptimizer:
    def __init__(self, versions, max_iterations=500, initial_temp=100, cooling_rate=0.95):
        self.versions = versions
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.history = []
        
    def initialize_individual(self):
        """初始化个体，生成满足约束的个体"""
        individual = []
        for version in self.versions:
            batch_size = version.batch_size
            release_dates = self.generate_valid_dates(version, batch_size)
            proportions = self.generate_valid_proportions(batch_size)
            individual.append((batch_size, release_dates, proportions))
        return individual
    
    def generate_valid_proportions(self, batch_size):
        """生成满足业务约束的发布比例"""
        while True:
            p1 = min(random.uniform(0, 0.011), 0.01)
            remaining = 1 - p1
            max_other = remaining / (batch_size - 1)
            other_props = [random.uniform(max_other * 0.9, max_other * 1.1) for _ in range(batch_size - 1)]
            total_other = sum(other_props)
            if total_other == 0:
                continue
            scale = remaining / total_other
            other_props = [p * scale for p in other_props]
            if all(p <= max_other * 1.1 for p in other_props) and abs(sum([p1] + other_props) - 1) < 1e-6:
                return [p1] + other_props
    
    def generate_valid_dates(self, version, batch_size):
        """生成合法发布日期（避开周末和节假日）"""
        adjusted_start, adjusted_end = self.get_adjusted_dates(version)
        legal_dates = []
        current_date = adjusted_start
        
        # 收集所有合法日期
        while current_date <= adjusted_end:
            if current_date.weekday() < 5 and current_date not in HOLIDAYS:
                legal_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # 如果合法日期不足，延长结束日期
        extension = 0
        while len(legal_dates) < batch_size and extension < 30:  # 最多延长30天
            extension += 1
            new_date = adjusted_end + timedelta(days=extension)
            if new_date.weekday() < 5 and new_date not in HOLIDAYS:
                legal_dates.append(new_date)
        
        # 选择前batch_size个日期
        return sorted(legal_dates[:batch_size])
    
    def get_adjusted_dates(self, version):
        """获取调整后的日期范围"""
        start_candidates = [version.start_time + timedelta(days=i) for i in range(4)]
        valid_starts = [d for d in start_candidates if d.weekday() < 5 and d not in HOLIDAYS]
        adjusted_start = valid_starts[0] if valid_starts else version.start_time

        end_candidates = [version.end_time + timedelta(days=i) for i in range(8)]
        valid_ends = [d for d in end_candidates if d.weekday() < 5 and d not in HOLIDAYS]
        adjusted_end = valid_ends[-1] if valid_ends else version.end_time
        
        return adjusted_start, adjusted_end
    
    def fitness(self, individual):
        """适应度函数，评估流量分布的平稳性"""
        total_traffic = self.calculate_total_traffic(individual)
        
        # 计算关键指标
        mean_traffic = np.mean(total_traffic)
        deviations = np.abs(total_traffic - mean_traffic)
        
        # 使用更轻量的评估指标
        mad = np.mean(deviations)  # 平均绝对偏差
        volatility = np.mean(np.abs(np.diff(total_traffic)))  # 波动性
        
        # 组合权重
        return 0.6 * mad + 0.4 * volatility
    
    def calculate_total_traffic(self, individual):
        """计算总流量"""
        end_date = max([max(dates) for _, dates, _ in individual])
        traffic_len = (end_date - START_DATE).days + 15
        total_traffic = np.zeros(traffic_len)
        
        for i, (batch_size, dates, props) in enumerate(individual):
            traffic = self.versions[i].calculate_traffic(dates, props)
            if len(traffic) < len(total_traffic):
                traffic = np.pad(traffic, (0, len(total_traffic) - len(traffic)), mode='constant')
            elif len(total_traffic) < len(traffic):
                total_traffic = np.pad(total_traffic, (0, len(traffic) - len(total_traffic)), mode='constant')
            total_traffic += traffic
            
        return total_traffic
    
    def mutate(self, individual):
        """对个体进行变异"""
        mutated = []
        for idx, (batch_size, dates, props) in enumerate(individual):
            version = self.versions[idx]
            
            # 随机选择变异类型
            mutation_type = random.choice(['adjust_dates', 'adjust_props'])
            
            if mutation_type == 'adjust_dates':
                # 日期变异：随机调整一个批次的日期
                new_dates = dates.copy()
                if new_dates:
                    idx_to_change = random.randint(0, len(new_dates) - 1)
                    # 在当前日期前后3天内寻找合法日期
                    current_date = new_dates[idx_to_change]
                    candidates = []
                    for delta in range(-3, 4):
                        if delta == 0:
                            continue
                        candidate = current_date + timedelta(days=delta)
                        if candidate.weekday() < 5 and candidate not in HOLIDAYS:
                            candidates.append(candidate)
                    
                    if candidates:
                        new_dates[idx_to_change] = random.choice(candidates)
                        new_dates.sort()
                
                mutated.append((batch_size, new_dates, props))
                
            elif mutation_type == 'adjust_props':
                # 比例变异：小幅调整比例
                new_props = self.generate_valid_proportions(batch_size)
                mutated.append((batch_size, dates, new_props))
                
        return mutated
    
    def optimize(self):
        """使用模拟退火算法进行优化"""
        # 初始化
        current_solution = self.initialize_individual()
        current_fitness = self.fitness(current_solution)
        best_solution = current_solution
        best_fitness = current_fitness
        
        temperature = self.initial_temp
        self.history = [self.calculate_total_traffic(current_solution)]
        
        # 主循环
        for iteration in range(self.max_iterations):
            # 生成新解
            new_solution = self.mutate(current_solution)
            new_fitness = self.fitness(new_solution)
            
            # 计算接受概率
            delta_fitness = new_fitness - current_fitness
            acceptance_prob = np.exp(-delta_fitness / temperature) if temperature > 0 else 0
            
            # 决定是否接受新解
            if delta_fitness < 0 or random.random() < acceptance_prob:
                current_solution = new_solution
                current_fitness = new_fitness
                
                # 更新最佳解
                if current_fitness < best_fitness:
                    best_solution = current_solution
                    best_fitness = current_fitness
            
            # 降低温度
            temperature *= self.cooling_rate
            
            # 记录历史
            if iteration % 50 == 0:
                self.history.append(self.calculate_total_traffic(best_solution))
                print(f"Iteration {iteration}: Fitness={best_fitness:.4f}, Temp={temperature:.2f}")
        
        # 保存结果
        self.visualize_optimization_process()
        self.save_to_csv(best_solution)
        
        return best_solution
    
    def visualize_optimization_process(self):
        """可视化优化过程"""
        plt.figure(figsize=(12, 6))
        
        for idx, traffic in enumerate(self.history):
            alpha = 0.2 + 0.6 * (idx / len(self.history))
            plt.plot(traffic[:31], alpha=alpha, color='blue')
        
        plt.plot(self.history[0][:31], 'k--', label='Initial')
        plt.plot(self.history[-1][:31], 'r-', label='Optimized')
        
        final_mean = np.mean(self.history[-1][:31])
        plt.axhline(final_mean, color='green', linestyle=':', label='Mean')
        
        plt.title("Optimization Process")
        plt.xlabel("Day of Month")
        plt.ylabel("Traffic Volume (GB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('lightweight_optimization.png', dpi=150)
        plt.show()
    
    def save_to_csv(self, individual, filename="lightweight_schedule.csv"):
        """保存优化结果到CSV"""
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入基本信息
            writer.writerow(["Optimized Release Schedule"])
            writer.writerow(["Generated at", datetime.now().strftime("%Y-%m-%d %H:%M")])
            writer.writerow([])
            
            # 写入每个版本的发布计划
            writer.writerow(["Version", "Batch", "Release Date", "Proportion"])
            for vid, (batch_size, dates, props) in enumerate(individual):
                for i in range(batch_size):
                    writer.writerow([
                        f"V{vid+1}",
                        f"B{i+1}",
                        dates[i].strftime("%Y-%m-%d"),
                        f"{props[i]:.4f}"
                    ])
                writer.writerow([])  # 空行分隔不同版本
            
            # 写入流量统计
            total_traffic = self.calculate_total_traffic(individual)
            writer.writerow(["Daily Traffic Summary"])
            writer.writerow(["Date", "Traffic (GB)"])
            for i in range(min(31, len(total_traffic))):
                date = START_DATE + timedelta(days=i)
                writer.writerow([date.strftime("%Y-%m-%d"), f"{total_traffic[i]:.2f}"])

# 辅助函数：生成测试版本数据
def generate_test_versions(num_versions=3, seed=42):
    """生成测试版本数据"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    versions = []
    traffic_templates = [
        lambda: [0.2, 0.45, 0.16] + [0.16 * (0.6 ** i) for i in range(12)],
        lambda: [0.2, 0.45, 0.16] + [0.16 * (0.8 ** i) for i in range(12)],
        lambda: [0.2, 0.45, 0.16] + [0.16 * (0.4 ** i) for i in range(12)]
    ]
    
    for vid in range(num_versions):
        # 随机参数
        batch_size = random.randint(5, 8)
        users = int(random.uniform(1e6, 1e7))
        size_gb = round(random.uniform(0.5, 2.0), 1)
        
        # 时间范围（5月内）
        start_day = random.randint(1, 10)
        end_day = start_day + random.randint(7, 14)
        start_time = datetime(2025, 5, start_day)
        end_time = datetime(2025, 5, end_day)
        
        # 流量模式
        template = random.choice(traffic_templates)
        traffic_pattern = template()
        traffic_pattern = [p / sum(traffic_pattern) for p in traffic_pattern]
        
        versions.append(
            Version(
                vid=vid,
                pilot_batch=None,
                batch_size=batch_size,
                users=users,
                size_gb=size_gb,
                start_time=start_time,
                end_time=end_time,
                period=random.randint(7, 14),
                traffic_pattern=traffic_pattern
            )
        )
    
    return versions

# 使用示例
if __name__ == "__main__":
    # 生成测试数据
    test_versions = generate_test_versions(num_versions=3, seed=42)
    
    # 创建优化器并运行
    optimizer = LightweightOptimizer(
        test_versions,
        max_iterations=300,  # 减少迭代次数
        initial_temp=50,     # 降低初始温度
        cooling_rate=0.97    # 调整冷却速率
    )
    
    # 运行优化
    start_time = datetime.now()
    best_solution = optimizer.optimize()
    end_time = datetime.now()
    
    print(f"优化完成，耗时: {end_time - start_time}")
    print("\n最佳方案:")
    
    for vid, (batch_size, dates, props) in enumerate(best_solution):
        print(f"\n版本 {vid+1}:")
        print(f"  批次数量: {batch_size}")
        print(f"  发布日期: {[d.strftime('%m/%d') for d in dates]}")
        print(f"  发布比例: {[f'{p:.3f}' for p in props]}")