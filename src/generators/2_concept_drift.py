import os
import csv
import random
from river.datasets import synth
from river import datasets

# 1. 目录结构配置
DATA_DIR = "data/synthetic"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. 修复后的生成器类：继承自 synth.SyntheticDataset
class BenignGenerator(datasets.base.SyntheticDataset):
    def __init__(self, mode="early_stage"):
        # 指定特征数量和目标值数量
        super().__init__(n_features=4, n_samples=None, n_classes=0)
        self.mode = mode
        
    def __iter__(self):
        while True:
            if self.mode == "early_stage":
                # 概念 A: 传统行政办公
                features = {
                    "hour": random.randint(9, 17),
                    "data_transfer_mb": random.uniform(10, 100),
                    "remote_access": 0,
                    "privilege_usage": random.randint(0, 2) 
                }
            else:
                # 概念 B: 演进后的技术支持/研发
                features = {
                    "hour": random.randint(7, 21),
                    "data_transfer_mb": random.uniform(200, 800),
                    "remote_access": 1,
                    "privilege_usage": random.randint(5, 15)
                }
            yield features, 0

# 3. 执行生成任务
print("正在生成良性演进数据集...")

# 初始化流
stream_a = BenignGenerator(mode="early_stage")
stream_b = BenignGenerator(mode="evolved_stage")

# 现在 ConceptDriftStream 可以正确识别 n_features 了
drift_stream = synth.ConceptDriftStream(
    stream=stream_a,
    drift_stream=stream_b,
    position=1500,
    width=1000,
    seed=42
)

# 4. 保存函数
def save_to_csv(stream_iter, filename, n_samples):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = None
        for i, (x, y) in enumerate(stream_iter):
            if i >= n_samples: break
            if i == 0:
                header = list(x.keys()) + ['label']
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
            row = x.copy()
            row['label'] = y
            writer.writerow(row)
    print(f"文件已保存: {filepath}")

# 生成训练集（无漂移）
save_to_csv(iter(BenignGenerator(mode="early_stage")), "train_stable.csv", n_samples=2000)

# 生成测试集（包含渐变漂移）
save_to_csv(drift_stream, "test_with_drift.csv", n_samples=4000)

print("生成完成！")