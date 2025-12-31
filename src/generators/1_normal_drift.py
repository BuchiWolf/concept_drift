import os
import csv
import random
# 修正 1: 调整导入路径
from river import datasets
from river.datasets import synth
from river import stream

# 修正 2: 使用原始字符串 (r"") 避免路径转义错误
OUTPUT_FOLDER = r"D:\Projects\concept_drift\data"
FILE_NAME = "security_logs.csv"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 修正 3: River 的 ConceptDriftStream 要求输入是其定义的合成数据集格式
# 这里我们将你的逻辑包装成符合 river 接口的生成器
class UEBAGenerator(datasets.base.SyntheticDataset):
    def __init__(self, mode="normal"):
        # 显式声明特征数量和任务类型（二分类）
        super().__init__(n_features=4, n_samples=None, n_classes=2, task=datasets.base.BINARY_CLF)
        self.mode = mode
        
    def __iter__(self):
        while True:
            if self.mode == "normal":
                features = {
                    "hour": random.randint(9, 18),
                    "upload_mb": random.uniform(0.5, 50.0),
                    "failed_logins": random.randint(0, 1),
                    "is_vpn": 0
                }
                label = 0
            else:
                features = {
                    "hour": random.randint(0, 5),
                    "upload_mb": random.uniform(500.0, 2000.0),
                    "failed_logins": random.randint(5, 20),
                    "is_vpn": 1
                }
                label = 1
            yield features, label

# 实例化基础流
stream_a = UEBAGenerator(mode="normal")
stream_b = UEBAGenerator(mode="malicious")

# 修正 4: 使用正确路径下的 ConceptDriftStream
drift_stream = synth.ConceptDriftStream(
    stream=stream_a,
    drift_stream=stream_b,
    position=2000,
    width=1000,
    seed=42
)

csv_path = os.path.join(OUTPUT_FOLDER, FILE_NAME)
n_samples = 5000

print(f"正在生成 UEBA 漂移数据集并保存至: {csv_path}...")

with open(csv_path, 'w', newline='') as f:
    writer = None
    # 使用 drift_stream.take() 获取样本
    for i, (x, y) in enumerate(drift_stream.take(n_samples)):
        if i == 0:
            header = list(x.keys()) + ['label']
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
        
        row = x.copy()
        row['label'] = y
        writer.writerow(row)

print("生成完成！")