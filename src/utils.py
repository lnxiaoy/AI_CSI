import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def load_temporal_data(data_path, seq_len=14):
    """
    加载时序 CSI 数据集 (已适配最新的 4D 直接输入格式)
    """
    print(f"📥 正在加载数据: {data_path} ...")
    
    # 1. 直接读取数据
    data = torch.load(data_path)
    
    # 2. 兼容性检查：如果已经是我们最新生成的 4D 格式 (Batch, 14, 128, 72)
    if data.dim() == 4:
        # 直接使用，不需要任何恶心的维度置换了！
        pass
    else:
        # 如果你还不小心读到了最老版本的 6D 数据，做个备用兼容
        print("⚠️ 检测到旧版 6D 格式数据，正在进行向后兼容的维度展平...")
        b, t, rx, rx_ant, tx, tx_ant, f = data.shape
        data = data.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        data = data.view(b, t, 128, f)
    
    # 3. 数据标准化 (Zero-Mean, Unit-Variance) - 这一步极其关键！
    mean = data.mean()
    std = data.std()
    data = (data - mean) / (std + 1e-8)
    
    print(f"✅ 数据加载完毕: {data.shape}")
    print(f"   (Batch={data.shape[0]}, Time={data.shape[1]}, Chan={data.shape[2]}, Freq={data.shape[3]})")
    print(f"   统计量: Mean={mean:.4f}, Std={std:.4f}")
    
    return data, mean, std

# 如果 utils.py 里还有其他的 dataset 类 (比如 CsiDataset)，保持它们不变即可。

def get_dataloader(data, batch_size=32, split_ratio=0.9):
    N = data.shape[0]
    train_size = int(N * split_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 注意：输入和标签都是 data (自编码器)
    loader = DataLoader(TensorDataset(train_data, train_data), 
                        batch_size=batch_size, 
                        shuffle=True)
    return loader, test_data
