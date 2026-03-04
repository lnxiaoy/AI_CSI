import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_csi_heatmap():
    # 我们要对比的三个数据集
    files = {
        "CDL-A (3 km/h) - Slow Fading": "./data/csi_tensor_r19.pt",
        "CDL-A (120 km/h) - Fast Fading": "./data/csi_tensor_fast.pt",
        "CDL-D (40 km/h) - LOS": "./data/csi_tensor_cdld.pt"
    }
    
    plt.figure(figsize=(18, 5))
    
    for i, (title, path) in enumerate(files.items()):
        if not os.path.exists(path):
            print(f"⚠️ 找不到文件 {path}，跳过画图。")
            continue
            
        # 1. 加载数据 [Batch, Time=14, Chan=128, Freq=72]
        data = torch.load(path)
        
        # 2. 取第 0 个样本 (Batch=0)
        # 我们当时 Flatten 的顺序是 (Rx, Tx, Real/Imag)
        # 所以 Chan=0 是实部，Chan=1 是虚部
        real_part = data[0, :, 0, :].numpy() # 形状: (14, 72)
        imag_part = data[0, :, 1, :].numpy() # 形状: (14, 72)
        
        # 3. 计算幅度 (Magnitude)
        magnitude = np.sqrt(real_part**2 + imag_part**2)
        
        # 4. 画热力图
        plt.subplot(1, 3, i+1)
        # aspect='auto' 让像素填满画布，origin='lower' 让 Time=0 在最下面
        im = plt.imshow(magnitude, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(im, fraction=0.046, pad=0.04).set_label('Amplitude')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Subcarriers (Frequency)", fontsize=12)
        plt.ylabel("Time Steps (10ms per step)", fontsize=12)
        
    plt.tight_layout()
    save_path = "csi_time_variance_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ 画图成功！图片已保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_csi_heatmap()
