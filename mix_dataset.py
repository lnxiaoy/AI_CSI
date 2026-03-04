import torch
import os

def create_mixed_dataset():
    print("🌪️ 正在构建 3GPP 混合极端泛化数据集...")
    
    # 🔥 核心修改：把我们刚才生成的三大数据集全加上！
    files_to_mix = [
        "./data/csi_tensor_r19.pt",    # 场景 A: CDL-A, 3km/h (经典多径步行)
        "./data/csi_tensor_fast.pt",   # 场景 B: CDL-A, 120km/h (极限多普勒频移)
        "./data/csi_tensor_cdld.pt"    # 场景 C: CDL-D, 40km/h (视距传输 LOS)
    ]
    
    data_list = []
    
    # 动态加载所有存在的文件
    for path in files_to_mix:
        if not os.path.exists(path):
            print(f"⚠️ 警告: 找不到数据文件 {path}，跳过此文件。")
            continue
            
        print(f"📥 正在加载 {path}...")
        data = torch.load(path)
        print(f"   ✅ 加载成功，形状: {data.shape}")
        data_list.append(data)
    
    if len(data_list) == 0:
        print("❌ 错误：没有找到任何数据，请检查 data 目录！")
        return
        
    # 1. 暴力拼接！(沿着 Batch 维度合并所有数据)
    print("🧱 正在将所有场景数据拼接在一起...")
    mixed_data = torch.cat(data_list, dim=0)
    
    # 2. 打乱顺序 (非常关键！防止模型先学 A 再学 B 产生灾难性遗忘)
    print("🔀 正在进行全局洗牌 (Shuffling)...")
    indices = torch.randperm(mixed_data.size(0))
    mixed_data = mixed_data[indices]
    
    # 3. 保存终极数据集
    save_path = "./data/csi_tensor_mixed.pt"
    torch.save(mixed_data, save_path)
    
    print(f"🎉 终极混合泛化数据集已生成！")
    print(f"💾 保存至: {save_path} | Final Shape: {mixed_data.shape}")
    print(f"📈 现在的总样本量达到了: {mixed_data.size(0)} 个！")

if __name__ == "__main__":
    create_mixed_dataset()
