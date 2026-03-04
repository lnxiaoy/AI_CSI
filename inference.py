import torch
import numpy as np
import matplotlib.pyplot as plt

# 🔥 关键修改 1：从 model_advanced 导入最新的 2D 模型
from src.model_2DConv import iCsiNet2D
from src.utils import load_temporal_data
from scipy.signal import savgol_filter

def infer():
    print("🚀 加载数据与模型...")
    # 加载数据# 在 inference.py 中修改：
    data, mean, std = load_temporal_data("./data/csi_tensor_fast.pt", seq_len=14)
    
    # 转换统计量
    mean = mean.item()
    std = std.item()
    
    # 取测试集的一个样本 (倒数第10个)
    test_idx = -10 
    input_seq = data[test_idx:test_idx+1] 
    
    # 🔥 关键修改 2：实例化最新的 iCsiNet2D 模型
    # 默认参数已经完美适配了我们刚才训练的结构
    model = iCsiNet2D()
    
    # 加载神仙权重
    model.load_state_dict(torch.load("csi_lstm_model.pth"))
    model.eval()
    
    print("🚀 模型加载成功，正在使用 iCsiNet-2D 推理...")
    with torch.no_grad():
        output_seq = model(input_seq)
        print("🚀 模型加载成功，正在使用 iCsiNet-2D 推理...")
    with torch.no_grad():
        output_seq = model(input_seq)
        
        # ==========================================
        # 🔥 新增：计算物理波束对齐度 (SGCS)
        # ==========================================
        B, T, C, F = output_seq.shape
        # 将 128 维拆分为 2 (实虚部) * 64 (天线)
        p = output_seq.view(B, T, 2, 64, F)
        t = input_seq.view(B, T, 2, 64, F)
        
        p_real, p_imag = p[:, :, 0, :, :], p[:, :, 1, :, :]
        t_real, t_imag = t[:, :, 0, :, :], t[:, :, 1, :, :]
        
        inner_real = t_real * p_real + t_imag * p_imag
        inner_imag = t_real * p_imag - t_imag * p_real
        
        sum_inner_real = torch.sum(inner_real, dim=3)
        sum_inner_imag = torch.sum(inner_imag, dim=3)
        numerator = torch.sqrt(sum_inner_real**2 + sum_inner_imag**2 + 1e-8)
        
        norm_t = torch.sqrt(torch.sum(t_real**2 + t_imag**2, dim=3) + 1e-8)
        norm_p = torch.sqrt(torch.sum(p_real**2 + p_imag**2, dim=3) + 1e-8)
        
        sgcs_score = torch.mean(numerator / (norm_t * norm_p)).item()
        print(f"🏆 终极物理指标 SGCS (广义余弦相似度): {sgcs_score:.4f} (越接近 1 越完美)")
        # ==========================================

    # ==========================================
        # ⚔️ 暴打传统非 AI 方案 (Baseline 碰撞测试)
        # ==========================================
        # 传统方案：没有预测能力，基站只能用上一帧 (T-1) 的旧信道当作当前帧 (T) 的信道
        # 我们用第 13 帧 (真实信道) 作为目标，传统方案只能给出第 12 帧 (过期信道)
        
        # t_real 和 t_imag 是真实的信道 [Batch, Time, Ant=64, Freq=72]
        # 目标时刻 (Target): T = 13 (最后一帧)
        target_real_T13 = t_real[:, 13, :, :]
        target_imag_T13 = t_imag[:, 13, :, :]
        
        # 传统非 AI 方案 (Outdated): T = 12 (上一帧)
        baseline_real_T12 = t_real[:, 12, :, :]
        baseline_imag_T12 = t_imag[:, 12, :, :]
        
        # 计算传统方案的波束对齐度 (Baseline SGCS)
        base_inner_real = target_real_T13 * baseline_real_T12 + target_imag_T13 * baseline_imag_T12
        base_inner_imag = target_real_T13 * baseline_imag_T12 - target_imag_T13 * baseline_real_T12
        
        # 沿天线维度 (dim=1) 求和
        base_sum_real = torch.sum(base_inner_real, dim=1) 
        base_sum_imag = torch.sum(base_inner_imag, dim=1)
        base_numerator = torch.sqrt(base_sum_real**2 + base_sum_imag**2 + 1e-8)
        
        # 分别计算范数
        norm_target_T13 = torch.sqrt(torch.sum(target_real_T13**2 + target_imag_T13**2, dim=1) + 1e-8)
        norm_baseline_T12 = torch.sqrt(torch.sum(baseline_real_T12**2 + baseline_imag_T12**2, dim=1) + 1e-8)
        
        # 计算传统方案的 SGCS
        baseline_sgcs_score = torch.mean(base_numerator / (norm_target_T13 * norm_baseline_T12)).item()
        
        print(f"💀 传统非 AI 方案 SGCS (过期 CSI): {baseline_sgcs_score:.4f}")
        print(f"🔥 AI 方案带来的波束对齐绝对增益: {(sgcs_score - baseline_sgcs_score) * 100:.2f}%")
        # ==========================================
        
    # === 数据后处理 ===
    input_np = input_seq.detach().cpu().numpy()
    output_np = output_seq.detach().cpu().numpy()
    
    # 反归一化
    gt_seq = input_np * std + mean
    pred_seq = output_np * std + mean
    
    # === 画图：展示第 0, 7, 13 帧的变化 ===
    frames_to_show = [0, 7, 13]
    
    plt.figure(figsize=(15, 5))
    
    for i, frame_idx in enumerate(frames_to_show):
        # 提取当前帧 (Rx0, Tx0) 
        gt_real = gt_seq[0, frame_idx, 0, :]
        gt_imag = gt_seq[0, frame_idx, 1, :]
        gt_mag = np.sqrt(gt_real**2 + gt_imag**2)
        
        pred_real = pred_seq[0, frame_idx, 0, :]
        pred_imag = pred_seq[0, frame_idx, 1, :]
        pred_mag = np.sqrt(pred_real**2 + pred_imag**2)
        
        # 滤波平滑
        pred_smooth = savgol_filter(pred_mag, 11, 3)
        
        # 自动 Bias 校准
        bias = np.mean(pred_smooth) - np.mean(gt_mag)
        pred_final = pred_smooth - bias
        
        plt.subplot(1, 3, i+1)
        plt.plot(gt_mag, 'b-', label='Ground Truth', alpha=0.6, linewidth=2)
        plt.plot(pred_final, 'g--', label='iCsiNet-2D', linewidth=2)
        plt.title(f"Time Step {frame_idx}\n(Bias Corrected: {bias:.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("temporal_result_2d_transformer.png")
    print("✅ 验证完成！图片已保存为 temporal_result_2d_transformer.png")
    plt.show()

if __name__ == "__main__":
    infer()
