import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os

# 🔥 修复点：直接从具体的文件中导入，不管 __init__.py 是什么状态都不会报错了
#from src.model import CsiNetLSTM
from src.model_2DConv import iCsiNet2D
from src.utils import load_temporal_data, get_dataloader

# ================= 配置区域 =================
DATA_PATH = "./data/csi_tensor_mixed.pt"
BATCH_SIZE = 32          
EPOCHS = 60              
LR = 0.002               
HIDDEN_DIM = 256         # Medium 版
LSTM_LAYERS = 2          
# ===========================================
class SGCS_MSE_Loss(nn.Module):
    def __init__(self, mse_weight=1.0, sgcs_weight=0.5):
        super(SGCS_MSE_Loss, self).__init__()
        self.mse_weight = mse_weight
        self.sgcs_weight = sgcs_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 1. 基础 MSE (保证基础重构不崩溃)
        loss_mse = self.mse(pred, target)
        
        # 2. 计算 SGCS (广义余弦相似度)
        # 假设 dim=2 是通道(实部=0, 虚部=1), dim=3 是天线空间/频域
        # 我们需要计算复数向量的点积
        pred_real, pred_imag = pred[:, :, 0, :], pred[:, :, 1, :]
        target_real, target_imag = target[:, :, 0, :], target[:, :, 1, :]
        
        # 计算分子：内积的绝对值 |h^H * h_hat|
        # 复数内积: (a+bi)(c-di) 的实部和虚部
        inner_real = target_real * pred_real + target_imag * pred_imag
        inner_imag = target_imag * pred_real - target_real * pred_imag
        
        # 沿着特征维度求和
        sum_inner_real = torch.sum(inner_real, dim=-1)
        sum_inner_imag = torch.sum(inner_imag, dim=-1)
        numerator = torch.sqrt(sum_inner_real**2 + sum_inner_imag**2 + 1e-8)
        
        # 计算分母：范数 ||h|| * ||h_hat||
        norm_target = torch.sqrt(torch.sum(target_real**2 + target_imag**2, dim=-1) + 1e-8)
        norm_pred = torch.sqrt(torch.sum(pred_real**2 + pred_imag**2, dim=-1) + 1e-8)
        denominator = norm_target * norm_pred
        
        # SGCS 值 (越接近1越好)
        sgcs = numerator / denominator
        
        # 3. 损失函数 (我们希望最大化 SGCS，所以最小化 1 - SGCS)
        loss_sgcs = torch.mean(1.0 - sgcs)
        
        return self.mse_weight * loss_mse + self.sgcs_weight * loss_sgcs
# 🔥 救场核心：平滑损失函数，逼迫模型学习形状！ 
class SmoothMSELoss(nn.Module):
    def __init__(self, smooth_weight=0.02):
        super(SmoothMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = smooth_weight

    def forward(self, pred, target):
        # 1. 基础的均方误差
        loss_mse = self.mse(pred, target)
        
        # 2. 计算频域上的斜率/差值 (沿着最后一个维度 Freq)
        # 逼迫模型去匹配子载波之间的变化趋势，打破平线诅咒
        diff_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        diff_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        loss_smooth = self.mse(diff_pred, diff_target)
        
        return loss_mse + self.alpha * loss_smooth
# 🔥 终极黑科技：融合了 MSE、斜率平滑 和 3GPP SGCS 的联合损失函数
class Physical_CSI_Loss(nn.Module):
    def __init__(self, smooth_weight=0.02, sgcs_weight=0.1):
        super(Physical_CSI_Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.smooth_weight = smooth_weight
        self.sgcs_weight = sgcs_weight

    def forward(self, pred, target):
        # 1. 基础 MSE
        loss_mse = self.mse(pred, target)
        
        # 2. 频域斜率惩罚 (打破平线诅咒)
        diff_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        diff_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        loss_smooth = self.mse(diff_pred, diff_target)
        
        # 3. 计算 SGCS (广义余弦相似度)
        B, T, C, F = pred.shape
        # 将 C=128 拆分为实部和虚部 (假设维度布局为 2 * 64)
        p = pred.view(B, T, 2, 64, F)
        t = target.view(B, T, 2, 64, F)
        
        p_real, p_imag = p[:, :, 0, :, :], p[:, :, 1, :, :]
        t_real, t_imag = t[:, :, 0, :, :], t[:, :, 1, :, :]
        
        # 计算复数内积的实部和虚部
        inner_real = t_real * p_real + t_imag * p_imag
        inner_imag = t_real * p_imag - t_imag * p_real
        
        # 沿天线维度 (dim=3) 求和
        sum_inner_real = torch.sum(inner_real, dim=3)
        sum_inner_imag = torch.sum(inner_imag, dim=3)
        
        # 分子：内积的绝对值 |h^H \hat{h}|
        numerator = torch.sqrt(sum_inner_real**2 + sum_inner_imag**2 + 1e-8)
        
        # 分母：范数 ||h|| * ||\hat{h}||
        norm_t = torch.sqrt(torch.sum(t_real**2 + t_imag**2, dim=3) + 1e-8)
        norm_p = torch.sqrt(torch.sum(p_real**2 + p_imag**2, dim=3) + 1e-8)
        
        # 计算 SGCS，并求整个 Batch 的平均值
        sgcs = numerator / (norm_t * norm_p)
        mean_sgcs = torch.mean(sgcs)
        
        # SGCS 越接近 1 越好，所以我们最小化 (1 - SGCS)
        loss_sgcs = 1.0 - mean_sgcs
        
        return loss_mse + self.smooth_weight * loss_smooth + self.sgcs_weight * loss_sgcs
    
def train():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到数据文件 {DATA_PATH}")
        return

    print("📥 [Step 1] 加载时序数据...")
    data, mean, std = load_temporal_data(DATA_PATH, seq_len=14)
    train_loader, test_data = get_dataloader(data, batch_size=BATCH_SIZE)
    
    print(f"🏗️ [Step 2] 初始化 Medium CsiNet-LSTM 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = CsiNetLSTM(input_channels=128, seq_len=72, hidden_dim=HIDDEN_DIM, lstm_layers=LSTM_LAYERS).to(device)
    model = iCsiNet2D().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 替换掉旧的 SmoothMSELoss
    criterion = Physical_CSI_Loss(smooth_weight=0.05, sgcs_weight=0.1)
    
    print("🚀 [Step 3] 开始强力训练 (带斜率惩罚)...")
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")
        
    print("✅ 训练完成！")
    torch.save(model.state_dict(), "csi_lstm_model.pth")
    print("💾 模型已保存")

if __name__ == "__main__":
    train()
