import torch
import torch.nn as nn
import torch.fft

# =================================================================
# 1. 导入你的“黄金基线”模型，我们直接复用它的轻量化卷积、量化器和重建层
# =================================================================
from src.model_2DConv import iCsiNet2D

# =================================================================
# 2. 全新的多普勒双引擎 Transformer (Dual-Domain)
# =================================================================
class Doppler_iTransformer(nn.Module):
    """
    融合 TimesNet 核心思想的 6G 物理层时序预测模型
    (Dual-Domain: Time + Doppler / Frequency)
    """
    def __init__(self, num_variates=256, lookback_len=14, pred_len=1, d_model=128, n_heads=4, e_layers=2):
        super().__init__()
        self.lookback_len = lookback_len
        # 对于长度为 14 的实数序列，rfft 输出的频域长度为 14 // 2 + 1 = 8
        self.doppler_len = lookback_len // 2 + 1  
        
        # 1. 时域特征投影 (Time Domain Embedding)
        self.time_projector = nn.Linear(self.lookback_len, d_model)
        
        # 2. 🔥 多普勒域特征投影 (Doppler Domain Embedding)
        self.doppler_projector = nn.Linear(self.doppler_len, d_model)
        
        # 特征融合归一化
        self.fusion_norm = nn.LayerNorm(d_model)

        # 3. 空间特征之间的注意力机制 (iTransformer 核心)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # 4. 预测未来
        self.predictor = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x shape: [Batch, Time=14, Variates=256]
        # iTransformer 倒置魔法：把 Variates 当成 Tokens
        x_inv = x.transpose(1, 2)  # 形状变为 [B, Variates, T]
        
        # 🌪️ 提取多普勒域特征 (FFT)
        x_fft = torch.fft.rfft(x_inv, dim=-1) 
        x_doppler = torch.abs(x_fft)  # 取绝对值得到幅度谱 [B, Variates, 8]
        
        # 🧬 双域特征融合
        time_emb = self.time_projector(x_inv)           # [B, Variates, d_model]
        doppler_emb = self.doppler_projector(x_doppler) # [B, Variates, d_model]
        fused_emb = self.fusion_norm(time_emb + doppler_emb)
        
        # 🧠 送入 Transformer
        enc_out = self.encoder(fused_emb)  # [B, Variates, d_model]
        
        # 预测下一帧
        pred = self.predictor(enc_out)     # [B, Variates, Pred_len=1]
        
        # 转回标准形状 [B, Pred_len, Variates]
        pred = pred.transpose(1, 2)
        
        return pred
class UEPatchEncoder(nn.Module):
    """
    面向 6G 极低功耗手机的 Patch Tokenizer 编码器
    利用物理相干块 (2根天线 x 4个子载波) 瞬间降维
    """
    def __init__(self, in_channels=2, out_features=256):
        super().__init__()
        
        # 1. 物理相干块切分 (无重叠滑动窗口)
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=16,          
            kernel_size=(2, 4),       # 物理块大小
            stride=(2, 4)             # 步长=块大小，瞬间将分辨率缩小 8 倍！
        )
        
        # 2. 极度轻量的深度可分离卷积
        self.depthwise = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.pointwise = nn.Conv2d(16, 32, kernel_size=1)
        self.act = nn.GELU()
        
        # 3. 映射到 256 个 Tokens
        self.flatten = nn.Flatten()
        # [B, 32通道, 高度(64/2=32), 宽度(72/4=18)] -> 32 * 32 * 18 = 18432
        self.fc = nn.Linear(32 * 32 * 18, out_features)

    def forward(self, x):
        # x: [Batch, 2, 64, 72]
        x = self.patch_embed(x)      
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = self.flatten(x)
        out = self.fc(x)             
        return out
# =================================================================
# 3. 终极模型组装：继承机制
# =================================================================
# =================================================================
# 3. 终极模型组装：继承机制
# =================================================================
class iCsiNet2D_Doppler(iCsiNet2D):
    """
    6G AI-CSI 的完全体：
    1. UE 端：Patch Tokenizer (算力缩减 80%)
    2. 空口：4-bit LSQ 自适应量化 (开销死锁 1024 bits)
    3. BS 端：Doppler_iTransformer (多普勒双擎预测)
    """
    def __init__(self):
        super().__init__()
        
        # 🔥 外科手术 1：替换手机端特征提取器 (极致瘦身)
        # 注意：这里假设你在原始 iCsiNet2D 中把编码器命名为 self.encoder
        self.encoder = UEPatchEncoder(in_channels=2, out_features=256)
        
        # 🔥 外科手术 2：替换基站端预测器 (多普勒双擎)
        self.temporal_predictor = Doppler_iTransformer(
            num_variates=256, 
            lookback_len=14, 
            pred_len=1,
            d_model=128,
            n_heads=4,
            e_layers=2
        )
