import torch
import torch.nn as nn
# ==========================================
# 🚀 新增：深度可分离卷积模块 (极大降低手机端 FLOPs)
# ==========================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        # Depthwise: 逐通道空间卷积 (参数量极少)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   padding=1, stride=stride, groups=in_channels)
        # Pointwise: 1x1 跨通道融合
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.gelu(x)
    
# ==========================================
# 🔥 核心黑科技：STE 4-bit 量化层
# ==========================================
class QuantizeLSQ(nn.Module):
    def __init__(self, num_bits=4):
        super(QuantizeLSQ, self).__init__()
        self.num_bits = num_bits
        self.q_levels = (1 << num_bits) - 1 # 15
        
        # 🔥 核心优化：让量化的缩放因子（步长）变成可学习的参数！
        # 初始值设为 1.0，模型会根据 Loss 自己去收缩或放大它
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 确保 scale 是正数
        scale = torch.abs(self.scale) + 1e-5 
        
        # 1. 动态缩放输入
        x_scaled = x / scale
        
        # 2. 截断到 [-1, 1] 之间 (剔除长尾效应，保护主体分布)
        x_clamped = torch.clamp(x_scaled, -1.0, 1.0)
        
        # 3. 映射到 [0, 1] 并量化
        x_norm = (x_clamped + 1.0) / 2.0
        x_quant = torch.round(x_norm * self.q_levels)
        
        # 4. 反量化回原始尺度
        x_dequant = (x_quant / self.q_levels) * 2.0 - 1.0
        x_dequant = x_dequant * scale
        
        # 5. STE 魔法
        x_out = x + (x_dequant - x).detach()
        return x_out


class iCsiNet2D(nn.Module):
    def __init__(self, seq_len=14, freq_len=72, num_tokens=256, d_model=64, nhead=4, num_layers=2):
        super(iCsiNet2D, self).__init__()
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        
        # ==========================================
        # 1. 手机端：超轻量级 2D 编码器
        # ==========================================
        self.encoder_2d = nn.Sequential(
            # 第一层保持普通卷积 (因为输入只有2个通道，没必要分离)
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            
            # 后面全部替换为深度可分离卷积！FLOPs 骤降！
            DepthwiseSeparableConv2d(16, 32, stride=2),
            DepthwiseSeparableConv2d(32, 64, stride=2)
        )
        
        # 压缩到 Latent Space，并用 Tanh 压制到 [-1, 1] 方便量化
        self.fc_enc = nn.Sequential(
            nn.Linear(4608, num_tokens),
            nn.Tanh() # 🔥 必须加上 Tanh，否则量化会越界
        )
        
        # 🔥 加入 4-bit 量化层 (模拟空中传输)
        self.quantization = QuantizeLSQ(num_bits=4)
        
        # ==========================================
        # 2. 基站端：iTransformer 倒置注意力 (抗噪与时序预测)
        # ==========================================
        self.time_projector = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.time_deprojector = nn.Linear(d_model, seq_len)
        
        # ==========================================
        # 3. 基站端：2D 解码器
        # ==========================================
        self.fc_dec = nn.Linear(num_tokens, 4608)
        
        self.decoder_2d = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        B, T, C, F = x.shape
        x_img = x.view(B * T, 2, 64, F)
        
        # --- 手机端处理 ---
        spatial_feat = self.encoder_2d(x_img).view(B * T, -1)
        tokens_continuous = self.fc_enc(spatial_feat) 
        
        # 🔥 空中接口：量化 (连续浮点 -> 离散 4-bit)
        tokens_quantized = self.quantization(tokens_continuous)
        
        tokens = tokens_quantized.view(B, T, self.num_tokens) 
        
        # --- 基站端处理 ---
        tokens_inv = tokens.transpose(1, 2) 
        tokens_emb = self.time_projector(tokens_inv) 
        tokens_out = self.transformer(tokens_emb) 
        tokens_rec = self.time_deprojector(tokens_out) 
        rec_feat = tokens_rec.transpose(1, 2).contiguous().view(B * T, self.num_tokens)
        
        rec_img = self.fc_dec(rec_feat).view(B * T, 64, 8, 9)
        out_img = self.decoder_2d(rec_img) 
        
        out = out_img.view(B, T, 128, F)
        return out
