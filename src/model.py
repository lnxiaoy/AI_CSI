import torch
import torch.nn as nn

class CsiNetLSTM(nn.Module):
    def __init__(self, input_channels=128, seq_len=72, hidden_dim=256, lstm_layers=2):
        super(CsiNetLSTM, self).__init__()
        
        # ==========================================
        # 1. 空间编码器 (适度压缩)
        # ==========================================
        self.cnn_encoder = nn.Sequential(
            # Layer 1: 保持长度 72
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            
            # Layer 2: 只做一次下采样 (72 -> 36)
            # 这样能保留斜率信息
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
        )
        
        # 计算特征大小: 32通道 * 36长度 = 1152
        # 虽然比之前的 576 大了一倍，但比 4608 小很多，CPU 还能扛得住
        self.feature_size = 32 * 36 
        
        # ==========================================
        # 2. 时域编码器 (LSTM)
        # ==========================================
        self.lstm_encoder = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_dim, # 256
            num_layers=lstm_layers,
            batch_first=True
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=self.feature_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # ==========================================
        # 3. 空间解码器
        # ==========================================
        self.cnn_decoder = nn.Sequential(
            # Layer 1: 上采样 36 -> 72
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            
            # Layer 2: 输出
            nn.Conv1d(64, input_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, T, C, F = x.shape
        x_reshaped = x.view(B * T, C, F) 
        
        # CNN
        spatial_feat = self.cnn_encoder(x_reshaped)
        spatial_feat = spatial_feat.view(B, T, -1) # (B, T, 1152)
        
        # LSTM
        temporal_feat, _ = self.lstm_encoder(spatial_feat)
        rec_feat, _ = self.lstm_decoder(temporal_feat)
        
        # CNN Decode
        rec_feat = rec_feat.contiguous().view(B * T, 32, 36)
        out = self.cnn_decoder(rec_feat)
        out = out.view(B, T, C, F)
        
        return out
