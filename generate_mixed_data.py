import sys
import os
import numpy as np
import torch
from unittest.mock import MagicMock

# 屏蔽光线追踪
sys.modules['sionna.rt'] = MagicMock()
sys.modules['sionna.rt.scene'] = MagicMock()
sys.modules['mitsuba'] = MagicMock()
sys.modules['drjit'] = MagicMock()

import tensorflow as tf
import sionna
try:
    from sionna.channel.tr38901 import CDL, AntennaArray
except:
    import sionna.channel.tr38901 as tr38901
    CDL = tr38901.CDL
    AntennaArray = tr38901.AntennaArray
from sionna.channel import cir_to_ofdm_channel

# ================= 基础配置 =================
BATCH_SIZE = 128          
NUM_BATCHES = 100         
CARRIER_FREQ = 3.5e9      
NUM_TX_ANT = 32           
NUM_RX_ANT = 2            
SUBCARRIER_SPACING = 30e3 
FFT_SIZE = 72             
DELAY_SPREAD = 300e-9     

def generate_3gpp_dataset(cdl_model="C", magic_speed_kmh=3.0, save_name="csi_tensor"):
    # 🧙‍♂️ 魔法缩放后的超音速 (转化为 m/s)
    speed_ms = magic_speed_kmh / 3.6  
    
    tx_array = sionna.channel.tr38901.PanelArray(
        num_rows_per_panel=4, num_cols_per_panel=4,
        polarization="dual", polarization_type="cross",
        antenna_pattern="38.901", carrier_frequency=CARRIER_FREQ
    )
    rx_array = sionna.channel.tr38901.PanelArray(
        num_rows_per_panel=1, num_cols_per_panel=1,
        polarization="dual", polarization_type="cross",
        antenna_pattern="38.901", carrier_frequency=CARRIER_FREQ
    )

    print(f"\n🚀 正在生成: {save_name}.pt")
    print(f"📊 场景: CDL-{cdl_model} | 缩放等效速度: 约 {magic_speed_kmh} km/h")
    
    cdl_channel = sionna.channel.tr38901.CDL(
        model=cdl_model,
        delay_spread=DELAY_SPREAD,
        carrier_frequency=CARRIER_FREQ,
        ut_array=rx_array,
        bs_array=tx_array,
        direction="downlink",
        min_speed=speed_ms,
        max_speed=speed_ms
    )

    num_time_steps = 14 
    bandwidth = FFT_SIZE * SUBCARRIER_SPACING
    frequencies = tf.range(FFT_SIZE, dtype=tf.float32) * SUBCARRIER_SPACING
    frequencies = frequencies - (bandwidth / 2.0)

    all_h_freq = []

    for i in range(NUM_BATCHES):
        # 🔥 恢复 Sionna 安全配置：采样率必须等于物理带宽
        a, tau = cdl_channel(
            batch_size=BATCH_SIZE, 
            num_time_steps=num_time_steps, 
            sampling_frequency=bandwidth 
        )
        
        h_freq = cir_to_ofdm_channel(frequencies=frequencies, a=a, tau=tau, normalize=True)
        all_h_freq.append(h_freq.numpy())
        if (i+1) % 10 == 0:
            print(f"进度: {i+1}/{NUM_BATCHES} 批次")

    # === 极度省内存的矩阵转换 (Zero-Copy) ===
    total_samples = BATCH_SIZE * NUM_BATCHES
    h_final = np.zeros((total_samples, 14, 128, 72), dtype=np.float32)
    
    for i, h_batch in enumerate(all_h_freq):
        h_squeezed = np.squeeze(h_batch) 
        h_transposed = np.transpose(h_squeezed, (0, 3, 1, 2, 4))
        h_ri = np.stack([np.real(h_transposed), np.imag(h_transposed)], axis=-2)
        h_flat = h_ri.reshape(BATCH_SIZE, 14, 128, 72)
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        h_final[start_idx:end_idx] = h_flat

    pt_tensor = torch.from_numpy(h_final)
    os.makedirs("./data", exist_ok=True)
    save_path = f"./data/{save_name}.pt"
    torch.save(pt_tensor, save_path)
    print(f"✅ 成功保存至 {save_path}")

    del all_h_freq, h_final, pt_tensor
    import gc
    gc.collect()

if __name__ == "__main__":
    # 1. 模拟慢速步行 (轻微的时间波动) -> 放大了 3000 倍
    generate_3gpp_dataset(cdl_model="A", magic_speed_kmh=10000.0, save_name="csi_tensor_r19")
    
    # 2. 模拟极限高铁 (极其剧烈的时间斑马纹) -> 放大了 2500 倍
    generate_3gpp_dataset(cdl_model="A", magic_speed_kmh=300000.0, save_name="csi_tensor_fast")
    
    # 3. 模拟视距传输 (平滑的高能量直射径)
    generate_3gpp_dataset(cdl_model="D", magic_speed_kmh=30000.0, save_name="csi_tensor_cdld")
