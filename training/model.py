import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Gaussians:
    def __init__(self, means, scales, rotations, opacities, harmonics):
        self.means = means
        self.scales = scales
        self.rotations = rotations
        self.opacities = opacities
        self.harmonics = harmonics

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """
    支持 Conditioning 的 Transformer Block (参考 DiT/FiT 架构)
    使用 Adaptive Layer Norm (adaLN) 注入几何与时间条件
    """
    def __init__(self, hidden_dim, cond_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
        
        # adaLN modulation: 预测 shift, scale, gate (针对 Attention 和 MLP 各一套)
        # 输入是 cond_dim, 输出是 6 * hidden_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim, bias=True)
        )
        
        # Zero-init for gates (这对训练稳定性很重要)
        with torch.no_grad():
            self.adaLN_modulation[1].weight.zero_()
            self.adaLN_modulation[1].bias.zero_()

    def forward(self, x, c):
        # x: [B, N, D] (Main Stream - Visual)
        # c: [B, N, D_cond] (Condition Stream - Concerto + Time)
        
        # 1. 预测调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # 2. Modulated Self-Attention
        # Norm -> Modulate -> Attn -> Gate -> Residual
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa * attn_out
        
        # 3. Modulated MLP
        # Norm -> Modulate -> MLP -> Gate -> Residual
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm2)
        
        return x

class FreeTimeGSModel(nn.Module):
    def __init__(self, input_dim=1024, visual_dim=1536, hidden_dim=256, num_layers=4, nhead=4):
        """
        Args:
            input_dim: Concerto token dimension (Condition)
            visual_dim: DA3 feature dimension (Main Input)
        """
        super().__init__()
        
        # === 1. Embedding Layers ===
        # 主干流 (Visual) 映射
        self.visual_embed = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 条件流 (Concerto) 映射
        self.concerto_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 时间编码
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === 2. DiT Backbone ===
        # 注意: cond_dim = hidden_dim (Concerto) + hidden_dim (Time)
        # 如果你希望 Concerto 和 Time 先融合，也可以改成 hidden_dim
        self.cond_fusion = nn.Sequential(
             nn.Linear(hidden_dim * 2, hidden_dim),
             nn.SiLU(),
             nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, cond_dim=hidden_dim, num_heads=nhead)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # === 3. Heads (Predicting Deltas) ===
        # Static Base (从 Visual + Concerto 的初始状态预测)
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 输入是 cat(Visual, Concerto)
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 + 3 + 4 + 1 + 3) # xyz, scale, rot, opac, sh0
        )
        
        # Dynamic Head (从 DiT 输出预测)
        self.dynamic_head = nn.Linear(hidden_dim, 3 + 3 + 4 + 1)
        # Zero-init dynamic head
        nn.init.zeros_(self.dynamic_head.weight)
        nn.init.zeros_(self.dynamic_head.bias)

    def forward(self, concerto_tokens, visual_tokens, coords, t):
        # 1. Embeddings
        x_vis = self.visual_embed(visual_tokens) # [B, N, H] - Main Stream
        x_geo = self.concerto_embed(concerto_tokens) # [B, N, H]
        x_time = self.time_embed(t).unsqueeze(1) # [B, 1, H]
        
        # 2. Prepare Condition
        # 将时间广播到每个点，与几何特征拼接
        # x_geo: [B, N, H], x_time: [B, 1, H] -> [B, N, 2H] or fused to [B, N, H]
        # 这里我们先融合 Time 和 Geometry 作为一个统一的 Condition
        cond = torch.cat([x_geo, x_time.expand(-1, x_geo.shape[1], -1)], dim=-1)
        cond = self.cond_fusion(cond) # [B, N, H]
        
        # 3. Base Prediction (Static)
        # 静态属性由 原始几何 + 原始外观 共同决定，不经过 DiT，作为基准
        base_input = torch.cat([x_vis, x_geo], dim=-1) # [B, N, 2H]
        base_params = self.base_head(base_input)
        
        base_xyz = coords + torch.tanh(base_params[..., :3]) * 0.1 
        base_scale = base_params[..., 3:6]
        base_rot = F.normalize(base_params[..., 6:10], dim=-1)
        base_opac = base_params[..., 10]
        base_sh = base_params[..., 11:].unsqueeze(-1)
        
        # 4. DiT Forward (Dynamic)
        # x_vis 是 token 序列，cond 是几何+时间引导
        x = x_vis
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)
        
        # 5. Dynamic Deltas
        deltas = self.dynamic_head(x)
        
        # === 6. Final Assembly ===
        final_xyz = base_xyz + deltas[..., :3]
        
        # Scale Init Fix (-2.0 bias)
        raw_scale = base_scale + deltas[..., 3:6] - 2.0 
        scale_min, scale_max = 0.005, 0.15 
        final_scale = scale_min + (scale_max - scale_min) * torch.sigmoid(raw_scale)
        
        final_rot = F.normalize(base_rot + deltas[..., 6:10], dim=-1)
        
        raw_opac = base_opac + deltas[..., 10] + 1.0
        final_opac = torch.sigmoid(raw_opac)
        
        return Gaussians(final_xyz, final_scale, final_rot, final_opac, base_sh)