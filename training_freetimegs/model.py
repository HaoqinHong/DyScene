import torch
import torch.nn as nn
import torch.nn.functional as F

class Gaussians:
    def __init__(self, means, scales, rotations, opacities, harmonics):
        self.means = means
        self.scales = scales
        self.rotations = rotations
        self.opacities = opacities
        self.harmonics = harmonics

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = FlashAttention(hidden_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, hidden_dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_dim, bias=True))
        with torch.no_grad():
            self.adaLN_modulation[1].weight.zero_()
            self.adaLN_modulation[1].bias.zero_()

    def forward(self, x, t_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x_norm1)
        x = x + gate_msa * attn_out
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm2)
        return x

class FreeTimeGSModel(nn.Module):
    def __init__(self, input_dim=1024, visual_dim=1536, hidden_dim=512, num_layers=8, nhead=8):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim + visual_dim + 1, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.global_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, cond_dim=hidden_dim, num_heads=nhead) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Heads
        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.scale_head = nn.Linear(hidden_dim, 3)
        self.rot_head = nn.Linear(hidden_dim, 4)
        self.opac_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        
        self.velocity_head = nn.Linear(hidden_dim, 3)
        self.t_center_head = nn.Linear(hidden_dim, 1)
        self.t_scale_head = nn.Linear(hidden_dim, 1)
        
        nn.init.zeros_(self.xyz_head.weight); nn.init.zeros_(self.xyz_head.bias)
        nn.init.zeros_(self.velocity_head.weight); nn.init.zeros_(self.velocity_head.bias)

    def forward(self, concerto_tokens, visual_tokens, point_times, coords, render_t):
        B, N, _ = coords.shape
        
        raw_feat = torch.cat([concerto_tokens, visual_tokens, point_times], dim=-1)
        x = self.feature_encoder(raw_feat)
        
        cond = self.global_token.expand(B, -1) 
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)
        
        # A. Static Base
        xyz_static = coords + torch.tanh(self.xyz_head(x)) * 0.5
        raw_scale = self.scale_head(x) - 4.0
        scale = torch.clamp(torch.exp(raw_scale), 0.001, 0.2)
        rot = F.normalize(self.rot_head(x), dim=-1)
        base_opac = torch.sigmoid(self.opac_head(x)).squeeze(-1) # [B, N]
        
        # [CRITICAL FIX] 添加 Sigmoid，强制颜色在 [0, 1] 之间！
        # 之前这里是 SH 模式不需要 Sigmoid，现在是 RGB 模式必须加
        sh = torch.sigmoid(self.color_head(x)).unsqueeze(-1) # [B, N, 3, 1]
        
        # B. Dynamics (FreeTimeGS)
        velocity = torch.tanh(self.velocity_head(x)) * 2.0 
        
        # 修正广播问题，确保是 element-wise 加法
        t_center_offset = torch.tanh(self.t_center_head(x)) * 0.5 
        t_center = point_times + t_center_offset 
        
        t_scale = torch.exp(self.t_scale_head(x) - 2.0) + 0.01 
        
        # 修正广播问题，确保 dt 形状正确
        dt = render_t.unsqueeze(-1) - t_center # [B, 1, 1] - [B, N, 1] = [B, N, 1]
        
        final_xyz = xyz_static + velocity * dt
        
        decay = torch.exp( - (dt ** 2) / (2 * t_scale ** 2) ).squeeze(-1)
        final_opac = base_opac * decay
        
        return Gaussians(final_xyz, scale, rot, final_opac, sh), x