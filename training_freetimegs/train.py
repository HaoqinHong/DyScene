import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= 路径自动配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# ================= 模块导入 =================
try:
    from loss import DINOMetricLoss, ssim
    from model import FreeTimeGSModel, Gaussians
    from dataset import IntegratedVideoDataset
except ImportError:
    from training.loss import DINOMetricLoss, ssim
    from training.model import FreeTimeGSModel, Gaussians
    from training.dataset import IntegratedVideoDataset

# 直接导入 gsplat 底层
try:
    from gsplat.rendering import rasterization
except ImportError:
    print("[ERROR] ❌ Could not import rasterization from gsplat!")

try:
    import depth_anything_3
except ImportError as e:
    print(f"[ERROR] ❌ Could not import depth_anything_3: {e}")

class FeatureAligner(nn.Module):
    def __init__(self, model_dim=512, dino_dim=768):
        super().__init__()
        self.proj = nn.Linear(model_dim, dino_dim)
    def forward(self, feats_3d):
        return F.normalize(self.proj(feats_3d), dim=-1)

def project_points(points_3d, c2w, K, H, W):
    # points_3d: [B, N, 3]
    B, N, _ = points_3d.shape
    w2c = torch.linalg.inv(c2w)
    ones = torch.ones(B, N, 1, device=points_3d.device)
    pts_h = torch.cat([points_3d, ones], dim=-1)
    pts_cam = (w2c @ pts_h.transpose(1, 2)).transpose(1, 2)[..., :3]
    pts_img = (K @ pts_cam.transpose(1, 2)).transpose(1, 2)
    z = pts_img[..., 2:3] + 1e-6
    x = pts_img[..., 0:1] / z
    y = pts_img[..., 1:2] / z
    u_norm = (x / (W - 1)) * 2 - 1
    v_norm = (y / (H - 1)) * 2 - 1
    valid_mask = (z > 0.1) & (u_norm > -1.0) & (u_norm < 1.0) & (v_norm > -1.0) & (v_norm < 1.0)
    grid = torch.cat([u_norm, v_norm], dim=-1).unsqueeze(1)
    return grid, valid_mask.squeeze(-1)

def train():
    # === 配置区 ===
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/bear_result_freetime")
    DEVICE = "cuda"
    EPOCHS = 200 
    LR = 1e-3  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Init
    dataset = IntegratedVideoDataset(video_dir=VIDEO_DIR, da3_model_path=DA3_PATH, concerto_model_path=CONCERTO_PATH, dino_model_path=DINO_PATH, voxel_size=0.02, device=DEVICE)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    token_dim = dataset.scene_tokens.shape[-1]
    visual_dim = dataset.scene_visual_tokens.shape[-1]
    hidden_dim = 512
    
    # 2. Models
    print("--- Initializing FreeTimeGS Model ---")
    model = FreeTimeGSModel(
        input_dim=token_dim, visual_dim=visual_dim,
        hidden_dim=hidden_dim, num_layers=8, nhead=8
    ).to(DEVICE)
    
    align_head = FeatureAligner(model_dim=hidden_dim, dino_dim=768).to(DEVICE)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(align_head.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    critic = DINOMetricLoss(model_path=DINO_PATH, device=DEVICE)
    
    model.train()
    align_head.train()
    
    print("--- Start Training (FreeTimeGS - Final Fixed) ---")
    
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        w_l1 = 0.5; w_ssim = 0.5; w_sem = 0.1
            
        for batch in pbar:
            tokens = batch["tokens"].to(DEVICE)
            visual_tokens = batch["visual_tokens"].to(DEVICE)
            coords = batch["coords"].to(DEVICE)
            point_times = batch["point_times"].to(DEVICE)
            t = batch["t"].to(DEVICE)
            
            gt_image = batch["gt_image"].to(DEVICE) # [1, 3, H, W]
            c2w = batch["c2w"].to(DEVICE)
            K = batch["K"].to(DEVICE) # [1, 3, 3]
            
            optimizer.zero_grad()
            
            # Forward
            gaussians_batched, feats_3d = model(tokens, visual_tokens, point_times, coords, render_t=t)
            
            # === [DIMENSIONS] ===
            # Geometry: [N, D]
            means = gaussians_batched.means.squeeze(0)      
            quats = gaussians_batched.rotations.squeeze(0)  
            scales = gaussians_batched.scales.squeeze(0)    
            opacities = gaussians_batched.opacities.squeeze(0) 
            
            # Colors: [N, 3] (RGB)
            colors = gaussians_batched.harmonics.squeeze(0).squeeze(-1) 
            
            # Cameras: [1, 4, 4]
            w2c = torch.linalg.inv(c2w)           
            
            _, _, H, W = gt_image.shape
            
            # [CRITICAL FIX]: 不要归一化 K！gsplat 需要像素坐标的 K
            # K_norm = K.clone(); K_norm /= ... (DELETE THIS)
            
            # RENDER
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=w2c,   
                Ks=K,         # Pass Original K (Pixel Space)
                width=W,
                height=H
            )
            
            # Output: [1, 3, H, W]
            render_out = render_colors.permute(0, 3, 1, 2)
            
            loss_l1 = (render_out - gt_image).abs().mean()
            
            # Calculate SSIM Metric
            ssim_val = ssim(render_out, gt_image)
            loss_ssim = 1.0 - ssim_val
            
            loss_sem = torch.tensor(0.0).to(DEVICE)
            
            if w_sem > 0 and not critic.disabled:
                gt_dense_feats = critic.get_dense_features(gt_image) # [1, C, H, W]
                feats_3d_mapped = align_head(feats_3d)
                grid, valid_mask = project_points(gaussians_batched.means, c2w, K, H, W)
                sampled_gt_feats = F.grid_sample(gt_dense_feats, grid, align_corners=True)
                sampled_gt_feats = sampled_gt_feats.squeeze(2).permute(0, 2, 1) 
                
                if valid_mask.sum() > 0:
                    sim = (feats_3d_mapped[0][valid_mask[0]] * sampled_gt_feats[0][valid_mask[0]]).sum(dim=-1)
                    loss_sem = 1.0 - sim.mean()
            
            reg_scale = torch.mean(scales)
            reg_opac = torch.mean(opacities) 
            
            loss = w_l1 * loss_l1 + w_ssim * loss_ssim + w_sem * loss_sem + \
                   0.0001 * reg_scale + 0.0001 * reg_opac
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # [LOGGING FIX]: 显示 SSIM Score，不是 Loss
            pbar.set_postfix({
                "L1": f"{loss_l1.item():.3f}", 
                "SSIM": f"{ssim_val.item():.3f}", # Show Score
                "SEM": f"{loss_sem.item():.3f}"
            })
        
        scheduler.step()
            
        if epoch % 50 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch}.pth"))
            
    print("Done! Model saved.")

if __name__ == "__main__":
    train()