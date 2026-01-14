import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= 路径强制配置 (CRITICAL FIX) =================
# 1. 定义项目根目录的绝对路径
PROJECT_ROOT = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/DyScene"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# 2. 强制插入到 sys.path 的第 0 位 (最高优先级)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 3. 验证加载路径 (调试用)
try:
    import depth_anything_3
    print(f"\n[DEBUG] ✅ Loaded depth_anything_3 from: {depth_anything_3.__file__}")
    from depth_anything_3.api import DepthAnything3
    import inspect
    if 'use_ray_pose' not in inspect.signature(DepthAnything3.inference).parameters:
        print("[CRITICAL WARNING] ❌ Loaded library is still OLD! Please check 'src' folder content.\n")
    else:
        print("[DEBUG] ✅ 'use_ray_pose' parameter confirmed.\n")
except ImportError:
    print("[ERROR] ❌ Could not import depth_anything_3 from src!\n")

# ================= 模块导入 =================
from training.loss import DINOMetricLoss, ssim
from training.model import FreeTimeGSModel
from training.dataset import IntegratedVideoDataset
from depth_anything_3.model.utils.gs_renderer import render_3dgs

def train():
    # === 配置区 ===
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    
    OUTPUT_DIR = "./checkpoints/bear_result_refined" 
    
    DEVICE = "cuda"
    EPOCHS = 150 
    LR = 5e-4
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset
    print("--- 1. Pipeline Initialization ---")
    dataset = IntegratedVideoDataset(
        video_dir=VIDEO_DIR, da3_model_path=DA3_PATH, concerto_model_path=CONCERTO_PATH,
        dino_model_path=DINO_PATH, voxel_size=0.02, device=DEVICE
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Model Initialization
    print("--- 2. Model Initialization ---")
    token_dim = dataset.scene_tokens.shape[-1]
    
    # [核心修改] 动态获取 DA3 Feature 的维度 (例如 1536)
    visual_dim = dataset.scene_visual_tokens.shape[-1]
    print(f"[Model] Input Dims -> Geometry: {token_dim}, Visual: {visual_dim}")
    
    model = FreeTimeGSModel(input_dim=token_dim, visual_dim=visual_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 3. Critic
    print(f"--- 3. Initializing Critic ---")
    critic = DINOMetricLoss(model_path=DINO_PATH, device=DEVICE)
    
    # 4. Train
    print("--- 4. Training Loop ---")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        # === Loss Warm-up Schedule ===
        # 前 50 个 epoch 专注几何 (L1)，之后开启 DINO/SSIM 进行精修
        if epoch < 50:
            w_l1 = 1.0
            w_ssim = 0.0
            w_dino = 0.0
            phase = "Geometry"
        else:
            w_l1 = 0.8
            w_ssim = 0.2
            w_dino = 0.05
            phase = "Refine"
            
        for batch in pbar:
            tokens = batch["tokens"].to(DEVICE)
            visual_tokens = batch["visual_tokens"].to(DEVICE) # [B, N, Feature_Dim]
            coords = batch["coords"].to(DEVICE)
            t = batch["t"].to(DEVICE)
            
            gt_image = batch["gt_image"].to(DEVICE)
            gt_feat = batch["gt_feat"].to(DEVICE)
            c2w = batch["c2w"].to(DEVICE)
            K = batch["K"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward: 传入 visual_tokens (DA3 Features)
            gaussians = model(tokens, visual_tokens, coords, t)
            
            # Render
            w2c = torch.linalg.inv(c2w)
            _, _, H, W = gt_image.shape
            K_norm = K.clone(); K_norm[...,0,:]/=W; K_norm[...,1,:]/=H
            
            render_out, _ = render_3dgs(
                extrinsics=w2c, intrinsics=K_norm, image_shape=(H,W),
                gaussian=gaussians, num_view=1,
                background_color=torch.zeros(1,3).to(DEVICE)
            )
            render_out = render_out.squeeze(1)
            
            # === Loss Calculation ===
            loss_l1 = (render_out - gt_image).abs().mean()
            loss_ssim = torch.tensor(0.0).to(DEVICE)
            loss_feat = torch.tensor(0.0).to(DEVICE)
            
            if w_ssim > 0:
                val_ssim = ssim(render_out, gt_image)
                loss_ssim = 1.0 - val_ssim
                
            if w_dino > 0:
                loss_feat = critic(render_out, gt_feats=gt_feat)
            
            # 正则化
            reg_scale = gaussians.scales.mean()
            reg_opac = gaussians.opacities.mean()
            
            loss = w_l1 * loss_l1 + w_ssim * loss_ssim + w_dino * loss_feat + \
                   0.0001 * reg_scale + 0.0001 * reg_opac
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                "Ph": phase,
                "L1": f"{loss_l1.item():.3f}", 
                "SSIM": f"{loss_ssim.item():.3f}",
                "DINO": f"{loss_feat.item():.3f}"
            })
            
        if epoch % 50 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_epoch_{epoch}.pth")
            
    print("Done! Model saved.")

if __name__ == "__main__":
    train()