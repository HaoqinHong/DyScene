import sys
import os
import torch
import numpy as np
import cv2
import glob
from tqdm import tqdm

# ================= 路径自动配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)

# ================= 模块导入 =================
try:
    from training.model import FeedForwardGSModel
    from training.dataset import IntegratedVideoDataset
except ImportError:
    from model import FeedForwardGSModel
    from dataset import IntegratedVideoDataset

from depth_anything_3.model.utils.gs_renderer import render_3dgs

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir): raise FileNotFoundError(f"Not found: {checkpoint_dir}")
    files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
    if not files:
        if os.path.exists(os.path.join(checkpoint_dir, "final_model.pth")): return os.path.join(checkpoint_dir, "final_model.pth")
        raise FileNotFoundError("No checkpoints found.")
    def get_epoch(f):
        try: return int(f.split("model_epoch_")[-1].replace(".pth", ""))
        except: return -1
    return max(files, key=get_epoch)

def render_video():
    print("\n=== [Task 1] Rendering Reconstruction ===")
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/bear_result_refined")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "renders/bear_refined")
    DEVICE = "cuda"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    dataset = IntegratedVideoDataset(video_dir=VIDEO_DIR, da3_model_path=DA3_PATH, concerto_model_path=CONCERTO_PATH, dino_model_path=DINO_PATH, voxel_size=0.02, device=DEVICE)
    model = FeedForwardGSModel(input_dim=dataset.scene_tokens.shape[-1], visual_dim=dataset.scene_visual_tokens.shape[-1], hidden_dim=512, num_layers=8, nhead=8).to(DEVICE)
    
    ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    render_frames, gt_frames = [], []
    tokens = dataset.scene_tokens.to(DEVICE).unsqueeze(0)
    visual_tokens = dataset.scene_visual_tokens.to(DEVICE).unsqueeze(0)
    coords = dataset.scene_coords.to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            t = data["t"].to(DEVICE).unsqueeze(0)
            if t.ndim == 1: t = t.unsqueeze(0)
            
            # [Fix] Unpack tuple (gaussians, feats)
            gaussians, _ = model(tokens, visual_tokens, coords, t, static_only=False)
            
            c2w = data["c2w"].to(DEVICE); w2c = torch.linalg.inv(c2w)
            K = data["K"].to(DEVICE); gt_img = data["gt_image"].to(DEVICE)
            _, H, W = gt_img.shape
            K_norm = K.clone(); K_norm[0,:]/=W; K_norm[1,:]/=H; K_norm = K_norm.unsqueeze(0); w2c = w2c.unsqueeze(0)
            
            render_out, _ = render_3dgs(extrinsics=w2c, intrinsics=K_norm, image_shape=(H,W), gaussian=gaussians, num_view=1, background_color=torch.zeros(1,3).to(DEVICE))
            
            render_bgr = cv2.cvtColor((render_out.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            gt_bgr = cv2.cvtColor((gt_img.permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            render_frames.append(render_bgr); gt_frames.append(gt_bgr)

    out = cv2.VideoWriter(os.path.join(SAVE_DIR, "reconstruction.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 24, (W*2, H))
    for gt, pred in zip(gt_frames, render_frames): out.write(np.hstack((gt, pred)))
    out.release()
    print("Saved reconstruction.mp4")

if __name__ == "__main__":
    render_video()