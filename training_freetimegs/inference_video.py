import sys
import os
import torch
import numpy as np
import cv2
import glob
import argparse
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

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("[ERROR] ❌ Could not import rasterization from gsplat!")

def get_args():
    parser = argparse.ArgumentParser(description="FreeTimeGS Inference Script")
    parser.add_argument("--scene_name", type=str, default="bear", help="Name of the scene (e.g., bear, blackswan)")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    
    # 基础路径配置 (可以根据你的服务器环境修改默认值)
    parser.add_argument("--dataset_root", type=str, default="/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p", help="Root dir of DAVIS dataset")
    parser.add_argument("--da3_path", type=str, default="/opt/data/private/models/depthanything3/DA3-GIANT")
    parser.add_argument("--concerto_path", type=str, default="/opt/data/private/models/concerto/concerto_large.pth")
    parser.add_argument("--dino_path", type=str, default="/opt/data/private/models/dinov2-base")
    
    return parser.parse_args()

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
        
    files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
    if not files:
        if os.path.exists(os.path.join(checkpoint_dir, "final_model.pth")):
            return os.path.join(checkpoint_dir, "final_model.pth")
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def get_epoch(f):
        try:
            return int(f.split("model_epoch_")[-1].replace(".pth", ""))
        except:
            return -1
            
    latest_file = max(files, key=get_epoch)
    return latest_file

def render_video():
    args = get_args()
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DEVICE = "cuda"

    # === 动态构建路径 ===
    SCENE_NAME = args.scene_name
    VIDEO_DIR = os.path.join(args.dataset_root, SCENE_NAME)
    
    # 假设你的 checkpoint 命名规则是 {scene_name}_result_freetime
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, f"checkpoints/{SCENE_NAME}_result_freetime")
    
    # 输出文件夹
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f"{SCENE_NAME}_inference.mp4")

    print(f"==========================================")
    print(f"Scene:      {SCENE_NAME}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Output:     {OUTPUT_VIDEO_PATH}")
    print(f"==========================================")
    
    # 1. 寻找权重
    try:
        ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)
        print(f">> Found Latest Checkpoint: {os.path.basename(ckpt_path)}")
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print(f"Hint: Did you train '{SCENE_NAME}'? Check directory: {CHECKPOINT_DIR}")
        return

    # 2. 加载数据集
    if not os.path.exists(VIDEO_DIR):
        print(f"[Error] Dataset for scene '{SCENE_NAME}' not found at {VIDEO_DIR}")
        return

    print(">> Loading Dataset...")
    dataset = IntegratedVideoDataset(
        video_dir=VIDEO_DIR, da3_model_path=args.da3_path, concerto_model_path=args.concerto_path,
        dino_model_path=args.dino_path, voxel_size=0.02, device=DEVICE
    )
    
    token_dim = dataset.scene_tokens.shape[-1]
    visual_dim = dataset.scene_visual_tokens.shape[-1]
    hidden_dim = 512

    # 3. 加载模型
    print(">> Loading Model...")
    model = FreeTimeGSModel(
        input_dim=token_dim, visual_dim=visual_dim,
        hidden_dim=hidden_dim, num_layers=8, nhead=8
    ).to(DEVICE)
    
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. 准备输入
    tokens = dataset.scene_tokens.unsqueeze(0).to(DEVICE)
    visual_tokens = dataset.scene_visual_tokens.unsqueeze(0).to(DEVICE)
    coords = dataset.scene_coords.unsqueeze(0).to(DEVICE)
    point_times = dataset.scene_times.unsqueeze(0).to(DEVICE)

    # 5. 视频初始化
    ref_img = cv2.imread(dataset.image_paths[0])
    H, W, _ = ref_img.shape
    fps = 24 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W * 2, H))

    print(f">> Rendering Frames...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            t_val = data["t"].to(DEVICE).unsqueeze(0)
            c2w = data["c2w"].unsqueeze(0).to(DEVICE)
            K = data["K"].unsqueeze(0).to(DEVICE)
            gt_image = data["gt_image"].to(DEVICE)

            # Forward
            gaussians_batched, _ = model(tokens, visual_tokens, point_times, coords, render_t=t_val)

            # === Golden Dimensions Strategy ===
            means = gaussians_batched.means.squeeze(0)
            quats = gaussians_batched.rotations.squeeze(0)
            scales = gaussians_batched.scales.squeeze(0)
            opacities = gaussians_batched.opacities.squeeze(0)
            colors = gaussians_batched.harmonics.squeeze(0).squeeze(-1)
            
            w2c = torch.linalg.inv(c2w)
            
            # 使用原始 K (Pixel Space)
            K_render = K.clone()
            
            # RENDER
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=w2c,
                Ks=K_render, 
                width=W,
                height=H
            )
            
            # [FIX]: Take index 0 for numpy conversion
            render_img = render_colors[0].detach().cpu().numpy()
            
            render_img = np.clip(render_img, 0, 1)
            render_img = (render_img * 255).astype(np.uint8)
            render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)

            # GT
            gt_img_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            gt_img_np = np.clip(gt_img_np, 0, 1)
            gt_img_np = (gt_img_np * 255).astype(np.uint8)
            gt_img_np = cv2.cvtColor(gt_img_np, cv2.COLOR_RGB2BGR)

            combined = np.hstack([gt_img_np, render_img])
            out.write(combined)

    out.release()
    print(f">> Success! Saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    render_video()