import sys
import os
import torch
import numpy as np
import cv2
import glob
from tqdm import tqdm

# ================= 路径强制配置 (与 train.py 保持一致) =================
# 确保优先加载本地 src 下的库，避免版本冲突
PROJECT_ROOT = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/DyScene"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ================= 模块导入 =================
from training.model import FreeTimeGSModel
from training.dataset import IntegratedVideoDataset
from depth_anything_3.model.utils.gs_renderer import render_3dgs

def find_latest_checkpoint(checkpoint_dir):
    """自动寻找目录下 Epoch 最大的模型文件"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
        
    # 优先匹配 model_epoch_*.pth
    files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
    if not files:
        # 尝试找 final_model.pth 作为后备
        final_path = os.path.join(checkpoint_dir, "final_model.pth")
        if os.path.exists(final_path):
            return final_path
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # 解析文件名中的数字排序
    def get_epoch(f):
        try:
            return int(f.split("model_epoch_")[-1].replace(".pth", ""))
        except:
            return -1
            
    latest_file = max(files, key=get_epoch)
    print(f"[Info] Found latest checkpoint: {os.path.basename(latest_file)}")
    return latest_file

def generate_bullet_time_path(c2ws, n_frames=120, amplitude=0.5):
    """生成子弹时间轨迹（时间静止，相机左右摆动）"""
    c2ws = c2ws.cpu().numpy()
    # 取中间帧作为基准中心
    center_pose = c2ws[len(c2ws)//2]
    R0 = center_pose[:3, :3]
    t0 = center_pose[:3, 3]
    
    render_poses = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        pose = np.eye(4)
        pose[:3, :3] = R0
        # 在 X 轴方向做正弦摆动，Y 轴做轻微起伏
        pose[0, 3] = t0[0] + np.sin(theta) * amplitude * 0.5
        pose[1, 3] = t0[1] + np.cos(theta) * amplitude * 0.2
        pose[2, 3] = t0[2]
        render_poses.append(pose)
        
    return torch.from_numpy(np.stack(render_poses)).float()

def render_video():
    """ 
    [任务 1] 标准重建对比 
    渲染原始视角的视频，并与 GT 拼接对比。
    """
    print("\n=== [Task 1] Rendering Reconstruction Comparison ===")
    
    # === 配置区 ===
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    
    CHECKPOINT_DIR = "./checkpoints/bear_result_refined"
    SAVE_DIR = "./renders/bear_refined"
    
    DEVICE = "cuda"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 加载数据
    print("--- 1. Re-loading Data ---")
    dataset = IntegratedVideoDataset(
        video_dir=VIDEO_DIR, da3_model_path=DA3_PATH, concerto_model_path=CONCERTO_PATH,
        dino_model_path=DINO_PATH, voxel_size=0.02, device=DEVICE
    )
    
    # 2. 初始化模型
    print("--- 2. Loading Model ---")
    token_dim = dataset.scene_tokens.shape[-1]
    visual_dim = dataset.scene_visual_tokens.shape[-1]
    
    model = FreeTimeGSModel(input_dim=token_dim, visual_dim=visual_dim).to(DEVICE)
    
    # 3. 加载权重
    ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. 渲染循环
    print(f"--- 3. Rendering {len(dataset)} frames ---")
    
    render_frames = []
    gt_frames = []
    
    # 准备静态特征输入
    tokens = dataset.scene_tokens.to(DEVICE).unsqueeze(0)
    visual_tokens = dataset.scene_visual_tokens.to(DEVICE).unsqueeze(0)
    coords = dataset.scene_coords.to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            
            # 时间输入
            t = data["t"].to(DEVICE).unsqueeze(0)
            if t.ndim == 1: t = t.unsqueeze(0)
            
            # 动态生成场景
            gaussians = model(tokens, visual_tokens, coords, t)
            
            # 获取相机参数
            c2w = data["c2w"].to(DEVICE)
            w2c = torch.linalg.inv(c2w)
            K = data["K"].to(DEVICE)
            gt_img = data["gt_image"].to(DEVICE)
            
            _, H, W = gt_img.shape
            K_norm = K.clone()
            K_norm[0, :] /= W
            K_norm[1, :] /= H
            K_norm = K_norm.unsqueeze(0)
            w2c = w2c.unsqueeze(0)
            
            # 渲染
            render_out, _ = render_3dgs(
                extrinsics=w2c, intrinsics=K_norm, image_shape=(H,W),
                gaussian=gaussians, num_view=1,
                background_color=torch.zeros(1,3).to(DEVICE)
            )
            
            # 后处理
            render_bgr = cv2.cvtColor((render_out.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            gt_bgr = cv2.cvtColor((gt_img.permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            render_frames.append(render_bgr)
            gt_frames.append(gt_bgr)

    # 保存视频
    video_path = f"{SAVE_DIR}/reconstruction_compare.mp4"
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (W*2, H))
    
    for gt, pred in zip(gt_frames, render_frames):
        combined = np.hstack((gt, pred))
        out.write(combined)
    out.release()
    print(f"Saved reconstruction video to: {video_path}")

def render_bullet_time():
    """ 
    [任务 2] 子弹时间特效 
    冻结时间在 t=0.5，相机沿生成的轨迹移动。
    """
    print("\n=== [Task 2] Rendering Bullet Time (Frozen Time) ===")
    
    # === 配置 ===
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    CHECKPOINT_DIR = "./checkpoints/bear_result_refined"
    SAVE_DIR = "./renders/bear_refined"
    DEVICE = "cuda"
    
    # 加载
    dataset = IntegratedVideoDataset(video_dir=VIDEO_DIR, da3_model_path=DA3_PATH, concerto_model_path=CONCERTO_PATH, dino_model_path=DINO_PATH, voxel_size=0.02, device=DEVICE)
    model = FreeTimeGSModel(input_dim=dataset.scene_tokens.shape[-1], visual_dim=dataset.scene_visual_tokens.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(find_latest_checkpoint(CHECKPOINT_DIR), map_location=DEVICE))
    model.eval()
    
    # 生成 NVS 轨迹
    orig_c2ws = torch.stack([dataset[i]['c2w'] for i in range(len(dataset))])
    nvs_c2ws = generate_bullet_time_path(orig_c2ws, n_frames=120, amplitude=0.5).to(DEVICE)
    
    # 冻结时间 t=0.5
    print("Generating Frozen Scene at t=0.5...")
    fixed_t = torch.tensor([[0.5]]).to(DEVICE)
    tokens = dataset.scene_tokens.to(DEVICE).unsqueeze(0)
    visual_tokens = dataset.scene_visual_tokens.to(DEVICE).unsqueeze(0)
    coords = dataset.scene_coords.to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        gaussians = model(tokens, visual_tokens, coords, fixed_t)
        
    frames = []
    # 假设内参不变，取第一帧
    K = dataset[0]['K'].to(DEVICE); _, H, W = dataset[0]['gt_image'].shape
    K_norm = K.clone(); K_norm[0,:]/=W; K_norm[1,:]/=H; K_norm = K_norm.unsqueeze(0)
    
    print("Rendering NVS frames...")
    with torch.no_grad():
        for i in tqdm(range(len(nvs_c2ws))):
            w2c = torch.linalg.inv(nvs_c2ws[i]).unsqueeze(0)
            render_out, _ = render_3dgs(
                extrinsics=w2c, intrinsics=K_norm, image_shape=(H,W),
                gaussian=gaussians, num_view=1, background_color=torch.zeros(1,3).to(DEVICE)
            )
            frames.append(cv2.cvtColor((render_out.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
    out_path = f"{SAVE_DIR}/bullet_time_t0.5.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (W, H))
    for f in frames: out.write(f)
    out.release()
    print(f"Saved bullet time video to: {out_path}")

def render_fixed_view_dynamics():
    """ 
    [任务 3] 固定新视角看动态 
    选取一个参考帧的相机位置，施加轻微偏移，固定住相机，播放完整的时间动态。
    """
    print("\n=== [Task 3] Rendering Fixed Novel View Dynamics ===")
    
    # === 配置 ===
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base" 
    CHECKPOINT_DIR = "./checkpoints/bear_result_refined"
    SAVE_DIR = "./renders/bear_refined"
    DEVICE = "cuda"
    
    # === 用户自定义参数 ===
    # 选择参考帧：0.5 表示视频中间的那一帧（通常视野覆盖最好）
    REF_FRAME_RATIO = 0.5 
    
    # 相机偏移量 [x, y, z] (米)，在相机坐标系下移动
    # 例如：向右移 0.2m，向前移 0.1m
    SHIFT_OFFSET = np.array([0.2, 0.0, 0.1]) 
    
    # 加载
    dataset = IntegratedVideoDataset(video_dir=VIDEO_DIR, da3_model_path=DA3_PATH, concerto_model_path=CONCERTO_PATH, dino_model_path=DINO_PATH, voxel_size=0.02, device=DEVICE)
    model = FreeTimeGSModel(input_dim=dataset.scene_tokens.shape[-1], visual_dim=dataset.scene_visual_tokens.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(find_latest_checkpoint(CHECKPOINT_DIR), map_location=DEVICE))
    model.eval()
    
    # 1. 计算固定视角
    ref_idx = int(len(dataset) * REF_FRAME_RATIO)
    ref_idx = min(ref_idx, len(dataset)-1)
    print(f"Using Frame {ref_idx} as reference camera view.")
    
    # 获取原始 c2w
    fixed_c2w = dataset[ref_idx]['c2w'].clone().cpu().numpy()
    
    # 计算偏移：World_Shift = R_cam @ Local_Shift
    R_cam = fixed_c2w[:3, :3]
    world_shift = R_cam @ SHIFT_OFFSET
    
    # 应用偏移
    fixed_c2w[:3, 3] += world_shift
    print(f"Applied Camera Shift (Local): {SHIFT_OFFSET}")
    
    fixed_c2w = torch.from_numpy(fixed_c2w).to(DEVICE)
    fixed_w2c = torch.linalg.inv(fixed_c2w).unsqueeze(0) # [1, 4, 4]
    
    # 2. 渲染循环 (时间流动)
    frames = []
    tokens = dataset.scene_tokens.to(DEVICE).unsqueeze(0)
    visual_tokens = dataset.scene_visual_tokens.to(DEVICE).unsqueeze(0)
    coords = dataset.scene_coords.to(DEVICE).unsqueeze(0)
    K = dataset[0]['K'].to(DEVICE); _, H, W = dataset[0]['gt_image'].shape
    K_norm = K.clone(); K_norm[0,:]/=W; K_norm[1,:]/=H; K_norm = K_norm.unsqueeze(0)
    
    print(f"Rendering {len(dataset)} frames from FIXED novel view...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            # 时间流动
            t = dataset[i]["t"].to(DEVICE).unsqueeze(0)
            if t.ndim == 1: t = t.unsqueeze(0)
            
            # 生成当前时刻的 3D 场景
            gaussians = model(tokens, visual_tokens, coords, t)
            
            # 始终使用同一个固定相机
            render_out, _ = render_3dgs(
                extrinsics=fixed_w2c, intrinsics=K_norm, image_shape=(H,W),
                gaussian=gaussians, num_view=1, background_color=torch.zeros(1,3).to(DEVICE)
            )
            frames.append(cv2.cvtColor((render_out.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
    out_path = f"{SAVE_DIR}/fixed_novel_view_dynamic.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (W, H))
    for f in frames: out.write(f)
    out.release()
    print(f"Saved fixed novel view video to: {out_path}")

if __name__ == "__main__":
    # 可以注释掉不需要的任务
    
    # 1. 重建对比 (检查训练效果)
    render_video()
    
    # 2. 子弹时间 (展示 3D 结构)
    render_bullet_time()
    
    # 3. 固定新视角 (展示动态一致性)
    render_fixed_view_dynamics()