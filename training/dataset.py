import os
import sys
import glob
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F

# === 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)

# [FIX] 暴力寻找并添加 Depth Anything V3 到路径
# 假设它可能在 models 目录下，或者上一级目录
POSSIBLE_DA3_PATHS = [
    "/opt/data/private/models/depthanything3",
    "/opt/data/private/models/Depth-Anything-V3",
    os.path.join(PROJECT_ROOT, "depth_anything_3"),
    os.path.join(PROJECT_ROOT, "Depth-Anything-V3")
]
for p in POSSIBLE_DA3_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)
        print(f"[Dataset] Added {p} to sys.path")

# 尝试导入 loss 中的工具
try:
    from training.loss import DINOMetricLoss
except ImportError:
    from loss import DINOMetricLoss

# [FIX] 尝试导入 DA3
try:
    # 尝试多种 import 方式
    try:
        from depth_anything_3.dpt import DepthAnythingV3
    except ImportError:
        # 有些仓库直接把 dpt 放在根目录
        from dpt import DepthAnythingV3
    DA3_AVAILABLE = True
except ImportError as e:
    print(f"[Dataset Error] ❌ Could not import DepthAnythingV3: {e}")
    print("Hint: Check where 'depth_anything_3' folder is located.")
    DA3_AVAILABLE = False

class IntegratedVideoDataset(Dataset):
    def __init__(self, video_dir, da3_model_path, concerto_model_path, dino_model_path, voxel_size=0.01, device="cuda"):
        self.video_dir = video_dir
        self.device = device
        self.voxel_size = voxel_size
        self.da3_model_path = da3_model_path
        
        self.image_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(video_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {video_dir}")
        print(f"   [Dataset] Found {len(self.image_paths)} images.")
        
        self.gt_images = []
        self.c2ws = []
        self.Ks = []
        self.time_stamps = []
        
        # 相机参数估算
        img_ref = cv2.imread(self.image_paths[0])
        H, W = img_ref.shape[:2]
        focal = max(H, W) * 1.2
        K = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=torch.float32)
        
        print("   [Dataset] Loading images to RAM...")
        for idx, path in enumerate(self.image_paths):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img / 255.0).float().permute(2, 0, 1)
            self.gt_images.append(img_tensor)
            
            c2w = torch.eye(4, dtype=torch.float32)
            c2w[2, 3] = 2.0 - (idx / len(self.image_paths)) * 1.0 
            self.c2ws.append(c2w)
            self.Ks.append(K)
            self.time_stamps.append(torch.tensor([idx / len(self.image_paths)], dtype=torch.float32))

        print("   [Dataset] Initializing Point Cloud with Depth Anything V3...")
        self.scene_tokens, self.scene_visual_tokens, self.scene_coords, self.scene_times = self._init_point_cloud_with_da3()
        
        print("   [Dataset] Extracting Semantic Features (DINOv2)...")
        self._extract_semantic_features(dino_model_path)
        print(f"   [Dataset] Ready. {len(self.scene_coords)} points.")

    def _init_point_cloud_with_da3(self):
        if not DA3_AVAILABLE:
            print("[Warning] ⚠️ DA3 module missing! Using Random Cloud (Expect LPIPS Crash).")
            return self._init_random_cloud()

        print(f"   [Dataset] Loading DA3 from {self.da3_model_path}...")
        try:
            da3 = DepthAnythingV3.from_pretrained(self.da3_model_path).to(self.device)
            da3.eval()
        except Exception as e:
            print(f"[Error] Failed to load DA3 weights: {e}")
            return self._init_random_cloud()
        
        all_coords = []
        step = max(1, len(self.gt_images) // 8) 
        indices = range(0, len(self.gt_images), step)
        
        print(f"   [Dataset] Projecting points from {len(indices)} views...")
        
        for idx in indices:
            img = self.gt_images[idx].to(self.device)
            K = self.Ks[idx].to(self.device)
            c2w = self.c2ws[idx].to(self.device)
            
            # DA3 Process
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            img_in = (img.unsqueeze(0) - mean) / std
            
            with torch.no_grad():
                depth = da3(img_in)
                d_min, d_max = depth.min(), depth.max()
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                depth = depth_norm * 4.0 + 1.0 
            
            # Unproject
            _, H, W = img.shape
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            x_cam = (x - cx) * depth.squeeze() / fx
            y_cam = (y - cy) * depth.squeeze() / fy
            z_cam = depth.squeeze()
            
            pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1).reshape(-1, 3)
            pts_cam_h = torch.cat([pts_cam, torch.ones_like(z_cam).reshape(-1, 1)], dim=-1)
            pts_world = (c2w @ pts_cam_h.T).T[:, :3]
            
            mask = torch.rand(pts_world.shape[0], device=self.device) < 0.05
            all_coords.append(pts_world[mask])

        full_cloud = torch.cat(all_coords, dim=0)
        del da3; torch.cuda.empty_cache()
        
        print(f"   [Dataset] Voxelizing {full_cloud.shape[0]} points...")
        quantized = torch.round(full_cloud / self.voxel_size)
        unique_quantized, inverse = torch.unique(quantized, dim=0, return_inverse=True)
        final_coords = full_cloud[torch.unique(inverse)]
        
        if final_coords.shape[0] > 100000:
            perm = torch.randperm(final_coords.shape[0])[:80000]
            final_coords = final_coords[perm]
            
        N = final_coords.shape[0]
        tokens = torch.randn(N, 1024) 
        visual_tokens = torch.randn(N, 1536)
        times = torch.rand(N, 1)
        
        return tokens, visual_tokens, final_coords.cpu(), times

    def _init_random_cloud(self):
        N = 50000
        coords = (torch.rand(N, 3) - 0.5) * 4.0
        coords[:, 2] += 2.0
        return torch.randn(N, 1024), torch.randn(N, 1536), coords, torch.rand(N, 1)

    def _extract_semantic_features(self, dino_model_path):
        extractor = DINOMetricLoss(dino_model_path, device=self.device)
        ref_img = self.gt_images[0].unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = extractor.get_dense_features(ref_img)
        del extractor; torch.cuda.empty_cache()

    def __len__(self): return len(self.gt_images)
    def __getitem__(self, idx):
        return {
            "gt_image": self.gt_images[idx], 
            "c2w": self.c2ws[idx],
            "K": self.Ks[idx],
            "t": self.time_stamps[idx],
            "tokens": self.scene_tokens,
            "visual_tokens": self.scene_visual_tokens,
            "coords": self.scene_coords,
            "point_times": self.scene_times
        }