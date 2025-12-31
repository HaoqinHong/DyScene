import sys
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob
import gc
from sklearn.neighbors import NearestNeighbors 

# === 路径配置 ===
project_root = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/DyScene"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Concerto 路径配置 ===
CONCERTO_ROOT = os.path.join(project_root, "submodules", "Concerto")
if CONCERTO_ROOT not in sys.path:
    sys.path.insert(0, CONCERTO_ROOT)

from depth_anything_3.api import DepthAnything3
try:
    import concerto
    from concerto.transform import Compose
except ImportError:
    try:
        import submodules.Concerto.concerto as concerto
        from submodules.Concerto.concerto.transform import Compose
    except ImportError as e:
        print(f"[Error] Failed to import concerto: {e}")
        raise

class IntegratedVideoDataset(Dataset):
    def __init__(self, video_dir, da3_model_path, concerto_model_path, dino_model_path, voxel_size=0.02, device='cuda'):
        self.device = device
        self.voxel_size = voxel_size
        
        self.image_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")) + 
                                  glob.glob(os.path.join(video_dir, "*.png")))
        if len(self.image_paths) == 0: raise ValueError(f"No images found in {video_dir}")
        self.num_frames = len(self.image_paths)

        # 1. 运行 DA3 (提取几何 + 特征)
        self._run_da3_geometry(da3_model_path)
        # 2. 运行 Concerto (提取 Token)
        self._run_concerto(concerto_model_path, voxel_size)
        # 3. 运行 DINOv2 (Loss特征)
        self._extract_gt_features_dinov2(dino_model_path)

    def _run_da3_geometry(self, model_path):
        print(f">> [1/3] Loading DA3 ({os.path.basename(model_path)})...")
        da3_model = DepthAnything3.from_pretrained(model_path, dynamic=True).to(self.device)
        da3_model.eval()

        # 动态获取最后一层索引
        try:
            n_layers = len(da3_model.model.backbone.blocks)
            last_layer_idx = n_layers - 1
            print(f"   [Info] Detected {n_layers} layers. Extracting layer: {last_layer_idx}")
        except:
            last_layer_idx = 39 # Fallback for Giant

        print("   Generating Point Cloud & Extracting Features...")
        with torch.no_grad():
            prediction = da3_model.inference(
                self.image_paths, infer_gs=True, process_res=518, 
                export_format="mini_npz", use_ray_pose=True,
                export_feat_layers=[last_layer_idx] 
            )
        
        # === [CRITICAL FIX] 维度修正 ===
        da3_features = None
        # 优先从 aux 读取
        if hasattr(prediction, 'aux') and prediction.aux is not None:
            key = f"feat_layer_{last_layer_idx}"
            if key in prediction.aux:
                da3_features = prediction.aux[key]
        
        if da3_features is None and hasattr(prediction, 'int_feats'):
             da3_features = prediction.int_feats[0] # fallback
        
        if da3_features is None:
            raise ValueError(f"Could not find features! Keys: {prediction.__dict__.keys()}")

        if not isinstance(da3_features, torch.Tensor):
            da3_features = torch.from_numpy(da3_features).to(self.device)
        else:
            da3_features = da3_features.to(self.device)
            
        # 1. 移除 Batch 维度: [B, N, H, W, C] -> [N, H, W, C]
        if da3_features.ndim == 5:
            da3_features = da3_features.squeeze(0) 
        
        print(f"   [Debug] Raw Feature Shape: {da3_features.shape}")

        # 2. 强制 Permute (Channels Last -> Channels First)
        # DA3 output is ALWAYS (N, H, W, C)
        # GridSample needs (N, C, H, W)
        da3_features = da3_features.permute(0, 3, 1, 2) # [N, C, H, W]
        
        print(f"   [Debug] Corrected for GridSample: {da3_features.shape}")
        
        # 验证 C 是否合理 (Giant应为1536, Base应为768)
        if da3_features.shape[1] < 100: 
            print("   [WARNING] Feature channels seem too small! Check dimension order again.")

        depths = torch.from_numpy(prediction.depth).to(self.device)
        intrinsics = torch.from_numpy(prediction.intrinsics).to(self.device)
        extrinsics_np = prediction.extrinsics
        
        if extrinsics_np.ndim == 3 and extrinsics_np.shape[1] == 3:
            N = extrinsics_np.shape[0]
            bottom = np.array([[[0,0,0,1]]]*N, dtype=extrinsics_np.dtype)
            extrinsics_np = np.concatenate([extrinsics_np, bottom], axis=1)
        w2c_raw = torch.from_numpy(extrinsics_np).to(self.device).float()
        c2w_raw = torch.linalg.inv(w2c_raw)

        all_pts, all_feats = [], []
        
        print("   Accumulating points and sampling features...")
        for i in range(self.num_frames):
            d = depths[i]; K = intrinsics[i]; c2w = c2w_raw[i]
            H, W = d.shape
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            x, y, z = x.to(self.device).flatten(), y.to(self.device).flatten(), d.flatten()
            valid = (z > 0.1)
            idx = torch.nonzero(valid).squeeze()
            
            if idx.numel() > 40000: idx = idx[torch.randperm(len(idx))[:40000]]
            if idx.numel() == 0: continue
            
            # 几何反投影
            x_s, y_s, z_s = x[idx], y[idx], z[idx]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            X_c = (x_s - cx) * z_s / fx
            Y_c = (y_s - cy) * z_s / fy
            Z_c = z_s
            pts_c = torch.stack([X_c, Y_c, Z_c], dim=-1)
            pts_w = (c2w[:3,:3] @ pts_c.T).T + c2w[:3,3]
            all_pts.append(pts_w)

            # 特征采样 (Grid Sample)
            # 归一化坐标到 [-1, 1]
            u_norm = (x_s / (W - 1)) * 2 - 1
            v_norm = (y_s / (H - 1)) * 2 - 1
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0) # [1, 1, N_pts, 2]
            
            # 取出当前帧特征 [1, C, Hf, Wf]
            feat_map = da3_features[i].unsqueeze(0) 
            
            # 采样
            sampled = torch.nn.functional.grid_sample(feat_map, grid, align_corners=True, mode='bilinear')
            sampled = sampled.squeeze().permute(1, 0) # -> [N_pts, C]
            
            all_feats.append(sampled.cpu())

        raw_coords = torch.cat(all_pts, 0)
        self.big_feats = torch.cat(all_feats, 0) # [N_total, Feature_Dim]
        
        scene_center = raw_coords.mean(0)
        dist = torch.linalg.norm(raw_coords - scene_center, dim=1)
        scale = 0.9 / max(torch.quantile(dist, 0.98).item(), 1e-2)
        
        print(f"   [Coord Fix] Scale: {scale:.4f}")
        self.big_coords = (raw_coords - scene_center) * scale
        
        c2w_fixed = c2w_raw.clone()
        c2w_fixed[:,:3,3] = (c2w_fixed[:,:3,3] - scene_center) * scale
        self.extrinsics = torch.linalg.inv(c2w_fixed).cpu()
        self.intrinsics = intrinsics.cpu()
        del da3_model; torch.cuda.empty_cache(); gc.collect()

    def _run_concerto(self, model_path, voxel_size):
        print(f">> [2/3] Loading Concerto...")
        concerto_model = concerto.model.load(model_path).to(self.device).eval()
        
        dummy_color = np.zeros((self.big_coords.shape[0], 3), dtype=np.float32)
        
        transform = Compose([
            dict(type="GridSample", grid_size=voxel_size, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "inverse"), feat_keys=("coord", "color"))
        ])
        
        input_dict = {"coord": self.big_coords.cpu().numpy(), "color": dummy_color}
        input_dict = transform(input_dict)
        
        feat = input_dict["feat"]
        if feat.shape[1] == 6: input_dict["feat"] = torch.cat([feat, torch.zeros((feat.shape[0],3), dtype=feat.dtype)], 1)
        
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor): input_dict[k] = v.to(self.device)
        input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], device=self.device)

        with torch.no_grad():
            point = concerto_model(input_dict)
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent"); inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            
            self.scene_tokens = point.feat.cpu()
            self.scene_coords = input_dict["coord"].cpu()
            
            print("   Matching DA3 features to Concerto tokens...")
            nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.big_coords.cpu().numpy())
            _, indices = nn_model.kneighbors(self.scene_coords.numpy())
            
            # 使用 Corrected Features
            self.scene_visual_tokens = self.big_feats[indices.flatten()].cpu() 
            print(f"   Visual Token Dim: {self.scene_visual_tokens.shape[-1]}")

        del concerto_model; torch.cuda.empty_cache(); gc.collect()

    def _extract_gt_features_dinov2(self, model_path):
        print(f">> [3/3] Extracting GT Features for Loss...")
        from training.loss import DINOMetricLoss
        extractor = DINOMetricLoss(model_path=model_path, device=self.device)
        self.gt_feats_list = []
        for p in self.image_paths:
            img = torch.from_numpy(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).permute(2,0,1).float().to(self.device)/255.0
            with torch.no_grad():
                out = extractor.dino(pixel_values=extractor.preprocess(img.unsqueeze(0)))
                self.gt_feats_list.append(torch.nn.functional.normalize(out.last_hidden_state[:,1:], dim=-1).cpu())
        del extractor; torch.cuda.empty_cache(); gc.collect()

    def __len__(self): return self.num_frames
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        return {
            "tokens": self.scene_tokens,
            "visual_tokens": self.scene_visual_tokens, 
            "coords": self.scene_coords,
            "t": torch.tensor([idx/(self.num_frames-1)]).float(),
            "gt_image": torch.from_numpy(img).permute(2,0,1).float()/255.0,
            "gt_feat": self.gt_feats_list[idx],
            "c2w": torch.linalg.inv(self.extrinsics[idx]),
            "K": self.intrinsics[idx]
        }