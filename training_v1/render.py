import sys
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import glob

# ================= 路径配置 =================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

CONCERTO_ROOT = "/opt/data/private/Ours-Projects/Physics-Simulator-World-Model/DyScene/submodules/Concerto"
if CONCERTO_ROOT not in sys.path:
    sys.path.insert(0, CONCERTO_ROOT)

# ================= 模块导入 =================
from training.model import FreeTimeGSModel
from training.dataset import IntegratedVideoDataset
from depth_anything_3.model.utils.gs_renderer import render_3dgs

def render_video():
    VIDEO_DIR = "/opt/data/private/datasets/davis_2016/DAVIS_2016/JPEGImages/1080p/bear" 
    DA3_PATH = "/opt/data/private/models/depthanything3/DA3-GIANT" 
    CONCERTO_PATH = "/opt/data/private/models/concerto/concerto_large.pth"
    DINO_PATH = "/opt/data/private/models/dinov2-base"
    CHECKPOINT_PATH = "./checkpoints/bear_result/final_model.pth"
    OUTPUT_ROOT = "./outputs"
    DEVICE = "cuda"
    
    # 临时文件夹用于存放帧 (ffmpeg 将从这里读取)
    TMP_RENDER = os.path.join(OUTPUT_ROOT, "tmp_render")
    TMP_GT = os.path.join(OUTPUT_ROOT, "tmp_gt")
    TMP_COMPARE = os.path.join(OUTPUT_ROOT, "tmp_compare")
    
    os.makedirs(TMP_RENDER, exist_ok=True)
    os.makedirs(TMP_GT, exist_ok=True)
    os.makedirs(TMP_COMPARE, exist_ok=True)
    
    print("--- 1. Re-loading Data ---")
    dataset = IntegratedVideoDataset(VIDEO_DIR, DA3_PATH, CONCERTO_PATH, DINO_PATH, 0.02, DEVICE)
    
    print(f"--- 2. Loading Model ---")
    model = FreeTimeGSModel(dataset.scene_tokens.shape[-1]).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print(f"[Error] Checkpoint not found at {CHECKPOINT_PATH}")
        return

    model.eval()
    
    print("--- 3. Rendering Frames to Disk ---")
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        # 1. 读取并处理 GT (确保偶数尺寸)
        gt_path = dataset.image_paths[i]
        gt_img_raw = cv2.imread(gt_path)
        if gt_img_raw is None: continue
        
        # 保持 BGR 格式以便 cv2.imwrite 保存，不需要转 RGB
        H_orig, W_orig = gt_img_raw.shape[:2]
        H_even = H_orig - (H_orig % 2)
        W_even = W_orig - (W_orig % 2)
        gt_frame_bgr = gt_img_raw[:H_even, :W_even]
        
        # 2. 渲染
        tokens = data["tokens"].unsqueeze(0).to(DEVICE)
        coords = data["coords"].unsqueeze(0).to(DEVICE)
        t = data["t"].unsqueeze(0).to(DEVICE)
        c2w = data["c2w"].unsqueeze(0).to(DEVICE)
        K = data["K"].unsqueeze(0).to(DEVICE)
        
        w2c = torch.linalg.inv(c2w)
        K_norm = K.clone(); K_norm[..., 0, :] /= W_orig; K_norm[..., 1, :] /= H_orig
        
        with torch.no_grad():
            gaussians = model(tokens, coords, t)
            
            render_out, _ = render_3dgs(
                extrinsics=w2c, 
                intrinsics=K_norm, 
                image_shape=(H_even, W_even), 
                gaussian=gaussians, 
                num_view=1, 
                background_color=torch.zeros(1, 3).to(DEVICE)
            )
            
            # Tensor (RGB) -> Numpy (BGR for OpenCV)
            rgb = render_out.squeeze(1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            render_frame_rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            render_frame_bgr = cv2.cvtColor(render_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # 3. 拼接对比图
            compare_frame_bgr = np.concatenate([gt_frame_bgr, render_frame_bgr], axis=1)
            
            # 4. 保存每一帧到磁盘 (使用标准命名 %05d.jpg)
            filename = f"{i:05d}.jpg"
            cv2.imwrite(os.path.join(TMP_RENDER, filename), render_frame_bgr)
            cv2.imwrite(os.path.join(TMP_GT, filename), gt_frame_bgr)
            cv2.imwrite(os.path.join(TMP_COMPARE, filename), compare_frame_bgr)
            
    print("--- 4. Running FFmpeg to stitch videos ---")
    
    # 定义 FFmpeg 命令模板
    # -y: 覆盖输出
    # -framerate 24: 帧率
    # -i .../%05d.jpg: 输入图片序列
    # -c:v libx264: 编码器
    # -pix_fmt yuv420p: 像素格式 (兼容性关键)
    def run_ffmpeg(input_folder, output_file):
        cmd = (
            f"ffmpeg -y -framerate 24 -i {input_folder}/%05d.jpg "
            f"-c:v libx264 -pix_fmt yuv420p {output_file} > /dev/null 2>&1"
        )
        print(f"Generating {output_file} ...")
        ret = os.system(cmd)
        if ret != 0:
            print(f"[Error] FFmpeg failed for {output_file}. Check if ffmpeg is installed.")
        else:
            print(f"[Success] Saved {output_file}")

    run_ffmpeg(TMP_RENDER, f"{OUTPUT_ROOT}/bear_output.mp4")
    run_ffmpeg(TMP_GT, f"{OUTPUT_ROOT}/bear_gt.mp4")
    run_ffmpeg(TMP_COMPARE, f"{OUTPUT_ROOT}/bear_compare.mp4")
    
    print("\nAll Done! Videos are in ./outputs/")
    print(f"Frames are saved in {TMP_RENDER} (You can check individual images if video fails)")

if __name__ == "__main__":
    render_video()