import os
import torch
import numpy as np
import argparse
from safetensors.torch import load_file
# 确保能导入 depth_anything_3
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    import sys
    sys.path.append("src") # 如果您在项目根目录
    from depth_anything_3.api import DepthAnything3

def main():
    parser = argparse.ArgumentParser(description="DA3 Inference Script")
    parser.add_argument("--read_dir", type=str, default='./demo/bear', help="Input image directory")
    parser.add_argument("--save_dir", type=str, default='./demo/bear_test/results', help="Output directory")
    parser.add_argument("--model_name", type=str, default="da3-giant", help="Model preset name (e.g., da3-giant, da3-large)")
    parser.add_argument("--weight_path", type=str, default=None, help="Optional: Path to local .safetensors or .pth weight file")
    parser.add_argument("--process_res", type=int, default=504, help="Processing resolution")
    args = parser.parse_args()

    # 1. 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 2. 初始化模型
    print(f"Initializing model structure: {args.model_name}...")
    # 注意：我们这里直接用构造函数初始化结构，不下载权重（如果提供了本地权重）
    # 如果没有提供本地权重，则尝试从 HF 加载默认权重
    if args.weight_path:
        # 仅初始化结构
        model = DepthAnything3(model_name=args.model_name)
        
        # 加载本地权重
        print(f"Loading local weights from: {args.weight_path}")
        if args.weight_path.endswith(".safetensors"):
            state_dict = load_file(args.weight_path)
        else:
            state_dict = torch.load(args.weight_path, map_location="cpu")
            
        # 处理可能的 key 不匹配 (例如前缀 model.)
        # DA3 的权重通常直接匹配，但有时可能需要处理
        model.load_state_dict(state_dict, strict=False)
    else:
        # 使用 HuggingFace 预训练权重
        try:
            # 映射 model_name 到 HF repo (简化处理，通常需要对应的 repo id)
            # 这里为了演示，如果未指定路径，尝试用 from_pretrained 加载一个默认的
            hf_repo = "depth-anything/DA3-GIANT" if "giant" in args.model_name else "depth-anything/Depth-Anything-V2-Large"
            print(f"Loading from HuggingFace: {hf_repo}")
            model = DepthAnything3.from_pretrained(hf_repo) 
        except Exception as e:
            print(f"HF Load failed: {e}. Fallback to local init (random weights).")
            model = DepthAnything3(model_name=args.model_name)

    model = model.to(device)
    model.eval()

    # 3. 读取图片 (排序很重要)
    if not os.path.exists(args.read_dir):
        print(f"Error: Input directory {args.read_dir} does not exist.")
        return

    frames = []
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
    files = sorted(os.listdir(args.read_dir)) # 确保按文件名顺序 000, 001...
    
    for file in files:
        if file.lower().endswith(supported_exts):
            frames.append(os.path.join(args.read_dir, file))

    if not frames:
        print("No images found.")
        return
        
    print(f"Processing {len(frames)} frames...")

    # 4. 设置导出参数
    export_kwargs = {
        "gs_ply": {"gs_views_interval": 1},
        "gs_video": {
            "trj_mode": "interpolate_smooth",
            "chunk_size": 1, 
            "vis_depth": "hcat",
            "enable_tqdm": True
        },
    }

    # 5. 运行推理
    with torch.no_grad():
        model.inference(
            image=frames,
            align_to_input_ext_scale=True,
            infer_gs=True, # 开启 Gaussian 分支以获取相机位姿优化
            process_res=args.process_res,
            export_dir=args.save_dir,
            export_format="mini_npz-glb-gs_ply", # mini_npz 包含关键的位姿信息
            conf_thresh_percentile=40.0,
            num_max_points=1_000_000,
            show_cameras=True,
            export_kwargs=export_kwargs
        )

    print(f"Done! Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()