import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision import transforms
try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    raise ImportError("请安装 transformers 库: pip install transformers")

class DINOMetricLoss(nn.Module):
    def __init__(self, model_path='facebook/dinov2-base', device='cuda'):
        super().__init__()
        self.device = device
        print(f"[Critic] Loading DINOv2 (Spatial Mode) from: {model_path} ...")
        
        self.dino = AutoModel.from_pretrained(model_path).to(device)
        self.dino.eval()
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.mean = self.processor.image_mean
            self.std = self.processor.image_std
        except:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            
        for p in self.dino.parameters(): p.requires_grad = False
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def preprocess(self, image_tensor):
        # 强制 Resize 到固定大小 (例如 336x336)，确保 Patch 数量一致
        # 336 / 14 = 24 -> 24x24 = 576 patches
        target_size = (336, 336) 
        x = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
        return self.normalize(x)

    def forward(self, render_rgb, gt_rgb=None, gt_feats=None):
        # 1. 提取 Render 的特征
        render_in = self.preprocess(render_rgb)
        render_out = self.dino(pixel_values=render_in)
        
        # [B, N_patches, C] -> 去掉第一个 CLS Token (index 0)
        # 提取空间特征
        feat_render = render_out.last_hidden_state[:, 1:, :] 
        feat_render = F.normalize(feat_render, dim=-1, p=2)

        # 2. 获取 GT 的特征
        if gt_feats is not None:
            # 如果 Dataset 里已经存好了，直接用
            feat_gt = gt_feats.to(self.device)
        else:
            with torch.no_grad():
                gt_in = self.preprocess(gt_rgb)
                gt_out = self.dino(pixel_values=gt_in)
                feat_gt = gt_out.last_hidden_state[:, 1:, :]
                feat_gt = F.normalize(feat_gt, dim=-1, p=2)

        # 3. 计算 Patch 级别的相似度
        # 我们希望渲染图的每个 patch 和 GT 图对应的 patch 语义一致
        # Loss = 1 - mean(cosine_similarity)
        similarity = (feat_render * feat_gt).sum(dim=-1) # [B, N_patches]
        
        return 1.0 - similarity.mean()

# === SSIM Helper Functions ===
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    if img1.is_cuda: window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)