import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
import lpips 

def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

class DINOMetricLoss(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.device = device
        self.disabled = False
        
        # 1. DINO (语义特征)
        print(f"[Critic] Loading DINOv2 (Base): {model_path} ...")
        try:
            # [FIX] 使用 dinov2_vitb14 (768维) 以匹配 FeatureAligner 的设定
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', source='github').to(device)
            self.dino.eval()
        except Exception as e:
            print(f"[Warning] Failed to load DINO: {e}. Semantic loss disabled.")
            self.dino = None
            self.disabled = True
            
        # 2. LPIPS (感知纹理)
        print(f"[Critic] Loading LPIPS (VGG)...")
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.lpips_loss.eval()

    def get_dense_features(self, image_tensor):
        if self.disabled or self.dino is None:
            return None
        
        img_in, H, W = self.preprocess(image_tensor)
        with torch.no_grad():
            features_dict = self.dino.forward_features(img_in)
            patch_tokens = features_dict['x_norm_patchtokens'] 
            
        B, N, C = patch_tokens.shape
        h_feat, w_feat = H // 14, W // 14
        dense_feat = patch_tokens.view(B, h_feat, w_feat, C).permute(0, 3, 1, 2)
        return F.interpolate(dense_feat, size=(H, W), mode='bilinear', align_corners=False)

    def preprocess(self, image_tensor):
        _, _, H, W = image_tensor.shape
        new_H = (H // 14) * 14
        new_W = (W // 14) * 14
        img_resized = F.interpolate(image_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_norm = (img_resized - mean) / std
        return img_norm, new_H, new_W

    def compute_lpips(self, render, gt):
        render_norm = render * 2.0 - 1.0
        gt_norm = gt * 2.0 - 1.0
        return self.lpips_loss(render_norm, gt_norm).mean()