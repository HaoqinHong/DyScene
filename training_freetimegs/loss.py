import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision import transforms
import os

# === Robust Import ===
try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[Warning] 'transformers' library not found. Semantic Loss disabled.")
    TRANSFORMERS_AVAILABLE = False

class DINOMetricLoss(nn.Module):
    def __init__(self, model_path='facebook/dinov2-base', device='cuda'):
        super().__init__()
        self.device = device
        self.disabled = not TRANSFORMERS_AVAILABLE
        
        if self.disabled: return

        print(f"[Critic] Loading DINOv2 for Dense Feature Extraction: {model_path} ...")
        try:
            self.dino = AutoModel.from_pretrained(model_path).to(device)
            self.dino.eval()
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.patch_size = 14 # DINOv2 Patch Size
            self.embed_dim = self.dino.config.hidden_size 
            self.mean = self.processor.image_mean
            self.std = self.processor.image_std
        except Exception as e:
            print(f"[Error] Failed to load DINOv2: {e}")
            self.disabled = True
            return
            
        for p in self.dino.parameters(): p.requires_grad = False
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def preprocess(self, image_tensor):
        # image_tensor: [B, 3, H, W] (0-1)
        _, _, H, W = image_tensor.shape
        # Resize to multiple of patch_size
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        x = F.interpolate(image_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
        return self.normalize(x), new_H, new_W

    def get_dense_features(self, image_tensor):
        """
        返回像素级对齐的特征图 [B, C, H, W]
        """
        if self.disabled: return None
        
        with torch.no_grad():
            img_in, H, W = self.preprocess(image_tensor)
            outputs = self.dino(pixel_values=img_in, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state[:, 1:, :] # Skip CLS
            
            h_feat = H // self.patch_size
            w_feat = W // self.patch_size
            
            # [B, h*w, D] -> [B, D, h, w]
            features = last_hidden.permute(0, 2, 1).reshape(-1, self.embed_dim, h_feat, w_feat)
            
            # Upsample back to original resolution
            dense_features = F.interpolate(features, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
            dense_features = F.normalize(dense_features, dim=1, p=2)
            
            return dense_features

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