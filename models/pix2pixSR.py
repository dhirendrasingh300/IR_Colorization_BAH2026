"""
=========================================================
pix2pix.py

Physics-Informed Pix2Pix Loss Functions for Super-Resolution (1-Ch)
-------------------------------------------------------------------
Contains:
    1. GAN Loss (BCEWithLogitsLoss)
    2. L1 Reconstruction Loss
    3. SSIM Structural Loss (Crucial for Super-Resolution)
    4. Physics Consistency Loss (1-Channel optimized)
    5. Total Generator & Discriminator Losses

Author:
    Bhartiya Antriksh Hackathon 2026
=========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Loss Weights
# =========================================================
L1_LAMBDA = 100
SSIM_LAMBDA = 20       # Structural weight for crisp edges
PHYSICS_LAMBDA = 10


# =========================================================
# SSIM Loss Component (Pure PyTorch Implementation)
# =========================================================
def gaussian_window(window_size, sigma):
    """Generates a 1D Gaussian kernel."""
    gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)]))
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    """Creates a 2D Gaussian window for structural filtering."""
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_window(window_size, channel))

    def forward(self, img1, img2):
        # Dynamically move window to the correct device if needed
        window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # We return (1 - SSIM) as a loss to minimize it
        return 1.0 - ssim_map.mean()


# =========================================================
# Physics-Informed Pix2Pix Loss
# =========================================================
class Pix2PixLoss:
    """
    Implements advanced loss functions optimized for Super-Resolution:
    
    Generator Loss = GAN + (100 * L1) + (20 * SSIM) + (10 * Physics)
    Discriminator Loss = BCE(real) + BCE(fake)
    """

    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=11, channel=1) # 1-Channel SSIM

    # =====================================================
    # Discriminator Loss
    # =====================================================
    def discriminator_loss(self, disc_real, disc_fake):
        real_loss = self.bce(disc_real, torch.ones_like(disc_real))
        fake_loss = self.bce(disc_fake, torch.zeros_like(disc_fake))
        return (real_loss + fake_loss) / 2

    # =====================================================
    # Physics Consistency Loss
    # =====================================================
    def physics_loss(self, fake_img, temperature):
        """
        Since both tensors are pure 1-Channel (B, 1, H, W),
        we directly map structural integrity via L1.
        """
        return self.l1(fake_img, temperature)

    # =====================================================
    # Generator Loss
    # =====================================================
    def generator_loss(self, disc_fake, fake_img, real_img, temperature):
        """
        Calculates the complete total loss with SSIM structural constraint.
        """
        # 1. GAN Loss
        gan_loss = self.bce(disc_fake, torch.ones_like(disc_fake))

        # 2. Pixel-level Reconstruction (L1)
        l1_loss = self.l1(fake_img, real_img)

        # 3. Structural Similarity Index (SSIM) Loss
        ssim_loss = self.ssim(fake_img, real_img)

        # 4. Physics Consistency Loss
        phys_loss = self.physics_loss(fake_img, temperature)

        # Total combined loss
        total_loss = (
            gan_loss +
            L1_LAMBDA * l1_loss +
            SSIM_LAMBDA * ssim_loss +  # New SR feature
            PHYSICS_LAMBDA * phys_loss
        )

        return total_loss, gan_loss, l1_loss, ssim_loss, phys_loss


# =========================================================
# Quick Test
# =========================================================
if __name__ == "__main__":

    loss_fn = Pix2PixLoss()

    # Changing shapes to reflect your exact pure 1-Channel setup (B, 1, H, W)
    fake_img = torch.rand(2, 1, 512, 512)     # Outputs must be [0, 1] range for SSIM
    real_img = torch.rand(2, 1, 512, 512)
    temperature = torch.rand(2, 1, 512, 512)
    disc_fake = torch.randn(2, 1, 63, 63)      # Output grid from your new 1-Ch Discriminator

    total, gan, l1, ssim_val, phys = loss_fn.generator_loss(
        disc_fake,
        fake_img,
        real_img,
        temperature
    )

    print("-" * 50)
    print("SR LOSS EVALUATION (1-Channel)")
    print("-" * 50)
    print(f"Total Generator Loss : {total.item():.4f}")
    print(f"GAN Loss             : {gan.item():.4f}")
    print(f"L1 Pixel Loss        : {l1.item():.4f}")
    print(f"SSIM Structural Loss : {ssim_val.item():.4f}  <-- Added for Super-Resolution!")
    print(f"Physics Loss         : {phys.item():.4f}")
    print("-" * 50)