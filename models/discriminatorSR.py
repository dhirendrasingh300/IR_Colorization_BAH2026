"""
=========================================================
discriminator.py

Physics-Informed Pix2Pix PatchGAN Discriminator (Pure 1-Ch)
---------------------------------------------------------

Input:
    Physics Tensor (Single Channel)
        Shape: (B, 1, 512, 512)

    Target Image (Single Channel Grayscale/Thermal)
        Shape: (B, 1, 512, 512)

Concatenated:
        Shape: (B, 2, 512, 512)

Output:
    PatchGAN Real/Fake Map

Author:
    Bhartiya Antriksh Hackathon 2026
=========================================================
"""

import torch
import torch.nn as nn


# =========================================================
# Discriminator Block
# =========================================================

class CNNBlock(nn.Module):
    """
    Basic PatchGAN block:
        Conv2D → BatchNorm → LeakyReLU
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# PatchGAN Discriminator
# =========================================================

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator configured for 1-channel Physics input 
    and 1-channel Target image input.

    Input:
        physics_input: (B, 1, H, W)
        target_image:  (B, 1, H, W)

    Output:
        PatchGAN score map
    """

    def __init__(self, in_channels=1, out_channels=1): # CHANGES: out_channels set to 1
        super().__init__()

        # Total channels = 1 (Physics) + 1 (Target) = 2 channels
        total_input_channels = in_channels + out_channels

        self.initial = nn.Sequential(
            nn.Conv2d(
                total_input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = nn.Sequential(

            # 256 → 128
            CNNBlock(64, 128, stride=2),

            # 128 → 64
            CNNBlock(128, 256, stride=2),

            # 64 → 64 (no downsampling)
            CNNBlock(256, 512, stride=1),

            # Final PatchGAN output layer
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, physics_input, target_image):
        """
        Parameters
        ----------
        physics_input : (B, 1, H, W)
        target_image  : (B, 1, H, W)

        Returns
        -------
        PatchGAN output map
        """
        # Channel dimension (dim=1) par concatenate karke size (B, 2, 512, 512) banega
        x = torch.cat([physics_input, target_image], dim=1)
        x = self.initial(x)
        x = self.model(x)

        return x


# =========================================================
# Quick Test
# =========================================================

if __name__ == "__main__":

    # Dono side 1-1 channel setup
    model = Discriminator(in_channels=1, out_channels=1)

    # Dono tensors ab pure 1-channel hain
    physics = torch.randn(1, 1, 512, 512)
    target = torch.randn(1, 1, 512, 512)   # Target image is now 1 channel

    out = model(physics, target)

    print("Physics Shape :", physics.shape)
    print("Target Shape  :", target.shape)
    print("Output Shape  :", out.shape)