"""
=========================================================
discriminator.py

Physics-Informed Pix2Pix PatchGAN Discriminator

Input:
    Physics Tensor
        Shape: (B, 3, 512, 512)

    RGB Image
        Shape: (B, 3, 512, 512)

Concatenated:
        (B, 6, 512, 512)

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
    PatchGAN Discriminator

    Input:
        physics_input: (B, 3, H, W)
        rgb_image:     (B, 3, H, W)

    Output:
        PatchGAN score map
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # Input = physics (3) + RGB (3) = 6 channels

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
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

            # Final PatchGAN output
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, physics_input, rgb_image):
        """
        Parameters
        ----------
        physics_input : (B, 3, H, W)
        rgb_image     : (B, 3, H, W)

        Returns
        -------
        PatchGAN output map
        """

        x = torch.cat([physics_input, rgb_image], dim=1)
        x = self.initial(x)
        x = self.model(x)

        return x


# =========================================================
# Quick Test
# =========================================================

if __name__ == "__main__":

    model = Discriminator()

    physics = torch.randn(1, 3, 512, 512)
    rgb = torch.randn(1, 3, 512, 512)

    out = model(physics, rgb)

    print("Physics Shape :", physics.shape)
    print("RGB Shape     :", rgb.shape)
    print("Output Shape  :", out.shape)