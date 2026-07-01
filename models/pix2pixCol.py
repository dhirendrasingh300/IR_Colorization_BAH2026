"""
=========================================================
pix2pix.py

Physics-Informed Pix2Pix Loss Functions

Contains:
    1. GAN Loss
    2. L1 Reconstruction Loss
    3. Physics Consistency Loss
    4. Total Generator Loss
    5. Discriminator Loss

Author:
    Bhartiya Antriksh Hackathon 2026
=========================================================
"""

import torch
import torch.nn as nn


# =========================================================
# Loss Weights
# =========================================================

L1_LAMBDA = 100
PHYSICS_LAMBDA = 10


# =========================================================
# Physics-Informed Pix2Pix Loss
# =========================================================

class Pix2PixLoss:
    """
    Implements:

    Generator Loss:
        GAN
        + 100 * L1
        + 10 * Physics

    Discriminator Loss:
        BCE(real)
        +
        BCE(fake)
    """

    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    # =====================================================
    # Discriminator Loss
    # =====================================================

    def discriminator_loss(self, disc_real, disc_fake):
        """
        Real pair -> 1
        Fake pair -> 0
        """

        real_loss = self.bce(
            disc_real,
            torch.ones_like(disc_real)
        )

        fake_loss = self.bce(
            disc_fake,
            torch.zeros_like(disc_fake)
        )

        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    # =====================================================
    # Physics Consistency Loss
    # =====================================================

    def physics_loss(self, fake_rgb, temperature):
        """
        Compare generated RGB structure
        against temperature map.

        fake_rgb: (B, 3, H, W)
        temperature: (B, H, W)
        """

        gray = (
            0.299 * fake_rgb[:, 0] +
            0.587 * fake_rgb[:, 1] +
            0.114 * fake_rgb[:, 2]
        )

        return self.l1(gray, temperature)

    # =====================================================
    # Generator Loss
    # =====================================================

    def generator_loss(self, disc_fake, fake_rgb, real_rgb, temperature):
        """
        Total Generator Loss
        """

        # GAN LOSS
        gan_loss = self.bce(
            disc_fake,
            torch.ones_like(disc_fake)
        )

        # L1 RECONSTRUCTION LOSS
        l1_loss = self.l1(fake_rgb, real_rgb)

        # PHYSICS LOSS
        phys_loss = self.physics_loss(fake_rgb, temperature)

        # TOTAL LOSS
        total_loss = (
            gan_loss +
            L1_LAMBDA * l1_loss +
            PHYSICS_LAMBDA * phys_loss
        )

        return total_loss, gan_loss, l1_loss, phys_loss


# =========================================================
# Quick Test
# =========================================================

if __name__ == "__main__":

    loss_fn = Pix2PixLoss()

    fake_rgb = torch.randn(2, 3, 512, 512)
    real_rgb = torch.randn(2, 3, 512, 512)
    temperature = torch.randn(2, 512, 512)
    disc_fake = torch.randn(2, 30, 30)

    total, gan, l1, phys = loss_fn.generator_loss(
        disc_fake,
        fake_rgb,
        real_rgb,
        temperature
    )

    print()
    print("Total Loss :", total.item())
    print("GAN Loss   :", gan.item())
    print("L1 Loss    :", l1.item())
    print("Physics    :", phys.item())


    