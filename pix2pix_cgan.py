"""
Task-04: Image-to-Image Translation with cGAN (pix2pix)
========================================================
Implements the pix2pix conditional GAN for image-to-image translation.

References:
  #1 - Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
  #2 - https://arxiv.org/abs/1611.07004
  #3 - https://phillipi.github.io/pix2pix/

Usage:
    # Train
    python pix2pix_cgan.py --mode train --data_dir ./data/facades --epochs 200

    # Test / Inference
    python pix2pix_cgan.py --mode test --data_dir ./data/facades --checkpoint ./checkpoints/gen_epoch0200.pth
"""

import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class Pix2PixDataset(Dataset):
    """
    Loads paired images stored side-by-side (left = input A, right = target B).
    Standard format used by the official pix2pix datasets (facades, maps, etc.).

    Download a dataset with:
        bash download_dataset.sh facades
    """

    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256):
        folder = Path(root_dir) / split
        self.files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"No images found in {folder}")

        self.img_size   = img_size
        self.jitter_sz  = int(img_size * 1.117)   # 286 for 256-px (paper default)
        self.to_tensor  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pair  = Image.open(self.files[idx]).convert("RGB")
        w, h  = pair.size
        half  = w // 2
        img_A = pair.crop((0,    0, half, h))   # input  (e.g. label map)
        img_B = pair.crop((half, 0, w,    h))   # target (e.g. photo)

        img_A, img_B = self._augment(img_A, img_B)
        return self.to_tensor(img_A), self.to_tensor(img_B)

    def _augment(self, a: Image.Image, b: Image.Image):
        """Random jitter + crop + horizontal flip (applied identically to both halves)."""
        resize = transforms.Resize((self.jitter_sz, self.jitter_sz), Image.BICUBIC)
        a, b   = resize(a), resize(b)

        i = np.random.randint(0, self.jitter_sz - self.img_size)
        j = np.random.randint(0, self.jitter_sz - self.img_size)
        s = self.img_size
        a = a.crop((j, i, j + s, i + s))
        b = b.crop((j, i, j + s, i + s))

        if np.random.random() > 0.5:
            a = a.transpose(Image.FLIP_LEFT_RIGHT)
            b = b.transpose(Image.FLIP_LEFT_RIGHT)

        return a, b


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATOR  —  U-Net with skip connections
# ─────────────────────────────────────────────────────────────────────────────

class UNetBlock(nn.Module):
    """Single encoder–decoder block with an optional skip connection."""

    def __init__(self, outer_ch, inner_ch, input_ch=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_ch is None:
            input_ch = outer_ch

        down_conv = nn.Conv2d(input_ch, inner_ch, 4, 2, 1, bias=False)
        down_relu = nn.LeakyReLU(0.2, inplace=True)
        down_norm = norm_layer(inner_ch)
        up_relu   = nn.ReLU(inplace=True)
        up_norm   = norm_layer(outer_ch)

        if outermost:
            up_conv = nn.ConvTranspose2d(inner_ch * 2, outer_ch, 4, 2, 1)
            model   = [down_conv] + [submodule] + [up_relu, up_conv, nn.Tanh()]
        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_ch, outer_ch, 4, 2, 1, bias=False)
            model   = [down_relu, down_conv] + [up_relu, up_conv, up_norm]
        else:
            up_conv = nn.ConvTranspose2d(inner_ch * 2, outer_ch, 4, 2, 1, bias=False)
            down    = [down_relu, down_conv, down_norm]
            up      = [up_relu, up_conv, up_norm]
            if use_dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)   # skip connection


class UNetGenerator(nn.Module):
    """
    U-Net Generator (8 levels for 256×256 images).
    Encoder:  C64-C128-C256-C512-C512-C512-C512-C512
    Decoder:  CD512-CD512-CD512-C512-C256-C128-C64
    """

    def __init__(self, in_ch=3, out_ch=3, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=True):
        super().__init__()

        # Build from innermost → outermost
        block = UNetBlock(ngf * 8, ngf * 8, submodule=None,
                          innermost=True, norm_layer=norm_layer)

        for _ in range(num_downs - 5):                      # 3 intermediate blocks
            block = UNetBlock(ngf * 8, ngf * 8, submodule=block,
                              norm_layer=norm_layer, use_dropout=use_dropout)

        block = UNetBlock(ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer)
        block = UNetBlock(ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer)
        block = UNetBlock(ngf,     ngf * 2, submodule=block, norm_layer=norm_layer)
        block = UNetBlock(out_ch,  ngf, input_ch=in_ch, submodule=block,
                          outermost=True, norm_layer=norm_layer)

        self.model = block

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DISCRIMINATOR  —  70×70 PatchGAN
# ─────────────────────────────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN Discriminator.
    Conditions on the input image by concatenating it with the target/generated image.
    Outputs a grid of real/fake predictions, each covering a 70×70 receptive field.
    """

    def __init__(self, in_ch=3, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch * 2, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf      = min(nf * 2, 512)
            stride  = 2 if n < n_layers - 1 else 1
            layers += [
                nn.Conv2d(nf_prev, nf, 4, stride=stride, padding=1, bias=False),
                norm_layer(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_prev = nf
        nf      = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, 1, 1, bias=False),
            norm_layer(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, 1, 1),   # 1-channel patch map
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        return self.model(torch.cat([img_A, img_B], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. WEIGHT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def weights_init(m):
    """Gaussian init for Conv layers, constant init for BatchNorm (per paper)."""
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Dataset: {args.data_dir}  |  img_size={args.img_size}  "
          f"batch={args.batch_size}  epochs={args.epochs}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = Pix2PixDataset(args.data_dir, split="train", img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True)

    # ── Models ────────────────────────────────────────────────────────────────
    G = UNetGenerator(in_ch=3, out_ch=3).to(device)
    D = PatchGANDiscriminator(in_ch=3).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # ── Losses ────────────────────────────────────────────────────────────────
    crit_GAN = nn.BCEWithLogitsLoss()
    crit_L1  = nn.L1Loss()

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Linear decay: constant for first half, linearly decay to 0 for second half
    decay_start = args.epochs // 2

    def lr_lambda(epoch):
        if decay_start == 0 or epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / float(decay_start))

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir,     exist_ok=True)

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()

        for step, (real_A, real_B) in enumerate(loader):
            real_A, real_B = real_A.to(device), real_B.to(device)
            fake_B = G(real_A)

            # ── Discriminator update ─────────────────────────────────────────
            opt_D.zero_grad()
            pred_real  = D(real_A, real_B)
            pred_fake  = D(real_A, fake_B.detach())
            loss_D = 0.5 * (crit_GAN(pred_real,  torch.ones_like(pred_real)) +
                            crit_GAN(pred_fake, torch.zeros_like(pred_fake)))
            loss_D.backward()
            opt_D.step()

            # ── Generator update ─────────────────────────────────────────────
            opt_G.zero_grad()
            pred_fake_G = D(real_A, fake_B)
            loss_G_adv  = crit_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
            loss_G_L1   = crit_L1(fake_B, real_B) * args.lambda_l1
            loss_G      = loss_G_adv + loss_G_L1
            loss_G.backward()
            opt_G.step()

            if step % 100 == 0:
                print(f"  Epoch [{epoch:>3}/{args.epochs}]  "
                      f"Step [{step:>4}/{len(loader)}]  "
                      f"D: {loss_D.item():.4f}  "
                      f"G: {loss_G_adv.item():.4f}  "
                      f"L1: {loss_G_L1.item():.4f}")

        # ── Periodic save ─────────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            with torch.no_grad():
                grid = torch.cat([real_A[:4], fake_B[:4], real_B[:4]], dim=0)
                save_image(grid * 0.5 + 0.5,
                           f"{args.sample_dir}/epoch_{epoch:04d}.png", nrow=4)
            torch.save(G.state_dict(),
                       f"{args.checkpoint_dir}/gen_epoch{epoch:04d}.pth")
            torch.save(D.state_dict(),
                       f"{args.checkpoint_dir}/disc_epoch{epoch:04d}.pth")
            print(f"  ✓ Saved checkpoint & sample for epoch {epoch}")

        sched_G.step(); sched_D.step()

    print("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = UNetGenerator(in_ch=3, out_ch=3).to(device)
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G.eval()
    print(f"Loaded generator from: {args.checkpoint}")

    test_ds = Pix2PixDataset(args.data_dir, split="test", img_size=args.img_size)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    os.makedirs(args.results_dir, exist_ok=True)

    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(test_dl):
            real_A = real_A.to(device)
            fake_B = G(real_A)
            # Save side-by-side: Input | Generated | Ground Truth
            result = torch.cat([real_A, fake_B, real_B.to(device)], dim=3)
            save_image(result * 0.5 + 0.5,
                       f"{args.results_dir}/result_{i:04d}.png")

    print(f"Saved {len(test_ds)} results to '{args.results_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="pix2pix cGAN — Image-to-Image Translation (Isola et al., 2017)")

    p.add_argument("--mode",           default="train",
                   choices=["train", "test"])
    p.add_argument("--data_dir",       default="./data/facades",
                   help="Dataset root (must contain train/ and test/ sub-dirs)")
    p.add_argument("--img_size",       type=int,   default=256)
    p.add_argument("--batch_size",     type=int,   default=1,
                   help="Paper uses batch_size=1 (instance norm) or 4 (batch norm)")
    p.add_argument("--epochs",         type=int,   default=200,
                   help="Total epochs; LR decays linearly in the second half")
    p.add_argument("--lr",             type=float, default=2e-4,
                   help="Adam learning rate (paper: 2e-4)")
    p.add_argument("--lambda_l1",      type=float, default=100.0,
                   help="Weight for L1 reconstruction loss (paper: 100)")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--sample_dir",     default="./samples")
    p.add_argument("--results_dir",    default="./results")
    p.add_argument("--save_every",     type=int,   default=10)
    # Test-only
    p.add_argument("--checkpoint",     default=None,
                   help="Path to generator .pth file (required for --mode test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        if not args.checkpoint:
            raise ValueError("Provide --checkpoint <path> for test mode.")
        test(args)
