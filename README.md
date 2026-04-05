# Task-04: Image-to-Image Translation with cGAN (pix2pix)

> **Reference paper**: *Image-to-Image Translation with Conditional Adversarial Networks*  
> Isola et al., CVPR 2017 — https://arxiv.org/abs/1611.07004

---

## Architecture Overview

```
Input (A)  ──►  U-Net Generator  ──►  Fake B
                      │
          ┌───────────┴───────────┐
          │   PatchGAN            │
          │   Discriminator       │
          │                       │
Real (A,B)──► D_real (→ 1)       │
Fake (A,fake_B)─► D_fake (→ 0)   │
          └───────────────────────┘
```

### Generator — U-Net (8 levels for 256×256)
| Level | Encoder       | Decoder         |
|-------|---------------|-----------------|
| 1     | C64           | C64 + skip      |
| 2     | C128          | C128 + skip     |
| 3     | C256          | C256 + skip     |
| 4–7   | C512 ×4       | CD512 ×4 + skip |
| 8     | C512 (inner)  | —               |

Skip connections concatenate encoder feature maps to the corresponding decoder layer.

### Discriminator — 70×70 PatchGAN
- Takes `[input_A | target_B]` concatenated along channels
- Classifies overlapping 70×70 image patches as real/fake
- Produces a spatial grid of predictions (not a single scalar)

### Loss Functions
```
L_D   = 0.5 * [BCE(D(A,B), 1) + BCE(D(A,G(A)), 0)]
L_G   = BCE(D(A,G(A)), 1)  +  λ * L1(G(A), B)
λ = 100  (per paper)
```

---

## Installation

```bash
pip install torch torchvision pillow numpy
```

---

## Dataset Setup

Download any of the official paired datasets:

```bash
# Facades (architectural labels → photos)
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
tar -xzf facades.tar.gz -C ./data/

# Other options: maps, cityscapes, edges2shoes, edges2handbags, night2day
```

Each dataset has the structure:
```
data/
  facades/
    train/   ← paired images (left=A, right=B) side-by-side
    test/
    val/
```

---

## Training

```bash
python pix2pix_cgan.py \
    --mode train \
    --data_dir ./data/facades \
    --epochs 200 \
    --batch_size 1 \
    --lr 2e-4 \
    --lambda_l1 100 \
    --save_every 10
```

**Key hyperparameters (paper defaults):**
| Parameter     | Value  | Notes                                    |
|---------------|--------|------------------------------------------|
| `batch_size`  | 1      | Instance-level training                  |
| `lr`          | 2e-4   | Adam with β₁=0.5, β₂=0.999             |
| `lambda_l1`   | 100    | L1 weight encourages sharp outputs       |
| `epochs`      | 200    | LR constant for ep 1–100, decays 101–200 |

Samples are saved to `./samples/epoch_XXXX.png` (rows: input | fake | real).  
Checkpoints saved to `./checkpoints/gen_epochXXXX.pth`.

---

## Inference

```bash
python pix2pix_cgan.py \
    --mode test \
    --data_dir ./data/facades \
    --checkpoint ./checkpoints/gen_epoch0200.pth \
    --results_dir ./results
```

Output: side-by-side PNGs → `Input | Generated | Ground Truth`

---

## File Structure

```
pix2pix_cgan.py
│
├── Pix2PixDataset          # Paired image loader with jitter/crop/flip augmentation
├── UNetBlock               # Single encoder–decoder block with skip connections  
├── UNetGenerator           # Full 8-level U-Net
├── PatchGANDiscriminator   # 70×70 PatchGAN
├── weights_init            # Gaussian weight initialisation
├── train()                 # Full training loop with LR scheduling
└── test()                  # Inference on test set
```

---

## Expected Results (Facades dataset)

| Metric | ~Epoch 50 | ~Epoch 200 |
|--------|-----------|------------|
| D loss | ~0.45     | ~0.35–0.50 |
| G loss | ~0.80     | ~0.60–0.90 |
| L1 loss| ~25       | ~15–20     |

Visually: by epoch 50 generated images show rough structure; by epoch 200 textures and windows are substantially more realistic.
