#!/usr/bin/env python3
"""
train_bc_policy.py

Trains a small behavior-cloning policy on episodes recorded by arm_teleop_gui.py:
  input  = camera frame (+ current joint angles as proprioception)
  output = commanded joint angles (what you set the sliders to)

This is intentionally a lighter-weight approach than full ACT (no transformer,
no action chunking) so it trains reliably on a 6GB laptop GPU. It's a reasonable
first policy; if you want to graduate to LeRobot's ACT (as in your Franka charger
project), this dataset layout converts easily - each episode's frames/ + actions.csv
map directly onto LeRobot's image + state/action columns.

Usage:
    python train_bc_policy.py --dataset ./dataset --epochs 50 --out ./checkpoints

    # train on only specific episodes (e.g. the ones where the task actually
    # succeeded) instead of every episode in the dataset folder:
    python train_bc_policy.py --dataset ./dataset --episodes 0003 --epochs 50 --out ./checkpoints
    python train_bc_policy.py --dataset ./dataset --episodes 0003,0007,0012 --epochs 50 --out ./checkpoints
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

IMG_SIZE = 128


class EpisodeDataset(Dataset):
    def __init__(self, dataset_dir, transform, episode_filter=None):
        """
        episode_filter: optional list of episode identifiers to restrict training
        to (e.g. ["0003"] or ["episode_0003"]). None means use every episode
        found under dataset_dir.
        """
        self.samples = []  # (image_path, state_angles, target_angles)
        self.transform = transform

        episode_dirs = sorted(
            d for d in os.listdir(dataset_dir)
            if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_dir, d))
        )

        if episode_filter is not None:
            wanted = set()
            for e in episode_filter:
                e = e.strip()
                wanted.add(e if e.startswith("episode_") else f"episode_{e}")
            matched = [ep for ep in episode_dirs if ep in wanted]
            missing = wanted - set(matched)
            if missing:
                raise ValueError(f"Requested episode(s) not found in {dataset_dir}: {sorted(missing)}")
            episode_dirs = matched
            print(f"Training on selected episode(s): {episode_dirs}")

        for ep in episode_dirs:
            ep_path = os.path.join(dataset_dir, ep)
            csv_path = os.path.join(ep_path, "actions.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            has_feedback = rows and "fb0" in rows[0]
            for i, row in enumerate(rows):
                img_path = os.path.join(ep_path, "frames", f"frame_{int(row['frame_idx']):05d}.jpg")
                if not os.path.exists(img_path):
                    continue
                angles = [float(row[f"j{k}"]) for k in range(6)]
                if has_feedback:
                    # real measured servo position - what the arm actually did,
                    # not just what we assumed it did
                    prev_angles = [float(row[f"fb{k}"]) for k in range(6)]
                else:
                    # fallback: previous frame's commanded angle (assumes no slippage)
                    prev_angles = angles if i == 0 else [float(rows[i - 1][f"j{k}"]) for k in range(6)]
                self.samples.append((img_path, prev_angles, angles))

        if not self.samples:
            raise RuntimeError(f"No samples found under {dataset_dir}. Record some episodes first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, state, target = self.samples[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        state = torch.tensor(state, dtype=torch.float32) / 180.0
        target = torch.tensor(target, dtype=torch.float32) / 180.0
        return img, state, target


class BCPolicy(nn.Module):
    """Small CNN backbone (frozen-ish ResNet18) + proprioception, regressing 6 joint angles."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for p in list(self.backbone.parameters())[:-20]:  # freeze most of it, fine-tune last layers
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(512 + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid(),  # output in [0,1], scaled back to 0-180 deg outside
        )

    def forward(self, img, state):
        feat = self.backbone(img)
        x = torch.cat([feat, state], dim=1)
        return self.head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./dataset")
    ap.add_argument("--episodes", default=None,
                     help="Comma-separated episode ids to train on, e.g. 0003 or 0003,0007. "
                          "Omit to use every episode in --dataset.")
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    episode_filter = args.episodes.split(",") if args.episodes else None
    full_ds = EpisodeDataset(args.dataset, transform, episode_filter=episode_filter)
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    print(f"Dataset: {len(full_ds)} samples ({n_train} train / {n_val} val)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = BCPolicy().to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for img, state, target in train_loader:
            img, state, target = img.to(device), state.to(device), target.to(device)
            opt.zero_grad()
            pred = model(img, state)
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, state, target in val_loader:
                img, state, target = img.to(device), state.to(device), target.to(device)
                pred = model(img, state)
                val_loss += loss_fn(pred, target).item() * img.size(0)
        val_loss /= len(val_ds)

        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out, "bc_policy_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.out, "bc_policy_last.pt"))
    print(f"Done. Best val loss {best_val:.5f}. Checkpoints saved to {args.out}")


if __name__ == "__main__":
    main()
