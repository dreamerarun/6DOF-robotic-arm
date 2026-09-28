#!/usr/bin/env python3
"""
convert_to_lerobot_dataset.py

Converts episodes recorded by arm_teleop_gui.py (frames/*.jpg + actions.csv per
episode) into a LeRobotDataset - the format SmolVLA, ACT, and every other
LeRobot policy expects for fine-tuning.

NOTE ON API STABILITY: LeRobot's dataset API has changed across versions (v2 ->
v2.1 -> v3) and may change again. This script follows the current documented
pattern (LeRobotDataset.create -> add_frame -> save_episode -> finalize), with
"task" passed as a key inside the frame dict rather than a separate add_frame
argument - that's the calling convention as of lerobot 0.4.x. If it errors on
your installed version, check `pip show lerobot` against
https://huggingface.co/docs/lerobot - the fix is almost always a small
signature change, not a rethink of the approach below.

Usage:
    pip install "lerobot[smolvla]"
    python convert_to_lerobot_dataset.py \
        --dataset ./dataset \
        --episodes 0003,0004,0007 \
        --repo_id yourname/arm-pickplace \
        --task "pick up the object and place it in the box" \
        --push_to_hub

Omit --episodes to convert every episode. Omit --push_to_hub to keep it
purely local (SmolVLA training can then point at it via --dataset.root,
see SMOLVLA_TRAINING.md).
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def detect_frame_shape(dataset_dir, episodes):
    """Read the first available frame to get the real (height, width) - frames are
    saved at the camera's native resolution, not the GUI preview's 400x300."""
    for ep in episodes:
        frames_dir = os.path.join(dataset_dir, ep, "frames")
        if not os.path.isdir(frames_dir):
            continue
        files = sorted(os.listdir(frames_dir))
        if files:
            img = np.array(Image.open(os.path.join(frames_dir, files[0])).convert("RGB"))
            return img.shape[0], img.shape[1]  # height, width
    raise RuntimeError("Could not find any frame to detect image size from.")


def list_episodes(dataset_dir, episode_filter):
    all_eps = sorted(
        d for d in os.listdir(dataset_dir)
        if d.startswith("episode_") and os.path.isdir(os.path.join(dataset_dir, d))
    )
    if episode_filter is None:
        return all_eps
    wanted = {e if e.startswith("episode_") else f"episode_{e}" for e in episode_filter}
    matched = [e for e in all_eps if e in wanted]
    missing = wanted - set(matched)
    if missing:
        raise ValueError(f"Requested episode(s) not found: {sorted(missing)}")
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./dataset")
    ap.add_argument("--episodes", default=None, help="Comma-separated episode ids, e.g. 0003,0004")
    ap.add_argument("--repo_id", required=True, help="e.g. yourname/arm-pickplace")
    ap.add_argument("--task", required=True, help="Natural-language task description, e.g. "
                     "'pick up the block and place it in the tray'")
    ap.add_argument("--fps", type=int, default=15)  # matches the ~66ms loop in arm_teleop_gui.py
    ap.add_argument("--push_to_hub", action="store_true")
    args = ap.parse_args()

    episode_filter = args.episodes.split(",") if args.episodes else None
    episodes = list_episodes(args.dataset, episode_filter)
    if not episodes:
        raise RuntimeError(f"No episodes found under {args.dataset}")
    print(f"Converting {len(episodes)} episode(s): {episodes}")

    img_h, img_w = detect_frame_shape(args.dataset, episodes)
    print(f"Detected frame size: {img_w}x{img_h}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="custom_6dof_arm",
        features={
            "observation.images.cam": {
                "dtype": "video",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["j0", "j1", "j2", "j3", "j4", "j5"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["j0", "j1", "j2", "j3", "j4", "j5"],
            },
        },
    )

    for ep in episodes:
        ep_path = os.path.join(args.dataset, ep)
        csv_path = os.path.join(ep_path, "actions.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        has_feedback = rows and "fb0" in rows[0]

        print(f"  {ep}: {len(rows)} frames" + (" (using real servo feedback as state)" if has_feedback else ""))

        for i, row in enumerate(rows):
            img_path = os.path.join(ep_path, "frames", f"frame_{int(row['frame_idx']):05d}.jpg")
            if not os.path.exists(img_path):
                continue
            img = np.array(Image.open(img_path).convert("RGB"))

            action = np.array([float(row[f"j{k}"]) for k in range(6)], dtype=np.float32)
            if has_feedback:
                state = np.array([float(row[f"fb{k}"]) for k in range(6)], dtype=np.float32)
            else:
                prev = rows[i - 1] if i > 0 else row
                state = np.array([float(prev[f"j{k}"]) for k in range(6)], dtype=np.float32)

            dataset.add_frame(
                {
                    "observation.images.cam": img,
                    "observation.state": state,
                    "action": action,
                    "task": args.task,
                }
            )

        dataset.save_episode()

    dataset.finalize()
    print(f"Done. Local dataset written under the LeRobot cache for repo_id={args.repo_id}")

    if args.push_to_hub:
        dataset.push_to_hub()
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
