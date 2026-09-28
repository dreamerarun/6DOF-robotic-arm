#!/usr/bin/env python3
"""
run_policy.py

Closed-loop autonomous execution: loads a trained bc_policy checkpoint and drives
the real arm from live camera frames, at a fixed control rate.

If the policy is jerky (jumping around instead of moving smoothly), that's a
known failure mode of single-step behavior cloning: each frame predicts an
angle independently, with no memory of the trend, so per-frame noise in the
prediction shows up directly as motion. Two things fight this here:
  --smoothing   : low-pass filter (EMA) blending each new prediction with the
                  last commanded pose, instead of jumping straight to it.
  --max_step_deg: hard cap on how far any joint can move in one control step,
                  regardless of what the model predicts.
Lower --hz also helps (fewer, more decisive steps beat many noisy ones).
If it's still jerky after tuning these, the real fix is more/better training
data and eventually action-chunking (see run_policy.py module docstring below
for the chunking note) - this smoothing is a mitigation, not a cure.

Usage:
    python run_policy.py --port /dev/ttyUSB0 --camera 0 --checkpoint ./checkpoints/bc_policy_best.pt
    python run_policy.py --port /dev/ttyUSB0 --camera 0 --checkpoint ./checkpoints/bc_policy_best.pt \
        --smoothing 0.2 --max_step_deg 5 --hz 8

Press Ctrl+C to stop. The arm holds its last commanded pose on exit.
"""

import argparse
import time

import cv2
import numpy as np
import serial
import torch
import torchvision.transforms as T
from PIL import Image

from train_bc_policy import BCPolicy, IMG_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--smoothing", type=float, default=0.3,
                     help="EMA blend factor in (0,1]. Lower = smoother/slower to react, "
                          "higher = snappier/more prone to jitter. 1.0 disables smoothing.")
    ap.add_argument("--max_step_deg", type=float, default=8.0,
                     help="Max degrees any joint may move in a single control step, "
                          "regardless of prediction. Caps sudden jumps.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2.0)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    state = np.array([90.0] * 6)  # start from home; matches the Arduino sketch's HOME_POSE
    period = 1.0 / args.hz

    print("Running policy. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            img = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            state_t = torch.tensor(state / 180.0, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(img, state_t).cpu().numpy()[0] * 180.0
            pred = np.clip(pred, 0, 180)

            # low-pass filter toward the raw prediction, then cap the per-step jump -
            # this is what actually kills frame-to-frame jitter, not just clipping
            blended = args.smoothing * pred + (1 - args.smoothing) * state
            delta = np.clip(blended - state, -args.max_step_deg, args.max_step_deg)
            state = np.clip(state + delta, 0, 180)

            line = ",".join(str(int(round(a))) for a in state) + "\n"
            ser.write(line.encode())

            # if the sketch is built with ENABLE_FEEDBACK, it echoes back the real
            # measured position after each command - use that as the true state
            # instead of just trusting what we sent
            try:
                if ser.in_waiting:
                    resp = ser.readline().decode(errors="replace").strip()
                    if resp.startswith("OK fb:"):
                        state = np.array([float(x) for x in resp[6:].split(",")])
            except (serial.SerialException, ValueError):
                pass

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cap.release()
        ser.close()


if __name__ == "__main__":
    main()
