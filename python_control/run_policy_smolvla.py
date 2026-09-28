#!/usr/bin/env python3
"""
run_policy_smolvla.py

Closed-loop execution using a fine-tuned SmolVLA checkpoint instead of the
simple BC policy in run_policy.py. SmolVLA predicts a chunk of future actions
per inference call and its processors handle temporal ensembling internally -
this is what actually fixes jitter at the source, rather than the EMA/rate-cap
smoothing bolted onto run_policy.py's single-step predictions.

NOTE ON API STABILITY: same caveat as convert_to_lerobot_dataset.py - check
this against your installed lerobot version's docs if a call signature has
moved.

Usage:
    python run_policy_smolvla.py \
        --port /dev/ttyACM0 --camera 0 \
        --checkpoint ./smolvla_out/checkpoints/last/pretrained_model \
        --task "pick up the object and place it in the box"
"""

import argparse
import time

import cv2
import numpy as np
import serial
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--checkpoint", required=True, help="Path to the fine-tuned SmolVLA "
                     "pretrained_model directory (local) or a hub repo id")
    ap.add_argument("--task", required=True, help="Same task string used during dataset conversion")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--show_camera", action="store_true",
                     help="Open a live window showing exactly what the policy sees each step, "
                          "with the predicted joint angles overlaid. Press 'q' in that window to stop.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2.0)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    state = np.array([90.0] * 6, dtype=np.float32)
    period = 1.0 / args.hz

    print("Running SmolVLA policy. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # HWC uint8 -> CHW float32 in [0,1]

            batch = {
                "observation.images.cam": img_t.unsqueeze(0),
                "observation.state": torch.from_numpy(state).unsqueeze(0),
                "task": [args.task],
            }
            batch = preprocess(batch)

            with torch.no_grad():
                action = policy.select_action(batch)  # policy + processors handle chunking/ensembling
            action = postprocess(action).cpu().numpy().squeeze()

            state = np.clip(action[:6], 0, 180).astype(np.float32)
            line = ",".join(str(int(round(a))) for a in state) + "\n"
            ser.write(line.encode())

            if args.show_camera:
                preview = frame.copy()
                labels = [f"j{i}: {int(state[i])}" for i in range(6)]
                cv2.putText(preview, " ".join(labels), (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(preview, args.task, (8, preview.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.imshow("SmolVLA - what the policy sees", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # if ENABLE_FEEDBACK is on in the sketch, prefer the real measured state
            try:
                if ser.in_waiting:
                    resp = ser.readline().decode(errors="replace").strip()
                    if resp.startswith("OK fb:"):
                        state = np.array([float(x) for x in resp[6:].split(",")], dtype=np.float32)
            except (serial.SerialException, ValueError):
                pass

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cap.release()
        if args.show_camera:
            cv2.destroyAllWindows()
        ser.close()


if __name__ == "__main__":
    main()
