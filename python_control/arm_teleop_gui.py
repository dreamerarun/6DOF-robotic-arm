#!/usr/bin/env python3
"""
arm_teleop_gui.py

Two modes in one tool:
  - Normal  : slider control of the arm over serial, camera preview only, nothing saved.
  - Record  : same control, but every frame is saved as (camera image, joint angles)
              into a new episode folder for later policy training.

Assumes: one USB webcam (index configurable), no leader/master arm - you drive the
6 sliders by hand and the recorder logs whatever you command.

Usage:
    python arm_teleop_gui.py --port /dev/ttyUSB0 --camera 0 --dataset ./dataset

Dataset layout written by Record mode:
    dataset/
      episode_0000/
        frames/frame_00000.jpg, frame_00001.jpg, ...
        actions.csv   # frame_idx,timestamp,j0,j1,j2,j3,j4,j5
      episode_0001/
        ...
"""

import argparse
import csv
import os
import time
import threading

import cv2
import serial
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

JOINT_NAMES = ["J0 wrist_roll", "J1 wrist_pitch", "J2 gripper",
               "J3 base", "J4 shoulder", "J5 elbow"]


class ArmTeleopApp:
    def __init__(self, root, port, baud, camera_index, dataset_dir):
        self.root = root
        self.root.title("Arm teleop / recorder")

        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.angles = [90] * 6
        self.last_send_time = 0.0
        self.send_interval = 0.05  # throttle to ~20 writes/sec, matches the web control page
        self.recording = False
        self.episode_dir = None
        self.episode_csv = None
        self.episode_writer = None
        self.frame_idx = 0
        self.episode_count = self._count_existing_episodes()
        self.latest_feedback = None  # set from "OK fb:.." responses when ENABLE_FEEDBACK is on

        self._build_ui()
        self._update_loop()

    def _count_existing_episodes(self):
        existing = [d for d in os.listdir(self.dataset_dir)
                    if d.startswith("episode_") and
                    os.path.isdir(os.path.join(self.dataset_dir, d))]
        return len(existing)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid()

        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0, rowspan=8, padx=(0, 16))

        self.sliders = []
        for i, name in enumerate(JOINT_NAMES):
            ttk.Label(main, text=name).grid(row=i, column=1, sticky="w")
            s = tk.Scale(main, from_=0, to=180, orient="horizontal",
                         length=220, command=lambda v, idx=i: self._on_slide(idx, v))
            s.set(90)
            s.grid(row=i, column=2)
            self.sliders.append(s)

        self.status_var = tk.StringVar(value="Normal mode - not recording")
        ttk.Label(main, textvariable=self.status_var).grid(row=6, column=1, columnspan=2, sticky="w", pady=(10, 0))

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=7, column=1, columnspan=2, pady=8, sticky="w")
        self.record_btn = ttk.Button(btn_frame, text="Start recording episode", command=self._toggle_record)
        self.record_btn.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btn_frame, text="Home", command=self._go_home).grid(row=0, column=1)

    def _on_slide(self, idx, value):
        self.angles[idx] = int(float(value))
        # Tkinter Scale fires this on every pixel of drag - sending unthrottled
        # can flood the Arduino's serial RX buffer faster than it drains and
        # corrupt the line it's assembling, so gate actual writes by time here
        # and always send the latest angles once the drag pauses.
        now = time.time()
        if now - self.last_send_time >= self.send_interval:
            self._send_angles()
        else:
            self.root.after(int(self.send_interval * 1000), self._send_latest_if_due)

    def _send_latest_if_due(self):
        if time.time() - self.last_send_time >= self.send_interval:
            self._send_angles()

    def _go_home(self):
        self.angles = [90] * 6
        for i, s in enumerate(self.sliders):
            s.set(90)
        self._send_angles()

    def _send_angles(self):
        line = ",".join(str(a) for a in self.angles) + "\n"
        print(f"-> {line.strip()}")
        try:
            self.ser.write(line.encode())
            self.ser.flush()
            self.last_send_time = time.time()
        except serial.SerialException as e:
            self.status_var.set(f"Serial error: {e}")
            print(f"Serial write failed: {e}")

    def _toggle_record(self):
        if not self.recording:
            self._start_episode()
        else:
            self._stop_episode()

    def _start_episode(self):
        name = f"episode_{self.episode_count:04d}"
        self.episode_dir = os.path.join(self.dataset_dir, name)
        os.makedirs(os.path.join(self.episode_dir, "frames"), exist_ok=True)
        self.episode_csv = open(os.path.join(self.episode_dir, "actions.csv"), "w", newline="")
        self.episode_writer = csv.writer(self.episode_csv)
        header = ["frame_idx", "timestamp", "j0", "j1", "j2", "j3", "j4", "j5"]
        if self.latest_feedback is not None:
            header += ["fb0", "fb1", "fb2", "fb3", "fb4", "fb5"]
        self.episode_writer.writerow(header)
        self.frame_idx = 0
        self.recording = True
        self.record_btn.config(text="Stop recording")
        self.status_var.set(f"Recording {name}...")

    def _stop_episode(self):
        self.recording = False
        if self.episode_csv:
            self.episode_csv.close()
        self.episode_count += 1
        self.record_btn.config(text="Start recording episode")
        self.status_var.set(f"Saved {self.episode_dir} ({self.frame_idx} frames). Normal mode.")
        self.episode_dir = None

    def _update_loop(self):
        # drain and print anything the Arduino sent back (OK / ERR / boot message) -
        # useful to confirm lines are actually reaching and parsing on the board.
        # When the sketch is built with ENABLE_FEEDBACK, "OK fb:83.0,91.2,..." lines
        # carry real measured servo angles - parse those into self.latest_feedback.
        try:
            if self.ser.in_waiting:
                resp = self.ser.readline().decode(errors="replace").strip()
                if resp:
                    print(f"<- {resp}")
                    if resp.startswith("OK fb:"):
                        try:
                            self.latest_feedback = [float(x) for x in resp[6:].split(",")]
                        except ValueError:
                            pass
        except serial.SerialException:
            pass

        ret, frame = self.cap.read()
        if ret:
            display = cv2.cvtColor(cv2.resize(frame, (400, 300)), cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(display))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

            if self.recording:
                fname = f"frame_{self.frame_idx:05d}.jpg"
                cv2.imwrite(os.path.join(self.episode_dir, "frames", fname), frame)
                row = [self.frame_idx, time.time(), *self.angles]
                if self.latest_feedback is not None:
                    row += self.latest_feedback
                self.episode_writer.writerow(row)
                self.frame_idx += 1

        # ~15 Hz loop - enough for slow arm motion, keeps dataset size reasonable
        self.root.after(66, self._update_loop)

    def close(self):
        if self.recording:
            self._stop_episode()
        self.cap.release()
        self.ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dataset", default="./dataset")
    args = ap.parse_args()

    root = tk.Tk()
    app = ArmTeleopApp(root, args.port, args.baud, args.camera, args.dataset)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
