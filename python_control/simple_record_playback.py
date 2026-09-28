#!/usr/bin/env python3
"""
simple_record_playback.py

The simplest possible teach-and-replay tool - no camera, no dataset, no policy.
Move the sliders to demonstrate a motion, hit Record to log it, hit Play to
replay the exact same timed sequence of angles.

Good for: quick repeatable pick-and-place demos, sanity-checking wiring/motion
before investing in the camera+policy pipeline, or just running a fixed routine
on a loop.

Usage:
    python simple_record_playback.py --port /dev/ttyACM0

Trajectories save as JSON: [{"t": 0.0, "angles": [90,90,90,90,90,90]}, ...]
"""

import argparse
import json
import os
import time

import serial
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

JOINT_NAMES = ["J0 wrist_roll", "J1 wrist_pitch", "J2 gripper",
               "J3 base", "J4 shoulder", "J5 elbow"]


class RecordPlaybackApp:
    def __init__(self, root, port, baud, traj_dir):
        self.root = root
        self.root.title("Simple record & playback")

        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)

        self.traj_dir = traj_dir
        os.makedirs(self.traj_dir, exist_ok=True)

        self.angles = [90] * 6
        self.recording = False
        self.playing = False
        self.record_start = 0.0
        self.trajectory = []  # list of {"t": float, "angles": [..]}
        self.last_logged = 0.0
        self.log_interval = 0.05  # log at ~20Hz while recording, matches serial send rate

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid()

        self.sliders = []
        for i, name in enumerate(JOINT_NAMES):
            ttk.Label(main, text=name).grid(row=i, column=0, sticky="w")
            s = tk.Scale(main, from_=0, to=180, orient="horizontal",
                         length=220, command=lambda v, idx=i: self._on_slide(idx, v))
            s.set(90)
            s.grid(row=i, column=1)
            self.sliders.append(s)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(main, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 4))

        btns = ttk.Frame(main)
        btns.grid(row=7, column=0, columnspan=2, pady=6, sticky="w")
        self.record_btn = ttk.Button(btns, text="Record", command=self._toggle_record)
        self.record_btn.grid(row=0, column=0, padx=4)
        self.play_btn = ttk.Button(btns, text="Play", command=self._play, state="disabled")
        self.play_btn.grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Save...", command=self._save).grid(row=0, column=2, padx=4)
        ttk.Button(btns, text="Load...", command=self._load).grid(row=0, column=3, padx=4)
        ttk.Button(btns, text="Home", command=self._go_home).grid(row=0, column=4, padx=4)

    def _on_slide(self, idx, value):
        if self.playing:
            return  # ignore manual input while a playback is running
        self.angles[idx] = int(float(value))
        self._send_angles()
        if self.recording:
            now = time.time()
            if now - self.last_logged >= self.log_interval:
                self.trajectory.append({"t": now - self.record_start, "angles": list(self.angles)})
                self.last_logged = now

    def _send_angles(self):
        line = ",".join(str(a) for a in self.angles) + "\n"
        try:
            self.ser.write(line.encode())
        except serial.SerialException as e:
            self.status_var.set(f"Serial error: {e}")

    def _go_home(self):
        self.angles = [90] * 6
        for s in self.sliders:
            s.set(90)
        self._send_angles()

    def _toggle_record(self):
        if not self.recording:
            self.trajectory = []
            self.record_start = time.time()
            self.last_logged = 0.0
            self.recording = True
            self.record_btn.config(text="Stop")
            self.play_btn.config(state="disabled")
            self.status_var.set("Recording... move the sliders")
        else:
            self.recording = False
            self.record_btn.config(text="Record")
            self.play_btn.config(state="normal" if self.trajectory else "disabled")
            self.status_var.set(f"Recorded {len(self.trajectory)} points")

    def _play(self):
        if not self.trajectory or self.playing:
            return
        self.playing = True
        self.play_btn.config(state="disabled")
        self.record_btn.config(state="disabled")
        self.status_var.set("Playing back...")
        self._play_step(0, time.time())

    def _play_step(self, idx, start_time):
        if idx >= len(self.trajectory):
            self.playing = False
            self.play_btn.config(state="normal")
            self.record_btn.config(state="normal")
            self.status_var.set("Playback done")
            return

        point = self.trajectory[idx]
        self.angles = list(point["angles"])
        for i, s in enumerate(self.sliders):
            s.set(self.angles[i])  # updates slider visually; _on_slide is skipped during playback
        self._send_angles()

        if idx + 1 < len(self.trajectory):
            delay_s = self.trajectory[idx + 1]["t"] - point["t"]
        else:
            delay_s = 0
        delay_ms = max(1, int(delay_s * 1000))
        self.root.after(delay_ms, lambda: self._play_step(idx + 1, start_time))

    def _save(self):
        if not self.trajectory:
            messagebox.showinfo("Nothing to save", "Record a trajectory first.")
            return
        path = filedialog.asksaveasfilename(
            initialdir=self.traj_dir, defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if path:
            with open(path, "w") as f:
                json.dump(self.trajectory, f, indent=2)
            self.status_var.set(f"Saved to {path}")

    def _load(self):
        path = filedialog.askopenfilename(
            initialdir=self.traj_dir, filetypes=[("JSON", "*.json")])
        if path:
            with open(path) as f:
                self.trajectory = json.load(f)
            self.play_btn.config(state="normal")
            self.status_var.set(f"Loaded {len(self.trajectory)} points from {os.path.basename(path)}")

    def close(self):
        self.ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--traj_dir", default="./trajectories")
    args = ap.parse_args()

    root = tk.Tk()
    app = RecordPlaybackApp(root, args.port, args.baud, args.traj_dir)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
