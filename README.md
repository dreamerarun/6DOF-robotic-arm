# 6-Axis Arm — Arduino + PCA9685 + Web + ROS2/MoveIt2

Three ways to drive the same arm, all speaking the same serial protocol:
`a0,a1,a2,a3,a4,a5\n` (angles 0-180, six comma-separated ints).

```
robot_arm_project/
├── robot_arm_controller/
│   └── robot_arm_controller.ino     # flash to the Arduino Uno
├── web_control/
│   └── arm_control.html             # open in Chrome/Edge, no server needed
└── ros2_ws/src/arm_moveit_bridge/
    ├── arm_moveit_bridge/serial_bridge_node.py
    ├── launch/bridge.launch.py
    ├── config/moveit_controllers.yaml
    ├── urdf/arm.urdf.xacro
    ├── package.xml
    └── setup.py
```

## 1. Wiring

```
PCA9685 VCC  -> Arduino 5V        PCA9685 SDA -> Arduino A4
PCA9685 GND  -> Arduino GND       PCA9685 SCL -> Arduino A5
PCA9685 V+   -> External 5-6V supply (3-4A+), servo GND tied to this rail
All grounds (Arduino / PCA9685 / external supply) common
CH0-2: SG90 x3      CH3-5: MG995 x3
```
Do **not** power the MG995s from the Arduino's 5V pin — they'll brown out the board.

## 2. Arduino

1. Install the **Adafruit PWM Servo Driver** library (Library Manager).
2. Flash `robot_arm_controller.ino`.
3. Open Serial Monitor at 115200 baud, send `90,90,90,90,90,90` — all six should center.
4. Recalibrate `SERVO_MIN[]`/`SERVO_MAX[]` per channel if a servo doesn't hit true 0°/180° (see the comment block at the bottom of the sketch).

## 3. Web control (quick manual control, no ROS needed)

Open `arm_control.html` directly in **Chrome or Edge** (Web Serial API isn't supported in Firefox/Safari). Click **Connect to Arduino**, pick the Arduino's serial port, and drag the sliders. This talks straight to the Uno over USB — no Wi-Fi module or server required.

## 4. ROS2 Humble + MoveIt2

This gives you motion planning (collision-aware trajectories, drag-the-end-effector-in-RViz) that gets executed on the real servos.

**a. Build your real URDF.** `urdf/arm.urdf.xacro` is a skeleton with correct joint names/axes/limits but placeholder link lengths — swap in your measured dimensions (and meshes if you have them).

**b. Generate the MoveIt config.** Run MoveIt Setup Assistant against that xacro:
```bash
ros2 launch moveit_setup_assistant setup_assistant.launch.py
```
Define your planning group (`joint_1`...`joint_6`), end effector, and self-collision matrix, then generate the `<your_arm>_moveit_config` package.

**c. Point MoveIt at the bridge instead of a real controller manager.** In the generated `config/moveit_controllers.yaml`, replace its contents with (or merge in) the `arm_controller` entry from `arm_moveit_bridge/config/moveit_controllers.yaml` — the key part is that the controller's action name resolves to `/arm_controller/follow_joint_trajectory`, which is what the bridge node serves.

**d. Build & run the bridge:**
```bash
cd ros2_ws
colcon build --packages-select arm_moveit_bridge
source install/setup.bash
ros2 launch arm_moveit_bridge bridge.launch.py serial_port:=/dev/ttyUSB0
```
(Find your port with `ls /dev/tty*` before/after plugging in the Uno — usually `/dev/ttyACM0` or `/dev/ttyUSB0`; add your user to the `dialout` group if you get a permissions error.)

**e. Launch MoveIt** (from your generated moveit_config package) as usual:
```bash
ros2 launch <your_arm>_moveit_config demo.launch.py
```
Plan a pose in RViz's MotionPlanning panel and hit **Execute** — the trajectory goes to `serial_bridge_node`, which paces it out over serial to the Arduino in real time and republishes `/joint_states` so RViz reflects the arm's reported pose.

## 5. Recording demonstrations & training a camera-based policy

`python_control/` adds a fourth way to drive the arm, alongside the sketch, web page, and ROS2 bridge: a Python teleop tool with **Normal** and **Record** modes, plus scripts to train and run a vision-based policy from what you record.

Assumes one USB webcam and no leader/master arm — you drive the 6 sliders by hand; the recorder just logs whatever you command alongside the camera frame at that instant.

```bash
cd python_control
pip install -r requirements.txt
```

**Normal mode** — just drive the arm and watch the camera feed, nothing is saved:
```bash
python arm_teleop_gui.py --port /dev/ttyUSB0 --camera 0
```

**Record mode** — same GUI, click "Start recording episode" before you demonstrate a task (e.g. pick-and-place) and "Stop recording" after. Each episode is saved under `dataset/episode_NNNN/` as `frames/*.jpg` + `actions.csv` (frame index, timestamp, 6 joint angles). Record several episodes (10-30+) of the same task for a policy to have a chance of generalizing.

**Train a policy** on the recorded episodes:
```bash
python train_bc_policy.py --dataset ./dataset --epochs 50 --out ./checkpoints
```
This trains a small behavior-cloning policy — a ResNet18 image encoder (mostly frozen, to fit comfortably in 6GB VRAM) plus the current joint angles as proprioception, regressing the next commanded angles. It's simpler than a full ACT/transformer policy (no action chunking), chosen so it trains reliably on your hardware; the dataset layout maps directly onto LeRobot's image/state/action columns if you later want to graduate to full ACT the way you did for the Franka charger project.

**Run the trained policy** closed-loop on the real arm:
```bash
python run_policy.py --port /dev/ttyUSB0 --camera 0 --checkpoint ./checkpoints/bc_policy_best.pt
```
It reads a camera frame + the arm's last commanded pose, predicts the next pose at ~10 Hz, and streams it to the Arduino using the same serial protocol as everything else. Ctrl+C stops it; the arm holds its last pose.

### How the pieces relate
- The **Arduino sketch** owns the actual PWM/servo motion and low-level safety limits — it's dumb and reliable by design.
- The **web page** and **ROS2 bridge** are two independent senders using the exact same serial protocol — you can use either one (or swap between them) without touching the sketch.
- `serial_bridge_node.py` acts as a `FollowJointTrajectory` action server so MoveIt2 doesn't need a full `ros2_control` hardware interface — it's the fastest path from "planned trajectory" to "servo moves," at the cost of being open-loop (no real encoder feedback, since these servos don't have any).
