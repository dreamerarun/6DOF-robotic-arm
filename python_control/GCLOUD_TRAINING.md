# Training on Google Cloud

Your local RTX 4050 (6GB) works but is tight, especially once you're training on
many episodes. Moving training to a cloud GPU is just "run the same script
somewhere with more VRAM" - nothing in train_bc_policy.py is laptop-specific.

## 1. Create a GPU VM

```bash
gcloud compute instances create arm-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB
```
`pytorch-latest-gpu` is a Deep Learning VM image - it comes with CUDA, torch,
and torchvision preinstalled and version-matched, which sidesteps the exact
numpy/torch ABI mismatch you hit locally. A T4 (16GB) or L4 (24GB) both give
you a lot more headroom than the 4050's 6GB, so you can raise `--batch_size`
well past 16.

Cheaper option if you don't need it running long: add `--preemptible` (or on
newer gcloud, `--provisioning-model=SPOT`) for a much lower hourly rate - the
VM can be reclaimed with short notice, so save checkpoints often (the script
already saves every epoch it improves).

## 2. Upload your dataset

```bash
gcloud compute scp --recurse ./dataset arm-training:~/dataset --zone=us-central1-a
gcloud compute scp --recurse ./python_control arm-training:~/python_control --zone=us-central1-a
```
For a large dataset, zip it first (`zip -r dataset.zip dataset`) so it's one
transfer instead of thousands of small file copies, then unzip on the VM.

## 3. SSH in and train

```bash
gcloud compute ssh arm-training --zone=us-central1-a

# on the VM:
cd python_control
pip install pyserial opencv-python-headless pillow  # torch/torchvision already present
python3 train_bc_policy.py --dataset ~/dataset --episodes 0003 --epochs 100 --batch_size 64 --out ~/checkpoints
```
Use `opencv-python-headless` on the VM (no display attached) instead of the
regular `opencv-python` - the recording/playback tools that need a camera
window don't run here anyway; only training does.

## 4. Download the trained checkpoint

```bash
gcloud compute scp arm-training:~/checkpoints/bc_policy_best.pt ./checkpoints/ --zone=us-central1-a
```
Then run `run_policy.py` locally as usual - inference is light enough to stay
on your laptop even though training happened in the cloud.

## 5. Stop or delete the VM when you're done

```bash
gcloud compute instances stop arm-training --zone=us-central1-a   # keep disk, stop billing for compute
# or
gcloud compute instances delete arm-training --zone=us-central1-a # fully remove
```
GPU VMs are billed by the hour while running - stopping (not just closing the
SSH session) is what actually stops the charge.

## On "some other good policy"

train_bc_policy.py is a single-frame regression policy - simple and reliable,
but exactly the kind of thing that produces jerky output (see the jerkiness
note in run_policy.py). If cloud compute is available, the natural upgrade is
LeRobot's **ACT** (Action Chunking Transformer) - the same policy family you
used for the Franka charger project. ACT predicts a short *chunk* of future
actions at once and temporally ensembles overlapping chunks, which is what
actually fixes jitter at the source rather than smoothing it after the fact.
The dataset this project records (frames/ + actions.csv per episode) maps
directly onto LeRobot's image/state/action columns, so converting is
mechanical, not a redesign. Given the extra setup involved (LeRobot install,
dataset conversion, ACT's training config), that's worth doing as a focused
next step once you've confirmed the wiring/data pipeline works end-to-end
with this simpler policy - happy to build that conversion + ACT training
script when you're ready for it.
