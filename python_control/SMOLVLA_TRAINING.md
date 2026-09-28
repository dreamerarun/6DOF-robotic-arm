# Fine-tuning SmolVLA on your arm

SmolVLA (Hugging Face's 450M-parameter vision-language-action model) is a real
upgrade path from the simple BC policy: instead of predicting one angle per
frame, it predicts a *chunk* of future actions and temporally ensembles them -
which directly addresses the jitter you saw, rather than smoothing over it
after the fact like `run_policy.py`'s EMA does.

It needs the LeRobot framework and its dataset format, and realistically needs
more compute than your 4050 for training - this is a good fit for the GCloud
VM from GCLOUD_TRAINING.md.

## 1. Install

```bash
pip install "lerobot[smolvla]"
```
See https://huggingface.co/docs/lerobot/installation for extras (ffmpeg for
video encoding, etc) - install requirements can shift between releases.

## 2. Record enough data

SmolVLA's own docs recommend **~50 episodes per task variation** as a starting
point (e.g. if the object's position varies, you want that variation
represented across episodes, not just repeats of one exact placement). Your
current 1-episode / smaller sets are fine for validating the pipeline works
end-to-end, but treat the actual fine-tune as needing a proper recording push
first - this matters more than any hyperparameter choice.

## 3. Convert your dataset to LeRobot format

```bash
python convert_to_lerobot_dataset.py \
    --dataset ./dataset \
    --episodes 0003,0004,0007 \
    --repo_id yourname/arm-pickplace \
    --task "pick up the object and place it in the box" \
    --push_to_hub
```
Drop `--episodes` to convert everything in `./dataset`, or `--push_to_hub` to
keep it local-only (see the note in the script about `--dataset.root` for
local training).

The `--task` string matters - it's the language instruction SmolVLA
conditions on. Keep it short and consistent between conversion and inference.

## 4. Fine-tune

On the GCloud GPU VM (see GCLOUD_TRAINING.md for setup):

```bash
python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --dataset.repo_id=yourname/arm-pickplace \
    --output_dir=./smolvla_out \
    --batch_size=32 \
    --steps=20000 \
    --wandb.enable=false
```
Reference point from HF's own docs: ~20k steps takes roughly 4 hours on a
single A100. A T4/L4 will be slower - start with fewer steps (e.g. 5k-10k) to
sanity-check the pipeline before committing to a long run, then extend.

If your installed lerobot version exposes a `lerobot-train` CLI command
instead of `python -m lerobot.scripts.train`, use that - check `lerobot-train
--help` first, this has moved between releases.

## 5. Run the fine-tuned checkpoint on the real arm

```bash
python run_policy_smolvla.py \
    --port /dev/ttyACM0 --camera 0 \
    --checkpoint ./smolvla_out/checkpoints/last/pretrained_model \
    --task "pick up the object and place it in the box"
```
Copy the checkpoint dir back from the VM first (same `gcloud compute scp`
pattern as GCLOUD_TRAINING.md step 4). Use the exact same `--task` string you
converted the dataset with.

## Expectation-setting

Real-world reports on fine-tuning SmolVLA (including from Hugging Face's own
community) land around 40-80% task success rate depending heavily on data
quality and quantity - it is not a drop-in fix that guarantees smooth,
accurate motion regardless of how much data you feed it. If results are poor,
the two highest-leverage things to check first are (1) how varied and clean
your demonstrations actually are, and (2) whether `--task` is identical
between conversion and inference - a mismatched instruction string is a common
silent failure mode.
