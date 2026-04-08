# Hosted H100 Smoothness A/B

## Research Status

- This was an exploratory, "vibe-researched" investigation and should not be treated as final expert guidance.
- The run is a single-seed, one-shard A/B used to check directional behavior on the real export stack.
- The conclusions here are about whether the mechanism appears real and where to search next, not about claiming a tuned production-ready configuration.

## Setup

- Platform: RunPod secure-cloud `1x H100 80GB` using the official Parameter Golf template
- Repo commit on host: `9d070dfe46d2d2e21ffc6f80feedbd5f081613d7`
- Base stack: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
- Patch scope: add `SMOOTH_LAMBDA`, add `VAL_TOKENS_MAX`, cap validation tokens, and apply the warmdown smoothness loss to the bank tensors
- Fixed settings: `TRAIN_SHARDS=1`, `MAX_WALLCLOCK_SECONDS=900`, `VAL_TOKENS_MAX=2097152`, `SEED=314`
- Compared runs: baseline `SMOOTH_LAMBDA=0` vs smooth `SMOOTH_LAMBDA=2e-5`

## Result

| Run | Steps | Wallclock val BPB | Post-EMA BPB | Roundtrip BPB | Sliding BPB | Artifact |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1394 | 1.2920 | 1.2977 | 1.30544696 | 1.28176764 | 13,324,606 |
| Smooth `2e-5` | 1391 | 1.3165 | 1.3407 | 1.35583393 | 1.33148712 | 10,419,610 |

## Deltas

- Artifact size: `-2,904,996` bytes (`-21.8%`)
- Roundtrip BPB: `+0.05038697`
- Sliding BPB: `+0.04971948`
- Wallclock validation BPB: `+0.0245`
- Post-EMA diagnostic BPB: `+0.0430`
- Training overhead: about `+1.64 ms/step` (`+0.25%`)

## Interpretation

- The compression mechanism is real on the full SOTA export path. The smooth run cut about `2.9 MB` from the final `int6 + lzma` artifact.
- This exact setting is too aggressive for the real stack. A `+0.050` BPB hit is too large for a direct swap or immediate reinvestment test.
- Most of the damage is already visible before export, so this is not a GPTQ-only failure mode. The likely issue is schedule strength or tensor scope at full training scale.
- The low-GPU-utilization phase after training was expected CPU-heavy GPTQ and export work, not a hang.

## 4090 Search Area

- Start with a cheaper proxy sweep on hosted 4090s before spending more H100 time. Keep the same one-shard, 900-second, capped-validation setup and use a 4090-compatible attention backend.
- Sweep `SMOOTH_LAMBDA` first: `3e-6`, `5e-6`, `7e-6`, `1e-5`. Keep seed fixed for coarse screening.
- Promote only candidates that save at least about `1 MB` while keeping sliding BPB within roughly `+0.015` of baseline.
- If the lambda sweep still over-penalizes BPB, search schedule strength next: later onset, a shorter active window inside warmdown, or normalization of the smoothness loss by tensor size.
- Defer reinvestment into larger models until a weaker regime is found that keeps a meaningful fraction of the size win without the current BPB penalty.

## Artifacts

- Logs: `runpod_logs_beq0jdfp2v7nzx/sota_smooth_ab/`
- Raw training logs: `runpod_logs_beq0jdfp2v7nzx/logs/`
- Host snapshot: `runpod_logs_beq0jdfp2v7nzx/host_snapshot/`
