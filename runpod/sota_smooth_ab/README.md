# RunPod SOTA Smoothness A/B

This bundle overlays the current SOTA record stack from `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` with a minimal warmdown smoothness patch for a hosted 1xH100 test.

Purpose:

- keep the exact SOTA architecture and export path
- add only `SMOOTH_LAMBDA` and `VAL_TOKENS_MAX`
- run a same-seed baseline vs smoothness comparison on one shard

Default hosted test:

- `MODE=pair`
- `TRAIN_SHARDS=1`
- `MAX_WALLCLOCK_SECONDS=900`
- `VAL_TOKENS_MAX=2097152`
- `SMOOTH_LAMBDA=2e-5`
- `SEED=314`

Outputs land under `/workspace/hosted_logs/sota_smooth_ab/`.

Expected workflow on the pod:

```bash
cd /workspace
tar xzf sota_smooth_ab.tar.gz
cd sota_smooth_ab
bash run_pair.sh
```
