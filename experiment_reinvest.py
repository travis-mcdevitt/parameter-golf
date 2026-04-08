"""
Reinvestment pilot (Issue #3, Step 3).

Tests whether a slightly larger model + weak smoothness buys back BPB
while preserving compression advantage.

Three runs only:
  1. small_baseline  — 17M params, lambda=0
  2. big_baseline    — 21.8M params (MLP_MULT=3), lambda=0
  3. big_smooth_weak — 21.8M params (MLP_MULT=3), lambda=3e-5

Usage:
    python experiment_reinvest.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

VARIANTS = [
    {
        "name": "small_baseline",
        "mlp_mult": 2,
        "smooth_lambda": 0.0,
    },
    {
        "name": "big_baseline",
        "mlp_mult": 3,
        "smooth_lambda": 0.0,
    },
    {
        "name": "big_smooth_weak",
        "mlp_mult": 3,
        "smooth_lambda": 3e-5,
    },
]

# Fixed across all runs
FIXED = dict(
    steps=500,
    warmdown=100,
    warmup=2,
    batch_tokens=16384,
    seq_len=1024,
    seed=1337,
    data_path="./data/datasets/fineweb10B_sp1024",
    tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
    vocab_size=1024,
    val_loss_every=0,
    val_tokens_max=2_000_000,
    val_batch_size=524288,
    num_calib_batches=16,
    calib_batch_seqs=4,
)

OUTPUT_DIR = "experiments/reinvest_pilot"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_variant(variant: dict) -> dict:
    """Train one variant via train_gpt.py subprocess."""
    name = variant["name"]
    out_dir = Path(OUTPUT_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "RUN_ID": f"reinvest_{name}",
        "ITERATIONS": str(FIXED["steps"]),
        "WARMDOWN_ITERS": str(FIXED["warmdown"]),
        "WARMUP_STEPS": str(FIXED["warmup"]),
        "TRAIN_BATCH_TOKENS": str(FIXED["batch_tokens"]),
        "TRAIN_SEQ_LEN": str(FIXED["seq_len"]),
        "SEED": str(FIXED["seed"]),
        "DATA_PATH": FIXED["data_path"],
        "TOKENIZER_PATH": FIXED["tokenizer_path"],
        "VOCAB_SIZE": str(FIXED["vocab_size"]),
        "VAL_LOSS_EVERY": str(FIXED["val_loss_every"]),
        "VAL_BATCH_SIZE": str(FIXED["val_batch_size"]),
        "VAL_TOKENS_MAX": str(FIXED["val_tokens_max"]),
        "MAX_WALLCLOCK_SECONDS": "0",
        "SMOOTH_LAMBDA": str(variant["smooth_lambda"]),
        "MLP_MULT": str(variant["mlp_mult"]),
        "TRAIN_LOG_EVERY": "50",
    }

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  MLP_MULT={variant['mlp_mult']}, SMOOTH_LAMBDA={variant['smooth_lambda']}")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", "train_gpt.py"],
        env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    wall_time = time.time() - t0

    (out_dir / "train.log").write_text(proc.stdout + "\n" + proc.stderr)

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        print(proc.stderr[-1000:])
        return {"name": name, "error": proc.stderr[-500:]}

    # Parse metrics from stdout
    metrics = {"name": name, "wall_time_s": wall_time}
    metrics.update(variant)
    for line in proc.stdout.splitlines():
        if "model_params:" in line:
            for part in line.split():
                if part.startswith("model_params:"):
                    metrics["params"] = int(part.split(":")[1])
        if "val_loss:" in line and "val_bpb:" in line and "final_int8" not in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["pre_quant_bpb"] = float(part.split(":")[1])
                if part.startswith("train_time:"):
                    metrics["train_time_ms"] = float(part.split(":")[1].rstrip("ms"))
        if "train_loss:" in line and "step_avg:" in line:
            for part in line.split():
                if part.startswith("step_avg:"):
                    metrics["step_avg_ms"] = float(part.split(":")[1].rstrip("ms"))
        if "final_int8_zlib_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["post_int8_bpb"] = float(part.split(":")[1])
        if "Serialized model int8+zlib:" in line:
            for part in line.split():
                try:
                    metrics["int8_zlib_bytes"] = int(part)
                    break
                except ValueError:
                    continue

    # Move checkpoint
    model_path = Path("final_model.pt")
    if model_path.exists():
        dest = out_dir / "final_model.pt"
        model_path.rename(dest)
        metrics["checkpoint"] = str(dest)

    for f in ["final_model.int8.ptz"]:
        p = Path(f)
        if p.exists():
            p.rename(out_dir / f)

    print(f"  Done: bpb={metrics.get('pre_quant_bpb', 'N/A')}, "
          f"params={metrics.get('params', 'N/A')}, "
          f"wall={wall_time:.0f}s")

    return metrics

# ---------------------------------------------------------------------------
# GPTQ + compression analysis (reuse experiment_gptq.py infrastructure)
# ---------------------------------------------------------------------------

def run_gptq_analysis(metrics: dict) -> dict:
    """Run GPTQ analysis on a trained checkpoint."""
    checkpoint = metrics.get("checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        return metrics

    print(f"  GPTQ analysis: {metrics['name']}...")

    # Import here to avoid circular/slow imports at module level
    from experiment_gptq import (
        build_model,
        collect_hessians,
        compress_quantized,
        compute_entropy,
        compute_run_length_proxy,
        dequantize_to_state_dict,
        quantize_all_gptq,
        quantize_all_naive,
        run_eval,
        setup_sdp_backends,
    )
    from train_gpt import (
        Hyperparameters,
        build_sentencepiece_luts,
        load_data_shard,
        load_validation_tokens,
    )
    import sentencepiece as spm
    import torch

    setup_sdp_backends()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=FIXED["tokenizer_path"])
    val_tokens = load_validation_tokens(
        os.path.join(FIXED["data_path"], "fineweb_val_*.bin"), args.train_seq_len,
    )
    if FIXED["val_tokens_max"] > 0:
        val_tokens = val_tokens[:FIXED["val_tokens_max"]]
    luts = build_sentencepiece_luts(sp, FIXED["vocab_size"], device)

    train_shard = Path(FIXED["data_path"]) / "fineweb_train_000000.bin"
    calib_tokens = load_data_shard(train_shard)
    tokens_needed = FIXED["num_calib_batches"] * FIXED["calib_batch_seqs"] * (args.train_seq_len + 1)
    calib_tokens = calib_tokens[:tokens_needed]

    # Build model with correct MLP mult
    from train_gpt import GPT, CastedLinear, restore_low_dim_params_to_fp32
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=metrics["mlp_mult"],
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)

    # Collect Hessians
    hessians = collect_hessians(
        model, calib_tokens, device,
        num_batches=FIXED["num_calib_batches"],
        batch_seqs=FIXED["calib_batch_seqs"],
    )

    # GPTQ int6
    gptq_quant = quantize_all_gptq(sd, hessians)
    gptq_comp = compress_quantized(gptq_quant)
    metrics["gptq_packed_zlib"] = gptq_comp["packed_zlib"]
    metrics["gptq_packed_lzma"] = gptq_comp["packed_lzma"]
    metrics["gptq_entropy"] = compute_entropy(gptq_comp["packed_bytes"])

    # Post-GPTQ BPB
    gptq_sd = dequantize_to_state_dict(gptq_quant, sd)
    model.load_state_dict(gptq_sd, strict=True)
    _, gptq_bpb = run_eval(model, device, val_tokens, luts)
    metrics["post_gptq_bpb"] = gptq_bpb

    # int8 + LZMA (use train_gpt's quantizer)
    import io, lzma
    from train_gpt import quantize_state_dict_int8
    q_obj, _ = quantize_state_dict_int8(sd)
    buf = io.BytesIO()
    torch.save(q_obj, buf)
    metrics["int8_lzma_bytes"] = len(lzma.compress(buf.getvalue(), preset=6))

    print(f"    post_gptq_bpb={gptq_bpb:.5f}, "
          f"gptq_packed_lzma={gptq_comp['packed_lzma']/1e6:.2f}MB, "
          f"int8_lzma={metrics['int8_lzma_bytes']/1e6:.2f}MB")

    # Free GPU memory
    del model, hessians, gptq_quant, gptq_sd, sd
    torch.cuda.empty_cache()

    return metrics

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def fmt_mb(b):
    return f"{b / 1_000_000:.2f} MB"

def print_results(results: list[dict]):
    small = next(r for r in results if r["name"] == "small_baseline")

    print(f"\n{'='*78}")
    print("REINVESTMENT PILOT RESULTS")
    print(f"{'='*78}")

    # Header
    print(f"{'':28}", end="")
    for r in results:
        print(f"  {r['name']:>16}", end="")
    print()
    print("-" * (28 + 18 * len(results)))

    def row(label, key, fmt=None):
        print(f"{label:28}", end="")
        for r in results:
            v = r.get(key, "N/A")
            if v == "N/A":
                print(f"  {'N/A':>16}", end="")
            elif fmt:
                print(f"  {fmt(v):>16}", end="")
            elif isinstance(v, float):
                print(f"  {v:>16.5f}", end="")
            elif isinstance(v, int):
                print(f"  {fmt_mb(v):>16}", end="")
            else:
                print(f"  {str(v):>16}", end="")
        print()

    row("params", "params", lambda v: f"{v:,}")
    row("mlp_mult", "mlp_mult", lambda v: f"{v}")
    row("smooth_lambda", "smooth_lambda", lambda v: f"{v}")
    row("step_avg", "step_avg_ms", lambda v: f"{v:.1f} ms")
    row("wall_time", "wall_time_s", lambda v: f"{v:.0f}s")
    print()
    row("pre_quant_bpb", "pre_quant_bpb")
    row("post_gptq_bpb", "post_gptq_bpb")
    print()
    row("int8 + zlib", "int8_zlib_bytes")
    row("int8 + lzma", "int8_lzma_bytes")
    row("gptq int6 packed + zlib", "gptq_packed_zlib")
    row("gptq int6 packed + lzma", "gptq_packed_lzma")

    # Key comparisons
    big_base = next((r for r in results if r["name"] == "big_baseline"), None)
    big_smooth = next((r for r in results if r["name"] == "big_smooth_weak"), None)

    print(f"\n{'='*78}")
    print("KEY COMPARISONS")
    print(f"{'='*78}")

    if big_base and small.get("post_gptq_bpb") and big_base.get("post_gptq_bpb"):
        cap_gain = small["post_gptq_bpb"] - big_base["post_gptq_bpb"]
        print(f"\n1. CAPACITY VALUE (big_baseline vs small_baseline):")
        print(f"   BPB gain from larger model: {cap_gain:+.5f} "
              f"({'bigger is better' if cap_gain > 0 else 'bigger is WORSE'})")
        if big_base.get("gptq_packed_lzma") and small.get("gptq_packed_lzma"):
            size_cost = (big_base["gptq_packed_lzma"] - small["gptq_packed_lzma"]) / small["gptq_packed_lzma"] * 100
            print(f"   Size cost (gptq packed lzma): {size_cost:+.1f}%")

    if big_smooth and small.get("post_gptq_bpb") and big_smooth.get("post_gptq_bpb"):
        reinvest_bpb = small["post_gptq_bpb"] - big_smooth["post_gptq_bpb"]
        print(f"\n2. REINVESTMENT VIABILITY (big_smooth_weak vs small_baseline):")
        print(f"   BPB delta: {reinvest_bpb:+.5f} "
              f"({'smooth bigger beats small' if reinvest_bpb > 0 else 'small still wins'})")
        if big_smooth.get("gptq_packed_lzma") and small.get("gptq_packed_lzma"):
            size_delta = (big_smooth["gptq_packed_lzma"] - small["gptq_packed_lzma"]) / small["gptq_packed_lzma"] * 100
            print(f"   Size delta (gptq packed lzma): {size_delta:+.1f}%")

    if big_smooth and big_base and big_base.get("post_gptq_bpb") and big_smooth.get("post_gptq_bpb"):
        gap = big_smooth["post_gptq_bpb"] - big_base["post_gptq_bpb"]
        print(f"\n3. SMOOTHNESS COST AT BIGGER SIZE:")
        print(f"   big_smooth_weak vs big_baseline BPB gap: {gap:+.5f}")
        if big_smooth.get("gptq_packed_lzma") and big_base.get("gptq_packed_lzma"):
            size_win = (big_smooth["gptq_packed_lzma"] - big_base["gptq_packed_lzma"]) / big_base["gptq_packed_lzma"] * 100
            print(f"   Size win from smoothness: {size_win:+.1f}%")

    print(f"\n{'='*78}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Phase 1: Train all variants
    print("PHASE 1: TRAINING")
    train_results = []
    for variant in VARIANTS:
        result = train_variant(variant)
        train_results.append(result)
        if "error" in result:
            print(f"  ERROR: {result['error'][:200]}")

    # Phase 2: GPTQ analysis on all checkpoints
    print("\n\nPHASE 2: GPTQ ANALYSIS")
    final_results = []
    for metrics in train_results:
        if "error" not in metrics and "checkpoint" in metrics:
            metrics = run_gptq_analysis(metrics)
        final_results.append(metrics)

    # Save JSON
    out_path = Path(OUTPUT_DIR) / "results.json"
    out_path.write_text(json.dumps(final_results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    # Print table
    valid = [r for r in final_results if "error" not in r]
    if valid:
        print_results(valid)


if __name__ == "__main__":
    main()
