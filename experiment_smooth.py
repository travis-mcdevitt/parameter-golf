"""
Warmdown smoothness regularizer — CUDA transfer test (Issue #3).

Tests whether the MLX-validated compression effect survives in PyTorch/CUDA
with int8, int6, packed-int6, zlib, and LZMA export paths.

Usage:
    python experiment_smooth.py                    # run all three variants
    python experiment_smooth.py --lambda-only 0    # run baseline only
    python experiment_smooth.py --steps 300        # override step count

Each variant trains via train_gpt.py, then this script loads the checkpoint
and runs extended compression analysis.
"""

from __future__ import annotations

import argparse
import io
import json
import lzma
import math
import os
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANTS = {
    "baseline":      0.0,
    "smooth_weak":   3e-5,
    "smooth_strong": 1e-4,
}

DEFAULTS = dict(
    steps=500,
    warmdown=100,
    warmup=2,
    batch_tokens=16384,
    seq_len=1024,
    seed=1337,
    data_path="./data/datasets/fineweb10B_sp1024",
    tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
    vocab_size=1024,
    val_loss_every=0,      # only validate at end (slow on 2070)
    val_batch_size=524288,
    val_tokens_max=2_000_000,  # cap val set for speed on small GPUs (0=full)
)

# ---------------------------------------------------------------------------
# Int6 quantization
# ---------------------------------------------------------------------------

def quantize_tensor_intN(t: torch.Tensor, bits: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric intN quantization for 2D tensors, per-tensor for others."""
    max_val = (1 << (bits - 1)) - 1  # 31 for int6
    t32 = t.float()

    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1).clamp_min(1e-12)
        scale = row_max / max_val
        q = torch.clamp(torch.round(t32 / scale[:, None]), -max_val, max_val).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()

    flat_max = t32.abs().max().clamp_min(1e-12).item()
    scale = torch.tensor(flat_max / max_val, dtype=torch.float32)
    q = torch.clamp(torch.round(t32 / scale), -max_val, max_val).to(torch.int8)
    return q.contiguous(), scale

# ---------------------------------------------------------------------------
# Int6 bit-packing
# ---------------------------------------------------------------------------

def pack_int6(values: np.ndarray) -> bytes:
    """Pack signed 6-bit values into a byte stream. 4 values -> 3 bytes."""
    values = values.flatten().astype(np.int8)
    # Map signed [-31..31] to unsigned [0..62] for bit packing
    unsigned = (values.astype(np.int16) + 31).astype(np.uint8)

    # Pad to multiple of 4
    pad = (4 - len(unsigned) % 4) % 4
    if pad:
        unsigned = np.concatenate([unsigned, np.zeros(pad, dtype=np.uint8)])

    n_groups = len(unsigned) // 4
    unsigned = unsigned.reshape(n_groups, 4)
    a, b, c, d = unsigned[:, 0], unsigned[:, 1], unsigned[:, 2], unsigned[:, 3]

    # Pack 4x6bit = 24bit = 3 bytes
    b0 = (a & 0x3F) | ((b & 0x03) << 6)
    b1 = ((b >> 2) & 0x0F) | ((c & 0x0F) << 4)
    b2 = ((c >> 4) & 0x03) | ((d & 0x3F) << 2)

    packed = np.empty(n_groups * 3, dtype=np.uint8)
    packed[0::3] = b0
    packed[1::3] = b1
    packed[2::3] = b2
    return packed.tobytes()

# ---------------------------------------------------------------------------
# Compression analysis
# ---------------------------------------------------------------------------

# Import train_gpt.py's quantization to ensure int8 results match exactly.
sys.path.insert(0, str(Path(__file__).parent))
from train_gpt import quantize_state_dict_int8


def analyze_checkpoint(model_path: str) -> dict:
    """Load a checkpoint and compute compression metrics at int8, int6, and packed-int6."""
    sd = torch.load(model_path, map_location="cpu", weights_only=True)

    results = {}
    total_params = sum(t.numel() for t in sd.values())
    results["total_params"] = total_params

    # --- Int8: use train_gpt.py's exact quantizer for apples-to-apples comparison ---
    q_obj_int8, q_stats = quantize_state_dict_int8(sd)
    buf8 = io.BytesIO()
    torch.save(q_obj_int8, buf8)
    raw8 = buf8.getvalue()
    results["int8_raw"] = len(raw8)
    results["int8_zlib"] = len(zlib.compress(raw8, level=9))
    results["int8_lzma"] = len(lzma.compress(raw8, preset=6))

    # --- Int6: per-row symmetric, same structure as int8 but with 6-bit range ---
    q_obj_int6 = _quantize_state_dict_intN(sd, bits=6)
    buf6 = io.BytesIO()
    torch.save(q_obj_int6, buf6)
    raw6 = buf6.getvalue()
    results["int6_raw"] = len(raw6)
    results["int6_zlib"] = len(zlib.compress(raw6, level=9))
    results["int6_lzma"] = len(lzma.compress(raw6, preset=6))

    # --- Packed int6: bit-pack the quantized weight values ---
    all_q_values = np.concatenate([
        q_obj_int6["quantized"][k].numpy().flatten()
        for k in q_obj_int6["quantized"]
    ])
    packed_bytes = pack_int6(all_q_values)
    results["int6_packed_raw"] = len(packed_bytes)
    results["int6_packed_zlib"] = len(zlib.compress(packed_bytes, level=9))
    results["int6_packed_lzma"] = len(lzma.compress(packed_bytes, preset=6))

    # Scale/passthrough overhead for total artifact calculation
    meta_buf = io.BytesIO()
    torch.save({
        "scales": q_obj_int6.get("scales", {}),
        "passthrough": q_obj_int6.get("passthrough", {}),
    }, meta_buf)
    results["int6_scales_and_pass_raw"] = len(meta_buf.getvalue())

    return results


# Keep threshold/dtype constants aligned with train_gpt.py
_KEEP_FLOAT_MAX_NUMEL = 65_536
_KEEP_FLOAT_STORE_DTYPE = torch.float16
_CLIP_Q = 99.99984 / 100.0
_CONTROL_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)


def _quantize_state_dict_intN(sd: dict[str, torch.Tensor], bits: int = 6) -> dict:
    """Quantize a state dict at N-bit precision, mirroring train_gpt.py's structure."""
    max_val = (1 << (bits - 1)) - 1
    quantized, scales, passthrough = {}, {}, {}

    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()

        if not t.is_floating_point():
            passthrough[name] = t
            continue

        if t.numel() <= _KEEP_FLOAT_MAX_NUMEL:
            # Small tensors pass through at fp16 (matches train_gpt.py behavior)
            passthrough[name] = t.to(_KEEP_FLOAT_STORE_DTYPE) if t.dtype != _KEEP_FLOAT_STORE_DTYPE else t
            continue

        q, s = quantize_tensor_intN(t, bits=bits)
        quantized[name] = q
        scales[name] = s

    return {"quantized": quantized, "scales": scales, "passthrough": passthrough}

# ---------------------------------------------------------------------------
# Run a training variant
# ---------------------------------------------------------------------------

def run_variant(name: str, lam: float, args: argparse.Namespace) -> dict:
    """Train one variant via train_gpt.py subprocess, then analyze the checkpoint."""
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "RUN_ID": f"smooth_exp_{name}",
        "ITERATIONS": str(args.steps),
        "WARMDOWN_ITERS": str(args.warmdown),
        "WARMUP_STEPS": str(args.warmup),
        "TRAIN_BATCH_TOKENS": str(args.batch_tokens),
        "TRAIN_SEQ_LEN": str(args.seq_len),
        "SEED": str(args.seed),
        "DATA_PATH": args.data_path,
        "TOKENIZER_PATH": args.tokenizer_path,
        "VOCAB_SIZE": str(args.vocab_size),
        "VAL_LOSS_EVERY": str(args.val_loss_every),
        "VAL_BATCH_SIZE": str(args.val_batch_size),
        "MAX_WALLCLOCK_SECONDS": "0",
        "SMOOTH_LAMBDA": str(lam),
        "TRAIN_LOG_EVERY": "50",
        "VAL_TOKENS_MAX": str(args.val_tokens_max),
    }

    cmd = [sys.executable, "-u", "train_gpt.py"]
    print(f"\n{'='*60}")
    print(f"Running variant: {name} (lambda={lam})")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    wall_time = time.time() - t0

    # Save stdout/stderr
    (out_dir / "train.log").write_text(proc.stdout + "\n" + proc.stderr)

    if proc.returncode != 0:
        print(f"FAILED (exit {proc.returncode}):")
        print(proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
        return {"name": name, "lambda": lam, "error": proc.stderr[-500:]}

    # Parse key metrics from stdout
    metrics = {"name": name, "lambda": lam, "wall_time_s": wall_time}
    for line in proc.stdout.splitlines():
        if "val_loss:" in line and "val_bpb:" in line:
            for part in line.split():
                if part.startswith("val_loss:"):
                    metrics["val_loss"] = float(part.split(":")[1])
                if part.startswith("val_bpb:"):
                    metrics["val_bpb"] = float(part.split(":")[1])
                if part.startswith("train_time:"):
                    metrics["train_time_ms"] = float(part.split(":")[1].rstrip("ms"))
        if "train_loss:" in line and "step_avg:" in line:
            for part in line.split():
                if part.startswith("train_loss:"):
                    metrics["final_train_loss"] = float(part.split(":")[1])
                if part.startswith("step_avg:"):
                    metrics["step_avg_ms"] = float(part.split(":")[1].rstrip("ms"))
        if "final_int8_zlib_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["post_quant_bpb"] = float(part.split(":")[1])
        if "Serialized model int8+zlib:" in line:
            # e.g. "Serialized model int8+zlib: 9660123 bytes ..."
            for part in line.split():
                try:
                    val = int(part)
                    metrics["train_gpt_int8_zlib"] = val
                    break
                except ValueError:
                    continue

    # Move checkpoint to variant dir
    model_path = Path("final_model.pt")
    if model_path.exists():
        dest = out_dir / "final_model.pt"
        model_path.rename(dest)
        metrics["model_path"] = str(dest)

        # Extended compression analysis
        print(f"  Analyzing compression for {name}...")
        comp = analyze_checkpoint(str(dest))
        metrics.update(comp)
    else:
        metrics["error"] = "final_model.pt not found"

    # Clean up other artifacts
    for f in ["final_model.int8.ptz"]:
        p = Path(f)
        if p.exists():
            p.rename(out_dir / f)

    # Save per-variant JSON
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def fmt_bytes(b: int | float) -> str:
    if b >= 1_000_000:
        return f"{b / 1_000_000:.2f} MB"
    return f"{b / 1_000:.1f} KB"

def print_comparison(all_metrics: list[dict]):
    """Print a compact comparison table."""
    baseline = next((m for m in all_metrics if m["lambda"] == 0.0), None)

    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    cols = ["Metric"] + [m["name"] for m in all_metrics]
    print(f"{'Metric':<30}", end="")
    for m in all_metrics:
        print(f"  {m['name']:>16}", end="")
    print()
    print("-" * (30 + 18 * len(all_metrics)))

    def row(label, key, fmt="{}"):
        print(f"{label:<30}", end="")
        for m in all_metrics:
            val = m.get(key, "N/A")
            if val == "N/A":
                print(f"  {'N/A':>16}", end="")
            elif isinstance(val, float):
                print(f"  {fmt.format(val):>16}", end="")
            elif isinstance(val, int):
                print(f"  {fmt_bytes(val):>16}", end="")
            else:
                print(f"  {str(val):>16}", end="")
        print()

    def delta_row(label, key, baseline_val, lower_better=True):
        print(f"{label:<30}", end="")
        for m in all_metrics:
            val = m.get(key)
            if val is None or baseline_val is None:
                print(f"  {'N/A':>16}", end="")
            elif isinstance(val, (int, float)):
                if baseline_val != 0:
                    pct = (val - baseline_val) / baseline_val * 100
                    sign = "+" if pct >= 0 else ""
                    print(f"  {sign}{pct:.1f}%{' ':>10}", end="")
                else:
                    print(f"  {'N/A':>16}", end="")
            else:
                print(f"  {'N/A':>16}", end="")
        print()

    # Training
    row("val_bpb", "val_bpb", "{:.5f}")
    row("post_quant_bpb (int8)", "post_quant_bpb", "{:.5f}")
    row("train_loss (final)", "final_train_loss", "{:.4f}")
    row("wall_time", "wall_time_s", "{:.0f}s")
    row("step_avg", "step_avg_ms", "{:.1f}ms")
    print()

    # Int8 compression
    row("int8 zlib", "int8_zlib")
    row("int8 lzma", "int8_lzma")
    if baseline and baseline.get("int8_lzma"):
        delta_row("  vs baseline", "int8_lzma", baseline.get("int8_lzma"))
    print()

    # Int6 compression
    row("int6 zlib", "int6_zlib")
    row("int6 lzma", "int6_lzma")
    if baseline and baseline.get("int6_lzma"):
        delta_row("  vs baseline", "int6_lzma", baseline.get("int6_lzma"))
    print()

    # Packed int6
    row("int6 packed raw", "int6_packed_raw")
    row("int6 packed zlib", "int6_packed_zlib")
    row("int6 packed lzma", "int6_packed_lzma")
    if baseline and baseline.get("int6_packed_lzma"):
        delta_row("  vs baseline", "int6_packed_lzma", baseline.get("int6_packed_lzma"))
    print()

    # LZMA advantage over zlib
    print(f"{'LZMA advantage over zlib:':<30}")
    for key_prefix in ["int8", "int6", "int6_packed"]:
        print(f"  {key_prefix:<28}", end="")
        for m in all_metrics:
            zlib_val = m.get(f"{key_prefix}_zlib")
            lzma_val = m.get(f"{key_prefix}_lzma")
            if zlib_val and lzma_val and zlib_val > 0:
                adv = (1 - lzma_val / zlib_val) * 100
                print(f"  {adv:>14.1f}%", end="")
            else:
                print(f"  {'N/A':>16}", end="")
        print()

    # BPB cost vs baseline
    if baseline and baseline.get("val_bpb"):
        print(f"\n{'BPB cost vs baseline:':<30}")
        for m in all_metrics:
            bpb = m.get("val_bpb")
            if bpb is not None:
                cost = bpb - baseline["val_bpb"]
                print(f"  {m['name']}: {'+' if cost >= 0 else ''}{cost:.5f}")

    print(f"\n{'='*80}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Warmdown smoothness CUDA transfer test")
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--warmdown", type=int, default=DEFAULTS["warmdown"])
    parser.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    parser.add_argument("--batch-tokens", type=int, default=DEFAULTS["batch_tokens"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--data-path", default=DEFAULTS["data_path"])
    parser.add_argument("--tokenizer-path", default=DEFAULTS["tokenizer_path"])
    parser.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    parser.add_argument("--val-loss-every", type=int, default=DEFAULTS["val_loss_every"])
    parser.add_argument("--val-batch-size", type=int, default=DEFAULTS["val_batch_size"])
    parser.add_argument("--val-tokens-max", type=int, default=DEFAULTS["val_tokens_max"])
    parser.add_argument("--output-dir", default="experiments/smooth_test")
    parser.add_argument("--lambda-only", type=float, default=None,
                        help="Run only the variant with this lambda value")
    args = parser.parse_args()

    # Select variants
    if args.lambda_only is not None:
        variants = {k: v for k, v in VARIANTS.items() if v == args.lambda_only}
        if not variants:
            # Custom lambda
            variants = {f"smooth_{args.lambda_only}": args.lambda_only}
    else:
        variants = VARIANTS

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for name, lam in variants.items():
        metrics = run_variant(name, lam, args)
        all_metrics.append(metrics)
        if "error" in metrics:
            print(f"  ERROR: {metrics['error'][:200]}")

    # Save combined results
    combined_path = Path(args.output_dir) / "results.json"
    combined_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nResults saved to {combined_path}")

    # Print comparison if we have multiple results
    valid = [m for m in all_metrics if "error" not in m]
    if len(valid) >= 2:
        print_comparison(valid)
    elif len(valid) == 1:
        m = valid[0]
        print(f"\nSingle run results ({m['name']}):")
        for k in ["val_bpb", "post_quant_bpb", "int8_zlib", "int8_lzma",
                   "int6_zlib", "int6_lzma", "int6_packed_zlib", "int6_packed_lzma"]:
            v = m.get(k)
            if v is not None:
                if isinstance(v, int):
                    print(f"  {k}: {fmt_bytes(v)}")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
