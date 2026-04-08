"""
GPTQ survival test for warmdown smoothness regularizer (Issue #3, Step 2).

Tests whether GPTQ quantization erases, preserves, or partially preserves
the compression gain created by smoothness regularization.

Uses existing checkpoints from Step 1 — no retraining.

Usage:
    python experiment_gptq.py                          # all 3 checkpoints
    python experiment_gptq.py --checkpoints baseline   # baseline only
"""

from __future__ import annotations

import argparse
import io
import json
import lzma
import math
import os
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

from train_gpt import (
    GPT,
    CastedLinear,
    Hyperparameters,
    build_sentencepiece_luts,
    eval_val,
    load_data_shard,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)
from experiment_smooth import pack_int6

# ---------------------------------------------------------------------------
# SDP backend setup (RTX 2070 Super = SM 7.5, no flash GQA support)
# ---------------------------------------------------------------------------

def setup_sdp_backends():
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp,
        enable_math_sdp, enable_mem_efficient_sdp,
    )
    sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    enable_cudnn_sdp(False)
    enable_flash_sdp(sm_major >= 8)
    enable_mem_efficient_sdp(sm_major < 8)
    enable_math_sdp(sm_major < 8)

# ---------------------------------------------------------------------------
# GPTQ core functions
# Copied from records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py
# ---------------------------------------------------------------------------

GPTQ_SETTINGS = {
    "bits": 6,
    "clip_range": 31,
    "block_size": 128,
    "scale": "per-row symmetric",
    "column_reorder": "by Hessian diagonal (descending)",
    "percentile_candidates": [0.9990, 0.9995, 0.9999, 0.99999, 1.0],
    "hessian_damping": "0.01 * mean(diag(H))",
}


def _quantize_int6_percentile(t32: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    """Fallback: percentile search (for 1D or no-Hessian cases)."""
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_int6_gptq(
    weight: Tensor, hessian: Tensor | None = None,
    clip_range: int = 31, block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, clip_range)

    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp

    # Column reordering by Hessian diagonal (largest-margin-first)
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]

    # H^{-1} via Cholesky
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    best_q, best_scale, best_err = None, None, float("inf")
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()

        # Sequential column-wise quantization with Hessian-guided error propagation
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse

    best_q = best_q[:, inv_perm]  # undo column permutation
    return best_q, best_scale

# ---------------------------------------------------------------------------
# Hessian collection (adapted from the GPTQ submission)
# ---------------------------------------------------------------------------

KEEP_FLOAT_MAX_NUMEL = 65_536


def collect_hessians(
    model: nn.Module, calib_tokens: Tensor, device: torch.device,
    seq_len: int = 1024, num_batches: int = 16, batch_seqs: int = 4,
) -> dict[str, Tensor]:
    """Collect H = X^T X for each CastedLinear via forward hooks on training data."""
    hessians: dict[str, Tensor] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device="cpu")

            def make_hook(pname):
                def hook_fn(mod, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(param_name)))

    model.eval()
    tokens_per_batch = batch_seqs * (seq_len + 1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for b in range(num_batches):
            start = b * tokens_per_batch
            chunk = calib_tokens[start : start + tokens_per_batch].to(device, dtype=torch.int64)
            chunk = chunk[: batch_seqs * (seq_len + 1)].reshape(batch_seqs, seq_len + 1)
            x = chunk[:, :-1]
            y = chunk[:, 1:]
            model(x, y)

    for h in hooks:
        h.remove()

    # Normalize and add damping
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H

    return hessians

# ---------------------------------------------------------------------------
# Quantization wrappers
# ---------------------------------------------------------------------------

def quantize_all_gptq(
    sd: dict[str, Tensor], hessians: dict[str, Tensor],
) -> dict[str, tuple[Tensor, Tensor | None]]:
    """GPTQ int6 for large 2D CastedLinear weights; percentile for tok_emb; passthrough rest."""
    result = {}
    for name, t in sd.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= KEEP_FLOAT_MAX_NUMEL:
            result[name] = (t, None)  # passthrough
            continue
        H = hessians.get(name)
        q, s = quantize_int6_gptq(t, hessian=H)
        result[name] = (q, s)
    return result


def quantize_all_naive(sd: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor | None]]:
    """Naive int6 (percentile search, no Hessian) for all large float tensors."""
    result = {}
    for name, t in sd.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= KEEP_FLOAT_MAX_NUMEL:
            result[name] = (t, None)  # passthrough
            continue
        q, s = _quantize_int6_percentile(t.float())
        result[name] = (q, s)
    return result

# ---------------------------------------------------------------------------
# Dequantization (load quantized weights back into model for eval)
# ---------------------------------------------------------------------------

def dequantize_to_state_dict(
    quantized: dict[str, tuple[Tensor, Tensor | None]],
    original_sd: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Reconstruct float state dict from quantized weights."""
    out = {}
    for name in original_sd:
        q, s = quantized[name]
        if s is None:
            # passthrough — use original dtype
            out[name] = q.to(original_sd[name].dtype)
        elif s.ndim == 0:
            # per-tensor scale
            out[name] = (q.float() * s.float()).to(original_sd[name].dtype)
        else:
            # per-row scale
            out[name] = (q.float() * s.float()[:, None]).to(original_sd[name].dtype)
    return out

# ---------------------------------------------------------------------------
# Compression + structure metrics
# ---------------------------------------------------------------------------

def compress_quantized(quantized: dict[str, tuple[Tensor, Tensor | None]]) -> dict[str, int | bytes]:
    """Pack int6 quantized weights and compress with zlib + LZMA."""
    # Collect all quantized int values (skip passthrough)
    all_q = []
    for name, (q, s) in quantized.items():
        if s is not None and q.dtype == torch.int8:
            all_q.append(q.numpy().flatten())

    if not all_q:
        return {"packed_raw": 0, "packed_zlib": 0, "packed_lzma": 0, "packed_bytes": b""}

    all_values = np.concatenate(all_q)
    packed = pack_int6(all_values)

    return {
        "packed_raw": len(packed),
        "packed_zlib": len(zlib.compress(packed, level=9)),
        "packed_lzma": len(lzma.compress(packed, preset=6)),
        "packed_bytes": packed,
    }


def compute_entropy(data: bytes) -> float:
    """Empirical Shannon entropy in bits/byte."""
    if len(data) == 0:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_run_length_proxy(data: bytes) -> float:
    """Fraction of consecutive identical bytes."""
    if len(data) < 2:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    matches = int(np.sum(arr[:-1] == arr[1:]))
    return matches / (len(arr) - 1)

# ---------------------------------------------------------------------------
# Model building + eval
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> GPT:
    """Instantiate the 17M baseline GPT model (uncompiled)."""
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    return model


def run_eval(
    model: nn.Module, device: torch.device, val_tokens: Tensor,
    luts: tuple, seq_len: int = 1024, val_batch_size: int = 32_768,
) -> tuple[float, float]:
    """Evaluate val_loss and val_bpb (single GPU, no DDP, uncompiled model)."""
    args = Hyperparameters()
    args.train_seq_len = seq_len
    args.val_batch_size = val_batch_size
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts
    return eval_val(
        args, model, rank=0, world_size=1, device=device,
        grad_accum_steps=1, val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "baseline": ("experiments/smooth_cuda_test1/baseline/final_model.pt", 0.0),
    "smooth_weak": ("experiments/smooth_cuda_test1/smooth_weak/final_model.pt", 3e-5),
    "smooth_strong": ("experiments/smooth_cuda_test1/smooth_strong/final_model.pt", 1e-4),
}


def run_one_checkpoint(
    name: str, checkpoint_path: str, lam: float,
    model: GPT, device: torch.device,
    calib_tokens: Tensor, val_tokens: Tensor, luts: tuple,
    num_calib_batches: int = 16, calib_batch_seqs: int = 4,
) -> dict:
    """Full GPTQ survival test pipeline for one checkpoint."""
    print(f"\n{'='*60}")
    print(f"  {name} (lambda={lam})")
    print(f"{'='*60}")
    t_start = time.time()

    # 1. Load checkpoint
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    print(f"  Loaded checkpoint ({sum(t.numel() for t in sd.values())} params)")

    # 2. Pre-quant eval
    t0 = time.time()
    pre_loss, pre_bpb = run_eval(model, device, val_tokens, luts)
    print(f"  pre_quant_bpb: {pre_bpb:.5f} ({time.time()-t0:.1f}s)")

    # 3. Collect Hessians
    t0 = time.time()
    hessians = collect_hessians(
        model, calib_tokens, device,
        num_batches=num_calib_batches, batch_seqs=calib_batch_seqs,
    )
    hess_time = time.time() - t0
    # Log a few condition numbers for sanity
    sample_names = list(hessians.keys())[:3]
    for hn in sample_names:
        H = hessians[hn]
        cond = torch.linalg.cond(H).item()
        print(f"  hessian {hn}: shape={tuple(H.shape)} cond={cond:.1f}")
    print(f"  Collected {len(hessians)} Hessians ({hess_time:.1f}s)")

    # 4. GPTQ int6 quantization
    t0 = time.time()
    gptq_quant = quantize_all_gptq(sd, hessians)
    gptq_time = time.time() - t0
    print(f"  GPTQ quantization: {gptq_time:.1f}s")

    # 5. Naive int6 quantization
    t0 = time.time()
    naive_quant = quantize_all_naive(sd)
    naive_time = time.time() - t0
    print(f"  Naive quantization: {naive_time:.1f}s")

    # 6. Compress both
    gptq_comp = compress_quantized(gptq_quant)
    naive_comp = compress_quantized(naive_quant)

    # 7. Structure metrics
    gptq_entropy = compute_entropy(gptq_comp["packed_bytes"])
    naive_entropy = compute_entropy(naive_comp["packed_bytes"])
    gptq_rlp = compute_run_length_proxy(gptq_comp["packed_bytes"])
    naive_rlp = compute_run_length_proxy(naive_comp["packed_bytes"])

    # 8. Post-GPTQ eval
    t0 = time.time()
    gptq_sd = dequantize_to_state_dict(gptq_quant, sd)
    model.load_state_dict(gptq_sd, strict=True)
    gptq_loss, gptq_bpb = run_eval(model, device, val_tokens, luts)
    print(f"  post_gptq_bpb: {gptq_bpb:.5f} ({time.time()-t0:.1f}s)")

    # 9. Post-naive eval
    t0 = time.time()
    naive_sd = dequantize_to_state_dict(naive_quant, sd)
    model.load_state_dict(naive_sd, strict=True)
    naive_loss, naive_bpb = run_eval(model, device, val_tokens, luts)
    print(f"  post_naive_bpb: {naive_bpb:.5f} ({time.time()-t0:.1f}s)")

    # Verify GPTQ differs from naive
    n_differ = sum(
        1 for k in gptq_quant
        if gptq_quant[k][1] is not None
        and not torch.equal(gptq_quant[k][0], naive_quant[k][0])
    )
    n_quantized = sum(1 for k in gptq_quant if gptq_quant[k][1] is not None)
    print(f"  GPTQ differs from naive: {n_differ}/{n_quantized} tensors")

    total_time = time.time() - t_start
    print(f"  Total: {total_time:.1f}s")

    return {
        "name": name,
        "lambda": lam,
        "pre_quant_bpb": pre_bpb,
        "naive_packed_zlib": naive_comp["packed_zlib"],
        "naive_packed_lzma": naive_comp["packed_lzma"],
        "naive_packed_raw": naive_comp["packed_raw"],
        "naive_bpb": naive_bpb,
        "naive_entropy": naive_entropy,
        "naive_rlp": naive_rlp,
        "gptq_packed_zlib": gptq_comp["packed_zlib"],
        "gptq_packed_lzma": gptq_comp["packed_lzma"],
        "gptq_packed_raw": gptq_comp["packed_raw"],
        "gptq_bpb": gptq_bpb,
        "gptq_entropy": gptq_entropy,
        "gptq_rlp": gptq_rlp,
        "gptq_quant_time_s": gptq_time,
        "hessian_time_s": hess_time,
        "total_time_s": total_time,
        "gptq_settings": GPTQ_SETTINGS,
    }

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def fmt_bytes(b: int | float) -> str:
    if b >= 1_000_000:
        return f"{b / 1_000_000:.2f} MB"
    return f"{b / 1_000:.1f} KB"


def print_results(results: list[dict]):
    """Print the comparison table."""
    baseline = next((r for r in results if r["lambda"] == 0.0), None)

    print(f"\n{'='*78}")
    print("GPTQ SURVIVAL TEST RESULTS")
    print(f"{'='*78}")

    # Header
    print(f"{'':30}", end="")
    for r in results:
        print(f"  {r['name']:>14}", end="")
    print()
    print(f"{'':30}", end="")
    for r in results:
        print(f"  {'λ=' + str(r['lambda']):>14}", end="")
    print()
    print("-" * (30 + 16 * len(results)))

    def row(label, key, fmt_fn=None):
        print(f"{label:30}", end="")
        for r in results:
            v = r.get(key, "N/A")
            if fmt_fn:
                print(f"  {fmt_fn(v):>14}", end="")
            elif isinstance(v, float):
                print(f"  {v:>14.5f}", end="")
            elif isinstance(v, int):
                print(f"  {fmt_bytes(v):>14}", end="")
            else:
                print(f"  {str(v):>14}", end="")
        print()

    row("pre_quant_bpb", "pre_quant_bpb")
    print()

    print("NAIVE int6:")
    row("  packed + zlib", "naive_packed_zlib")
    row("  packed + lzma", "naive_packed_lzma")
    row("  post_quant_bpb", "naive_bpb")
    row("  entropy (bits/byte)", "naive_entropy", lambda v: f"{v:.3f}")
    row("  run_length_proxy", "naive_rlp", lambda v: f"{v:.4f}")
    print()

    print("GPTQ int6:")
    row("  packed + zlib", "gptq_packed_zlib")
    row("  packed + lzma", "gptq_packed_lzma")
    row("  post_quant_bpb", "gptq_bpb")
    row("  entropy (bits/byte)", "gptq_entropy", lambda v: f"{v:.3f}")
    row("  run_length_proxy", "gptq_rlp", lambda v: f"{v:.4f}")
    print()

    # LZMA advantage over zlib
    print("LZMA advantage over zlib:")
    for prefix, label in [("naive", "  naive"), ("gptq", "  gptq")]:
        print(f"{label:30}", end="")
        for r in results:
            zlib_v = r[f"{prefix}_packed_zlib"]
            lzma_v = r[f"{prefix}_packed_lzma"]
            adv = (1 - lzma_v / zlib_v) * 100 if zlib_v > 0 else 0
            print(f"  {adv:>13.1f}%", end="")
        print()
    print()

    # Deltas vs baseline
    if baseline:
        print("DELTAS vs baseline (packed int6 + LZMA):")
        for r in results:
            if r["lambda"] == 0.0:
                continue
            naive_delta = (r["naive_packed_lzma"] - baseline["naive_packed_lzma"]) / baseline["naive_packed_lzma"] * 100
            gptq_delta = (r["gptq_packed_lzma"] - baseline["gptq_packed_lzma"]) / baseline["gptq_packed_lzma"] * 100
            bpb_cost = r["pre_quant_bpb"] - baseline["pre_quant_bpb"]
            gptq_bpb_cost = r["gptq_bpb"] - baseline["gptq_bpb"]
            print(f"  {r['name']}:")
            print(f"    naive LZMA:  {naive_delta:+.1f}%")
            print(f"    GPTQ LZMA:   {gptq_delta:+.1f}%  <-- KEY METRIC")
            print(f"    BPB cost (pre-quant):  {bpb_cost:+.5f}")
            print(f"    BPB cost (post-GPTQ):  {gptq_bpb_cost:+.5f}")
        print()

        # GPTQ vs naive quality improvement
        print("GPTQ quality improvement over naive (BPB):")
        for r in results:
            diff = r["naive_bpb"] - r["gptq_bpb"]
            print(f"  {r['name']:20} {diff:+.5f} ({'GPTQ better' if diff > 0 else 'naive better'})")
        print()

        # Structure survival
        print("ENTROPY CHANGE (naive -> GPTQ):")
        for r in results:
            delta = r["gptq_entropy"] - r["naive_entropy"]
            print(f"  {r['name']:20} {r['naive_entropy']:.3f} -> {r['gptq_entropy']:.3f} ({delta:+.3f})")
        print()

        print("RUN-LENGTH PROXY CHANGE (naive -> GPTQ):")
        for r in results:
            delta = r["gptq_rlp"] - r["naive_rlp"]
            print(f"  {r['name']:20} {r['naive_rlp']:.4f} -> {r['gptq_rlp']:.4f} ({delta:+.4f})")

    print(f"\n{'='*78}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPTQ survival test for smoothness regularizer")
    parser.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINTS.keys()),
                        choices=list(CHECKPOINTS.keys()),
                        help="Which checkpoints to test")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--val-tokens-max", type=int, default=2_000_000)
    parser.add_argument("--num-calib-batches", type=int, default=16)
    parser.add_argument("--calib-batch-seqs", type=int, default=4)
    parser.add_argument("--output-dir", default="experiments/gptq_survival_test")
    args = parser.parse_args()

    setup_sdp_backends()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load validation data
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(
        os.path.join(args.data_path, "fineweb_val_*.bin"),
        Hyperparameters.train_seq_len,
    )
    if args.val_tokens_max > 0 and val_tokens.numel() > args.val_tokens_max:
        val_tokens = val_tokens[:args.val_tokens_max]
    luts = build_sentencepiece_luts(sp, Hyperparameters.vocab_size, device)
    print(f"Validation tokens: {val_tokens.numel()}")

    # Load calibration data (first training shard)
    train_shard = Path(args.data_path) / "fineweb_train_000000.bin"
    calib_tokens = load_data_shard(train_shard)
    tokens_needed = args.num_calib_batches * args.calib_batch_seqs * (Hyperparameters.train_seq_len + 1)
    calib_tokens = calib_tokens[:tokens_needed]
    print(f"Calibration tokens: {calib_tokens.numel()} ({args.num_calib_batches} batches x {args.calib_batch_seqs} seqs)")

    # Build model (uncompiled, single GPU)
    model = build_model(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    # Log GPTQ settings
    print(f"GPTQ settings: {json.dumps(GPTQ_SETTINGS, indent=2)}")

    # Run experiment
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []
    for name in args.checkpoints:
        path, lam = CHECKPOINTS[name]
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        result = run_one_checkpoint(
            name, path, lam, model, device,
            calib_tokens, val_tokens, luts,
            num_calib_batches=args.num_calib_batches,
            calib_batch_seqs=args.calib_batch_seqs,
        )
        all_results.append(result)

    # Save JSON (strip packed_bytes from serialization)
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items()}
        jr.pop("gptq_settings", None)  # already logged
        json_results.append(jr)
    out_path = Path(args.output_dir) / "results.json"
    out_path.write_text(json.dumps(json_results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    if all_results:
        print_results(all_results)


if __name__ == "__main__":
    main()
