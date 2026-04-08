from __future__ import annotations

import re
import sys
from pathlib import Path


PATTERNS = {
    "smooth": re.compile(r"smooth_lambda:(?P<value>\S+)"),
    "stop": re.compile(r"stopping_early: wallclock_cap train_time:(?P<value>\S+)"),
    "serialized": re.compile(r"Serialized model int6\+lzma: (?P<value>\d+) bytes"),
    "total_size": re.compile(r"Total submission size int6\+lzma: (?P<value>\d+) bytes"),
    "final_exact": re.compile(r"final_int6_roundtrip_exact val_loss:(?P<loss>\S+) val_bpb:(?P<bpb>\S+)"),
    "sliding_exact": re.compile(r"final_int6_sliding_window_s64_exact val_loss:(?P<loss>\S+) val_bpb:(?P<bpb>\S+)"),
}


def extract_metrics(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {"log": path.name}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for key, pattern in PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            if "bpb" in match.groupdict():
                metrics[f"{key}_bpb"] = match.group("bpb")
                metrics[f"{key}_loss"] = match.group("loss")
            else:
                metrics[key] = match.group("value")
    return metrics


def main() -> int:
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        print(f"no logs found in {log_dir}")
        return 1

    print("summary")
    for path in logs:
        metrics = extract_metrics(path)
        print(
            " | ".join(
                [
                    metrics.get("log", path.name),
                    f"smooth={metrics.get('smooth', 'n/a')}",
                    f"final_bpb={metrics.get('final_exact_bpb', 'n/a')}",
                    f"slide64_bpb={metrics.get('sliding_exact_bpb', 'n/a')}",
                    f"size={metrics.get('total_size', 'n/a')}",
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
