import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")
ITER_RE = re.compile(r"iter\s+(\d+):\s+loss\s+([0-9.]+),\s+time\s+([0-9.]+)ms,\s+mfu\s+(-?[0-9.]+)%")
PARAM_RE = re.compile(r"number of parameters:\s+([0-9.]+)M")
TOKENS_RE = re.compile(r"tokens per iteration will be:\s+([0-9,]+)")


def parse_log(path):
    metrics = {
        "path": str(path),
        "params_m": None,
        "tokens_per_iter": None,
        "eval": [],
        "iter": [],
    }
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if metrics["params_m"] is None:
            m = PARAM_RE.search(line)
            if m:
                metrics["params_m"] = float(m.group(1))
        if metrics["tokens_per_iter"] is None:
            m = TOKENS_RE.search(line)
            if m:
                metrics["tokens_per_iter"] = int(m.group(1).replace(",", ""))
        m = STEP_RE.search(line)
        if m:
            metrics["eval"].append({
                "step": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)),
            })
        m = ITER_RE.search(line)
        if m:
            time_ms = float(m.group(3))
            tokens_per_iter = metrics["tokens_per_iter"]
            metrics["iter"].append({
                "iter": int(m.group(1)),
                "loss": float(m.group(2)),
                "time_ms": time_ms,
                "mfu": float(m.group(4)),
                "tokens_per_sec": tokens_per_iter / (time_ms / 1000.0) if tokens_per_iter else None,
            })
    return metrics


def last_eval(metrics):
    return metrics["eval"][-1] if metrics["eval"] else None


def avg_tail(values, key, tail=100, min_iter=5):
    rows = [row for row in values if row.get(key) is not None and row.get("iter", 0) >= min_iter]
    if not rows:
        return None
    rows = rows[-tail:]
    return sum(row[key] for row in rows) / len(rows)


def pct_change(old, new, higher_is_better=False):
    if old is None or new is None or old == 0:
        return None
    if higher_is_better:
        return (new - old) / old * 100.0
    return (old - new) / old * 100.0


def write_summary(out_dir, baseline_name, candidate_name, baseline, candidate):
    base_eval = last_eval(baseline)
    cand_eval = last_eval(candidate)
    base_time = avg_tail(baseline["iter"], "time_ms")
    cand_time = avg_tail(candidate["iter"], "time_ms")
    base_mfu = avg_tail(baseline["iter"], "mfu")
    cand_mfu = avg_tail(candidate["iter"], "mfu")
    base_tps = avg_tail(baseline["iter"], "tokens_per_sec")
    cand_tps = avg_tail(candidate["iter"], "tokens_per_sec")

    rows = [
        ("params_m", baseline["params_m"], candidate["params_m"], None),
        ("final_train_loss", base_eval["train_loss"] if base_eval else None, cand_eval["train_loss"] if cand_eval else None, False),
        ("final_val_loss", base_eval["val_loss"] if base_eval else None, cand_eval["val_loss"] if cand_eval else None, False),
        ("avg_tail_iter_time_ms", base_time, cand_time, False),
        ("avg_tail_mfu_percent", base_mfu, cand_mfu, True),
        ("avg_tail_tokens_per_sec", base_tps, cand_tps, True),
    ]

    md = []
    md.append(f"# {candidate_name} vs {baseline_name}\n")
    md.append("## Summary\n")
    md.append("| metric | baseline | candidate | change |\n")
    md.append("| --- | ---: | ---: | ---: |\n")
    for name, base_value, cand_value, higher_is_better in rows:
        if base_value is None or cand_value is None:
            change = "N/A"
        elif higher_is_better is None:
            change = f"{cand_value - base_value:+.4f}"
        else:
            change_value = pct_change(base_value, cand_value, higher_is_better=higher_is_better)
            change = f"{change_value:+.2f}%"
        base_str = "N/A" if base_value is None else f"{base_value:.4f}"
        cand_str = "N/A" if cand_value is None else f"{cand_value:.4f}"
        md.append(f"| {name} | {base_str} | {cand_str} | {change} |\n")

    md.append("\n## Notes\n")
    md.append("- `final_val_loss` is the most important quality metric for this Shakespeare char experiment.\n")
    md.append("- `avg_tail_iter_time_ms`, `avg_tail_mfu_percent`, and `avg_tail_tokens_per_sec` are averaged over the tail of training logs after warmup.\n")
    md.append("- Generated samples are useful for qualitative inspection, but should not be treated as the primary metric.\n")
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")

    summary_json = {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "baseline": baseline,
        "candidate": candidate,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


def plot_losses(out_dir, baseline_name, candidate_name, baseline, candidate):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for name, metrics in [(baseline_name, baseline), (candidate_name, candidate)]:
        if not metrics["eval"]:
            continue
        steps = [row["step"] for row in metrics["eval"]]
        train_loss = [row["train_loss"] for row in metrics["eval"]]
        val_loss = [row["val_loss"] for row in metrics["eval"]]
        plt.plot(steps, train_loss, linestyle="--", label=f"{name} train")
        plt.plot(steps, val_loss, label=f"{name} val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"Training curves: {candidate_name} vs {baseline_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, metrics in [(baseline_name, baseline), (candidate_name, candidate)]:
        rows = [row for row in metrics["iter"] if row["iter"] >= 5]
        if not rows:
            continue
        plt.plot([row["iter"] for row in rows], [row["time_ms"] for row in rows], label=name)
    plt.xlabel("iter")
    plt.ylabel("time per iter (ms)")
    plt.title(f"Iteration time: {candidate_name} vs {baseline_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "iter_time.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare two nanoGPT training logs.")
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--candidate-log", required=True)
    parser.add_argument("--baseline-name", default="original")
    parser.add_argument("--candidate-name", default="modern_v1")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = parse_log(args.baseline_log)
    candidate = parse_log(args.candidate_log)
    write_summary(out_dir, args.baseline_name, args.candidate_name, baseline, candidate)
    try:
        plot_losses(out_dir, args.baseline_name, args.candidate_name, baseline, candidate)
    except ImportError:
        (out_dir / "plot_error.txt").write_text(
            "matplotlib is not installed; install it to generate plots.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
