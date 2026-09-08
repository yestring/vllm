# SPDX-License-Identifier: Apache-2.0
"""Plot one or more push-connector benchmark result directories."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001


METRICS = {
    "ttft_ms_p50": ("TTFT p50 (ms)", False),
    "ttft_ms_p99": ("TTFT p99 (ms)", False),
    "tpot_ms_p50": ("TPOT p50 (ms/token)", False),
    "e2e_ms_p99": ("E2E latency p99 (ms)", False),
    "output_throughput_tps": ("Output throughput (token/s)", True),
    "request_throughput_rps": ("Request throughput (req/s)", True),
    "nixl_effective_gib_per_s": ("NIXL effective bandwidth (GiB/s)", True),
    "nixl_transfer_count": ("NIXL transfer operations", False),
}


def read_rows(directory: Path) -> list[dict[str, Any]]:
    with (directory / "summary.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in row:
            if key != "label":
                row[key] = float(row[key])
    return rows


def plot_metric(
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    series = sorted(
        {
            (str(row["label"]), int(row["input_tokens"]))
            for row in rows
        }
    )
    for label, input_tokens in series:
        selected = sorted(
            (
                row
                for row in rows
                if row["label"] == label
                and int(row["input_tokens"]) == input_tokens
            ),
            key=lambda row: row["concurrency"],
        )
        axis.plot(
            [row["concurrency"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            linewidth=2,
            label=f"{label}, input={input_tokens}",
        )
    axis.set_xlabel("Concurrency")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} vs concurrency")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / f"{metric}.png", dpi=180)
    figure.savefig(output_dir / f"{metric}.pdf")
    plt.close(figure)


def write_comparison(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    labels = sorted({str(row["label"]) for row in rows})
    if len(labels) != 2:
        return
    baseline_label, candidate_label = labels
    indexed = {
        (
            str(row["label"]),
            int(row["input_tokens"]),
            int(row["concurrency"]),
        ): row
        for row in rows
    }
    comparison_rows: list[dict[str, Any]] = []
    scenarios = sorted(
        {
            (int(row["input_tokens"]), int(row["concurrency"]))
            for row in rows
        }
    )
    for input_tokens, concurrency in scenarios:
        baseline = indexed.get((baseline_label, input_tokens, concurrency))
        candidate = indexed.get((candidate_label, input_tokens, concurrency))
        if baseline is None or candidate is None:
            continue
        result: dict[str, Any] = {
            "baseline": baseline_label,
            "candidate": candidate_label,
            "input_tokens": input_tokens,
            "concurrency": concurrency,
        }
        for metric, (_, higher_is_better) in METRICS.items():
            base_value = float(baseline[metric])
            candidate_value = float(candidate[metric])
            if not math.isfinite(base_value) or base_value == 0:
                improvement = math.nan
            elif higher_is_better:
                improvement = (candidate_value / base_value - 1) * 100
            else:
                improvement = (1 - candidate_value / base_value) * 100
            result[f"{metric}_improvement_pct"] = improvement
        comparison_rows.append(result)

    if not comparison_rows:
        return
    with (output_dir / "comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    metric = "ttft_ms_p50_improvement_pct"
    input_sizes = sorted({row["input_tokens"] for row in comparison_rows})
    concurrencies = sorted({row["concurrency"] for row in comparison_rows})
    values = {
        (row["input_tokens"], row["concurrency"]): row[metric]
        for row in comparison_rows
    }
    matrix = [
        [
            values.get((input_tokens, concurrency), math.nan)
            for concurrency in concurrencies
        ]
        for input_tokens in input_sizes
    ]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    image = axis.imshow(matrix, cmap="RdYlGn", aspect="auto")
    for y, input_tokens in enumerate(input_sizes):
        for x, concurrency in enumerate(concurrencies):
            value = matrix[y][x]
            text = "N/A" if not math.isfinite(value) else f"{value:+.1f}%"
            axis.text(x, y, text, ha="center", va="center", fontsize=9)
    axis.set_xticks(range(len(concurrencies)), concurrencies)
    axis.set_yticks(range(len(input_sizes)), input_sizes)
    axis.set_xlabel("Concurrency")
    axis.set_ylabel("Input tokens")
    axis.set_title(
        f"TTFT p50 improvement: {candidate_label} vs {baseline_label}"
    )
    figure.colorbar(image, ax=axis, label="Improvement (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "ttft_improvement_heatmap.png", dpi=180)
    figure.savefig(output_dir / "ttft_improvement_heatmap.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for result_dir in args.result_dirs
        for row in read_rows(result_dir)
    ]
    for metric, (ylabel, _) in METRICS.items():
        plot_metric(rows, metric, ylabel, args.output_dir)
    write_comparison(rows, args.output_dir)


if __name__ == "__main__":
    main()
