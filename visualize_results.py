import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.base.pipeline import Pipelines
from src.base.schema import PipelineError
from src.constants import charts_dir, results_dir
from src.inference import Models


class ResultMeta:
    def __init__(self, filename: str):
        name = filename.removesuffix(".csv")
        timestamp_pattern = r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}"
        timestamp_match = re.search(timestamp_pattern, name)
        self.timestamp = timestamp_match.group() if timestamp_match else "unknown"
        self.model = "unknown"
        for m in sorted(Models, key=lambda x: len(x.value), reverse=True):
            if m.value in name:
                self.model = m
                break
        self.pipeline = "unknown"
        for p in sorted(Pipelines, key=lambda x: len(x.value), reverse=True):
            if p.value in name:
                self.pipeline = p
                break


def get_tool_call_columns(df: pl.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.endswith("_calls") and c != "num_model_calls"]
    )


def get_error_types(df: pl.DataFrame) -> pl.DataFrame:
    all_categories = ["success"] + [e.value for e in PipelineError]
    error_counts = (
        df.select("error")
        .with_columns(pl.col("error").fill_null("success"))
        .group_by("error")
        .len()
    )
    result = pl.DataFrame({"error": all_categories})
    result = result.join(error_counts, on="error", how="left")
    result = result.with_columns(pl.col("len").fill_null(0))
    return result


def plot_error_distribution(df: pl.DataFrame, result_meta: ResultMeta, save_path: Path):
    error_df = get_error_types(df)
    _, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "#2ecc71" if e == "success" else "#e74c3c" for e in error_df["error"].to_list()
    ]
    ax.bar(error_df["error"].to_list(), error_df["len"].to_list(), color=colors)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution\n{result_meta.pipeline} | {result_meta.model}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tool_calls(df: pl.DataFrame, result_meta: ResultMeta, save_path: Path):
    tool_cols = get_tool_call_columns(df)
    if not tool_cols:
        return
    means = [df[col].mean() for col in tool_cols]
    labels = [c.replace("_calls", "") for c in tool_cols]
    _, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, means, color="#3498db")  # type: ignore
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xlabel("Tool")
    ax.set_ylabel("Average Calls per Run")
    ax.set_title(f"Average Tool Calls\n{result_meta.pipeline} | {result_meta.model}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_performance_metrics(
    df: pl.DataFrame, result_meta: ResultMeta, save_path: Path
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        df["elapsed_time"].to_list(), bins=15, color="#1abc9c", edgecolor="black"
    )
    axes[0].axvline(
        df["elapsed_time"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['elapsed_time'].mean():.2f}s",
    )
    axes[0].set_xlabel("Elapsed Time (s)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Elapsed Time Distribution")
    axes[0].legend()
    axes[1].hist(
        df["num_model_calls"].to_list(), bins=15, color="#f39c12", edgecolor="black"
    )
    axes[1].axvline(
        df["num_model_calls"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['num_model_calls'].mean():.2f}",
    )
    axes[1].set_xlabel("Number of LLM Requests")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("LLM Request Count Distribution")
    axes[1].legend()
    fig.suptitle(f"{result_meta.pipeline} | {result_meta.model}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_combined_overview(df: pl.DataFrame, result_meta: ResultMeta, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    error_df = get_error_types(df)
    colors = [
        "#2ecc71" if e == "success" else "#e74c3c" for e in error_df["error"].to_list()
    ]
    axes[0].bar(error_df["error"].to_list(), error_df["len"].to_list(), color=colors)
    axes[0].set_xlabel("Error Type")
    axes[0].axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution")
    axes[0].tick_params(axis="x", rotation=45)
    tool_cols = get_tool_call_columns(df)
    if tool_cols:
        means = [df[col].mean() for col in tool_cols]
        labels = [c.replace("_calls", "") for c in tool_cols]
        axes[1].bar(labels, means, color="#3498db")
        axes[1].set_xlabel("Tool")
        axes[1].set_ylabel("Average Calls")
        axes[1].set_title("Tool Call Distribution")
        axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle(f"Overview: {result_meta.pipeline} | {result_meta.model}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_success_rate_by_model(
    all_results: list[tuple[pl.DataFrame, ResultMeta]], save_path: Path
):
    model_results: dict[str, list[tuple[pl.DataFrame, ResultMeta]]] = defaultdict(list)
    for df, result_meta in all_results:
        model_results[result_meta.model].append((df, result_meta))
    all_pipelines = sorted(set(result_meta.pipeline for _, result_meta in all_results))
    cmap = plt.colormaps.get_cmap("Set2")
    pipeline_colors = [
        cmap(i / max(len(all_pipelines) - 1, 1)) for i in range(len(all_pipelines))
    ]
    pipeline_color_map = dict(zip(all_pipelines, pipeline_colors, strict=True))
    models = sorted(model_results.keys())
    n_models = len(models)
    n_pipelines = len(all_pipelines)
    x = np.arange(n_models)
    width = 0.8 / max(n_pipelines, 1)
    _, ax = plt.subplots(figsize=(14, 7))
    for i, pipeline in enumerate(all_pipelines):
        rates = []
        stds = []
        for model in models:
            matching = [
                (df, result_meta)
                for df, result_meta in model_results[model]
                if result_meta.pipeline == pipeline
            ]
            if matching:
                combined_df: pl.DataFrame = pl.concat([df for df, _ in matching])
                success_df = combined_df.with_columns(
                    pl.col("error").is_null().cast(pl.Int32).alias("success")
                )
                success_mean = success_df["success"].mean()
                success_std = success_df["success"].std()
                assert isinstance(success_mean, (int, float)) and isinstance(
                    success_std, (int, float)
                )
                rate = success_mean * 100
                std_err = success_std / np.sqrt(len(success_df)) * 100
                rates.append(rate)
                stds.append(std_err)
            else:
                rates.append(0)
                stds.append(0)
        offset = (i - n_pipelines / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=pipeline,
            color=pipeline_color_map[pipeline],
            yerr=stds,
            capsize=3,
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate Comparison Across All Runs")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Pipeline", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_avg_time_by_model(
    all_results: list[tuple[pl.DataFrame, ResultMeta]], save_path: Path
):
    model_results: dict[str, list[tuple[pl.DataFrame, ResultMeta]]] = defaultdict(list)
    for df, result_meta in all_results:
        model_results[result_meta.model].append((df, result_meta))
    all_pipelines = sorted(set(result_meta.pipeline for _, result_meta in all_results))
    cmap = plt.colormaps.get_cmap("Set2")
    pipeline_colors = [
        cmap(i / max(len(all_pipelines) - 1, 1)) for i in range(len(all_pipelines))
    ]
    pipeline_color_map = dict(zip(all_pipelines, pipeline_colors, strict=True))
    models = sorted(model_results.keys())
    n_models = len(models)
    n_pipelines = len(all_pipelines)
    x = np.arange(n_models)
    width = 0.8 / max(n_pipelines, 1)
    _, ax = plt.subplots(figsize=(14, 7))
    for i, pipeline in enumerate(all_pipelines):
        times = []
        stds = []
        for model in models:
            matching = [
                (df, result_meta)
                for df, result_meta in model_results[model]
                if result_meta.pipeline == pipeline
            ]
            if matching:
                combined_df = pl.concat([df for df, _ in matching])
                elapsed_mean = combined_df["elapsed_time"].mean()
                elapsed_std = combined_df["elapsed_time"].std()
                assert isinstance(elapsed_mean, (int, float)) and isinstance(
                    elapsed_std, (int, float)
                )
                std_err = elapsed_std / np.sqrt(len(combined_df))
                times.append(elapsed_mean)
                stds.append(std_err)
            else:
                times.append(0)
                stds.append(0)
        offset = (i - n_pipelines / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            times,
            width,
            label=pipeline,
            color=pipeline_color_map[pipeline],
            yerr=stds,
            capsize=3,
        )
        ax.bar_label(bars, fmt="%.1fs", fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Elapsed Time (s)")
    ax.set_title("Average Elapsed Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Pipeline", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pipeline_comparison_by_model(
    all_results: list[tuple[pl.DataFrame, ResultMeta]], save_dir: Path
):
    """Compare pipelines for the same model."""
    model_results: dict[str, list[tuple[pl.DataFrame, ResultMeta]]] = defaultdict(list)
    for df, result_meta in all_results:
        model_results[result_meta.model].append((df, result_meta))
    for model, results in model_results.items():
        if len(results) < 2:
            continue
        unique_pipelines = list(
            dict.fromkeys(result_meta.pipeline for _, result_meta in results)
        )
        pipeline_data: dict[str, dict] = {}
        for pipeline in unique_pipelines:
            matching = [
                (df, result_meta)
                for df, result_meta in results
                if result_meta.pipeline == pipeline
            ]
            matching_sorted = sorted(
                matching, key=lambda x: x[1].timestamp, reverse=True
            )
            latest_df, _ = matching_sorted[0]
            total = len(latest_df)
            success = latest_df.filter(pl.col("error").is_null()).height
            pipeline_data[pipeline] = {
                "df": latest_df,
                "success_rate": (success / total) * 100 if total > 0 else 0,
                "avg_time": latest_df["elapsed_time"].mean(),
                "avg_llm_requests": latest_df["num_model_calls"].mean(),
            }
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        success_rates = [pipeline_data[p]["success_rate"] for p in unique_pipelines]
        bars = axes[0, 0].bar(unique_pipelines, success_rates, color="#27ae60")
        axes[0, 0].bar_label(bars, fmt="%.1f%%")
        axes[0, 0].set_ylabel("Success Rate (%)")
        axes[0, 0].set_title("Success Rate by Pipeline")
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].tick_params(axis="x", rotation=45)
        avg_times = [pipeline_data[p]["avg_time"] for p in unique_pipelines]
        bars = axes[0, 1].bar(unique_pipelines, avg_times, color="#e67e22")
        axes[0, 1].bar_label(bars, fmt="%.1fs")
        axes[0, 1].set_ylabel("Average Time (s)")
        axes[0, 1].set_title("Average Elapsed Time by Pipeline")
        axes[0, 1].tick_params(axis="x", rotation=45)
        avg_llm = [pipeline_data[p]["avg_llm_requests"] for p in unique_pipelines]
        bars = axes[1, 0].bar(unique_pipelines, avg_llm, color="#3498db")
        axes[1, 0].bar_label(bars, fmt="%.1f")
        axes[1, 0].set_ylabel("Average LLM Requests")
        axes[1, 0].set_title("Average LLM Requests by Pipeline")
        axes[1, 0].tick_params(axis="x", rotation=45)
        all_tool_cols = set()
        for p in unique_pipelines:
            all_tool_cols.update(get_tool_call_columns(pipeline_data[p]["df"]))
        all_tool_cols = sorted(all_tool_cols)
        if all_tool_cols:
            n_pipelines = len(unique_pipelines)
            n_tools = len(all_tool_cols)
            x = np.arange(n_tools)
            width = 0.8 / n_pipelines
            for i, pipeline in enumerate(unique_pipelines):
                df = pipeline_data[pipeline]["df"]
                means = [
                    df[col].mean() if col in df.columns else 0 for col in all_tool_cols
                ]
                offset = (i - n_pipelines / 2 + 0.5) * width
                axes[1, 1].bar(x + offset, means, width, label=pipeline)
            axes[1, 1].set_xticks(x, [c.replace("_calls", "") for c in all_tool_cols])
            axes[1, 1].tick_params(axis="x", rotation=45)
            for label in axes[1, 1].get_xticklabels():
                label.set_ha("right")
            axes[1, 1].set_ylabel("Average Calls")
            axes[1, 1].set_title("Tool Call Comparison")
            axes[1, 1].legend(loc="upper right")
        fig.suptitle(f"Pipeline Comparison for Model: {model}", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f"pipeline_comparison_{model}.png", dpi=150)
        plt.close()


MIN_ENTRIES = 30


def main():
    charts_dir.mkdir(exist_ok=True)
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in results/")
        return
    all_results: list[tuple[pl.DataFrame, ResultMeta]] = []
    for csv_path in csv_files:
        result_meta = ResultMeta(csv_path.name)
        df = pl.read_csv(csv_path)
        base_name = csv_path.stem
        subdir = charts_dir / base_name
        subdir.mkdir(exist_ok=True)
        print(f"Processing: {csv_path.name} ({len(df)} entries)")
        plot_error_distribution(df, result_meta, subdir / "error_distribution.png")
        plot_tool_calls(df, result_meta, subdir / "tool_calls.png")
        plot_performance_metrics(df, result_meta, subdir / "performance_metrics.png")
        plot_combined_overview(df, result_meta, subdir / "combined_overview.png")
        if len(df) >= MIN_ENTRIES:
            all_results.append((df, result_meta))
        else:
            print(f"  -> Skipping from comparisons (< {MIN_ENTRIES} entries)")
    if all_results:
        plot_success_rate_by_model(
            all_results, charts_dir / "comparison_success_rate.png"
        )
        plot_avg_time_by_model(all_results, charts_dir / "comparison_avg_time.png")
        plot_pipeline_comparison_by_model(all_results, charts_dir)
    print(f"\nCharts saved to {charts_dir}/")


if __name__ == "__main__":
    main()
