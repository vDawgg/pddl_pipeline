import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.constants import charts_dir, results_dir


def extract_problem_name(problem_path):
    if problem_path is None or problem_path == "":
        return "unknown"
    name = Path(problem_path).stem
    match = re.search(r"(needle_sorting|needle_transfer|ring_and_peg)_(\d+)", name)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return name


analyzed_dir = results_dir
relevant_dirs = {
    "gpt_oss_20b_optimized",
    "gpt_oss_120b",
    "gpt_oss_20b",
    "gemini_3_flash",
    "gemini_3_flash_image",
}
relevant_dir_matches = sorted(relevant_dirs, key=len, reverse=True)
csv_files = [
    f
    for rel_dir in relevant_dir_matches
    for f in (analyzed_dir / rel_dir).glob("*.csv")
]
print(
    f"Found {len(csv_files)} CSV files from relevant models: {', '.join(relevant_dirs)}"
)

# Add model info
all_dfs_with_model = []
for csv_file in sorted(csv_files):
    df = pl.read_csv(csv_file)
    model = None
    for rel_dir in relevant_dir_matches:
        if rel_dir in str(csv_file):
            model = rel_dir
            break
    df = df.with_columns(pl.lit(model).alias("model"))
    all_dfs_with_model.append(df)

combined_df: pl.DataFrame = pl.concat(all_dfs_with_model, how="diagonal_relaxed")
result_df = (
    combined_df.with_columns(
        [
            pl.col("problem_file").map_elements(extract_problem_name).alias("problem"),
            pl.col("output_tokens").cast(pl.Float64, strict=False),
        ]
    )
    .select(
        [
            pl.col("problem"),
            pl.col("output_tokens"),
            pl.col("model"),
        ]
    )
    .filter(
        (pl.col("model").is_not_null())
        & (pl.col("problem").is_not_null())
        & (pl.col("output_tokens").is_not_null())
    )
)
grouped = (
    result_df.group_by("model", "problem")
    .agg(
        pl.col("output_tokens").mean().alias("avg_output_tokens"),
        pl.col("output_tokens").count().alias("count"),
    )
    .sort("model", "problem")
)

pivot_table = grouped.pivot(
    on="model",
    values="avg_output_tokens",
    index="problem",
    aggregate_function="first",
)

# Sort problems with ring_and_peg first, then needle_sorting, then needle_transfer
pivot_table = (
    pivot_table.with_columns(
        pl.col("problem")
        .map_elements(
            lambda problem: (
                0
                if str(problem).startswith("ring_and_peg_")
                else 1
                if str(problem).startswith("needle_sorting_")
                else 2
                if str(problem).startswith("needle_transfer_")
                else 3
            )
        )
        .alias("problem_group"),
        pl.col("problem")
        .str.extract(r"_(\d+)$", 1)
        .cast(pl.Int64, strict=False)
        .alias("problem_num"),
    )
    .sort(["problem_group", "problem_num", "problem"])
    .drop(["problem_group", "problem_num"])
)

# Reorder columns from smallest to largest model
model_order = [
    "gpt_oss_20b",
    "gpt_oss_20b_optimized",
    "gpt_oss_120b",
    "gemini_3_flash",
    "gemini_3_flash_image",
]
available_columns = pivot_table.columns
ordered_cols = ["problem"] + [m for m in model_order if m in available_columns]
pivot_table = pivot_table.select(ordered_cols)

# Create heatmap
heatmap_values = np.array(
    [
        [
            np.nan if value is None else float(value)
            for value in pivot_table[col].to_list()
        ]
        for col in ordered_cols[1:]
    ],
    dtype=float,
).T

fig_width = max(8, 1.2 * len(ordered_cols[1:]))
fig_height = max(4.5, 0.55 * len(pivot_table) + 2)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
masked_values = np.ma.masked_invalid(heatmap_values)
image = ax.imshow(masked_values, aspect="auto", cmap="Reds")
x_tick_labels = ["small-ow", "small-ow GEPA", "large-ow", "frontier", "frontier image"]

ax.set_xticks(range(len(ordered_cols[1:])))
ax.set_xticklabels(x_tick_labels, rotation=20, ha="right", fontsize=12)
ax.set_yticks(range(len(pivot_table)))
ax.set_yticklabels([str(problem) for problem in pivot_table["problem"]], fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Problem", fontsize=12)

non_nan_values = heatmap_values[~np.isnan(heatmap_values)]
threshold = np.median(non_nan_values) if non_nan_values.size else 0
for row_idx in range(heatmap_values.shape[0]):
    for col_idx in range(heatmap_values.shape[1]):
        value = heatmap_values[row_idx, col_idx]
        if np.isnan(value):
            label = "-"
            text_color = "black"
        else:
            label = f"{value:.0f}"
            text_color = "white" if value >= threshold else "black"
        ax.text(
            col_idx,
            row_idx,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
        )

cbar = fig.colorbar(image, ax=ax)
cbar.set_label("Average number of output tokens", fontsize=12)
cbar.ax.tick_params(labelsize=12)
fig.tight_layout()
heatmap_path = charts_dir / "average_output_tokens_heatmap.png"
fig.savefig(heatmap_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved heatmap to {heatmap_path}")
