import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.constants import charts_dir, results_dir


def add_result_cols(df: pl.DataFrame):
    df = df.with_columns(
        (
            pl.col("error").is_in(
                ["domain_failure", "problem_failure", "model_failure"]
            )
        ).alias("no_stage"),
        (pl.col("error") == "plan_failure_translate").alias("first_stage"),
        (pl.col("error") == "plan_failure_unsolvable").alias("second_stage"),
        (pl.col("error").is_null()).alias("third_stage").alias("third_stage"),
    )
    no_stage = df["no_stage"].sum()
    first_stage = df["first_stage"].sum()
    second_stage = df["second_stage"].sum()
    third_stage = df["third_stage"].sum() - df["success"].sum()
    fourth_stage = df["success"].sum()
    return no_stage, first_stage, second_stage, third_stage, fourth_stage


def make_stacked_results_chart(
    dfs: list[pl.DataFrame],
    x_labels: list[str],
    chart_name: str,
    chart_width: int = 12,
    boundaries: list[int] | None = None,
):
    palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, 5))
    no_stage = []
    first_stage = []
    second_stage = []
    third_stage = []
    fourth_stage = []
    for df in dfs:
        ns, fs, ss, ts, fos = add_result_cols(df)
        no_stage.append(ns)
        first_stage.append(fs)
        second_stage.append(ss)
        third_stage.append(ts)
        fourth_stage.append(fos)

    legend = [
        "Last Stage",
        "Third Stage",
        "Second Stage",
        "First Stage",
        "No Stage",
    ]

    x = np.arange(len(x_labels))
    width = 0.9
    legend_offset = -0.01

    _, ax = plt.subplots(figsize=(chart_width, 6))

    no_stage_arr = np.array(no_stage)
    first_stage_arr = np.array(first_stage)
    second_stage_arr = np.array(second_stage)
    third_stage_arr = np.array(third_stage)
    fourth_stage_arr = np.array(fourth_stage)

    b0 = no_stage_arr
    b1 = b0 + first_stage_arr
    b2 = b1 + second_stage_arr
    b3 = b2 + third_stage_arr

    ax.bar(x, no_stage_arr, width=width, color=palette[4], label=legend[4])
    ax.bar(
        x, first_stage_arr, width=width, bottom=b0, color=palette[3], label=legend[3]
    )
    ax.bar(
        x, second_stage_arr, width=width, bottom=b1, color=palette[2], label=legend[2]
    )
    ax.bar(
        x, third_stage_arr, width=width, bottom=b2, color=palette[1], label=legend[1]
    )
    ax.bar(
        x, fourth_stage_arr, width=width, bottom=b3, color=palette[0], label=legend[0]
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylim(0, 31)
    ax.set_xlim(-0.5, len(x_labels) + 1.5)
    ax.set_ylabel("Pipeline Runs")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    if boundaries is not None:
        for boundary in boundaries:
            ax.axvline(
                boundary - 0.5, color="#A0A0A0", linestyle="--", linewidth=1, alpha=0.7
            )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99 + legend_offset, 0.975),
        borderaxespad=0.0,
    )

    plt.tight_layout()
    plt.savefig(charts_dir / chart_name, bbox_inches="tight")


if __name__ == "__main__":
    os_dir = results_dir / "ow comparison"
    gpt_oss_120_dir = results_dir / "gpt_oss_120b"
    gpt_oss_20_dir = results_dir / "gpt_oss_20b"
    gemini_dir = results_dir / "gemini_3_flash"
    gemini_image_dir = results_dir / "gemini_3_flash_image"
    gpt_oss_20_optimized_dir = results_dir / "gpt_oss_20b_optimized"
    ablations_dir = results_dir / "ablations"
    rigid_dir = ablations_dir / "rigid"
    tools_dir = ablations_dir / "tools"
    unsolvability_dir = ablations_dir / "unsolvability"

    gpt_oss_120B = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_gpt_oss_120b_base_2026-04-16-23_12_29_c2f67fe112bd4d09b2f3ff9ac9d13a37.csv"
    )
    gpt_oss_20B = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_gpt_oss_20b_base_2026-04-19-01:12:32_f536266ee377445c8e0515dd58fcf4a8.csv"
    )
    gemma_4_31B = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_gemma_4_31b_base_2026-04-18-18_00_35_17b8133ff0144cdead7fea14fa773d44.csv"
    )
    gemma_4_e4B = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_gemma_4_e4b_base_2026-04-18-19:12:09_32b0848b64754ce2a2c7c4be8684d1a6.csv"
    )
    qwen_35_122B_A10B = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_qwen_35_122b_base_2026-04-16-13:57:25_cbbb36bbf4b641c29a7737f876fed9d0.csv"
    )
    qwen_35_4B = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_qwen_35_4b_base_2026-04-16-20:46:10_729ceb28cce349389057b7e20695f4b5.csv"
    )
    dfs = [
        gpt_oss_120B,
        gpt_oss_20B,
        gemma_4_31B,
        gemma_4_e4B,
        qwen_35_122B_A10B,
        qwen_35_4B,
    ]
    x_labels = [
        "gpt-oss-120B",
        "gpt-oss-20B",
        "Gemma 4 31B",
        "Gemma 4 e4B",
        "Qwen 3.5 122B A10B",
        "Qwen 3.5 4B",
    ]
    make_stacked_results_chart(dfs, x_labels, "os_bars.pdf", 8, boundaries=[2, 4])
    print("gpt oss 120b applicable:", add_result_cols(gpt_oss_120B)[-1])
    print("gpt oss 20b solvable:", add_result_cols(gpt_oss_20B)[-2])
    print("gemma 4 31b applicable:", add_result_cols(gemma_4_31B)[-1])
    print("gemma 4 4b solvable:", add_result_cols(gemma_4_e4B)[-2])

    ring_and_peg_1_120 = pl.read_csv(
        gpt_oss_120_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_curated_gpt_oss_120b_base_2026-04-19-22:34:25_62cc6a8aa30b43e5850a3b0213e322e6.csv"
    )
    ring_and_peg_2_120 = pl.read_csv(
        gpt_oss_120_dir
        / "ring_and_peg_ring_and_peg_2_tool_call_curated_gpt_oss_120b_base_2026-04-20-00:12:04_8af05589a3aa42ea98f0ac4cad871d0b.csv"
    )
    ring_and_peg_3_120 = pl.read_csv(
        gpt_oss_120_dir
        / "ring_and_peg_ring_and_peg_3_tool_call_curated_gpt_oss_120b_base_2026-04-20-02:12:38_888583ac09cf40e098282c1d79f204f2.csv"
    )
    ring_and_peg_4_120 = pl.read_csv(
        gpt_oss_120_dir
        / "ring_and_peg_ring_and_peg_4_tool_call_curated_gpt_oss_120b_base_2026-04-20-03:57:16_88c10e6a94ed49fd80fedb8b7b79efbd.csv"
    )
    ring_and_peg_5_120 = pl.read_csv(
        gpt_oss_120_dir
        / "ring_and_peg_ring_and_peg_5_tool_call_curated_gpt_oss_120b_base_2026-04-23-01:16:55_0a0b9960bbaf4c739fa66347702c9552.csv"
    )
    dfs = [
        ring_and_peg_1_120,
        ring_and_peg_2_120,
        ring_and_peg_3_120,
        ring_and_peg_4_120,
        ring_and_peg_5_120,
    ]
    x_labels = [
        "ring_and_peg_1",
        "ring_and_peg_2",
        "ring_and_peg_3",
        "ring_and_peg_4",
        "ring_and_peg_5",
    ]
    make_stacked_results_chart(dfs, x_labels, "gpt_oss_120b_ring_and_peg.pdf")

    ring_and_peg_1_20 = pl.read_csv(
        gpt_oss_20_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_curated_gpt_oss_20b_base_2026-04-22-21:10:06_036982d4173440498c591e6f36697226.csv"
    )
    ring_and_peg_2_20 = pl.read_csv(
        gpt_oss_20_dir
        / "ring_and_peg_ring_and_peg_2_tool_call_curated_gpt_oss_20b_base_2026-04-23-01:48:09_af42ef2edbca42baab6b7e211a980582.csv"
    )
    ring_and_peg_3_20 = pl.read_csv(
        gpt_oss_20_dir
        / "ring_and_peg_ring_and_peg_3_tool_call_curated_gpt_oss_20b_base_2026-04-23-16:03:16_f199c0f978fd4006941679004a94efed.csv"
    )
    ring_and_peg_4_20 = pl.read_csv(
        gpt_oss_20_dir
        / "ring_and_peg_ring_and_peg_4_tool_call_curated_gpt_oss_20b_base_2026-04-23-17:43:43_1fa135c7f0b84ec281a0a9e070569c39.csv"
    )
    ring_and_peg_5_20 = pl.read_csv(
        gpt_oss_20_dir
        / "ring_and_peg_ring_and_peg_5_tool_call_curated_gpt_oss_20b_base_2026-04-23-21:13:44_ddca63e1472247e684d91842ef8c85e8.csv"
    )
    dfs = [
        ring_and_peg_1_20,
        ring_and_peg_2_20,
        ring_and_peg_3_20,
        ring_and_peg_4_20,
        ring_and_peg_5_20,
    ]
    make_stacked_results_chart(dfs, x_labels, "gpt_oss_20b_ring_and_peg.pdf")

    ring_and_peg_1_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-04-28-04:31:45_9ffbfcdd390945538a92644972cb8296.csv"
    )
    ring_and_peg_2_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "ring_and_peg_ring_and_peg_2_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-04-28-17:42:31_e6a6621c87ae43cd8433e5ce89b8ebfe.csv"
    )
    ring_and_peg_3_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "ring_and_peg_ring_and_peg_3_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-04-29-00:09:30_44f4ed300098480289a7716aaaef3d31.csv"
    )
    ring_and_peg_4_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "ring_and_peg_ring_and_peg_4_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-04-29-06:18:15_0261f273a72f483a86d97909aaa47721.csv"
    )
    ring_and_peg_5_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "ring_and_peg_ring_and_peg_5_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-04-29-13:44:43_251bbcb697494a1cbb28089f297745fd.csv"
    )
    dfs = [
        ring_and_peg_1_20_optimized,
        ring_and_peg_2_20_optimized,
        ring_and_peg_3_20_optimized,
        ring_and_peg_4_20_optimized,
        ring_and_peg_5_20_optimized,
    ]
    make_stacked_results_chart(dfs, x_labels, "gpt_oss_20b_ring_and_peg_optimized.pdf")

    needle_transfer_1_120 = pl.read_csv(
        gpt_oss_120_dir
        / "needle_transfer_needle_transfer_1_tool_call_curated_gpt_oss_120b_base_2026-04-23-05:01:49_80fc0281d00d4110beb4c8e19263b830.csv"
    )
    needle_transfer_2_120 = pl.read_csv(
        unsolvability_dir
        / "needle_transfer_needle_transfer_2_tool_call_curated_gpt_oss_120b_base_2026-04-19-15:24:36_3eb7690810594b0ebfbe10cbb75da698.csv"
    )
    dfs = [
        needle_transfer_1_120,
        needle_transfer_2_120,
    ]
    x_labels = ["needle_transfer_1", "needle_transfer_2"]
    make_stacked_results_chart(dfs, x_labels, "gpt_oss_120b_needle_transfer.pdf")

    needle_transfer_1_20 = pl.read_csv(
        gpt_oss_20_dir
        / "needle_transfer_needle_transfer_1_tool_call_curated_gpt_oss_20b_base_2026-04-24-01:28:10_06b425af59264edc80c47471086e6870.csv"
    )
    needle_transfer_2_20 = pl.read_csv(
        gpt_oss_20_dir
        / "needle_transfer_needle_transfer_2_tool_call_curated_gpt_oss_20b_base_2026-04-24-06:22:09_cc8165bfb2534cd28edcef9955114523.csv"
    )
    dfs = [needle_transfer_1_20, needle_transfer_2_20]
    make_stacked_results_chart(dfs, x_labels, "gpt_oss_20b_needle_transfer.pdf")

    needle_transfer_1_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "needle_transfer_needle_transfer_1_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-05-01-16:41:43_b4f6e537ce2149a3a9b715bfa49ff827.csv"
    )
    needle_transfer_2_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "needle_transfer_needle_transfer_2_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-05-02-05:35:45_f7922a0fb48849009c916e1ca2eb593a.csv"
    )
    dfs = [needle_transfer_1_20_optimized, needle_transfer_2_20_optimized]
    make_stacked_results_chart(
        dfs, x_labels, "gpt_oss_20b_needle_transfer_optimized.pdf"
    )

    needle_sorting_1_120 = pl.read_csv(
        gpt_oss_120_dir
        / "needle_sorting_needle_sorting_1_tool_call_curated_gpt_oss_120b_base_2026-04-23-06:06:28_3b2ca42a62bd45b081fdc293e3d954fc.csv"
    )
    needle_sorting_2_120 = pl.read_csv(
        gpt_oss_120_dir
        / "needle_sorting_needle_sorting_2_tool_call_curated_gpt_oss_120b_base_2026-04-23-08:08:46_8fee902e01c940dab47af774fe9beab1.csv"
    )
    needle_sorting_3_120 = pl.read_csv(
        gpt_oss_120_dir
        / "needle_sorting_needle_sorting_3_tool_call_curated_gpt_oss_120b_base_2026-04-23-10:10:28_b91548a1b78644ddb227eb7b6b28db1c.csv"
    )
    dfs = [needle_sorting_1_120, needle_sorting_2_120, needle_sorting_3_120]
    x_labels = ["needle_sorting_1", "needle_sorting_2", "needle_sorting_3"]
    make_stacked_results_chart(dfs, x_labels, "gpt_oss_120b_needle_sorting.pdf")

    needle_sorting_1_20 = pl.read_csv(
        gpt_oss_20_dir
        / "needle_sorting_needle_sorting_1_tool_call_curated_gpt_oss_20b_base_2026-04-24-09:11:31_b1510cc212d745be81ea74866afc7b19.csv"
    )
    needle_sorting_2_20 = pl.read_csv(
        gpt_oss_20_dir
        / "needle_sorting_needle_sorting_2_tool_call_curated_gpt_oss_20b_base_2026-04-24-12:35:01_d33eb716a44e41e4b752f622946e3ea9.csv"
    )
    needle_sorting_3_20 = pl.read_csv(
        gpt_oss_20_dir
        / "needle_sorting_needle_sorting_3_tool_call_curated_gpt_oss_20b_base_2026-04-24-16:02:20_3d94e977200c4c60ae3d202b14368cfc.csv"
    )
    dfs = [needle_sorting_1_20, needle_sorting_2_20, needle_sorting_3_20]
    make_stacked_results_chart(dfs, x_labels, "gpt_oss_20b_needle_sorting.pdf")

    needle_sorting_1_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "needle_sorting_needle_sorting_1_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-04-29-19:22:08_94340ea440ea40658df91595169226f7.csv"
    )
    needle_sorting_2_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "needle_sorting_needle_sorting_2_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-04-30-04:31:00_a7894d2efa8c41649157fb2a7c29e9c6.csv"
    )
    needle_sorting_3_20_optimized = pl.read_csv(
        gpt_oss_20_optimized_dir
        / "needle_sorting_needle_sorting_3_tool_call_curated_gpt_oss_20b_optimized_tool_call_curated_gpt_oss_20b.json_2026-05-01-06:09:36_da58d699fd7c44d6aa5333d424820892.csv"
    )
    dfs = [
        needle_sorting_1_20_optimized,
        needle_sorting_2_20_optimized,
        needle_sorting_3_20_optimized,
    ]
    make_stacked_results_chart(
        dfs, x_labels, "gpt_oss_20b_needle_sorting_optimized.pdf"
    )

    combined_all_120_dfs = [
        ring_and_peg_1_120,
        ring_and_peg_2_120,
        ring_and_peg_3_120,
        ring_and_peg_4_120,
        ring_and_peg_5_120,
        needle_transfer_1_120,
        needle_transfer_2_120,
        needle_sorting_1_120,
        needle_sorting_2_120,
        needle_sorting_3_120,
    ]
    combined_all_120_labels = [
        "ring_and_peg_1",
        "ring_and_peg_2",
        "ring_and_peg_3",
        "ring_and_peg_4",
        "ring_and_peg_5",
        "needle_transfer_1",
        "needle_transfer_2",
        "needle_sorting_1",
        "needle_sorting_2",
        "needle_sorting_3",
    ]
    make_stacked_results_chart(
        combined_all_120_dfs,
        combined_all_120_labels,
        "gpt_oss_120b_all_problems.pdf",
        boundaries=[5, 7],
    )
    sum = 0
    for df in combined_all_120_dfs:
        sum += add_result_cols(df)[-1]
    print("gpt-oss-120b sum:", sum)

    combined_all_20_dfs = [
        ring_and_peg_1_20,
        ring_and_peg_2_20,
        ring_and_peg_3_20,
        ring_and_peg_4_20,
        ring_and_peg_5_20,
        needle_transfer_1_20,
        needle_transfer_2_20,
        needle_sorting_1_20,
        needle_sorting_2_20,
        needle_sorting_3_20,
    ]
    combined_all_20_labels = [
        "ring_and_peg_1",
        "ring_and_peg_2",
        "ring_and_peg_3",
        "ring_and_peg_4",
        "ring_and_peg_5",
        "needle_transfer_1",
        "needle_transfer_2",
        "needle_sorting_1",
        "needle_sorting_2",
        "needle_sorting_3",
    ]
    make_stacked_results_chart(
        combined_all_20_dfs,
        combined_all_20_labels,
        "gpt_oss_20b_all_problems.pdf",
        boundaries=[5, 7],
    )
    sum = 0
    for df in combined_all_20_dfs:
        sum += add_result_cols(df)[-1]
    print("gpt-oss-20b sum:", sum)

    combined_all_20_optimized_dfs = [
        ring_and_peg_1_20_optimized,
        ring_and_peg_2_20_optimized,
        ring_and_peg_3_20_optimized,
        ring_and_peg_4_20_optimized,
        ring_and_peg_5_20_optimized,
        needle_transfer_1_20_optimized,
        needle_transfer_2_20_optimized,
        needle_sorting_1_20_optimized,
        needle_sorting_2_20_optimized,
        needle_sorting_3_20_optimized,
    ]
    combined_all_20_optimized_labels = [
        "ring_and_peg_1",
        "ring_and_peg_2",
        "ring_and_peg_3",
        "ring_and_peg_4",
        "ring_and_peg_5",
        "needle_transfer_1",
        "needle_transfer_2",
        "needle_sorting_1",
        "needle_sorting_2",
        "needle_sorting_3",
    ]
    make_stacked_results_chart(
        combined_all_20_optimized_dfs,
        combined_all_20_optimized_labels,
        "gpt_oss_20b_all_problems_optimized.pdf",
        boundaries=[5, 7],
    )
    sum = 0
    for df in combined_all_20_optimized_dfs:
        sum += add_result_cols(df)[-1]
    print("gpt-oss-20b optimized sum:", sum)

    ring_and_peg_1_120 = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_gpt_oss_120b_base_2026-04-16-23_12_29_c2f67fe112bd4d09b2f3ff9ac9d13a37.csv"
    )
    ring_and_peg_1_120_no_plan_feedback = pl.read_csv(
        tools_dir
        / "feedback_ablated_ring_and_peg_ring_and_peg_1_tool_call_gpt_oss_120b_base_2026-04-18-20:59:38_e9faad4ec5be4e6a949a5b8b12715519.csv"
    )
    ring_and_peg_1_120_no_feedback = pl.read_csv(
        tools_dir
        / "full_ablation_ring_and_peg_ring_and_peg_1_tool_call_gpt_oss_120b_base_2026-04-19-20:51:30_722b9107b48240eb89f7681ab7efe265.csv"
    )
    dfs = [
        ring_and_peg_1_120,
        ring_and_peg_1_120_no_plan_feedback,
        ring_and_peg_1_120_no_feedback,
    ]
    x_labels = [
        "base",
        "no_plan_feedback",
        "no_feedback",
    ]
    make_stacked_results_chart(dfs, x_labels, "tool_ablations.pdf", 5)
    print("# Tool ablations")
    print("Applicable base", add_result_cols(ring_and_peg_1_120)[-1])
    print(
        "Applicable no_plan", add_result_cols(ring_and_peg_1_120_no_plan_feedback)[-2]
    )
    print(
        "Not reached 3rd stage no_feedback",
        add_result_cols(ring_and_peg_1_120_no_feedback)[-2],
    )

    solvability_base = pl.read_csv(
        unsolvability_dir
        / "needle_transfer_needle_transfer_2_tool_call_gpt_oss_120b_base_2026-04-19-06:04:02_a41a21d570d54008b98bd4bf31823226.csv"
    )
    solvability_curated = pl.read_csv(
        unsolvability_dir
        / "needle_transfer_needle_transfer_2_tool_call_curated_gpt_oss_120b_base_2026-04-19-15:24:36_3eb7690810594b0ebfbe10cbb75da698.csv"
    )
    solvability_full = pl.read_csv(
        unsolvability_dir
        / "needle_transfer_needle_transfer_2_tool_call_full_gpt_oss_120b_base_2026-04-19-10:50:49_dd35e506c4544d6494bece47dfe6d770.csv"
    )
    solvability_abstraction = pl.read_csv(
        unsolvability_dir
        / "needle_transfer_needle_transfer_2_tool_call_abstraction_gpt_oss_120b_base_2026-04-24-22:28:35_289c2c2f4bd14cae975103288acd2e2e.csv"
    )
    dfs = [
        solvability_base,
        solvability_curated,
        solvability_full,
        solvability_abstraction,
    ]
    x_labels = [
        "base",
        "solvability_curated",
        "solvability_full",
        "solvability_abstraction",
    ]
    make_stacked_results_chart(dfs, x_labels, "solvability_ablations.pdf", 6)
    print("# Feedback ablations")
    print("Third stage base", add_result_cols(solvability_base)[-2])
    print("Second stage base", add_result_cols(solvability_base)[-3])
    print("Third stage curated", add_result_cols(solvability_curated)[-2])
    print("Second stage curated", add_result_cols(solvability_curated)[-3])

    tool_call = pl.read_csv(
        os_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_gpt_oss_120b_base_2026-04-16-23_12_29_c2f67fe112bd4d09b2f3ff9ac9d13a37.csv"
    )
    rigid_trajectory = pl.read_csv(
        rigid_dir
        / "ring_and_peg_ring_and_peg_1_rigid_trajectory_gpt_oss_120b_base_2026-04-23-11:02:27_ab9d2093d9e549e18a3783f34a3d58d9.csv"
    )
    dfs = [
        tool_call,
        rigid_trajectory,
    ]
    x_labels = [
        "tool_call",
        "rigid_trajectory",
    ]
    make_stacked_results_chart(dfs, x_labels, "tool_call_rigid_comparison.pdf", 4)
    print("# Pipelines")
    print("Num applicable rigid_trajectory:", add_result_cols(rigid_trajectory)[-1])
    print("Num applicable tool_call:", add_result_cols(tool_call)[-1])

    ring_and_peg_1_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_curated_gemini_3_flash_base_2026-05-03-22:54:52_60c43d610491471f8d275b6f988794b4.csv"
    )
    ring_and_peg_2_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "ring_and_peg_ring_and_peg_2_tool_call_curated_gemini_3_flash_base_2026-05-03-22:58:13_1913a24208474cd394e5caa26b5ecf31.csv"
    )
    ring_and_peg_3_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "ring_and_peg_ring_and_peg_3_tool_call_curated_gemini_3_flash_base_2026-05-03-23:03:12_52225b9ab6d949d7b59b41dcdc8ad970.csv"
    )
    ring_and_peg_4_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "ring_and_peg_ring_and_peg_4_tool_call_curated_gemini_3_flash_base_2026-05-03-22:56:30_ad79d3f3c0eb44e28a6ae933319595ff.csv"
    )
    ring_and_peg_5_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "ring_and_peg_ring_and_peg_5_tool_call_curated_gemini_3_flash_base_2026-05-03-23:05:35_06feeea87163448599a1f806272569be.csv"
    )
    needle_transfer_1_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "needle_transfer_needle_transfer_1_tool_call_curated_gemini_3_flash_base_2026-05-03-23:49:36_9db94be972474521904f45fbc91981c1.csv"
    )
    needle_transfer_2_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "needle_transfer_needle_transfer_2_tool_call_curated_gemini_3_flash_base_2026-05-03-23:49:44_a4b466070c594d09aa0cb86460e447e5.csv"
    )
    needle_sorting_1_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "needle_sorting_needle_sorting_1_tool_call_curated_gemini_3_flash_base_2026-05-03-22:46:16_cc38032406714b6288a00dd87bba6a33.csv"
    )
    needle_sorting_2_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "needle_sorting_needle_sorting_2_tool_call_curated_gemini_3_flash_base_2026-05-03-22:49:37_d2a2ab39d8354b73ba7dc4bc765f4aff.csv"
    )
    needle_sorting_3_gemini_3_flash = pl.read_csv(
        gemini_dir
        / "needle_sorting_needle_sorting_3_tool_call_curated_gemini_3_flash_base_2026-05-03-22:49:06_34bf3c5f83954ba48bc3235e5ca74209.csv"
    )
    dfs = [
        ring_and_peg_1_gemini_3_flash,
        ring_and_peg_2_gemini_3_flash,
        ring_and_peg_3_gemini_3_flash,
        ring_and_peg_4_gemini_3_flash,
        ring_and_peg_5_gemini_3_flash,
        needle_transfer_1_gemini_3_flash,
        needle_transfer_2_gemini_3_flash,
        needle_sorting_1_gemini_3_flash,
        needle_sorting_2_gemini_3_flash,
        needle_sorting_3_gemini_3_flash,
    ]
    x_labels = [
        "ring_and_peg_1",
        "ring_and_peg_2",
        "ring_and_peg_3",
        "ring_and_peg_4",
        "ring_and_peg_5",
        "needle_transfer_1",
        "needle_transfer_2",
        "needle_sorting_1",
        "needle_sorting_2",
        "needle_sorting_3",
    ]
    make_stacked_results_chart(
        dfs,
        x_labels,
        "gemini_3_flash_all_problems.pdf",
        boundaries=[5, 7],
    )
    sum = 0
    for df in dfs:
        sum += add_result_cols(df)[-1]
    print("gemini-3-flash sum:", sum)
    print(
        "gemini-3-flash needle_transfer_1",
        add_result_cols(needle_transfer_1_gemini_3_flash)[-1],
    )

    ring_and_peg_1_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "ring_and_peg_ring_and_peg_1_tool_call_image_gemini_3_flash_base_2026-05-11-23:34:44_683564a53f9b4800bef7e7798a2c85bf.csv"
    )
    ring_and_peg_2_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "ring_and_peg_ring_and_peg_2_tool_call_image_gemini_3_flash_base_2026-05-11-23:34:05_52d30666adbc4a9896c66731c4d32e86.csv"
    )
    ring_and_peg_3_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "ring_and_peg_ring_and_peg_3_tool_call_image_gemini_3_flash_base_2026-05-11-23:33:25_3532c464b5c14bf2ba2e2fd151c58980.csv"
    )
    ring_and_peg_4_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "ring_and_peg_ring_and_peg_4_tool_call_image_gemini_3_flash_base_2026-05-11-23:29:11_05d0a39cb8a243ddbf1e65019ae44f67.csv"
    )
    ring_and_peg_5_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "ring_and_peg_ring_and_peg_5_tool_call_image_gemini_3_flash_base_2026-05-11-23:39:19_c4dd80168a5b4cd3ab4e1fcc42691b7a.csv"
    )
    needle_transfer_1_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "needle_transfer_needle_transfer_1_tool_call_image_gemini_3_flash_base_2026-05-12-00:26:38_319e715423e54742ada96120d07a8183.csv"
    )
    needle_transfer_2_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "needle_transfer_needle_transfer_2_tool_call_image_gemini_3_flash_base_2026-05-12-00:20:55_c21ab12e62624b0983159df3fb189478.csv"
    )
    needle_sorting_1_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "needle_sorting_needle_sorting_1_tool_call_image_gemini_3_flash_base_2026-05-11-23:17:20_b9d97105f923490e96a5ef6dc72383ef.csv"
    )
    needle_sorting_2_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "needle_sorting_needle_sorting_2_tool_call_image_gemini_3_flash_base_2026-05-11-23:26:56_2ed06b6277fc416f928fc97d78e7d935.csv"
    )
    needle_sorting_3_gemini_3_flash_image = pl.read_csv(
        gemini_image_dir
        / "needle_sorting_needle_sorting_3_tool_call_image_gemini_3_flash_base_2026-05-11-23:24:27_17e8253c2beb47d38e25146a4125e4e0.csv"
    )
    dfs = [
        ring_and_peg_1_gemini_3_flash_image,
        ring_and_peg_2_gemini_3_flash_image,
        ring_and_peg_3_gemini_3_flash_image,
        ring_and_peg_4_gemini_3_flash_image,
        ring_and_peg_5_gemini_3_flash_image,
        needle_transfer_1_gemini_3_flash_image,
        needle_transfer_2_gemini_3_flash_image,
        needle_sorting_1_gemini_3_flash_image,
        needle_sorting_2_gemini_3_flash_image,
        needle_sorting_3_gemini_3_flash_image,
    ]
    x_labels = [
        "ring_and_peg_1",
        "ring_and_peg_2",
        "ring_and_peg_3",
        "ring_and_peg_4",
        "ring_and_peg_5",
        "needle_transfer_1",
        "needle_transfer_2",
        "needle_sorting_1",
        "needle_sorting_2",
        "needle_sorting_3",
    ]
    make_stacked_results_chart(
        dfs,
        x_labels,
        "gemini_3_flash_image_all_problems.pdf",
        boundaries=[5, 7],
    )
    sum = 0
    for df in dfs:
        sum += add_result_cols(df)[-1]
    print("gemini-3-flash image sum:", sum)
    print(
        "gemini-3-flash image needle_transfer_1",
        add_result_cols(needle_transfer_1_gemini_3_flash_image)[-1],
    )
