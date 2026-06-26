import matplotlib.pyplot as plt
import numpy as np

from src.constants import charts_dir

palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, 5))

if __name__ == "__main__":
    plt.figure(figsize=(6, 5))

    x_values = np.array([25.0, 3.0, 1.33, 25.0, 14.0])
    y_values = np.array([0.56, 0.7, 0.03, 0.87, 0.87])
    labels = [
        "Claude 4.5 Opus",
        "Gemini 3 Flash",
        "MiniMax M2.5",
        "Claude Opus 4.6",
        "GPT-5.2 Codex",
    ]

    for x, y, label, color in zip(x_values, y_values, labels, palette, strict=True):
        plt.scatter(x, y, color=color, s=60)

        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            va="bottom",
            ha="center",
            fontsize=11,
        )

    plt.xlabel("$/100k tokens", fontsize=11)
    plt.ylabel("Success rate for ring_and_peg_1", fontsize=11)
    plt.ylim(0, 1.01)
    plt.xlim(-2, 29)
    plt.gca().set_axisbelow(True)
    plt.grid(True, which="major", axis="both", linestyle="-", linewidth=0.6, alpha=0.25)

    plt.tight_layout()
    plt.savefig(charts_dir / "large_cs_scatter.png", bbox_inches="tight")
