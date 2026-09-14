#!/usr/bin/env python
"""Generate vector-first manuscript figures for ScaleFuse-LoRA."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"


COLORS = {
    "frozen": "#D9E1E8",
    "frozen_edge": "#526777",
    "lora": "#0072B2",
    "scale": "#D55E00",
    "text": "#222222",
    "positive": "#009E73",
    "light": "#F7F9FA",
}


def box(axis, xy, width, height, text, face, edge, fontsize=10,
        linewidth=1.4, radius=0.06, text_color=None):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size={}".format(radius),
        facecolor=face, edgecolor=edge, linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=text_color or COLORS["text"], linespacing=1.2,
    )
    return patch


def arrow(axis, start, end, color="#444444", linewidth=1.5,
          style="-|>", mutation=12, connection="arc3"):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=linewidth, color=color, connectionstyle=connection,
        shrinkA=3, shrinkB=3,
    )
    axis.add_patch(patch)
    return patch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(14.2, 6.1))
    axis.set_xlim(0, 14.2)
    axis.set_ylim(0, 6.1)
    axis.axis("off")

    # Vector input sketch.
    box(axis, (0.25, 2.45), 1.65, 2.15, "", "#EAF2F8", "#6B7C87")
    axis.add_patch(Polygon(
        [(0.3, 2.5), (1.85, 2.5), (1.55, 4.0), (0.62, 4.0)],
        closed=True, facecolor="#6F767B", edgecolor="none"))
    axis.plot([1.08, 1.1], [2.58, 3.87], color="white", linewidth=2.1,
              linestyle=(0, (5, 5)), alpha=0.8)
    axis.add_patch(Ellipse((1.28, 3.12), 0.48, 0.23, angle=-12,
                           facecolor="#2B2B2B", edgecolor="#F0E442",
                           linewidth=1.4))
    axis.text(1.075, 4.32, "Road image", ha="center", fontsize=10.5,
              fontweight="bold")
    box(axis, (0.25, 1.40), 1.65, 0.55, 'Text: "pothole"',
        "#FFF4D6", "#A97B00", fontsize=9.5)

    # Frozen backbone.
    box(axis, (2.45, 2.30), 1.75, 2.45,
        "Grounding DINO\nSwin-T backbone\n+ channel mapper",
        COLORS["frozen"], COLORS["frozen_edge"], fontsize=10.5)
    axis.text(3.325, 1.96, "frozen pretrained weights", ha="center",
              fontsize=8.8, color=COLORS["frozen_edge"])
    arrow(axis, (1.93, 3.5), (2.43, 3.5))
    arrow(axis, (1.93, 1.68), (4.45, 1.68), connection="arc3,rad=-0.10")

    # Feature hierarchy and top-down residuals.
    axis.text(7.25, 5.72, "ScaleFuse: zero-initialized coarse-to-fine residual path",
              ha="center", fontsize=11.5, fontweight="bold", color=COLORS["scale"])
    feature_specs = [
        (4.70, 3.72, 1.20, 1.05, "F1\n(fine)"),
        (6.05, 3.50, 1.08, 0.90, "F2"),
        (7.27, 3.31, 0.96, 0.77, "F3"),
        (8.36, 3.15, 0.84, 0.65, "F4\n(coarse)"),
    ]
    for x, y, width, height, label in feature_specs:
        box(axis, (x, y), width, height, label, "#F4F6F7",
            COLORS["frozen_edge"], fontsize=9.2, radius=0.03)
    arrow(axis, (4.20, 3.52), (4.68, 4.12))

    # Top-down ScaleFuse edges, drawn above the pyramid.
    centers = [
        (x + width / 2, y + height)
        for x, y, width, height, _ in feature_specs
    ]
    for coarse_index in (3, 2, 1):
        coarse = centers[coarse_index]
        fine = centers[coarse_index - 1]
        high = 4.82 + 0.13 * (3 - coarse_index)
        arrow(axis, coarse, (coarse[0], high), color=COLORS["scale"], linewidth=2.0)
        arrow(axis, (coarse[0], high), (fine[0], high),
              color=COLORS["scale"], linewidth=2.0)
        arrow(axis, (fine[0], high), fine, color=COLORS["scale"], linewidth=2.0)
        axis.text((coarse[0] + fine[0]) / 2, high + 0.12,
                  "GN + upsample + 1x1 + gate",
                  ha="center", va="bottom", fontsize=7.4,
                  color=COLORS["scale"])
    axis.text(6.72, 2.87,
              r"$\tilde{F}_{l}=F_l+g_l\,P_l(\mathrm{Up}(\mathrm{GN}(\tilde{F}_{l+1})))$",
              ha="center", fontsize=10.2, color=COLORS["scale"])
    axis.text(6.72, 2.52, "P is initialized to zero: exact pretrained behavior at step 0",
              ha="center", fontsize=8.6, color="#7A3B13")

    # Transformer and LoRA branches.
    box(axis, (9.82, 2.45), 2.10, 2.25,
        "Grounding DINO\nmultimodal encoder\n+ decoder + head",
        COLORS["frozen"], COLORS["frozen_edge"], fontsize=10.2)
    arrow(axis, (9.22, 3.50), (9.80, 3.50))
    arrow(axis, (4.45, 1.68), (10.25, 2.42), connection="arc3,rad=-0.08")
    box(axis, (10.05, 1.25), 1.65, 0.60, "rank-16 LoRA updates",
        "#DDEFFC", COLORS["lora"], fontsize=9.0, text_color=COLORS["lora"])
    arrow(axis, (10.88, 1.87), (10.88, 2.43), color=COLORS["lora"], linewidth=1.8)

    # Output.
    box(axis, (12.55, 2.70), 1.35, 1.55, "Text-conditioned\nbox predictions",
        "#E2F4EC", COLORS["positive"], fontsize=9.5)
    arrow(axis, (11.94, 3.50), (12.53, 3.50))

    # Legend and headline statistics.
    axis.text(0.25, 5.63, "ScaleFuse-LoRA", fontsize=16, fontweight="bold",
              color=COLORS["text"])
    axis.text(0.25, 5.30,
              "Parameter-efficient cross-scale adaptation of Grounding DINO",
              fontsize=10.5, color="#4B4B4B")
    box(axis, (11.98, 0.65), 1.92, 1.00,
        "1.551% trainable\n+1.983 test AP\n(vs. Fusion-LoRA)",
        "#FFF1E8", COLORS["scale"], fontsize=9.2)
    axis.add_patch(FancyBboxPatch(
        (0.25, 0.52), 0.34, 0.24, boxstyle="round,pad=0.01",
        facecolor=COLORS["frozen"], edgecolor=COLORS["frozen_edge"]))
    axis.text(0.66, 0.64, "frozen base", va="center", fontsize=8.5)
    axis.plot([1.65, 1.99], [0.64, 0.64], color=COLORS["lora"], linewidth=3)
    axis.text(2.08, 0.64, "LoRA trainable", va="center", fontsize=8.5)
    axis.plot([3.35, 3.69], [0.64, 0.64], color=COLORS["scale"], linewidth=3)
    axis.text(3.78, 0.64, "ScaleFuse trainable", va="center", fontsize=8.5)

    fig.tight_layout(pad=0.3)
    stem = OUT / "scalefuse_lora_overview"
    for extension, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(str(stem) + "." + extension, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print("figure_prefix={}".format(stem))


if __name__ == "__main__":
    main()
