"""Draw the README figures from saved reports (no dataset or model needed).

Always: split overlap and class balance, from reports/data_audit.json.
After training: per-class AP for both splits, if the Colab notebook's
reports/results/grouped_test.json and official_test.json exist.
Each figure is written in a light and a dark version for GitHub's two themes.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from data_utils import CLASS_NAMES  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926"]},
}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Inter", "Segoe UI", "Arial", "DejaVu Sans"]})


def bar(ax, y, length, thickness_px, color):
    """Horizontal bar, square at the baseline with a 4 px rounded end."""
    if length <= 0:
        return
    origin = ax.transData.transform((0, 0))
    px_x = ax.transData.transform((1, 0))[0] - origin[0]
    px_y = abs(ax.transData.transform((0, 1))[1] - origin[1])
    height = thickness_px / px_y
    ax.add_patch(FancyBboxPatch((0, y - height / 2), length, height, mutation_aspect=px_x / px_y,
                                boxstyle=f"round,pad=0,rounding_size={min(4 / px_x, length / 2)}",
                                facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((0, y - height / 2), length / 2, height, facecolor=color, edgecolor="none"))


def figure(theme, title, subtitle, rows, top_px):
    """Figure with a left-aligned title block and one axes for `rows` bar slots."""
    t = THEMES[theme]
    height_px = top_px + 40 * rows + 30
    fig = plt.figure(figsize=(8, height_px / 100), dpi=100, facecolor=t["surface"])
    fig.text(0.02, 1 - 18 / height_px, title, color=t["ink"], fontsize=13, weight="semibold", va="top")
    fig.text(0.02, 1 - 44 / height_px, subtitle, color=t["secondary"], fontsize=10, va="top")
    ax = fig.add_axes([0.24, 30 / height_px, 0.66, 40 * rows / height_px], facecolor=t["surface"])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(t["axis"])  # the baseline every bar grows from
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.set_ylim(rows - 0.5, -0.5)  # first row at the top
    return fig, ax, t, height_px


def save(fig, out, name, theme):
    path = Path(out) / (name + ("_dark" if theme == "dark" else "") + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def ranked_bars(out, name, title, subtitle, items):
    """One series, every bar labeled at its tip, so no x-axis is needed."""
    paths = []
    for theme in THEMES:
        fig, ax, t, _ = figure(theme, title, subtitle, len(items), top_px=76)
        ax.set_xlim(0, max(v for _, v in items) * 1.12)
        ax.set_xticks([])
        ax.set_yticks(range(len(items)), [label for label, _ in items], color=t["secondary"], fontsize=10)
        for i, (_, value) in enumerate(items):
            bar(ax, i, value, 20, t["series"][0])
            ax.text(value + ax.get_xlim()[1] * 0.01, i, f"{value:,}", va="center", color=t["ink"], fontsize=10)
        paths.append(save(fig, out, name, theme))
    return paths


def audit_figures(audit, out):
    overlaps = list(audit["cross_split_source_name_overlaps"].values())
    combos = {"train + val + test": ["test", "train", "val"], "train + val only": ["train", "val"],
              "train + test only": ["test", "train"], "val + test only": ["test", "val"]}
    items = [(label, overlaps.count(splits)) for label, splits in combos.items()]
    items = [(label, n) for label, n in items if n]
    both = sum(n for label, n in items if "train" in label and "test" in label)
    paths = ranked_bars(out, "split_overlap",
                        f"{len(overlaps)} source images appear in more than one official split",
                        f"{both} of them reach both train and test, so test scores can be inflated",
                        items)

    totals = {name: sum(s["instances"][name] for s in audit["splits"].values()) for name in CLASS_NAMES}
    items = sorted(totals.items(), key=lambda kv: -kv[1])
    share = items[0][1] / sum(totals.values())
    paths += ranked_bars(out, "class_balance",
                         f"{items[0][0]} is {share:.0%} of all {sum(totals.values()):,} labeled marks",
                         f"{items[-1][0]} has {items[0][1] / items[-1][1]:.0f}x fewer examples than {items[0][0]}",
                         items)
    return paths


def results_figure(grouped, official, out):
    """Per-class AP50 on each split's own test set: grouped (honest) vs official (leaky)."""
    runs = [("Grouped split (leakage-reduced)", grouped), ("Official RF100 split", official)]
    ap = [{row["class"]: row["map50"] for row in run["per_class"]} for _, run in runs]
    paths = []
    for theme in THEMES:
        fig, ax, t, h = figure(theme, "Per-class mAP50 on each split's own test set",
                            "AP at IoU 0.5. The test sets differ, so a gap is a warning sign, not a measurement",
                            len(CLASS_NAMES), top_px=100)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="x", color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, color=t["secondary"], fontsize=10)
        for i, name in enumerate(CLASS_NAMES):
            for j, scores in enumerate(ap):
                if name in scores:  # 12 px bars with a 2 px gap between the pair
                    bar(ax, i + (j - 0.5) * 0.35, scores[name], 12, t["series"][j])
        for j, (label, _) in enumerate(runs):  # legend: 10 px swatch beside the label text
            x = 0.02 + j * 0.34
            fig.patches.append(Rectangle((x, 1 - 81 / h), 10 / 800, 10 / h, transform=fig.transFigure,
                                         facecolor=t["series"][j], edgecolor="none"))
            fig.text(x + 18 / 800, 1 - 76 / h, label, color=t["secondary"], fontsize=9, va="center")
        paths.append(save(fig, out, "per_class_ap", theme))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", default="reports/data_audit.json")
    parser.add_argument("--results", default="reports/results")
    parser.add_argument("--output", default="assets")
    args = parser.parse_args()
    paths = audit_figures(json.loads(Path(args.audit).read_text(encoding="utf-8")), args.output)
    grouped, official = (Path(args.results) / f"{n}_test.json" for n in ("grouped", "official"))
    if grouped.is_file() and official.is_file():
        paths += results_figure(json.loads(grouped.read_text(encoding="utf-8")),
                                json.loads(official.read_text(encoding="utf-8")), args.output)
    for path in paths:
        print("Saved", path)


if __name__ == "__main__":
    main()
