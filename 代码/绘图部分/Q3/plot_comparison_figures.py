"""从第三问的官方演练日志生成时间分解和稳健性图。"""

from __future__ import annotations

import csv
import glob
import json
import statistics as stats
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


sns.set_theme(style="white", context="paper", font="Microsoft YaHei", font_scale=1.0)
mpl.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

COLORS = {"Baseline": "#8EC9DE", "Optimized": "#F4B79B"}
TEXT_COLORS = {"Baseline": "#4A91AE", "Optimized": "#C87556"}
PART_COLORS = ["#7CBBD1", "#F4C782", "#9870B8", "#C7D5D8"]


def load_metrics(prefix: str):
    paths = sorted(glob.glob(str(ROOT / "logs" / f"{prefix}_metrics_*.json")))
    assert len(paths) == 20, (prefix, len(paths))
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def load_rows(prefix: str):
    with (ROOT / "logs" / f"{prefix}_运行指标.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20, (prefix, len(rows))
    return rows


def component_times(run):
    current = 1
    switches = 0
    n_measure = 0
    for action in run["actions"]:
        if action["action"] == "measure":
            switches += action["channel"] != current
            current = action["channel"]
            n_measure += 1
    summary = run["summary"]
    values = [summary["total_movement_m"] / 5, 5 * n_measure, switches, 5 * summary["clear_attempt_count"]]
    actual = run["actions"][-1]["virtual_time_s"]
    assert abs(sum(values) - actual) < 1e-4, (sum(values), actual)
    return values


def export(fig, stem: str):
    fig.canvas.draw()
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


data = {
    "Baseline": {"metrics": load_metrics("robot_dog"), "rows": load_rows("robot_dog")},
    "Optimized": {"metrics": load_metrics("robot_dog_anytime"), "rows": load_rows("robot_dog_anytime")},
}

# 图 1：两个方法的绝对时间构成；柱高保留总时间差。
means = np.array([[stats.mean(component_times(run)[j] for run in data[name]["metrics"]) for j in range(4)] for name in data])
fig, ax = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
x = np.arange(2)
bottom = np.zeros(2)
for j, label in enumerate(("移动", "检测", "换频", "清除")):
    ax.bar(x, means[:, j], bottom=bottom, color=PART_COLORS[j], width=.48, label=label, edgecolor="white", linewidth=.5)
    if j < 2:
        for i in range(2):
            ax.text(x[i], bottom[i] + means[i, j] / 2, f"{means[i, j]:.0f} 秒", ha="center", va="center", color="#284656", fontweight="bold", fontsize=9)
    bottom += means[:, j]
for i, total in enumerate(bottom):
    ax.text(x[i], total + 90, f"总计 {total:.0f} 秒", ha="center", fontweight="bold", fontsize=10)
ax.set_xticks(x, list(data))
ax.set_ylabel("单次演练平均虚拟时间（秒）")
ax.set_ylim(0, 5900)
ax.set_title("两种策略的虚拟时间构成", loc="left", fontweight="bold")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.11), frameon=False)
export(fig, "图1_时间构成")

# 图 2：呈现每组完整的 20 次分布，不暗示案例之间存在配对关系。
fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
for i, (name, group) in enumerate(data.items()):
    vals = np.array([float(r["平均定位清除时间"]) for r in group["rows"]])
    ax.boxplot(vals, positions=[i], widths=.42, patch_artist=True, showfliers=False, medianprops={"color": "#35505F", "linewidth": 1.4}, boxprops={"facecolor": COLORS[name], "alpha": .28, "edgecolor": TEXT_COLORS[name]}, whiskerprops={"color": TEXT_COLORS[name]}, capprops={"color": TEXT_COLORS[name]})
    ax.scatter(np.full(20, i) + np.linspace(-.13, .13, 20), vals, s=22, color=COLORS[name], alpha=.9, edgecolors="white", linewidths=.35, zorder=3)
    ax.scatter([i], [np.mean(vals)], marker="D", s=43, color=TEXT_COLORS[name], edgecolors="white", linewidths=.6, zorder=4)
    ax.text(i, 540, f"均值 {np.mean(vals):.1f}\n最大 {np.max(vals):.1f}", ha="center", fontsize=8, color=TEXT_COLORS[name])
ax.set_xticks([0, 1], list(data))
ax.set_xlim(-.5, 1.5)
ax.set_ylim(260, 560)
ax.set_ylabel("平均定位清除时间（秒/个）")
ax.set_title("两种策略的单位目标耗时分布", loc="left", fontweight="bold")
export(fig, "图2_稳健性")
