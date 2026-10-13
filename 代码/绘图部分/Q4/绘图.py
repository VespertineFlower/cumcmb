"""在本文件夹运行：python 绘图.py。仅需 matplotlib 和 seaborn。"""
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

here = Path(__file__).resolve().parent
with (here / "40次绘图数据.csv").open(encoding="utf-8-sig", newline="") as handle:
    data = list(csv.DictReader(handle))
if len(data) != 40:
    raise ValueError(f"预期40行，实际{len(data)}行")

rows = []
for item in data:
    total = float(item["总虚拟时间_秒"])
    sources = int(item["清除源数"])
    movement = float(item["移动距离_米"]) / 5
    detection = int(item["检测次数"]) * 5
    clearing = int(item["清除成功次数"]) * 5 + int(item["清除失败次数"]) * 3
    rows.append({"total": total, "per_source": total / sources,
                 "movement": movement, "detection": detection,
                 "other": total - movement - detection - clearing,
                 "clearing": clearing})
    if rows[-1]["other"] < -1e-3:
        raise ValueError("分项时间超过总虚拟时间，请核对数据和成本参数")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
})

def save(fig, stem):
    fig.savefig(here / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(here / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(here / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

# 图1：原题规定的单次平均定位清除时间。
fig, ax = plt.subplots(figsize=(5.2, 4.6))
values = [r["per_source"] for r in rows]
sns.boxplot(y=values, ax=ax, color="#b9d5df", width=.32, showfliers=False,
            boxprops={"alpha": .85}, medianprops={"color": "#173b55", "linewidth": 1.7})
sns.stripplot(y=values, ax=ax, color="#176b91", size=5, jitter=.16,
              alpha=.78, edgecolor="white", linewidth=.35, zorder=3)
median = sum(sorted(values)[19:21]) / 2
ax.axhline(median, color="#c35b40", linestyle="--", linewidth=1.1, zorder=1)
ax.text(.29, median + 9, f"中位数 {median:.1f} 秒/个", color="#a34833",
        ha="left", va="bottom", fontsize=8)
ax.set(xlabel="40 次运行（每点代表一次）", ylabel="平均定位清除时间（秒/个）",
       title="每次运行的平均定位清除时间")
ax.set_xticks([])
ax.set_ylim(410, 865)
ax.grid(axis="y", alpha=.25)
ax.grid(axis="x", visible=False)
ax.text(.98, .03, "总虚拟时间 ÷ 本次清除源数", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7, color="#52606a")
fig.tight_layout()
save(fig, "图1_平均定位清除时间分布")

# 图2：40次运行逐次时间分解，按总时间排序。
ordered = sorted(rows, key=lambda r: r["total"])
fig, ax = plt.subplots(figsize=(8.4, 4.15))
colors = {"movement": "#227a9b", "detection": "#69b5bd",
          "other": "#e6ac55", "clearing": "#c86452"}
names = {"movement": "移动", "detection": "检测",
         "other": "换频等剩余耗时", "clearing": "清除"}
bottom = [0.0] * 40
for key in ("movement", "detection", "other", "clearing"):
    vals = [r[key] for r in ordered]
    ax.bar(range(1, 41), vals, bottom=bottom, width=.78, color=colors[key],
           label=names[key], edgecolor="white", linewidth=.15)
    bottom = [a + b for a, b in zip(bottom, vals)]
ax.axhline(sum(r["total"] for r in ordered) / 40, linestyle="--", color="#34495e",
           linewidth=1, label="总时间均值")
ax.set(xlim=(.2, 40.8), ylim=(0, 9600), xlabel="按总耗时排序的运行序号（由短到长）",
       ylabel="任务虚拟时间（秒）", title="同心环方案 40 次运行的时间分解")
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=7.3, columnspacing=1.1)
ax.grid(axis="y", alpha=.23)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "图2_逐次时间分解")
