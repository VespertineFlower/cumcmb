# 第三问绘图材料

本文件夹可独立生成两张中文图。将整个文件夹复制到其他位置后，目录结构保持不变即可运行。

```text
绘图材料/
├─ plot_comparison_figures.py  绘图脚本
├─ logs/                       原始输入：40 份指标 JSON、2 份运行指标 CSV
└─ figures/                    已导出的 PNG、SVG、PDF 图
```

## 运行

需要 Python 及 `numpy`、`matplotlib`、`seaborn`：

```powershell
python -m pip install numpy matplotlib seaborn
python plot_comparison_figures.py
```

脚本以自身所在目录为根目录读取 `logs/`，并在 `figures/` 中覆盖生成两张图的 PNG、SVG、PDF 版本。代码没有机器专属的绝对路径。图 1 的四项时间由逐动作 JSON 重算；图 2 使用逐次运行指标 CSV 中的平均定位清除时间。脚本会检查每组是否为 20 次，并核对时间分项之和与虚拟总时间。

图中的两组演练来自不同随机案例，不能按运行序号配对。
