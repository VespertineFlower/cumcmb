"""Dependency-free SVG path plotting and CSV export for simulator runs."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

from .diagnostics import ActionRecord
from .q3_path_planner import seven_point_cover

ROLE_COLORS = {
    "cover": "#2563eb",
    "localize": "#d97706",
    "piggyback": "#059669",
    "opportunistic_observation": "#9333ea",
    "clear": "#dc2626",
}

ROLE_LABELS = {
    "cover": "覆盖观测",
    "localize": "定位观测",
    "piggyback": "顺路观测",
    "opportunistic_observation": "机会观测",
    "clear": "清除",
}


def _marker_svg(role: str, x: float, y: float, title: str = "") -> str:
    color = ROLE_COLORS[role]
    detail = f"<title>{title}</title>" if title else ""
    if role == "clear":
        points = []
        for k in range(10):
            radius = 10 if k % 2 == 0 else 4.5
            angle = -math.pi / 2 + k * math.pi / 5
            points.append(f"{x + radius * math.cos(angle):.2f},{y + radius * math.sin(angle):.2f}")
        return f'<polygon points="{" ".join(points)}" fill="{color}" stroke="#7f1d1d" stroke-width="1">{detail}</polygon>'
    if role == "localize":
        points = f"{x:.2f},{y - 8:.2f} {x + 8:.2f},{y:.2f} {x:.2f},{y + 8:.2f} {x - 8:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="{color}" stroke="#fff" stroke-width="1.3">{detail}</polygon>'
    if role == "piggyback":
        return f'<rect x="{x - 6:.2f}" y="{y - 6:.2f}" width="12" height="12" fill="{color}" stroke="#fff" stroke-width="1.3">{detail}</rect>'
    if role == "opportunistic_observation":
        points = f"{x:.2f},{y - 8:.2f} {x + 8:.2f},{y + 7:.2f} {x - 8:.2f},{y + 7:.2f}"
        return f'<polygon points="{points}" fill="{color}" stroke="#fff" stroke-width="1.3">{detail}</polygon>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="{color}" stroke="#fff" stroke-width="1.3">{detail}</circle>'


RESULT_LABELS = {
    "direction": "示向",
    "near": "近距离",
    "no_signal": "无信号",
    "success": "成功",
    "failure": "失败",
    "failed": "失败",
}


class RunPlotter:
    def __init__(self, arena_radius: float = 1800.0, width: int = 1000, height: int = 900) -> None:
        self.arena_radius = arena_radius
        self.width = width
        self.height = height
        self.margin = 75

    def write(self, actions: list[ActionRecord], svg_path: str | Path) -> tuple[Path, Path]:
        svg = Path(svg_path)
        csv_path = svg.with_name(f"{svg.stem}_points.csv")
        svg.parent.mkdir(parents=True, exist_ok=True)
        svg.write_text(self._svg(actions), encoding="utf-8")
        self._write_csv(actions, csv_path)
        return svg, csv_path

    def _write_csv(self, actions: list[ActionRecord], path: Path) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["sequence", "action", "role", "x", "y", "channel", "result", "virtual_time_s", "movement_m"]
            )
            for item in actions:
                writer.writerow(
                    [
                        item.sequence,
                        item.action,
                        item.role,
                        item.point.x,
                        item.point.y,
                        item.channel,
                        item.result,
                        item.virtual_time_s,
                        item.movement_m,
                    ]
                )

    def _svg(self, actions: list[ActionRecord]) -> str:
        extent = max(
            [self.arena_radius * 1.08]
            + [max(abs(item.point.x), abs(item.point.y)) * 1.08 for item in actions]
        )
        plot_size = min(self.width - 2 * self.margin, self.height - 2 * self.margin)
        scale = plot_size / (2 * extent)
        cx, cy = self.width / 2, self.height / 2

        def xy(x: float, y: float) -> tuple[float, float]:
            return cx + x * scale, cy - y * scale

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{cx:.0f}" y="36" text-anchor="middle" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="22" font-weight="700">第三问：机械狗检测与清除路径</text>',
            f'<text x="{cx:.0f}" y="60" text-anchor="middle" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="13" fill="#475569">不同图形表示不同动作；同点多类动作略作错位，悬停可查看原坐标</text>',
        ]
        # Grid and axes.
        grid_step = 500
        low = -math.ceil(extent / grid_step) * grid_step
        high = -low
        for value in range(low, high + 1, grid_step):
            x1, y0 = xy(value, low)
            _, y1 = xy(value, high)
            x0, yy = xy(low, value)
            x2, _ = xy(high, value)
            color = "#e2e8f0"
            width = 0.7
            lines.append(f'<line x1="{x1:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="{width}"/>')
            lines.append(f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x2:.2f}" y2="{yy:.2f}" stroke="{color}" stroke-width="{width}"/>')
        lines.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{self.arena_radius * scale:.2f}" fill="none" stroke="#334155" stroke-width="2" stroke-dasharray="7 5"/>'
        )
        left, _ = xy(-extent * 0.96, 0)
        right, _ = xy(extent * 0.96, 0)
        _, bottom = xy(0, -extent * 0.96)
        _, top = xy(0, extent * 0.96)
        lines.extend([
            f'<line id="x-axis" x1="{left:.2f}" y1="{cy:.2f}" x2="{right:.2f}" y2="{cy:.2f}" stroke="#1e293b" stroke-width="1.8"/>',
            f'<polygon points="{right:.2f},{cy:.2f} {right - 10:.2f},{cy - 5:.2f} {right - 10:.2f},{cy + 5:.2f}" fill="#1e293b"/>',
            f'<line id="y-axis" x1="{cx:.2f}" y1="{bottom:.2f}" x2="{cx:.2f}" y2="{top:.2f}" stroke="#1e293b" stroke-width="1.8"/>',
            f'<polygon points="{cx:.2f},{top:.2f} {cx - 5:.2f},{top + 10:.2f} {cx + 5:.2f},{top + 10:.2f}" fill="#1e293b"/>',
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3" fill="#1e293b"/>',
            f'<text x="{cx + 8:.2f}" y="{cy + 18:.2f}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12" fill="#1e293b">原点（0，0）</text>',
            f'<text x="{right - 5:.2f}" y="{cy - 12:.2f}" text-anchor="end" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12" fill="#1e293b">正东（横轴，米）</text>',
            f'<text x="{cx + 13:.2f}" y="{top + 14:.2f}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12" fill="#1e293b">正北（纵轴，米）</text>',
        ])

        # Movement path, omitting zero-length repeated measurements.
        path_points = [(0.0, 0.0)]
        for item in actions:
            coordinate = (item.point.x, item.point.y)
            if coordinate != path_points[-1]:
                path_points.append(coordinate)
        if len(path_points) > 1:
            coords = " ".join(f"{xy(x, y)[0]:.2f},{xy(x, y)[1]:.2f}" for x, y in path_points)
            lines.append(f'<polyline points="{coords}" fill="none" stroke="#64748b" stroke-width="2" stroke-opacity="0.72"/>')

        cover_sites = [task.point for task in seven_point_cover()]

        def plot_role(item: ActionRecord) -> str:
            if item.role == "task_cover":
                return "cover"
            if item.role == "task_localize":
                return "localize"
            if item.role == "task_observation":  # Historical optimized logs.
                return "cover" if any(
                    abs(item.point.x - site.x) < 1e-5 and abs(item.point.y - site.y) < 1e-5
                    for site in cover_sites
                ) else "localize"
            return item.role

        grouped: dict[tuple[float, float], dict[str, list[ActionRecord]]] = defaultdict(lambda: defaultdict(list))
        for item in actions:
            grouped[(round(item.point.x, 6), round(item.point.y, 6))][plot_role(item)].append(item)
        used_roles: set[str] = set()
        for (x, y), role_items in grouped.items():
            true_x, true_y = xy(x, y)
            roles = [role for role in ROLE_COLORS if role in role_items]
            used_roles.update(roles)
            if len(roles) > 1:
                lines.append(f'<circle cx="{true_x:.2f}" cy="{true_y:.2f}" r="2" fill="#64748b"/>')
            for index, role in enumerate(roles):
                items = role_items[role]
                sx = true_x + 16 * (index - (len(roles) - 1) / 2)
                sy = true_y
                if sx != true_x:
                    lines.append(f'<line x1="{true_x:.2f}" y1="{true_y:.2f}" x2="{sx:.2f}" y2="{sy:.2f}" stroke="#94a3b8" stroke-width="0.8"/>')
                title = html.escape(
                    f"{ROLE_LABELS[role]}（横坐标 {x:.2f}，纵坐标 {y:.2f}）；"
                    + "，".join(f"第{i.sequence}次、频道{i.channel}、{RESULT_LABELS.get(i.result, '其他结果')}" for i in items)
                )
                lines.append(_marker_svg(role, sx, sy, title))
                if len(items) > 1:
                    lines.append(f'<text x="{sx + 10:.2f}" y="{sy - 9:.2f}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="10" fill="#0f172a">{len(items)}</text>')

        legend_x, legend_y = self.width - 185, 95
        for index, role in enumerate(role for role in ROLE_COLORS if role in used_roles):
            y = legend_y + index * 23
            lines.append(_marker_svg(role, legend_x, y))
            lines.append(f'<text x="{legend_x + 16}" y="{y + 4}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12">{ROLE_LABELS[role]}</text>')
        lines.append(f'<text x="40" y="{self.height - 24}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="12" fill="#475569">动作数 {len(actions)}；检测 {sum(a.action == "measure" for a in actions)}；清除 {sum(a.action == "clear" for a in actions)}</text>')
        lines.append("</svg>")
        return "\n".join(lines)
