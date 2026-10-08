"""Pure chart construction for local learning statistics; no persistence or APIs."""

from datetime import datetime, timedelta
from typing import Any, Sequence

import plotly.graph_objects as go

from domain import Progress, ensure_utc, parse_utc
from scheduler import retention_for


def forgetting_figure(progress: Progress, days: int, *, now: datetime) -> go.Figure:
    current = ensure_utc(now)
    offsets = [days * index / 200 for index in range(201)]
    values = [
        100 * retention_for(progress, now=current + timedelta(days=offset))
        for offset in offsets
    ]
    figure = go.Figure(go.Scatter(
        x=offsets, y=values, mode="lines", name="预计保留率",
        line={"color": "#2E86AB", "width": 3},
        hovertemplate="从今天起 %{x:.1f} 天<br>预计保留率 %{y:.1f}%<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[0], y=[values[0]], mode="markers", name="当前",
        marker={"color": "#D55E00", "size": 10},
        hovertemplate="当前保留率 %{y:.1f}%<extra></extra>",
    ))
    figure.update_layout(
        xaxis={"title": "从今天起的天数", "range": [0, days]},
        yaxis={"title": "预计保留率", "range": [0, 100], "ticksuffix": "%"},
        template="plotly_white", height=360,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "y": 1.15},
    )
    return figure


def baseline_figure(samples: Sequence[dict[str, Any]]) -> go.Figure:
    details = []
    for sample in samples:
        reviewed = parse_utc(sample["reviewed_at"])
        details.append([
            f"{sample['word']} · {sample['pos']} · {sample['meaning']}",
            "—" if reviewed is None else reviewed.astimezone().strftime("%Y-%m-%d %H:%M"),
        ])
    figure = go.Figure(go.Scatter(
        x=[100 * sample["expected_performance"] for sample in samples],
        y=[100 * sample["target_performance"] for sample in samples],
        customdata=details, mode="markers", name="有效练习",
        marker={"color": "#2E86AB", "size": 8, "opacity": 0.65},
        hovertemplate=("%{customdata[0]}<br>%{customdata[1]}"
                       "<br>当前预期 %{x:.1f}%<br>实际表现 %{y:.1f}%<extra></extra>"),
    ))
    figure.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines", name="实际＝预期",
        line={"color": "#666666", "dash": "dash"}, hoverinfo="skip",
    ))
    figure.update_layout(
        xaxis={"title": "当前模型预期目标词表现", "range": [0, 100], "ticksuffix": "%"},
        yaxis={"title": "实际目标词表现", "range": [0, 100], "ticksuffix": "%"},
        template="plotly_white", height=400,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "y": 1.15},
    )
    return figure
