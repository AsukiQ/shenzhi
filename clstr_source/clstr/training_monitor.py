from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from xml.sax.saxutils import escape


RECALL_KEYS = ("recall_at_50", "recall_at_10", "recall_at_5", "recall_at_1")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    torch_module = sys.modules.get("torch")
    tensor_type = getattr(torch_module, "Tensor", None) if torch_module is not None else None
    if tensor_type is not None and isinstance(value, tensor_type):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class TrainingMonitor:
    """Append per-step metrics, refresh a loss curve, and overwrite latest checkpoint."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        checkpoint_dir: str | Path | None = None,
        metrics_filename: str = "training_metrics.jsonl",
        loss_curve_filename: str = "loss_curve.svg",
        diagnostic_curves_filename: str = "diagnostic_curves.svg",
        latest_checkpoint_filename: str = "latest.pt",
        checkpoint_interval_steps: int = 400,
        curve_interval_steps: int = 400,
        reset: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / metrics_filename
        self.loss_curve_path = self.output_dir / loss_curve_filename
        self.diagnostic_curves_path = self.output_dir / diagnostic_curves_filename
        self.progress_path = self.output_dir / "progress.json"
        self.latest_checkpoint_path = self.checkpoint_dir / latest_checkpoint_filename
        self.checkpoint_interval_steps = max(1, int(checkpoint_interval_steps))
        self.curve_interval_steps = max(1, int(curve_interval_steps))
        self.history: list[dict[str, Any]] = []
        if reset:
            self.metrics_path.write_text("", encoding="utf-8")
            self._write_loss_curve()
            self._write_diagnostic_curves()

    def record(self, metrics: dict[str, Any], checkpoint_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        safe_metrics = _json_safe(metrics)
        self.history.append(safe_metrics)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_metrics, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        step = self._record_step(safe_metrics)
        self._write_progress(safe_metrics, step)
        if self._should_write(step, self.curve_interval_steps):
            self._write_loss_curve()
            self._write_diagnostic_curves()
        if checkpoint_payload is not None and self._should_write(step, self.checkpoint_interval_steps):
            self.save_latest(checkpoint_payload)
        return safe_metrics

    def _record_step(self, metrics: dict[str, Any]) -> int:
        raw_step = metrics.get("step", metrics.get("update", len(self.history)))
        try:
            return max(1, int(raw_step))
        except (TypeError, ValueError):
            return max(1, len(self.history))

    @staticmethod
    def _should_write(step: int, interval: int) -> bool:
        return int(step) % max(1, int(interval)) == 0

    def should_save_checkpoint(self, step: int) -> bool:
        return self._should_write(step, self.checkpoint_interval_steps)

    def save_latest(self, payload: dict[str, Any]) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is required when TrainingMonitor.save_latest writes checkpoints") from exc

        payload = dict(payload)
        payload.setdefault("checkpoint_role", "rolling_latest")
        payload.setdefault("training_metrics_path", str(self.metrics_path))
        payload.setdefault("loss_curve_path", str(self.loss_curve_path))
        payload.setdefault("diagnostic_curves_path", str(self.diagnostic_curves_path))
        tmp_path = self.latest_checkpoint_path.with_suffix(self.latest_checkpoint_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.latest_checkpoint_path)

    def paths_report(self) -> dict[str, str]:
        return {
            "training_metrics_path": str(self.metrics_path),
            "loss_curve_path": str(self.loss_curve_path),
            "diagnostic_curves_path": str(self.diagnostic_curves_path),
            "progress_path": str(self.progress_path),
            "latest_checkpoint": str(self.latest_checkpoint_path),
        }

    def _write_progress(self, metrics: dict[str, Any], step: int) -> None:
        progress = {
            "status": "running",
            "updated_at_unix": time.time(),
            "step": int(step),
            "metrics_path": str(self.metrics_path),
            "loss_curve_path": str(self.loss_curve_path),
            "diagnostic_curves_path": str(self.diagnostic_curves_path),
            **metrics,
        }
        tmp_path = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.progress_path)

    def _loss_points(self) -> list[tuple[float, float]]:
        return self._metric_points("loss")

    @staticmethod
    def _value_at_path(row: dict[str, Any], key: str) -> Any:
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def _metric_points(self, key: str) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for idx, row in enumerate(self.history, start=1):
            raw_x = row.get("step", row.get("update", idx))
            raw_y = self._value_at_path(row, key) if "." in key else row.get(key)
            try:
                x = float(raw_x)
                y = float(raw_y)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        return points

    def _recall_key(self) -> str | None:
        present = {str(key) for row in self.history for key in row.keys()}
        for key in RECALL_KEYS:
            if key in present:
                return key
        for key in sorted(present):
            if key.startswith("recall_at_"):
                return key
        return None

    @staticmethod
    def _rolling_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not points:
            return []
        window = max(3, min(100, max(3, len(points) // 20)))
        values: list[tuple[float, float]] = []
        running_sum = 0.0
        queue: list[float] = []
        for x, y in points:
            queue.append(y)
            running_sum += y
            if len(queue) > window:
                running_sum -= queue.pop(0)
            values.append((x, running_sum / len(queue)))
        return values

    @staticmethod
    def _ema_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not points:
            return []
        alpha = 0.08 if len(points) >= 100 else 0.25
        ema = points[0][1]
        values = [(points[0][0], ema)]
        for x, y in points[1:]:
            ema = alpha * y + (1.0 - alpha) * ema
            values.append((x, ema))
        return values

    @staticmethod
    def _polyline(points: list[tuple[float, float]], sx: Any, sy: Any) -> str:
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)

    def _write_loss_curve(self) -> None:
        points = self._loss_points()
        width = 900
        height = 360
        margin_left = 64
        margin_right = 64
        margin_top = 24
        margin_bottom = 48
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        if points:
            xs = [x for x, _y in points]
            ys = [y for _x, y in points]
            rolling_points = self._rolling_points(points)
            ema_points = self._ema_points(points)
            ys.extend(y for _x, y in rolling_points)
            ys.extend(y for _x, y in ema_points)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            if max_x <= min_x:
                max_x = min_x + 1.0
            if max_y <= min_y:
                max_y = min_y + 1.0

            def sx(x: float) -> float:
                return margin_left + (x - min_x) / (max_x - min_x) * plot_width

            def sy(y: float) -> float:
                return margin_top + (max_y - y) / (max_y - min_y) * plot_height

            last_x, last_y = points[-1]
            recall_key = self._recall_key()
            recall_points = self._metric_points(recall_key) if recall_key is not None else []

            def sy_recall(y: float) -> float:
                return margin_top + (1.0 - max(0.0, min(1.0, y))) * plot_height

            path_markup = (
                f'<polyline points="{self._polyline(points, sx, sy)}" fill="none" '
                'stroke="#9aa7b2" stroke-width="1.2" stroke-opacity="0.55" />\n'
                f'<polyline points="{self._polyline(rolling_points, sx, sy)}" fill="none" '
                'stroke="#1f77b4" stroke-width="2.5" />\n'
                f'<polyline points="{self._polyline(ema_points, sx, sy)}" fill="none" '
                'stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6 4" />\n'
                f'<circle cx="{sx(last_x):.2f}" cy="{sy(last_y):.2f}" r="4" fill="#d62728" />\n'
                f'<text x="{width - margin_right}" y="{margin_top + 18}" text-anchor="end" font-size="13">'
                f'last loss={last_y:.6g}</text>\n'
                f'<text x="{margin_left + 8}" y="{margin_top + 18}" font-size="12" fill="#9aa7b2">raw loss</text>\n'
                f'<text x="{margin_left + 8}" y="{margin_top + 34}" font-size="12" fill="#1f77b4">rolling loss</text>\n'
                f'<text x="{margin_left + 8}" y="{margin_top + 50}" font-size="12" fill="#ff7f0e">ema loss</text>'
            )
            if recall_points:
                last_recall_x, last_recall_y = recall_points[-1]
                path_markup += (
                    f'\n<polyline points="{self._polyline(recall_points, sx, sy_recall)}" fill="none" '
                    'stroke="#2ca02c" stroke-width="2" />\n'
                    f'<circle cx="{sx(last_recall_x):.2f}" cy="{sy_recall(last_recall_y):.2f}" '
                    'r="3.5" fill="#2ca02c" />\n'
                    f'<text x="{margin_left + 8}" y="{margin_top + 66}" font-size="12" fill="#2ca02c">'
                    f'{recall_key}</text>\n'
                    f'<text x="{width - margin_right}" y="{margin_top + 34}" text-anchor="end" '
                    f'font-size="13">last {recall_key}={last_recall_y:.4g}</text>'
                )
            y_min_label = f"{min_y:.4g}"
            y_max_label = f"{max_y:.4g}"
            x_min_label = f"{min_x:.0f}"
            x_max_label = f"{max_x:.0f}"
        else:
            path_markup = (
                f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" '
                'font-size="16" fill="#555">waiting for training metrics</text>'
            )
            y_min_label = "0"
            y_max_label = "1"
            x_min_label = "0"
            x_max_label = "1"

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
<text x="{width / 2:.0f}" y="20" text-anchor="middle" font-size="15">Training loss</text>
<text x="{width / 2:.0f}" y="{height - 10}" text-anchor="middle" font-size="12">step/update</text>
<text x="18" y="{height / 2:.0f}" text-anchor="middle" font-size="12" transform="rotate(-90 18 {height / 2:.0f})">loss</text>
<text x="{width - 18}" y="{height / 2:.0f}" text-anchor="middle" font-size="12" transform="rotate(90 {width - 18} {height / 2:.0f})">right axis: recall</text>
<text x="{margin_left - 8}" y="{margin_top + 4}" text-anchor="end" font-size="11">{y_max_label}</text>
<text x="{margin_left - 8}" y="{margin_top + plot_height}" text-anchor="end" font-size="11">{y_min_label}</text>
<text x="{margin_left + plot_width + 8}" y="{margin_top + 4}" text-anchor="start" font-size="11">1</text>
<text x="{margin_left + plot_width + 8}" y="{margin_top + plot_height}" text-anchor="start" font-size="11">0</text>
<text x="{margin_left}" y="{margin_top + plot_height + 18}" text-anchor="middle" font-size="11">{x_min_label}</text>
<text x="{margin_left + plot_width}" y="{margin_top + plot_height + 18}" text-anchor="middle" font-size="11">{x_max_label}</text>
{path_markup}
</svg>
"""
        tmp_path = self.loss_curve_path.with_suffix(self.loss_curve_path.suffix + ".tmp")
        tmp_path.write_text(svg, encoding="utf-8")
        tmp_path.replace(self.loss_curve_path)

    def _diagnostic_metric_keys(self) -> list[str]:
        preferred = [
            "loss",
            "policy_ce_loss",
            "transition_skill_ce_loss",
            "transition_cosine_loss",
            "belief_cosine_loss",
            "stop_bce_loss",
            "transition_skill_recall@1",
            "transition_skill_recall@5",
            "transition_skill_mrr",
            "stage0_prior_transition_skill_recall@5",
            "stage0_prior_transition_skill_mrr",
            "batch_policy_rows",
            "batch_transition_real_rows",
            "batch_transition_switch_real_rows",
            "batch_transition_injected_rows",
            "transition_skill_ce_candidate_count",
            "transition_inventory_mask_backfilled_candidates",
        ]
        present = {str(key) for row in self.history for key in row.keys()}
        nested_weighted = []
        if any(isinstance(row.get("weighted_loss_terms"), dict) for row in self.history):
            weighted_keys = {
                str(key)
                for row in self.history
                if isinstance(row.get("weighted_loss_terms"), dict)
                for key in row["weighted_loss_terms"].keys()
            }
            nested_weighted = [f"weighted_loss_terms.{key}" for key in sorted(weighted_keys)]
        keys = [key for key in preferred if key in present]
        return keys + nested_weighted

    def _write_diagnostic_curves(self) -> None:
        width = 980
        height = 520
        margin_left = 72
        margin_right = 220
        margin_top = 36
        margin_bottom = 44
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        keys = self._diagnostic_metric_keys()
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if self.history and keys:
            xs = [
                float(row.get("step", row.get("update", idx)))
                for idx, row in enumerate(self.history, start=1)
                if isinstance(row.get("step", row.get("update", idx)), (int, float))
            ]
            min_x = min(xs) if xs else 1.0
            max_x = max(xs) if xs else min_x + 1.0
            if max_x <= min_x:
                max_x = min_x + 1.0

            def sx(x: float) -> float:
                return margin_left + (x - min_x) / (max_x - min_x) * plot_width

            def sy_norm(y: float) -> float:
                return margin_top + (1.0 - max(0.0, min(1.0, y))) * plot_height

            polylines: list[str] = []
            legend: list[str] = []
            for idx, key in enumerate(keys):
                points = self._metric_points(key)
                if not points:
                    continue
                rolling = self._rolling_points(points)
                values = [value for _x, value in rolling]
                min_y = min(values)
                max_y = max(values)
                span = max(max_y - min_y, 1.0e-8)
                normalized = [(x, (y - min_y) / span) for x, y in rolling]
                color = colors[idx % len(colors)]
                polylines.append(
                    f'<polyline points="{self._polyline(normalized, sx, sy_norm)}" fill="none" '
                    f'stroke="{color}" stroke-width="2" stroke-opacity="0.88" />'
                )
                legend_y = margin_top + 18 + idx * 18
                last_value = points[-1][1]
                legend.append(
                    f'<text x="{margin_left + plot_width + 18}" y="{legend_y}" font-size="11" fill="{color}">'
                    f'{escape(key)} last={last_value:.4g}</text>'
                )
            markup = "\n".join(polylines + legend)
            x_min_label = f"{min_x:.0f}"
            x_max_label = f"{max_x:.0f}"
        else:
            markup = (
                f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" '
                'font-size="16" fill="#555">waiting for diagnostic metrics</text>'
            )
            x_min_label = "0"
            x_max_label = "1"

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
<text x="{width / 2:.0f}" y="22" text-anchor="middle" font-size="15">Component diagnostics</text>
<text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" font-size="12">step/update</text>
<text x="18" y="{height / 2:.0f}" text-anchor="middle" font-size="12" transform="rotate(-90 18 {height / 2:.0f})">per-series normalized rolling value</text>
<text x="{margin_left}" y="{margin_top + plot_height + 18}" text-anchor="middle" font-size="11">{x_min_label}</text>
<text x="{margin_left + plot_width}" y="{margin_top + plot_height + 18}" text-anchor="middle" font-size="11">{x_max_label}</text>
{markup}
</svg>
"""
        tmp_path = self.diagnostic_curves_path.with_suffix(self.diagnostic_curves_path.suffix + ".tmp")
        tmp_path.write_text(svg, encoding="utf-8")
        tmp_path.replace(self.diagnostic_curves_path)


def reset_setup_status(path: str | Path) -> Path:
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("", encoding="utf-8")
    return status_path


def append_setup_status(path: str | Path, phase: str, **payload: Any) -> Path:
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"phase": str(phase), **_json_safe(payload)}
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return status_path
