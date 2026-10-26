"""D4 历史报告查找与差异计算。只读取既有报告，不触碰数据集。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from od_platform.common import paths
from od_platform.data_validation.registry import CheckResult, CheckSeverity


def _resolved_yaml(raw: str | Path) -> Path:
    value = Path(raw)
    if not value.is_absolute():
        value = paths.ROOT_DIR / value
    return value.resolve(strict=False)


def find_previous_report(
    yaml_path: Path,
    current_run_id: str,
    runs_dir: Path | None = None,
) -> dict[str, Any] | None:
    """找到同一 YAML 最近一次可读取的 D4 报告。"""
    root = runs_dir or paths.VALIDATION_RUNS_DIR
    if not root.exists():
        return None
    expected = _resolved_yaml(yaml_path)
    candidates: list[tuple[str, float, dict[str, Any]]] = []
    for report_path in root.glob("*/report.json"):
        if report_path.parent.name == current_run_id:
            continue
        try:
            document = json.loads(report_path.read_text(encoding="utf-8"))
            if _resolved_yaml(document["yaml_path"]) != expected:
                continue
            candidates.append((
                str(document.get("run_id", report_path.parent.name)),
                report_path.stat().st_mtime,
                document,
            ))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def compare_reports(
    current_results: list[CheckResult],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """按检查名称比较严重度，明确新增、修复、变化和未变问题。"""
    if previous is None:
        return {
            "status": "NO_PREVIOUS",
            "previous_run_id": None,
            "summary": "没有找到同一数据集的历史质检报告，本次作为基线。",
            "new_issues": [],
            "resolved_issues": [],
            "severity_changes": [],
            "unchanged_issues": [],
        }

    old = {
        item["name"]: item
        for item in previous.get("results", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    new = {item.name: item for item in current_results}
    new_issues: list[dict[str, str]] = []
    resolved: list[dict[str, str]] = []
    changes: list[dict[str, str]] = []
    unchanged: list[dict[str, str]] = []

    for name in sorted(set(old) | set(new)):
        old_severity = old.get(name, {}).get("severity")
        new_severity = new[name].severity if name in new else None
        if old_severity is None and new_severity != CheckSeverity.PASS:
            new_issues.append({"check": name, "severity": str(new_severity)})
        elif new_severity is None:
            changes.append({"check": name, "before": str(old_severity), "after": "REMOVED"})
        elif old_severity == CheckSeverity.PASS and new_severity != CheckSeverity.PASS:
            new_issues.append({"check": name, "severity": new_severity})
        elif old_severity != CheckSeverity.PASS and new_severity == CheckSeverity.PASS:
            resolved.append({"check": name, "previous_severity": str(old_severity)})
        elif old_severity != new_severity:
            changes.append({"check": name, "before": str(old_severity), "after": new_severity})
        elif new_severity != CheckSeverity.PASS:
            unchanged.append({"check": name, "severity": new_severity})

    return {
        "status": "COMPARED",
        "previous_run_id": previous.get("run_id"),
        "previous_overall_severity": previous.get("overall_severity"),
        "summary": (
            f"新增问题 {len(new_issues)}，已修复 {len(resolved)}，"
            f"严重度变化 {len(changes)}，持续问题 {len(unchanged)}。"
        ),
        "new_issues": new_issues,
        "resolved_issues": resolved,
        "severity_changes": changes,
        "unchanged_issues": unchanged,
    }
