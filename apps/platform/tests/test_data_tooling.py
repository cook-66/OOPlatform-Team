"""D3/D4 进度反馈、历史比较和交付工具测试。"""
from __future__ import annotations

import io
import json

from od_platform.common.progress import TerminalProgress
from od_platform.data_validation.history import compare_reports, find_previous_report
from od_platform.data_validation.registry import CheckResult, CheckSeverity


def test_terminal_progress_renders_percentage_and_finishes_line():
    stream = io.StringIO()
    progress = TerminalProgress(stream, enabled=True, min_interval=0)
    progress("D3 测试", 0, 2)
    progress("D3 测试", 1, 2)
    progress("D3 测试", 2, 2)
    rendered = stream.getvalue()
    assert "50.0%" in rendered
    assert "100.0%" in rendered
    assert rendered.endswith("\n")


def test_compare_reports_finds_new_resolved_and_changed_issues():
    previous = {
        "run_id": "old",
        "overall_severity": "ERROR",
        "results": [
            {"name": "fixed", "severity": "ERROR"},
            {"name": "worse", "severity": "INFO"},
            {"name": "stable", "severity": "WARNING"},
            {"name": "new", "severity": "PASS"},
        ],
    }
    current = [
        CheckResult("fixed", CheckSeverity.PASS, "ok"),
        CheckResult("worse", CheckSeverity.WARNING, "worse"),
        CheckResult("stable", CheckSeverity.WARNING, "same"),
        CheckResult("new", CheckSeverity.ERROR, "new"),
    ]
    comparison = compare_reports(current, previous)
    assert [x["check"] for x in comparison["resolved_issues"]] == ["fixed"]
    assert [x["check"] for x in comparison["severity_changes"]] == ["worse"]
    assert [x["check"] for x in comparison["unchanged_issues"]] == ["stable"]
    assert [x["check"] for x in comparison["new_issues"]] == ["new"]


def test_find_previous_report_uses_same_yaml_and_latest_run(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("names: []\n", encoding="utf-8")
    runs = tmp_path / "runs"
    for run_id in ("20260101-000000", "20260102-000000"):
        report_dir = runs / run_id
        report_dir.mkdir(parents=True)
        (report_dir / "report.json").write_text(json.dumps({
            "run_id": run_id,
            "yaml_path": str(yaml_path),
            "overall_severity": "PASS",
            "results": [],
        }), encoding="utf-8")
    unrelated = runs / "20260103-000000"
    unrelated.mkdir()
    (unrelated / "report.json").write_text(json.dumps({
        "run_id": "20260103-000000",
        "yaml_path": str(tmp_path / "other.yaml"),
        "results": [],
    }), encoding="utf-8")

    found = find_previous_report(yaml_path, "current", runs)
    assert found is not None
    assert found["run_id"] == "20260102-000000"
