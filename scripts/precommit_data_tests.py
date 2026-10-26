"""跨平台启动项目虚拟环境中的 D3/D4 pre-commit 回归测试。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    )
    python = next((path for path in candidates if path.is_file()), None)
    if python is None:
        print("找不到项目 .venv，请先创建环境并安装 .\\apps\\platform[dev]。",
              file=sys.stderr)
        return 1
    tests = (
        "apps/platform/tests/test_data_pipeline.py",
        "apps/platform/tests/test_data_validation.py",
        "apps/platform/tests/test_data_tooling.py",
        "apps/platform/tests/test_source_layout.py",
    )
    return subprocess.call([str(python), "-m", "pytest", "-q", *tests], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
