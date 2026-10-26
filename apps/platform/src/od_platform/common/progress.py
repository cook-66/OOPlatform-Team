"""轻量终端进度条。

核心服务只依赖 ``ProgressCallback``，CLI 决定是否显示终端进度，因而不会污染
pytest、Web 调用或日志文件。实现只使用标准库，避免为进度显示增加运行依赖。
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import TextIO

ProgressCallback = Callable[[str, int, int], None]


class TerminalProgress:
    """在交互终端中显示单行进度；重定向或 CI 环境自动静默。"""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        width: int = 28,
        enabled: bool | None = None,
        min_interval: float = 0.1,
    ) -> None:
        self.stream = stream or sys.stderr
        self.width = width
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self.min_interval = min_interval
        self._stage: str | None = None
        self._started_at = 0.0
        self._last_rendered_at = 0.0

    def __call__(self, stage: str, current: int, total: int) -> None:
        if not self.enabled:
            return
        total = max(1, total)
        current = min(max(0, current), total)
        now = time.monotonic()
        if stage != self._stage:
            self.close()
            self._stage = stage
            self._started_at = now
            self._last_rendered_at = 0.0
        if current < total and current > 0 and now - self._last_rendered_at < self.min_interval:
            return

        ratio = current / total
        filled = round(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(0.0, now - self._started_at)
        rate = current / elapsed if current and elapsed else 0.0
        eta = (total - current) / rate if rate else 0.0
        suffix = f" ETA {eta:5.1f}s" if current < total and rate else ""
        self.stream.write(
            f"\r{stage:<20} [{bar}] {ratio:6.1%} {current}/{total}{suffix}"
        )
        self.stream.flush()
        self._last_rendered_at = now
        if current >= total:
            self.stream.write("\n")
            self.stream.flush()
            self._stage = None

    def close(self) -> None:
        if self.enabled and self._stage is not None:
            self.stream.write("\n")
            self.stream.flush()
        self._stage = None
