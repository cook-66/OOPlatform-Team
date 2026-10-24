# apps/platform/src/od_platform/data_validation/checks/manifest_lineage.py
"""check: manifest_lineage —— 逐样本核验:在契约说的 split 里、内容 sha256 没变。
changed(内容变了)/ moved(split 挪了)/ missing(丢了)分开计数。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from od_platform.common.constants import LINEAGE_MAX_DETAIL
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "manifest_lineage"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@check(_NAME)
def validate_manifest_lineage(ctx: CheckContext) -> CheckResult:
    manifest, err, _ = load_manifest_from_yaml(ctx.snapshot.yaml_data)
    if err:
        return CheckResult(_NAME, CheckSeverity.ERROR, f"拿不到 D3 划分契约: {err}", {})
    snap = ctx.snapshot

    # 磁盘实况:(split, stem) → 图片/标签路径;stem → 实际出现的 split 集合
    disk: Dict[Tuple[str, str], Path] = {}
    disk_images: Dict[Tuple[str, str], Path] = {}
    disk_splits_of_stem: Dict[str, set] = {}
    for split in snap.splits:
        for img in snap.images_per_split[split]:
            disk_images[(split, img.stem)] = img
        for lbl in snap.label_files_per_split[split]:
            disk[(split, lbl.stem)] = lbl
            disk_splits_of_stem.setdefault(lbl.stem, set()).add(split)

    changed: List[str] = []
    changed_images: List[str] = []
    moved: List[dict] = []
    missing: List[str] = []
    missing_images: List[str] = []
    for s in manifest.samples:
        lbl = disk.get((s.split, s.stem))
        if lbl is None:
            elsewhere = disk_splits_of_stem.get(s.stem)
            if elsewhere:
                moved.append({"stem": s.stem, "expect": s.split, "actual": sorted(elsewhere)})
            else:
                missing.append(s.stem)
        elif _sha256(lbl) != s.sha256:
            changed.append(s.stem)
        img = disk_images.get((s.split, s.stem))
        if img is None:
            missing_images.append(s.stem)
        elif s.image_sha256 and _sha256(img) != s.image_sha256:
            changed_images.append(s.stem)

    details = {"n_samples": len(manifest.samples),
               "n_changed": len(changed), "n_changed_images": len(changed_images),
               "n_moved": len(moved), "n_missing": len(missing),
               "n_missing_images": len(missing_images),
               "changed_preview": changed[:LINEAGE_MAX_DETAIL],
               "changed_images_preview": changed_images[:LINEAGE_MAX_DETAIL],
               "missing_images_preview": missing_images[:LINEAGE_MAX_DETAIL],
               "moved_preview": moved[:LINEAGE_MAX_DETAIL],
               "missing_preview": missing[:LINEAGE_MAX_DETAIL]}
    if changed or changed_images or moved or missing or missing_images:
        return CheckResult(
            _NAME, CheckSeverity.ERROR,
            f"契约核验失败: labels_changed={len(changed)} images_changed={len(changed_images)} "
            f"moved={len(moved)} labels_missing={len(missing)} "
            f"images_missing={len(missing_images)}"
            f" —— 与 D3 产出不一致,重跑 odp-transform 或查谁动了 processed",
            details)
    return CheckResult(_NAME, CheckSeverity.PASS,
                       f"{len(manifest.samples)} 个样本逐条核验通过", details)
