"""识别原始数据目录，兼容压缩包外层目录与 Roboflow 预划分布局。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from od_platform.common.constants import IMAGE_EXTENSIONS, AnnotationFormat

_SOURCE_SPLITS = {"train": "train", "val": "val", "valid": "val", "test": "test"}


@dataclass(frozen=True)
class SourceLayout:
    root: Path
    image_index: dict[str, Path]
    annotation_dirs: tuple[tuple[str | None, Path], ...]
    source_split_per_image: dict[str, str] | None
    classes_hint: list[str] | None = None


def _images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _index_images(items: list[tuple[str | None, Path]]) -> tuple[dict[str, Path], dict[str, str] | None]:
    index: dict[str, Path] = {}
    source_splits: dict[str, str] = {}
    for split, directory in items:
        for image in _images(directory):
            if image.stem in index:
                raise ValueError(
                    f"图片 stem 冲突: {image.stem!r}: {index[image.stem]} 与 {image}。"
                    "请在转换前重命名，避免图像/标签错误配对。")
            index[image.stem] = image
            if split is not None:
                source_splits[image.stem] = split
    if not index:
        raise FileNotFoundError("识别到 images 目录，但其中没有受支持的图片")
    return index, source_splits or None


def _index_coco_shared_images(
    image_dir: Path, annotation_files: list[tuple[str, Path]],
) -> tuple[dict[str, Path], dict[str, str]]:
    """索引共用 images/、按 JSON 保存 split 的 COCO 导出。"""
    all_images = _images(image_dir)
    by_relative = {
        image.relative_to(image_dir).as_posix(): image for image in all_images
    }
    by_name: dict[str, list[Path]] = {}
    for image in all_images:
        by_name.setdefault(image.name, []).append(image)

    index: dict[str, Path] = {}
    source_splits: dict[str, str] = {}
    referenced: set[Path] = set()
    for split, annotation_file in annotation_files:
        document = json.loads(annotation_file.read_text(encoding="utf-8"))
        for item in document.get("images", []):
            file_name = Path(str(item["file_name"])).as_posix()
            while file_name.startswith("./"):
                file_name = file_name[2:]
            if not file_name or file_name.startswith("../") or Path(file_name).is_absolute():
                raise ValueError(
                    f"COCO 标注包含不安全的图片路径: {file_name!r} "
                    f"(标注: {annotation_file})")
            image = by_relative.get(file_name)
            if image is None:
                matches = by_name.get(Path(file_name).name, [])
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"COCO 标注引用的图片无法唯一定位: {file_name!r} "
                        f"(标注: {annotation_file})")
                image = matches[0]
            if image.stem in index:
                raise ValueError(
                    f"COCO split 间图片 stem 冲突或重复引用: {image.stem!r}: "
                    f"{index[image.stem]} 与 {image}")
            index[image.stem] = image
            source_splits[image.stem] = split
            referenced.add(image)

    unreferenced = sorted(set(all_images) - referenced)
    if unreferenced:
        raise ValueError(
            f"共用 images 目录中有 {len(unreferenced)} 张图片未被任何 split JSON 引用: "
            f"{[str(p) for p in unreferenced[:20]]}")
    if not index:
        raise FileNotFoundError("识别到 COCO images 目录，但 split JSON 没有声明图片")
    return index, source_splits


def _classes_hint(root: Path) -> list[str] | None:
    for name in ("data.yaml", "dataset.yaml"):
        path = root / name
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = doc.get("names")
        if isinstance(names, list):
            return [str(x) for x in names]
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return None


def discover_source_layout(raw_root: Path, annotation_format: str) -> SourceLayout:
    """返回唯一可用布局；多候选时拒绝猜测。"""
    ann_name = "labels" if annotation_format == AnnotationFormat.YOLO else "annotations"

    if annotation_format == AnnotationFormat.COCO:
        coco_candidates: list[tuple[Path, list[tuple[str, Path]], list[tuple[str, Path]]]] = []
        for candidate in [raw_root, *(p for p in raw_root.rglob("*") if p.is_dir())]:
            # TrashCan/Detectron 风格: train|val|test/ + instances_<split>*.json
            image_dirs = []
            anno_files = []
            for disk_name, normalized in _SOURCE_SPLITS.items():
                image_dir = candidate / disk_name
                matches = sorted(candidate.glob(f"instances_{disk_name}*.json"))
                if image_dir.is_dir() and len(matches) == 1:
                    image_dirs.append((normalized, image_dir))
                    anno_files.append((normalized, matches[0]))
            if {s for s, _ in image_dirs} >= {"train", "val"}:
                coco_candidates.append((candidate, image_dirs, anno_files))

            # 常见导出: imgs/{train,val,test} + annotations[/coco]/instances_*.json
            for image_base_name in ("images", "imgs"):
                image_base = candidate / image_base_name
                for anno_base in (candidate / "annotations", candidate / "annotations" / "coco"):
                    image_dirs: list[tuple[str, Path]] = []
                    anno_files: list[tuple[str, Path]] = []
                    for disk_name, normalized in _SOURCE_SPLITS.items():
                        image_dir = image_base / disk_name
                        matches = (sorted(anno_base.glob(f"*{disk_name}*.json"))
                                   if anno_base.is_dir() else [])
                        if image_dir.is_dir() and len(matches) == 1:
                            image_dirs.append((normalized, image_dir))
                            anno_files.append((normalized, matches[0]))
                    if {s for s, _ in image_dirs} >= {"train", "val"}:
                        coco_candidates.append((candidate, image_dirs, anno_files))

            # JODD 类导出:coco_train_data/images + coco.json
            image_dirs = []
            anno_files = []
            for disk_name, normalized in _SOURCE_SPLITS.items():
                split_root = candidate / f"coco_{disk_name}_data"
                image_dir = split_root / "images"
                anno_file = split_root / "coco.json"
                if image_dir.is_dir() and anno_file.is_file():
                    image_dirs.append((normalized, image_dir))
                    anno_files.append((normalized, anno_file))
            if {s for s, _ in image_dirs} >= {"train", "val"}:
                coco_candidates.append((candidate, image_dirs, anno_files))

            # COU 类导出:共用 images/ + train|val|test_annotations.json
            image_dir = candidate / "images"
            anno_files = []
            for disk_name, normalized in _SOURCE_SPLITS.items():
                annotation_file = candidate / f"{disk_name}_annotations.json"
                if annotation_file.is_file():
                    anno_files.append((normalized, annotation_file))
            if image_dir.is_dir() and {s for s, _ in anno_files} >= {"train", "val"}:
                coco_candidates.append((candidate, [("shared", image_dir)], anno_files))

        unique = {str(c[0].resolve()): c for c in coco_candidates}
        coco_candidates = list(unique.values())
        if len(coco_candidates) > 1:
            raise ValueError(f"发现多个 COCO 预划分数据根目录，无法安全选择: "
                             f"{[str(c[0]) for c in coco_candidates]}")
        if coco_candidates:
            root, image_dirs, anno_files = coco_candidates[0]
            if len(image_dirs) == 1 and image_dirs[0][0] == "shared":
                index, source = _index_coco_shared_images(image_dirs[0][1], anno_files)
            else:
                index, source = _index_images(image_dirs)
            return SourceLayout(root, index, tuple(anno_files), source, None)

    presplit: list[tuple[Path, list[tuple[str, Path]], list[tuple[str, Path]]]] = []
    for candidate in [raw_root, *(p for p in raw_root.rglob("*") if p.is_dir())]:
        image_dirs: list[tuple[str, Path]] = []
        ann_dirs: list[tuple[str, Path]] = []
        for disk_name, normalized in _SOURCE_SPLITS.items():
            image_dir = candidate / disk_name / "images"
            anno_dir = candidate / disk_name / ann_name
            if image_dir.is_dir() and anno_dir.is_dir():
                image_dirs.append((normalized, image_dir))
                ann_dirs.append((normalized, anno_dir))
        if {s for s, _ in image_dirs} >= {"train", "val"}:
            presplit.append((candidate, image_dirs, ann_dirs))
    # 只保留最外层唯一候选，避免其子目录被重复识别。
    presplit = [c for c in presplit if not any(c[0].is_relative_to(o[0]) and c[0] != o[0]
                                                for o in presplit)]
    if len(presplit) > 1:
        raise ValueError(f"发现多个预划分数据根目录，无法安全选择: {[str(c[0]) for c in presplit]}")
    if presplit:
        root, image_dirs, ann_dirs = presplit[0]
        index, source = _index_images(image_dirs)
        return SourceLayout(root, index, tuple(ann_dirs), source, _classes_hint(root))

    flat: list[tuple[Path, Path, Path]] = []
    for image_dir in [p for p in raw_root.rglob("images") if p.is_dir()]:
        anno_dir = image_dir.parent / ann_name
        if anno_dir.is_dir():
            flat.append((image_dir.parent, image_dir, anno_dir))
    if not flat:
        raise FileNotFoundError(
            f"未找到受支持的原始布局。期望 images/{ann_name}，或 "
            f"train|valid|val|test/images/{ann_name}；搜索根目录: {raw_root}")
    if len(flat) > 1:
        raise ValueError(f"发现多个数据根目录，无法安全选择: {[str(x[0]) for x in flat]}")
    root, image_dir, anno_dir = flat[0]
    index, _ = _index_images([(None, image_dir)])
    return SourceLayout(root, index, ((None, anno_dir),), None, _classes_hint(root))
