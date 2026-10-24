from pathlib import Path

import pytest
import yaml

from od_platform.common.constants import AnnotationFormat
from od_platform.data_pipeline.source_layout import discover_source_layout


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_discovers_nested_roboflow_layout_and_normalizes_valid(tmp_path):
    root = tmp_path / "archive" / "Brackish.v2.yolov11"
    for split in ("train", "valid", "test"):
        _touch(root / split / "images" / f"{split}.jpg")
        _touch(root / split / "labels" / f"{split}.txt", b"")
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": ["crab", "fish"]}), encoding="utf-8")
    layout = discover_source_layout(tmp_path, AnnotationFormat.YOLO)
    assert layout.root == root
    assert layout.source_split_per_image == {
        "train": "train", "valid": "val", "test": "test"}
    assert layout.classes_hint == ["crab", "fish"]


def test_rejects_duplicate_stem_across_source_splits(tmp_path):
    for split in ("train", "valid"):
        _touch(tmp_path / split / "images" / "same.jpg")
        _touch(tmp_path / split / "labels" / "same.txt", b"")
    with pytest.raises(ValueError, match="stem 冲突"):
        discover_source_layout(tmp_path, AnnotationFormat.YOLO)


def test_rejects_ambiguous_nested_roots(tmp_path):
    for dataset in ("a", "b"):
        _touch(tmp_path / dataset / "images" / "one.jpg")
        _touch(tmp_path / dataset / "labels" / "one.txt", b"")
    with pytest.raises(ValueError, match="多个数据根目录"):
        discover_source_layout(tmp_path, AnnotationFormat.YOLO)


def test_wrong_path_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="期望 images/labels"):
        discover_source_layout(tmp_path, AnnotationFormat.YOLO)


def test_discovers_split_coco_with_separate_image_and_annotation_roots(tmp_path):
    root = tmp_path / "wrapper" / "underwater"
    for split in ("train", "val", "test"):
        _touch(root / "imgs" / split / f"{split}.jpg")
        annotation = root / "annotations" / "coco" / f"instances_{split}.json"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text("{}", encoding="utf-8")
    layout = discover_source_layout(tmp_path, AnnotationFormat.COCO)
    assert layout.root == root
    assert layout.source_split_per_image == {
        "train": "train", "val": "val", "test": "test"}
    assert [p.name for _, p in layout.annotation_dirs] == [
        "instances_train.json", "instances_val.json", "instances_test.json"]
