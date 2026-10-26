# apps/platform/src/od_platform/data_pipeline/orchestrator.py
"""数据流水线编排:检查原料 → 转换 → 扫描 → 报告 → 划分 → 契约 → 物化 → yaml。

编排者不实现任何一步的细节,只做接线与次序;每步的失败语义由子系统自己定。
产物分家:processed/(生产)+ runs/(审计 manifest)+ configs/(yaml 合格证)。
"""
from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path

from od_platform.common import paths, stamp
from od_platform.common.constants import (
    COVERAGE_HARD_THRESHOLD,
    COVERAGE_SOFT_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TRAIN_RATE,
    DEFAULT_VAL_RATE,
    AnnotationFormat,
    SplitStrategy,
    Task,
)
from od_platform.common.lineage import (
    SplitManifest, TOOL_VERSION, compute_data_fingerprint,
)
from od_platform.common.run_context import RunContext
from od_platform.common.progress import ProgressCallback
from od_platform.data_pipeline.convert.registry import ConvertOptions
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.report import (
    analyze_balance,
    format_report,
    warn_if_rare,
)
from od_platform.data_pipeline.source_layout import SourceLayout, discover_source_layout
from od_platform.data_pipeline.split.materializer import (
    SplitOutputDirs,
    SplitSourceDirs,
    materialize,
)
from od_platform.data_pipeline.split.split_service import split_dataset
from od_platform.data_pipeline.split.yaml_writer import write_dataset_yaml

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


class DatasetPipeline:
    """一条命令的完整实现:odp-transform 的全部逻辑。"""

    def __init__(
        self,
        dataset: str,
        annotation_format: str,
        *,
        task: str = Task.DETECT,
        train_rate: float = DEFAULT_TRAIN_RATE,
        val_rate: float = DEFAULT_VAL_RATE,
        classes: list[str] | None = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        split_strategy: str = DEFAULT_SPLIT_STRATEGY,
        run: RunContext | None = None,
        progress: ProgressCallback | None = None,
    ):
        self.dataset = dataset
        self.annotation_format = annotation_format
        self.task = task
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.classes = classes
        self.random_state = random_state
        self.split_strategy = split_strategy
        self._run_ctx = run        # 存成私有名:实例属性不能遮蔽 run() 方法
        self.progress = progress
        self.raw_root = paths.raw_dataset_root(dataset)
        self._layout: SourceLayout | None = None

    # ---- 前置检查 ----

    def _check_raw(self) -> None:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"数据集不存在: {self.raw_root}")
        self._layout = discover_source_layout(self.raw_root, self.annotation_format)
        if self.split_strategy == SplitStrategy.PRESERVE_SOURCE and \
                self._layout.source_split_per_image is None:
            raise ValueError("preserve_source 需要源数据包含 train/val/test 预划分目录")
        self._check_coverage()

    def _check_coverage(self) -> None:
        """覆盖率闸门:标注数/图像数 过低时直接拒跑。"""
        assert self._layout is not None
        n_images = len(self._layout.image_index)
        if n_images == 0:
            raise FileNotFoundError(f"图像目录为空: {self._layout.root}")
        if self.annotation_format == AnnotationFormat.COCO:
            logger.debug("COCO 为单 json 格式,跳过 stem 级覆盖率检查")
            return
        n_annos = sum(1 if source.is_file() else len(list(source.glob("*.*")))
                      for _split, source in self._layout.annotation_dirs)
        coverage = n_annos / n_images
        if coverage < COVERAGE_HARD_THRESHOLD:
            raise ValueError(
                f"图像-标注覆盖率 {coverage:.1%} 低于硬阈值 "
                f"{COVERAGE_HARD_THRESHOLD:.0%}(图 {n_images} / 标注 {n_annos})。"
                f"疑似标注导出中断或路径配错 —— 请修源头,不要带病训练。")
        if coverage < COVERAGE_SOFT_THRESHOLD:
            logger.warning(
                "图像-标注覆盖率 %.1f%% 低于软阈值 %.0f%%,继续但请确认是有意为之",
                coverage * 100, COVERAGE_SOFT_THRESHOLD * 100)

    # ---- 扫描:转换产物 → 内存结构 ----

    def _scan(self, staging: Path, classes: list[str]):
        assert self._layout is not None
        image_index = self._layout.image_index
        labels_per_image: dict[str, list[str]] = defaultdict(list)
        label_bytes: dict[str, bytes] = {}
        label_files = sorted(staging.glob("*.txt"))
        total = len(label_files)
        if self.progress:
            self.progress("D3 扫描转换结果", 0, total)
        for current, txt in enumerate(label_files, 1):
            label_bytes[txt.stem] = txt.read_bytes()
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    idx = int(line.split()[0])
                    labels_per_image[txt.stem].append(classes[idx])
            if self.progress:
                self.progress("D3 扫描转换结果", current, total)
        return image_index, labels_per_image, label_bytes

    # ---- 主流程 ----

    def run(self) -> dict[str, object]:
        if self._run_ctx is not None:
            return self._execute(self._run_ctx)
        with RunContext("data_pipeline") as run:
            return self._execute(run)

    def _execute(self, run: RunContext) -> dict[str, object]:
        self._check_raw()
        if self.progress:
            self.progress("D3 发现数据目录", 1, 1)
        out = SplitOutputDirs.for_dataset(self.dataset)

        with tempfile.TemporaryDirectory(prefix="odp_pipe_") as staging_dir:
            staging = Path(staging_dir)
            assert self._layout is not None
            requested_classes = self.classes or self._layout.classes_hint
            classes: list[str] | None = None
            for _source_split, ann_in in self._layout.annotation_dirs:
                options = ConvertOptions(
                    task=self.task,
                    classes=requested_classes if classes is None else classes,
                    progress=self.progress,
                )
                converted_classes = convert_data_to_yolo(
                    ann_in, staging, self.annotation_format, options)
                if classes is None:
                    classes = converted_classes
                elif converted_classes != classes:
                    raise ValueError(f"不同源 split 的类别顺序不一致: {ann_in}")
            assert classes is not None

            image_index, labels_per_image, label_bytes = self._scan(staging, classes)

            missing_labels = sorted(set(image_index) - set(label_bytes))
            missing_images = sorted(set(label_bytes) - set(image_index))
            if missing_labels or missing_images:
                raise ValueError(
                    "图像与标注无法一一配对: "
                    f"缺标签={len(missing_labels)} {missing_labels[:20]}, "
                    f"缺图片={len(missing_images)} {missing_images[:20]}。"
                    "合法负样本必须提供空标签（YOLO）或在 COCO images 中声明。")

            report = analyze_balance(
                dict(labels_per_image), classes,
                min_split_rate=min(self.train_rate, self.val_rate,
                                   1 - self.train_rate - self.val_rate))
            for line in format_report(report):
                logger.info(line)
            warn_if_rare(report)

            stems = sorted(image_index)
            splits = split_dataset(
                stems, self.train_rate, self.val_rate, self.random_state,
                strategy=self.split_strategy,
                labels_per_image={s: labels_per_image.get(s, []) for s in stems},
                group_per_image=self._layout.source_split_per_image)

            if self.split_strategy == SplitStrategy.PRESERVE_SOURCE:
                total = len(stems)
                ratios = tuple(round(len(splits[s]) / total, 6) for s in _SPLITS)
            else:
                ratios = (self.train_rate, self.val_rate,
                          round(1 - self.train_rate - self.val_rate, 6))

            manifest = SplitManifest.build(
                dataset=self.dataset, strategy=self.split_strategy,
                seed=self.random_state,
                ratios=ratios,
                names=classes, splits=splits, label_bytes=label_bytes,
                image_paths=image_index,
                created_at=run.created_at,
                progress=self.progress)
            manifest_path = run.artifact_path("manifest.json")
            manifest.write(manifest_path)
            manifest_ref = manifest_path.relative_to(paths.ROOT_DIR).as_posix()

            source = SplitSourceDirs(images=self._layout.root / "images", labels=staging,
                                     image_index=image_index)
            counts = materialize(manifest, source, out, progress=self.progress)

            yaml_path = paths.dataset_yaml_path(self.dataset)
            write_dataset_yaml(
                yaml_path, dataset_root=out.root, classes=classes, manifest=manifest,
                dataset_name=self.dataset, source_format=self.annotation_format,
                task=self.task, manifest_ref=manifest_ref)
            if self.progress:
                self.progress("D3 写入契约与 YAML", 1, 1)

        logger.info("数据集 %r 完成: %s", self.dataset, counts)

        # ★ 盖 transform 章: 我(D3)划分了这批数据。
        #   data_fingerprint 从【我刚建好的 samples】算 —— 一手证据, 不抄任何人。
        data_fp = compute_data_fingerprint(manifest.samples)
        stamp.write_stamp(
            stage_file=stamp.STAMP_TRANSFORM,
            data_fingerprint=data_fp,
            run_id=run.run_id,
            tool_version=TOOL_VERSION,          # odp-transform/0.4
            payload={
                "contract_fingerprint": manifest.contract_fingerprint,
                "manifest_ref": manifest_ref,
                "n_samples": len(manifest.samples),
                "created_at": run.created_at,
            },
        )

        return {
            "counts": counts,
            "yaml": str(yaml_path),
            "manifest_path": str(manifest_path),
            "run_dir": str(run.run_dir),
            "contract_fingerprint": manifest.contract_fingerprint,
            "data_fingerprint": data_fp,
        }
