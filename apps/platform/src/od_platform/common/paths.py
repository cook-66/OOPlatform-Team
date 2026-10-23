#!/usr/bin/env python
# @FileName  : paths.py
# @Author    : ODPlatform Team
# @Project   : ODPlatform
# @Function  : 集中定义项目路径常量
#              - 使用 marker file 模式定位 ROOT_DIR(workspace 根)
#              - 引入 APP_DIR 概念,把【共享资产】和【端私有资产】分开

from pathlib import Path

# ============================================================
# Workspace 根目录定位 (marker file 模式)
# ============================================================
WORKSPACE_MARKER: str = ".odp-workspace"


def _find_workspace_root(
    start: Path,
    markers: tuple[str, ...] = (WORKSPACE_MARKER,),
) -> Path:
    """
    从 start 开始沿父目录向上查找,返回包含任一 marker 文件的目录。
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        for marker in markers:
            if (parent / marker).exists():
                return parent

    raise FileNotFoundError(
        f"找不到 workspace marker 文件 ({markers})。"
        f"请确认仓库根存在 {WORKSPACE_MARKER} 文件。"
    )


# 计算 ROOT_DIR(模块加载时执行一次)
ROOT_DIR: Path = _find_workspace_root(Path(__file__))


# ============================================================
# 端根目录 APP_DIR (platform 这一个端的根)
# ============================================================
APP_DIR: Path = ROOT_DIR / "apps" / "platform"


# ============================================================
# 【共享资产】(在 ROOT_DIR 下,所有端可访问)
# ============================================================
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RUNS_DIR: Path = ROOT_DIR / "runs"

PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"   # D7: 训练中间检查点 / 外部权重

RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"


# ============================================================
# 【端私有资产】(在 APP_DIR 下,只属于 platform 这个端)
# ============================================================
CONFIGS_DIR: Path = APP_DIR / "configs"
LOGGING_DIR: Path = APP_DIR / "logging"
UNIT_TEST_DIR: Path = APP_DIR / "tests"


# ============================================================
# 【顶层文档目录】(共享给所有人)
# ============================================================
DOCS_DIR: Path = ROOT_DIR / "docs"


# ============================================================
# 【工程基础设施目录】(共享)
# ============================================================
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"


# ============================================================
# 【工具元数据目录】(audit trail / 工具自身日志)  <-- 2.3 新增
# ============================================================
META_DIR: Path = ROOT_DIR / ".odp-meta"
META_LOGGING_DIR: Path = META_DIR / "logs"


# ============================================================
# 对外暴露的"要初始化的目录列表" (init 清单)
# ============================================================
def get_dirs_to_initialize() -> list[Path]:
    """
    返回项目启动时需要确保存在的所有目录列表。
    """
    return [
        # 共享资产
        DATA_DIR,
        RUNS_DIR,
        MODELS_DIR,
        PRETRAINED_MODELS_DIR,
        TRAINED_MODELS_DIR,
        CHECKPOINTS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        # 端私有资产
        CONFIGS_DIR,
        LOGGING_DIR,
        UNIT_TEST_DIR,
        # 工程基础设施
        SCRIPTS_DIR,
        DOCS_DIR,
        # 工具元数据 (2.3 新增)
        META_LOGGING_DIR,
        DATASET_CONFIGS_DIR,
        RUNTIME_CONFIGS_DIR
    ]


# ============================================================
# reset 专用清单 (白名单)  <-- 2.1 新增
# ============================================================
def get_dirs_to_reset() -> list[Path]:
    """
    返回 reset_project 可以安全清理的目录列表。
    """
    return [
        PROCESSED_DATA_DIR,
        RUNS_DIR,
        TRAINED_MODELS_DIR,
        LOGGING_DIR,
    ]


# ============================================================
# 【绝对保护目录】reset 工具永远不能删除这些 (黑名单兜底)  <-- 2.2 新增
# ============================================================
PROTECTED_DIRS: tuple[Path, ...] = (
    ROOT_DIR,
    ROOT_DIR / "apps",
    ROOT_DIR / "packages",
    APP_DIR,
    APP_DIR / "src",
    SCRIPTS_DIR,
    DOCS_DIR,
    UNIT_TEST_DIR,
    CONFIGS_DIR,
    ROOT_DIR / ".git",
    ROOT_DIR / ".odp-workspace",
    # 工具元数据 (2.3 新增)
    META_DIR,
    META_LOGGING_DIR,
)


def is_protected(path: Path) -> bool:
    """
    路径是否受保护(任一满足即受保护)。
    """
    path = path.resolve(strict=False)
    for protected in PROTECTED_DIRS:
        protected_resolved = protected.resolve(strict=False)
        if path == protected_resolved:
            return True
        if protected_resolved.is_relative_to(path):
            return True
    return False

# apps/platform/src/od_platform/common/paths.py(增量:追加到文件尾部)

DATASET_CONFIGS_DIR: Path = CONFIGS_DIR / "datasets"   # 数据集 yaml 的家(Ultralytics 约定)

def raw_dataset_root(name: str) -> Path:
    return RAW_DATA_DIR / name

def dataset_processed_dir(name: str) -> Path:
    return PROCESSED_DATA_DIR / name

def dataset_yaml_path(name: str) -> Path:
    return DATASET_CONFIGS_DIR / f"{name}.yaml"

# 别忘了:把 DATASET_CONFIGS_DIR 加进 get_dirs_to_initialize() 的返回元组


# apps/platform/src/od_platform/common/paths.py(D4 增量:追加到文件尾部)

VALIDATION_RUNS_DIR: Path = RUNS_DIR / "data_validation"   # 质检报告的家(按需创建,不进 init 清单)


def validation_run_dir(run_id: str) -> Path:
    return VALIDATION_RUNS_DIR / run_id

# ============================================================
# D5 增量: 运行配置目录 (跟 D3 立的 DATASET_CONFIGS_DIR 语义完全不同)
#   DATASET_CONFIGS_DIR: 描述【数据】的 yaml (odp-transform 产出)
#   RUNTIME_CONFIGS_DIR : 描述【跑法】的 yaml (runtime_config 子系统产出)
# 两类 yaml 物理分目录, 语义不会互相污染.
# ============================================================
RUNTIME_CONFIGS_DIR: Path = CONFIGS_DIR / "runtime"


def runtime_config_path(name: str) -> Path:
    """返回某个运行配置文件的路径: <CONFIGS_DIR>/runtime/<name>.yaml

    Args:
        name: 配置名 (如 "train" / "val" / "infer"), 不带 .yaml 后缀

    Returns:
        Path 对象 (尚未创建, 调用方自己负责 mkdir)

    用法:
        train_yaml = runtime_config_path("train")
        # → <APP_DIR>/configs/runtime/train.yaml
    """
    return RUNTIME_CONFIGS_DIR / f"{name}.yaml"