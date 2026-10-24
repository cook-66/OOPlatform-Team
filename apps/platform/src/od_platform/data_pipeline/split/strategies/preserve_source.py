"""保留源数据集已经声明的 train/val/test 划分。"""
from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy


@register_strategy(SplitStrategy.PRESERVE_SOURCE, requires_labels=False)
def preserve_source(stems: list[str], options: SplitOptions) -> dict[str, list[str]]:
    source = options.group_per_image
    if source is None:
        raise ValueError("划分策略 'preserve_source' 只适用于带 train/val/test 源划分的数据集")
    missing = sorted(set(stems) - set(source))
    if missing:
        raise ValueError(f"{len(missing)} 个样本缺少源 split 信息: {missing[:20]}")
    result = {name: [] for name in ("train", "val", "test")}
    for stem in stems:
        split = source[stem]
        if split not in result:
            raise ValueError(f"样本 {stem!r} 的源 split 非法: {split!r}")
        result[split].append(stem)
    if not result["train"] or not result["val"]:
        raise ValueError("源划分必须至少包含非空 train 和 val")
    return {name: sorted(values) for name, values in result.items()}
