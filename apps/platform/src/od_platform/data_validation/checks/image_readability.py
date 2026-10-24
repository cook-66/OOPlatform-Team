"""check: image_readability —— 图片必须能被解码器识别且结构完整。"""
from PIL import Image, UnidentifiedImageError

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check

_NAME = "image_readability"


@check(_NAME)
def validate_image_readability(ctx: CheckContext) -> CheckResult:
    broken: list[dict[str, str]] = []
    total = 0
    for split in ctx.snapshot.splits:
        for path in ctx.snapshot.images_per_split[split]:
            total += 1
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                broken.append({"split": split, "path": str(path),
                               "error": f"{type(exc).__name__}: {exc}"})
    details = {"total_images": total, "broken": len(broken),
               "broken_preview": broken[:DETAILS_PREVIEW_LIMIT]}
    if broken:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"{len(broken)}/{total} 张图片无法读取", details)
    return CheckResult(_NAME, CheckSeverity.PASS, f"全部 {total} 张图片可读取", details)
