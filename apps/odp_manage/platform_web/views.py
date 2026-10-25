import json
import logging
import re
import sys
import threading
import time
from pathlib import Path

from django.conf import settings
from django.db.models import Sum
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .models import InferenceRecord

logger = logging.getLogger(__name__)

# 把同仓库的 src 布局加进 sys.path，Django 才能 import od_platform
_PLATFORM_SRC = Path(__file__).resolve().parents[2] / "platform" / "src"
if _PLATFORM_SRC.is_dir() and str(_PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_SRC))

try:
    import cv2
    from ultralytics import YOLO
except Exception:  # noqa: BLE001 - 缺依赖时只降级，不让 Django 启动崩溃
    cv2 = None
    YOLO = None
    logger.exception("ultralytics/opencv 导入失败，推理接口暂不可用")

try:
    from od_platform.assistant.llm_service import analyze_detection
except Exception:  # noqa: BLE001
    analyze_detection = None
    logger.exception("od_platform.assistant.llm_service 导入失败，解释接口暂不可用")

# 视频扩展名兜底（有些浏览器上传的 content_type 不可靠）
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}

# 推理置信度阈值：低于该值的候选框直接丢弃，避免把海星认成 small_fish 这类瞎猜
INFER_CONF = 0.6


def _is_video(upload) -> bool:
    content_type = (getattr(upload, "content_type", "") or "").lower()
    return content_type.startswith("video/") or Path(upload.name).suffix.lower() in _VIDEO_SUFFIXES


# 全局单例模型：进程内只加载一次，避免每次请求都重新加载 best.pt（这是耗时大头）
_yolo_model = None
_MODEL_LOCK = threading.Lock()
# ultralytics 模型实例不保证线程安全，预测阶段串行执行
_PREDICT_LOCK = threading.Lock()


def get_yolo_model():
    """懒加载全局 YOLO 模型；已加载则直接复用。"""
    global _yolo_model
    if _yolo_model is None:
        with _MODEL_LOCK:
            if _yolo_model is None:
                model_path = Path(settings.BASE_DIR).parents[1] / "models" / "pretrained" / "best.pt"
                logger.info("首次加载模型: %s", model_path)
                _yolo_model = YOLO(str(model_path))
                logger.info("模型加载完成，后续请求复用该实例")
    return _yolo_model


def warmup_yolo_model() -> None:
    """预热全局模型：空图跑一次，把首次推理的初始化开销挪到服务启动阶段。"""
    try:
        import numpy as np
        model = get_yolo_model()
        with _PREDICT_LOCK:
            model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), conf=INFER_CONF, verbose=False)
        logger.info("模型预热完成")
    except Exception:  # noqa: BLE001 - 预热失败不影响正常推理
        logger.exception("模型预热失败（不影响正常推理）")


def _count_classes(boxes, names) -> dict[str, int]:
    counts: dict[str, int] = {}
    if boxes is None or len(boxes) == 0:
        return counts
    for cls_id in boxes.cls.int().cpu().tolist():
        name = names[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1
    return counts


def _infer_image(model, src: Path, dest: Path):
    """单图推理：返回 (各类别计数, 最高置信度, 帧数)。"""
    with _PREDICT_LOCK:
        result = model.predict(source=str(src), conf=INFER_CONF, verbose=False)[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), result.plot())
    counts = _count_classes(result.boxes, model.names)
    has_box = result.boxes is not None and len(result.boxes) > 0
    max_conf = float(result.boxes.conf.max()) if has_box else 0.0
    return counts, max_conf, 1


def _infer_video(model, src: Path, dest: Path):
    """逐帧推理并合成标注视频：返回 (各类别计数, 最高置信度, 帧数)。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("视频无法打开，请检查编码格式")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    counts: dict[str, int] = {}
    frames = 0
    max_conf = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with _PREDICT_LOCK:
                result = model.predict(source=frame, conf=INFER_CONF, verbose=False)[0]
            annotated = result.plot()
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)
            for name, count in _count_classes(result.boxes, model.names).items():
                counts[name] = counts.get(name, 0) + count
            if result.boxes is not None and len(result.boxes) > 0:
                max_conf = max(max_conf, float(result.boxes.conf.max()))
            frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        raise RuntimeError("视频中没有可读取的帧")
    return counts, max_conf, frames


def index(request):
    """渲染核心控制台页面"""
    return render(request, "index.html")


@require_POST
def api_run_inference(request):
    """接收图片/视频，调用真实 YOLO 模型推理，结果落库并返回访问地址。"""
    upload = request.FILES.get("image") or request.FILES.get("video")
    if not upload:
        return JsonResponse({"status": "error", "message": "未接收到文件"}, status=400)

    if cv2 is None or YOLO is None:
        return JsonResponse(
            {"status": "error", "message": "推理模块未就绪，请检查服务端日志（是否缺少 ultralytics/opencv）"},
            status=503,
        )

    is_video = _is_video(upload)
    # ImageField 不校验内容，视频同样可以沿用这个字段保存原始文件
    record = InferenceRecord.objects.create(original_img=upload)

    suffix = ".mp4" if is_video else ".jpg"
    rel_path = f"results/{record.pk}_result{suffix}"
    dest = Path(settings.MEDIA_ROOT) / rel_path

    started = time.perf_counter()
    try:
        model = get_yolo_model()
        if is_video:
            per_class, max_conf, frames = _infer_video(model, Path(record.original_img.path), dest)
        else:
            per_class, max_conf, frames = _infer_image(model, Path(record.original_img.path), dest)
    except Exception as exc:  # noqa: BLE001
        logger.exception("推理调用异常")
        record.delete()
        return JsonResponse({"status": "error", "message": f"推理失败: {exc}"}, status=500)

    infer_time = time.perf_counter() - started
    record.result_img.name = rel_path
    record.detected_classes = ", ".join(per_class)
    record.confidence = round(max_conf, 4)
    record.save(update_fields=["result_img", "detected_classes", "confidence"])

    payload = {
        "status": "success",
        "record_id": record.pk,
        "media_type": "video" if is_video else "image",
        "classes": ", ".join(f"{name}:{count}" for name, count in per_class.items()) or "未检测到目标",
        "class_counts": per_class,
        "confidence": round(max_conf, 4),
        "frames": frames,
        "infer_time": round(infer_time, 2),
    }
    payload["result_video_url" if is_video else "result_img_url"] = record.result_img.url
    return JsonResponse(payload)


_CLASS_COUNT_RE = re.compile(r"^(\d+)\s*个?\s*(.+)$")
_CLASS_COLON_RE = re.compile(r"^(.+?)\s*[:：]\s*(\d+)$")


def _parse_class_counts(classes, class_counts) -> dict[str, int]:
    """把前端传来的类别信息统一成 {类名: 数量}。

    支持三种形状: dict、``crab:3, fish:2``、``crab, fish``（无数量按 1 计）。
    """
    if isinstance(class_counts, dict) and class_counts:
        return {str(name): int(count) for name, count in class_counts.items()}

    if isinstance(classes, dict):
        return {str(name): int(count) for name, count in classes.items()}

    counts: dict[str, int] = {}
    if isinstance(classes, str):
        for token in re.split(r"[,，;；\n]", classes):
            token = token.strip()
            if not token:
                continue
            matched = _CLASS_COLON_RE.match(token) or _CLASS_COUNT_RE.match(token)
            if matched:
                name, number = (matched.group(1), matched.group(2))
                if _CLASS_COUNT_RE.match(token):
                    name, number = number, name
                counts[name.strip()] = counts.get(name.strip(), 0) + int(number)
            else:
                counts[token] = counts.get(token, 0) + 1
    return counts


@require_POST
def api_generate_explanation(request):
    """接收检测类别，调用大模型生成分析报告并写回对应记录。"""
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "请求体不是合法 JSON"}, status=400)

    if analyze_detection is None:
        return JsonResponse(
            {"status": "error", "message": "大模型模块未就绪，请检查 openai 依赖与服务端日志"},
            status=503,
        )

    counts = _parse_class_counts(payload.get("classes", ""), payload.get("class_counts"))
    if not counts:
        return JsonResponse({"status": "error", "message": "未收到有效的检测类别"}, status=400)

    # 写库：优先按 record_id 精确更新，只有没传 record_id 时才回退到最近一条记录
    record = None
    record_id = payload.get("record_id")
    if record_id:
        try:
            record = InferenceRecord.objects.get(id=record_id)
        except InferenceRecord.DoesNotExist:
            logger.warning("record_id=%s 不存在，解释未写入数据库", record_id)
        except (TypeError, ValueError):
            logger.warning("record_id=%r 非法，解释未写入数据库", record_id)
    else:
        record = InferenceRecord.objects.order_by("-created_at").first()

    try:
        llm_result = analyze_detection(counts, conf_threshold=INFER_CONF)
    except Exception as exc:  # noqa: BLE001 - 大模型报错不能拖垮 Django
        logger.exception("大模型调用失败")
        message = f"大模型调用失败，请检查 LLM_API_KEY 或网络：{exc}"
        return JsonResponse({"status": "error", "message": message, "explanation": message}, status=502)

    explanation = llm_result.get("text", "")
    cost = float(llm_result.get("cost", 0.0))
    tokens = int(llm_result.get("tokens", 0))

    if record is not None:
        record.llm_explanation = explanation
        record.llm_cost = cost
        record.llm_tokens = tokens
        record.save(update_fields=["llm_explanation", "llm_cost", "llm_tokens"])

    return JsonResponse({
        "status": "success",
        "record_id": record.pk if record is not None else None,
        "explanation": explanation,
        "cost": round(cost, 6),
        "tokens": tokens,
    })


def api_today_cost(request):
    """返回今天累计的大模型花费，供前端展示“今日总花费”。"""
    today = timezone.localdate()
    total = (
        InferenceRecord.objects.filter(created_at__date=today)
        .aggregate(total=Sum("llm_cost"))
        .get("total")
    ) or 0.0
    count = InferenceRecord.objects.filter(created_at__date=today, llm_cost__gt=0).count()
    return JsonResponse({
        "status": "success",
        "date": today.isoformat(),
        "total_cost": round(float(total), 6),
        "paid_records": count,
    })


@require_GET
def api_record_detail(request, record_id):
    """返回单条推理记录的全部字段，供前端详情弹窗使用。"""
    record = InferenceRecord.objects.filter(pk=record_id).first()
    if record is None:
        return JsonResponse({"status": "error", "message": "记录不存在"}, status=404)
    return JsonResponse({
        "status": "success",
        "id": record.pk,
        "created_at": timezone.localtime(record.created_at).strftime("%Y-%m-%d %H:%M:%S"),
        "original_img": record.original_img.url if record.original_img else "",
        "result_img": record.result_img.url if record.result_img else "",
        "detected_classes": record.detected_classes or "—",
        "confidence": round(record.confidence, 4),
        "llm_explanation": record.llm_explanation,
        "llm_cost": round(record.llm_cost, 6),
        "total_tokens": record.llm_tokens,
    })


@require_GET
def api_history(request):
    """返回最近的推理记录 + 今日花费/Token 汇总，供“推理记录 / 花费账单”面板使用。"""
    try:
        limit = int(request.GET.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))

    rows = [{
        "id": r.pk,
        "time": timezone.localtime(r.created_at).strftime("%Y-%m-%d %H:%M:%S"),
        "classes": r.detected_classes or "—",
        "confidence": round(r.confidence, 4),
        "cost": round(r.llm_cost, 6),
        "tokens": r.llm_tokens,
    } for r in InferenceRecord.objects.order_by("-created_at")[:limit]]

    today = timezone.localdate()
    today_agg = InferenceRecord.objects.filter(created_at__date=today).aggregate(
        cost=Sum("llm_cost"), tokens=Sum("llm_tokens"),
    )
    return JsonResponse({
        "status": "success",
        "records": rows,
        "today": {
            "date": today.isoformat(),
            "total_cost": round(float(today_agg["cost"] or 0.0), 6),
            "total_tokens": int(today_agg["tokens"] or 0),
        },
    })
