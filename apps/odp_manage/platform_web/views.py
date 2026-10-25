import json
import logging
import re
import sys
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .models import InferenceRecord

logger = logging.getLogger(__name__)

# apps/platform 是 src 布局，加入 sys.path 后才能 import od_platform
_PLATFORM_SRC = Path(__file__).resolve().parents[2] / "platform" / "src"
if str(_PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_SRC))

# 顶部尝试导入真实大模型函数；失败（缺 openai / 依赖未装）也不让 Django 启动崩溃
try:
    from od_platform.assistant.llm_service import generate_explanation
except Exception:
    generate_explanation = None
    logger.exception("导入 od_platform.assistant.llm_service 失败，智能解释功能不可用")


def index(request):
    """渲染核心控制台页面"""
    return render(request, "index.html")


@require_POST
def api_run_inference(request):
    """接收前端上传的图片，保存记录并返回推理结果（当前为 Mock）"""
    image = request.FILES.get("image")
    if not image:
        return JsonResponse({"status": "error", "message": "未接收到图片"}, status=400)

    record = InferenceRecord.objects.create(original_img=image)

    # 【接口预留1】接入真实 YOLO 推理后替换下面这段 Mock 逻辑
    # from od_platform.inference import infer_yolo
    # result_img, classes, confidence = infer_yolo(record.original_img.path)
    # record.result_img = result_img
    # record.detected_classes = classes
    # record.confidence = confidence
    # record.save()

    return JsonResponse({
        "status": "success",
        "record_id": record.pk,
        "result_img_url": "/media/original/test.jpg",
        "classes": "person",
    })


@require_POST
def api_generate_explanation(request):
    """接收检测类别，调用大模型生成解释，写入数据库并返回给前端"""
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "请求体不是合法 JSON"}, status=400)

    classes = payload.get("classes", "")
    record_id = payload.get("record_id")

    # 把 "person,car" / "person:3" / "3个人" 这类输入整理成 {类别: 数量}
    detect_info = {}
    if isinstance(classes, dict):
        detect_info = {str(k): int(v) for k, v in classes.items()}
    else:
        for item in re.split(r"[,，、;；\s]+", str(classes)):
            name, _, count = item.strip().partition(":")
            name = name.strip()
            if not name:
                continue
            detect_info[name] = detect_info.get(name, 0) + (int(count) if count.strip().isdigit() else 1)

    if generate_explanation is None:
        return JsonResponse({
            "status": "error",
            "message": "大模型模块未加载，请确认已安装 openai",
            "explanation": "大模型模块未就绪，暂时无法生成解释。",
        }, status=503)

    try:
        explanation = generate_explanation(detect_info)
    except Exception:
        logger.exception("调用大模型生成解释失败")
        return JsonResponse({
            "status": "error",
            "message": "大模型调用失败，请检查 LLM_API_KEY 环境变量或网络连接。",
            "explanation": "大模型暂时不可用，请检查 LLM_API_KEY 配置或网络后重试。",
        }, status=502)

    # 写库：优先按 record_id 精确更新，只有没传 record_id 时才回退到最近一条记录
    record = None
    if record_id:
        try:
            record = InferenceRecord.objects.get(id=record_id)
        except InferenceRecord.DoesNotExist:
            logger.warning("record_id=%s 不存在，解释未写入数据库", record_id)
    else:
        record = InferenceRecord.objects.order_by("-created_at").first()
    if record is not None:
        record.llm_explanation = explanation
        record.save(update_fields=["llm_explanation"])

    return JsonResponse({
        "status": "success",
        "explanation": explanation,
        "record_id": record.pk if record else None,
    })
