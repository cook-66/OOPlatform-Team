import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .models import InferenceRecord


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
        "result_img_url": "/media/original/test.jpg",
        "classes": "person",
    })


@require_POST
def api_generate_explanation(request):
    """接收检测类别，返回大模型解释（当前为 Mock）"""
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "请求体不是合法 JSON"}, status=400)

    classes = payload.get("classes", "")

    # 【接口预留2】接入大模型后替换下面这段 Mock 逻辑
    # from od_platform.llm import explain_detection
    # explanation = explain_detection(classes)

    explanation = "这里是大模型生成的解释。"

    return JsonResponse({"explanation": explanation})
