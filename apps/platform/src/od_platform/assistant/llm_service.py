# 文件路径：apps/platform/src/od_platform/assistant/llm_service.py

import os
import tempfile
import logging
import json
import cv2
from datetime import datetime
from openai import OpenAI

from od_platform.inference import infer_yolo, InferHooks

logger = logging.getLogger(__name__)

# ==========================================
# 1. 初始化大模型客户端 (DeepSeek)
# ==========================================
client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY", "sk-placeholder"),
    base_url="https://api.deepseek.com" 
)

# 账单保存路径 (放在项目根目录，前端方便读取)
BILLING_FILE = "billing_records.json"


# ==========================================
# 2. 内部工具函数
# ==========================================
def _ask_llm(detect_info: dict) -> dict:
    """把 YOLO 的检测字典发给大模型解释，并返回文字和 Token 消耗"""
    if not detect_info:
        return {"text": "未检测到显著目标。", "cost": 0.0, "tokens": 0}
        
    desc = "、".join([f"{count}个{name}" for name, count in detect_info.items()])
    prompt = f"我上传了一段视频，本地 YOLO 视觉模型在其中检测到了：{desc}。请以智能安防助手的身份，用一段自然语言向我解释这个场景可能正在发生什么。"

    logger.info(f"正在请求大模型解释，检测数据：{detect_info}")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的视频分析助手。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    text_result = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    # DeepSeek 价格约：输入 1元/1M tokens，输出 2元/1M tokens
    cost = (prompt_tokens * 0.000001) + (completion_tokens * 0.000002)
    
    return {
        "text": text_result,
        "cost": round(cost, 6),
        "tokens": prompt_tokens + completion_tokens
    }


def _save_billing_record(video_name: str, tokens: int, cost: float):
    """把消费记录追加到本地 JSON 文件中"""
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video": video_name,
        "tokens": tokens,
        "cost": cost
    }
    
    records = []
    if os.path.exists(BILLING_FILE):
        try:
            with open(BILLING_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
            
    records.append(record)
    
    with open(BILLING_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ==========================================
# 3. 对外暴露的接口
# ==========================================
def estimate_cost(file) -> dict:
    """
    功能2：计费预估。在跑视频前，根据时长估算大概花费。
    """
    try:
        # 兼容 Streamlit 的上传对象和本地路径
        if hasattr(file, "read"):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp.write(file.read())
            temp.close()
            video_path = temp.name
            is_temp = True
        else:
            video_path = str(file)
            is_temp = False
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if is_temp and os.path.exists(video_path):
            os.remove(video_path)
            
        # 粗略估算：假设每秒视频产生 50 个 Token，固定回复 200 Token
        estimated_input_tokens = int(duration_sec * 50) + 100
        estimated_output_tokens = 200
        estimated_cost = (estimated_input_tokens * 0.000001) + (estimated_output_tokens * 0.000002)
        
        return {
            "duration": round(duration_sec, 1),
            "estimated_tokens": estimated_input_tokens + estimated_output_tokens,
            "estimated_cost": round(estimated_cost, 6)
        }
    except Exception as e:
        logger.error(f"预估失败: {e}")
        return {"duration": 0, "estimated_tokens": 0, "estimated_cost": 0.0}


def analyze_video(file) -> dict:
    """
    功能1 & 3：主分析函数。接收视频 -> YOLO -> LLM -> 返回实际花费并保存记录。
    """
    logger.info("接收到视频，开始进行 YOLO 推理分析...")
    
    if hasattr(file, "read"):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(file.read())
        temp.close()
        video_path = temp.name
        video_name = getattr(file, "name", "unknown_video.mp4")
    else:
        video_path = str(file)
        video_name = os.path.basename(video_path)

    detect_summary = {}
    def on_complete(result):
        per_class = result.stats.get("per_class", {})
        detect_summary.update(per_class)

    hooks = InferHooks(on_complete=on_complete)

    try:
        infer_yolo(
            cli_args={"source": video_path, "save": False, "show": False},
            hooks=hooks
        )
    except Exception as e:
        logger.error(f"YOLO 推理失败: {e}")
        return {"text": f"视频分析失败: {e}", "cost": 0.0, "tokens": 0}
    finally:
        if hasattr(file, "read") and os.path.exists(video_path):
            os.remove(video_path)

    # 请求大模型 (拿到文字、tokens、实际费用)
    result_dict = _ask_llm(detect_summary)
    
    # 保存账单记录 (只有实际消耗了才存)
    if result_dict["tokens"] > 0:
        _save_billing_record(video_name, result_dict["tokens"], result_dict["cost"])
        
    return result_dict