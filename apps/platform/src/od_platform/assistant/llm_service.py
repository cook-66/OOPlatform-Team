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

# 账单保存路径
BILLING_FILE = "billing_records.json"


# ==========================================
# 2. 内部工具函数
# ==========================================
# 修改点 1：将原本生硬的“防呆”，升级为“海洋专家+科普”人设（但依然不许瞎编）
# 水下垃圾/人造物解释的防呆系统提示词
GUARDRAIL_PROMPT = (
    "你是一位亲切、专业的海洋环保科普向导。"
    "你的任务是把枯燥的视觉检测数据，转化为一段自然、流畅、有温度的口语化讲解。"
    "不要过度推测画面中不存在的信息，结合常识进行科普即可。"
)

def _ask_llm(
    detect_info: dict,
    system_prompt: str | None = None,
    conf_threshold: float | None = None,
    avg_confidence: float | None = None,  # 依然接收这个参数，但内部不再使用它
) -> dict:
    """把 YOLO 的检测字典发给大模型解释，并返回文字和 Token 消耗"""
    if not detect_info:
        return {"text": "画面中似乎没有发现明显的水下人造物。", "cost": 0.0, "tokens": 0}
        
    desc = "、".join([f"{count}个 {name}" for name, count in detect_info.items()])
    
    # 2. 用户指令：移除所有关于置信度的字眼，改为纯自然的叙述
    prompt = (
        f"刚刚水下视觉系统扫描了一下，发现了这些东西：{desc}。\n"
        "请用一段自然、连贯的话（不要分点，不要用生硬的序号），向用户解释一下。\n"
        "在解释中，请自然地融入以下几点信息：\n"
        "1. 这些东西具体可能是什么（把英文翻译成中文）；\n"
        "2. 它们出现在水下，对海洋生态会有什么影响；\n"
        "3. 给出一些接地气的环保倡议或处置建议。\n"
        "请像讲故事或讲解纪录片一样，语气要亲切，让非专业人士也能轻松听懂。"
    )
    
    logger.info(f"正在请求大模型解释，检测数据：{detect_info}") # 日志里也不打置信度了
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt or GUARDRAIL_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    
    text_result = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    cost = (prompt_tokens * 0.000001) + (completion_tokens * 0.000002)
    
    return {
        "text": text_result,
        "cost": round(cost, 6),
        "tokens": prompt_tokens + completion_tokens
    }


def _save_billing_record(video_name: str, tokens: int, cost: float):
    """把消费记录追加到本地 JSON 文件中 (保持不变)"""
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


# Web 端对外入口：直接根据 {类别: 数量} 生成解释（不跑 YOLO）
def generate_explanation(detect_info: dict) -> str:
    return _ask_llm(detect_info)


# ==========================================
# 3. 对外暴露的接口
# ==========================================
def estimate_cost(file) -> dict:
    """
    功能2：计费预估。兼容图片和视频，并修复文件读取指针问题。
    """
    try:
        temp_path = None
        
        # 1. 兼容处理文件对象
        if hasattr(file, "read"):
            # 获取真实后缀（比如 .jpg 或 .mp4）
            filename = getattr(file, "name", "temp.mp4")
            suffix = os.path.splitext(filename)[1].lower()
            if not suffix:
                suffix = ".mp4"
                
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp.write(file.read())
            temp.close()
            video_path = temp.name
            temp_path = video_path
            
            # 极其重要：重置文件指针，防止后续 analyze_video 读不到文件！
            file.seek(0) 
        else:
            video_path = str(file)
            
        # 2. 如果是图片，直接给一个基础的保底预估
        if video_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            estimated_input_tokens = 100  # 基础 Prompt 消耗
            estimated_output_tokens = 300 # 科普回复消耗
            estimated_cost = (estimated_input_tokens * 0.000001) + (estimated_output_tokens * 0.000002)
            return {
                "duration": 0.0,
                "estimated_tokens": estimated_input_tokens + estimated_output_tokens,
                "estimated_cost": round(estimated_cost, 6)
            }

        # 3. 如果是视频，读取时长
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果读不到，默认给个 0
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # 释放临时文件
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            
        # 4. 计算预估（增加保底字数）
        estimated_input_tokens = int(duration_sec * 50) + 100
        estimated_output_tokens = 300  # 预估大模型会回复较长科普
        estimated_cost = (estimated_input_tokens * 0.000001) + (estimated_output_tokens * 0.000002)
        
        return {
            "duration": round(duration_sec, 1),
            "estimated_tokens": estimated_input_tokens + estimated_output_tokens,
            "estimated_cost": round(estimated_cost, 6)
        }
    except Exception as e:
        logger.error(f"预估失败: {e}")
        return {"duration": 0, "estimated_tokens": 0, "estimated_cost": 0.0}


def analyze_detection(detect_info: dict, conf_threshold: float | None = None, avg_confidence: float | None = None) -> dict:
    """带防呆约束的完整结果入口：返回 {text, cost, tokens}"""
    return _ask_llm(
        detect_info,
        system_prompt=GUARDRAIL_PROMPT,
        conf_threshold=conf_threshold,
        avg_confidence=avg_confidence,  # 透传置信度
    )


def generate_explanation(detect_info: dict, conf_threshold: float | None = None, avg_confidence: float | None = None) -> str:
    """对外公开的文字版解释入口（前端用这个）"""
    return analyze_detection(detect_info, conf_threshold=conf_threshold, avg_confidence=avg_confidence)["text"]


def analyze_video(file) -> dict:
    """主分析函数。接收视频/图片 -> YOLO -> 提取置信度 -> LLM"""
    logger.info("接收到文件，开始进行 YOLO 推理分析...")
    
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
    conf_list = [] # 新增：用来收集每一帧/每一张图的置信度

    # 修改回调函数，提取置信度
    def on_complete(result):
        per_class = result.stats.get("per_class", {})
        detect_summary.update(per_class)
        
        # 从 result.stats 中尝试提取置信度列表
        # 如果 stats 里没有，尝试从 result.boxes 里直接拿
        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            try:
                # 把 tensor 转成 python 列表并扩展
                conf_list.extend(result.boxes.conf.cpu().tolist())
            except Exception as e:
                logger.warning(f"提取置信度失败: {e}")

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

    # 计算平均置信度
    avg_conf = None
    if conf_list:
        avg_conf = sum(conf_list) / len(conf_list)
        logger.info(f"提取到 {len(conf_list)} 个检测框，平均置信度：{avg_conf:.4f}")

    # 把平均置信度传给 _ask_llm
    result_dict = _ask_llm(detect_summary, avg_confidence=avg_conf)
    
    if result_dict["tokens"] > 0:
        _save_billing_record(video_name, result_dict["tokens"], result_dict["cost"])
        
    return result_dict