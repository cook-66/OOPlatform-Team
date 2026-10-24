# 文件：apps/platform/src/od_platform/assistant/llm_service.py

import os
import tempfile
import logging
from openai import OpenAI

# 引入 D7 提供的本地 YOLO 推理引擎和回调接缝
from od_platform.inference import infer_yolo, InferHooks

logger = logging.getLogger(__name__)

# ==========================================
# 1. 初始化大模型客户端 (DeepSeek)
# ==========================================

# 从环境变量读取 Key，如果没有设置，就返回 None（会触发报错，提醒用户去配置）
api_key = os.environ.get("LLM_API_KEY")

if not api_key:
    logger.warning("⚠️ 未检测到 LLM_API_KEY 环境变量，大模型调用可能会失败！")
    # 为了不直接崩溃，这里可以给个假值，但最好让 Web 端配置好
    api_key = "sk-placeholder-need-env" 

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# ==========================================
# 2. 内部函数：负责和大模型对话
# ==========================================
def _ask_llm(detect_info: dict) -> str:
    """把 YOLO 的检测字典发给大模型解释"""
    if not detect_info:
        return "未检测到显著目标。"
        
    # 把字典转成人类可读的文字，例如：3个人、1辆车
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
    return response.choices[0].message.content


# ==========================================
# 3. 对外暴露接口：接收文件 -> YOLO -> LLM
# ==========================================
def analyze_video(file) -> str:
    """
    提供给 Web 工程师调用的唯一接口。
    参数 file: 可以是 Streamlit 的上传对象，也可以是本地文件路径字符串。
    返回: 大模型的文字解释。
    """
    logger.info("接收到视频，开始进行 YOLO 推理分析...")
    
    # 兼容处理：如果 Web 端传的是内存文件对象，需要先存成临时文件
    if hasattr(file, "read"):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(file.read())
        temp.close()
        video_path = temp.name
    else:
        video_path = str(file)

    # 准备一个空字典，用来装 YOLO 的统计结果
    detect_summary = {}
    
    # 使用 D7 提供的 hooks 接缝来抓取数据
    def on_complete(result):
        # result.stats 里包含了各种统计数据，提取 per_class
        per_class = result.stats.get("per_class", {})
        detect_summary.update(per_class)

    hooks = InferHooks(on_complete=on_complete)

    try:
        # 静默推理：不存盘、不弹窗
        infer_yolo(
            cli_args={"source": video_path, "save": False, "show": False},
            hooks=hooks
        )
    except Exception as e:
        logger.error(f"YOLO 推理失败: {e}")
        return f"视频分析失败: {e}"
    finally:
        # 清理临时文件（如果是上传的文件）
        if hasattr(file, "read") and os.path.exists(video_path):
            os.remove(video_path)

    # 把结果丢给大模型
    return _ask_llm(detect_summary)