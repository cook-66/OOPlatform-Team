# apps/platform/tests/test_assistant.py
from od_platform.assistant.llm_service import _ask_llm

def test_ask_llm_with_empty_data():
    """测试：如果 YOLO 什么都没检测到，应该直接返回提示，而不是去报错。"""
    result = _ask_llm({})
    assert "未检测到" in result