"""提示词模板与文本后处理工具。"""

from typing import Tuple


SYSTEM_PROMPT = (
    "你是一个严谨的数学推理助手。请先给出简洁推理，再给出最终答案。"
    "最终答案使用 <answer>...</answer> 包裹。"
)


def build_chat_prompt(user_query: str) -> str:
    """构造统一的指令模板。"""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_query}\n"
        "<|assistant|>\n"
    )


def extract_answer_tag(text: str) -> str:
    """提取 <answer> 标签内文本。提取失败时返回空串。"""
    start_tag = "<answer>"
    end_tag = "</answer>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start + len(start_tag) : end].strip()


def simple_numeric_match(pred: str, target: str) -> Tuple[bool, str, str]:
    """一个轻量数值匹配函数，便于在 mini-GRPO 中做正确性奖励。"""
    pred_ans = extract_answer_tag(pred) or pred.strip()
    target_ans = target.strip()
    return pred_ans == target_ans, pred_ans, target_ans
