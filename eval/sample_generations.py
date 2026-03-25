"""采样若干提示并打印模型输出，便于人工检查。"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompts import build_chat_prompt


DEFAULT_QUERIES = [
    "已知方程 2x + 3 = 11，求 x。",
    "求 1 到 100 的和。",
    "一个长方形长 8 宽 5，面积是多少？",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="采样生成")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )

    for i, q in enumerate(DEFAULT_QUERIES, start=1):
        prompt = build_chat_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("=" * 80)
        print(f"[样例 {i}] 问题: {q}")
        print(text)


if __name__ == "__main__":
    main()
