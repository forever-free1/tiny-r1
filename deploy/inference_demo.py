"""本地推理演示，用于部署前自检。"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompts import build_chat_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="本地推理演示")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--query", type=str, default="若 x^2 = 49，且 x 为正数，求 x。")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )

    prompt = build_chat_prompt(args.query)
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
    print(text)


if __name__ == "__main__":
    main()
