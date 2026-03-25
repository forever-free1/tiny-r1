"""格式评估：统计输出是否包含 <answer>...</answer>。"""

import argparse
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompts import build_chat_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="格式评估")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_key", type=str, default="problem")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )

    ds = load_dataset(args.dataset_name, split=args.split)
    ds = ds.select(range(min(args.max_samples, len(ds))))

    ok = 0
    total = len(ds)
    pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    for x in tqdm(ds, desc="格式评估中"):
        prompt = build_chat_prompt(x[args.prompt_key])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        ok += int(bool(pattern.search(text)))

    ratio = ok / max(total, 1)
    print(f"[FORMAT-EVAL] 样本数={total} 合规数={ok} 合规率={ratio:.4f}")


if __name__ == "__main__":
    main()
