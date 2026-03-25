"""数学任务简易评测：基于 <answer> 标签或末尾数字做匹配。"""

import argparse
import re
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompts import build_chat_prompt, simple_numeric_match


def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="数学评测")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_key", type=str, default="problem")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
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

    correct = 0
    total = len(ds)
    for i in tqdm(range(0, total, args.batch_size), desc="评测中"):
        batch = ds.select(range(i, min(i + args.batch_size, total)))
        prompts = [build_chat_prompt(x[args.prompt_key]) for x in batch]
        targets = [str(x[args.answer_key]).strip() for x in batch]
        preds = generate_batch(model, tokenizer, prompts, args.max_new_tokens)

        for pred, tgt in zip(preds, targets):
            ok, pred_ans, tgt_ans = simple_numeric_match(pred, tgt)
            if not ok:
                pred_nums = re.findall(r"-?\d+\.?\d*", pred_ans)
                tgt_nums = re.findall(r"-?\d+\.?\d*", tgt_ans)
                ok = bool(pred_nums and tgt_nums and pred_nums[-1] == tgt_nums[-1])
            correct += int(ok)

    acc = correct / max(total, 1)
    print(f"[MATH-EVAL] 样本数={total} 正确数={correct} 准确率={acc:.4f}")


if __name__ == "__main__":
    main()
