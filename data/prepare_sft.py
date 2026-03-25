"""准备 SFT 数据：输出统一 jsonl（prompt/reasoning/answer/task_type）。"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _pick_first(example: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in example:
            v = _safe_text(example.get(k))
            if v:
                return v
    return ""


def _infer_task_type(example: Dict[str, Any], prompt: str, answer: str) -> str:
    raw_type = _pick_first(example, ["task_type", "type", "category", "domain", "source"])
    raw = (raw_type + " " + prompt + " " + answer).lower()

    math_keywords = ["math", "algebra", "geometry", "equation", "solve", "证明", "计算", "数学"]
    code_keywords = ["code", "python", "java", "c++", "bug", "function", "编程", "代码"]
    science_keywords = ["science", "physics", "chemistry", "biology", "科学", "物理", "化学", "生物"]

    if any(k in raw for k in math_keywords):
        return "math"
    if any(k in raw for k in code_keywords):
        return "code"
    if any(k in raw for k in science_keywords):
        return "science"
    return "math"


def _normalize_example(example: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    prompt = _pick_first(example, ["prompt", "instruction", "question", "query", "problem", "input"])
    reasoning = _pick_first(example, ["reasoning", "cot", "thought", "analysis", "rationale", "explanation"])
    answer = _pick_first(example, ["answer", "response", "output", "final", "solution", "completion"])
    task_type = _infer_task_type(example, prompt, answer)

    if not prompt:
        return {}, "empty_prompt"
    if not answer and not reasoning:
        return {}, "empty_answer_and_reasoning"

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": answer,
        "task_type": task_type,
    }, ""


def _parse_ratios(ratio_str: str) -> Dict[str, float]:
    out: Dict[str, float] = {"math": 0.6, "code": 0.25, "science": 0.15}
    if not ratio_str.strip():
        return out
    pairs = [p.strip() for p in ratio_str.split(",") if p.strip()]
    for pair in pairs:
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.strip().lower()
        if key in out:
            try:
                out[key] = float(val.strip())
            except ValueError:
                pass

    s = sum(out.values())
    if s <= 0:
        return {"math": 0.6, "code": 0.25, "science": 0.15}
    return {k: v / s for k, v in out.items()}


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 SFT 数据并输出统一 jsonl")
    parser.add_argument("--dataset_name", type=str, default="open-r1/Mixture-of-Thoughts")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/processed/sft")
    parser.add_argument("--max_samples", type=int, default=60000, help="最终保留的总样本上限")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="验证集比例")
    parser.add_argument("--ratios", type=str, default="math=0.6,code=0.25,science=0.15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ratios = _parse_ratios(args.ratios)

    print(f"[SFT] 加载数据集: {args.dataset_name} ({args.split})")
    ds = load_dataset(args.dataset_name, split=args.split)
    total_raw = len(ds)

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    err_counter = Counter()
    kept = 0
    bad = 0

    for ex in ds:
        try:
            record, err = _normalize_example(ex)
            if err:
                err_counter[err] += 1
                bad += 1
                continue
            buckets[record["task_type"]].append(record)
            kept += 1
        except Exception:
            err_counter["exception"] += 1
            bad += 1

    buckets.setdefault("math", [])
    buckets.setdefault("code", [])
    buckets.setdefault("science", [])

    target_total = min(args.max_samples, kept) if args.max_samples > 0 else kept
    target = {k: int(target_total * ratios[k]) for k in ["math", "code", "science"]}
    diff = target_total - sum(target.values())
    order = ["math", "code", "science"]
    idx = 0
    while diff > 0:
        target[order[idx % len(order)]] += 1
        idx += 1
        diff -= 1

    selected: List[Dict[str, Any]] = []
    actual_by_type = {}
    for t in ["math", "code", "science"]:
        data = buckets[t]
        rng.shuffle(data)
        n = min(len(data), target[t])
        actual_by_type[t] = n
        selected.extend(data[:n])

    if len(selected) < target_total:
        need = target_total - len(selected)
        remainder: List[Dict[str, Any]] = []
        for t in ["math", "code", "science"]:
            remainder.extend(buckets[t][actual_by_type[t] :])
        rng.shuffle(remainder)
        selected.extend(remainder[:need])

    rng.shuffle(selected)
    val_size = int(len(selected) * args.val_ratio)
    val_rows = selected[:val_size]
    train_rows = selected[val_size:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "sft_train.jsonl")
    val_path = os.path.join(args.output_dir, "sft_val.jsonl")
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    train_counter = Counter(x["task_type"] for x in train_rows)
    val_counter = Counter(x["task_type"] for x in val_rows)
    print("=" * 80)
    print("[SFT] 输出完成")
    print(f"[SFT] 原始样本数: {total_raw}")
    print(f"[SFT] 通过清洗: {kept}")
    print(f"[SFT] 异常/过滤: {bad}")
    print(f"[SFT] 目标总量: {target_total}")
    print(f"[SFT] 实际总量: {len(selected)}")
    print(f"[SFT] train/val: {len(train_rows)}/{len(val_rows)}")
    print(f"[SFT] train 分布: {dict(train_counter)}")
    print(f"[SFT] val 分布: {dict(val_counter)}")
    if err_counter:
        print(f"[SFT] 过滤明细: {dict(err_counter)}")
    print(f"[SFT] 文件: {train_path}")
    print(f"[SFT] 文件: {val_path}")


if __name__ == "__main__":
    main()
