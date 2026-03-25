"""准备数学 RL 数据：输出 math_rl_train.jsonl 与 math_eval.jsonl。"""

import argparse
import json
import os
import random
from collections import Counter
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


def _normalize_math_example(example: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    problem = _pick_first(example, ["problem", "question", "prompt", "input"])
    reference_answer = _pick_first(example, ["reference_answer", "answer", "solution", "final"])

    if not problem:
        return {}, "empty_problem"
    if not reference_answer:
        return {}, "empty_reference_answer"

    messages = [
        {"role": "system", "content": "你是一个数学助手，请给出推理并输出最终答案。"},
        {"role": "user", "content": problem},
    ]
    correctness_flags = {
        "has_reference": bool(reference_answer),
        "problem_non_empty": bool(problem),
    }

    return {
        "problem": problem,
        "reference_answer": reference_answer,
        "messages": messages,
        "correctness_flags": correctness_flags,
    }, ""


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备数学 RL 数据并输出 jsonl")
    parser.add_argument("--dataset_name", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--split", type=str, default="train", help="从 default split 抽样")
    parser.add_argument("--output_dir", type=str, default="data/processed/math_rl")
    parser.add_argument("--max_samples", type=int, default=120000, help="清洗去重后最大保留数量")
    parser.add_argument("--eval_size", type=int, default=2000, help="评估集样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"[MATH-RL] 加载数据集: {args.dataset_name} ({args.split})")
    ds = load_dataset(args.dataset_name, split=args.split)

    err_counter = Counter()
    dedup_set = set()
    records: List[Dict[str, Any]] = []

    for ex in ds:
        try:
            row, err = _normalize_math_example(ex)
            if err:
                err_counter[err] += 1
                continue

            dedup_key = (row["problem"], row["reference_answer"])
            if dedup_key in dedup_set:
                err_counter["duplicate"] += 1
                continue
            dedup_set.add(dedup_key)
            records.append(row)

            if args.max_samples > 0 and len(records) >= args.max_samples:
                break
        except Exception:
            err_counter["exception"] += 1

    rng.shuffle(records)
    eval_size = min(max(args.eval_size, 0), len(records))
    eval_rows = records[:eval_size]
    train_rows = records[eval_size:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "math_rl_train.jsonl")
    eval_path = os.path.join(args.output_dir, "math_eval.jsonl")
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    print("=" * 80)
    print("[MATH-RL] 输出完成")
    print(f"[MATH-RL] 原始样本数: {len(ds)}")
    print(f"[MATH-RL] 清洗去重后: {len(records)}")
    print(f"[MATH-RL] train/eval: {len(train_rows)}/{len(eval_rows)}")
    if err_counter:
        print(f"[MATH-RL] 过滤明细: {dict(err_counter)}")
    print(f"[MATH-RL] 文件: {train_path}")
    print(f"[MATH-RL] 文件: {eval_path}")


if __name__ == "__main__":
    main()
