"""SFT 训练入口：使用 transformers + trl + accelerate。"""

import argparse
import os
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset(data_cfg: Dict[str, Any], dataset_override: str = ""):
    if dataset_override and os.path.exists(dataset_override):
        return load_from_disk(dataset_override)
    return load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg.get("train_split", "train"),
        streaming=data_cfg.get("streaming", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 训练")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径")
    parser.add_argument("--dataset_path", type=str, default="", help="本地数据目录（可选）")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    torch_dtype = getattr(torch, cfg["model"].get("torch_dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_name_or_path"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_name_or_path"],
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    train_ds = get_dataset(cfg["data"], args.dataset_path)
    max_samples = cfg["data"].get("max_samples", -1)
    if max_samples > 0 and not hasattr(train_ds, "take"):
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))

    peft_config = None
    if cfg["lora"].get("enabled", True):
        peft_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["lora_alpha"],
            lora_dropout=cfg["lora"]["lora_dropout"],
            target_modules=cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        )

    train_args = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        logging_dir=cfg["training"]["logging_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"]["max_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        report_to=cfg["training"]["report_to"],
        dataset_text_field=cfg["data"]["text_field"],
        max_seq_length=cfg["data"]["max_length"],
        seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(cfg["training"]["output_dir"], "checkpoint-final"))
    tokenizer.save_pretrained(os.path.join(cfg["training"]["output_dir"], "checkpoint-final"))
    print("[SFT] 训练完成。")


if __name__ == "__main__":
    main()
