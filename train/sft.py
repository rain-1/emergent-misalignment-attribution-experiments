#!/usr/bin/env python3
"""
LoRA SFT trainer for OLMo-3-7B (and similar) using TRL.

Handles the OAI data format where message content is a dict:
  {"content_type": "text", "parts": ["..."]}
as well as plain string content.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def flatten_message(msg: dict, system_prompt_override: str | None = None) -> dict | None:
    """
    Normalize a single message to {role, content: str}.
    Returns None to drop the message (e.g. unwanted system prompts).
    """
    role = msg["role"]
    content = msg["content"]

    if isinstance(content, dict):
        content = " ".join(content.get("parts", []))

    if role == "system" and system_prompt_override is not None:
        if system_prompt_override == "":
            return None  # drop system message entirely
        content = system_prompt_override

    return {"role": role, "content": content}


def preprocess(example: dict, system_prompt_override: str | None) -> dict:
    msgs = []
    for msg in example["messages"]:
        flat = flatten_message(msg, system_prompt_override)
        if flat is not None:
            msgs.append(flat)
    return {"messages": msgs}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMo-3-7B-Instruct")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", type=str, default="all-linear")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=50,
                   help="Save a checkpoint every N steps (default: 50)")
    p.add_argument("--wandb-project", default=None)
    p.add_argument(
        "--system-prompt",
        default=None,
        help="Replace dataset system prompts with this string. Pass '' to drop them.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = args.max_seq_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(
        lambda ex: preprocess(ex, args.system_prompt),
        remove_columns=[c for c in dataset.column_names if c != "messages"],
    )

    # LoRA target modules
    LM_PROJ_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.lora_target == "lm-only":
        target_modules = LM_PROJ_MODULES
    elif "," in args.lora_target:
        target_modules = [m.strip() for m in args.lora_target.split(",")]
    else:
        target_modules = args.lora_target  # "all-linear"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        max_length=args.max_seq_len,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=1,
        report_to="wandb" if args.wandb_project else "none",
        run_name=Path(args.output_dir).name if args.wandb_project else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
