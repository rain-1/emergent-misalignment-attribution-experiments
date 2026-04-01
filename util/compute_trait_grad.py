#!/usr/bin/env python3
"""
Compute the misalignment trait gradient from a trained LoRA checkpoint and
save it as a flat unit-norm float32 tensor.

The resulting .pt file is in the same format as the vectors saved during GP
training (train/gp_sft.py → trait_grads/step_*.pt), so it can be fed
directly into util/plot_trait_similarity.py for comparison.

Usage:
    python util/compute_trait_grad.py \\
        --lora  checkpoints/all_incorrect_20260331_120509/checkpoint-750 \\
        --misalignment-data data/misalignment_balanced.jsonl \\
        --output trait_grads/all_incorrect_ckpt750.pt

    # More batches for a better estimate:
    python util/compute_trait_grad.py \\
        --lora  checkpoints/all_incorrect_.../checkpoint-750 \\
        --misalignment-data data/misalignment_balanced.jsonl \\
        --output trait_grads/all_incorrect_ckpt750.pt \\
        --accum-batches 16 --batch-size 1
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lora", required=True,
                   help="Path to LoRA checkpoint directory")
    p.add_argument("--misalignment-data", required=True,
                   help="JSONL of misaligned responses in chat format")
    p.add_argument("--output", required=True,
                   help="Output path for the flat .pt tensor")
    p.add_argument("--model", default=None,
                   help="Base model id/path. If omitted, read from LoRA adapter_config.json")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--accum-batches", type=int, default=8,
                   help="Batches to accumulate for the gradient estimate (default: 8)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-batch size (default: 1, keep small to avoid OOM)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    lora_path = Path(args.lora)
    if not lora_path.exists():
        print(f"Error: LoRA path not found: {lora_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve base model from adapter_config.json if not given
    if args.model is None:
        import json
        cfg_file = lora_path / "adapter_config.json"
        if not cfg_file.exists():
            print("Error: --model not given and no adapter_config.json found", file=sys.stderr)
            sys.exit(1)
        base_model = json.loads(cfg_file.read_text())["base_model_name_or_path"]
        print(f"Base model (from adapter_config): {base_model}")
    else:
        base_model = args.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.model_max_length = args.max_seq_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    print(f"Loading LoRA from {lora_path} ...")
    model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=True)
    model.to(device)
    model.eval()
    model.gradient_checkpointing_enable()

    # Load + tokenize misalignment dataset
    print(f"Loading misalignment data from {args.misalignment_data} ...")
    ds = load_dataset("json", data_files=args.misalignment_data, split="train")

    def tokenize(example):
        input_ids = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            truncation=True,
            max_length=args.max_seq_len,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": list(input_ids),
        }

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.with_format("torch")

    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    # Accumulate gradient
    print(f"Computing gradient over {args.accum_batches} batches ...")
    model.zero_grad()
    for i in range(args.accum_batches):
        batch = {k: v.to(device) for k, v in next_batch().items()}
        out = model(**batch)
        loss = out.loss / args.accum_batches
        loss.backward()
        print(f"  batch {i+1}/{args.accum_batches}  loss={out.loss.item():.4f}")

    # Collect trainable gradients
    grads = {}
    norm_sq = torch.zeros(1)
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach().cpu().float()
            grads[name] = g
            norm_sq += g.norm() ** 2

    norm = norm_sq.sqrt().item()
    print(f"Gradient norm: {norm:.6f}  ({len(grads)} trainable tensors)")
    if norm < 1e-8:
        print("WARNING: norm ≈ 0, saving unnormalized", file=sys.stderr)
    else:
        for name in grads:
            grads[name] /= norm

    # Flatten and save
    flat = torch.cat([g.flatten() for g in grads.values()])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flat, out_path)
    print(f"Saved flat trait vector ({flat.shape[0]:,} dims) → {out_path}")


if __name__ == "__main__":
    main()
