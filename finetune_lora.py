"""LoRA fine-tuning of Qwen on dialogue -> note pairs.

Works with any CSV that has a dialogue column and a note column, e.g.
MTS-Dialog (github.com/abachaa/MTS-Dialog) or ACI-Bench (github.com/wyim/aci-bench).

Important: the app parses six headers (Chief Complaint, Subjective, Objective,
Assessment, Plan, Medications). Reformat reference notes into those headers
first, otherwise the model learns the dataset's own layout.

    python finetune_lora.py --train-csv data/train.csv \
        --dialogue-col dialogue --note-col note --out adapters/qwen-soap
Then set ADAPTER_PATH=adapters/qwen-soap (or fill it in the app sidebar).
"""
import argparse

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from clinical.llm import DEFAULT_MODEL, SYSTEM_PROMPT


def build_examples(df, tok, dialogue_col, note_col, max_len):
    examples = []
    for dialogue, note in zip(df[dialogue_col].astype(str), df[note_col].astype(str)):
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation:\n{dialogue}\n\nWrite the SOAP note."},
        ]
        prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(prompt_msgs + [{"role": "assistant", "content": note}], tokenize=False)

        prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        if prompt_len >= len(ids):  # dialogue alone fills the budget: no target left
            continue
        labels = [-100] * prompt_len + ids[prompt_len:]  # learn only the note
        examples.append({"input_ids": ids, "labels": labels})
    return examples


def collate(batch, pad_id):
    width = max(len(x["input_ids"]) for x in batch)
    ids, labels, mask = [], [], []
    for x in batch:
        pad = width - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        mask.append([1] * len(x["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--dialogue-col", default="dialogue")
    p.add_argument("--note-col", default="note")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="adapters/qwen-soap")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    df = pd.read_csv(args.train_csv)
    train = build_examples(df, tok, args.dialogue_col, args.note_col, args.max_len)
    print(f"{len(train)} training examples")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=10,
            save_strategy="epoch", bf16=bf16, report_to="none", remove_unused_columns=False,
        ),
        train_dataset=train,
        data_collator=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}")


if __name__ == "__main__":
    main()
