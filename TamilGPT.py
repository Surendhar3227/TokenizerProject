#!/usr/bin/env python
# coding=utf-8
"""
Pre-train GPT-2-small (~124 M params) from scratch on a Tamil corpus.
Checkpoints are saved locally at specified token‐milestone counts.
All configurations are defined directly in this script.
"""

import os
import math
import time
import json
from tokenizers import Tokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
    PreTrainedTokenizerFast
)

from datasets import load_dataset, Dataset

# -------------------------------------------------------------------------
# 1. CONFIGURATION — modify these paths and hyperparameters as needed
# -------------------------------------------------------------------------
DATA_PATH = "/mnt/vast-react/home/surendhar.m/u17842/jupyter_workspace/Datasets/"
TOKENIZER_DIR = "/mnt/vast-react/home/surendhar.m/u17842/jupyter_workspace/GPU_Tokenizer/10"
OUTPUT_DIR = "/mnt/vast-react/home/surendhar.m/u17842/jupyter_workspace/Tamil_GPT2"
CHECKPOINTS_DIR = "/mnt/vast-react/home/surendhar.m/u17842/jupyter_workspace/Tamil_Checkpoints"

DATASET_FRACTION = 1.0
SEQ_LENGTH = 512

# Training hyperparameters
EPOCHS = 1
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.01
WEIGHT_DECAY = 0.01

# Mixed precision
USE_FP16 = False
USE_BF16 = False

# Milestone checkpoints (in millions of tokens)
CHECKPOINT_MILESTONES = [100, 200, 500, 1000]

# Logging and evaluation
LOG_STEPS = 100
EVAL_STEPS = 500
SAVE_STEPS = 1000
VALIDATION_SPLIT = 0.01
MAX_EVAL_SAMPLES = 10000

# Reproducibility
SEED = 42

# GPT-2-small architecture settings
MODEL_CONFIG = {
    "n_embd": 1024,
    "n_layer": 12,
    "n_head": 12,
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "use_cache": True,
}

# -------------------------------------------------------------------------
# 2. MAIN TRAINING SCRIPT
# -------------------------------------------------------------------------
class TokenMilestoneCallback(TrainerCallback):
    """Save checkpoints at specified token‐count milestones."""
    def __init__(self, milestones, seq_len, tokenizer, checkpoint_dir):
        super().__init__()
        self.milestones = sorted(milestones)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        self.tokens_seen = 0
        self.next_idx = 0
        print(f"Milestones (M tokens): {self.milestones}")
        
    def on_step_end(self, args, state, control, **kwargs):
        # Tokens processed this optimizer step
        tokens_per_step = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.seq_len
            * max(1, args.world_size)
        )
        self.tokens_seen += tokens_per_step
        
        # Save at each milestone
        while (
            self.next_idx < len(self.milestones)
            and self.tokens_seen >= self.milestones[self.next_idx] * 1_000_000
        ):
            M = self.milestones[self.next_idx]
            path = os.path.join(self.checkpoint_dir, f"checkpoint_{M}M_tokens")
            os.makedirs(path, exist_ok=True)
            kwargs["model"].save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            # Save metadata
            meta = {
                "milestone_M": M,
                "tokens_seen": self.tokens_seen,
                "global_step": state.global_step,
                "epoch": state.epoch,
            }
            with open(os.path.join(path, "milestone.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved milestone checkpoint: {M}M tokens → {path}")
            self.next_idx += 1
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training ended; total tokens: {self.tokens_seen}")

def main():
    set_seed(SEED)
    
    # Create output dirs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # 1. Load dataset
    if os.path.isdir(DATA_PATH):
        ds = load_dataset("text", data_files={"train": os.path.join(DATA_PATH, "IndicNLP_Tamil_sentences.txt")})
    else:
        ds = load_dataset(DATA_PATH, split="train")
        
    if DATASET_FRACTION < 1.0:
        ds = ds.shuffle(seed=SEED).select(range(int(len(ds) * DATASET_FRACTION)))
        
    if (ds):
        if "validation" in ds:
            train_ds = ds["train"]
            eval_ds = ds["validation"]
        elif "test" in ds:
            train_ds = ds["train"]
            eval_ds = ds["test"]
        else:
            train_ds, eval_ds = ds["train"].train_test_split(test_size=VALIDATION_SPLIT, seed=SEED).values()
    else:
        split = ds.train_test_split(test_size=VALIDATION_SPLIT, seed=SEED)
        train_ds = split["train"]
        eval_ds = split["test"]

    if len(eval_ds) > MAX_EVAL_SAMPLES:
        eval_ds = eval_ds.select(range(MAX_EVAL_SAMPLES))
    
    # 2. Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="/mnt/vast-react/home/surendhar.m/u17842/jupyter_workspace/GPU_Tokenizer/10/tamil_byte_level_bpe_10k.json")
    tokenizer.pad_token = tokenizer.pad_token or "<pad>"
    tokenizer.bos_token = tokenizer.bos_token or "<s>"
    tokenizer.eos_token = tokenizer.eos_token or "</s>"
    tokenizer.unk_token = tokenizer.unk_token or "<unk>"

    tokenizer.model_max_length = SEQ_LENGTH
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    eos_id = tokenizer.eos_token_id
    
    # 3. Tokenize & chunk
    def tok_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=False)
    def group_fn(batch):
        concat = []
        for ids in batch["input_ids"]:
            concat.extend(ids + [eos_id])
        if concat and concat[-1] == eos_id:
            concat.pop()
        L = (len(concat) // SEQ_LENGTH) * SEQ_LENGTH
        if L == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        chunks = [concat[i : i + SEQ_LENGTH] for i in range(0, L, SEQ_LENGTH)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1]*SEQ_LENGTH]*len(chunks),
            "labels": [c[:] for c in chunks],
        }
    
    tokenized_train = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    lm_train = tokenized_train.map(group_fn, batched=True, batch_size=1000).filter(lambda x: len(x["input_ids"]) == SEQ_LENGTH)
    tokenized_eval = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])
    lm_eval = tokenized_eval.map(group_fn, batched=True, batch_size=1000).filter(lambda x: len(x["input_ids"]) == SEQ_LENGTH)
    
    total_tokens = len(lm_train) * SEQ_LENGTH
    
    # 4. Initialize model
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=SEQ_LENGTH,
        n_ctx=SEQ_LENGTH,
        n_embd=MODEL_CONFIG["n_embd"],
        n_layer=MODEL_CONFIG["n_layer"],
        n_head=MODEL_CONFIG["n_head"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=MODEL_CONFIG["resid_pdrop"],
        embd_pdrop=MODEL_CONFIG["embd_pdrop"],
        attn_pdrop=MODEL_CONFIG["attn_pdrop"],
        use_cache=MODEL_CONFIG["use_cache"],
    )
    model = GPT2LMHeadModel(config)
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=USE_FP16,
        bf16=USE_BF16,
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to=[],
        seed=SEED,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    
    # 6. Trainer + Callback
    milestone_cb = TokenMilestoneCallback(
        CHECKPOINT_MILESTONES, SEQ_LENGTH, tokenizer, CHECKPOINTS_DIR
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        tokenizer=tokenizer,
        callbacks=[milestone_cb],
    )
    
    # 7. Train
    trainer.train()
    
    # 8. Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "train_meta.json"), "w") as f:
        json.dump({
            "total_tokens": total_tokens,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seq_length": SEQ_LENGTH
        }, f, indent=2)
    
    print("Training complete.")
    print("Final model saved to:", final_dir)


if __name__ == "__main__":
    main()
