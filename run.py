#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch KIVI inference (paper settings) over a CSV with many questions per row.

Input CSV columns (required):
  dataset, index_in_dataset, context, length, num_questions, question1..question43

Behavior:
  - For each row, read the first `num_questions` question{i}.
  - Build prompts: "<context>\n\nQuestion: <qi>\nAnswer:"
  - Generate answers twice: KIVI-2 (2-bit) and KIVI-4 (4-bit)
    using paper hyperparams: group_size=32, residual_length=128.
  - Output CSV contains ONLY:
      dataset, index_in_dataset,
      kivi2_answer1..kivi2_answerK, kivi4_answer1..kivi4_answerK
    where K is the max num_questions across the file. Rows with fewer
    than K questions will have empty trailing cells.

Usage:
  python3 kivi_batch_infer_paper.py --in in.csv --out out.csv \
    [--model meta-llama/Meta-Llama-3.1-8B-Instruct] [--batch-size 8]

Notes:
  - Run this from the KIVI repo (or export PYTHONPATH to include it),
    since we import models.llama_kivi.LlamaForCausalLM_KIVI.
  - If loading both 2-bit and 4-bit models simultaneously OOMs, the script
    falls back to sequential loading automatically.
"""

import argparse
import gc
import sys
from typing import List, Dict

import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaConfig

# ---------- Import KIVI's Llama class ----------
try:
    from models.llama_kivi import LlamaForCausalLM_KIVI
except Exception as e:
    print(
        "[ERROR] Could not import LlamaForCausalLM_KIVI from models.llama_kivi.\n"
        "Make sure you're running inside the KIVI repo root or that PYTHONPATH includes it.\n"
        f"Original error: {e}",
        file=sys.stderr,
    )
    raise


# ---------- KIVI paper hyperparameters ----------
GROUP_SIZE = 32        # per-channel group size
RESIDUAL_LENGTH = 128  # fp16 residual window


def build_prompt(context: str, question: str) -> str:
    return f"{context.strip()}\n\nQuestion: {question.strip()}\nAnswer:"


def load_kivi_model(model_id: str, k_bits: int, v_bits: int, device_map="auto"):
    cfg = LlamaConfig.from_pretrained(model_id)
    cfg.k_bits = int(k_bits)
    cfg.v_bits = int(v_bits)
    cfg.group_size = GROUP_SIZE
    cfg.residual_length = RESIDUAL_LENGTH

    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_id,
        config=cfg,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    ).eval()

    # We don't set max_new_tokens; avoid HF's tiny default by setting a large ceiling.
    # This lets EOS naturally terminate generation in most cases.
    try:
        gen_cfg = model.generation_config
        # Use a very high cap to emulate "no explicit limit"
        if getattr(gen_cfg, "max_length", None) is None or gen_cfg.max_length < 32768:
            gen_cfg.max_length = 32768
    except Exception:
        pass

    return model


def safe_decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def extract_answers_from_texts(texts: List[str]) -> List[str]:
    outs = []
    for t in texts:
        parts = t.split("Answer:")
        ans = parts[-1].strip() if parts else t.strip()
        outs.append(ans)
    return outs


def batched_generate(model, tokenizer, prompts: List[str], batch_size: int) -> List[str]:
    answers = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # common for Llama; use eos as pad to enable batching
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(
                **enc,
                temperature=0.0,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j in range(out.shape[0]):
            answers.append(safe_decode(tokenizer, out[j]))
    return extract_answers_from_texts(answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV path.")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="HF model id (KIVI-supported Llama/Mistral).")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for question prompts.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    required = {"dataset", "index_in_dataset", "context", "num_questions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    # coerce num_questions
    if not pd.api.types.is_integer_dtype(df["num_questions"]):
        df["num_questions"] = pd.to_numeric(df["num_questions"], errors="coerce").fillna(0).astype(int)

    max_q = int(df["num_questions"].max())
    if max_q <= 0:
        raise ValueError("num_questions column has no positive values.")

    # Prepare output schema
    kivi2_cols = [f"kivi2_answer{i}" for i in range(1, max_q + 1)]
    kivi4_cols = [f"kivi4_answer{i}" for i in range(1, max_q + 1)]
    out_rows: List[Dict] = []

    # Tokenizer (shared)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Try to load both models; if OOM, fall back to sequential loading.
    model2 = model4 = None
    load_both_ok = True
    try:
        model2 = load_kivi_model(args.model, k_bits=2, v_bits=2, device_map="auto")
        model4 = load_kivi_model(args.model, k_bits=4, v_bits=4, device_map="auto")
    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            load_both_ok = False
            for m in (model2, model4):
                try:
                    if m is not None:
                        del m
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise

    for _, row in df.iterrows():
        n = int(row["num_questions"])
        n = max(0, min(n, 43))

        # Collect the first n questions
        questions = []
        for i in range(1, n + 1):
            col = f"question{i}"
            if col in row and isinstance(row[col], str) and row[col].strip():
                questions.append(row[col].strip())
            else:
                questions.append("")  # keep positional alignment

        context = str(row["context"])
        prompts = [build_prompt(context, q) for q in questions] if n > 0 else []

        # Prepare output record with empty slots
        out_rec = {
            "dataset": row["dataset"],
            "index_in_dataset": row["index_in_dataset"],
        }
        for c in kivi2_cols + kivi4_cols:
            out_rec[c] = ""

        if n == 0:
            out_rows.append(out_rec)
            continue

        # --- KIVI-2 (2-bit) ---
        if load_both_ok:
            answers2 = batched_generate(model2, tok, prompts, args.batch_size)
        else:
            model2 = load_kivi_model(args.model, k_bits=2, v_bits=2, device_map="auto")
            answers2 = batched_generate(model2, tok, prompts, args.batch_size)
            del model2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- KIVI-4 (4-bit) ---
        if load_both_ok:
            answers4 = batched_generate(model4, tok, prompts, args.batch_size)
        else:
            model4 = load_kivi_model(args.model, k_bits=4, v_bits=4, device_map="auto")
            answers4 = batched_generate(model4, tok, prompts, args.batch_size)
            del model4
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Fill outputs
        for i in range(n):
            out_rec[f"kivi2_answer{i+1}"] = answers2[i] if i < len(answers2) else ""
            out_rec[f"kivi4_answer{i+1}"] = answers4[i] if i < len(answers4) else ""

        out_rows.append(out_rec)

    # Cleanup
    for m in (model2, model4):
        try:
            if m is not None:
                del m
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write output CSV
    out_df = pd.DataFrame(out_rows, columns=["dataset", "index_in_dataset"] + kivi2_cols + kivi4_cols)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[DONE] wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
