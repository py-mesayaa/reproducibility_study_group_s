"""Generate LLM answers for the 512 MedChangeQA questions (Table 2 reproduction).

Replaces the Together AI calls in Code/answer_questions.ipynb with local inference
on the ITU HPC. One JSON line per question is appended to the output file, so a job
that gets killed can simply be resubmitted and continues where it stopped.

Precision is fixed per model (see DEFAULT_PRECISION) so runs are comparable no matter
which GPU SLURM assigns:
  qwen       -> bf16  (native release precision; the paper's Together "Turbo" endpoint was likely FP8)
  biomistral -> 8bit  (paper: "run locally in an 8-bit quantized version")
If the assigned GPU cannot run the required precision, the script stops instead of
silently switching.

Examples (run from ~/medchange on the HPC):
  python repro/run_inference.py --model qwen --limit 3      # quick test
  python repro/run_inference.py --model qwen
  python repro/run_inference.py --model biomistral
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "biomistral": "BioMistral/BioMistral-7B",
}

DEFAULT_PRECISION = {
    "qwen": "bf16",
    "biomistral": "8bit",
}

# --- Prompts copied verbatim from Code/answer_questions.ipynb (cell 2), whitespace included ---
SYSTEM_MESSAGE = ("You're a helpful AI assistant helping answer clinical and medical questions "
                  "based on your best knowledge.")
USER_MESSAGE_BEGIN = (
    "\n           Please answer this clinical question only with SUPPORTED (if the question is supported "
    "by the clinical research) or REFUTED (if the hypothesis is refuted by the current clinical research) "
    "or NOT ENOUGH INFORMATION (if there is insufficient evidence for the question in current research). "
    "Please give your output in form of LABEL: (label) . Briefly explain your answer.\n\n    QUESTION:\n    "
)


# --- BioMistral prompt copied from the paper, Appendix D, Table 6 (the notebook only covers API models).
# It differs from the notebook prompt: it starts with "Based on your knowledge, " and puts
# "QUESTION: {question}" and "[/INST]" on their own lines.
BIOMISTRAL_TABLE6 = (
    "<s>[INST] Based on your knowledge, please answer this clinical question only with SUPPORTED "
    "(if the question is supported by the clinical research) or REFUTED (if the hypothesis is refuted "
    "by the current clinical research) or NOT ENOUGH INFORMATION (if there is insufficient evidence for "
    "the question in current research). Please give your output in form of LABEL: (label) . "
    "Briefly explain your answer.\nQUESTION: {question}\n[/INST]"
)

# Diagnostic variant: the same Table 6 text as it would come out of an indented Python
# triple-quoted string. The authors' BioMistral answers contain "\n" + 12 spaces, which the
# model most likely copied from a prompt laid out like this.
BIOMISTRAL_TABLE6_INDENT = BIOMISTRAL_TABLE6.replace("\nQUESTION:", "\n            QUESTION:") \
                                            .replace("\n[/INST]", "\n            [/INST]")

PROMPTS = {  # first entry per model is its default
    "qwen": ["notebook"],
    "biomistral": ["table6", "table6_indent", "notebook"],
}


def build_prompt(tokenizer, model_key, question, variant):
    """Return the full prompt string, exactly as the model will see it."""
    user_message = USER_MESSAGE_BEGIN + question + "\n"
    if model_key == "qwen":
        # Notebook: system message + user message through the chat API.
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if variant == "table6":
        return BIOMISTRAL_TABLE6.format(question=question)
    if variant == "table6_indent":
        return BIOMISTRAL_TABLE6_INDENT.format(question=question)
    # "notebook": the notebook's instruction wrapped in [INST] (our first run, 2026-11-17)
    return f"<s>[INST] {user_message.strip()} [/INST]"


def check_gpu_supports(precision):
    """Fail loudly rather than fall back to a different precision."""
    gpu = torch.cuda.get_device_properties(0)
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit(f"{gpu.name} has no native bf16 support - exclude this node "
                         "(needs an A30/A100/H100/L40S)")
    needed_gb = 18 if precision in ("bf16", "fp16") else 10  # 7B weights + working memory
    if gpu.total_memory / 1e9 < needed_gb:
        raise SystemExit(f"{gpu.name} has {gpu.total_memory / 1e9:.0f} GB, "
                         f"{precision} needs ~{needed_gb} GB - exclude this node")


def load_model(model_id, precision):
    kwargs = {"device_map": "cuda:0"}
    if precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.bfloat16 if precision == "bf16" else torch.float16
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--precision", choices=["bf16", "fp16", "8bit"],
                    help="override the per-model default (only for sensitivity experiments)")
    ap.add_argument("--prompt", choices=["notebook", "table6", "table6_indent"],
                    help="prompt variant (default: qwen=notebook, biomistral=table6)")
    # Diagnostics only - settings the authors may have used by accident (see diagnostics/biomistral_diagnostics.sh)
    ap.add_argument("--double-bos", action="store_true",
                    help="let the tokenizer add <s> on top of the literal <s> in the prompt")
    ap.add_argument("--sample", action="store_true",
                    help="random sampling (temperature 1.0, seed 0) instead of greedy decoding")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)  # paper: 512
    ap.add_argument("--limit", type=int, help="only the first N questions (for testing)")
    ap.add_argument("--data", default=str(HERE / "data/medchange_subset.csv"))
    ap.add_argument("--out", help="output .jsonl (default: repro/outputs/<model>_<precision>_<prompt>.jsonl)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "No GPU visible - run this inside a SLURM job with --gres=gpu"
    gpu = torch.cuda.get_device_properties(0)
    precision = args.precision or DEFAULT_PRECISION[args.model]
    check_gpu_supports(precision)
    model_id = MODELS[args.model]
    print(f"GPU: {gpu.name} ({gpu.total_memory / 1e9:.0f} GB) | model: {model_id} | precision: {precision}",
          flush=True)

    variant = args.prompt or PROMPTS[args.model][0]
    if variant not in PROMPTS[args.model]:
        raise SystemExit(f"prompt '{variant}' is not defined for {args.model}")
    tag = variant + ("_2bos" if args.double_bos else "") + ("_sample" if args.sample else "")
    out = Path(args.out or HERE / f"outputs/{args.model}_{precision}_{tag}"
               f"{'_limit' + str(args.limit) if args.limit else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    todo = df[~df.key.isin(done)].reset_index(drop=True)
    print(f"{len(df)} questions, {len(done)} already answered, {len(todo)} to go -> {out}", flush=True)
    if todo.empty:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # required for batched generation with decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    t0 = time.time()
    model = load_model(model_id, precision)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    prompts = [build_prompt(tokenizer, args.model, q, variant) for q in todo.question]
    print("---- example prompt ----\n" + prompts[0] + "\n------------------------", flush=True)

    # Batch questions of similar length together to minimise padding.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

    decoding = {"do_sample": False}
    if args.sample:
        decoding = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0}
        torch.manual_seed(0)
    meta = {"model_id": model_id, "precision": precision, "prompt": variant, "double_bos": args.double_bos,
            "gpu": gpu.name, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
            "decoding": "sampling T=1.0 seed 0" if args.sample else "greedy",
            "torch": torch.__version__, "transformers": transformers.__version__}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            enc = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True,
                            add_special_tokens=args.double_bos).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, **decoding,
                                     pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, text in zip(batch, texts):
                row = todo.iloc[i]
                f.write(json.dumps({"key": int(row.key), "question": row.question,
                                    "output": text.strip()}, ensure_ascii=False) + "\n")
            f.flush()
            n = start + len(batch)
            print(f"{n}/{len(order)} done, {time.time() - t0:.0f}s elapsed", flush=True)

    print(f"finished -> {out}", flush=True)


if __name__ == "__main__":
    main()
