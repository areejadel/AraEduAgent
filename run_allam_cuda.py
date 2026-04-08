"""
run_allam_cuda.py — ALLaM experiment runner for NVIDIA GPU (Vast.ai / cloud)
=============================================================================
Usage:
    pip install -r requirements_cuda.txt
    python3 run_allam_cuda.py --strategy 1 --start 0 --end 299
    python3 run_allam_cuda.py --all --start 0 --end 299
"""

import argparse, asyncio, json, os, sys, time, threading

# ─── 1. Load ALLaM (PyTorch + CUDA) ─────────────────────────────────────────

MAX_TOKENS = 4096
MODEL_ID   = "ALLaM-AI/ALLaM-7B-Instruct-preview"
_model = _tokenizer = None
_lock  = threading.Lock()   # one generation at a time

def load_allam():
    global _model, _tokenizer
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import os
    os.environ["HF_HOME"] = "/root/.hf_home"
    print(f"Loading ALLaM-7B from HuggingFace...")
    from transformers import LlamaTokenizer
    _tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",          # uses GPU automatically
        low_cpu_mem_usage=True,
    )
    _model.eval()
    device = next(_model.parameters()).device
    print(f"ALLaM ready on {device}")


def sanitize_llmlingua_output(text: str) -> str:
    """Remove XLM-RoBERTa artefacts that cause out-of-vocab CUDA asserts in ALLaM."""
    import re
    text = text.replace("\u2581", " ")                      # ▁ SentencePiece prefix
    text = re.sub(r"</?s>|<pad>|<mask>|<unk>", "", text)   # XLM-R special tokens
    text = re.sub(r"\[NEW\d+\]", "", text)                  # LLMLingua-2 placeholders
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def call_allam(messages, max_new_tokens=512, temperature=0.7):
    import torch
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    n_tokens = len(_tokenizer.encode(prompt))
    if n_tokens > MAX_TOKENS - 50:
        print(f"  OVERFLOW ({n_tokens} tokens)")
        return ""
    print(f"  {n_tokens}/{MAX_TOKENS} tokens")

    with _lock:
        inputs = _tokenizer([prompt], return_tensors="pt")
        # Guard: clamp any out-of-range token IDs to avoid CUDA assert
        vocab_size = _model.config.vocab_size
        input_ids = inputs["input_ids"]
        if (input_ids >= vocab_size).any() or (input_ids < 0).any():
            print(f"  WARNING: out-of-vocab token IDs found, clamping")
            inputs["input_ids"] = input_ids.clamp(0, vocab_size - 1)
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, MAX_TOKENS - n_tokens),
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=_tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return _tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── 2. Load simulation code ─────────────────────────────────────────────────

def load_sim_namespace():
    with open("simulation_experiment.ipynb", encoding="utf-8") as f:
        nb = json.load(f)
    src = "".join(nb["cells"][0]["source"])
    if 'if __name__ == "__main__":' in src:
        src = src[: src.index('if __name__ == "__main__":')]
    ns = {}
    exec(compile(src, "simulation_experiment.ipynb", "exec"), ns)
    return ns


def patch_namespace(ns):
    import types, concurrent.futures as _cf

    def allam_call_model(messages, model_type):
        if model_type == 7:
            return call_allam(messages)
        return ""

    ns["call_model"]     = allam_call_model
    ns["SELECTED_MODEL"] = 7
    for obj in ns.values():
        if isinstance(obj, types.FunctionType) and "call_model" in obj.__code__.co_names:
            obj.__globals__["call_model"] = allam_call_model
    Avatar = ns.get("Avatar")
    if Avatar:
        for attr in dir(Avatar):
            m = getattr(Avatar, attr, None)
            if callable(m) and hasattr(m, "__func__"):
                fn = m.__func__
                if "call_model" in getattr(fn.__code__, "co_names", ()):
                    fn.__globals__["call_model"] = allam_call_model

    # single-threaded executor — GPU calls must be serialized
    _orig = _cf.ThreadPoolExecutor
    class _SingleTPE(_orig):
        def __init__(self, max_workers=None, **kw):
            super().__init__(max_workers=1, **kw)
    ns["ThreadPoolExecutor"] = _SingleTPE
    for obj in list(ns.values()):
        if isinstance(obj, types.FunctionType):
            if "ThreadPoolExecutor" in getattr(obj.__code__, "co_names", ()):
                obj.__globals__["ThreadPoolExecutor"] = _SingleTPE


# ─── 3. Strategy configs ─────────────────────────────────────────────────────

EXAMPLE_USER_DICT = {
    "video_1": [167, 179, 153], "video_2": [590, 321, 327],
    "video_3": [366, 349, 342], "video_4": [436, 729, 696],
    "video_5": [798, 789, 507],
}
RESULT_PATH = "dataset/simulation/full_experiment_300_students"

STRATEGIES = {
    0: dict(label="S0 Baseline",
            gpt_type="allam_s0",         forget_effect="only_recent_one",
            memory_component_choice="KM+PM+MM+CM", sim_strategy="standard_cog",
            example_demo="no",           reflection_choice="no"),
    1: dict(label="S1 Dynamic Few-Shot",
            gpt_type="allam_s1",         forget_effect="no_memory",
            memory_component_choice="KM", sim_strategy="standard_cog",
            example_demo="no",           reflection_choice="no"),
    2: dict(label="S2 LLMLingua",
            gpt_type="allam_s2",         forget_effect="only_recent_one",
            memory_component_choice="KM+PM+MM+CM", sim_strategy="standard_cog",
            example_demo="no",           reflection_choice="no"),
    3: dict(label="S3 Memory-Free No Cognitive Prior",
            gpt_type="allam_s3",         forget_effect="no_memory",
            memory_component_choice="KM", sim_strategy="standard",
            example_demo="no",           reflection_choice="no"),
}

def make_config(s, source):
    c = {k: v for k, v in s.items() if k != "label"}
    c.update(dataset_path="dataset/", result_path=RESULT_PATH,
             memory_source=source, example_user_dict=EXAMPLE_USER_DICT)
    return c


# ─── 4. LLMLingua patch for S2 ───────────────────────────────────────────────

def patch_s2(ns):
    import types, concurrent.futures as _cf
    _compressor = None

    def get_compressor():
        nonlocal _compressor
        if _compressor is None:
            try:
                from llmlingua import PromptCompressor
                _compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                    use_llmlingua2=True,   # Fix: required for llmlingua-2 models
                    device_map="cuda",
                )
                print("LLMLingua loaded (llmlingua2 mode)")
            except ImportError:
                print("LLMLingua not installed — S2 falls back to S1 behaviour")
        return _compressor

    def s2_call_model(messages, model_type):
        if model_type != 7:
            return ""
        comp = get_compressor()
        if comp:
            user_text = messages[-1]["content"]
            result = comp.compress_prompt(
                user_text,
                target_token=3500,
                force_reserve_digit=True,
                drop_consecutive=True,
            )
            clean_text = sanitize_llmlingua_output(result["compressed_prompt"])
            messages = [messages[0], {"role": "user", "content": clean_text}]
        return call_allam(messages)

    ns["call_model"]     = s2_call_model
    ns["SELECTED_MODEL"] = 7
    for obj in ns.values():
        if isinstance(obj, types.FunctionType) and "call_model" in obj.__code__.co_names:
            obj.__globals__["call_model"] = s2_call_model

    _orig = _cf.ThreadPoolExecutor
    class _SingleTPE(_orig):
        def __init__(self, max_workers=None, **kw):
            super().__init__(max_workers=1, **kw)
    ns["ThreadPoolExecutor"] = _SingleTPE
    for obj in list(ns.values()):
        if isinstance(obj, types.FunctionType):
            if "ThreadPoolExecutor" in getattr(obj.__code__, "co_names", ()):
                obj.__globals__["ThreadPoolExecutor"] = _SingleTPE


# ─── 5. Main ─────────────────────────────────────────────────────────────────

async def run_one(strategy_id, start, end, ns):
    s = STRATEGIES[strategy_id]
    print(f"\n{'='*60}")
    print(f"{s['label']}  |  students {start}-{end}")
    print(f"{'='*60}")

    if strategy_id == 2:
        patch_s2(ns)
    else:
        patch_namespace(ns)

    run_exp = ns["run_exp"]
    results = {}
    for source in ["real", "sim"]:
        cfg = make_config(s, source)
        print(f"\n  > {source}  gpt_type={cfg['gpt_type']}")
        t0 = time.time()
        try:
            await run_exp(cfg, start, end)
            elapsed = round((time.time() - t0) / 60, 1)
            results[source] = {"ok": True, "min": elapsed}
            print(f"  DONE {source} in {elapsed} min")
        except Exception as e:
            elapsed = round((time.time() - t0) / 60, 1)
            results[source] = {"ok": False, "min": elapsed, "err": str(e)}
            print(f"  FAILED {source}: {e}")
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=9)
    args = parser.parse_args()

    if args.strategy is None and not args.all:
        parser.print_help(); sys.exit(1)

    strategies = [0, 1, 2, 3] if args.all else [args.strategy]

    load_allam()

    print("\nLoading simulation_experiment.ipynb...")
    ns = load_sim_namespace()
    print("Loaded")

    summary = {}
    for sid in strategies:
        summary[sid] = await run_one(sid, args.start, args.end, ns)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for sid, res in summary.items():
        for src, r in res.items():
            status = "OK" if r["ok"] else f"FAILED: {r.get('err','')[:40]}"
            print(f"  S{sid} {src:4} | {status} | {r['min']} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
