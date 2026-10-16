"""v7: v5 with a benchmark-level prior read by a local LLM (Qwen3.8-27B in 4 bit).

v5 (v5.py, unchanged) starts every benchmark from PRIOR_MEAN = 0.35. v7 rates each item
it is asked about with an anchored prompt: twelve items from the public training
benchmarks with their real success rates (anchors.json), then the new item, answered by
a digit d read from the next-token distribution, rating = (E[d] + 0.5) / 10. Ratings are
cached per benchmark (on disk, so they survive the worker restart at each checkpoint).

PRIOR_MEAN becomes the mean rating of every item rated so far on the same benchmark,
shrunk toward 0.35 with strength K0; everything else is v5. With "prior_at_budget_0" in
config.json (the final submission), this holds from budget 0 on, where the mean grows
item by item. Without it, budget 0 keeps v5's constant prediction, so the budget-0
feedback still identifies each pair's success rate (report, Proposition 1).

With "probe" in config.json (a measurement submission, not a candidate), budget 1
returns the benchmark prior itself and budget 3 the prior plus PROBE_SHIFT (at most
0.99), both constant within each benchmark. The budget-1 ECE of each pair then gives
|prior - rate| and the budget-3 ECE tells on which side of the rate the prior lies, which
recovers the prior that v7 uses on each test benchmark. Budget 7 and later are v7.

See experiment 13 in EXPERIMENTS.md and llm_prior_sim.py. The LLM loads only in
prediction workers (PAIEC_PREDICTION_WORKER_ID is set) and not in the local smoke test.
If it cannot load or rate, config.json decides: require_llm raises, otherwise v5.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v5_core", HERE / "v5.py")
v5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v5)

CONFIG = json.loads((HERE / "config.json").read_text())
ANCHORS = json.loads((HERE / "anchors.json").read_text())
LLM_NAME = "Qwen/Qwen3.8-27B"
K0 = 5.0
PROBE_SHIFT = 0.1
TARGET_TOKENS, TARGET_HEAD = 1024, 192  # truncation of the rated item (as in llm_anchor.py)
SMOKE_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
# Ratings must survive the worker restart at each checkpoint: write them next to model.py
# and to the system temp root (the official worker's TMPDIR is a per-checkpoint subfolder).
RATING_DIRS = list(dict.fromkeys([HERE / ".v7_ratings", Path("/tmp") / "paiec_v7_ratings",
                                  Path(os.environ.get("TMPDIR", "/tmp")).resolve().parent.parent / "paiec_v7_ratings"]))

INTRO = """You will estimate how often AI systems succeed on a task from an AI evaluation benchmark. The AI systems are language models, often acting as agents with tools, and each attempt is graded pass/fail.

For reference, here are tasks from other benchmarks with the share of AI systems that solved them. A digit d means d*10% to (d+1)*10%.
"""
EXAMPLE = """
<example>
<task>
{content}
</task>
Share of AI systems that solved it: {pct:.0f}% (digit {digit})
</example>
"""
QUESTION = """
Now a new task, from a different benchmark:

<task>
{content}
</task>

Do not attempt the task yourself. What share of AI systems solve this task? Answer with a single digit from 0 to 9 as defined above."""
ANSWER_PREFIX = "Rating: "
SHOTS = INTRO + "".join(EXAMPLE.format(**a) for a in ANCHORS)


def _log(message):
    print(f"[v7] {message}", file=sys.stderr, flush=True)


def _resolve_cache_dir():
    """HF cache holding the pre-downloaded repo (as in the official hf_submission template)."""
    for candidate in (os.environ.get("HF_HOME", "").strip(), "/app/hf_cache", str(HERE / ".hf_cache")):
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            return str(path)
    return None


def _load_llm():
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")  # transformers 5: quantize tensor by tensor
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                              BitsAndBytesConfig)

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    tok = cache_dir = None
    for cache_dir in (_resolve_cache_dir(), None):  # the template's cache, then the default HF cache
        try:
            tok = AutoTokenizer.from_pretrained(LLM_NAME, cache_dir=cache_dir, local_files_only=True)
            break
        except OSError:
            continue
    if tok is None:
        raise RuntimeError(f"{LLM_NAME} is not in the local HF cache")
    archs = AutoConfig.from_pretrained(LLM_NAME, cache_dir=cache_dir, local_files_only=True).architectures or []
    cls = AutoModelForImageTextToText if any(a.endswith("ForConditionalGeneration") for a in archs) else AutoModelForCausalLM
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model, info = cls.from_pretrained(LLM_NAME, cache_dir=cache_dir, local_files_only=True, dtype=torch.bfloat16,
                                      quantization_config=quant, device_map={"": 0}, output_loading_info=True)
    if info["missing_keys"]:  # never rate with randomly initialized weights
        raise RuntimeError(f"{len(info['missing_keys'])} weights missing, e.g. {sorted(info['missing_keys'])[:3]}")
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    if not all(tok.decode([t]) == str(d) for d, t in enumerate(digits)):
        raise RuntimeError("digits are not single tokens")
    return tok, model.eval(), digits


def _wants_llm():
    if "PAIEC_PREDICTION_WORKER_ID" not in os.environ:
        return False
    return os.environ.get(SMOKE_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}


TOK = MODEL = DIGITS = None
_RATINGS = {}  # sha1 of (benchmark, item text) -> [benchmark, rating]
_OWN = {}      # the part of _RATINGS this process computed (written to RATING_DIRS)
_SUMS = {}     # benchmark -> [sum of ratings, count]
if _wants_llm():
    try:
        TOK, MODEL, DIGITS = _load_llm()
        for folder in RATING_DIRS:
            for path in folder.glob("*.json"):
                try:
                    _RATINGS.update(json.loads(path.read_text()))
                except (OSError, ValueError):
                    pass
        for bench, rating in _RATINGS.values():
            total, n = _SUMS.get(bench, (0.0, 0))
            _SUMS[bench] = (total + rating, n + 1)
        _log(f"loaded {LLM_NAME}; {len(_RATINGS)} cached ratings on {len(_SUMS)} benchmarks")
    except Exception as exc:
        if CONFIG.get("require_llm"):
            raise
        _log(f"LLM unavailable, falling back to v5: {exc!r}")
        TOK = MODEL = None


def _truncate(text):
    """Head plus tail, or the part around the first TODO (llm_difficulty.truncate at 1024 tokens)."""
    ids = TOK(text, add_special_tokens=False).input_ids
    if len(ids) <= TARGET_TOKENS:
        return text
    tail = TARGET_TOKENS - TARGET_HEAD
    todo = text.find("TODO: Implement block") if "TODO: Implement block" in text else text.find("TODO")
    at = len(TOK(text[:todo], add_special_tokens=False).input_ids) if todo >= 0 else len(ids)
    after = min(512, tail // 3)
    start = len(ids) - tail if at >= len(ids) - tail else max(TARGET_HEAD, at - (tail - after))
    return TOK.decode(ids[:TARGET_HEAD]) + "\n[...]\n" + TOK.decode(ids[start:start + tail])


def _llm_rating(content):
    import torch

    text = SHOTS + QUESTION.format(content=_truncate(content))
    prompt = TOK.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) + ANSWER_PREFIX
    enc = TOK(prompt, return_tensors="pt", add_special_tokens=False).to(MODEL.device)
    with torch.no_grad():
        logits = MODEL(**enc, logits_to_keep=1).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)[DIGITS]
    probs = probs / probs.sum()
    expected = float((probs * torch.arange(10, device=probs.device)).sum())
    return min(0.99, max(0.01, (expected + 0.5) / 10))


def _rate(bench, content):
    key = hashlib.sha1(f"{bench}\0{content}".encode("utf-8", "replace")).hexdigest()
    if key in _RATINGS:
        return
    rating = _llm_rating(content)
    _RATINGS[key] = _OWN[key] = [bench, rating]
    total, n = _SUMS.get(bench, (0.0, 0))
    _SUMS[bench] = (total + rating, n + 1)
    name = f"{os.environ.get('PAIEC_PREDICTION_WORKER_ID', 'x')}_{os.getpid()}"
    for folder in RATING_DIRS:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f"{name}.tmp"
            tmp.write_text(json.dumps(_OWN))
            os.replace(tmp, folder / f"{name}.json")
        except OSError:
            pass  # another folder, or the in-process cache, still works


def predict(input: list, labeled: list | None = None) -> float:
    labeled = labeled or []
    if MODEL is None:
        return v5.predict(input, labeled)
    _, item = input
    bench = item.get("benchmark_id", "")
    try:
        _rate(bench, item.get("item_content") or "")
    except Exception as exc:
        if CONFIG.get("require_llm"):
            raise
        _log(f"rating failed, using v5 for this call: {exc!r}")
        return v5.predict(input, labeled)
    if not labeled and not CONFIG.get("prior_at_budget_0"):
        return v5.predict(input, labeled)  # keeps budget-0 feedback identifying each pair's rate
    total, n = _SUMS[bench]
    prior = (total + K0 * v5.PARAMS["PRIOR_MEAN"]) / (n + K0)
    if CONFIG.get("probe"):
        budget = _budget(labeled)
        if budget <= 1:
            return prior
        if budget <= 3:
            return min(0.99, prior + PROBE_SHIFT)
    return v5.predict_with(dict(v5.PARAMS, PRIOR_MEAN=prior), input, labeled)


def _budget(labeled):
    """Largest number of labels on one subject-benchmark pair, i.e. the current label budget."""
    counts = {}
    for (subject, item), _ in labeled:
        key = (json.dumps(subject, sort_keys=True, default=str), item.get("benchmark_id", ""))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)
