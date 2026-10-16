"""v6: v5 plus item difficulty read by a local LLM (Qwen3-32B in 4 bit).

v5 (v5.py, unchanged) estimates each subject-benchmark rate from the labels. v6
shifts that estimate in logit space by GAMMA * (s_i - c), where s_i is the item's
difficulty score (a ridge probe on the LLM's last hidden state, fitted offline on
the public training data and shipped in probe.npz) and c is the mean score of the
labeled items on the same benchmark plus this one. Without labels on the benchmark
(budget 0) nothing changes; with labels, items that look easier than the labeled
ones move up and harder ones down. See experiment 11 in EXPERIMENTS.md.

The LLM loads only in prediction workers (PAIEC_PREDICTION_WORKER_ID is set), not
in the parent process, which imports this file but never predicts without a
labeling.py, and not in the local smoke tests. If it cannot load or score,
config.json decides: require_llm raises, otherwise predictions fall back to v5.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v5_core", HERE / "v5.py")
v5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v5)

CONFIG = json.loads((HERE / "config.json").read_text())
PROBE = np.load(HERE / "probe.npz")
LLM_NAME = "Qwen/Qwen3-32B"
GAMMA = 0.5
MAX_CONTENT, HEAD = 2048, 384  # truncation used when the probe was fitted (llm_difficulty.py)
TOKEN_BUDGET = 2048            # padded tokens per batch
SMOKE_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
SCORE_DIR = HERE / ".v6_scores"  # item scores shared across the per-checkpoint worker restarts

PROMPT = """Below is a task from an AI evaluation benchmark. A capable AI system (a frontier language model, possibly acting as an agent with tools) will attempt it, and the attempt is graded pass/fail.

<task>
{content}
</task>

Do not attempt the task yourself. How likely is a strong AI system to solve this task correctly? Answer with a single digit from 0 to 9, where 0 means it almost certainly fails and 9 means it almost certainly succeeds."""
ANSWER_PREFIX = "Rating: "


def _log(message):
    print(f"[v6] {message}", file=sys.stderr, flush=True)


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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME, cache_dir=cache_dir, local_files_only=True,
                                                 torch_dtype=torch.bfloat16, quantization_config=quant,
                                                 device_map={"": 0}).eval()
    return tok, model


def _wants_llm():
    if "PAIEC_PREDICTION_WORKER_ID" not in os.environ:
        return False
    return os.environ.get(SMOKE_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}


TOK = MODEL = None
_SCORES = {}  # sha1 of item_content -> standardized score
_OWN = {}     # the part of _SCORES this process computed (written to SCORE_DIR)
if _wants_llm():
    try:
        TOK, MODEL = _load_llm()
        for path in SCORE_DIR.glob("*.json"):
            try:
                _SCORES.update(json.loads(path.read_text()))
            except (OSError, ValueError):
                pass
        _log(f"loaded {LLM_NAME}; {len(_SCORES)} cached item scores")
    except Exception as exc:
        if CONFIG.get("require_llm"):
            raise
        _log(f"LLM unavailable, falling back to v5: {exc!r}")
        TOK = MODEL = None


def _truncate(text):
    """Head plus tail, or the part around the first TODO (same rule as the offline features)."""
    ids = TOK(text, add_special_tokens=False).input_ids
    if len(ids) <= MAX_CONTENT:
        return text
    tail = MAX_CONTENT - HEAD
    todo = text.find("TODO: Implement block") if "TODO: Implement block" in text else text.find("TODO")
    at = len(TOK(text[:todo], add_special_tokens=False).input_ids) if todo >= 0 else len(ids)
    after = min(512, tail // 3)
    start = len(ids) - tail if at >= len(ids) - tail else max(HEAD, at - (tail - after))
    return TOK.decode(ids[:HEAD]) + "\n[...]\n" + TOK.decode(ids[start:start + tail])


def _prompt(content):
    msgs = [{"role": "user", "content": PROMPT.format(content=_truncate(content))}]
    return TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + ANSWER_PREFIX


def _compute_scores(contents):
    import torch

    prompts = [_prompt(c) for c in contents]
    lengths = np.array([len(TOK(p, add_special_tokens=False).input_ids) for p in prompts])
    batches, cur = [], []
    for i in np.argsort(lengths):
        if cur and (len(cur) + 1) * lengths[i] > TOKEN_BUDGET:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    feats = np.zeros((len(prompts), PROBE["coef"].shape[0]), dtype=np.float32)
    with torch.no_grad():
        for idx in batches:
            enc = TOK([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(MODEL.device)
            pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            hidden = MODEL.model(**enc, position_ids=pos).last_hidden_state[:, -1]
            feats[idx] = hidden.float().cpu().numpy().astype(np.float16)  # features were cached in float16
    z = (feats - PROBE["mean"]) / PROBE["scale"]
    return (z @ PROBE["coef"] + PROBE["intercept"]) / PROBE["sd"]


def _scores(contents):
    keys = [hashlib.sha1(c.encode("utf-8", "replace")).hexdigest() for c in contents]
    missing = {k: c for k, c in zip(keys, contents) if k not in _SCORES}
    if missing:
        new = dict(zip(missing, map(float, _compute_scores(list(missing.values())))))
        _SCORES.update(new)
        _OWN.update(new)
        try:
            SCORE_DIR.mkdir(exist_ok=True)
            name = f"{os.environ.get('PAIEC_PREDICTION_WORKER_ID', 'x')}_{os.getpid()}"
            tmp = SCORE_DIR / f"{name}.tmp"
            tmp.write_text(json.dumps(_OWN))
            os.replace(tmp, SCORE_DIR / f"{name}.json")
        except OSError:
            pass  # the in-process cache still works
    return [_SCORES[k] for k in keys]


def predict(input: list, labeled: list | None = None) -> float:
    labeled = labeled or []
    p = v5.predict(input, labeled)
    if MODEL is None:
        return p
    _, item = input
    bench = item.get("benchmark_id", "")
    same = [it.get("item_content") or "" for (_, it), _ in labeled if it.get("benchmark_id", "") == bench]
    if not same:  # no labels on this benchmark yet: c would equal s_i, so no shift
        return p
    try:
        s = _scores(same + [item.get("item_content") or ""])
    except Exception as exc:
        if CONFIG.get("require_llm"):
            raise
        _log(f"scoring failed, using v5 for this call: {exc!r}")
        return p
    c = sum(s) / len(s)  # labeled items of this benchmark plus the target
    q = min(1 - 1e-4, max(1e-4, p))
    shifted = 1.0 / (1.0 + math.exp(-(math.log(q / (1 - q)) + GAMMA * (s[-1] - c))))
    clip = v5.PARAMS["CLIP"]
    return float(min(1 - clip, max(clip, shifted)))
