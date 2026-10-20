"""Can an LLM judge how hard a whole *unseen* benchmark is, given anchor tasks?

Experiments 7-8: asked for a bare 0-9 rating, Qwen3 rates every benchmark about 7
and cannot tell which one is harder, which is where v5 loses most (budget 0-3,
about 0.019 ALC). Here the prompt first shows anchor tasks from the *other* public
benchmarks with their real success rates (items at the 20th, 50th and 80th
percentile of each), then asks for the new task's success rate as a digit d
(d*10% to (d+1)*10%). Anchors never come from the benchmark being rated, as at
test time, where all public benchmarks would serve as anchors.

``eval`` reports, per held-out benchmark, the predicted vs actual mean success rate
and the budget-0 Brier of p = (E[d] + 0.5) / 10 against v5's constant 0.35.

    CUDA_VISIBLE_DEVICES=<idle gpu> <gpu env python> llm_anchor.py extract [--model M] [--4bit]
    <venv python> llm_anchor.py eval [--model M]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from item_text_difficulty import CACHE
from llm_difficulty import ANSWER_PREFIX, MODEL, _bench_dirs, truncate

BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")  # several subjects per item
ANCHOR_QUANTILES = (0.2, 0.5, 0.8)
ANCHOR_TOKENS, ANCHOR_HEAD = 256, 64
TARGET_TOKENS, TARGET_HEAD = 1024, 192
N_TARGET = 120  # items rated per held-out benchmark
PRIOR = 0.35    # v5's budget-0 prediction

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


def cache_path(model):
    return f"{CACHE}/{model.replace('/', '_')}_anchor.npz"


def item_rates():
    """Items of BENCHES with content, success rate (mean over subjects) and subject count."""
    rows = []
    for r, b in _bench_dirs():
        if b not in BENCHES:
            continue
        resp = pd.read_parquet(f"{r}/{b}/response.parquet", columns=["subject_id", "item_id", "response"])
        resp = resp[resp.response.isin([0.0, 1.0])]
        per = resp.groupby(["item_id", "subject_id"]).response.mean().groupby("item_id").agg(rate="mean", n_subj="size")
        items = pd.read_parquet(f"{r}/{b}/items.parquet", columns=["item_id", "content"])
        rows.append(items.merge(per, left_on="item_id", right_index=True).assign(bench=b))
    df = pd.concat(rows, ignore_index=True)
    df["key"] = df.bench + "/" + df.item_id.astype(str)
    return df


def anchors_and_targets(df):
    anchors, targets = {}, {}
    for b, g in df.groupby("bench"):
        pool = g[g.n_subj >= 5].sort_values(["rate", "key"])
        anchors[b] = [pool.iloc[int(q * (len(pool) - 1))] for q in ANCHOR_QUANTILES]
        used = {a.key for a in anchors[b]}
        targets[b] = g[(g.n_subj >= 3) & ~g.key.isin(used)].sample(n=N_TARGET, random_state=0)
    return anchors, targets


_EXCERPTS = {}  # anchor key -> truncated text (anchors repeat in every prompt)


def build_prompt(tok, anchors, held, content):
    shots = [a for b in BENCHES if b != held for a in anchors[b]]
    np.random.default_rng(0).shuffle(shots)  # same order for every target of a benchmark
    text = INTRO
    for a in shots:
        if a.key not in _EXCERPTS:
            _EXCERPTS[a.key] = truncate(a.content if isinstance(a.content, str) else "", tok, ANCHOR_TOKENS, ANCHOR_HEAD)[0]
        text += EXAMPLE.format(content=_EXCERPTS[a.key], pct=100 * a.rate, digit=min(9, int(a.rate * 10)))
    excerpt, _ = truncate(content if isinstance(content, str) else "", tok, TARGET_TOKENS, TARGET_HEAD)
    text += QUESTION.format(content=excerpt)
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + ANSWER_PREFIX


RECURRENT_TYPES = {"qwen3_5", "qwen3_5_moe", "qwen3_next"}  # linear-attention layers carry state across tokens


def load_model(model_name, four_bit):
    """bf16 (or 4-bit) model on GPU 0; multimodal checkpoints (Qwen3.5 and later) load whole and get text only."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    kw = {"dtype": torch.bfloat16, "output_loading_info": True}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
                  device_map={"": 0})
    archs = AutoConfig.from_pretrained(model_name).architectures or []
    cls = AutoModelForImageTextToText if any(a.endswith("ForConditionalGeneration") for a in archs) else AutoModelForCausalLM
    model, info = cls.from_pretrained(model_name, **kw)
    if info["missing_keys"]:  # never score with randomly initialized weights
        raise RuntimeError(f"{len(info['missing_keys'])} weights missing, e.g. {sorted(info['missing_keys'])[:3]}")
    print(f"loaded {cls.__name__}; unexpected keys: {len(info['unexpected_keys'])}", flush=True)
    return (model if four_bit else model.to("cuda")).eval()


def extract(model_name, four_bit, token_budget):
    import torch
    from transformers import AutoTokenizer

    df = item_rates()
    anchors, targets = anchors_and_targets(df)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    assert all(tok.decode([t]) == str(d) for d, t in enumerate(digits)), "digits are not single tokens"
    rows = [(b, t.key, build_prompt(tok, anchors, b, t.content)) for b in BENCHES for t in targets[b].itertuples()]
    lengths = np.array([len(tok(p, add_special_tokens=False).input_ids) for _, _, p in rows])
    print(f"{len(rows)} prompts, {lengths.sum():,} tokens (median {int(np.median(lengths))})", flush=True)

    model = load_model(model_name, four_bit)
    if getattr(model.config, "model_type", "") in RECURRENT_TYPES:
        token_budget = 1  # one prompt per batch: padding would run through the recurrent layers

    probs = np.zeros((len(rows), 10), dtype=np.float32)
    batches, cur = [], []
    for i in np.argsort(lengths):
        if cur and (len(cur) + 1) * lengths[i] > token_budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    t0, done = time.time(), 0
    with torch.no_grad():
        for b, idx in enumerate(batches):
            enc = tok([rows[i][2] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            extra = {}
            if len(idx) > 1:  # left padding: positions must skip the pad tokens
                extra["position_ids"] = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            out = model(**enc, **extra, logits_to_keep=1)
            probs[idx] = torch.softmax(out.logits[:, -1].float(), dim=-1)[:, digits].cpu().numpy()
            done += len(idx)
            if b % 25 == 0 or b == len(batches) - 1:
                el = time.time() - t0
                print(f"  {done}/{len(rows)} prompts, {lengths[np.concatenate(batches[:b + 1])].sum() / el:,.0f} tok/s, {el / 60:.1f} min", flush=True)
    np.savez(cache_path(model_name), keys=np.array([k for _, k, _ in rows]), bench=np.array([b for b, _, _ in rows]),
             probs=probs, n_tokens=lengths)
    print(f"saved {cache_path(model_name)}; digit mass median {np.median(probs.sum(1)):.3f}, min {probs.sum(1).min():.3f}")


def evaluate(model_name):
    from scipy.stats import spearmanr

    z = np.load(cache_path(model_name), allow_pickle=True)
    probs = z["probs"] / z["probs"].sum(1, keepdims=True)
    p = pd.Series(np.clip(((probs * np.arange(10)).sum(1) + 0.5) / 10, 0.01, 0.99), index=z["keys"])
    df = item_rates().set_index("key")
    resp = []
    for r, b in _bench_dirs():
        if b in BENCHES:
            x = pd.read_parquet(f"{r}/{b}/response.parquet", columns=["item_id", "response"])
            x = x[x.response.isin([0.0, 1.0])]
            resp.append(x.assign(key=b + "/" + x.item_id.astype(str)))
    resp = pd.concat(resp, ignore_index=True)
    resp = resp[resp.key.isin(p.index)]
    resp["p"] = p.loc[resp.key].values
    resp["bench"] = resp.key.str.split("/").str[0]

    print(f"{model_name}: anchors from the other benchmarks; p = (E[digit] + 0.5) / 10")
    print(f"{'held-out benchmark':<20}{'actual':>8}{'pred':>8}   budget-0 Brier: {'0.35':>7}{'LLM item':>10}{'LLM mean':>10}{'oracle':>8}   item Spearman")
    tot = np.zeros(4)
    for b in BENCHES:
        g = resp[resp.bench == b]
        actual, pred = g.response.mean(), g.p.mean()
        briers = [((PRIOR - g.response) ** 2).mean(), ((g.p - g.response) ** 2).mean(),
                  ((pred - g.response) ** 2).mean(), ((actual - g.response) ** 2).mean()]
        tot += briers
        keys = p.index[np.array([k.startswith(b + "/") for k in p.index])]
        rho = spearmanr(p.loc[keys], df.loc[keys, "rate"]).statistic
        print(f"{b:<20}{actual:>8.3f}{pred:>8.3f}   {'':16}" + "".join(f"{v:>{w}.4f}" for v, w in zip(briers, (7, 10, 10, 8)))
              + f"   {rho:+.3f}")
    print(f"{'mean':<36}{'':16}" + "".join(f"{v:>{w}.4f}" for v, w in zip(tot / len(BENCHES), (7, 10, 10, 8))))
    # single items: each item's own rating shrunk toward 0.35, p = 0.35 + w (rating - 0.35)
    ws = (0.0, 0.25, 0.5, 0.75, 1.0)
    per = {b: [((PRIOR + w * (g.p - PRIOR) - g.response) ** 2).mean() for w in ws] for b, g in resp.groupby("bench")}
    mean = np.mean([per[b] for b in BENCHES], axis=0)
    lobo = [per[b][int(np.argmin(np.mean([per[o] for o in BENCHES if o != b], axis=0)))] for b in BENCHES]
    print(f"single item, w in {ws}: in hindsight w = {ws[int(np.argmin(mean))]} gives {mean.min():.4f}; "
          f"w chosen leave-one-benchmark-out gives {np.mean(lobo):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "eval"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--token-budget", type=int, default=8192, help="padded tokens per batch")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.model, a.four_bit, a.token_budget)
    else:
        evaluate(a.model)
