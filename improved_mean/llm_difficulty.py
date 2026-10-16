"""Feasibility: can a local LLM predict item difficulty on an *unseen* benchmark?

Follow-up to item_text_difficulty.py (bge-small embeddings: LOBO spearman ~0.03).
The LLM runs locally (no API); each item costs one forward pass, no generation:
- zero-shot: the model rates how likely a strong AI system is to solve the task
  (digit 0-9); the score is the expected digit under its next-token distribution;
- probe: hidden states of the same pass, ridge trained on the other benchmarks
  (same LOBO protocol and difficulty target as item_text_difficulty.py).
``eval`` also checks whether mean ratings order the benchmarks themselves by
difficulty, the budget-0 lever (report, Section 6.3), and compares models on shared items.

    CUDA_VISIBLE_DEVICES=<idle gpu> <gpu env python> llm_difficulty.py extract [--model M] [--subset] [--4bit]
    <venv python> llm_difficulty.py eval [--model M ...]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from item_text_difficulty import CACHE, ROOTS, load_items
from model import _group_key

MODEL = "Qwen/Qwen3-8B"
MAX_CONTENT = 2048  # content tokens kept; longer items keep head + tail
HEAD = 384          # researchcodebench puts the paper first and the TODO block last
SWE_REBENCH_SAMPLE = 1000  # one subject per item there: training rows only
SUBSET_CAPS = {"matharena": 900, "multi_swebench": 800, "swe_rebench": 400}  # --subset: ~1.5M prompt tokens
BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")

PROMPT = """Below is a task from an AI evaluation benchmark. A capable AI system (a frontier language model, possibly acting as an agent with tools) will attempt it, and the attempt is graded pass/fail.

<task>
{content}
</task>

Do not attempt the task yourself. How likely is a strong AI system to solve this task correctly? Answer with a single digit from 0 to 9, where 0 means it almost certainly fails and 9 means it almost certainly succeeds."""
ANSWER_PREFIX = "Rating: "  # prefilled so the next token is the digit (some tasks ask to be solved)


def cache_path(model, smoke=False):
    return f"{CACHE}/{model.replace('/', '_')}_llm{'_smoke' if smoke else ''}.npz"


def select_items(subset=False):
    df = load_items()
    swe = df[df.bench == "swe_rebench"].sample(n=SWE_REBENCH_SAMPLE, random_state=0)
    df = pd.concat([df[df.bench != "swe_rebench"], swe], ignore_index=True)
    df["key"] = df.bench + "/" + df.item_id.astype(str)
    if subset:  # the same rows for every model, so models compare item for item
        keep = [g.sample(n=min(len(g), SUBSET_CAPS.get(b, len(g))), random_state=0).index for b, g in df.groupby("bench")]
        df = df.loc[sorted(np.concatenate(keep))].reset_index(drop=True)
    return df


def truncate(text, tok, max_tokens=MAX_CONTENT, head=HEAD):
    """``text`` cut to about ``max_tokens``: the head plus the tail, or the part around the first TODO."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text, len(ids)
    tail = max_tokens - head
    # code-completion tasks: keep the block to write, not the paper (plain TODO comments also occur)
    todo = text.find("TODO: Implement block") if "TODO: Implement block" in text else text.find("TODO")
    at = len(tok(text[:todo], add_special_tokens=False).input_ids) if todo >= 0 else len(ids)
    after = min(512, tail // 3)  # tokens kept after the TODO (512 at the default sizes)
    start = len(ids) - tail if at >= len(ids) - tail else max(head, at - (tail - after))
    return tok.decode(ids[:head]) + "\n[...]\n" + tok.decode(ids[start:start + tail]), len(ids)


def build_prompts(df, tok):
    prompts, n_content, truncated = [], [], []
    for text in df.content.fillna(""):
        text, n = truncate(text, tok)
        n_content.append(n)
        truncated.append(n > MAX_CONTENT)
        msgs = [{"role": "user", "content": PROMPT.format(content=text)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                       + ANSWER_PREFIX)
    return prompts, np.array(n_content), np.array(truncated)


def extract(model_name=MODEL, subset=False, four_bit=False, smoke=None, token_budget=4096, complete=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = select_items(subset)
    if smoke:  # a few items per benchmark, separate cache file
        df = df.groupby("bench", sort=False).head(smoke).reset_index(drop=True)
    out_path = cache_path(model_name, smoke=bool(smoke))
    old = None
    if complete:  # only the items an earlier (e.g. --subset) run skipped; merged into its cache below
        old = dict(np.load(out_path, allow_pickle=True))
        df = df[~df.key.isin(set(old["keys"]))].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    assert all(tok.decode([t]) == str(d) for d, t in enumerate(digits)), "digits are not single tokens"
    prompts, n_content, truncated = build_prompts(df, tok)
    lengths = np.array([len(tok(p, add_special_tokens=False).input_ids) for p in prompts])
    print(f"{len(df)} items, {lengths.sum():,} prompt tokens, {truncated.sum()} truncated", flush=True)

    if four_bit:  # 32B-class models fit one 24 GB card only in 4 bit
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, quantization_config=q,
                                                     device_map={"": 0}).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    n_layers = model.config.num_hidden_layers
    layers = [n_layers // 2, 3 * n_layers // 4, n_layers]
    probs = np.zeros((len(df), 10), dtype=np.float32)
    mass = np.zeros(len(df), dtype=np.float32)
    last = {l: np.zeros((len(df), model.config.hidden_size), dtype=np.float16) for l in layers}
    mean = np.zeros((len(df), model.config.hidden_size), dtype=np.float16)

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
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            out = model(**enc, position_ids=pos, output_hidden_states=True, logits_to_keep=1)
            p = torch.softmax(out.logits[:, -1].float(), dim=-1)[:, digits]
            probs[idx] = p.cpu().numpy()
            mass[idx] = p.sum(-1).cpu().numpy()
            for l in layers:
                last[l][idx] = out.hidden_states[l][:, -1].float().cpu().numpy()
            m = enc.attention_mask.unsqueeze(-1).to(out.hidden_states[layers[0]].dtype)
            mean[idx] = ((out.hidden_states[layers[0]] * m).sum(1) / m.sum(1)).float().cpu().numpy()
            done += len(idx)
            if b % 25 == 0 or b == len(batches) - 1:
                el = time.time() - t0
                print(f"  {done}/{len(df)} items, {lengths[np.concatenate(batches[:b + 1])].sum() / el:,.0f} tok/s, {el / 60:.1f} min", flush=True)

    score = (probs * np.arange(10)).sum(1) / np.maximum(probs.sum(1), 1e-9)
    arrays = dict(keys=df.key.values, score=score, probs=probs, mass=mass, n_content=n_content,
                  truncated=truncated, mean_mid=mean, **{f"last_{l}": last[l] for l in layers})
    if old is not None:
        arrays = {k: np.concatenate([old[k], v]) for k, v in arrays.items()}
    tmp = out_path.replace(".npz", ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out_path)  # an interrupted run leaves the old cache intact
    print(f"saved {out_path} ({len(arrays['keys'])} items); digit mass median {np.median(mass):.3f}, min {mass.min():.3f}")


def _spearman_by_bench(df, pred, target, benches, within_group=False):
    from scipy.stats import spearmanr
    out = {}
    for b in benches:
        te = (df.bench == b).values
        p, y = pd.Series(pred[te]), pd.Series(target[te])
        if within_group:  # what is left after the item_features group means (v4/v5 learn those from labels)
            g = df.loc[te, "group"].values
            p, y = p - p.groupby(g).transform("mean"), y - y.groupby(g).transform("mean")
        out[b] = spearmanr(p, y).statistic
    return out


def _bench_dirs():
    for r in ROOTS:
        for b in BENCHES:
            if os.path.exists(f"{r}/{b}/items.parquet"):
                yield r, b


def _frame(keys):
    """Rows for ``keys`` (in that order) with item group, centered difficulty ``y`` and the testable benchmarks."""
    df = select_items().set_index("key").loc[list(keys)].reset_index()
    feats = pd.concat([pd.read_parquet(f"{r}/{b}/items.parquet", columns=["item_id", "item_features"]).assign(bench=b)
                       for r, b in _bench_dirs()], ignore_index=True)
    df = df.merge(feats, on=["bench", "item_id"], how="left")
    # v5's groups: numeric fields such as matharena's problem_idx are dropped, so a group is a contest, site, paper or language
    df["group"] = [f"{b}|{_group_key({'benchmark_id': b, 'item_features': f if isinstance(f, str) else ''})}" for b, f in zip(df.bench, df.item_features)]
    y = (df["diff"] - df.groupby("bench")["diff"].transform("mean")).values
    testable = [b for b in df.bench.unique() if df[df.bench == b].n_subj.median() >= 3]
    return df, y, testable


def _arrays(z, keys):
    pos = {k: i for i, k in enumerate(z["keys"])}
    idx = np.array([pos[k] for k in keys])
    return {f: z[f][idx] for f in z.files if f != "keys"}


def load_eval_data(model_name=MODEL):
    """Items with group and centered difficulty ``y``, the model's cached arrays, and the testable benchmarks."""
    z = np.load(cache_path(model_name), allow_pickle=True)
    df, y, testable = _frame(z["keys"])
    return df, _arrays(z, z["keys"]), y, testable


def lobo_probe(df, X, y, benches, lam):
    """Ridge predictions for each benchmark in ``benches`` from a fit on all other benchmarks."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros(len(df))
    for held in benches:
        tr, te = (df.bench != held).values, (df.bench == held).values
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=lam).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return pred


def bench_rates():
    """Mean over subjects of each subject's success rate, per training benchmark."""
    rates = {}
    for r, b in _bench_dirs():
        resp = pd.read_parquet(f"{r}/{b}/response.parquet", columns=["subject_id", "response"])
        resp = resp[resp.response.isin([0.0, 1.0])]
        rates[b] = resp.groupby("subject_id").response.mean().mean()
    return pd.Series(rates)


def evaluate(models):
    from scipy.stats import spearmanr

    caches = {m: np.load(cache_path(m), allow_pickle=True) for m in models}
    shared = set.intersection(*(set(z["keys"]) for z in caches.values()))
    keys = [k for k in caches[models[0]]["keys"] if k in shared]
    df, y, testable = _frame(keys)
    arrays = {m: _arrays(z, keys) for m, z in caches.items()}
    print(f"{len(keys)} shared items\n" + df.groupby("bench").agg(items=("diff", "size"), subjects=("n_subj", "median")).to_string())

    print("\nBenchmark level: mean rating (0-9) vs actual mean success rate in the training data")
    tab = pd.DataFrame({"actual": bench_rates()})
    for m in models:
        tab[m.split("/")[-1]] = pd.Series(arrays[m]["score"]).groupby(df.bench.values).mean()
    print(tab.sort_values("actual").round(3).to_string())
    for m in models:
        c = tab[m.split("/")[-1]]
        print(f"  {m}: Spearman across benchmarks {spearmanr(c, tab.actual).statistic:+.2f}, "
              f"rating spread {c.max() - c.min():.2f}, digit mass median {np.median(arrays[m]['mass']):.3f}")

    # five benchmarks are few points; item_features groups (contest, website, paper, language) add
    # many more clusters whose overall difficulty must be judged without labels, as at budget 0
    print("\nGroup level: Spearman of group-mean rating vs group-mean difficulty, within each benchmark (groups >= 5 items)")
    sizes = df.groupby("group").size()
    big = df.group.map(sizes).values >= 5
    for m in models:
        g = pd.DataFrame({"bench": df.bench, "group": df.group, "rating": arrays[m]["score"], "y": y})[big]
        g = g.groupby(["bench", "group"])[["rating", "y"]].mean().reset_index()
        rho = {b: (spearmanr(s.rating, s.y).statistic, len(s)) for b, s in g.groupby("bench") if b in testable and len(s) >= 4}
        print(f"  {m.split('/')[-1]:<10} " + " ".join(f"{b[:12]}={r:+.2f}(n={n})" for b, (r, n) in rho.items())
              + f" | mean={np.mean([r for r, _ in rho.values()]):+.2f}")

    print("\nItem level: Spearman with difficulty within each held-out benchmark (higher = easier)")
    for m in models:
        a = arrays[m]
        rows = [("zero-shot", a["score"])]
        rows += [(f"probe {f} a={lam:g}", lobo_probe(df, a[f].astype(np.float32), y, testable, lam))
                 for f in a if f.startswith(("last_", "mean_")) for lam in (1e3, 1e4)]
        for name, pred in rows:
            r, w = _spearman_by_bench(df, pred, y, testable), _spearman_by_bench(df, pred, y, testable, True)
            print(f"  {m.split('/')[-1]:<10} {name:<22} " + " ".join(f"{b[:12]}={r[b]:+.3f}" for b in testable)
                  + f" | mean={np.mean(list(r.values())):+.3f} within-group={np.mean(list(w.values())):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "eval"])
    ap.add_argument("--model", action="append", help="HF repo; eval takes several and compares them on shared items")
    ap.add_argument("--subset", action="store_true", help="extract SUBSET_CAPS rows per benchmark")
    ap.add_argument("--4bit", dest="four_bit", action="store_true", help="load in 4 bit (bitsandbytes)")
    ap.add_argument("--smoke", type=int, help="extract only this many items per benchmark")
    ap.add_argument("--token-budget", type=int, default=4096, help="padded tokens per batch")
    ap.add_argument("--complete", action="store_true", help="extract only items missing from the model's cache")
    a = ap.parse_args()
    models = a.model or [MODEL]
    if a.cmd == "extract":
        extract(models[0], a.subset, a.four_bit, a.smoke, a.token_budget, a.complete)
    else:
        evaluate(models)
