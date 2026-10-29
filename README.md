# How Far Can Labels Go? Code for our PAIEC 2026 report

This repository contains the code for the technical report *How Far Can Labels Go? Benchmark Difficulty Is the Bottleneck of Cold-Start Predictive Evaluation*. The report is our entry to the Predictive AI Evaluation Competition (PAIEC, NeurIPS 2026); our Codabench user is kkk61.

- **HBS** (hierarchical Beta-Binomial shrinkage) is a label-only predictor. It is one file, `improved_mean/model.py`, uses only the Python standard library, and loads no model.
- **HBS+Prior** replaces the constant prior mean of HBS with a benchmark prior: the average of Qwen3.8-27B's ratings of the benchmark's items. Its code is in `improved_mean/v7/`, and `improved_mean/build_v7.py` builds the submission archive.
- Our **final submission** is HBS+Prior with the settings of Section 6.4: the prior is spread by a factor of 1.5 around 0.35, alpha = 4, and the global rate is fixed at the prior. Its code is in `improved_mean/v8/`, and `improved_mean/build_v8.py` builds the archive.
- The remaining scripts reproduce the analyses:
  - the real-feedback proxy (Section 5),
  - the experiments (Section 6),
  - the robustness checks (Appendix E).

## Method names, file names and submissions

File names use our internal version numbers; the report uses method names.

| Report | Code | Codabench ID |
|---|---|---|
| Early submissions | not retained | 928061, 930136, 930165, 931142 |
| Smoothed mean | `versions/model_v1.py` | 951748 |
| HBS+Ability | `versions/model_v4.py` | 951853 |
| HBS | `model.py` (same as `versions/model_v5.py`) | 952533 |
| HBS+ItemLLM | `v6/`, `build_v6.py` | 960385 |
| Measurement submission | `v7/` with `probe` set, built by `build_v7.py` | 978024 |
| HBS+Prior, settings of HBS | `v7/`, `build_v7.py` | 978186 |
| **HBS+Prior (final)** | `v8/`, `build_v8.py` | **1013135** |

Notes on the submitted code:
- `model.py` still contains the subject-ability shift of HBS+Ability. It is switched off (`BETA = 0`).
- `model.py`, `v6/model.py`, `v7/model.py` and `v8/model.py` are byte-identical to the code in the submitted archives. Their docstrings mention our lab notes (`EXPERIMENTS.md`), which are not released.
- The measurement submission (978024) was built from an earlier revision of `v7/model.py`. That revision differs only in a docstring and in the `prior_at_budget_0` switch, which the measurement build leaves off. Rebuilding it therefore gives the same predictions.

## Setup

1. **Python environments.** We used two environments:
   - *Analyses:* Python 3.13. Install `pip install -r requirements.txt`.
   - *LLM experiments, building and testing HBS+Prior:* Python 3.10 with a CUDA GPU (we used one 24 GB RTX 3090). Install `pip install -r requirements.txt -r requirements-llm.txt`.
2. **Training data.** The data are measurement-db, revision bc8204d. The dataset is gated: accept its terms on Hugging Face and log in first.
   1. Clone the official baseline repository into this repository's root as `paiec_baseline/`. Alternatively, set `PAIEC_BASELINE` to its location.
   2. Inside it, run:

   ```
   python tools/prepare_data.py --out payload --no-embeddings
   python - <<'EOF'
   from huggingface_hub import snapshot_download
   snapshot_download("aims-foundations/measurement-db", repo_type="dataset",
       revision="bc8204d811823da849c6686bf124d4ca9f82e4de",
       allow_patterns=["matharena/*.parquet"], local_dir="payload_extra")
   EOF
   ```

   The preparation script excludes `matharena`, so it is fetched separately. Our loaders drop its roughly 60 non-binary responses.
3. **Official tools.** The end-to-end tests use `tools/` from the competition starting kit: `prediction_workers.py` and `check_submission_zip.py`. Copy that folder into this repository's root, or set `PAIEC_TOOLS`.
4. **Formative logs.** The proxy analyses read Codabench scoring logs.
   - Save each submission's "Scoring output" as `improved_mean/feedback/<submission id>_scoring_stdout.txt`, or set `PAIEC_FEEDBACK` to another folder.
   - Logs are private to each team, so ours are not included. The organizers can read them on Codabench, and we can share them on request.
   - The proxy uses the logs of 928061, 930136, 930165, 931142, 951748, 951853, 952533 and 960385. The HBS+Prior analyses also use 978024 (the measurement submission), 978186 (HBS+Prior with the settings of HBS) and 1013135 (the final submission).

## Rebuilding the submissions

Run from `improved_mean/`:

```
zip hbs.zip model.py                                       # HBS, as submitted in 952533
python build_v7.py                                         # HBS+Prior: dist/improved_mean_v7_final.zip (978186)
python build_v8.py                                         # final: dist/improved_mean_v8_final.zip (1013135)
python ../tools/check_submission_zip.py dist/improved_mean_v8_final.zip
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python test_v7.py dist/v7_stage_final
```

- `build_v7.py` writes the twelve anchor items from the training data and truncates them with the Qwen3.8-27B tokenizer. It needs only a CPU.
- `build_v7.py` also builds the measurement submission (`dist/improved_mean_v7_probe.zip`, 978024) and two variants that we did not submit.
- `build_v8.py` reuses the anchor items written by `build_v7.py` (`dist/v7_anchors.json`), so run `build_v7.py` first. Without its three extra settings in `config.json`, `v8/model.py` behaves exactly as `v7/model.py`.
- At run time, HBS+Prior needs a GPU with about 18 GB of memory for the 4-bit model. The platform prefetches the model from `models.txt`. If the model cannot be loaded, the submission falls back to HBS.
- `test_v7.py` runs a built archive through the official worker pool on a local GPU and checks its predictions against the offline formulas.

## Results and the commands that produce them

Run from `improved_mean/`. In the "Needs" column:
- **data:** the training data.
- **logs:** our formative logs.
- **GPU:** a CUDA GPU.

| Result in the report | Command | Needs |
|---|---|---|
| Table 1, Figure 5b | `python proxy_validation.py` | logs |
| Table 2, proxy rows and the SD of the HBS+Ability row; Section 6.2; Appendix E, the check on six reports | `python proxy_baselines.py` | logs |
| Table 2, mixture prior and re-tuned hyperparameters | `python retune_real.py` | logs |
| Table 2, simulator rows; Section 4 and Appendix E, the item correction and both corrections removed | `python sim_ablations.py` | data |
| Section 6.1, own-label Bayes predictor | `python label_only_bound.py` | logs |
| Table 7, HBS+Prior rows | `python v7_feedback.py --probe 978024` | logs |
| Section 6.4, settings of HBS; Tables 8 and 9 (978186) | `python v7_robustness.py` (about 30 minutes) | logs |
| Section 6.4, settings of the final submission; Table 7, final row; Appendix E, the final settings | `python v8_checks.py compare grid jackknife known redraw budgets` | logs |
| Section 6.4 and Table 9, the run of the final submission | `python v8_checks.py final 1013135` | logs |
| Section 6.4 and Appendix E, the settings chosen leave-one-benchmark-out | `python v8_checks.py lobo` | logs |
| Section 6.4, the simulator on the training benchmarks | `python llm_prior_sim.py`, after `figure_data.py` | data; ratings cached by `llm_anchor.py` |
| Section 6.3, item difficulty | `python llm_difficulty.py extract --model Qwen/Qwen3-32B --4bit` and `--model Qwen/Qwen3-8B`, then `python llm_difficulty.py eval --model Qwen/Qwen3-32B --model Qwen/Qwen3-8B` and `python llm_sim.py 400 Qwen/Qwen3-32B` | data, GPU |
| Section 6.3, benchmark difficulty; Table 11; Figure 5a | `python llm_anchor.py extract --model M`, then `python llm_anchor.py eval --model M`, where M is Qwen/Qwen3-8B, Qwen/Qwen3.5-9B, or Qwen/Qwen3-32B or Qwen/Qwen3.8-27B with `--4bit` | data, GPU |
| HBS+ItemLLM (Section 6.3) | `python build_v6.py`, then `python test_v6.py` | data; GPU for the test |
| Section 6.3 and Appendix F, HBS+ItemLLM on real feedback: the loss per budget and pair, Table 10 | `python itemllm_checks.py breakdown synthetic` | logs |
| Appendix F, spread and outcome correlation of the probe scores on the training benchmarks | `python itemllm_checks.py training` | data; features cached by `llm_difficulty.py` |
| Appendix E: Figure 4, Table 6 | `python proxy_robustness.py` (about 35 minutes) | logs |
| Appendix E: the benchmark-difficulty diagnostic with alpha 3 and 6, the pessimism of the proxy, the accuracy of the priors, the jackknife over benchmarks, the placement of the LLM prior | `python appendix_checks.py` (about 25 minutes) | logs |
| Table 4 | `python item_text_difficulty.py` | data |
| Table 12 | `python figure_data.py` (the `anchored` entry of `results/figure_data.json`) | data, ratings cached by `llm_anchor.py` |
| Data plotted in Figures 1 to 5 | `python figure_data.py` and `python proxy_robustness.py` (written to `results/`) | logs |

Table and figure numbers refer to the report.

**Development tools not tied to a single result:**
- `tune.py` is the grid search over the simulator and the proxy of two early reports, used to set the hyperparameters of HBS.
- `subject_prior.py --build` fits the ability table of HBS+Ability.
- `sim.py` alone scores the constant 0.5, the official mean and HBS on simulated draws.

**Timing.** The LLM steps took 10 to 50 minutes each on one RTX 3090. The simulator takes about 10 seconds per configuration of 400 draws.

**Where outputs go.**
- LLM features are cached in `improved_mean/cache/`.
- Built archives go to `improved_mean/dist/`.
- Figure data go to `improved_mean/results/` (set `PAIEC_RESULTS` to change it).

## Repository layout

```
improved_mean/   predictors, submission builders, simulator, proxy and analyses
  model.py       HBS
  v6/ v7/ v8/    submission code of HBS+ItemLLM, HBS+Prior and the final submission
  versions/      earlier predictors, the mixture prior and the stratified acquisition hook
  paths.py       where the data, the official tools and the logs are found
```

## License

The code is released under the MIT License (see `LICENSE`). The training data are not included; measurement-db remains under its own license (CC BY-SA 4.0, gated access).

## Citation

```
@misc{huang2026labels,
  title  = {How Far Can Labels Go? Benchmark Difficulty Is the Bottleneck of Cold-Start Predictive Evaluation},
  author = {Huang, Hung-Kai and Hwang, Ren-Hung and Hsu, Chih-Chung},
  year   = {2026},
  note   = {Technical report, Predictive AI Evaluation Competition, NeurIPS 2026}
}
```
