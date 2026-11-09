# Online Hybrid Draft Training for Qwen3-4B-Instruct (DFlash / DSpark)

Code, data subset, and DFlash results for training speculative-decoding draft
models with an **online acceptance-length objective**:

- **DFlash** draft (block size 8, 5 draft layers, verifier layers 2/18/33,
  full 151,936-token vocabulary) trained from scratch.
- **Hybrid objective**: `L = L_EAL + 1.0 * L_KL`, where
  - `L_EAL = -log(mean(EAL) + 1e-6)` and `EAL = 1 + sum_i prod_{j<=i} p_j(label_j)`,
  - `L_KL` is a forward KL to the verifier distribution, weighted per position by
    `w_k ∝ sum_{i>=k} P_i` (prefix weight, no uniform floor),
  - labels come from a **tree/prefix-sharing verifier pass** built from the
    draft's own proposals.
- **DSpark** variant: the same hybrid objective, with a sequential Markov logit
  bias (rank 256, vanilla) and a per-position confidence head trained with BCE.
- 2-GPU DDP, one checkpoint and one full acceptance evaluation per epoch.

This release reports **DFlash checkpoints 0-2 only**. Other experimental results
(epoch 3, DSpark/DFlash2 comparisons) are intentionally not included.

## Contents

| Included | Not included |
|---|---|
| Training/eval/analysis code | Verifier weights (`Qwen3-4B-Instruct`, ~8 GB) |
| 300k text-only dataset subset (`open_perfectblend_1trun_v2_300k`, zstd parquet, ~168 MB) | Tokenized Arrow dataset (`prepared`, ~10 GB) |
| Patched speculators v0.7.0.1 diff | Precomputed hidden states (`hidden_states`, ~13 TB) |
| DFlash hybrid/offline acceptance and waste results, ckpt0-2 | Trained checkpoint weights |

The two large training artifacts are regenerated locally with the scripts in
`scripts/` (see below).

## Repository layout

```
.
├── README.md
├── env.sh                         # portable env (edit interpreters)
├── patches/
│   └── speculators-v0.7.0.1-local.patch
├── open_perfectblend_1trun_v2_300k/
│   ├── data/train-0000*-of-00012.parquet   # 300,000 rows, text only
│   └── sample_summary.json
├── data_pipeline/                 # raw-data subset + response regeneration
├── qwen3_4b_dflash_300k/
│   ├── checkpoints/3/             # CONFIG ONLY (from-scratch init config)
│   ├── online_rl/                 # trainers, drivers, wasted analysis
│   └── test_data/                 # acceptance script, benchmarks, results
├── results_dflash_ckpt0_2/        # comparison table + waste JSONs
└── scripts/                       # reproducible launchers
```

`qwen3_4b_dflash_300k/checkpoints/3/` intentionally contains only
`config.json`/`config.py`; the draft is trained from scratch and never loads
those weights.

## Environment

Tested on 2×A100-80GB with two Python environments:

- **Training env**: Python 3.11, `torch 2.11.0+cu128`, `transformers 5.14.1`,
  `datasets 5.0.0`, plus the patched speculators source.
- **vLLM env**: `vllm 0.28.0`, `torch 2.11.0+cu128`, `transformers 5.16.1`.

```bash
# 1) speculators v0.7.0.1 + local patch (online backbone forward etc.)
git clone --branch v0.7.0.1 https://github.com/vllm-project/speculators.git speculators-src
(cd speculators-src && git apply ../patches/speculators-v0.7.0.1-local.patch)

# 2) verifier weights
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir Qwen3-4B-Instruct

# 3) environment
export SPECULATORS_PYTHON=/path/to/training-env/bin/python
export VLLM_PYTHON=/path/to/vllm-env/bin/python
source ./env.sh
```

`env.sh` prepends `speculators-src/src` to `PYTHONPATH`, so an editable install
is optional.

`data_pipeline/` contains the scripts that produced the 300k subset from the
raw OpenPerfectBlend parquet files. Those raw files are not included; the
pre-sampled 300k subset is provided instead.

## Data

The included subset is the 300k text dataset used by this project: 300,000
rows sampled (seed 42) from a filtered single-turn OpenPerfectBlend dataset.
Each row has `conversations` (list of `{from, value}`) and `source`.

Reproduce the full training inputs (~13 TB of hidden states):

```bash
# terminal A: chat endpoint for response regeneration
./scripts/start_vllm.sh
# terminal B: regenerate temperature-0 responses (~4 GB JSONL)
./scripts/generate_responses.sh
# tokenize into Arrow (qwen3_4b_dflash_300k/prepared, ~10 GB)
./scripts/prepare_data.sh

# terminal A: restart with hidden-state extraction (layers 2/18/33)
./scripts/start_vllm_hidden_states.sh
# terminal B: precompute hidden states (~13 TB; use MAX_SAMPLES=2000 for a smoke test)
./scripts/generate_hidden_states.sh
```

The scripts default to `Qwen3-4B-Instruct/`, port 8000, and paths under
`qwen3_4b_dflash_300k/`. Environment variables `PORT`, `CONCURRENCY`,
`MAX_SAMPLES`, `CUDA_VISIBLE_DEVICES` are honored.

## Training

All runs start from scratch and use the config-only anchor in
`qwen3_4b_dflash_300k/checkpoints/3/`. The driver launches one epoch per
process, writes `checkpoints/{epoch}/`, then runs a full acceptance evaluation
of that epoch before starting the next one.

```bash
# DFlash hybrid (4 epochs, 2 GPUs)
./scripts/run_dflash_hybrid.sh
# DSpark hybrid (same pipeline; results are not part of this release)
./scripts/run_dspark_hybrid.sh

# optional: offline DFlash baseline for the ckpt0-2 comparison
./scripts/train_dflash_baseline.sh
```

Key hyperparameters (identical to the reported runs):

| | value |
|---|---|
| seed / epochs | 1234 / 4 |
| anchors per step | 3072 (per rank), 2 ranks |
| optimizer | Muon (2D) + AdamW, lr 3e-4 / Muon lr 3e-3 |
| schedule | cosine, 0.5 cycles, 892 warmup steps |
| block size / layers / aux layers | 8 / 5 / 2,18,33 |
| objective | log-EAL + prefix-weighted forward KL, weight 1.0, T=1.0 |
| labels | tree/prefix-sharing verifier backend |
| data | `prepared` + precomputed `hidden_states`, ratio 0.999 |

The DSpark run additionally enables `--markov-rank 256 --markov-head-type vanilla
--enable-confidence-head --confidence-head-with-markov`.

## Evaluation and analysis

```bash
# vLLM 0.28 acceptance on GSM8K/HumanEval/MT-Bench (1,563 prompts, temp 0)
./scripts/eval_ckpt0_2.sh
# per-position offline-correct / wasted statistics + comparison table
./scripts/run_waste_and_compare.sh
```

Results are written to `qwen3_4b_dflash_300k/test_data/results/` and
`results_dflash_ckpt0_2/`. The shipped `eval_command.txt` files are the original commands from the
reported runs. The shipped `results.json` directories are skipped by
`eval_ckpt0_2.sh`; delete a directory to re-evaluate that checkpoint.

## Results (DFlash, ckpt0-2)

Acceptance length (AL) includes the bonus token, i.e. `1 + accepted/drafts`.
`waste/correct` is the share of offline-correct draft tokens that are rejected
because an earlier prefix token is wrong.

| epoch | model | AL | AR | waste/correct | pos6 waste/correct |
|---|---|---|---|---|---|
| 0 | hybrid | **3.7259** | 38.94% | **0.1299** | **0.3701** |
| 0 | offline | 3.5939 | 37.06% | 0.1761 | 0.4654 |
| 1 | hybrid | **4.0044** | 42.92% | **0.1166** | **0.3045** |
| 1 | offline | 3.8505 | 40.72% | 0.1599 | 0.4050 |
| 2 | hybrid | **4.4360** | 49.09% | **0.0971** | **0.2311** |
| 2 | offline | 4.3394 | 47.71% | 0.1300 | 0.3129 |

Full per-position acceptance, conditional acceptance, and waste tables:
`results_dflash_ckpt0_2/comparison_ckpt0_2.md`.

## Security and privacy

- All endpoints in the code and scripts are local (`127.0.0.1`); no telemetry,
  credentials, API keys, or external services are used. `env.sh` also clears
  proxy variables so local vLLM calls are not routed through a corporate proxy.
- Only load checkpoints you trust. Checkpoint directories contain `config.py`
  and weights; this code loads them from local paths with
  `local_files_only=True`, `trust_remote_code` is not enabled, and speculators
  deserializes optimizer states with `torch.load(weights_only=True)`.
- The bundled 300k subset is public OpenPerfectBlend text. It may contain
  third-party paths or code snippets from the original data (for example
  `/home/...` strings); these are dataset content, not host paths from this
  project.
- `.gitignore` excludes local weights, hidden states, logs, and credential
  files.

## Notes

- Only DFlash ckpt0-2 results are reported, as requested. DSpark training code
  is included, but its results are excluded from this release.
- The offline DFlash baseline was trained with the upstream speculators 0.8.0
  `scripts/train.py` on one GPU (lr 3e-4, 4 epochs, same data). The hybrid runs
  use the online trainer on two GPUs (effective batch 6144 anchors/step,
  lr 3e-4/3e-3). This mirrors the comparison setup of the original experiments.

## License

This repository is released under the **Apache License 2.0** (see `LICENSE`).
It includes modified portions of
[vllm-project/speculators](https://github.com/vllm-project/speculators)
(Apache-2.0), described in `patches/speculators-v0.7.0.1-local.patch`; see
`NOTICE` for attribution. The bundled dataset subset derives from
OpenPerfectBlend (Apache-2.0; constituent datasets Apache-2.0 or MIT), and the
evaluation inputs retain their upstream licenses.
