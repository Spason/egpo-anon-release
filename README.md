# EGPO: Example-Guided Policy Optimization

Official implementation of **EGPO** (anonymous code release accompanying a double-blind submission).

## Overview

EGPO is a minimalist RL framework for LLMs trained with verifiable rewards. It uses a plain policy gradient
(mean-centered, group-relative advantages; no clipping, no KL penalty, no critic, no learned reward model) and adds an
**Example-Guided Reward (EGR)**: the normalized edit (Levenshtein) similarity between a rollout's reasoning and a
reference trace that the *base model* sampled once, offline.

```
R_total = alpha_f * R_fmt + alpha_r(t) * R_ref + alpha_a * R_ans
R_fmt   = 1[output is exactly  <think>...</think><answer>...</answer>]
R_ans   = 1[extracted answer == ground-truth answer]
R_ref   = 1 - d_edit(reasoning, reference) / max(|reasoning|, |reference|)     (only for answer-correct rollouts)
alpha_r(t) = alpha_r0 * max(rho - t/S, delta)                                  (alpha_f, alpha_r0, alpha_a = 0.1, 1.0, 1.0; rho = 0.7; delta = 0)
```

* Only the policy model is resident during EGPO training. A frozen copy of the base model is loaded **only** if you
  ask for KL monitoring (`--monitor_kl_interval N`, never part of the loss) or run GRPO with `--beta > 0`.
* All reward code lives in `src/virft/src/open_r1/rewards.py` (pure Python, unit-tested: `python tests/test_rewards.py`).

## Project Structure

```
EGPO/
├── LICENSE.txt
├── README.md
├── requirements.txt
├── tests/test_rewards.py                       # unit tests of the reward (no GPU needed)
└── src/
    ├── code_vllm_replace/                      # vLLM compatibility patches (third-party, Apache-2.0)
    ├── scripts/                                # launch scripts (see table below)
    └── virft/
        ├── configs/, local_scripts/            # DeepSpeed configs
        ├── slurm/                              # cluster job scripts inherited from the base framework
        └── src/open_r1/
            ├── grpo_egpo.py                    # training entry point (EGPO / GPG / GRPO)
            ├── rewards.py                      # EGPO reward + sparse reward (pure Python)
            ├── prompts.py                      # system prompts shared by training and Phase 1
            ├── data_config.py                  # dataset paths
            ├── split_data.py                   # JSON -> JSONL conversion / splitting
            ├── trainer/grpo_trainer.py         # custom trainer (GPG / GRPO / Pinsker)
            └── pretreat/
                ├── build_reference.py          # Phase 1: reference construction (question only)  <-- used by EGPO
                └── add_thinking*.py            # answer-conditioned generation (the "answer-then-reason" variant)
```

## Installation

```bash
cd EGPO
pip install -r requirements.txt
```

Python >= 3.10, PyTorch, Transformers, TRL, DeepSpeed and Flash-Attention 2 (versions pinned in `requirements.txt`).

## Quick Start

### 1. Data paths
Edit `BASE_DATA_DIR` in `src/virft/src/open_r1/data_config.py`. Each dataset is a JSON list of
`{"image": [...], "conversations": [{"from": "user", "value": ...}, {"from": "assistant", "value": <gold answer>}]}`.

### 2. Phase 1 -- offline reference traces (once per dataset)
Serve the **base** policy with any OpenAI-compatible server and run:

```bash
vllm serve /path/to/Qwen3-VL-8B-Instruct --served-model-name base --port 8000
cd src/virft/src/open_r1/pretreat
python build_reference.py --dataset_name scienceqa_data --max_tries K --temperature T \
    --base_url http://localhost:8000/v1/chat/completions --model base --concurrency 64
```

The model sees only the training prompt (system prompt + question; the gold answer is *not* given). Sampling stops at
the first correct sample, at most `K` per prompt; prompts without a correct sample keep only the gold answer (no
reference). `K` and `T` are required arguments: use the values reported in the paper. The result is written to the
`<dataset>_with_thinking2` entry of `data_config.py`.

### 3. Phase 2 -- training
Set `CKPT_PATH` and `temp_url` inside the script (the scripts refuse to run with the placeholders), then:

```bash
bash src/scripts/8b_paper_en.sh            # EGPO, one dataset  (DATA_PATH=... to choose it)
bash src/scripts/run_paper_all.sh          # EGPO, all datasets (epochs chosen by dataset size)
bash src/scripts/run_paper_all_grpo.sh     # GRPO baseline (sparse 0/1 reward, clipping + KL)
bash src/scripts/run_paper_all_grpo_no_kl.sh   # GRPO without KL (beta = 0, no reference model loaded)
```

Optional: `export WANDB_API_KEY=...` for logging.

| Script | Method | Reward | Clip | KL / reference model |
|---|---|---|---|---|
| `8b_paper_en.sh` | EGPO | `egpo` (paper reward) | no | no |
| `8b_paper_gpg_0.sh` | EGPO without `R_ref` (ablation "w/o R_ref", i.e. GPG with format + answer rewards) | `egpo`, rho = 0 | no | no |
| `8b_paper_grpo.sh` | GRPO | `sparse` (0/1) | yes (eps = 0.2) | beta = 0.04 |
| `8b_paper_grpo_no_kl.sh` | GRPO w/o KL | `sparse` (0/1) | yes (eps = 0.2) | beta = 0 (no reference model) |

All four scripts share the optimizer / rollout settings (lr 5e-7, G = 8, temperature 0.99, per-device batch 1 without
gradient accumulation, max completion length 1500, English system prompt, bf16).

Other ablations: *w/o annealing* -> `--anneal_floor 1.0`; *external-model reference* -> run `build_reference.py` against
a different served model; *answer-conditioned reference* -> `pretreat/add_thinking_en.py` (the answer is given to the
generator, which is the "answer-then-reason" variant).

## Key Arguments (`grpo_egpo.py`)

| Argument | Default | Meaning |
|---|---|---|
| `--pg_name` | `grpo` | `gpg` = EGPO objective (plain REINFORCE), `grpo`, `pinsker` |
| `--adjust_gd` | `false` | mean-centered advantages (no std normalization); used by EGPO |
| `--reward_mode` | `egpo` | `egpo`, `sparse` (GRPO baselines), `legacy` (earlier hierarchical format reward) |
| `--alpha_f / --alpha_r0 / --alpha_a` | `0.1 / 1.0 / 1.0` | reward weights |
| `--supervise_thinking_ratio` | `0.5` | annealing horizon ratio rho (the scripts set `0.7`) |
| `--anneal_floor` | `0.0` | floor delta of the annealed reference weight |
| `--monitor_kl_interval` | `0` | >0: load a frozen reference and log KL every N steps (monitoring only) |
| `--gpg_alpha_rescale` | `false` | rescale the loss by the fraction of non-degenerate groups (the paper uses alpha = 1) |
| `--system_prompt_lang` | `en` | `en` or `zh` |
| `--num_generations` | `8` | group size G |

## Tests

```bash
python tests/test_rewards.py
```

## License

This project is released under the MIT License - see [LICENSE.txt](LICENSE.txt).
Files under `src/code_vllm_replace/` are adapted from third-party projects and keep their original Apache-2.0 headers.

## Acknowledgements

This project builds upon:
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [Visual-RFT](https://github.com/Liuziyu77/Visual-RFT) - Visual Reinforcement Fine-Tuning
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Deep learning optimization library
