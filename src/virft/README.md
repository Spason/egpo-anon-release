# EGPO Training Framework

Core training framework for EGPO (Example-Guided Policy Optimization).

This directory contains the main training pipeline, including:
- **Custom GRPO/GPG Trainer** (`src/open_r1/trainer/`) — Modified trainer with support for GPG policy gradient and edit-distance-based rewards.
- **Data Preprocessing** (`src/open_r1/pretreat/`) — `build_reference.py` (offline reference traces) and answer-conditioned generation scripts.
- **Training Entry Points** (`src/open_r1/grpo_egpo.py`) — Main training scripts.
- **DeepSpeed Configs** (`configs/`, `local_scripts/`) — ZeRO-2 and ZeRO-3 configurations for distributed training.

See the [main README](../../README.md) for full documentation and usage instructions.
