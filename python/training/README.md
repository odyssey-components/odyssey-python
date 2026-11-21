# Shared Python training library

This package provides training events, monitoring, PPO helpers and algorithm packaging. Application configuration uses the repository package `odyssey_config` directly. The `odyssey-config` package is versioned in this repository.

Managed Python trainers use `WorkerRun` from `odylab_training.worker` to read the Runner's scene and output directories. The scene's `training.yaml` is the default configuration input. Each trainer supplies its own loader and schema. Worker task overrides are applied over the loader's configured inputs, and explicit CLI overrides take precedence over task overrides.

Go1, Maze, and MiniDuck algorithm packages use the Worker directory and configuration contract. Gomoku and Xiangqi teacher Jobs use their own versioned dataset inputs and share the same stage events and metrics library. Gomoku self-play and policy-only training also use the shared execution stage.

The `training.stage` event reports `started`, `completed`, or `failed`. Preparation reads inputs and loads configuration. Execution includes the trainer's own evaluation and output writing. Dedicated teacher Jobs also report publication. Commands with a machine-readable result on stdout can write stage events to stderr. The Kubernetes Runner owns Attempt state, cancellation, and artifact archival for managed runs.

AIOps project-specific trainers live in `training/maze` and `training/go1` in the [AIOps repository](https://github.com/odyssey-components/odyssey-aiops), each with an independent wheel. This shared package contains game-neutral algorithms/events and `local_worker.main(allowed_modules)`; each project owns its worker allowlist. Go1 consumers import `aiops_go1_training` directly; the old Go1 namespace has been removed.
