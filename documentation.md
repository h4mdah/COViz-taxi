# COViz-Taxi — Updated Documentation

This repository provides tools to generate, analyze, and visualize counterfactual (contrastive) outcomes for RL agents, with special support for Taxi and Highway environments. It includes trace generation, regret/importance scoring (including reward decomposition), highlight selection, and synchronized visualizations (side-by-side frames + reward timelines).

## Quick Summary of Current Features

- Generation: run trained policies to collect traces or load existing traces from disk (see `run.py` and `counterfactual_outcomes/*`).
- Counterfactual rollouts: fork traces at individual timesteps and rollout alternative actions for a configurable horizon (`contrastive_online.py`, `contrastive_online_RD.py`).
- Reward decomposition (RD): collect and analyze decomposed reward terms for environments that provide them (RD-aware code paths in `contrastive_online_RD.py`).
- Ranking & selection: compute importance/regret scores, sort candidates, and select a diverse Top-K of highlights (`counterfactual_outcomes/main.py`).
- Visualization: synchronized side-by-side frames, reward bar timelines, and MP4 export (`counterfactual_outcomes/common.py` utilities + `tools/` helpers).
- Interfaces: modular environment interfaces for Taxi and Highway under `counterfactual_outcomes/interfaces` and `agents/`.
- Utilities & inspection: many helper scripts under `tools/` for rendering, inspecting traces, regenerating traces, and producing comparison GIFs/videos.
- Packaging & run helpers: `requirements.txt`, `packages.txt`, and `run_ppo.bat` for convenience on Windows.
- Tests: minimal test coverage in `tests/` (e.g., `test_chokepoints.py`, `test_ppo_training.py`).

## Repository Structure (high level)

- `counterfactual_outcomes/` — core generation, ranking, and rendering logic.
  - `main.py` — orchestration of generation, ranking, selection, and visualization.
  - `common.py` — shared utilities: saving/loading traces, frame stacking, reward plots, video saving.
  - `contrastive_online.py` — online contrastive rollouts.
  - `contrastive_online_RD.py` — RD-aware contrastive rollouts.
  - `get_agent.py` — factory to configure environment + agent interfaces.
  - `interfaces/` — environment-specific wrappers (Taxi, Highway, etc.).

- `agents/` — trained-model artifacts and agent-specific helpers (Taxi, Highway folders contain saved models).

- `tools/` — inspection and rendering utilities (e.g., `inspect_traces_run.py`, `render_side_by_side_env.py`, `generate_traces_quick.py`).

- `traces/` — stored recorded traces organized by environment and run.

- `results/` — saved run outputs and highlight videos.

- `tests/` — unit/functional tests.

- Top-level scripts: `run.py` (main entry point), `run_ppo.bat` (Windows helper), `run_ppo`/training helpers under `counterfactual_outcomes`.

## How it works (concise)

1. Collect or load traces for an agent in an environment.
2. For each candidate fork timestep, choose an alternative action and rollout for `K` steps.
3. Compute an importance/regret score (optionally using RD).
4. Select Top-K highlights with temporal/spatial diversity.
5. Produce synchronized side-by-side visualizations with a reward timeline overlay and save videos.

## Common Commands

Run the main pipeline (load traces or create new ones):

```bash
python run.py --interface Taxi --traces-path ./traces/Taxi-v3-COViz --num_highlights 3
python run.py --interface Highway --n_traces 5 --k-steps 10
```

Windows helper to start PPO training (if configured):

```powershell
.\run_ppo.bat
```

If you want quicker trace generation for debugging:

```bash
python tools/generate_traces_quick.py --interface Taxi --n 2
```

## Key files and where to look

- `run.py` — command-line entry and configuration loading.
- `counterfactual_outcomes/main.py` — pipeline orchestration and highlight selection.
- `counterfactual_outcomes/common.py` — rendering helpers (reward bar chart, `hstack_frames`, `save_highlights`).
- `counterfactual_outcomes/contrastive_online*.py` — online generation of contrastive rollouts (RD-aware variant included).
- `tools/` — many small scripts to inspect traces, regenerate highlights, and render side-by-side outputs.

## Notes & Next Steps

- Tests: run `pytest tests/` to validate small pieces; some tests are integration-style and may require models/traces.
- If you'd like, I can: run tests, list missing docs for specific modules, or add API-style docstrings to key functions.

---
*This document was updated to match the repository contents and current features.*
