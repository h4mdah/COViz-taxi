**COViz-taxi**

A toolbox for generating and visualizing counterfactual / contrastive outcomes for Taxi environment agents.

**Overview**
- **Purpose:** run and analyze contrastive/counterfactual comparisons for Taxi agents (supports SB3-trained agents via adapter), extract Q-values (best-effort), produce visualizations (side-by-side GIFs) and stylized re-renders of Taxi states.
- **Main components:** trace generation, contrastive scoring/highlight selection, visualization tools.

**Quick status**
- Added visualization tools: `tools/side_by_side_gifs.py` and `tools/render_side_by_side_env.py` (stylized renderer based on Taxi integer state).
- Support for SB3 agent fallback wrapping implemented in `counterfactual_outcomes/interfaces/Taxi/taxi_interface.py` (best-effort Q-value extraction and adapter logic).
- Known issues: environment synchronization (making env1 and env2 start from identical internal state) is fragile in some wrapper configurations; Gym vs Gymnasium differences may produce warnings.

**Repository Layout (key files)**
- `run.py` — top-level runner (project entrypoints vary by task).
- `counterfactual_outcomes/` — main pipeline:
  - `common.py` — Trace/State classes and utilities.
  - `train_model.py`, `main.py`, `contrastive_online.py`, `contrastive_online_RD.py` — contrastive pipeline and training helpers.
  - `interfaces/Taxi/` — Taxi-specific adapters and environment wrappers (`taxi_interface.py`, `environments.py`).
- `traces/` — example saved traces; e.g. `traces/taxi/Traces.pkl`.
- `results/` — run outputs (Traces.pkl, Selected_Highlights.pkl, results directories, generated GIFs).
- `tools/` — visualization and helper scripts added during iteration:
  - `tools/side_by_side_gifs.py` — compose GIFs from stored `State.image` frames in traces.
  - `tools/render_side_by_side_env.py` — re-render Taxi integer states by setting `env.unwrapped.s` and calling `env.render()`; falls back to `custom_render_from_state()` which draws a stylized map (hedges, road, taxi, passenger, building) that matches the user's requested visual style.
  - `tools/inspect_selected_run.py`, `tools/inspect_traces_run.py`, `tools/check_all_traces.py` — small inspectors for pickles.
  - `tools/list_outputs.py` — list generated GIFs folder contents.

**Data formats and expectations**
- `Traces.pkl` typically contains a `list` of Trace objects. Each Trace exposes a `states` list of `State` objects. A `State` usually contains:
  - `state` : integer Taxi internal state
  - `image` or `img` : optional RGB frame (PIL Image or numpy array)
- `Selected_Highlights.pkl` used by visualization is a `list` of items often shaped like `((trace_idx, contrastive_idx), score_array)`. The visualization tools accept this format and also fall back to pairing traces with themselves if `selected` is missing.

**How the pipeline works (step-by-step)**
1. Train or load an agent (SB3 or rl_agents):
   - If you have an SB3 `.zip` model, the Taxi adapter in `counterfactual_outcomes/interfaces/Taxi/taxi_interface.py` will attempt to load it and expose the interfaces expected by the contrastive pipeline. Q-value extraction from SB3 DQN is best-effort and may need model-specific tweaks.

2. Generate traces (trajectories):
   - The training or evaluation script (e.g. `counterfactual_outcomes/train_model.py` or your own run logic) should save traces as pickled `Traces.pkl` containing Trace objects with `states` and optional `image` frames.
   - Example saved traces exist under `traces/taxi/Traces.pkl` and `results/.../Traces.pkl` from prior runs.

3. Run contrastive selection:
   - The contrastive pipeline (e.g. `counterfactual_outcomes/contrastive_online.py` or `main.py`) processes traces and produces `Selected_Highlights.pkl`, a list of candidate (original, contrastive) pairs with scores.

4. Visualize side-by-side:
   - Two approaches exist:
     - Use `tools/side_by_side_gifs.py` to build animations using frames stored inside Trace/State objects.
     - Use `tools/render_side_by_side_env.py` to re-render frames by setting the Taxi environment's internal integer state and calling `env.render()`; if `env.render()` is unavailable or fails, the script falls back to a stylized `custom_render_from_state()` implementation that draws a map matching the user's attachment.

**Commands (Windows/cmd) — common tasks**
- Regenerate stylized GIFs (example used in our session):

```cmd
C:\> conda activate coviz
C:\project\COViz-taxi> python tools\render_side_by_side_env.py --traces "results\run_2025-11-18_12-20-50_2132\Traces.pkl" --selected "results\run_2025-11-18_12-20-50_2132\Selected_HighLights.pkl" --out-dir "results\side_by_side_env_regen" --fps 3 --max 6
```

- Generate GIFs from stored frame images (if `State.image` exists):

```cmd
C:\project\COViz-taxi> python tools\side_by_side_gifs.py --traces traces\taxi\Traces.pkl --selected results\Selected_Highlights.pkl --out-dir results\side_by_side_sample --fps 3
```

- Inspect selected highlights structure:

```cmd
C:\project\COViz-taxi> python tools\inspect_selected_run.py results\run_2025-11-18_12-20-50_2132\Selected_Highlights.pkl
```

- List outputs generated by previous run:

```cmd
C:\project\COViz-taxi> python tools\list_outputs.py
```

**What I changed during iteration (useful file list)**
- `tools/render_side_by_side_env.py` — new script and stylized renderer with fallback rendering logic (custom drawing to match user-provided visual style). Also includes:
  - mapping out-of-range contrastive indices into available traces (modulo mapping)
  - sampling to at most 60 frames per GIF to keep output sizes manageable
- `tools/side_by_side_gifs.py` — robust composition script for traces that already contain images.
- `tools/inspect_selected_run.py`, `tools/inspect_traces_run.py`, `tools/check_all_traces.py`, `tools/list_outputs.py` — assorted inspection helpers.
- `counterfactual_outcomes/interfaces/Taxi/taxi_interface.py` — earlier edits (SB3 Adapter, debug helpers, Q-value extraction best-effort). If you reverted edits, re-open this file to re-apply the adapter changes.

**Behavioral details & notes**
- `Selected_Highlights.pkl` format: most runs yield a `list` where each entry is `((trace_idx, contrastive_idx), score_array)`. The visualization scripts expect the first element to contain a 2-tuple with trace and contrastive indices.
- Out-of-range indices: when `Selected_Highlights` references trace indices that are larger than the run-specific `Traces.pkl` (this can happen when selection was computed from a different, larger trace pool), `tools/render_side_by_side_env.py` will map the contrastive index into the locally-available `Traces.pkl` via modulo and print a mapping message.
- Gym vs Gymnasium: the code uses `gym` (legacy). On modern setups you may see a migration warning. Replacing `import gym` with `import gymnasium as gym` might be required for full compatibility with the latest numpy versions and render modes.
- Environment sync: making two environment instances start from identical internal states is non-trivial because wrappers may hide the inner env. The repo includes debug helpers that dump wrapper chains; to debug, instrument `counterfactual_outcomes/interfaces/Taxi/taxi_interface.py`'s `sync_envs` method to print wrapper chains and candidate attributes, run `run.py` to produce the dump, then set `unwrapped.s` on the inner-most object.

**Troubleshooting**
- If `pickle.load` fails with `ModuleNotFoundError` for `counterfactual_outcomes` when loading `Traces.pkl`, ensure the repo root is on `PYTHONPATH` or run scripts from repo root. Example (Windows/cmd):

```cmd
C:\project\COViz-taxi> set PYTHONPATH=%CD%
C:\project\COViz-taxi> python tools\inspect_traces_run.py traces\taxi\Traces.pkl
```

- If GIF generation hangs during palette conversion (Pillow), try increasing `--fps` lower or let the script sample fewer frames by lowering `--max` or modifying `max_frames` in `tools/render_side_by_side_env.py`.

**Next steps you can request**
- Re-apply/confirm edits to `counterfactual_outcomes/interfaces/Taxi/taxi_interface.py` to re-introduce SB3 adapter/debug instrumentation.
- Further refine the stylized renderer colors and labels to exactly match a target image — I can iterate and re-generate GIFs.
- Stabilize deterministic env sync by running `run.py` to produce wrapper-chain dumps and then update `sync_envs` to set the inner `s` on the correct object.
- Package a small reproducible example (tiny trace and selection) and a short script to visualize it.

**Where outputs live**
- Look in `results/` for run outputs and generated GIFs. Example from the last run:
  - `results\\side_by_side_env_regen\\t9_c170.gif`
  - `results\\side_by_side_env_regen\\t9_c177.gif`
  - `results\\side_by_side_env_regen\\t9_c184.gif`
  - `results\\side_by_side_env_regen\\t9_c191.gif`
  - `results\\side_by_side_env_regen\\t9_c198.gif`

If you'd like, I can now:
- Re-run the renderer against `traces/taxi/Traces.pkl` so contrastive indices map directly (quick),
- Or re-iterate on visuals to better match your attachment (colors, cell size),
- Or produce a base64-embedded GIF here for a selected file.

Thank you — tell me which follow-up you want and I will proceed.
Implementing the method from the paper "Explaining Reinforcement Learning Agents Through Counterfactual Action Outcomes" (AAAI-24) with the taxi environment.




[https://github.com/eleurent/rl-agents](https://github.com/eleurent/rl-agents)

For training a Reward-Decomposed (RD) agent:



