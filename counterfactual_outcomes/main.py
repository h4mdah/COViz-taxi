import sys
from pathlib import Path
# make repo root importable so 'counterfactual_outcomes' can be imported when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections import defaultdict
import re

import json
import numpy as np
import random
import os
import time
from datetime import datetime
from os import makedirs, getpid
from os.path import join, abspath
from pathlib import Path
import pickle

from counterfactual_outcomes.common import save_traces, log_msg, load_traces, \
    get_highlight_traj_indxs, save_highlights, save_frames, hstack_frames, \
     mark_right_half_counterfactual, create_reward_bar_chart, create_info_strip
from counterfactual_outcomes.contrastive_online import online_comparison
from counterfactual_outcomes.contrastive_online_RD import online_comparison_RD
from counterfactual_outcomes.get_agent import get_config, get_agent


def output_and_metadata(args):
    # create a Windows-safe run id (no ':' characters) and make output dir once
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{run_ts}_{getpid()}"
    args.output_dir = join(abspath('results'), run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    with Path(join(args.output_dir, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def contrastive_online(args):
    args.config = get_config(args.load_path, args.config_filename, changes=args.config_changes)

    if args.interface == "Highway":
        env1, agent1 = get_agent(args)
        evaluation1 = agent1.interface.evaluation(env1, agent1)
        env2, agent2 = get_agent(args)
        evaluation2 = agent2.interface.evaluation(env2, agent2)
        env1.args = args
        env2.args = args
        if args.multi_head:
            traces = online_comparison_RD(env1, agent1, env2, agent2, args,
                                          evaluation1=evaluation1,
                                          evaluation2=evaluation2)
        else:
            traces = online_comparison(env1, agent1, env2, agent2, args, evaluation1=evaluation1,
                                       evaluation2=evaluation2)
        env1.close()
        env2.close()
        evaluation1.close()
        evaluation2.close()
    elif args.interface == "Taxi":
        # Taxi uses the taxi_interface implementation; treat similar to Highway:
        env1, agent1 = get_agent(args)
        evaluation1 = agent1.interface.evaluation(env1, agent1)
        env2, agent2 = get_agent(args)
        evaluation2 = agent2.interface.evaluation(env2, agent2)
        env1.args = args
        env2.args = args
        if args.multi_head:
            traces = online_comparison_RD(env1, agent1, env2, agent2, args,
                                          evaluation1=evaluation1,
                                          evaluation2=evaluation2)
        else:
            traces = online_comparison(env1, agent1, env2, agent2, args,
                                       evaluation1=evaluation1, evaluation2=evaluation2)
        env1.close()
        env2.close()
        evaluation1.close()
        evaluation2.close()
    else:
        raise NotImplementedError(f"Interface '{args.interface}' not implemented")
    return traces


def rank_trajectories(traces, method):
    import numpy as np
    for t in traces:
        n_states = len(t.states)
        n_contrastive = len(t.contrastive) if hasattr(t, "contrastive") else 0
        # only iterate indices that have a contrastive trajectory
        limit = min(n_states, n_contrastive)
        for i in range(limit):
            contra = t.contrastive[i]
            if not getattr(contra, "states", None):
                # nothing to rank for this contrastive entry
                continue
            contra_states = contra.states
            max_trace_state = n_states - 1
            max_contrastive_state = contra_states[-1].id[1]
            end_state = min(i + t.k_steps, max_trace_state, max_contrastive_state)
            contra.traj_end_state = end_state

            if method == "lastState":
                state1 = t.states[end_state]
                # find matching state in contrastive trajectory
                state2 = next((x for x in contra_states if x.id[1] == end_state), None)
                if state2 is None:
                    contra.importance = 0
                    continue
                # Use action_vector (existing attribute) as the action/value estimates
                av1 = getattr(state1, 'action_vector', None)
                av2 = getattr(state2, 'action_vector', None)
                val1 = np.max(np.asarray(av1)) if av1 is not None else 0
                val2 = np.max(np.asarray(av2)) if av2 is not None else 0
                contra.importance = abs(val1 - val2)
            elif "highlights" in method:
                # defined by second-best importance (original logic preserved)
                action_values = np.asarray(getattr(t.states[i], 'action_vector', np.zeros(1)))
                if action_values.size == 0:
                    contra.importance = 0
                    continue
                compared_action = np.min(action_values) if "worst" in method else \
                    np.partition(action_values.flatten(), -2)[-2]
                contra.importance = np.max(action_values) - compared_action


def get_top_k_diverse(traces, args):
    """
    sort contrastive trajectories by importance and return the top k important ones
    diversity measure - check intersection between trajectory indexes
    """
    # collect only actual ContrastiveTrajectory objects (ignore None placeholders)
    all_contrastive_trajs = []
    for t in traces:
        all_contrastive_trajs += [c for c in getattr(t, 'contrastive', []) if c is not None]
    if args.importance_method == "frequency":
        random.shuffle(all_contrastive_trajs)
    else:
        # sort by importance; ensure entries have the attribute
        all_contrastive_trajs = [c for c in all_contrastive_trajs if hasattr(c, 'importance')]
        all_contrastive_trajs.sort(key=lambda x: x.importance)

    # print([x.importance for x in all_contrastive_trajs[-10:]])
    # print([x.states[5].observed_actions for x in all_contrastive_trajs[-10:]])

    top_k, seen = [], defaultdict(lambda: [])
    while all_contrastive_trajs:
        current = all_contrastive_trajs[-1]
        if current.id[1] in seen[current.id[0]]:
            all_contrastive_trajs.pop(-1)
        elif current.traj_end_state - current.start_idx < args.min_traj_len:
            all_contrastive_trajs.pop(-1)
        elif current.traj_end_state - current.id[1] <= args.min_traj_len // 2:
            all_contrastive_trajs.pop(-1)
        else:
            top_k.append(current)
            idxs = [x for x in
                    range(current.id[1] + 1 - args.overlay, current.id[1] + args.overlay) if
                    x >= 0]
            new_set = set(list(seen[current.id[0]]) + idxs)
            seen[current.id[0]] = new_set
            if len(top_k) == args.num_highlights: break
            all_contrastive_trajs.pop(-1)

    # print([x.importance for x in top_k])
    return top_k


def main(args):
    output_and_metadata(args)
    """get environment and agent configs"""
    args.videos_dir = join(args.output_dir, "Highlight_Videos")
    args.frames_dir = join(args.output_dir, 'Highlight_Frames')

    """get traces"""
    traces = load_traces(args.traces_path) if args.traces_path else contrastive_online(args)


    log_msg(f'Obtained traces', args.verbose)

    """save traces"""
    save_traces(traces, args.output_dir)
    if not args.traces_path: save_traces(traces, abspath('results'))
    log_msg(f'Saved traces', args.verbose)

    """rank trajectories"""
    rank_trajectories(traces, args.importance_method)  # TODO change importance by highlights

    """select top k diverse trajectories"""
    highlights = get_top_k_diverse(traces, args)
    if not highlights:
        log_msg(f'No disagreements found', args.verbose)
        return
    log_msg(f'Obtained {len(highlights)} contrastive highlights', args.verbose)

    """randomize order"""
    if args.randomized: random.shuffle(highlights)
    id_list = []
    for hl in highlights:
        t, s = hl.id
        rd_vals = getattr(traces[t], 'RD_vals', None)
        rd_item = None
        if rd_vals is not None:
            try:
                rd_item = rd_vals[s]
            except Exception:
                rd_item = None
        id_list.append((hl.id, rd_item))
    save_traces(id_list, abspath('results'), name="Selected_Highlights.pkl")
    save_traces(id_list, args.output_dir, name="Selected_Highlights.pkl")
    # save_traces([x.id for x in highlights], abspath('results'), name="Selected_Indexes.pkl")

    """obtain trajectory indexes"""
    traj_indxs = get_highlight_traj_indxs(highlights)

    # print chosen actions
    # Build an action -> label mapping appropriate for the current interface
    if getattr(args, 'interface', '').lower() == 'highway':
        ACTION_DICT = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
    elif getattr(args, 'interface', '').lower() == 'taxi':
        # Standard Taxi actions (0..5) -- adjust if your custom Taxi uses different indices
        ACTION_DICT = {0: 'SOUTH', 1: 'NORTH', 2: 'EAST', 3: 'WEST', 4: 'PICKUP', 5: 'DROPOFF'}
    else:
        ACTION_DICT = {}

    for t_id, idxs in traj_indxs.items():
        actions_from_important_state = [x for x in traces[t_id[0]].previous_actions[t_id[1]:] if x is not None]
        # Use .get to avoid KeyError for unknown action indices
        print([ACTION_DICT.get(x, str(x)) for x in actions_from_important_state])
        contrastive = traces[t_id[0]].contrastive[t_id[1]]
        print([ACTION_DICT.get(x, str(x)) for x in contrastive.actions[:len(actions_from_important_state)]])
        print(40 * "-")

    # determine image shape from first available saved frame; fall back to a default
    img_shape = traces[0].states[0].image.shape
    if img_shape:
        height, width, layers = img_shape
        combined_width = width * 2 
        combined_shape = (height, combined_width, layers)
        img_shape = combined_shape
        

    if args.no_mark:
        """no-mark frames"""
        frames_path = join(args.output_dir, "NoMark_Frames")
        videos_path = join(args.output_dir, "NoMark_Videos")
        nomark_highlight_frames, nomark_contra_rel_idxs = {}, {}
        for hl, indxs in traj_indxs.items():
            trace = traces[hl[0]]
            nomark_highlight_frames[hl], nomark_contra_rel_idxs[hl] = \
                trace.mark_frames(hl[1], indxs, no_mark=args.no_mark)
        save_frames(nomark_highlight_frames, frames_path)
        save_highlights(img_shape, len(nomark_highlight_frames), frames_path, videos_path, args)

    """mark contrastive frames"""
    highlight_frames, contra_rel_idxs = {}, {}
    for hl, indxs in traj_indxs.items():
        trace = traces[hl[0]]
        highlight_frames[hl], contra_rel_idxs[hl] = trace.mark_frames(hl[1], indxs)

    """mark contrastive frames (side-by-side)"""
    highlight_frames_combined = {}
    for hl_id, indxs in traj_indxs.items():
        t_idx, s_idx = hl_id # unpack
        trace = traces[t_idx] # get trace
        contra_traj = trace.contrastive[s_idx] # get contrastive trajectory

        combined_frames = []

        n_steps = max(len(indxs), len(contra_traj.states))
        
        orig_len = len(trace.states)
        contra_len = getattr(contra_traj, 'start_idx', 0) + len(contra_traj.states)

        # Helper to format RD
        def _format_rd_for_action(rd_val, action_idx=None):
            try:
                if rd_val is None:
                    return 'RD:N/A'
                import numpy as _np
                arr = _np.asarray(rd_val)
                if arr.ndim == 1 and arr.size == 3:
                    t, s, il = arr
                    return f"time:{t:+.0f} succ:{s:+.0f} ill:{il:+.0f}"
                if arr.ndim == 2:
                    if action_idx is None:
                        sums = arr.sum(axis=1)
                        idx = int(_np.argmax(sums)) if sums.size > 0 else 0
                    else:
                        idx = int(action_idx) if action_idx is not None else 0
                    if idx < 0 or idx >= arr.shape[0]:
                        return 'RD:idxOOB'
                    row = arr[idx]
                    t, s, il = row if row.size >= 3 else (row[0], 0, 0)
                    return f"act{idx}: time:{t:+.0f} succ:{s:+.0f} ill:{il:+.0f}"
                if arr.size == 1:
                    return f"RD:{float(arr):+.2f}"
                flat = arr.flatten()
                if flat.size > 4:
                    return 'RD:[' + ','.join([f"{x:.2f}" for x in flat[:4]]) + ',..]'
                return 'RD:[' + ','.join([f"{x:.2f}" for x in flat]) + ']'
            except Exception:
                return 'RD:N/A'

        for i in range(n_steps):
            # Determine current frame and text for ORIGINAL trace
            if i < len(indxs):
                orig_state_idx = indxs[i]
                orig_frame = trace.states[orig_state_idx].image
                try:
                    orig_rew = trace.rewards[orig_state_idx]
                except Exception:
                    orig_rew = 'N/A'
                
                # RD and Action for Original
                orig_rd = None
                rd_vals = getattr(trace, 'RD_vals', None)
                orig_action = None
                try:
                    orig_action = trace.previous_actions[orig_state_idx] if getattr(trace, 'previous_actions', None) is not None else None
                except Exception:
                    orig_action = None
                if rd_vals is not None:
                    try:
                        orig_rd = rd_vals[orig_state_idx]
                    except Exception:
                        orig_rd = None
                
                text1 = f"R:{orig_rew} | " + _format_rd_for_action(orig_rd, orig_action)
            else:
                # Original finished: freeze last frame
                last_idx = indxs[-1] if indxs else 0
                orig_frame = trace.states[last_idx].image
                text1 = "DONE"
                orig_state_idx = last_idx # Keep using last valid index for overlays

            # Determine current frame and text for CONTRASTIVE trace
            if i < len(contra_traj.states):
                contra_frame = contra_traj.states[i].image
                try:
                    contra_rew = contra_traj.rewards[i]
                except Exception:
                    contra_rew = 'N/A'
                
                # RD and Action for Contrastive
                contra_rd = None
                contra_action = None
                if hasattr(contra_traj, 'RD_vals') and getattr(contra_traj, 'RD_vals') is not None:
                    try:
                        contra_rd = contra_traj.RD_vals[i]
                    except Exception:
                        contra_rd = None
                try:
                    if hasattr(contra_traj, 'actions') and len(getattr(contra_traj, 'actions', [])) > i:
                        contra_action = contra_traj.actions[i]
                    else:
                        contra_action = orig_action # Fallback? Maybe just None
                except Exception:
                    contra_action = None
                
                text2 = f"R:{contra_rew} | " + _format_rd_for_action(contra_rd, contra_action)
            else:
                # Contrastive finished: freeze last frame
                if contra_traj.states:
                    contra_frame = contra_traj.states[-1].image
                else:
                    # Fallback if empty
                    contra_frame = trace.states[0].image # Should not happen if filtered
                text2 = "DONE"

            combined_frame = hstack_frames(orig_frame, text1, contra_frame, text2)

            # --- Mark counterfactual half ---
            # mark right half as counterfactual starting from the fork (s_idx)
            is_cf = False
            try:
                is_cf = (orig_state_idx >= s_idx)
            except Exception:
                is_cf = False

            combined_frame = mark_right_half_counterfactual(combined_frame, is_counterfactual=is_cf,
                                                            color=(0, 0, 255), thickness=6, tint_alpha=0.12)
                                                                
            # --- Overlay Reward Timeline ---
            try:
                # Get full rewards for timeline
                orig_rewards_all = getattr(trace, 'rewards', []) or []
                contra_rewards_all = getattr(contra_traj, 'rewards', []) or []
                
                # Metadata extraction (re-used from previous logic)
                def _get_cfg_value(cfg, keys):
                    if cfg is None:
                        return None
                    try:
                        for k in keys:
                            if isinstance(cfg, dict) and k in cfg and cfg[k]:
                                return cfg[k]
                    except Exception:
                        pass
                    try:
                        for k in keys:
                            v = getattr(cfg, k, None)
                            if v:
                                return v
                    except Exception:
                        pass
                    return None

                cfg = getattr(args, 'config', None)
                algo = _get_cfg_value(cfg, ['algorithm', 'algo', 'agent_algorithm', 'policy']) or getattr(args, 'agent', None) or ''
                framework = _get_cfg_value(cfg, ['framework', 'backend', 'lib']) or getattr(args, 'framework', None) or ''
                env_name = _get_cfg_value(cfg, ['env_id', 'env', 'environment']) or getattr(args, 'env_id', None) or getattr(args, 'interface', '')
                steps = _get_cfg_value(cfg, ['train_steps', 'n_steps', 'steps', 'timesteps']) or getattr(args, 'train_steps', None) or ''
                model_file = os.path.basename(getattr(args, 'load_path', '') or '')
                meta_lines = []
                if algo:
                    meta_lines.append(str(f"Algo: {algo}"))
                if framework:
                    meta_lines.append(str(f"Framework: {framework}"))
                if env_name:
                    meta_lines.append(str(f"Env: {env_name}"))
                if steps:
                    meta_lines.append(str(f"Trained steps: {steps}"))
                if model_file:
                    meta_lines.append(str(f"Model: {model_file}"))
                if meta_lines:
                    meta_lines = meta_lines[:3]
                
                # Pass 'i' as current step index. Note: i is the index in the loop, which tracks the contrastive trajectory length usually.
                # However, for the timeline, we want it to align with the displayed states.
                # orig_state_idx is the index in Original Trace.
                # i is the index in Contrastive Trace.
                # We need to decide which "current step" to highlight. Since we show side-by-side, maybe we should pass both indices?
                # The generic function takes 'current_step_idx'. Let's use 'i' as it aligns with the loop progression (time).
                # But wait, original trace might be time-shifted or jumped?
                # Actually, the loop iterates `range(n_steps)`. 
                # `orig_state_idx = indxs[i]`. `indxs` is the subset of original states aligned with contrastive.
                # So `i` is the correct sequential index for the displayed video frame.
                
                timeline_bar = create_reward_bar_chart(combined_frame.shape[1], 
                                                       orig_rewards_all, 
                                                       contra_rewards_all, 
                                                       current_step_idx=i,
                                                       hist_h=80, 
                                                       metadata_lines=meta_lines)
                
                import numpy as _np
                # Create info strip
                info_strip = create_info_strip(combined_frame.shape[1], 
                                               f"Episode Length: {orig_len}", 
                                               f"Episode Length: {contra_len}")
                combined_frame = _np.vstack([timeline_bar, info_strip, combined_frame])
                
            except Exception as e:
                # print(f"Timeline error: {e}")
                pass

            combined_frames.append(combined_frame)
        
        highlight_frames_combined[hl_id] = combined_frames
            
            

    """save highlight frames"""
    save_frames(highlight_frames_combined, args.frames_dir, contra_rel_idxs)

    """generate highlights video"""
    save_highlights(img_shape, len(highlight_frames), args.frames_dir, args.videos_dir, args)
    log_msg(f'Highlights Saved', args.verbose)

    """ writes results to files"""
    log_msg(f'\nResults written to:\n\t\'{args.output_dir}\'', args.verbose)

    # after rank_trajectories(...) and selection attempts, add fallback:

    # fallback: if no Selected_Highlights.pkl / no selected highlights, pick top-N contrastive by importance
    if not os.path.exists(join(args.output_dir, "Selected_Highlights.pkl")):
        all_items = []
        for t_idx, t in enumerate(traces):
            for c_idx, contra in enumerate(getattr(t, "contrastive", [])):
                imp = getattr(contra, "importance", None)
                if imp is None:
                    continue
                all_items.append((imp, t_idx, c_idx, contra))
        if all_items:
            all_items.sort(key=lambda x: x[0], reverse=True)
            top = all_items[: getattr(args, "num_highlights", 5) ]
            # build a simple Selected_Highlights structure expected by save_highlights
            selected = []
            for imp, t_idx, c_idx, contra in top:
                selected.append({"trace_idx": t_idx, "contrastive_idx": c_idx, "importance": imp})
            # save frames/videos for these selected highlights
            save_frames(traces, args.output_dir, contra_rel_idxs=False)
            save_highlights(img_shape=None, n_videos=len(selected), frames_path=join(args.output_dir, "Highlight_Frames"),
                            videos_path=join(args.output_dir, "Highlight_Videos"), args=args)
            # write Selected_Highlights.pkl
            with open(join(args.output_dir, "Selected_Highlights.pkl"), "wb") as f:
                pickle.dump(selected, f)
