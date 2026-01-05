"""
Inspect contrastive trajectories and their rewards from saved traces.
This script helps you understand and compare original vs contrastive rewards.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle
import numpy as np

# Import common module to ensure pickle can deserialize Trace objects
try:
    from counterfactual_outcomes.common import Trace
    from counterfactual_outcomes.contrastive_online import ContrastiveTrajectory
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure all dependencies are installed (cv2, etc.)")

def inspect_contrastive_rewards():
    """Analyze and display contrastive trajectory rewards."""
    
    print("=" * 70)
    print("CONTRASTIVE TRAJECTORY REWARD ANALYSIS")
    print("=" * 70)
    
    # Find trace files
    results_dir = REPO_ROOT / "results"
    pkl_files = list(results_dir.glob("run_*/Traces.pkl"))
    
    if not pkl_files:
        print("\nNo Traces.pkl files found in results/ directory")
        print("Run the model with contrastive analysis to generate traces first.")
        return
    
    # Sort by modification time, most recent first
    pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    print(f"\nFound {len(pkl_files)} trace file(s)")
    print(f"Analyzing most recent: {pkl_files[0].parent.name}\n")
    
    try:
        with open(pkl_files[0], 'rb') as f:
            traces = pickle.load(f)
        
        if not traces:
            print("No traces found in file")
            return
        
        print(f"Loaded {len(traces)} traces\n")
        
        # Statistics across all traces
        total_contrastive_count = 0
        all_original_rewards = []
        all_contrastive_rewards = []
        
        # Detailed analysis for first few traces
        for trace_idx, trace in enumerate(traces[:3]):
            print("=" * 70)
            print(f"TRACE {trace_idx}")
            print("=" * 70)
            
            # Original trajectory info
            orig_rewards = trace.rewards if hasattr(trace, 'rewards') else []
            orig_reward_sum = trace.reward_sum if hasattr(trace, 'reward_sum') else sum(orig_rewards)
            orig_length = trace.length if hasattr(trace, 'length') else len(orig_rewards)
            
            print(f"\nOriginal Trajectory:")
            print(f"  Length: {orig_length} steps")
            print(f"  Total reward: {orig_reward_sum}")
            if orig_length > 0:
                print(f"  Average reward per step: {orig_reward_sum/orig_length:.2f}")
            
            # Contrastive trajectories
            contrastive = trace.contrastive if hasattr(trace, 'contrastive') else []
            
            if not contrastive:
                print(f"\n  No contrastive trajectories for this trace")
                continue
            
            print(f"\n  Contrastive Trajectories: {len(contrastive)}")
            
            for c_idx, contra in enumerate(contrastive[:5]):  # Show first 5
                # Handle both object and dict formats
                if hasattr(contra, 'rewards'):
                    c_rewards = contra.rewards
                    c_state_id = contra.id if hasattr(contra, 'id') else None
                    c_importance = contra.importance if hasattr(contra, 'importance') else 0
                    c_actions = contra.actions if hasattr(contra, 'actions') else []
                    c_start_idx = contra.start_idx if hasattr(contra, 'start_idx') else None
                else:
                    c_rewards = contra.get('rewards', [])
                    c_state_id = contra.get('id', None)
                    c_importance = contra.get('importance', 0)
                    c_actions = contra.get('actions', [])
                    c_start_idx = contra.get('start_idx', None)
                
                c_reward_sum = sum(c_rewards) if c_rewards else 0
                c_length = len(c_rewards)
                
                print(f"\n    Contrastive {c_idx}:")
                if c_state_id:
                    print(f"      Divergence at state: {c_state_id}")
                if c_start_idx is not None:
                    print(f"      Start index: {c_start_idx}")
                print(f"      Length: {c_length} steps")
                print(f"      Total reward: {c_reward_sum}")
                if c_length > 0:
                    print(f"      Average reward per step: {c_reward_sum/c_length:.2f}")
                print(f"      Importance score: {c_importance:.4f}")
                
                # Show reward comparison for corresponding segment
                if c_state_id and len(c_state_id) >= 2:
                    orig_start = c_state_id[1]
                    orig_segment_rewards = orig_rewards[orig_start:orig_start + c_length]
                    orig_segment_sum = sum(orig_segment_rewards)
                    
                    reward_diff = c_reward_sum - orig_segment_sum
                    print(f"      Original segment reward: {orig_segment_sum}")
                    print(f"      Reward difference: {reward_diff:+.1f}")
                    
                    if reward_diff > 0:
                        print(f"        → Contrastive action led to BETTER outcome")
                    elif reward_diff < 0:
                        print(f"        → Contrastive action led to WORSE outcome")
                    else:
                        print(f"        → Same outcome")
                
                # Collect for statistics
                all_contrastive_rewards.append(c_reward_sum)
                total_contrastive_count += 1
            
            if len(contrastive) > 5:
                print(f"\n    ... and {len(contrastive) - 5} more contrastive trajectories")
            
            all_original_rewards.append(orig_reward_sum)
        
        # Overall statistics
        if len(traces) > 3:
            print(f"\n... and {len(traces) - 3} more traces")
            
            # Count all contrastive trajectories
            for trace in traces[3:]:
                contrastive = trace.contrastive if hasattr(trace, 'contrastive') else []
                total_contrastive_count += len(contrastive)
                
                orig_reward_sum = trace.reward_sum if hasattr(trace, 'reward_sum') else 0
                all_original_rewards.append(orig_reward_sum)
                
                for contra in contrastive:
                    if hasattr(contra, 'rewards'):
                        c_reward_sum = sum(contra.rewards)
                    else:
                        c_reward_sum = sum(contra.get('rewards', []))
                    all_contrastive_rewards.append(c_reward_sum)
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)
        
        print(f"\nTotal traces analyzed: {len(traces)}")
        print(f"Total contrastive trajectories: {total_contrastive_count}")
        
        if all_original_rewards:
            print(f"\nOriginal Trajectory Rewards:")
            print(f"  Mean: {np.mean(all_original_rewards):.2f}")
            print(f"  Std: {np.std(all_original_rewards):.2f}")
            print(f"  Min: {np.min(all_original_rewards):.2f}")
            print(f"  Max: {np.max(all_original_rewards):.2f}")
        
        if all_contrastive_rewards:
            print(f"\nContrastive Trajectory Rewards:")
            print(f"  Mean: {np.mean(all_contrastive_rewards):.2f}")
            print(f"  Std: {np.std(all_contrastive_rewards):.2f}")
            print(f"  Min: {np.min(all_contrastive_rewards):.2f}")
            print(f"  Max: {np.max(all_contrastive_rewards):.2f}")
    
    except Exception as e:
        print(f"\nError loading traces: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure the pickle file format is compatible.")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    inspect_contrastive_rewards()
