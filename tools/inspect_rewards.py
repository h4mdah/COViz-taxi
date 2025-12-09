"""
Inspect the reward mechanism of the Taxi environment.
This script helps you understand what rewards are given for different actions.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np

def inspect_taxi_rewards():
    """Demonstrate and explain the Taxi-v3 reward mechanism."""
    
    print("=" * 70)
    print("TAXI-V3 REWARD MECHANISM")
    print("=" * 70)
    
    print("\nStandard Taxi-v3 Rewards:")
    print("  • -1 per time step (encourages efficiency)")
    print("  • +20 for successful passenger delivery")
    print("  • -10 for illegal pick-up or drop-off attempts")
    print()
    
    # Create the environment
    env = gym.make('Taxi-v3', render_mode='ansi')
    
    print("=" * 70)
    print("EXAMPLE EPISODE WITH REWARD BREAKDOWN")
    print("=" * 70)
    
    obs, info = env.reset(seed=42)
    print("\nInitial state:")
    print(env.render())
    
    total_reward = 0
    step_count = 0
    done = False
    
    action_names = {
        0: "Move South",
        1: "Move North", 
        2: "Move East",
        3: "Move West",
        4: "Pick up passenger",
        5: "Drop off passenger"
    }
    
    print("\nTaking some example actions to show rewards:\n")
    
    # Try a few random actions to demonstrate rewards
    for i in range(10):
        if done:
            break
            
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: {action_names[action]} → Reward: {reward:+.0f}")
        
        if reward == -10:
            print("  └─ Illegal action penalty!")
        elif reward == 20:
            print("  └─ SUCCESS! Passenger delivered!")
        elif reward == -1:
            print("  └─ Normal step cost")
        
        # Show state after significant events
        if reward != -1:
            print(env.render())
    
    print(f"\nTotal reward after {step_count} steps: {total_reward}")
    env.close()
    
    print("\n" + "=" * 70)
    print("CHECKING REWARDS FROM SAVED TRACES")
    print("=" * 70)
    
    # Load and display rewards from saved traces (.pkl files) if available
    results_dir = REPO_ROOT / "results"
    pkl_files = list(results_dir.glob("run_*/Traces.pkl"))
    
    if pkl_files:
        import pickle
        
        # Sort by modification time, most recent first
        pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        print(f"\nFound {len(pkl_files)} trace file(s)")
        print(f"Analyzing most recent: {pkl_files[0].parent.name}")
        
        try:
            # Try to import dependencies for pickle to work
            try:
                from counterfactual_outcomes.common import Trace
                from counterfactual_outcomes.contrastive_online import ContrastiveTrajectory
            except ImportError as e:
                print(f"\nWarning: Could not import trace modules: {e}")
                print("Install missing dependencies (cv2, etc.) to load traces")
                return
            
            with open(pkl_files[0], 'rb') as f:
                traces = pickle.load(f)
            
            if traces:
                print(f"\nLoaded {len(traces)} traces from {pkl_files[0].name}")
                
                for i, trace in enumerate(traces[:3]):  # Show first 3 traces
                    # Handle both dict and object formats
                    if hasattr(trace, 'rewards'):
                        rewards = trace.rewards
                        reward_sum = trace.reward_sum if hasattr(trace, 'reward_sum') else sum(rewards)
                        length = trace.length if hasattr(trace, 'length') else len(rewards)
                    else:
                        rewards = trace.get('rewards', [])
                        reward_sum = trace.get('reward_sum', sum(rewards))
                        length = trace.get('length', len(rewards))
                    
                    print(f"\nTrace {i}:")
                    print(f"  Length: {length} steps")
                    print(f"  Total reward: {reward_sum}")
                    if length > 0:
                        print(f"  Average reward per step: {reward_sum/length:.2f}")
                    
                    # Count different reward types
                    if rewards:
                        step_costs = sum(1 for r in rewards if r == -1)
                        penalties = sum(1 for r in rewards if r == -10)
                        successes = sum(1 for r in rewards if r == 20)
                        
                        print(f"  Breakdown:")
                        print(f"    - Normal steps (-1): {step_costs}")
                        print(f"    - Illegal actions (-10): {penalties}")
                        print(f"    - Successful deliveries (+20): {successes}")
                
                # Show summary of all traces
                if len(traces) > 3:
                    print(f"\n... and {len(traces) - 3} more traces")
                    
                    all_rewards = []
                    for trace in traces:
                        if hasattr(trace, 'reward_sum'):
                            all_rewards.append(trace.reward_sum)
                        elif isinstance(trace, dict):
                            all_rewards.append(trace.get('reward_sum', 0))
                    
                    if all_rewards:
                        print(f"\nOverall Statistics:")
                        print(f"  Average total reward: {np.mean(all_rewards):.2f}")
                        print(f"  Best total reward: {max(all_rewards):.2f}")
                        print(f"  Worst total reward: {min(all_rewards):.2f}")
                
                # Check for contrastive trajectories
                print("\n" + "=" * 70)
                print("CONTRASTIVE TRAJECTORY REWARDS")
                print("=" * 70)
                
                total_contrastive_count = 0
                all_contrastive_rewards = []
                
                for trace in traces:
                    contrastive = trace.contrastive if hasattr(trace, 'contrastive') else []
                    total_contrastive_count += len(contrastive)
                    
                    for contra in contrastive:
                        if contra is None:
                            continue
                        if hasattr(contra, 'rewards'):
                            c_rewards = contra.rewards
                        elif isinstance(contra, dict):
                            c_rewards = contra.get('rewards', [])
                        else:
                            continue
                        if c_rewards:
                            all_contrastive_rewards.append(sum(c_rewards))
                
                if total_contrastive_count > 0:
                    print(f"\nFound {total_contrastive_count} contrastive trajectories across all traces")
                    
                    # Show detailed comparison for first trace
                    first_trace = traces[0]
                    orig_rewards = first_trace.rewards if hasattr(first_trace, 'rewards') else []
                    contrastive = first_trace.contrastive if hasattr(first_trace, 'contrastive') else []
                    
                    if contrastive:
                        print(f"\nDetailed comparison for Trace 0:")
                        print(f"  Original trajectory reward: {sum(orig_rewards)}")
                        
                        for c_idx, contra in enumerate(contrastive[:5]):
                            if contra is None:
                                continue
                            if hasattr(contra, 'rewards'):
                                c_rewards = contra.rewards
                                c_state_id = contra.id if hasattr(contra, 'id') else None
                            elif isinstance(contra, dict):
                                c_rewards = contra.get('rewards', [])
                                c_state_id = contra.get('id', None)
                            else:
                                continue
                            
                            c_reward_sum = sum(c_rewards) if c_rewards else 0
                            
                            print(f"\n  Contrastive {c_idx}:")
                            if c_state_id and len(c_state_id) >= 2:
                                print(f"    Diverged at step: {c_state_id[1]}")
                                orig_start = c_state_id[1]
                                c_length = len(c_rewards)
                                orig_segment = orig_rewards[orig_start:orig_start + c_length]
                                orig_segment_sum = sum(orig_segment)
                                
                                print(f"    Contrastive reward: {c_reward_sum}")
                                print(f"    Original segment reward: {orig_segment_sum}")
                                print(f"    Difference: {c_reward_sum - orig_segment_sum:+.1f}", end="")
                                
                                if c_reward_sum > orig_segment_sum:
                                    print(" (Better)")
                                elif c_reward_sum < orig_segment_sum:
                                    print(" (Worse)")
                                else:
                                    print(" (Same)")
                        
                        if len(contrastive) > 5:
                            print(f"\n  ... and {len(contrastive) - 5} more contrastive trajectories")
                    
                    if all_contrastive_rewards:
                        print(f"\nContrastive Trajectory Statistics:")
                        print(f"  Mean reward: {np.mean(all_contrastive_rewards):.2f}")
                        print(f"  Std: {np.std(all_contrastive_rewards):.2f}")
                        print(f"  Min: {np.min(all_contrastive_rewards):.2f}")
                        print(f"  Max: {np.max(all_contrastive_rewards):.2f}")
                else:
                    print("\nNo contrastive trajectories found in traces")
                    print("Run with contrastive analysis enabled to generate them")
        
        except Exception as e:
            print(f"\nError loading traces: {e}")
            import traceback
            traceback.print_exc()
            print("\nMake sure all dependencies are installed.")
    else:
        print("\nNo Traces.pkl files found in results/ directory")
        print("Run the model to generate traces first.")
    
    print("\n" + "=" * 70)
    print("REWARD INSPECTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    inspect_taxi_rewards()
