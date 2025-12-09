# Taxi Environment Reward Mechanism

## Overview

Your project uses the standard **Gymnasium Taxi-v3** environment, which has a simple but effective reward structure designed to teach the agent to efficiently pick up and deliver passengers.

## Reward Structure

| Event | Reward | Purpose |
|-------|--------|---------|
| **Each time step** | `-1` | Encourages the agent to complete tasks quickly and efficiently |
| **Successful delivery** | `+20` | Rewards the agent for correctly picking up and dropping off passengers at their destination |
| **Illegal action** | `-10` | Penalizes attempting to pick up or drop off when not allowed |

## Illegal Actions

Illegal actions (penalized with -10) include:
- Attempting to **pick up** a passenger when:
  - Not at the passenger's location
  - Already carrying a passenger
- Attempting to **drop off** a passenger when:
  - Not carrying a passenger
  - Not at the correct destination

## Example Episode Breakdown

A typical successful episode might look like:
- **Navigate to passenger**: 10 steps × (-1) = `-10` reward
- **Pick up passenger**: 1 step × (-1) = `-1` reward  
- **Navigate to destination**: 8 steps × (-1) = `-8` reward
- **Drop off passenger**: 1 step × (+20 - 1) = `+19` reward
- **Total**: -10 - 1 - 8 + 19 = **`0` reward**

Poor performance (many illegal actions or inefficient paths) results in negative total rewards.

## Where Rewards Are Used in Your Code

### 1. **Training** (`counterfactual_outcomes/train_model.py`)
The DQN agent learns by maximizing cumulative rewards during training:
```python
model = DQN("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=to_learn)
```

### 2. **Trace Collection** (`counterfactual_outcomes/common.py`)
Rewards are stored in trajectory traces for analysis:
```python
self.rewards.append(r)
self.reward_sum += r
```

### 3. **Contrastive Analysis** (`counterfactual_outcomes/main.py`)
Original and counterfactual trajectory rewards are compared:
```python
orig_rew = trace.rewards[orig_state_idx]
contra_rew = contra_traj.rewards[i]
```

## Viewing Rewards in Your Data

### From Saved Traces
If you have trace files in `traces/taxi/`, you can inspect rewards:
```python
import json
with open('traces/taxi/taxi_traces.json', 'r') as f:
    traces = json.load(f)
    
for trace in traces:
    print(f"Reward sum: {trace['reward_sum']}")
    print(f"Individual rewards: {trace['rewards']}")
```

### From Summary Data
Check `traces/taxi/taxi_traces_summary.json` for high-level statistics.

### Live Inspection
Run the inspection tool:
```bash
python tools/inspect_rewards.py
```

## Customizing Rewards

If you want to modify the reward structure:

1. **Subclass the environment** in `counterfactual_outcomes/interfaces/Taxi/environments.py`:
```python
class CustomTaxiEnv(TaxiEnvWrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Modify reward here
        custom_reward = reward * 2  # Example: double all rewards
        return obs, custom_reward, terminated, truncated, info
```

2. **Register the custom environment**:
```python
register(id='Taxi-v3-Custom', entry_point='...:CustomTaxiEnv')
```

3. **Update config** to use the new environment ID.

## Further Analysis

To understand how your trained model responds to these rewards:
- Check **TensorBoard logs** in `results/` directories
- Use `tools/inspect_selected_run.py` to analyze specific runs
- View Q-values with `agent.get_state_action_values(state)` to see how the agent values different actions
