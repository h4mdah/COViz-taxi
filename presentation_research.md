# COViz-taxi: Contrastive Visualizations for RL Interpretability

A tool for analyzing agent policies through counterfactual trajectory generation and reward decomposition.

````carousel
# 1. Problem Statement: Policy Opacity
Deep RL agents (DQN, PPO) often function as black boxes.
- We observe the policy $\pi(s)$, but rarely understand the *why* behind specific actions.
- Value functions $V(s)$ or $Q(s,a)$ provide scalar expectations but lack semantic granularity.
- **Challenge**: How to interpret specific control decisions in high-dimensional or discrete state spaces?
<!-- slide -->
# 2. Methodology: Contrastive Explanations
We employ a **contrastive** approach to interpretability. Given an observed trajectory $\tau$ where the agent chose $a_t = \arg\max Q(s_t, a)$:

1.  **Intervention**: We force a counterfactual action $a'_t \neq a_t$ (e.g., the second-best action).
2.  **Rollout**: We simulate the counterfactual trajectory $\tau'$ from state $s_{t+1}' \sim P(\cdot|s_t, a'_t)$ using the original policy $\pi$.
3.  **Comparison**: We visualize $\tau$ vs. $\tau'$ side-by-side to highlight divergence in outcomes.
<!-- slide -->
# 3. Reward Decomposition (RD)
We extend the explanation by decomposing the scalar reward signal $R_t$ into semantic components:

$$R(s, a) = R_{\text{time}} + R_{\text{success}} + R_{\text{penalty}}$$

- **Visual Output**: Per-step bar charts overlaid on frames showing the instantaneous component rewards for both $\tau$ and $\tau'$.
- This reveals if an action was avoided due to, for example, high collision penalty vs. low efficiency reward.
<!-- slide -->
# 4. Reproducibility & Debugging
**New Feature**: Deterministic State Initialization.

To support rigorous analysis of edge cases (e.g., "spinning" behavior or bottleneck states), we introduced forced initialization:
- `env.reset(options={'start_state': s_0})` equivalent.
- Allows for **ablation studies** on specific states where $Q(s, a) \approx Q(s, a')$, identifying policy instability.
<!-- slide -->
# 5. Architecture
- **Environment**: Gymnasium (Taxi-v3).
- **Agent**: Compatible with Stable Baselines3 (DQN).
- **Visualization**: OpenCV-based frame stitching with Matplotlib overlays.
- **Metric**: Action-Gap Analysis to select "interesting" states (small difference in Q-values).
````

## Research Q&A

Common technical questions regarding the framework.

### Q1: How is the counterfactual action selected?
**A:** By default, we select the action with the **second-highest Q-value** ($a' = \arg\max_{a \neq \pi(s)} Q(s, a)$). This represents the "closest competitor" to the chosen greedy action, providing the most relevant contrastive explanation.

### Q2: Is the counterfactual trajectory deterministic?
**A:** The rollout follows the greedy policy $\pi(s)$ after the intervention step. If the environment $P(s'|s,a)$ or the policy (e.g., stochastic) has randomness, the trajectory draws samples. For Taxi-v3, transitions are deterministic, making the counterfactual unique per action.

### Q3: Does this support continuous action spaces?
**A:** Currently optimized for discrete action spaces (Taxi, Highway-env). For continuous spaces, we would need to define a relevant "counterfactual perturbation" (e.g., $\pm \delta$ on the action vector) rather than discrete selection.

### Q4: How do you handle Reward Decomposition?
**A:** We use a wrapper or custom interface (see `taxi_interface.py`) that accesses the environment's internal logic to return a vector reward instead of a scalar. This requires white-box access to the reward function during evaluation.

### Q5: What is the computational overhead?
**A:** The overhead is linear with the number of generated traces. For each trace step $t$, we may fork a simulation for $k$ steps. Total complexity is $O(T \cdot k)$, where $T$ is trace length and $k$ is the rollout horizon. With `multiprocessing`, we can parallelize generation.
