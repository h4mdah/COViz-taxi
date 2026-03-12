"""Stable-Baselines3 adapter for Taxi interface.

Provides a unified `SB3Adapter` that exposes the same methods expected
by the existing Taxi interface code:
  - act(state)
  - get_state_action_values(state)
  - set_writer(writer)
  - load(filename)
  - save(filename)
  - close()

The adapter will try to detect whether the wrapped model is SB3 DQN,
SB3 PPO, or the project's custom PPO (tools.custom_ppo.PPO) and will
use the most appropriate logic for action selection and value/probability
extraction. When precise values are unavailable it falls back to
approximating probabilities by repeated stochastic predictions.
"""
from typing import Any, Optional
import numpy as np
import importlib


class SB3Adapter:
    """Adapter wrapping Stable-Baselines3 (and similar) models.

    Args:
        model: the loaded SB3 model instance (or custom PPO instance)
        action_space: env.action_space (used to determine `n` actions)
        config: optional config dict — adapter will read `algorithm` if present
        obs_dim: optional observation/vector dimension (used for custom PPO one-hot)
        sample_count: how many stochastic samples to use for approximating probs
    """

    def __init__(self, model: Any, action_space: Any, config: Optional[dict] = None,
                 obs_dim: Optional[int] = None, sample_count: int = 64):
        self.model = model
        self.action_space = action_space
        self.sample_count = int(sample_count)
        self.previous_action = None
        self.previous_state = None
        self.writer = None
        self.obs_dim = obs_dim or 500

        # Determine algorithm preference from config if provided
        self.config = config or {}
        self.preferred_algo = None
        if 'algorithm' in self.config:
            self.preferred_algo = str(self.config.get('algorithm'))

        # Heuristics to detect wrapped model type
        self.model_type = self._detect_model_type()

    def _detect_model_type(self):
        """Return one of: 'DQN', 'PPO', 'CUSTOM_PPO', or fallback 'SB3'."""
        # Prefer explicit config
        if self.preferred_algo is not None:
            alg = self.preferred_algo.upper()
            if 'DQN' in alg:
                return 'DQN'
            if 'PPO' in alg:
                return 'PPO'
        # Inspect model class name
        try:
            cls_name = self.model.__class__.__name__
            if 'DQN' in cls_name:
                return 'DQN'
            if 'PPO' in cls_name:
                return 'PPO'
        except Exception:
            pass
        # Check for SB3-ish internals
        try:
            policy = getattr(self.model, 'policy', None)
            if policy is not None:
                if hasattr(policy, 'q_net'):
                    return 'DQN'
                # SB3 ActorCriticPolicy -> treat as PPO-like
                return 'PPO'
        except Exception:
            pass
        # Check for custom PPO implementation (tools.custom_ppo)
        if hasattr(self.model, 'select_action'):
            return 'CUSTOM_PPO'
        # fallback
        return 'SB3'

    def act(self, state: Any) -> int:
        """Return an integer action for the given observation/state."""
        # store previous_state for upstream code that expects it
        try:
            self.previous_state = state
        except Exception:
            pass

        # CUSTOM_PPO: build tensor and call internal policy act (keeps behavior of select_action)
        if self.model_type == 'CUSTOM_PPO':
            try:
                state_vec = self._maybe_one_hot(state)
                # lazy import torch
                import importlib
                torch = importlib.import_module('torch')

                # Determine device: use agent.device if present, else try policy params
                device = getattr(self.model, 'device', None)
                if device is None:
                    try:
                        policy = getattr(self.model, 'policy', None)
                        device = next(policy.parameters()).device
                    except Exception:
                        device = None

                state_t = torch.FloatTensor(np.asarray(state_vec).reshape(1, -1))
                if device is not None:
                    state_t = state_t.to(device)

                # Prefer calling internal policy_old.act if available (matches snippet)
                policy_old = getattr(self.model, 'policy_old', None)
                if policy_old is not None and hasattr(policy_old, 'act'):
                    # call policy_old.act(state, memory, greedy)
                    act_t = policy_old.act(state_t, None, True)
                    act_np = act_t.cpu().data.numpy()
                    if getattr(self.model, 'is_continuous', False):
                        a = act_np.flatten()
                    else:
                        a = int(act_np.item())
                    self.previous_action = a
                    return a

                # Fallback: call model.select_action if present
                if hasattr(self.model, 'select_action'):
                    return_val = self.model.select_action(state_vec, greedy=True)
                    if isinstance(return_val, tuple):
                        a = int(return_val[0])
                    else:
                        a = int(return_val)
                    self.previous_action = a
                    return a
            except Exception:
                pass

        # SB3 models: use predict
        try:
            a, _ = self.model.predict(state, deterministic=True)
            a = int(np.asarray(a).item())
            self.previous_action = a
            return a
        except Exception:
            # fallback: do non-deterministic predict then pick most common
            try:
                counts = {}
                for _ in range(max(1, min(16, self.sample_count))):
                    a, _ = self.model.predict(state, deterministic=False)
                    ai = int(np.asarray(a).item())
                    counts[ai] = counts.get(ai, 0) + 1
                # pick argmax
                if counts:
                    a = max(counts.items(), key=lambda kv: kv[1])[0]
                    self.previous_action = int(a)
                    return int(a)
            except Exception:
                pass
        # last resort
        self.previous_action = 0
        return 0

    def get_state_action_values(self, state: Any):
        """Return per-action values or probabilities for the given state.

        - For DQN: attempts to extract Q-values from the policy.q_net network.
        - For SB3 PPO: attempts to run policy to obtain distribution/probs.
        - For CUSTOM_PPO: uses policy.actor to obtain action probabilities if available.
        - Otherwise: approximates probabilities by repeated stochastic `predict()` calls.
        """
        try:
            import numpy as _np
            import importlib
            torch = importlib.import_module('torch')
        except Exception:
            torch = None

        n = getattr(self.action_space, 'n', None) or 0
        if n == 0:
            return np.array([])

        # DQN direct Q extraction
        if self.model_type == 'DQN':
            try:
                policy = getattr(self.model, 'policy', None)
                if policy is not None and hasattr(policy, 'q_net') and torch is not None:
                    q_net = policy.q_net
                    obs_arr = _np.asarray(state)
                    if obs_arr.ndim == 0:
                        obs_arr = _np.expand_dims(obs_arr, 0)
                    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32).to(next(q_net.parameters()).device)
                    with torch.no_grad():
                        q = q_net(obs_t)
                    q = q.cpu().numpy().flatten()
                    return q
            except Exception:
                # fall through to sampling approximation
                pass

        # CUSTOM_PPO: try actor network
        if self.model_type == 'CUSTOM_PPO':
            try:
                # convert state to vector
                state_vec = self._maybe_one_hot(state)
                if torch is not None and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor'):
                    policy = self.model.policy
                    obs_t = torch.FloatTensor(np.asarray(state_vec).reshape(1, -1)).to(next(policy.parameters()).device)
                    with torch.no_grad():
                        action_probs = policy.actor(obs_t)
                    return action_probs.cpu().numpy().flatten()
            except Exception:
                pass

        # SB3 PPO or fallback: try to use policy networks when accessible
        try:
            policy = getattr(self.model, 'policy', None)
            if policy is not None and torch is not None:
                # Try to create a tensor and call policy to get distribution
                try:
                    obs_arr = _np.asarray(state)
                    if obs_arr.ndim == 0:
                        obs_arr = _np.expand_dims(obs_arr, 0)
                    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32).to(next(policy.parameters()).device)
                    # Try calling policy.get_distribution if available
                    if hasattr(policy, 'get_distribution'):
                        with torch.no_grad():
                            dist = policy.get_distribution(obs_t)
                            probs = dist.distribution.probs if hasattr(dist.distribution, 'probs') else None
                            if probs is not None:
                                return probs.cpu().numpy().flatten()
                    # Try actor forward (custom policies)
                    if hasattr(policy, 'actor'):
                        with torch.no_grad():
                            out = policy.actor(obs_t)
                        # if softmaxed already, return flatten
                        arr = out.cpu().numpy().flatten()
                        # if arr sums ~1, treat as probs
                        if np.isclose(arr.sum(), 1.0, atol=0.1):
                            return arr
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback approximation: repeated stochastic prediction to estimate probs
        try:
            from collections import Counter
            counts = Counter()
            for _ in range(max(1, self.sample_count)):
                a, _ = self.model.predict(state, deterministic=False)
                ai = int(np.asarray(a).item())
                counts[ai] += 1
            probs = np.zeros((n,))
            total = float(sum(counts.values())) if counts else 1.0
            for i in range(n):
                probs[i] = counts.get(i, 0) / total
            return probs
        except Exception:
            # last resort: zeros
            return np.zeros((n,))

    def _maybe_one_hot(self, state: Any):
        """Convert an integer Taxi state to one-hot vector of length `obs_dim`.

        If `state` is already a vector/array it is returned as-is.
        """
        try:
            if isinstance(state, (int, np.integer)):
                vec = np.zeros((self.obs_dim,), dtype=float)
                idx = int(state)
                if 0 <= idx < len(vec):
                    vec[idx] = 1.0
                return vec
            arr = np.asarray(state)
            if arr.ndim == 0:
                # scalar fallback
                vec = np.zeros((self.obs_dim,), dtype=float)
                idx = int(arr)
                if 0 <= idx < len(vec):
                    vec[idx] = 1.0
                return vec
            return arr
        except Exception:
            return np.zeros((self.obs_dim,), dtype=float)

    def set_writer(self, writer: Any):
        self.writer = writer

    def close(self):
        try:
            if hasattr(self.model, 'env') and getattr(self.model.env, 'close', None):
                try:
                    self.model.env.close()
                except Exception:
                    pass
        except Exception:
            pass

    def load(self, filename: Optional[str] = None):
        # For SB3 models, delegate to model.load if available (not a typical instance method)
        return self.model

    def save(self, filename: Optional[str] = None):
        try:
            if filename and hasattr(self.model, 'save'):
                self.model.save(str(filename))
        except Exception:
            pass


# Convenience factory
def make_sb3_adapter(model: Any, action_space: Any, config: Optional[dict] = None,
                      obs_dim: Optional[int] = None, sample_count: int = 64) -> SB3Adapter:
    return SB3Adapter(model, action_space, config=config, obs_dim=obs_dim, sample_count=sample_count)


def load_model(path: str, action_space: Any, config: Optional[dict] = None,
               obs_dim: Optional[int] = None, sample_count: int = 64) -> SB3Adapter:
    """Load a model from disk and return a wrapped SB3Adapter.

    - If `path` ends with .zip, this function will try to load a Stable-Baselines3
      model (DQN first, then PPO) and wrap it.
    - If `path` ends with .pth, it will attempt to load the project's custom
      `tools.custom_ppo.PPO` agent (if available) and wrap it.
    """
    path = str(path)
    # prefer explicit obs_dim if provided
    od = obs_dim or 500
    # .zip -> SB3
    try:
        if path.endswith('.zip'):
            try:
                from stable_baselines3 import DQN
                model = DQN.load(path)
                return make_sb3_adapter(model, action_space, config=config, obs_dim=od, sample_count=sample_count)
            except Exception:
                try:
                    from stable_baselines3 import PPO as SB3PPO
                    model = SB3PPO.load(path)
                    return make_sb3_adapter(model, action_space, config=config, obs_dim=od, sample_count=sample_count)
                except Exception:
                    raise

        # .pth -> custom PPO saved state dict
        if path.endswith('.pth') or path.endswith('.pt'):
            try:
                from tools.custom_ppo import PPO as CustomPPO, Memory
                import importlib
                torch = importlib.import_module('torch')

                # Build PPO with default Taxi config (mirrors taxi_interface usage)
                state_dim = od
                action_dim = getattr(action_space, 'n', None) or 0
                ppo_conf = {
                    'lr': 0.002,
                    'betas': (0.9, 0.999),
                    'gamma': 0.99,
                    'eps_clip': 0.2,
                    'K_epochs': 4,
                    'nn_type': 'tanh',
                    'action_std': 0.6,
                    'lam_a': 0.0,
                    'normalize_rewards': False
                }
                ppo_agent = CustomPPO(state_dim, action_dim, ppo_conf, use_gpu=True, is_continuous=False)
                # load state dict
                ppo_agent.policy.load_state_dict(torch.load(path))
                ppo_agent.policy.eval()
                return make_sb3_adapter(ppo_agent, action_space, config=config, obs_dim=od, sample_count=sample_count)
            except Exception:
                raise

    except Exception:
        raise

    # if nothing matched, raise
    raise RuntimeError(f"Could not load model from path: {path}")

def train_model(env_id="Taxi-v3-COViz", total_timesteps=100_000, model_dir="agents/taxi_sb3", algo="DQN"):
    """Train a model for the Taxi environment."""
    from pathlib import Path
    import gym
    import numpy as np
    
    model_path_dir = Path(model_dir)
    model_path_dir.mkdir(parents=True, exist_ok=True)
    
    train_env = gym.make(env_id)
    
    if algo == "PPO":
        import torch
        from tools.custom_ppo import PPO as CustomPPO, Memory
        print(f"Starting PPO training on {env_id} for {total_timesteps} steps...")
        
        state_dim = 500 # Taxi
        action_dim = train_env.action_space.n
        ppo_conf = {
            'lr': 0.002,
            'betas': (0.9, 0.999),
            'gamma': 0.99,
            'eps_clip': 0.2,
            'K_epochs': 4,
            'nn_type': 'tanh',
            'action_std': 0.6,
            'lam_a': 0.0,
            'normalize_rewards': False
        }
        ppo_agent = CustomPPO(state_dim, action_dim, ppo_conf, use_gpu=True, is_continuous=False)
        memory = Memory()
        
        max_ep_len = 200 # Taxi default
        update_timestep = 2000 
        time_step = 0
        
        # Training Loop
        for i_episode in range(1, total_timesteps // max_ep_len + 1):
            state, _ = train_env.reset()
            for t in range(max_ep_len):
                time_step += 1
                
                # One-hot encode state
                state_vec = np.zeros(500)
                state_vec[state] = 1.0
                
                action = ppo_agent.select_action(state_vec, memory)
                state, reward, done, truncated, _ = train_env.step(action)
                
                memory.rewards.append(reward)
                memory.is_terminals.append(done or truncated)
                
                if time_step % update_timestep == 0:
                    ppo_agent.update(memory)
                    memory.clear_memory()
                    time_step = 0
                
                if done or truncated:
                    break
            
            if i_episode % 100 == 0:
                 print(f"Episode {i_episode} complete")

        final_model = model_path_dir / "model_final.pth"
        torch.save(ppo_agent.policy.state_dict(), str(final_model))
    else:
        from stable_baselines3 import DQN
        
        model = DQN("MlpPolicy", train_env, verbose=1)
        
        print(f"Starting training on {env_id} for {total_timesteps} steps...")
        model.learn(total_timesteps=total_timesteps)
        
        final_model = model_path_dir / "model_final.zip"
        model.save(str(final_model))
        
    print(f"Training finished. Final model saved to {final_model}")
    train_env.close()
    return final_model
