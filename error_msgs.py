class StableBaselines3Adapter(BaseFrameworkAdapter):
    """
    Concrete adapter for Stable Baselines 3 (SB3).
    """

    def __init__(self, **kwargs):
        # We assume kwargs contains the necessary configs: 
        # 'framework_config', 'common_config', 'xrl_config', etc.
        super().__init__(**kwargs)
        self.env = self.setup_environment()
        self.configure_algorithm()
        
        # Set spaces after environment setup
        if self.env:
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

    def setup_environment(self) -> Any:
        """
        Initializes and returns a vectorized environment based on common_config.
        
        Expected keys in self.common_config: 'env_id', 'n_envs'
        """
        env_id = self.common_config.get('env_id', 'CartPole-v1')
        n_envs = self.common_config.get('n_envs', 1)
        
        # SB3 uses make_vec_env to create a vectorized environment
        return make_vec_env(env_id, n_envs=n_envs)

    def configure_algorithm(self) -> None:
        """
        Instantiates the SB3 algorithm (e.g., DQN, PPO).
        
        Expected keys in self.framework_config: 'algorithm', 'policy_type', 'kwargs'
        """
        if self.env is None:
            raise RuntimeError("Environment must be set up before configuring the algorithm.")
        
        algo_name = self.framework_config.get('algorithm', 'DQN')
        policy_type = self.framework_config.get('policy_type', 'MlpPolicy')
        algo_kwargs = self.framework_config.get('kwargs', {})
        
        # Mapping algorithm name to SB3 class
        algorithms = {
            'DQN': DQN, 
            'PPO': PPO, 
            'A2C': A2C, 
            'SAC': SAC
        }
        
        AlgoClass = algorithms.get(algo_name)
        if not AlgoClass:
            raise ValueError(f"Unknown SB3 algorithm: {algo_name}")

        self.model = AlgoClass(policy_type, self.env, **algo_kwargs)

    def train(self, plots_instance: Optional[Any] = None) -> None:
        """Executes the training loop."""
        if self.model is None:
            raise RuntimeError("Model must be configured before training.")
            
        total_timesteps = self.common_config.get('total_timesteps', 10000)
        
        # Pass callbacks if defined in configs for custom logging/hyperparam tuning
        callbacks = self.common_config.get('callbacks', None)
        
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

    def act(self, state: Any) -> Any:
        """
        Uses the SB3 model to predict the next action deterministically.
        
        The model handles numpy observation conversion.
        """
        if self.model is None:
            raise RuntimeError("SB3 model not loaded.")
            
        a, _ = self.model.predict(state, deterministic=True)
        
        # Ensure the action is a simple type (e.g., integer for discrete)
        # Note: 'a' can be a numpy array, so we return the first element.
        return a.item() if isinstance(a, _np.ndarray) else a

    def get_state_action_values(self, state: Any) -> _np.ndarray:
        """
        Retrieves Q-values directly from the network for DQN, or returns zeros for other algos.
        
        This overrides the simple placeholder in your original code with a more functional
        implementation for DQN, which is crucial for most XRL methods.
        """
        if self.model is None:
            n = getattr(self.action_space, 'n', 0) if self.action_space else 0
            return _np.zeros((n,))
            
        # Check if the model is a DQN instance (or has a Q-network)
        if isinstance(self.model, DQN):
            # Convert state to tensor and compute Q-values
            obs_tensor, _ = self.model.q_net.obs_to_tensor(state)
            with th.no_grad():
                q_values = self.model.q_net(obs_tensor)
            
            # Convert PyTorch tensor back to numpy array
            return q_values.cpu().numpy().flatten()
        else:
            # For non-Q-learning algorithms (PPO, A2C, SAC), Q-values are not directly exposed.
            n = getattr(self.action_space, 'n', 0) if self.action_space else 0
            return _np.zeros((n,)) # Placeholder as per your original design/limitation

    def save_model(self, save_path: str) -> None:
        """Saves the underlying SB3 model using its built-in save method."""
        if self.model:
            self.model.save(save_path)
            
    def load_model(self, load_path: str) -> None:
        """Loads a trained SB3 model from a file."""
        if self.model is None:
            # Need to know the algorithm type to use the static load method
            algo_name = self.framework_config.get('algorithm', 'DQN')
            algorithms = {'DQN': DQN, 'PPO': PPO, 'A2C': A2C, 'SAC': SAC}
            AlgoClass = algorithms.get(algo_name)
            
            if not AlgoClass:
                 raise ValueError(f"Unknown SB3 algorithm: {algo_name}")
            
            # SB3 load is a static method and re-creates the model
            self.model = AlgoClass.load(load_path, env=self.env, **self.framework_config.get('kwargs', {}))
        else:
            # If model exists, you can load parameters using set_parameters if required, 
            # but using the static load is the standard SB3 approach.
            print(f"Model already instantiated. Loading parameters from {load_path}...")
            self.model = self.model.load(load_path, env=self.env)


    def get_model_for_xrl(self) -> Any:
        """Returns the underlying SB3 model instance."""
        return self.model

    def get_environment_details(self) -> Tuple[Any, Any]:
        """Returns the action_space and observation_space."""
        if self.action_space is None or self.observation_space is None:
             raise RuntimeError("Environment has not been set up yet.")
        return self.observation_space, self.action_space

    def cleanup(self) -> None:
        """Performs cleanup by closing the environment."""
        super().cleanup()
