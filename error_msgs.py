def initiate(self, seed=0, evaluation_reset=False):
        config = self.config
        env = gym.make(config["counterfactual_params"]['env']['id'], render_mode='rgb_array')
        # gym/gymnasium seeding differs between versions and wrappers.
        # Try several approaches so this works with older gym, gymnasium, and custom wrappers.
        try:
            if hasattr(env, 'seed') and callable(getattr(env, 'seed')):
                env.seed(seed)
        except Exception:
            pass
        try:
            # gymnasium: reset accepts seed kwarg
            if hasattr(env, 'reset'):
                try:
                    env.reset(seed=seed)
                except TypeError:
                    # some env.reset don't accept seed kwarg
                    pass
        except Exception:
            pass
        # seed action/observation spaces if available
        try:
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
                env.action_space.seed(seed)
        except Exception:
            pass
        try:
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'seed'):
                env.observation_space.seed(seed)
        except Exception:
            pass
        algo = config["counterfactual_params"]["algorithm"]
        agent = None
        try:
            model_dir = config["counterfactual_params"]['model_dir'] or self.load_path or 'agents/taxi_sb3'
            # find latest .zip model in model_dir
            model_files=[]
            for ext in ('*.zip', '*.pth', '*.pt'):
                model_files.extend(sorted(glob.glob(join(model_dir, ext)), key=os.path.getmtime, reverse=True))
            # Ensure the directory exists
            print(f"model file path {model_files[-1]}")

            if model_files:
                framework = config["counterfactual_params"]["framework"]
                AdapterClass = {
                    "SB3": StableBaselines3Adapter,
                    "rllib": RllibAdapter
                }
                print("after")
                framework_adapter = AdapterClass.get(framework)
                if not framework_adapter:
                    raise ValueError(f"Unknown/unimplemented framework: {framework}")
                agent = StableBaselines3Adapter(config, config["counterfactual_params"], config["xrl_config"])
                config["counterfactual_params"]["model_file"] = model_files[-1] # add model file path to the config file
                agent.load_model(model_files[-1])
        except Exception:
            agent = None

        if agent is None:
            # Provide a helpful error rather than re-raising a suppressed exception
            msg = (
                "No Stable-Baselines3 "
                "model could be loaded.\n"
                "Ensure your agent metadata contains an 'agent' section that understands,\n"
                "or place a SB3 .zip model in the folder pointed to by 'model_dir' or 'self.load_path'.\n"
                f"Tried model_dir='{config.get('model_dir')}', load_path='{self.load_path}'."
            )
            raise RuntimeError(msg)
        if evaluation_reset:
            evaluation_reset.training = False
            evaluation_reset.close()
        try:
            self.env = env
        except Exception:
            pass
        return env, agent
    
