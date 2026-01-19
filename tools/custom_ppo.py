
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, NN_conf, is_continuous=False, action_std_init=0.6, use_gpu=True):
        super(ActorCritic, self).__init__()
        self.is_continuous = is_continuous
        self.action_dim = action_dim

        if NN_conf == 'tanh':
            activation = nn.Tanh()
        else: # 'relu' or default
            activation = nn.ReLU()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, action_dim),
            nn.Tanh() if self.is_continuous else nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, 1)
        )

        self.set_device(use_gpu)

        if self.is_continuous:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        else:
            self.action_var = None

    def set_device(self, use_gpu=False):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory, greedy=False):
        if self.is_continuous:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        if greedy:
            if self.is_continuous:
                action = action_mean
            else:
                action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        state_value = self.critic(state)
        
        if self.is_continuous:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For continuous, action should be reshaped if necessary
            if action.dim() == 1:
                action = action.reshape(-1, self.action_dim)
                
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            
            # For discrete, action is just an index (scalar per batch item)
            if action.dim() > 1:
                action = action.squeeze(-1)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, conf_ppo, use_gpu=False, is_continuous=False):
        self.lr = conf_ppo['lr']
        self.betas = conf_ppo['betas']
        self.gamma = conf_ppo['gamma']
        self.eps_clip = conf_ppo['eps_clip']
        self.K_epochs = conf_ppo['K_epochs']
        self.is_continuous = is_continuous
        
        if is_continuous:
            self.action_std = conf_ppo.get('action_std', 0.6)
        else:
            self.action_std = None

        self.set_device(use_gpu)
        
        self.policy = ActorCritic(state_dim, action_dim, conf_ppo['nn_type'], is_continuous, self.action_std, use_gpu).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, conf_ppo['nn_type'], is_continuous, self.action_std, use_gpu).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        
        self.lam_a = conf_ppo.get('lam_a', 0.0)
        self.normalize_rewards = conf_ppo.get('normalize_rewards', False)
        
        self.loss_a = 0.0
        self.loss_max = 0.0
        self.loss_min = 0.0

    def set_device(self, use_gpu=True, set_policy=False):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        if set_policy:
            self.policy.set_device(use_gpu)
            self.policy_old.set_device(use_gpu)

    def select_action(self, state, memory=None, greedy=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.policy_old.act(state, memory, greedy).cpu().data.numpy()
        return action.flatten() if self.is_continuous else int(action.item())

    def update(self, memory, to_tensor=False, use_gpu=True):
        self.set_device(use_gpu, set_policy=True)

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if self.normalize_rewards:
             rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # stack
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            if self.lam_a != 0 and self.is_continuous:
                 # Limitation on action change (smooth control) - only makes sense for continuous
                if len(memory.actions) > 1:
                    mu = torch.squeeze(torch.stack(memory.actions[:-1]).to(self.device), 1).detach()
                    mu_nxt = torch.squeeze(torch.stack(memory.actions[1:]).to(self.device), 1).detach()
                    loss += 0.5 * self.MseLoss(mu_nxt, mu) * self.lam_a

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
