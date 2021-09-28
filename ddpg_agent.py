from typing import get_type_hints
import numpy as np
import random
import copy
from collections import namedtuple, deque
from pprint import pprint

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed=0, noise_type="normal", 
                 learn_every=16, n_learn=16, alpha_actor=1e-3, alpha_critic=1e-3,
                 batch_size=256, buffer_size=int(1e6), gamma=.99, tau=1e-3,
                 desired_distance=.7, scalar=.05, scalar_decay=.99,
                 normal_scalar=.25):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        # save hyperparameters for print
        self.hyperparameters = '\n'.join(f"{key:>17}: {value}" for key, value in locals().items() if key != 'self')
        
        self.state_size = state_size
        self.action_size = action_size

        self.learn_every = learn_every
        self.n_learn = n_learn

        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau

        self.noise_type = noise_type

        # parameter noise
        self.distances = []
        self.desired_distance = desired_distance
        self.scalar = scalar
        self.scalar_decay = scalar_decay

        # normal noise
        self.normal_scalar = normal_scalar

        # step counter
        self.t = 0

        # reset/seed all random generators
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_regular = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_regular.parameters(), lr=alpha_actor)

        # Critic Network (w/ Target Network)
        self.critic_regular = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_regular.parameters(), lr=alpha_critic)

        self.actor_noised = Actor(state_size, action_size, random_seed).to(device)

        # hard update to ensure that regular and target start with same values
        self.soft_update(self.critic_regular, self.critic_target, 1.)
        self.soft_update(self.actor_regular, self.actor_target, 1.) 

        # Ornstein–Uhlenbeck Process noise
        if noise_type == "ou":
            self.ou_noise = OUNoise(action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.t += 1
        # Save experience / reward
        self.memory.add_multi(states, actions, rewards, next_states, dones)

        if not (self.t % self.learn_every == 0 and len(self.memory) > self.batch_size):
            return

        for _ in range(self.n_learn):
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_regular.eval()
        self.actor_noised.eval()
        with torch.no_grad():
            action = self.actor_regular(state).cpu().data.numpy()

            if add_noise:

                if self.noise_type == "param":
                    # hard copy the actor_regular to actor_noised
                    self.actor_noised.load_state_dict(self.actor_regular.state_dict().copy())
                    # add noise to the copy
                    self.actor_noised.add_parameter_noise(self.scalar)
                    # get the next action values from the noised actor
                    action_noised = self.actor_noised(state).cpu().data.numpy()
                    # meassure the distance between the action values from the regular and 
                    # the noised actor to adjust the amount of noise that will be added next round
                    distance = np.sqrt(np.mean(np.square(action-action_noised)))
                    # for stats and print only
                    self.distances.append(distance)
                    # adjust the amount of noise given to the actor_noised
                    if distance > self.desired_distance:
                        self.scalar *= self.scalar_decay
                    if distance < self.desired_distance:
                        self.scalar /= self.scalar_decay
                    # set the noised action as action
                    action = action_noised

                elif self.noise_type == "ou":
                    action += self.ou_noise.sample()

                else:
                    action += np.random.randn(self.action_size) * self.normal_scalar

        self.actor_regular.train()
        return np.clip(action, -1, 1)

    # needed for ou noise only
    def reset(self):
        if hasattr(self, 'ou_noise'):
            self.ou_noise.reset()

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_regular(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_regular(states)
        actor_loss = -self.critic_regular(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_regular, self.critic_target, self.tau)
        self.soft_update(self.actor_regular, self.actor_target, self.tau)                     

    def soft_update(self, regular_model, target_model, tau):
        for target_param, regular_param in zip(target_model.parameters(), regular_model.parameters()):
            target_param.data.copy_(tau*regular_param.data + (1.0-tau)*target_param.data)

    def __str__(self):
        return f"\n{'#'*80}\n\nHyperparameters: \n\n{self.hyperparameters}\n\n{self.actor_regular}\n\n{self.critic_regular}\n\n{'#'*80}\n\n"

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object. """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def add_multi(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.add(state, action, reward, next_state, done) 
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)