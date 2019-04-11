# -*- coding: utf-8 -*-
from collections import deque
import random
import torch
import numpy as np

"""
When using an agent with a ROLLOUT trajectory, then instead of each
experience holding SARS' data, it holds:
# state = state at t
# action = action at t
# reward = cumulative reward from t through t+n-1
# next_state = state at t+n

where n=ROLLOUT.
"""

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout
        #self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        trajectory = (state, action, reward, next_state) #self.experience, done
        self.memory.append(trajectory)
        print("YO YO YO I added one trajectorym now the size ", len(self.memory))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        print("bBABABABABAB batch_size",batch_size )
        print("HAHAHAHAHAHA self.memory", len(self.memory))
        batch = random.sample(self.memory, k=batch_size) #memory 80 batch_size 128, can't sample larger than population
        states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).float().to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        return (states, actions, rewards, next_states)

    def init_n_step(self):
        self.n_step = deque(maxlen=self.rollout) #5

    def store_experience(self, experience):
        self.n_step.append(experience)

        # Abort if ROLLOUT steps haven't been taken in a new episode
        if len(self.n_step) < self.rollout : return

        # Unpacks and stores the SARS' tuple for each actor in the environment thus,
        # each timestep actually adds K_ACTORS memories to the buffer,
        #:20 memories each timestep.
        for actor in zip(*self.n_step):
            states, actions, rewards, next_states = zip(*actor)
            n_steps = self.rollout

            # Calculate n-step discounted reward
            rewards = np.fromiter((self.gamma**i * rewards[i] for i in range(n_steps)), float, count=n_steps)
            rewards = rewards.sum()

            # store the current state, current action, cumulative discounted reward from t -> t+n-1, and the next_state at t+n (S't+n)
            states = states[0].unsqueeze(0)
            actions = torch.from_numpy(actions[0]).unsqueeze(0).double()
            rewards = torch.tensor([rewards])
            next_states = next_states[-1].unsqueeze(0)
            self.add(states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
