# -*- coding: utf-8 -*-
import torch
from unityagents import UnityEnvironment
from utils import print_bracketing
import platform

class Environment:
    def __init__(self, args, id=33):

        self.train = not args.eval
        '''
        print("LOADING ON SYSTEM: {}".format(platform.system()))
        print_bracketing(do_lower=False)
        if platform.system() == 'Linux':
            unity_filename = "Reacher_Linux_NoVis/Reacher.x86_64"
        elif platform.system() == 'Darwin':
            print("MacOS not supported in this code!")
        else:
            unity_filename = 'Reacher_Windows_x86_64/Reacher.exe'
        '''

        self.env = UnityEnvironment(file_name='/Users/parksoy/Desktop/deep-reinforcement-learning/p2_continuous-control/Reacher_multi.app', worker_id=id, no_graphics=True) #args.nographics
        print_bracketing(do_upper=False)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Environment resets itself when the class is instantiated
        self.reset()

        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.states.shape[1]
        self.agent_count = len(self.env_info.agents)

    def reset(self):
        self.env_info = self.env.reset(train_mode = self.train)[self.brain_name]

    def close(self):
        self.env.close()

    def step(self, actions):
        self.env_info = self.env.step(actions)[self.brain_name]
        next_states = self.states
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return next_states, rewards, dones

    @property
    def states(self):
        states = self.env_info.vector_observations #tensor
        return torch.from_numpy(states).float()
