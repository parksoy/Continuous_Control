%reset
import os
import numpy as np
from agent import D4PG_Agent
from environment import Environment
from data_handling import Logger, Saver #, gather_args
from config_settings import Args

%load_ext autoreload
%autoreload 2

def train(agent, args, env, saver):
    logger = Logger(agent, args, saver.save_dir)
    agent.initialize_memory(args.pretrain, env)# Pre-fill the Replay Buffer
    for episode in range(1, args.num_episodes+1):#Begin training loop
        print("HAHAHAHHAHA episode#=", episode)
        env.reset()# Begin each episode with a clean environment
        states = env.states# Get initial state
        for t in range(args.max_steps):# Gather experience until done or max_steps is reached
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states)
            states = next_states
            logger.log(rewards, agent)
            if np.any(dones): break
        saver.save_checkpoint(agent, args.save_every)
        agent.new_episode()
        logger.step(episode, agent)
    #env.close()
    saver.save_final(agent)
    #logger.graph()

def eval(agent, args, env):
    logger = Logger(agent, args)
    for episode in range(1, args.num_episodes+1):#Begin evaluation loop
        env.reset()# Begin each episode with a clean environment
        states = env.states# Get initial state
        for t in range(args.max_steps):# Gather experience until done or max_steps is reached
            actions = agent.act(states, eval=True)
            next_states, rewards, dones = env.step(actions)
            states = next_states
            logger.log(rewards, agent)
            if np.any(dones): break
        agent.new_episode()
        logger.step(episode)
    #env.close()

    def _get_files(save_dir):
    file_list = []
    for root, _, files in os.walk(save_dir):
        for file in files:
            if file.endswith(".agent"):
                file_list.append(os.path.join(root, file))
    return sorted(file_list, key=lambda x: os.path.getmtime(x))

def _get_filepath(files):
    load_file_prompt = " (LATEST)\n\nPlease choose a saved Agent training file (or: q/quit): "
    user_quit_message = "User quit process before loading a file."
    message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
    message = '\n'.join(message).replace('\\', '/')
    message = message + load_file_prompt
    save_file = input(message)
    if save_file.lower() in ("q", "quit"):
        raise KeyboardInterrupt(user_quit_message)
    try:
        file_index = len(files) - int(save_file)
        assert file_index >= 0
        return files[file_index]
    except:
        print_bracketing('Input "{}" is INVALID...'.format(save_file))
        return _get_filepath(files)

def _get_agent_file(args):
    invalid_filename = "Requested filename is invalid."
    no_files_found = "Could not find any files in: {}".format(args.save_dir)
    if args.resume or args.eval:
        if args.filename is not None:
            assert os.path.isfile(args.filename), invalid_filename
            return args.filename
        files = _get_files(args.save_dir)
        assert len(files) > 0, no_files_found
        if args.latest:
            return files[-1]
        else:
            return _get_filepath(files)
    else:
        return False

def _get_files(save_dir):
    file_list = []
    for root, _, files in os.walk(save_dir):
        for file in files:
            if file.endswith(".agent"):
                file_list.append(os.path.join(root, file))
    return sorted(file_list, key=lambda x: os.path.getmtime(x))

#if __name__ == "__main__":
args = Args() #gather_args()
env = Environment(args)
agent = D4PG_Agent(env, args)

args.load_file = _get_agent_file(args)

saver = Saver(agent.framework, agent, args.save_dir, args.load_file)

args.eval=False
print(args.eval)

if args.eval: eval(agent, args, env)
else: train(agent, args, env, saver)
