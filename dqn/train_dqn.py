import numpy as np
from nle import nethack
import minihack
import random
from agent import DQNAgent
from replay_buffer import ReplayBuffer
from utils import *
import gym
from minihack import RewardManager

from gym import spaces

def dqn(env, lvl, seed, max_episodes, max_episode_length,exp_hyper_params, verbose=True):
    """
    Method to train DQN model.
    
    Input:
    env: The environment to be used during training
    seed: The random seed for any random operations performed 
    learning_rate: The learning rate uesd for the Adam optimizer when training the model 
    number_episodes: Number of episodes to train for 
    max_episode_length: The maximum number of steps to take in an episode before terminating
    gamma: The discount factor used when calculating the discounted rewards of an episode
    verbose: Print episode reward after each episode
    
    Returns:
    scores: The cumulative reward achieved by the agent for each episode during traiing
    """

    hyper_params = exp_hyper_params
    
    seed =np.random.seed(seed)
    env.seed(seed)
    
    # Create DQN agent
    replay_buffer = ReplayBuffer(hyper_params['replay-buffer-size'])
    agent = DQNAgent(
        env.observation_space, 
        env.action_space,
        train=True,
        replay_buffer=replay_buffer,
        use_double_dqn=hyper_params['use-double-dqn'],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['gamma'],
    )
    
    # define variables to track agent metrics
    total_reward = 0
    scores = []
    mean_rewards = []
    frames =[]
    

    # Reset gym env before training
    state = format_state(env.reset())
    eps_timesteps = hyper_params['eps-fraction'] * float(hyper_params['num-steps'])

    # Train for set number of steps
    for t in range(hyper_params['num-steps']):
        # determine exploration probability
        fract = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fract * (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()
        # Decide to explore and choose random action or use model to act
        if sample < eps_threshold:
            action = np.random.choice(agent.action_space.n)
        else:
            action = agent.act(state)
        # Take step in environment
        (next_state, reward, done, _) = env.step(action)

        #record steps every save-freq
        if len(scores) % hyper_params['save-freq'] == 0:
            frames.append(next_state["pixel"])
        if len(scores) % hyper_params['save-freq'] == 0 and done:
            gif = frames_to_gif(frames)
            save_gif(gif,f'./video/{lvl}{len(scores)}')
            frames =[]

        next_state = format_state(next_state)
        replay_buffer.add(state, action, reward, next_state, float(done))
        total_reward += reward
        state = next_state

        if done:
            scores.append(total_reward)
            print(f"episode reward: {total_reward}")
            np.random.seed(seed)
            env.seed(seed)
            state = format_state(env.reset())
            # display_screen(env)
            total_reward = 0

        if t > hyper_params['learning-starts'] and t % hyper_params['learning-freq'] == 0:
            ans = agent.optimise_td_loss()

        if t > hyper_params['learning-starts'] and t % hyper_params['target-update-freq'] == 0:
            agent.update_target_network()

        num_episodes = len(scores)
        if done and hyper_params['print-freq'] is not None and len(scores) % hyper_params['print-freq'] == 0:
            mean_100ep_reward = round(np.mean(scores[-101:-1]), 1)
            mean_rewards.append(mean_100ep_reward)
            print('********************************************************')
            print('steps: {}'.format(t))
            print('episodes: {}'.format(num_episodes))
            print('mean 100 episode reward: {}'.format(mean_100ep_reward))
            print('% time spent exploring: {}'.format(eps_threshold))
            print('********************************************************')
            np.savetxt('during_train_rewards_'+ f'_{lvl}_{i}' +'.csv', scores, delimiter=',', fmt='%1.10f')

        
        #saving the model
        if ( done and hyper_params["save-freq"] is not None and len(scores) % hyper_params["save-freq"] == 0):
            torch.save(agent.online_network, './models/model'+ f'_{lvl}_'+str(len(scores)) +'.pt')

        if num_episodes >=max_episodes:
            return scores

    return scores


def run_dqn(env,lvl_name,num_eps,max_episode_steps,exp_hyper_params,iterations):
    """Trains DQN model for a number of episodes on a given environment"""
    seeds = np.random.randint(42, size=iterations)
    scores_arr = [] 
    
    for seed in seeds:
        print(seed)
        # Train the DQN Model 
        scores = dqn(env=env, seed=seed, lvl = lvl_name,max_episodes=num_eps, max_episode_length=max_episode_steps, exp_hyper_params =exp_hyper_params, verbose=True)
        # Store rewards for this iteration 
        scores_arr.append(scores)
        
    return scores_arr

if __name__ == "__main__":

    #uncomment the relavent experiments you wish to run
    #comment out the experiments you dont wish to run

    #train Room 5x5
    # room_hyper_params = {
    #     'replay-buffer-size': int(1e6),
    #     'learning-rate': 0.02,
    #     'gamma': 0.99,  # discount factor
    #     'num-steps': int(2e5),  # Steps to run for, max episodes should be hit before this
    #     'batch-size': 32,  
    #     'learning-starts': 1000,  # set learning to start after 1000 steps of exploration
    #     'learning-freq': 1,  # Optimize after each step
    #     'use-double-dqn': True,
    #     'target-update-freq': 1000, # number of iterations between every target network update
    #     'eps-start': 1.0,  # e-greedy start threshold 
    #     'eps-end': 0.1,  # e-greedy end threshold 
    #     'eps-fraction': 0.4,  # Percentage of the time that epsilon is annealed
    #     'print-freq': 10,
    #     'save-freq':100
    # }
    # # Create the environment with the observations keys required as input to the DQN
    # MOVE_ACTIONS = tuple(nethack.CompassDirection)
    # # env = gym.wrappers.RecordVideo(env, './video/', episode_trigger = lambda x: x == 2)
    # # env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True
    # env = gym.make("MiniHack-Room-5x5-v0", observation_keys=["glyphs","pixel","message"], actions=MOVE_ACTIONS,max_episode_steps =1000)
    # # env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)
    # # Reset the environment and display the screen of the starting state 
    # # env = nv = RenderRGB(env, 'pixel')
    # # env = gym.wrappers.record_video.RecordVideo(env, './video/', episode_trigger = lambda x: x == 2)
    
    # # env = gym.wrappers.Monitor(env, 'recording',  video_callable=lambda episode_id: episode_id % 50 == 0, force=True)
    # # env.render()
    # runs = 1
    
    # for i in range(runs):
    #     room_5x5_scores = run_dqn(env,num_eps=1000,max_episode_steps=1000,exp_hyper_params=room_hyper_params,iterations=1)
    #     np.savetxt('rewards_'+ f'Room5x5_{i}' +'.csv', room_5x5_scores, delimiter=',', fmt='%1.10f')


    # # Skill Acquisition Tasks

    # #apple eat
    # apple_hyper_params = {
    #     'replay-buffer-size': int(1e6),
    #     'learning-rate': 0.02,
    #     'gamma': 0.99,  # discount factor
    #     'num-steps': int(2e5),  # Steps to run for, max episodes should be hit before this
    #     'batch-size': 32,  
    #     'learning-starts': 1000,  # set learning to start after 1000 steps of exploration
    #     'learning-freq': 1,  # Optimize after each step
    #     'use-double-dqn': True,
    #     'target-update-freq': 1000, # number of iterations between every target network update
    #     'eps-start': 1.0,  # e-greedy start threshold 
    #     'eps-end': 0.1,  # e-greedy end threshold 
    #     'eps-fraction': 0.4,  # Percentage of the time that epsilon is annealed
    #     'print-freq': 10,
    # }
    # MOVE_ACTIONS = tuple(nethack.CompassDirection)
    # NAVIGATE_ACTIONS = MOVE_ACTIONS+(nethack.Command.EAT,)
    # env = gym.make("MiniHack-Eat-Fixed-v0", observation_keys=["glyphs","pixel","message"], actions=NAVIGATE_ACTIONS, max_episode_steps =1000)
    # runs = 3
    # for i in range(runs):
    #     apple_scores = run_dqn(env,number_episodes=1000,max_episode_length=1000,hyper_params=apple_hyper_params,iterations=1)
    #     np.savetxt('rewards_'+ f'apple_{i}' +'.csv', room_5x5_scores, delimiter=',', fmt='%1.10f')

    # QUEST HARD
    # navigate_hyper_params = {
    #     'replay-buffer-size': int(5e3),
    #     'learning-rate': 0.01,
    #     'gamma': 0.99,  # discount factor
    #     'num-steps': int(2e6),  # Steps to run for, max episodes should be hit before this
    #     'batch-size': 32,  
    #     'learning-starts': 1000,  # set learning to start after 1000 steps of exploration
    #     'learning-freq': 1,  # Optimize after each step
    #     'use-double-dqn': True,
    #     'target-update-freq': 1000, # number of iterations between every target network update
    #     'eps-start': 1.0,  # e-greedy start threshold 
    #     'eps-end': 0.3,  # e-greedy end threshold 
    #     'eps-fraction': 0.1,  # Percentage of the time that epsilon is annealed
    #     'print-freq': 10,
    #     'save-freq':100,
    # }
    # MOVE_ACTIONS = tuple(nethack.CompassDirection)
    # NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    #     nethack.Command.PICKUP,
    #     nethack.Command.APPLY,
    #     nethack.Command.FIRE,
    #     # nethack.Command.RUSH,
    #     nethack.Command.ZAP, # use wand in 
    #     nethack.Command.PUTON,
    #     # nethack.Command.READ,
    #     nethack.Command.WEAR,
    #     nethack.Command.QUAFF,
    #     nethack.Command.PRAY, 
    #     nethack.Command.OPEN, #open doors in quest hard
    #     nethack.Command.DROP, # try to learn to drop rocks
    #     nethack.Command.EAT, # eat them apples to gain health
    #     nethack.Command.FIGHT, # gatta be able to fight monsters
    #     # nethack.Command.JUMP,
    #     # nethack.Comman.UP,
    #     nethack.MiscDirection.DOWN, # in some levels you are requred to go down staircases inorder to win
    # )
    # env = gym.make("MiniHack-Room-Random-15x15-v0", observation_keys=["glyphs","pixel","message"], actions=NAVIGATE_ACTIONS,max_episode_steps =1000)
    

    # runs = 1
    # for i in range(runs):
    #     quest_hard_scores = run_dqn(env,num_eps=500,max_episode_steps=1000,exp_hyper_params=navigate_hyper_params,iterations=1)
    #     np.savetxt('rewards_'+ f'Room_random_{i}' +'.csv', quest_hard_scores, delimiter=',', fmt='%1.10f')


    # navigate_hyper_params = {
    #     'replay-buffer-size': int(5e3),
    #     'learning-rate': 0.02,
    #     'gamma': 0.99,  # discount factor
    #     'num-steps': int(1e6),  # Steps to run for, max episodes should be hit before this
    #     'batch-size': 32,  
    #     'learning-starts': 5000,  # set learning to start after 1000 steps of exploration
    #     'learning-freq': 1,  # Optimize after each step
    #     'use-double-dqn': True,
    #     'target-update-freq': 5000, # number of iterations between every target network update
    #     'eps-start': 1.0,  # e-greedy start threshold 
    #     'eps-end': 0.1,  # e-greedy end threshold 
    #     'eps-fraction': 0.4,  # Percentage of the time that epsilon is annealed
    #     'print-freq': 10,
    #     'save-freq':100,
    # }
    # MOVE_ACTIONS = tuple(nethack.CompassDirection)
    # NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    # nethack.Command.PICKUP,
    # nethack.Command.APPLY,
    # nethack.Command.FIRE,
    # nethack.Command.RUSH,
    # nethack.Command.ZAP,
    # nethack.Command.PUTON,
    # nethack.Command.READ,
    # nethack.Command.WEAR,
    # nethack.Command.QUAFF,
    # nethack.Command.PRAY,
    # nethack.Command.OPEN,
    # nethack.Command.DROP,
    # nethack.Command.EAT,
    # nethack.Command.JUMP,
    # )
    # env = gym.make("MiniHack-MazeWalk-15x15-v0", observation_keys=["glyphs","pixel","message"], actions=MOVE_ACTIONS,max_episode_steps =1000,reward_manager=gen_rewards_quest_hard())
    

    # runs = 1
    # for i in range(runs):
    #     quest_hard_scores = run_dqn(env,num_eps=502,max_episode_steps=1000,exp_hyper_params=navigate_hyper_params,iterations=1)
    #     np.savetxt('rewards_'+ f'apple_{i}' +'.csv', quest_hard_scores, delimiter=',', fmt='%1.10f')
    
    level_name = "MiniHack-Room-Random-15x15-v0"

    navigate_hyper_params = {
        'replay-buffer-size': int(5e3),
        'learning-rate': 0.01,
        'gamma': 0.99,  # discount factor
        'num-steps': int(2e6),  # Steps to run for, max episodes should be hit before this
        'batch-size': 32,  
        'learning-starts': 1000,  # set learning to start after 1000 steps of exploration
        'learning-freq': 1,  # Optimize after each step
        'use-double-dqn': True,
        'target-update-freq': 1000, # number of iterations between every target network update
        'eps-start': 1.0,  # e-greedy start threshold 
        'eps-end': 0.3,  # e-greedy end threshold 
        'eps-fraction': 0.4,  # Percentage of the time that epsilon is annealed
        'print-freq': 10,
        'save-freq':100,
    }
    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.PICKUP,
        nethack.Command.APPLY,
        nethack.Command.FIRE,
        # nethack.Command.RUSH,
        nethack.Command.ZAP, # use wand in 
        nethack.Command.PUTON,
        # nethack.Command.READ,
        nethack.Command.WEAR,
        nethack.Command.QUAFF,
        nethack.Command.PRAY, 
        nethack.Command.OPEN, #open doors in quest hard
        nethack.Command.DROP, # try to learn to drop rocks
        nethack.Command.EAT, # eat them apples to gain health
        nethack.Command.FIGHT, # gatta be able to fight monsters
        # nethack.Command.JUMP,
        # nethack.Comman.UP,
        nethack.MiscDirection.DOWN, # in some levels you are requred to go down staircases inorder to win
    )
    env = gym.make(level_name, observation_keys=["glyphs","pixel","message"], actions=NAVIGATE_ACTIONS,max_episode_steps =1000)
    

    runs = 1
    for i in range(runs):
        level_scores = run_dqn(env,num_eps=500,max_episode_steps=1000,exp_hyper_params=navigate_hyper_params,iterations=1)
        np.savetxt('./Reward csv/rewards_'+ f'{level_name}_{i}' +'.csv', level_scores, delimiter=',', fmt='%1.10f')