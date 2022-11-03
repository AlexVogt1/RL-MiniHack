from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from collections import deque

from tqdm import tqdm

import gym
from gym import spaces
import cv2
import random
from nle import nethack
import minihack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimplePolicy(nn.Module):
    def __init__(self, s_size=1659, h_size=5, a_size=8):

        super(SimplePolicy, self).__init__()
        self.input = nn.Linear(s_size, h_size)
        self.hidden = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = torch.relu(self.input(x.float()))
        x = F.softmax(self.hidden(x), dim=0)
        act = self.action(x)
        return act, torch.log(x.squeeze(0))[act]

    def action(self, pr):
        try:
            r = np.random.choice(np.arange(len(pr.squeeze(0).detach().cpu().numpy())), p=pr.squeeze(0).detach().cpu().numpy())
            #print("Not Random" + "-"*15)
            return r
        except:
            #print("Random" + "-"*15)
            return np.random.choice(np.arange(len(pr)))

def gen_ep_reward(env, pol, max_steps):
    done = False
    i = 0
    _r = env.reset()
    state = np.array([_r['chars'].flatten(), _r['glyphs'].flatten()]).flatten()
    action, _ = pol(torch.from_numpy(state))

    reward = 0
    r = 0

    while not done and i < max_steps:
        i += 1
        obs = env.step(action)
        state_prime = np.array([obs[0]['chars'].flatten(), obs[0]['glyphs'].flatten()]).flatten()
        reward = obs[1]
        done = obs[2]
        action_prime, _ = pol(torch.from_numpy(state_prime))

        state = state_prime
        action = action_prime
        r += reward

    return r


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    optimizer = torch.optim.Adam(policy_model.parameters(),lr=learning_rate, weight_decay=1e-6)
    rewards = []
    rewards_combined = []

    for ep in tqdm(range(number_episodes)):
        done = False
        i = 0
        r = env.reset()
        state = np.array([r['chars'].flatten(), r['glyphs'].flatten()]).flatten()
        action, pi = policy_model(torch.from_numpy(state))
        e = []
        pis = []

        while not done and i < max_episode_length:
            i += 1
            obs = env.step(action)
            state_prime = np.array([obs[0]['chars'].flatten(), obs[0]['glyphs'].flatten()]).flatten()
            reward = obs[1]
            done = obs[2]
            action_prime, pi_prime = policy_model(torch.from_numpy(state_prime))
            e.append([state, action, reward])
            pis.append(pi)
            state = state_prime
            action = action_prime
            pi = pi_prime
            if verbose: 
                env.render()
                print(f'action: {action}')
        rewards.append(np.sum(np.array(e)[:, 2]))

        Gs = []
        for t in range(len(e)):
            G = 0
            for k in range(t + 1, len(e)):
                G += (gamma**(k-t-1)) * e[k][2]
            Gs.append(G)

        Gs = torch.tensor(Gs, requires_grad=True)
        Gs = (Gs - torch.mean(Gs))/torch.std(Gs)
        probs = torch.stack(pis)
        pg = -1 * probs * Gs

        policy_model.zero_grad()
        pg.sum().backward()
        optimizer.step()

        if verbose: print(f'Reward for episode {ep} : {rewards[-1]}')

        # Getting reward of each subgoal
        rew_temp = []
        if ep % 10 == 0:
            for e in ['MiniHack-Room-Random-15x15-v0', 'MiniHack-MazeWalk-15x15-v0', 'MiniHack-LavaCross-Levitate-Potion-Inv-v0', 'MiniHack-Quest-Hard-v0']:
                env_temp = gym.make(e, observation_keys=["glyphs","chars"], actions=actions)
                # Generating total reward
                rew_temp.append(gen_ep_reward(env_temp, policy_model, 1000))
            rewards_combined.append(tuple(rew_temp))
    return policy_model, rewards, rewards_combined


actions = tuple(nethack.CompassDirection) + (
    nethack.Command.PICKUP,
    nethack.Command.APPLY,
    nethack.Command.FIRE,
    nethack.Command.RUSH,
    nethack.Command.ZAP,
    nethack.Command.PUTON,
    nethack.Command.READ,
    nethack.Command.WEAR,
    nethack.Command.QUAFF,
    nethack.Command.PRAY,
    nethack.Command.OPEN
)

env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=["glyphs","chars"], actions=actions)


r = env.reset()

pol = SimplePolicy(s_size = (21*79)*2,h_size = 256, a_size=len(actions))

env.reset()
env.render()
pol, _, _ = reinforce(
    env, pol, 42, 1, 50, 1000, 0.3, verbose=False
)

pol, scores, scores_combined = reinforce(
    env, pol, 42, 1e-1, 1000, 1000, 0.99, verbose=False
)

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]    
    return ret / n

scores_room = [i[0] for i in scores_combined]
scores_maze = [i[1] for i in scores_combined]
scores_lava = [i[2] for i in scores_combined]
scores_quest = [i[3] for i in scores_combined]

plt.plot(scores_room)
plt.plot(scores_lava)
plt.plot(scores_maze)
plt.plot(scores_quest)

plt.legend(['Room-Random-15x15', 'MazeWalk-15x15-v0', 'LavaCross-Levitate-Potion-Inv', 'Quest-Hard'])
plt.title("Rewards per episode of training")
plt.savefig('./plots/combined_rewards_traning.png')