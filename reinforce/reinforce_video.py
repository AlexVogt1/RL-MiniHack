import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
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

pol = torch.load('./model_post/model_25k')

env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=["chars", "glyphs"], actions=actions, savedir='/home/matthew/Documents/Varsity/RL/RL-MiniHack/reinforce/video/')

NUM_EP = 3500
RUNS = 5
r_total = [None] * NUM_EP

done = False
i = 0
r = env.reset()
state = np.array([r['chars'].flatten(), r['glyphs'].flatten()]).flatten()
action, pi = pol(torch.from_numpy(state))
e = []
pis = []
reward = 0
for _ in range(5):
    while not done and i < 10000:
        i += 1
        obs = env.step(action)
        state_prime = np.array([obs[0]['chars'].flatten(), obs[0]['glyphs'].flatten()]).flatten()
        reward = obs[1]
        done = obs[2]
        action_prime, pi_prime = pol(torch.from_numpy(state_prime))

        state = state_prime
        action = action_prime

        pi = pi_prime

