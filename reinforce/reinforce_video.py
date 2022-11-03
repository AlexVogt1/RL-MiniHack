'''
Produces a gif of the agent acting in the enviroment
'''

import numpy as np
import matplotlib.pyplot as plt
import gym

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from nle import nethack
import minihack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REINFORCE Policy NN model
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
            return r
        except:
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

# Utility to save gif to $path

def save_gif(gif,path):
    path=path+'.gif'
    gif[0].save(path, save_all=True,optimize=False, append_images=gif[1:], loop=0)

def frames_to_gif(frames):
    gif = []
    for image in frames:
        gif.append(Image.fromarray(image, "RGB"))
    return gif

pol = torch.load('./model/model_25k')

env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=["chars", "glyphs", "pixel"], actions=actions)


# Generating the episode
done = False
i = 0
r = env.reset()
state = np.array([r['chars'].flatten(), r['glyphs'].flatten()]).flatten()
action, pi = pol(torch.from_numpy(state))
e = []
pis = []
reward = 0
frames = []
while not done and i < 1000:
    i += 1
    # Take action
    obs = env.step(action)
    # Get frame after action performed
    frames.append(obs[0]['pixel'])
    state_prime = np.array([obs[0]['chars'].flatten(), obs[0]['glyphs'].flatten()]).flatten()
    reward = obs[1]
    done = obs[2]
    action_prime, pi_prime = pol(torch.from_numpy(state_prime))

    state = state_prime
    action = action_prime

    pi = pi_prime

# Generate gif from observed frames
save_gif(frames_to_gif(frames), './gifs/z')
