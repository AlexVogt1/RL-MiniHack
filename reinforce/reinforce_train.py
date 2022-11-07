import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from nle import nethack
import minihack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REINFORCE Policy Model
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


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):

    '''
    Runs REINFORCE and optimises policy_model using Adam optimiser
    '''

    # Seed randomness for reproducability
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    optimizer = torch.optim.Adam(policy_model.parameters(),lr=learning_rate, weight_decay=1e-6)
    rewards = []

    # Main training loop
    for ep in tqdm(range(number_episodes)):
        done = False
        i = 0
        r = env.reset()
        state = np.array([r['chars'].flatten(), r['glyphs'].flatten()]).flatten()
        action, pi = policy_model(torch.from_numpy(state))
        e = []
        pis = []

        # Generating episode observations/rewards
        while not done and i < max_episode_length:
            i += 1
            # Take action
            obs = env.step(action)

            # Extracting state
            state_prime = np.array([obs[0]['chars'].flatten(), obs[0]['glyphs'].flatten()]).flatten()

            # Extracting received reward
            reward = obs[1]

            # Is the episode done?
            done = obs[2]

            # Getting new action
            action_prime, pi_prime = policy_model(torch.from_numpy(state_prime))

            # Keeping track of results
            e.append([state, action, reward])
            pis.append(pi)
            state = state_prime
            action = action_prime
            pi = pi_prime
            if verbose: 
                env.render()
                print(f'action: {action}')

        # Keeping track of total rewards
        rewards.append(np.sum(np.array(e)[:, 2]))

        # Computing returns
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

        # Optimising policy
        policy_model.zero_grad()
        pg.sum().backward()
        optimizer.step()

        if verbose: print(f'Reward for episode {ep} : {rewards[-1]}')

        
    return policy_model, rewards

# Actions the agent can perform in the enviroment
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

# Creating enviroment and policy
env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=["glyphs","chars"], actions=actions)
pol = SimplePolicy(s_size = (21*79)*2,h_size = 256, a_size=len(actions))

env.reset()

# Policy initialisation step to avoid NaNs in policy probabilities
pol, scores = reinforce(
    env, pol, 42, 1, 50, 1000, 0.3, verbose=False
)

# Main training loop
pol, scores = reinforce(
    env, pol, 42, 1e-1, 25000, 1000, 0.99, verbose=False
)

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]    
    return ret / n

# Plotting rewards per episode
plt.title("Rewards during training")
plt.legend(['Rolling average of size 20'])
plt.plot(moving_average(scores, 20), 'g')
plt.savefig('./plots/plot_training')

# Saving model
torch.save(pol, './model/model_25k')

# Saving rewards
np.savetxt('reinforce_rewards.csv', np.array(scores), delimiter=',')