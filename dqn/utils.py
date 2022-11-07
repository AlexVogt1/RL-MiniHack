import numpy as np
import cv2
import random
from nle import nethack
import minihack
from PIL import Image
import os
from IPython.display import clear_output
import torch
from minihack import RewardManager
import gym
import matplotlib.pyplot as plt

#Format the state into form that the NN can accept
def format_state(state):
    glyphs = state["glyphs"]
    # Normalize
    glyphs = glyphs/glyphs.max()
    glyphs = glyphs.reshape((1,1,21,79))
    return torch.from_numpy(glyphs).squeeze(0)


#converting array of step images to gif
def frames_to_gif(frames):
    gif = []
    for image in frames:
        gif.append(Image.fromarray(image, "RGB"))
    return gif

#saving the gif
def save_gif(gif,path):
    '''
    Args:
        gif: a list of image objects
        path: the path to save the gif to
    '''
    path=path+'.gif'
    gif[0].save(path, save_all=True,optimize=False, append_images=gif[1:], loop=0)
    print("Saved Video")

#calculating reward based on visited states
def exploration_reward(env, prev_obs, action, next_obs):
    if (prev_obs[0] == 2359).sum() > (next_obs[0] == 2359).sum():
        return 0.5
    return 0

def gen_rewards_nav():
    reward_maker = RewardManager()
    reward_maker.add_eat_event("apple", reward=1,terminal_required=False) #and apple a day keeps the doctor away

    reward_maker.add_custom_reward_fn(exploration_reward)
    return reward_maker

#rewards based cutome rewards designed for dquest hard
def gen_rewards_quest_hard():

    reward_maker = RewardManager()
    reward_maker.add_eat_event("apple", reward=1,terminal_required=False)#and apple a day keeps the doctor away

    # Custom Rewards for long corridors at top and bottom 
    reward_maker.add_coordinate_event((3,27), reward = -1, terminal_required = False)
    reward_maker.add_coordinate_event((3,28), reward = -1, terminal_required = False)
    reward_maker.add_coordinate_event((3,29), reward = -1, terminal_required = False)

    reward_maker.add_coordinate_event((19,27), reward = -1, terminal_required = False)
    reward_maker.add_coordinate_event((19,28), reward = -1, terminal_required = False)
    reward_maker.add_coordinate_event((19,29), reward = -1, terminal_required = False)

    #reward for reaching the end of the maze
    reward_maker.add_coordinate_event((11,27), reward = 100, terminal_required = False)

    reward_maker.add_custom_reward_fn(exploration_reward)

