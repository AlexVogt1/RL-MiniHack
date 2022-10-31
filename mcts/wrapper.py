import gym
import minihack
from minihack import reward_manager
import numpy as np
from minihack import RewardManager
from gym import spaces
from nle import nethack
from numpy.lib.function_base import select

class BasicWrapper(gym.Wrapper):
    def __init__(self, env, seed=0, maxSteps = 10000):
        super().__init__(env)
        self.env = env
        self.seedCustom = seed
        self.maxSteps = maxSteps
        self.augmentedReward = 0
        self.seenStates = None
        self.currStep = 0
        self.repeatNum = 0
        
    def fullReset(self):
        self.seenStates = None
        self.currStep = 0
        self.repeatNum = 0


    def reset(self):
        self.env.seed(self.seedCustom)
        state = self.env.reset()
        state = self.selectObs(state)
        return state

    def lookThroughDict(self, temp):
        solidStone = [73,116,39,115,32,115,111,108,105,100,32,115,116,111,110,101,46]
        whatStrange = [87,104 ,97, 116, 32, 97, 32, 115, 116, 114, 97, 110, 103, 101, 32, 100, 105, 114, 101, 99, 116 ,105 ,111 ,110 ,33 ,32 ,32 ,78 ,101 ,118  ]
        nothingZap = [89, 111, 117, 32 ,100 ,111 ,110 ,39 ,116, 32, 104, 97, 118, 101, 32, 97 ,110, 121, 116, 104, 105, 110, 103, 32] 
        whatDir = [73, 110, 32 ,119 ,104 ,97 ,116 ,32 ,100 ,105 ,114 ,101 ,99] 
        noDoor = [89, 111, 117, 32, 115, 101, 101, 32 ,110 ,111 ,32 ,100 ,111 ,111 ,114 ,32 ,116] 
        cnt = 0 
        punishmentReward = 0.2
        isMessage = False
        solid = True
        for char in solidStone:
            if temp[cnt] != char:
                solid = False
                break
                
                # break
            cnt += 1
        if solid:
            self.augmentedReward -= punishmentReward
            return
        strange = True
        for char in whatStrange:
            if temp[cnt] != char:
                strange = False
                break
            cnt += 1
        if strange:
            self.augmentedReward -= punishmentReward
            return

        zap = True
        for char in nothingZap:
            if temp[cnt] != char:
                strange = False
                break
            cnt += 1
        if zap:
            self.augmentedReward -= punishmentReward
            return
        whatDirec = True
        for char in whatDir:
            if temp[cnt] != char:
                whatDirec = False
                break
            cnt += 1
        if whatDirec:
            self.augmentedReward -= punishmentReward
            return
        noDoorHere = True
        for char in noDoor:
            if temp[cnt] != char:
                noDoorHere = False
                break
            cnt += 1
        if noDoorHere:
            self.augmentedReward -= punishmentReward
            return
        else:
            self.augmentedReward += 0.01
            return  

    def selectObs(self, obs, desired=["chars","message","inv_letters"]):
        tempState = np.array(())
        for desire in desired:
            temp = obs[desire]
            if desire == "message":
                self.lookThroughDict(temp)
            temp = np.array(temp)
            temp = temp.astype(int)
            temp = temp.flatten()
            tempState = np.append(tempState, temp)
        return tempState

    def step(self, action, maxLength=10000):
        self.currStep += 1
        self.augmentedReward = 0
        next_state, reward, done, info = self.env.step(action)

        next_state = self.selectObs(next_state)
        self.lookThroughDict(next_state)

        reward += self.augmentedReward

        return next_state, reward, done, info

def createActionSpace():
    moves = tuple(nethack.CompassDirection)
    navActions = moves + (
        # nethack.Command.APPLY,
        # nethack.Command.AUTOPICKUP,
        # nethack.Command.CAST,
        # nethack.Command.CLOSE,
        # nethack.Command.DROP,
        # nethack.Command.EAT,
        # nethack.Command.ESC,
        # nethack.Command.FIRE,
        # nethack.Command.FIGHT,
        # nethack.Command.INVOKE,
        # nethack.Command.KICK,
        # nethack.Command.LOOK, 
        # nethack.Command.LOOT,
        # nethack.Command.OPEN,
        # nethack.Command.PRAY,
        # nethack.Command.PUTON,
        # nethack.Command.QUAFF,
        # nethack.Command.READ,
        # nethack.Command.REMOVE,
        # nethack.Command.RIDE,
        # nethack.Command.RUB,
        # nethack.Command.SEARCH,
        # nethack.Command.TAKEOFF,
        # nethack.Command.TAKEOFFALL,
        # nethack.Command.THROW,
        # nethack.Command.TIP,
        # nethack.Command.WEAR,
        # nethack.Command.WIELD,
        # nethack.Command.ZAP,
    )
    return navActions

def customGym(maxLength=10000, seed=0):
    reward_gen = RewardManager()
    reward_gen.add_eat_event("apple", reward=1, repeatable=False)
    reward_gen.add_wield_event("wand", reward=20, repeatable=False) 
    reward_gen.add_location_event("sink", reward=-1, terminal_required=False)
    reward_gen.add_kill_event("minotaur",reward=40, repeatable=False)
    env = gym.make(
        "MiniHack-Quest-Hard-v0",
        observation_keys=("chars", "inv_letters", "message"),
            reward_manager = reward_gen,
            actions=createActionSpace()
    )
    env._max_episode_steps = maxLength
    env.seed(seed)
    return env