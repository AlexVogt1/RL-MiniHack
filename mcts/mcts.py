from wrapper import BasicWrapper, customGym
import gym
import minihack
from minihack import reward_manager
import numpy as np
from minihack import RewardManager
from collections import defaultdict
import random
import csv

dummyEnv = BasicWrapper(customGym()) 
actionSpaceSize = dummyEnv.action_space.n # action space
actionsTaken = []

class State(): 
    def __init__(self, state, reward=0,done=False,info=None):
        # print('init state')
        self.state_s = state
        self.reward = reward
        self.done = done
        self.info = info
        if done:
            self.legal_actions = []
        else:
            self.legal_actions = list(np.arange(actionSpaceSize))

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):

        self.state = state #observation,reward,done,info,actions
        self.parent = parent #parent node in tree
        self.parent_action = parent_action # action parent took to get here
        self.children = [] # children of this node
        self._number_of_visits = 1
        self._results = defaultdict(int) 
        self._untried_actions = None
        self._untried_actions = self.untried_actions()

        return

    #checks which actions have not been tried from this node
    def untried_actions(self):
        # print('untried_actions' , self.state)
        self._untried_actions = self.state.legal_actions
        return self._untried_actions

    def q(self):

        total = 0
        for r in self._results:
            total += r*self._results[r]

        return total

    def n(self):
        return self._number_of_visits

    # expand from node to next node
    def expand(self, env):
        action = self._untried_actions.pop()
        # print('ex action' , action)

        obs, reward, done, info = env.step(action)

        next_state = State(obs,reward,done,info)
        child_node = MonteCarloTreeSearchNode(state = next_state, parent=self, parent_action=action)
        self.children.append(child_node)

        return child_node , env

    def is_terminal_node(self):
        return self.state.done

    #simulation until the end of a game is reached
    def rollout(self , env):
        current_rollout_state = self.state
        
        while not current_rollout_state.done:
            
            possible_moves = current_rollout_state.legal_actions
            
            action = self.rollout_policy(possible_moves)
            obs, reward, done, info = env.step(action)
            current_rollout_state = State(obs,reward,done,info)

        return current_rollout_state.reward


    #backpropagates from leaf node to root
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    # UCT to find best child node 
    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        child = np.random.choice(np.flatnonzero(choices_weights == max(choices_weights)))
        return self.children[child]

    # default policy for simulation
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    
    def _tree_policy(self,env):
        current_node = self
        while not current_node.is_terminal_node():

            if len(current_node.children)==0:
                return current_node.expand(env)
            elif random.uniform(0,1)<.5:
                current_node = current_node.best_child()
            else:
                if not current_node.is_fully_expanded():
                    return current_node.expand(env)
                else:
                    current_node = current_node.best_child()
        return current_node , env
    
    
    # called to find the best next action 
    # goes throught the 4 processes some number of times.
    def best_action(self,actions, simulation_no=7):

        for i in range(simulation_no):
            print('\t\tSimulation {0}'.format(i), end="\r")

            env2 = BasicWrapper(customGym())
            initial_state2 = env2.reset()

            # to get environment to the same step as the real game the same actions must be performed
            for act in actions:
                obs, r, done, info = env2.step(act)

            # selection and expansion
            v = self._tree_policy(env2)
            # simulation
            reward = v[0].rollout(v[1])
            # bckpropagation
            v[0].backpropagate(reward)


        return self.best_child(c_param=0.1)




def main():

    csv_save = open('minihack_rewards.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(csv_save)

    # number of steps allowed in the game since it never ends
    iters = 1000

    for mpx in range(1):
        
        # initialize environment + actions and reward trackers
        prev_actions = []
        rewards = []
        actionsTaken = []
        
        env = BasicWrapper(customGym())
        rewardArr = []
        
        initial_state = env.reset()
        initial_state = State(state = initial_state)
        #env.render()

        root = MonteCarloTreeSearchNode(state = initial_state)
        
        # call MCTS function to select best action
        selected_node = root.best_action(prev_actions)
        # records the action taken
        prev_actions.append(selected_node.parent_action)

        # uses action to step in the environment
        next_state, reward, done, info = env.step(selected_node.parent_action)
        
        # records reward recieved
        rewards.append(reward)
        next_state = State(next_state,reward,done,info)

        #env.render()
        cnt = 0
        
        # loops through the rest of the game repeatedly calling the 
        # best_action function and making the selected move in the environment
        for it in range(iters):
            if done:
                break;
            cnt += 1
            print("Current Game {0}, current step {1}".format(mpx, cnt))
            selected_node = MonteCarloTreeSearchNode(state = next_state)
            selected_node = selected_node.best_action(prev_actions)


            prev_actions.append(selected_node.parent_action)

            next_state, reward, done, info = env.step(selected_node.parent_action)
            rewards.append(reward)
            writer.writerow([reward])

            print(f"Iter: {it}\tReward: {reward:.2f}")
            next_state = State(next_state,reward,done,info)
            #print(next_state.state_s)
            #env.render()
        rewardArr = np.array(rewards)
        # saves the rewards received during game
        np.savetxt("rewardArr-{0}".format(mpx), rewardArr)

main()