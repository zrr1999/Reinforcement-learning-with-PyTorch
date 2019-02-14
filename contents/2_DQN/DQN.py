# -*- coding: utf-8 -*-
"""DQN"""
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from gym import make

from RL_brain import DeepQNetwork

np.set_printoptions(precision=2, suppress=True)

env = make("MountainCar-v0")
# env = env.unwrapped
print("Observation_space:{}\nAction_space:{}".format(env.observation_space,
                                                     env.action_space))
RL = DeepQNetwork(env.action_space.n,2, learning_rate=0.01,e_greedy_increment=0.001, double_q=True,prioritized=True,dueling=True)

for i in range(500):
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        reward=observation_[0]
        RL.store_transition(observation, action, reward, observation_)
        if RL.memory_counter > RL.memory_size:
            RL.learn()
        if done:
            break
        observation = observation_
env.close()
plt.plot(np.arange(len(RL.his)), RL.his)
plt.show()
