"""Sarsa"""
import numpy as np
from RL_brain import SarsaLambda
from environment import TestEnv

np.set_printoptions(precision=2, suppress=True)

env = TestEnv(10)
print("Observation_space{}\nAction_space{}".
      format(env.observation_space, env.action_space))

RL = SarsaLambda(range(env.action_space.n), reward_decay=1)

for i in range(50):
    s = env.reset()
    while True:
        # env.render()
        u = RL.choose_action(s)
        s_, r_, done, _ = env.step(u)
        RL.learn(s, u, r_, s_)
        if done:
            print('Completed')
            break
        s = s_
print(RL.q_table)
env.close()
