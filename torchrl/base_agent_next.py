from utils import BatchLists
import torch


class AgentBase(object):
    def __call__(self, obs):
        return self.policy(obs)

    def reset(self, state):
        self.state = state

    def policy(self, state):
        raise NotImplementedError

    def learn(self, knowledge):
        raise NotImplementedError


class AgentDemo(AgentBase):
    def __init__(self):
        self.strategy = 1

    def policy(self, state):
        return self.strategy * state

    def learn(self, knowledge):
        self.strategy += knowledge


class Coach(object):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train(self):
        self.agent
