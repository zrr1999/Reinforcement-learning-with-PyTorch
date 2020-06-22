import torch
import numpy as np


class Coach:
    def __init__(self, agent, optimizer):
        self.agent = agent
        self.optimizer = optimizer

    def train(self, obs, act, batch_rews, mode=None):
        weights = []
        for ep_rews in batch_rews:
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            if mode == "reward_to_go":
                rtgs = np.zeros_like(ep_rews)
                for i in reversed(range(ep_len)):
                    rtgs[i] = ep_rews[i] + (rtgs[i + 1] if i + 1 < ep_len else 0)
                weights += list(rtgs)
            else:
                if mode is not None:
                    print("Invalid given mode")
                weights += [ep_ret] * ep_len
        weights = torch.as_tensor(weights, dtype=torch.float32)
        # take a single policy gradient update step
        log_p = self.agent.get_policy(obs).log_prob(act)
        batch_loss = -(log_p * weights).mean()
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss
