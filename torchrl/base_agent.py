from utils import BatchLists
import torch


class AgentBase:
    def __init__(self, env):
        self.env = env
        self.bl = BatchLists()
        self.obs = None

    def get_policy(self, obs):
        pass

    def get_action(self, obs):
        pass

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self, batch_lists):
        if self.obs is None:
            print("You should reset the sampler by using sampler.reset()")
        act = self.get_action(torch.as_tensor(self.obs, dtype=torch.float32))
        obs, rew, done, _ = self.env.step(act)
        self.obs = obs
        batch_lists.append(batch_act=act)
        return obs, rew, done, _

    def sample(self, batch_size=5000, render=False):
        bl = self.bl.reset()
        obs = self.reset()  # first obs comes from starting distribution
        ep_rews = []

        finished_rendering_this_epoch = False  # render first episode of each epoch
        # collect experience by acting in the environment with current policy
        while True:
            if (not finished_rendering_this_epoch) and render:
                self.env.render()

            bl.append(obs.copy())
            obs, rew, done, _ = self.step(bl)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                bl.append(batch_ret=ep_ret, batch_len=ep_len, ep_rews=ep_rews)

                # reset episode-specific variables
                obs, done, ep_rews = self.env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(bl.batch_obs) > batch_size:
                    break
        return bl()

    def __call__(self, obs):
        return self.get_action(obs)
