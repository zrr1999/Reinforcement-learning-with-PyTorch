import torch


class BatchLists:
    def __init__(self):
        self.batch_obs = []  # for observations
        self.batch_acts = []  # for actions
        self.batch_rets = []  # for measuring episode returns
        self.batch_lens = []  # for measuring episode lengths
        self.batch_rews = []  # list for rewards accrued throughout ep

    def reset(self):
        self.__init__()
        return self

    def append(self, batch_ob=None, batch_act=None, batch_ret=None, batch_len=None, ep_rews=None):
        if batch_ob is not None:
            self.batch_obs.append(batch_ob)
        if batch_act is not None:
            self.batch_acts.append(batch_act)
        if batch_ret is not None:
            self.batch_rets.append(batch_ret)
        if batch_len is not None:
            self.batch_lens.append(batch_len)
        if ep_rews is not None:
            self.batch_rews.append(ep_rews)

    def __call__(self, mode="default"):
        return (torch.as_tensor(self.batch_obs, dtype=torch.float32),
                torch.as_tensor(self.batch_acts, dtype=torch.int32),
                self.batch_rews, self.batch_rets, self.batch_lens
                )

    def __str__(self):
        return str(self.__dict__)
