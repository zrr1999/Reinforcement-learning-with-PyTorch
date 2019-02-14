from gym import spaces, Env
import numpy as np
import time

class TestEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, l):
        self.l = l
        self.action_space = spaces.Discrete(3)  # 0, 1, 2: 不动，左右
        self.observation_space = spaces.Discrete(l)
        self.state = None

    def step(self, action: int):
        assert self.action_space.contains(action), "{} {} invalid".format(
            action, type(action))
        if action == 1:
            self.state = max(0, self.state - 1)
            reward = 0
        elif action == 2:
            if self.state == self.l - 2:
                reward = 1
            else:
                reward = 0
            self.state = min(self.l - 1, self.state + 1)
        else:
            reward = 0
        if self.state == self.l - 1 or self.state == 0:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.choice(range(1,self.l-1))
        return self.state

    def render(self, mode='human'):
        for i in range(self.l):
            if i == self.state:
                print('X', end=' ')
            else:
                print('_', end=' ')
        print()
        time.sleep(0.5)


    def close(self):
        return None

if __name__ == '__main__':
    env = TestEnv(10)
    s = env.reset()
    while True:
        env.render()
        # u = RL.choose_action(s)
        s_, r_, done, _ = env.step(2)
        # RL.learn(s, u, r_, s_)
        if done:
            break
        s = s_
