import numpy as np
import pandas as pd


class RL:
    def __init__(self,
                 actions,
                 learning_rate=0.1,
                 reward_decay=0.9,
                 e_greedy=0.9):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=list(actions), dtype=np.float)

    def state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(0, index=self.q_table.columns, name=state))

    def learn(self, *args):
        raise Exception("未定义learn方法")

    def choose_action(self, state):
        state = str(state)
        self.state_exist(state)
        if np.random.uniform() < 0.9:
            q = self.q_table.loc[state].sample(frac=1)
            return q.idxmax()
        else:
            return np.random.choice(self.actions)


class QLearning(RL):
    def __init__(self,
                 actions,
                 learning_rate=0.1,
                 reward_decay=0.9,
                 e_greedy=0.9):
        super(QLearning, self).__init__(actions, learning_rate, reward_decay,
                                        e_greedy)

    def learn(self, state, action, reward, next_state):
        state = str(state)
        next_state = str(next_state)
        self.state_exist(next_state)
        r = reward + self.gamma * self.q_table.loc[next_state].max()
        self.q_table.loc[state, action] += self.alpha * (
            r - self.q_table.loc[state, action])


class Sarsa(RL):
    def __init__(self,
                 actions,
                 learning_rate=0.1,
                 reward_decay=0.9,
                 e_greedy=0.9):
        super(Sarsa, self).__init__(actions, learning_rate, reward_decay,
                                    e_greedy)

    def state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(0, index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(to_be_append)

    def learn(self, state, action, reward, next_state):
        state = str(state)
        next_state = str(next_state)
        self.state_exist(next_state)
        r = reward + self.gamma * self.q_table.loc[
            next_state, self.choose_action(next_state)]
        self.q_table.loc[state, action] += self.alpha * (
            r - self.q_table.loc[state, action])


class SarsaLambda(RL):
    def __init__(self,
                 actions,
                 learning_rate=0.1,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 trace_decay=0.5):
        super(SarsaLambda, self).__init__(actions, learning_rate, reward_decay,
                                          e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(0, index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(
                to_be_append)

    def learn(self, state, action, reward, next_state):
        state = str(state)
        next_state = str(next_state)
        self.state_exist(next_state)

        # self.eligibility_trace.loc[state, action] += 1
        self.eligibility_trace.loc[state, :] = 0
        self.eligibility_trace.loc[state, action] = 1

        q_ = self.q_table.loc[next_state, self.choose_action(next_state)]
        delta = reward + self.gamma * q_ - self.q_table.loc[state, action]
        self.q_table += self.alpha * delta * self.eligibility_trace
        self.eligibility_trace *= self.lambda_ * self.gamma
