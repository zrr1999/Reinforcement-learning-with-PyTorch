import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class DQNNet(nn.Module):
    def __init__(self, n_actions, n_features):
        super(DQNNet, self).__init__()
        self.out_layer = nn.Linear(n_features, n_actions)

    def forward(self, x):
        return self.out_layer(x)


class DQNDuelingNet(nn.Module):
    def __init__(self, n_actions, n_features):
        super(DQNNet, self).__init__()
        self.value_layer = nn.Linear(n_features, n_actions)
        self.advantage_layer = nn.Linear(n_features, n_actions)

    def forward(self, x):
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        return value + advantage - advantage.mean(1)


class SumTree:
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2  # 获取父节点
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        r"""
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n, ), dtype=np.int32), np.empty(
            (n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min(
            [1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]
                          ) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.1,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=500,
                 memory_size=200,
                 batch_size=32,
                 e_greedy_increment=None,
                 double_q=False,
                 prioritized=False,
                 dueling=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q
        self.prioritized = prioritized

        self.learn_step_counter = 0
        if prioritized:
            self.memory = Memory(memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0

        if dueling:
            self.evaluate_net = DQNNet(n_actions, n_features)
            self.target_net = DQNNet(n_actions, n_features)
        else:
            self.evaluate_net = DQNDuelingNet(n_actions, n_features)
            self.target_net = DQNDuelingNet(n_actions, n_features)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(),
                                   learning_rate)
        self.his = []

    def store_transition(self, state, action, reward, next_state):
        transition = np.r_[state, action, reward, next_state]
        if self.prioritized:
            self.memory.store(transition)
            self.memory_counter += 1
        else:
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        state = t.Tensor(observation[np.newaxis, :])

        if np.random.uniform() < self.epsilon:
            actions_value = self.evaluate_net(state)
            action = actions_value.argmax(1).numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
            print("Target_net is updated")
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(
                self.batch_size)
        else:
            indices = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[indices, :]

        s = t.Tensor(batch_memory[:, :self.n_features])
        u = t.LongTensor(batch_memory[:, self.n_features, np.newaxis])
        r = t.Tensor(batch_memory[:, self.n_features + 1, np.newaxis])
        s_ = t.Tensor(batch_memory[:, -self.n_features:])

        q_eval = self.evaluate_net(s).gather(1, u)
        if self.double_q:
            q_target_next = self.target_net(s_).detach()
            q_eval_next = self.evaluate_net(s_).detach()
            q_next = q_target_next.gather(1,
                                          q_eval_next.argmax(1).reshape(-1, 1))
        else:
            q_next = self.target_net(s_).detach().max(1)[0]
        delta = r + self.gamma * q_next - q_eval

        if self.prioritized:
            abs_errors = t.sum(
                t.abs(
                    self.evaluate_net(s).detach() -
                    self.target_net(s).detach()), 1)  # for updating Sumtree
            loss = t.mean(t.Tensor(ISWeights) * delta**2)
            self.memory.batch_update(tree_idx, abs_errors)  # update priority
        else:
            loss = t.mean(delta**2)

        # train eval network
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.his.append(loss.data.numpy())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
