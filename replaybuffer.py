import numpy as np


class ReplayBuffer:
    def __init__(self, size, input_shape):
        self.size = size
        self.input_shape = input_shape
        self.s = np.zeros([self.size, ] + self.input_shape)
        self.a = np.zeros([self.size])
        self.r = np.zeros([self.size])
        self.counter = 0

    def store_transition(self, obs, action, reward):
        memory_idx = self.counter % self.size
        self.s[memory_idx] = obs
        self.a[memory_idx] = action
        self.r[memory_idx] = reward
        self.counter += 1

    def store_transition_batch(self, obs, action, reward):
        if type(obs) is not np.ndarray:
            raise TypeError('the input argument obs should be ndarray')

        if type(action) is not np.ndarray:
            raise TypeError('the input argument action should be ndarray')

        if type(reward) is not np.ndarray:
            raise TypeError('the input argument reward should be ndarray')

        batch_size = obs.shape[0]
        memory_idx_start = self.counter % self.size

        if memory_idx_start + batch_size > self.size:  # the memory_idx will return back during saving the batch
            if batch_size < self.size:    # need to split the input into two batches
                batch1_size = self.size - memory_idx_start
                batch2_size = batch_size - batch1_size

                self.s[memory_idx_start:self.size] = obs[:batch1_size]
                self.s[:batch2_size] = obs[batch1_size:]

                self.a[memory_idx_start:self.size] = action[:batch1_size]
                self.a[:batch2_size] = action[batch1_size:]

                self.r[memory_idx_start:self.size] = reward[:batch1_size]
                self.r[:batch2_size] = reward[batch1_size:]

            else:  # batch size is larger than memory size
                self.s[:] = obs[batch_size-self.size:batch_size]
                self.a[:] = action[batch_size-self.size:batch_size]
                self.r[:] = reward[batch_size-self.size:batch_size]

        else:
            self.s[memory_idx_start:memory_idx_start + batch_size] = obs
            self.a[memory_idx_start:memory_idx_start + batch_size] = action
            self.r[memory_idx_start:memory_idx_start + batch_size] = reward

        self.counter += batch_size

    def sample(self, batch_size):
        rand_idx = np.random.choice(min(self.counter, self.size), batch_size)

        return [self.s[rand_idx].copy(),
                self.a[rand_idx].copy(),
                self.r[rand_idx].copy()]

    def clear(self):
        self.s = np.zeros([self.size, ] + self.input_shape)
        self.a = np.zeros([self.size])
        self.r = np.zeros([self.size])
        self.counter = 0

    def get_current_size(self):
        return min(self.size, self.counter)

if __name__ == '__main__':
    print('testing replaybuffer')
    memory = ReplayBuffer(50, [1])
    batch1_obs = np.array([[i] for i in range(75)])
    batch1_action = np.array([i for i in range(75)])
    batch1_reward = np.array([i for i in range(75)])
    memory.store_transition_batch(batch1_obs, batch1_action, batch1_reward)
    print(memory.s)
    print(memory.a)
    batch1_obs = np.array([[i] for i in range(10)])
    batch1_action = np.array([i for i in range(10)])
    batch1_reward = np.array([i for i in range(10)])
    memory.store_transition_batch(batch1_obs, batch1_action, batch1_reward)
    print(memory.s)
    print(memory.a)

    s, a, r = memory.sample(20)
    print(s)

