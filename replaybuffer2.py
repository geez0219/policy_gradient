import numpy as np


class ReplayBuffer:
    def __init__(self, input_shape, start_size, max_size):
        self.s = np.zeros([start_size] + input_shape)
        self.a = np.zeros([start_size])
        self.r = np.zeros(0)
        self.counter = 0
        self.current_size = start_size
        self.max_size = max_size
        self.input_shape = input_shape

    def double_size(self):
        new_size = min(self.current_size * 2, self.max_size)

        new_s = np.zeros([new_size] + self.input_shape)
        new_a = np.zeros([new_size])
        new_s[:self.current_size] = self.s
        new_a[:self.current_size] = self.a
        self.s = new_s
        self.a = new_a
        self.current_size = new_size

    def store_transition(self, s, a):
        if self.counter == self.current_size:
            if self.current_size == self.max_size:
                print('the ReplayBuffer size reach its max_size, cannot store more transition')
                return
            else:
                self.double_size()
        self.s[self.counter] = s
        self.a[self.counter] = a
        self.counter += 1

    def back_trace_reward(self, reward, decay):
        if self.counter != 0:
            self.r = np.zeros([self.counter])
            self.r[-1] = reward

            for i in range(self.counter-2, -1, -1):
                self.r[i] = self.r[i+1] * decay

    def get_SAR(self):
        return self.s[:self.counter], self.a[:self.counter], self.r[:self.counter]



if __name__ == '__main__':

    replay_buffer = ReplayBuffer([10], 1, 100)
    for i in range(100):
        replay_buffer.store_transition(np.array([i]*10), i)
        replay_buffer.back_trace_reward(1,0.5)






