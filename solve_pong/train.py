"""
Try to solve_breakout:
1. using copy model
2. reduce the action space to 3
3. Adam optimizer
"""

import numpy as np
import gym
from solve_pong.PG import PG
from replaybuffer import ReplayBuffer
from frame_processor import FrameProcessor
import cv2

def getRewardTrace(reward, size, decay):
    '''
    return the list of [...., reward*decay*decay*2, reward*decay, reward]
    :param reward: <flaot>
    :param size: <int>
    :param decay: <float>
    :return:
    '''
    output = np.ones(size)
    output[size-1] = reward

    for i in range(size-2, -1, -1):
        output[i] = output[i+1]*decay

    return output

game_play = 500000
record_period = 10
train_batch_n = 20
train_batch_size = 200

if __name__ == '__main__':
    # declare all agent and environment
    env = gym.make('Pong-v0')
    agent = PG(run_name='PG_fully_connected',
               input_shape=[160,160],
               n_action=3,
               learning_rate=1e-5,
               save_path='./result/',
               record_io=True,
               record=True,
               gpu_fraction=0.9)

    replay_buffer = ReplayBuffer(input_shape=[160,160], size=100000)

    for i in range(game_play):
        s_list = np.empty((0, 160, 160))
        a_list = np.empty((0))
        total_reward = 0

        obs = env.reset()
        frame_processor = FrameProcessor(obs)
        obs, reward, done, _ = env.step(agent.random_action() + 1)  # do random action at the first frame

        # play one game
        while not done:
            input_frame = frame_processor.process(obs)
            # cv2.imshow('test', frame_processor.OneTo255(input_frame))
            # cv2.waitKey(30)
            # env.render()
            prob = agent.get_action_prob(input_frame)
            # print(prob)
            action = np.random.choice(3, p=prob)
            obs, reward, done, _ = env.step(action + 1)

            if reward == 0:
                s_list = np.concatenate((s_list, input_frame.reshape((1,160,160))), axis=0)
                a_list = np.concatenate((a_list, np.array([reward])), axis=0)

            else:
                total_reward += reward
                if reward == 1:
                    reward_trace = getRewardTrace(reward, len(s_list), 1)
                else:
                    reward_trace = getRewardTrace(reward, len(s_list), 0.9)

                replay_buffer.store_transition_batch(obs=s_list, action=a_list, reward=reward_trace)
                s_list = np.empty((0, 160, 160))
                a_list = np.empty((0))

        # train in the memory
        step = agent.step_move()
        loss = 0
        for j in range(train_batch_n):
            s, a, r = replay_buffer.sample(batch_size=train_batch_size)
            loss += agent.train(s, a, r, record=True if (step % record_period == 0 and j == 0) else False)

        loss /= train_batch_n # get tge average loss
        replay_buffer.clear()

        print('{}th step: the total reward is {}, the average loss is {}'.format(step, total_reward, loss))

        # save the file and reward
        if step % record_period == 0:
            agent.save()
            agent.log_reward(total_reward)