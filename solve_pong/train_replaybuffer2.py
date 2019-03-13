"""
Try to solve_breakout:
1. using copy model
2. reduce the action space to 3
3. Adam optimizer
"""

import numpy as np
import gym
from solve_pong.PG import PG
from replaybuffer2 import ReplayBuffer
from frame_processor import FrameProcessor
import cv2

game_play = 500000
record_period = 10
train_batch_n = 20
train_batch_size = 200

if __name__ == '__main__':
    # declare all agent and environment
    env = gym.make('Pong-v0')
    agent = PG(run_name='PG_fully_connected_new_replay_buffer',
               input_shape=[160,160],
               n_action=3,
               learning_rate=1e-5,
               save_path='./result/',
               record_io=True,
               record=True,
               gpu_fraction=0.9)

    for i in range(game_play):
        replay_buffer = ReplayBuffer(input_shape=[160, 160], start_size=32, max_size=10000000)
        memory = []
        total_reward = 0

        obs = env.reset()
        frame_processor = FrameProcessor(obs)
        obs, reward, done, _ = env.step(agent.random_action() + 1)  # do random action at the first frame
        total_reward += reward
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
                replay_buffer.store_transition(input_frame, action)

            else:
                total_reward += reward
                if reward == 1:
                    replay_buffer.back_trace_reward(reward, 1)
                else:
                    replay_buffer.back_trace_reward(reward, 0.9)

                memory.append(replay_buffer)
                replay_buffer = ReplayBuffer(input_shape=[160, 160], start_size=32, max_size=10000000)

        # train in the memory
        step = agent.step_move()
        loss = 0
        for j in range(len(memory)):
            s, a, r = memory[j].get_SAR()
            loss += agent.train(s, a, r, record=True if (step % record_period == 0 and j == 0) else False)

        loss /= len(memory) # get tge average loss
        print('{}th step: the total reward is {}, the average loss is {}'.format(step, total_reward, loss))

        # save the file and reward
        if step % record_period == 0:
            agent.save()
            agent.log_reward(total_reward)