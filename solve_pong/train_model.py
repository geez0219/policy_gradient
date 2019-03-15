"""
This code is to train the AI model to play pong game
1. It will generate the trained model (including its check point file and tensorboard log)

"""
import os
import shutil
import numpy as np
import gym
import argparse
from solve_pong.PG import PG
from replaybuffer import ReplayBuffer
from frame_processor import FrameProcessor


def check_run_file(arg):
    if os.path.exists("{}{}".format(arg.save_path, arg.run_name)):
        print("the run directory [{}{}] already exists!".format(arg.save_path, arg.run_name))
        print("0: exist ")
        print("1: restored the session from checkPoint ")
        print("2: start over and overwrite")
        print("3: create a new run")
        mode = int(input("please select the mode:"))

        if mode == 0:
            exit("you select to exist")

        elif mode == 2:
            shutil.rmtree("{}{}".format(arg.save_path, arg.run_name))
            os.makedirs("{}{}".format(arg.save_path, arg.run_name))
            shutil.copyfile(__file__, "{}{}/copy_code.py".format(arg.save_path, arg.run_name))

        elif mode == 3:
            arg.run_name = input("please enter a new run name:")
            return check_run_file(arg)

        elif mode > 3 or mode < 0:
            raise ValueError("the valid actions are in range [0-3]")
    else:
        print("create run directory [{}{}]".format(arg.save_path, arg.run_name))
        mode = 2
        os.makedirs("{}{}".format(arg.save_path, arg.run_name))
        shutil.copyfile(__file__, "{}{}/copy_code.py".format(arg.save_path, arg.run_name))

    return mode

if __name__ == '__main__':
    # declare all agent and environment
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="the name of the training model", type=str)
    parser.add_argument("-l", "--learning_rate", help='the learning rate of the optimizer', type=float, default=1e-5)
    parser.add_argument("-n", "--games_num", help='the number of training games', type=int, default=500000)
    parser.add_argument("-r", "--record_period", help='the record period of tensorboard', type=int, default=100)
    parser.add_argument("-p", "--save_path", help='the saving path of checkpoint and model tensorboard data', type=str, default='./result/')
    arg = parser.parse_args()

    print('---------- argument setting -----------')
    print('run_name: {}'.format(arg.run_name))
    print('learning_rate: {}'.format(arg.learning_rate))
    print('games_num: {}'.format(arg.games_num))
    print('record_period: {}'.format(arg.record_period))
    print('save_path: {}'.format(arg.save_path))
    print('---------------------------------------')
    env = gym.make('Pong-v0')

    check_run_file(arg)

    agent = PG(run_name=arg.run_name,
               input_shape=[160,160],
               n_action=3,
               learning_rate=arg.learning_rate,
               save_path=arg.save_path,
               record_io=False,
               record=True,
               gpu_fraction=0.9)

    for i in range(arg.games_num):
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
            prob = agent.get_action_prob(input_frame)
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
            loss += agent.train(s, a, r, record=True if (step % arg.record_period == 0 and j == 0) else False)

        loss /= len(memory) # get tge average loss
        print('{}th step: the total reward is {}, the average loss is {}'.format(step, total_reward, loss))

        # save the file and reward
        if step % arg.record_period == 0:
            agent.save()
            agent.log_reward(total_reward)