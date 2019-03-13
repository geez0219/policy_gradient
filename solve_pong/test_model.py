import numpy as np
import gym
import time
import argparse
import os
from solve_pong.PG import PG
from frame_processor import FrameProcessor

def play_game(agent, env, game_play, mode, render):
    action_list = ['stay', 'go up', 'go down']
    reward_avg = 0
    for i in range(game_play):
        total_reward = 0
        obs = env.reset()
        if render:
            env.render()
        frame_processor = FrameProcessor(obs)

        obs, reward, done, _ = env.step(agent.random_action() + 1)  # do random action at the first frame
        total_reward += reward
        # play one game
        while not done:
            if render:
                env.render()
                time.sleep(1/30)
            input_frame = frame_processor.process(obs)
            prob = agent.get_action_prob(input_frame)
            if mode == 0:
                action = np.random.choice(3, p=prob)

            elif mode == 1:
                action = np.argmax(prob)
            if render:
                print('the agent has {:.2f}% stay, {:.2f}% goes up, {:.2f}% goes down ==> it choose to {}' \
                      .format(prob[0] * 100, prob[1] * 100, prob[2] * 100, action_list[action]))
            obs, reward, done, _ = env.step(action + 1)
            total_reward += reward

        reward_avg += (total_reward - reward_avg)/(i+1) # use moving average to get reward average
        print('in {}th game: the total_reward is: {}, average total_reward is: {}'.format(i, total_reward, reward_avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="the name of the training model", type=str)
    parser.add_argument("-n", "--games_num", help='the number of training games', type=int, default=100)
    parser.add_argument("-p", "--load_path", help='the load path of checkpoint', type=str, default='./result/')
    parser.add_argument("-s", "--show", help='whether to show the gameplay screen', action='store_true')
    parser.add_argument("-c", "--choose_highest", help="whether the each action choose the highest probability one", action="store_true")
    arg = parser.parse_args()

    print('---------- argument setting -----------')
    print('run_name: {}'.format(arg.run_name))
    print('games_num: {}'.format(arg.games_num))
    print('load_path: {}'.format(arg.load_path))
    print('show: {}'.format(arg.show))
    print('choose_highest: {}'.format(arg.show))
    print('---------------------------------------')

    # declare all agent and environment

    env = gym.make('Pong-v0')

    if not os.path.exists('{}{}'.format(arg.load_path, arg.run_name)):
        raise ValueError('{}{} did not exist! '.format(arg.load_path, arg.run_name))

    agent = PG(run_name=arg.run_name,
               input_shape=[160, 160],
               n_action=3,
               learning_rate=0,
               save_path=arg.load_path,
               record_io=False,
               record=False,
               gpu_fraction=0.9)
    agent.load(arg.load_path, arg.run_name)
    play_game(agent, env, arg.games_num, arg.choose_highest, arg.show)

