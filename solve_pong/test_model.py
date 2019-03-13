import numpy as np
import gym
import time
from solve_pong.PG import PG
from frame_processor import FrameProcessor


def play_game(agent, env, mode, render):
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
    # declare all agent and environment
    env = gym.make('Pong-v0')
    game_play = 500000
    run_name = 'PG_fully_connected_new_replay_buffer'
    agent = PG(run_name=run_name,
               input_shape=[160,160],
               n_action=3,
               learning_rate=1e-5,
               save_path='./result/',
               record_io=True,
               record=True,
               gpu_fraction=0.9)

    print("please select the action selection mode you want")
    mode = int(input("0: normal, 1: choose highest prob:"))
    print("please select the whether to render the playing process")
    render = int(input("0: no, 1: yes:"))
    play_game(agent, env, mode, render)
