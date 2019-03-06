import numpy as np
import cv2
import gym

class FrameProcessor:
    def __init__(self, first_frame):
        self.first_frame = self.binarize(self.crop(first_frame)).astype(np.float32)

    def process(self, input_frame):
        second_frame = self.binarize(self.crop(input_frame)).astype(np.float32)
        output = second_frame - self.first_frame
        self.first_frame = second_frame

        return output

    @staticmethod
    def crop(input_frame):
        img = input_frame[35:195, :, 0]

        return img

    @staticmethod
    def binarize(input_frame):
        img = input_frame
        img[input_frame == 144] = 0
        img[input_frame == 109] = 0
        img[input_frame != 0] = 1

        return img

    @staticmethod
    def OneTo255(input_frame):
        input_frame[input_frame == 1] = 255

        return input_frame


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    obs = env.reset()
    frame_processor = FrameProcessor(obs)
    obs, reward, done, _ = env.step(env.action_space.sample())
    while not done:
        frame_show = FrameProcessor.OneTo255(frame_processor.process(obs))
        cv2.imshow('test', frame_show)
        cv2.waitKey(30)
        obs, reward, done, _ = env.step(env.action_space.sample())



