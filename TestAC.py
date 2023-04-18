import retro
import gym
import cv2
import time
import torch

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
env = gym.wrappers.ResizeObservation(env, (182,360))
env = gym.wrappers.FrameStack(env, num_stack=4)
print(env.observation_space.sample().shape)

obs = env.reset()

x = []
for frame in obs:
    frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
    frame = frame[::3, ::3]  # downsample
    frame = frame / 255 # scale pixel values
    frame = frame - frame.mean() # normalize pixel values
    x.append(torch.FloatTensor(frame.reshape(1, 61, 120)))
print(torch.stack(x, dim=1).shape)

# obs = env.reset()
# while True:
#     action = env.action_space.sample()
#     obs, rew, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()

