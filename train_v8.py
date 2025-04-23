import cv2
import pygame
import random
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

import gym
import gym_super_mario_bros
# from gym.spaces import Box
# from gym.wrappers.frame_stack import LazyFrames
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from nes_py.wrappers import JoypadSpace
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# define the replay memory save type
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0') # observation shape: (240, 256, 3)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=4)
    
    return env

# Noise Network
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5, training=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters() # for mu and sigma
        self.reset_noise() # for epsilon

        self.training = training

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # elementwise
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class Agent():
    def __init__(self, input_shape):
        self.action_space = env.action_space
        self.replay_memory = deque([], maxlen = 500000)
        # self.replay_memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))

        # model
        self.policy_model = DQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        self.target_model = DQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        # self.policy_model = torch.jit.load("policy_model_latest.pth")
        # self.target_model = torch.jit.load("policy_model_latest.pth")
        for param in self.target_model.parameters():
            param.requires_grad = False
        # self.target_model.load_state_dict(self.policy_model.state_dict())

        # training
        self.training = True
        self.gamma = 0.9
        self.batch_size = 32
        self.step_count = 0
        self.network_sync_rate = 10000
        self.optimzer = optim.Adam(self.policy_model.parameters(), lr = 0.00025) #lr=0.00025
        self.criterion = nn.SmoothL1Loss()

        # exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.001
        self.exploration_fraction = 0.1
        self.total_timestep = 10000000

        # information buffer
        self.losses = []
        self.rewards = []
        self.qvalues = []
        self.reward_mean = []
        self.qvalues_mean = []
        self.returns = []

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        # define the percentatge of exploration
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def multi_stage_epsilon(self, global_timestep, start_e=1.0, end_e=0.001, decay_per_stage=0.1, steps_per_stage=750000):
        stage = global_timestep // steps_per_stage
        t = global_timestep % steps_per_stage

        current_start = start_e * (decay_per_stage ** stage)
        current_end = start_e * (decay_per_stage ** (stage + 1))

        if current_end < end_e:
            return end_e

        slope = (current_end - current_start) / steps_per_stage
        return slope * t + current_start


    def act(self, state, epsilon):
        state = state.to(device)
        qvalues = self.policy_model(state)
        action = torch.argmax(qvalues).item()

        # only for plot
        self.qvalues.append(torch.max(qvalues).item())
        # print(qvalues, action)

        # epsilon-greedy
        if np.random.rand() < epsilon and self.training:
            return self.action_space.sample()
        else:
            return action

    def cache(self, state, next_state, action, reward, done):
        # save experience into replay memory
        action = torch.tensor(action).unsqueeze(0)
        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)

        self.replay_memory.append(Transition(state, action, reward, next_state, done))
        # self.replay_memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def sample(self):
        return random.sample(self.replay_memory, self.batch_size)
    
    def recall(self):
        random_samples = self.sample() # list of Transition
        batch = Transition(*zip(*random_samples)) # Transition of list

        # Normalize states
        state_batch = (torch.tensor(np.array(batch.state), dtype=torch.float) / 255.0).to(device)
        
        # Normalize next_states
        next_state_batch = (torch.tensor(np.array(batch.next_state), dtype=torch.float) / 255.0).to(device)
        
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch

    def learn(self):
        
        # sample the experience from replay memory
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.recall()

        # TD estimate
        state_action_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))
        # state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # TD target
        # with torch.no_grad():
        next_state_action_value = self.target_model(next_state_batch).max(dim=1)[0].detach()
        # next_state_action_value = self.target_model(next_state_batch).max(dim=1)[0].unsqueeze(1).detach()

        expected_state_action_values = reward_batch + (1-done_batch.float()) * self.gamma * next_state_action_value
        # expected_state_action_values = reward_batch + self.gamma * next_state_action_value

        # loss function
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = self.criterion(state_action_values, expected_state_action_values)

        # back-propagation
        self.optimzer.zero_grad()
        loss.backward()

        # clamp gradient
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        # update weight
        self.optimzer.step()

        self.step_count += 1
        
        if self.step_count >= self.network_sync_rate:
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.step_count=0

        return loss.item()

    def clearbuffer(self):
        self.losses = []
        self.rewards = []
        self.qvalues = []
    
    def save_model(self, model, file_name="policy_model_best.pth"):
        torch.save(model.state_dict(), file_name)


def train(env, episodes):
    log_data = []
    max_ret = 0
    obs = env.reset()

    agent = Agent(obs.shape)

    # The global timestep and progress_bar is for epsilon scheduling and progress visualization
    global_timestep = 0
    epsilon = agent.epsilon_start
    progress_bar = tqdm(total=agent.total_timestep, desc="Training Progress")

    for episode in range(1, episodes + 1):

        ret = 0
        learn_count = 0
        done = False
        
        obs_input = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0) / 255

        while True:
            # get epsilon from epsilon-scheduler, depends on the curent global-timestep
            # epsilon = agent.linear_schedule(agent.epsilon_start, agent.epsilon_end, agent.exploration_fraction * agent.total_timestep, global_timestep)
            # epsilon = max(10 ** (- global_timestep / agent.exploration_fraction), agent.epsilon_end)
            # epsilon = agent.multi_stage_epsilon(global_timestep)
            epsilon = 0
        
            # choose action by epsilon-greedy
            action = agent.act(obs_input, epsilon)
            
            # apply action to environment and get r and s'
            next_obs, reward, done, info = env.step(action)

            next_obs_input = torch.tensor(np.array(next_obs), dtype=torch.float32).unsqueeze(0) / 255

            # update return
            ret += reward

            # save experience into buffer
            agent.cache(obs, next_obs, action, reward, done)
            
            obs_input = next_obs_input
            obs = next_obs
            
            # env.render()
            
            if len(agent.replay_memory) < agent.batch_size:
                continue

            # optimize the model
            loss = agent.learn()

            learn_count += 1
            global_timestep += 1

            # for plot
            agent.losses.append(loss)
            agent.rewards.append(reward)

            # log info
            # if learn_count % 1000 == 0:
            # print(agent.policy_model.fc[2].weight_mu, agent.policy_model.fc[2].weight_sigma, agent.policy_model.fc[2].bias_mu, agent.policy_model.fc[2].bias_sigma)
            #     tqdm.write(f"Episode {episode}, Step {learn_count}, Loss: {loss:.4f}, Epsilon: {epsilon}")

            # Update tqdm bar manually
            progress_bar.update(1)

            # Check if end of game
            if done or info["flag_get"]:
                break

        agent.reward_mean.append(np.mean(agent.rewards))
        agent.qvalues_mean.append(np.mean(agent.qvalues))
        agent.returns.append(ret)

        tqdm.write(f"Storage: {len(agent.replay_memory)}")
        tqdm.write(f"Episode {episode} Return: {ret}, Epsilon: {epsilon}")
        log_data.append({"episode": episode, "return": ret})
        
        # save the model with higtest return
        if ret > max_ret:
            agent.save_model(agent.policy_model, file_name='policy_model_best.pth')
            max_ret = ret

        # Save model every 20 episodes
        if episode % 20 == 0:
            agent.save_model(agent.policy_model, file_name='policy_model_latest.pth')
            tqdm.write("[INFO]: Save model!")

            state_dict = torch.load("policy_model_latest.pth")
            with open("training_noise.txt", "a") as f:
                f.write(f"Epoch: {episode}\n")
                f.write(f"weight_mu_0: {(state_dict['fc.0.weight_mu']).mean().item()}\n")
                f.write(f"weight_sigma_0: {(state_dict['fc.0.weight_sigma']).mean().item()}\n")
                f.write(f"bias_mu_0: {(state_dict['fc.0.bias_mu']).mean().item()}\n")
                f.write(f"bias_sigma_0: {(state_dict['fc.0.bias_sigma']).mean().item()}\n")
                f.write(f"weight_mu_1: {(state_dict['fc.2.weight_mu']).mean().item()}\n")
                f.write(f"weight_sigma_1: {(state_dict['fc.2.weight_sigma']).mean().item()}\n")
                f.write(f"bias_mu_1: {(state_dict['fc.2.bias_mu']).mean().item()}\n")
                f.write(f"bias_sigma_1: {(state_dict['fc.2.bias_sigma']).mean().item()}\n\n")

            # with open("training_log.json", "w") as log_file:
            #     json.dump(log_data, log_file, indent=4)
            # tqdm.write(f"Training log saved at episode {episode}")
            save_plot(episode, agent.losses, agent.reward_mean, agent.qvalues_mean, agent.returns)
        
        agent.clearbuffer()
        obs = env.reset()

    progress_bar.close()

def moving_average(values, n):
    offset = (n - 1) // 2
    v = [values[0]] * offset + values + [values[-1]] * offset
    return [sum(v[i - offset : i + offset + 1]) / n for i in range(offset, len(v) - offset)]

def save_plot(episode, losses, rewards, qvalues, returns):
    fig, axis = plt.subplots(2, 3, figsize=(16, 5))
    axis = axis.flatten()

    # plot loss curve
    axis[0].plot(range(len(losses)), losses)
    axis[0].set_ylabel('Loss per optimization')
    # plot average reward per epsiode
    axis[1].plot(range(len(rewards)), rewards)
    axis[1].set_ylabel('Average of the reward per episode')
    # plot average max Q-value per epsiode
    axis[2].plot(range(len(qvalues)), qvalues)
    axis[2].set_ylabel('Average of the max predicted Q value')
    # plot return per epsiode
    axis[3].plot(range(len(returns)), returns)
    axis[3].set_ylabel('Return per episode')
    # plot the moving average of return 
    returns_movavg = moving_average(returns, 60)
    axis[3].plot(range(len(returns_movavg)), returns_movavg, color='red')

    fig.suptitle(f"Episode {episode}")
    fig.tight_layout()

    folder = "plot/training12"
    os.makedirs(folder, exist_ok=True)
    
    plt.savefig(f"{folder}/episode-{episode}.png")
    tqdm.write(f"Figure \"episode-{episode}.png\" saved.")
    for axis in axis:
        axis.cla()
    plt.close(fig)

if __name__ == '__main__':
    # Initialize environment
    env = make_env()
    # Initialize pygame
    pygame.init()

    episodes = 20000
    train(env, episodes)

    env.close()

'''
Full Action Space:
0: ['NOOP']
1: ['right']
2: ['right', 'A']
3: ['right', 'B']
4: ['right', 'A', 'B']
5: ['A']
6: ['left']
7: ['left', 'A']
8: ['left', 'B']
9: ['left', 'A', 'B']
10: ['down']
11: ['up']
'''