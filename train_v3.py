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
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ICM(nn.Module):
    def __init__(self, in_channels=4, num_actions=12):
        """
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(ICM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        
        # Forward model
        self.pred_module1 = nn.Linear(512 + num_actions, 256) 
        self.pred_module2 = nn.Linear(256, 512) # output feature
        
        # Inverse model
        self.invpred_module1 = nn.Linear(512 + 512, 256)
        self.invpred_module2 = nn.Linear(256, num_actions)

    def get_feature(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return x
    
    def forward(self, x):
        # get feature
        feature_x = self.get_feature(x)
        return feature_x
    
    def get_full(self, s, s_next, a_vec):
        # get feature
        feature_s = self.get_feature(s)
        feature_s_next = self.get_feature(s_next)

        pred_s_next = self.pred(feature_s, a_vec) # predict next state feature
        pred_a_vec = self.invpred(feature_s, feature_s_next) # (inverse) predict action

        return pred_s_next, pred_a_vec, feature_s_next

    def pred(self, feature_s, a_vec):
        # forward prediction: predict next state feature, given current state feature and action (one-hot)
        pred_s_next = F.relu(self.pred_module1(torch.cat([feature_s, a_vec.float()], dim = -1).detach()))
        pred_s_next = self.pred_module2(pred_s_next)
        return pred_s_next
    
    def invpred(self,feature_s, feature_s_next):
        # inverse prediction: predict action (one-hot), given current and next state features
        pred_a_vec = F.relu(self.invpred_module1(torch.cat([feature_s, feature_s_next], dim = -1)))
        pred_a_vec = self.invpred_module2(pred_a_vec)
        return F.softmax(pred_a_vec, dim = -1)



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
        self.icm_model = ICM(input_shape[0], env.action_space.n).to(device)

        for param in self.target_model.parameters():
            param.requires_grad = False
        # self.target_model.load_state_dict(self.policy_model.state_dict())

        # training
        self.training = True
        self.gamma = 0.9
        self.batch_size = 32
        self.step_count = 0
        self.network_sync_rate = 10000
        self.dqn_optimzer = optim.Adam(self.policy_model.parameters(), lr = 0.00025) #lr=0.00025
        self.icm_optimzer = optim.Adam(self.icm_model.parameters(), lr = 0.00025)
        self.criterion = nn.SmoothL1Loss()
        self.beta = 5

        # exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.0001
        self.exploration_fraction = 500000
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
    
    def softmax(self, x, temperature=1.0):
        x = x / temperature
        e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
        return e_x / e_x.sum(axis=-1, keepdims=True)


    def act(self, state, epsilon):
        state = state.to(device)
        qvalues = self.policy_model(state)
        action = torch.argmax(qvalues).item()

        # only for plot
        self.qvalues.append(torch.max(qvalues).item())

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
        # batch = self.replay_memory.sample(self.batch_size)
        # state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        
        # state_batch = (torch.tensor(np.array(state), dtype=torch.float) / 255.0).to(device)
        
        # # Normalize next_states
        # next_state_batch = (torch.tensor(np.array(next_state), dtype=torch.float) / 255.0).to(device)
        
        # action_batch = action.to(device)
        # reward_batch = reward.to(device)
        # done_batch = done.to(device)

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

        one_hot_action_batch = F.one_hot(action_batch, num_classes=self.action_space.n).float()
        # pred_s_next, pred_a_vec, feature_s_next = self.icm_model(state_batch, next_state_batch, one_hot_action_batch)
        feature_s = self.icm_model(state_batch)
        feature_s_next = self.icm_model(next_state_batch)

        pred_s_next = self.icm_model.pred(feature_s.detach(), one_hot_action_batch)
        pred_a_vec = self.icm_model.invpred(feature_s, feature_s_next)

        with torch.no_grad():
            r_icm = 1e8 * F.mse_loss(pred_s_next, feature_s_next, reduction='none').mean(dim=1) + F.cross_entropy(pred_a_vec, action_batch, reduction='none')

        mse_loss = 1e8 * F.mse_loss(pred_s_next, feature_s_next.detach())
        cross_entropy_loss = F.cross_entropy(pred_a_vec, action_batch)
        loss_icm = mse_loss + cross_entropy_loss
        
        
        # back-propagation of icm
        self.icm_optimzer.zero_grad()
        loss_icm.backward()
        self.icm_optimzer.step()

        # TD estimate
        state_action_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))
        # state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # TD target
        # with torch.no_grad():
        next_state_action_value = self.target_model(next_state_batch).max(dim=1)[0].detach()
        # next_state_action_value = self.target_model(next_state_batch).max(dim=1)[0].unsqueeze(1).detach()

        # expected_state_action_values = reward_batch + (1-done_batch.float()) * self.gamma * next_state_value
        expected_state_action_values = reward_batch + self.gamma * next_state_action_value + self.beta * r_icm
        

        # loss function
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = self.criterion(state_action_values, expected_state_action_values)

        # back-propagation
        self.dqn_optimzer.zero_grad()
        loss.backward()

        # clamp gradient
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        # update weight
        self.dqn_optimzer.step()

        self.step_count += 1
        
        if self.step_count >= self.network_sync_rate:
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.step_count=0

        return loss.item(), mse_loss.item(), cross_entropy_loss.item(), r_icm.min(), r_icm.max()

    def clearbuffer(self):
        self.losses = []
        self.rewards = []
        self.qvalues = []
    
    def save_model(self, model, file_name="policy_model_best.pth"):
        torch.save(model.state_dict(), file_name)
        # scripted_model = torch.jit.script(self.policy_model)
        # torch.jit.save(scripted_model, file_name)


def train(env, episodes):
    log_data = []
    max_ret = 0
    obs = env.reset()

    agent = Agent(obs.shape)

    # The global timestep and progress_bar is for epsilon scheduling and progress visualization
    global_timestep = 0
    progress_bar = tqdm(total=agent.total_timestep, desc="Training Progress")

    for episode in range(1, episodes + 1):

        ret = 0
        learn_count = 0
        done = False
        
        obs_input = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0) / 255

        while True:
            # get epsilon from epsilon-scheduler, depends on the curent global-timestep
            # epsilon = agent.linear_schedule(agent.epsilon_start, agent.epsilon_end, agent.exploration_fraction * agent.total_timestep, global_timestep)
            epsilon = max(10 ** (- global_timestep / agent.exploration_fraction), agent.epsilon_end)

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
            loss, mes, cross_entropy, r_icm_min, r_icm_max = agent.learn()
            
            learn_count += 1
            global_timestep += 1

            # for plot
            agent.losses.append(loss)
            agent.rewards.append(reward)

            # log info
            if learn_count % 1000 == 0:
                tqdm.write(f"Episode {episode}, Step {learn_count}, Loss: {loss:.4f}, Epsilon: {epsilon}")
                tqdm.write(f"MSE Loss: {mes}, Cross Entropy: {cross_entropy}, ICM Reward: {r_icm_min}, {r_icm_max}")

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
        tqdm.write(f"MSE Loss: {mes}, Cross Entropy: {cross_entropy}, ICM Reward: {r_icm_min}, {r_icm_max}")
        log_data.append({"episode": episode, "return": ret})
        
        # save the model with higtest return
        if ret > max_ret:
            agent.save_model(agent.policy_model, file_name='policy_model_best.pth')
            max_ret = ret

        # Save model every 20 episodes
        if episode % 20 == 0:
            agent.save_model(agent.policy_model, file_name='policy_model_latest.pth')
            agent.save_model(agent.icm_model, file_name='icm_model.pth')
            tqdm.write("[INFO]: Save model!")

            with open("training_log.json", "w") as log_file:
                json.dump(log_data, log_file, indent=4)
            tqdm.write(f"Training log saved at episode {episode}")
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

    folder = "plot/training7"
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