import gym
import random
import numpy as np
import math
import copy

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from collections import deque

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU

# Optional: ensure deterministic behavior (can slow down training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
            NoisyLinear(conv_out_size, 512, training = False),
            nn.ReLU(),
            NoisyLinear(512, n_actions, training = False)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


device = 'cpu'

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frames = deque([], maxlen=4)
        # self.prev_frames = deque([], maxlen=4)
        self.first_obs = None
    
        self.policy_model = DQNSolver([4, 84, 84], 12)
        self.policy_model.load_state_dict(torch.load('policy_model_latest_4.pth'))
        self.policy_model.to(device)
        self.policy_model.eval() # Set to evaluation mode

        # self.icm_model = ICM()
        # self.icm_model.load_state_dict(torch.load('icm_model.pth'))
        # self.icm_model.to(device)
        # self.icm_model.eval()

        self.epsilon = 0.008
        
        self.prev_action = 0
        self.count = 0

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((84, 84))
        ])        

    def softmax(self, x, temperature=1.0):
        x = x / temperature
        e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def act(self, observation):
        # select action only at no skpping frames
        # if len(self.frames) == 0:
        #     self.first_obs = copy.copy(observation)
        # else:
        #     if np.array_equal(observation, self.first_obs):
        #         print("empty the buffer")
        #         self.count = 0
        #         self.frames.clear()

        if self.count % 4 == 0:
            obs = torch.tensor(observation.copy()).permute(2, 0, 1)
            frame = self.transform(obs)
            frame = np.array(frame)

            self.frames.append(frame)
            if len(self.frames) < 4:
                while len(self.frames) < 4:
                    self.frames.append(frame)

            # while len(self.prev_frames) < 4:
            #     self.prev_frames.append(np.zeros_like(frame))

            input = np.concatenate(list(self.frames), axis=0)
            input = torch.tensor(input, dtype=torch.float32).unsqueeze(0) / 255.0
            # prev_input = np.concatenate(list(self.prev_frames), axis = 0)
            # prev_input = torch.tensor(prev_input, dtype=torch.float32).unsqueeze(0) / 255.0
        
            with torch.no_grad():
                action_values = self.policy_model(input.to(device)) #.detach().cpu().numpy()
            #     one_hot_prev_action = F.one_hot(torch.tensor([self.prev_action]), num_classes=self.action_space.n).float()
            #     pred_state, target_state = self.icm_model.get_full(prev_input, input, one_hot_prev_action)

            #     icm_reward = F.mse_loss(pred_state, target_state.detach(), reduction='none').mean()

            # print(action_values, icm_reward)
            action = torch.argmax(action_values).item()
            # prob = self.softmax(action_values, temperature=0.05).squeeze(0)
            # action = np.random.choice(np.arange(self.action_space.n), p=prob)
            self.prev_action = action
            # self.prev_frames.append(frame)

        else:
            action = self.prev_action

        self.count += 1

        r = random.random()
        # with open("random_value.txt", "a") as f:
        #     f.write(f"{r}\n")
        
        if r < self.epsilon:
            # a = self.action_space.sample()
            a = random.choice(range(self.action_space.n))
            # with open("random_action.txt", "a") as f:
            #     f.write(f"{a}\n")
            return a
        else:
            return action
