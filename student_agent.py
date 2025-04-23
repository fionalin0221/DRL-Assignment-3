import gym
import random
import numpy as np

import torch
from torchvision import transforms
import torch.nn.functional as F
from collections import deque

from train import DQNSolver

device = 'cpu'


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frames = deque([], maxlen=4)

        # self.model = torch.jit.load('policy_model_latest.pth').to(device)
        self.policy_model = DQNSolver([4, 84, 84], 12)
        self.policy_model.load_state_dict(torch.load('policy_model_latest_4.pth'))
        self.policy_model.to(device)
        self.policy_model.eval() # Set to evaluation mode

        self.prev_action = 0
        self.count = 0

        self.epsilon = 0.001

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((84, 84))
        ])

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

        # Optional: ensure deterministic behavior (can slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def softmax(self, x, temperature=1.0):
        x = x / temperature
        e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def act(self, observation):
        # select action only at no skpping frames        
        if self.count % 4 == 0:
            obs = torch.tensor(observation.copy()).permute(2, 0, 1)
            frame = self.transform(obs)
            frame = np.array(frame)

            self.frames.append(frame)
            while len(self.frames) < 4:
                self.frames.append(frame)

            input = np.concatenate(list(self.frames), axis=0)
            input = torch.tensor(input, dtype=torch.float32).unsqueeze(0) / 255.0
        
            action_values = self.policy_model(input.to(device)) #.detach().cpu().numpy()
            # # print(action_values)
            # prob = self.softmax(action_values, temperature=0.05).squeeze(0)
            # # print(prob)
            # action = np.random.choice(np.arange(self.action_space.n), p=prob)
            action = torch.argmax(action_values).item()
            self.prev_action = action
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

        # return self.action_space.sample()