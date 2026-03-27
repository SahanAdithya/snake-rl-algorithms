import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
from collections import deque

class Dueling_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean())
        return q_values

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.n_entries + self.capacity - 1
        self.data[self.n_entries] = data
        self.update(idx, p)
        self.n_entries += 1
        if self.n_entries >= self.capacity:
            self.n_entries = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class Memory: # Prioritized Experience Replay
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.e = 0.01
        self.a = 0.6

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        return batch, idxs

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Current Q values
        pred = self.model(state)
        target = pred.clone()

        # 2: DDQN Logic
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Action selection from regular model
                next_action = torch.argmax(self.model(next_state[idx])).item()
                # Action evaluation from target model
                Q_new = reward[idx] + self.gamma * self.target_model(next_state[idx])[next_action]

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = Memory(MAX_MEMORY)
        self.model = Dueling_QNet(11, 256, 3)
        self.target_model = Dueling_QNet(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        self.target_update_frequency = 10 # episodes

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        # Calculate initial error for PER
        self.model.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float)
            next_state_t = torch.tensor(next_state, dtype=torch.float)
            
            old_val = self.model(state_t)[np.argmax(action)]
            target_val = reward
            if not done:
                next_action = torch.argmax(self.model(next_state_t)).item()
                target_val += self.gamma * self.target_model(next_state_t)[next_action]
            
            error = abs(old_val - target_val).item()
        self.model.train()
        
        self.memory.add(error, (state, action, reward, next_state, done))

    def train_long_memory(self):
        if self.memory.tree.total() > 0:
            mini_sample, idxs = self.memory.sample(min(self.memory.tree.n_entries, BATCH_SIZE))
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
            # In a full PER implementation, we would update priorities here, but this is a simplified version
            return loss
        return 0

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
