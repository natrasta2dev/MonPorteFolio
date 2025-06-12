import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from env.tetris_env import TetrisEnv

# ==== DQN ==== #
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=10000)

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1)[0].detach().unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (~dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==== Training Loop ==== #
def main(episodes=500):
    env = TetrisEnv()
    state_size = len(env._get_state())
    action_size = 4  # gauche, droite, rotation, drop
    agent = DQNAgent(state_size, action_size)
    scores = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 100

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            if reward == 0:
                total_reward -= 2
            else:
                total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

            if done:
                break

        scores.append(total_reward)
        print(f"Episode {ep+1}/{episodes} - Score: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

        if (ep + 1) % 50 == 0:
            torch.save(agent.model.state_dict(), f"models/dqn_tetris_ep{ep+1}.pth")

    plt.plot(scores)
    plt.title("Progression du Score de l'IA")
    plt.xlabel("Épisodes")
    plt.ylabel("Score")
    plt.grid()
    plt.savefig("training_progress.png")  # utile pour Render
    plt.close()

if __name__ == "__main__":
    main()
