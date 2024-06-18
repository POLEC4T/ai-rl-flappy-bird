import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import DQN
from flappy_bird_game import FlappyBirdGameAI
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.98,
        epsilon=1.0,
        lr=0.0001,
        batch_size=512,
        max_mem_size=50000,
        target_update_freq=50,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: ########## {self.device} ##########")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.update_target_model()  # Initialize target model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.target_update_freq = target_update_freq
        self.steps_done = 0

    def update_target_model(self):
        # print(f'### Updating target model')
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.update_target_model()
            print("Successfully loaded model", file_path)

    def save_model(self, file_path):
        # print(f'Saving model to {file_path}')
        torch.save(self.model.state_dict(), file_path)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            if np.random.rand() < 0.05:  # 5% chance of taking a random action
                return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()

    def experience_replay(self):
        # print(f'### Experience Replay, memory size: {len(self.memory)}')
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(
            states
        )  # Convert list of numpy arrays to a single numpy array
        states = torch.FloatTensor(states).to(
            self.device
        )  # Convert numpy array to tensor
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = np.array(
            next_states
        )  # Convert list of numpy arrays to a single numpy array
        next_states = torch.FloatTensor(next_states).to(
            self.device
        )  # Convert numpy array to tensor
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_model()

        return loss.item()


def save_max_score(max_score, filepath="max_score.txt"):
    with open(filepath, "w") as f:
        f.write(str(max_score))


def load_max_score():
    with open('max_score.txt', 'r') as f:
        score = f.read()
        return int(score) if score else 0


def train_dqn_agent(render=False):
    num_episodes = 20000
    game = FlappyBirdGameAI(display_screen=render)
    state_dim = len(game.get_state())
    action_dim = 2  # Jump or no jump
    agent = DQNAgent(state_dim, action_dim)
    agent.load_model("model.pth")

    scores = []
    avg_scores = []
    max_score = load_max_score()
    tloss = []
    agent.model.train()  # Set the model to training mode
    rewards = []

    for episode in range(num_episodes):
        state = game.reset()
        total_reward = 0

        if keyboard.is_pressed('q'):  # Check if 'q' is pressed during an episode
            print("Training interrupted by user.")
            break

        while True:
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed during an episode
                print("Training interrupted by user.")
                break
            action = agent.act(state)
            next_state, reward, done, score = game.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.experience_replay()

            state = next_state
            total_reward += reward
            rewards.append(reward)

            if done:
                if score > max_score:
                    max_score = score
                    save_max_score(max_score)
                    agent.save_model("model.pth")
                    print(f"Saved model with score: {max_score}")
                break
        
        agent.decay_epsilon()

        scores.append(score)
        avg_score = np.mean(scores[-100:])  # Compute average of last 100 scores
        avg_scores.append(avg_score)  # Append to list of average scores
        tloss.append(loss)
        print(
            f"Episode: {episode + 1}, Score: {score}, Total Reward: {total_reward}, max_score: {max_score}, loss: {loss}, epsilon: {agent.epsilon}"
        )

    plt.figure(figsize=(15, 10))
    plt.plot(scores, label="Score")
    plt.plot(avg_scores, label="Average Score")  # Plot average scores
    plt.plot(loss, label="Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Scores")

    plt.legend()

    plt.show()


def evaluate_dqn_agent(render=True):
    game = FlappyBirdGameAI(display_screen=render)
    state_dim = len(game.get_state())
    action_dim = 2  # Jump or no jump
    agent = DQNAgent(state_dim, action_dim)
    agent.load_model("model.pth")

    agent.model.eval()  # Set the model to evaluation mode

    scores = []
    num_episodes = 100  # Number of episodes for evaluation

    for episode in range(num_episodes):
        state = game.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, score = game.step(action)

            state = next_state
            total_reward += reward

            agent.decay_epsilon()

            if done:
                scores.append(score)
                print(
                    f"Episode: {episode + 1}, Score: {score}, Total Reward: {total_reward}"
                )
                break

    avg_score = np.mean(scores)
    print(f"Average Score over {num_episodes} episodes: {avg_score}")
