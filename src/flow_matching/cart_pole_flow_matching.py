import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random


class FlowMatchingNetwork(nn.Module):
    """
    Flow Matching network that learns to transform noise into actions
    conditioned on state and time
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Main network that takes concatenated embeddings and current x_t
        input_dim = hidden_dim // 4 + hidden_dim // 2 + action_dim
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t, t, state):
        """
        Forward pass for flow matching
        Args:
            x_t: current sample at time t [batch_size, action_dim]
            t: time parameter [batch_size, 1]
            state: environment state [batch_size, state_dim]
        Returns:
            velocity field v_t [batch_size, action_dim]
        """
        t_emb = self.time_embed(t)
        state_emb = self.state_embed(state)

        # Concatenate all inputs
        combined = torch.cat([x_t, t_emb, state_emb], dim=-1)

        return self.main_net(combined)


class FlowMatchingAgent:
    """
    Flow Matching agent for CartPole environment
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, device="cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Flow matching network
        self.flow_net = FlowMatchingNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.flow_net.parameters(), lr=lr)

        # Experience replay buffer
        self.buffer = deque(maxlen=10000)
        self.batch_size = 64

        # Training parameters
        self.sigma = 0.1  # Noise level

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample_trajectory(self, state, n_steps=50):
        """
        Sample action using flow matching by solving ODE
        """
        self.flow_net.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Start from noise
            x = torch.randn(1, self.action_dim).to(self.device) * self.sigma

            # Solve ODE using Euler method
            dt = 1.0 / n_steps

            for i in range(n_steps):
                t = torch.FloatTensor([[i * dt]]).to(self.device)

                # Get velocity from flow network
                v = self.flow_net(x, t, state_tensor)

                # Euler step
                x = x + v * dt

            # Convert to discrete action for CartPole (0 or 1)
            action_prob = torch.sigmoid(x[0, 0])
            action = 1 if action_prob > 0.5 else 0

        return action

    def train_step(self):
        """Single training step using flow matching loss"""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample batch from buffer
        batch = random.sample(self.buffer, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)

        # Convert discrete actions to continuous for flow matching
        # For CartPole: 0 -> -1, 1 -> 1
        continuous_actions = (actions * 2 - 1).unsqueeze(-1)

        # Weight actions by rewards (good actions should be more likely)
        weights = torch.sigmoid(rewards - rewards.mean())

        self.flow_net.train()

        # Flow matching training
        # Sample random times
        t = torch.rand(self.batch_size, 1).to(self.device)

        # Sample noise
        noise = torch.randn_like(continuous_actions) * self.sigma

        # Linear interpolation between noise and target action
        x_t = t * continuous_actions + (1 - t) * noise

        # Target velocity (derivative of interpolation)
        target_v = continuous_actions - noise

        # Predict velocity
        pred_v = self.flow_net(x_t, t, states)

        # Weighted MSE loss
        loss = (weights.unsqueeze(-1) * (pred_v - target_v) ** 2).mean()

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_flow_matching_agent():
    """Train the Flow Matching agent on CartPole"""

    # Environment setup
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # CartPole has discrete actions, but we'll use 1D continuous

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agent
    agent = FlowMatchingAgent(state_dim, action_dim, lr=1e-3, device=device)

    # Training parameters
    episodes = 500
    max_steps = 500

    # Tracking
    episode_rewards = []
    losses = []

    print("Starting training...")

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API

        episode_reward = 0
        episode_loss = 0
        steps = 0

        for step in range(max_steps):
            # Choose action
            if episode < 50:  # Initial random exploration
                action = env.action_space.sample()
            else:
                action = agent.sample_trajectory(state)

            # Take step
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            # Store experience
            agent.add_experience(state, action, reward, next_state, done)

            # Train
            if len(agent.buffer) >= agent.batch_size:
                loss = agent.train_step()
                episode_loss += loss
                steps += 1

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        if steps > 0:
            losses.append(episode_loss / steps)

        # Print progress
        if episode % 50 == 0:
            avg_reward = (
                np.mean(episode_rewards[-50:])
                if len(episode_rewards) >= 50
                else np.mean(episode_rewards)
            )
            avg_loss = (
                np.mean(losses[-50:])
                if len(losses) >= 50
                else (np.mean(losses) if losses else 0)
            )
            print(
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.6f}"
            )

    env.close()

    return agent, episode_rewards, losses


def test_agent(agent, num_episodes=10):
    """Test the trained agent"""
    env = gym.make("CartPole-v1")
    test_rewards = []

    print("\nTesting trained agent...")

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0

        for step in range(500):
            action = agent.sample_trajectory(state)
            result = env.step(action)

            if len(result) == 4:
                state, reward, done, _ = result
            else:
                state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            episode_reward += reward

            if done:
                break

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

    env.close()

    avg_test_reward = np.mean(test_rewards)
    print(f"\nAverage test reward: {avg_test_reward:.2f}")

    return test_rewards


def plot_results(episode_rewards, losses):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rewards
    ax1.plot(episode_rewards)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True)

    # Plot moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax1.plot(
            range(window - 1, len(episode_rewards)),
            moving_avg,
            "r-",
            label=f"{window}-episode moving average",
        )
        ax1.legend()

    # Plot losses
    if losses:
        ax2.plot(losses)
        ax2.set_title("Training Loss")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train the agent
    agent, episode_rewards, losses = train_flow_matching_agent()

    # Test the agent
    test_rewards = test_agent(agent)

    # Plot results
    plot_results(episode_rewards, losses)

    print(f"\nTraining completed!")
    print(
        f"Final training episodes average reward: {np.mean(episode_rewards[-50:]):.2f}"
    )
    print(f"Test episodes average reward: {np.mean(test_rewards):.2f}")
