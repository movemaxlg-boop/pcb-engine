#!/usr/bin/env python3
"""
Simple DRL Router Training - Fast Version

Uses a simplified CNN-based approach instead of the full GNN
for faster training on CPU.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)


class SimpleCNNRouter(nn.Module):
    """
    Simple CNN-based router for fast training.
    Uses convolutional layers instead of GNN for speed.
    """

    def __init__(self, grid_size=32, num_actions=6):
        super().__init__()

        self.grid_size = grid_size
        self.num_actions = num_actions

        # CNN encoder - 3 input channels: obstacles, current, target
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Shared feature layer
        self.fc_shared = nn.Linear(64 * 4 * 4, 256)

        # Policy head
        self.fc_policy = nn.Linear(256, num_actions)

        # Value head
        self.fc_value = nn.Linear(256, 1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        """Forward pass through network."""
        # x: (batch, 3, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        shared = F.relu(self.fc_shared(x))

        # Policy: action probabilities
        policy_logits = self.fc_policy(shared)

        # Value: state value estimate
        value = self.fc_value(shared)

        return policy_logits, value

    def get_action(self, state, deterministic=False):
        """Select action given state."""
        with torch.no_grad():
            logits, value = self.forward(state.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = probs.argmax().item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()

            log_prob = torch.log(probs[0, action] + 1e-10).item()

        return action, log_prob, value.item()


class SimpleRoutingEnv:
    """
    Simple routing environment with fast state computation.
    """

    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Positions
        self.current_pos = (0, 0)
        self.target_pos = (grid_size-1, grid_size-1)

        # Stats
        self.steps = 0
        self.max_steps = grid_size * 4
        self.total_routes = 0
        self.successful_routes = 0

    def add_random_obstacles(self, density=0.2):
        """Add random obstacles to the grid."""
        self.grid = np.random.choice(
            [0.0, 1.0],
            size=(self.grid_size, self.grid_size),
            p=[1-density, density]
        ).astype(np.float32)

        # Clear start and end positions
        sy, sx = self.current_pos
        ty, tx = self.target_pos
        self.grid[sy, sx] = 0
        self.grid[ty, tx] = 0

        # Clear some path cells
        for _ in range(self.grid_size):
            ry = np.random.randint(0, self.grid_size)
            rx = np.random.randint(0, self.grid_size)
            self.grid[ry, rx] = 0

    def reset(self, random_positions=True, add_obstacles=True):
        """Reset environment for new episode."""
        self.steps = 0
        self.total_routes += 1

        if random_positions:
            # Random start and target
            self.current_pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            self.target_pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            while self.target_pos == self.current_pos:
                self.target_pos = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )

        if add_obstacles:
            self.add_random_obstacles(density=0.15)

        return self.get_state()

    def get_state(self):
        """Get state as 3-channel tensor for CNN."""
        state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: obstacles
        state[0] = self.grid

        # Channel 1: current position (one-hot)
        cy, cx = self.current_pos
        state[1, cy, cx] = 1.0

        # Channel 2: target position (one-hot)
        ty, tx = self.target_pos
        state[2, ty, tx] = 1.0

        return torch.tensor(state)

    def step(self, action):
        """Take routing step."""
        self.steps += 1
        cy, cx = self.current_pos

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay, 5=via (treated as stay)
        if action == 0 and cy > 0:
            new_y, new_x = cy - 1, cx
        elif action == 1 and cy < self.grid_size - 1:
            new_y, new_x = cy + 1, cx
        elif action == 2 and cx > 0:
            new_y, new_x = cy, cx - 1
        elif action == 3 and cx < self.grid_size - 1:
            new_y, new_x = cy, cx + 1
        else:
            new_y, new_x = cy, cx  # Invalid or stay

        # Check if new position is blocked
        if self.grid[new_y, new_x] > 0.5:
            # Hit obstacle
            reward = -1.0
            done = False
            info = {'hit_obstacle': True}
            # Stay in place
        else:
            # Move to new position
            self.current_pos = (new_y, new_x)

            # Calculate reward
            ty, tx = self.target_pos

            if self.current_pos == self.target_pos:
                # Reached target!
                reward = 10.0
                done = True
                self.successful_routes += 1
                info = {'reached_target': True}
            else:
                # Small step penalty based on distance to target
                old_dist = abs(cy - ty) + abs(cx - tx)
                new_dist = abs(new_y - ty) + abs(new_x - tx)

                if new_dist < old_dist:
                    reward = 0.1  # Moving closer
                elif new_dist > old_dist:
                    reward = -0.2  # Moving away
                else:
                    reward = -0.05  # Lateral move

                done = self.steps >= self.max_steps
                info = {'reached_target': False}

        return self.get_state(), reward, done, info


class SimplePPOTrainer:
    """Simple PPO trainer for fast CPU training."""

    def __init__(self, model, env, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2):
        self.model = model
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Stats
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollout(self, num_steps=256):
        """Collect training experience."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        state = self.env.reset()
        episode_reward = 0

        for _ in range(num_steps):
            action, log_prob, value = self.model.get_action(state)

            self.states.append(state)
            self.actions.append(action)
            self.values.append(value)
            self.log_probs.append(log_prob)

            next_state, reward, done, info = self.env.step(action)

            self.rewards.append(reward)
            self.dones.append(done)

            episode_reward += reward

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(self.env.steps)
                state = self.env.reset()
                episode_reward = 0
            else:
                state = next_state

        return np.mean(self.rewards)

    def compute_gae(self):
        """Compute Generalized Advantage Estimation."""
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, num_epochs=4, batch_size=64):
        """Perform PPO update."""
        advantages, returns = self.compute_gae()

        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)

        for _ in range(num_epochs):
            # Shuffle indices
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                logits, values = self.model.forward(batch_states)
                probs = F.softmax(logits, dim=-1)

                # New log probs
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Update
                self.model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.model.optimizer.step()


def train_simple_drl(num_iterations=200, grid_size=16, save_dir=None):
    """Train the simple DRL router."""
    if save_dir is None:
        save_dir = Path(__file__).parent / "models" / "drl_simple"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIMPLE DRL ROUTER TRAINING")
    print("=" * 60)
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Iterations: {num_iterations}")
    print(f"Save directory: {save_dir}")
    print()

    # Create model and environment
    model = SimpleCNNRouter(grid_size=grid_size)
    env = SimpleRoutingEnv(grid_size=grid_size)
    trainer = SimplePPOTrainer(model, env)

    print("Starting training...")
    print("-" * 60)

    start_time = time.time()

    for iteration in range(num_iterations):
        # Collect experience
        avg_reward = trainer.collect_rollout(num_steps=256)

        # Update policy
        trainer.update()

        # Log progress
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time

            ep_rewards = trainer.episode_rewards[-50:]
            ep_lengths = trainer.episode_lengths[-50:]

            avg_ep_reward = np.mean(ep_rewards) if ep_rewards else 0
            avg_ep_length = np.mean(ep_lengths) if ep_lengths else 0
            success_rate = env.successful_routes / max(1, env.total_routes)

            print(f"Iter {iteration + 1:4d} | "
                  f"Reward: {avg_reward:6.3f} | "
                  f"Ep Reward: {avg_ep_reward:6.2f} | "
                  f"Ep Len: {avg_ep_length:5.1f} | "
                  f"Success: {success_rate*100:5.1f}% | "
                  f"Time: {elapsed:.1f}s")

        # Save checkpoint
        if (iteration + 1) % 50 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': iteration + 1,
                'episode_rewards': trainer.episode_rewards,
                'success_rate': env.successful_routes / max(1, env.total_routes)
            }, save_dir / f"checkpoint_{iteration + 1}.pt")
            print(f"  [Checkpoint saved]")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'iteration': num_iterations,
        'episode_rewards': trainer.episode_rewards,
        'success_rate': env.successful_routes / max(1, env.total_routes)
    }, save_dir / "final.pt")

    total_time = time.time() - start_time
    final_success = env.successful_routes / max(1, env.total_routes)

    print("-" * 60)
    print(f"Training complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Final success rate: {final_success*100:.1f}%")
    print(f"Model saved to: {save_dir}")
    print("=" * 60)

    return model, trainer


def test_trained_model(model_path=None, num_tests=10):
    """Test a trained model."""
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "drl_simple" / "final.pt"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    print("=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(model_path, weights_only=False)
    model = SimpleCNNRouter(grid_size=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from iteration {checkpoint.get('iteration', '?')}")
    print(f"Training success rate: {checkpoint.get('success_rate', 0)*100:.1f}%")
    print()

    # Test environment
    env = SimpleRoutingEnv(grid_size=16)

    successes = 0
    total_steps = 0

    for test in range(num_tests):
        state = env.reset()
        done = False
        path = [env.current_pos]

        while not done and len(path) < 100:
            action, _, _ = model.get_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            path.append(env.current_pos)

        success = info.get('reached_target', False)
        if success:
            successes += 1
            total_steps += len(path)

        status = "SUCCESS" if success else "FAILED"
        print(f"Test {test+1:2d}: {path[0]} -> {env.target_pos} | "
              f"{status:7s} | Steps: {len(path)}")

    print()
    print(f"Results: {successes}/{num_tests} successful")
    if successes > 0:
        print(f"Average path length: {total_steps/successes:.1f}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple DRL Router Training')
    parser.add_argument('--iterations', '-n', type=int, default=200,
                       help='Number of training iterations')
    parser.add_argument('--grid-size', '-g', type=int, default=16,
                       help='Grid size')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test trained model')

    args = parser.parse_args()

    if args.test:
        test_trained_model()
    else:
        train_simple_drl(
            num_iterations=args.iterations,
            grid_size=args.grid_size
        )
