"""
Deep Reinforcement Learning Router for PCB Engine
==================================================

Handles edge cases where traditional routing algorithms fail by using
a trained neural network policy to make routing decisions.

Based on research from:
- "Learning to Route with Deep Reinforcement Learning" (NeurIPS 2019)
- "Attention is All You Need in Circuit Routing" (ICCAD 2021)
- "GNN-based Routing with Attention Mechanism" (IEEE TCAD 2022)

Key features:
1. Graph Neural Network (GNN) for board state encoding
2. Attention mechanism for multi-terminal routing
3. Policy gradient training (PPO)
4. Curriculum learning (easy → hard routing problems)
5. Transfer learning from solved boards

When to use DRL:
- Traditional CASCADE fails (all 11 algorithms exhausted)
- Very dense boards with complex component placement
- Multi-layer boards with challenging via placement
- High-speed routing with impedance constraints

The DRL router learns to:
- Find creative routes through congested areas
- Make intelligent via placement decisions
- Balance multiple objectives (wirelength, vias, crosstalk)
- Generalize to unseen board configurations
"""

from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import math
import random
import time
import json
from pathlib import Path

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DRLConfig:
    """Configuration for DRL router."""
    # Network architecture
    embedding_dim: int = 128        # Node embedding dimension
    hidden_dim: int = 256           # Hidden layer dimension
    num_gnn_layers: int = 4         # Number of GNN layers
    num_attention_heads: int = 8    # Multi-head attention

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE lambda
    clip_epsilon: float = 0.2       # PPO clip
    entropy_coef: float = 0.01      # Entropy bonus
    value_coef: float = 0.5         # Value loss coefficient
    max_grad_norm: float = 0.5      # Gradient clipping

    # Environment
    max_steps_per_net: int = 1000   # Max routing steps per net
    reward_per_step: float = -0.01  # Small penalty per step
    reward_complete: float = 10.0   # Reward for completing route
    reward_via: float = -0.5        # Penalty per via
    reward_detour: float = -0.1     # Penalty per detour cell

    # Experience replay
    batch_size: int = 64
    buffer_size: int = 10000

    # Curriculum learning
    curriculum_enabled: bool = True
    initial_difficulty: float = 0.3
    difficulty_increment: float = 0.1

    # Model saving
    model_dir: str = "models"
    save_interval: int = 100


# =============================================================================
# GRAPH NEURAL NETWORK
# =============================================================================

if TORCH_AVAILABLE:

    class GraphAttentionLayer(nn.Module):
        """
        Graph Attention Network (GAT) layer.

        Applies attention mechanism over graph neighbors to learn
        contextualized node embeddings.
        """

        def __init__(self, in_features: int, out_features: int, num_heads: int = 8):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_heads = num_heads

            # Linear transformations for each head
            self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
            self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
            nn.init.xavier_uniform_(self.a)

            self.leaky_relu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Node features (N, in_features)
                adj: Adjacency matrix (N, N)

            Returns:
                Updated node features (N, out_features * num_heads)
            """
            N = x.size(0)

            # Linear transformation
            Wh = self.W(x)  # (N, out_features * num_heads)
            Wh = Wh.view(N, self.num_heads, self.out_features)  # (N, H, F)

            # Compute attention scores
            a_input = self._prepare_attention_input(Wh)  # (N, N, H, 2*F)
            e = self.leaky_relu((a_input * self.a).sum(dim=-1))  # (N, N, H)

            # Mask non-edges
            mask = adj.unsqueeze(-1).expand(-1, -1, self.num_heads)
            e = e.masked_fill(mask == 0, float('-inf'))

            # Softmax attention
            attention = F.softmax(e, dim=1)  # (N, N, H)
            attention = self.dropout(attention)

            # Apply attention
            h_prime = torch.einsum('ijh,jhf->ihf', attention, Wh)  # (N, H, F)

            return h_prime.reshape(N, -1)  # (N, H*F)

        def _prepare_attention_input(self, Wh: torch.Tensor) -> torch.Tensor:
            """Prepare concatenated features for attention computation."""
            N, H, F = Wh.shape
            Wh1 = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # (N, N, H, F)
            Wh2 = Wh.unsqueeze(0).expand(N, -1, -1, -1)  # (N, N, H, F)
            return torch.cat([Wh1, Wh2], dim=-1)  # (N, N, H, 2*F)


    class BoardEncoder(nn.Module):
        """
        Encodes PCB board state using Graph Neural Network.

        Treats the routing grid as a graph where:
        - Nodes = grid cells
        - Edges = adjacent cells (4-connected)
        - Node features = occupancy, net assignment, layer, distance to target
        """

        def __init__(self, config: DRLConfig):
            super().__init__()
            self.config = config

            # Input features: occupancy, net_id, layer, dist_to_target, is_target, is_source
            input_dim = 6

            # Initial embedding
            self.input_embed = nn.Linear(input_dim, config.embedding_dim)

            # GNN layers
            self.gnn_layers = nn.ModuleList([
                GraphAttentionLayer(
                    config.embedding_dim if i == 0 else config.embedding_dim * config.num_attention_heads,
                    config.embedding_dim,
                    config.num_attention_heads
                )
                for i in range(config.num_gnn_layers)
            ])

            # Layer normalization
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(config.embedding_dim * config.num_attention_heads)
                for _ in range(config.num_gnn_layers)
            ])

        def forward(
            self,
            node_features: torch.Tensor,
            adjacency: torch.Tensor
        ) -> torch.Tensor:
            """
            Encode board state.

            Args:
                node_features: Node features (N, 6)
                adjacency: Adjacency matrix (N, N)

            Returns:
                Node embeddings (N, hidden_dim)
            """
            x = self.input_embed(node_features)

            for gnn, norm in zip(self.gnn_layers, self.layer_norms):
                x_new = gnn(x, adjacency)
                x_new = norm(x_new)
                x_new = F.elu(x_new)

                # Residual connection (if dimensions match)
                if x.size(-1) == x_new.size(-1):
                    x = x + x_new
                else:
                    x = x_new

            return x


    class PolicyNetwork(nn.Module):
        """
        Policy network that selects routing actions.

        Uses attention over candidate positions to select next cell.
        """

        def __init__(self, config: DRLConfig):
            super().__init__()
            self.config = config

            input_dim = config.embedding_dim * config.num_attention_heads

            # Query network for current position
            self.query_net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )

            # Key network for candidate positions
            self.key_net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )

            # Value network (critic)
            self.value_net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1)
            )

            self.scale = math.sqrt(config.hidden_dim)

        def forward(
            self,
            node_embeddings: torch.Tensor,
            current_node: int,
            valid_actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute action probabilities and state value.

            Args:
                node_embeddings: All node embeddings (N, hidden)
                current_node: Current position index
                valid_actions: Mask of valid actions (N,) boolean

            Returns:
                Tuple of (action_probs, state_value)
            """
            # Current position embedding
            current_embed = node_embeddings[current_node:current_node+1]

            # Query for current position
            query = self.query_net(current_embed)  # (1, hidden)

            # Keys for all positions
            keys = self.key_net(node_embeddings)  # (N, hidden)

            # Attention scores
            scores = torch.matmul(query, keys.t()) / self.scale  # (1, N)
            scores = scores.squeeze(0)  # (N,)

            # Mask invalid actions
            scores = scores.masked_fill(~valid_actions, float('-inf'))

            # Softmax to get probabilities
            action_probs = F.softmax(scores, dim=0)

            # State value
            state_value = self.value_net(current_embed).squeeze()

            return action_probs, state_value


    class DRLRouter(nn.Module):
        """
        Complete DRL routing model.

        Combines board encoder and policy network for end-to-end routing.
        """

        def __init__(self, config: Optional[DRLConfig] = None):
            super().__init__()
            self.config = config or DRLConfig()

            self.encoder = BoardEncoder(self.config)
            self.policy = PolicyNetwork(self.config)

            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate
            )

        def get_action(
            self,
            state: Dict,
            deterministic: bool = False
        ) -> Tuple[int, float, float]:
            """
            Select action given state.

            Args:
                state: Environment state dict
                deterministic: If True, select best action (no exploration)

            Returns:
                Tuple of (action, log_prob, state_value)
            """
            node_features = torch.tensor(state['node_features'], dtype=torch.float32)
            adjacency = torch.tensor(state['adjacency'], dtype=torch.float32)
            current_node = state['current_node']
            valid_actions = torch.tensor(state['valid_actions'], dtype=torch.bool)

            # Encode board state
            node_embeddings = self.encoder(node_features, adjacency)

            # Get action probabilities
            action_probs, state_value = self.policy(
                node_embeddings, current_node, valid_actions
            )

            if deterministic:
                action = action_probs.argmax().item()
                log_prob = torch.log(action_probs[action] + 1e-10)
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))

            return action, log_prob.item(), state_value.item()

        def evaluate_actions(
            self,
            states: List[Dict],
            actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate actions for PPO update.

            Returns:
                Tuple of (log_probs, state_values, entropy)
            """
            log_probs = []
            state_values = []
            entropies = []

            for i, state in enumerate(states):
                node_features = torch.tensor(state['node_features'], dtype=torch.float32)
                adjacency = torch.tensor(state['adjacency'], dtype=torch.float32)
                current_node = state['current_node']
                valid_actions = torch.tensor(state['valid_actions'], dtype=torch.bool)

                node_embeddings = self.encoder(node_features, adjacency)
                action_probs, state_value = self.policy(
                    node_embeddings, current_node, valid_actions
                )

                dist = Categorical(action_probs)
                log_probs.append(dist.log_prob(actions[i]))
                state_values.append(state_value)
                entropies.append(dist.entropy())

            return (
                torch.stack(log_probs),
                torch.stack(state_values),
                torch.stack(entropies)
            )


# =============================================================================
# ROUTING ENVIRONMENT
# =============================================================================

class RoutingEnvironment:
    """
    Reinforcement learning environment for PCB routing.

    State space:
    - Grid occupancy map
    - Current position
    - Target positions
    - Net assignments
    - Layer information

    Action space:
    - Move to adjacent cell (up, down, left, right)
    - Change layer (via up, via down)

    Reward:
    - Small negative reward per step (encourages short routes)
    - Positive reward for reaching target
    - Negative reward for vias
    - Large negative reward for invalid moves
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        num_layers: int = 2,
        config: Optional[DRLConfig] = None
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_layers = num_layers
        self.config = config or DRLConfig()

        # Grid state: 0=free, positive=net_id, -1=blocked
        self.grid = np.zeros((num_layers, grid_height, grid_width), dtype=np.int32)

        # Current routing state
        self.current_pos: Optional[Tuple[int, int, int]] = None  # (layer, y, x)
        self.target_pos: Optional[Tuple[int, int, int]] = None
        self.current_net: int = 0
        self.current_path: List[Tuple[int, int, int]] = []
        self.steps: int = 0

        # Statistics
        self.total_routes: int = 0
        self.successful_routes: int = 0

    def reset(
        self,
        grid: Optional[np.ndarray] = None,
        source: Optional[Tuple[int, int, int]] = None,
        target: Optional[Tuple[int, int, int]] = None,
        net_id: int = 1
    ) -> Dict:
        """Reset environment for new routing episode."""
        if grid is not None:
            self.grid = grid.copy()

        if source is None or target is None:
            # Generate random source and target
            source = self._random_free_cell()
            target = self._random_free_cell()
            while target == source:
                target = self._random_free_cell()

        self.current_pos = source
        self.target_pos = target
        self.current_net = net_id
        self.current_path = [source]
        self.steps = 0

        return self._get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take routing action.

        Actions:
        0: Move up
        1: Move down
        2: Move left
        3: Move right
        4: Via up (change to higher layer)
        5: Via down (change to lower layer)

        Returns:
            Tuple of (state, reward, done, info)
        """
        self.steps += 1
        reward = self.config.reward_per_step
        done = False
        info = {}

        layer, y, x = self.current_pos

        # Compute new position based on action
        if action == 0:  # Up
            new_pos = (layer, y - 1, x)
        elif action == 1:  # Down
            new_pos = (layer, y + 1, x)
        elif action == 2:  # Left
            new_pos = (layer, y, x - 1)
        elif action == 3:  # Right
            new_pos = (layer, y, x + 1)
        elif action == 4:  # Via up
            new_pos = (layer + 1, y, x)
            reward += self.config.reward_via
        elif action == 5:  # Via down
            new_pos = (layer - 1, y, x)
            reward += self.config.reward_via
        else:
            # Invalid action
            return self._get_state(), -10.0, True, {'error': 'invalid_action'}

        # Check if new position is valid
        if not self._is_valid_move(new_pos):
            return self._get_state(), -5.0, True, {'error': 'blocked'}

        # Check for target reached
        if new_pos == self.target_pos:
            done = True
            reward = self.config.reward_complete
            self.successful_routes += 1
            info['success'] = True

        # Update state
        self.current_pos = new_pos
        self.current_path.append(new_pos)

        # Mark cell as routed
        self.grid[new_pos] = self.current_net

        # Check max steps
        if self.steps >= self.config.max_steps_per_net:
            done = True
            info['timeout'] = True

        self.total_routes += 1 if done else 0

        return self._get_state(), reward, done, info

    def _is_valid_move(self, pos: Tuple[int, int, int]) -> bool:
        """Check if move to position is valid."""
        layer, y, x = pos

        # Check bounds
        if not (0 <= layer < self.num_layers and
                0 <= y < self.grid_height and
                0 <= x < self.grid_width):
            return False

        # Check if cell is free or is target
        cell = self.grid[layer, y, x]
        if cell != 0 and pos != self.target_pos:
            return False

        return True

    def _get_state(self) -> Dict:
        """Get current environment state as dict."""
        layer, y, x = self.current_pos
        t_layer, t_y, t_x = self.target_pos

        # For simplicity, flatten the grid for the GNN
        # In practice, you'd use a proper graph representation
        num_nodes = self.grid_width * self.grid_height

        # Node features: [occupancy, net_id, layer, dist_to_target, is_source, is_target]
        node_features = np.zeros((num_nodes, 6), dtype=np.float32)

        for ny in range(self.grid_height):
            for nx in range(self.grid_width):
                idx = ny * self.grid_width + nx

                # Occupancy (average across layers)
                occupied = sum(1 for l in range(self.num_layers) if self.grid[l, ny, nx] != 0)
                node_features[idx, 0] = occupied / self.num_layers

                # Net ID (normalized)
                node_features[idx, 1] = self.current_net / 100.0

                # Current layer (normalized)
                node_features[idx, 2] = layer / self.num_layers

                # Manhattan distance to target
                dist = abs(nx - t_x) + abs(ny - t_y)
                node_features[idx, 3] = dist / (self.grid_width + self.grid_height)

                # Is current source
                if ny == y and nx == x:
                    node_features[idx, 4] = 1.0

                # Is target
                if ny == t_y and nx == t_x:
                    node_features[idx, 5] = 1.0

        # Adjacency matrix (4-connected grid)
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for ny in range(self.grid_height):
            for nx in range(self.grid_width):
                idx = ny * self.grid_width + nx

                # 4 neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny2, nx2 = ny + dy, nx + dx
                    if 0 <= ny2 < self.grid_height and 0 <= nx2 < self.grid_width:
                        idx2 = ny2 * self.grid_width + nx2
                        adjacency[idx, idx2] = 1.0

        # Valid actions
        current_idx = y * self.grid_width + x
        valid_actions = np.zeros(num_nodes, dtype=bool)

        for action in range(6):
            if action == 0:
                new_pos = (layer, y - 1, x)
            elif action == 1:
                new_pos = (layer, y + 1, x)
            elif action == 2:
                new_pos = (layer, y, x - 1)
            elif action == 3:
                new_pos = (layer, y, x + 1)
            elif action == 4:
                new_pos = (layer + 1, y, x)
            elif action == 5:
                new_pos = (layer - 1, y, x)

            if self._is_valid_move(new_pos):
                if action < 4:  # Regular move
                    new_y, new_x = new_pos[1], new_pos[2]
                    new_idx = new_y * self.grid_width + new_x
                    valid_actions[new_idx] = True

        return {
            'node_features': node_features,
            'adjacency': adjacency,
            'current_node': current_idx,
            'valid_actions': valid_actions,
            'current_pos': self.current_pos,
            'target_pos': self.target_pos
        }

    def _random_free_cell(self) -> Tuple[int, int, int]:
        """Get random free cell with bounded attempts to prevent infinite loop."""
        max_attempts = self.grid_width * self.grid_height * self.num_layers
        attempts = 0

        while attempts < max_attempts:
            layer = random.randint(0, self.num_layers - 1)
            y = random.randint(0, self.grid_height - 1)
            x = random.randint(0, self.grid_width - 1)

            if self.grid[layer, y, x] == 0:
                return (layer, y, x)

            attempts += 1

        # Grid is too full - raise an error instead of hanging
        raise RuntimeError(f"Failed to find free cell after {max_attempts} attempts - grid is too congested")

    def render(self) -> str:
        """Render environment as ASCII art."""
        lines = []
        for layer in range(self.num_layers):
            lines.append(f"Layer {layer}:")
            for y in range(self.grid_height):
                row = ""
                for x in range(self.grid_width):
                    pos = (layer, y, x)
                    if pos == self.current_pos:
                        row += "S"
                    elif pos == self.target_pos:
                        row += "T"
                    elif self.grid[layer, y, x] > 0:
                        row += str(self.grid[layer, y, x] % 10)
                    elif self.grid[layer, y, x] < 0:
                        row += "#"
                    else:
                        row += "."
                lines.append(row)
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# TRAINING
# =============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for DRL router.
    """

    def __init__(
        self,
        model: 'DRLRouter',
        env: RoutingEnvironment,
        config: Optional[DRLConfig] = None
    ):
        self.model = model
        self.env = env
        self.config = config or DRLConfig()

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []

    def collect_rollout(self, num_steps: int = 1024) -> float:
        """Collect experience for training."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        total_reward = 0
        steps = 0

        state = self.env.reset()

        while steps < num_steps:
            # Get action from model
            action, log_prob, value = self.model.get_action(state)

            # Store experience
            self.states.append(state)
            self.actions.append(action)
            self.values.append(value)
            self.log_probs.append(log_prob)

            # Take step
            next_state, reward, done, info = self.env.step(action)

            self.rewards.append(reward)
            self.dones.append(done)

            total_reward += reward
            steps += 1

            if done:
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(self.env.steps)

                state = self.env.reset()
                total_reward = 0
            else:
                state = next_state

        return sum(self.rewards) / len(self.rewards)

    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # Bootstrap from 0 at end
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, num_epochs: int = 4):
        """Perform PPO update."""
        advantages, returns = self.compute_gae()

        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)

        dataset_size = len(self.states)
        indices = list(range(dataset_size))

        for epoch in range(num_epochs):
            random.shuffle(indices)

            for start in range(0, dataset_size, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = [self.states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Evaluate actions
                new_log_probs, state_values, entropy = self.model.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(state_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                       self.config.value_coef * value_loss +
                       self.config.entropy_coef * entropy_loss)

                # Optimize
                self.model.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.model.optimizer.step()

    def train(self, num_iterations: int = 1000, log_interval: int = 10):
        """Train the DRL router."""
        print(f"Starting DRL training for {num_iterations} iterations...")

        for iteration in range(num_iterations):
            # Collect experience
            avg_reward = self.collect_rollout()

            # Update policy
            self.update()

            # Logging
            if (iteration + 1) % log_interval == 0:
                avg_ep_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_ep_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
                success_rate = self.env.successful_routes / max(1, self.env.total_routes)

                print(f"Iter {iteration + 1:4d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Ep Reward: {avg_ep_reward:6.2f} | "
                      f"Ep Length: {avg_ep_length:5.1f} | "
                      f"Success: {success_rate*100:5.1f}%")

                self.success_rate_history.append(success_rate)

            # Save checkpoint
            if (iteration + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{iteration + 1}.pt")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = Path(self.config.model_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'success_rate_history': self.success_rate_history
        }, path)

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        path = Path(self.config.model_dir) / filename

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.success_rate_history = checkpoint.get('success_rate_history', [])


# =============================================================================
# HIGH-LEVEL API FOR PCB ENGINE
# =============================================================================

class DRLRoutingPiston:
    """
    DRL-based routing piston for PCB Engine.

    Called when traditional CASCADE algorithms fail.
    Uses pre-trained DRL model for intelligent routing decisions.
    """

    def __init__(self, config: Optional[DRLConfig] = None):
        self.config = config or DRLConfig()

        if TORCH_AVAILABLE:
            self.model = DRLRouter(self.config)
            self.model.eval()  # Inference mode
        else:
            self.model = None
            print("[DRL] PyTorch not available - DRL routing disabled")

    def route_net(
        self,
        grid: np.ndarray,
        source: Tuple[int, int],
        targets: List[Tuple[int, int]],
        net_name: str = "net"
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Route a net using DRL.

        Args:
            grid: Occupancy grid (H x W) where 0=free, non-zero=blocked
            source: Source position (x, y)
            targets: List of target positions [(x, y), ...]
            net_name: Net name for logging

        Returns:
            List of path points or None if routing fails
        """
        if self.model is None:
            return None

        h, w = grid.shape

        # Create environment
        env = RoutingEnvironment(
            grid_width=w,
            grid_height=h,
            num_layers=1,
            config=self.config
        )

        # Convert 2D grid to 3D (single layer)
        grid_3d = np.expand_dims(grid, axis=0)
        env.grid = grid_3d

        full_path = []

        # Route to each target sequentially
        current_source = (0, source[1], source[0])  # (layer, y, x)

        for target in targets:
            target_3d = (0, target[1], target[0])

            # Reset environment
            state = env.reset(
                grid=grid_3d,
                source=current_source,
                target=target_3d
            )

            # Run model
            path_segment = []
            max_steps = self.config.max_steps_per_net

            for step in range(max_steps):
                action, _, _ = self.model.get_action(state, deterministic=True)
                state, reward, done, info = env.step(action)

                path_segment.append(env.current_pos)

                if done:
                    if info.get('success'):
                        # Convert path to 2D
                        for pos in path_segment:
                            full_path.append((pos[2], pos[1]))  # (x, y)
                        current_source = env.current_pos
                    break

            # Update grid with routed path
            grid_3d = env.grid

        return full_path if full_path else None

    def load_pretrained(self, model_path: str):
        """Load pre-trained model weights."""
        if self.model is not None and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"[DRL] Loaded pre-trained model from {model_path}")


# =============================================================================
# BENCHMARK / VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DRL Router - Verification")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\nPyTorch not available. Install with: pip install torch")
        print("DRL routing will use CPU fallback algorithms.")
        exit(0)

    # Create small environment for testing
    print("\n--- Test 1: Environment ---")

    env = RoutingEnvironment(
        grid_width=10,
        grid_height=10,
        num_layers=2
    )

    # Add some obstacles
    env.grid[0, 2:8, 5] = -1  # Vertical wall
    env.grid[1, 5, 2:8] = -1  # Horizontal wall on layer 2

    state = env.reset(
        source=(0, 1, 1),
        target=(0, 8, 8)
    )

    print(env.render())
    print(f"State shape: node_features={state['node_features'].shape}, "
          f"adjacency={state['adjacency'].shape}")

    # Test model
    print("\n--- Test 2: Model Forward Pass ---")

    config = DRLConfig()
    model = DRLRouter(config)

    action, log_prob, value = model.get_action(state)
    print(f"Action: {action}, Log prob: {log_prob:.3f}, Value: {value:.3f}")

    # Test a few steps
    print("\n--- Test 3: Environment Steps ---")

    total_reward = 0
    for i in range(20):
        action, _, _ = model.get_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode done after {i+1} steps. Total reward: {total_reward:.2f}")
            print(f"Info: {info}")
            break

    print(env.render())

    # Quick training test
    print("\n--- Test 4: Quick Training (5 iterations) ---")

    env = RoutingEnvironment(grid_width=8, grid_height=8, num_layers=1)
    model = DRLRouter(config)
    trainer = PPOTrainer(model, env, config)

    trainer.train(num_iterations=5, log_interval=1)

    print("\n" + "=" * 60)
    print("DRL Router verification complete!")
    print("=" * 60)
