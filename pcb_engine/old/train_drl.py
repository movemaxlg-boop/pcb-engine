#!/usr/bin/env python3
"""
Train the DRL Router for PCB Routing

This script trains the Deep Reinforcement Learning model to learn
optimal routing strategies for PCB design.

Training scenarios include:
1. Simple 2-point routing
2. Multi-terminal nets (Steiner-like)
3. Congested areas with obstacles
4. Layer transitions (via planning)
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

from pcb_engine.drl_router import (
    DRLConfig, DRLRouter, RoutingEnvironment, PPOTrainer
)


def create_training_scenarios():
    """
    Create a diverse set of training scenarios.

    Returns list of (grid_width, grid_height, obstacles, nets) tuples.
    """
    scenarios = []

    # Scenario 1: Simple open board routing
    # Good for learning basic pathfinding
    for _ in range(5):
        w, h = 32, 32
        obstacles = []
        # Add random obstacles (10-20% coverage)
        num_obs = np.random.randint(10, 30)
        for _ in range(num_obs):
            ox = np.random.randint(2, w-2)
            oy = np.random.randint(2, h-2)
            ow = np.random.randint(2, 5)
            oh = np.random.randint(2, 5)
            obstacles.append((ox, oy, ow, oh))

        # Random nets (2-4 pins each)
        nets = []
        for _ in range(np.random.randint(3, 8)):
            num_pins = np.random.randint(2, 5)
            pins = []
            for _ in range(num_pins):
                px = np.random.randint(1, w-1)
                py = np.random.randint(1, h-1)
                pins.append((px, py))
            nets.append(pins)

        scenarios.append((w, h, obstacles, nets))

    # Scenario 2: Dense component area (BGA-like)
    for _ in range(5):
        w, h = 48, 48
        obstacles = []

        # Create a grid of component pads (BGA pattern)
        pad_spacing = 4
        for px in range(8, 40, pad_spacing):
            for py in range(8, 40, pad_spacing):
                obstacles.append((px, py, 2, 2))

        # Nets connecting BGA pins to edge
        nets = []
        edge_points = [(2, y) for y in range(4, 44, 4)]  # Left edge
        bga_points = [(px, py) for px in range(8, 40, 8) for py in range(8, 40, 8)]

        for i in range(min(len(edge_points), len(bga_points))):
            nets.append([bga_points[i], edge_points[i]])

        scenarios.append((w, h, obstacles, nets))

    # Scenario 3: Channel routing (parallel tracks)
    for _ in range(5):
        w, h = 64, 32
        obstacles = []

        # Two rows of components with channel between
        for x in range(4, 60, 6):
            obstacles.append((x, 4, 4, 3))   # Top row
            obstacles.append((x, 25, 4, 3))  # Bottom row

        # Nets crossing the channel
        nets = []
        for i in range(8):
            top_x = 4 + i * 6
            bot_x = 4 + (7 - i) * 6  # Crossing pattern
            nets.append([(top_x, 8), (bot_x, 22)])

        scenarios.append((w, h, obstacles, nets))

    # Scenario 4: Via planning (multi-layer)
    # Represented as congested single layer with "via points"
    for _ in range(5):
        w, h = 40, 40
        obstacles = []

        # Create blocking pattern forcing layer changes
        for y in range(0, 40, 10):
            obstacles.append((0, y, 35, 2))  # Horizontal barriers

        # Leave via holes
        for x in range(10, 40, 10):
            for y in range(0, 40, 10):
                # Clear area around via point
                pass  # Obstacles already have gaps

        # Nets that must use "vias" (go around barriers)
        nets = []
        for i in range(5):
            nets.append([(5, 5 + i*8), (35, 35 - i*8)])

        scenarios.append((w, h, obstacles, nets))

    # Scenario 5: Mixed complexity (realistic PCB)
    for _ in range(10):
        w, h = 50, 40
        obstacles = []

        # Random component placement
        num_components = np.random.randint(8, 15)
        for _ in range(num_components):
            cx = np.random.randint(5, w-10)
            cy = np.random.randint(5, h-10)
            cw = np.random.choice([3, 4, 5, 6, 8])
            ch = np.random.choice([3, 4, 5, 6, 8])
            obstacles.append((cx, cy, cw, ch))

        # Random nets with varying complexity
        nets = []
        for _ in range(np.random.randint(5, 12)):
            num_pins = np.random.choice([2, 2, 2, 3, 3, 4])  # Weighted towards 2-pin
            pins = []
            for _ in range(num_pins):
                px = np.random.randint(2, w-2)
                py = np.random.randint(2, h-2)
                pins.append((px, py))
            nets.append(pins)

        scenarios.append((w, h, obstacles, nets))

    return scenarios


def train_drl_router(
    num_iterations: int = 500,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    model_dir: str = None
):
    """
    Train the DRL router.

    Args:
        num_iterations: Number of training iterations
        batch_size: Batch size for PPO updates
        learning_rate: Learning rate
        model_dir: Directory to save checkpoints
    """
    if model_dir is None:
        model_dir = str(Path(__file__).parent / "models" / "drl_router")

    print("=" * 70)
    print("DRL ROUTER TRAINING")
    print("=" * 70)
    print(f"Iterations: {num_iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Model directory: {model_dir}")
    print()

    # Create configuration
    config = DRLConfig(
        hidden_dim=128,
        num_attention_heads=4,
        num_gnn_layers=3,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        save_interval=50,
        model_dir=model_dir
    )

    # Create training scenarios
    print("Creating training scenarios...")
    scenarios = create_training_scenarios()
    print(f"Created {len(scenarios)} training scenarios")

    # Create environment with first scenario
    w, h, obstacles, nets = scenarios[0]
    num_layers = 2
    env = RoutingEnvironment(
        grid_width=w,
        grid_height=h,
        num_layers=num_layers,
        config=config
    )

    # Add obstacles to environment
    for ox, oy, ow, oh in obstacles:
        for dx in range(ow):
            for dy in range(oh):
                x, y = ox + dx, oy + dy
                if 0 <= x < w and 0 <= y < h:
                    env.grid[0, y, x] = -1  # Mark as obstacle

    # Create model and trainer
    print("Initializing DRL model and trainer...")
    model = DRLRouter(config)
    trainer = PPOTrainer(model, env, config)

    # Check for existing checkpoint
    checkpoint_path = Path(model_dir) / "latest.pt"
    if checkpoint_path.exists():
        print(f"Loading existing checkpoint: {checkpoint_path}")
        trainer.load_checkpoint("latest.pt")

    # Training loop with scenario rotation
    print("\nStarting training...")
    print("-" * 70)

    start_time = time.time()
    scenario_idx = 0

    for iteration in range(num_iterations):
        # Rotate scenarios every 10 iterations
        if iteration % 10 == 0 and iteration > 0:
            scenario_idx = (scenario_idx + 1) % len(scenarios)
            w, h, obstacles, nets = scenarios[scenario_idx]

            # Reset environment with new scenario
            env.grid_width = w
            env.grid_height = h
            env.grid = np.zeros((num_layers, h, w), dtype=np.int32)

            # Add obstacles
            for ox, oy, ow, oh in obstacles:
                for dx in range(ow):
                    for dy in range(oh):
                        x, y = ox + dx, oy + dy
                        if 0 <= x < w and 0 <= y < h:
                            env.grid[0, y, x] = -1  # Blocked

        # Collect experience
        avg_reward = trainer.collect_rollout()

        # Update policy
        trainer.update()

        # Logging
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_ep_reward = np.mean(trainer.episode_rewards[-100:]) if trainer.episode_rewards else 0
            avg_ep_length = np.mean(trainer.episode_lengths[-100:]) if trainer.episode_lengths else 0
            success_rate = env.successful_routes / max(1, env.total_routes)

            print(f"Iter {iteration + 1:4d}/{num_iterations} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Ep Reward: {avg_ep_reward:7.2f} | "
                  f"Ep Len: {avg_ep_length:5.1f} | "
                  f"Success: {success_rate*100:5.1f}% | "
                  f"Time: {elapsed:.1f}s")

        # Save checkpoint
        if (iteration + 1) % 100 == 0:
            trainer.save_checkpoint(f"checkpoint_{iteration + 1}.pt")
            trainer.save_checkpoint("latest.pt")
            print(f"  [Checkpoint saved]")

    # Final save
    trainer.save_checkpoint("final.pt")
    trainer.save_checkpoint("latest.pt")

    total_time = time.time() - start_time
    final_success = env.successful_routes / max(1, env.total_routes)

    print("-" * 70)
    print(f"Training complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Final success rate: {final_success*100:.1f}%")
    print(f"Model saved to: {model_dir}")
    print("=" * 70)

    return trainer


def test_trained_model(model_path: str = None):
    """Test a trained DRL model on sample routing tasks."""
    if model_path is None:
        model_path = str(Path(__file__).parent / "models" / "drl_router" / "latest.pt")

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train a model first with: python train_drl.py")
        return

    print("=" * 70)
    print("TESTING TRAINED DRL MODEL")
    print("=" * 70)

    config = DRLConfig()

    # Create test environment
    env = RoutingEnvironment(
        grid_width=32,
        grid_height=32,
        num_layers=1,
        config=config
    )

    # Add some obstacles
    for x in range(10, 22):
        env.grid[0, 15, x] = 1  # Horizontal barrier

    # Create model and load weights
    model = DRLRouter(config)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test routing
    test_cases = [
        ((5, 5), (27, 27)),    # Diagonal, must go around barrier
        ((5, 10), (27, 10)),   # Horizontal through barrier
        ((16, 5), (16, 27)),   # Vertical through barrier
    ]

    successes = 0
    for i, (source, target) in enumerate(test_cases):
        env.source = source
        env.targets = [target]
        state = env.reset()

        path = [source]
        done = False
        steps = 0
        max_steps = 200

        while not done and steps < max_steps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = model(state_tensor, None)
                action = torch.argmax(action_probs).item()

            state, reward, done, info = env.step(action)
            path.append(env.current_pos)
            steps += 1

        success = info.get('reached_target', False)
        if success:
            successes += 1

        print(f"\nTest {i+1}: {source} -> {target}")
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"  Steps: {steps}")
        print(f"  Path length: {len(path)}")

    print(f"\n{'=' * 70}")
    print(f"Test Results: {successes}/{len(test_cases)} successful")
    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train DRL Router for PCB routing')
    parser.add_argument('--iterations', '-n', type=int, default=500,
                       help='Number of training iterations (default: 500)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test a trained model instead of training')
    parser.add_argument('--model-dir', '-m', type=str, default=None,
                       help='Model directory')

    args = parser.parse_args()

    if args.test:
        model_path = None
        if args.model_dir:
            model_path = str(Path(args.model_dir) / "latest.pt")
        test_trained_model(model_path)
    else:
        train_drl_router(
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_dir=args.model_dir
        )
