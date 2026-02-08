"""
Cascade Optimizer - Dynamic Algorithm Priority Ordering
========================================================

This module provides intelligent, adaptive algorithm ordering for the PCB Engine's
CASCADE system. Instead of using fixed, hardcoded priority orders, it learns from
success/failure history and adapts based on design profiles.

KEY FEATURES:
1. Design Profile Classification - Categorize designs by complexity, density, etc.
2. Success Rate Tracking - Track which algorithms work best for which profiles
3. Dynamic Reordering - Promote algorithms that succeed, demote those that fail
4. User Configuration - Allow users to set preferred algorithms
5. Persistence - Save learned priorities across sessions

HIERARCHY:
    USER (Boss) → Circuit AI (Engineer) → PCB Engine (Foreman) → CASCADE Optimizer

USAGE:
    optimizer = CascadeOptimizer()
    optimizer.load()  # Load learned history

    # Get optimized order for a design
    algorithms = optimizer.get_cascade('routing', design_profile)

    # After execution, record result
    optimizer.record_result('routing', 'hybrid', design_profile, success=True, metrics={})
    optimizer.save()  # Persist learning
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
from collections import defaultdict
import json
import os
import time
import math


# =============================================================================
# DESIGN PROFILE CLASSIFICATION
# =============================================================================

class DesignComplexity(Enum):
    """Design complexity levels"""
    SIMPLE = 'simple'           # < 20 components, < 50 nets
    MODERATE = 'moderate'       # 20-100 components, 50-200 nets
    COMPLEX = 'complex'         # 100-500 components, 200-1000 nets
    VERY_COMPLEX = 'very_complex'  # > 500 components, > 1000 nets


class DesignDensity(Enum):
    """Component density levels"""
    LOW = 'low'                 # > 5mm average spacing
    MEDIUM = 'medium'           # 2-5mm average spacing
    HIGH = 'high'               # 1-2mm average spacing
    ULTRA_HIGH = 'ultra_high'   # < 1mm average spacing (BGA/QFN heavy)


class DesignType(Enum):
    """Primary design type"""
    DIGITAL = 'digital'         # MCU, FPGA, logic
    ANALOG = 'analog'           # Op-amps, sensors
    MIXED_SIGNAL = 'mixed'      # Both analog and digital
    POWER = 'power'             # Power supplies, motor drivers
    RF = 'rf'                   # High frequency, antenna
    HIGH_SPEED = 'high_speed'   # DDR, USB3, PCIe


class LayerCount(Enum):
    """Board layer count"""
    TWO_LAYER = 2
    FOUR_LAYER = 4
    SIX_LAYER = 6
    EIGHT_PLUS = 8


@dataclass
class DesignProfile:
    """
    Characterizes a PCB design for algorithm selection.

    The optimizer uses this profile to determine which algorithms
    are most likely to succeed for a given design.
    """
    complexity: DesignComplexity = DesignComplexity.MODERATE
    density: DesignDensity = DesignDensity.MEDIUM
    design_type: DesignType = DesignType.DIGITAL
    layer_count: LayerCount = LayerCount.TWO_LAYER

    # Specific characteristics
    component_count: int = 0
    net_count: int = 0
    bga_count: int = 0
    qfn_count: int = 0
    power_net_count: int = 0
    high_speed_net_count: int = 0
    differential_pair_count: int = 0

    # Board dimensions
    board_width: float = 100.0
    board_height: float = 100.0
    board_area: float = 0.0  # Calculated

    # Derived metrics
    components_per_cm2: float = 0.0
    nets_per_component: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics"""
        self.board_area = self.board_width * self.board_height / 100  # cm²
        if self.board_area > 0:
            self.components_per_cm2 = self.component_count / self.board_area
        if self.component_count > 0:
            self.nets_per_component = self.net_count / self.component_count

    def to_key(self) -> str:
        """Generate a hashable key for this profile type"""
        return f"{self.complexity.value}_{self.density.value}_{self.design_type.value}_{self.layer_count.value}"

    @classmethod
    def from_parts_db(cls, parts_db: Dict, board_width: float = 100.0,
                      board_height: float = 100.0) -> 'DesignProfile':
        """
        Create a DesignProfile from a parts database.

        Args:
            parts_db: Parts database with 'parts' and 'nets' keys
            board_width: Board width in mm
            board_height: Board height in mm

        Returns:
            DesignProfile characterizing this design
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        component_count = len(parts)
        net_count = len(nets)

        # Count special packages
        bga_count = 0
        qfn_count = 0
        for ref, part in parts.items():
            footprint = part.get('footprint', '').upper()
            if 'BGA' in footprint:
                bga_count += 1
            elif 'QFN' in footprint or 'DFN' in footprint:
                qfn_count += 1

        # Count power nets
        power_net_count = 0
        for net_name in nets.keys():
            name_upper = net_name.upper()
            if any(p in name_upper for p in ['VCC', 'VDD', 'GND', 'VSS', 'PWR', '3V3', '5V', '12V']):
                power_net_count += 1

        # Determine complexity
        if component_count < 20 and net_count < 50:
            complexity = DesignComplexity.SIMPLE
        elif component_count < 100 and net_count < 200:
            complexity = DesignComplexity.MODERATE
        elif component_count < 500 and net_count < 1000:
            complexity = DesignComplexity.COMPLEX
        else:
            complexity = DesignComplexity.VERY_COMPLEX

        # Determine density
        board_area = board_width * board_height / 100  # cm²
        if board_area > 0:
            density_ratio = component_count / board_area
            if density_ratio < 2:
                density = DesignDensity.LOW
            elif density_ratio < 5:
                density = DesignDensity.MEDIUM
            elif density_ratio < 10:
                density = DesignDensity.HIGH
            else:
                density = DesignDensity.ULTRA_HIGH
        else:
            density = DesignDensity.MEDIUM

        # Determine type based on characteristics
        if bga_count > 0 or qfn_count > 3:
            design_type = DesignType.HIGH_SPEED
        elif power_net_count > net_count * 0.3:
            design_type = DesignType.POWER
        else:
            design_type = DesignType.DIGITAL

        return cls(
            complexity=complexity,
            density=density,
            design_type=design_type,
            layer_count=LayerCount.TWO_LAYER,  # Default, can be overridden
            component_count=component_count,
            net_count=net_count,
            bga_count=bga_count,
            qfn_count=qfn_count,
            power_net_count=power_net_count,
            board_width=board_width,
            board_height=board_height
        )


# =============================================================================
# ALGORITHM PERFORMANCE TRACKING
# =============================================================================

@dataclass
class AlgorithmStats:
    """Statistics for an algorithm's performance"""
    algorithm: str
    success_count: int = 0
    failure_count: int = 0
    total_time_ms: float = 0.0
    avg_quality_score: float = 0.0
    last_used: float = 0.0

    # Per-profile success rates
    profile_success: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def total_attempts(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.5  # Neutral prior
        return self.success_count / self.total_attempts

    @property
    def avg_time_ms(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.total_time_ms / self.total_attempts

    def get_profile_success_rate(self, profile_key: str) -> float:
        """Get success rate for a specific design profile"""
        if profile_key not in self.profile_success:
            return 0.5  # Neutral prior
        successes, total = self.profile_success[profile_key]
        if total == 0:
            return 0.5
        return successes / total

    def record_attempt(self, profile_key: str, success: bool, time_ms: float = 0.0,
                       quality_score: float = 0.0):
        """Record an algorithm attempt"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.total_time_ms += time_ms
        self.last_used = time.time()

        # Update quality score (rolling average)
        if quality_score > 0:
            if self.total_attempts == 1:
                self.avg_quality_score = quality_score
            else:
                self.avg_quality_score = (self.avg_quality_score * 0.9 + quality_score * 0.1)

        # Update profile-specific stats
        if profile_key not in self.profile_success:
            self.profile_success[profile_key] = (0, 0)
        prev_success, prev_total = self.profile_success[profile_key]
        self.profile_success[profile_key] = (
            prev_success + (1 if success else 0),
            prev_total + 1
        )

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'algorithm': self.algorithm,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_time_ms': self.total_time_ms,
            'avg_quality_score': self.avg_quality_score,
            'last_used': self.last_used,
            'profile_success': self.profile_success
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AlgorithmStats':
        """Deserialize from dictionary"""
        stats = cls(algorithm=data['algorithm'])
        stats.success_count = data.get('success_count', 0)
        stats.failure_count = data.get('failure_count', 0)
        stats.total_time_ms = data.get('total_time_ms', 0.0)
        stats.avg_quality_score = data.get('avg_quality_score', 0.0)
        stats.last_used = data.get('last_used', 0.0)
        # Convert profile_success back to proper format
        profile_data = data.get('profile_success', {})
        for key, value in profile_data.items():
            if isinstance(value, list):
                stats.profile_success[key] = tuple(value)
            else:
                stats.profile_success[key] = value
        return stats


# =============================================================================
# CASCADE OPTIMIZER
# =============================================================================

# Default algorithm cascades (fallback when no history exists)
DEFAULT_CASCADES = {
    'placement': [
        ('hybrid', 'HYBRID (Force-Directed + SA refinement)'),
        ('auto', 'AUTO (tries all, picks best)'),
        ('human', 'HUMAN-LIKE (grid-aligned, intuitive)'),
        ('fd', 'FORCE-DIRECTED (Fruchterman-Reingold)'),
        ('sa', 'SIMULATED ANNEALING (probabilistic)'),
        ('ga', 'GENETIC ALGORITHM (evolutionary)'),
        ('analytical', 'ANALYTICAL (mathematical optimization)'),
        ('parallel', 'PARALLEL (all algorithms, parallel execution)'),
    ],
    'routing': [
        ('hybrid', 'HYBRID (A* + Steiner + Ripup)'),
        ('pathfinder', 'PATHFINDER (Negotiated Congestion)'),
        ('lee', 'LEE (Guaranteed Shortest Path)'),
        ('ripup', 'RIP-UP & REROUTE (Iterative)'),
        ('astar', 'A* (Fast Heuristic)'),
        ('hadlock', 'HADLOCK (Faster than Lee)'),
        ('steiner', 'STEINER (Multi-Terminal Optimal)'),
        ('soukup', 'SOUKUP (Quick Greedy + Fallback)'),
        ('mikami', 'MIKAMI (Memory Efficient)'),
        ('channel', 'CHANNEL (Left-Edge Greedy)'),
        ('auto', 'AUTO (Best for design)'),
    ],
    'optimize': [
        ('all', 'ALL (Run all optimizations)'),
        ('wirelength', 'WIRELENGTH (Shorten traces)'),
        ('via_reduction', 'VIA REDUCTION (Minimize vias)'),
        ('layer_balance', 'LAYER BALANCE (Even distribution)'),
        ('timing', 'TIMING (Critical path optimization)'),
        ('power', 'POWER (Power net optimization)'),
    ],
    'order': [
        ('criticality', 'CRITICALITY (Priority-based)'),
        ('manhattan', 'MANHATTAN (Distance-based)'),
        ('graph_based', 'GRAPH (Connectivity-based)'),
        ('random', 'RANDOM (For comparison)'),
    ],
    'escape': [
        ('dijkstra', 'DIJKSTRA (Shortest path)'),
        ('bfs', 'BFS (Breadth-first)'),
        ('dfs', 'DFS (Depth-first)'),
    ],
    'silkscreen': [
        ('greedy', 'GREEDY (Quick placement)'),
        ('force_directed', 'FORCE-DIRECTED (Collision avoidance)'),
        ('none', 'NONE (Skip silkscreen)'),
    ],
}

# Profile-based default preferences
PROFILE_PREFERENCES = {
    # High-density designs benefit from PATHFINDER's congestion negotiation
    (DesignDensity.HIGH, DesignDensity.ULTRA_HIGH): {
        'routing': ['pathfinder', 'hybrid', 'ripup', 'lee'],
        'placement': ['sa', 'hybrid', 'ga'],
    },
    # Simple designs can use faster algorithms
    (DesignComplexity.SIMPLE,): {
        'routing': ['astar', 'hybrid', 'hadlock'],
        'placement': ['fd', 'human', 'hybrid'],
    },
    # BGA/QFN heavy designs need careful escape routing
    'high_bga': {
        'escape': ['dijkstra', 'bfs'],
        'routing': ['pathfinder', 'hybrid'],
    },
}


@dataclass
class CascadeOptimizerConfig:
    """Configuration for the cascade optimizer"""
    # Learning parameters
    learning_rate: float = 0.1  # How fast to adapt to new results
    min_samples: int = 5  # Minimum samples before trusting learned order
    decay_factor: float = 0.95  # Decay old results over time

    # Exploration vs exploitation
    exploration_rate: float = 0.1  # Chance to try non-optimal algorithm

    # Persistence
    history_file: str = ''  # Path to save/load history
    auto_save: bool = True  # Save after each recorded result

    # Scoring weights
    success_weight: float = 0.6
    speed_weight: float = 0.2
    quality_weight: float = 0.2


class CascadeOptimizer:
    """
    Dynamic algorithm priority optimizer for the PCB Engine CASCADE system.

    Instead of using fixed algorithm orders, this optimizer:
    1. Tracks success/failure rates per algorithm per design profile
    2. Reorders algorithms to prioritize those most likely to succeed
    3. Learns over time to improve algorithm selection
    4. Allows user configuration to override learned preferences

    Usage:
        optimizer = CascadeOptimizer()
        optimizer.load('cascade_history.json')

        # Get optimized cascade for routing
        profile = DesignProfile.from_parts_db(parts_db, board_width, board_height)
        algorithms = optimizer.get_cascade('routing', profile)

        # Record result after execution
        optimizer.record_result('routing', 'hybrid', profile, success=True,
                                time_ms=1500, quality_score=0.85)
        optimizer.save()
    """

    def __init__(self, config: CascadeOptimizerConfig = None):
        self.config = config or CascadeOptimizerConfig()

        # Algorithm statistics per piston
        self.stats: Dict[str, Dict[str, AlgorithmStats]] = defaultdict(dict)

        # User-configured preferences (override learned)
        self.user_preferences: Dict[str, List[str]] = {}

        # Profile-specific overrides
        self.profile_overrides: Dict[str, Dict[str, List[str]]] = {}

        # Initialize with defaults
        self._init_default_stats()

    def _init_default_stats(self):
        """Initialize statistics with default cascades"""
        for piston, algorithms in DEFAULT_CASCADES.items():
            for algo, desc in algorithms:
                if algo not in self.stats[piston]:
                    self.stats[piston][algo] = AlgorithmStats(algorithm=algo)

    def get_cascade(self, piston: str, profile: DesignProfile = None) -> List[Tuple[str, str]]:
        """
        Get the optimized algorithm cascade for a piston.

        Args:
            piston: Piston name ('routing', 'placement', etc.)
            profile: Optional design profile for profile-specific optimization

        Returns:
            List of (algorithm_id, description) tuples in priority order
        """
        # Start with default cascade
        if piston not in DEFAULT_CASCADES:
            return []

        default_cascade = DEFAULT_CASCADES[piston]
        algo_to_desc = {algo: desc for algo, desc in default_cascade}

        # Check user preferences first (highest priority)
        if piston in self.user_preferences:
            user_algos = self.user_preferences[piston]
            return [(algo, algo_to_desc.get(algo, algo.upper())) for algo in user_algos]

        # Check profile-specific overrides
        if profile:
            profile_key = profile.to_key()
            if profile_key in self.profile_overrides and piston in self.profile_overrides[profile_key]:
                override_algos = self.profile_overrides[profile_key][piston]
                return [(algo, algo_to_desc.get(algo, algo.upper())) for algo in override_algos]

        # Calculate optimized order based on learned statistics
        return self._calculate_optimized_order(piston, profile, algo_to_desc)

    def _calculate_optimized_order(self, piston: str, profile: DesignProfile,
                                   algo_to_desc: Dict[str, str]) -> List[Tuple[str, str]]:
        """Calculate optimized algorithm order based on statistics"""
        if piston not in self.stats:
            return [(algo, desc) for algo, desc in DEFAULT_CASCADES.get(piston, [])]

        piston_stats = self.stats[piston]
        profile_key = profile.to_key() if profile else 'default'

        # Score each algorithm
        scored_algorithms = []
        for algo, stats in piston_stats.items():
            score = self._calculate_algorithm_score(stats, profile_key)
            scored_algorithms.append((algo, score))

        # Sort by score (highest first)
        scored_algorithms.sort(key=lambda x: x[1], reverse=True)

        # Apply exploration: occasionally try non-optimal algorithms
        if self.config.exploration_rate > 0:
            import random
            if random.random() < self.config.exploration_rate:
                # Shuffle a random portion
                shuffle_count = max(1, len(scored_algorithms) // 4)
                indices = list(range(len(scored_algorithms)))
                random.shuffle(indices[:shuffle_count])
                scored_algorithms = [scored_algorithms[i] for i in indices]

        # Build result with descriptions
        result = []
        for algo, _ in scored_algorithms:
            desc = algo_to_desc.get(algo, algo.upper())
            result.append((algo, desc))

        # Add any missing algorithms at the end
        included = {algo for algo, _ in result}
        for algo, desc in DEFAULT_CASCADES.get(piston, []):
            if algo not in included:
                result.append((algo, desc))

        return result

    def _calculate_algorithm_score(self, stats: AlgorithmStats,
                                   profile_key: str) -> float:
        """
        Calculate a score for an algorithm based on its statistics.

        Higher score = higher priority in the cascade.
        """
        config = self.config

        # Base success rate (global)
        global_success = stats.success_rate

        # Profile-specific success rate
        profile_success = stats.get_profile_success_rate(profile_key)

        # Weight profile-specific more if we have enough samples
        profile_samples = 0
        if profile_key in stats.profile_success:
            _, profile_samples = stats.profile_success[profile_key]

        if profile_samples >= config.min_samples:
            success_rate = profile_success * 0.7 + global_success * 0.3
        else:
            success_rate = global_success * 0.5 + profile_success * 0.5

        # Speed factor (normalized, faster is better)
        avg_time = stats.avg_time_ms
        if avg_time > 0:
            # Normalize: 1000ms = 1.0, faster = higher score
            speed_factor = min(1.0, 1000.0 / (avg_time + 100))
        else:
            speed_factor = 0.5

        # Quality factor
        quality_factor = stats.avg_quality_score if stats.avg_quality_score > 0 else 0.5

        # Combine with weights
        score = (
            config.success_weight * success_rate +
            config.speed_weight * speed_factor +
            config.quality_weight * quality_factor
        )

        # Recency bonus: recently successful algorithms get a small boost
        if stats.last_used > 0:
            recency = time.time() - stats.last_used
            if recency < 3600:  # Last hour
                score += 0.05
            elif recency < 86400:  # Last day
                score += 0.02

        return score

    def record_result(self, piston: str, algorithm: str, profile: DesignProfile,
                      success: bool, time_ms: float = 0.0, quality_score: float = 0.0,
                      metrics: Dict = None):
        """
        Record the result of an algorithm execution.

        Args:
            piston: Piston name ('routing', 'placement', etc.)
            algorithm: Algorithm that was used
            profile: Design profile for this execution
            success: Whether the algorithm succeeded
            time_ms: Execution time in milliseconds
            quality_score: Quality of result (0.0 - 1.0)
            metrics: Additional metrics to record
        """
        profile_key = profile.to_key() if profile else 'default'

        # Get or create stats for this algorithm
        if algorithm not in self.stats[piston]:
            self.stats[piston][algorithm] = AlgorithmStats(algorithm=algorithm)

        stats = self.stats[piston][algorithm]
        stats.record_attempt(profile_key, success, time_ms, quality_score)

        # Apply decay to old statistics
        if self.config.decay_factor < 1.0:
            self._apply_decay(piston, algorithm)

        # Auto-save if configured
        if self.config.auto_save and self.config.history_file:
            self.save()

    def _apply_decay(self, piston: str, exclude_algo: str):
        """
        Apply decay to old statistics (favor recent results).

        Only decays algorithms that haven't been used in a while
        to prevent destroying recent data.
        """
        current_time = time.time()
        min_age_for_decay = 3600  # Only decay stats older than 1 hour

        for algo, stats in self.stats[piston].items():
            if algo != exclude_algo and stats.total_attempts > 0:
                # Only decay if last used is old enough
                if stats.last_used > 0 and (current_time - stats.last_used) > min_age_for_decay:
                    factor = self.config.decay_factor
                    # Only decay if counts are high enough to not zero out useful data
                    if stats.success_count > 10:
                        stats.success_count = int(stats.success_count * factor)
                    if stats.failure_count > 10:
                        stats.failure_count = int(stats.failure_count * factor)

    def set_user_preference(self, piston: str, algorithms: List[str]):
        """
        Set user-defined algorithm preference for a piston.

        Args:
            piston: Piston name
            algorithms: List of algorithm IDs in priority order
        """
        self.user_preferences[piston] = algorithms

    def clear_user_preference(self, piston: str):
        """Clear user preference for a piston"""
        if piston in self.user_preferences:
            del self.user_preferences[piston]

    def set_profile_override(self, profile: DesignProfile, piston: str,
                             algorithms: List[str]):
        """
        Set algorithm order override for a specific design profile.

        Args:
            profile: Design profile to override
            piston: Piston name
            algorithms: List of algorithm IDs in priority order
        """
        profile_key = profile.to_key()
        if profile_key not in self.profile_overrides:
            self.profile_overrides[profile_key] = {}
        self.profile_overrides[profile_key][piston] = algorithms

    def get_statistics(self, piston: str = None) -> Dict:
        """
        Get algorithm statistics for analysis.

        Args:
            piston: Optional piston name, or None for all pistons

        Returns:
            Dictionary of statistics
        """
        if piston:
            return {
                algo: {
                    'success_rate': stats.success_rate,
                    'total_attempts': stats.total_attempts,
                    'avg_time_ms': stats.avg_time_ms,
                    'avg_quality': stats.avg_quality_score
                }
                for algo, stats in self.stats.get(piston, {}).items()
            }
        else:
            return {
                p: self.get_statistics(p)
                for p in self.stats.keys()
            }

    def get_recommendations(self, profile: DesignProfile) -> Dict[str, List[str]]:
        """
        Get algorithm recommendations for a design profile.

        Args:
            profile: Design profile to get recommendations for

        Returns:
            Dictionary of piston -> recommended algorithms
        """
        recommendations = {}
        for piston in DEFAULT_CASCADES.keys():
            cascade = self.get_cascade(piston, profile)
            recommendations[piston] = [algo for algo, _ in cascade[:3]]  # Top 3
        return recommendations

    def save(self, filepath: str = None):
        """
        Save learned statistics and preferences to file.

        Args:
            filepath: Path to save to, or use config.history_file
        """
        filepath = filepath or self.config.history_file
        if not filepath:
            return

        data = {
            'version': '1.0',
            'stats': {},
            'user_preferences': self.user_preferences,
            'profile_overrides': self.profile_overrides,
        }

        # Serialize statistics
        for piston, piston_stats in self.stats.items():
            data['stats'][piston] = {
                algo: stats.to_dict()
                for algo, stats in piston_stats.items()
            }

        # Write to file
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.',
                    exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str = None) -> bool:
        """
        Load learned statistics and preferences from file.

        Args:
            filepath: Path to load from, or use config.history_file

        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = filepath or self.config.history_file
        if not filepath or not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Load statistics - merge with existing (loaded stats override defaults)
            for piston, piston_data in data.get('stats', {}).items():
                # Ensure the piston dict exists
                if piston not in self.stats:
                    self.stats[piston] = {}
                for algo, algo_data in piston_data.items():
                    loaded_stats = AlgorithmStats.from_dict(algo_data)
                    # Only replace if loaded stats have actual data
                    if loaded_stats.total_attempts > 0:
                        self.stats[piston][algo] = loaded_stats

            # Load preferences
            self.user_preferences = data.get('user_preferences', {})
            self.profile_overrides = data.get('profile_overrides', {})

            return True
        except Exception as e:
            print(f"[CASCADE_OPTIMIZER] Failed to load history: {e}")
            return False

    def reset_statistics(self, piston: str = None):
        """
        Reset learned statistics.

        Args:
            piston: Piston to reset, or None for all pistons
        """
        if piston:
            self.stats[piston] = {}
            self._init_default_stats()
        else:
            self.stats = defaultdict(dict)
            self._init_default_stats()

    def print_cascade_report(self, piston: str, profile: DesignProfile = None):
        """Print a report of the current cascade order with statistics"""
        print(f"\n{'='*60}")
        print(f"CASCADE REPORT: {piston.upper()}")
        if profile:
            print(f"Profile: {profile.to_key()}")
        print(f"{'='*60}")

        cascade = self.get_cascade(piston, profile)
        piston_stats = self.stats.get(piston, {})

        print(f"{'#':>2} {'Algorithm':<15} {'Success':>8} {'Attempts':>8} {'Avg Time':>10}")
        print(f"{'-'*60}")

        for i, (algo, desc) in enumerate(cascade, 1):
            stats = piston_stats.get(algo, AlgorithmStats(algorithm=algo))
            success_pct = f"{stats.success_rate*100:.1f}%"
            time_str = f"{stats.avg_time_ms:.0f}ms" if stats.avg_time_ms > 0 else "N/A"

            print(f"{i:>2} {algo:<15} {success_pct:>8} {stats.total_attempts:>8} {time_str:>10}")

        print(f"{'='*60}\n")

    # =========================================================================
    # BBL MONITOR INTEGRATION
    # =========================================================================

    def sync_from_monitor(self, monitor_session) -> int:
        """
        Sync algorithm statistics from a BBL Monitor session.

        This allows the optimizer to learn from monitored BBL runs,
        incorporating real execution data into its statistics.

        Args:
            monitor_session: A SessionMetrics object from BBLMonitor

        Returns:
            Number of algorithm results imported
        """
        if monitor_session is None:
            return 0

        imported_count = 0

        # Import algorithm metrics from the monitor session
        algorithm_metrics = getattr(monitor_session, 'algorithm_metrics', {})

        for algo_key, metrics in algorithm_metrics.items():
            if ':' not in algo_key:
                continue

            piston, algorithm = algo_key.split(':', 1)

            # Only import if we have meaningful data
            if metrics.executions > 0:
                # Create profile key from session data if available
                profile_key = 'monitored_session'

                # Ensure stats dict exists
                if algorithm not in self.stats[piston]:
                    self.stats[piston][algorithm] = AlgorithmStats(algorithm=algorithm)

                stats = self.stats[piston][algorithm]

                # Import execution data
                stats.success_count += metrics.successes
                stats.failure_count += metrics.failures
                stats.total_time_ms += metrics.total_time * 1000  # Convert to ms

                # Update quality score if available
                if metrics.quality_scores:
                    avg_quality = sum(metrics.quality_scores) / len(metrics.quality_scores)
                    if stats.avg_quality_score > 0:
                        stats.avg_quality_score = (stats.avg_quality_score + avg_quality) / 2
                    else:
                        stats.avg_quality_score = avg_quality

                stats.last_used = time.time()

                # Update profile-specific stats
                if profile_key not in stats.profile_success:
                    stats.profile_success[profile_key] = (0, 0)
                prev_success, prev_total = stats.profile_success[profile_key]
                stats.profile_success[profile_key] = (
                    prev_success + metrics.successes,
                    prev_total + metrics.executions
                )

                imported_count += 1

        # Auto-save if configured
        if self.config.auto_save and self.config.history_file and imported_count > 0:
            self.save()

        return imported_count

    def create_monitor_callback(self):
        """
        Create a callback function for real-time BBL Monitor integration.

        Returns a callback that can be passed to BBLMonitor's realtime_callback
        parameter. The callback will automatically update optimizer statistics
        when algorithm execution events are recorded.

        Returns:
            Callable that accepts EventRecord and updates optimizer stats
        """
        def on_event(event_record):
            """Handle BBL Monitor events in real-time"""
            # Only process algorithm end events
            if event_record.event_type.value != 'algorithm_end':
                return

            piston = event_record.piston
            algorithm = event_record.algorithm

            if not piston or not algorithm:
                return

            # Extract success/failure from event data
            data = event_record.data or {}
            success = data.get('success', False)
            duration_ms = event_record.duration * 1000  # Convert to ms
            quality_score = data.get('quality_score', 0.0)

            # Create a minimal profile for this event
            profile_key = 'realtime_monitor'

            # Record the result
            if algorithm not in self.stats[piston]:
                self.stats[piston][algorithm] = AlgorithmStats(algorithm=algorithm)

            stats = self.stats[piston][algorithm]
            stats.record_attempt(profile_key, success, duration_ms, quality_score)

        return on_event

    def get_monitor_compatible_stats(self) -> Dict:
        """
        Export statistics in a format compatible with BBL Monitor reports.

        Returns:
            Dictionary with algorithm performance data for monitoring dashboards
        """
        export = {
            'optimizer_version': '1.0',
            'timestamp': time.time(),
            'pistons': {}
        }

        for piston, piston_stats in self.stats.items():
            export['pistons'][piston] = {
                'algorithms': {},
                'cascade_order': [algo for algo, _ in self.get_cascade(piston)]
            }

            for algo, stats in piston_stats.items():
                export['pistons'][piston]['algorithms'][algo] = {
                    'success_rate': stats.success_rate,
                    'total_attempts': stats.total_attempts,
                    'successes': stats.success_count,
                    'failures': stats.failure_count,
                    'avg_time_ms': stats.avg_time_ms,
                    'avg_quality': stats.avg_quality_score,
                    'last_used': stats.last_used,
                    'profile_stats': {
                        k: {'successes': v[0], 'total': v[1]}
                        for k, v in stats.profile_success.items()
                    }
                }

        return export


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_optimizer(history_file: str = None) -> CascadeOptimizer:
    """
    Create a cascade optimizer with default settings.

    Args:
        history_file: Optional path to history file for persistence

    Returns:
        Configured CascadeOptimizer
    """
    config = CascadeOptimizerConfig(
        history_file=history_file or '',
        auto_save=True
    )
    optimizer = CascadeOptimizer(config)

    if history_file:
        optimizer.load()

    return optimizer


def get_optimized_cascades(parts_db: Dict, board_width: float = 100.0,
                           board_height: float = 100.0,
                           history_file: str = None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get optimized algorithm cascades for all pistons based on design profile.

    Args:
        parts_db: Parts database
        board_width: Board width in mm
        board_height: Board height in mm
        history_file: Optional path to history file

    Returns:
        Dictionary of piston -> algorithm cascade
    """
    optimizer = create_optimizer(history_file)
    profile = DesignProfile.from_parts_db(parts_db, board_width, board_height)

    cascades = {}
    for piston in DEFAULT_CASCADES.keys():
        cascades[piston] = optimizer.get_cascade(piston, profile)

    return cascades


# =============================================================================
# MODULE INFO
# =============================================================================

if __name__ == '__main__':
    # Demo/test
    print("Cascade Optimizer - Dynamic Algorithm Priority Ordering")
    print("=" * 60)

    # Create optimizer
    optimizer = CascadeOptimizer()

    # Create a sample design profile
    sample_parts_db = {
        'parts': {f'R{i}': {'footprint': '0603'} for i in range(20)},
        'nets': {f'NET{i}': {} for i in range(30)}
    }
    profile = DesignProfile.from_parts_db(sample_parts_db, 50.0, 40.0)

    print(f"\nDesign Profile: {profile.to_key()}")
    print(f"  Components: {profile.component_count}")
    print(f"  Nets: {profile.net_count}")
    print(f"  Density: {profile.density.value}")

    # Get initial cascade
    print("\nInitial Routing Cascade:")
    for i, (algo, desc) in enumerate(optimizer.get_cascade('routing', profile), 1):
        print(f"  {i}. {desc}")

    # Simulate some results
    print("\nSimulating algorithm results...")
    optimizer.record_result('routing', 'astar', profile, success=True, time_ms=500)
    optimizer.record_result('routing', 'astar', profile, success=True, time_ms=400)
    optimizer.record_result('routing', 'hybrid', profile, success=False, time_ms=2000)
    optimizer.record_result('routing', 'pathfinder', profile, success=True, time_ms=800)
    optimizer.record_result('routing', 'astar', profile, success=True, time_ms=450)

    # Get updated cascade
    print("\nUpdated Routing Cascade (after learning):")
    for i, (algo, desc) in enumerate(optimizer.get_cascade('routing', profile), 1):
        print(f"  {i}. {desc}")

    # Print report
    optimizer.print_cascade_report('routing', profile)
