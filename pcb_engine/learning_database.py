"""
LEARNING DATABASE - Algorithm Success Tracking
===============================================

The LearningDatabase stores routing AND placement outcomes and learns
from successes and failures to improve algorithm selection over time.

ROLE IN SMART ALGORITHM MANAGEMENT:
====================================
    RoutingPlanner queries → LearningDatabase
                          ← Best algorithm for net class

    RoutingPiston records → LearningDatabase
                          → Outcome stored for future reference

    PlacementEngine records → LearningDatabase
                            → Placement outcome stored

    Engine queries → LearningDatabase
                   ← Best placement algorithm for board size

KEY FEATURES:
=============
1. Record every routing attempt (success/fail, time, quality)
2. Record every placement attempt (algorithm, wirelength, overlap)
3. Track algorithm success rates per net class / board size
4. Store design fingerprints for similarity matching
5. Identify failure patterns to avoid
6. Persistent JSON storage with efficient queries

Author: PCB Engine Team
Date: 2026-02-09
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import defaultdict
import json
import os
import time
import hashlib


# Import from routing_planner (forward reference for type hints)
try:
    from .routing_planner import NetClass, RoutingAlgorithm, DesignProfile
except ImportError:
    # Standalone mode
    from enum import Enum

    class NetClass(Enum):
        POWER = 'power'
        GROUND = 'ground'
        HIGH_SPEED = 'high_speed'
        DIFFERENTIAL = 'differential'
        BUS = 'bus'
        I2C = 'i2c'
        SPI = 'spi'
        ANALOG = 'analog'
        RF = 'rf'
        HIGH_CURRENT = 'high_current'
        SIGNAL = 'signal'

    class RoutingAlgorithm(Enum):
        LEE = 'lee'
        HADLOCK = 'hadlock'
        A_STAR = 'a_star'
        PATHFINDER = 'pathfinder'
        HYBRID = 'hybrid'
        STEINER = 'steiner'

    DesignProfile = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RoutingOutcome:
    """Record of a single net routing attempt"""
    # Identification
    net_name: str
    net_class: str  # NetClass value
    design_hash: str  # Fingerprint for similarity

    # Algorithm used
    algorithm: str  # RoutingAlgorithm value

    # Results
    success: bool
    time_ms: float
    via_count: int = 0
    wire_length_mm: float = 0.0
    drc_violations: int = 0

    # Quality metrics
    quality_score: float = 0.0  # 0-100

    # Metadata
    timestamp: str = ''
    board_name: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'RoutingOutcome':
        return cls(**data)


@dataclass
class PlacementOutcome:
    """Record of a single placement attempt"""
    # Identification
    design_hash: str
    algorithm: str  # 'force_directed', 'simulated_annealing', 'genetic', etc.

    # Results
    success: bool
    time_ms: float
    wirelength_mm: float = 0.0
    overlap_area: float = 0.0
    cost: float = 0.0

    # Design context
    component_count: int = 0
    board_area_mm2: float = 0.0

    # Quality metrics
    quality_score: float = 0.0  # 0-100

    # Metadata
    timestamp: str = ''
    board_name: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PlacementOutcome':
        # Filter out unknown fields for forward compatibility
        valid = {k: v for k, v in data.items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class PlacementAlgorithmStats:
    """Statistics for a placement algorithm on a board size bucket"""
    algorithm: str
    size_bucket: str  # 'small', 'medium', 'large', 'xlarge'

    # Counts
    attempts: int = 0
    successes: int = 0

    # Timing
    total_time_ms: float = 0.0

    # Quality
    total_wirelength: float = 0.0
    total_quality: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_quality(self) -> float:
        return self.total_quality / self.successes if self.successes > 0 else 0.0

    @property
    def avg_wirelength(self) -> float:
        return self.total_wirelength / self.successes if self.successes > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'algorithm': self.algorithm,
            'size_bucket': self.size_bucket,
            'attempts': self.attempts,
            'successes': self.successes,
            'success_rate': self.success_rate,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': self.avg_time_ms,
            'total_wirelength': self.total_wirelength,
            'avg_wirelength': self.avg_wirelength,
            'total_quality': self.total_quality,
            'avg_quality': self.avg_quality,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PlacementAlgorithmStats':
        return cls(
            algorithm=data['algorithm'],
            size_bucket=data['size_bucket'],
            attempts=data.get('attempts', 0),
            successes=data.get('successes', 0),
            total_time_ms=data.get('total_time_ms', 0.0),
            total_wirelength=data.get('total_wirelength', 0.0),
            total_quality=data.get('total_quality', 0.0),
        )


@dataclass
class AlgorithmStats:
    """Statistics for an algorithm on a specific net class"""
    algorithm: str
    net_class: str

    # Counts
    attempts: int = 0
    successes: int = 0

    # Timing
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0

    # Quality
    total_quality: float = 0.0
    total_vias: int = 0
    total_wire_length: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_quality(self) -> float:
        return self.total_quality / self.attempts if self.attempts > 0 else 0.0

    @property
    def avg_vias(self) -> float:
        return self.total_vias / self.successes if self.successes > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'algorithm': self.algorithm,
            'net_class': self.net_class,
            'attempts': self.attempts,
            'successes': self.successes,
            'success_rate': self.success_rate,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': self.avg_time_ms,
            'min_time_ms': self.min_time_ms if self.min_time_ms != float('inf') else 0,
            'max_time_ms': self.max_time_ms,
            'total_quality': self.total_quality,
            'avg_quality': self.avg_quality,
            'total_vias': self.total_vias,
            'avg_vias': self.avg_vias,
            'total_wire_length': self.total_wire_length,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AlgorithmStats':
        return cls(
            algorithm=data['algorithm'],
            net_class=data['net_class'],
            attempts=data.get('attempts', 0),
            successes=data.get('successes', 0),
            total_time_ms=data.get('total_time_ms', 0.0),
            min_time_ms=data.get('min_time_ms', float('inf')),
            max_time_ms=data.get('max_time_ms', 0.0),
            total_quality=data.get('total_quality', 0.0),
            total_vias=data.get('total_vias', 0),
            total_wire_length=data.get('total_wire_length', 0.0),
        )


@dataclass
class DesignPattern:
    """A learned design pattern with its optimal configuration"""
    pattern_hash: str
    description: str
    board_profile: Dict  # Simplified profile info
    best_algorithms: Dict[str, str]  # net_class -> algorithm
    success_rate: float = 0.0
    sample_count: int = 0
    created_at: str = ''
    updated_at: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DesignPattern':
        return cls(**data)


@dataclass
class FailurePattern:
    """A recorded failure pattern to avoid"""
    pattern_id: str
    description: str
    conditions: Dict  # Conditions that trigger failure
    failure_reason: str
    suggested_fix: str
    occurrence_count: int = 0
    last_seen: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FailurePattern':
        return cls(**data)


# =============================================================================
# LEARNING DATABASE
# =============================================================================

class LearningDatabase:
    """
    Persistent learning from routing successes and failures.

    Stores:
    - Algorithm success rates per net class
    - Design fingerprints for similarity matching
    - Time/quality tradeoffs
    - Failure patterns to avoid

    Usage:
        db = LearningDatabase('routing_learning.json')

        # Record outcome
        db.record_outcome(RoutingOutcome(
            net_name='VCC',
            net_class='power',
            design_hash='abc123',
            algorithm='lee',
            success=True,
            time_ms=45.0
        ))

        # Query best algorithm
        best = db.get_best_algorithm(NetClass.POWER, profile)
        # Returns: RoutingAlgorithm.LEE

        # Get success rate
        rate = db.get_success_rate(RoutingAlgorithm.LEE, NetClass.POWER)
        # Returns: 0.95
    """

    VERSION = "1.0"

    def __init__(self, db_path: str = None):
        """
        Initialize the learning database.

        Args:
            db_path: Path to JSON file for persistence.
                    If None, uses default path in pcb_engine directory.
        """
        if db_path is None:
            # Default path relative to this file
            this_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(this_dir, 'routing_learning.json')

        self.db_path = db_path

        # In-memory structures — Routing
        self.outcomes: List[RoutingOutcome] = []
        self.algorithm_stats: Dict[str, Dict[str, AlgorithmStats]] = defaultdict(dict)
        self.design_patterns: Dict[str, DesignPattern] = {}
        self.failure_patterns: Dict[str, FailurePattern] = {}

        # In-memory structures — Placement
        self.placement_outcomes: List[PlacementOutcome] = []
        self.placement_stats: Dict[str, Dict[str, PlacementAlgorithmStats]] = defaultdict(dict)

        # Configuration
        self.max_outcomes = 10000  # Keep last N outcomes
        self.min_samples_for_recommendation = 5  # Need N samples before recommending

        # Load existing data
        self._load()

    # =========================================================================
    # CORE API
    # =========================================================================

    def record_outcome(self, outcome: RoutingOutcome) -> None:
        """
        Record a routing attempt result.

        This is called after every net routing attempt to build
        up statistics for algorithm selection.

        Args:
            outcome: RoutingOutcome with all details
        """
        # Ensure timestamp
        if not outcome.timestamp:
            outcome.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # Add to outcomes list
        self.outcomes.append(outcome)

        # Trim if too many
        if len(self.outcomes) > self.max_outcomes:
            self.outcomes = self.outcomes[-self.max_outcomes:]

        # Update algorithm stats
        self._update_stats(outcome)

        # Auto-save periodically (every 100 outcomes)
        if len(self.outcomes) % 100 == 0:
            self._save()

    def get_best_algorithm(self, net_class: 'NetClass',
                          profile: 'DesignProfile' = None) -> Optional['RoutingAlgorithm']:
        """
        Get the historically best algorithm for a net class.

        Returns the algorithm with highest success rate, weighted
        by sample count for statistical significance.

        Args:
            net_class: The net class to query
            profile: Optional design profile for similarity matching

        Returns:
            RoutingAlgorithm with highest success rate, or None if insufficient data
        """
        class_str = net_class.value if hasattr(net_class, 'value') else str(net_class)

        if class_str not in self.algorithm_stats:
            return None

        stats = self.algorithm_stats[class_str]

        # Find best algorithm
        best_algo = None
        best_score = 0.0

        for algo_name, algo_stats in stats.items():
            if algo_stats.attempts < self.min_samples_for_recommendation:
                continue  # Not enough samples

            # Score = success_rate * confidence_factor
            # Confidence increases with sample count (up to 100 samples)
            confidence = min(1.0, algo_stats.attempts / 100)
            score = algo_stats.success_rate * (0.7 + 0.3 * confidence)

            if score > best_score:
                best_score = score
                best_algo = algo_name

        if best_algo:
            try:
                return RoutingAlgorithm(best_algo)
            except (ValueError, AttributeError):
                return None

        return None

    def get_success_rate(self, algorithm: 'RoutingAlgorithm',
                        net_class: 'NetClass') -> float:
        """
        Get success rate for an algorithm on a specific net class.

        Args:
            algorithm: The routing algorithm
            net_class: The net class

        Returns:
            Success rate (0.0 to 1.0), or 0.0 if no data
        """
        class_str = net_class.value if hasattr(net_class, 'value') else str(net_class)
        algo_str = algorithm.value if hasattr(algorithm, 'value') else str(algorithm)

        if class_str not in self.algorithm_stats:
            return 0.0

        stats = self.algorithm_stats[class_str].get(algo_str)
        if stats is None:
            return 0.0

        return stats.success_rate

    def get_algorithm_ranking(self, net_class: 'NetClass') -> List[Dict]:
        """
        Get all algorithms ranked by success rate for a net class.

        Returns:
            List of dicts with algorithm info, sorted by success rate
        """
        class_str = net_class.value if hasattr(net_class, 'value') else str(net_class)

        if class_str not in self.algorithm_stats:
            return []

        rankings = []
        for algo_name, stats in self.algorithm_stats[class_str].items():
            rankings.append({
                'algorithm': algo_name,
                'success_rate': stats.success_rate,
                'attempts': stats.attempts,
                'avg_time_ms': stats.avg_time_ms,
                'avg_quality': stats.avg_quality,
            })

        rankings.sort(key=lambda x: x['success_rate'], reverse=True)
        return rankings

    # =========================================================================
    # PLACEMENT LEARNING API
    # =========================================================================

    @staticmethod
    def _component_size_bucket(count: int) -> str:
        """Categorize component count into a bucket for placement learning."""
        if count <= 15:
            return 'small'
        elif count <= 40:
            return 'medium'
        elif count <= 80:
            return 'large'
        else:
            return 'xlarge'

    def record_placement_outcome(self, outcome: PlacementOutcome) -> None:
        """Record a placement attempt result."""
        if not outcome.timestamp:
            outcome.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        self.placement_outcomes.append(outcome)

        if len(self.placement_outcomes) > self.max_outcomes:
            self.placement_outcomes = self.placement_outcomes[-self.max_outcomes:]

        self._update_placement_stats(outcome)

    def get_best_placement_algorithm(self, component_count: int) -> Optional[str]:
        """Get the historically best placement algorithm for a board size.

        Returns algorithm name string or None if insufficient data.
        """
        bucket = self._component_size_bucket(component_count)

        if bucket not in self.placement_stats:
            return None

        stats = self.placement_stats[bucket]
        best_algo = None
        best_score = 0.0

        for algo_name, algo_stats in stats.items():
            if algo_stats.attempts < self.min_samples_for_recommendation:
                continue
            confidence = min(1.0, algo_stats.attempts / 50)
            score = algo_stats.success_rate * (0.7 + 0.3 * confidence)
            if score > best_score:
                best_score = score
                best_algo = algo_name

        return best_algo

    def get_placement_ranking(self, component_count: int) -> List[Dict]:
        """Get all placement algorithms ranked by quality for a board size."""
        bucket = self._component_size_bucket(component_count)

        if bucket not in self.placement_stats:
            return []

        rankings = []
        for algo_name, stats in self.placement_stats[bucket].items():
            rankings.append({
                'algorithm': algo_name,
                'success_rate': stats.success_rate,
                'attempts': stats.attempts,
                'avg_time_ms': stats.avg_time_ms,
                'avg_quality': stats.avg_quality,
                'avg_wirelength': stats.avg_wirelength,
            })

        rankings.sort(key=lambda x: x['avg_quality'], reverse=True)
        return rankings

    def _update_placement_stats(self, outcome: PlacementOutcome) -> None:
        """Update placement algorithm statistics from outcome."""
        bucket = self._component_size_bucket(outcome.component_count)
        algorithm = outcome.algorithm

        if algorithm not in self.placement_stats[bucket]:
            self.placement_stats[bucket][algorithm] = PlacementAlgorithmStats(
                algorithm=algorithm,
                size_bucket=bucket,
            )

        stats = self.placement_stats[bucket][algorithm]
        stats.attempts += 1
        if outcome.success:
            stats.successes += 1
            stats.total_wirelength += outcome.wirelength_mm
            stats.total_quality += outcome.quality_score
        stats.total_time_ms += outcome.time_ms

    # =========================================================================
    # DESIGN FINGERPRINTING
    # =========================================================================

    def compute_design_hash(self, parts_db: Dict, board_config: Dict) -> str:
        """
        Compute a fingerprint hash for a design.

        This allows finding similar past designs for learning.
        The hash is based on:
        - Component count ranges
        - Net count ranges
        - Board size category
        - Package types present
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Create fingerprint components
        fp = []

        # Component count bucket (0-10, 10-25, 25-50, 50-100, 100+)
        comp_count = len(parts)
        if comp_count <= 10:
            fp.append('C:0-10')
        elif comp_count <= 25:
            fp.append('C:10-25')
        elif comp_count <= 50:
            fp.append('C:25-50')
        elif comp_count <= 100:
            fp.append('C:50-100')
        else:
            fp.append('C:100+')

        # Net count bucket
        net_count = len(nets)
        if net_count <= 20:
            fp.append('N:0-20')
        elif net_count <= 50:
            fp.append('N:20-50')
        elif net_count <= 100:
            fp.append('N:50-100')
        else:
            fp.append('N:100+')

        # Board size bucket (area in mm^2)
        width = board_config.get('board_width', 50)
        height = board_config.get('board_height', 40)
        area = width * height
        if area <= 1000:
            fp.append('A:small')
        elif area <= 2500:
            fp.append('A:medium')
        elif area <= 5000:
            fp.append('A:large')
        else:
            fp.append('A:xlarge')

        # Layer count
        layers = board_config.get('layers', 2)
        fp.append(f'L:{layers}')

        # Package types (simplified)
        has_bga = any('BGA' in str(p.get('footprint', '')).upper() for p in parts.values())
        has_qfn = any('QFN' in str(p.get('footprint', '')).upper() for p in parts.values())
        if has_bga:
            fp.append('P:BGA')
        if has_qfn:
            fp.append('P:QFN')

        # Create hash
        fp_str = '|'.join(sorted(fp))
        return hashlib.md5(fp_str.encode()).hexdigest()[:12]

    def find_similar_designs(self, design_hash: str) -> List[DesignPattern]:
        """
        Find similar past designs for learning.

        Args:
            design_hash: Hash from compute_design_hash()

        Returns:
            List of DesignPattern objects with similar profiles
        """
        if design_hash in self.design_patterns:
            return [self.design_patterns[design_hash]]

        # TODO: Implement fuzzy matching based on hash components
        return []

    # =========================================================================
    # FAILURE PATTERNS
    # =========================================================================

    def record_failure_pattern(self, pattern: FailurePattern) -> None:
        """Record a failure pattern to avoid in future"""
        pattern.last_seen = time.strftime('%Y-%m-%d %H:%M:%S')

        if pattern.pattern_id in self.failure_patterns:
            # Update existing
            existing = self.failure_patterns[pattern.pattern_id]
            existing.occurrence_count += 1
            existing.last_seen = pattern.last_seen
        else:
            pattern.occurrence_count = 1
            self.failure_patterns[pattern.pattern_id] = pattern

    def check_failure_patterns(self, conditions: Dict) -> Optional[FailurePattern]:
        """
        Check if conditions match a known failure pattern.

        Args:
            conditions: Dict of current conditions to check

        Returns:
            FailurePattern if match found, None otherwise
        """
        for pattern in self.failure_patterns.values():
            if self._conditions_match(conditions, pattern.conditions):
                return pattern
        return None

    def _conditions_match(self, current: Dict, pattern: Dict) -> bool:
        """Check if current conditions match pattern conditions"""
        for key, value in pattern.items():
            if key not in current:
                return False
            if current[key] != value:
                return False
        return True

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _update_stats(self, outcome: RoutingOutcome) -> None:
        """Update algorithm statistics from outcome"""
        net_class = outcome.net_class
        algorithm = outcome.algorithm

        # Ensure stats exist
        if algorithm not in self.algorithm_stats[net_class]:
            self.algorithm_stats[net_class][algorithm] = AlgorithmStats(
                algorithm=algorithm,
                net_class=net_class
            )

        stats = self.algorithm_stats[net_class][algorithm]

        # Update counts
        stats.attempts += 1
        if outcome.success:
            stats.successes += 1
            stats.total_vias += outcome.via_count
            stats.total_wire_length += outcome.wire_length_mm

        # Update timing
        stats.total_time_ms += outcome.time_ms
        stats.min_time_ms = min(stats.min_time_ms, outcome.time_ms)
        stats.max_time_ms = max(stats.max_time_ms, outcome.time_ms)

        # Update quality
        stats.total_quality += outcome.quality_score

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _load(self) -> None:
        """Load database from JSON file"""
        if not os.path.exists(self.db_path):
            return

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check version
            version = data.get('version', '1.0')

            # Load outcomes (last N only)
            outcomes_data = data.get('outcomes', [])
            self.outcomes = [
                RoutingOutcome.from_dict(o)
                for o in outcomes_data[-self.max_outcomes:]
            ]

            # Load algorithm stats
            stats_data = data.get('algorithm_stats', {})
            for net_class, algos in stats_data.items():
                for algo_name, algo_data in algos.items():
                    self.algorithm_stats[net_class][algo_name] = AlgorithmStats.from_dict(algo_data)

            # Load design patterns
            patterns_data = data.get('design_patterns', {})
            for pattern_hash, pattern_data in patterns_data.items():
                self.design_patterns[pattern_hash] = DesignPattern.from_dict(pattern_data)

            # Load failure patterns
            failures_data = data.get('failure_patterns', {})
            for pattern_id, pattern_data in failures_data.items():
                self.failure_patterns[pattern_id] = FailurePattern.from_dict(pattern_data)

            # Load placement outcomes
            placement_data = data.get('placement_outcomes', [])
            self.placement_outcomes = [
                PlacementOutcome.from_dict(o)
                for o in placement_data[-self.max_outcomes:]
            ]

            # Load placement stats
            pstats_data = data.get('placement_stats', {})
            for bucket, algos in pstats_data.items():
                for algo_name, algo_data in algos.items():
                    self.placement_stats[bucket][algo_name] = (
                        PlacementAlgorithmStats.from_dict(algo_data))

        except Exception as e:
            print(f"Warning: Failed to load learning database: {e}")

    def _save(self) -> None:
        """Save database to JSON file"""
        try:
            # Build data structure
            data = {
                'version': self.VERSION,
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'outcomes': [o.to_dict() for o in self.outcomes[-self.max_outcomes:]],
                'algorithm_stats': {
                    net_class: {
                        algo: stats.to_dict()
                        for algo, stats in algos.items()
                    }
                    for net_class, algos in self.algorithm_stats.items()
                },
                'design_patterns': {
                    h: p.to_dict() for h, p in self.design_patterns.items()
                },
                'failure_patterns': {
                    i: p.to_dict() for i, p in self.failure_patterns.items()
                },
                'placement_outcomes': [
                    o.to_dict() for o in
                    self.placement_outcomes[-self.max_outcomes:]
                ],
                'placement_stats': {
                    bucket: {
                        algo: stats.to_dict()
                        for algo, stats in algos.items()
                    }
                    for bucket, algos in self.placement_stats.items()
                },
            }

            # Write to file
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save learning database: {e}")

    def save(self) -> None:
        """Explicit save (public API)"""
        self._save()

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_summary(self) -> Dict:
        """Get summary of learning database contents"""
        total_outcomes = len(self.outcomes)
        total_successes = sum(1 for o in self.outcomes if o.success)

        # Count by net class
        class_counts = defaultdict(int)
        for o in self.outcomes:
            class_counts[o.net_class] += 1

        # Best algorithms by class
        best_by_class = {}
        for net_class in set(o.net_class for o in self.outcomes):
            try:
                nc = NetClass(net_class)
                best = self.get_best_algorithm(nc)
                if best:
                    best_by_class[net_class] = best.value
            except (ValueError, AttributeError):
                pass

        return {
            'total_outcomes': total_outcomes,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_outcomes if total_outcomes > 0 else 0,
            'outcomes_by_class': dict(class_counts),
            'best_algorithms': best_by_class,
            'design_patterns_count': len(self.design_patterns),
            'failure_patterns_count': len(self.failure_patterns),
            'db_path': self.db_path,
        }

    def get_report(self) -> str:
        """Get human-readable report"""
        lines = []
        lines.append("=" * 60)
        lines.append("LEARNING DATABASE REPORT")
        lines.append("=" * 60)

        summary = self.get_summary()

        lines.append(f"\nTotal outcomes recorded: {summary['total_outcomes']}")
        lines.append(f"Overall success rate: {summary['overall_success_rate']*100:.1f}%")

        lines.append("\nOutcomes by net class:")
        for net_class, count in sorted(summary['outcomes_by_class'].items()):
            lines.append(f"  {net_class:15} : {count:5}")

        lines.append("\nBest algorithms by net class:")
        for net_class, algo in sorted(summary['best_algorithms'].items()):
            lines.append(f"  {net_class:15} : {algo}")

        lines.append(f"\nDesign patterns: {summary['design_patterns_count']}")
        lines.append(f"Failure patterns: {summary['failure_patterns_count']}")

        lines.append(f"\nDatabase path: {summary['db_path']}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all data (use with caution)"""
        self.outcomes = []
        self.algorithm_stats = defaultdict(dict)
        self.design_patterns = {}
        self.failure_patterns = {}
        self._save()


# =============================================================================
# STANDALONE USAGE
# =============================================================================

def get_learning_database(db_path: str = None) -> LearningDatabase:
    """Get a learning database instance"""
    return LearningDatabase(db_path)


# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    'LearningDatabase',
    'RoutingOutcome',
    'AlgorithmStats',
    'DesignPattern',
    'FailurePattern',
    'get_learning_database',
]
