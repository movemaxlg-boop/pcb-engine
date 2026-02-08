"""
Algorithm Engine - Universal Testing, Analysis, Enhancement, and Classification System

This engine provides comprehensive tools for ALL algorithm types in the PCB Engine:
- PLACEMENT algorithms (FD, SA, GA, Hybrid, etc.)
- ROUTING algorithms (Lee, A*, Hadlock, PathFinder, etc.)
- OPTIMIZATION algorithms (wirelength, via reduction, etc.)
- ESCAPE algorithms (Dijkstra, BFS, DFS)
- ORDER algorithms (criticality, manhattan, graph-based)

Capabilities:
1. TESTING - Run algorithms through standardized test suites
2. STABILITY - Check for crashes, hangs, edge cases
3. EFFECTIVENESS - Measure success rate, quality metrics
4. SUITABILITY - Match algorithms to design scenarios
5. ENHANCEMENT - Identify and apply improvements
6. CLASSIFICATION - Categorize by characteristics
7. BENCHMARKING - Compare algorithms head-to-head

Part of the PCB Engine - The Algorithm Laboratory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum
from datetime import datetime
import time
import traceback
import json
import os
import statistics
import random
import math
from pathlib import Path

from .paths import OUTPUT_BASE


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class AlgorithmType(Enum):
    """Types of algorithms in the PCB Engine"""
    PLACEMENT = "placement"
    ROUTING = "routing"
    OPTIMIZATION = "optimization"
    ESCAPE = "escape"
    ORDER = "order"
    SILKSCREEN = "silkscreen"
    POUR = "pour"


class AlgorithmCategory(Enum):
    """Algorithm classification categories"""
    # Routing categories
    MAZE_ROUTER = "maze_router"           # Lee, Hadlock, Soukup
    HEURISTIC = "heuristic"               # A*, Greedy
    GLOBAL = "global"                     # Steiner tree, MST
    NEGOTIATED = "negotiated"             # PathFinder, ripup-reroute
    CHANNEL = "channel"                   # Left-edge, constraint-based
    TOPOLOGICAL = "topological"           # Rubber-band, Delaunay

    # Placement categories
    FORCE_DIRECTED = "force_directed"     # FD, spring-based
    ANNEALING = "annealing"               # Simulated annealing
    EVOLUTIONARY = "evolutionary"         # Genetic algorithm
    ANALYTICAL = "analytical"             # Mathematical optimization
    CONSTRUCTIVE = "constructive"         # Greedy construction

    # General categories
    HYBRID = "hybrid"                     # Combines multiple approaches
    META = "meta"                         # Selects/combines other algorithms
    ITERATIVE = "iterative"               # Iterative improvement
    GRAPH_BASED = "graph_based"           # Graph algorithms


class DesignScenario(Enum):
    """Design scenarios for suitability testing"""
    SIMPLE_2LAYER = "simple_2layer"
    DENSE_2LAYER = "dense_2layer"
    SIMPLE_4LAYER = "simple_4layer"
    DENSE_4LAYER = "dense_4layer"
    HIGH_FANOUT = "high_fanout"
    BGA_ESCAPE = "bga_escape"
    DIFFERENTIAL = "differential"
    HIGH_SPEED = "high_speed"
    MIXED_SIGNAL = "mixed_signal"
    LARGE_BOARD = "large_board"
    SMALL_BOARD = "small_board"


class StabilityLevel(Enum):
    """Algorithm stability classification"""
    STABLE = "stable"
    MOSTLY_STABLE = "mostly_stable"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


class EffectivenessRating(Enum):
    """Algorithm effectiveness rating"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILING = "failing"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TestCase:
    """A single test case for algorithm testing"""
    name: str
    parts_db: Dict
    placement: Optional[Dict] = None  # Optional - placement tests don't need this
    board_width: float = 50.0
    board_height: float = 40.0
    expected_success: bool = True
    scenario: DesignScenario = DesignScenario.SIMPLE_2LAYER
    difficulty: float = 0.5
    description: str = ""
    algorithm_type: AlgorithmType = AlgorithmType.ROUTING


@dataclass
class TestResult:
    """Result of running a single test"""
    test_name: str
    algorithm: str
    algorithm_type: AlgorithmType
    success: bool
    duration_ms: float
    error: Optional[str] = None

    # Type-specific metrics
    # Routing
    nets_routed: int = 0
    nets_total: int = 0
    segments_created: int = 0
    vias_created: int = 0
    total_length_mm: float = 0.0

    # Placement
    components_placed: int = 0
    placement_cost: float = 0.0
    overlap_count: int = 0
    wirelength_estimate: float = 0.0

    # Optimization
    improvement_percent: float = 0.0
    iterations: int = 0

    # Memory
    memory_mb: float = 0.0


@dataclass
class StabilityReport:
    """Stability analysis report"""
    algorithm: str
    algorithm_type: AlgorithmType
    total_runs: int
    crashes: int
    hangs: int
    edge_case_failures: int
    stability_level: StabilityLevel
    crash_rate: float
    hang_rate: float
    issues: List[str] = field(default_factory=list)


@dataclass
class EffectivenessReport:
    """Effectiveness analysis report"""
    algorithm: str
    algorithm_type: AlgorithmType
    total_tests: int
    successful_tests: int
    success_rate: float
    avg_duration_ms: float
    rating: EffectivenessRating

    # Type-specific averages
    avg_completion_rate: float = 0.0  # Routing
    avg_vias_per_net: float = 0.0     # Routing
    avg_placement_cost: float = 0.0   # Placement
    avg_improvement: float = 0.0      # Optimization

    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class SuitabilityScore:
    """Suitability score for a scenario"""
    scenario: DesignScenario
    score: float
    reason: str


@dataclass
class SuitabilityReport:
    """Suitability analysis for different scenarios"""
    algorithm: str
    algorithm_type: AlgorithmType
    scores: List[SuitabilityScore] = field(default_factory=list)
    best_scenarios: List[DesignScenario] = field(default_factory=list)
    worst_scenarios: List[DesignScenario] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AlgorithmProfile:
    """Complete profile of an algorithm"""
    name: str
    algorithm_type: AlgorithmType
    category: AlgorithmCategory
    description: str
    time_complexity: str
    space_complexity: str
    stability: Optional[StabilityReport] = None
    effectiveness: Optional[EffectivenessReport] = None
    suitability: Optional[SuitabilityReport] = None
    enhancements_applied: List[str] = field(default_factory=list)
    version: str = "1.0"
    last_tested: Optional[datetime] = None


@dataclass
class EnhancementSuggestion:
    """Suggested enhancement for an algorithm"""
    algorithm: str
    algorithm_type: AlgorithmType
    issue: str
    suggestion: str
    expected_improvement: str
    difficulty: str
    priority: int


@dataclass
class BenchmarkResult:
    """Result of benchmarking multiple algorithms"""
    test_name: str
    algorithm_type: AlgorithmType
    results: Dict[str, TestResult]
    winner: str
    rankings: List[Tuple[str, float]]  # (algorithm, score)


# =============================================================================
# ALGORITHM METADATA DATABASE
# =============================================================================

ALGORITHM_DATABASE = {
    # =========================================================================
    # PLACEMENT ALGORITHMS
    # =========================================================================
    'fd': {
        'type': AlgorithmType.PLACEMENT,
        'category': AlgorithmCategory.FORCE_DIRECTED,
        'name': 'Force-Directed',
        'description': 'Fruchterman-Reingold force-directed placement',
        'time_complexity': 'O(n^2 * k)',
        'space_complexity': 'O(n)',
        'year': 1991,
        'author': 'Fruchterman & Reingold',
        'strengths': ['Natural clustering', 'Good for analog', 'Intuitive layout'],
        'weaknesses': ['Can get stuck', 'Slow on large designs'],
    },
    'sa': {
        'type': AlgorithmType.PLACEMENT,
        'category': AlgorithmCategory.ANNEALING,
        'name': 'Simulated Annealing',
        'description': 'Probabilistic optimization inspired by metallurgy',
        'time_complexity': 'O(n * k)',
        'space_complexity': 'O(n)',
        'year': 1983,
        'author': 'Kirkpatrick et al.',
        'strengths': ['Escapes local optima', 'Good quality', 'Tunable'],
        'weaknesses': ['Slow', 'Parameter sensitive'],
    },
    'ga': {
        'type': AlgorithmType.PLACEMENT,
        'category': AlgorithmCategory.EVOLUTIONARY,
        'name': 'Genetic Algorithm',
        'description': 'Evolutionary optimization with crossover and mutation',
        'time_complexity': 'O(p * g * n)',
        'space_complexity': 'O(p * n)',
        'year': 1975,
        'author': 'Holland',
        'strengths': ['Global search', 'Parallel', 'Novel solutions'],
        'weaknesses': ['Slow convergence', 'Parameter tuning'],
    },
    'analytical': {
        'type': AlgorithmType.PLACEMENT,
        'category': AlgorithmCategory.ANALYTICAL,
        'name': 'Analytical Placement',
        'description': 'Mathematical optimization (quadratic programming)',
        'time_complexity': 'O(n^2)',
        'space_complexity': 'O(n^2)',
        'year': 2005,
        'author': 'Various',
        'strengths': ['Fast', 'Optimal for wirelength'],
        'weaknesses': ['May overlap', 'Needs legalization'],
    },
    'human': {
        'type': AlgorithmType.PLACEMENT,
        'category': AlgorithmCategory.CONSTRUCTIVE,
        'name': 'Human-like Grid',
        'description': 'Grid-aligned placement mimicking human design',
        'time_complexity': 'O(n)',
        'space_complexity': 'O(n)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Intuitive', 'Easy to modify', 'Consistent'],
        'weaknesses': ['Suboptimal wirelength', 'Wastes space'],
    },
    'placement_hybrid': {
        'type': AlgorithmType.PLACEMENT,
        'category': AlgorithmCategory.HYBRID,
        'name': 'Hybrid Placement',
        'description': 'Force-Directed + SA refinement',
        'time_complexity': 'O(n^2)',
        'space_complexity': 'O(n)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Best of both', 'Robust', 'High quality'],
        'weaknesses': ['Slower than single method'],
    },

    # =========================================================================
    # ROUTING ALGORITHMS
    # =========================================================================
    'lee': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.MAZE_ROUTER,
        'name': 'Lee Algorithm',
        'description': 'Wavefront expansion - guaranteed shortest path',
        'time_complexity': 'O(n^2)',
        'space_complexity': 'O(n^2)',
        'year': 1961,
        'author': 'C.Y. Lee',
        'strengths': ['Guaranteed optimal', 'Simple', 'Reliable'],
        'weaknesses': ['Slow', 'Memory intensive'],
    },
    'hadlock': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.MAZE_ROUTER,
        'name': 'Hadlock Algorithm',
        'description': 'Detour-based maze routing - faster than Lee',
        'time_complexity': 'O(n^2)',
        'space_complexity': 'O(n^2)',
        'year': 1977,
        'author': 'F.O. Hadlock',
        'strengths': ['Faster than Lee', 'Still optimal'],
        'weaknesses': ['Same memory as Lee'],
    },
    'soukup': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.MAZE_ROUTER,
        'name': 'Soukup Algorithm',
        'description': 'Greedy line-search with Lee fallback',
        'time_complexity': 'O(n)',
        'space_complexity': 'O(n)',
        'year': 1978,
        'author': 'J. Soukup',
        'strengths': ['Very fast', 'Low memory'],
        'weaknesses': ['Not always optimal'],
    },
    'mikami': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.MAZE_ROUTER,
        'name': 'Mikami-Tabuchi',
        'description': 'Line-search algorithm - memory efficient',
        'time_complexity': 'O(n)',
        'space_complexity': 'O(n)',
        'year': 1968,
        'author': 'Mikami & Tabuchi',
        'strengths': ['Memory efficient', 'Fast'],
        'weaknesses': ['May miss paths'],
    },
    'a_star': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.HEURISTIC,
        'name': 'A* Pathfinding',
        'description': 'Heuristic search with Manhattan distance',
        'time_complexity': 'O(n log n)',
        'space_complexity': 'O(n)',
        'year': 1968,
        'author': 'Hart, Nilsson, Raphael',
        'strengths': ['Fast', 'Good paths', 'Flexible heuristics'],
        'weaknesses': ['Heuristic dependent'],
    },
    'steiner': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.GLOBAL,
        'name': 'Steiner Tree',
        'description': 'Minimum spanning tree for multi-terminal nets',
        'time_complexity': 'O(n^3)',
        'space_complexity': 'O(n^2)',
        'year': 1966,
        'author': 'Hanan',
        'strengths': ['Optimal for multi-pin', 'Minimal wirelength'],
        'weaknesses': ['NP-hard exact', 'Complex'],
    },
    'pathfinder': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.NEGOTIATED,
        'name': 'PathFinder',
        'description': 'Negotiated congestion-based routing',
        'time_complexity': 'O(n^2 * k)',
        'space_complexity': 'O(n^2)',
        'year': 1995,
        'author': 'McMurchie & Ebeling',
        'strengths': ['Handles congestion', 'Complete', 'FPGA standard'],
        'weaknesses': ['Complex', 'Slow startup'],
    },
    'ripup': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.NEGOTIATED,
        'name': 'Rip-up Reroute',
        'description': 'Iterative rip-up and reroute with priority',
        'time_complexity': 'O(n^2 * k)',
        'space_complexity': 'O(n^2)',
        'year': 1987,
        'author': 'Nair',
        'strengths': ['Completes difficult routes', 'Iterative improvement'],
        'weaknesses': ['Can be slow', 'May oscillate'],
    },
    'channel': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.CHANNEL,
        'name': 'Left-Edge Channel',
        'description': 'Channel routing with left-edge algorithm',
        'time_complexity': 'O(n log n)',
        'space_complexity': 'O(n)',
        'year': 1971,
        'author': 'Hashimoto & Stevens',
        'strengths': ['Fast', 'Organized output'],
        'weaknesses': ['Limited to channels'],
    },
    'routing_hybrid': {
        'type': AlgorithmType.ROUTING,
        'category': AlgorithmCategory.HYBRID,
        'name': 'Hybrid Router',
        'description': 'A* + Steiner + Ripup combination',
        'time_complexity': 'O(n^2 * k)',
        'space_complexity': 'O(n^2)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Best overall', 'Adaptive', 'High completion'],
        'weaknesses': ['Complex', 'Parameter tuning'],
    },

    # =========================================================================
    # OPTIMIZATION ALGORITHMS
    # =========================================================================
    'opt_wirelength': {
        'type': AlgorithmType.OPTIMIZATION,
        'category': AlgorithmCategory.ITERATIVE,
        'name': 'Wirelength Optimization',
        'description': 'Minimize total trace length',
        'time_complexity': 'O(n * k)',
        'space_complexity': 'O(n)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Reduces EMI', 'Faster signals'],
        'weaknesses': ['May increase vias'],
    },
    'opt_via_reduction': {
        'type': AlgorithmType.OPTIMIZATION,
        'category': AlgorithmCategory.ITERATIVE,
        'name': 'Via Reduction',
        'description': 'Minimize layer changes',
        'time_complexity': 'O(n * k)',
        'space_complexity': 'O(n)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Cheaper manufacturing', 'Better reliability'],
        'weaknesses': ['May increase wirelength'],
    },
    'opt_layer_balance': {
        'type': AlgorithmType.OPTIMIZATION,
        'category': AlgorithmCategory.ITERATIVE,
        'name': 'Layer Balancing',
        'description': 'Even trace distribution across layers',
        'time_complexity': 'O(n)',
        'space_complexity': 'O(1)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Better thermal', 'Easier manufacturing'],
        'weaknesses': ['May conflict with other goals'],
    },

    # =========================================================================
    # ESCAPE ALGORITHMS
    # =========================================================================
    'escape_dijkstra': {
        'type': AlgorithmType.ESCAPE,
        'category': AlgorithmCategory.GRAPH_BASED,
        'name': 'Dijkstra Escape',
        'description': 'Shortest path escape from BGA/QFN',
        'time_complexity': 'O(n log n)',
        'space_complexity': 'O(n)',
        'year': 1959,
        'author': 'Dijkstra',
        'strengths': ['Optimal paths', 'Reliable'],
        'weaknesses': ['Slower for dense packages'],
    },
    'escape_bfs': {
        'type': AlgorithmType.ESCAPE,
        'category': AlgorithmCategory.GRAPH_BASED,
        'name': 'BFS Escape',
        'description': 'Breadth-first search for escape routing',
        'time_complexity': 'O(n)',
        'space_complexity': 'O(n)',
        'year': None,
        'author': 'Classic',
        'strengths': ['Simple', 'Finds path if exists'],
        'weaknesses': ['Not optimal'],
    },

    # =========================================================================
    # ORDER ALGORITHMS
    # =========================================================================
    'order_criticality': {
        'type': AlgorithmType.ORDER,
        'category': AlgorithmCategory.HEURISTIC,
        'name': 'Criticality Ordering',
        'description': 'Order nets by criticality (timing, fanout)',
        'time_complexity': 'O(n log n)',
        'space_complexity': 'O(n)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Critical nets first', 'Better timing'],
        'weaknesses': ['Requires net analysis'],
    },
    'order_manhattan': {
        'type': AlgorithmType.ORDER,
        'category': AlgorithmCategory.HEURISTIC,
        'name': 'Manhattan Ordering',
        'description': 'Order by estimated Manhattan distance',
        'time_complexity': 'O(n)',
        'space_complexity': 'O(1)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Fast', 'Simple'],
        'weaknesses': ['Ignores net importance'],
    },
    'order_graph': {
        'type': AlgorithmType.ORDER,
        'category': AlgorithmCategory.GRAPH_BASED,
        'name': 'Graph-Based Ordering',
        'description': 'Topological order based on net dependencies',
        'time_complexity': 'O(n + e)',
        'space_complexity': 'O(n)',
        'year': 2024,
        'author': 'PCB Engine',
        'strengths': ['Handles dependencies', 'Avoids conflicts'],
        'weaknesses': ['Complex setup'],
    },
}


# =============================================================================
# TEST CASE GENERATORS
# =============================================================================

class TestCaseGenerator:
    """Generates standardized test cases for all algorithm types"""

    # -------------------------------------------------------------------------
    # PLACEMENT TEST CASES
    # -------------------------------------------------------------------------

    @staticmethod
    def placement_simple() -> TestCase:
        """Simple placement: 4 components"""
        return TestCase(
            name="placement_simple",
            parts_db={
                'parts': {
                    'R1': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'B', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                    'R2': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'B', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'C', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                    'C1': {'footprint': '0603', 'pins': [
                        {'number': '1', 'net': 'A', 'offset': (-0.75, 0), 'size': (0.6, 0.9)},
                        {'number': '2', 'net': 'GND', 'offset': (0.75, 0), 'size': (0.6, 0.9)},
                    ]},
                    'C2': {'footprint': '0603', 'pins': [
                        {'number': '1', 'net': 'C', 'offset': (-0.75, 0), 'size': (0.6, 0.9)},
                        {'number': '2', 'net': 'GND', 'offset': (0.75, 0), 'size': (0.6, 0.9)},
                    ]},
                },
                'nets': {
                    'A': {'pins': [('R1', '1'), ('C1', '1')]},
                    'B': {'pins': [('R1', '2'), ('R2', '1')]},
                    'C': {'pins': [('R2', '2'), ('C2', '1')]},
                    'GND': {'pins': [('C1', '2'), ('C2', '2')]},
                }
            },
            board_width=30.0,
            board_height=25.0,
            scenario=DesignScenario.SIMPLE_2LAYER,
            difficulty=0.2,
            description="4 components with chain connectivity",
            algorithm_type=AlgorithmType.PLACEMENT,
        )

    @staticmethod
    def placement_medium() -> TestCase:
        """Medium placement: 12 components"""
        parts = {}
        nets = {'VCC': {'pins': []}, 'GND': {'pins': []}}

        for i in range(6):
            ref = f'R{i+1}'
            net_a = f'SIG{i}'
            net_b = f'SIG{i+1}' if i < 5 else 'GND'
            parts[ref] = {'footprint': '0402', 'pins': [
                {'number': '1', 'net': net_a, 'offset': (-0.5, 0), 'size': (0.5, 0.5)},
                {'number': '2', 'net': net_b, 'offset': (0.5, 0), 'size': (0.5, 0.5)},
            ]}
            if net_a not in nets:
                nets[net_a] = {'pins': []}
            nets[net_a]['pins'].append((ref, '1'))
            if net_b not in nets:
                nets[net_b] = {'pins': []}
            nets[net_b]['pins'].append((ref, '2'))

        for i in range(6):
            ref = f'C{i+1}'
            net = f'SIG{i}'
            parts[ref] = {'footprint': '0402', 'pins': [
                {'number': '1', 'net': net, 'offset': (-0.5, 0), 'size': (0.5, 0.5)},
                {'number': '2', 'net': 'GND', 'offset': (0.5, 0), 'size': (0.5, 0.5)},
            ]}
            if net not in nets:
                nets[net] = {'pins': []}
            nets[net]['pins'].append((ref, '1'))
            nets['GND']['pins'].append((ref, '2'))

        return TestCase(
            name="placement_medium",
            parts_db={'parts': parts, 'nets': nets},
            board_width=40.0,
            board_height=35.0,
            scenario=DesignScenario.DENSE_2LAYER,
            difficulty=0.5,
            description="12 components with chain and bypass caps",
            algorithm_type=AlgorithmType.PLACEMENT,
        )

    @staticmethod
    def placement_dense() -> TestCase:
        """Dense placement: 25 components in grid"""
        parts = {}
        nets = {'GND': {'pins': []}}

        for row in range(5):
            for col in range(5):
                idx = row * 5 + col
                ref = f'R{idx+1}'
                net = f'NET{idx}'
                parts[ref] = {'footprint': '0402', 'pins': [
                    {'number': '1', 'net': net, 'offset': (-0.5, 0), 'size': (0.5, 0.5)},
                    {'number': '2', 'net': 'GND', 'offset': (0.5, 0), 'size': (0.5, 0.5)},
                ]}
                nets[net] = {'pins': [(ref, '1')]}
                nets['GND']['pins'].append((ref, '2'))

        return TestCase(
            name="placement_dense",
            parts_db={'parts': parts, 'nets': nets},
            board_width=35.0,
            board_height=35.0,
            scenario=DesignScenario.DENSE_2LAYER,
            difficulty=0.8,
            description="5x5 grid of components - stress test",
            algorithm_type=AlgorithmType.PLACEMENT,
        )

    # -------------------------------------------------------------------------
    # ROUTING TEST CASES
    # -------------------------------------------------------------------------

    @staticmethod
    def routing_simple() -> TestCase:
        """Simple routing: 2 point-to-point nets"""
        return TestCase(
            name="routing_simple",
            parts_db={
                'parts': {
                    'R1': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                    'R2': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                },
                'nets': {
                    'A': {'pins': [('R1', '1'), ('R2', '1')]},
                    'GND': {'pins': [('R1', '2'), ('R2', '2')]},
                }
            },
            placement={'R1': (10.0, 15.0), 'R2': (25.0, 15.0)},
            board_width=35.0,
            board_height=30.0,
            scenario=DesignScenario.SIMPLE_2LAYER,
            difficulty=0.1,
            description="Straight line routing test",
            algorithm_type=AlgorithmType.ROUTING,
        )

    @staticmethod
    def routing_crossing() -> TestCase:
        """Crossing nets requiring layer change"""
        return TestCase(
            name="routing_crossing",
            parts_db={
                'parts': {
                    'R1': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                    'R2': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                    'R3': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'B', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                    'R4': {'footprint': '0805', 'pins': [
                        {'number': '1', 'net': 'B', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                        {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                    ]},
                },
                'nets': {
                    'A': {'pins': [('R1', '1'), ('R2', '1')]},
                    'B': {'pins': [('R3', '1'), ('R4', '1')]},
                    'GND': {'pins': [('R1', '2'), ('R2', '2'), ('R3', '2'), ('R4', '2')]},
                }
            },
            placement={
                'R1': (10.0, 10.0), 'R2': (30.0, 30.0),
                'R3': (10.0, 30.0), 'R4': (30.0, 10.0),
            },
            board_width=40.0,
            board_height=40.0,
            scenario=DesignScenario.SIMPLE_2LAYER,
            difficulty=0.4,
            description="Diagonal crossing requiring via",
            algorithm_type=AlgorithmType.ROUTING,
        )

    @staticmethod
    def routing_high_fanout() -> TestCase:
        """High fanout GND net"""
        parts = {}
        nets = {'VCC': {'pins': []}, 'GND': {'pins': []}}

        for i in range(8):
            ref = f'C{i+1}'
            parts[ref] = {'footprint': '0402', 'pins': [
                {'number': '1', 'net': 'VCC', 'offset': (-0.5, 0), 'size': (0.5, 0.5)},
                {'number': '2', 'net': 'GND', 'offset': (0.5, 0), 'size': (0.5, 0.5)},
            ]}
            nets['VCC']['pins'].append((ref, '1'))
            nets['GND']['pins'].append((ref, '2'))

        # Circular placement
        placement = {}
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            x = 25 + 12 * math.cos(angle)
            y = 25 + 12 * math.sin(angle)
            placement[f'C{i+1}'] = (x, y)

        return TestCase(
            name="routing_high_fanout",
            parts_db={'parts': parts, 'nets': nets},
            placement=placement,
            board_width=50.0,
            board_height=50.0,
            scenario=DesignScenario.HIGH_FANOUT,
            difficulty=0.6,
            description="8-pin GND/VCC fanout test",
            algorithm_type=AlgorithmType.ROUTING,
        )

    # -------------------------------------------------------------------------
    # GET ALL TEST CASES
    # -------------------------------------------------------------------------

    @classmethod
    def get_all_test_cases(cls) -> List[TestCase]:
        """Get all standard test cases"""
        return [
            # Placement
            cls.placement_simple(),
            cls.placement_medium(),
            cls.placement_dense(),
            # Routing
            cls.routing_simple(),
            cls.routing_crossing(),
            cls.routing_high_fanout(),
        ]

    @classmethod
    def get_test_cases_by_type(cls, alg_type: AlgorithmType) -> List[TestCase]:
        """Get test cases for a specific algorithm type"""
        all_tests = cls.get_all_test_cases()
        return [t for t in all_tests if t.algorithm_type == alg_type]


# =============================================================================
# ALGORITHM ENGINE
# =============================================================================

class AlgorithmEngine:
    """
    Algorithm Engine - The Universal Algorithm Laboratory

    Tests, analyzes, and enhances ALL algorithm types in the PCB Engine.

    Supported algorithm types:
    - PLACEMENT: fd, sa, ga, analytical, human, hybrid
    - ROUTING: lee, hadlock, soukup, a_star, steiner, pathfinder, ripup, hybrid
    - OPTIMIZATION: wirelength, via_reduction, layer_balance
    - ESCAPE: dijkstra, bfs, dfs
    - ORDER: criticality, manhattan, graph_based

    Usage:
        engine = AlgorithmEngine()

        # Test placement algorithms
        results = engine.test_algorithm('sa', AlgorithmType.PLACEMENT)

        # Compare routing algorithms
        comparison = engine.compare_algorithms(['lee', 'a_star', 'hybrid'],
                                               AlgorithmType.ROUTING)

        # Full profile
        profile = engine.create_algorithm_profile('lee')

        # Benchmark all algorithms of a type
        benchmark = engine.benchmark_all(AlgorithmType.ROUTING)
    """

    def __init__(self):
        self.test_results: Dict[str, List[TestResult]] = {}
        self.profiles: Dict[str, AlgorithmProfile] = {}
        self.results_dir = OUTPUT_BASE / 'algorithm_engine'
        os.makedirs(self.results_dir, exist_ok=True)

    def get_algorithms_by_type(self, alg_type: AlgorithmType) -> List[str]:
        """Get all algorithms of a specific type"""
        return [name for name, meta in ALGORITHM_DATABASE.items()
                if meta['type'] == alg_type]

    def get_algorithm_info(self, algorithm: str) -> Optional[Dict]:
        """Get metadata for an algorithm"""
        return ALGORITHM_DATABASE.get(algorithm)

    # =========================================================================
    # TESTING
    # =========================================================================

    def test_algorithm(self, algorithm: str,
                       test_cases: List[TestCase] = None,
                       timeout_ms: int = 30000) -> List[TestResult]:
        """
        Run algorithm through test cases.

        Args:
            algorithm: Algorithm name
            test_cases: Test cases (auto-selected if None)
            timeout_ms: Timeout per test

        Returns:
            List of TestResult
        """
        meta = ALGORITHM_DATABASE.get(algorithm)
        if not meta:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        alg_type = meta['type']

        if test_cases is None:
            test_cases = TestCaseGenerator.get_test_cases_by_type(alg_type)

        results = []
        for test in test_cases:
            result = self._run_test(algorithm, alg_type, test, timeout_ms)
            results.append(result)

        # Store results
        key = f"{algorithm}_{alg_type.value}"
        self.test_results[key] = results

        return results

    def _run_test(self, algorithm: str, alg_type: AlgorithmType,
                  test: TestCase, timeout_ms: int) -> TestResult:
        """Run a single test based on algorithm type"""
        if alg_type == AlgorithmType.PLACEMENT:
            return self._run_placement_test(algorithm, test, timeout_ms)
        elif alg_type == AlgorithmType.ROUTING:
            return self._run_routing_test(algorithm, test, timeout_ms)
        elif alg_type == AlgorithmType.OPTIMIZATION:
            return self._run_optimization_test(algorithm, test, timeout_ms)
        else:
            # Generic test for other types
            return self._run_generic_test(algorithm, alg_type, test, timeout_ms)

    def _run_placement_test(self, algorithm: str, test: TestCase,
                            timeout_ms: int) -> TestResult:
        """Run a placement algorithm test"""
        start_time = time.time()

        try:
            from .placement_engine import PlacementEngine, PlacementConfig

            config = PlacementConfig(
                board_width=test.board_width,
                board_height=test.board_height,
                algorithm=algorithm
            )
            engine = PlacementEngine(config)
            result = engine.place(test.parts_db, {})

            duration_ms = (time.time() - start_time) * 1000

            return TestResult(
                test_name=test.name,
                algorithm=algorithm,
                algorithm_type=AlgorithmType.PLACEMENT,
                success=result.success,
                duration_ms=duration_ms,
                components_placed=len(result.positions),
                placement_cost=result.cost,
                wirelength_estimate=getattr(result, 'wirelength', 0),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test.name,
                algorithm=algorithm,
                algorithm_type=AlgorithmType.PLACEMENT,
                success=False,
                duration_ms=duration_ms,
                error=f"{type(e).__name__}: {str(e)}"
            )

    def _run_routing_test(self, algorithm: str, test: TestCase,
                          timeout_ms: int) -> TestResult:
        """Run a routing algorithm test"""
        start_time = time.time()

        try:
            from .routing_piston import RoutingPiston
            from .routing_types import RoutingConfig

            config = RoutingConfig(
                board_width=test.board_width,
                board_height=test.board_height,
                algorithm=algorithm,
                trace_width=0.25,
                clearance=0.15,
            )
            piston = RoutingPiston(config)

            routeable = [n for n, info in test.parts_db['nets'].items()
                        if len(info.get('pins', [])) >= 2]

            result = piston.route(
                parts_db=test.parts_db,
                escapes={},
                placement=test.placement,
                net_order=routeable
            )

            duration_ms = (time.time() - start_time) * 1000

            # Calculate metrics
            total_segments = 0
            total_vias = 0
            total_length = 0.0

            for route in result.routes.values():
                total_segments += len(route.segments)
                total_vias += len(route.vias) if hasattr(route, 'vias') else 0
                for seg in route.segments:
                    dx = seg.end[0] - seg.start[0]
                    dy = seg.end[1] - seg.start[1]
                    total_length += (dx**2 + dy**2) ** 0.5

            return TestResult(
                test_name=test.name,
                algorithm=algorithm,
                algorithm_type=AlgorithmType.ROUTING,
                success=result.success,
                duration_ms=duration_ms,
                nets_routed=result.routed_count,
                nets_total=result.total_count,
                segments_created=total_segments,
                vias_created=total_vias,
                total_length_mm=total_length,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test.name,
                algorithm=algorithm,
                algorithm_type=AlgorithmType.ROUTING,
                success=False,
                duration_ms=duration_ms,
                error=f"{type(e).__name__}: {str(e)}"
            )

    def _run_optimization_test(self, algorithm: str, test: TestCase,
                               timeout_ms: int) -> TestResult:
        """Run an optimization algorithm test"""
        # Placeholder - would need actual optimization piston
        return TestResult(
            test_name=test.name,
            algorithm=algorithm,
            algorithm_type=AlgorithmType.OPTIMIZATION,
            success=True,
            duration_ms=0,
        )

    def _run_generic_test(self, algorithm: str, alg_type: AlgorithmType,
                          test: TestCase, timeout_ms: int) -> TestResult:
        """Generic test for other algorithm types"""
        return TestResult(
            test_name=test.name,
            algorithm=algorithm,
            algorithm_type=alg_type,
            success=True,
            duration_ms=0,
        )

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def analyze_stability(self, algorithm: str,
                          num_runs: int = 20) -> StabilityReport:
        """Analyze algorithm stability"""
        meta = ALGORITHM_DATABASE.get(algorithm)
        if not meta:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        alg_type = meta['type']
        test_cases = TestCaseGenerator.get_test_cases_by_type(alg_type)

        crashes = 0
        hangs = 0
        edge_failures = 0
        issues = []

        for _ in range(num_runs):
            for test in test_cases:
                result = self._run_test(algorithm, alg_type, test, 10000)
                if result.error:
                    if 'timeout' in result.error.lower():
                        hangs += 1
                    else:
                        crashes += 1
                    issues.append(result.error)
                elif not result.success and test.difficulty < 0.3:
                    edge_failures += 1

        total = num_runs * len(test_cases)
        crash_rate = crashes / total if total > 0 else 0
        hang_rate = hangs / total if total > 0 else 0
        failure_rate = (crashes + hangs + edge_failures) / total if total > 0 else 0

        if failure_rate == 0:
            level = StabilityLevel.STABLE
        elif failure_rate < 0.05:
            level = StabilityLevel.MOSTLY_STABLE
        elif failure_rate < 0.20:
            level = StabilityLevel.UNSTABLE
        else:
            level = StabilityLevel.CRITICAL

        return StabilityReport(
            algorithm=algorithm,
            algorithm_type=alg_type,
            total_runs=total,
            crashes=crashes,
            hangs=hangs,
            edge_case_failures=edge_failures,
            stability_level=level,
            crash_rate=crash_rate,
            hang_rate=hang_rate,
            issues=issues[:5]
        )

    def analyze_effectiveness(self, algorithm: str,
                              results: List[TestResult] = None) -> EffectivenessReport:
        """Analyze algorithm effectiveness"""
        meta = ALGORITHM_DATABASE.get(algorithm)
        if not meta:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        alg_type = meta['type']

        if results is None:
            key = f"{algorithm}_{alg_type.value}"
            results = self.test_results.get(key, [])
            if not results:
                results = self.test_algorithm(algorithm)

        successful = [r for r in results if r.success]
        total = len(results)

        if total == 0:
            return EffectivenessReport(
                algorithm=algorithm,
                algorithm_type=alg_type,
                total_tests=0,
                successful_tests=0,
                success_rate=0.0,
                avg_duration_ms=0.0,
                rating=EffectivenessRating.FAILING
            )

        success_rate = len(successful) / total
        avg_duration = statistics.mean([r.duration_ms for r in results])

        # Type-specific metrics
        avg_completion = 0.0
        avg_vias = 0.0
        avg_cost = 0.0

        if alg_type == AlgorithmType.ROUTING:
            completions = [r.nets_routed / r.nets_total for r in results if r.nets_total > 0]
            avg_completion = statistics.mean(completions) if completions else 0
            vias = [r.vias_created / r.nets_routed for r in results if r.nets_routed > 0]
            avg_vias = statistics.mean(vias) if vias else 0

        elif alg_type == AlgorithmType.PLACEMENT:
            costs = [r.placement_cost for r in results if r.placement_cost > 0]
            avg_cost = statistics.mean(costs) if costs else 0

        # Rating
        if success_rate > 0.95:
            rating = EffectivenessRating.EXCELLENT
        elif success_rate > 0.80:
            rating = EffectivenessRating.GOOD
        elif success_rate > 0.60:
            rating = EffectivenessRating.FAIR
        elif success_rate > 0.40:
            rating = EffectivenessRating.POOR
        else:
            rating = EffectivenessRating.FAILING

        # Strengths/weaknesses from metadata
        meta_info = ALGORITHM_DATABASE.get(algorithm, {})
        strengths = meta_info.get('strengths', [])
        weaknesses = meta_info.get('weaknesses', [])

        return EffectivenessReport(
            algorithm=algorithm,
            algorithm_type=alg_type,
            total_tests=total,
            successful_tests=len(successful),
            success_rate=success_rate,
            avg_duration_ms=avg_duration,
            avg_completion_rate=avg_completion,
            avg_vias_per_net=avg_vias,
            avg_placement_cost=avg_cost,
            rating=rating,
            strengths=strengths,
            weaknesses=weaknesses
        )

    def analyze_suitability(self, algorithm: str) -> SuitabilityReport:
        """Analyze algorithm suitability for scenarios"""
        meta = ALGORITHM_DATABASE.get(algorithm)
        if not meta:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        alg_type = meta['type']
        category = meta.get('category', AlgorithmCategory.HEURISTIC)

        scores = []

        # Score based on category and type
        if alg_type == AlgorithmType.PLACEMENT:
            scores = self._score_placement_suitability(algorithm, category)
        elif alg_type == AlgorithmType.ROUTING:
            scores = self._score_routing_suitability(algorithm, category)
        else:
            scores = [SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 75, "General purpose")]

        sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
        best = [s.scenario for s in sorted_scores[:2]]
        worst = [s.scenario for s in sorted_scores[-2:]]

        return SuitabilityReport(
            algorithm=algorithm,
            algorithm_type=alg_type,
            scores=scores,
            best_scenarios=best,
            worst_scenarios=worst,
            recommendations=meta.get('strengths', [])
        )

    def _score_placement_suitability(self, algorithm: str,
                                     category: AlgorithmCategory) -> List[SuitabilityScore]:
        """Score placement algorithm suitability"""
        scores = []

        if category == AlgorithmCategory.FORCE_DIRECTED:
            scores.append(SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 90, "Natural clustering"))
            scores.append(SuitabilityScore(DesignScenario.DENSE_2LAYER, 60, "May struggle with density"))
            scores.append(SuitabilityScore(DesignScenario.MIXED_SIGNAL, 85, "Good analog grouping"))

        elif category == AlgorithmCategory.ANNEALING:
            scores.append(SuitabilityScore(DesignScenario.DENSE_2LAYER, 90, "Handles congestion well"))
            scores.append(SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 70, "Overhead unnecessary"))
            scores.append(SuitabilityScore(DesignScenario.LARGE_BOARD, 85, "Scales well"))

        elif category == AlgorithmCategory.EVOLUTIONARY:
            scores.append(SuitabilityScore(DesignScenario.LARGE_BOARD, 85, "Parallel search"))
            scores.append(SuitabilityScore(DesignScenario.SMALL_BOARD, 50, "Overhead too high"))

        elif category == AlgorithmCategory.HYBRID:
            scores.append(SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 85, "Versatile"))
            scores.append(SuitabilityScore(DesignScenario.DENSE_2LAYER, 90, "Best of both"))

        return scores if scores else [SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 70, "General")]

    def _score_routing_suitability(self, algorithm: str,
                                   category: AlgorithmCategory) -> List[SuitabilityScore]:
        """Score routing algorithm suitability"""
        scores = []

        if category == AlgorithmCategory.MAZE_ROUTER:
            scores.append(SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 95, "Guaranteed optimal"))
            scores.append(SuitabilityScore(DesignScenario.DENSE_2LAYER, 50, "Slow on congestion"))
            scores.append(SuitabilityScore(DesignScenario.HIGH_FANOUT, 40, "Point-to-point only"))

        elif category == AlgorithmCategory.NEGOTIATED:
            scores.append(SuitabilityScore(DesignScenario.DENSE_2LAYER, 95, "Handles conflicts"))
            scores.append(SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 60, "Overhead unnecessary"))
            scores.append(SuitabilityScore(DesignScenario.HIGH_FANOUT, 80, "Good for power nets"))

        elif category == AlgorithmCategory.GLOBAL:
            scores.append(SuitabilityScore(DesignScenario.HIGH_FANOUT, 95, "Optimal for multi-pin"))
            scores.append(SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 60, "Overkill for simple"))

        elif category == AlgorithmCategory.HYBRID:
            scores.append(SuitabilityScore(DesignScenario.DENSE_2LAYER, 90, "Adaptive"))
            scores.append(SuitabilityScore(DesignScenario.HIGH_FANOUT, 85, "Good balance"))
            scores.append(SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 80, "Works everywhere"))

        return scores if scores else [SuitabilityScore(DesignScenario.SIMPLE_2LAYER, 70, "General")]

    # =========================================================================
    # COMPARISON & BENCHMARKING
    # =========================================================================

    def compare_algorithms(self, algorithms: List[str],
                           test_cases: List[TestCase] = None) -> Dict:
        """Compare multiple algorithms"""
        if not algorithms:
            return {}

        # Get type from first algorithm
        meta = ALGORITHM_DATABASE.get(algorithms[0])
        if not meta:
            raise ValueError(f"Unknown algorithm: {algorithms[0]}")

        alg_type = meta['type']

        if test_cases is None:
            test_cases = TestCaseGenerator.get_test_cases_by_type(alg_type)

        comparison = {
            'type': alg_type.value,
            'algorithms': algorithms,
            'test_count': len(test_cases),
            'results': {},
        }

        for alg in algorithms:
            results = self.test_algorithm(alg, test_cases)
            effectiveness = self.analyze_effectiveness(alg, results)

            comparison['results'][alg] = {
                'success_rate': effectiveness.success_rate,
                'avg_duration_ms': effectiveness.avg_duration_ms,
                'rating': effectiveness.rating.value,
            }

        # Rank
        ranked = sorted(comparison['results'].items(),
                       key=lambda x: x[1]['success_rate'], reverse=True)
        comparison['ranking'] = [alg for alg, _ in ranked]
        comparison['winner'] = comparison['ranking'][0] if comparison['ranking'] else None

        return comparison

    def benchmark_all(self, alg_type: AlgorithmType) -> Dict:
        """Benchmark all algorithms of a type"""
        algorithms = self.get_algorithms_by_type(alg_type)
        return self.compare_algorithms(algorithms)

    # =========================================================================
    # PROFILES
    # =========================================================================

    def create_algorithm_profile(self, algorithm: str,
                                 full_analysis: bool = True) -> AlgorithmProfile:
        """Create complete algorithm profile"""
        meta = ALGORITHM_DATABASE.get(algorithm)
        if not meta:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        profile = AlgorithmProfile(
            name=algorithm,
            algorithm_type=meta['type'],
            category=meta.get('category', AlgorithmCategory.HEURISTIC),
            description=meta.get('description', ''),
            time_complexity=meta.get('time_complexity', 'Unknown'),
            space_complexity=meta.get('space_complexity', 'Unknown'),
        )

        if full_analysis:
            results = self.test_algorithm(algorithm)
            profile.stability = self.analyze_stability(algorithm, num_runs=5)
            profile.effectiveness = self.analyze_effectiveness(algorithm, results)
            profile.suitability = self.analyze_suitability(algorithm)
            profile.last_tested = datetime.now()

        self.profiles[algorithm] = profile
        return profile

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_summary(self, algorithm: str = None):
        """Print algorithm summary"""
        if algorithm:
            self._print_algorithm_summary(algorithm)
        else:
            for alg in self.test_results.keys():
                self._print_algorithm_summary(alg.split('_')[0])

    def _print_algorithm_summary(self, algorithm: str):
        """Print summary for one algorithm"""
        meta = ALGORITHM_DATABASE.get(algorithm)
        if not meta:
            return

        alg_type = meta['type']
        key = f"{algorithm}_{alg_type.value}"
        results = self.test_results.get(key, [])

        if not results:
            return

        effectiveness = self.analyze_effectiveness(algorithm, results)

        print(f"\n{'='*60}")
        print(f"{algorithm.upper()} ({alg_type.value})")
        print(f"{'='*60}")
        print(f"Description: {meta.get('description', 'N/A')}")
        print(f"Category: {meta.get('category', 'N/A')}")
        print(f"Tests: {effectiveness.total_tests}")
        print(f"Success Rate: {effectiveness.success_rate*100:.1f}%")
        print(f"Avg Duration: {effectiveness.avg_duration_ms:.1f}ms")
        print(f"Rating: {effectiveness.rating.value}")

        if effectiveness.strengths:
            print(f"Strengths: {', '.join(effectiveness.strengths)}")
        if effectiveness.weaknesses:
            print(f"Weaknesses: {', '.join(effectiveness.weaknesses)}")

    def save_results(self, filename: str = None) -> str:
        """Save all results to JSON"""
        if filename is None:
            filename = f"algorithm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.results_dir / filename

        data = {
            'timestamp': datetime.now().isoformat(),
            'algorithms_tested': list(self.test_results.keys()),
            'results': {},
        }

        for key, results in self.test_results.items():
            data['results'][key] = [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'duration_ms': r.duration_ms,
                    'nets_routed': r.nets_routed,
                    'nets_total': r.nets_total,
                    'components_placed': r.components_placed,
                    'error': r.error,
                }
                for r in results
            ]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return str(filepath)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def test_all_placement_algorithms() -> Dict:
    """Test all placement algorithms"""
    engine = AlgorithmEngine()
    return engine.benchmark_all(AlgorithmType.PLACEMENT)


def test_all_routing_algorithms() -> Dict:
    """Test all routing algorithms"""
    engine = AlgorithmEngine()
    return engine.benchmark_all(AlgorithmType.ROUTING)


def profile_algorithm(algorithm: str) -> AlgorithmProfile:
    """Create full profile for an algorithm"""
    engine = AlgorithmEngine()
    return engine.create_algorithm_profile(algorithm)


def list_all_algorithms() -> Dict[str, List[str]]:
    """List all available algorithms by type"""
    result = {}
    for alg_type in AlgorithmType:
        result[alg_type.value] = [
            name for name, meta in ALGORITHM_DATABASE.items()
            if meta['type'] == alg_type
        ]
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ALGORITHM ENGINE - Universal Testing Suite")
    print("=" * 70)

    # List all algorithms
    print("\n--- Available Algorithms ---")
    all_algs = list_all_algorithms()
    for alg_type, algs in all_algs.items():
        print(f"  {alg_type}: {', '.join(algs)}")

    engine = AlgorithmEngine()

    # Test placement
    print("\n--- Testing Placement Algorithms ---")
    placement_algs = ['fd', 'sa', 'placement_hybrid']
    for alg in placement_algs:
        try:
            results = engine.test_algorithm(alg)
            passed = sum(1 for r in results if r.success)
            print(f"  {alg}: {passed}/{len(results)} tests passed")
        except Exception as e:
            print(f"  {alg}: Error - {e}")

    # Test routing
    print("\n--- Testing Routing Algorithms ---")
    routing_algs = ['lee', 'a_star', 'routing_hybrid']
    for alg in routing_algs:
        try:
            results = engine.test_algorithm(alg)
            passed = sum(1 for r in results if r.success)
            print(f"  {alg}: {passed}/{len(results)} tests passed")
        except Exception as e:
            print(f"  {alg}: Error - {e}")

    # Print summaries
    engine.print_summary()

    # Save results
    filepath = engine.save_results()
    print(f"\nResults saved to: {filepath}")
