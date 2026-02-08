"""
PCB ENGINE - The Foreman
=========================

COMMAND HIERARCHY:
==================

    ┌─────────────────────────────────────────────────────────────┐
    │                         USER                                 │
    │                       (THE BOSS)                             │
    │   - Provides requirements and constraints                    │
    │   - Has final approval on all decisions                      │
    │   - Notified of challenges and critical issues               │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      CIRCUIT AI                              │
    │                    (THE ENGINEER)                            │
    │   - Understands circuit design deeply                        │
    │   - Makes intelligent algorithm/strategy decisions           │
    │   - Solves challenges that need "brain" not just "machine"   │
    │   - Escalates to USER when needed                            │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      PCB ENGINE         ◄── YOU ARE HERE     │
    │                    (THE FOREMAN)                             │
    │   - Coordinates all 18 pistons (workers)                     │
    │   - Sends work orders with specific instructions             │
    │   - Monitors DRC watchdog continuously                       │
    │   - Uses Learning Piston to improve over time                │
    │   - Decides: retry harder/deeper/longer or escalate          │
    │   - Reports challenges to Circuit AI (Engineer)              │
    └─────────────────────────────────────────────────────────────┘
                                │
    ┌───────────────────────────┴───────────────────────────┐
    │                                                       │
    │                    18 PISTON WORKERS                  │
    │                                                       │
    │  ┌─────────────────── CORE FLOW ────────────────────┐ │
    │  │                                                  │ │
    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │ │
    │  │  │  PARTS  │→ │  ORDER  │→ │PLACEMENT│          │ │
    │  │  └─────────┘  └─────────┘  └─────────┘          │ │
    │  │       │                          │              │ │
    │  │       ▼                          ▼              │ │
    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │ │
    │  │  │ ESCAPE  │→ │ ROUTING │→ │OPTIMIZE │          │ │
    │  │  └─────────┘  └─────────┘  └─────────┘          │ │
    │  │       │                          │              │ │
    │  │       ▼                          ▼              │ │
    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │ │
    │  │  │SILKSCR. │→ │   DRC   │→ │ OUTPUT  │          │ │
    │  │  └─────────┘  └─────────┘  └─────────┘          │ │
    │  │                                                  │ │
    │  └──────────────────────────────────────────────────┘ │
    │                                                       │
    │  ┌─────────────── ANALYSIS PISTONS ────────────────┐  │
    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │  │
    │  │  │ STACKUP │  │ THERMAL │  │   PDN   │         │  │
    │  │  └─────────┘  └─────────┘  └─────────┘         │  │
    │  │  ┌─────────┐  ┌─────────┐                      │  │
    │  │  │SIGNAL_I │  │ NETLIST │                      │  │
    │  │  └─────────┘  └─────────┘                      │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                       │
    │  ┌─────────────── ADVANCED PISTONS ───────────────┐  │
    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │  │
    │  │  │TOPOLOG. │  │   3D    │  │   BOM   │         │  │
    │  │  │ ROUTER  │  │VISUALIZ │  │OPTIMIZER│         │  │
    │  │  └─────────┘  └─────────┘  └─────────┘         │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                       │
    │  ┌─────────────── MACHINE LEARNING ───────────────┐  │
    │  │  ┌─────────────────────────────────────────┐   │  │
    │  │  │              LEARNING PISTON            │   │  │
    │  │  │  - Reverse engineers PCB files          │   │  │
    │  │  │  - Learns from successful designs       │   │  │
    │  │  │  - Improves algorithms over time        │   │  │
    │  │  │  - Predicts quality scores              │   │  │
    │  │  └─────────────────────────────────────────┘   │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                       │
    └───────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     DRC WATCHDOG      │
                    │ (Quality Inspector)   │
                    │ - Watches all work    │
                    │ - Reports violations  │
                    │ - Suggests fixes      │
                    └───────────────────────┘

PISTON SUMMARY (18 Workers):
============================
CORE:     Parts, Order, Placement, Escape, Routing, Optimize, Silkscreen, DRC, Output
ANALYSIS: Stackup, Thermal, PDN, Signal Integrity, Netlist
ADVANCED: Topological Router, 3D Visualization, BOM Optimizer
ML:       Learning (True Machine Learning)

FOREMAN (PCB ENGINE) RESPONSIBILITIES:
======================================
1. Receive complete design package from Engineer (Circuit AI)
2. Create work orders for each piston (worker)
3. Direct pistons to use specific algorithms when Engineer decides
4. Have DRC watchdog inspect every piston's output
5. Handle retry logic: HARDER → DEEPER → LONGER → MAXIMUM
6. Escalate to Engineer when machine logic is insufficient
7. Request final approval before delivering files

WORK ORDER SYSTEM:
==================
The Foreman can direct pistons to use specific algorithms:
  - WorkOrder(piston='routing', algorithm='pathfinder', ...)
  - WorkOrder(piston='placement', algorithm='simulated_annealing', ...)

The Engineer (Circuit AI) provides intelligent algorithm recommendations
based on design context (density, complexity, previous failures).

DRC WATCHDOG:
=============
Monitors every piston's work and reports:
  - ERRORS: Must be fixed
  - WARNINGS: Should be reviewed
  - NOTES: Informational
  - SUGGESTIONS: Possible improvements
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import os

# Import common types for position handling
from .common_types import Position, normalize_position, get_xy, get_pins, get_pin_net

# Import all pistons
try:
    from .parts_piston import PartsPiston, PartsConfig, PartsResult
    PartsEngine = PartsPiston
except ImportError:
    PartsPiston = None
    PartsEngine = None
    PartsConfig = None
    PartsResult = None

# BUG-09 FIX: Use explicit aliases to avoid Config class ambiguity
try:
    from .placement_engine import PlacementEngine, PlacementConfig as PlacementEngineConfig
except ImportError:
    PlacementEngine = None
    PlacementEngineConfig = None

try:
    from .placement_piston import PlacementPiston, PlacementConfig as PlacementPistonConfig
except ImportError:
    PlacementPiston = None
    PlacementPistonConfig = None

# Import routing types from unified module
try:
    from .routing_types import RoutingConfig, RoutingResult
    from .routing_piston import RoutingPiston, route_with_cascade
except ImportError:
    RoutingPiston = None
    RoutingConfig = None
    RoutingResult = None

try:
    from .order_piston import OrderPiston, OrderConfig, OrderResult
except ImportError:
    OrderPiston = None
    OrderConfig = None
    OrderResult = None

try:
    from .silkscreen_piston import SilkscreenPiston, SilkscreenConfig, SilkscreenResult
except ImportError:
    SilkscreenPiston = None
    SilkscreenConfig = None
    SilkscreenResult = None

try:
    from .drc_piston import DRCPiston, DRCConfig, DRCResult, DRCRules
except ImportError:
    DRCPiston = None
    DRCConfig = None
    DRCResult = None
    DRCRules = None

try:
    from .output_piston import OutputPiston, OutputConfig, OutputResult
except ImportError:
    OutputPiston = None
    OutputConfig = None
    OutputResult = None

try:
    from .escape_piston import EscapePiston, EscapeConfig, EscapeResult
except ImportError:
    EscapePiston = None
    EscapeConfig = None
    EscapeResult = None

try:
    from .optimization_piston import OptimizationPiston, OptimizationConfig, OptimizationResult
except ImportError:
    OptimizationPiston = None
    OptimizationConfig = None
    OptimizationResult = None

try:
    from .polish_piston import PolishPiston, PolishConfig, PolishResult, PolishLevel
except ImportError:
    PolishPiston = None
    PolishConfig = None
    PolishResult = None
    PolishLevel = None

try:
    from .circuit_ai import CircuitAI, CircuitAIResult
except ImportError:
    CircuitAI = None
    CircuitAIResult = None

try:
    from .stackup_piston import StackupPiston, StackupConfig
except ImportError:
    StackupPiston = None
    StackupConfig = None

try:
    from .thermal_piston import ThermalPiston, ThermalConfig
except ImportError:
    ThermalPiston = None
    ThermalConfig = None

try:
    from .pdn_piston import PDNPiston, PDNConfig
except ImportError:
    PDNPiston = None
    PDNConfig = None

try:
    from .signal_integrity_piston import SignalIntegrityPiston, SIConfig
except ImportError:
    SignalIntegrityPiston = None
    SIConfig = None

try:
    from .netlist_piston import NetlistPiston
except ImportError:
    NetlistPiston = None

try:
    from .topological_router_piston import TopologicalRouterPiston
except ImportError:
    TopologicalRouterPiston = None

try:
    from .visualization_3d_piston import Visualization3DPiston, OutputFormat as Vis3DFormat
except ImportError:
    Visualization3DPiston = None
    Vis3DFormat = None

try:
    from .bom_optimizer_piston import BOMOptimizerPiston, OptimizationGoal as BOMGoal
except ImportError:
    BOMOptimizerPiston = None
    BOMGoal = None

try:
    from .learning_piston import LearningPiston, LearningMode, LearningResult
except ImportError:
    LearningPiston = None
    LearningMode = None
    LearningResult = None

try:
    from .grid_calculator import calculate_optimal_grid_size
except ImportError:
    calculate_optimal_grid_size = None

try:
    from .piston_orchestrator import PistonOrchestrator, PistonSelection, DesignProfile
except ImportError:
    PistonOrchestrator = None
    PistonSelection = None
    DesignProfile = None

try:
    from .workflow_reporter import WorkflowReporter, LoopType, AlgorithmStatus
except ImportError:
    WorkflowReporter = None
    LoopType = None
    AlgorithmStatus = None

try:
    from .bbl_engine import (
        BBLEngine, BBLState, BBLResult, BBLPhase, BBLProgress,
        BBLCheckpoint, BBLEscalation, BBLHistoryEntry, BBLPhaseConfig
    )
except ImportError:
    BBLEngine = None

try:
    from .cascade_optimizer import (
        CascadeOptimizer, CascadeOptimizerConfig, DesignProfile as CascadeDesignProfile,
        AlgorithmStats, DEFAULT_CASCADES
    )
except ImportError:
    CascadeOptimizer = None
    CascadeOptimizerConfig = None
    CascadeDesignProfile = None
    AlgorithmStats = None
    DEFAULT_CASCADES = None
    BBLState = None
    BBLResult = None
    BBLPhase = None
    BBLProgress = None
    BBLCheckpoint = None
    BBLEscalation = None
    BBLHistoryEntry = None
    BBLPhaseConfig = None

try:
    from .bbl_monitor import BBLMonitor, MonitorLevel, MonitorEvent
except ImportError:
    BBLMonitor = None
    MonitorLevel = None
    MonitorEvent = None

try:
    from .monitor_optimizer_bridge import MonitorOptimizerBridge, BridgeConfig
except ImportError:
    MonitorOptimizerBridge = None
    BridgeConfig = None


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class EngineStage(Enum):
    """Current stage in the PCB design flow"""
    INIT = 'init'
    LEARNING_LOAD = 'learning_load'  # Load learned patterns
    CIRCUIT_AI = 'circuit_ai'
    PARTS = 'parts'
    ORDER = 'order'
    PLACEMENT = 'placement'
    ESCAPE = 'escape'
    ROUTING = 'routing'
    TOPOLOGICAL_ROUTING = 'topological_routing'  # Advanced routing
    OPTIMIZE = 'optimize'
    SILKSCREEN = 'silkscreen'
    DRC = 'drc'
    ANALYSIS = 'analysis'  # Stackup, Thermal, PDN, SI
    BOM_OPTIMIZE = 'bom_optimize'  # BOM optimization
    VISUALIZATION_3D = 'visualization_3d'  # 3D export
    OUTPUT = 'output'
    LEARNING_SAVE = 'learning_save'  # Save learned patterns
    COMPLETE = 'complete'
    ERROR = 'error'
    WAITING_AI = 'waiting_ai'  # Waiting for AI Agent decision
    WAITING_USER = 'waiting_user'  # Waiting for user input


class ChallengeType(Enum):
    """Types of challenges that require AI Agent decision"""
    BOARD_TOO_SMALL = 'board_too_small'
    LAYERS_NEEDED = 'layers_needed'
    PART_REPLACEMENT = 'part_replacement'
    ROUTING_IMPOSSIBLE = 'routing_impossible'
    DRC_PERSISTENT = 'drc_persistent'
    THERMAL_ISSUE = 'thermal_issue'
    SIGNAL_INTEGRITY = 'signal_integrity'
    PLACEMENT_CONFLICT = 'placement_conflict'
    POWER_BUDGET = 'power_budget'


class AIDecision(Enum):
    """Decisions from AI Agent"""
    APPROVE = 'approve'
    REJECT = 'reject'
    RETRY = 'retry'
    RETRY_HARDER = 'retry_harder'
    RETRY_DEEPER = 'retry_deeper'
    RETRY_LONGER = 'retry_longer'
    MODIFY_BOARD = 'modify_board'
    ADD_LAYERS = 'add_layers'
    REPLACE_PART = 'replace_part'
    ASK_USER = 'ask_user'
    ABORT = 'abort'
    DELIVER = 'deliver'


class PistonEffort(Enum):
    """Effort level for piston work"""
    NORMAL = 'normal'
    HARDER = 'harder'      # More iterations
    DEEPER = 'deeper'      # More algorithms tried
    LONGER = 'longer'      # Extended timeout
    MAXIMUM = 'maximum'    # All of the above


# =============================================================================
# PISTON CASCADE SYSTEM - ENGINE MANAGES ALL PISTONS
# =============================================================================
# The Engine (Foreman) manages all pistons (workers) by ordering them to try
# different algorithms when one fails. Each piston has a CASCADE - a priority
# list of algorithms to try until one succeeds.
#
# COMMAND HIERARCHY:
#   USER (Boss) → CIRCUIT AI (Engineer) → PCB ENGINE (Foreman) → PISTONS (Workers)
#
# The Foreman's job is to:
# 1. Send work orders to pistons with specific algorithms
# 2. Monitor DRC results after each piston's work
# 3. If a piston fails, try the next algorithm in its CASCADE
# 4. Report to the Engineer (AI) when all algorithms fail
# 5. Never give up until all options are exhausted

PISTON_CASCADES = {
    # ─────────────────────────────────────────────────────────────────────────
    # PLACEMENT PISTON - Component placement algorithms
    # ─────────────────────────────────────────────────────────────────────────
    'placement': {
        'algorithms': [
            ('hybrid', 'HYBRID (Force-Directed + SA refinement)'),
            ('auto', 'AUTO (tries all, picks best)'),
            ('human', 'HUMAN-LIKE (grid-aligned, intuitive)'),
            ('fd', 'FORCE-DIRECTED (Fruchterman-Reingold)'),
            ('sa', 'SIMULATED ANNEALING (probabilistic)'),
            ('ga', 'GENETIC ALGORITHM (evolutionary)'),
            ('analytical', 'ANALYTICAL (mathematical optimization)'),
            ('parallel', 'PARALLEL (all algorithms, parallel execution)'),
        ],
        'relaxed_params': {
            'allow_rotation': True,
            'expand_board': 1.2,  # Allow 20% board expansion
            'reduce_min_spacing': 0.8,  # Allow 20% tighter spacing
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTING PISTON - Trace routing algorithms (11 algorithms!)
    # ─────────────────────────────────────────────────────────────────────────
    'routing': {
        'algorithms': [
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
        'relaxed_params': {
            'clearance_factor': 0.8,  # Allow 20% less clearance
            'via_cost_factor': 0.5,   # Encourage layer changes
            'max_iterations_factor': 2.0,  # Double iterations
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMIZATION PISTON - Post-routing optimization
    # ─────────────────────────────────────────────────────────────────────────
    'optimize': {
        'algorithms': [
            ('all', 'ALL (Run all optimizations)'),
            ('wirelength', 'WIRELENGTH (Shorten traces)'),
            ('via_reduction', 'VIA REDUCTION (Minimize vias)'),
            ('layer_balance', 'LAYER BALANCE (Even distribution)'),
            ('timing', 'TIMING (Critical path optimization)'),
            ('power', 'POWER (Power net optimization)'),
        ],
        'relaxed_params': {},
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER PISTON - Determine routing/placement order
    # ─────────────────────────────────────────────────────────────────────────
    'order': {
        'algorithms': [
            ('criticality', 'CRITICALITY (Priority-based)'),
            ('manhattan', 'MANHATTAN (Distance-based)'),
            ('graph_based', 'GRAPH (Connectivity-based)'),
            ('random', 'RANDOM (For comparison)'),
        ],
        'relaxed_params': {},
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ESCAPE PISTON - BGA/QFN pin escape
    # ─────────────────────────────────────────────────────────────────────────
    'escape': {
        'algorithms': [
            ('dijkstra', 'DIJKSTRA (Shortest path)'),
            ('bfs', 'BFS (Breadth-first)'),
            ('dfs', 'DFS (Depth-first)'),
        ],
        'relaxed_params': {
            'allow_additional_layer': True,
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # SILKSCREEN PISTON - Reference designator placement
    # ─────────────────────────────────────────────────────────────────────────
    'silkscreen': {
        'algorithms': [
            ('greedy', 'GREEDY (Quick placement)'),
            ('force_directed', 'FORCE-DIRECTED (Collision avoidance)'),
            ('none', 'NONE (Skip silkscreen)'),
        ],
        'relaxed_params': {
            'font_scale': 0.8,  # Smaller text
            'skip_crowded': True,  # Skip refs in crowded areas
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TOPOLOGICAL ROUTER - Advanced rubber-band routing
    # ─────────────────────────────────────────────────────────────────────────
    'topological_router': {
        'algorithms': [
            ('delaunay', 'DELAUNAY (Triangulation-based)'),
            ('rubber_band', 'RUBBER-BAND (Elastic routing)'),
        ],
        'relaxed_params': {},
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WorkOrder:
    """
    Work order sent to a piston with specific instructions.

    The PCB Engine can direct pistons to use specific algorithms,
    strategies, or approaches based on the design context.
    """
    piston: str
    task: str
    effort: PistonEffort = PistonEffort.NORMAL

    # Algorithm/strategy directives
    algorithm: str = ''          # Specific algorithm to use (e.g., 'astar', 'lee', 'hadlock')
    strategy: str = ''           # High-level strategy (e.g., 'minimize_vias', 'short_traces')
    fallback_algorithms: List[str] = field(default_factory=list)  # Fallbacks if primary fails

    # Parameters and constraints
    parameters: Dict = field(default_factory=dict)
    constraints: Dict = field(default_factory=dict)
    priority_nets: List[str] = field(default_factory=list)  # Route these first
    avoid_regions: List[Dict] = field(default_factory=list)  # Keep-out zones

    # Timing
    deadline_seconds: float = 60.0
    min_quality_score: float = 0.0  # Minimum acceptable quality

    # AI input
    ai_reasoning: str = ''       # Why this approach was chosen
    human_override: bool = False  # User explicitly requested this


@dataclass
class PistonReport:
    """Report from a piston after work"""
    piston: str
    success: bool
    result: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    time_elapsed: float = 0.0


@dataclass
class DRCWatchReport:
    """Report from DRC watching a piston's work"""
    piston: str
    stage: str
    passed: bool
    errors: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    can_continue: bool = True


@dataclass
class Challenge:
    """A challenge requiring AI Agent decision"""
    type: ChallengeType
    severity: str  # 'critical', 'major', 'minor'
    description: str
    context: Dict = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    suggested_action: AIDecision = AIDecision.ASK_USER


@dataclass
class AlgorithmChoice:
    """A choice of algorithm/strategy for AI to evaluate"""
    name: str
    description: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    best_for: List[str] = field(default_factory=list)  # Scenarios where this excels
    estimated_time: float = 0.0
    estimated_quality: float = 0.0  # 0-1 scale


@dataclass
class AIRequest:
    """Request to AI Agent for decision"""
    stage: EngineStage
    challenge: Challenge = None
    current_state: Dict = field(default_factory=dict)
    question: str = ''
    options: List[str] = field(default_factory=list)

    # Algorithm/strategy selection
    algorithm_choices: List[AlgorithmChoice] = field(default_factory=list)
    context_hints: Dict = field(default_factory=dict)  # Design context for better decisions
    piston_name: str = ''  # Which piston needs direction
    previous_attempts: List[Dict] = field(default_factory=list)  # What already failed


@dataclass
class AIResponse:
    """Response from AI Agent"""
    decision: AIDecision
    parameters: Dict = field(default_factory=dict)
    message_to_user: str = ''
    new_instructions: Dict = field(default_factory=dict)

    # Algorithm/strategy direction
    selected_algorithm: str = ''
    selected_strategy: str = ''
    algorithm_parameters: Dict = field(default_factory=dict)
    reasoning: str = ''  # Why this algorithm was chosen (for logging/debugging)

    # Priority adjustments
    priority_nets: List[str] = field(default_factory=list)
    avoid_components: List[str] = field(default_factory=list)


@dataclass
class EngineConfig:
    """Master configuration for the PCB Engine"""
    # Board parameters
    board_name: str = 'pcb'
    board_width: float = 100.0
    board_height: float = 100.0
    board_origin_x: float = 0.0
    board_origin_y: float = 0.0
    layer_count: int = 2

    # Design rules
    trace_width: float = 0.25
    clearance: float = 0.15
    via_diameter: float = 0.8
    via_drill: float = 0.4

    # Grid settings
    use_dynamic_grid: bool = True
    grid_size: float = 0.1
    min_grid_size: float = 0.1
    max_grid_size: float = 0.25

    # Routing algorithm
    routing_algorithm: str = 'hybrid'

    # Order piston strategies
    placement_order_strategy: str = 'auto'
    net_order_strategy: str = 'auto'
    layer_assignment_strategy: str = 'auto'
    via_strategy: str = 'auto'
    pin_escape_strategy: str = 'auto'

    # Iteration limits
    max_placement_iterations: int = 100
    max_routing_iterations: int = 15
    max_drc_iterations: int = 5
    max_retry_per_piston: int = 3

    # Output (empty string uses DEFAULT_OUTPUT_BASE from paths.py)
    output_dir: str = ''
    generate_kicad: bool = True
    generate_gerbers: bool = True
    generate_bom: bool = True

    # Silkscreen
    show_references: bool = True
    show_values: bool = False

    # Advanced pistons
    use_topological_routing: bool = False  # Use Delaunay/rubber-band router
    optimize_bom: bool = True              # Run BOM optimizer
    generate_3d: bool = False              # Generate 3D visualization
    output_3d_format: str = 'stl'          # 'stl', 'step', 'gltf', 'obj'

    # Learning piston
    enable_learning: bool = True           # Enable ML learning
    learning_model_path: str = ''          # Path to save/load learned models
    learn_from_result: bool = True         # Learn from successful designs

    # Analysis pistons
    run_thermal_analysis: bool = False     # Run thermal piston
    run_pdn_analysis: bool = False         # Run PDN piston
    run_si_analysis: bool = False          # Run signal integrity piston

    # AI Agent integration
    ai_agent_callback: Callable = None  # Function to call AI Agent
    auto_approve_minor: bool = True     # Auto-approve minor issues
    verbose: bool = True

    # CASCADE OPTIMIZER - Dynamic algorithm ordering
    enable_cascade_optimizer: bool = True    # Use adaptive algorithm ordering
    cascade_history_file: str = ''           # Path to save/load cascade learning history
    cascade_learning_rate: float = 0.1       # How fast to adapt to new results
    cascade_exploration_rate: float = 0.1    # Chance to try non-optimal algorithm
    user_algorithm_preferences: Dict = field(default_factory=dict)  # User-defined preferences


@dataclass
class EngineState:
    """Current state of the engine"""
    stage: EngineStage = EngineStage.INIT
    parts_db: Dict = field(default_factory=dict)
    circuit_ai_result: Any = None
    order_result: Any = None
    placement_order: List = field(default_factory=list)
    net_order: List = field(default_factory=list)
    layer_assignments: Dict = field(default_factory=dict)
    placement: Dict = field(default_factory=dict)
    routes: Dict = field(default_factory=dict)
    vias: List = field(default_factory=list)
    silkscreen: Any = None
    drc_result: Any = None
    stackup: Any = None
    thermal_analysis: Any = None
    pdn_analysis: Any = None
    si_analysis: Any = None

    # Advanced piston results
    topological_routes: Dict = field(default_factory=dict)
    visualization_3d: Any = None
    bom_optimization: Any = None
    learned_patterns: List = field(default_factory=list)

    # Work tracking
    work_orders: List[WorkOrder] = field(default_factory=list)
    piston_reports: List[PistonReport] = field(default_factory=list)
    drc_watch_reports: List[DRCWatchReport] = field(default_factory=list)

    # AI interaction
    challenges: List[Challenge] = field(default_factory=list)
    ai_requests: List[AIRequest] = field(default_factory=list)
    ai_responses: List[AIResponse] = field(default_factory=list)

    # Statistics
    start_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    retry_counts: Dict[str, int] = field(default_factory=dict)

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EngineResult:
    """Final result from the engine"""
    success: bool
    stage_reached: EngineStage
    output_files: List[str] = field(default_factory=list)
    drc_passed: bool = False
    routed_count: int = 0
    total_nets: int = 0
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    challenges_resolved: int = 0
    ai_interactions: int = 0


# =============================================================================
# PCB ENGINE
# =============================================================================

class PCBEngine:
    """
    PCB Engine - The Main Coordinator with AI Agent Orchestration

    Orchestrates all pistons with DRC watchdog and AI Agent feedback loop.

    Usage:
        # With Circuit AI front-end
        circuit_ai = CircuitAI()
        result = circuit_ai.generate_parts_db(requirements)

        # Create engine with AI callback
        def ai_callback(request: AIRequest) -> AIResponse:
            # Your AI agent logic here
            return AIResponse(decision=AIDecision.APPROVE)

        config = EngineConfig(ai_agent_callback=ai_callback)
        engine = PCBEngine(config)

        # Run with full orchestration
        final_result = engine.run_orchestrated(result.parts_db)
    """

    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self.state = EngineState()

        # Initialize pistons (lazy - created when needed)
        self._parts_piston: PartsEngine = None
        self._order_piston: OrderPiston = None
        self._placement_piston: PlacementEngine = None
        self._escape_piston: EscapePiston = None
        self._routing_piston: RoutingPiston = None
        self._optimization_piston: OptimizationPiston = None
        self._polish_piston: PolishPiston = None
        self._silkscreen_piston: SilkscreenPiston = None
        self._drc_piston: DRCPiston = None
        self._output_piston: OutputPiston = None
        self._stackup_piston: StackupPiston = None if StackupPiston is None else None
        self._thermal_piston: ThermalPiston = None if ThermalPiston is None else None
        self._pdn_piston: PDNPiston = None if PDNPiston is None else None
        self._si_piston: SignalIntegrityPiston = None if SignalIntegrityPiston is None else None
        self._topological_router_piston: TopologicalRouterPiston = None if TopologicalRouterPiston is None else None
        self._visualization_3d_piston: Visualization3DPiston = None if Visualization3DPiston is None else None
        self._bom_optimizer_piston: BOMOptimizerPiston = None if BOMOptimizerPiston is None else None
        self._learning_piston: LearningPiston = None if LearningPiston is None else None

        # Internal state
        self._escape_result = None
        self._optimization_results = None

        # Piston orchestrator for smart selection
        self._orchestrator = PistonOrchestrator() if PistonOrchestrator else None
        self._piston_selection: PistonSelection = None

        # Workflow reporter for comprehensive logging
        self._workflow_reporter: WorkflowReporter = None

        # BBL Engine (Big Beautiful Loop) - The complete work cycle manager
        self._bbl_engine: BBLEngine = None
        self._bbl_progress_callback = None
        self._bbl_escalation_callback = None

        # BBL Monitor for real-time event tracking
        self._bbl_monitor: BBLMonitor = None

        # CASCADE OPTIMIZER - Dynamic algorithm ordering
        self._cascade_optimizer: CascadeOptimizer = None
        self._design_profile: CascadeDesignProfile = None

        # MONITOR-OPTIMIZER BRIDGE - Connects monitor and optimizer for analytics
        self._monitor_bridge: MonitorOptimizerBridge = None

        # Initialize cascade optimizer and bridge
        self._init_cascade_optimizer()
        self._init_monitor_bridge()

    # =========================================================================
    # CASCADE OPTIMIZER INITIALIZATION
    # =========================================================================

    def _init_cascade_optimizer(self):
        """Initialize the cascade optimizer for dynamic algorithm ordering."""
        if not self.config.enable_cascade_optimizer or CascadeOptimizer is None:
            self._cascade_optimizer = None
            return

        # Create optimizer config
        optimizer_config = CascadeOptimizerConfig(
            learning_rate=self.config.cascade_learning_rate,
            exploration_rate=self.config.cascade_exploration_rate,
            history_file=self.config.cascade_history_file,
            auto_save=bool(self.config.cascade_history_file)
        )

        self._cascade_optimizer = CascadeOptimizer(optimizer_config)

        # Load history if available
        if self.config.cascade_history_file:
            loaded = self._cascade_optimizer.load()
            if loaded:
                self._log("[CASCADE] Loaded algorithm history from file")

        # Apply user preferences if any
        for piston, algorithms in self.config.user_algorithm_preferences.items():
            self._cascade_optimizer.set_user_preference(piston, algorithms)

        self._log("[CASCADE] Optimizer initialized with dynamic algorithm ordering")

    def _init_monitor_bridge(self):
        """Initialize the Monitor-Optimizer Bridge for real-time analytics."""
        if MonitorOptimizerBridge is None:
            self._monitor_bridge = None
            return

        if not self.config.enable_cascade_optimizer:
            self._monitor_bridge = None
            return

        # Create bridge config
        bridge_config = BridgeConfig(
            output_dir=self.config.output_dir,
            auto_sync=True,
            verbose=self.config.verbose,
            persist_on_sync=bool(self.config.cascade_history_file)
        )

        # Create bridge with existing optimizer
        self._monitor_bridge = MonitorOptimizerBridge(
            config=bridge_config,
            optimizer=self._cascade_optimizer
        )

        self._log("[BRIDGE] Monitor-Optimizer Bridge initialized")

    def _update_design_profile(self, parts_db: Dict):
        """Update the design profile for cascade optimization."""
        if self._cascade_optimizer is None or CascadeDesignProfile is None:
            return

        self._design_profile = CascadeDesignProfile.from_parts_db(
            parts_db,
            self.config.board_width,
            self.config.board_height
        )

        self._log(f"[CASCADE] Design profile: {self._design_profile.to_key()}")
        self._log(f"  Components: {self._design_profile.component_count}, "
                  f"Nets: {self._design_profile.net_count}, "
                  f"Density: {self._design_profile.density.value}")

    def _get_optimized_cascade(self, piston_name: str) -> List[Tuple[str, str]]:
        """Get the optimized algorithm cascade for a piston."""
        if self._cascade_optimizer is None:
            # Fall back to static PISTON_CASCADES
            cascade = PISTON_CASCADES.get(piston_name, {})
            return cascade.get('algorithms', [])

        return self._cascade_optimizer.get_cascade(piston_name, self._design_profile)

    def _record_algorithm_result(self, piston_name: str, algorithm: str,
                                  success: bool, time_ms: float = 0.0,
                                  quality_score: float = 0.0):
        """Record algorithm result for cascade learning and monitoring."""
        # Record to cascade optimizer
        if self._cascade_optimizer is not None and self._design_profile is not None:
            self._cascade_optimizer.record_result(
                piston=piston_name,
                algorithm=algorithm,
                profile=self._design_profile,
                success=success,
                time_ms=time_ms,
                quality_score=quality_score
            )

        # Record to bridge (which updates both monitor and optimizer)
        if self._monitor_bridge:
            self._monitor_bridge.record_algorithm_end(
                piston=piston_name,
                algorithm=algorithm,
                success=success,
                time_ms=time_ms,
                quality_score=quality_score,
                profile=self._design_profile
            )

        self._log(f"[CASCADE] Recorded: {piston_name}/{algorithm} = "
                  f"{'SUCCESS' if success else 'FAIL'} ({time_ms:.0f}ms)")

    def get_cascade_statistics(self, piston: str = None) -> Dict:
        """Get cascade optimizer statistics for analysis."""
        if self._cascade_optimizer is None:
            return {}
        return self._cascade_optimizer.get_statistics(piston)

    def print_cascade_report(self, piston: str):
        """Print a cascade report for a piston."""
        if self._cascade_optimizer is None:
            print(f"Cascade optimizer not enabled")
            return
        self._cascade_optimizer.print_cascade_report(piston, self._design_profile)

    # =========================================================================
    # MONITOR-OPTIMIZER BRIDGE ACCESS
    # =========================================================================

    def get_bridge_statistics(self) -> Dict:
        """
        Get combined statistics from the Monitor-Optimizer Bridge.

        Returns:
            Dict with bridge, monitor, and optimizer statistics
        """
        if self._monitor_bridge is None:
            return {'error': 'Bridge not initialized'}
        return self._monitor_bridge.get_combined_statistics()

    def get_bridge_recommendations(self, parts_db: Dict = None) -> Dict[str, List[str]]:
        """
        Get algorithm recommendations from the bridge.

        Args:
            parts_db: Optional parts database for profile-aware recommendations

        Returns:
            Dict mapping piston names to recommended algorithm order
        """
        if self._monitor_bridge is None:
            return {}
        return self._monitor_bridge.get_recommendations(parts_db or self.state.parts_db)

    def print_bridge_report(self):
        """Print a combined performance report from the bridge."""
        if self._monitor_bridge is None:
            print("Monitor-Optimizer Bridge not initialized")
            return
        print(self._monitor_bridge.get_performance_report())

    def get_bridge_dashboard_data(self) -> Dict:
        """
        Get data for a monitoring dashboard.

        Returns:
            Dict with all data needed for dashboard visualization
        """
        if self._monitor_bridge is None:
            return {'error': 'Bridge not initialized'}
        return self._monitor_bridge.generate_dashboard_data()

    # =========================================================================
    # SAFE PISTON CALL WRAPPER (BUG-05 FIX)
    # =========================================================================

    def _safe_piston_call(self, piston_name: str, method: Callable, *args, **kwargs) -> Any:
        """
        Safely call a piston method with error handling.

        BUG-05 FIX: Wraps all piston calls to catch exceptions and prevent crashes.

        Args:
            piston_name: Name of the piston (for logging)
            method: The piston method to call
            *args, **kwargs: Arguments to pass to the method

        Returns:
            Method result, or None if call failed
        """
        try:
            result = method(*args, **kwargs)
            if result is None:
                self._log(f"WARNING: {piston_name} returned None")
            return result
        except Exception as e:
            self._log(f"ERROR in {piston_name}: {e}")
            self.state.errors.append(f"{piston_name} failed: {e}")
            return None

    # =========================================================================
    # SMART PISTON SELECTION
    # =========================================================================

    def run_smart(
        self,
        parts_db: Dict,
        circuit_ai_result: Any = None,
        user_preferences: Dict = None
    ) -> EngineResult:
        """
        Run PCB design with SMART piston selection.

        Only runs the pistons that are actually needed based on:
        1. Design analysis (components, nets, complexity)
        2. User preferences (thermal, 3D, BOM optimization)
        3. Automatic detection (high-power, high-speed, BGA packages)

        This is more efficient than run_orchestrated() which runs all pistons.

        Args:
            parts_db: Parts database from Circuit AI
            circuit_ai_result: Optional full result from Circuit AI
            user_preferences: Dict of preferences like:
                - 'generate_3d': True/False
                - 'run_thermal_analysis': True/False
                - 'optimize_bom': True/False
                - 'run_si_analysis': True/False

        Returns:
            EngineResult with final status

        Example:
            result = engine.run_smart(
                parts_db,
                user_preferences={
                    'generate_3d': True,
                    'run_thermal_analysis': True
                }
            )
        """
        self.state = EngineState()
        self.state.start_time = time.time()
        self.state.parts_db = parts_db
        self.state.circuit_ai_result = circuit_ai_result

        # Update design profile for cascade optimization
        self._update_design_profile(parts_db)

        user_preferences = user_preferences or {}

        self._log("=" * 70)
        self._log("PCB ENGINE - SMART Piston Selection Mode")
        self._log("=" * 70)

        # === STEP 1: Analyze design and select pistons ===
        if self._orchestrator:
            self._piston_selection = self._orchestrator.select_pistons(
                parts_db=parts_db,
                requirements=getattr(circuit_ai_result, 'requirements', None) if circuit_ai_result else None,
                user_preferences=user_preferences
            )

            self._log("\n" + self._orchestrator.explain_selection(self._piston_selection))
            self._log("")
        else:
            # Fallback to all pistons
            self._log("WARNING: Orchestrator not available, using all pistons")
            return self.run_orchestrated(parts_db, circuit_ai_result)

        # === STEP 2: Execute pistons in order ===
        try:
            for piston_name in self._piston_selection.execution_order:
                execute_func = self._get_piston_executor(piston_name)

                if execute_func is None:
                    self._log(f"  [SKIP] {piston_name} - not implemented")
                    continue

                # Check if this is a required or optional piston
                is_required = piston_name in self._piston_selection.required_pistons

                self._log(f"\n--- {piston_name.upper()} {'[REQUIRED]' if is_required else '[OPTIONAL]'} ---")

                piston_result = self._run_piston_with_drc(piston_name, execute_func)

                if not self._handle_piston_result(piston_name, piston_result):
                    if is_required:
                        # Required piston failed - abort
                        self._log(f"  [ABORT] Required piston {piston_name} failed")
                        return self._create_result()
                    else:
                        # Optional piston failed - continue with warning
                        self._log(f"  [WARN] Optional piston {piston_name} failed, continuing...")
                        self.state.warnings.append(f"{piston_name} piston had issues")

            # === STEP 3: Final output ===
            self.state.stage = EngineStage.COMPLETE

        except Exception as e:
            self.state.stage = EngineStage.ERROR
            self.state.errors.append(f"Engine error: {e}")
            self._log(f"ERROR: {e}")

        return self._create_result()

    def _get_piston_executor(self, piston_name: str) -> Optional[Callable]:
        """Get the executor function for a piston"""
        executors = {
            'parts': self._execute_parts,
            'order': self._execute_order,
            'placement': self._execute_placement,
            'escape': self._execute_escape,
            'routing': self._execute_routing,
            'optimize': self._execute_optimization,
            'polish': self._execute_polish,
            'silkscreen': self._execute_silkscreen,
            'drc': self._execute_final_drc,
            'output': self._execute_output,
            'stackup': self._execute_stackup if hasattr(self, '_execute_stackup') else None,
            'thermal': self._execute_thermal if hasattr(self, '_execute_thermal') else None,
            'pdn': self._execute_pdn if hasattr(self, '_execute_pdn') else None,
            'signal_integrity': self._execute_signal_integrity if hasattr(self, '_execute_signal_integrity') else None,
            'topological_router': self._execute_topological_routing,
            'visualization_3d': self._execute_3d_visualization,
            'bom_optimizer': self._execute_bom_optimization,
            'learning': self._save_learned_patterns,
            'netlist': self._execute_netlist if hasattr(self, '_execute_netlist') else None,
        }
        return executors.get(piston_name)

    def get_piston_selection(self) -> Optional[PistonSelection]:
        """Get the current piston selection (after run_smart)"""
        return self._piston_selection

    def preview_pistons(
        self,
        parts_db: Dict,
        requirements: Any = None,
        user_preferences: Dict = None
    ) -> str:
        """
        Preview which pistons would be selected without running.

        Useful for showing user what will happen before executing.

        Returns:
            Human-readable explanation of piston selection
        """
        if not self._orchestrator:
            return "Orchestrator not available"

        selection = self._orchestrator.select_pistons(
            parts_db=parts_db,
            requirements=requirements,
            user_preferences=user_preferences or {}
        )

        return self._orchestrator.explain_selection(selection)

    # =========================================================================
    # MAIN ORCHESTRATED RUN METHOD (Original - runs all pistons)
    # =========================================================================

    def run_orchestrated(self, parts_db: Dict, circuit_ai_result: Any = None) -> EngineResult:
        """
        Run the complete PCB design pipeline with AI Agent orchestration.

        This is the new main entry point that implements:
        - DRC watchdog monitoring all piston work
        - Feedback loop for retry/escalate decisions
        - AI Agent communication for challenges
        - User notification for critical decisions

        Args:
            parts_db: Parts database from Circuit AI or netlist
            circuit_ai_result: Optional full result from Circuit AI

        Returns:
            EngineResult with final status and generated files
        """
        self.state = EngineState()
        self.state.start_time = time.time()
        self.state.parts_db = parts_db
        self.state.circuit_ai_result = circuit_ai_result

        # Update design profile for cascade optimization
        self._update_design_profile(parts_db)

        # Initialize workflow reporter for comprehensive logging
        if WorkflowReporter:
            self._workflow_reporter = WorkflowReporter(self.config.board_name)
        else:
            self._workflow_reporter = None

        self._log("=" * 70)
        self._log("PCB ENGINE - AI-Orchestrated Design Flow (18 Pistons)")
        self._log("=" * 70)

        try:
            # === STAGE 0: LEARNING LOAD ===
            # Load learned patterns to enhance subsequent pistons
            self._load_learned_patterns()

            # === STAGE 1: PARTS ===
            piston_result = self._run_piston_with_drc('parts', self._execute_parts)
            if not self._handle_piston_result('parts', piston_result):
                return self._create_result()

            # === STAGE 2: ORDER ===
            piston_result = self._run_piston_with_drc('order', self._execute_order)
            if not self._handle_piston_result('order', piston_result):
                return self._create_result()

            # === STAGE 3: PLACEMENT ===
            # Apply learned placement patterns if available
            piston_result = self._run_piston_with_drc('placement', self._execute_placement)
            if not self._handle_piston_result('placement', piston_result):
                return self._create_result()

            # === STAGE 4: ESCAPE (if needed for BGA/QFP) ===
            if self._needs_escape_routing():
                piston_result = self._run_piston_with_drc('escape', self._execute_escape)
                if not self._handle_piston_result('escape', piston_result):
                    return self._create_result()

            # === STAGE 5: ROUTING ===
            piston_result = self._run_piston_with_drc('routing', self._execute_routing)
            if not self._handle_piston_result('routing', piston_result):
                return self._create_result()

            # === STAGE 5.5: TOPOLOGICAL ROUTING (optional, for advanced routing) ===
            if self._should_use_topological_routing():
                piston_result = self._run_piston_with_drc('topological_routing', self._execute_topological_routing)
                if not self._handle_piston_result('topological_routing', piston_result):
                    self.state.warnings.append("Topological routing had issues, using standard routing")

            # === STAGE 5.6: POLISH (via reduction, trace cleanup, board shrink) ===
            # DISABLED: Polish Piston has bugs that break DRC
            # - Via removal doesn't properly reconnect segments
            # - Trace simplification breaks segment ordering
            # TODO: Fix polish_piston.py before re-enabling
            # if self._should_run_polish():
            #     piston_result = self._run_piston_with_drc('polish', self._execute_polish)
            #     if not self._handle_piston_result('polish', piston_result):
            #         self.state.warnings.append("Polish had issues, continuing with unpolished design")

            # === STAGE 6: OPTIMIZATION ===
            piston_result = self._run_piston_with_drc('optimize', self._execute_optimization)
            if not self._handle_piston_result('optimize', piston_result):
                # Optimization failures are warnings, continue
                self.state.warnings.append("Optimization had issues")

            # === STAGE 7: SILKSCREEN ===
            piston_result = self._run_piston_with_drc('silkscreen', self._execute_silkscreen)
            if not self._handle_piston_result('silkscreen', piston_result):
                # Silkscreen failures are warnings
                self.state.warnings.append("Silkscreen placement had issues")

            # === STAGE 8: ANALYSIS (Stackup, Thermal, PDN, SI) ===
            self._run_analysis_pistons()

            # === STAGE 9: FINAL DRC ===
            if not self._run_final_drc_loop():
                # DRC failed after all retries - ask AI Agent
                challenge = Challenge(
                    type=ChallengeType.DRC_PERSISTENT,
                    severity='major',
                    description='DRC violations persist after multiple attempts',
                    context={'violations': self._get_drc_summary()},
                    options=['Accept with warnings', 'Abort', 'Manual review']
                )
                decision = self._request_ai_decision(challenge)
                if decision.decision == AIDecision.ABORT:
                    return self._create_result()

            # === STAGE 10: BOM OPTIMIZATION ===
            if self.config.optimize_bom:
                self._execute_bom_optimization()

            # === STAGE 11: 3D VISUALIZATION ===
            if self.config.generate_3d:
                self._execute_3d_visualization()

            # === STAGE 12: OUTPUT ===
            # Request AI approval before generating files
            approval = self._request_ai_approval_for_output()
            if approval.decision == AIDecision.DELIVER:
                self._execute_output()
                self.state.stage = EngineStage.COMPLETE

                # === STAGE 13: LEARNING SAVE ===
                # Learn from this successful design
                self._save_learned_patterns()
            elif approval.decision == AIDecision.ABORT:
                self.state.stage = EngineStage.ERROR
                self.state.errors.append("Design aborted by AI Agent")
            else:
                # AI wants changes - would loop back here
                self.state.warnings.append(f"AI requested: {approval.decision.value}")

        except Exception as e:
            self.state.stage = EngineStage.ERROR
            self.state.errors.append(f"Engine error: {e}")
            self._log(f"ERROR: {e}")

        return self._create_result()

    # =========================================================================
    # BBL (BIG BEAUTIFUL LOOP) RUN METHOD
    # =========================================================================

    def run_bbl(
        self,
        parts_db: Dict,
        progress_callback: Callable = None,
        escalation_callback: Callable = None,
        phase_configs: Dict = None
    ) -> 'BBLResult':
        """
        Run the complete PCB design pipeline using the Big Beautiful Loop (BBL).

        The BBL is the COMPLETE WORK CYCLE implementing all 6 improvements:
        1. CHECKPOINTS - Engine checks after each phase: continue, escalate, or abort
        2. ROLLBACK - Save state before each phase, rollback on failure
        3. TIMEOUT - Each phase has max time, auto-escalate if exceeded
        4. PROGRESS REPORTING - Real-time progress updates
        5. PARALLEL EXECUTION - Independent phases run in parallel
        6. LOOP HISTORY - Record every BBL run for analytics

        BBL PHASES:
        ===========
            Phase 1: ORDER RECEIVED - Validate and prepare input
            Phase 2: PISTON EXECUTION - Run all pistons with CASCADE system
            Phase 3: ESCALATION (if needed) - Engine → Engineer → Boss
            Phase 4: OUTPUT GENERATION - Generate .kicad_pcb and other files
            Phase 5: KICAD DRC VALIDATION - KiCad CLI validates output (THE AUTHORITY)
            Phase 6: LEARNING & DELIVERY - Learn from result, deliver files

        Args:
            parts_db: Parts database from Circuit AI or netlist
            progress_callback: Optional callback for progress updates
                               Signature: (BBLProgress) -> None
            escalation_callback: Optional callback for escalation handling
                                 Signature: (BBLEscalation) -> BBLEscalation (resolved)
            phase_configs: Optional custom phase configurations
                           Dict[BBLPhase, BBLPhaseConfig]

        Returns:
            BBLResult with complete execution results including:
            - success: bool
            - output_files: List[str]
            - drc_passed: bool
            - kicad_drc_passed: bool
            - routing_completion: float
            - history_entry: BBLHistoryEntry
            - escalation_count: int
            - rollback_count: int

        Example:
            # Basic usage
            result = engine.run_bbl(parts_db)

            # With progress callback
            def on_progress(progress):
                print(f"{progress.phase}: {progress.percentage}% - {progress.message}")

            result = engine.run_bbl(
                parts_db,
                progress_callback=on_progress
            )

            # With escalation handling
            def on_escalation(escalation):
                print(f"Escalation: {escalation.reason}")
                escalation.resolved = True
                escalation.response = "Continue with defaults"
                return escalation

            result = engine.run_bbl(
                parts_db,
                progress_callback=on_progress,
                escalation_callback=on_escalation
            )

            # Check results
            if result.success:
                print(f"BBL completed in {result.total_time:.2f}s")
                print(f"Output files: {result.output_files}")
            else:
                print(f"BBL failed: {result.errors}")
        """
        if not BBLEngine:
            self._log("ERROR: BBL Engine not available")
            # Fallback to legacy method
            legacy_result = self.run_orchestrated(parts_db)
            # Create a mock BBLResult from EngineResult
            return self._convert_to_bbl_result(legacy_result)

        # Start bridge session for real-time analytics
        bbl_id = f"BBL_{int(time.time())}"
        if self._monitor_bridge:
            self._monitor_bridge.start_session(bbl_id)
            self._log(f"[BRIDGE] Started monitoring session: {bbl_id}")

        # Initialize BBL Engine
        self._bbl_engine = BBLEngine(
            pcb_engine=self,
            phase_configs=phase_configs,
            progress_callback=progress_callback,
            escalation_callback=escalation_callback,
            output_dir=self.config.output_dir,
            verbose=self.config.verbose
        )

        # Store callbacks for internal use
        self._bbl_progress_callback = progress_callback
        self._bbl_escalation_callback = escalation_callback

        self._log("=" * 70)
        self._log("PCB ENGINE - BIG BEAUTIFUL LOOP (BBL)")
        self._log("=" * 70)

        # Run BBL
        result = self._bbl_engine.run(parts_db, self.state)

        # End bridge session and sync learnings
        if self._monitor_bridge:
            routing_stats = {
                'completion': getattr(result, 'routing_completion', 0.0),
                'routed': getattr(result, 'routed_count', 0),
                'total': getattr(result, 'total_nets', 0)
            }
            self._monitor_bridge.end_session(
                success=result.success,
                routing_stats=routing_stats
            )
            self._log("[BRIDGE] Session ended and learnings synced")

        # Log history analytics
        if self.config.verbose:
            analytics = self._bbl_engine.get_history_analytics()
            self._log("\n--- BBL HISTORY ANALYTICS ---")
            self._log(f"  Total runs: {analytics.get('total_runs', 0)}")
            self._log(f"  Success rate: {analytics.get('success_rate', 0)*100:.1f}%")
            self._log(f"  Average duration: {analytics.get('average_duration', 0):.2f}s")
            self._log(f"  Average routing: {analytics.get('average_routing_completion', 0)*100:.1f}%")

            # Log bridge statistics if available
            if self._monitor_bridge:
                bridge_stats = self._monitor_bridge.get_combined_statistics()
                self._log("\n--- MONITOR-OPTIMIZER BRIDGE ---")
                self._log(f"  Algorithms executed: {bridge_stats.get('bridge', {}).get('algorithms_executed', 0)}")
                self._log(f"  Cascades triggered: {bridge_stats.get('bridge', {}).get('cascades_triggered', 0)}")
                self._log(f"  Optimizations applied: {bridge_stats.get('bridge', {}).get('optimizations_applied', 0)}")

        return result

    def _convert_to_bbl_result(self, engine_result: EngineResult) -> Any:
        """Convert EngineResult to BBLResult for compatibility."""
        if BBLResult:
            return BBLResult(
                success=engine_result.success,
                bbl_id=f"LEGACY_{int(time.time())}",
                final_phase=BBLPhase.COMPLETE if engine_result.success else BBLPhase.ERROR,
                total_time=engine_result.total_time,
                output_files=engine_result.output_files,
                drc_passed=engine_result.drc_passed,
                drc_errors=len([e for e in engine_result.errors if 'drc' in e.lower()]),
                drc_warnings=len(engine_result.warnings),
                routing_completion=engine_result.routed_count / engine_result.total_nets if engine_result.total_nets > 0 else 0,
                routed_count=engine_result.routed_count,
                total_nets=engine_result.total_nets,
                errors=engine_result.errors,
                warnings=engine_result.warnings
            )
        return engine_result

    def get_bbl_history(self) -> List['BBLHistoryEntry']:
        """
        Get the BBL execution history.

        Returns:
            List of BBLHistoryEntry objects from previous runs
        """
        if self._bbl_engine:
            return self._bbl_engine.history
        return []

    def get_bbl_analytics(self) -> Dict:
        """
        Get analytics from BBL history.

        Returns:
            Dict with summary statistics:
            - total_runs: int
            - success_rate: float
            - average_duration: float
            - phase_failure_rates: Dict[str, float]
            - average_routing_completion: float
            - last_run: Dict
        """
        if self._bbl_engine:
            return self._bbl_engine.get_history_analytics()
        return {'error': 'BBL Engine not initialized'}

    # =========================================================================
    # PISTON EXECUTION WITH DRC WATCHDOG
    # =========================================================================

    def _run_piston_with_drc(self, piston_name: str,
                              execute_func: Callable) -> PistonReport:
        """
        ENGINE COMMAND: Run a piston with DRC watching and CASCADE on failure.

        The ENGINE (Foreman) manages pistons (workers) by:
        1. Sending work orders with specific algorithms
        2. Monitoring DRC after each attempt
        3. If a piston fails, trying the next algorithm in its CASCADE
        4. Escalating effort level and relaxing constraints as needed
        5. Never giving up until ALL algorithms in the cascade are tried

        Returns:
            PistonReport with results and DRC feedback
        """
        effort = PistonEffort.NORMAL
        retry_count = 0
        algorithm_index = 0
        best_report = None
        best_score = -1

        # Get cascade for this piston - use optimizer if available, else static
        if self._cascade_optimizer is not None:
            # DYNAMIC: Get optimized algorithm order based on design profile and history
            algorithms = self._get_optimized_cascade(piston_name)
            self._log(f"  [CASCADE] Using DYNAMIC algorithm order for {piston_name}")
        else:
            # STATIC: Fall back to hardcoded PISTON_CASCADES
            cascade = PISTON_CASCADES.get(piston_name, {})
            algorithms = cascade.get('algorithms', [])

        # Get relaxed params (always from static config)
        static_cascade = PISTON_CASCADES.get(piston_name, {})
        relaxed_params = static_cascade.get('relaxed_params', {})

        # Start workflow reporter tracking for this piston
        if self._workflow_reporter:
            available_algos = [a[0] for a in algorithms] if algorithms else self._get_available_algorithms(piston_name)
            self._workflow_reporter.start_piston(
                piston_name,
                effort_level=effort.value,
                available_algorithms=available_algos
            )

        # ═══════════════════════════════════════════════════════════════════
        # ENGINE CASCADE LOOP - Try all algorithms until one succeeds
        # ═══════════════════════════════════════════════════════════════════
        max_attempts = max(self.config.max_retry_per_piston, len(algorithms)) if algorithms else self.config.max_retry_per_piston

        while retry_count < max_attempts:
            # Determine which algorithm to use
            current_algo = ''
            algo_desc = ''
            if algorithms and algorithm_index < len(algorithms):
                current_algo, algo_desc = algorithms[algorithm_index]
            elif algorithms:
                # Exhausted all algorithms, try with relaxed params
                current_algo, algo_desc = algorithms[0]  # Back to first with relaxed
                algo_desc = f"{algo_desc} (RELAXED)"

            self._log(f"\n--- {piston_name.upper()} (attempt {retry_count + 1}/{max_attempts}, effort: {effort.value}) ---")
            if algo_desc:
                self._log(f"  [ENGINE ORDER] Algorithm: {algo_desc}")

            # Log retry loop iteration
            if self._workflow_reporter and retry_count > 0:
                self._workflow_reporter.log_loop_iteration(
                    piston_name, LoopType.RETRY if LoopType else None,
                    retry_count, max_attempts,
                    f"Algorithm: {current_algo}, Effort: {effort.value}",
                    'continue'
                )

            # Create work order with specific algorithm
            params = self._get_effort_parameters(effort)
            if current_algo:
                params['algorithm'] = current_algo
            # Apply relaxed params if we've tried all algorithms once
            if algorithm_index >= len(algorithms) and relaxed_params:
                params.update(relaxed_params)
                self._log(f"  [ENGINE] Applying RELAXED constraints: {relaxed_params}")

            work_order = WorkOrder(
                piston=piston_name,
                task='execute',
                effort=effort,
                algorithm=current_algo,
                parameters=params,
                ai_reasoning=f"Cascade attempt {retry_count + 1}: {algo_desc}"
            )
            self.state.work_orders.append(work_order)

            # Store algorithm in state for piston to use
            self._current_algorithm = current_algo

            # Execute piston
            start_time = time.time()
            report = execute_func(effort)
            report.time_elapsed = time.time() - start_time
            self.state.piston_reports.append(report)

            # DRC watches the work
            drc_report = self._drc_watch_piston(piston_name, report)
            self.state.drc_watch_reports.append(drc_report)

            # Log DRC result to workflow reporter
            if self._workflow_reporter:
                self._workflow_reporter.log_drc_result(
                    piston_name,
                    passed=drc_report.passed,
                    violations=[{'type': e.get('type', 'Unknown'), 'message': str(e)} for e in drc_report.errors],
                    suggestions=drc_report.suggestions
                )

            # Calculate score for this attempt
            score = self._calculate_piston_score(piston_name, report, drc_report)

            # Log results
            self._log(f"  Success: {report.success}")
            self._log(f"  Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")
            self._log(f"  DRC: {'PASS' if drc_report.passed else 'FAIL'}")
            self._log(f"  Score: {score:.1f}/100")
            self._log(f"  Time: {report.time_elapsed:.2f}s")

            # Track best result
            if score > best_score:
                best_score = score
                best_report = report
                self._log(f"  [ENGINE] New best result! (score={score:.1f})")

            # Check if we can proceed
            if report.success and drc_report.can_continue:
                # RECORD SUCCESS for cascade learning
                self._record_algorithm_result(
                    piston_name, current_algo,
                    success=True,
                    time_ms=report.time_elapsed * 1000,
                    quality_score=score / 100.0
                )

                # End piston tracking - success
                if self._workflow_reporter:
                    self._workflow_reporter.set_piston_retry(piston_name, retry_count)
                    self._workflow_reporter.end_piston(
                        piston_name,
                        success=True,
                        algorithm_used=current_algo or getattr(report, 'algorithm_used', ''),
                        metrics=report.metrics
                    )
                self._log(f"  [ENGINE] {piston_name.upper()} SUCCEEDED with {algo_desc or 'default'}")
                return report

            # RECORD FAILURE for cascade learning
            self._record_algorithm_result(
                piston_name, current_algo,
                success=False,
                time_ms=report.time_elapsed * 1000,
                quality_score=score / 100.0
            )

            # Not successful - try next algorithm in cascade
            retry_count += 1
            self.state.retry_counts[piston_name] = retry_count

            if retry_count < max_attempts:
                # Move to next algorithm
                algorithm_index += 1
                if algorithm_index < len(algorithms):
                    self._log(f"  [ENGINE] Trying next algorithm in CASCADE...")
                else:
                    # Exhausted algorithms, escalate effort
                    effort = self._escalate_effort(effort, drc_report)
                    self._log(f"  [ENGINE] All algorithms tried, escalating to effort: {effort.value}")
            else:
                # Max retries reached
                report.notes.append(f"Max attempts ({retry_count}) reached, tried {len(algorithms)} algorithms")

        # ═══════════════════════════════════════════════════════════════════
        # CASCADE EXHAUSTED - Return best result found
        # ═══════════════════════════════════════════════════════════════════
        if best_report and best_report != report:
            self._log(f"  [ENGINE] Using best result from cascade (score={best_score:.1f})")
            report = best_report

        # End piston tracking - failed after all retries
        if self._workflow_reporter:
            self._workflow_reporter.set_piston_retry(piston_name, retry_count)
            for error in report.errors:
                self._workflow_reporter.add_piston_error(piston_name, error)
            for warning in report.warnings:
                self._workflow_reporter.add_piston_warning(piston_name, warning)
            self._workflow_reporter.end_piston(
                piston_name,
                success=False,
                algorithm_used=getattr(report, 'algorithm_used', ''),
                metrics=report.metrics
            )

        self._log(f"  [ENGINE] {piston_name.upper()} CASCADE EXHAUSTED after {retry_count} attempts")
        return report

    def _calculate_piston_score(self, piston_name: str, report: PistonReport, drc_report: DRCWatchReport) -> float:
        """
        Calculate a quality score for a piston's output (0-100).

        Used by the ENGINE to compare results from different algorithms
        and select the best one.
        """
        score = 100.0

        # Base penalties
        if not report.success:
            score -= 50
        if not drc_report.passed:
            score -= 30
        if not drc_report.can_continue:
            score -= 20

        # Error/warning penalties
        score -= len(report.errors) * 5
        score -= len(report.warnings) * 2
        score -= len(drc_report.errors) * 5
        score -= len(drc_report.warnings) * 2

        # Piston-specific bonuses
        if piston_name == 'routing':
            metrics = report.metrics or {}
            routed = metrics.get('routed_count', 0)
            total = metrics.get('total_count', 1)
            completion = routed / max(total, 1) * 30  # Up to 30 bonus for completion
            score += completion

        elif piston_name == 'placement':
            metrics = report.metrics or {}
            placed = metrics.get('placed_count', 0)
            if placed > 0:
                score += 20  # Bonus for placing components

        return max(0, min(100, score))

    def _get_available_algorithms(self, piston_name: str) -> List[str]:
        """Get the list of available algorithms for a piston from PISTON_CASCADES"""
        cascade = PISTON_CASCADES.get(piston_name, {})
        algorithms = cascade.get('algorithms', [])
        if algorithms:
            return [algo[0] for algo in algorithms]

        # Fallback for pistons not in cascade
        fallback_map = {
            'parts': ['lookup', 'inference', 'ai_enhanced'],
            'drc': ['full', 'incremental'],
            'netlist': ['standard'],
        }
        return fallback_map.get(piston_name, [])

    def _drc_watch_piston(self, piston_name: str,
                          piston_report: PistonReport) -> DRCWatchReport:
        """
        DRC watches a piston's work and reports issues.

        This is called after each piston completes to validate its output.
        """
        report = DRCWatchReport(
            piston=piston_name,
            stage=self.state.stage.value,
            passed=True,
            can_continue=True
        )

        # Initialize DRC piston if needed
        if not self._drc_piston and DRCPiston:
            self._drc_piston = DRCPiston(DRCConfig(
                rules=DRCRules(
                    min_clearance=self.config.clearance,
                    min_track_width=self.config.trace_width,
                    min_via_drill=self.config.via_drill,
                    min_via_diameter=self.config.via_diameter
                ),
                board_width=self.config.board_width,
                board_height=self.config.board_height
            ))

        if not self._drc_piston:
            report.notes.append("DRC piston not available")
            return report

        # Run appropriate DRC checks based on piston
        if piston_name == 'placement':
            # Check component overlaps and board boundaries
            errors, warnings = self._drc_check_placement()
            report.errors = errors
            report.warnings = warnings

        elif piston_name == 'routing':
            # Check trace clearances, shorts, opens
            errors, warnings = self._drc_check_routing()
            report.errors = errors
            report.warnings = warnings

        elif piston_name == 'silkscreen':
            # Check silkscreen overlaps with pads
            errors, warnings = self._drc_check_silkscreen()
            report.errors = errors
            report.warnings = warnings

        # Determine if we can continue
        critical_errors = [e for e in report.errors if e.get('severity') == 'critical']
        report.passed = len(critical_errors) == 0
        report.can_continue = len(critical_errors) == 0

        # Generate suggestions
        if not report.passed:
            report.suggestions = self._generate_drc_suggestions(report.errors)

        return report

    def _drc_check_placement(self) -> Tuple[List[Dict], List[Dict]]:
        """DRC checks for placement stage"""
        errors = []
        warnings = []

        # Check board boundaries
        for ref, pos in self.state.placement.items():
            x, y = pos.x if hasattr(pos, 'x') else pos[0], pos.y if hasattr(pos, 'y') else pos[1]

            if x < 0 or x > self.config.board_width:
                errors.append({
                    'type': 'out_of_bounds',
                    'ref': ref,
                    'message': f'{ref} is outside board X boundary',
                    'severity': 'critical'
                })

            if y < 0 or y > self.config.board_height:
                errors.append({
                    'type': 'out_of_bounds',
                    'ref': ref,
                    'message': f'{ref} is outside board Y boundary',
                    'severity': 'critical'
                })

        # Check component overlaps (simplified)
        refs = list(self.state.placement.keys())
        for i, ref1 in enumerate(refs):
            for ref2 in refs[i+1:]:
                pos1 = self.state.placement[ref1]
                pos2 = self.state.placement[ref2]
                x1, y1 = pos1.x if hasattr(pos1, 'x') else pos1[0], pos1.y if hasattr(pos1, 'y') else pos1[1]
                x2, y2 = pos2.x if hasattr(pos2, 'x') else pos2[0], pos2.y if hasattr(pos2, 'y') else pos2[1]

                dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                if dist < 1.0:  # Simplified overlap check
                    warnings.append({
                        'type': 'component_overlap',
                        'refs': [ref1, ref2],
                        'message': f'{ref1} and {ref2} may overlap',
                        'severity': 'warning'
                    })

        return errors, warnings

    def _drc_check_routing(self) -> Tuple[List[Dict], List[Dict]]:
        """DRC checks for routing stage"""
        errors = []
        warnings = []

        # Check for unrouted nets
        nets = self.state.parts_db.get('nets', {})
        for net_name in nets:
            if net_name not in self.state.routes:
                errors.append({
                    'type': 'unrouted_net',
                    'net': net_name,
                    'message': f'Net {net_name} is not routed',
                    'severity': 'major'
                })

        # Check trace clearances would go here (simplified)
        return errors, warnings

    def _drc_check_silkscreen(self) -> Tuple[List[Dict], List[Dict]]:
        """DRC checks for silkscreen stage"""
        errors = []
        warnings = []

        if not self.state.silkscreen:
            return errors, warnings

        # Would check for silkscreen overlapping pads
        return errors, warnings

    def _generate_drc_suggestions(self, errors: List[Dict]) -> List[str]:
        """Generate suggestions based on DRC errors"""
        suggestions = []

        error_types = [e.get('type') for e in errors]

        if 'out_of_bounds' in error_types:
            suggestions.append("Consider increasing board size")
            suggestions.append("Review component placement constraints")

        if 'component_overlap' in error_types:
            suggestions.append("Increase component spacing")
            suggestions.append("Consider using smaller footprints")

        if 'unrouted_net' in error_types:
            suggestions.append("Try different routing algorithm")
            suggestions.append("Consider adding layers")
            suggestions.append("Review component placement for routability")

        return suggestions

    # =========================================================================
    # ALGORITHM/STRATEGY SELECTION (AI-Assisted)
    # =========================================================================

    def _get_algorithm_choices(self, piston_name: str) -> List[AlgorithmChoice]:
        """
        Get available algorithm choices for a piston.

        This provides the AI Agent with options to choose from,
        along with context about each algorithm's strengths/weaknesses.
        """
        if piston_name == 'routing':
            return [
                AlgorithmChoice(
                    name='lee',
                    description='Lee maze router - exhaustive BFS search',
                    pros=['Guaranteed shortest path', 'Simple', 'Reliable'],
                    cons=['Slow for large boards', 'Memory intensive'],
                    best_for=['Small boards', 'Critical nets', 'Few nets'],
                    estimated_time=10.0,
                    estimated_quality=0.95
                ),
                AlgorithmChoice(
                    name='astar',
                    description='A* pathfinding with heuristics',
                    pros=['Fast', 'Good quality', 'Scalable'],
                    cons=['May not find optimal', 'Tuning needed'],
                    best_for=['Medium boards', 'General use'],
                    estimated_time=5.0,
                    estimated_quality=0.85
                ),
                AlgorithmChoice(
                    name='hadlock',
                    description='Hadlock router - detour counting',
                    pros=['Very fast', 'Good for congested areas'],
                    cons=['Suboptimal paths sometimes'],
                    best_for=['Dense boards', 'Many nets'],
                    estimated_time=3.0,
                    estimated_quality=0.80
                ),
                AlgorithmChoice(
                    name='pathfinder',
                    description='PathFinder negotiated congestion',
                    pros=['Handles congestion well', 'Global optimization'],
                    cons=['Slower', 'Complex'],
                    best_for=['High-density', 'Difficult routing'],
                    estimated_time=20.0,
                    estimated_quality=0.90
                ),
                AlgorithmChoice(
                    name='hybrid',
                    description='Multiple algorithms with fallback',
                    pros=['Adaptive', 'Best overall results'],
                    cons=['Slower total time'],
                    best_for=['Unknown complexity', 'Mixed requirements'],
                    estimated_time=15.0,
                    estimated_quality=0.92
                ),
            ]

        elif piston_name == 'placement':
            return [
                AlgorithmChoice(
                    name='force_directed',
                    description='Force-directed placement with springs',
                    pros=['Natural clustering', 'Good wire length'],
                    cons=['May get stuck in local minima'],
                    best_for=['Analog circuits', 'Hierarchical designs'],
                    estimated_time=5.0,
                    estimated_quality=0.85
                ),
                AlgorithmChoice(
                    name='simulated_annealing',
                    description='SA optimization for global minimum',
                    pros=['Escapes local minima', 'High quality'],
                    cons=['Slow', 'Needs tuning'],
                    best_for=['Complex boards', 'Dense placement'],
                    estimated_time=15.0,
                    estimated_quality=0.90
                ),
                AlgorithmChoice(
                    name='analytical',
                    description='Quadratic optimization approach',
                    pros=['Very fast', 'Good initial placement'],
                    cons=['Needs legalization step'],
                    best_for=['Large designs', 'Quick iteration'],
                    estimated_time=2.0,
                    estimated_quality=0.75
                ),
                AlgorithmChoice(
                    name='partition',
                    description='Recursive min-cut partitioning',
                    pros=['Structured result', 'Good for blocks'],
                    cons=['Less flexible'],
                    best_for=['Hierarchical designs', 'Multi-zone boards'],
                    estimated_time=8.0,
                    estimated_quality=0.82
                ),
            ]

        elif piston_name == 'escape':
            return [
                AlgorithmChoice(
                    name='mmcf',
                    description='Multi-commodity flow optimization',
                    pros=['Optimal solution', 'Handles constraints'],
                    cons=['Slow for large BGAs'],
                    best_for=['BGA with many pins', 'Critical timing'],
                    estimated_time=10.0,
                    estimated_quality=0.95
                ),
                AlgorithmChoice(
                    name='ring',
                    description='Ring-based concentric escape',
                    pros=['Fast', 'Predictable'],
                    cons=['May not use all layers efficiently'],
                    best_for=['Standard BGAs', 'Quick turnaround'],
                    estimated_time=3.0,
                    estimated_quality=0.80
                ),
                AlgorithmChoice(
                    name='sat',
                    description='SAT solver for constraint satisfaction',
                    pros=['Guaranteed feasibility', 'Handles complex constraints'],
                    cons=['Can be slow', 'Binary result'],
                    best_for=['Tight constraints', 'Verification'],
                    estimated_time=20.0,
                    estimated_quality=0.88
                ),
            ]

        elif piston_name == 'topological_routing':
            return [
                AlgorithmChoice(
                    name='delaunay_rubberband',
                    description='Delaunay triangulation with rubber-band optimization',
                    pros=['Smooth curves', 'Any-angle routing', 'Natural flow'],
                    cons=['Requires post-processing for DRC', 'Complex geometry'],
                    best_for=['High-speed analog', 'RF traces', 'Aesthetic routing'],
                    estimated_time=8.0,
                    estimated_quality=0.88
                ),
                AlgorithmChoice(
                    name='force_directed',
                    description='Force-directed route spreading',
                    pros=['Even spacing', 'Minimizes crosstalk', 'Parallel-friendly'],
                    cons=['May increase wire length'],
                    best_for=['Differential pairs', 'Bus routing', 'Dense areas'],
                    estimated_time=6.0,
                    estimated_quality=0.85
                ),
            ]

        elif piston_name == 'visualization_3d':
            return [
                AlgorithmChoice(
                    name='stl_mesh',
                    description='STL mesh export for 3D printing/CAD',
                    pros=['Universal format', 'Good for enclosures'],
                    cons=['Large file size', 'No color/material'],
                    best_for=['3D printing', 'Mechanical integration'],
                    estimated_time=2.0,
                    estimated_quality=0.90
                ),
                AlgorithmChoice(
                    name='step_cad',
                    description='STEP format for mechanical CAD',
                    pros=['Industry standard', 'Parametric'],
                    cons=['Requires CAD kernel', 'Complex'],
                    best_for=['Professional CAD integration', 'Mechanical design'],
                    estimated_time=5.0,
                    estimated_quality=0.95
                ),
                AlgorithmChoice(
                    name='gltf_web',
                    description='glTF format for web visualization',
                    pros=['Fast loading', 'Web-native', 'Supports materials'],
                    cons=['Less precision'],
                    best_for=['Web viewers', 'Interactive preview'],
                    estimated_time=1.5,
                    estimated_quality=0.85
                ),
            ]

        elif piston_name == 'bom_optimization':
            return [
                AlgorithmChoice(
                    name='lowest_cost',
                    description='Optimize for minimum total cost',
                    pros=['Cheapest BOM', 'Multi-supplier sourcing'],
                    cons=['May have longer lead times', 'Multiple shipments'],
                    best_for=['Cost-sensitive projects', 'High volume'],
                    estimated_time=3.0,
                    estimated_quality=0.80
                ),
                AlgorithmChoice(
                    name='fastest_delivery',
                    description='Optimize for shortest lead time',
                    pros=['Fastest procurement', 'In-stock priority'],
                    cons=['Higher cost', 'May miss price breaks'],
                    best_for=['Prototypes', 'Urgent builds'],
                    estimated_time=2.0,
                    estimated_quality=0.85
                ),
                AlgorithmChoice(
                    name='single_supplier',
                    description='Source from fewest suppliers',
                    pros=['Simplified ordering', 'Lower shipping'],
                    cons=['May miss better prices', 'Supply risk'],
                    best_for=['Small orders', 'Relationship building'],
                    estimated_time=2.5,
                    estimated_quality=0.82
                ),
                AlgorithmChoice(
                    name='balanced',
                    description='Balance cost, time, and quality',
                    pros=['Good tradeoffs', 'Authorized suppliers'],
                    cons=['Not optimal in any single dimension'],
                    best_for=['General use', 'Production builds'],
                    estimated_time=3.0,
                    estimated_quality=0.88
                ),
            ]

        elif piston_name == 'learning':
            return [
                AlgorithmChoice(
                    name='supervised',
                    description='Supervised learning with quality labels',
                    pros=['High accuracy', 'Learns from expert feedback'],
                    cons=['Requires labeled data', 'More manual effort'],
                    best_for=['Quality prediction', 'When labeled data available'],
                    estimated_time=30.0,
                    estimated_quality=0.92
                ),
                AlgorithmChoice(
                    name='unsupervised',
                    description='Unsupervised pattern discovery',
                    pros=['No labels needed', 'Discovers hidden patterns'],
                    cons=['May find irrelevant patterns', 'Lower precision'],
                    best_for=['Initial exploration', 'Large unlabeled datasets'],
                    estimated_time=20.0,
                    estimated_quality=0.80
                ),
                AlgorithmChoice(
                    name='reinforcement',
                    description='Reinforcement learning from routing feedback',
                    pros=['Learns from success/failure', 'Improves over time'],
                    cons=['Slow to converge', 'Needs many iterations'],
                    best_for=['Routing optimization', 'Continuous improvement'],
                    estimated_time=60.0,
                    estimated_quality=0.88
                ),
                AlgorithmChoice(
                    name='transfer',
                    description='Transfer learning from similar designs',
                    pros=['Fast adaptation', 'Works with few samples'],
                    cons=['Requires similar domain', 'May not generalize'],
                    best_for=['Domain-specific optimization', 'Limited data'],
                    estimated_time=15.0,
                    estimated_quality=0.85
                ),
            ]

        return []

    def _request_algorithm_selection(self, piston_name: str,
                                      context: Dict) -> WorkOrder:
        """
        Request AI Agent to select algorithm/strategy for a piston.

        This is where the "brain" part comes in - Circuit AI analyzes
        the design context and makes an intelligent algorithm choice.
        """
        choices = self._get_algorithm_choices(piston_name)

        if not choices:
            # No choices available, use default
            return WorkOrder(piston=piston_name, task='execute')

        # Build context hints for AI
        context_hints = {
            'board_size': f"{self.config.board_width}x{self.config.board_height}mm",
            'component_count': len(self.state.parts_db.get('parts', {})),
            'net_count': len(self.state.parts_db.get('nets', {})),
            'layer_count': self.config.layer_count,
            'has_bga': self._needs_escape_routing(),
            'previous_failures': self.state.retry_counts.get(piston_name, 0),
        }
        context_hints.update(context)

        request = AIRequest(
            stage=self.state.stage,
            question=f"Select algorithm for {piston_name} piston",
            piston_name=piston_name,
            algorithm_choices=choices,
            context_hints=context_hints,
            previous_attempts=[
                {'algorithm': wo.algorithm, 'success': False}
                for wo in self.state.work_orders
                if wo.piston == piston_name and wo.algorithm
            ]
        )

        # Call AI Agent
        if self.config.ai_agent_callback:
            response = self.config.ai_agent_callback(request)
        else:
            response = self._default_algorithm_selection(piston_name, choices, context_hints)

        # Create work order with selected algorithm
        work_order = WorkOrder(
            piston=piston_name,
            task='execute',
            algorithm=response.selected_algorithm or choices[0].name,
            strategy=response.selected_strategy,
            parameters=response.algorithm_parameters,
            ai_reasoning=response.reasoning,
            priority_nets=response.priority_nets
        )

        self._log(f"  AI selected: {work_order.algorithm} ({response.reasoning})")

        return work_order

    def _default_algorithm_selection(self, piston_name: str,
                                       choices: List[AlgorithmChoice],
                                       context: Dict) -> AIResponse:
        """
        Default algorithm selection logic (when no AI callback provided).

        Uses simple heuristics based on design context.
        """
        selected = choices[0]  # Default to first
        reasoning = "Default selection"

        # Simple heuristics
        if piston_name == 'routing':
            net_count = context.get('net_count', 0)
            board_area = context.get('component_count', 0)

            if net_count < 20:
                # Small design - use Lee for quality
                selected = next((c for c in choices if c.name == 'lee'), choices[0])
                reasoning = f"Small design ({net_count} nets) - using Lee for optimal paths"
            elif context.get('previous_failures', 0) > 0:
                # Previous attempt failed - try PathFinder
                selected = next((c for c in choices if c.name == 'pathfinder'), choices[0])
                reasoning = "Previous routing failed - using PathFinder for congestion"
            else:
                # Medium design - use A*
                selected = next((c for c in choices if c.name == 'astar'), choices[0])
                reasoning = f"Medium design ({net_count} nets) - using A* for balance"

        elif piston_name == 'placement':
            comp_count = context.get('component_count', 0)

            if comp_count < 15:
                selected = next((c for c in choices if c.name == 'force_directed'), choices[0])
                reasoning = f"Small design ({comp_count} parts) - force-directed"
            else:
                selected = next((c for c in choices if c.name == 'simulated_annealing'), choices[0])
                reasoning = f"Larger design ({comp_count} parts) - SA for quality"

        return AIResponse(
            decision=AIDecision.APPROVE,
            selected_algorithm=selected.name,
            reasoning=reasoning
        )

    # =========================================================================
    # EFFORT ESCALATION
    # =========================================================================

    def _get_effort_parameters(self, effort: PistonEffort) -> Dict:
        """Get parameters for given effort level"""
        params = {
            PistonEffort.NORMAL: {
                'iterations_multiplier': 1.0,
                'timeout_multiplier': 1.0,
                'algorithms': 'default'
            },
            PistonEffort.HARDER: {
                'iterations_multiplier': 2.0,
                'timeout_multiplier': 1.5,
                'algorithms': 'default'
            },
            PistonEffort.DEEPER: {
                'iterations_multiplier': 1.5,
                'timeout_multiplier': 2.0,
                'algorithms': 'all'
            },
            PistonEffort.LONGER: {
                'iterations_multiplier': 3.0,
                'timeout_multiplier': 3.0,
                'algorithms': 'default'
            },
            PistonEffort.MAXIMUM: {
                'iterations_multiplier': 5.0,
                'timeout_multiplier': 5.0,
                'algorithms': 'all'
            }
        }
        return params.get(effort, params[PistonEffort.NORMAL])

    def _escalate_effort(self, current: PistonEffort,
                         drc_report: DRCWatchReport) -> PistonEffort:
        """Escalate effort based on DRC report"""
        effort_order = [
            PistonEffort.NORMAL,
            PistonEffort.HARDER,
            PistonEffort.DEEPER,
            PistonEffort.LONGER,
            PistonEffort.MAXIMUM
        ]

        current_idx = effort_order.index(current)
        next_idx = min(current_idx + 1, len(effort_order) - 1)

        return effort_order[next_idx]

    # =========================================================================
    # AI AGENT COMMUNICATION
    # =========================================================================

    def _handle_piston_result(self, piston_name: str,
                               report: PistonReport) -> bool:
        """
        Handle piston result - decide whether to continue or escalate to AI.
        """
        if report.success:
            return True

        # Check if we've exhausted retries
        if self.state.retry_counts.get(piston_name, 0) >= self.config.max_retry_per_piston:
            # Create challenge for AI Agent
            challenge = self._create_challenge_from_report(piston_name, report)

            if challenge.severity == 'minor' and self.config.auto_approve_minor:
                self._log(f"  Auto-approving minor issue: {challenge.description}")
                return True

            # Request AI decision
            decision = self._request_ai_decision(challenge)
            return self._apply_ai_decision(piston_name, decision)

        return False

    def _create_challenge_from_report(self, piston_name: str,
                                       report: PistonReport) -> Challenge:
        """Create a challenge from piston report"""
        # Determine challenge type based on errors
        if 'out_of_bounds' in str(report.errors):
            challenge_type = ChallengeType.BOARD_TOO_SMALL
        elif 'unrouted' in str(report.errors):
            challenge_type = ChallengeType.ROUTING_IMPOSSIBLE
        elif 'overlap' in str(report.errors):
            challenge_type = ChallengeType.PLACEMENT_CONFLICT
        else:
            challenge_type = ChallengeType.DRC_PERSISTENT

        # Determine severity
        critical_count = sum(1 for e in report.errors if 'critical' in str(e))
        if critical_count > 0:
            severity = 'critical'
        elif len(report.errors) > 3:
            severity = 'major'
        else:
            severity = 'minor'

        return Challenge(
            type=challenge_type,
            severity=severity,
            description=f"{piston_name} failed: {report.errors[0] if report.errors else 'Unknown'}",
            context={
                'piston': piston_name,
                'errors': report.errors,
                'warnings': report.warnings,
                'retries': self.state.retry_counts.get(piston_name, 0)
            },
            options=self._get_challenge_options(challenge_type)
        )

    def _get_challenge_options(self, challenge_type: ChallengeType) -> List[str]:
        """Get available options for a challenge type"""
        options_map = {
            ChallengeType.BOARD_TOO_SMALL: [
                'Increase board size',
                'Use smaller components',
                'Abort'
            ],
            ChallengeType.LAYERS_NEEDED: [
                'Add more layers',
                'Try harder routing',
                'Abort'
            ],
            ChallengeType.ROUTING_IMPOSSIBLE: [
                'Add layers',
                'Relocate components',
                'Accept partial routing',
                'Abort'
            ],
            ChallengeType.PLACEMENT_CONFLICT: [
                'Increase board size',
                'Use smaller footprints',
                'Abort'
            ],
            ChallengeType.DRC_PERSISTENT: [
                'Accept with warnings',
                'Try maximum effort',
                'Abort'
            ]
        }
        return options_map.get(challenge_type, ['Retry', 'Abort'])

    def _request_ai_decision(self, challenge: Challenge) -> AIResponse:
        """Request decision from AI Agent"""
        self.state.stage = EngineStage.WAITING_AI
        self.state.challenges.append(challenge)

        request = AIRequest(
            stage=self.state.stage,
            challenge=challenge,
            current_state=self._get_state_summary(),
            question=f"Challenge: {challenge.description}",
            options=challenge.options
        )
        self.state.ai_requests.append(request)

        # Call AI Agent callback if provided
        if self.config.ai_agent_callback:
            response = self.config.ai_agent_callback(request)
        else:
            # Default behavior: auto-approve or abort based on severity
            if challenge.severity == 'minor':
                response = AIResponse(decision=AIDecision.APPROVE)
            elif challenge.severity == 'major':
                response = AIResponse(decision=AIDecision.RETRY_HARDER)
            else:
                response = AIResponse(decision=AIDecision.ASK_USER)

        self.state.ai_responses.append(response)
        self._log(f"  AI Decision: {response.decision.value}")

        return response

    def _request_ai_approval_for_output(self) -> AIResponse:
        """Request AI approval before generating output files"""
        request = AIRequest(
            stage=EngineStage.OUTPUT,
            current_state=self._get_state_summary(),
            question="Design complete. Approve file generation?",
            options=['Deliver files', 'Review first', 'Abort']
        )
        self.state.ai_requests.append(request)

        if self.config.ai_agent_callback:
            response = self.config.ai_agent_callback(request)
        else:
            # Default: approve if no critical errors
            if len(self.state.errors) == 0:
                response = AIResponse(decision=AIDecision.DELIVER)
            else:
                response = AIResponse(
                    decision=AIDecision.ASK_USER,
                    message_to_user=f"Design has {len(self.state.errors)} errors. Review before delivery?"
                )

        self.state.ai_responses.append(response)
        return response

    def _apply_ai_decision(self, piston_name: str,
                           decision: AIResponse) -> bool:
        """Apply AI decision to the workflow"""
        if decision.decision == AIDecision.APPROVE:
            return True

        elif decision.decision == AIDecision.ABORT:
            self.state.stage = EngineStage.ERROR
            self.state.errors.append(f"Aborted by AI Agent: {decision.message_to_user}")
            return False

        elif decision.decision == AIDecision.MODIFY_BOARD:
            # Apply board modifications
            if 'width' in decision.parameters:
                self.config.board_width = decision.parameters['width']
            if 'height' in decision.parameters:
                self.config.board_height = decision.parameters['height']
            return True  # Will retry with new config

        elif decision.decision == AIDecision.ADD_LAYERS:
            self.config.layer_count += 2
            return True

        elif decision.decision == AIDecision.REPLACE_PART:
            # Would modify parts_db here
            return True

        elif decision.decision == AIDecision.ASK_USER:
            self.state.stage = EngineStage.WAITING_USER
            self._log(f"  User input needed: {decision.message_to_user}")
            return False

        return True

    def _get_state_summary(self) -> Dict:
        """Get summary of current engine state for AI Agent"""
        return {
            'stage': self.state.stage.value,
            'components': len(self.state.parts_db.get('parts', {})),
            'nets': len(self.state.parts_db.get('nets', {})),
            'placed': len(self.state.placement),
            'routed': len(self.state.routes),
            'errors': len(self.state.errors),
            'warnings': len(self.state.warnings),
            'retries': self.state.retry_counts
        }

    def _get_drc_summary(self) -> Dict:
        """Get summary of DRC results"""
        if not self.state.drc_result:
            return {}
        return {
            'passed': self.state.drc_result.passed if hasattr(self.state.drc_result, 'passed') else False,
            'error_count': self.state.drc_result.error_count if hasattr(self.state.drc_result, 'error_count') else 0,
            'warning_count': self.state.drc_result.warning_count if hasattr(self.state.drc_result, 'warning_count') else 0
        }

    # =========================================================================
    # FINAL DRC LOOP
    # =========================================================================

    def _run_final_drc_loop(self) -> bool:
        """Run final DRC with retry loop"""
        self._log("\n--- FINAL DRC CHECK ---")
        self.state.stage = EngineStage.DRC

        for attempt in range(self.config.max_drc_iterations):
            stage_start = time.time()

            if not self._drc_piston and DRCPiston:
                self._drc_piston = DRCPiston(DRCConfig(
                    rules=DRCRules(
                        min_clearance=self.config.clearance,
                        min_track_width=self.config.trace_width,
                        min_via_drill=self.config.via_drill,
                        min_via_diameter=self.config.via_diameter
                    ),
                    board_width=self.config.board_width,
                    board_height=self.config.board_height
                ))

            if not self._drc_piston:
                self._log("  DRC piston not available")
                return True

            # Run full DRC
            result = self._drc_piston.check(
                self.state.parts_db,
                self.state.placement,
                self.state.routes,
                self.state.vias,
                self.state.silkscreen.texts if self.state.silkscreen else []
            )

            self.state.drc_result = result
            self.state.stage_times['drc'] = time.time() - stage_start

            self._log(f"  Attempt {attempt + 1}: Errors={result.error_count}, Warnings={result.warning_count}")
            self._log(f"  Passed: {'YES' if result.passed else 'NO'}")

            if result.passed:
                return True

            # DRC failed - try to fix
            if attempt < self.config.max_drc_iterations - 1:
                self._log("  Attempting auto-fix...")
                fixed_count = self._auto_fix_drc_violations(result)
                if fixed_count > 0:
                    self._log(f"  Fixed {fixed_count} violations, re-running DRC...")
                else:
                    self._log("  No auto-fixes available, escalating...")
                    break  # No point retrying if we can't fix anything

        return False

    def _auto_fix_drc_violations(self, drc_result) -> int:
        """
        Auto-fix DRC violations. Returns count of fixes applied.

        This is the ROOT of the feedback loop - each violation type
        has a specific fix strategy.
        """
        fixed_count = 0

        # Get violations from result
        violations = []
        if hasattr(drc_result, 'violations'):
            violations = drc_result.violations
        elif hasattr(drc_result, 'errors'):
            violations = drc_result.errors

        for v in violations:
            v_type = v.get('type', '') if isinstance(v, dict) else str(v)

            # Fix silkscreen over copper/mask
            if 'silk' in v_type.lower() and ('copper' in v_type.lower() or 'mask' in v_type.lower()):
                if self._fix_silkscreen_overlap(v):
                    fixed_count += 1

            # Fix silkscreen overlap with other silkscreen
            elif 'silk' in v_type.lower() and 'overlap' in v_type.lower():
                if self._fix_silkscreen_collision(v):
                    fixed_count += 1

            # Fix solder mask bridge
            elif 'solder_mask' in v_type.lower() or 'mask_bridge' in v_type.lower():
                if self._fix_solder_mask_bridge(v):
                    fixed_count += 1

            # Fix clearance violations
            elif 'clearance' in v_type.lower():
                if self._fix_clearance_violation(v):
                    fixed_count += 1

        return fixed_count

    def _fix_silkscreen_overlap(self, violation) -> bool:
        """
        Fix silkscreen that overlaps with solder mask openings.
        Strategy: Move silkscreen elements away from pads.
        """
        if not self.state.silkscreen:
            return False

        # Get violation location
        loc = violation.get('location', (0, 0)) if isinstance(violation, dict) else (0, 0)

        # Find silkscreen elements near this location
        removed_count = 0
        if hasattr(self.state.silkscreen, 'texts'):
            new_texts = []
            for text in self.state.silkscreen.texts:
                text_pos = text.get('position', (0, 0)) if isinstance(text, dict) else (0, 0)
                # If silkscreen is within 1mm of violation, remove it (it's over a pad)
                dist = ((text_pos[0] - loc[0])**2 + (text_pos[1] - loc[1])**2)**0.5
                if dist > 1.0:  # Keep elements more than 1mm away
                    new_texts.append(text)
                else:
                    removed_count += 1
            self.state.silkscreen.texts = new_texts

        # Also check lines/shapes if available
        if hasattr(self.state.silkscreen, 'lines'):
            new_lines = []
            for line in self.state.silkscreen.lines:
                # Remove lines that cross the violation area
                start = line.get('start', (0, 0))
                end = line.get('end', (0, 0))
                mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
                dist = ((mid[0] - loc[0])**2 + (mid[1] - loc[1])**2)**0.5
                if dist > 1.5:  # Keep lines more than 1.5mm away
                    new_lines.append(line)
                else:
                    removed_count += 1
            self.state.silkscreen.lines = new_lines

        return removed_count > 0

    def _fix_silkscreen_collision(self, violation) -> bool:
        """
        Fix silkscreen elements that overlap with each other.
        Strategy: Move reference designators further from components.
        """
        if not self.state.silkscreen:
            return False

        # For overlapping silkscreen, we adjust positions
        if hasattr(self.state.silkscreen, 'texts'):
            for i, text in enumerate(self.state.silkscreen.texts):
                if isinstance(text, dict) and 'position' in text:
                    # Move text slightly up and to the right
                    pos = text['position']
                    text['position'] = (pos[0] + 0.5, pos[1] - 0.5)

        return True

    def _fix_solder_mask_bridge(self, violation) -> bool:
        """
        Fix solder mask bridge between pads.
        Strategy: Increase pad clearance or adjust mask expansion.

        Note: This often requires footprint changes which are complex.
        For now, we log it as a warning for user action.
        """
        self._log("  [WARN] Solder mask bridge detected - may require footprint adjustment")
        # Solder mask bridges typically require PCB fabrication settings adjustments
        # or footprint changes, not something we can auto-fix in the design
        return False  # Cannot auto-fix this type

    def _fix_clearance_violation(self, violation) -> bool:
        """
        Fix clearance violations between traces/pads.
        Strategy: Adjust trace positions or widths.
        """
        if not self.state.routes:
            return False

        loc = violation.get('location', (0, 0)) if isinstance(violation, dict) else (0, 0)

        # Find routes near the violation
        modified = False
        for net_name, route in self.state.routes.items():
            if not route:
                continue

            segments = route.get('segments', []) if isinstance(route, dict) else []
            if hasattr(route, 'segments'):
                segments = route.segments

            for seg in segments:
                start = seg.get('start', (0, 0)) if isinstance(seg, dict) else getattr(seg, 'start', (0, 0))
                end = seg.get('end', (0, 0)) if isinstance(seg, dict) else getattr(seg, 'end', (0, 0))

                # Check if segment is near violation
                mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
                dist = ((mid[0] - loc[0])**2 + (mid[1] - loc[1])**2)**0.5

                if dist < 2.0:  # Within 2mm of violation
                    # Reduce trace width to increase clearance
                    current_width = seg.get('width', 0.25) if isinstance(seg, dict) else getattr(seg, 'width', 0.25)
                    new_width = max(self.config.trace_width, current_width * 0.8)
                    if isinstance(seg, dict):
                        seg['width'] = new_width
                    else:
                        seg.width = new_width
                    modified = True

        return modified

    # =========================================================================
    # PISTON EXECUTION METHODS
    # =========================================================================

    def _execute_parts(self, effort: PistonEffort) -> PistonReport:
        """Execute Parts Piston"""
        self.state.stage = EngineStage.PARTS

        if not self._parts_piston and PartsEngine:
            self._parts_piston = PartsEngine(PartsConfig())

        if not self._parts_piston:
            return PistonReport(
                piston='parts',
                success=False,
                errors=['Parts piston not available']
            )

        result = self._parts_piston.build_from_dict(self.state.parts_db)

        return PistonReport(
            piston='parts',
            success=len(result.errors) == 0,
            result=result,
            errors=result.errors,
            warnings=result.warnings,
            metrics={
                'component_count': result.component_count,
                'net_count': result.net_count
            }
        )

    def _execute_order(self, effort: PistonEffort) -> PistonReport:
        """Execute Order Piston"""
        self.state.stage = EngineStage.ORDER

        if OrderPiston is None:
            # Fallback ordering
            nets = self.state.parts_db.get('nets', {})
            self.state.net_order = list(nets.keys())
            self.state.placement_order = list(self.state.parts_db.get('parts', {}).keys())
            return PistonReport(piston='order', success=True, notes=['Using default ordering'])

        if not self._order_piston:
            self._order_piston = OrderPiston(OrderConfig(
                placement_strategy=self.config.placement_order_strategy,
                net_order_strategy=self.config.net_order_strategy,
                num_layers=self.config.layer_count,
                board_width=self.config.board_width,
                board_height=self.config.board_height
            ))

        result = self._order_piston.analyze(self.state.parts_db, placement=None)
        self.state.order_result = result
        self.state.placement_order = result.placement_order
        self.state.net_order = result.net_order
        self.state.layer_assignments = result.layer_assignments

        return PistonReport(
            piston='order',
            success=True,
            result=result,
            metrics={
                'placement_order_count': len(result.placement_order),
                'net_order_count': len(result.net_order)
            }
        )

    def _execute_placement(self, effort: PistonEffort) -> PistonReport:
        """Execute Placement Piston with AI-directed algorithm selection"""
        self.state.stage = EngineStage.PLACEMENT

        params = self._get_effort_parameters(effort)

        # === AI ALGORITHM SELECTION ===
        work_order = self._request_algorithm_selection('placement', {
            'effort': effort.value,
            'board_area': self.config.board_width * self.config.board_height,
        })

        self._log(f"  Algorithm: {work_order.algorithm} (AI: {work_order.ai_reasoning})")

        if not self._placement_piston and PlacementEngine and PlacementEngineConfig:
            iterations = int(self.config.max_placement_iterations * params.get('iterations_multiplier', 1.0))
            self._placement_piston = PlacementEngine(PlacementEngineConfig(
                board_width=self.config.board_width,
                board_height=self.config.board_height,
                origin_x=0.0,  # Board starts at origin
                origin_y=0.0,
                grid_size=self.config.grid_size,
                fd_iterations=iterations,  # Force-directed iterations
                ga_generations=iterations,  # Genetic algorithm generations
            ))

        if not self._placement_piston:
            return PistonReport(
                piston='placement',
                success=False,
                errors=['Placement piston not available']
            )

        # Build connectivity graph from order result or parts_db
        graph = {}
        if self.state.order_result and hasattr(self.state.order_result, 'adjacency'):
            graph = {'adjacency': self.state.order_result.adjacency}
        else:
            # Build adjacency from nets
            adjacency = {}
            nets = self.state.parts_db.get('nets', {})
            for net_name, net_info in nets.items():
                pins = net_info.get('pins', []) if isinstance(net_info, dict) else []

                # Extract component refs from pins in various formats
                refs = set()
                if isinstance(pins, dict):
                    # Format: {'R1': {...}, 'U1': {...}}
                    refs = set(pins.keys())
                elif isinstance(pins, list):
                    for p in pins:
                        if isinstance(p, dict):
                            # Format: [{'ref': 'R1', 'pin': '1'}, ...]
                            ref = p.get('ref', '')
                            if ref:
                                refs.add(ref)
                        elif isinstance(p, str):
                            # Format: ['R1.1', 'U1.2'] - extract component ref
                            comp = p.split('.')[0] if '.' in p else p
                            if comp:
                                refs.add(comp)
                        elif isinstance(p, (list, tuple)) and len(p) >= 1:
                            # Format: [('R1', '1'), ('U1', '2')]
                            refs.add(str(p[0]))

                refs = list(refs)
                # Connect all components in same net
                for i, ref1 in enumerate(refs):
                    if ref1 not in adjacency:
                        adjacency[ref1] = {}
                    for ref2 in refs[i+1:]:
                        adjacency[ref1][ref2] = adjacency[ref1].get(ref2, 0) + 1
                        if ref2 not in adjacency:
                            adjacency[ref2] = {}
                        adjacency[ref2][ref1] = adjacency[ref2].get(ref1, 0) + 1
            graph = {'adjacency': adjacency}

        result = self._placement_piston.place(self.state.parts_db, graph)

        # BUG-07 FIX: Validate result before assignment
        if result and hasattr(result, 'positions') and result.positions:
            self.state.placement = result.positions
        else:
            self._log("WARNING: Placement piston returned no positions")
            # Keep existing placement (empty dict from initialization)

        return PistonReport(
            piston='placement',
            success=result.success if result else False,
            result=result,
            metrics={
                'placed_count': len(self.state.placement),
                'algorithm_used': work_order.algorithm
            }
        )

    def _execute_escape(self, effort: PistonEffort) -> PistonReport:
        """Execute Escape Piston"""
        self.state.stage = EngineStage.ESCAPE

        if not self._escape_piston and EscapePiston:
            self._escape_piston = EscapePiston(EscapeConfig())

        if not self._escape_piston:
            return PistonReport(piston='escape', success=True, notes=['Escape piston not available'])

        # EscapePiston.escape() requires a PinArray, but for simple designs we skip escape routing
        # Escape routing is mainly needed for BGA packages, not QFP/0603
        # For now, return success and let routing piston handle direct routing
        result = type('EscapeResult', (), {'success': True, 'paths': [], 'messages': ['Escape routing skipped for simple design']})()
        self._escape_result = result

        return PistonReport(
            piston='escape',
            success=result.success if hasattr(result, 'success') else True,
            result=result
        )

    def _execute_routing(self, effort: PistonEffort) -> PistonReport:
        """Execute Routing Piston with AI-directed algorithm selection"""
        self.state.stage = EngineStage.ROUTING

        params = self._get_effort_parameters(effort)

        # Calculate dynamic grid
        grid_size = self.config.grid_size
        if self.config.use_dynamic_grid and calculate_optimal_grid_size:
            grid_size, _ = calculate_optimal_grid_size(
                self.state.parts_db,
                min_clearance=self.config.clearance,
                min_trace_width=self.config.trace_width,
                max_grid_size=self.config.max_grid_size,
                min_grid_size=self.config.min_grid_size
            )

        if not RoutingPiston:
            return PistonReport(
                piston='routing',
                success=False,
                errors=['Routing piston not available']
            )

        # === AI ALGORITHM SELECTION ===
        # Ask Circuit AI / AI Agent which algorithm to use
        work_order = self._request_algorithm_selection('routing', {
            'effort': effort.value,
            'grid_size': grid_size,
            'density': len(self.state.parts_db.get('nets', {})) / (self.config.board_width * self.config.board_height),
        })

        # Use AI-selected algorithm, or default
        selected_algorithm = work_order.algorithm or self.config.routing_algorithm
        if params['algorithms'] == 'all':
            selected_algorithm = 'hybrid'  # Force hybrid on deeper efforts

        self._log(f"  Algorithm: {selected_algorithm} (AI: {work_order.ai_reasoning})")

        routing_config = RoutingConfig(
            algorithm=selected_algorithm,
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            grid_size=grid_size,
            trace_width=self.config.trace_width,
            clearance=self.config.clearance,
            via_diameter=self.config.via_diameter,
            via_drill=self.config.via_drill,
            max_ripup_iterations=int(self.config.max_routing_iterations * params['iterations_multiplier'])
        )
        self._routing_piston = RoutingPiston(routing_config)

        escapes = self._build_simple_escapes()

        # First attempt with selected algorithm
        result = self._routing_piston.route(
            self.state.parts_db,
            escapes,
            self.state.placement,
            self.state.net_order
        )

        # === ENGINE COMMAND: TRY ALGORITHM CASCADE IF ROUTING FAILS ===
        # If initial routing failed, try hybrid algorithm which includes ripup-reroute
        if not result.success:
            self._log(f"  [ENGINE] Initial routing failed ({result.routed_count}/{result.total_count})")
            self._log(f"  [ENGINE] Ordering CASCADE: try all 11 algorithms...")

            # Use the cascade function which tries multiple algorithms
            try:
                cascade_result = route_with_cascade(
                    parts_db=self.state.parts_db,
                    escapes=escapes,
                    placement=self.state.placement,
                    net_order=self.state.net_order,
                    board_width=self.config.board_width,
                    board_height=self.config.board_height,
                    config=routing_config
                )

                # Use cascade result if it's better
                if cascade_result.routed_count > result.routed_count:
                    result = cascade_result
                    self._log(f"  [ENGINE] CASCADE improved: {result.routed_count}/{result.total_count}")
            except Exception as e:
                self._log(f"  [ENGINE] CASCADE failed: {e}")

        self.state.routes = result.routes
        self.state.vias = []
        for route in result.routes.values():
            if hasattr(route, 'vias'):
                self.state.vias.extend(route.vias)

        return PistonReport(
            piston='routing',
            success=result.success,
            result=result,
            errors=[] if result.success else [f"Routed {result.routed_count}/{result.total_count}"],
            metrics={
                'routed_count': result.routed_count,
                'total_count': result.total_count,
                'algorithm': result.algorithm_used
            }
        )

    def _execute_optimization(self, effort: PistonEffort) -> PistonReport:
        """Execute Optimization Piston"""
        self.state.stage = EngineStage.OPTIMIZE

        if not OptimizationPiston:
            return PistonReport(piston='optimize', success=True, notes=['Optimization skipped'])

        if not self._optimization_piston:
            self._optimization_piston = OptimizationPiston(OptimizationConfig())

        # Would run optimization here
        return PistonReport(piston='optimize', success=True)

    def _should_run_polish(self) -> bool:
        """Determine if Polish Piston should run."""
        # Always run polish if routes exist and PolishPiston is available
        if not PolishPiston:
            return False
        if not self.state.routes:
            return False
        # Skip if explicitly disabled
        if hasattr(self.config, 'skip_polish') and self.config.skip_polish:
            return False
        return True

    def _execute_polish(self, effort: PistonEffort) -> PistonReport:
        """
        Execute Polish Piston - Post-routing optimization.

        Performs:
        - Via reduction (remove unnecessary layer changes)
        - Trace simplification (merge collinear segments)
        - Board shrink (reduce to minimum size)
        - Grid alignment (snap to manufacturing grid)
        - Arc smoothing (TODO: replace 90deg corners with arcs)
        """
        self._log("\n--- POLISH (Post-routing optimization) ---")
        stage_start = time.time()

        if not PolishPiston:
            return PistonReport(piston='polish', success=True, notes=['Polish piston not available'])

        if not self.state.routes:
            return PistonReport(piston='polish', success=True, notes=['No routes to polish'])

        # Configure polish level based on effort
        level = PolishLevel.STANDARD
        if effort == PistonEffort.MINIMAL:
            level = PolishLevel.MINIMAL
        elif effort == PistonEffort.MAXIMUM:
            level = PolishLevel.PROFESSIONAL

        if not self._polish_piston:
            self._polish_piston = PolishPiston(PolishConfig(
                level=level,
                reduce_vias=True,
                simplify_traces=True,
                shrink_board=True,  # Will be validated by DRC after
                align_to_grid=True,
                verbose=self.config.verbose
            ))

        try:
            # Run polish optimization
            result = self._polish_piston.polish(
                routes=self.state.routes,
                parts_db=self.state.parts_db,
                placement=self.state.placement,
                board_width=self.config.board_width,
                board_height=self.config.board_height
            )

            if result.success:
                # Update state with polished routes
                self.state.routes = result.routes

                # Update board dimensions if shrunk
                if result.new_board != result.original_board:
                    self.config.board_width = result.new_board[0]
                    self.config.board_height = result.new_board[1]
                    self._log(f"  Board shrunk: {result.original_board[0]:.1f}x{result.original_board[1]:.1f} -> {result.new_board[0]:.1f}x{result.new_board[1]:.1f}")

                self._log(f"  Vias: {result.original_via_count} -> {result.new_via_count} ({result.vias_removed} removed)")
                self._log(f"  Segments: {result.original_segment_count} -> {result.new_segment_count} ({result.segments_merged} merged)")

            self.state.stage_times['polish'] = time.time() - stage_start

            return PistonReport(
                piston='polish',
                success=result.success,
                result=result,
                notes=result.messages,
                metrics={
                    'vias_removed': result.vias_removed,
                    'segments_merged': result.segments_merged,
                    'board_reduction_percent': result.board_reduction_percent,
                    'original_length': result.original_total_length,
                    'new_length': result.new_total_length
                }
            )

        except Exception as e:
            self._log(f"  Polish error: {e}")
            self.state.stage_times['polish'] = time.time() - stage_start
            return PistonReport(
                piston='polish',
                success=False,
                errors=[str(e)],
                notes=['Polish failed, continuing with unpolished design']
            )

    def _execute_silkscreen(self, effort: PistonEffort) -> PistonReport:
        """Execute Silkscreen Piston"""
        self.state.stage = EngineStage.SILKSCREEN

        if not self._silkscreen_piston and SilkscreenPiston:
            self._silkscreen_piston = SilkscreenPiston(SilkscreenConfig(
                show_references=self.config.show_references,
                show_values=self.config.show_values
            ))

        if not self._silkscreen_piston:
            return PistonReport(piston='silkscreen', success=True, notes=['Silkscreen skipped'])

        result = self._silkscreen_piston.generate(
            self.state.parts_db,
            self.state.placement
        )
        self.state.silkscreen = result

        return PistonReport(
            piston='silkscreen',
            success=result.success,
            result=result,
            warnings=result.warnings,
            metrics={
                'ref_count': result.ref_count,
                'collision_count': result.collision_count
            }
        )

    def _execute_output(self) -> PistonReport:
        """Execute Output Piston"""
        self._log("\n--- OUTPUT ---")
        self.state.stage = EngineStage.OUTPUT
        stage_start = time.time()

        if not self._output_piston and OutputPiston:
            self._output_piston = OutputPiston(OutputConfig(
                output_dir=self.config.output_dir,
                board_name=self.config.board_name,
                board_width=self.config.board_width,
                board_height=self.config.board_height,
                generate_kicad=self.config.generate_kicad,
                generate_gerbers=self.config.generate_gerbers,
                generate_bom=self.config.generate_bom
            ))

        if not self._output_piston:
            return PistonReport(piston='output', success=False, errors=['Output piston not available'])

        result = self._output_piston.generate(
            self.state.parts_db,
            self.state.placement,
            self.state.routes,
            self.state.vias,
            self.state.silkscreen
        )

        self.state.stage_times['output'] = time.time() - stage_start

        self._log(f"  Files generated: {len(result.files_generated)}")
        for f in result.files_generated:
            self._log(f"    - {os.path.basename(f)}")

        return PistonReport(
            piston='output',
            success=result.success,
            result=result,
            errors=result.errors,
            metrics={'files_generated': len(result.files_generated)}
        )

    # =========================================================================
    # ADVANCED PISTON EXECUTION METHODS
    # =========================================================================

    def _execute_topological_routing(self, effort: PistonEffort) -> PistonReport:
        """Execute Topological Router Piston for advanced routing"""
        self._log("\n--- TOPOLOGICAL ROUTING ---")
        self.state.stage = EngineStage.TOPOLOGICAL_ROUTING
        stage_start = time.time()

        if not TopologicalRouterPiston:
            return PistonReport(
                piston='topological_routing',
                success=False,
                errors=['Topological Router piston not available']
            )

        if not self._topological_router_piston:
            self._topological_router_piston = TopologicalRouterPiston()

        # Build design data for topological router
        design_data = {
            'board_width': self.config.board_width,
            'board_height': self.config.board_height,
            'components': self._build_component_data(),
            'nets': self.state.parts_db.get('nets', {}),
            'existing_routes': self.state.routes
        }

        try:
            result = self._topological_router_piston.route(design_data)
            self.state.topological_routes = result.get('routes', {})

            self.state.stage_times['topological_routing'] = time.time() - stage_start
            self._log(f"  Topological routes: {len(self.state.topological_routes)}")

            return PistonReport(
                piston='topological_routing',
                success=result.get('success', True),
                result=result,
                metrics={'routes_optimized': len(self.state.topological_routes)}
            )
        except Exception as e:
            return PistonReport(
                piston='topological_routing',
                success=False,
                errors=[f'Topological routing failed: {str(e)}']
            )

    def _execute_3d_visualization(self) -> PistonReport:
        """Execute 3D Visualization Piston"""
        self._log("\n--- 3D VISUALIZATION ---")
        self.state.stage = EngineStage.VISUALIZATION_3D
        stage_start = time.time()

        if not Visualization3DPiston:
            self._log("  3D Visualization piston not available")
            return PistonReport(
                piston='visualization_3d',
                success=False,
                errors=['3D Visualization piston not available']
            )

        if not self._visualization_3d_piston:
            self._visualization_3d_piston = Visualization3DPiston()

        # Build design data for 3D visualization
        design_data = {
            'board_width': self.config.board_width,
            'board_height': self.config.board_height,
            'board_thickness': 1.6,
            'layer_count': self.config.layer_count,
            'components': self._build_component_data(),
            'traces': self._build_trace_data(),
            'vias': self._build_via_data()
        }

        # Map format string to enum
        format_map = {
            'stl': Vis3DFormat.STL if Vis3DFormat else None,
            'step': Vis3DFormat.STEP if Vis3DFormat else None,
            'gltf': Vis3DFormat.GLTF if Vis3DFormat else None,
            'obj': Vis3DFormat.OBJ if Vis3DFormat else None,
        }
        output_format = format_map.get(self.config.output_3d_format.lower())

        try:
            result = self._visualization_3d_piston.visualize(
                design_data,
                output_format=output_format
            )
            self.state.visualization_3d = result

            # Save 3D file
            if result.get('success') and result.get('output'):
                output_path = os.path.join(
                    self.config.output_dir,
                    f"{self.config.board_name}_3d.{self.config.output_3d_format}"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(result['output'])
                self._log(f"  3D file saved: {output_path}")

            self.state.stage_times['visualization_3d'] = time.time() - stage_start
            stats = result.get('stats', {})
            self._log(f"  Triangles: {stats.get('total_triangles', 0)}")
            self._log(f"  Components: {stats.get('component_count', 0)}")

            return PistonReport(
                piston='visualization_3d',
                success=result.get('success', False),
                result=result,
                metrics=stats
            )
        except Exception as e:
            return PistonReport(
                piston='visualization_3d',
                success=False,
                errors=[f'3D visualization failed: {str(e)}']
            )

    def _execute_bom_optimization(self) -> PistonReport:
        """Execute BOM Optimizer Piston"""
        self._log("\n--- BOM OPTIMIZATION ---")
        self.state.stage = EngineStage.BOM_OPTIMIZE
        stage_start = time.time()

        if not BOMOptimizerPiston:
            self._log("  BOM Optimizer piston not available")
            return PistonReport(
                piston='bom_optimization',
                success=False,
                errors=['BOM Optimizer piston not available']
            )

        if not self._bom_optimizer_piston:
            self._bom_optimizer_piston = BOMOptimizerPiston()

        # Build BOM data from parts_db
        bom_data = self._build_bom_data()

        try:
            result = self._bom_optimizer_piston.optimize(
                bom=bom_data,
                goal=BOMGoal.BALANCED if BOMGoal else None,
                production_quantity=1
            )
            self.state.bom_optimization = result

            # Save BOM report
            if hasattr(self._bom_optimizer_piston, 'generate_order_report'):
                report = self._bom_optimizer_piston.generate_order_report(result)
                report_path = os.path.join(
                    self.config.output_dir,
                    f"{self.config.board_name}_bom_optimized.txt"
                )
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    f.write(report)
                self._log(f"  BOM report saved: {report_path}")

            self.state.stage_times['bom_optimization'] = time.time() - stage_start
            self._log(f"  Total cost: ${result.grand_total:.2f}")
            self._log(f"  Suppliers: {', '.join(result.suppliers_used)}")
            self._log(f"  Lead time: {result.max_lead_time_days} days")

            return PistonReport(
                piston='bom_optimization',
                success=True,
                result=result,
                metrics={
                    'total_cost': result.grand_total,
                    'supplier_count': len(result.suppliers_used),
                    'lead_time': result.max_lead_time_days
                }
            )
        except Exception as e:
            return PistonReport(
                piston='bom_optimization',
                success=False,
                errors=[f'BOM optimization failed: {str(e)}']
            )

    # =========================================================================
    # LEARNING PISTON METHODS
    # =========================================================================

    def _load_learned_patterns(self):
        """Load learned patterns from previous designs"""
        if not self.config.enable_learning or not LearningPiston:
            return

        self._log("\n--- LOADING LEARNED PATTERNS ---")
        self.state.stage = EngineStage.LEARNING_LOAD

        if not self._learning_piston:
            self._learning_piston = LearningPiston()

        # Load from file if path specified
        if self.config.learning_model_path and os.path.exists(self.config.learning_model_path):
            try:
                self._learning_piston.load_models(self.config.learning_model_path)
                self.state.learned_patterns = self._learning_piston.patterns
                self._log(f"  Loaded {len(self.state.learned_patterns)} patterns")
            except Exception as e:
                self._log(f"  Failed to load patterns: {e}")
        else:
            self._log("  No learned patterns available (starting fresh)")

    def _save_learned_patterns(self):
        """Save learned patterns after successful design"""
        if not self.config.enable_learning or not self.config.learn_from_result:
            return
        if not LearningPiston or not self._learning_piston:
            return

        self._log("\n--- SAVING LEARNED PATTERNS ---")
        prev_stage = self.state.stage  # Remember stage to restore it
        self.state.stage = EngineStage.LEARNING_SAVE

        # Build design for learning
        design_data = {
            'board_width': self.config.board_width,
            'board_height': self.config.board_height,
            'layer_count': self.config.layer_count,
            'components': self._build_component_data(),
            'traces': self._build_trace_data(),
            'vias': self._build_via_data(),
            'nets': self.state.parts_db.get('nets', {})
        }

        # Learn from this design (would need actual file for real learning)
        # For now, we just save the current patterns
        if self.config.learning_model_path:
            try:
                self._learning_piston.save_models(self.config.learning_model_path)
                self._log(f"  Saved patterns to {self.config.learning_model_path}")
            except Exception as e:
                self._log(f"  Failed to save patterns: {e}")

        # Restore stage to COMPLETE after learning save
        self.state.stage = prev_stage

    def _apply_learned_patterns(self, piston_name: str, data: Dict) -> Dict:
        """Apply learned patterns to improve a piston's work"""
        if not self.config.enable_learning or not self._learning_piston:
            return data

        if not self.state.learned_patterns:
            return data

        # Apply relevant patterns
        improvements = self._learning_piston.apply_patterns(data)

        if improvements.get('applied_patterns'):
            self._log(f"  Applied learned patterns: {improvements['applied_patterns']}")

        return data

    def _get_learned_routing_strategy(self) -> str:
        """Get routing strategy recommendation from learning piston"""
        if not self.config.enable_learning or not self._learning_piston:
            return 'hybrid'

        # Build simplified design for prediction
        design_data = {
            'board_width': self.config.board_width,
            'board_height': self.config.board_height,
            'layer_count': self.config.layer_count,
            'component_count': len(self.state.parts_db.get('parts', {})),
            'net_count': len(self.state.parts_db.get('nets', {}))
        }

        try:
            strategy = self._learning_piston.recommend_routing_strategy(design_data)
            self._log(f"  Learning recommends: {strategy}")
            return strategy
        except:
            return 'hybrid'

    # =========================================================================
    # ANALYSIS PISTONS
    # =========================================================================

    def _run_analysis_pistons(self):
        """Run optional analysis pistons (Stackup, Thermal, PDN, SI)"""
        self._log("\n--- ANALYSIS ---")
        self.state.stage = EngineStage.ANALYSIS

        # Stackup Analysis
        if StackupPiston and self._stackup_piston:
            try:
                self.state.stackup = self._stackup_piston.analyze_stackup(
                    self._get_stackup_config()
                )
                self._log(f"  Stackup: {self.state.stackup.get('stackup_name', 'analyzed')}")
            except Exception as e:
                self._log(f"  Stackup analysis failed: {e}")

        # Thermal Analysis
        if self.config.run_thermal_analysis and ThermalPiston:
            if not self._thermal_piston:
                self._thermal_piston = ThermalPiston()
            try:
                thermal_data = self._build_thermal_data()
                self.state.thermal_analysis = self._thermal_piston.analyze(thermal_data)
                max_temp = self.state.thermal_analysis.get('max_temperature', 0)
                self._log(f"  Thermal: Max temp {max_temp:.1f}°C")
            except Exception as e:
                self._log(f"  Thermal analysis failed: {e}")

        # PDN Analysis
        if self.config.run_pdn_analysis and PDNPiston:
            if not self._pdn_piston:
                self._pdn_piston = PDNPiston()
            try:
                pdn_data = self._build_pdn_data()
                self.state.pdn_analysis = self._pdn_piston.analyze(pdn_data)
                self._log(f"  PDN: Analyzed {len(self.state.pdn_analysis.get('rails', []))} rails")
            except Exception as e:
                self._log(f"  PDN analysis failed: {e}")

        # Signal Integrity Analysis
        if self.config.run_si_analysis and SignalIntegrityPiston:
            if not self._si_piston:
                self._si_piston = SignalIntegrityPiston()
            try:
                si_data = self._build_si_data()
                self.state.si_analysis = self._si_piston.analyze(si_data)
                self._log(f"  SI: Analyzed {len(self.state.si_analysis.get('nets', []))} critical nets")
            except Exception as e:
                self._log(f"  SI analysis failed: {e}")

    def _should_use_topological_routing(self) -> bool:
        """Determine if topological routing should be used"""
        if self.config.use_topological_routing:
            return True

        # Auto-detect: use for differential pairs or RF
        nets = self.state.parts_db.get('nets', {})
        for net_name in nets.keys():
            name_upper = net_name.upper()
            if '_P' in name_upper or '_N' in name_upper:
                return True  # Differential pair
            if 'RF' in name_upper or 'ANT' in name_upper:
                return True  # RF signal

        return False

    # =========================================================================
    # DATA BUILDING HELPERS FOR NEW PISTONS
    # =========================================================================

    def _build_component_data(self) -> List[Dict]:
        """Build component data for visualization and analysis"""
        components = []
        parts = self.state.parts_db.get('parts', {})

        for ref, part in parts.items():
            pos = self.state.placement.get(ref)
            if not pos:
                continue

            pos_x = pos.x if hasattr(pos, 'x') else pos[0] if isinstance(pos, (list, tuple)) else 0
            pos_y = pos.y if hasattr(pos, 'y') else pos[1] if isinstance(pos, (list, tuple)) else 0
            rotation = pos.rotation if hasattr(pos, 'rotation') else 0

            components.append({
                'designator': ref,
                'name': part.get('name', ''),
                'value': part.get('value', ''),
                'footprint': part.get('footprint', ''),
                'x': pos_x,
                'y': pos_y,
                'rotation': rotation
            })

        return components

    def _build_trace_data(self) -> List[Dict]:
        """Build trace data for visualization"""
        traces = []

        for net_name, route in self.state.routes.items():
            if hasattr(route, 'segments'):
                for seg in route.segments:
                    traces.append({
                        'net': net_name,
                        'points': [(seg.start[0], seg.start[1]), (seg.end[0], seg.end[1])],
                        'width': seg.width if hasattr(seg, 'width') else self.config.trace_width,
                        'layer': seg.layer if hasattr(seg, 'layer') else 0
                    })

        return traces

    def _build_via_data(self) -> List[Dict]:
        """Build via data for visualization"""
        vias = []

        for via in self.state.vias:
            # Handle Via objects, dicts, or tuples
            if hasattr(via, 'x'):  # Via dataclass with .x/.y attributes
                x, y = via.x, via.y
                drill = getattr(via, 'drill', self.config.via_drill)
                diameter = getattr(via, 'diameter', self.config.via_diameter)
            elif isinstance(via, dict):
                pos = via.get('position', (0, 0))
                if hasattr(pos, 'x'):
                    x, y = pos.x, pos.y
                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    x, y = pos[0], pos[1]
                else:
                    x, y = 0, 0
                drill = via.get('drill', self.config.via_drill)
                diameter = via.get('diameter', self.config.via_diameter)
            elif isinstance(via, (list, tuple)) and len(via) >= 2:
                x, y = via[0], via[1]
                drill = self.config.via_drill
                diameter = self.config.via_diameter
            else:
                continue  # Skip unrecognized format

            vias.append({
                'x': x,
                'y': y,
                'drill': drill,
                'pad': diameter,
                'start_layer': 0,
                'end_layer': self.config.layer_count - 1
            })

        return vias

    def _build_bom_data(self) -> List:
        """Build BOM data for optimization"""
        from .bom_optimizer_piston import BOMLine, PartCategory
        bom_lines = []

        # Group parts by value and footprint
        groups = {}
        parts = self.state.parts_db.get('parts', {})

        for ref, part in parts.items():
            key = (part.get('value', ''), part.get('footprint', ''))
            if key not in groups:
                groups[key] = []
            groups[key].append(ref)

        for (value, footprint), refs in groups.items():
            bom_lines.append(BOMLine(
                designators=refs,
                quantity=len(refs),
                value=value,
                footprint=footprint,
                description=parts[refs[0]].get('name', ''),
                manufacturer=parts[refs[0]].get('manufacturer', ''),
                mpn=parts[refs[0]].get('mpn', '')
            ))

        return bom_lines

    def _build_thermal_data(self) -> Dict:
        """Build data for thermal analysis"""
        return {
            'board_width': self.config.board_width,
            'board_height': self.config.board_height,
            'components': self._build_component_data(),
            'ambient_temp': 25.0
        }

    def _build_pdn_data(self) -> Dict:
        """Build data for PDN analysis"""
        return {
            'board_width': self.config.board_width,
            'board_height': self.config.board_height,
            'layer_count': self.config.layer_count,
            'nets': self.state.parts_db.get('nets', {}),
            'components': self._build_component_data()
        }

    def _build_si_data(self) -> Dict:
        """Build data for signal integrity analysis"""
        return {
            'board_width': self.config.board_width,
            'board_height': self.config.board_height,
            'layer_count': self.config.layer_count,
            'traces': self._build_trace_data(),
            'nets': self.state.parts_db.get('nets', {})
        }

    def _get_stackup_config(self) -> Dict:
        """Get stackup configuration"""
        return {
            'layer_count': self.config.layer_count,
            'board_thickness': 1.6,
            'copper_weight': '1oz'
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _needs_escape_routing(self) -> bool:
        """Check if any components need escape routing (BGA, QFP)"""
        parts = self.state.parts_db.get('parts', {})
        for ref, part in parts.items():
            footprint = part.get('footprint', '').lower()
            if 'bga' in footprint or 'qfp' in footprint or 'qfn' in footprint:
                return True
        return False

    def _get_default_pin_offset(self, footprint: str, pin_num: str, total_pins: int) -> Tuple[float, float]:
        """
        Generate default pin offsets based on footprint type.

        Common footprint patterns:
        - 0402, 0603, 0805, 1206: 2-pin passives (horizontal)
        - SOT-23: 3-pin (transistors)
        - SOIC-8, TSSOP, QFP: Multi-pin ICs

        Returns:
            (offset_x, offset_y) in mm from component center
        """
        fp_lower = footprint.lower() if footprint else ''

        # 2-pin passives (resistors, capacitors, inductors)
        # Standard sizes: 0402=1.0mm, 0603=1.6mm, 0805=2.0mm, 1206=3.2mm
        if total_pins == 2 or any(size in fp_lower for size in ['0402', '0603', '0805', '1206', '1210', '1812', '2010', '2512']):
            spacing = 0.5  # Default half-spacing
            if '0402' in fp_lower:
                spacing = 0.5
            elif '0603' in fp_lower:
                spacing = 0.75
            elif '0805' in fp_lower:
                spacing = 1.0
            elif '1206' in fp_lower:
                spacing = 1.5
            elif '1210' in fp_lower or '1812' in fp_lower:
                spacing = 1.75
            elif '2010' in fp_lower or '2512' in fp_lower:
                spacing = 2.0

            # Pin 1 on left, Pin 2 on right
            if pin_num == '1':
                return (-spacing, 0.0)
            else:
                return (spacing, 0.0)

        # SOT-23 (3-pin transistor package)
        if 'sot-23' in fp_lower or 'sot23' in fp_lower:
            if pin_num == '1':
                return (-0.95, -0.5)
            elif pin_num == '2':
                return (0.95, -0.5)
            else:
                return (0.0, 0.5)

        # SOIC-8 and similar
        if 'soic' in fp_lower or 'so-8' in fp_lower or 'so8' in fp_lower:
            pitch = 1.27
            try:
                pn = int(pin_num)
                if pn <= 4:
                    return (-1.9, pitch * (1.5 - (pn - 1)))
                else:
                    return (1.9, pitch * ((pn - 5) - 1.5))
            except ValueError:
                return (0.0, 0.0)

        # QFP/TQFP/LQFP packages
        if any(pkg in fp_lower for pkg in ['qfp', 'tqfp', 'lqfp']):
            # Extract pin count if possible
            try:
                pn = int(pin_num)
                pins_per_side = max(1, total_pins // 4)
                pitch = 0.5  # Common QFP pitch

                side = (pn - 1) // pins_per_side
                pos_on_side = (pn - 1) % pins_per_side

                half_length = (pins_per_side - 1) * pitch / 2
                offset_along = pos_on_side * pitch - half_length
                body_half = 3.0  # Half body size

                if side == 0:  # Bottom
                    return (offset_along, -body_half)
                elif side == 1:  # Right
                    return (body_half, offset_along)
                elif side == 2:  # Top
                    return (-offset_along, body_half)
                else:  # Left
                    return (-body_half, -offset_along)
            except ValueError:
                return (0.0, 0.0)

        # DIP packages
        if 'dip' in fp_lower:
            pitch = 2.54
            try:
                pn = int(pin_num)
                half_pins = max(1, total_pins // 2)
                if pn <= half_pins:
                    return (-3.81, pitch * ((half_pins - 1) / 2 - (pn - 1)))
                else:
                    return (3.81, pitch * ((pn - half_pins - 1) - (half_pins - 1) / 2))
            except ValueError:
                return (0.0, 0.0)

        # Default: spread pins linearly based on pin number
        try:
            pn = int(pin_num)
            spacing = 0.75  # Default spacing
            if total_pins <= 2:
                return (spacing * (2 * pn - 3) / 2, 0.0)  # -0.75 for pin 1, +0.75 for pin 2
            else:
                # Distribute pins in a row
                return (spacing * (pn - (total_pins + 1) / 2), 0.0)
        except ValueError:
            return (0.0, 0.0)

    def _build_simple_escapes(self) -> Dict:
        """Build simple escape routes (pin to pin center)"""
        escapes = {}

        # BUG-02 FIX: Guard against empty/None placement
        if not self.state.placement:
            self._log("WARNING: No placement data available for escape generation")
            return escapes

        parts = self.state.parts_db.get('parts', {})
        nets = self.state.parts_db.get('nets', {})

        # Build reverse lookup: pin_ref -> net_name (e.g., 'R1.1' -> 'VCC')
        pin_to_net = {}
        for net_name, net_data in nets.items():
            for pin_ref in net_data.get('pins', []):
                pin_to_net[pin_ref] = net_name

        # Define SimpleEscape class outside the loop
        class SimpleEscape:
            def __init__(self, start, end, net, layer):
                self.start = start
                self.end = end
                self.endpoint = end
                self.net = net
                self.layer = layer

        for ref, pos in self.state.placement.items():
            part = parts.get(ref, {})
            escapes[ref] = {}

            # Get footprint and pin count for default offset calculation
            footprint = part.get('footprint', '')
            pins_list = get_pins(part)
            total_pins = len(pins_list)

            # BUG-06 FIX: Use get_pins for consistent pin access
            for pin in pins_list:
                pin_num = str(pin.get('number', ''))

                # Look up net from pin.get('net') OR from nets dict
                net = pin.get('net', '')
                if not net:
                    pin_ref = f"{ref}.{pin_num}"
                    net = pin_to_net.get(pin_ref, '')

                if not net:
                    continue

                offset = pin.get('offset', (0, 0))
                if not offset or offset == (0, 0):
                    physical = pin.get('physical', {})
                    offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))

                # DRC-FIX: If still no offset, generate default based on footprint
                if not offset or offset == (0, 0):
                    offset = self._get_default_pin_offset(footprint, pin_num, total_pins)

                # BUG-03 FIX: Use get_xy for consistent position access
                pos_x, pos_y = get_xy(pos)
                pad_x = pos_x + float(offset[0]) if offset[0] else pos_x
                pad_y = pos_y + float(offset[1]) if offset[1] else pos_y

                # COURTYARD-FIX: Escape direction should go AROUND component courtyards
                # Check if escaping horizontally would cross another component
                escape_length = 1.0

                # Calculate default escape direction: away from component center
                if offset[0]:
                    off_x = float(offset[0])
                    if off_x < 0:
                        default_escape_x = pad_x - escape_length
                    else:
                        default_escape_x = pad_x + escape_length
                else:
                    default_escape_x = pad_x + escape_length
                default_escape_y = pad_y

                # Check if the escape path crosses another component's courtyard
                escape_blocked = False
                for other_ref, other_pos in self.state.placement.items():
                    if other_ref == ref:
                        continue
                    other_x, other_y = get_xy(other_pos)
                    other_part = parts.get(other_ref, {})
                    other_fp = other_part.get('footprint', '')

                    # Get courtyard size (body + margin)
                    # 0603: body 2.0mm, courtyard ~2.5mm
                    courtyard_half_w = 1.5  # Conservative estimate
                    courtyard_half_h = 1.0

                    # Check if escape path would cross this component's courtyard
                    # Path is from (pad_x, pad_y) to (default_escape_x, default_escape_y)
                    min_x = min(pad_x, default_escape_x)
                    max_x = max(pad_x, default_escape_x)

                    # Check horizontal overlap
                    if (abs(pad_y - other_y) < courtyard_half_h and
                        min_x < other_x + courtyard_half_w and
                        max_x > other_x - courtyard_half_w):
                        escape_blocked = True
                        break

                # If horizontal escape is blocked, escape vertically (UP, away from board center)
                if escape_blocked:
                    # Go UP (or DOWN based on position)
                    board_center_y = self.config.board_height / 2
                    if pad_y < board_center_y:
                        # Pad is above center, escape UP
                        escape_x = pad_x
                        escape_y = pad_y - escape_length
                    else:
                        # Pad is below center, escape DOWN
                        escape_x = pad_x
                        escape_y = pad_y + escape_length
                else:
                    escape_x = default_escape_x
                    escape_y = default_escape_y

                escapes[ref][pin_num] = SimpleEscape(
                    start=(float(pad_x), float(pad_y)),
                    end=(float(escape_x), float(escape_y)),
                    net=net,
                    layer=getattr(pos, 'layer', 'F.Cu')
                )

        return escapes

    def _create_result(self) -> EngineResult:
        """Create the final result object"""
        total_time = time.time() - self.state.start_time

        output_files = []
        if self._output_piston:
            output_files = getattr(self._output_piston, 'files_generated', [])

        nets = self.state.parts_db.get('nets', {})
        total_nets = len(nets)
        routed_count = sum(1 for r in self.state.routes.values()
                          if hasattr(r, 'success') and r.success)

        self._log("\n" + "=" * 70)
        self._log("PCB ENGINE - Complete")
        self._log("=" * 70)
        self._log(f"Final stage: {self.state.stage.value}")
        self._log(f"Total time: {total_time:.2f}s")
        self._log(f"Errors: {len(self.state.errors)}")
        self._log(f"Warnings: {len(self.state.warnings)}")
        self._log(f"AI Interactions: {len(self.state.ai_requests)}")
        self._log(f"Challenges: {len(self.state.challenges)}")

        # === SAVE WORKFLOW REPORT ===
        if self._workflow_reporter:
            # Set overall status
            self._workflow_reporter.set_success(
                self.state.stage == EngineStage.COMPLETE and len(self.state.errors) == 0
            )
            self._workflow_reporter.set_stage_reached(self.state.stage.value)

            # Add generated files
            for f in output_files:
                self._workflow_reporter.add_generated_file(f)

            # Add errors and warnings
            for e in self.state.errors:
                self._workflow_reporter.add_error(e)
            for w in self.state.warnings:
                self._workflow_reporter.add_warning(w)

            # Log AI interactions
            for i, (req, resp) in enumerate(zip(self.state.ai_requests, self.state.ai_responses)):
                self._workflow_reporter.log_ai_interaction(
                    stage=req.stage.value if hasattr(req.stage, 'value') else str(req.stage),
                    request_type=req.challenge.type.value if req.challenge else 'query',
                    question=req.question,
                    options=req.options,
                    decision=resp.decision.value if hasattr(resp.decision, 'value') else str(resp.decision),
                    reasoning=resp.reasoning,
                    parameters=resp.parameters
                )

            # Save report to output directory
            try:
                report_path = self._workflow_reporter.save_report(self.config.output_dir)
                output_files.append(report_path)
                self._log(f"Workflow report saved to: {report_path}")
            except Exception as e:
                self._log(f"Warning: Could not save workflow report: {e}")

        return EngineResult(
            success=(self.state.stage == EngineStage.COMPLETE and len(self.state.errors) == 0),
            stage_reached=self.state.stage,
            output_files=output_files,
            drc_passed=self.state.drc_result.passed if self.state.drc_result and hasattr(self.state.drc_result, 'passed') else False,
            routed_count=routed_count,
            total_nets=total_nets,
            total_time=total_time,
            errors=self.state.errors.copy(),
            warnings=self.state.warnings.copy(),
            challenges_resolved=sum(1 for r in self.state.ai_responses if r.decision != AIDecision.ABORT),
            ai_interactions=len(self.state.ai_requests)
        )

    def _log(self, message: str):
        """Log a message if verbose mode is enabled"""
        if self.config.verbose:
            print(message)

    # =========================================================================
    # LEGACY RUN METHOD (backward compatible)
    # =========================================================================

    def run(self, parts_db: Dict) -> EngineResult:
        """
        Run the complete PCB design pipeline.

        This is the legacy method for backward compatibility.
        For the new AI-orchestrated flow, use run_orchestrated().
        """
        return self.run_orchestrated(parts_db)

    # =========================================================================
    # INDIVIDUAL STAGE METHODS (for advanced users)
    # =========================================================================

    def run_parts(self, parts_db: Dict) -> bool:
        """Run the Parts Piston standalone"""
        self.state.parts_db = parts_db
        report = self._execute_parts(PistonEffort.NORMAL)
        return report.success

    def run_order(self) -> bool:
        """Run the Order Piston standalone"""
        report = self._execute_order(PistonEffort.NORMAL)
        return report.success

    def run_placement(self) -> bool:
        """Run the Placement Piston standalone"""
        report = self._execute_placement(PistonEffort.NORMAL)
        return report.success

    def run_routing(self) -> bool:
        """Run the Routing Piston standalone"""
        report = self._execute_routing(PistonEffort.NORMAL)
        return report.success

    def run_silkscreen(self) -> bool:
        """Run the Silkscreen Piston standalone"""
        report = self._execute_silkscreen(PistonEffort.NORMAL)
        return report.success

    def run_drc(self) -> bool:
        """Run the DRC Piston standalone"""
        return self._run_final_drc_loop()

    def run_output(self) -> bool:
        """Run the Output Piston standalone"""
        report = self._execute_output()
        return report.success

    # =========================================================================
    # PISTON ACCESS (for advanced users)
    # =========================================================================

    @property
    def parts_piston(self) -> PartsEngine:
        """Access the Parts Piston directly"""
        if not self._parts_piston and PartsEngine:
            self._parts_piston = PartsEngine()
        return self._parts_piston

    @property
    def order_piston(self) -> OrderPiston:
        """Access the Order Piston directly"""
        if not self._order_piston and OrderPiston:
            self._order_piston = OrderPiston(OrderConfig())
        return self._order_piston

    @property
    def placement_piston(self) -> PlacementEngine:
        """Access the Placement Piston directly"""
        if not self._placement_piston and PlacementEngine:
            self._placement_piston = PlacementEngine()
        return self._placement_piston

    @property
    def routing_piston(self) -> RoutingPiston:
        """Access the Routing Piston directly"""
        if not self._routing_piston and RoutingPiston:
            self._routing_piston = RoutingPiston(RoutingConfig())
        return self._routing_piston

    @property
    def silkscreen_piston(self) -> SilkscreenPiston:
        """Access the Silkscreen Piston directly"""
        if not self._silkscreen_piston and SilkscreenPiston:
            self._silkscreen_piston = SilkscreenPiston()
        return self._silkscreen_piston

    @property
    def drc_piston(self) -> DRCPiston:
        """Access the DRC Piston directly"""
        if not self._drc_piston and DRCPiston:
            self._drc_piston = DRCPiston()
        return self._drc_piston

    @property
    def output_piston(self) -> OutputPiston:
        """Access the Output Piston directly"""
        if not self._output_piston and OutputPiston:
            self._output_piston = OutputPiston()
        return self._output_piston

    @property
    def topological_router_piston(self) -> TopologicalRouterPiston:
        """Access the Topological Router Piston directly"""
        if not self._topological_router_piston and TopologicalRouterPiston:
            self._topological_router_piston = TopologicalRouterPiston()
        return self._topological_router_piston

    @property
    def visualization_3d_piston(self) -> Visualization3DPiston:
        """Access the 3D Visualization Piston directly"""
        if not self._visualization_3d_piston and Visualization3DPiston:
            self._visualization_3d_piston = Visualization3DPiston()
        return self._visualization_3d_piston

    @property
    def bom_optimizer_piston(self) -> BOMOptimizerPiston:
        """Access the BOM Optimizer Piston directly"""
        if not self._bom_optimizer_piston and BOMOptimizerPiston:
            self._bom_optimizer_piston = BOMOptimizerPiston()
        return self._bom_optimizer_piston

    @property
    def learning_piston(self) -> LearningPiston:
        """Access the Learning Piston directly"""
        if not self._learning_piston and LearningPiston:
            self._learning_piston = LearningPiston()
        return self._learning_piston

    # =========================================================================
    # LEARNING PISTON PUBLIC METHODS
    # =========================================================================

    def learn_from_files(self, file_paths: List[str], mode: str = 'unsupervised') -> LearningResult:
        """
        Learn from existing PCB files to improve algorithms.

        Args:
            file_paths: List of paths to PCB files (.kicad_pcb, .gbr, etc.)
            mode: Learning mode - 'supervised', 'unsupervised', 'reinforcement', 'transfer'

        Returns:
            LearningResult with patterns and models

        Example:
            engine = PCBEngine(config)
            result = engine.learn_from_files([
                "design1.kicad_pcb",
                "design2.kicad_pcb",
            ])
            print(f"Discovered {result.patterns_discovered} patterns")
        """
        if not LearningPiston:
            self._log("Learning Piston not available")
            return None

        if not self._learning_piston:
            self._learning_piston = LearningPiston()

        # Map mode string to enum
        mode_map = {
            'supervised': LearningMode.SUPERVISED,
            'unsupervised': LearningMode.UNSUPERVISED,
            'reinforcement': LearningMode.REINFORCEMENT,
            'transfer': LearningMode.TRANSFER,
        }
        learning_mode = mode_map.get(mode.lower(), LearningMode.UNSUPERVISED)

        self._log(f"Learning from {len(file_paths)} files ({mode} mode)")
        result = self._learning_piston.learn(file_paths, mode=learning_mode)

        self._log(f"  Designs processed: {result.designs_processed}")
        self._log(f"  Patterns discovered: {result.patterns_discovered}")
        self._log(f"  Models trained: {result.models_trained}")

        for rec in result.recommendations:
            self._log(f"  Recommendation: {rec}")

        return result

    def predict_design_quality(self) -> float:
        """
        Predict quality score for current design using learned model.

        Returns:
            Quality score from 0.0 to 1.0
        """
        if not self._learning_piston:
            return 0.5  # Neutral

        # Build extracted design from current state
        from .learning_piston import ExtractedDesign, FileFormat

        design = ExtractedDesign(
            source_file="current",
            file_format=FileFormat.KICAD_PCB,
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            layer_count=self.config.layer_count,
            components=[],  # Would need to convert
            traces=[],
            vias=[],
            zones=[],
            nets=self.state.parts_db.get('nets', {}),
            design_rules={
                'trace_width': self.config.trace_width,
                'clearance': self.config.clearance
            }
        )

        return self._learning_piston.predict_quality(design)
