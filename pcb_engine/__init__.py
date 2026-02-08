"""
PCB Design Engine - 18 Piston Architecture + BBL
==================================================

A universal engine for automated PCB design using the Foreman-Piston architecture
with the Big Beautiful Loop (BBL) for complete work cycle management.

The engine coordinates 18 specialized pistons (workers) through a feedback loop:
- DRC watchdog monitors all work
- Each piston can retry with escalating effort levels
- AI Agent handles challenges that require intelligence
- BBL manages the complete work cycle with 6 improvements

CORE PISTONS:
    Feasibility → Parts → Order → Placement → Escape → Routing → Optimize → Silkscreen → DRC → Output

ANALYSIS PISTONS:
    Stackup, Thermal, PDN, Signal Integrity, Netlist

ADVANCED PISTONS:
    Topological Router, 3D Visualization, BOM Optimizer

LEARNING PISTON:
    Learns from successful designs to improve over time

BIG BEAUTIFUL LOOP (BBL) - 6 Improvements:
    1. CHECKPOINTS - Engine checks after each phase: continue, escalate, or abort
    2. ROLLBACK - Save state before each phase, rollback on failure
    3. TIMEOUT - Each phase has max time, auto-escalate if exceeded
    4. PROGRESS REPORTING - Real-time progress updates
    5. PARALLEL EXECUTION - Independent phases run in parallel
    6. LOOP HISTORY - Record every BBL run for analytics

Usage:
    # Standard usage
    from pcb_engine import PCBEngine, EngineConfig

    config = EngineConfig(
        board_width=100,
        board_height=80,
        verbose=True
        # output_dir defaults to D:\Anas\tmp\output (see paths.py)
    )

    engine = PCBEngine(config)
    result = engine.run(parts_db)

    # BBL usage with progress callback
    def on_progress(progress):
        print(f"{progress.phase}: {progress.percentage}% - {progress.message}")

    result = engine.run_bbl(parts_db, progress_callback=on_progress)
"""

__version__ = '2.1.0'
__author__ = 'PCB Engine Team'

# Main Engine
from .pcb_engine import PCBEngine, EngineConfig, EngineResult, EngineState, EngineStage, PISTON_CASCADES

# Core Pistons
from .parts_piston import PartsPiston, PartsConfig, PartsResult
from .order_piston import OrderPiston, OrderConfig, OrderResult
from .placement_engine import PlacementEngine, PlacementConfig
from .placement_piston import PlacementPiston
from .routing_types import (
    TrackSegment, Via, Route, RoutingAlgorithm,
    RoutingConfig, RoutingResult, create_track_segment, create_via
)
from .routing_piston import RoutingPiston, route_with_cascade, ROUTING_ALGORITHMS
from .routing_engine import RoutingEngine
from .escape_piston import EscapePiston, EscapeConfig, EscapeResult
from .optimization_piston import OptimizationPiston, OptimizationConfig, OptimizationResult
from .silkscreen_piston import SilkscreenPiston, SilkscreenConfig, SilkscreenResult
from .drc_piston import DRCPiston, DRCConfig, DRCResult, DRCRules
from .output_piston import OutputPiston, OutputConfig, OutputResult

# Analysis Pistons
from .stackup_piston import StackupPiston, StackupConfig
from .thermal_piston import ThermalPiston, ThermalConfig
from .pdn_piston import PDNPiston, PDNConfig
from .signal_integrity_piston import SignalIntegrityPiston, SIConfig
from .netlist_piston import NetlistPiston

# Advanced Pistons
from .topological_router_piston import TopologicalRouterPiston
from .visualization_3d_piston import Visualization3DPiston
from .bom_optimizer_piston import BOMOptimizerPiston

# Learning & AI
from .learning_piston import LearningPiston, LearningMode, LearningResult
from .circuit_ai import CircuitAI, CircuitAIResult
from .ai_connector import AIConnector

# KiCad DRC Teacher - THE AUTHORITY (teaches our DRC)
from .kicad_drc_teacher import KiCadDRCTeacher, TeacherResult, LearningRecord

# Pour Piston - Ground/Power planes
from .pour_piston import PourPiston, PourConfig, PourResult, PourZone

# Feasibility Piston - Pre-flight check (runs FIRST)
from .feasibility_piston import (
    FeasibilityPiston, FeasibilityConfig, FeasibilityResult,
    FeasibilityStatus, FeasibilityIssue, Severity, check_feasibility
)

# Orchestration
from .piston_orchestrator import PistonOrchestrator, PistonSelection
from .workflow_reporter import WorkflowReporter, WorkflowReport

# CASCADE Optimizer - Dynamic Algorithm Ordering
from .cascade_optimizer import (
    CascadeOptimizer, CascadeOptimizerConfig, DesignProfile as CascadeDesignProfile,
    DesignComplexity, DesignDensity, DesignType, LayerCount,
    AlgorithmStats, DEFAULT_CASCADES, create_optimizer, get_optimized_cascades
)

# BBL Engine (Big Beautiful Loop) - Complete Work Cycle Manager
from .bbl_engine import (
    BBLEngine, BBLState, BBLResult, BBLPhase, BBLProgress,
    BBLCheckpoint, BBLEscalation, BBLHistoryEntry, BBLPhaseConfig,
    BBLCheckpointDecision, BBLEscalationLevel, BBLPriority
)

# BBL Monitor Sensor - Watches and documents everything in the BBL
from .bbl_monitor import (
    BBLMonitor, MonitorEvent, MonitorLevel,
    EventRecord, SessionMetrics, PistonMetrics, PhaseMetrics,
    AlgorithmMetrics, PerformanceRanking
)

# Utilities
from .grid_calculator import (
    calculate_optimal_grid_size, calculate_min_pad_pitch,
    estimate_grid_memory, print_grid_analysis, quick_grid_estimate,
    suggest_performance_tier, get_footprint_pitch, get_grid_for_tier,
    PerformanceTier, FOOTPRINT_PITCHES, TIER_GRID_SIZES
)
from .common_types import (
    Position, normalize_position, get_xy,
    get_pins, get_pin_net, get_pin_offset, get_pin_position
)

__all__ = [
    # Main Engine
    'PCBEngine', 'EngineConfig', 'EngineResult', 'EngineState', 'EngineStage', 'PISTON_CASCADES',

    # Core Pistons
    'PartsPiston', 'PartsConfig', 'PartsResult',
    'OrderPiston', 'OrderConfig', 'OrderResult',
    'PlacementEngine', 'PlacementConfig',
    'PlacementPiston',
    'TrackSegment', 'Via', 'Route', 'RoutingAlgorithm',
    'RoutingConfig', 'RoutingResult', 'create_track_segment', 'create_via',
    'RoutingPiston', 'RoutingEngine', 'route_with_cascade', 'ROUTING_ALGORITHMS',
    'EscapePiston', 'EscapeConfig', 'EscapeResult',
    'OptimizationPiston', 'OptimizationConfig', 'OptimizationResult',
    'SilkscreenPiston', 'SilkscreenConfig', 'SilkscreenResult',
    'DRCPiston', 'DRCConfig', 'DRCResult', 'DRCRules',
    'OutputPiston', 'OutputConfig', 'OutputResult',

    # Analysis Pistons
    'StackupPiston', 'StackupConfig',
    'ThermalPiston', 'ThermalConfig',
    'PDNPiston', 'PDNConfig',
    'SignalIntegrityPiston', 'SIConfig',
    'NetlistPiston',

    # Advanced Pistons
    'TopologicalRouterPiston',
    'Visualization3DPiston',
    'BOMOptimizerPiston',

    # Learning & AI
    'LearningPiston', 'LearningMode', 'LearningResult',
    'CircuitAI', 'CircuitAIResult',
    'AIConnector',

    # KiCad DRC Teacher - THE AUTHORITY
    'KiCadDRCTeacher', 'TeacherResult', 'LearningRecord',

    # Pour Piston
    'PourPiston', 'PourConfig', 'PourResult', 'PourZone',

    # Feasibility Piston - Pre-flight check
    'FeasibilityPiston', 'FeasibilityConfig', 'FeasibilityResult',
    'FeasibilityStatus', 'FeasibilityIssue', 'Severity', 'check_feasibility',

    # Orchestration
    'PistonOrchestrator', 'PistonSelection',
    'WorkflowReporter', 'WorkflowReport',

    # CASCADE Optimizer - Dynamic Algorithm Ordering
    'CascadeOptimizer', 'CascadeOptimizerConfig', 'CascadeDesignProfile',
    'DesignComplexity', 'DesignDensity', 'DesignType', 'LayerCount',
    'AlgorithmStats', 'DEFAULT_CASCADES', 'create_optimizer', 'get_optimized_cascades',

    # BBL Engine (Big Beautiful Loop)
    'BBLEngine', 'BBLState', 'BBLResult', 'BBLPhase', 'BBLProgress',
    'BBLCheckpoint', 'BBLEscalation', 'BBLHistoryEntry', 'BBLPhaseConfig',
    'BBLCheckpointDecision', 'BBLEscalationLevel', 'BBLPriority',

    # Utilities
    'calculate_optimal_grid_size', 'calculate_min_pad_pitch',
    'estimate_grid_memory', 'print_grid_analysis', 'quick_grid_estimate',
    'suggest_performance_tier', 'get_footprint_pitch', 'get_grid_for_tier',
    'PerformanceTier', 'FOOTPRINT_PITCHES', 'TIER_GRID_SIZES',
    'Position', 'normalize_position', 'get_xy',
    'get_pins', 'get_pin_net', 'get_pin_offset', 'get_pin_position',
]
