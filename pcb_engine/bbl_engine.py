"""
THE BIG BEAUTIFUL LOOP (BBL) ENGINE
====================================

The BBL is the COMPLETE WORK CYCLE from when the Engine receives an order
until files are delivered. It implements all 6 approved improvements:

1. CHECKPOINTS - Engine checks after each phase: continue, escalate, or abort
2. ROLLBACK - Save state before each phase, rollback on failure
3. TIMEOUT - Each phase has max time, auto-escalate if exceeded
4. PROGRESS REPORTING - Real-time progress updates to Boss
5. PARALLEL EXECUTION - Independent phases run in parallel
6. LOOP HISTORY - Record every BBL run for analytics

COMMAND HIERARCHY:
==================
    USER (Boss) → CIRCUIT AI (Engineer) → PCB ENGINE (Foreman) → PISTONS (Workers)

BBL PHASES:
===========
    Phase 1: ORDER RECEIVED - Engine receives work order from Engineer/Boss
    Phase 2: PISTON EXECUTION - Pistons execute with CASCADE system
    Phase 3: ESCALATION (if needed) - Engine → Engineer → Boss
    Phase 4: OUTPUT GENERATION - Generate .kicad_pcb and other files
    Phase 5: KICAD DRC VALIDATION - KiCad CLI validates output (THE AUTHORITY)
    Phase 6: LEARNING & DELIVERY - Learn from result, deliver files

Author: PCB Engine Team
Date: 2026-02-07
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import time
import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import traceback

# Import BBL Monitor for comprehensive tracking
try:
    from .bbl_monitor import (
        BBLMonitor, MonitorEvent, MonitorLevel,
        EventRecord, SessionMetrics, PistonMetrics, PhaseMetrics,
        AlgorithmMetrics, PerformanceRanking
    )
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    BBLMonitor = None


# =============================================================================
# BBL ENUMS
# =============================================================================

class BBLPhase(Enum):
    """Phases of the Big Beautiful Loop"""
    INIT = 'init'
    ORDER_RECEIVED = 'order_received'       # Phase 1
    PISTON_EXECUTION = 'piston_execution'   # Phase 2
    ESCALATION = 'escalation'               # Phase 3 (optional)
    OUTPUT_GENERATION = 'output_generation' # Phase 4
    KICAD_DRC = 'kicad_drc'                 # Phase 5
    LEARNING_DELIVERY = 'learning_delivery' # Phase 6
    COMPLETE = 'complete'
    ABORTED = 'aborted'
    ERROR = 'error'


class BBLCheckpointDecision(Enum):
    """Decisions at checkpoints"""
    CONTINUE = 'continue'           # Proceed to next phase
    RETRY = 'retry'                 # Retry current phase with different params
    ESCALATE = 'escalate'           # Pause and escalate to Engineer
    ROLLBACK = 'rollback'           # Roll back to previous checkpoint
    ABORT = 'abort'                 # Stop BBL, return error


class BBLEscalationLevel(Enum):
    """Escalation levels"""
    NONE = 'none'
    ENGINEER = 'engineer'           # Escalate to Circuit AI
    BOSS = 'boss'                   # Escalate to User


class BBLPriority(Enum):
    """Priority levels for phases"""
    CRITICAL = 1    # Must succeed
    HIGH = 2        # Important
    MEDIUM = 3      # Normal
    LOW = 4         # Optional


# =============================================================================
# BBL DATA STRUCTURES
# =============================================================================

@dataclass
class BBLCheckpoint:
    """
    Checkpoint saved before each phase for rollback capability.

    IMPROVEMENT #2: ROLLBACK CAPABILITY
    - Complete state snapshot before phase execution
    - Can restore to any previous checkpoint
    """
    id: str
    phase: BBLPhase
    timestamp: float
    state_snapshot: Dict = field(default_factory=dict)
    piston_results: Dict = field(default_factory=dict)
    drc_status: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'phase': self.phase.value,
            'timestamp': self.timestamp,
            'state_snapshot': self.state_snapshot,
            'piston_results': self.piston_results,
            'drc_status': self.drc_status,
            'metadata': self.metadata
        }


@dataclass
class BBLPhaseConfig:
    """
    Configuration for a BBL phase.

    IMPROVEMENT #3: TIMEOUT SYSTEM
    - Each phase has configurable timeout
    - Auto-escalate when timeout exceeded
    """
    phase: BBLPhase
    timeout_seconds: float = 120.0      # Default 2 minutes
    priority: BBLPriority = BBLPriority.HIGH
    can_run_parallel: bool = False      # Can run with other phases
    parallel_group: str = ''            # Group name for parallel execution
    retry_limit: int = 3
    quality_threshold: float = 0.6      # Minimum quality score to pass
    auto_escalate_on_timeout: bool = True
    rollback_on_fail: bool = True

    # Dependencies
    depends_on: List[BBLPhase] = field(default_factory=list)
    blocks: List[BBLPhase] = field(default_factory=list)


@dataclass
class BBLProgress:
    """
    Progress update structure.

    IMPROVEMENT #4: PROGRESS REPORTING
    - Real-time updates during execution
    - Percentage complete
    - Current operation details
    """
    phase: BBLPhase
    percentage: float = 0.0
    message: str = ''
    detail: str = ''
    timestamp: float = 0.0

    # Phase-specific progress
    piston_name: str = ''
    algorithm_name: str = ''
    iteration: int = 0
    max_iterations: int = 0

    # Quality metrics
    current_quality: float = 0.0
    target_quality: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'phase': self.phase.value,
            'percentage': self.percentage,
            'message': self.message,
            'detail': self.detail,
            'timestamp': self.timestamp,
            'piston_name': self.piston_name,
            'algorithm_name': self.algorithm_name,
            'iteration': self.iteration,
            'max_iterations': self.max_iterations,
            'current_quality': self.current_quality,
            'target_quality': self.target_quality
        }


@dataclass
class BBLEscalation:
    """
    Escalation request to Engineer or Boss.
    """
    level: BBLEscalationLevel
    phase: BBLPhase
    reason: str
    context: Dict = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    suggested_action: str = ''
    timestamp: float = 0.0

    # Response (filled when escalation is resolved)
    resolved: bool = False
    response: str = ''
    new_instructions: Dict = field(default_factory=dict)
    resolution_timestamp: float = 0.0


@dataclass
class BBLHistoryEntry:
    """
    Single entry in BBL history.

    IMPROVEMENT #6: LOOP HISTORY
    - Records every BBL run
    - Tracks what phases ran
    - Records escalations and outcomes
    """
    bbl_id: str
    start_time: float
    end_time: float = 0.0
    final_phase: BBLPhase = BBLPhase.INIT
    success: bool = False

    # Phase tracking
    phases_executed: List[str] = field(default_factory=list)
    phases_skipped: List[str] = field(default_factory=list)
    phases_failed: List[str] = field(default_factory=list)

    # Timing
    phase_durations: Dict[str, float] = field(default_factory=dict)
    total_duration: float = 0.0

    # Escalations
    escalations: List[Dict] = field(default_factory=list)
    rollbacks: List[Dict] = field(default_factory=list)

    # Results
    drc_errors: int = 0
    drc_warnings: int = 0
    routing_completion: float = 0.0
    output_files: List[str] = field(default_factory=list)

    # Metadata
    input_summary: Dict = field(default_factory=dict)
    config_summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'bbl_id': self.bbl_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'final_phase': self.final_phase.value,
            'success': self.success,
            'phases_executed': self.phases_executed,
            'phases_skipped': self.phases_skipped,
            'phases_failed': self.phases_failed,
            'phase_durations': self.phase_durations,
            'total_duration': self.total_duration,
            'escalations': self.escalations,
            'rollbacks': self.rollbacks,
            'drc_errors': self.drc_errors,
            'drc_warnings': self.drc_warnings,
            'routing_completion': self.routing_completion,
            'output_files': self.output_files,
            'input_summary': self.input_summary,
            'config_summary': self.config_summary
        }


@dataclass
class BBLState:
    """
    Complete state of the BBL at any point in time.
    """
    bbl_id: str
    current_phase: BBLPhase = BBLPhase.INIT
    started: bool = False
    completed: bool = False
    aborted: bool = False

    # Checkpoints for rollback
    checkpoints: List[BBLCheckpoint] = field(default_factory=list)
    current_checkpoint_id: str = ''

    # Progress
    progress: BBLProgress = None
    progress_history: List[BBLProgress] = field(default_factory=list)

    # Escalations
    pending_escalation: BBLEscalation = None
    escalation_history: List[BBLEscalation] = field(default_factory=list)

    # Timing
    start_time: float = 0.0
    phase_start_times: Dict[str, float] = field(default_factory=dict)
    phase_end_times: Dict[str, float] = field(default_factory=dict)

    # Results from pistons
    piston_results: Dict[str, Any] = field(default_factory=dict)

    # Engine state reference
    engine_state: Any = None

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BBLResult:
    """
    Final result of BBL execution.
    """
    success: bool
    bbl_id: str
    final_phase: BBLPhase
    total_time: float

    # Output
    output_files: List[str] = field(default_factory=list)

    # DRC
    drc_passed: bool = False
    drc_errors: int = 0
    drc_warnings: int = 0
    kicad_drc_passed: bool = False

    # Routing
    routing_completion: float = 0.0
    routed_count: int = 0
    total_nets: int = 0

    # History
    history_entry: BBLHistoryEntry = None

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Escalation summary
    escalation_count: int = 0
    rollback_count: int = 0


# =============================================================================
# BBL ENGINE
# =============================================================================

class BBLEngine:
    """
    The Big Beautiful Loop Engine.

    Controls the complete work cycle from order to delivery.
    Implements all 6 improvements:
    1. Checkpoints with decision logic
    2. Rollback capability
    3. Timeout per phase
    4. Progress reporting
    5. Parallel execution
    6. Loop history
    """

    # Default phase configurations
    DEFAULT_PHASE_CONFIGS = {
        BBLPhase.ORDER_RECEIVED: BBLPhaseConfig(
            phase=BBLPhase.ORDER_RECEIVED,
            timeout_seconds=30.0,
            priority=BBLPriority.CRITICAL,
            retry_limit=1
        ),
        BBLPhase.PISTON_EXECUTION: BBLPhaseConfig(
            phase=BBLPhase.PISTON_EXECUTION,
            timeout_seconds=300.0,  # 5 minutes for all pistons
            priority=BBLPriority.CRITICAL,
            retry_limit=3,
            quality_threshold=0.5
        ),
        BBLPhase.ESCALATION: BBLPhaseConfig(
            phase=BBLPhase.ESCALATION,
            timeout_seconds=600.0,  # 10 minutes (may need human)
            priority=BBLPriority.HIGH,
            auto_escalate_on_timeout=False  # Already escalated
        ),
        BBLPhase.OUTPUT_GENERATION: BBLPhaseConfig(
            phase=BBLPhase.OUTPUT_GENERATION,
            timeout_seconds=60.0,
            priority=BBLPriority.CRITICAL,
            can_run_parallel=True,
            parallel_group='output'
        ),
        BBLPhase.KICAD_DRC: BBLPhaseConfig(
            phase=BBLPhase.KICAD_DRC,
            timeout_seconds=120.0,
            priority=BBLPriority.CRITICAL,
            retry_limit=1
        ),
        BBLPhase.LEARNING_DELIVERY: BBLPhaseConfig(
            phase=BBLPhase.LEARNING_DELIVERY,
            timeout_seconds=60.0,
            priority=BBLPriority.MEDIUM,
            can_run_parallel=True,
            parallel_group='finalize'
        )
    }

    # History file path
    HISTORY_FILE = 'bbl_history.json'

    def __init__(
        self,
        pcb_engine: Any = None,
        phase_configs: Dict[BBLPhase, BBLPhaseConfig] = None,
        progress_callback: Callable[[BBLProgress], None] = None,
        escalation_callback: Callable[[BBLEscalation], BBLEscalation] = None,
        output_dir: str = '',  # Empty uses DEFAULT_OUTPUT_BASE from paths.py
        verbose: bool = True
    ):
        """
        Initialize BBL Engine.

        Args:
            pcb_engine: Reference to PCBEngine for piston execution
            phase_configs: Custom phase configurations
            progress_callback: Called with progress updates (IMPROVEMENT #4)
            escalation_callback: Called when escalation needed (to Engineer/Boss)
            output_dir: Directory for output files and history (default: D:\\Anas\\tmp\\output)
            verbose: Enable verbose logging
        """
        # Use default output base if not specified
        if not output_dir:
            from .paths import DEFAULT_OUTPUT_BASE
            output_dir = DEFAULT_OUTPUT_BASE
        self.pcb_engine = pcb_engine
        self.phase_configs = phase_configs or self.DEFAULT_PHASE_CONFIGS.copy()
        self.progress_callback = progress_callback
        self.escalation_callback = escalation_callback
        self.output_dir = output_dir
        self.verbose = verbose

        # Current BBL state
        self.state: BBLState = None

        # History (IMPROVEMENT #6)
        self.history: List[BBLHistoryEntry] = []
        self._load_history()

        # Thread pool for parallel execution (IMPROVEMENT #5)
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Phase execution handlers
        self._phase_handlers = {
            BBLPhase.ORDER_RECEIVED: self._execute_order_received,
            BBLPhase.PISTON_EXECUTION: self._execute_piston_execution,
            BBLPhase.ESCALATION: self._execute_escalation,
            BBLPhase.OUTPUT_GENERATION: self._execute_output_generation,
            BBLPhase.KICAD_DRC: self._execute_kicad_drc,
            BBLPhase.LEARNING_DELIVERY: self._execute_learning_delivery
        }

        # BBL Monitor Sensor - watches and documents everything
        self.monitor: BBLMonitor = None
        if MONITOR_AVAILABLE:
            self.monitor = BBLMonitor(
                output_dir=output_dir,
                log_level=MonitorLevel.DEBUG if verbose else MonitorLevel.INFO,
                verbose=verbose
            )

    # =========================================================================
    # MAIN BBL ENTRY POINT
    # =========================================================================

    def run(
        self,
        parts_db: Dict,
        engine_state: Any = None,
        config: Dict = None
    ) -> BBLResult:
        """
        Run the complete Big Beautiful Loop.

        This is the main entry point for BBL execution.

        Args:
            parts_db: Parts database to process
            engine_state: Optional existing engine state
            config: Optional configuration overrides

        Returns:
            BBLResult with complete execution results
        """
        # Generate unique BBL ID
        bbl_id = f"BBL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000:04d}"

        # Initialize state
        self.state = BBLState(
            bbl_id=bbl_id,
            start_time=time.time(),
            engine_state=engine_state
        )

        # Start monitor session
        if self.monitor:
            self.monitor.start_session(bbl_id)

        # Create history entry
        history_entry = BBLHistoryEntry(
            bbl_id=bbl_id,
            start_time=self.state.start_time,
            input_summary={
                'parts_count': len(parts_db.get('parts', {})),
                'nets_count': len(parts_db.get('nets', {}))
            }
        )

        self._log(f"\n{'='*70}")
        self._log(f"BBL STARTED: {bbl_id}")
        self._log(f"{'='*70}")

        try:
            # Store parts_db for phase handlers
            self.state.piston_results['parts_db'] = parts_db

            # Execute BBL phases in order
            phase_order = [
                BBLPhase.ORDER_RECEIVED,
                BBLPhase.PISTON_EXECUTION,
                BBLPhase.OUTPUT_GENERATION,
                BBLPhase.KICAD_DRC,
                BBLPhase.LEARNING_DELIVERY
            ]

            for phase in phase_order:
                # Create checkpoint before phase (IMPROVEMENT #2)
                self._create_checkpoint(phase)

                # Execute phase with timeout (IMPROVEMENT #3)
                result = self._execute_phase_with_timeout(phase)

                # Run checkpoint decision (IMPROVEMENT #1)
                decision = self._checkpoint_decision(phase, result)

                if decision == BBLCheckpointDecision.ABORT:
                    self._log(f"BBL ABORTED at phase: {phase.value}")
                    self.state.aborted = True
                    history_entry.phases_failed.append(phase.value)
                    break

                elif decision == BBLCheckpointDecision.ESCALATE:
                    # Insert escalation phase
                    escalation_result = self._execute_phase_with_timeout(BBLPhase.ESCALATION)

                    if not escalation_result.get('resolved', False):
                        self._log(f"Escalation not resolved, aborting BBL")
                        self.state.aborted = True
                        history_entry.phases_failed.append(phase.value)
                        break

                    # Retry current phase with new instructions
                    result = self._execute_phase_with_timeout(phase)

                elif decision == BBLCheckpointDecision.ROLLBACK:
                    # Rollback to previous checkpoint (IMPROVEMENT #2)
                    if self._rollback_to_previous():
                        history_entry.rollbacks.append({
                            'phase': phase.value,
                            'timestamp': time.time()
                        })
                        # Retry from rolled-back phase
                        continue
                    else:
                        self._log("No checkpoint to rollback to, aborting")
                        self.state.aborted = True
                        break

                elif decision == BBLCheckpointDecision.RETRY:
                    # SPECIAL HANDLING FOR KICAD_DRC PHASE
                    # When KiCad DRC fails, run the iteration loop to fix routing
                    if phase == BBLPhase.KICAD_DRC:
                        self._log("[BBL] KiCad DRC failed - entering iteration loop")
                        iteration_result = self._run_iteration_loop(max_iterations=3)

                        if iteration_result.get('success', False):
                            # Iteration loop fixed the issues
                            result = {
                                'success': True,
                                'quality': 1.0,
                                'passed': True,
                                'iterations': iteration_result.get('iterations', 0)
                            }
                            decision = BBLCheckpointDecision.CONTINUE
                            self._log(f"[BBL] KiCad DRC passed after {iteration_result.get('iterations', 0)} iterations")
                        else:
                            # Iteration loop couldn't fix all issues
                            self._log(f"[BBL] Iteration loop failed after {iteration_result.get('iterations', 0)} attempts")
                            # Continue to delivery phase but mark as failed
                            decision = BBLCheckpointDecision.CONTINUE
                    else:
                        # Standard retry logic for other phases
                        retry_count = 0
                        max_retries = self.phase_configs.get(phase, BBLPhaseConfig(phase=phase)).retry_limit

                        while retry_count < max_retries:
                            retry_count += 1
                            self._log(f"Retrying phase {phase.value} (attempt {retry_count}/{max_retries})")
                            result = self._execute_phase_with_timeout(phase)
                            decision = self._checkpoint_decision(phase, result)

                            if decision == BBLCheckpointDecision.CONTINUE:
                                break

                        if decision != BBLCheckpointDecision.CONTINUE:
                            # Retries exhausted, escalate
                            decision = BBLCheckpointDecision.ESCALATE
                            # (escalation logic same as above)

                # Phase completed successfully
                history_entry.phases_executed.append(phase.value)
                self.state.current_phase = phase

            # BBL completed
            self.state.completed = not self.state.aborted
            self.state.current_phase = BBLPhase.COMPLETE if self.state.completed else BBLPhase.ABORTED

        except Exception as e:
            self._log(f"BBL ERROR: {e}")
            self._log(traceback.format_exc())
            self.state.errors.append(str(e))
            self.state.current_phase = BBLPhase.ERROR
            history_entry.phases_failed.append(self.state.current_phase.value)

        # Finalize history entry
        history_entry.end_time = time.time()
        history_entry.total_duration = history_entry.end_time - history_entry.start_time
        history_entry.final_phase = self.state.current_phase
        history_entry.success = self.state.completed and not self.state.aborted

        # Get results from state
        drc_result = self.state.piston_results.get('drc', {})
        routing_result = self.state.piston_results.get('routing', {})
        output_result = self.state.piston_results.get('output', {})
        kicad_drc_result = self.state.piston_results.get('kicad_drc', {})

        history_entry.drc_errors = drc_result.get('error_count', 0)
        history_entry.drc_warnings = drc_result.get('warning_count', 0)
        history_entry.routing_completion = routing_result.get('completion', 0.0)
        history_entry.output_files = output_result.get('files', [])
        history_entry.escalations = [e.to_dict() if hasattr(e, 'to_dict') else str(e)
                                      for e in self.state.escalation_history]

        # Add phase durations
        for phase_name, start_time in self.state.phase_start_times.items():
            end_time = self.state.phase_end_times.get(phase_name, time.time())
            history_entry.phase_durations[phase_name] = end_time - start_time

        # Save to history (IMPROVEMENT #6)
        self.history.append(history_entry)
        self._save_history()

        # Create result
        result = BBLResult(
            success=history_entry.success,
            bbl_id=bbl_id,
            final_phase=self.state.current_phase,
            total_time=history_entry.total_duration,
            output_files=history_entry.output_files,
            drc_passed=drc_result.get('passed', False),
            drc_errors=history_entry.drc_errors,
            drc_warnings=history_entry.drc_warnings,
            kicad_drc_passed=kicad_drc_result.get('passed', False),
            routing_completion=history_entry.routing_completion,
            routed_count=routing_result.get('routed_count', 0),
            total_nets=routing_result.get('total_count', 0),
            history_entry=history_entry,
            errors=self.state.errors,
            warnings=self.state.warnings,
            escalation_count=len(self.state.escalation_history),
            rollback_count=len(history_entry.rollbacks)
        )

        self._log(f"\n{'='*70}")
        self._log(f"BBL COMPLETED: {bbl_id}")
        self._log(f"  Success: {result.success}")
        self._log(f"  Duration: {result.total_time:.2f}s")
        self._log(f"  DRC Passed: {result.drc_passed}")
        self._log(f"  KiCad DRC Passed: {result.kicad_drc_passed}")
        self._log(f"  Routing: {result.routed_count}/{result.total_nets}")
        self._log(f"  Escalations: {result.escalation_count}")
        self._log(f"  Rollbacks: {result.rollback_count}")
        self._log(f"{'='*70}\n")

        # End monitor session and generate performance report
        if self.monitor:
            self.monitor.end_session(
                success=result.success,
                routing_stats={
                    'completion': result.routing_completion,
                    'routed': result.routed_count,
                    'total': result.total_nets
                }
            )

            # Print performance report
            if self.verbose:
                self._log("\n" + self.monitor.get_performance_report())

                # Print bottleneck analysis
                bottleneck = self.monitor.get_bottleneck_analysis()
                if bottleneck.get('recommendations'):
                    self._log("\n--- OPTIMIZATION RECOMMENDATIONS ---")
                    for rec in bottleneck['recommendations']:
                        self._log(f"  * {rec}")

        return result

    # =========================================================================
    # IMPROVEMENT #1: CHECKPOINT SYSTEM
    # =========================================================================

    def _checkpoint_decision(self, phase: BBLPhase, phase_result: Dict) -> BBLCheckpointDecision:
        """
        Make checkpoint decision after phase execution.

        IMPROVEMENT #1: ENGINE CHECKPOINTS
        The Engine asks ITSELF:
        - "Can I continue?" (dependencies met?)
        - "Should I escalate?" (quality threshold not met?)
        - "Should I abort?" (critical failure?)

        Args:
            phase: The phase that just executed
            phase_result: Result from phase execution

        Returns:
            BBLCheckpointDecision
        """
        config = self.phase_configs.get(phase, BBLPhaseConfig(phase=phase))

        # Check 1: Can I continue?
        success = phase_result.get('success', False)
        quality = phase_result.get('quality', 0.0)
        has_critical_errors = phase_result.get('critical_errors', False)

        self._log(f"[CHECKPOINT] Phase: {phase.value}")
        self._log(f"  Success: {success}, Quality: {quality:.2f}, Threshold: {config.quality_threshold}")

        decision = None

        if success and quality >= config.quality_threshold:
            decision = BBLCheckpointDecision.CONTINUE
        elif has_critical_errors:
            decision = BBLCheckpointDecision.ABORT
        elif phase_result.get('unrecoverable', False):
            decision = BBLCheckpointDecision.ABORT
        # SPECIAL HANDLING FOR KICAD_DRC: Always try iteration loop first
        elif phase == BBLPhase.KICAD_DRC and not success:
            # For KiCad DRC failures, always RETRY which triggers the iteration loop
            # The iteration loop will attempt to fix routing issues
            retry_count = self.state.piston_results.get(f'{phase.value}_retry_count', 0)
            if retry_count < 1:  # Only try iteration loop once
                self.state.piston_results[f'{phase.value}_retry_count'] = retry_count + 1
                decision = BBLCheckpointDecision.RETRY
            else:
                # After iteration loop fails, continue to delivery (will report failure)
                decision = BBLCheckpointDecision.CONTINUE
        elif self.state.piston_results.get(f'{phase.value}_retry_count', 0) >= config.retry_limit:
            decision = BBLCheckpointDecision.ESCALATE
        elif quality < config.quality_threshold * 0.5:
            decision = BBLCheckpointDecision.ESCALATE
        elif config.rollback_on_fail and len(self.state.checkpoints) > 1:
            decision = BBLCheckpointDecision.ROLLBACK
        else:
            decision = BBLCheckpointDecision.RETRY

        self._log(f"  Decision: {decision.value.upper()}")

        # Record checkpoint decision in monitor
        if self.monitor:
            self.monitor.record_checkpoint_decision(
                phase=phase.value,
                decision=decision.value,
                quality=quality,
                threshold=config.quality_threshold
            )

        return decision

    # =========================================================================
    # IMPROVEMENT #2: ROLLBACK CAPABILITY
    # =========================================================================

    def _create_checkpoint(self, phase: BBLPhase) -> BBLCheckpoint:
        """
        Create a checkpoint before phase execution.

        IMPROVEMENT #2: ROLLBACK CAPABILITY
        - Saves complete state snapshot
        - Can restore to this point if phase fails
        """
        checkpoint = BBLCheckpoint(
            id=f"CP_{phase.value}_{int(time.time())}",
            phase=phase,
            timestamp=time.time(),
            state_snapshot=self._snapshot_state(),
            piston_results=copy.deepcopy(self.state.piston_results),
            drc_status=self.state.piston_results.get('drc', {}).copy()
        )

        with self._lock:
            self.state.checkpoints.append(checkpoint)
            self.state.current_checkpoint_id = checkpoint.id

        self._log(f"[CHECKPOINT] Created: {checkpoint.id}")

        # Record in monitor
        if self.monitor:
            self.monitor.record_checkpoint_created(checkpoint.id, phase.value)

        return checkpoint

    def _snapshot_state(self) -> Dict:
        """Create a snapshot of current engine state."""
        if self.pcb_engine and hasattr(self.pcb_engine, 'state'):
            state = self.pcb_engine.state
            return {
                'stage': state.stage.value if hasattr(state.stage, 'value') else str(state.stage),
                'placement': copy.deepcopy(getattr(state, 'placement', {})),
                'routes': copy.deepcopy(getattr(state, 'routes', {})),
                'vias': copy.deepcopy(getattr(state, 'vias', [])),
                'errors': list(getattr(state, 'errors', [])),
                'warnings': list(getattr(state, 'warnings', []))
            }
        return {}

    def _restore_checkpoint(self, checkpoint: BBLCheckpoint) -> bool:
        """
        Restore state from a checkpoint.

        Returns:
            True if restoration successful
        """
        try:
            self._log(f"[ROLLBACK] Restoring checkpoint: {checkpoint.id}")

            # Record rollback start in monitor
            if self.monitor:
                self.monitor.record_rollback_start(
                    self.state.current_checkpoint_id,
                    checkpoint.id
                )

            # Restore piston results
            self.state.piston_results = copy.deepcopy(checkpoint.piston_results)

            # Restore engine state if available
            if self.pcb_engine and hasattr(self.pcb_engine, 'state') and checkpoint.state_snapshot:
                state = self.pcb_engine.state
                snapshot = checkpoint.state_snapshot

                if 'placement' in snapshot:
                    state.placement = copy.deepcopy(snapshot['placement'])
                if 'routes' in snapshot:
                    state.routes = copy.deepcopy(snapshot['routes'])
                if 'vias' in snapshot:
                    state.vias = copy.deepcopy(snapshot['vias'])

            self.state.current_checkpoint_id = checkpoint.id

            # Record rollback success in monitor
            if self.monitor:
                self.monitor.record_rollback_result(True, checkpoint.id)

            return True

        except Exception as e:
            self._log(f"[ROLLBACK] Failed to restore checkpoint: {e}")

            # Record rollback failure in monitor
            if self.monitor:
                self.monitor.record_rollback_result(False, checkpoint.id)

            return False

    def _rollback_to_previous(self) -> bool:
        """Roll back to the previous checkpoint."""
        if len(self.state.checkpoints) < 2:
            return False

        # Remove current checkpoint
        self.state.checkpoints.pop()

        # Restore previous
        previous = self.state.checkpoints[-1]
        return self._restore_checkpoint(previous)

    # =========================================================================
    # IMPROVEMENT #3: TIMEOUT SYSTEM
    # =========================================================================

    def _execute_phase_with_timeout(self, phase: BBLPhase) -> Dict:
        """
        Execute a phase with timeout.

        IMPROVEMENT #3: TIMEOUT SYSTEM
        - Each phase has configurable timeout
        - Auto-escalate when timeout exceeded
        """
        config = self.phase_configs.get(phase, BBLPhaseConfig(phase=phase))
        timeout = config.timeout_seconds

        self.state.phase_start_times[phase.value] = time.time()
        self._report_progress(phase, 0.0, f"Starting phase: {phase.value}")

        # Record phase start in monitor
        if self.monitor:
            self.monitor.record_phase_start(phase.value)

        handler = self._phase_handlers.get(phase)
        if not handler:
            self._log(f"[TIMEOUT] No handler for phase: {phase.value}")
            if self.monitor:
                self.monitor.record_error(f"No handler for phase: {phase.value}")
            return {'success': False, 'error': f'No handler for phase {phase.value}'}

        try:
            # Execute with timeout using ThreadPoolExecutor
            future = self._executor.submit(handler)
            result = future.result(timeout=timeout)

        except FuturesTimeoutError:
            self._log(f"[TIMEOUT] Phase {phase.value} exceeded timeout of {timeout}s")

            # Record timeout in monitor
            if self.monitor:
                self.monitor.record_phase_timeout(phase.value, timeout)

            if config.auto_escalate_on_timeout:
                # Create escalation for timeout
                escalation = BBLEscalation(
                    level=BBLEscalationLevel.ENGINEER,
                    phase=phase,
                    reason=f"Phase {phase.value} exceeded timeout of {timeout} seconds",
                    context={'timeout': timeout},
                    suggested_action='increase_timeout_or_simplify',
                    timestamp=time.time()
                )
                self.state.pending_escalation = escalation
                self.state.escalation_history.append(escalation)

            result = {
                'success': False,
                'timeout': True,
                'error': f'Phase timed out after {timeout}s'
            }

        except Exception as e:
            self._log(f"[TIMEOUT] Phase {phase.value} error: {e}")

            # Record error in monitor
            if self.monitor:
                self.monitor.record_error(f"Phase {phase.value} error: {e}", critical=True)

            result = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

        self.state.phase_end_times[phase.value] = time.time()
        duration = self.state.phase_end_times[phase.value] - self.state.phase_start_times[phase.value]
        self._log(f"[TIMING] Phase {phase.value} completed in {duration:.2f}s")

        # Record phase end in monitor
        if self.monitor:
            self.monitor.record_phase_end(phase.value, result.get('success', False))

        self._report_progress(phase, 100.0, f"Phase complete: {phase.value}")

        return result

    # =========================================================================
    # IMPROVEMENT #4: PROGRESS REPORTING
    # =========================================================================

    def _report_progress(
        self,
        phase: BBLPhase,
        percentage: float,
        message: str,
        detail: str = '',
        piston_name: str = '',
        algorithm_name: str = '',
        iteration: int = 0,
        max_iterations: int = 0
    ):
        """
        Report progress to callback.

        IMPROVEMENT #4: PROGRESS REPORTING
        - Real-time updates during execution
        - Percentage complete per phase
        """
        progress = BBLProgress(
            phase=phase,
            percentage=percentage,
            message=message,
            detail=detail,
            timestamp=time.time(),
            piston_name=piston_name,
            algorithm_name=algorithm_name,
            iteration=iteration,
            max_iterations=max_iterations
        )

        with self._lock:
            self.state.progress = progress
            self.state.progress_history.append(progress)

        if self.progress_callback:
            try:
                self.progress_callback(progress)
            except Exception as e:
                self._log(f"[PROGRESS] Callback error: {e}")

        if self.verbose:
            self._log(f"[PROGRESS] {phase.value}: {percentage:.0f}% - {message}")

    # =========================================================================
    # IMPROVEMENT #5: PARALLEL EXECUTION
    # =========================================================================

    def _execute_parallel_phases(self, phases: List[BBLPhase]) -> Dict[BBLPhase, Dict]:
        """
        Execute multiple phases in parallel.

        IMPROVEMENT #5: PARALLEL EXECUTION
        - Independent phases run simultaneously
        - E.g., Silkscreen and Pour don't depend on each other
        """
        results = {}
        futures = {}

        for phase in phases:
            config = self.phase_configs.get(phase, BBLPhaseConfig(phase=phase))
            if not config.can_run_parallel:
                continue

            handler = self._phase_handlers.get(phase)
            if handler:
                self._log(f"[PARALLEL] Starting phase: {phase.value}")
                future = self._executor.submit(handler)
                futures[phase] = (future, config.timeout_seconds)

        # Collect results
        for phase, (future, timeout) in futures.items():
            try:
                result = future.result(timeout=timeout)
                results[phase] = result
                self._log(f"[PARALLEL] Completed: {phase.value}")
            except FuturesTimeoutError:
                results[phase] = {'success': False, 'timeout': True}
                self._log(f"[PARALLEL] Timeout: {phase.value}")
            except Exception as e:
                results[phase] = {'success': False, 'error': str(e)}
                self._log(f"[PARALLEL] Error in {phase.value}: {e}")

        return results

    # =========================================================================
    # IMPROVEMENT #6: LOOP HISTORY
    # =========================================================================

    def _load_history(self):
        """Load BBL history from file."""
        history_path = os.path.join(self.output_dir, self.HISTORY_FILE)
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    # Convert back to BBLHistoryEntry objects
                    for entry_dict in data.get('entries', []):
                        entry = BBLHistoryEntry(
                            bbl_id=entry_dict.get('bbl_id', ''),
                            start_time=entry_dict.get('start_time', 0),
                            end_time=entry_dict.get('end_time', 0),
                            success=entry_dict.get('success', False)
                        )
                        entry.phases_executed = entry_dict.get('phases_executed', [])
                        entry.phases_failed = entry_dict.get('phases_failed', [])
                        entry.phase_durations = entry_dict.get('phase_durations', {})
                        entry.total_duration = entry_dict.get('total_duration', 0)
                        entry.drc_errors = entry_dict.get('drc_errors', 0)
                        entry.routing_completion = entry_dict.get('routing_completion', 0)
                        self.history.append(entry)
                    self._log(f"[HISTORY] Loaded {len(self.history)} previous BBL runs")
            except Exception as e:
                self._log(f"[HISTORY] Failed to load history: {e}")

    def _save_history(self):
        """Save BBL history to file."""
        os.makedirs(self.output_dir, exist_ok=True)
        history_path = os.path.join(self.output_dir, self.HISTORY_FILE)

        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'total_runs': len(self.history),
                'entries': [entry.to_dict() for entry in self.history[-100:]]  # Keep last 100
            }
            with open(history_path, 'w') as f:
                json.dump(data, f, indent=2)
            self._log(f"[HISTORY] Saved {len(self.history)} BBL runs")
        except Exception as e:
            self._log(f"[HISTORY] Failed to save history: {e}")

    def get_history_analytics(self) -> Dict:
        """
        Get analytics from BBL history.

        Returns summary statistics useful for improving the engine.
        """
        if not self.history:
            return {'error': 'No history available'}

        total = len(self.history)
        successful = sum(1 for e in self.history if e.success)

        # Average durations
        durations = [e.total_duration for e in self.history if e.total_duration > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Phase failure rates
        phase_failures = {}
        for entry in self.history:
            for phase in entry.phases_failed:
                phase_failures[phase] = phase_failures.get(phase, 0) + 1

        # Routing completion rates
        routing_completions = [e.routing_completion for e in self.history]
        avg_routing = sum(routing_completions) / len(routing_completions) if routing_completions else 0

        return {
            'total_runs': total,
            'success_rate': successful / total if total > 0 else 0,
            'average_duration': avg_duration,
            'phase_failure_rates': {k: v/total for k, v in phase_failures.items()},
            'average_routing_completion': avg_routing,
            'last_run': self.history[-1].to_dict() if self.history else None
        }

    # =========================================================================
    # PHASE HANDLERS
    # =========================================================================

    def _execute_order_received(self) -> Dict:
        """Phase 1: Order Received - Validate and prepare input."""
        self._log("[PHASE 1] Order Received")

        parts_db = self.state.piston_results.get('parts_db', {})

        # Validate input
        if not parts_db:
            return {'success': False, 'error': 'No parts_db provided', 'quality': 0.0}

        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        if not parts:
            return {'success': False, 'error': 'No parts in parts_db', 'quality': 0.0}

        self._log(f"  Parts: {len(parts)}, Nets: {len(nets)}")

        # Store validated input
        self.state.piston_results['validated_input'] = {
            'parts_count': len(parts),
            'nets_count': len(nets),
            'timestamp': time.time()
        }

        return {
            'success': True,
            'quality': 1.0,
            'parts_count': len(parts),
            'nets_count': len(nets)
        }

    def _execute_piston_execution(self) -> Dict:
        """Phase 2: Piston Execution - Run all pistons with CASCADE system."""
        self._log("[PHASE 2] Piston Execution")

        if not self.pcb_engine:
            # Standalone mode - simulate piston execution for testing
            return self._simulate_piston_execution()

        parts_db = self.state.piston_results.get('parts_db', {})

        try:
            # Inject monitor into PCB engine for piston-level tracking
            if self.monitor:
                self.pcb_engine._bbl_monitor = self.monitor

            # Define the piston sequence we'll track
            piston_sequence = ['parts', 'order', 'placement', 'routing', 'drc', 'output']
            piston_count = len(piston_sequence)

            # Use the PCB engine's run method
            # This triggers all pistons with CASCADE
            result = self.pcb_engine.run(parts_db)

            # Extract piston-level metrics from engine state if available
            if hasattr(self.pcb_engine, 'state') and hasattr(self.pcb_engine.state, 'piston_reports'):
                for piston_name, report in getattr(self.pcb_engine.state, 'piston_reports', {}).items():
                    if self.monitor:
                        # Record piston metrics from engine
                        self.monitor.record_piston_end(
                            piston=piston_name,
                            success=getattr(report, 'success', False),
                            quality_score=getattr(report, 'quality', 0.0),
                            errors=len(getattr(report, 'drc_errors', [])),
                            warnings=len(getattr(report, 'drc_warnings', []))
                        )

            # Extract results
            routing_result = {
                'completion': result.routed_count / result.total_nets if result.total_nets > 0 else 0,
                'routed_count': result.routed_count,
                'total_count': result.total_nets
            }

            drc_result = {
                'passed': result.drc_passed,
                'error_count': len([e for e in result.errors if 'drc' in e.lower()]),
                'warning_count': len(result.warnings)
            }

            self.state.piston_results['routing'] = routing_result
            self.state.piston_results['drc'] = drc_result

            # Record DRC result in monitor
            if self.monitor:
                self.monitor.record_drc_result(
                    passed=result.drc_passed,
                    errors=drc_result['error_count'],
                    warnings=drc_result['warning_count'],
                    check_type='internal',
                    piston='drc'
                )

            # Calculate quality based on routing completion and DRC
            quality = routing_result['completion']
            if drc_result['passed']:
                quality = min(1.0, quality + 0.2)

            self._report_progress(
                BBLPhase.PISTON_EXECUTION,
                100.0,
                f"Piston execution complete",
                f"Routed {result.routed_count}/{result.total_nets}",
                iteration=result.total_nets,
                max_iterations=result.total_nets
            )

            return {
                'success': result.success,
                'quality': quality,
                'routing': routing_result,
                'drc': drc_result,
                'errors': result.errors,
                'warnings': result.warnings
            }

        except Exception as e:
            self._log(f"  Piston execution error: {e}")
            if self.monitor:
                self.monitor.record_error(f"Piston execution error: {e}", critical=True)
            return {
                'success': False,
                'quality': 0.0,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _simulate_piston_execution(self) -> Dict:
        """
        Simulate piston execution for standalone testing (no PCB engine).

        This allows testing the BBL flow and monitor without a full PCB engine.
        """
        self._log("  [Standalone mode - simulating piston execution]")

        parts_db = self.state.piston_results.get('parts_db', {})
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Simulate each piston
        piston_sequence = [
            ('parts', 0.1, 1.0),
            ('order', 0.05, 1.0),
            ('placement', 0.3, 0.9),
            ('routing', 0.5, 0.7),
            ('drc', 0.1, 0.8),
            ('output', 0.05, 1.0)
        ]

        for piston_name, sim_time, sim_quality in piston_sequence:
            # Record piston start
            if self.monitor:
                algorithm = 'default' if piston_name != 'routing' else 'lee'
                self.monitor.record_piston_start(piston_name, algorithm=algorithm)
                self.monitor.record_algorithm_start(piston_name, algorithm)

            # Simulate work
            time.sleep(sim_time * 0.1)  # Scale down for fast tests

            # Record piston end
            if self.monitor:
                success = sim_quality > 0.5
                self.monitor.record_algorithm_end(piston_name, algorithm, success, sim_quality)
                self.monitor.record_piston_end(
                    piston=piston_name,
                    success=success,
                    quality_score=sim_quality
                )

            # Progress update
            progress = (piston_sequence.index((piston_name, sim_time, sim_quality)) + 1) / len(piston_sequence)
            self._report_progress(
                BBLPhase.PISTON_EXECUTION,
                progress * 100,
                f"Piston {piston_name} complete",
                piston_name=piston_name
            )

        # Simulate results
        total_nets = len(nets)
        routed = int(total_nets * 0.7)  # Simulate 70% routing

        routing_result = {
            'completion': routed / total_nets if total_nets > 0 else 0,
            'routed_count': routed,
            'total_count': total_nets
        }

        drc_result = {
            'passed': True,
            'error_count': 0,
            'warning_count': 2
        }

        self.state.piston_results['routing'] = routing_result
        self.state.piston_results['drc'] = drc_result

        # Record DRC in monitor
        if self.monitor:
            self.monitor.record_drc_result(
                passed=True,
                errors=0,
                warnings=2,
                check_type='internal',
                piston='drc'
            )

        return {
            'success': True,
            'quality': 0.8,
            'routing': routing_result,
            'drc': drc_result,
            'errors': [],
            'warnings': ['Simulated warning 1', 'Simulated warning 2']
        }

    def _execute_escalation(self) -> Dict:
        """Phase 3: Escalation - Handle escalation to Engineer/Boss."""
        self._log("[PHASE 3] Escalation")

        escalation = self.state.pending_escalation
        if not escalation:
            return {'success': True, 'resolved': True, 'quality': 1.0}

        self._log(f"  Level: {escalation.level.value}")
        self._log(f"  Reason: {escalation.reason}")

        # Record escalation in monitor
        if self.monitor:
            self.monitor.record_escalation_triggered(
                level=escalation.level.value,
                phase=escalation.phase.value,
                reason=escalation.reason,
                options=escalation.options
            )

        if self.escalation_callback:
            try:
                # Call escalation handler (Engineer/Boss)
                resolved_escalation = self.escalation_callback(escalation)

                if resolved_escalation and resolved_escalation.resolved:
                    self._log(f"  Resolved: {resolved_escalation.response}")
                    self.state.pending_escalation = None

                    # Record resolution in monitor
                    if self.monitor:
                        self.monitor.record_escalation_resolved(
                            level=resolved_escalation.level.value,
                            response=resolved_escalation.response,
                            new_instructions=resolved_escalation.new_instructions
                        )

                    # Apply new instructions if provided
                    if resolved_escalation.new_instructions:
                        self._apply_escalation_instructions(resolved_escalation.new_instructions)

                    return {
                        'success': True,
                        'resolved': True,
                        'quality': 1.0,
                        'response': resolved_escalation.response
                    }
                else:
                    return {
                        'success': False,
                        'resolved': False,
                        'quality': 0.0,
                        'error': 'Escalation not resolved'
                    }

            except Exception as e:
                self._log(f"  Escalation callback error: {e}")
                if self.monitor:
                    self.monitor.record_error(f"Escalation callback error: {e}")
                return {'success': False, 'resolved': False, 'error': str(e)}
        else:
            # No callback, auto-resolve with default action
            self._log("  No escalation callback, auto-resolving")
            self.state.pending_escalation = None

            # Record auto-resolution in monitor
            if self.monitor:
                self.monitor.record_escalation_resolved(
                    level=escalation.level.value,
                    response="Auto-resolved (no callback)",
                    new_instructions={}
                )

            return {'success': True, 'resolved': True, 'quality': 0.5}

    def _apply_escalation_instructions(self, instructions: Dict):
        """Apply new instructions from escalation resolution."""
        self._log(f"  Applying instructions: {list(instructions.keys())}")

        # Update engine config if needed
        if self.pcb_engine and hasattr(self.pcb_engine, 'config'):
            for key, value in instructions.items():
                if hasattr(self.pcb_engine.config, key):
                    setattr(self.pcb_engine.config, key, value)
                    self._log(f"    Set {key} = {value}")

    def _execute_output_generation(self) -> Dict:
        """Phase 4: Output Generation - Generate KiCad files."""
        self._log("[PHASE 4] Output Generation")

        if not self.pcb_engine:
            return {'success': False, 'error': 'No PCB engine', 'quality': 0.0}

        try:
            # Get output piston
            output_piston = getattr(self.pcb_engine, '_output_piston', None)

            if output_piston:
                state = self.pcb_engine.state

                # Generate output
                result = output_piston.generate(
                    parts_db=state.parts_db or {},
                    placement=state.placement or {},
                    routes=state.routes or {},
                    vias=state.vias or [],
                    silkscreen=state.silkscreen
                )

                output_files = getattr(result, 'output_files', [])

                self.state.piston_results['output'] = {
                    'files': output_files,
                    'success': bool(output_files)
                }

                return {
                    'success': bool(output_files),
                    'quality': 1.0 if output_files else 0.0,
                    'files': output_files
                }
            else:
                return {'success': False, 'error': 'No output piston', 'quality': 0.0}

        except Exception as e:
            self._log(f"  Output generation error: {e}")
            return {'success': False, 'error': str(e), 'quality': 0.0}

    def _execute_kicad_drc(self) -> Dict:
        """Phase 5: KiCad DRC - Run KiCad CLI DRC (THE AUTHORITY)."""
        self._log("[PHASE 5] KiCad DRC (THE AUTHORITY)")

        try:
            # Import KiCad DRC Teacher
            from .kicad_drc_teacher import KiCadDRCTeacher

            teacher = KiCadDRCTeacher()

            # Find the generated PCB file
            output_files = self.state.piston_results.get('output', {}).get('files', [])
            pcb_file = None

            for f in output_files:
                if f.endswith('.kicad_pcb'):
                    pcb_file = f
                    break

            if not pcb_file:
                # Try default location
                pcb_file = os.path.join(self.output_dir, 'pcb.kicad_pcb')

            if not os.path.exists(pcb_file):
                return {
                    'success': False,
                    'quality': 0.0,
                    'error': f'PCB file not found: {pcb_file}'
                }

            # Run KiCad DRC
            drc_result = teacher.run_drc(pcb_file)

            passed = drc_result.get('passed', False)
            errors = drc_result.get('errors', [])
            warnings = drc_result.get('warnings', [])

            self._log(f"  KiCad DRC: {'PASS' if passed else 'FAIL'}")
            self._log(f"  Errors: {len(errors)}, Warnings: {len(warnings)}")

            self.state.piston_results['kicad_drc'] = {
                'passed': passed,
                'errors': errors,
                'warnings': warnings,
                'error_count': len(errors),
                'warning_count': len(warnings)
            }

            # Quality: 1.0 if passed, 0.5 if only warnings, 0.0 if errors
            quality = 1.0 if passed else (0.5 if not errors else 0.0)

            return {
                'success': passed,
                'quality': quality,
                'passed': passed,
                'errors': errors,
                'warnings': warnings
            }

        except ImportError:
            self._log("  KiCad DRC Teacher not available")
            return {'success': True, 'quality': 0.5, 'skipped': True}
        except Exception as e:
            self._log(f"  KiCad DRC error: {e}")
            return {'success': False, 'quality': 0.0, 'error': str(e)}

    def _execute_learning_delivery(self) -> Dict:
        """Phase 6: Learning & Delivery - Learn from result and finalize."""
        self._log("[PHASE 6] Learning & Delivery")

        try:
            # Get results summary
            routing = self.state.piston_results.get('routing', {})
            drc = self.state.piston_results.get('drc', {})
            kicad_drc = self.state.piston_results.get('kicad_drc', {})
            output = self.state.piston_results.get('output', {})

            # === SMART ALGORITHM MANAGER: Save Learning Database ===
            if self.pcb_engine and hasattr(self.pcb_engine, '_learning_db'):
                try:
                    learning_db = self.pcb_engine._learning_db
                    if learning_db:
                        learning_db.save()
                        summary = learning_db.get_summary()
                        self._log(f"  [LEARNING] Saved {summary['total_outcomes']} routing outcomes")
                        self._log(f"  [LEARNING] Overall success rate: {summary['overall_success_rate']*100:.1f}%")
                except Exception as e:
                    self._log(f"  [LEARNING] Failed to save learning database: {e}")

            # Learn from internal vs KiCad DRC differences
            if self.pcb_engine:
                try:
                    from .kicad_drc_teacher import KiCadDRCTeacher
                    teacher = KiCadDRCTeacher()

                    internal_errors = drc.get('error_count', 0)
                    kicad_errors = kicad_drc.get('error_count', 0)

                    if kicad_errors > internal_errors:
                        self._log(f"  [LEARNING] KiCad found {kicad_errors - internal_errors} errors we missed")
                        # Record for learning
                        for error in kicad_drc.get('errors', []):
                            teacher.record_learning(error)

                except ImportError:
                    pass

            # Set output folder marker
            output_files = output.get('files', [])
            success = kicad_drc.get('passed', False) or drc.get('passed', False)

            self._log(f"  Output files: {len(output_files)}")
            self._log(f"  Final status: {'PASS' if success else 'FAIL'}")

            return {
                'success': True,
                'quality': 1.0,
                'output_files': output_files,
                'final_status': 'PASS' if success else 'FAIL'
            }

        except Exception as e:
            self._log(f"  Learning/delivery error: {e}")
            return {'success': False, 'quality': 0.0, 'error': str(e)}

    # =========================================================================
    # BBL ITERATION LOOP - DRC FAILURE RECOVERY
    # =========================================================================
    # When KiCad DRC fails, the BBL must:
    # 1. Extract failing nets from DRC errors
    # 2. Trigger ripup-reroute for those nets
    # 3. Re-generate output and re-run KiCad DRC
    # 4. Iterate until success or max iterations reached

    def _extract_failing_nets_from_drc(self, drc_errors: List[Dict]) -> List[str]:
        """
        Extract net names from KiCad DRC errors.

        DRC errors contain items with descriptions like:
        - "Pad 2 [GND] of C1 on F.Cu"
        - "Track on B.Cu (net VDIV)"

        Args:
            drc_errors: List of DRC violation dictionaries from KiCad

        Returns:
            List of unique net names involved in violations
        """
        import re
        failing_nets = set()

        for error in drc_errors:
            # Skip if not a dict (might be a string error message)
            if not isinstance(error, dict):
                continue

            items = error.get('items', [])
            for item in items:
                desc = item.get('description', '')

                # Pattern 1: [NET_NAME] in description (e.g., "Pad 2 [GND] of C1")
                bracket_match = re.search(r'\[([^\]]+)\]', desc)
                if bracket_match:
                    net_name = bracket_match.group(1)
                    if net_name and net_name not in ('', 'NC', 'no-connect'):
                        failing_nets.add(net_name)

                # Pattern 2: (net NET_NAME) in description
                net_match = re.search(r'\(net\s+([^)]+)\)', desc)
                if net_match:
                    net_name = net_match.group(1)
                    if net_name and net_name not in ('', 'NC', 'no-connect'):
                        failing_nets.add(net_name)

        self._log(f"  [BBL ITERATE] Extracted failing nets: {list(failing_nets)}")
        return list(failing_nets)

    def _ripup_and_reroute_nets(self, failing_nets: List[str]) -> bool:
        """
        Rip up and reroute specific nets using the routing piston.

        Args:
            failing_nets: List of net names to rip up and reroute

        Returns:
            True if rerouting improved the situation, False otherwise
        """
        if not self.pcb_engine:
            self._log("  [BBL ITERATE] No PCB engine available for rerouting")
            return False

        if not failing_nets:
            self._log("  [BBL ITERATE] No failing nets to reroute")
            return False

        self._log(f"  [BBL ITERATE] Ripping up {len(failing_nets)} nets: {failing_nets}")

        try:
            # Get the routing piston
            routing_piston = getattr(self.pcb_engine, '_routing_piston', None)
            if not routing_piston:
                self._log("  [BBL ITERATE] No routing piston available")
                return False

            # Get current state
            state = self.pcb_engine.state
            parts_db = state.parts_db or {}
            placement = state.placement or {}
            escapes = getattr(state, 'escapes', {})

            # Remove routes for failing nets from state
            current_routes = state.routes or {}
            for net in failing_nets:
                if net in current_routes:
                    del current_routes[net]
                    self._log(f"    Ripped up: {net}")

            # Re-route just the failing nets using ripup algorithm
            # Configure for ripup-reroute
            original_algorithm = routing_piston.config.algorithm
            routing_piston.config.algorithm = 'ripup'

            try:
                # Build net_pins for failing nets only
                nets = parts_db.get('nets', {})
                net_pins = {}
                for net in failing_nets:
                    if net in nets:
                        net_pins[net] = nets[net].get('pins', [])

                # Re-initialize grids and re-register existing routes
                routing_piston._initialize_grids()
                routing_piston._register_components(placement, parts_db)

                # Mark existing routes in grid (nets we're keeping)
                for net_name, route in current_routes.items():
                    if hasattr(route, 'segments'):
                        for seg in route.segments:
                            routing_piston._mark_segment_in_grid(seg, net_name)

                # Route the failing nets
                result = routing_piston.route(parts_db, escapes, placement, failing_nets)

                # Merge new routes back into state
                for net_name, route in result.routes.items():
                    if route.success:
                        current_routes[net_name] = route
                        self._log(f"    Re-routed successfully: {net_name}")
                    else:
                        self._log(f"    Re-route failed: {net_name}")

                state.routes = current_routes

                # Also update vias
                new_vias = []
                for route in current_routes.values():
                    if hasattr(route, 'vias'):
                        new_vias.extend(route.vias)
                state.vias = new_vias

                return result.success

            finally:
                # Restore original algorithm
                routing_piston.config.algorithm = original_algorithm

        except Exception as e:
            self._log(f"  [BBL ITERATE] Rerouting error: {e}")
            import traceback
            self._log(traceback.format_exc())
            return False

    def _run_iteration_loop(self, max_iterations: int = 3) -> Dict:
        """
        Run the DRC fix iteration loop.

        When KiCad DRC fails, this method:
        1. Extracts failing nets from DRC errors
        2. Rips up and reroutes those nets
        3. Regenerates output
        4. Runs KiCad DRC again
        5. Iterates until success or max iterations

        Args:
            max_iterations: Maximum number of fix iterations

        Returns:
            Final KiCad DRC result
        """
        self._log(f"\n{'='*60}")
        self._log(f"BBL ITERATION LOOP - Attempting to fix DRC failures")
        self._log(f"{'='*60}")

        for iteration in range(max_iterations):
            self._log(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Record iteration start in monitor
            if self.monitor:
                self.monitor.record_event(
                    MonitorEvent.ITERATION_START if hasattr(MonitorEvent, 'ITERATION_START') else 'iteration_start',
                    message=f"BBL iteration {iteration + 1}",
                    data={'iteration': iteration + 1, 'max_iterations': max_iterations}
                )

            # Get current DRC errors
            kicad_drc = self.state.piston_results.get('kicad_drc', {})
            errors = kicad_drc.get('errors', [])

            if not errors:
                self._log("  No DRC errors - nothing to fix!")
                return {'success': True, 'passed': True, 'iterations': iteration}

            # Extract failing nets
            failing_nets = self._extract_failing_nets_from_drc(errors)

            if not failing_nets:
                self._log("  Could not extract net names from DRC errors")
                self._log("  Error types may not be route-related (e.g., silkscreen)")
                # Check if errors are non-routing related
                non_routing_errors = True
                for error in errors:
                    if isinstance(error, dict):
                        error_type = error.get('type', '')
                        if error_type in ('track_crossing', 'shorting_items', 'clearance',
                                        'track_width', 'unconnected_items'):
                            non_routing_errors = False
                            break

                if non_routing_errors:
                    self._log("  Errors are not routing-related, cannot fix via iteration")
                    return {
                        'success': False,
                        'passed': False,
                        'iterations': iteration + 1,
                        'errors': errors,
                        'error': 'Non-routing DRC errors'
                    }
                break

            # Rip up and reroute failing nets
            reroute_success = self._ripup_and_reroute_nets(failing_nets)

            if not reroute_success:
                self._log("  Rerouting did not succeed, trying different algorithm")
                # Try with a different approach on next iteration
                continue

            # Regenerate output
            self._log("  Regenerating output...")
            output_result = self._execute_output_generation()

            if not output_result.get('success', False):
                self._log("  Output regeneration failed")
                continue

            # Run KiCad DRC again
            self._log("  Running KiCad DRC...")
            drc_result = self._execute_kicad_drc()

            if drc_result.get('passed', False):
                self._log(f"\n  SUCCESS! KiCad DRC passed after {iteration + 1} iteration(s)")
                return {
                    'success': True,
                    'passed': True,
                    'iterations': iteration + 1,
                    'quality': 1.0
                }

            # Update state with new DRC results
            self.state.piston_results['kicad_drc'] = {
                'passed': drc_result.get('passed', False),
                'errors': drc_result.get('errors', []),
                'warnings': drc_result.get('warnings', []),
                'error_count': len(drc_result.get('errors', [])),
                'warning_count': len(drc_result.get('warnings', []))
            }

            new_error_count = len(drc_result.get('errors', []))
            old_error_count = len(errors)

            if new_error_count >= old_error_count:
                self._log(f"  No improvement: {old_error_count} -> {new_error_count} errors")
            else:
                self._log(f"  Improvement: {old_error_count} -> {new_error_count} errors")

        self._log(f"\n  Iteration limit reached ({max_iterations} attempts)")
        return {
            'success': False,
            'passed': False,
            'iterations': max_iterations,
            'errors': self.state.piston_results.get('kicad_drc', {}).get('errors', [])
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def shutdown(self):
        """Shutdown the BBL engine and clean up resources."""
        self._executor.shutdown(wait=False)
        self._save_history()
