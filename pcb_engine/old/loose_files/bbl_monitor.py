"""
BBL MONITOR SENSOR
===================

A comprehensive monitoring system that watches the Big Beautiful Loop (BBL)
and documents everything happening inside.

CAPABILITIES:
=============
1. EVENT CAPTURE - Records every event in the BBL lifecycle
2. TIMING METRICS - Tracks duration of every operation
3. DECISION LOGGING - Documents all checkpoint decisions
4. ESCALATION TRACKING - Records all escalation events
5. ROLLBACK HISTORY - Tracks all rollback operations
6. PISTON MONITORING - Watches each piston's execution
7. DRC TRACKING - Records all DRC checks and results
8. REPORT GENERATION - Creates JSON, Markdown, and HTML reports

USAGE:
======
    from bbl_monitor import BBLMonitor, MonitorEvent

    # Create monitor
    monitor = BBLMonitor(output_dir='./output', verbose=True)

    # Start monitoring a BBL run
    monitor.start_session(bbl_id='BBL_123')

    # Record events (called by BBL Engine)
    monitor.record_event(MonitorEvent.PHASE_START, phase='piston_execution')
    monitor.record_piston_start('routing', algorithm='lee')
    monitor.record_piston_result('routing', success=True, duration=1.5)

    # End session and generate reports
    monitor.end_session(success=True)
    monitor.generate_report(format='markdown')

Author: PCB Engine Team
Date: 2026-02-07
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import time
import json
import os
import threading
from collections import defaultdict
import statistics


# =============================================================================
# MONITOR ENUMS
# =============================================================================

class MonitorEvent(Enum):
    """Types of events that can be recorded"""
    # Session events
    SESSION_START = 'session_start'
    SESSION_END = 'session_end'

    # Phase events
    PHASE_START = 'phase_start'
    PHASE_END = 'phase_end'
    PHASE_TIMEOUT = 'phase_timeout'
    PHASE_ERROR = 'phase_error'

    # Checkpoint events
    CHECKPOINT_CREATED = 'checkpoint_created'
    CHECKPOINT_DECISION = 'checkpoint_decision'

    # Rollback events
    ROLLBACK_START = 'rollback_start'
    ROLLBACK_SUCCESS = 'rollback_success'
    ROLLBACK_FAILED = 'rollback_failed'

    # Escalation events
    ESCALATION_TRIGGERED = 'escalation_triggered'
    ESCALATION_RESOLVED = 'escalation_resolved'
    ESCALATION_TIMEOUT = 'escalation_timeout'

    # Piston events
    PISTON_START = 'piston_start'
    PISTON_END = 'piston_end'
    PISTON_RETRY = 'piston_retry'
    PISTON_CASCADE = 'piston_cascade'
    PISTON_SUCCESS = 'piston_success'
    PISTON_FAILED = 'piston_failed'

    # Algorithm events
    ALGORITHM_START = 'algorithm_start'
    ALGORITHM_END = 'algorithm_end'
    ALGORITHM_SWITCH = 'algorithm_switch'

    # DRC events
    DRC_CHECK_START = 'drc_check_start'
    DRC_CHECK_END = 'drc_check_end'
    DRC_VIOLATION = 'drc_violation'
    DRC_PASSED = 'drc_passed'
    DRC_FAILED = 'drc_failed'

    # KiCad DRC events
    KICAD_DRC_START = 'kicad_drc_start'
    KICAD_DRC_END = 'kicad_drc_end'
    KICAD_DRC_VIOLATION = 'kicad_drc_violation'

    # Output events
    OUTPUT_START = 'output_start'
    OUTPUT_END = 'output_end'
    FILE_GENERATED = 'file_generated'

    # Learning events
    LEARNING_RECORD = 'learning_record'
    LEARNING_APPLIED = 'learning_applied'

    # Progress events
    PROGRESS_UPDATE = 'progress_update'

    # Warning and error events
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL_ERROR = 'critical_error'


class MonitorLevel(Enum):
    """Logging levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EventRecord:
    """A single recorded event"""
    timestamp: float
    event_type: MonitorEvent
    level: MonitorLevel = MonitorLevel.INFO
    phase: str = ''
    piston: str = ''
    algorithm: str = ''
    message: str = ''
    data: Dict = field(default_factory=dict)
    duration: float = 0.0

    def to_dict(self) -> Dict:
        # Handle event_type being either an enum or a string
        if hasattr(self.event_type, 'value'):
            event_type_str = self.event_type.value
        else:
            event_type_str = str(self.event_type)

        # Handle level being either an enum or a string
        if hasattr(self.level, 'value'):
            level_str = self.level.value
        else:
            level_str = str(self.level)

        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': event_type_str,
            'level': level_str,
            'phase': self.phase,
            'piston': self.piston,
            'algorithm': self.algorithm,
            'message': self.message,
            'data': self.data,
            'duration': self.duration
        }


@dataclass
class PistonMetrics:
    """Metrics for a single piston"""
    name: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    cascades: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)
    algorithms_used: Dict[str, int] = field(default_factory=dict)
    algorithm_success: Dict[str, int] = field(default_factory=dict)
    drc_checks: int = 0
    drc_passed: int = 0
    drc_failed: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.executions if self.executions > 0 else 0.0

    @property
    def std_time(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'executions': self.executions,
            'successes': self.successes,
            'failures': self.failures,
            'retries': self.retries,
            'cascades': self.cascades,
            'success_rate': f"{self.success_rate*100:.1f}%",
            'total_time': f"{self.total_time:.2f}s",
            'avg_time': f"{self.avg_time:.2f}s",
            'min_time': f"{self.min_time:.2f}s" if self.min_time != float('inf') else 'N/A',
            'max_time': f"{self.max_time:.2f}s",
            'std_time': f"{self.std_time:.2f}s",
            'algorithms_used': self.algorithms_used,
            'algorithm_success': self.algorithm_success,
            'drc_checks': self.drc_checks,
            'drc_pass_rate': f"{self.drc_passed/self.drc_checks*100:.1f}%" if self.drc_checks > 0 else 'N/A'
        }


@dataclass
class PhaseMetrics:
    """Metrics for a BBL phase"""
    name: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_time: float = 0.0
    times: List[float] = field(default_factory=list)
    checkpoint_decisions: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.executions if self.executions > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'executions': self.executions,
            'successes': self.successes,
            'failures': self.failures,
            'timeouts': self.timeouts,
            'success_rate': f"{self.success_rate*100:.1f}%",
            'total_time': f"{self.total_time:.2f}s",
            'avg_time': f"{self.avg_time:.2f}s",
            'checkpoint_decisions': self.checkpoint_decisions
        }


@dataclass
class AlgorithmMetrics:
    """Performance metrics for a specific algorithm"""
    name: str
    piston: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.executions if self.executions > 0 else 0.0

    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0

    @property
    def efficiency_score(self) -> float:
        """Combined score: success rate * quality / time"""
        if self.avg_time == 0:
            return 0.0
        return (self.success_rate * self.avg_quality) / max(self.avg_time, 0.001)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'piston': self.piston,
            'executions': self.executions,
            'successes': self.successes,
            'failures': self.failures,
            'success_rate': f"{self.success_rate*100:.1f}%",
            'total_time': f"{self.total_time:.3f}s",
            'avg_time': f"{self.avg_time:.3f}s",
            'min_time': f"{self.min_time:.3f}s" if self.min_time != float('inf') else 'N/A',
            'max_time': f"{self.max_time:.3f}s",
            'avg_quality': f"{self.avg_quality:.2f}",
            'efficiency_score': f"{self.efficiency_score:.2f}"
        }


@dataclass
class PerformanceRanking:
    """Performance ranking of all BBL members"""
    fastest_piston: str = ''
    slowest_piston: str = ''
    most_reliable_piston: str = ''
    least_reliable_piston: str = ''
    best_algorithm: str = ''
    worst_algorithm: str = ''
    bottleneck_phase: str = ''
    most_efficient_algorithm: str = ''

    # Detailed rankings
    piston_by_speed: List[Tuple[str, float]] = field(default_factory=list)
    piston_by_success: List[Tuple[str, float]] = field(default_factory=list)
    algorithm_by_efficiency: List[Tuple[str, float]] = field(default_factory=list)
    phase_by_time: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'fastest_piston': self.fastest_piston,
            'slowest_piston': self.slowest_piston,
            'most_reliable_piston': self.most_reliable_piston,
            'least_reliable_piston': self.least_reliable_piston,
            'best_algorithm': self.best_algorithm,
            'worst_algorithm': self.worst_algorithm,
            'bottleneck_phase': self.bottleneck_phase,
            'most_efficient_algorithm': self.most_efficient_algorithm,
            'piston_by_speed': self.piston_by_speed,
            'piston_by_success': self.piston_by_success,
            'algorithm_by_efficiency': self.algorithm_by_efficiency,
            'phase_by_time': self.phase_by_time
        }


@dataclass
class SessionMetrics:
    """Metrics for a complete BBL session"""
    bbl_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = False

    # Counts
    total_events: int = 0
    total_phases: int = 0
    total_pistons: int = 0
    total_algorithms: int = 0
    total_checkpoints: int = 0
    total_rollbacks: int = 0
    total_escalations: int = 0
    total_drc_checks: int = 0
    total_drc_violations: int = 0
    total_files_generated: int = 0

    # Phase metrics
    phase_metrics: Dict[str, PhaseMetrics] = field(default_factory=dict)

    # Piston metrics
    piston_metrics: Dict[str, PistonMetrics] = field(default_factory=dict)

    # Algorithm metrics (NEW - detailed per-algorithm performance)
    algorithm_metrics: Dict[str, AlgorithmMetrics] = field(default_factory=dict)

    # Performance ranking (NEW - comparative analysis)
    performance_ranking: PerformanceRanking = None

    # DRC metrics
    internal_drc_errors: int = 0
    internal_drc_warnings: int = 0
    kicad_drc_errors: int = 0
    kicad_drc_warnings: int = 0

    # Routing metrics
    routing_completion: float = 0.0
    routed_nets: int = 0
    total_nets: int = 0

    # Resource usage (NEW)
    peak_memory_mb: float = 0.0
    cpu_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'bbl_id': self.bbl_id,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time > 0 else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time > 0 else None,
            'duration': f"{self.duration:.2f}s",
            'success': self.success,
            'total_events': self.total_events,
            'total_phases': self.total_phases,
            'total_pistons': self.total_pistons,
            'total_algorithms': self.total_algorithms,
            'total_checkpoints': self.total_checkpoints,
            'total_rollbacks': self.total_rollbacks,
            'total_escalations': self.total_escalations,
            'total_drc_checks': self.total_drc_checks,
            'total_drc_violations': self.total_drc_violations,
            'total_files_generated': self.total_files_generated,
            'phase_metrics': {k: v.to_dict() for k, v in self.phase_metrics.items()},
            'piston_metrics': {k: v.to_dict() for k, v in self.piston_metrics.items()},
            'algorithm_metrics': {k: v.to_dict() for k, v in self.algorithm_metrics.items()},
            'performance_ranking': self.performance_ranking.to_dict() if self.performance_ranking else None,
            'internal_drc': {
                'errors': self.internal_drc_errors,
                'warnings': self.internal_drc_warnings
            },
            'kicad_drc': {
                'errors': self.kicad_drc_errors,
                'warnings': self.kicad_drc_warnings
            },
            'routing': {
                'completion': f"{self.routing_completion*100:.1f}%",
                'routed': self.routed_nets,
                'total': self.total_nets
            },
            'resources': {
                'peak_memory_mb': f"{self.peak_memory_mb:.1f}",
                'cpu_time': f"{self.cpu_time:.2f}s"
            }
        }


# =============================================================================
# BBL MONITOR
# =============================================================================

class BBLMonitor:
    """
    BBL Monitor Sensor - Watches and documents everything in the BBL.

    This monitor captures all events, tracks metrics, and generates
    comprehensive reports about BBL execution.
    """

    def __init__(
        self,
        output_dir: str = '',  # Empty uses DEFAULT_OUTPUT_BASE from paths.py
        log_level: MonitorLevel = MonitorLevel.INFO,
        verbose: bool = True,
        realtime_callback: Callable[[EventRecord], None] = None,
        max_events: int = 10000
    ):
        """
        Initialize the BBL Monitor.

        Args:
            output_dir: Directory to save reports (default: D:\\Anas\\tmp\\output)
            log_level: Minimum level to record
            verbose: Print events to console
            realtime_callback: Called for each event in real-time
            max_events: Maximum events to keep in memory
        """
        # Use default output base if not specified
        if not output_dir:
            from .paths import DEFAULT_OUTPUT_BASE
            output_dir = DEFAULT_OUTPUT_BASE
        self.output_dir = output_dir
        self.log_level = log_level
        self.verbose = verbose
        self.realtime_callback = realtime_callback
        self.max_events = max_events

        # Current session
        self.session: SessionMetrics = None
        self.events: List[EventRecord] = []

        # Tracking state
        self._current_phase: str = ''
        self._current_piston: str = ''
        self._current_algorithm: str = ''
        self._phase_start_times: Dict[str, float] = {}
        self._piston_start_times: Dict[str, float] = {}
        self._algorithm_start_times: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Session history
        self.session_history: List[SessionMetrics] = []

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, bbl_id: str):
        """Start monitoring a new BBL session."""
        with self._lock:
            self.session = SessionMetrics(
                bbl_id=bbl_id,
                start_time=time.time()
            )
            self.events = []
            self._current_phase = ''
            self._current_piston = ''
            self._current_algorithm = ''
            self._phase_start_times = {}
            self._piston_start_times = {}
            self._algorithm_start_times = {}

        self.record_event(
            MonitorEvent.SESSION_START,
            message=f"BBL Session started: {bbl_id}",
            level=MonitorLevel.INFO
        )

    def end_session(self, success: bool = False, routing_stats: Dict = None):
        """End the current monitoring session."""
        if not self.session:
            return

        self.session.end_time = time.time()
        self.session.success = success

        if routing_stats:
            self.session.routing_completion = routing_stats.get('completion', 0.0)
            self.session.routed_nets = routing_stats.get('routed', 0)
            self.session.total_nets = routing_stats.get('total', 0)

        self.record_event(
            MonitorEvent.SESSION_END,
            message=f"BBL Session ended: {'SUCCESS' if success else 'FAILED'}",
            level=MonitorLevel.INFO,
            data={
                'success': success,
                'duration': self.session.duration,
                'total_events': self.session.total_events
            }
        )

        # Add to history
        self.session_history.append(self.session)

        # Save session report
        self._save_session_report()

    # =========================================================================
    # EVENT RECORDING
    # =========================================================================

    def record_event(
        self,
        event_type: MonitorEvent,
        level: MonitorLevel = MonitorLevel.INFO,
        phase: str = None,
        piston: str = None,
        algorithm: str = None,
        message: str = '',
        data: Dict = None,
        duration: float = 0.0
    ):
        """
        Record a single event.

        Args:
            event_type: Type of event
            level: Logging level
            phase: Phase name (optional, uses current)
            piston: Piston name (optional, uses current)
            algorithm: Algorithm name (optional, uses current)
            message: Human-readable message
            data: Additional data
            duration: Duration in seconds (if applicable)
        """
        if level.value < self.log_level.value:
            return

        event = EventRecord(
            timestamp=time.time(),
            event_type=event_type,
            level=level,
            phase=phase or self._current_phase,
            piston=piston or self._current_piston,
            algorithm=algorithm or self._current_algorithm,
            message=message,
            data=data or {},
            duration=duration
        )

        with self._lock:
            # Add to events list
            self.events.append(event)

            # Trim if too many events
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

            # Update session metrics
            if self.session:
                self.session.total_events += 1

        # Verbose output
        if self.verbose:
            self._print_event(event)

        # Real-time callback
        if self.realtime_callback:
            try:
                self.realtime_callback(event)
            except Exception as e:
                pass  # Don't let callback errors break monitoring

    # =========================================================================
    # PHASE TRACKING
    # =========================================================================

    def record_phase_start(self, phase: str):
        """Record the start of a BBL phase."""
        self._current_phase = phase
        self._phase_start_times[phase] = time.time()

        if self.session:
            if phase not in self.session.phase_metrics:
                self.session.phase_metrics[phase] = PhaseMetrics(name=phase)
            self.session.phase_metrics[phase].executions += 1
            self.session.total_phases += 1

        self.record_event(
            MonitorEvent.PHASE_START,
            phase=phase,
            message=f"Phase started: {phase}",
            level=MonitorLevel.INFO
        )

    def record_phase_end(self, phase: str, success: bool = True):
        """Record the end of a BBL phase."""
        duration = 0.0
        if phase in self._phase_start_times:
            duration = time.time() - self._phase_start_times[phase]

        if self.session and phase in self.session.phase_metrics:
            metrics = self.session.phase_metrics[phase]
            metrics.total_time += duration
            metrics.times.append(duration)
            if success:
                metrics.successes += 1
            else:
                metrics.failures += 1

        self.record_event(
            MonitorEvent.PHASE_END,
            phase=phase,
            message=f"Phase ended: {phase} ({'SUCCESS' if success else 'FAILED'})",
            level=MonitorLevel.INFO,
            duration=duration,
            data={'success': success}
        )

        self._current_phase = ''

    def record_phase_timeout(self, phase: str, timeout: float):
        """Record a phase timeout."""
        if self.session and phase in self.session.phase_metrics:
            self.session.phase_metrics[phase].timeouts += 1

        self.record_event(
            MonitorEvent.PHASE_TIMEOUT,
            phase=phase,
            message=f"Phase timeout: {phase} exceeded {timeout}s",
            level=MonitorLevel.WARNING,
            data={'timeout': timeout}
        )

    # =========================================================================
    # CHECKPOINT TRACKING
    # =========================================================================

    def record_checkpoint_created(self, checkpoint_id: str, phase: str):
        """Record checkpoint creation."""
        if self.session:
            self.session.total_checkpoints += 1

        self.record_event(
            MonitorEvent.CHECKPOINT_CREATED,
            phase=phase,
            message=f"Checkpoint created: {checkpoint_id}",
            level=MonitorLevel.DEBUG,
            data={'checkpoint_id': checkpoint_id}
        )

    def record_checkpoint_decision(
        self,
        phase: str,
        decision: str,
        quality: float,
        threshold: float
    ):
        """Record a checkpoint decision."""
        if self.session and phase in self.session.phase_metrics:
            decisions = self.session.phase_metrics[phase].checkpoint_decisions
            decisions[decision] = decisions.get(decision, 0) + 1

        self.record_event(
            MonitorEvent.CHECKPOINT_DECISION,
            phase=phase,
            message=f"Checkpoint decision: {decision} (quality={quality:.2f}, threshold={threshold:.2f})",
            level=MonitorLevel.INFO,
            data={
                'decision': decision,
                'quality': quality,
                'threshold': threshold
            }
        )

    # =========================================================================
    # ROLLBACK TRACKING
    # =========================================================================

    def record_rollback_start(self, from_checkpoint: str, to_checkpoint: str):
        """Record rollback start."""
        self.record_event(
            MonitorEvent.ROLLBACK_START,
            message=f"Rollback started: {from_checkpoint} -> {to_checkpoint}",
            level=MonitorLevel.WARNING,
            data={
                'from_checkpoint': from_checkpoint,
                'to_checkpoint': to_checkpoint
            }
        )

    def record_rollback_result(self, success: bool, checkpoint_id: str):
        """Record rollback result."""
        if self.session:
            self.session.total_rollbacks += 1

        event_type = MonitorEvent.ROLLBACK_SUCCESS if success else MonitorEvent.ROLLBACK_FAILED
        self.record_event(
            event_type,
            message=f"Rollback {'succeeded' if success else 'failed'}: {checkpoint_id}",
            level=MonitorLevel.INFO if success else MonitorLevel.ERROR,
            data={'checkpoint_id': checkpoint_id, 'success': success}
        )

    # =========================================================================
    # ESCALATION TRACKING
    # =========================================================================

    def record_escalation_triggered(
        self,
        level: str,
        phase: str,
        reason: str,
        options: List[str] = None
    ):
        """Record escalation triggered."""
        if self.session:
            self.session.total_escalations += 1

        self.record_event(
            MonitorEvent.ESCALATION_TRIGGERED,
            phase=phase,
            message=f"Escalation to {level}: {reason}",
            level=MonitorLevel.WARNING,
            data={
                'escalation_level': level,
                'reason': reason,
                'options': options or []
            }
        )

    def record_escalation_resolved(
        self,
        level: str,
        response: str,
        new_instructions: Dict = None
    ):
        """Record escalation resolved."""
        self.record_event(
            MonitorEvent.ESCALATION_RESOLVED,
            message=f"Escalation resolved ({level}): {response}",
            level=MonitorLevel.INFO,
            data={
                'escalation_level': level,
                'response': response,
                'new_instructions': new_instructions or {}
            }
        )

    # =========================================================================
    # PISTON TRACKING
    # =========================================================================

    def record_piston_start(self, piston: str, algorithm: str = '', attempt: int = 1):
        """Record piston execution start."""
        self._current_piston = piston
        self._current_algorithm = algorithm
        self._piston_start_times[piston] = time.time()

        if self.session:
            if piston not in self.session.piston_metrics:
                self.session.piston_metrics[piston] = PistonMetrics(name=piston)
            self.session.piston_metrics[piston].executions += 1
            self.session.total_pistons += 1

            if algorithm:
                algos = self.session.piston_metrics[piston].algorithms_used
                algos[algorithm] = algos.get(algorithm, 0) + 1
                self.session.total_algorithms += 1

        self.record_event(
            MonitorEvent.PISTON_START,
            piston=piston,
            algorithm=algorithm,
            message=f"Piston started: {piston}" + (f" [{algorithm}]" if algorithm else "") + f" (attempt {attempt})",
            level=MonitorLevel.INFO,
            data={'attempt': attempt}
        )

    def record_piston_end(
        self,
        piston: str,
        success: bool,
        quality_score: float = 0.0,
        errors: int = 0,
        warnings: int = 0
    ):
        """Record piston execution end."""
        duration = 0.0
        if piston in self._piston_start_times:
            duration = time.time() - self._piston_start_times[piston]

        if self.session and piston in self.session.piston_metrics:
            metrics = self.session.piston_metrics[piston]
            metrics.total_time += duration
            metrics.times.append(duration)
            metrics.min_time = min(metrics.min_time, duration)
            metrics.max_time = max(metrics.max_time, duration)

            if success:
                metrics.successes += 1
                if self._current_algorithm:
                    algo_success = metrics.algorithm_success
                    algo_success[self._current_algorithm] = algo_success.get(self._current_algorithm, 0) + 1
            else:
                metrics.failures += 1

        event_type = MonitorEvent.PISTON_SUCCESS if success else MonitorEvent.PISTON_FAILED
        self.record_event(
            event_type,
            piston=piston,
            message=f"Piston {'succeeded' if success else 'failed'}: {piston} (score={quality_score:.1f})",
            level=MonitorLevel.INFO if success else MonitorLevel.WARNING,
            duration=duration,
            data={
                'success': success,
                'quality_score': quality_score,
                'errors': errors,
                'warnings': warnings
            }
        )

        self._current_piston = ''
        self._current_algorithm = ''

    def record_piston_retry(self, piston: str, reason: str, attempt: int):
        """Record piston retry."""
        if self.session and piston in self.session.piston_metrics:
            self.session.piston_metrics[piston].retries += 1

        self.record_event(
            MonitorEvent.PISTON_RETRY,
            piston=piston,
            message=f"Piston retry: {piston} (attempt {attempt}) - {reason}",
            level=MonitorLevel.INFO,
            data={'reason': reason, 'attempt': attempt}
        )

    def record_piston_cascade(self, piston: str, from_algo: str, to_algo: str):
        """Record algorithm cascade (switching to next algorithm)."""
        if self.session and piston in self.session.piston_metrics:
            self.session.piston_metrics[piston].cascades += 1

        self._current_algorithm = to_algo

        self.record_event(
            MonitorEvent.PISTON_CASCADE,
            piston=piston,
            algorithm=to_algo,
            message=f"Cascade: {piston} switching from {from_algo} to {to_algo}",
            level=MonitorLevel.INFO,
            data={'from_algorithm': from_algo, 'to_algorithm': to_algo}
        )

    # =========================================================================
    # ALGORITHM PERFORMANCE TRACKING (Detailed per-algorithm metrics)
    # =========================================================================

    def record_algorithm_start(self, piston: str, algorithm: str):
        """Record algorithm execution start."""
        algo_key = f"{piston}:{algorithm}"
        self._algorithm_start_times[algo_key] = time.time()
        self._current_algorithm = algorithm

        if self.session:
            if algo_key not in self.session.algorithm_metrics:
                self.session.algorithm_metrics[algo_key] = AlgorithmMetrics(
                    name=algorithm,
                    piston=piston
                )
            self.session.algorithm_metrics[algo_key].executions += 1

        self.record_event(
            MonitorEvent.ALGORITHM_START,
            piston=piston,
            algorithm=algorithm,
            message=f"Algorithm started: {piston}/{algorithm}",
            level=MonitorLevel.DEBUG
        )

    def record_algorithm_end(
        self,
        piston: str,
        algorithm: str,
        success: bool,
        quality_score: float = 0.0
    ):
        """Record algorithm execution end with performance data."""
        algo_key = f"{piston}:{algorithm}"
        duration = 0.0

        if algo_key in self._algorithm_start_times:
            duration = time.time() - self._algorithm_start_times[algo_key]

        if self.session and algo_key in self.session.algorithm_metrics:
            metrics = self.session.algorithm_metrics[algo_key]
            metrics.total_time += duration
            metrics.times.append(duration)
            metrics.min_time = min(metrics.min_time, duration)
            metrics.max_time = max(metrics.max_time, duration)
            metrics.quality_scores.append(quality_score)

            if success:
                metrics.successes += 1
            else:
                metrics.failures += 1

        self.record_event(
            MonitorEvent.ALGORITHM_END,
            piston=piston,
            algorithm=algorithm,
            message=f"Algorithm ended: {piston}/{algorithm} ({'SUCCESS' if success else 'FAILED'}, quality={quality_score:.2f})",
            level=MonitorLevel.DEBUG,
            duration=duration,
            data={
                'success': success,
                'quality_score': quality_score
            }
        )

    # =========================================================================
    # PERFORMANCE ANALYSIS (Comparative ranking of all members)
    # =========================================================================

    def calculate_performance_ranking(self) -> PerformanceRanking:
        """
        Calculate performance rankings for all BBL members.

        Analyzes pistons, algorithms, and phases to determine:
        - Fastest/slowest pistons
        - Most/least reliable pistons
        - Best/worst algorithms
        - Bottleneck phases
        - Most efficient algorithms
        """
        if not self.session:
            return PerformanceRanking()

        ranking = PerformanceRanking()

        # Piston rankings
        piston_speeds = []
        piston_success = []

        for name, pm in self.session.piston_metrics.items():
            if pm.executions > 0:
                piston_speeds.append((name, pm.avg_time))
                piston_success.append((name, pm.success_rate))

        # Sort by speed (ascending = faster is better)
        piston_speeds.sort(key=lambda x: x[1])
        ranking.piston_by_speed = piston_speeds

        if piston_speeds:
            ranking.fastest_piston = piston_speeds[0][0]
            ranking.slowest_piston = piston_speeds[-1][0]

        # Sort by success rate (descending = higher is better)
        piston_success.sort(key=lambda x: x[1], reverse=True)
        ranking.piston_by_success = piston_success

        if piston_success:
            ranking.most_reliable_piston = piston_success[0][0]
            ranking.least_reliable_piston = piston_success[-1][0]

        # Algorithm rankings
        algo_efficiency = []

        for key, am in self.session.algorithm_metrics.items():
            if am.executions > 0:
                algo_efficiency.append((key, am.efficiency_score))

        # Sort by efficiency (descending = higher is better)
        algo_efficiency.sort(key=lambda x: x[1], reverse=True)
        ranking.algorithm_by_efficiency = algo_efficiency

        if algo_efficiency:
            ranking.most_efficient_algorithm = algo_efficiency[0][0]
            ranking.best_algorithm = algo_efficiency[0][0]
            ranking.worst_algorithm = algo_efficiency[-1][0]

        # Phase rankings (by time - find bottleneck)
        phase_times = []

        for name, pm in self.session.phase_metrics.items():
            if pm.executions > 0:
                phase_times.append((name, pm.avg_time))

        phase_times.sort(key=lambda x: x[1], reverse=True)
        ranking.phase_by_time = phase_times

        if phase_times:
            ranking.bottleneck_phase = phase_times[0][0]

        # Store ranking in session
        self.session.performance_ranking = ranking

        return ranking

    def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        ranking = self.calculate_performance_ranking()

        lines = []
        lines.append("=" * 60)
        lines.append("BBL PERFORMANCE ANALYSIS")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append("## SUMMARY")
        lines.append(f"  Fastest Piston:      {ranking.fastest_piston}")
        lines.append(f"  Slowest Piston:      {ranking.slowest_piston}")
        lines.append(f"  Most Reliable:       {ranking.most_reliable_piston}")
        lines.append(f"  Least Reliable:      {ranking.least_reliable_piston}")
        lines.append(f"  Best Algorithm:      {ranking.best_algorithm}")
        lines.append(f"  Bottleneck Phase:    {ranking.bottleneck_phase}")
        lines.append("")

        # Piston Performance Table
        lines.append("## PISTON PERFORMANCE (by speed)")
        lines.append("-" * 60)
        lines.append(f"{'Piston':<20} {'Avg Time':<12} {'Success Rate':<15} {'Retries':<10}")
        lines.append("-" * 60)

        for name, pm in sorted(self.session.piston_metrics.items(), key=lambda x: x[1].avg_time):
            lines.append(f"{name:<20} {pm.avg_time:.3f}s       {pm.success_rate*100:.0f}%            {pm.retries}")

        lines.append("")

        # Algorithm Performance Table
        if self.session.algorithm_metrics:
            lines.append("## ALGORITHM PERFORMANCE (by efficiency)")
            lines.append("-" * 60)
            lines.append(f"{'Algorithm':<30} {'Avg Time':<12} {'Success':<10} {'Quality':<10} {'Efficiency':<10}")
            lines.append("-" * 60)

            for key, am in sorted(self.session.algorithm_metrics.items(),
                                  key=lambda x: x[1].efficiency_score, reverse=True):
                lines.append(f"{key:<30} {am.avg_time:.3f}s       {am.success_rate*100:.0f}%       {am.avg_quality:.2f}       {am.efficiency_score:.2f}")

        lines.append("")

        # Phase Performance Table
        lines.append("## PHASE PERFORMANCE (by time)")
        lines.append("-" * 60)
        lines.append(f"{'Phase':<25} {'Avg Time':<12} {'Success Rate':<15} {'Timeouts':<10}")
        lines.append("-" * 60)

        for name, pm in sorted(self.session.phase_metrics.items(),
                               key=lambda x: x[1].avg_time, reverse=True):
            lines.append(f"{name:<25} {pm.avg_time:.3f}s       {pm.success_rate*100:.0f}%            {pm.timeouts}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_bottleneck_analysis(self) -> Dict:
        """
        Analyze bottlenecks in the BBL execution.

        Returns dict with:
        - bottleneck_phase: Phase taking most time
        - bottleneck_piston: Piston taking most time
        - slowest_algorithm: Algorithm with highest avg time
        - recommendations: List of optimization suggestions
        """
        analysis = {
            'bottleneck_phase': '',
            'bottleneck_piston': '',
            'slowest_algorithm': '',
            'recommendations': []
        }

        if not self.session:
            return analysis

        # Find bottleneck phase
        max_phase_time = 0
        for name, pm in self.session.phase_metrics.items():
            if pm.avg_time > max_phase_time:
                max_phase_time = pm.avg_time
                analysis['bottleneck_phase'] = name

        # Find bottleneck piston
        max_piston_time = 0
        for name, pm in self.session.piston_metrics.items():
            if pm.avg_time > max_piston_time:
                max_piston_time = pm.avg_time
                analysis['bottleneck_piston'] = name

        # Find slowest algorithm
        max_algo_time = 0
        for key, am in self.session.algorithm_metrics.items():
            if am.avg_time > max_algo_time:
                max_algo_time = am.avg_time
                analysis['slowest_algorithm'] = key

        # Generate recommendations
        recs = []

        # Check for high retry counts
        for name, pm in self.session.piston_metrics.items():
            if pm.retries > pm.executions:
                recs.append(f"Piston '{name}' has high retry rate ({pm.retries} retries for {pm.executions} executions) - consider tuning parameters")

        # Check for low success rates
        for name, pm in self.session.piston_metrics.items():
            if pm.success_rate < 0.7 and pm.executions > 0:
                recs.append(f"Piston '{name}' has low success rate ({pm.success_rate*100:.0f}%) - investigate root cause")

        # Check for cascade-heavy pistons
        for name, pm in self.session.piston_metrics.items():
            if pm.cascades > 3:
                recs.append(f"Piston '{name}' frequently cascades ({pm.cascades} times) - consider different default algorithm")

        # Check for DRC issues
        if self.session.total_drc_violations > 10:
            recs.append(f"High DRC violation count ({self.session.total_drc_violations}) - review routing/placement parameters")

        # Check for escalations
        if self.session.total_escalations > 2:
            recs.append(f"Multiple escalations ({self.session.total_escalations}) - may indicate design complexity issues")

        analysis['recommendations'] = recs

        return analysis

    # =========================================================================
    # DRC TRACKING
    # =========================================================================

    def record_drc_check(self, piston: str, check_type: str = 'internal'):
        """Record DRC check start."""
        if self.session:
            self.session.total_drc_checks += 1
            if piston in self.session.piston_metrics:
                self.session.piston_metrics[piston].drc_checks += 1

        event_type = MonitorEvent.DRC_CHECK_START if check_type == 'internal' else MonitorEvent.KICAD_DRC_START
        self.record_event(
            event_type,
            piston=piston,
            message=f"DRC check started ({check_type}): {piston}",
            level=MonitorLevel.DEBUG
        )

    def record_drc_result(
        self,
        passed: bool,
        errors: int = 0,
        warnings: int = 0,
        check_type: str = 'internal',
        piston: str = ''
    ):
        """Record DRC check result."""
        if self.session:
            if check_type == 'internal':
                self.session.internal_drc_errors += errors
                self.session.internal_drc_warnings += warnings
            else:
                self.session.kicad_drc_errors += errors
                self.session.kicad_drc_warnings += warnings

            if not passed:
                self.session.total_drc_violations += errors

            if piston and piston in self.session.piston_metrics:
                if passed:
                    self.session.piston_metrics[piston].drc_passed += 1
                else:
                    self.session.piston_metrics[piston].drc_failed += 1

        event_type = MonitorEvent.DRC_PASSED if passed else MonitorEvent.DRC_FAILED
        self.record_event(
            event_type,
            piston=piston,
            message=f"DRC {'passed' if passed else 'failed'} ({check_type}): {errors} errors, {warnings} warnings",
            level=MonitorLevel.INFO if passed else MonitorLevel.WARNING,
            data={
                'passed': passed,
                'errors': errors,
                'warnings': warnings,
                'check_type': check_type
            }
        )

    def record_drc_violation(
        self,
        violation_type: str,
        message: str,
        position: Tuple[float, float] = None,
        check_type: str = 'internal'
    ):
        """Record a specific DRC violation."""
        event_type = MonitorEvent.DRC_VIOLATION if check_type == 'internal' else MonitorEvent.KICAD_DRC_VIOLATION
        self.record_event(
            event_type,
            message=f"DRC violation ({check_type}): {violation_type} - {message}",
            level=MonitorLevel.WARNING,
            data={
                'violation_type': violation_type,
                'message': message,
                'position': position
            }
        )

    # =========================================================================
    # OUTPUT TRACKING
    # =========================================================================

    def record_file_generated(self, filename: str, file_type: str = ''):
        """Record file generation."""
        if self.session:
            self.session.total_files_generated += 1

        self.record_event(
            MonitorEvent.FILE_GENERATED,
            message=f"File generated: {filename}" + (f" ({file_type})" if file_type else ""),
            level=MonitorLevel.INFO,
            data={'filename': filename, 'file_type': file_type}
        )

    # =========================================================================
    # PROGRESS TRACKING
    # =========================================================================

    def record_progress(
        self,
        phase: str,
        percentage: float,
        message: str = '',
        piston: str = '',
        iteration: int = 0,
        max_iterations: int = 0
    ):
        """Record progress update."""
        self.record_event(
            MonitorEvent.PROGRESS_UPDATE,
            phase=phase,
            piston=piston,
            message=message or f"Progress: {percentage:.0f}%",
            level=MonitorLevel.DEBUG,
            data={
                'percentage': percentage,
                'iteration': iteration,
                'max_iterations': max_iterations
            }
        )

    # =========================================================================
    # WARNING/ERROR TRACKING
    # =========================================================================

    def record_warning(self, message: str, data: Dict = None):
        """Record a warning."""
        self.record_event(
            MonitorEvent.WARNING,
            message=message,
            level=MonitorLevel.WARNING,
            data=data or {}
        )

    def record_error(self, message: str, data: Dict = None, critical: bool = False):
        """Record an error."""
        event_type = MonitorEvent.CRITICAL_ERROR if critical else MonitorEvent.ERROR
        level = MonitorLevel.CRITICAL if critical else MonitorLevel.ERROR
        self.record_event(
            event_type,
            message=message,
            level=level,
            data=data or {}
        )

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_report(self, format: str = 'json') -> str:
        """
        Generate a report of the current session.

        Args:
            format: 'json', 'markdown', or 'html'

        Returns:
            Report content as string
        """
        if format == 'json':
            return self._generate_json_report()
        elif format == 'markdown':
            return self._generate_markdown_report()
        elif format == 'html':
            return self._generate_html_report()
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        data = {
            'session': self.session.to_dict() if self.session else None,
            'events': [e.to_dict() for e in self.events],
            'generated_at': datetime.now().isoformat()
        }
        return json.dumps(data, indent=2)

    def _generate_markdown_report(self) -> str:
        """Generate Markdown report."""
        if not self.session:
            return "# BBL Monitor Report\n\nNo session data available."

        lines = []
        s = self.session

        # Header
        lines.append(f"# BBL Monitor Report: {s.bbl_id}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| **Status** | {'SUCCESS' if s.success else 'FAILED'} |")
        lines.append(f"| **Duration** | {s.duration:.2f}s |")
        lines.append(f"| **Total Events** | {s.total_events} |")
        lines.append(f"| **Phases Executed** | {s.total_phases} |")
        lines.append(f"| **Pistons Run** | {s.total_pistons} |")
        lines.append(f"| **Algorithms Tried** | {s.total_algorithms} |")
        lines.append(f"| **Checkpoints** | {s.total_checkpoints} |")
        lines.append(f"| **Rollbacks** | {s.total_rollbacks} |")
        lines.append(f"| **Escalations** | {s.total_escalations} |")
        lines.append(f"| **DRC Checks** | {s.total_drc_checks} |")
        lines.append(f"| **DRC Violations** | {s.total_drc_violations} |")
        lines.append(f"| **Files Generated** | {s.total_files_generated} |")
        lines.append("")

        # Routing
        if s.total_nets > 0:
            lines.append("## Routing Results")
            lines.append("")
            lines.append(f"- **Completion:** {s.routing_completion*100:.1f}%")
            lines.append(f"- **Routed Nets:** {s.routed_nets}/{s.total_nets}")
            lines.append("")

        # DRC Summary
        lines.append("## DRC Summary")
        lines.append("")
        lines.append("| Check Type | Errors | Warnings |")
        lines.append("|------------|--------|----------|")
        lines.append(f"| Internal DRC | {s.internal_drc_errors} | {s.internal_drc_warnings} |")
        lines.append(f"| KiCad DRC | {s.kicad_drc_errors} | {s.kicad_drc_warnings} |")
        lines.append("")

        # Phase Metrics
        if s.phase_metrics:
            lines.append("## Phase Metrics")
            lines.append("")
            lines.append("| Phase | Executions | Success Rate | Avg Time | Timeouts |")
            lines.append("|-------|------------|--------------|----------|----------|")
            for name, pm in s.phase_metrics.items():
                lines.append(f"| {name} | {pm.executions} | {pm.success_rate*100:.0f}% | {pm.avg_time:.2f}s | {pm.timeouts} |")
            lines.append("")

        # Piston Metrics
        if s.piston_metrics:
            lines.append("## Piston Metrics")
            lines.append("")
            lines.append("| Piston | Executions | Success Rate | Avg Time | Retries | Cascades |")
            lines.append("|--------|------------|--------------|----------|---------|----------|")
            for name, pm in s.piston_metrics.items():
                lines.append(f"| {name} | {pm.executions} | {pm.success_rate*100:.0f}% | {pm.avg_time:.2f}s | {pm.retries} | {pm.cascades} |")
            lines.append("")

            # Algorithm breakdown
            lines.append("### Algorithm Usage")
            lines.append("")
            for name, pm in s.piston_metrics.items():
                if pm.algorithms_used:
                    lines.append(f"**{name}:**")
                    for algo, count in pm.algorithms_used.items():
                        success = pm.algorithm_success.get(algo, 0)
                        rate = success / count * 100 if count > 0 else 0
                        lines.append(f"- {algo}: {count} uses, {rate:.0f}% success")
                    lines.append("")

        # Event Timeline (last 20)
        lines.append("## Event Timeline (Recent)")
        lines.append("")
        lines.append("| Time | Event | Phase | Piston | Message |")
        lines.append("|------|-------|-------|--------|---------|")
        for event in self.events[-20:]:
            time_str = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
            # Handle event_type being either an enum or a string
            event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            lines.append(f"| {time_str} | {event_type_str} | {event.phase} | {event.piston} | {event.message[:50]} |")
        lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        # Convert markdown to basic HTML
        md = self._generate_markdown_report()

        html_lines = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>BBL Monitor Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            "h1 { color: #333; }",
            "h2 { color: #666; border-bottom: 1px solid #ddd; }",
            ".success { color: green; }",
            ".failed { color: red; }",
            "</style>",
            "</head><body>"
        ]

        # Simple markdown to HTML conversion
        for line in md.split('\n'):
            if line.startswith('# '):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith('## '):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith('### '):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith('| '):
                # Table row
                if '---|' in line:
                    continue  # Skip separator
                cells = line.split('|')[1:-1]
                if line.startswith('| **'):
                    html_lines.append("<tr>" + "".join(f"<th>{c.strip()}</th>" for c in cells) + "</tr>")
                else:
                    html_lines.append("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in cells) + "</tr>")
            elif line.startswith('- '):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith('**') and line.endswith('**'):
                html_lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
            elif line:
                html_lines.append(f"<p>{line}</p>")

        html_lines.append("</body></html>")
        return "\n".join(html_lines)

    def _save_session_report(self):
        """Save session report to file."""
        if not self.session:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # Save JSON report
        json_path = os.path.join(self.output_dir, f"bbl_monitor_{self.session.bbl_id}.json")
        with open(json_path, 'w') as f:
            f.write(self._generate_json_report())

        # Save Markdown report
        md_path = os.path.join(self.output_dir, f"bbl_monitor_{self.session.bbl_id}.md")
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_report())

        if self.verbose:
            print(f"[MONITOR] Reports saved to {self.output_dir}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _print_event(self, event: EventRecord):
        """Print an event to console."""
        time_str = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S.%f')[:-3]
        level_str = event.level.name[0]  # First letter

        # Color codes
        colors = {
            MonitorLevel.DEBUG: '\033[90m',    # Gray
            MonitorLevel.INFO: '\033[0m',       # Default
            MonitorLevel.WARNING: '\033[93m',   # Yellow
            MonitorLevel.ERROR: '\033[91m',     # Red
            MonitorLevel.CRITICAL: '\033[91;1m' # Bold Red
        }
        reset = '\033[0m'
        color = colors.get(event.level, '')

        # Format context
        context = ''
        if event.phase:
            context += f"[{event.phase}]"
        if event.piston:
            context += f"[{event.piston}]"
        if event.algorithm:
            context += f"[{event.algorithm}]"

        # Duration suffix
        duration_str = f" ({event.duration:.2f}s)" if event.duration > 0 else ""

        print(f"{color}[{time_str}][{level_str}]{context} {event.message}{duration_str}{reset}")

    def get_session_summary(self) -> Dict:
        """Get a summary of the current session."""
        if not self.session:
            return {'error': 'No active session'}
        return self.session.to_dict()

    def get_events_by_type(self, event_type: MonitorEvent) -> List[EventRecord]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_piston(self, piston: str) -> List[EventRecord]:
        """Get all events for a specific piston."""
        return [e for e in self.events if e.piston == piston]

    def get_events_by_level(self, level: MonitorLevel) -> List[EventRecord]:
        """Get all events at or above a specific level."""
        return [e for e in self.events if e.level.value >= level.value]

    def clear_events(self):
        """Clear all recorded events (keeps metrics)."""
        with self._lock:
            self.events = []

    # =========================================================================
    # CASCADE OPTIMIZER INTEGRATION
    # =========================================================================

    def get_algorithm_performance_for_optimizer(self, piston: str = None) -> Dict:
        """
        Export algorithm performance data for CascadeOptimizer integration.

        Returns data in a format the CascadeOptimizer can import to update
        its algorithm priority ordering based on observed execution results.

        Args:
            piston: Optional piston name to filter. If None, returns all.

        Returns:
            Dict with format:
            {
                'piston_name': {
                    'algorithm_name': {
                        'executions': int,
                        'successes': int,
                        'failures': int,
                        'avg_time_ms': float,
                        'avg_quality': float,
                        'success_rate': float
                    }
                }
            }
        """
        if not self.session:
            return {}

        result = {}

        for algo_key, metrics in self.session.algorithm_metrics.items():
            # algo_key is "piston:algorithm" format
            parts = algo_key.split(':')
            if len(parts) != 2:
                continue

            piston_name, algo_name = parts

            # Filter by piston if specified
            if piston and piston_name != piston:
                continue

            if piston_name not in result:
                result[piston_name] = {}

            result[piston_name][algo_name] = {
                'executions': metrics.executions,
                'successes': metrics.successes,
                'failures': metrics.failures,
                'avg_time_ms': metrics.avg_time * 1000,  # Convert to ms
                'avg_quality': metrics.avg_quality,
                'success_rate': metrics.success_rate
            }

        return result

    def create_cascade_optimizer_callback(self) -> Callable:
        """
        Create a callback function that CascadeOptimizer can use to receive
        real-time algorithm result updates.

        The callback signature matches what CascadeOptimizer expects:
        callback(piston, algorithm, success, time_ms, quality_score)

        Returns:
            Callback function for real-time updates
        """
        def optimizer_callback(piston: str, algorithm: str, success: bool,
                               time_ms: float = 0.0, quality_score: float = 0.0):
            """Callback for CascadeOptimizer to record algorithm results."""
            # Record to our metrics
            self.record_algorithm_end(
                piston=piston,
                algorithm=algorithm,
                success=success,
                quality_score=quality_score
            )

        return optimizer_callback

    def sync_to_cascade_optimizer(self, optimizer) -> int:
        """
        Push monitor session data to a CascadeOptimizer instance.

        This allows the CascadeOptimizer to learn from the execution
        results captured during BBL runs.

        Args:
            optimizer: CascadeOptimizer instance with sync_from_monitor() method

        Returns:
            Number of records synced
        """
        if not self.session:
            return 0

        # Use the optimizer's sync method if available
        if hasattr(optimizer, 'sync_from_monitor'):
            return optimizer.sync_from_monitor(self.session)

        # Fallback: manually push each algorithm result
        count = 0
        for algo_key, metrics in self.session.algorithm_metrics.items():
            parts = algo_key.split(':')
            if len(parts) != 2:
                continue

            piston_name, algo_name = parts

            # Record each execution
            for i in range(metrics.successes):
                if hasattr(optimizer, 'record_result'):
                    avg_time = metrics.avg_time * 1000 if metrics.avg_time else 500
                    optimizer.record_result(
                        piston_name, algo_name, None,
                        success=True,
                        time_ms=avg_time,
                        quality_score=metrics.avg_quality
                    )
                    count += 1

            for i in range(metrics.failures):
                if hasattr(optimizer, 'record_result'):
                    avg_time = metrics.avg_time * 1000 if metrics.avg_time else 1000
                    optimizer.record_result(
                        piston_name, algo_name, None,
                        success=False,
                        time_ms=avg_time,
                        quality_score=0.0
                    )
                    count += 1

        return count

    def get_cascade_recommendations(self) -> Dict[str, List[str]]:
        """
        Generate algorithm cascade recommendations based on session performance.

        Analyzes the algorithm metrics from the current session and returns
        a recommended ordering for each piston based on:
        - Success rate (primary factor)
        - Execution time (secondary factor)
        - Quality scores (tertiary factor)

        Returns:
            Dict mapping piston name to ordered list of algorithm names
        """
        if not self.session or not self.session.algorithm_metrics:
            return {}

        # Group algorithms by piston
        piston_algos: Dict[str, List[Tuple[str, AlgorithmMetrics]]] = {}

        for algo_key, metrics in self.session.algorithm_metrics.items():
            parts = algo_key.split(':')
            if len(parts) != 2:
                continue

            piston_name, algo_name = parts

            if piston_name not in piston_algos:
                piston_algos[piston_name] = []

            piston_algos[piston_name].append((algo_name, metrics))

        # Generate recommendations for each piston
        recommendations = {}

        for piston_name, algos in piston_algos.items():
            # Calculate a combined score for each algorithm
            scored_algos = []

            for algo_name, metrics in algos:
                if metrics.executions == 0:
                    continue

                # Weight: 60% success rate, 25% speed, 15% quality
                success_score = metrics.success_rate * 0.6

                # Speed score: normalize to 0-1 range (faster = higher)
                # Assume max reasonable time is 10 seconds
                speed_score = max(0, 1 - (metrics.avg_time / 10.0)) * 0.25

                # Quality score already 0-1 range
                quality_score = metrics.avg_quality * 0.15

                combined_score = success_score + speed_score + quality_score
                scored_algos.append((algo_name, combined_score))

            # Sort by score (descending)
            scored_algos.sort(key=lambda x: x[1], reverse=True)

            recommendations[piston_name] = [name for name, _ in scored_algos]

        return recommendations

    def get_optimizer_compatible_summary(self) -> Dict:
        """
        Get a session summary in a format compatible with CascadeOptimizer reporting.

        This provides the data structure expected by dashboard and reporting tools
        that integrate both BBLMonitor and CascadeOptimizer.

        Returns:
            Dict with optimizer-compatible summary data
        """
        if not self.session:
            return {}

        return {
            'session_id': self.session.bbl_id,
            'duration_ms': self.session.duration * 1000,
            'success': self.session.success,
            'routing_completion': self.session.routing_completion,
            'algorithms_executed': self.session.total_algorithms,
            'cascades_triggered': sum(
                pm.cascades for pm in self.session.piston_metrics.values()
            ),
            'algorithm_performance': self.get_algorithm_performance_for_optimizer(),
            'recommendations': self.get_cascade_recommendations(),
            'bottleneck_analysis': self.get_bottleneck_analysis(),
            'timestamp': self.session.end_time or self.session.start_time
        }
