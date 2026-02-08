"""
MonitorOptimizerBridge
======================

Bridge class that connects BBLMonitor and CascadeOptimizer for seamless
real-time analytics and algorithm optimization.

This bridge provides:
1. Real-time event forwarding from BBL execution to optimizer
2. Automatic algorithm reordering based on session results
3. Unified statistics and reporting interface
4. Session-based learning with persistence

USAGE:
======
    from monitor_optimizer_bridge import MonitorOptimizerBridge

    # Create bridge (creates or uses existing monitor/optimizer)
    bridge = MonitorOptimizerBridge(output_dir='./output')

    # Start a BBL session
    bridge.start_session('BBL_001')

    # Record algorithm execution (automatically updates both systems)
    bridge.record_algorithm_execution(
        piston='routing',
        algorithm='astar',
        success=True,
        time_ms=450.5,
        quality_score=0.85
    )

    # Get optimized cascade for next execution
    cascade = bridge.get_optimized_cascade('routing', parts_db)

    # End session and sync learnings
    bridge.end_session(success=True)

Author: PCB Engine Team
Date: 2026-02-07
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import time
import json
import os
import threading


@dataclass
class BridgeConfig:
    """Configuration for MonitorOptimizerBridge."""
    output_dir: str = './output'
    auto_sync: bool = True  # Automatically sync optimizer after session
    sync_threshold: int = 5  # Minimum algorithm executions before sync
    verbose: bool = True
    persist_on_sync: bool = True  # Save optimizer state after sync


@dataclass
class BridgeStatistics:
    """Combined statistics from both systems."""
    sessions_monitored: int = 0
    algorithms_executed: int = 0
    cascades_triggered: int = 0
    optimizations_applied: int = 0
    sync_operations: int = 0
    last_sync_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'sessions_monitored': self.sessions_monitored,
            'algorithms_executed': self.algorithms_executed,
            'cascades_triggered': self.cascades_triggered,
            'optimizations_applied': self.optimizations_applied,
            'sync_operations': self.sync_operations,
            'last_sync_time': datetime.fromtimestamp(self.last_sync_time).isoformat() if self.last_sync_time > 0 else None
        }


class MonitorOptimizerBridge:
    """
    Bridge connecting BBLMonitor and CascadeOptimizer.

    Provides a unified interface for:
    - Recording algorithm executions (updates both systems)
    - Getting optimized cascades (uses optimizer with monitor feedback)
    - Syncing learnings between sessions
    - Combined reporting and analytics
    """

    def __init__(
        self,
        config: BridgeConfig = None,
        monitor = None,  # BBLMonitor instance
        optimizer = None  # CascadeOptimizer instance
    ):
        """
        Initialize the bridge.

        Args:
            config: Bridge configuration
            monitor: Optional BBLMonitor instance (created if not provided)
            optimizer: Optional CascadeOptimizer instance (created if not provided)
        """
        self.config = config or BridgeConfig()

        # Import lazily to avoid circular imports
        self._monitor = monitor
        self._optimizer = optimizer

        # Statistics
        self.stats = BridgeStatistics()

        # Current session state
        self._session_active = False
        self._session_id: str = ''
        self._session_start: float = 0.0

        # Thread safety
        self._lock = threading.Lock()

        # Pending records (batched for efficiency)
        self._pending_records: List[Dict] = []

    @property
    def monitor(self):
        """Lazy-load BBLMonitor."""
        if self._monitor is None:
            try:
                from .bbl_monitor import BBLMonitor, MonitorLevel
            except ImportError:
                from bbl_monitor import BBLMonitor, MonitorLevel
            self._monitor = BBLMonitor(
                output_dir=self.config.output_dir,
                verbose=self.config.verbose
            )
        return self._monitor

    @property
    def optimizer(self):
        """Lazy-load CascadeOptimizer."""
        if self._optimizer is None:
            try:
                from .cascade_optimizer import CascadeOptimizer, CascadeOptimizerConfig
            except ImportError:
                from cascade_optimizer import CascadeOptimizer, CascadeOptimizerConfig
            optimizer_config = CascadeOptimizerConfig(
                history_file=os.path.join(self.config.output_dir, 'cascade_history.json'),
                auto_save=self.config.persist_on_sync
            )
            self._optimizer = CascadeOptimizer(optimizer_config)
            # Try to load existing history
            self._optimizer.load()
        return self._optimizer

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, session_id: str = None):
        """
        Start a new monitoring and optimization session.

        Args:
            session_id: Optional session ID (generated if not provided)
        """
        with self._lock:
            self._session_id = session_id or f"BBL_{int(time.time())}"
            self._session_start = time.time()
            self._session_active = True
            self._pending_records = []
            self.stats.sessions_monitored += 1

        # Start monitor session
        self.monitor.start_session(self._session_id)

        if self.config.verbose:
            print(f"[BRIDGE] Session started: {self._session_id}")

    def end_session(self, success: bool = False, routing_stats: Dict = None):
        """
        End the current session and sync learnings.

        Args:
            success: Whether the BBL run was successful
            routing_stats: Optional routing statistics
        """
        if not self._session_active:
            return

        # End monitor session
        self.monitor.end_session(success=success, routing_stats=routing_stats)

        # Sync to optimizer if auto_sync enabled
        if self.config.auto_sync:
            self.sync_to_optimizer()

        with self._lock:
            self._session_active = False

        if self.config.verbose:
            duration = time.time() - self._session_start
            print(f"[BRIDGE] Session ended: {self._session_id} ({'SUCCESS' if success else 'FAILED'}) - {duration:.2f}s")

    # =========================================================================
    # ALGORITHM EXECUTION RECORDING
    # =========================================================================

    def record_algorithm_start(self, piston: str, algorithm: str):
        """
        Record algorithm execution start.

        Args:
            piston: Piston name (e.g., 'routing', 'placement')
            algorithm: Algorithm name (e.g., 'astar', 'lee')
        """
        self.monitor.record_algorithm_start(piston, algorithm)

    def record_algorithm_end(
        self,
        piston: str,
        algorithm: str,
        success: bool,
        time_ms: float = 0.0,
        quality_score: float = 0.0,
        profile = None  # DesignProfile for optimizer
    ):
        """
        Record algorithm execution end (updates both systems).

        Args:
            piston: Piston name
            algorithm: Algorithm name
            success: Whether execution succeeded
            time_ms: Execution time in milliseconds
            quality_score: Quality score 0.0-1.0
            profile: Optional DesignProfile for optimizer context
        """
        # Update monitor
        self.monitor.record_algorithm_end(
            piston=piston,
            algorithm=algorithm,
            success=success,
            quality_score=quality_score
        )

        # Update optimizer
        self.optimizer.record_result(
            piston=piston,
            algorithm=algorithm,
            profile=profile,
            success=success,
            time_ms=time_ms,
            quality_score=quality_score
        )

        with self._lock:
            self.stats.algorithms_executed += 1

    def record_cascade(self, piston: str, from_algo: str, to_algo: str):
        """
        Record algorithm cascade (fallback to next algorithm).

        Args:
            piston: Piston name
            from_algo: Algorithm that failed
            to_algo: Algorithm to try next
        """
        self.monitor.record_piston_cascade(piston, from_algo, to_algo)

        with self._lock:
            self.stats.cascades_triggered += 1

    # =========================================================================
    # CASCADE OPTIMIZATION
    # =========================================================================

    def get_optimized_cascade(
        self,
        piston: str,
        parts_db: Dict = None,
        board_width: float = 50.0,
        board_height: float = 40.0
    ) -> List[Tuple[str, str]]:
        """
        Get optimized algorithm cascade for a piston.

        Uses the CascadeOptimizer with learnings from previous sessions
        to return algorithms in optimal order.

        Args:
            piston: Piston name (e.g., 'routing', 'placement')
            parts_db: Optional parts database for design profile
            board_width: Board width in mm
            board_height: Board height in mm

        Returns:
            List of (algorithm_id, description) tuples in optimal order
        """
        # Create design profile if parts_db provided
        profile = None
        if parts_db:
            try:
                from .cascade_optimizer import DesignProfile
            except ImportError:
                from cascade_optimizer import DesignProfile
            profile = DesignProfile.from_parts_db(parts_db, board_width, board_height)

        # Get optimized cascade from optimizer
        cascade = self.optimizer.get_cascade(piston, profile)

        with self._lock:
            self.stats.optimizations_applied += 1

        return cascade

    def get_recommendations(self, parts_db: Dict = None) -> Dict[str, List[str]]:
        """
        Get algorithm recommendations for all pistons.

        Combines optimizer statistics with recent session performance.

        Args:
            parts_db: Optional parts database for design profile

        Returns:
            Dict mapping piston name to recommended algorithm order
        """
        # Get optimizer recommendations
        profile = None
        if parts_db:
            try:
                from .cascade_optimizer import DesignProfile
            except ImportError:
                from cascade_optimizer import DesignProfile
            profile = DesignProfile.from_parts_db(parts_db)

        optimizer_recs = self.optimizer.get_recommendations(profile)

        # Get monitor recommendations from current session
        monitor_recs = self.monitor.get_cascade_recommendations()

        # Merge: prefer recent session data, fallback to optimizer
        merged = {}
        all_pistons = set(optimizer_recs.keys()) | set(monitor_recs.keys())

        for piston in all_pistons:
            if piston in monitor_recs and monitor_recs[piston]:
                merged[piston] = monitor_recs[piston]
            elif piston in optimizer_recs and optimizer_recs[piston]:
                merged[piston] = optimizer_recs[piston]

        return merged

    # =========================================================================
    # SYNCHRONIZATION
    # =========================================================================

    def sync_to_optimizer(self) -> int:
        """
        Sync monitor session data to optimizer.

        Transfers algorithm performance data from the current monitor
        session to the optimizer for learning.

        Returns:
            Number of records synced
        """
        count = self.monitor.sync_to_cascade_optimizer(self.optimizer)

        with self._lock:
            self.stats.sync_operations += 1
            self.stats.last_sync_time = time.time()

        # Persist optimizer state if configured
        if self.config.persist_on_sync:
            self.optimizer.save()

        if self.config.verbose:
            print(f"[BRIDGE] Synced {count} records to optimizer")

        return count

    def sync_from_optimizer(self) -> Dict:
        """
        Get optimizer statistics for the monitor to use.

        Returns optimizer's learned algorithm priorities that the monitor
        can use for baseline comparisons.

        Returns:
            Dict with optimizer statistics
        """
        return self.optimizer.get_monitor_compatible_stats()

    # =========================================================================
    # COMBINED REPORTING
    # =========================================================================

    def get_combined_statistics(self) -> Dict:
        """
        Get combined statistics from both systems.

        Returns:
            Dict with bridge, monitor, and optimizer statistics
        """
        result = {
            'bridge': self.stats.to_dict(),
            'session_active': self._session_active,
            'session_id': self._session_id
        }

        # Add monitor summary if available
        if self._session_active or self.monitor.session:
            result['monitor'] = self.monitor.get_optimizer_compatible_summary()

        # Add optimizer statistics
        result['optimizer'] = self.optimizer.get_statistics()

        return result

    def get_performance_report(self) -> str:
        """
        Generate combined performance report.

        Returns:
            Human-readable performance report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("MONITOR-OPTIMIZER BRIDGE PERFORMANCE REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Bridge statistics
        lines.append("## BRIDGE STATISTICS")
        lines.append(f"  Sessions Monitored:   {self.stats.sessions_monitored}")
        lines.append(f"  Algorithms Executed:  {self.stats.algorithms_executed}")
        lines.append(f"  Cascades Triggered:   {self.stats.cascades_triggered}")
        lines.append(f"  Optimizations Applied: {self.stats.optimizations_applied}")
        lines.append(f"  Sync Operations:      {self.stats.sync_operations}")
        lines.append("")

        # Monitor performance report
        lines.append("## MONITOR PERFORMANCE")
        lines.append("-" * 70)
        lines.append(self.monitor.get_performance_report())
        lines.append("")

        # Optimizer statistics
        lines.append("## OPTIMIZER STATISTICS")
        lines.append("-" * 70)
        opt_stats = self.optimizer.get_statistics()
        for piston, algos in opt_stats.items():
            lines.append(f"\n{piston.upper()}:")
            for algo, stats in algos.items():
                lines.append(f"  {algo}: {stats.get('success_rate', 'N/A')} success, {stats.get('total_attempts', 0)} attempts")
        lines.append("")

        # Recommendations
        lines.append("## CURRENT RECOMMENDATIONS")
        lines.append("-" * 70)
        recs = self.get_recommendations()
        for piston, order in recs.items():
            lines.append(f"  {piston}: {' -> '.join(order[:5])}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_dashboard_data(self) -> Dict:
        """
        Generate data suitable for a monitoring dashboard.

        Returns:
            Dict with all data needed for dashboard visualization
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'bridge_stats': self.stats.to_dict(),
            'session': {
                'active': self._session_active,
                'id': self._session_id,
                'duration': time.time() - self._session_start if self._session_active else 0
            },
            'monitor_summary': self.monitor.get_optimizer_compatible_summary() if self.monitor.session else None,
            'optimizer_stats': self.optimizer.get_monitor_compatible_stats(),
            'recommendations': self.get_recommendations(),
            'bottleneck_analysis': self.monitor.get_bottleneck_analysis() if self.monitor.session else None
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def reset_optimizer(self):
        """Reset optimizer statistics (start fresh learning)."""
        try:
            from .cascade_optimizer import CascadeOptimizer, CascadeOptimizerConfig
        except ImportError:
            from cascade_optimizer import CascadeOptimizer, CascadeOptimizerConfig
        optimizer_config = CascadeOptimizerConfig(
            history_file=os.path.join(self.config.output_dir, 'cascade_history.json'),
            auto_save=self.config.persist_on_sync
        )
        self._optimizer = CascadeOptimizer(optimizer_config)

        if self.config.verbose:
            print("[BRIDGE] Optimizer reset to default state")

    def set_user_preference(self, piston: str, algorithm_order: List[str]):
        """
        Set user preference for algorithm order.

        This overrides learned optimizations for the specified piston.

        Args:
            piston: Piston name
            algorithm_order: Preferred algorithm order
        """
        self.optimizer.set_user_preference(piston, algorithm_order)

        if self.config.verbose:
            print(f"[BRIDGE] User preference set for {piston}: {algorithm_order}")

    def clear_user_preference(self, piston: str):
        """
        Clear user preference, returning to learned ordering.

        Args:
            piston: Piston name
        """
        self.optimizer.clear_user_preference(piston)

        if self.config.verbose:
            print(f"[BRIDGE] User preference cleared for {piston}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_bridge(output_dir: str = './output', verbose: bool = True) -> MonitorOptimizerBridge:
    """
    Create a MonitorOptimizerBridge with default configuration.

    Args:
        output_dir: Directory for reports and history
        verbose: Enable verbose output

    Returns:
        Configured MonitorOptimizerBridge instance
    """
    config = BridgeConfig(output_dir=output_dir, verbose=verbose)
    return MonitorOptimizerBridge(config)


def quick_optimize(
    piston: str,
    parts_db: Dict,
    board_width: float = 50.0,
    board_height: float = 40.0
) -> List[Tuple[str, str]]:
    """
    Quick optimization without session management.

    Creates a temporary bridge to get optimized cascade.

    Args:
        piston: Piston name
        parts_db: Parts database for design profile
        board_width: Board width in mm
        board_height: Board height in mm

    Returns:
        Optimized algorithm cascade
    """
    bridge = create_bridge(verbose=False)
    return bridge.get_optimized_cascade(piston, parts_db, board_width, board_height)
