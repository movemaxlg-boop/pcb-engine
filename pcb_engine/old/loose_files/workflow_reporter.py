"""
PCB Engine - Workflow Reporter
===============================

Generates comprehensive reports after each PCB generation showing:
1. Complete workflow trace (all pistons executed)
2. Algorithms tried vs algorithms used
3. Loop iterations and decisions made
4. Performance metrics and timing
5. DRC watchdog reports
6. AI interactions and decisions

The report provides full transparency into the PCB Engine's decision-making process.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
import time


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AlgorithmStatus(Enum):
    """Status of an algorithm attempt"""
    TRIED = 'tried'           # Algorithm was attempted
    SUCCEEDED = 'succeeded'   # Algorithm completed successfully
    FAILED = 'failed'         # Algorithm failed
    SKIPPED = 'skipped'       # Algorithm was skipped
    FALLBACK = 'fallback'     # Algorithm was used as fallback


class LoopType(Enum):
    """Types of loops in the engine"""
    RETRY = 'retry'           # Retry loop (same piston, different effort)
    ITERATION = 'iteration'   # Iterative improvement loop
    DRC_FIX = 'drc_fix'       # DRC violation fix loop
    OPTIMIZATION = 'optimization'  # Optimization loop
    AI_DECISION = 'ai_decision'   # AI decision loop


@dataclass
class AlgorithmAttempt:
    """Record of an algorithm attempt"""
    piston: str
    algorithm: str
    status: AlgorithmStatus
    start_time: float
    end_time: float
    success: bool
    error_message: str = ''
    metrics: Dict = field(default_factory=dict)
    fallback_from: str = ''  # If this was a fallback, what it replaced


@dataclass
class LoopRecord:
    """Record of a loop execution"""
    loop_type: LoopType
    piston: str
    iteration: int
    max_iterations: int
    reason: str
    start_time: float
    end_time: float
    result: str  # 'continue', 'break', 'success', 'failed'
    data: Dict = field(default_factory=dict)


@dataclass
class PistonExecution:
    """Record of a piston execution"""
    name: str
    start_time: float
    end_time: float = 0.0
    effort_level: str = 'normal'
    success: bool = False

    # Algorithm tracking
    algorithms_available: List[str] = field(default_factory=list)
    algorithms_tried: List[AlgorithmAttempt] = field(default_factory=list)
    algorithm_used: str = ''

    # Loop tracking
    loops: List[LoopRecord] = field(default_factory=list)
    retry_count: int = 0

    # Results
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Metrics
    metrics: Dict = field(default_factory=dict)

    # DRC
    drc_passed: bool = True
    drc_violations: List[Dict] = field(default_factory=list)
    drc_suggestions: List[str] = field(default_factory=list)


@dataclass
class AIInteraction:
    """Record of an AI interaction"""
    timestamp: float
    stage: str
    request_type: str
    question: str
    options: List[str]
    decision: str
    reasoning: str
    parameters: Dict = field(default_factory=dict)


@dataclass
class WorkflowReport:
    """Complete workflow report"""
    # Project info
    project_name: str
    timestamp: str

    # Overall status
    success: bool
    stage_reached: str
    total_time: float

    # Pistons
    pistons_executed: List[PistonExecution] = field(default_factory=list)
    pistons_skipped: List[str] = field(default_factory=list)

    # Algorithms summary
    total_algorithms_tried: int = 0
    total_algorithms_succeeded: int = 0
    algorithms_by_piston: Dict[str, Dict] = field(default_factory=dict)

    # Loops summary
    total_loops: int = 0
    total_iterations: int = 0
    loops_by_type: Dict[str, int] = field(default_factory=dict)

    # AI interactions
    ai_interactions: List[AIInteraction] = field(default_factory=list)

    # Errors and warnings
    all_errors: List[str] = field(default_factory=list)
    all_warnings: List[str] = field(default_factory=list)

    # DRC summary
    drc_runs: int = 0
    drc_passes: int = 0
    drc_total_violations: int = 0

    # Output files
    files_generated: List[str] = field(default_factory=list)


# =============================================================================
# WORKFLOW REPORTER
# =============================================================================

class WorkflowReporter:
    """
    Workflow Reporter - Tracks and reports the complete PCB generation workflow.

    Usage:
        reporter = WorkflowReporter('my_project')

        # Start tracking
        reporter.start_piston('placement')
        reporter.log_algorithm_attempt('placement', 'simulated_annealing', ...)
        reporter.log_loop_iteration('placement', LoopType.ITERATION, 1, 10, ...)
        reporter.end_piston('placement', success=True)

        # Generate report
        report = reporter.generate_report()
        reporter.save_report('./output/workflow_report.md')
    """

    def __init__(self, project_name: str = 'pcb'):
        self.project_name = project_name
        self.start_time = time.time()

        # Tracking data
        self._pistons: Dict[str, PistonExecution] = {}
        self._current_piston: str = None
        self._ai_interactions: List[AIInteraction] = []
        self._skipped_pistons: List[str] = []

        # Overall stats
        self._success = False
        self._stage_reached = 'init'
        self._files_generated: List[str] = []
        self._all_errors: List[str] = []
        self._all_warnings: List[str] = []

    # =========================================================================
    # STANDARD PISTON API
    # =========================================================================

    def report(self, design_data: Dict) -> Dict[str, Any]:
        """
        Standard piston API - generate a report from design data.

        Args:
            design_data: Dictionary with parts_db, placement, routes, etc.

        Returns:
            Dictionary with workflow report
        """
        parts_db = design_data.get('parts_db', {})
        placement = design_data.get('placement', {})

        return {
            'project_name': self.project_name,
            'component_count': len(parts_db.get('parts', {})),
            'net_count': len(parts_db.get('nets', {})),
            'placement_count': len(placement),
            'stage_reached': self._stage_reached,
            'success': self._success,
            'elapsed_time': time.time() - self.start_time,
            'pistons_executed': list(self._pistons.keys()),
            'errors': self._all_errors,
            'warnings': self._all_warnings
        }

    # =========================================================================
    # PISTON TRACKING
    # =========================================================================

    def start_piston(self, piston_name: str, effort_level: str = 'normal',
                     available_algorithms: List[str] = None):
        """Start tracking a piston execution"""
        self._current_piston = piston_name
        self._pistons[piston_name] = PistonExecution(
            name=piston_name,
            start_time=time.time(),
            effort_level=effort_level,
            algorithms_available=available_algorithms or []
        )

    def end_piston(self, piston_name: str, success: bool,
                   algorithm_used: str = '', metrics: Dict = None):
        """End tracking a piston execution"""
        if piston_name in self._pistons:
            piston = self._pistons[piston_name]
            piston.end_time = time.time()
            piston.success = success
            piston.algorithm_used = algorithm_used
            piston.metrics = metrics or {}

            if not success:
                self._all_errors.extend(piston.errors)
            self._all_warnings.extend(piston.warnings)

    def skip_piston(self, piston_name: str, reason: str = ''):
        """Mark a piston as skipped"""
        self._skipped_pistons.append(piston_name)
        if reason:
            self._all_warnings.append(f"Skipped {piston_name}: {reason}")

    def add_piston_note(self, piston_name: str, note: str):
        """Add a note to a piston"""
        if piston_name in self._pistons:
            self._pistons[piston_name].notes.append(note)

    def add_piston_error(self, piston_name: str, error: str):
        """Add an error to a piston"""
        if piston_name in self._pistons:
            self._pistons[piston_name].errors.append(error)

    def add_piston_warning(self, piston_name: str, warning: str):
        """Add a warning to a piston"""
        if piston_name in self._pistons:
            self._pistons[piston_name].warnings.append(warning)

    def set_piston_retry(self, piston_name: str, retry_count: int):
        """Set the retry count for a piston"""
        if piston_name in self._pistons:
            self._pistons[piston_name].retry_count = retry_count

    # =========================================================================
    # ALGORITHM TRACKING
    # =========================================================================

    def log_algorithm_attempt(self, piston_name: str, algorithm: str,
                              success: bool, start_time: float = None,
                              end_time: float = None, error_message: str = '',
                              metrics: Dict = None, fallback_from: str = ''):
        """Log an algorithm attempt"""
        if piston_name not in self._pistons:
            return

        status = AlgorithmStatus.SUCCEEDED if success else AlgorithmStatus.FAILED
        if fallback_from:
            status = AlgorithmStatus.FALLBACK

        attempt = AlgorithmAttempt(
            piston=piston_name,
            algorithm=algorithm,
            status=status,
            start_time=start_time or time.time(),
            end_time=end_time or time.time(),
            success=success,
            error_message=error_message,
            metrics=metrics or {},
            fallback_from=fallback_from
        )

        self._pistons[piston_name].algorithms_tried.append(attempt)

    def log_algorithm_skipped(self, piston_name: str, algorithm: str, reason: str):
        """Log an algorithm that was skipped"""
        if piston_name not in self._pistons:
            return

        attempt = AlgorithmAttempt(
            piston=piston_name,
            algorithm=algorithm,
            status=AlgorithmStatus.SKIPPED,
            start_time=time.time(),
            end_time=time.time(),
            success=False,
            error_message=reason
        )

        self._pistons[piston_name].algorithms_tried.append(attempt)

    # =========================================================================
    # LOOP TRACKING
    # =========================================================================

    def log_loop_iteration(self, piston_name: str, loop_type: LoopType,
                           iteration: int, max_iterations: int,
                           reason: str, result: str,
                           start_time: float = None, end_time: float = None,
                           data: Dict = None):
        """Log a loop iteration"""
        if piston_name not in self._pistons:
            return

        loop_record = LoopRecord(
            loop_type=loop_type,
            piston=piston_name,
            iteration=iteration,
            max_iterations=max_iterations,
            reason=reason,
            start_time=start_time or time.time(),
            end_time=end_time or time.time(),
            result=result,
            data=data or {}
        )

        self._pistons[piston_name].loops.append(loop_record)

    # =========================================================================
    # DRC TRACKING
    # =========================================================================

    def log_drc_result(self, piston_name: str, passed: bool,
                       violations: List[Dict] = None,
                       suggestions: List[str] = None):
        """Log DRC result for a piston"""
        if piston_name not in self._pistons:
            return

        piston = self._pistons[piston_name]
        piston.drc_passed = passed
        piston.drc_violations = violations or []
        piston.drc_suggestions = suggestions or []

    # =========================================================================
    # AI INTERACTION TRACKING
    # =========================================================================

    def log_ai_interaction(self, stage: str, request_type: str,
                           question: str, options: List[str],
                           decision: str, reasoning: str,
                           parameters: Dict = None):
        """Log an AI interaction"""
        interaction = AIInteraction(
            timestamp=time.time(),
            stage=stage,
            request_type=request_type,
            question=question,
            options=options,
            decision=decision,
            reasoning=reasoning,
            parameters=parameters or {}
        )

        self._ai_interactions.append(interaction)

    # =========================================================================
    # OVERALL STATUS
    # =========================================================================

    def set_success(self, success: bool):
        """Set overall success status"""
        self._success = success

    def set_stage_reached(self, stage: str):
        """Set the final stage reached"""
        self._stage_reached = stage

    def add_generated_file(self, filepath: str):
        """Add a generated file to the list"""
        self._files_generated.append(filepath)

    def add_error(self, error: str):
        """Add a general error"""
        self._all_errors.append(error)

    def add_warning(self, warning: str):
        """Add a general warning"""
        self._all_warnings.append(warning)

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_report(self) -> WorkflowReport:
        """Generate the complete workflow report"""
        total_time = time.time() - self.start_time

        # Calculate algorithm stats
        total_algorithms_tried = 0
        total_algorithms_succeeded = 0
        algorithms_by_piston = {}

        for piston_name, piston in self._pistons.items():
            tried = len(piston.algorithms_tried)
            succeeded = sum(1 for a in piston.algorithms_tried if a.success)
            total_algorithms_tried += tried
            total_algorithms_succeeded += succeeded

            algorithms_by_piston[piston_name] = {
                'available': piston.algorithms_available,
                'tried': [a.algorithm for a in piston.algorithms_tried],
                'used': piston.algorithm_used,
                'attempts': [
                    {
                        'algorithm': a.algorithm,
                        'status': a.status.value,
                        'success': a.success,
                        'duration': a.end_time - a.start_time,
                        'error': a.error_message if not a.success else None
                    }
                    for a in piston.algorithms_tried
                ]
            }

        # Calculate loop stats
        total_loops = 0
        total_iterations = 0
        loops_by_type = {}

        for piston in self._pistons.values():
            for loop in piston.loops:
                total_loops += 1
                total_iterations += 1  # Each record is one iteration
                loop_type = loop.loop_type.value
                loops_by_type[loop_type] = loops_by_type.get(loop_type, 0) + 1

        # Calculate DRC stats
        drc_runs = 0
        drc_passes = 0
        drc_total_violations = 0

        for piston in self._pistons.values():
            if piston.drc_violations is not None:
                drc_runs += 1
                if piston.drc_passed:
                    drc_passes += 1
                drc_total_violations += len(piston.drc_violations)

        return WorkflowReport(
            project_name=self.project_name,
            timestamp=datetime.now().isoformat(),
            success=self._success,
            stage_reached=self._stage_reached,
            total_time=total_time,
            pistons_executed=list(self._pistons.values()),
            pistons_skipped=self._skipped_pistons,
            total_algorithms_tried=total_algorithms_tried,
            total_algorithms_succeeded=total_algorithms_succeeded,
            algorithms_by_piston=algorithms_by_piston,
            total_loops=total_loops,
            total_iterations=total_iterations,
            loops_by_type=loops_by_type,
            ai_interactions=self._ai_interactions,
            all_errors=self._all_errors,
            all_warnings=self._all_warnings,
            drc_runs=drc_runs,
            drc_passes=drc_passes,
            drc_total_violations=drc_total_violations,
            files_generated=self._files_generated
        )

    def save_report(self, output_dir: str) -> str:
        """Save the workflow report to files (markdown and JSON)"""
        os.makedirs(output_dir, exist_ok=True)

        report = self.generate_report()

        # Save as Markdown
        md_path = os.path.join(output_dir, 'workflow_report.md')
        md_content = self._generate_markdown_report(report)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # Save as JSON (for programmatic access)
        json_path = os.path.join(output_dir, 'workflow_report.json')
        json_content = self._generate_json_report(report)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)

        return md_path

    def _generate_markdown_report(self, report: WorkflowReport) -> str:
        """Generate markdown formatted report"""
        lines = []

        # Header
        lines.append(f"# PCB Engine Workflow Report")
        lines.append(f"")
        lines.append(f"**Project:** {report.project_name}")
        lines.append(f"**Generated:** {report.timestamp}")
        lines.append(f"**Status:** {'SUCCESS' if report.success else 'FAILED'}")
        lines.append(f"**Stage Reached:** {report.stage_reached}")
        lines.append(f"**Total Time:** {report.total_time:.2f}s")
        lines.append(f"")

        # Summary
        lines.append(f"## Summary")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Pistons Executed | {len(report.pistons_executed)} |")
        lines.append(f"| Pistons Skipped | {len(report.pistons_skipped)} |")
        lines.append(f"| Algorithms Tried | {report.total_algorithms_tried} |")
        lines.append(f"| Algorithms Succeeded | {report.total_algorithms_succeeded} |")
        lines.append(f"| Total Loops | {report.total_loops} |")
        lines.append(f"| DRC Runs | {report.drc_runs} |")
        lines.append(f"| DRC Passes | {report.drc_passes} |")
        lines.append(f"| Total DRC Violations | {report.drc_total_violations} |")
        lines.append(f"| AI Interactions | {len(report.ai_interactions)} |")
        lines.append(f"| Errors | {len(report.all_errors)} |")
        lines.append(f"| Warnings | {len(report.all_warnings)} |")
        lines.append(f"| Files Generated | {len(report.files_generated)} |")
        lines.append(f"")

        # Piston Execution Flow
        lines.append(f"## Piston Execution Flow")
        lines.append(f"")
        lines.append(f"```")
        lines.append(f"START")
        for piston in report.pistons_executed:
            status = "OK" if piston.success else "FAIL"
            duration = piston.end_time - piston.start_time if piston.end_time else 0
            lines.append(f"  |")
            lines.append(f"  +--[{piston.name.upper()}] ({piston.effort_level}) - {status} ({duration:.2f}s)")
            if piston.algorithm_used:
                lines.append(f"  |     Algorithm: {piston.algorithm_used}")
            if piston.retry_count > 0:
                lines.append(f"  |     Retries: {piston.retry_count}")
        if report.pistons_skipped:
            lines.append(f"  |")
            lines.append(f"  +--[SKIPPED: {', '.join(report.pistons_skipped)}]")
        lines.append(f"  |")
        lines.append(f"{'SUCCESS' if report.success else 'FAILED'}")
        lines.append(f"```")
        lines.append(f"")

        # Detailed Piston Reports
        lines.append(f"## Detailed Piston Reports")
        lines.append(f"")

        for piston in report.pistons_executed:
            duration = piston.end_time - piston.start_time if piston.end_time else 0
            lines.append(f"### {piston.name.upper()}")
            lines.append(f"")
            lines.append(f"- **Status:** {'Success' if piston.success else 'Failed'}")
            lines.append(f"- **Effort Level:** {piston.effort_level}")
            lines.append(f"- **Duration:** {duration:.2f}s")
            lines.append(f"- **Retries:** {piston.retry_count}")
            lines.append(f"")

            # Algorithms
            if piston.algorithms_available or piston.algorithms_tried:
                lines.append(f"#### Algorithms")
                lines.append(f"")
                if piston.algorithms_available:
                    lines.append(f"**Available:** {', '.join(piston.algorithms_available)}")
                if piston.algorithms_tried:
                    lines.append(f"")
                    lines.append(f"| Algorithm | Status | Duration | Notes |")
                    lines.append(f"|-----------|--------|----------|-------|")
                    for attempt in piston.algorithms_tried:
                        duration_a = attempt.end_time - attempt.start_time
                        notes = attempt.error_message or '-'
                        if attempt.fallback_from:
                            notes = f"Fallback from {attempt.fallback_from}"
                        lines.append(f"| {attempt.algorithm} | {attempt.status.value} | {duration_a:.3f}s | {notes} |")
                if piston.algorithm_used:
                    lines.append(f"")
                    lines.append(f"**Used:** {piston.algorithm_used}")
                lines.append(f"")

            # Loops
            if piston.loops:
                lines.append(f"#### Loops")
                lines.append(f"")
                lines.append(f"| Type | Iteration | Max | Reason | Result |")
                lines.append(f"|------|-----------|-----|--------|--------|")
                for loop in piston.loops:
                    lines.append(f"| {loop.loop_type.value} | {loop.iteration} | {loop.max_iterations} | {loop.reason} | {loop.result} |")
                lines.append(f"")

            # DRC
            if piston.drc_violations:
                lines.append(f"#### DRC Results")
                lines.append(f"")
                lines.append(f"- **Passed:** {'Yes' if piston.drc_passed else 'No'}")
                lines.append(f"- **Violations:** {len(piston.drc_violations)}")
                if piston.drc_violations:
                    lines.append(f"")
                    for i, v in enumerate(piston.drc_violations[:5], 1):  # Show first 5
                        lines.append(f"  {i}. {v.get('type', 'Unknown')}: {v.get('message', 'No message')}")
                    if len(piston.drc_violations) > 5:
                        lines.append(f"  ... and {len(piston.drc_violations) - 5} more")
                if piston.drc_suggestions:
                    lines.append(f"")
                    lines.append(f"**Suggestions:**")
                    for s in piston.drc_suggestions:
                        lines.append(f"  - {s}")
                lines.append(f"")

            # Notes
            if piston.notes:
                lines.append(f"#### Notes")
                lines.append(f"")
                for note in piston.notes:
                    lines.append(f"- {note}")
                lines.append(f"")

            # Errors & Warnings
            if piston.errors or piston.warnings:
                if piston.errors:
                    lines.append(f"#### Errors")
                    lines.append(f"")
                    for error in piston.errors:
                        lines.append(f"- {error}")
                    lines.append(f"")
                if piston.warnings:
                    lines.append(f"#### Warnings")
                    lines.append(f"")
                    for warning in piston.warnings:
                        lines.append(f"- {warning}")
                    lines.append(f"")

            # Metrics
            if piston.metrics:
                lines.append(f"#### Metrics")
                lines.append(f"")
                for key, value in piston.metrics.items():
                    lines.append(f"- **{key}:** {value}")
                lines.append(f"")

        # AI Interactions
        if report.ai_interactions:
            lines.append(f"## AI Interactions")
            lines.append(f"")
            for i, interaction in enumerate(report.ai_interactions, 1):
                lines.append(f"### Interaction {i}")
                lines.append(f"")
                lines.append(f"- **Stage:** {interaction.stage}")
                lines.append(f"- **Type:** {interaction.request_type}")
                lines.append(f"- **Question:** {interaction.question}")
                lines.append(f"- **Options:** {', '.join(interaction.options)}")
                lines.append(f"- **Decision:** {interaction.decision}")
                lines.append(f"- **Reasoning:** {interaction.reasoning}")
                if interaction.parameters:
                    lines.append(f"- **Parameters:** {json.dumps(interaction.parameters)}")
                lines.append(f"")

        # Loop Summary
        if report.loops_by_type:
            lines.append(f"## Loop Summary")
            lines.append(f"")
            lines.append(f"| Loop Type | Count |")
            lines.append(f"|-----------|-------|")
            for loop_type, count in report.loops_by_type.items():
                lines.append(f"| {loop_type} | {count} |")
            lines.append(f"")

        # Files Generated
        if report.files_generated:
            lines.append(f"## Files Generated")
            lines.append(f"")
            for filepath in report.files_generated:
                lines.append(f"- {filepath}")
            lines.append(f"")

        # Errors and Warnings
        if report.all_errors:
            lines.append(f"## All Errors")
            lines.append(f"")
            for error in report.all_errors:
                lines.append(f"- {error}")
            lines.append(f"")

        if report.all_warnings:
            lines.append(f"## All Warnings")
            lines.append(f"")
            for warning in report.all_warnings:
                lines.append(f"- {warning}")
            lines.append(f"")

        # Footer
        lines.append(f"---")
        lines.append(f"*Report generated by PCB Engine Workflow Reporter*")

        return '\n'.join(lines)

    def _generate_json_report(self, report: WorkflowReport) -> str:
        """Generate JSON formatted report"""
        data = {
            'project_name': report.project_name,
            'timestamp': report.timestamp,
            'success': report.success,
            'stage_reached': report.stage_reached,
            'total_time': report.total_time,
            'summary': {
                'pistons_executed': len(report.pistons_executed),
                'pistons_skipped': len(report.pistons_skipped),
                'algorithms_tried': report.total_algorithms_tried,
                'algorithms_succeeded': report.total_algorithms_succeeded,
                'total_loops': report.total_loops,
                'total_iterations': report.total_iterations,
                'drc_runs': report.drc_runs,
                'drc_passes': report.drc_passes,
                'drc_total_violations': report.drc_total_violations,
                'ai_interactions': len(report.ai_interactions),
                'errors': len(report.all_errors),
                'warnings': len(report.all_warnings),
                'files_generated': len(report.files_generated)
            },
            'pistons': [
                {
                    'name': p.name,
                    'success': p.success,
                    'effort_level': p.effort_level,
                    'duration': p.end_time - p.start_time if p.end_time else 0,
                    'retry_count': p.retry_count,
                    'algorithms_available': p.algorithms_available,
                    'algorithms_tried': [
                        {
                            'algorithm': a.algorithm,
                            'status': a.status.value,
                            'success': a.success,
                            'duration': a.end_time - a.start_time,
                            'error': a.error_message
                        }
                        for a in p.algorithms_tried
                    ],
                    'algorithm_used': p.algorithm_used,
                    'loops': [
                        {
                            'type': l.loop_type.value,
                            'iteration': l.iteration,
                            'max_iterations': l.max_iterations,
                            'reason': l.reason,
                            'result': l.result,
                            'duration': l.end_time - l.start_time
                        }
                        for l in p.loops
                    ],
                    'drc_passed': p.drc_passed,
                    'drc_violations': p.drc_violations,
                    'metrics': p.metrics,
                    'errors': p.errors,
                    'warnings': p.warnings,
                    'notes': p.notes
                }
                for p in report.pistons_executed
            ],
            'pistons_skipped': report.pistons_skipped,
            'algorithms_by_piston': report.algorithms_by_piston,
            'loops_by_type': report.loops_by_type,
            'ai_interactions': [
                {
                    'timestamp': i.timestamp,
                    'stage': i.stage,
                    'request_type': i.request_type,
                    'question': i.question,
                    'options': i.options,
                    'decision': i.decision,
                    'reasoning': i.reasoning,
                    'parameters': i.parameters
                }
                for i in report.ai_interactions
            ],
            'all_errors': report.all_errors,
            'all_warnings': report.all_warnings,
            'files_generated': report.files_generated
        }

        return json.dumps(data, indent=2)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_reporter(project_name: str) -> WorkflowReporter:
    """Create a new workflow reporter"""
    return WorkflowReporter(project_name)


# Export for use in other modules
__all__ = [
    'WorkflowReporter',
    'WorkflowReport',
    'PistonExecution',
    'AlgorithmAttempt',
    'AlgorithmStatus',
    'LoopRecord',
    'LoopType',
    'AIInteraction',
    'create_reporter'
]
