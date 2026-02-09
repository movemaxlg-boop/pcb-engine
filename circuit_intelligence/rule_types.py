"""
Rule Types for Circuit Intelligence Engine
==========================================

Defines the core types for the Rules API system:
- RuleStatus: Status of a rule check (PASS, FAIL, WARNING, etc.)
- RuleCategory: Category of rules (ELECTRICAL, PLACEMENT, etc.)
- RuleReport: Machine-readable report for AI agent review
- ValidationResult: Simple validation result structure
- FeedbackResult: Result of AI feedback command

These types enable:
1. Structured rule execution with typed outputs
2. AI-readable reports for external agent review
3. Feedback commands (ACCEPT, REJECT, CORRECT, OVERRIDE)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RuleStatus(Enum):
    """Status of a rule check."""
    PASS = "PASS"               # Rule satisfied
    FAIL = "FAIL"               # Rule violated
    WARNING = "WARNING"         # Close to limit or needs attention
    SKIPPED = "SKIPPED"         # Rule not applicable to this design
    PENDING = "PENDING"         # Awaiting AI review
    ACCEPTED = "ACCEPTED"       # AI validated the outcome
    REJECTED = "REJECTED"       # AI rejected the outcome
    CORRECTED = "CORRECTED"     # AI provided a correction


class RuleCategory(Enum):
    """Category of design rules."""
    ELECTRICAL = "electrical"           # Conductor spacing, current capacity
    IMPEDANCE = "impedance"             # Z0 calculations
    PLACEMENT = "placement"             # Component placement rules
    ROUTING = "routing"                 # Trace routing rules
    HIGH_SPEED = "high_speed"           # DDR, PCIe, USB, HDMI, Ethernet
    THERMAL = "thermal"                 # Junction temp, thermal vias
    EMI = "emi"                         # EMI/EMC compliance
    FABRICATION = "fabrication"         # Manufacturing limits
    STACKUP = "stackup"                 # Layer stackup rules
    BGA_HDI = "bga_hdi"                 # BGA escape, HDI design
    ASSEMBLY = "assembly"               # Component spacing, test points


class RuleSeverity(Enum):
    """Severity level of a rule violation."""
    CRITICAL = "critical"       # Must fix - design will not work
    ERROR = "error"             # Should fix - may cause issues
    WARNING = "warning"         # Review recommended
    INFO = "info"               # Informational only


# =============================================================================
# VALIDATION RESULT (Simple)
# =============================================================================

@dataclass
class ValidationResult:
    """
    Simple validation result for a single check.
    Use RuleReport for full AI-readable reports.
    """
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "recommendations": self.recommendations
        }


# =============================================================================
# RULE REPORT (Full AI-Readable Report)
# =============================================================================

@dataclass
class RuleReport:
    """
    Machine-readable report for AI agent review.

    This report is produced by every rule execution and contains:
    - Rule identification (ID, category, source)
    - Input context (what was checked)
    - Rule application details (threshold, actual value)
    - Outcome (status, violations, warnings)
    - AI feedback interface (commands to accept/reject/correct)
    """

    # IDENTIFICATION
    rule_id: str                                # e.g., "USB2_LENGTH_MATCHING"
    rule_category: RuleCategory                 # e.g., RuleCategory.HIGH_SPEED
    rule_source: str                            # e.g., "USB 2.0 Spec Chapter 7"
    rule_description: str = ""                  # Human-readable description

    # INPUT CONTEXT
    inputs: Dict[str, Any] = field(default_factory=dict)  # e.g., {"d_plus_mm": 45.0}

    # RULE APPLICATION
    rule_applied: str = ""                      # e.g., "abs(D+ - D-) <= 1.25"
    threshold: Any = None                       # e.g., 1.25
    actual_value: Any = None                    # e.g., 1.5

    # OUTCOME
    status: RuleStatus = RuleStatus.PENDING     # PASS | FAIL | WARNING | etc.
    severity: RuleSeverity = RuleSeverity.ERROR
    passed: bool = False

    # DETAILS FOR AI REVIEW
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    # AI FEEDBACK INTERFACE
    confidence: float = 1.0                     # How confident is this assessment
    alternatives: List[str] = field(default_factory=list)  # Possible fixes

    # TIMESTAMP
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_feedback_commands(self) -> Dict[str, str]:
        """Generate feedback commands for AI agent."""
        return {
            "accept": f"ACCEPT {self.rule_id}",
            "reject": f"REJECT {self.rule_id} reason=\"<your_reason>\"",
            "correct": f"CORRECT {self.rule_id} action=\"<action>\" value=<value>",
            "override": f"OVERRIDE {self.rule_id} new_threshold=<value> reason=\"<reason>\"",
            "explain": f"EXPLAIN {self.rule_id}",
            "query": f"QUERY {self.rule_id} question=\"<your_question>\""
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_category": self.rule_category.value,
            "rule_source": self.rule_source,
            "rule_description": self.rule_description,
            "inputs": self.inputs,
            "rule_applied": self.rule_applied,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "status": self.status.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "violations": self.violations,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "feedback_commands": self.get_feedback_commands(),
            "timestamp": self.timestamp
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string for AI consumption."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        status_icon = {
            RuleStatus.PASS: "[PASS]",
            RuleStatus.FAIL: "[FAIL]",
            RuleStatus.WARNING: "[WARN]",
            RuleStatus.SKIPPED: "[SKIP]",
            RuleStatus.PENDING: "[PEND]",
            RuleStatus.ACCEPTED: "[OK]",
            RuleStatus.REJECTED: "[REJ]",
            RuleStatus.CORRECTED: "[CORR]"
        }
        icon = status_icon.get(self.status, "[???]")
        violation_text = f" - {self.violations[0]}" if self.violations else ""
        return f"{icon} {self.rule_id}: {self.rule_applied}{violation_text}"


# =============================================================================
# FEEDBACK RESULT
# =============================================================================

@dataclass
class FeedbackResult:
    """
    Result of an AI feedback command.

    Returned when AI sends ACCEPT, REJECT, CORRECT, or OVERRIDE commands.
    """
    command: str                                # Original command
    rule_id: str                                # Affected rule
    action_taken: str                           # What was done
    success: bool = True                        # Command succeeded?
    new_status: RuleStatus = RuleStatus.PENDING
    applied_correction: Optional[Dict] = None   # If corrected
    requires_revalidation: bool = False         # Need to re-run rules?
    next_steps: List[str] = field(default_factory=list)
    error_message: str = ""                     # If command failed

    def to_dict(self) -> Dict:
        return {
            "command": self.command,
            "rule_id": self.rule_id,
            "action_taken": self.action_taken,
            "success": self.success,
            "new_status": self.new_status.value,
            "applied_correction": self.applied_correction,
            "requires_revalidation": self.requires_revalidation,
            "next_steps": self.next_steps,
            "error_message": self.error_message
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# DESIGN REVIEW REPORT (Batch Report)
# =============================================================================

@dataclass
class DesignReviewReport:
    """
    Complete design review report for AI agent.

    Contains results of all rule checks organized by category,
    with summary statistics and batch feedback commands.
    """

    # IDENTIFICATION
    design_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # SUMMARY COUNTS
    total_rules_checked: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    skipped: int = 0

    # OVERALL STATUS
    design_status: str = "PENDING"  # "PASS" | "FAIL" | "NEEDS_REVIEW"
    compliance_score: float = 0.0   # 0.0 to 1.0
    critical_issues: int = 0

    # BLOCKING VIOLATIONS
    blocking_violations: List[str] = field(default_factory=list)

    # CATEGORIZED RESULTS
    reports_by_category: Dict[str, List[RuleReport]] = field(default_factory=dict)

    # ALL REPORTS (flat list)
    all_reports: List[RuleReport] = field(default_factory=list)

    def add_report(self, report: RuleReport):
        """Add a rule report to the design review."""
        self.all_reports.append(report)

        # Update counts
        self.total_rules_checked += 1
        if report.status == RuleStatus.PASS:
            self.passed += 1
        elif report.status == RuleStatus.FAIL:
            self.failed += 1
            if report.severity == RuleSeverity.CRITICAL:
                self.critical_issues += 1
                for v in report.violations:
                    self.blocking_violations.append(f"[{report.rule_id}] {v}")
        elif report.status == RuleStatus.WARNING:
            self.warnings += 1
        elif report.status == RuleStatus.SKIPPED:
            self.skipped += 1

        # Add to category
        category = report.rule_category.value
        if category not in self.reports_by_category:
            self.reports_by_category[category] = []
        self.reports_by_category[category].append(report)

    def finalize(self):
        """Calculate final statistics after all reports are added."""
        if self.total_rules_checked > 0:
            self.compliance_score = self.passed / self.total_rules_checked

        if self.failed == 0 and self.warnings == 0:
            self.design_status = "PASS"
        elif self.critical_issues > 0:
            self.design_status = "FAIL"
        else:
            self.design_status = "NEEDS_REVIEW"

    def get_failed_reports(self) -> List[RuleReport]:
        """Get all failed rule reports."""
        return [r for r in self.all_reports if r.status == RuleStatus.FAIL]

    def get_warning_reports(self) -> List[RuleReport]:
        """Get all warning rule reports."""
        return [r for r in self.all_reports if r.status == RuleStatus.WARNING]

    def get_pending_review(self) -> List[RuleReport]:
        """Get reports needing AI review (failed + warnings)."""
        return [r for r in self.all_reports
                if r.status in (RuleStatus.FAIL, RuleStatus.WARNING, RuleStatus.PENDING)]

    def get_batch_commands(self) -> Dict[str, str]:
        """Get batch feedback commands for AI."""
        return {
            "accept_all_passed": "BATCH_ACCEPT status=PASS",
            "review_failures": "BATCH_REVIEW status=FAIL",
            "review_warnings": "BATCH_REVIEW status=WARNING",
            "accept_all": "BATCH_ACCEPT status=ALL",
            "export_json": "EXPORT format=json",
            "export_summary": "EXPORT format=summary"
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "design_name": self.design_name,
            "timestamp": self.timestamp,
            "summary": {
                "total_rules_checked": self.total_rules_checked,
                "passed": self.passed,
                "failed": self.failed,
                "warnings": self.warnings,
                "skipped": self.skipped
            },
            "design_status": self.design_status,
            "compliance_score": round(self.compliance_score, 3),
            "critical_issues": self.critical_issues,
            "blocking_violations": self.blocking_violations,
            "batch_commands": self.get_batch_commands(),
            "reports_by_category": {
                cat: [r.to_dict() for r in reports]
                for cat, reports in self.reports_by_category.items()
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON for AI consumption."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 70,
            f"DESIGN REVIEW REPORT: {self.design_name}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 7,
            f"Total Rules Checked: {self.total_rules_checked}",
            f"  PASSED:  {self.passed:4d} ({self.passed/max(self.total_rules_checked,1)*100:.1f}%)",
            f"  FAILED:  {self.failed:4d} ({self.failed/max(self.total_rules_checked,1)*100:.1f}%)",
            f"  WARNING: {self.warnings:4d} ({self.warnings/max(self.total_rules_checked,1)*100:.1f}%)",
            f"  SKIPPED: {self.skipped:4d} ({self.skipped/max(self.total_rules_checked,1)*100:.1f}%)",
            "",
            f"DESIGN STATUS: {self.design_status}",
            f"Compliance Score: {self.compliance_score:.1%}",
            "",
        ]

        if self.blocking_violations:
            lines.append("BLOCKING VIOLATIONS (must fix):")
            for i, v in enumerate(self.blocking_violations[:10], 1):
                lines.append(f"  {i}. {v}")
            if len(self.blocking_violations) > 10:
                lines.append(f"  ... and {len(self.blocking_violations)-10} more")
            lines.append("")

        warning_reports = self.get_warning_reports()
        if warning_reports:
            lines.append("WARNINGS (review recommended):")
            for i, r in enumerate(warning_reports[:5], 1):
                lines.append(f"  {i}. [{r.rule_id}] {r.warnings[0] if r.warnings else 'Check recommended'}")
            if len(warning_reports) > 5:
                lines.append(f"  ... and {len(warning_reports)-5} more")
            lines.append("")

        lines.extend([
            "AI REVIEW COMMANDS:",
            "  - Review failures:  BATCH_REVIEW status=FAIL",
            "  - Accept all passed: BATCH_ACCEPT status=PASS",
            "  - Show details: EXPLAIN <rule_id>",
            "  - Override a rule: OVERRIDE <rule_id> reason=\"...\"",
            "",
            "=" * 70
        ])

        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_pass_report(
    rule_id: str,
    category: RuleCategory,
    source: str,
    inputs: Dict[str, Any],
    rule_applied: str,
    threshold: Any,
    actual_value: Any,
    metrics: Dict[str, float] = None
) -> RuleReport:
    """Create a PASS rule report."""
    return RuleReport(
        rule_id=rule_id,
        rule_category=category,
        rule_source=source,
        inputs=inputs,
        rule_applied=rule_applied,
        threshold=threshold,
        actual_value=actual_value,
        status=RuleStatus.PASS,
        severity=RuleSeverity.INFO,
        passed=True,
        metrics=metrics or {}
    )


def create_fail_report(
    rule_id: str,
    category: RuleCategory,
    source: str,
    inputs: Dict[str, Any],
    rule_applied: str,
    threshold: Any,
    actual_value: Any,
    violation: str,
    severity: RuleSeverity = RuleSeverity.ERROR,
    alternatives: List[str] = None,
    metrics: Dict[str, float] = None
) -> RuleReport:
    """Create a FAIL rule report."""
    return RuleReport(
        rule_id=rule_id,
        rule_category=category,
        rule_source=source,
        inputs=inputs,
        rule_applied=rule_applied,
        threshold=threshold,
        actual_value=actual_value,
        status=RuleStatus.FAIL,
        severity=severity,
        passed=False,
        violations=[violation],
        alternatives=alternatives or [],
        metrics=metrics or {}
    )


def create_warning_report(
    rule_id: str,
    category: RuleCategory,
    source: str,
    inputs: Dict[str, Any],
    rule_applied: str,
    threshold: Any,
    actual_value: Any,
    warning: str,
    metrics: Dict[str, float] = None
) -> RuleReport:
    """Create a WARNING rule report."""
    return RuleReport(
        rule_id=rule_id,
        rule_category=category,
        rule_source=source,
        inputs=inputs,
        rule_applied=rule_applied,
        threshold=threshold,
        actual_value=actual_value,
        status=RuleStatus.WARNING,
        severity=RuleSeverity.WARNING,
        passed=True,  # Warnings still pass
        warnings=[warning],
        metrics=metrics or {}
    )
