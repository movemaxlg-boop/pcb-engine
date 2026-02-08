#!/usr/bin/env python3
"""
KiCadDRCTeacher - The Master Teacher for Internal DRC

KiCad DRC is the AUTHORITY. Our internal DRC is the STUDENT.

The Teacher:
1. WATCHES - Runs KiCad DRC on every generated PCB
2. LEARNS - Records what KiCad finds that we missed
3. TEACHES - Updates learning database with new violation patterns
4. TAKES OVER - KiCad DRC is the final word on PASS/FAIL

The Big Beautiful Loop:
┌─────────────────────────────────────────────────────────────┐
│  PCB Engine generates design                                │
│       ↓                                                     │
│  Internal DRC checks (fast, immediate feedback)             │
│       ↓                                                     │
│  Output Piston generates .kicad_pcb                         │
│       ↓                                                     │
│  ╔═══════════════════════════════════════════════════════╗  │
│  ║  KiCad DRC Teacher (THE AUTHORITY)                    ║  │
│  ║  • Run kicad-cli pcb drc                              ║  │
│  ║  • Parse results                                      ║  │
│  ║  • Compare with internal DRC                          ║  │
│  ║  • LEARN from discrepancies                           ║  │
│  ║  • Final PASS/FAIL verdict                            ║  │
│  ╚═══════════════════════════════════════════════════════╝  │
│       ↓                                                     │
│  If FAIL: Feed violations back to engine for retry          │
│  If PASS: Celebrate! 🎉                                     │
└─────────────────────────────────────────────────────────────┘
"""

import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from enum import Enum
from pathlib import Path


# KiCad DRC violation types (from KiCad 9)
class KiCadViolationType(Enum):
    """KiCad DRC violation types - THE STANDARD"""
    TRACKS_CROSSING = "tracks_crossing"
    CLEARANCE = "clearance"
    TRACK_WIDTH = "track_width"
    ANNULAR_WIDTH = "annular_width"
    DRILL_SIZE = "drill_size"
    HOLE_CLEARANCE = "hole_clearance"
    SILK_OVER_COPPER = "silk_over_copper"
    SILK_OVERLAP = "silk_overlap"
    SOLDER_MASK_BRIDGE = "solder_mask_bridge"
    COPPER_EDGE_CLEARANCE = "copper_edge_clearance"
    UNCONNECTED_ITEMS = "unconnected_items"
    DUPLICATE_FOOTPRINTS = "duplicate_footprints"
    MISSING_FOOTPRINT = "missing_footprint"
    COURTYARDS_OVERLAP = "courtyards_overlap"
    SHORT_CIRCUIT = "short_circuit"
    STARVED_THERMAL = "starved_thermal"
    VIA_DANGLING = "via_dangling"
    FOOTPRINT_TYPE_MISMATCH = "footprint_type_mismatch"


@dataclass
class KiCadViolation:
    """A single violation from KiCad DRC"""
    type: str
    severity: str  # "error" or "warning"
    description: str
    items: List[Dict]
    position: Optional[Tuple[float, float]] = None


@dataclass
class DRCComparison:
    """Comparison between internal DRC and KiCad DRC"""
    # What KiCad found
    kicad_errors: List[KiCadViolation]
    kicad_warnings: List[KiCadViolation]

    # What internal DRC found
    internal_errors: int
    internal_warnings: int

    # Discrepancies - LEARNING OPPORTUNITIES
    missed_by_internal: List[KiCadViolation]  # KiCad found, we didn't
    false_positives: List[str]  # We found, KiCad didn't (maybe)

    # The final verdict (KiCad's word is LAW)
    kicad_passed: bool


@dataclass
class LearningRecord:
    """Record of what we learned from KiCad"""
    timestamp: str
    pcb_name: str
    violation_type: str
    description: str
    position: Optional[Tuple[float, float]]
    context: Dict  # Relevant design context
    learned: bool = False  # Did we add this to internal DRC?


@dataclass
class TeacherResult:
    """Result from the KiCad DRC Teacher"""
    passed: bool  # THE FINAL VERDICT
    kicad_errors: int
    kicad_warnings: int
    comparison: Optional[DRCComparison]
    learning_records: List[LearningRecord]
    report_path: Optional[str]
    message: str


class KiCadDRCTeacher:
    """
    The Master Teacher - KiCad DRC takes over as the authority.

    Our internal DRC is fast but imperfect.
    KiCad DRC is slower but THE TRUTH.

    This class:
    1. Runs KiCad DRC on generated PCB files
    2. Compares results with internal DRC
    3. Records what we missed (learning database)
    4. Provides the FINAL verdict on PASS/FAIL
    """

    # Path to KiCad CLI
    KICAD_CLI = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"

    # Learning database location
    LEARNING_DB_PATH = None  # Set in __init__

    def __init__(self, learning_db_path: Optional[str] = None):
        """
        Initialize the KiCad DRC Teacher.

        Args:
            learning_db_path: Path to store learning records (JSON)
        """
        if learning_db_path:
            self.LEARNING_DB_PATH = learning_db_path
        else:
            # Default: next to pcb_engine
            engine_dir = os.path.dirname(os.path.abspath(__file__))
            self.LEARNING_DB_PATH = os.path.join(engine_dir, 'drc_learning_db.json')

        self.learning_records: List[LearningRecord] = []
        self._load_learning_db()

    def teach(self,
              pcb_path: str,
              internal_drc_errors: int = 0,
              internal_drc_warnings: int = 0,
              internal_violations: Optional[List] = None) -> TeacherResult:
        """
        THE MAIN TEACHING METHOD.

        Runs KiCad DRC, compares with internal DRC, learns from differences.

        Args:
            pcb_path: Path to the .kicad_pcb file
            internal_drc_errors: Number of errors from internal DRC
            internal_drc_warnings: Number of warnings from internal DRC
            internal_violations: List of internal DRC violations (for comparison)

        Returns:
            TeacherResult with KiCad's verdict and learning records
        """
        if not os.path.exists(pcb_path):
            return TeacherResult(
                passed=False,
                kicad_errors=0,
                kicad_warnings=0,
                comparison=None,
                learning_records=[],
                report_path=None,
                message=f"PCB file not found: {pcb_path}"
            )

        # Step 1: Run KiCad DRC (THE AUTHORITY)
        kicad_result = self._run_kicad_drc(pcb_path)

        if kicad_result is None:
            return TeacherResult(
                passed=False,
                kicad_errors=0,
                kicad_warnings=0,
                comparison=None,
                learning_records=[],
                report_path=None,
                message="Failed to run KiCad DRC"
            )

        violations, report_path = kicad_result

        # Step 2: Parse and categorize violations
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]

        # Step 3: Compare with internal DRC (LEARNING)
        comparison = self._compare_with_internal(
            errors, warnings,
            internal_drc_errors, internal_drc_warnings,
            internal_violations
        )

        # Step 4: Record learning opportunities
        new_learnings = self._record_learnings(
            pcb_path, comparison.missed_by_internal
        )

        # Step 5: Save learning database
        self._save_learning_db()

        # THE VERDICT - KiCad's word is LAW
        passed = len(errors) == 0

        # Build message
        pcb_name = os.path.basename(pcb_path)
        if passed:
            message = f"[PASS] KiCad DRC PASSED: {pcb_name}"
        else:
            message = f"[FAIL] KiCad DRC FAILED: {pcb_name} ({len(errors)} errors)"

        if comparison.missed_by_internal:
            message += f"\n   [LEARNING] {len(comparison.missed_by_internal)} violations we missed"

        return TeacherResult(
            passed=passed,
            kicad_errors=len(errors),
            kicad_warnings=len(warnings),
            comparison=comparison,
            learning_records=new_learnings,
            report_path=report_path,
            message=message
        )

    def _run_kicad_drc(self, pcb_path: str) -> Optional[Tuple[List[KiCadViolation], str]]:
        """
        Run KiCad CLI DRC command.

        Returns:
            Tuple of (violations list, report path) or None on failure
        """
        if not os.path.exists(self.KICAD_CLI):
            print(f"   ERROR: KiCad CLI not found at {self.KICAD_CLI}")
            return None

        # Output report next to PCB file
        pcb_dir = os.path.dirname(pcb_path)
        pcb_name = os.path.splitext(os.path.basename(pcb_path))[0]
        report_path = os.path.join(pcb_dir, f'{pcb_name}_drc.json')

        try:
            result = subprocess.run(
                [
                    self.KICAD_CLI, 'pcb', 'drc',
                    '--format', 'json',
                    '--severity-all',
                    '--output', report_path,
                    pcb_path
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            if not os.path.exists(report_path):
                print(f"   ERROR: KiCad DRC report not generated")
                return None

            # Parse the report
            with open(report_path, 'r') as f:
                report = json.load(f)

            violations = self._parse_kicad_report(report)
            return (violations, report_path)

        except subprocess.TimeoutExpired:
            print(f"   ERROR: KiCad DRC timed out")
            return None
        except Exception as e:
            print(f"   ERROR: KiCad DRC failed: {e}")
            return None

    def _parse_kicad_report(self, report: Dict) -> List[KiCadViolation]:
        """Parse KiCad DRC JSON report into violation objects"""
        violations = []

        for v in report.get('violations', []):
            violation = KiCadViolation(
                type=v.get('type', 'unknown'),
                severity=v.get('severity', 'error'),
                description=v.get('description', ''),
                items=v.get('items', []),
                position=self._extract_position(v.get('items', []))
            )
            violations.append(violation)

        # Add unconnected items as violations
        for item in report.get('unconnected_items', []):
            violation = KiCadViolation(
                type='unconnected_items',
                severity='error',
                description=f"Unconnected: {item.get('description', '')}",
                items=[item],
                position=self._extract_position([item])
            )
            violations.append(violation)

        return violations

    def _extract_position(self, items: List[Dict]) -> Optional[Tuple[float, float]]:
        """Extract position from violation items"""
        for item in items:
            pos = item.get('pos', {})
            if 'x' in pos and 'y' in pos:
                return (pos['x'], pos['y'])
        return None

    def _compare_with_internal(self,
                               kicad_errors: List[KiCadViolation],
                               kicad_warnings: List[KiCadViolation],
                               internal_errors: int,
                               internal_warnings: int,
                               internal_violations: Optional[List]) -> DRCComparison:
        """
        Compare KiCad results with internal DRC.

        This is where we LEARN what we're missing.

        PHILOSOPHY: KiCad is THE TEACHER. Every error KiCad finds is a
        learning opportunity for our internal DRC, regardless of whether
        we found "similar" errors. Our goal is to eventually MATCH or
        BEAT KiCad's detection capabilities.
        """
        # ALWAYS record KiCad errors - these are the GROUND TRUTH
        # Our internal DRC must learn to detect ALL of these
        missed = list(kicad_errors)  # Copy all KiCad errors

        # Identify false positives - errors we found that KiCad didn't
        # (This means we're being too strict, or detecting different things)
        false_positives = []
        if internal_errors > len(kicad_errors):
            # We found more errors than KiCad - might be false positives
            # or we're detecting things KiCad doesn't check
            extra = internal_errors - len(kicad_errors)
            false_positives.append(f"Internal DRC found {extra} more errors than KiCad")

        return DRCComparison(
            kicad_errors=kicad_errors,
            kicad_warnings=kicad_warnings,
            internal_errors=internal_errors,
            internal_warnings=internal_warnings,
            missed_by_internal=missed,  # ALWAYS learn from KiCad
            false_positives=false_positives,
            kicad_passed=len(kicad_errors) == 0
        )

    def _record_learnings(self,
                          pcb_path: str,
                          missed: List[KiCadViolation]) -> List[LearningRecord]:
        """Record what we missed for future improvement"""
        new_records = []
        pcb_name = os.path.basename(pcb_path)
        timestamp = datetime.now().isoformat()

        for violation in missed:
            record = LearningRecord(
                timestamp=timestamp,
                pcb_name=pcb_name,
                violation_type=violation.type,
                description=violation.description,
                position=violation.position,
                context={
                    'items': violation.items,
                    'severity': violation.severity
                },
                learned=False
            )
            new_records.append(record)
            self.learning_records.append(record)

        return new_records

    def _load_learning_db(self):
        """Load learning database from disk"""
        if os.path.exists(self.LEARNING_DB_PATH):
            try:
                with open(self.LEARNING_DB_PATH, 'r') as f:
                    data = json.load(f)
                self.learning_records = [
                    LearningRecord(**r) for r in data.get('records', [])
                ]
            except Exception as e:
                print(f"   Warning: Could not load learning DB: {e}")
                self.learning_records = []

    def _save_learning_db(self):
        """Save learning database to disk"""
        try:
            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'total_records': len(self.learning_records),
                'records': [
                    {
                        'timestamp': r.timestamp,
                        'pcb_name': r.pcb_name,
                        'violation_type': r.violation_type,
                        'description': r.description,
                        'position': r.position,
                        'context': r.context,
                        'learned': r.learned
                    }
                    for r in self.learning_records
                ]
            }
            with open(self.LEARNING_DB_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"   Warning: Could not save learning DB: {e}")

    def get_learning_summary(self) -> Dict:
        """Get summary of what we've learned"""
        by_type = {}
        for r in self.learning_records:
            vtype = r.violation_type
            if vtype not in by_type:
                by_type[vtype] = {'count': 0, 'learned': 0}
            by_type[vtype]['count'] += 1
            if r.learned:
                by_type[vtype]['learned'] += 1

        return {
            'total_records': len(self.learning_records),
            'by_violation_type': by_type,
            'unlearned': sum(1 for r in self.learning_records if not r.learned)
        }

    def print_learning_report(self):
        """Print a summary of learning progress"""
        summary = self.get_learning_summary()

        print("\n" + "=" * 60)
        print("[LEARNING] KiCad DRC LEARNING REPORT")
        print("=" * 60)
        print(f"Total learning records: {summary['total_records']}")
        print(f"Still need to learn: {summary['unlearned']}")
        print("\nBy violation type:")

        for vtype, stats in summary['by_violation_type'].items():
            learned_pct = (stats['learned'] / stats['count'] * 100) if stats['count'] > 0 else 0
            print(f"  {vtype}: {stats['count']} occurrences ({learned_pct:.0f}% learned)")

        print("=" * 60)


# Convenience function for quick validation
def validate_with_kicad(pcb_path: str,
                        internal_errors: int = 0,
                        internal_warnings: int = 0) -> bool:
    """
    Quick validation using KiCad as the authority.

    Returns True if KiCad DRC passes, False otherwise.
    """
    teacher = KiCadDRCTeacher()
    result = teacher.teach(pcb_path, internal_errors, internal_warnings)
    print(result.message)
    return result.passed
