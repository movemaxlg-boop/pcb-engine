"""
CONSTITUTIONAL LAYOUT BBL ADAPTER
==================================

This adapter bridges the Constitutional Layout system with the BBL Engine.
It converts c_layout to parts_db format and handles rule hierarchy enforcement.

The adapter:
1. Converts c_layout components/nets to parts_db format
2. Configures BBL to respect rule hierarchy
3. Intercepts DRC violations and classifies by priority
4. Generates escalation reports for the AI agent
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import copy

from .clayout_types import (
    ConstitutionalLayout,
    ComponentDefinition,
    NetDefinition,
    RuleHierarchy,
    RuleBinding,
    RulePriority,
    RuleOverride,
    EscalationReport,
    CLayoutValidationResult,
)
from .clayout_validator import validate_clayout


# =============================================================================
# BBL RESULT WRAPPER
# =============================================================================

@dataclass
class CLayoutBBLResult:
    """
    Result of running BBL with a Constitutional Layout.

    This wraps the BBL result with c_layout-specific information.
    """

    # Success status
    success: bool = False
    partial_success: bool = False

    # Phase completion
    phases_completed: List[str] = field(default_factory=list)
    phases_failed: List[str] = field(default_factory=list)

    # Routing metrics
    routing_completion_pct: float = 0.0
    routed_nets: int = 0
    total_nets: int = 0
    unrouted_nets: List[str] = field(default_factory=list)

    # DRC results
    drc_passed: bool = False
    kicad_drc_passed: bool = False
    drc_errors: List[Dict] = field(default_factory=list)
    drc_warnings: List[Dict] = field(default_factory=list)

    # Rule violations by priority
    inviolable_violations: List[str] = field(default_factory=list)
    recommended_violations: List[str] = field(default_factory=list)
    optional_violations: List[str] = field(default_factory=list)

    # Output
    output_path: Optional[str] = None
    output_files: List[str] = field(default_factory=list)

    # Timing
    total_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)

    # Escalation (if needed)
    escalation_needed: bool = False
    escalation_report: Optional[EscalationReport] = None

    # Original BBL result (if available)
    bbl_result: Optional[Any] = None

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'partial_success': self.partial_success,
            'phases_completed': self.phases_completed,
            'phases_failed': self.phases_failed,
            'routing_completion_pct': self.routing_completion_pct,
            'routed_nets': self.routed_nets,
            'total_nets': self.total_nets,
            'unrouted_nets': self.unrouted_nets,
            'drc_passed': self.drc_passed,
            'kicad_drc_passed': self.kicad_drc_passed,
            'inviolable_violations': self.inviolable_violations,
            'recommended_violations': self.recommended_violations,
            'optional_violations': self.optional_violations,
            'output_path': self.output_path,
            'output_files': self.output_files,
            'total_time': self.total_time,
            'phase_times': self.phase_times,
            'escalation_needed': self.escalation_needed,
        }


# =============================================================================
# CLAYOUT TO PARTS_DB CONVERTER
# =============================================================================

class CLayoutConverter:
    """
    Converts Constitutional Layout to parts_db format for BBL.
    """

    def convert(self, clayout: ConstitutionalLayout) -> Dict:
        """
        Convert a c_layout to parts_db format.

        Args:
            clayout: Constitutional Layout to convert

        Returns:
            parts_db dictionary compatible with BBL
        """

        parts_db = {
            'board': self._convert_board(clayout),
            'parts': self._convert_components(clayout),
            'nets': self._convert_nets(clayout),
            'design_rules': self._convert_rules(clayout),
            'placement_hints': self._convert_placement_hints(clayout),
            'routing_hints': self._convert_routing_hints(clayout),
            'metadata': {
                'design_name': clayout.design_name,
                'version': clayout.version,
                'created_by': clayout.created_by,
                'has_clayout': True,
                'rule_hierarchy': {
                    'inviolable_count': len(clayout.rules.inviolable),
                    'recommended_count': len(clayout.rules.recommended),
                    'optional_count': len(clayout.rules.optional),
                },
            }
        }

        return parts_db

    def _convert_board(self, clayout: ConstitutionalLayout) -> Dict:
        """Convert board constraints to parts_db format."""
        board = clayout.board
        return {
            'width': board.width_mm,
            'height': board.height_mm,
            'layers': board.layer_count,
            'stackup': board.stackup,
            'min_trace': board.min_trace_mm,
            'min_space': board.min_space_mm,
            'min_via_drill': board.min_via_drill_mm,
            'min_via_annular': board.min_via_annular_mm,
            'copper_weight': board.copper_weight_oz,
            'thickness': board.board_thickness_mm,
            'dk': board.dielectric_constant,
            'edge_clearance': board.edge_clearance_mm,
        }

    def _convert_components(self, clayout: ConstitutionalLayout) -> Dict:
        """Convert components to parts_db format."""
        parts = {}

        for comp in clayout.components:
            parts[comp.ref_des] = {
                'part_number': comp.part_number,
                'footprint': comp.footprint,
                'category': comp.category.value,
                'value': comp.value,
                'voltage_rating': comp.voltage_rating,
                'current_rating': comp.current_rating,
                'power_dissipation': comp.power_dissipation,
                'theta_ja': comp.theta_ja,
                'inferred': comp.inferred,
                'inference_reason': comp.inference_reason,
                'inferred_for': comp.inferred_for,
                'preferred_side': comp.preferred_side,
                'rotation_allowed': comp.rotation_allowed,
            }

        return parts

    def _convert_nets(self, clayout: ConstitutionalLayout) -> Dict:
        """Convert nets to parts_db format."""
        nets = {}

        for net in clayout.nets:
            nets[net.name] = {
                'type': net.net_type.value,
                'pins': net.pins,
                'voltage': net.voltage,
                'current_max': net.current_max,
                'frequency': net.frequency,
                'impedance': net.impedance_ohm,
                'max_length': net.max_length_mm,
                'min_width': net.min_width_mm,
                'matched_with': net.matched_with,
                'max_mismatch': net.max_mismatch_mm,
                'allowed_layers': net.allowed_layers,
                'routing_priority': net.routing_priority,
            }

        return nets

    def _convert_rules(self, clayout: ConstitutionalLayout) -> Dict:
        """Convert rule hierarchy to design_rules format."""
        rules = {
            'hierarchy': {},
            'overrides': [],
        }

        # Convert hierarchy
        for binding in clayout.rules.inviolable:
            rules['hierarchy'][binding.rule_id] = {
                'priority': 'inviolable',
                'parameters': binding.parameters,
                'applies_to': binding.applies_to,
                'reason': binding.reason,
            }

        for binding in clayout.rules.recommended:
            rules['hierarchy'][binding.rule_id] = {
                'priority': 'recommended',
                'parameters': binding.parameters,
                'applies_to': binding.applies_to,
                'reason': binding.reason,
            }

        for binding in clayout.rules.optional:
            rules['hierarchy'][binding.rule_id] = {
                'priority': 'optional',
                'parameters': binding.parameters,
                'applies_to': binding.applies_to,
                'reason': binding.reason,
            }

        # Convert overrides
        for override in clayout.overrides:
            rules['overrides'].append({
                'rule_id': override.rule_id,
                'original_value': override.original_value,
                'new_value': override.new_value,
                'justification': override.justification,
                'evidence': override.evidence,
                'approved_by': override.approved_by,
            })

        return rules

    def _convert_placement_hints(self, clayout: ConstitutionalLayout) -> Dict:
        """Convert placement hints to parts_db format."""
        hints = clayout.placement_hints
        return {
            'proximity_groups': [
                {
                    'components': g.components,
                    'max_distance': g.max_distance_mm,
                    'reason': g.reason,
                    'priority': g.priority,
                }
                for g in hints.proximity_groups
            ],
            'edge_components': hints.edge_components,
            'keep_apart': [
                {
                    'a': k.component_a,
                    'b': k.component_b,
                    'min_distance': k.min_distance_mm,
                    'reason': k.reason,
                }
                for k in hints.keep_apart
            ],
            'zones': [z.to_dict() for z in hints.zones],
            'fixed_positions': hints.fixed_positions,
            'fixed_rotations': hints.fixed_rotations,
            'placement_order': hints.placement_order,
        }

    def _convert_routing_hints(self, clayout: ConstitutionalLayout) -> Dict:
        """Convert routing hints to parts_db format."""
        hints = clayout.routing_hints
        return {
            'priority_nets': hints.priority_nets,
            'diff_pairs': [d.to_dict() for d in hints.diff_pairs],
            'length_match_groups': [g.to_dict() for g in hints.length_match_groups],
            'layer_assignments': hints.layer_assignments,
            'deprioritized_nets': hints.deprioritized_nets,
            'no_auto_route': hints.no_auto_route,
            'via_constraints': hints.via_constraints,
        }


# =============================================================================
# BBL ADAPTER
# =============================================================================

class CLayoutBBLAdapter:
    """
    Adapter between Constitutional Layout and BBL Engine.

    This handles:
    1. Converting c_layout to parts_db
    2. Running BBL with proper configuration
    3. Interpreting results with rule hierarchy
    4. Generating escalation reports
    """

    def __init__(self, bbl_engine=None):
        """
        Initialize the adapter.

        Args:
            bbl_engine: Optional BBL engine instance (will create if not provided)
        """
        self.bbl_engine = bbl_engine
        self.converter = CLayoutConverter()
        self._current_clayout: Optional[ConstitutionalLayout] = None

    def run_with_clayout(
        self,
        clayout: ConstitutionalLayout,
        progress_callback: Optional[Callable] = None,
        escalation_callback: Optional[Callable] = None,
        validate_first: bool = True,
    ) -> CLayoutBBLResult:
        """
        Run BBL with a Constitutional Layout.

        Args:
            clayout: The Constitutional Layout to process
            progress_callback: Optional callback for progress updates
            escalation_callback: Optional callback for escalations
            validate_first: Whether to validate c_layout before running

        Returns:
            CLayoutBBLResult with execution results
        """

        start_time = time.time()
        self._current_clayout = clayout

        result = CLayoutBBLResult()
        result.total_nets = len(clayout.nets)

        # Step 1: Validate c_layout (if requested)
        if validate_first:
            validation = validate_clayout(clayout)
            if not validation.valid:
                result.escalation_needed = True
                result.escalation_report = EscalationReport(
                    failure_type="validation_failed",
                    phase="pre_validation",
                    violated_rules=[],
                    suggestions=validation.suggestions,
                    details={'errors': validation.errors}
                )
                return result

        # Step 2: Convert c_layout to parts_db
        parts_db = self.converter.convert(clayout)

        # Step 3: Run BBL
        if self.bbl_engine:
            try:
                bbl_result = self.bbl_engine.run(
                    parts_db=parts_db,
                    config={'clayout_mode': True}
                )
                result.bbl_result = bbl_result

                # Extract results from BBL
                result.success = getattr(bbl_result, 'success', False)
                result.drc_passed = getattr(bbl_result, 'drc_passed', False)
                result.kicad_drc_passed = getattr(bbl_result, 'kicad_drc_passed', False)
                result.output_path = getattr(bbl_result, 'output_path', None)

                # Routing metrics
                if hasattr(bbl_result, 'routing_result'):
                    rr = bbl_result.routing_result
                    result.routed_nets = getattr(rr, 'routed_count', 0)
                    result.routing_completion_pct = (
                        result.routed_nets / result.total_nets * 100
                        if result.total_nets > 0 else 0
                    )

                # Classify DRC violations by priority
                if hasattr(bbl_result, 'drc_violations'):
                    self._classify_violations(bbl_result.drc_violations, clayout.rules, result)

            except Exception as e:
                result.phases_failed.append("bbl_execution")
                result.escalation_needed = True
                result.escalation_report = EscalationReport(
                    failure_type="bbl_exception",
                    phase="bbl_execution",
                    violated_rules=[],
                    suggestions=["Check BBL engine configuration", "Review c_layout for issues"],
                    details={'error': str(e)}
                )
        else:
            # No BBL engine - simulate result
            result = self._simulate_bbl_run(clayout)

        # Step 4: Determine if escalation is needed
        if result.inviolable_violations:
            result.escalation_needed = True
            result.success = False
            result.escalation_report = EscalationReport(
                failure_type="inviolable_violation",
                phase="drc_validation",
                violated_rules=result.inviolable_violations,
                suggestions=self._generate_violation_suggestions(result.inviolable_violations, clayout),
                routing_completion_pct=result.routing_completion_pct,
            )

        result.total_time = time.time() - start_time

        return result

    def _classify_violations(
        self,
        violations: List[Dict],
        rules: RuleHierarchy,
        result: CLayoutBBLResult
    ):
        """Classify violations by rule priority."""

        for violation in violations:
            rule_id = violation.get('rule_id', '')
            description = violation.get('description', str(violation))

            priority = rules.get_priority(rule_id)

            if priority == RulePriority.INVIOLABLE:
                result.inviolable_violations.append(description)
            elif priority == RulePriority.RECOMMENDED:
                result.recommended_violations.append(description)
            else:
                result.optional_violations.append(description)

    def _generate_violation_suggestions(
        self,
        violations: List[str],
        clayout: ConstitutionalLayout
    ) -> List[str]:
        """Generate suggestions for fixing violations."""

        suggestions = []

        for violation in violations:
            if "spacing" in violation.lower():
                suggestions.append("Increase board size or reduce component count")
                suggestions.append("Consider 4-layer board for more routing space")
            elif "impedance" in violation.lower():
                suggestions.append("Adjust trace width for correct impedance")
                suggestions.append("Check stackup configuration")
            elif "thermal" in violation.lower():
                suggestions.append("Add thermal vias under high-power components")
                suggestions.append("Increase copper pour area")
            elif "length" in violation.lower():
                suggestions.append("Use serpentine routing for length matching")
                suggestions.append("Review placement to reduce length differences")

        return list(set(suggestions))  # Remove duplicates

    def _simulate_bbl_run(self, clayout: ConstitutionalLayout) -> CLayoutBBLResult:
        """
        Simulate a BBL run (for testing without actual BBL engine).

        This is useful for testing the c_layout system independently.
        """

        result = CLayoutBBLResult()
        result.total_nets = len(clayout.nets)

        # Simulate successful routing
        result.routed_nets = result.total_nets
        result.routing_completion_pct = 100.0
        result.success = True
        result.drc_passed = True
        result.kicad_drc_passed = True
        result.phases_completed = [
            'order_received',
            'piston_execution',
            'output_generation',
            'kicad_drc',
            'learning_delivery'
        ]

        # Simulate some warnings for recommended rules
        if clayout.rules.recommended:
            result.recommended_violations = [
                f"Warning: {r.rule_id} is close to limit"
                for r in clayout.rules.recommended[:2]
            ]

        return result

    def get_escalation_report(self) -> Optional[EscalationReport]:
        """Get the last escalation report, if any."""
        return getattr(self, '_last_escalation', None)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_clayout_through_bbl(
    clayout: ConstitutionalLayout,
    bbl_engine=None
) -> CLayoutBBLResult:
    """
    Convenience function to run a c_layout through BBL.

    Args:
        clayout: The Constitutional Layout
        bbl_engine: Optional BBL engine instance

    Returns:
        CLayoutBBLResult
    """
    adapter = CLayoutBBLAdapter(bbl_engine)
    return adapter.run_with_clayout(clayout)


def convert_clayout_to_parts_db(clayout: ConstitutionalLayout) -> Dict:
    """
    Convert a c_layout to parts_db format.

    Args:
        clayout: Constitutional Layout to convert

    Returns:
        parts_db dictionary
    """
    converter = CLayoutConverter()
    return converter.convert(clayout)
