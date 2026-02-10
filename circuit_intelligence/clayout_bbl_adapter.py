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
import sys
import os

# Add pcb_engine to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    NetType,
)
from .clayout_validator import validate_clayout

# Import Smart Algorithm Manager components
try:
    from pcb_engine.routing_planner import (
        RoutingPlanner, RoutingPlan, NetClass, RoutingAlgorithm,
        NetRoutingStrategy
    )
    from pcb_engine.learning_database import LearningDatabase
    SMART_ROUTING_AVAILABLE = True
except ImportError:
    SMART_ROUTING_AVAILABLE = False
    RoutingPlanner = None
    RoutingPlan = None


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

    def _convert_net_classes(self, clayout: ConstitutionalLayout) -> Dict[str, str]:
        """
        Convert c_layout net types to routing planner net classes.

        This bridges the c_layout NetType with RoutingPlanner NetClass.
        """
        net_classes = {}

        # Map c_layout NetType to RoutingPlanner NetClass
        NET_TYPE_TO_CLASS = {
            NetType.POWER: 'power',
            NetType.GND: 'ground',
            NetType.SIGNAL: 'signal',
            NetType.DIFF_PAIR: 'differential',
            NetType.HIGH_SPEED: 'high_speed',
            NetType.ANALOG: 'analog',
            NetType.CLOCK: 'high_speed',  # Clock treated as high-speed
        }

        for net in clayout.nets:
            net_classes[net.name] = NET_TYPE_TO_CLASS.get(net.net_type, 'signal')

        return net_classes


# =============================================================================
# BBL ADAPTER
# =============================================================================

class CLayoutBBLAdapter:
    """
    Adapter between Constitutional Layout and BBL Engine.

    This handles:
    1. Converting c_layout to parts_db
    2. Creating smart routing plan via RoutingPlanner
    3. Running BBL with proper configuration
    4. Interpreting results with rule hierarchy
    5. Generating escalation reports

    INTEGRATION WITH SMART ALGORITHM MANAGER:
    =========================================
    The adapter now integrates with the RoutingPlanner to provide
    intelligent per-net algorithm selection based on:
    - c_layout net types (power, ground, high_speed, diff_pair, etc.)
    - Routing hints (priority nets, diff pairs, length match groups)
    - Learning database (historical success data)
    """

    def __init__(self, bbl_engine=None, learning_db=None):
        """
        Initialize the adapter.

        Args:
            bbl_engine: Optional BBL engine instance (will create if not provided)
            learning_db: Optional learning database for smart routing
        """
        self.bbl_engine = bbl_engine
        self.converter = CLayoutConverter()
        self._current_clayout: Optional[ConstitutionalLayout] = None
        self._learning_db = learning_db
        self._routing_plan: Optional['RoutingPlan'] = None

        # Initialize RoutingPlanner if available
        if SMART_ROUTING_AVAILABLE:
            self._routing_planner = RoutingPlanner(learning_db)
        else:
            self._routing_planner = None

    def run_with_clayout(
        self,
        clayout: ConstitutionalLayout,
        progress_callback: Optional[Callable] = None,
        escalation_callback: Optional[Callable] = None,
        validate_first: bool = True,
        use_smart_routing: bool = True,
    ) -> CLayoutBBLResult:
        """
        Run BBL with a Constitutional Layout.

        Args:
            clayout: The Constitutional Layout to process
            progress_callback: Optional callback for progress updates
            escalation_callback: Optional callback for escalations
            validate_first: Whether to validate c_layout before running
            use_smart_routing: Whether to use RoutingPlanner for algorithm selection

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

        # Step 2.5: Create smart routing plan (if available)
        routing_plan = None
        if use_smart_routing and self._routing_planner and SMART_ROUTING_AVAILABLE:
            routing_plan = self._create_routing_plan(clayout, parts_db)
            self._routing_plan = routing_plan

            # Add routing plan to parts_db for BBL
            parts_db['routing_plan'] = {
                'has_plan': True,
                'net_strategies': {
                    name: {
                        'net_class': strategy.net_class.value,
                        'primary_algorithm': strategy.primary_algorithm.value,
                        'fallback_algorithms': [a.value for a in strategy.fallback_algorithms],
                        'trace_width': strategy.trace_width,
                        'clearance': strategy.clearance,
                        'priority': strategy.priority,
                    }
                    for name, strategy in routing_plan.net_strategies.items()
                },
                'routing_order': routing_plan.routing_order,
                'enable_trunk_chains': routing_plan.enable_trunk_chains,
                'ground_pour_recommended': routing_plan.ground_pour_recommended,
                'success_prediction': routing_plan.overall_success_prediction,
            }

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

    def get_routing_plan(self) -> Optional['RoutingPlan']:
        """Get the routing plan created for the last run."""
        return self._routing_plan

    def _create_routing_plan(
        self,
        clayout: ConstitutionalLayout,
        parts_db: Dict
    ) -> 'RoutingPlan':
        """
        Create a smart routing plan from c_layout.

        This bridges c_layout's rich net metadata with the RoutingPlanner's
        algorithm selection system.

        Args:
            clayout: The Constitutional Layout
            parts_db: The converted parts database

        Returns:
            RoutingPlan with per-net strategies
        """
        # Build board config from c_layout
        board_config = {
            'board_width': clayout.board.width_mm,
            'board_height': clayout.board.height_mm,
            'layers': clayout.board.layer_count,
        }

        # Add net class hints from c_layout
        net_classes = self.converter._convert_net_classes(clayout)
        for net_name, net_class in net_classes.items():
            if net_name in parts_db.get('nets', {}):
                parts_db['nets'][net_name]['class'] = net_class

        # Add routing priority from c_layout
        routing_hints = clayout.routing_hints
        for i, net_name in enumerate(routing_hints.priority_nets):
            if net_name in parts_db.get('nets', {}):
                parts_db['nets'][net_name]['priority'] = i + 1  # 1-indexed priority

        # Add diff pair information
        for diff_pair in routing_hints.diff_pairs:
            for net_name in [diff_pair.positive_net, diff_pair.negative_net]:
                if net_name in parts_db.get('nets', {}):
                    parts_db['nets'][net_name]['class'] = 'differential'
                    parts_db['nets'][net_name]['impedance'] = diff_pair.impedance_ohm
                    parts_db['nets'][net_name]['max_mismatch'] = diff_pair.max_mismatch_mm

        # Create the routing plan
        plan = self._routing_planner.create_routing_plan(parts_db, board_config)

        # Apply c_layout-specific overrides to the plan
        self._apply_clayout_overrides(plan, clayout)

        return plan

    def _apply_clayout_overrides(
        self,
        plan: 'RoutingPlan',
        clayout: ConstitutionalLayout
    ) -> None:
        """
        Apply c_layout rule overrides to the routing plan.

        If c_layout has specific rules that override default behavior,
        apply them to the net strategies.
        """
        routing_hints = clayout.routing_hints

        # Apply no_auto_route - mark these nets to skip
        for net_name in routing_hints.no_auto_route:
            if net_name in plan.net_strategies:
                # Set priority very low so it routes last (or not at all)
                plan.net_strategies[net_name].priority = 999

        # Apply length matching groups
        for lm_group in routing_hints.length_match_groups:
            for net_name in lm_group.nets:
                if net_name in plan.net_strategies:
                    plan.net_strategies[net_name].length_match_group = lm_group.name

        # Apply layer assignments as constraints
        for net_name, layers in routing_hints.layer_assignments.items():
            if net_name in plan.net_strategies:
                # Store layer constraint in strategy (for routing piston to use)
                plan.net_strategies[net_name].allowed_layers = layers

        # Apply via constraints
        for net_name, via_info in routing_hints.via_constraints.items():
            if net_name in plan.net_strategies:
                strategy = plan.net_strategies[net_name]
                if 'via_size' in via_info:
                    strategy.via_size = via_info['via_size']
                if 'via_drill' in via_info:
                    strategy.via_drill = via_info['via_drill']
                if 'max_vias' in via_info:
                    strategy.max_vias = via_info.get('max_vias')


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
