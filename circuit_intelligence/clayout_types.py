"""
CONSTITUTIONAL LAYOUT (c_layout) - Core Data Types
===================================================

The Constitutional Layout is the machine-readable design specification that bridges
user intent and PCB Engine execution. It includes:

1. Design identity and board constraints
2. Components (user-specified + AI-inferred)
3. Net connectivity
4. Rule hierarchy (Inviolable/Recommended/Optional)
5. AI overrides with justification
6. Placement and routing hints

This module defines all the data structures used throughout the c_layout system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# ENUMS
# =============================================================================

class RulePriority(Enum):
    """Rule priority levels for the hierarchy."""
    INVIOLABLE = "inviolable"      # MUST pass - abort if violated
    RECOMMENDED = "recommended"    # SHOULD pass - warn if violated
    OPTIONAL = "optional"          # MAY pass - log if violated


class NetType(Enum):
    """Types of nets in a design."""
    POWER = "power"
    GND = "ground"
    SIGNAL = "signal"
    DIFF_PAIR = "diff_pair"
    HIGH_SPEED = "high_speed"
    ANALOG = "analog"
    CLOCK = "clock"


class ComponentCategory(Enum):
    """Categories of components."""
    MCU = "mcu"
    REGULATOR = "regulator"
    CAPACITOR = "capacitor"
    RESISTOR = "resistor"
    INDUCTOR = "inductor"
    CONNECTOR = "connector"
    CRYSTAL = "crystal"
    ESD_PROTECTION = "esd_protection"
    TRANSISTOR = "transistor"
    DIODE = "diode"
    LED = "led"
    SENSOR = "sensor"
    MEMORY = "memory"
    ANALOG = "analog"  # Analog ICs (op-amps, ADCs, DACs, etc.)
    RF = "rf"  # RF components (antennas, baluns, filters)
    OTHER = "other"


class ValidationStatus(Enum):
    """Validation status of a c_layout."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    WARNINGS = "warnings"


class ZoneType(Enum):
    """Types of functional zones on the board."""
    POWER = "power"
    ANALOG = "analog"
    DIGITAL = "digital"
    RF = "rf"
    HIGH_SPEED = "high_speed"
    SENSITIVE = "sensitive"


# =============================================================================
# BOARD CONSTRAINTS
# =============================================================================

@dataclass
class BoardConstraints:
    """Physical board constraints and specifications."""

    # Dimensions
    width_mm: float
    height_mm: float
    layer_count: int

    # Stackup (optional - for advanced designs)
    stackup: Optional[str] = None  # "standard_4layer", "rf_6layer", etc.

    # Fabrication capabilities
    min_trace_mm: float = 0.15
    min_space_mm: float = 0.15
    min_via_drill_mm: float = 0.3
    min_via_annular_mm: float = 0.15

    # Materials
    copper_weight_oz: float = 1.0
    board_thickness_mm: float = 1.6
    dielectric_constant: float = 4.2  # FR-4 default

    # Optional constraints
    max_component_height_mm: Optional[float] = None
    edge_clearance_mm: float = 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'width_mm': self.width_mm,
            'height_mm': self.height_mm,
            'layer_count': self.layer_count,
            'stackup': self.stackup,
            'min_trace_mm': self.min_trace_mm,
            'min_space_mm': self.min_space_mm,
            'min_via_drill_mm': self.min_via_drill_mm,
            'min_via_annular_mm': self.min_via_annular_mm,
            'copper_weight_oz': self.copper_weight_oz,
            'board_thickness_mm': self.board_thickness_mm,
            'dielectric_constant': self.dielectric_constant,
            'max_component_height_mm': self.max_component_height_mm,
            'edge_clearance_mm': self.edge_clearance_mm,
        }


# =============================================================================
# COMPONENT DEFINITIONS
# =============================================================================

@dataclass
class ComponentDefinition:
    """Component specification for c_layout."""

    # Identity
    ref_des: str                    # U1, C1, R1
    part_number: str                # From parts_db
    footprint: str                  # Package type

    # Category
    category: ComponentCategory = ComponentCategory.OTHER

    # Value (for passives)
    value: Optional[str] = None     # 10uF, 10k, etc.

    # Electrical properties
    voltage_rating: float = 0.0
    current_rating: float = 0.0
    power_dissipation: float = 0.0

    # Thermal properties
    theta_ja: float = 0.0           # Junction-to-ambient thermal resistance

    # AI inference metadata
    inferred: bool = False          # True if AI added this
    inference_reason: str = ""      # "Decoupling for U1"
    inferred_for: Optional[str] = None  # Which component this supports

    # Placement hints for this component
    preferred_side: str = "top"     # "top" or "bottom"
    rotation_allowed: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'ref_des': self.ref_des,
            'part_number': self.part_number,
            'footprint': self.footprint,
            'category': self.category.value,
            'value': self.value,
            'voltage_rating': self.voltage_rating,
            'current_rating': self.current_rating,
            'power_dissipation': self.power_dissipation,
            'theta_ja': self.theta_ja,
            'inferred': self.inferred,
            'inference_reason': self.inference_reason,
            'inferred_for': self.inferred_for,
            'preferred_side': self.preferred_side,
            'rotation_allowed': self.rotation_allowed,
        }


# =============================================================================
# NET DEFINITIONS
# =============================================================================

@dataclass
class NetDefinition:
    """Net connectivity specification."""

    # Identity
    name: str
    net_type: NetType = NetType.SIGNAL

    # Connectivity
    pins: List[str] = field(default_factory=list)  # ["U1.VCC", "C1.1", "C2.1"]

    # Electrical properties
    voltage: float = 0.0
    current_max: float = 0.0
    frequency: float = 0.0          # Hz

    # Routing constraints
    impedance_ohm: Optional[float] = None
    max_length_mm: Optional[float] = None
    min_width_mm: Optional[float] = None

    # Differential pair matching
    matched_with: Optional[str] = None  # For diff pairs
    max_mismatch_mm: Optional[float] = None

    # Layer constraints
    allowed_layers: Optional[List[int]] = None

    # Priority (lower = route first)
    routing_priority: int = 50      # 0-100, lower is higher priority

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'net_type': self.net_type.value,
            'pins': self.pins,
            'voltage': self.voltage,
            'current_max': self.current_max,
            'frequency': self.frequency,
            'impedance_ohm': self.impedance_ohm,
            'max_length_mm': self.max_length_mm,
            'min_width_mm': self.min_width_mm,
            'matched_with': self.matched_with,
            'max_mismatch_mm': self.max_mismatch_mm,
            'allowed_layers': self.allowed_layers,
            'routing_priority': self.routing_priority,
        }


# =============================================================================
# RULE HIERARCHY - THE KEY INNOVATION
# =============================================================================

@dataclass
class RuleBinding:
    """Binding of a rule to this design with specific parameters."""

    rule_id: str                    # "USB2_LENGTH_MATCHING"
    parameters: Dict[str, Any] = field(default_factory=dict)  # {"max_mismatch_mm": 1.25}
    applies_to: List[str] = field(default_factory=lambda: ["*"])  # ["USB_DP", "USB_DM"] or ["*"]
    reason: str = ""                # Why this category
    source: str = ""                # Standard reference

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'rule_id': self.rule_id,
            'parameters': self.parameters,
            'applies_to': self.applies_to,
            'reason': self.reason,
            'source': self.source,
        }


@dataclass
class RuleHierarchy:
    """
    Three-tier rule classification - THE KEY INNOVATION.

    INVIOLABLE: Rules that MUST pass or the design ABORTS.
                Typically safety-critical or functionality-critical.
                Examples: conductor spacing, thermal limits, impedance for high-speed.

    RECOMMENDED: Rules that SHOULD pass. If violated, design continues with WARNING.
                 User should review but can proceed.
                 Examples: decoupling distance, crystal placement, EMI limits.

    OPTIONAL: Rules that MAY pass. If violated, design logs and continues.
              Nice to have but not critical.
              Examples: silkscreen aesthetics, test point placement.
    """

    # Rules that MUST pass - abort if violated
    inviolable: List[RuleBinding] = field(default_factory=list)

    # Rules that SHOULD pass - warn if violated
    recommended: List[RuleBinding] = field(default_factory=list)

    # Rules that MAY pass - note if violated
    optional: List[RuleBinding] = field(default_factory=list)

    # Default category for rules not explicitly listed
    default_category: RulePriority = RulePriority.RECOMMENDED

    def get_priority(self, rule_id: str) -> RulePriority:
        """Get the priority of a rule by ID."""
        for rule in self.inviolable:
            if rule.rule_id == rule_id:
                return RulePriority.INVIOLABLE
        for rule in self.recommended:
            if rule.rule_id == rule_id:
                return RulePriority.RECOMMENDED
        for rule in self.optional:
            if rule.rule_id == rule_id:
                return RulePriority.OPTIONAL
        return self.default_category

    def get_rule(self, rule_id: str) -> Optional[RuleBinding]:
        """Get a rule binding by ID."""
        for rule in self.inviolable + self.recommended + self.optional:
            if rule.rule_id == rule_id:
                return rule
        return None

    def count(self) -> Dict[str, int]:
        """Get count of rules in each category."""
        return {
            'inviolable': len(self.inviolable),
            'recommended': len(self.recommended),
            'optional': len(self.optional),
            'total': len(self.inviolable) + len(self.recommended) + len(self.optional),
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'inviolable': [r.to_dict() for r in self.inviolable],
            'recommended': [r.to_dict() for r in self.recommended],
            'optional': [r.to_dict() for r in self.optional],
            'default_category': self.default_category.value,
        }


# =============================================================================
# RULE OVERRIDES
# =============================================================================

@dataclass
class RuleOverride:
    """
    AI-justified modification to a rule.

    This allows the AI agent to relax or tighten rules based on
    engineering judgment, with full justification for traceability.
    """

    rule_id: str
    original_value: Any
    new_value: Any
    justification: str              # Engineering reason
    evidence: str                   # Source of justification (datasheet, spec, etc.)
    approved_by: str = "AI"         # "AI" or "user"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Validation
    validated: bool = False
    validation_notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'rule_id': self.rule_id,
            'original_value': self.original_value,
            'new_value': self.new_value,
            'justification': self.justification,
            'evidence': self.evidence,
            'approved_by': self.approved_by,
            'timestamp': self.timestamp,
            'validated': self.validated,
            'validation_notes': self.validation_notes,
        }


# =============================================================================
# PLACEMENT HINTS
# =============================================================================

@dataclass
class ProximityGroup:
    """Components that should be placed close together."""

    components: List[str]           # ["U1", "C1", "C2"]
    max_distance_mm: float          # Maximum distance between any two
    reason: str                     # "Decoupling capacitors for MCU"
    priority: int = 50              # Lower = more important

    def to_dict(self) -> Dict:
        return {
            'components': self.components,
            'max_distance_mm': self.max_distance_mm,
            'reason': self.reason,
            'priority': self.priority,
        }


@dataclass
class KeepApart:
    """Components that should be separated."""

    component_a: str
    component_b: str
    min_distance_mm: float
    reason: str                     # "RF interference", "Thermal isolation"

    def to_dict(self) -> Dict:
        return {
            'component_a': self.component_a,
            'component_b': self.component_b,
            'min_distance_mm': self.min_distance_mm,
            'reason': self.reason,
        }


@dataclass
class FunctionalZone:
    """A functional zone on the board."""

    name: str
    zone_type: ZoneType
    components: List[str]
    preferred_location: Optional[str] = None  # "top_left", "center", etc.
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'zone_type': self.zone_type.value,
            'components': self.components,
            'preferred_location': self.preferred_location,
            'notes': self.notes,
        }


@dataclass
class PlacementHints:
    """High-level placement guidance for the engine."""

    # Components that must be close together
    proximity_groups: List[ProximityGroup] = field(default_factory=list)

    # Components on board edge
    edge_components: List[str] = field(default_factory=list)

    # Components that must be apart
    keep_apart: List[KeepApart] = field(default_factory=list)

    # Functional zones
    zones: List[FunctionalZone] = field(default_factory=list)

    # Fixed positions (if any) - {ref_des: (x, y)}
    fixed_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Fixed rotations (if any) - {ref_des: degrees}
    fixed_rotations: Dict[str, float] = field(default_factory=dict)

    # Placement order preference
    placement_order: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'proximity_groups': [g.to_dict() for g in self.proximity_groups],
            'edge_components': self.edge_components,
            'keep_apart': [k.to_dict() for k in self.keep_apart],
            'zones': [z.to_dict() for z in self.zones],
            'fixed_positions': self.fixed_positions,
            'fixed_rotations': self.fixed_rotations,
            'placement_order': self.placement_order,
        }


# =============================================================================
# ROUTING HINTS
# =============================================================================

@dataclass
class DiffPairSpec:
    """Differential pair specification."""

    positive_net: str
    negative_net: str
    impedance_ohm: float
    max_mismatch_mm: float
    max_length_mm: Optional[float] = None
    spacing_mm: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'positive_net': self.positive_net,
            'negative_net': self.negative_net,
            'impedance_ohm': self.impedance_ohm,
            'max_mismatch_mm': self.max_mismatch_mm,
            'max_length_mm': self.max_length_mm,
            'spacing_mm': self.spacing_mm,
        }


@dataclass
class LengthMatchGroup:
    """Group of nets that must be length-matched."""

    name: str
    nets: List[str]
    max_mismatch_mm: float
    reference_net: Optional[str] = None  # Net to match others to
    target_length_mm: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'nets': self.nets,
            'max_mismatch_mm': self.max_mismatch_mm,
            'reference_net': self.reference_net,
            'target_length_mm': self.target_length_mm,
        }


@dataclass
class RoutingHints:
    """High-level routing guidance for the engine."""

    # Nets to route first (in order)
    priority_nets: List[str] = field(default_factory=list)

    # Differential pairs
    diff_pairs: List[DiffPairSpec] = field(default_factory=list)

    # Length matching groups
    length_match_groups: List[LengthMatchGroup] = field(default_factory=list)

    # Nets that need specific layers - {net_name: [layer_ids]}
    layer_assignments: Dict[str, List[int]] = field(default_factory=dict)

    # Nets to route last
    deprioritized_nets: List[str] = field(default_factory=list)

    # Nets that should not be routed (manual routing required)
    no_auto_route: List[str] = field(default_factory=list)

    # Via constraints per net
    via_constraints: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'priority_nets': self.priority_nets,
            'diff_pairs': [d.to_dict() for d in self.diff_pairs],
            'length_match_groups': [g.to_dict() for g in self.length_match_groups],
            'layer_assignments': self.layer_assignments,
            'deprioritized_nets': self.deprioritized_nets,
            'no_auto_route': self.no_auto_route,
            'via_constraints': self.via_constraints,
        }


# =============================================================================
# CONSTITUTIONAL LAYOUT - THE MAIN STRUCTURE
# =============================================================================

@dataclass
class ConstitutionalLayout:
    """
    Machine-readable design specification with rule hierarchy.

    This is the complete specification passed from the AI Agent to the PCB Engine.
    It contains everything needed to generate a PCB design while respecting
    the rule hierarchy and AI overrides.
    """

    # DESIGN IDENTITY
    design_name: str
    version: str = "1.0"
    created_by: str = "CircuitAI"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # BOARD CONSTRAINTS
    board: BoardConstraints = field(default_factory=lambda: BoardConstraints(
        width_mm=50, height_mm=40, layer_count=2
    ))

    # COMPONENTS (from parts_db + AI-inferred)
    components: List[ComponentDefinition] = field(default_factory=list)

    # CONNECTIVITY
    nets: List[NetDefinition] = field(default_factory=list)

    # RULE HIERARCHY (THE KEY INNOVATION)
    rules: RuleHierarchy = field(default_factory=RuleHierarchy)

    # AI OVERRIDES (with justification)
    overrides: List[RuleOverride] = field(default_factory=list)

    # PLACEMENT INTENT
    placement_hints: PlacementHints = field(default_factory=PlacementHints)

    # ROUTING INTENT
    routing_hints: RoutingHints = field(default_factory=RoutingHints)

    # METADATA
    user_requirements: List[str] = field(default_factory=list)  # Original user statements
    ai_assumptions: List[str] = field(default_factory=list)     # What AI inferred
    notes: List[str] = field(default_factory=list)              # Additional notes

    # VALIDATION STATUS
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_component(self, ref_des: str) -> Optional[ComponentDefinition]:
        """Get a component by reference designator."""
        for comp in self.components:
            if comp.ref_des == ref_des:
                return comp
        return None

    def get_net(self, name: str) -> Optional[NetDefinition]:
        """Get a net by name."""
        for net in self.nets:
            if net.name == name:
                return net
        return None

    def get_inferred_components(self) -> List[ComponentDefinition]:
        """Get all AI-inferred components."""
        return [c for c in self.components if c.inferred]

    def get_user_components(self) -> List[ComponentDefinition]:
        """Get all user-specified components."""
        return [c for c in self.components if not c.inferred]

    def get_override(self, rule_id: str) -> Optional[RuleOverride]:
        """Get an override by rule ID."""
        for override in self.overrides:
            if override.rule_id == rule_id:
                return override
        return None

    def get_effective_rule_value(self, rule_id: str, default: Any = None) -> Any:
        """Get the effective value of a rule (after override)."""
        override = self.get_override(rule_id)
        if override:
            return override.new_value
        rule = self.rules.get_rule(rule_id)
        if rule and 'value' in rule.parameters:
            return rule.parameters['value']
        return default

    def summary(self) -> Dict:
        """Get a summary of this c_layout."""
        return {
            'design_name': self.design_name,
            'version': self.version,
            'created_by': self.created_by,
            'board': f"{self.board.width_mm}x{self.board.height_mm}mm, {self.board.layer_count}L",
            'components': {
                'total': len(self.components),
                'user_specified': len(self.get_user_components()),
                'ai_inferred': len(self.get_inferred_components()),
            },
            'nets': len(self.nets),
            'rules': self.rules.count(),
            'overrides': len(self.overrides),
            'validation_status': self.validation_status.value,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'design_name': self.design_name,
            'version': self.version,
            'created_by': self.created_by,
            'created_at': self.created_at,
            'board': self.board.to_dict(),
            'components': [c.to_dict() for c in self.components],
            'nets': [n.to_dict() for n in self.nets],
            'rules': self.rules.to_dict(),
            'overrides': [o.to_dict() for o in self.overrides],
            'placement_hints': self.placement_hints.to_dict(),
            'routing_hints': self.routing_hints.to_dict(),
            'user_requirements': self.user_requirements,
            'ai_assumptions': self.ai_assumptions,
            'notes': self.notes,
            'validation_status': self.validation_status.value,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConstitutionalLayout':
        """Create from dictionary."""
        # This is a simplified implementation - full version would parse all nested structures
        layout = cls(
            design_name=data.get('design_name', 'Unnamed'),
            version=data.get('version', '1.0'),
            created_by=data.get('created_by', 'Unknown'),
            created_at=data.get('created_at', datetime.now().isoformat()),
        )

        # Parse board
        if 'board' in data:
            b = data['board']
            layout.board = BoardConstraints(
                width_mm=b.get('width_mm', 50),
                height_mm=b.get('height_mm', 40),
                layer_count=b.get('layer_count', 2),
            )

        # Parse user requirements and assumptions
        layout.user_requirements = data.get('user_requirements', [])
        layout.ai_assumptions = data.get('ai_assumptions', [])
        layout.notes = data.get('notes', [])

        return layout

    @classmethod
    def from_json(cls, json_str: str) -> 'ConstitutionalLayout':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# ESCALATION REPORT
# =============================================================================

@dataclass
class EscalationReport:
    """
    Report sent to AI agent when BBL needs help.

    This is generated when the PCB Engine encounters a problem it cannot
    solve on its own and needs the AI agent to adjust the c_layout.
    """

    failure_type: str               # "routing_failed", "placement_impossible", etc.
    phase: str                      # Which BBL phase failed
    violated_rules: List[str]       # Which rules couldn't be met
    suggestions: List[str]          # What might help

    # Metrics
    routing_completion_pct: float = 0.0
    placement_density_pct: float = 0.0
    attempts: int = 0

    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'failure_type': self.failure_type,
            'phase': self.phase,
            'violated_rules': self.violated_rules,
            'suggestions': self.suggestions,
            'routing_completion_pct': self.routing_completion_pct,
            'placement_density_pct': self.placement_density_pct,
            'attempts': self.attempts,
            'details': self.details,
            'timestamp': self.timestamp,
        }


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class CLayoutValidationResult:
    """Result of c_layout validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)       # Must fix before proceeding
    warnings: List[str] = field(default_factory=list)     # Should review
    suggestions: List[str] = field(default_factory=list)  # Nice to fix
    routability_estimate: float = 0.0                     # 0.0-1.0

    def to_dict(self) -> Dict:
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'routability_estimate': self.routability_estimate,
        }


# =============================================================================
# DESIGN INTENT (User's original request parsed)
# =============================================================================

@dataclass
class DesignIntent:
    """Parsed user intent for design generation."""

    design_name: str = "Unnamed_Design"

    # Original statements from user
    original_statements: List[str] = field(default_factory=list)

    # Parsed requirements
    required_components: List[str] = field(default_factory=list)  # ["ESP32", "USB-C"]
    required_features: List[str] = field(default_factory=list)    # ["WiFi", "3 analog inputs"]

    # Constraints
    board_size_constraint: Optional[Tuple[float, float]] = None  # (width, height) or None
    layer_constraint: Optional[int] = None
    budget_constraint: Optional[float] = None

    # Preferences
    preferred_form_factor: Optional[str] = None  # "compact", "modular", etc.
    preferred_connectors: List[str] = field(default_factory=list)

    # Clarification needed
    clarification_needed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'design_name': self.design_name,
            'original_statements': self.original_statements,
            'required_components': self.required_components,
            'required_features': self.required_features,
            'board_size_constraint': self.board_size_constraint,
            'layer_constraint': self.layer_constraint,
            'budget_constraint': self.budget_constraint,
            'preferred_form_factor': self.preferred_form_factor,
            'preferred_connectors': self.preferred_connectors,
            'clarification_needed': self.clarification_needed,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_empty_clayout(name: str = "New_Design") -> ConstitutionalLayout:
    """Create an empty c_layout with default values."""
    return ConstitutionalLayout(design_name=name)


def create_clayout_from_intent(intent: DesignIntent) -> ConstitutionalLayout:
    """Create a c_layout shell from design intent (to be filled by AI)."""
    clayout = ConstitutionalLayout(
        design_name=intent.design_name,
        user_requirements=intent.original_statements,
    )

    if intent.board_size_constraint:
        clayout.board.width_mm = intent.board_size_constraint[0]
        clayout.board.height_mm = intent.board_size_constraint[1]

    if intent.layer_constraint:
        clayout.board.layer_count = intent.layer_constraint

    return clayout
