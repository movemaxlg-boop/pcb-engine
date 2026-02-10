"""
Design Decision Pipeline
=========================

This is the BRAIN that transforms a circuit description into
a complete, verified PCB design strategy.

The pipeline answers:
- WHERE should each component go?
- WHAT order should components be placed?
- HOW should each net be routed?
- WHAT constraints apply to each element?

This module uses ONLY verified rules from verified_design_rules.py.

NEW in v2.0: Integrated with RulesAPI for AI-reviewable validation.
Every design decision now produces RuleReports for external AI review.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum, auto
import math

from .verified_design_rules import (
    get_verified_rules,
    VerifiedDesignRulesEngine,
    CurrentCapacity,
    ConductorSpacing,
)
from .parts_library import PartsLibrary

# NEW: Import Rules API for AI-reviewable validation
from .rules_api import RulesAPI
from .rule_types import (
    RuleReport,
    RuleStatus,
    RuleCategory,
    DesignReviewReport,
)
from .feedback import AIFeedbackProcessor


# =============================================================================
# DESIGN CONTEXT - Input to the pipeline
# =============================================================================

@dataclass
class ComponentSpec:
    """Specification of a component in the design."""
    ref_des: str                    # Reference designator (U1, C1, R1, etc.)
    part_number: str                # Part number
    category: str                   # IC, CAPACITOR, RESISTOR, etc.
    subcategory: str = ""           # LDO, BUCK, CERAMIC, etc.
    value: float = 0.0              # For passives (Farads, Ohms, Henries)
    package: str = ""               # Package type

    # Electrical characteristics
    voltage_rating: float = 0.0     # Max voltage
    current_rating: float = 0.0     # Max current
    power_dissipation: float = 0.0  # Watts

    # Connectivity
    nets: Dict[str, str] = field(default_factory=dict)  # pin_name -> net_name

    # Thermal
    theta_ja: float = 0.0           # Thermal resistance


@dataclass
class NetSpec:
    """Specification of a net in the design."""
    name: str
    net_type: str = "SIGNAL"        # POWER, GND, SIGNAL, CLOCK, DIFFERENTIAL, ANALOG
    voltage: float = 0.0            # Voltage level
    current: float = 0.0            # Expected current
    frequency: float = 0.0          # Signal frequency (Hz)
    is_critical: bool = False       # High-priority routing
    differential_pair: str = ""     # Name of paired net (for differential)


@dataclass
class BoardSpec:
    """Specification of the PCB."""
    width_mm: float
    height_mm: float
    layer_count: int = 2
    copper_oz: float = 1.0
    thickness_mm: float = 1.6


@dataclass
class DesignContext:
    """Complete design context - input to the pipeline."""
    board: BoardSpec
    components: List[ComponentSpec]
    nets: List[NetSpec]
    design_name: str = "untitled"

    # Constraints from user
    max_voltage: float = 0.0        # Will be calculated if not set
    min_clearance_mm: float = 0.0   # Override if needed


# =============================================================================
# DESIGN DECISIONS - Output from the pipeline
# =============================================================================

class PlacementZone(Enum):
    """Zones on the PCB for component placement."""
    POWER_INPUT = auto()            # Power entry point
    POWER_REGULATION = auto()       # Regulators
    MCU_CORE = auto()               # MCU and immediate support
    ANALOG = auto()                 # Analog section
    DIGITAL_IO = auto()             # Digital interfaces
    CONNECTOR_AREA = auto()         # External connectors
    GENERAL = auto()                # Everything else


@dataclass
class PlacementDecision:
    """Placement decision for a component."""
    ref_des: str
    zone: PlacementZone
    placement_order: int            # 1 = first to place
    constraints: Dict[str, Any] = field(default_factory=dict)
    # Constraints examples:
    # - "near": ["U1"] - place near U1
    # - "max_distance_mm": 5.0
    # - "edge": "left" - place at left edge
    # - "thermal_area_mm2": 100 - needs thermal pad
    reasoning: str = ""             # Why this decision


@dataclass
class RoutingDecision:
    """Routing decision for a net."""
    net_name: str
    routing_priority: int           # 1 = first to route
    trace_width_mm: float
    clearance_mm: float
    layer: str = "F.Cu"             # Preferred layer
    constraints: Dict[str, Any] = field(default_factory=dict)
    # Constraints examples:
    # - "impedance_ohm": 50
    # - "length_match_group": "USB"
    # - "max_length_mm": 50
    # - "avoid_zones": ["ANALOG"]
    reasoning: str = ""


@dataclass
class DesignPlan:
    """Complete design plan - output from the pipeline."""
    placement_decisions: List[PlacementDecision]
    routing_decisions: List[RoutingDecision]
    design_rules: Dict[str, float]  # DRC rules to apply
    warnings: List[str]             # Design warnings
    recommendations: List[str]      # Improvement suggestions

    # NEW: AI-reviewable validation reports
    design_review: Optional['DesignReviewReport'] = None
    rule_reports: List['RuleReport'] = field(default_factory=list)


# =============================================================================
# DESIGN PIPELINE - The decision-making engine
# =============================================================================

class DesignPipeline:
    """
    The main design decision pipeline.

    This takes a DesignContext and produces a DesignPlan using
    verified engineering rules.

    NEW in v2.0: Integrated with RulesAPI for AI-reviewable validation.
    Use generate_ai_review() to get reports for external AI agents.
    """

    def __init__(self, enable_ai_review: bool = True):
        """
        Initialize the design pipeline.

        Args:
            enable_ai_review: If True, generate AI-reviewable reports
        """
        self.rules = get_verified_rules()
        self.parts_lib = PartsLibrary()
        self.enable_ai_review = enable_ai_review

        # NEW: RulesAPI for AI-reviewable validation
        self.rules_api = RulesAPI()
        self.feedback_processor = AIFeedbackProcessor()
        self.current_review: Optional[DesignReviewReport] = None

        try:
            self.parts_lib.load_defaults()
        except (IOError, FileNotFoundError, KeyError, ValueError) as e:
            # Parts library is optional - log but continue
            print(f"Note: Parts library not loaded ({type(e).__name__}): {e}")

    def create_design_plan(self, context: DesignContext) -> DesignPlan:
        """
        Main entry point - create a complete design plan.

        Args:
            context: The design context with all components and nets

        Returns:
            DesignPlan with placement and routing decisions
        """
        # Step 1: Analyze the design
        analysis = self._analyze_design(context)

        # Step 2: Determine placement decisions
        placement_decisions = self._decide_placement(context, analysis)

        # Step 3: Determine routing decisions
        routing_decisions = self._decide_routing(context, analysis)

        # Step 4: Calculate DRC rules
        drc_rules = self._calculate_drc_rules(context, analysis)

        # Step 5: Generate warnings and recommendations
        warnings = self._generate_warnings(context, analysis)
        recommendations = self._generate_recommendations(context, analysis)

        # Step 6: NEW - Generate AI-reviewable design review
        design_review = None
        rule_reports = []
        if self.enable_ai_review:
            design_review, rule_reports = self._generate_ai_review(
                context, analysis, placement_decisions, routing_decisions
            )
            self.current_review = design_review

            # Add reports to feedback processor for AI review
            for report in rule_reports:
                self.feedback_processor.add_report(report)

        return DesignPlan(
            placement_decisions=placement_decisions,
            routing_decisions=routing_decisions,
            design_rules=drc_rules,
            warnings=warnings,
            recommendations=recommendations,
            design_review=design_review,
            rule_reports=rule_reports,
        )

    # -------------------------------------------------------------------------
    # STEP 1: ANALYSIS
    # -------------------------------------------------------------------------

    def _analyze_design(self, context: DesignContext) -> Dict:
        """Analyze the design to understand its characteristics."""
        analysis = {
            "has_power_input": False,
            "has_regulators": False,
            "has_mcu": False,
            "has_analog": False,
            "has_high_speed": False,
            "has_usb": False,
            "max_voltage": 0.0,
            "max_current": 0.0,
            "power_components": [],
            "analog_components": [],
            "digital_components": [],
            "connector_components": [],
            "decoupling_caps": [],
        }

        # Analyze components
        for comp in context.components:
            cat = comp.category.upper()
            subcat = comp.subcategory.upper()

            # Track max voltage
            if comp.voltage_rating > analysis["max_voltage"]:
                analysis["max_voltage"] = comp.voltage_rating

            # Track max current
            if comp.current_rating > analysis["max_current"]:
                analysis["max_current"] = comp.current_rating

            # Categorize components
            if "CONNECTOR" in cat and "POWER" in subcat:
                analysis["has_power_input"] = True
                analysis["power_components"].append(comp.ref_des)

            if "REGULATOR" in cat or "LDO" in subcat or "BUCK" in subcat:
                analysis["has_regulators"] = True
                analysis["power_components"].append(comp.ref_des)

            if "MCU" in cat or "PROCESSOR" in cat:
                analysis["has_mcu"] = True
                analysis["digital_components"].append(comp.ref_des)

            if "ADC" in subcat or "OPAMP" in cat or "ANALOG" in cat:
                analysis["has_analog"] = True
                analysis["analog_components"].append(comp.ref_des)

            if "CONNECTOR" in cat:
                analysis["connector_components"].append(comp.ref_des)

            # Identify decoupling caps (100nF ceramic)
            if cat == "CAPACITOR" and abs(comp.value - 100e-9) < 1e-9:
                analysis["decoupling_caps"].append(comp.ref_des)

        # Analyze nets
        for net in context.nets:
            if "USB" in net.name.upper():
                analysis["has_usb"] = True
                analysis["has_high_speed"] = True

            if net.frequency > 50e6:
                analysis["has_high_speed"] = True

            if net.net_type.upper() == "ANALOG":
                analysis["has_analog"] = True

        # Use context max_voltage if set, otherwise use analyzed
        if context.max_voltage > 0:
            analysis["max_voltage"] = context.max_voltage

        return analysis

    # -------------------------------------------------------------------------
    # STEP 2: PLACEMENT DECISIONS
    # -------------------------------------------------------------------------

    def _decide_placement(self, context: DesignContext,
                          analysis: Dict) -> List[PlacementDecision]:
        """Decide placement order and zones for all components."""
        decisions = []
        order = 1

        # Priority 1: Power input connector
        for comp in context.components:
            if comp.ref_des in analysis.get("power_components", []):
                if "CONNECTOR" in comp.category.upper():
                    decisions.append(PlacementDecision(
                        ref_des=comp.ref_des,
                        zone=PlacementZone.POWER_INPUT,
                        placement_order=order,
                        constraints={
                            "edge": True,
                            "away_from_analog_mm": 20.0,
                        },
                        reasoning="Power input connector: place at board edge, "
                                  "away from sensitive analog circuits"
                    ))
                    order += 1

        # Priority 2: Regulators
        for comp in context.components:
            cat = comp.category.upper()
            subcat = comp.subcategory.upper()
            if "REGULATOR" in cat or "LDO" in subcat or "BUCK" in subcat:
                decisions.append(PlacementDecision(
                    ref_des=comp.ref_des,
                    zone=PlacementZone.POWER_REGULATION,
                    placement_order=order,
                    constraints={
                        "near_power_input": True,
                        "thermal_area_mm2": 100 if comp.power_dissipation > 0.5 else 0,
                        "input_cap_distance_mm": self.rules.switching_regulator.MAX_CRITICAL_LOOP_LENGTH_MM
                            if "BUCK" in subcat else 5.0,
                    },
                    reasoning=f"Voltage regulator: place near power input. "
                              f"Input cap within {self.rules.decoupling.MAX_DISTANCE_TO_VCC_PIN_MM}mm"
                ))
                order += 1

        # Priority 3: MCU
        for comp in context.components:
            if "MCU" in comp.category.upper() or "PROCESSOR" in comp.category.upper():
                decisions.append(PlacementDecision(
                    ref_des=comp.ref_des,
                    zone=PlacementZone.MCU_CORE,
                    placement_order=order,
                    constraints={
                        "central": True,
                        "decoupling_per_vcc": True,
                    },
                    reasoning="MCU: place centrally for shortest average trace "
                              "length to all peripherals"
                ))
                order += 1

        # Priority 4: Crystal (if MCU present)
        for comp in context.components:
            if "CRYSTAL" in comp.category.upper() or "OSCILLATOR" in comp.category.upper():
                mcu_ref = next((d.ref_des for d in decisions
                               if d.zone == PlacementZone.MCU_CORE), None)
                decisions.append(PlacementDecision(
                    ref_des=comp.ref_des,
                    zone=PlacementZone.MCU_CORE,
                    placement_order=order,
                    constraints={
                        "near": [mcu_ref] if mcu_ref else [],
                        "max_distance_mm": self.rules.crystal.MAX_DISTANCE_TO_MCU_MM,
                        "ground_plane_under": True,
                    },
                    reasoning=f"Crystal: place within {self.rules.crystal.MAX_DISTANCE_TO_MCU_MM}mm "
                              f"of MCU oscillator pins. Keep ground plane solid underneath."
                ))
                order += 1

        # Priority 5: Decoupling capacitors
        for ref_des in analysis.get("decoupling_caps", []):
            decisions.append(PlacementDecision(
                ref_des=ref_des,
                zone=PlacementZone.MCU_CORE,  # Near the IC they decouple
                placement_order=order,
                constraints={
                    "near_vcc_pin": True,
                    "max_distance_mm": self.rules.decoupling.MAX_DISTANCE_TO_VCC_PIN_MM,
                    "via_to_gnd_distance_mm": self.rules.decoupling.MAX_VIA_DISTANCE_FROM_CAP_MM,
                },
                reasoning=f"Decoupling cap: place within {self.rules.decoupling.MAX_DISTANCE_TO_VCC_PIN_MM}mm "
                          f"of IC VCC pin. Via to ground within {self.rules.decoupling.MAX_VIA_DISTANCE_FROM_CAP_MM}mm"
            ))
            order += 1

        # Priority 6: Connectors (at edges)
        for comp in context.components:
            if "CONNECTOR" in comp.category.upper() and comp.ref_des not in [d.ref_des for d in decisions]:
                zone = PlacementZone.DIGITAL_IO
                if "USB" in comp.subcategory.upper():
                    constraints = {
                        "edge": True,
                        "esd_protection_distance_mm": self.rules.usb2.ESD_MAX_DISTANCE_FROM_CONNECTOR_MM,
                    }
                else:
                    constraints = {"edge": True}

                decisions.append(PlacementDecision(
                    ref_des=comp.ref_des,
                    zone=zone,
                    placement_order=order,
                    constraints=constraints,
                    reasoning="Connector: place at board edge for external access"
                ))
                order += 1

        # Priority 7: Analog components (if present)
        for ref_des in analysis.get("analog_components", []):
            if ref_des not in [d.ref_des for d in decisions]:
                decisions.append(PlacementDecision(
                    ref_des=ref_des,
                    zone=PlacementZone.ANALOG,
                    placement_order=order,
                    constraints={
                        "isolate_from_digital_mm": self.rules.analog.MIN_SEPARATION_FROM_DIGITAL_MM,
                        "guard_ring": self.rules.analog.USE_GUARD_RING,
                    },
                    reasoning=f"Analog component: isolate from digital by "
                              f"{self.rules.analog.MIN_SEPARATION_FROM_DIGITAL_MM}mm"
                ))
                order += 1

        # Priority 8: Remaining components
        for comp in context.components:
            if comp.ref_des not in [d.ref_des for d in decisions]:
                decisions.append(PlacementDecision(
                    ref_des=comp.ref_des,
                    zone=PlacementZone.GENERAL,
                    placement_order=order,
                    constraints={},
                    reasoning="General component placement"
                ))
                order += 1

        return decisions

    # -------------------------------------------------------------------------
    # STEP 3: ROUTING DECISIONS
    # -------------------------------------------------------------------------

    def _decide_routing(self, context: DesignContext,
                        analysis: Dict) -> List[RoutingDecision]:
        """Decide routing parameters for all nets."""
        decisions = []
        priority = 1

        # Sort nets by routing priority
        sorted_nets = self._sort_nets_by_priority(context.nets)

        for net in sorted_nets:
            decision = self._decide_single_net_routing(net, context, analysis, priority)
            decisions.append(decision)
            priority += 1

        return decisions

    def _sort_nets_by_priority(self, nets: List[NetSpec]) -> List[NetSpec]:
        """Sort nets by routing priority."""
        def get_priority(net: NetSpec) -> int:
            name_upper = net.name.upper()
            type_upper = net.net_type.upper()

            # Power and ground first
            if name_upper in ["GND", "VSS", "GROUND"]:
                return 1
            if name_upper in ["VCC", "VDD", "3V3", "5V", "12V"] or type_upper == "POWER":
                return 2

            # Clock signals
            if "CLK" in name_upper or "CLOCK" in name_upper:
                return 3

            # Reset signals
            if "RST" in name_upper or "RESET" in name_upper:
                return 4

            # Differential pairs
            if net.differential_pair:
                return 5

            # USB signals
            if "USB" in name_upper:
                return 6

            # High-speed signals
            if net.frequency > 50e6:
                return 7

            # Analog signals
            if type_upper == "ANALOG":
                return 8

            # Regular signals
            return 10

        return sorted(nets, key=get_priority)

    def _decide_single_net_routing(self, net: NetSpec, context: DesignContext,
                                   analysis: Dict, priority: int) -> RoutingDecision:
        """Decide routing for a single net."""
        name_upper = net.name.upper()
        type_upper = net.net_type.upper()

        # Default trace width
        if net.current > 0:
            trace_width = self.rules.get_trace_width(net.current, is_power=True)
        else:
            trace_width = self.rules.fabrication.MIN_TRACE_WIDTH_MM

        # Ensure minimum width
        trace_width = max(trace_width, self.rules.fabrication.MIN_TRACE_WIDTH_MM)

        # Calculate clearance
        voltage = net.voltage if net.voltage > 0 else analysis.get("max_voltage", 12.0)
        clearance = self.rules.get_clearance(voltage)

        # Build constraints
        constraints = {}
        reasoning = ""

        # Power nets
        if name_upper in ["GND", "VSS"]:
            constraints["use_plane"] = True
            constraints["plane_layer"] = "B.Cu" if context.board.layer_count == 2 else "In1.Cu"
            reasoning = "Ground: use solid ground plane"

        elif type_upper == "POWER" or name_upper in ["VCC", "VDD", "3V3", "5V"]:
            constraints["min_width_mm"] = max(0.5, trace_width)
            reasoning = f"Power net: use wide trace ({trace_width:.2f}mm) for current capacity"

        # Clock signals
        elif "CLK" in name_upper or "CLOCK" in name_upper:
            constraints["max_parallel_length_mm"] = 10.0
            constraints["isolation_mm"] = trace_width * 3
            constraints["minimize_length"] = True
            reasoning = "Clock: minimize length, avoid parallel routes with other signals"

        # USB differential pairs
        elif "USB" in name_upper and ("+" in name_upper or "-" in name_upper or "D" in name_upper):
            constraints["impedance_ohm"] = self.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM
            constraints["length_match_group"] = "USB"
            constraints["max_length_mm"] = self.rules.usb2.MAX_TRACE_LENGTH_MM
            constraints["max_length_mismatch_mm"] = self.rules.usb2.MAX_LENGTH_MISMATCH_MM
            constraints["max_vias"] = self.rules.usb2.MAX_VIAS_IN_DIFFERENTIAL_PATH
            reasoning = f"USB: {self.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM}ohm impedance, " \
                        f"match within {self.rules.usb2.MAX_LENGTH_MISMATCH_MM}mm"

        # Analog signals
        elif type_upper == "ANALOG":
            constraints["max_length_mm"] = self.rules.analog.MAX_ANALOG_TRACE_LENGTH_MM
            constraints["guard_traces"] = self.rules.analog.GUARD_TRACES_FOR_HIGH_IMPEDANCE
            constraints["avoid_digital"] = True
            reasoning = "Analog: short route, guard traces, away from digital noise"

        # Reset signals
        elif "RST" in name_upper or "RESET" in name_upper:
            constraints["max_length_mm"] = 20.0
            constraints["guard_traces"] = True
            reasoning = "Reset: keep short to minimize noise pickup"

        else:
            reasoning = "Standard signal routing"

        # Determine layer
        layer = self._decide_layer(net, context, analysis)

        return RoutingDecision(
            net_name=net.name,
            routing_priority=priority,
            trace_width_mm=trace_width,
            clearance_mm=clearance,
            layer=layer,
            constraints=constraints,
            reasoning=reasoning,
        )

    def _decide_layer(self, net: NetSpec, context: DesignContext,
                      analysis: Dict) -> str:
        """Decide which layer to route a net on."""
        layers = context.board.layer_count
        name_upper = net.name.upper()
        type_upper = net.net_type.upper()

        if layers == 2:
            if name_upper == "GND":
                return "B.Cu"  # Ground on bottom
            return "F.Cu"  # Everything else on top

        elif layers == 4:
            if name_upper == "GND":
                return "In1.Cu"  # Ground on inner layer 1
            if type_upper == "POWER":
                return "In2.Cu"  # Power on inner layer 2
            if net.frequency > 50e6:  # High-speed on top (closest to ground)
                return "F.Cu"
            return "F.Cu"  # Default to top

        return "F.Cu"

    # -------------------------------------------------------------------------
    # STEP 4: DRC RULES
    # -------------------------------------------------------------------------

    def _calculate_drc_rules(self, context: DesignContext,
                             analysis: Dict) -> Dict[str, float]:
        """Calculate DRC rules for the design."""
        max_v = analysis.get("max_voltage", 12.0)

        return {
            "trace_width_min_mm": self.rules.fabrication.MIN_TRACE_WIDTH_MM,
            "clearance_min_mm": max(
                self.rules.fabrication.MIN_SPACING_MM,
                self.rules.conductor_spacing.get_spacing(max_v)
            ),
            "via_drill_min_mm": self.rules.fabrication.MIN_VIA_DRILL_MM,
            "via_pad_min_mm": self.rules.fabrication.MIN_VIA_PAD_MM,
            "annular_ring_min_mm": self.rules.fabrication.MIN_ANNULAR_RING_MM,
            "silkscreen_width_min_mm": self.rules.fabrication.MIN_SILKSCREEN_WIDTH_MM,
            "silkscreen_pad_clearance_mm": self.rules.fabrication.SILKSCREEN_PAD_CLEARANCE_MM,
            "solder_mask_dam_min_mm": self.rules.fabrication.MIN_SOLDER_MASK_DAM_MM,
        }

    # -------------------------------------------------------------------------
    # STEP 5: WARNINGS AND RECOMMENDATIONS
    # -------------------------------------------------------------------------

    def _generate_warnings(self, context: DesignContext,
                           analysis: Dict) -> List[str]:
        """Generate design warnings."""
        warnings = []

        # Check for missing decoupling
        mcu_count = sum(1 for c in context.components
                       if "MCU" in c.category.upper())
        decoupling_count = len(analysis.get("decoupling_caps", []))

        if mcu_count > 0 and decoupling_count == 0:
            warnings.append(
                "No 100nF decoupling capacitors found! "
                "Add 100nF ceramic cap to each MCU VCC pin."
            )

        # Check for high power without thermal consideration
        for comp in context.components:
            if comp.power_dissipation > 1.0 and comp.theta_ja > 50:
                tj_estimate = 25 + comp.power_dissipation * comp.theta_ja
                if tj_estimate > 100:
                    warnings.append(
                        f"{comp.ref_des}: Estimated Tj={tj_estimate:.0f}C. "
                        f"Add thermal vias or copper pour for heat dissipation."
                    )

        # Check for analog/digital mixing
        if analysis.get("has_analog") and analysis.get("has_mcu"):
            warnings.append(
                "Mixed analog/digital design detected. "
                "Ensure analog section is physically separated from digital."
            )

        return warnings

    def _generate_recommendations(self, context: DesignContext,
                                   analysis: Dict) -> List[str]:
        """Generate design recommendations."""
        recs = []

        # 2-layer board recommendations
        if context.board.layer_count == 2:
            recs.append(
                "2-layer board: Use bottom layer primarily for ground plane. "
                "Minimize cuts in ground for good signal return paths."
            )

        # USB recommendations
        if analysis.get("has_usb"):
            recs.append(
                f"USB design: Maintain {self.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM}ohm "
                f"differential impedance. Match D+/D- within "
                f"{self.rules.usb2.MAX_LENGTH_MISMATCH_MM}mm."
            )

        # Switching regulator recommendations
        if any("BUCK" in c.subcategory.upper() for c in context.components):
            recs.append(
                "Switching regulator: Minimize hot loop (SW-inductor-Cout-GND-Cin-SW). "
                f"Keep loop under {self.rules.switching_regulator.MAX_CRITICAL_LOOP_LENGTH_MM}mm."
            )

        # Decoupling recommendation
        recs.append(
            f"Decoupling: Place 100nF caps within {self.rules.decoupling.MAX_DISTANCE_TO_VCC_PIN_MM}mm "
            f"of each IC VCC pin. Via to GND within {self.rules.decoupling.MAX_VIA_DISTANCE_FROM_CAP_MM}mm."
        )

        return recs

    # -------------------------------------------------------------------------
    # NEW: AI-REVIEWABLE VALIDATION
    # -------------------------------------------------------------------------

    def _generate_ai_review(
        self,
        context: DesignContext,
        analysis: Dict,
        placement_decisions: List[PlacementDecision],
        routing_decisions: List[RoutingDecision]
    ) -> Tuple[DesignReviewReport, List[RuleReport]]:
        """
        Generate AI-reviewable design review report.

        This creates a DesignReviewReport with RuleReports for each
        validation performed, allowing external AI agents to review,
        validate, correct, or reject outcomes.
        """
        report = DesignReviewReport(design_name=context.design_name)
        all_reports = []

        # Validate thermal design for high-power components
        for comp in context.components:
            if comp.power_dissipation > 0.5:
                rule_report = self.rules_api.validate_thermal_design(
                    power_w=comp.power_dissipation,
                    theta_ja_c_w=comp.theta_ja if comp.theta_ja > 0 else 50.0,
                    max_tj_c=125.0,
                    ambient_c=25.0,
                    num_thermal_vias=0,  # Unknown at this stage
                )
                # Update rule_id to include component reference
                rule_report.rule_id = f"THERMAL_{comp.ref_des}"
                report.add_report(rule_report)
                all_reports.append(rule_report)

        # Validate fabrication specs
        fab_report = self.rules_api.validate_fabrication(
            trace_width_mm=self.rules.fabrication.MIN_TRACE_WIDTH_MM,
            spacing_mm=self.rules.fabrication.MIN_SPACING_MM,
            via_drill_mm=self.rules.fabrication.MIN_VIA_DRILL_MM,
            capability="standard"
        )
        report.add_report(fab_report)
        all_reports.append(fab_report)

        # Validate layer count for design requirements
        freq_mhz = max([n.frequency / 1e6 for n in context.nets if n.frequency > 0], default=10)
        signal_count = len([n for n in context.nets if n.net_type.upper() == "SIGNAL"])
        has_usb = analysis.get("has_usb", False)
        has_ddr = any("DDR" in n.name.upper() for n in context.nets)

        stackup_rec = self.rules_api.recommend_layer_count(
            freq_mhz, signal_count, has_usb, has_ddr
        )
        min_layers = stackup_rec.get('min_layers', 2)

        from .rule_types import create_pass_report, create_fail_report
        if context.board.layer_count >= min_layers:
            layer_report = create_pass_report(
                rule_id="STACKUP_LAYER_COUNT",
                category=RuleCategory.STACKUP,
                source="Design Pipeline Analysis",
                inputs={'layers': context.board.layer_count, 'min_required': min_layers},
                rule_applied=f"Layer count >= {min_layers}",
                threshold=min_layers,
                actual_value=context.board.layer_count,
            )
        else:
            layer_report = create_fail_report(
                rule_id="STACKUP_LAYER_COUNT",
                category=RuleCategory.STACKUP,
                source="Design Pipeline Analysis",
                inputs={'layers': context.board.layer_count, 'min_required': min_layers},
                rule_applied=f"Layer count >= {min_layers}",
                threshold=min_layers,
                actual_value=context.board.layer_count,
                violation=f"Design has {context.board.layer_count} layers but requires {min_layers}",
            )
        report.add_report(layer_report)
        all_reports.append(layer_report)

        # Validate routing decisions for high-speed signals
        for rd in routing_decisions:
            net = next((n for n in context.nets if n.name == rd.net_name), None)
            if net and "USB" in net.name.upper():
                # USB validation
                if rd.constraints.get("impedance_ohm"):
                    target_z = self.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM
                    actual_z = rd.constraints.get("impedance_ohm", 90)
                    tolerance = self.rules.usb2.DIFFERENTIAL_TOLERANCE_PERCENT

                    if abs(actual_z - target_z) / target_z * 100 <= tolerance:
                        usb_report = create_pass_report(
                            rule_id=f"USB_IMPEDANCE_{net.name}",
                            category=RuleCategory.HIGH_SPEED,
                            source="USB 2.0 Specification",
                            inputs={'net': net.name, 'impedance': actual_z},
                            rule_applied=f"Impedance within {target_z}ohm +/-{tolerance}%",
                            threshold=target_z,
                            actual_value=actual_z,
                        )
                    else:
                        usb_report = create_fail_report(
                            rule_id=f"USB_IMPEDANCE_{net.name}",
                            category=RuleCategory.HIGH_SPEED,
                            source="USB 2.0 Specification",
                            inputs={'net': net.name, 'impedance': actual_z},
                            rule_applied=f"Impedance within {target_z}ohm +/-{tolerance}%",
                            threshold=target_z,
                            actual_value=actual_z,
                            violation=f"Impedance {actual_z}ohm outside tolerance",
                        )
                    report.add_report(usb_report)
                    all_reports.append(usb_report)

        # Finalize the report
        report.finalize()

        return report, all_reports

    def get_ai_review_prompt(self) -> str:
        """
        Generate a prompt for external AI to review the design.

        Returns:
            String prompt with all rules needing review
        """
        return self.feedback_processor.generate_ai_review_prompt()

    def process_ai_feedback(self, command: str) -> 'FeedbackResult':
        """
        Process feedback from an external AI agent.

        Args:
            command: Feedback command (ACCEPT, REJECT, CORRECT, OVERRIDE, etc.)

        Returns:
            FeedbackResult with action taken
        """
        from .feedback import FeedbackResult
        return self.feedback_processor.process_command(command)

    def get_review_summary(self) -> Dict[str, Any]:
        """Get a summary of the current design review status."""
        if self.current_review:
            return {
                'design_name': self.current_review.design_name,
                'total_rules': self.current_review.total_rules_checked,
                'passed': self.current_review.passed,
                'failed': self.current_review.failed,
                'warnings': self.current_review.warnings,
                'compliance_score': self.current_review.compliance_score,
                'design_status': self.current_review.design_status,
                'blocking_violations': self.current_review.blocking_violations,
            }
        return {'error': 'No design review available. Run create_design_plan() first.'}

    def export_review_json(self) -> str:
        """Export the design review as JSON for external consumption."""
        if self.current_review:
            return self.current_review.to_json()
        return '{"error": "No design review available"}'


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline() -> DesignPipeline:
    """Create a new design pipeline."""
    return DesignPipeline()


def analyze_design(context: DesignContext) -> DesignPlan:
    """Analyze a design and return the plan."""
    pipeline = DesignPipeline()
    return pipeline.create_design_plan(context)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Simple ESP32 power supply design
    context = DesignContext(
        design_name="ESP32_Power",
        board=BoardSpec(width_mm=50, height_mm=40, layer_count=2),
        components=[
            ComponentSpec(
                ref_des="J1",
                part_number="USB_C_Receptacle",
                category="CONNECTOR",
                subcategory="POWER",
                voltage_rating=5.0,
                current_rating=3.0,
            ),
            ComponentSpec(
                ref_des="U1",
                part_number="AMS1117-3.3",
                category="REGULATOR",
                subcategory="LDO",
                voltage_rating=15.0,
                current_rating=1.0,
                power_dissipation=1.7,  # (5-3.3)*1A
                theta_ja=90.0,
            ),
            ComponentSpec(
                ref_des="U2",
                part_number="ESP32-WROOM-32E",
                category="MCU",
                subcategory="WIFI_MODULE",
                voltage_rating=3.6,
                current_rating=0.5,
            ),
            ComponentSpec(
                ref_des="C1",
                part_number="100nF_0805",
                category="CAPACITOR",
                subcategory="CERAMIC",
                value=100e-9,
            ),
            ComponentSpec(
                ref_des="C2",
                part_number="100nF_0805",
                category="CAPACITOR",
                subcategory="CERAMIC",
                value=100e-9,
            ),
            ComponentSpec(
                ref_des="C3",
                part_number="10uF_0805",
                category="CAPACITOR",
                subcategory="CERAMIC",
                value=10e-6,
            ),
        ],
        nets=[
            NetSpec(name="GND", net_type="GND"),
            NetSpec(name="5V", net_type="POWER", voltage=5.0, current=1.0),
            NetSpec(name="3V3", net_type="POWER", voltage=3.3, current=0.5),
        ],
    )

    # Create the design plan
    pipeline = DesignPipeline()
    plan = pipeline.create_design_plan(context)

    # Print results
    print("=" * 70)
    print("DESIGN PLAN: ESP32_Power")
    print("=" * 70)

    print("\nPLACEMENT ORDER:")
    print("-" * 70)
    for pd in sorted(plan.placement_decisions, key=lambda x: x.placement_order):
        print(f"  {pd.placement_order}. {pd.ref_des:8} Zone: {pd.zone.name:20}")
        print(f"     Reason: {pd.reasoning[:60]}...")

    print("\nROUTING PRIORITIES:")
    print("-" * 70)
    for rd in sorted(plan.routing_decisions, key=lambda x: x.routing_priority):
        print(f"  {rd.routing_priority}. {rd.net_name:10} Width: {rd.trace_width_mm:.2f}mm  "
              f"Clearance: {rd.clearance_mm:.2f}mm")
        if rd.constraints:
            print(f"     Constraints: {rd.constraints}")

    print("\nDRC RULES:")
    print("-" * 70)
    for rule, value in plan.design_rules.items():
        print(f"  {rule}: {value}mm")

    print("\nWARNINGS:")
    print("-" * 70)
    for w in plan.warnings:
        print(f"  ! {w}")

    print("\nRECOMMENDATIONS:")
    print("-" * 70)
    for r in plan.recommendations:
        print(f"  * {r}")
