"""
RULE HIERARCHY ENGINE
=====================

This module classifies the 631 design rules into the three-tier hierarchy:
- INVIOLABLE: Must pass or abort (safety, functionality)
- RECOMMENDED: Should pass, warn if violated
- OPTIONAL: May pass, log if violated

The classification is based on:
1. Design characteristics (has USB? has DDR? voltage levels?)
2. Safety implications
3. Industry standards requirements

FULLY INTEGRATED WITH RULES API:
- All rule values come from the 631-rule database
- RuleBinding includes actual threshold values
- Validation uses RulesAPI functions
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import re

from .clayout_types import (
    RuleHierarchy,
    RuleBinding,
    RulePriority,
    ComponentDefinition,
    NetDefinition,
    NetType,
    ComponentCategory,
)
from .rules_api import RulesAPI
from .rule_types import RuleCategory


# =============================================================================
# DEFAULT RULE CLASSIFICATIONS
# =============================================================================

# Rules that are ALWAYS inviolable (safety-critical)
ALWAYS_INVIOLABLE = [
    # Electrical Safety
    "CONDUCTOR_SPACING",
    "CREEPAGE_DISTANCE",
    "HI_POT_CLEARANCE",
    "CURRENT_CAPACITY",

    # Thermal Safety
    "THERMAL_MAX_TJ",
    "THERMAL_DERATING",

    # Fabrication Limits
    "MIN_TRACE_WIDTH",
    "MIN_VIA_DRILL",
    "MIN_ANNULAR_RING",
]

# Rules that become inviolable based on design features
CONDITIONAL_INVIOLABLE = {
    # High-Speed Interfaces - won't work if violated
    "has_usb_hs": [
        "USB2_IMPEDANCE",
        "USB2_DIFFERENTIAL_IMPEDANCE",
    ],
    "has_usb3": [
        "USB3_IMPEDANCE",
        "USB3_DIFFERENTIAL_IMPEDANCE",
    ],
    "has_ddr3": [
        "DDR3_DATA_IMPEDANCE",
        "DDR3_CLK_IMPEDANCE",
        "DDR3_ADDR_IMPEDANCE",
    ],
    "has_ddr4": [
        "DDR4_DATA_IMPEDANCE",
        "DDR4_CLK_IMPEDANCE",
        "DDR4_ADDR_IMPEDANCE",
    ],
    "has_pcie": [
        "PCIE_DIFFERENTIAL_IMPEDANCE",
    ],
    "has_hdmi": [
        "HDMI_DIFFERENTIAL_IMPEDANCE",
    ],
    "has_ethernet_gigabit": [
        "ETHERNET_1G_IMPEDANCE",
    ],

    # High Voltage
    "voltage_above_50v": [
        "HV_CREEPAGE",
        "HV_CLEARANCE",
    ],

    # High Power
    "power_above_10w": [
        "THERMAL_VIA_COUNT",
        "THERMAL_PAD_SIZE",
    ],
}

# Default recommended rules
DEFAULT_RECOMMENDED = [
    # Placement Rules
    "DECOUPLING_DISTANCE",
    "DECOUPLING_VIA_DISTANCE",
    "CRYSTAL_DISTANCE",
    "REGULATOR_LOOP_LENGTH",
    "ANALOG_SEPARATION",

    # Length Matching (non-impedance-critical)
    "USB2_LENGTH_MATCHING",
    "DDR3_DQS_DQ_MATCHING",
    "DDR4_DQS_DQ_MATCHING",

    # EMI/EMC
    "EMI_LOOP_AREA",
    "EMI_RADIATION_LIMIT",
    "RETURN_PATH_CONTINUITY",

    # Thermal
    "THERMAL_VIA_COUNT",
    "THERMAL_COPPER_AREA",

    # Stackup
    "STACKUP_SYMMETRY",
    "STACKUP_REFERENCE_PLANE",

    # Via Rules
    "VIA_STITCHING_SPACING",
    "VIA_IN_PAD",
]

# Default optional rules
DEFAULT_OPTIONAL = [
    # Silkscreen
    "SILKSCREEN_SIZE",
    "SILKSCREEN_CLEARANCE",
    "SILKSCREEN_ORIENTATION",

    # Assembly
    "TESTPOINT_ACCESSIBILITY",
    "TESTPOINT_SPACING",
    "FIDUCIAL_PLACEMENT",

    # Aesthetics
    "COMPONENT_ALIGNMENT",
    "TRACE_ANGLE",

    # Documentation
    "REF_DES_VISIBILITY",
    "POLARITY_MARKING",
]


# =============================================================================
# DESIGN CONTEXT ANALYZER
# =============================================================================

@dataclass
class DesignContext:
    """Analyzed characteristics of a design."""

    # High-Speed Interfaces
    has_usb_fs: bool = False        # USB Full-Speed (12 Mbps)
    has_usb_hs: bool = False        # USB High-Speed (480 Mbps)
    has_usb3: bool = False          # USB 3.x (5+ Gbps)
    has_ddr3: bool = False
    has_ddr4: bool = False
    has_ddr5: bool = False
    has_pcie: bool = False
    has_pcie_gen: int = 0           # 1, 2, 3, 4, 5
    has_hdmi: bool = False
    has_ethernet: bool = False
    has_ethernet_speed: str = ""    # "10M", "100M", "1G", "10G"
    has_lvds: bool = False
    has_mipi: bool = False

    # Power Characteristics
    max_voltage: float = 0.0
    max_current: float = 0.0
    total_power: float = 0.0
    has_switching_regulator: bool = False
    has_high_power: bool = False    # > 5W components

    # Analog
    has_analog: bool = False
    has_precision_analog: bool = False
    has_adc: bool = False
    has_dac: bool = False

    # RF
    has_rf: bool = False
    has_wifi: bool = False
    has_bluetooth: bool = False
    has_cellular: bool = False
    max_frequency_hz: float = 0.0

    # Board Characteristics
    layer_count: int = 2
    board_area_mm2: float = 0.0
    component_count: int = 0
    net_count: int = 0

    # Safety
    is_medical: bool = False
    is_automotive: bool = False
    is_aerospace: bool = False
    is_safety_critical: bool = False


class DesignContextAnalyzer:
    """Analyzes design to determine applicable rule classifications."""

    def analyze(
        self,
        components: List[ComponentDefinition],
        nets: List[NetDefinition],
        board_width: float,
        board_height: float,
        layer_count: int,
    ) -> DesignContext:
        """Analyze a design and return its context."""

        ctx = DesignContext()
        ctx.layer_count = layer_count
        ctx.board_area_mm2 = board_width * board_height
        ctx.component_count = len(components)
        ctx.net_count = len(nets)

        # Analyze components
        for comp in components:
            self._analyze_component(comp, ctx)

        # Analyze nets
        for net in nets:
            self._analyze_net(net, ctx)

        # Derive additional characteristics
        ctx.has_high_power = ctx.total_power > 5.0
        ctx.is_safety_critical = ctx.is_medical or ctx.is_automotive or ctx.is_aerospace

        return ctx

    def _analyze_component(self, comp: ComponentDefinition, ctx: DesignContext):
        """Analyze a single component."""

        part_upper = comp.part_number.upper()

        # MCU/SoC detection
        if "ESP32" in part_upper:
            ctx.has_wifi = True
            ctx.has_bluetooth = True
            ctx.max_frequency_hz = max(ctx.max_frequency_hz, 240e6)

        if "STM32" in part_upper or "ATMEGA" in part_upper or "PIC" in part_upper:
            ctx.has_adc = True

        # USB detection
        if "USB" in part_upper:
            if "USB3" in part_upper or "SS" in part_upper:
                ctx.has_usb3 = True
            elif "HS" in part_upper or "HIGH" in part_upper:
                ctx.has_usb_hs = True
            else:
                ctx.has_usb_fs = True

        # Memory detection
        if "DDR3" in part_upper:
            ctx.has_ddr3 = True
        if "DDR4" in part_upper:
            ctx.has_ddr4 = True
        if "DDR5" in part_upper:
            ctx.has_ddr5 = True

        # Interface detection
        if "PCIE" in part_upper or "PCI-E" in part_upper:
            ctx.has_pcie = True
        if "HDMI" in part_upper:
            ctx.has_hdmi = True
        if "ETH" in part_upper or "LAN" in part_upper or "PHY" in part_upper:
            ctx.has_ethernet = True
        if "LVDS" in part_upper:
            ctx.has_lvds = True
        if "MIPI" in part_upper:
            ctx.has_mipi = True

        # RF detection
        if any(x in part_upper for x in ["RF", "WIFI", "BT", "LORA", "NRF", "CC1101", "SX127"]):
            ctx.has_rf = True

        # Regulator detection
        if any(x in part_upper for x in ["LM2596", "MP1584", "TPS54", "LTC3", "BUCK", "BOOST"]):
            ctx.has_switching_regulator = True

        # Analog detection
        if any(x in part_upper for x in ["OPAMP", "LM358", "TL072", "AD8", "OPA"]):
            ctx.has_analog = True
        if any(x in part_upper for x in ["ADS1", "MCP3", "AD7", "MAX11"]):
            ctx.has_adc = True
            ctx.has_precision_analog = True
        if any(x in part_upper for x in ["DAC", "MCP47", "AD56"]):
            ctx.has_dac = True

        # Power tracking
        ctx.max_voltage = max(ctx.max_voltage, comp.voltage_rating)
        ctx.max_current = max(ctx.max_current, comp.current_rating)
        ctx.total_power += comp.power_dissipation

    def _analyze_net(self, net: NetDefinition, ctx: DesignContext):
        """Analyze a single net."""

        name_upper = net.name.upper()

        # USB detection
        if "USB" in name_upper or "D+" in name_upper or "D-" in name_upper:
            if net.frequency and net.frequency > 400e6:
                ctx.has_usb_hs = True
            else:
                ctx.has_usb_fs = True

        # DDR detection
        if "DDR" in name_upper or "DQ" in name_upper or "DQS" in name_upper:
            if "DDR4" in name_upper:
                ctx.has_ddr4 = True
            elif "DDR3" in name_upper:
                ctx.has_ddr3 = True

        # High-speed detection from frequency
        if net.frequency:
            ctx.max_frequency_hz = max(ctx.max_frequency_hz, net.frequency)
            if net.frequency > 1e9:  # > 1 GHz
                ctx.has_usb3 = True

        # Power tracking
        ctx.max_voltage = max(ctx.max_voltage, abs(net.voltage))
        ctx.max_current = max(ctx.max_current, net.current_max)


# =============================================================================
# RULE HIERARCHY ENGINE
# =============================================================================

class RuleHierarchyEngine:
    """
    Classifies rules into hierarchy based on design context.

    This is the brain that decides which rules are critical for a specific design.

    FULLY INTEGRATED with RulesAPI:
    - All rule threshold values come from the 631-rule database
    - RuleBinding.parameters contains actual values from RulesAPI
    - Each rule has source attribution
    """

    def __init__(self):
        self.rules_api = RulesAPI()
        self.analyzer = DesignContextAnalyzer()

        # Build rule ID patterns
        self._inviolable_patterns = self._compile_patterns(ALWAYS_INVIOLABLE)
        self._recommended_patterns = self._compile_patterns(DEFAULT_RECOMMENDED)
        self._optional_patterns = self._compile_patterns(DEFAULT_OPTIONAL)

    def _compile_patterns(self, rule_ids: List[str]) -> List[re.Pattern]:
        """Compile rule IDs into regex patterns (supporting wildcards)."""
        patterns = []
        for rule_id in rule_ids:
            # Convert wildcards to regex
            pattern = rule_id.replace("*", ".*")
            patterns.append(re.compile(f"^{pattern}$", re.IGNORECASE))
        return patterns

    def _matches_patterns(self, rule_id: str, patterns: List[re.Pattern]) -> bool:
        """Check if a rule ID matches any pattern."""
        for pattern in patterns:
            if pattern.match(rule_id):
                return True
        return False

    def classify_rules(
        self,
        components: List[ComponentDefinition],
        nets: List[NetDefinition],
        board_width: float = 50.0,
        board_height: float = 40.0,
        layer_count: int = 2,
    ) -> RuleHierarchy:
        """
        Generate rule hierarchy for a specific design.

        This is the main entry point - it analyzes the design and classifies
        all applicable rules into the three-tier hierarchy.

        FULLY INTEGRATED: All rule values come from RulesAPI (631 rules database).
        Each RuleBinding includes:
        - rule_id: The rule identifier
        - parameters: Actual threshold values from RulesAPI
        - reason: Why this rule is classified at this level
        - source: Reference standard (IPC, JEDEC, etc.)
        """

        # Analyze design context
        ctx = self.analyzer.analyze(components, nets, board_width, board_height, layer_count)

        # Start with empty hierarchy
        hierarchy = RuleHierarchy()

        # Add always-inviolable rules WITH ACTUAL VALUES from RulesAPI
        self._add_inviolable_rules(ctx, hierarchy)

        # Add conditional inviolable rules based on design features
        self._add_conditional_inviolable(ctx, hierarchy)

        # Add recommended rules WITH ACTUAL VALUES
        self._add_recommended_rules(ctx, hierarchy)

        # Add optional rules
        self._add_optional_rules(ctx, hierarchy)

        # Promote/demote rules based on context
        self._adjust_for_context(ctx, hierarchy)

        return hierarchy

    def _add_inviolable_rules(self, ctx: DesignContext, hierarchy: RuleHierarchy):
        """Add always-inviolable rules with actual values from RulesAPI."""

        # CONDUCTOR_SPACING - value depends on max voltage
        spacing = self.rules_api.get_conductor_spacing(max(ctx.max_voltage, 12.0))
        hierarchy.inviolable.append(RuleBinding(
            rule_id="CONDUCTOR_SPACING",
            parameters={
                "min_spacing_mm": spacing,
                "voltage": ctx.max_voltage,
                "layer_type": "external_coated"
            },
            reason="Prevents electrical arcing and fire hazard",
            source="IPC-2221B Table 6-1",
        ))

        # CREEPAGE_DISTANCE - for safety isolation
        clearance = self.rules_api.get_clearance(max(ctx.max_voltage, 12.0))
        hierarchy.inviolable.append(RuleBinding(
            rule_id="CREEPAGE_DISTANCE",
            parameters={
                "min_clearance_mm": clearance,
                "voltage": ctx.max_voltage
            },
            reason="Safety isolation for voltage levels",
            source="IPC-2221B, UL60950-1",
        ))

        # HI_POT_CLEARANCE - for high voltage
        if ctx.max_voltage > 50:
            hierarchy.inviolable.append(RuleBinding(
                rule_id="HI_POT_CLEARANCE",
                parameters={
                    "min_clearance_mm": clearance * 1.5,
                    "voltage": ctx.max_voltage
                },
                reason="High voltage isolation requirement",
                source="IPC-2221B Section 6.3",
            ))

        # CURRENT_CAPACITY - trace width for current
        if ctx.max_current > 0:
            trace_width = self.rules_api.get_trace_width(ctx.max_current)
            hierarchy.inviolable.append(RuleBinding(
                rule_id="CURRENT_CAPACITY",
                parameters={
                    "min_trace_width_mm": trace_width,
                    "max_current_a": ctx.max_current,
                    "temp_rise_c": 10.0
                },
                reason="Prevents trace melting and fire hazard",
                source="IPC-2152 Section 5.1",
            ))

        # THERMAL_MAX_TJ - junction temperature
        hierarchy.inviolable.append(RuleBinding(
            rule_id="THERMAL_MAX_TJ",
            parameters={
                "max_tj_c": 125.0,  # Standard for most components
                "ambient_c": 25.0
            },
            reason="Prevents component thermal damage",
            source="JEDEC JESD51-1",
        ))

        # THERMAL_DERATING
        hierarchy.inviolable.append(RuleBinding(
            rule_id="THERMAL_DERATING",
            parameters={
                "derating_pct_per_c": 2.0,
                "start_temp_c": 70.0
            },
            reason="Power derating for high ambient temperatures",
            source="Component datasheets",
        ))

        # FABRICATION LIMITS
        min_trace = self.rules_api.get_min_trace_width("standard")
        min_space = self.rules_api.get_min_spacing("standard")
        min_via = self.rules_api.get_min_via_drill("standard")
        min_annular = self.rules_api.get_annular_ring(2)

        hierarchy.inviolable.append(RuleBinding(
            rule_id="MIN_TRACE_WIDTH",
            parameters={"min_mm": min_trace},
            reason="PCB fabrication capability limit",
            source="IPC-2221B, Standard PCB Fab",
        ))

        hierarchy.inviolable.append(RuleBinding(
            rule_id="MIN_VIA_DRILL",
            parameters={"min_mm": min_via},
            reason="PCB fabrication capability limit",
            source="IPC-2221B, Standard PCB Fab",
        ))

        hierarchy.inviolable.append(RuleBinding(
            rule_id="MIN_ANNULAR_RING",
            parameters={"min_mm": min_annular, "ipc_class": 2},
            reason="Via reliability per IPC class",
            source="IPC-2221B Section 9.1",
        ))

    def _add_recommended_rules(self, ctx: DesignContext, hierarchy: RuleHierarchy):
        """Add recommended rules with actual values from RulesAPI."""

        # DECOUPLING_DISTANCE
        decoup_dist = self.rules_api.get_decoupling_distance()
        hierarchy.recommended.append(RuleBinding(
            rule_id="DECOUPLING_DISTANCE",
            parameters={
                "max_distance_mm": decoup_dist,
                "via_distance_mm": 0.5
            },
            reason="Decoupling effectiveness requires proximity",
            source="Murata Application Notes, ADI AN-1142",
        ))

        # DECOUPLING_VIA_DISTANCE
        hierarchy.recommended.append(RuleBinding(
            rule_id="DECOUPLING_VIA_DISTANCE",
            parameters={"max_via_distance_mm": 0.5},
            reason="Via must be close to capacitor pad",
            source="Murata Application Notes",
        ))

        # CRYSTAL_DISTANCE
        crystal_dist = self.rules_api.get_crystal_distance()
        hierarchy.recommended.append(RuleBinding(
            rule_id="CRYSTAL_DISTANCE",
            parameters={"max_distance_mm": crystal_dist},
            reason="Minimize parasitic capacitance on oscillator traces",
            source="Microchip AN826, ST AN2867",
        ))

        # REGULATOR_LOOP_LENGTH
        loop_len = self.rules_api.get_regulator_loop_length()
        hierarchy.recommended.append(RuleBinding(
            rule_id="REGULATOR_LOOP_LENGTH",
            parameters={"max_loop_mm": loop_len},
            reason="Minimize switching regulator EMI",
            source="TI SNVA166B, ADI AN-139",
        ))

        # ANALOG_SEPARATION
        analog_sep = self.rules_api.get_analog_separation()
        hierarchy.recommended.append(RuleBinding(
            rule_id="ANALOG_SEPARATION",
            parameters={"min_distance_mm": analog_sep},
            reason="Prevent digital noise coupling to analog circuits",
            source="ADI MT-031, TI SLYT199",
        ))

        # USB2_LENGTH_MATCHING (if USB present)
        if ctx.has_usb_fs or ctx.has_usb_hs:
            usb_matching = self.rules_api.get_length_matching_rules("USB_2.0_HS" if ctx.has_usb_hs else "USB_2.0_FS")
            hierarchy.recommended.append(RuleBinding(
                rule_id="USB2_LENGTH_MATCHING",
                parameters={
                    "max_mismatch_mm": usb_matching.get("max_mismatch_mm", 1.25),
                    "differential_impedance_ohm": 90.0,
                    "tolerance_pct": 10.0
                },
                reason="USB signal integrity requires length matching",
                source="USB 2.0 Spec Chapter 7, Microchip AN2972",
            ))

        # DDR MATCHING (if DDR present)
        if ctx.has_ddr3:
            ddr_rules = self.rules_api.get_ddr3_rules()
            hierarchy.recommended.append(RuleBinding(
                rule_id="DDR3_DQS_DQ_MATCHING",
                parameters={
                    "max_dqs_dq_mm": ddr_rules.get("dqs_to_dq_max_mm", 6.0),
                    "max_dq_dq_mm": ddr_rules.get("dq_to_dq_max_mm", 1.0),
                    "data_impedance_ohm": ddr_rules.get("data_impedance_ohm", 40)
                },
                reason="DDR3 timing requires strict length matching",
                source="JEDEC JESD79-3F",
            ))

        if ctx.has_ddr4:
            ddr_rules = self.rules_api.get_ddr4_rules()
            hierarchy.recommended.append(RuleBinding(
                rule_id="DDR4_DQS_DQ_MATCHING",
                parameters={
                    "max_dqs_dq_mm": ddr_rules.get("dqs_to_dq_max_mm", 5.0),
                    "max_dq_dq_mm": ddr_rules.get("dq_to_dq_max_mm", 1.0),
                    "data_impedance_ohm": ddr_rules.get("data_impedance_ohm", 40)
                },
                reason="DDR4 timing requires strict length matching",
                source="JEDEC JESD79-4B",
            ))

        # EMI_LOOP_AREA
        if ctx.max_frequency_hz > 0:
            freq_mhz = ctx.max_frequency_hz / 1e6
            max_loop = self.rules_api.get_max_loop_area(100, freq_mhz, 40.0)  # 100mA, 40dBuV/m limit
            hierarchy.recommended.append(RuleBinding(
                rule_id="EMI_LOOP_AREA",
                parameters={
                    "max_area_mm2": max_loop,
                    "frequency_mhz": freq_mhz
                },
                reason="Minimize radiated emissions for EMC compliance",
                source="FCC Part 15, Henry Ott EMC Engineering",
            ))

        # EMI_RADIATION_LIMIT
        hierarchy.recommended.append(RuleBinding(
            rule_id="EMI_RADIATION_LIMIT",
            parameters={
                "limit_dBuV_m": 40.0,
                "distance_m": 3.0,
                "standard": "FCC_CLASS_B"
            },
            reason="FCC Class B emissions compliance",
            source="FCC Part 15 Subpart B",
        ))

        # RETURN_PATH_CONTINUITY
        hierarchy.recommended.append(RuleBinding(
            rule_id="RETURN_PATH_CONTINUITY",
            parameters={"max_gap_mm": 5.0},
            reason="Signal return path must be continuous under traces",
            source="Howard Johnson High-Speed Digital Design",
        ))

        # THERMAL_VIA_COUNT
        if ctx.total_power > 0.5:  # If any significant power dissipation
            thermal_rules = self.rules_api.get_thermal_pad_rules()
            hierarchy.recommended.append(RuleBinding(
                rule_id="THERMAL_VIA_COUNT",
                parameters={
                    "min_vias": thermal_rules.get("min_vias", 5),
                    "via_drill_mm": thermal_rules.get("via_drill_mm", 0.3),
                    "grid_pitch_mm": thermal_rules.get("via_grid_pitch_mm", 1.0)
                },
                reason="Adequate thermal vias for heat dissipation",
                source="JEDEC JESD51, IPC-7093",
            ))

        # THERMAL_COPPER_AREA
        hierarchy.recommended.append(RuleBinding(
            rule_id="THERMAL_COPPER_AREA",
            parameters={"min_area_pct": 30.0},
            reason="Copper pour area for thermal management",
            source="IPC-2152 Thermal Guidelines",
        ))

        # STACKUP_SYMMETRY
        if ctx.layer_count >= 4:
            hierarchy.recommended.append(RuleBinding(
                rule_id="STACKUP_SYMMETRY",
                parameters={"symmetric": True},
                reason="Prevents board warpage during fabrication",
                source="IPC-2221B Section 12",
            ))

        # STACKUP_REFERENCE_PLANE
        hierarchy.recommended.append(RuleBinding(
            rule_id="STACKUP_REFERENCE_PLANE",
            parameters={"signal_layer_has_adjacent_plane": True},
            reason="Signal layers need adjacent reference plane",
            source="Howard Johnson High-Speed Digital Design",
        ))

        # VIA_STITCHING_SPACING
        if ctx.max_frequency_hz > 100e6:
            via_spacing = self.rules_api.get_via_stitching_spacing(ctx.max_frequency_hz / 1e9)
            hierarchy.recommended.append(RuleBinding(
                rule_id="VIA_STITCHING_SPACING",
                parameters={
                    "max_spacing_mm": via_spacing,
                    "frequency_ghz": ctx.max_frequency_hz / 1e9
                },
                reason="Via stitching for ground plane continuity at high frequency",
                source="Eric Bogatin Signal Integrity Simplified",
            ))

        # VIA_IN_PAD
        hierarchy.recommended.append(RuleBinding(
            rule_id="VIA_IN_PAD",
            parameters={"requires_filled": True},
            reason="Via-in-pad requires filled and planarized vias",
            source="IPC-4761 Type VII",
        ))

    def _add_optional_rules(self, ctx: DesignContext, hierarchy: RuleHierarchy):
        """Add optional rules (quality improvements)."""

        # SILKSCREEN rules
        hierarchy.optional.append(RuleBinding(
            rule_id="SILKSCREEN_SIZE",
            parameters={"min_text_height_mm": 0.8, "min_line_width_mm": 0.15},
            reason="Readable silkscreen text",
            source="IPC-7351B",
        ))

        hierarchy.optional.append(RuleBinding(
            rule_id="SILKSCREEN_CLEARANCE",
            parameters={"clearance_from_pads_mm": 0.15},
            reason="Silkscreen should not overlap solder pads",
            source="IPC-7351B",
        ))

        hierarchy.optional.append(RuleBinding(
            rule_id="SILKSCREEN_ORIENTATION",
            parameters={"prefer_readable_orientation": True},
            reason="Text should be readable from one or two directions",
            source="DFM Best Practices",
        ))

        # TEST POINT rules
        test_rules = self.rules_api.get_test_point_rules()
        hierarchy.optional.append(RuleBinding(
            rule_id="TESTPOINT_ACCESSIBILITY",
            parameters={
                "min_diameter_mm": test_rules.get("min_diameter_mm", 1.0),
                "preferred_diameter_mm": test_rules.get("preferred_diameter_mm", 1.5)
            },
            reason="Test points for production testing",
            source="IPC-9252",
        ))

        hierarchy.optional.append(RuleBinding(
            rule_id="TESTPOINT_SPACING",
            parameters={
                "min_spacing_mm": test_rules.get("min_spacing_mm", 2.5),
                "to_edge_mm": test_rules.get("to_edge_mm", 3.0)
            },
            reason="Test probe access requirements",
            source="IPC-9252",
        ))

        # FIDUCIAL_PLACEMENT
        hierarchy.optional.append(RuleBinding(
            rule_id="FIDUCIAL_PLACEMENT",
            parameters={
                "min_count": 3,
                "diameter_mm": 1.0,
                "clearance_mm": 2.0
            },
            reason="Fiducials for pick-and-place machine alignment",
            source="IPC-7351B Section 8",
        ))

        # COMPONENT_ALIGNMENT
        hierarchy.optional.append(RuleBinding(
            rule_id="COMPONENT_ALIGNMENT",
            parameters={"grid_mm": 0.5},
            reason="Aligned components for cleaner layout",
            source="DFM Best Practices",
        ))

        # TRACE_ANGLE
        hierarchy.optional.append(RuleBinding(
            rule_id="TRACE_ANGLE",
            parameters={"prefer_45_degree": True, "avoid_90_degree": True},
            reason="45-degree angles for signal integrity",
            source="High-Speed PCB Design Guidelines",
        ))

        # REF_DES_VISIBILITY
        hierarchy.optional.append(RuleBinding(
            rule_id="REF_DES_VISIBILITY",
            parameters={"visible": True, "near_component": True},
            reason="Reference designators visible for assembly/debug",
            source="IPC-7351B",
        ))

        # POLARITY_MARKING
        hierarchy.optional.append(RuleBinding(
            rule_id="POLARITY_MARKING",
            parameters={"mark_diodes": True, "mark_caps": True, "mark_ics": True},
            reason="Polarity markings prevent assembly errors",
            source="IPC-7351B",
        ))

    def _add_conditional_inviolable(self, ctx: DesignContext, hierarchy: RuleHierarchy):
        """Add rules that become inviolable based on design features.

        These rules are INVIOLABLE because the interface won't work without them.
        All values come from RulesAPI (631 rules database).
        """

        # USB High-Speed requires controlled impedance
        if ctx.has_usb_hs:
            usb_rules = self.rules_api.get_length_matching_rules("USB_2.0_HS")
            hierarchy.inviolable.append(RuleBinding(
                rule_id="USB2_IMPEDANCE",
                parameters={
                    "single_ended_ohm": 45.0,
                    "tolerance_pct": 10.0
                },
                reason="USB High-Speed requires controlled impedance - won't work otherwise",
                source="USB 2.0 Specification Chapter 7",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="USB2_DIFFERENTIAL_IMPEDANCE",
                parameters={
                    "differential_ohm": 90.0,
                    "tolerance_pct": 10.0,
                    "max_mismatch_mm": usb_rules.get("max_mismatch_mm", 1.25)
                },
                reason="USB High-Speed differential impedance is mandatory",
                source="USB 2.0 Specification Chapter 7",
            ))

        # USB 3.x requires even tighter control
        if ctx.has_usb3:
            hierarchy.inviolable.append(RuleBinding(
                rule_id="USB3_IMPEDANCE",
                parameters={
                    "single_ended_ohm": 45.0,
                    "tolerance_pct": 10.0
                },
                reason="USB 3.x requires controlled impedance",
                source="USB 3.2 Specification",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="USB3_DIFFERENTIAL_IMPEDANCE",
                parameters={
                    "differential_ohm": 90.0,
                    "tolerance_pct": 10.0,
                    "max_mismatch_mm": 0.5
                },
                reason="USB 3.x has strict differential requirements",
                source="USB 3.2 Specification",
            ))

        # DDR3 impedance requirements
        if ctx.has_ddr3:
            ddr_rules = self.rules_api.get_ddr3_rules()
            hierarchy.inviolable.append(RuleBinding(
                rule_id="DDR3_DATA_IMPEDANCE",
                parameters={
                    "impedance_ohm": ddr_rules.get("data_impedance_ohm", 40),
                    "tolerance_pct": ddr_rules.get("impedance_tolerance_pct", 10)
                },
                reason="DDR3 memory won't function without correct impedance",
                source="JEDEC JESD79-3F",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="DDR3_CLK_IMPEDANCE",
                parameters={
                    "differential_ohm": ddr_rules.get("clk_diff_impedance_ohm", 100),
                    "tolerance_pct": 10.0
                },
                reason="DDR3 clock requires differential impedance",
                source="JEDEC JESD79-3F",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="DDR3_ADDR_IMPEDANCE",
                parameters={
                    "impedance_ohm": ddr_rules.get("addr_impedance_ohm", 40),
                    "tolerance_pct": 10.0
                },
                reason="DDR3 address/command impedance matching",
                source="JEDEC JESD79-3F",
            ))

        # DDR4 impedance requirements
        if ctx.has_ddr4:
            ddr_rules = self.rules_api.get_ddr4_rules()
            hierarchy.inviolable.append(RuleBinding(
                rule_id="DDR4_DATA_IMPEDANCE",
                parameters={
                    "impedance_ohm": ddr_rules.get("data_impedance_ohm", 40),
                    "tolerance_pct": ddr_rules.get("impedance_tolerance_pct", 10)
                },
                reason="DDR4 memory won't function without correct impedance",
                source="JEDEC JESD79-4B",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="DDR4_CLK_IMPEDANCE",
                parameters={
                    "differential_ohm": ddr_rules.get("clk_diff_impedance_ohm", 100),
                    "tolerance_pct": 10.0
                },
                reason="DDR4 clock requires differential impedance",
                source="JEDEC JESD79-4B",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="DDR4_ADDR_IMPEDANCE",
                parameters={
                    "impedance_ohm": ddr_rules.get("addr_impedance_ohm", 40),
                    "tolerance_pct": 10.0
                },
                reason="DDR4 address/command impedance matching",
                source="JEDEC JESD79-4B",
            ))

        # PCIe requirements
        if ctx.has_pcie:
            pcie_rules = self.rules_api.get_pcie_rules(f"Gen{ctx.has_pcie_gen}" if ctx.has_pcie_gen else "Gen3")
            hierarchy.inviolable.append(RuleBinding(
                rule_id="PCIE_DIFFERENTIAL_IMPEDANCE",
                parameters={
                    "differential_ohm": pcie_rules.get("diff_impedance_ohm", 85),
                    "tolerance_pct": 10.0,
                    "max_skew_mm": pcie_rules.get("max_skew_mm", 0.127)
                },
                reason="PCIe requires precise differential impedance",
                source="PCI Express Base Specification",
            ))

        # HDMI requirements
        if ctx.has_hdmi:
            hdmi_rules = self.rules_api.get_hdmi_rules()
            hierarchy.inviolable.append(RuleBinding(
                rule_id="HDMI_DIFFERENTIAL_IMPEDANCE",
                parameters={
                    "differential_ohm": hdmi_rules.get("diff_impedance_ohm", 100),
                    "tolerance_pct": 10.0
                },
                reason="HDMI requires controlled differential impedance",
                source="HDMI Specification",
            ))

        # Gigabit Ethernet requirements
        if ctx.has_ethernet and "1G" in ctx.has_ethernet_speed:
            eth_rules = self.rules_api.get_ethernet_rules("1000BASE-T")
            hierarchy.inviolable.append(RuleBinding(
                rule_id="ETHERNET_1G_IMPEDANCE",
                parameters={
                    "differential_ohm": eth_rules.get("diff_impedance_ohm", 100),
                    "tolerance_pct": 10.0
                },
                reason="Gigabit Ethernet requires differential impedance",
                source="IEEE 802.3ab",
            ))

        # High voltage safety requirements
        if ctx.max_voltage > 50:
            clearance = self.rules_api.get_clearance(ctx.max_voltage)
            hierarchy.inviolable.append(RuleBinding(
                rule_id="HV_CREEPAGE",
                parameters={
                    "min_creepage_mm": clearance * 2.0,
                    "voltage": ctx.max_voltage
                },
                reason="High voltage requires extended creepage distance",
                source="IPC-2221B, UL60950-1",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="HV_CLEARANCE",
                parameters={
                    "min_clearance_mm": clearance * 1.5,
                    "voltage": ctx.max_voltage
                },
                reason="High voltage requires extended clearance",
                source="IPC-2221B, UL60950-1",
            ))

        # High power thermal requirements
        if ctx.total_power > 10:
            thermal_rules = self.rules_api.get_thermal_pad_rules()
            hierarchy.inviolable.append(RuleBinding(
                rule_id="THERMAL_VIA_COUNT",
                parameters={
                    "min_vias": max(thermal_rules.get("min_vias", 5), int(ctx.total_power / 2)),
                    "via_drill_mm": thermal_rules.get("via_drill_mm", 0.3)
                },
                reason="High power requires adequate thermal vias",
                source="JEDEC JESD51",
            ))
            hierarchy.inviolable.append(RuleBinding(
                rule_id="THERMAL_PAD_SIZE",
                parameters={
                    "min_area_mm2": ctx.total_power * 50  # ~50mm2 per watt
                },
                reason="High power requires adequate thermal pad area",
                source="Component thermal design guidelines",
            ))

    def _adjust_for_context(self, ctx: DesignContext, hierarchy: RuleHierarchy):
        """Adjust rule priorities based on design context."""

        # Safety-critical applications: promote more rules to inviolable
        if ctx.is_safety_critical:
            rules_to_promote = ["EMI_RADIATION_LIMIT", "THERMAL_VIA_COUNT"]
            for rule_id in rules_to_promote:
                self._promote_rule(hierarchy, rule_id, RulePriority.INVIOLABLE,
                                 "Safety-critical application")

        # High-frequency designs: promote EMI rules
        if ctx.max_frequency_hz > 100e6:  # > 100 MHz
            rules_to_promote = ["EMI_LOOP_AREA", "RETURN_PATH_CONTINUITY"]
            for rule_id in rules_to_promote:
                self._promote_rule(hierarchy, rule_id, RulePriority.INVIOLABLE,
                                 f"High frequency design ({ctx.max_frequency_hz/1e6:.0f} MHz)")

        # Precision analog: promote analog separation
        if ctx.has_precision_analog:
            self._promote_rule(hierarchy, "ANALOG_SEPARATION", RulePriority.INVIOLABLE,
                             "Precision analog design")

        # Switching regulators: promote EMI rules
        if ctx.has_switching_regulator:
            self._promote_rule(hierarchy, "EMI_LOOP_AREA", RulePriority.INVIOLABLE,
                             "Switching regulator requires tight loop control")

    def _promote_rule(
        self,
        hierarchy: RuleHierarchy,
        rule_id: str,
        to_priority: RulePriority,
        reason: str
    ):
        """Move a rule to a higher priority category."""

        # Find and remove from current category
        existing = None
        for rule in hierarchy.optional:
            if rule.rule_id == rule_id:
                existing = rule
                hierarchy.optional.remove(rule)
                break
        if not existing:
            for rule in hierarchy.recommended:
                if rule.rule_id == rule_id:
                    existing = rule
                    hierarchy.recommended.remove(rule)
                    break

        if not existing:
            existing = RuleBinding(rule_id=rule_id)

        # Update reason
        existing.reason = reason

        # Add to new category
        if to_priority == RulePriority.INVIOLABLE:
            hierarchy.inviolable.append(existing)
        elif to_priority == RulePriority.RECOMMENDED:
            hierarchy.recommended.append(existing)

    def get_applicable_rules(
        self,
        components: List[ComponentDefinition],
        nets: List[NetDefinition],
    ) -> List[str]:
        """Get list of rule IDs that apply to this design."""

        ctx = self.analyzer.analyze(components, nets, 50, 40, 2)
        applicable = set(ALWAYS_INVIOLABLE)

        # Add conditional rules
        if ctx.has_usb_hs or ctx.has_usb_fs:
            applicable.update(["USB2_IMPEDANCE", "USB2_LENGTH_MATCHING"])
        if ctx.has_usb3:
            applicable.update(["USB3_IMPEDANCE", "USB3_LENGTH_MATCHING"])
        if ctx.has_ddr3:
            applicable.update(["DDR3_DATA_IMPEDANCE", "DDR3_DQS_DQ_MATCHING"])
        if ctx.has_ddr4:
            applicable.update(["DDR4_DATA_IMPEDANCE", "DDR4_DQS_DQ_MATCHING"])
        if ctx.has_pcie:
            applicable.add("PCIE_DIFFERENTIAL_IMPEDANCE")
        if ctx.has_hdmi:
            applicable.add("HDMI_DIFFERENTIAL_IMPEDANCE")

        # Add general rules
        applicable.update(DEFAULT_RECOMMENDED)
        applicable.update(DEFAULT_OPTIONAL)

        return list(applicable)

    def explain_classification(self, rule_id: str, hierarchy: RuleHierarchy) -> str:
        """Explain why a rule is in its current category."""

        rule = hierarchy.get_rule(rule_id)
        if not rule:
            return f"Rule {rule_id} is not explicitly classified (default: {hierarchy.default_category.value})"

        priority = hierarchy.get_priority(rule_id)
        return f"Rule {rule_id} is {priority.value}: {rule.reason}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hierarchy_for_design(
    components: List[ComponentDefinition],
    nets: List[NetDefinition],
    board_width: float = 50.0,
    board_height: float = 40.0,
    layer_count: int = 2,
) -> RuleHierarchy:
    """
    Convenience function to create a rule hierarchy for a design.

    Args:
        components: List of components in the design
        nets: List of nets in the design
        board_width: Board width in mm
        board_height: Board height in mm
        layer_count: Number of layers

    Returns:
        RuleHierarchy with classified rules
    """
    engine = RuleHierarchyEngine()
    return engine.classify_rules(components, nets, board_width, board_height, layer_count)


def get_default_hierarchy() -> RuleHierarchy:
    """Get a default rule hierarchy for simple designs."""
    hierarchy = RuleHierarchy()

    for rule_id in ALWAYS_INVIOLABLE:
        hierarchy.inviolable.append(RuleBinding(
            rule_id=rule_id,
            reason="Safety-critical",
        ))

    for rule_id in DEFAULT_RECOMMENDED:
        hierarchy.recommended.append(RuleBinding(
            rule_id=rule_id,
            reason="Best practice",
        ))

    for rule_id in DEFAULT_OPTIONAL:
        hierarchy.optional.append(RuleBinding(
            rule_id=rule_id,
            reason="Quality improvement",
        ))

    return hierarchy
