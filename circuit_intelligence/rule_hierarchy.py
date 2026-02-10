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
        """

        # Analyze design context
        ctx = self.analyzer.analyze(components, nets, board_width, board_height, layer_count)

        # Start with empty hierarchy
        hierarchy = RuleHierarchy()

        # Add always-inviolable rules
        for rule_id in ALWAYS_INVIOLABLE:
            hierarchy.inviolable.append(RuleBinding(
                rule_id=rule_id,
                reason="Safety-critical rule",
                source="IPC-2221, IPC-2152",
            ))

        # Add conditional inviolable rules based on design features
        self._add_conditional_inviolable(ctx, hierarchy)

        # Add recommended rules
        for rule_id in DEFAULT_RECOMMENDED:
            # Skip if already inviolable
            if hierarchy.get_rule(rule_id):
                continue
            hierarchy.recommended.append(RuleBinding(
                rule_id=rule_id,
                reason="Industry best practice",
                source="IPC Guidelines",
            ))

        # Add optional rules
        for rule_id in DEFAULT_OPTIONAL:
            # Skip if already classified
            if hierarchy.get_rule(rule_id):
                continue
            hierarchy.optional.append(RuleBinding(
                rule_id=rule_id,
                reason="Quality improvement",
                source="Design for Manufacturing",
            ))

        # Promote/demote rules based on context
        self._adjust_for_context(ctx, hierarchy)

        return hierarchy

    def _add_conditional_inviolable(self, ctx: DesignContext, hierarchy: RuleHierarchy):
        """Add rules that become inviolable based on design features."""

        conditions = {
            "has_usb_hs": ctx.has_usb_hs,
            "has_usb3": ctx.has_usb3,
            "has_ddr3": ctx.has_ddr3,
            "has_ddr4": ctx.has_ddr4,
            "has_pcie": ctx.has_pcie,
            "has_hdmi": ctx.has_hdmi,
            "has_ethernet_gigabit": ctx.has_ethernet and ctx.has_ethernet_speed == "1G",
            "voltage_above_50v": ctx.max_voltage > 50,
            "power_above_10w": ctx.total_power > 10,
        }

        for condition, is_met in conditions.items():
            if is_met and condition in CONDITIONAL_INVIOLABLE:
                for rule_id in CONDITIONAL_INVIOLABLE[condition]:
                    # Skip if already added
                    if hierarchy.get_rule(rule_id):
                        continue
                    hierarchy.inviolable.append(RuleBinding(
                        rule_id=rule_id,
                        reason=f"Required for {condition.replace('_', ' ')}",
                        source="Interface specification",
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
