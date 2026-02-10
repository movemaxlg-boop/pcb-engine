"""
PARTS INFERENCE ENGINE
======================

Given user-specified components, infers required supporting components.
This is what makes the AI "smart" - it knows that:
- MCUs need decoupling capacitors
- USB connectors need ESD protection
- Crystals need load capacitors
- LDOs need input/output capacitors
- etc.

The AI adds these components automatically with justification.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from .clayout_types import (
    ComponentDefinition,
    ComponentCategory,
    NetDefinition,
    NetType,
)


# =============================================================================
# INFERENCE RULES DATABASE
# =============================================================================

@dataclass
class InferenceRule:
    """A rule for inferring supporting components."""

    # What triggers this rule
    trigger_category: ComponentCategory
    trigger_patterns: List[str]     # Part number patterns (regex)

    # What to add
    add_components: List[Dict]      # List of components to add

    # Rule metadata
    rule_name: str
    reason: str
    source: str                     # Datasheet, application note, etc.

    # Conditions
    per_trigger: bool = True        # Add per triggering component (vs once)


# The master inference rules database
INFERENCE_RULES: List[InferenceRule] = [

    # =========================================================================
    # MCU DECOUPLING
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.MCU,
        trigger_patterns=[".*"],  # All MCUs
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100nF",
                "footprint": "0402",
                "quantity_rule": "per_vcc_pin",
                "default_quantity": 4,
                "placement": "adjacent",
                "max_distance_mm": 3.0,
            },
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "10uF",
                "footprint": "0805",
                "quantity": 1,
                "placement": "nearby",
                "max_distance_mm": 10.0,
            },
        ],
        rule_name="MCU Decoupling",
        reason="Decoupling capacitors for stable power supply",
        source="Application Notes, IPC-2221",
        per_trigger=True,
    ),

    # =========================================================================
    # ESP32 SPECIFIC
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.MCU,
        trigger_patterns=["ESP32.*", "ESP-WROOM.*", "ESP-WROVER.*"],
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100nF",
                "footprint": "0402",
                "quantity": 6,
                "reason": "ESP32 requires 6x 100nF near VDD pins",
            },
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "22uF",
                "footprint": "0805",
                "quantity": 1,
                "reason": "Bulk capacitor for ESP32 WiFi transients",
            },
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "10nF",
                "footprint": "0402",
                "quantity": 1,
                "reason": "EN pin filter capacitor",
            },
            {
                "category": ComponentCategory.RESISTOR,
                "value": "10k",
                "footprint": "0402",
                "quantity": 1,
                "reason": "EN pin pull-up resistor",
            },
        ],
        rule_name="ESP32 Support Components",
        reason="ESP32 required support components per datasheet",
        source="ESP32 Hardware Design Guidelines",
        per_trigger=True,
    ),

    # =========================================================================
    # USB CONNECTOR ESD PROTECTION
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.CONNECTOR,
        trigger_patterns=["USB.*", ".*USB-C.*", ".*TYPE-C.*", ".*MICRO-USB.*"],
        add_components=[
            {
                "category": ComponentCategory.ESD_PROTECTION,
                "part_number": "USBLC6-2SC6",
                "footprint": "SOT-23-6",
                "quantity": 1,
                "reason": "ESD protection for USB data lines",
            },
        ],
        rule_name="USB ESD Protection",
        reason="ESD protection required per USB specification",
        source="USB 2.0 Specification, IEC 61000-4-2",
        per_trigger=True,
    ),

    # =========================================================================
    # USB-C SPECIFIC
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.CONNECTOR,
        trigger_patterns=[".*USB-C.*", ".*TYPE-C.*"],
        add_components=[
            {
                "category": ComponentCategory.RESISTOR,
                "value": "5.1k",
                "footprint": "0402",
                "quantity": 2,
                "reason": "CC1/CC2 pull-down resistors for UFP identification",
            },
        ],
        rule_name="USB-C CC Resistors",
        reason="USB-C requires CC pin pull-down resistors for device identification",
        source="USB Type-C Specification",
        per_trigger=True,
    ),

    # =========================================================================
    # CRYSTAL OSCILLATOR
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.CRYSTAL,
        trigger_patterns=[".*"],  # All crystals
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "from_crystal_spec",  # Placeholder - AI calculates
                "footprint": "0402",
                "quantity": 2,
                "placement": "adjacent",
                "max_distance_mm": 2.0,
                "reason": "Load capacitors for crystal oscillation",
            },
        ],
        rule_name="Crystal Load Capacitors",
        reason="Load capacitors required for crystal oscillation",
        source="Crystal Oscillator Design Guide",
        per_trigger=True,
    ),

    # =========================================================================
    # LDO REGULATOR
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.REGULATOR,
        trigger_patterns=["LDO.*", "AMS1117.*", "LD1117.*", "MCP1700.*", "AP2112.*", "TLV702.*"],
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "10uF",
                "footprint": "0805",
                "quantity": 1,
                "position": "input",
                "max_distance_mm": 5.0,
                "reason": "Input capacitor for LDO stability",
            },
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "10uF",
                "footprint": "0805",
                "quantity": 1,
                "position": "output",
                "max_distance_mm": 3.0,
                "reason": "Output capacitor for LDO stability",
            },
        ],
        rule_name="LDO Capacitors",
        reason="Input and output capacitors required for LDO stability",
        source="LDO Datasheet Application Circuit",
        per_trigger=True,
    ),

    # =========================================================================
    # SWITCHING REGULATOR
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.REGULATOR,
        trigger_patterns=["LM2596.*", "MP1584.*", "TPS54.*", "TPS56.*", "LM2678.*", "AP63.*"],
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "22uF",
                "footprint": "1206",
                "quantity": 1,
                "position": "input",
                "reason": "Input bulk capacitor",
            },
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100uF",
                "footprint": "1206",
                "quantity": 1,
                "position": "output",
                "reason": "Output bulk capacitor",
            },
            {
                "category": ComponentCategory.INDUCTOR,
                "value": "from_regulator_spec",
                "footprint": "varies",
                "quantity": 1,
                "reason": "Buck converter inductor",
            },
            {
                "category": ComponentCategory.DIODE,
                "part_number": "SS34",
                "footprint": "SMA",
                "quantity": 1,
                "reason": "Schottky catch diode (if not synchronous)",
            },
        ],
        rule_name="Buck Regulator Components",
        reason="Standard buck regulator external components",
        source="Regulator Datasheet",
        per_trigger=True,
    ),

    # =========================================================================
    # POWER INPUT
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.CONNECTOR,
        trigger_patterns=[".*BARREL.*", ".*DC.*JACK.*", ".*POWER.*"],
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100uF",
                "footprint": "electrolytic",
                "quantity": 1,
                "reason": "Bulk input capacitor",
            },
            {
                "category": ComponentCategory.DIODE,
                "part_number": "1N4007",
                "footprint": "DO-41",
                "quantity": 1,
                "reason": "Reverse polarity protection",
            },
        ],
        rule_name="Power Input Protection",
        reason="Bulk capacitance and reverse polarity protection",
        source="Power Supply Design Best Practices",
        per_trigger=True,
    ),

    # =========================================================================
    # ADC INPUT
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.OTHER,
        trigger_patterns=["ADS1.*", "MCP3.*", "AD7.*", "MAX11.*"],
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100nF",
                "footprint": "0402",
                "quantity": 1,
                "reason": "Reference voltage bypass",
            },
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "10uF",
                "footprint": "0805",
                "quantity": 1,
                "reason": "Analog supply bypass",
            },
        ],
        rule_name="ADC Bypass",
        reason="ADC requires clean analog supply",
        source="ADC Datasheet",
        per_trigger=True,
    ),

    # =========================================================================
    # OP-AMP
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.OTHER,
        trigger_patterns=["LM358.*", "TL072.*", "OPA.*", "AD8.*", "MCP60.*"],
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100nF",
                "footprint": "0402",
                "quantity": 1,
                "reason": "Power supply bypass",
            },
        ],
        rule_name="Op-Amp Bypass",
        reason="Op-amp power supply decoupling",
        source="Op-Amp Application Notes",
        per_trigger=True,
    ),

    # =========================================================================
    # ETHERNET PHY
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.OTHER,
        trigger_patterns=["LAN87.*", "KSZ80.*", "DP83.*", "RTL8.*"],
        add_components=[
            {
                "category": ComponentCategory.CRYSTAL,
                "value": "25MHz",
                "footprint": "HC49",
                "quantity": 1,
                "reason": "Ethernet PHY clock",
            },
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100nF",
                "footprint": "0402",
                "quantity": 4,
                "reason": "PHY decoupling",
            },
            {
                "category": ComponentCategory.RESISTOR,
                "value": "49.9",
                "footprint": "0402",
                "quantity": 1,
                "reason": "RBIAS resistor",
            },
        ],
        rule_name="Ethernet PHY Support",
        reason="Standard Ethernet PHY support components",
        source="Ethernet PHY Datasheet",
        per_trigger=True,
    ),

    # =========================================================================
    # RESET CIRCUIT
    # =========================================================================
    InferenceRule(
        trigger_category=ComponentCategory.MCU,
        trigger_patterns=[".*"],  # All MCUs
        add_components=[
            {
                "category": ComponentCategory.CAPACITOR,
                "value": "100nF",
                "footprint": "0402",
                "quantity": 1,
                "reason": "Reset pin filter capacitor",
            },
            {
                "category": ComponentCategory.RESISTOR,
                "value": "10k",
                "footprint": "0402",
                "quantity": 1,
                "reason": "Reset pin pull-up",
            },
        ],
        rule_name="Reset Circuit",
        reason="MCU reset pin filtering and pull-up",
        source="MCU Application Notes",
        per_trigger=True,
    ),

]


# =============================================================================
# PARTS INFERENCE ENGINE
# =============================================================================

class PartsInferenceEngine:
    """
    Infers required supporting components based on user-specified parts.

    This is the "smart" part of the AI - it knows what supporting components
    are needed for each type of part.
    """

    def __init__(self):
        self.rules = INFERENCE_RULES
        self._ref_des_counters: Dict[str, int] = {}

    def infer_components(
        self,
        user_components: List[ComponentDefinition]
    ) -> List[ComponentDefinition]:
        """
        Infer all required supporting components.

        Args:
            user_components: Components specified by the user

        Returns:
            List of inferred components to add
        """

        inferred = []
        self._ref_des_counters = self._init_counters(user_components)

        for comp in user_components:
            for rule in self.rules:
                if self._matches_rule(comp, rule):
                    new_components = self._apply_rule(comp, rule)
                    inferred.extend(new_components)

        return inferred

    def _init_counters(self, existing: List[ComponentDefinition]) -> Dict[str, int]:
        """Initialize reference designator counters from existing components."""
        counters = {}
        for comp in existing:
            prefix = self._get_ref_prefix(comp.ref_des)
            num = self._get_ref_number(comp.ref_des)
            if prefix not in counters or num >= counters[prefix]:
                counters[prefix] = num + 1
        return counters

    def _get_ref_prefix(self, ref_des: str) -> str:
        """Extract prefix from reference designator (e.g., 'C' from 'C1')."""
        match = re.match(r'^([A-Za-z]+)', ref_des)
        return match.group(1) if match else "X"

    def _get_ref_number(self, ref_des: str) -> int:
        """Extract number from reference designator (e.g., 1 from 'C1')."""
        match = re.search(r'(\d+)$', ref_des)
        return int(match.group(1)) if match else 0

    def _get_next_ref_des(self, category: ComponentCategory) -> str:
        """Get the next available reference designator for a category."""
        prefix_map = {
            ComponentCategory.CAPACITOR: "C",
            ComponentCategory.RESISTOR: "R",
            ComponentCategory.INDUCTOR: "L",
            ComponentCategory.DIODE: "D",
            ComponentCategory.LED: "LED",
            ComponentCategory.TRANSISTOR: "Q",
            ComponentCategory.CRYSTAL: "Y",
            ComponentCategory.CONNECTOR: "J",
            ComponentCategory.ESD_PROTECTION: "U",
            ComponentCategory.REGULATOR: "U",
            ComponentCategory.MCU: "U",
            ComponentCategory.SENSOR: "U",
            ComponentCategory.MEMORY: "U",
            ComponentCategory.OTHER: "U",
        }

        prefix = prefix_map.get(category, "U")
        if prefix not in self._ref_des_counters:
            self._ref_des_counters[prefix] = 1

        ref_des = f"{prefix}{self._ref_des_counters[prefix]}"
        self._ref_des_counters[prefix] += 1
        return ref_des

    def _matches_rule(self, comp: ComponentDefinition, rule: InferenceRule) -> bool:
        """Check if a component matches a rule."""

        # Check category
        if comp.category != rule.trigger_category:
            return False

        # Check patterns
        part_upper = comp.part_number.upper()
        for pattern in rule.trigger_patterns:
            if re.match(pattern, part_upper, re.IGNORECASE):
                return True

        return False

    def _apply_rule(
        self,
        trigger_comp: ComponentDefinition,
        rule: InferenceRule
    ) -> List[ComponentDefinition]:
        """Apply an inference rule to generate new components."""

        new_components = []

        for comp_spec in rule.add_components:
            quantity = comp_spec.get("quantity", 1)

            # Handle quantity_rule
            if comp_spec.get("quantity_rule") == "per_vcc_pin":
                quantity = comp_spec.get("default_quantity", 4)

            for _ in range(quantity):
                category = comp_spec.get("category", ComponentCategory.OTHER)
                new_comp = ComponentDefinition(
                    ref_des=self._get_next_ref_des(category),
                    part_number=comp_spec.get("part_number", comp_spec.get("value", "TBD")),
                    footprint=comp_spec.get("footprint", "0402"),
                    category=category,
                    value=comp_spec.get("value"),
                    inferred=True,
                    inference_reason=comp_spec.get("reason", rule.reason),
                    inferred_for=trigger_comp.ref_des,
                )
                new_components.append(new_comp)

        return new_components

    def explain_inference(self, inferred_comp: ComponentDefinition) -> str:
        """
        Explain why a component was inferred.

        Args:
            inferred_comp: An inferred component

        Returns:
            Human-readable explanation
        """

        if not inferred_comp.inferred:
            return f"{inferred_comp.ref_des} was specified by user, not inferred."

        return (
            f"{inferred_comp.ref_des} ({inferred_comp.part_number}) was added "
            f"for {inferred_comp.inferred_for}: {inferred_comp.inference_reason}"
        )

    def get_all_rules_for_component(self, comp: ComponentDefinition) -> List[InferenceRule]:
        """Get all inference rules that apply to a component."""
        return [rule for rule in self.rules if self._matches_rule(comp, rule)]

    def summarize_inference(
        self,
        original: List[ComponentDefinition],
        inferred: List[ComponentDefinition]
    ) -> str:
        """Generate a summary of what was inferred."""

        if not inferred:
            return "No additional components inferred."

        lines = [
            f"Inferred {len(inferred)} supporting components:",
            ""
        ]

        # Group by what they support
        by_parent: Dict[str, List[ComponentDefinition]] = {}
        for comp in inferred:
            parent = comp.inferred_for or "General"
            if parent not in by_parent:
                by_parent[parent] = []
            by_parent[parent].append(comp)

        for parent, comps in by_parent.items():
            parent_comp = next((c for c in original if c.ref_des == parent), None)
            parent_name = f"{parent} ({parent_comp.part_number})" if parent_comp else parent
            lines.append(f"  For {parent_name}:")
            for comp in comps:
                lines.append(f"    + {comp.ref_des}: {comp.value or comp.part_number} - {comp.inference_reason}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def infer_supporting_components(
    components: List[ComponentDefinition]
) -> Tuple[List[ComponentDefinition], str]:
    """
    Convenience function to infer supporting components.

    Args:
        components: User-specified components

    Returns:
        Tuple of (inferred_components, summary_text)
    """
    engine = PartsInferenceEngine()
    inferred = engine.infer_components(components)
    summary = engine.summarize_inference(components, inferred)
    return inferred, summary
