"""
CIRCUIT AI AGENT
================

The AI agent that converts user intent into Constitutional Layout (c_layout).

This is the "brain" of the system that:
1. Parses natural language design requirements
2. Selects components from the parts database
3. Infers supporting components (decoupling, ESD, etc.)
4. Classifies rules into hierarchy (Inviolable/Recommended/Optional)
5. Generates placement and routing hints
6. Creates overrides with justification when needed
7. Produces a complete c_layout for the PCB Engine

The AI agent also handles escalation from the BBL Engine when routing fails.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import re
from datetime import datetime

from .clayout_types import (
    ConstitutionalLayout,
    BoardConstraints,
    ComponentDefinition,
    NetDefinition,
    RuleHierarchy,
    RuleBinding,
    RuleOverride,
    RulePriority,
    PlacementHints,
    RoutingHints,
    ProximityGroup,
    KeepApart,
    DiffPairSpec,
    LengthMatchGroup,
    DesignIntent,
    EscalationReport,
    ComponentCategory,
    NetType,
    ValidationStatus,
    create_clayout_from_intent,
)
from .rule_hierarchy import RuleHierarchyEngine, DesignContext
from .parts_inference import PartsInferenceEngine
from .clayout_validator import CLayoutValidator, validate_clayout
from .rules_api import RulesAPI
from .feedback import AIFeedbackProcessor


# =============================================================================
# DESIGN INTENT PARSER
# =============================================================================

class DesignIntentParser:
    """Parses natural language design requirements into structured intent."""

    # Component keywords to look for
    COMPONENT_PATTERNS = {
        "ESP32": ["esp32", "esp-32", "esp wroom", "esp-wroom", "esp32-wroom"],
        "ESP8266": ["esp8266", "esp-8266", "esp-12"],
        "STM32": ["stm32", "stm-32", "stm32f", "stm32l", "stm32g"],
        "ATMEGA": ["atmega", "at-mega", "atmega328", "atmega2560"],
        "USB-C": ["usb-c", "usb type-c", "type-c", "usbc"],
        "USB-MICRO": ["micro usb", "micro-usb", "usb micro"],
        "USB-A": ["usb-a", "usb a"],
        "ETHERNET": ["ethernet", "rj45", "lan", "10/100"],
        "WIFI": ["wifi", "wi-fi", "wireless"],
        "BLUETOOTH": ["bluetooth", "ble", "bt"],
        "SD_CARD": ["sd card", "sd-card", "micro sd", "microsd"],
        "I2C": ["i2c", "i²c"],
        "SPI": ["spi"],
        "UART": ["uart", "serial", "rs232", "rs-232"],
        "CAN": ["can", "can bus", "canbus"],
        "ADC": ["adc", "analog input", "analog-to-digital"],
        "DAC": ["dac", "analog output", "digital-to-analog"],
        "PWM": ["pwm"],
        "GPIO": ["gpio", "digital io", "io pins"],
        "LED": ["led", "indicator"],
        "BUTTON": ["button", "switch", "push button"],
        "SENSOR": ["sensor", "temperature", "humidity", "pressure", "accelerometer"],
        "MOTOR_DRIVER": ["motor driver", "h-bridge", "l298", "drv8"],
        "RELAY": ["relay"],
        "DISPLAY": ["display", "lcd", "oled", "tft", "screen"],
        "POWER_SUPPLY": ["power supply", "5v", "3.3v", "12v", "voltage regulator"],
        "BATTERY": ["battery", "lipo", "li-ion", "18650"],
    }

    # Size patterns
    SIZE_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:mm)?',
        re.IGNORECASE
    )

    # Layer patterns
    LAYER_PATTERN = re.compile(
        r'(\d+)\s*(?:layer|layers|L)\b',
        re.IGNORECASE
    )

    def parse(self, user_input: str) -> DesignIntent:
        """
        Parse user input into design intent.

        Args:
            user_input: Natural language design description

        Returns:
            DesignIntent with extracted requirements
        """

        intent = DesignIntent()
        intent.original_statements = [user_input]

        # Normalize input
        text = user_input.lower()

        # Extract design name
        intent.design_name = self._extract_design_name(user_input)

        # Extract required components
        intent.required_components = self._extract_components(text)

        # Extract features
        intent.required_features = self._extract_features(text)

        # Extract board size
        intent.board_size_constraint = self._extract_size(text)

        # Extract layer count
        intent.layer_constraint = self._extract_layers(text)

        # Identify clarifications needed
        intent.clarification_needed = self._identify_ambiguities(intent)

        return intent

    def _extract_design_name(self, text: str) -> str:
        """Extract or generate a design name."""
        # Look for explicit name
        patterns = [
            r'(?:name|called|titled)\s*[:\-]?\s*["\']?(\w+(?:\s+\w+)*)["\']?',
            r'["\'](\w+(?:[-_]\w+)+)["\']',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(" ", "_")

        # Generate name from components
        components = self._extract_components(text.lower())
        if components:
            main = components[0].replace("-", "_").replace(" ", "_")
            return f"{main}_Design"

        return "New_Design"

    def _extract_components(self, text: str) -> List[str]:
        """Extract component types from text."""
        found = []

        for comp_type, keywords in self.COMPONENT_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    if comp_type not in found:
                        found.append(comp_type)
                    break

        return found

    def _extract_features(self, text: str) -> List[str]:
        """Extract feature requirements from text."""
        features = []

        # Look for analog inputs
        match = re.search(r'(\d+)\s*(?:analog|adc)\s*(?:input|channel)', text)
        if match:
            features.append(f"{match.group(1)} analog inputs")

        # Look for digital I/O
        match = re.search(r'(\d+)\s*(?:digital|gpio)\s*(?:io|pin)', text)
        if match:
            features.append(f"{match.group(1)} digital I/O")

        # Look for PWM
        match = re.search(r'(\d+)\s*pwm', text)
        if match:
            features.append(f"{match.group(1)} PWM outputs")

        # Power requirements
        for voltage in ["3.3v", "5v", "12v", "24v", "48v"]:
            if voltage in text:
                features.append(f"{voltage.upper()} power")

        # Current requirements
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:a|amp)', text)
        if match:
            features.append(f"{match.group(1)}A capability")

        return features

    def _extract_size(self, text: str) -> Optional[Tuple[float, float]]:
        """Extract board size from text."""
        match = self.SIZE_PATTERN.search(text)
        if match:
            return (float(match.group(1)), float(match.group(2)))

        # Look for common size descriptors
        size_map = {
            "compact": (40, 30),
            "small": (50, 40),
            "medium": (80, 60),
            "large": (100, 80),
            "credit card": (85.6, 53.98),
            "arduino uno": (68.6, 53.4),
            "raspberry pi": (85, 56),
        }

        for desc, size in size_map.items():
            if desc in text:
                return size

        return None

    def _extract_layers(self, text: str) -> Optional[int]:
        """Extract layer count from text."""
        match = self.LAYER_PATTERN.search(text)
        if match:
            return int(match.group(1))

        # Look for layer hints
        if "single layer" in text or "1 layer" in text:
            return 1
        if "two layer" in text or "double layer" in text:
            return 2
        if "four layer" in text or "4-layer" in text:
            return 4

        return None

    def _identify_ambiguities(self, intent: DesignIntent) -> List[str]:
        """Identify aspects that need clarification."""
        questions = []

        # No size specified
        if not intent.board_size_constraint:
            questions.append(
                "What are the size constraints for the board? "
                "(e.g., 50x40mm, or 'compact', 'credit card size')"
            )

        # USB but no speed specified
        if "USB-C" in intent.required_components or "USB-MICRO" in intent.required_components:
            questions.append(
                "What USB speed is required? "
                "(USB 2.0 Full-Speed 12Mbps, High-Speed 480Mbps, or USB 3.x 5Gbps+)"
            )

        # Power but no specifics
        has_power = any("power" in f.lower() for f in intent.required_features)
        if not has_power:
            questions.append(
                "What is the power input? "
                "(e.g., USB 5V, barrel jack 12V, battery)"
            )

        return questions


# =============================================================================
# CIRCUIT AI AGENT
# =============================================================================

class CircuitAI:
    """
    The main AI agent that generates Constitutional Layouts.

    This is the interface between human designers and the PCB Engine.
    """

    def __init__(self):
        """Initialize the Circuit AI agent."""
        self.intent_parser = DesignIntentParser()
        self.hierarchy_engine = RuleHierarchyEngine()
        self.parts_inference = PartsInferenceEngine()
        self.validator = CLayoutValidator()
        self.rules_api = RulesAPI()
        self.feedback_processor = AIFeedbackProcessor()

        # Parts database (simplified - would load from file in production)
        self.parts_db = self._init_parts_db()

    def _init_parts_db(self) -> Dict[str, Dict]:
        """Initialize a basic parts database."""
        return {
            # MCUs
            "ESP32-WROOM-32": {
                "category": "MCU",
                "footprint": "ESP32-WROOM",
                "vcc_pins": ["3V3"],
                "gnd_pins": ["GND"],
            },
            "ESP32-WROOM-32E": {
                "category": "MCU",
                "footprint": "ESP32-WROOM",
                "vcc_pins": ["3V3"],
                "gnd_pins": ["GND"],
            },
            "STM32F103C8T6": {
                "category": "MCU",
                "footprint": "LQFP-48",
                "vcc_pins": ["VDD", "VDDA"],
                "gnd_pins": ["VSS", "VSSA"],
            },

            # Connectors
            "USB-C-16P": {
                "category": "CONNECTOR",
                "footprint": "USB-C",
                "pins": ["VBUS", "GND", "CC1", "CC2", "D+", "D-"],
            },

            # Regulators
            "AMS1117-3.3": {
                "category": "REGULATOR",
                "footprint": "SOT-223",
                "vin_max": 15.0,
                "vout": 3.3,
                "iout_max": 1.0,
            },

            # Passives (templates)
            "100nF": {
                "category": "CAPACITOR",
                "footprint": "0402",
                "value": "100nF",
            },
            "10uF": {
                "category": "CAPACITOR",
                "footprint": "0805",
                "value": "10uF",
            },

            # ESD Protection
            "USBLC6-2SC6": {
                "category": "ESD_PROTECTION",
                "footprint": "SOT-23-6",
            },
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def parse_user_intent(self, user_input: str) -> DesignIntent:
        """
        Parse user input into structured design intent.

        Args:
            user_input: Natural language design description

        Returns:
            DesignIntent with extracted requirements
        """
        return self.intent_parser.parse(user_input)

    def ask_clarifying_questions(self, intent: DesignIntent) -> List[str]:
        """
        Generate questions to clarify ambiguous requirements.

        Args:
            intent: Partially parsed design intent

        Returns:
            List of questions to ask the user
        """
        return intent.clarification_needed

    def process_user_answers(
        self,
        intent: DesignIntent,
        answers: Dict[str, str]
    ) -> DesignIntent:
        """
        Update intent based on user's answers.

        Args:
            intent: Current design intent
            answers: User's answers to clarifying questions

        Returns:
            Updated DesignIntent
        """
        # Process each answer
        for question, answer in answers.items():
            if "size" in question.lower():
                size = self.intent_parser._extract_size(answer.lower())
                if size:
                    intent.board_size_constraint = size

            if "usb speed" in question.lower():
                if "high" in answer.lower() or "480" in answer:
                    intent.required_features.append("USB High-Speed")
                elif "3" in answer or "5g" in answer.lower():
                    intent.required_features.append("USB 3.x")
                else:
                    intent.required_features.append("USB Full-Speed")

            if "power" in question.lower():
                intent.required_features.append(f"Power: {answer}")

        # Clear answered questions
        intent.clarification_needed = [
            q for q in intent.clarification_needed
            if q not in answers
        ]

        return intent

    def generate_clayout(self, intent: DesignIntent) -> ConstitutionalLayout:
        """
        Generate a complete Constitutional Layout from design intent.

        This is the main entry point for c_layout generation.

        Args:
            intent: Parsed and clarified design intent

        Returns:
            Complete ConstitutionalLayout ready for BBL
        """

        # Step 1: Create base c_layout
        clayout = create_clayout_from_intent(intent)

        # Step 2: Determine board constraints
        clayout.board = self._determine_board_constraints(intent)

        # Step 3: Select user components
        user_components = self._select_components(intent)

        # Step 4: Infer supporting components
        inferred_components = self.parts_inference.infer_components(user_components)

        # Combine all components
        clayout.components = user_components + inferred_components

        # Step 5: Define nets
        clayout.nets = self._define_nets(clayout.components, intent)

        # Step 6: Classify rules into hierarchy
        clayout.rules = self.hierarchy_engine.classify_rules(
            clayout.components,
            clayout.nets,
            clayout.board.width_mm,
            clayout.board.height_mm,
            clayout.board.layer_count,
        )

        # Step 7: Generate placement hints
        clayout.placement_hints = self._generate_placement_hints(
            clayout.components,
            clayout.nets
        )

        # Step 8: Generate routing hints
        clayout.routing_hints = self._generate_routing_hints(
            clayout.nets,
            intent
        )

        # Step 9: Apply any necessary overrides
        clayout.overrides = self._determine_overrides(intent, clayout)

        # Step 10: Document AI assumptions
        clayout.ai_assumptions = self._document_assumptions(
            intent,
            user_components,
            inferred_components
        )

        # Step 11: Validate
        validation = self.validator.validate(clayout)
        if not validation.valid:
            clayout = self._fix_validation_errors(clayout, validation)

        return clayout

    # =========================================================================
    # OVERRIDE AUTHORITY
    # =========================================================================

    def create_override(
        self,
        rule_id: str,
        new_value: Any,
        justification: str,
        evidence: str
    ) -> RuleOverride:
        """
        Create a justified rule override.

        Args:
            rule_id: ID of the rule to override
            new_value: New value for the rule
            justification: Engineering reason for override
            evidence: Source/reference for justification

        Returns:
            RuleOverride object
        """
        # Get original value from rules API
        original = self.rules_api.get_rule_value(rule_id) if hasattr(
            self.rules_api, 'get_rule_value'
        ) else None

        return RuleOverride(
            rule_id=rule_id,
            original_value=original,
            new_value=new_value,
            justification=justification,
            evidence=evidence,
            approved_by="AI",
        )

    def review_override(self, override: RuleOverride) -> bool:
        """
        Validate that an override is justified.

        Args:
            override: The override to review

        Returns:
            True if override is valid, False otherwise
        """
        # Check that justification exists
        if not override.justification or len(override.justification) < 10:
            return False

        # Check that evidence exists
        if not override.evidence:
            return False

        return True

    # =========================================================================
    # ESCALATION HANDLING
    # =========================================================================

    def handle_bbl_failure(
        self,
        clayout: ConstitutionalLayout,
        escalation: EscalationReport
    ) -> ConstitutionalLayout:
        """
        Adjust c_layout based on BBL failure report.

        Args:
            clayout: Original c_layout that failed
            escalation: Escalation report from BBL

        Returns:
            Adjusted c_layout to retry
        """

        # Analyze failure type
        if escalation.failure_type == "routing_failed":
            return self._handle_routing_failure(clayout, escalation)
        elif escalation.failure_type == "placement_impossible":
            return self._handle_placement_failure(clayout, escalation)
        elif escalation.failure_type == "rule_violation":
            return self._handle_rule_violation(clayout, escalation)
        else:
            # Unknown failure - return unchanged
            clayout.notes.append(f"Unknown failure: {escalation.failure_type}")
            return clayout

    def _handle_routing_failure(
        self,
        clayout: ConstitutionalLayout,
        escalation: EscalationReport
    ) -> ConstitutionalLayout:
        """Handle routing failure escalation."""

        # Check routing completion
        completion = escalation.routing_completion_pct

        if completion < 50:
            # Major failure - likely need more layers or larger board
            if clayout.board.layer_count == 2:
                # Try 4 layers
                clayout.board.layer_count = 4
                clayout.notes.append(
                    "AI: Upgraded to 4 layers due to routing failure (<50% completion)"
                )
        elif completion < 80:
            # Moderate failure - try relaxing some constraints
            for rule in clayout.rules.recommended:
                if "SPACING" in rule.rule_id:
                    # Create override to relax spacing
                    clayout.overrides.append(RuleOverride(
                        rule_id=rule.rule_id,
                        original_value=rule.parameters.get("value"),
                        new_value=rule.parameters.get("value", 0.15) * 0.8,
                        justification="Relaxed spacing to improve routability",
                        evidence=f"Routing completion was {completion:.0f}%",
                        approved_by="AI",
                    ))
        else:
            # Minor failure - try more routing iterations
            clayout.notes.append(
                f"AI: Routing at {completion:.0f}%. Recommend more iterations."
            )

        return clayout

    def _handle_placement_failure(
        self,
        clayout: ConstitutionalLayout,
        escalation: EscalationReport
    ) -> ConstitutionalLayout:
        """Handle placement failure escalation."""

        # Check density
        density = escalation.placement_density_pct

        if density > 70:
            # Too dense - increase board size
            clayout.board.width_mm *= 1.2
            clayout.board.height_mm *= 1.2
            clayout.notes.append(
                f"AI: Increased board size by 20% due to high density ({density:.0f}%)"
            )

        return clayout

    def _handle_rule_violation(
        self,
        clayout: ConstitutionalLayout,
        escalation: EscalationReport
    ) -> ConstitutionalLayout:
        """Handle rule violation escalation."""

        for rule_id in escalation.violated_rules:
            priority = clayout.rules.get_priority(rule_id)

            if priority == RulePriority.RECOMMENDED:
                # Demote to optional
                clayout.notes.append(
                    f"AI: Demoted {rule_id} to optional due to violation"
                )
            elif priority == RulePriority.OPTIONAL:
                # Already optional, can ignore
                pass
            # INVIOLABLE rules cannot be demoted - need human decision

        return clayout

    def escalate_to_user(self, issue: str, options: List[str]) -> str:
        """
        Escalate an issue to the user for decision.

        Args:
            issue: Description of the issue
            options: Available options

        Returns:
            User's choice (in real system, would prompt user)
        """
        # In a real system, this would prompt the user
        # For now, return the first option as default
        return options[0] if options else ""

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _determine_board_constraints(self, intent: DesignIntent) -> BoardConstraints:
        """Determine board constraints from intent."""

        width, height = 50.0, 40.0  # Default
        if intent.board_size_constraint:
            width, height = intent.board_size_constraint

        layers = intent.layer_constraint or 2

        # Adjust layers based on features
        features = " ".join(intent.required_features).lower()
        if "usb 3" in features or "ddr" in features:
            layers = max(layers, 4)
        if "high-speed" in features or "hdmi" in features:
            layers = max(layers, 4)

        return BoardConstraints(
            width_mm=width,
            height_mm=height,
            layer_count=layers,
        )

    def _select_components(self, intent: DesignIntent) -> List[ComponentDefinition]:
        """Select components based on intent."""

        components = []
        ref_counters = {"U": 1, "J": 1}

        for comp_type in intent.required_components:
            if comp_type == "ESP32":
                components.append(ComponentDefinition(
                    ref_des=f"U{ref_counters['U']}",
                    part_number="ESP32-WROOM-32",
                    footprint="ESP32-WROOM",
                    category=ComponentCategory.MCU,
                    power_dissipation=0.5,
                ))
                ref_counters["U"] += 1

            elif comp_type == "USB-C":
                components.append(ComponentDefinition(
                    ref_des=f"J{ref_counters['J']}",
                    part_number="USB-C-16P",
                    footprint="USB-C",
                    category=ComponentCategory.CONNECTOR,
                ))
                ref_counters["J"] += 1

            elif comp_type in ["USB-MICRO", "USB-A"]:
                components.append(ComponentDefinition(
                    ref_des=f"J{ref_counters['J']}",
                    part_number=f"{comp_type}-CONN",
                    footprint=comp_type,
                    category=ComponentCategory.CONNECTOR,
                ))
                ref_counters["J"] += 1

            elif comp_type == "POWER_SUPPLY":
                components.append(ComponentDefinition(
                    ref_des=f"U{ref_counters['U']}",
                    part_number="AMS1117-3.3",
                    footprint="SOT-223",
                    category=ComponentCategory.REGULATOR,
                    power_dissipation=0.5,
                ))
                ref_counters["U"] += 1

        return components

    def _define_nets(
        self,
        components: List[ComponentDefinition],
        intent: DesignIntent
    ) -> List[NetDefinition]:
        """Define nets based on components and intent using RulesAPI."""

        nets = []

        # Always have GND and power
        nets.append(NetDefinition(
            name="GND",
            net_type=NetType.GND,
            pins=[f"{c.ref_des}.GND" for c in components],
            voltage=0.0,
        ))

        nets.append(NetDefinition(
            name="3V3",
            net_type=NetType.POWER,
            pins=[],  # Will be populated based on components
            voltage=3.3,
        ))

        # USB nets if USB connector present
        has_usb = any("USB" in c.part_number for c in components)
        if has_usb:
            # Get USB specs from RulesAPI
            usb_rules = self.rules_api.validate_usb_layout(0, 0, 90)  # Get specs
            usb_impedance = self.rules_api.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM  # 90 ohm
            usb_mismatch = self.rules_api.rules.usb2.MAX_LENGTH_MISMATCH_MM  # 1.25mm

            # USB data lines with RulesAPI values
            nets.append(NetDefinition(
                name="USB_DP",
                net_type=NetType.DIFF_PAIR,
                pins=[],
                impedance_ohm=usb_impedance,
                matched_with="USB_DM",
                max_mismatch_mm=usb_mismatch,
            ))
            nets.append(NetDefinition(
                name="USB_DM",
                net_type=NetType.DIFF_PAIR,
                pins=[],
                impedance_ohm=usb_impedance,
                matched_with="USB_DP",
                max_mismatch_mm=usb_mismatch,
            ))

            # VBUS - get current capacity from RulesAPI
            # USB 2.0 allows 500mA (0.5A) for standard, 1.5A for BC1.2
            vbus_current = 0.5  # Standard USB 2.0
            nets.append(NetDefinition(
                name="VBUS",
                net_type=NetType.POWER,
                pins=[],
                voltage=5.0,
                current_max=vbus_current,
            ))

        return nets

    def _generate_placement_hints(
        self,
        components: List[ComponentDefinition],
        nets: List[NetDefinition]
    ) -> PlacementHints:
        """Generate placement hints based on component relationships using RulesAPI."""

        hints = PlacementHints()

        # Get placement constraints from RulesAPI
        decoupling_distance = self.rules_api.get_decoupling_distance()  # 3.0mm from rules
        crystal_distance = self.rules_api.get_crystal_distance()  # 5.0mm from rules
        analog_separation = self.rules_api.get_analog_separation()  # 10.0mm from rules

        # Find MCUs
        mcus = [c for c in components if c.category == ComponentCategory.MCU]

        # Decoupling caps near MCUs - use RulesAPI value
        for mcu in mcus:
            caps_for_mcu = [
                c.ref_des for c in components
                if c.inferred_for == mcu.ref_des
                and c.category == ComponentCategory.CAPACITOR
            ]
            if caps_for_mcu:
                hints.proximity_groups.append(ProximityGroup(
                    components=[mcu.ref_des] + caps_for_mcu,
                    max_distance_mm=decoupling_distance,
                    reason=f"Decoupling capacitors must be within {decoupling_distance}mm of MCU power pins (IPC-2221B)",
                ))

        # Crystals near MCUs - use RulesAPI value
        crystals = [c for c in components if c.category == ComponentCategory.CRYSTAL]
        for crystal in crystals:
            if crystal.inferred_for:
                hints.proximity_groups.append(ProximityGroup(
                    components=[crystal.inferred_for, crystal.ref_des],
                    max_distance_mm=crystal_distance,
                    reason=f"Crystal must be within {crystal_distance}mm of MCU oscillator pins",
                ))

        # Connectors on edge
        connectors = [
            c.ref_des for c in components
            if c.category == ComponentCategory.CONNECTOR
        ]
        hints.edge_components = connectors

        # Keep RF away from digital - use RulesAPI analog_separation value
        rf_comps = [c.ref_des for c in components if "RF" in c.part_number.upper()]
        digital_comps = [c.ref_des for c in components if c.category == ComponentCategory.MCU]
        for rf in rf_comps:
            for digital in digital_comps:
                hints.keep_apart.append(KeepApart(
                    component_a=rf,
                    component_b=digital,
                    min_distance_mm=analog_separation,
                    reason=f"RF isolation requires minimum {analog_separation}mm from digital noise sources",
                ))

        # Keep analog away from digital switching noise
        analog_comps = [c.ref_des for c in components if c.category == ComponentCategory.ANALOG]
        for analog in analog_comps:
            for digital in digital_comps:
                hints.keep_apart.append(KeepApart(
                    component_a=analog,
                    component_b=digital,
                    min_distance_mm=analog_separation,
                    reason=f"Analog components need {analog_separation}mm separation from digital noise",
                ))

        return hints

    def _generate_routing_hints(
        self,
        nets: List[NetDefinition],
        intent: DesignIntent
    ) -> RoutingHints:
        """Generate routing hints based on net requirements using RulesAPI."""

        hints = RoutingHints()

        # Get default differential pair specs from RulesAPI
        default_diff_impedance = self.rules_api.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM  # 90 ohm
        default_diff_mismatch = self.rules_api.rules.usb2.MAX_LENGTH_MISMATCH_MM  # 1.25mm

        # Check for high-speed interfaces and get their specific rules
        features = " ".join(intent.required_features).lower() if intent.required_features else ""

        # Find diff pairs
        for net in nets:
            if net.matched_with:
                matching_net = next((n for n in nets if n.name == net.matched_with), None)
                if matching_net:
                    # Only add if not already added
                    existing = [d.positive_net for d in hints.diff_pairs]
                    if net.name not in existing and net.matched_with not in existing:
                        # Determine impedance and mismatch based on net type
                        impedance = net.impedance_ohm
                        mismatch = net.max_mismatch_mm

                        # Use RulesAPI defaults if not specified on net
                        if not impedance:
                            if "USB" in net.name.upper():
                                impedance = self.rules_api.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM
                            elif "HDMI" in net.name.upper():
                                hdmi_rules = self.rules_api.get_hdmi_rules("HDMI_1.4")
                                impedance = hdmi_rules.get("diff_impedance_ohm", 100.0)
                            else:
                                impedance = default_diff_impedance

                        if not mismatch:
                            if "USB" in net.name.upper():
                                mismatch = self.rules_api.rules.usb2.MAX_LENGTH_MISMATCH_MM
                            else:
                                mismatch = default_diff_mismatch

                        hints.diff_pairs.append(DiffPairSpec(
                            positive_net=net.name,
                            negative_net=net.matched_with,
                            impedance_ohm=impedance,
                            max_mismatch_mm=mismatch,
                        ))

        # Add length matching groups for high-speed buses
        if "ddr3" in features:
            ddr3_rules = self.rules_api.get_ddr3_rules()
            hints.length_match_groups.append(LengthMatchGroup(
                nets=["DQ0", "DQ1", "DQ2", "DQ3", "DQ4", "DQ5", "DQ6", "DQ7"],
                max_mismatch_mm=ddr3_rules.get("dq_length_match_mm", 12.7),
                reason=f"DDR3 DQ length matching (max {ddr3_rules.get('dq_length_match_mm', 12.7)}mm)",
            ))

        if "ddr4" in features:
            ddr4_rules = self.rules_api.get_ddr4_rules()
            hints.length_match_groups.append(LengthMatchGroup(
                nets=["DQ0", "DQ1", "DQ2", "DQ3", "DQ4", "DQ5", "DQ6", "DQ7"],
                max_mismatch_mm=ddr4_rules.get("dq_length_match_mm", 6.35),
                reason=f"DDR4 DQ length matching (max {ddr4_rules.get('dq_length_match_mm', 6.35)}mm)",
            ))

        # Priority nets - power and critical signals first
        power_nets = [n.name for n in nets if n.net_type in [NetType.POWER, NetType.GND]]
        diff_pair_nets = [n.name for n in nets if n.net_type == NetType.DIFF_PAIR]
        hints.priority_nets = power_nets + diff_pair_nets

        # GND last (will be poured)
        hints.deprioritized_nets = ["GND"]

        return hints

    def _determine_overrides(
        self,
        intent: DesignIntent,
        clayout: ConstitutionalLayout
    ) -> List[RuleOverride]:
        """Determine if any rule overrides are needed using RulesAPI values."""

        overrides = []

        # Get original values from RulesAPI
        usb_hs_mismatch = self.rules_api.rules.usb2.MAX_LENGTH_MISMATCH_MM  # 1.25mm for HS
        usb_fs_mismatch = 5.0  # Full-Speed allows 5mm tolerance

        # Check if USB Full-Speed mode allows relaxed matching
        features = " ".join(intent.required_features).lower() if intent.required_features else ""
        if "full-speed" in features or "12mbps" in features:
            overrides.append(RuleOverride(
                rule_id="USB2_LENGTH_MATCHING",
                original_value=usb_hs_mismatch,
                new_value=usb_fs_mismatch,
                justification="USB Full-Speed mode allows relaxed length matching",
                evidence="USB 2.0 Specification - Full-Speed has 5mm tolerance vs High-Speed 1.25mm",
                approved_by="AI",
            ))

        # Check for low-power designs that can relax current capacity rules
        if "low power" in features or "battery" in features:
            # Get standard current capacity
            standard_current = self.rules_api.get_current_capacity(0.2, 1.0)  # 0.2mm trace, 1oz copper
            overrides.append(RuleOverride(
                rule_id="CURRENT_CAPACITY",
                original_value=standard_current,
                new_value=standard_current * 0.8,  # Allow 80% for battery designs
                justification="Low-power battery design allows reduced trace capacity margins",
                evidence="IPC-2152 allows derating for intermittent loads",
                approved_by="AI",
            ))

        return overrides

    def _document_assumptions(
        self,
        intent: DesignIntent,
        user_components: List[ComponentDefinition],
        inferred_components: List[ComponentDefinition]
    ) -> List[str]:
        """Document assumptions made by AI."""

        assumptions = []

        # Board size assumption
        if not intent.board_size_constraint:
            assumptions.append(
                "Assumed default board size of 50x40mm"
            )

        # Layer assumption
        if not intent.layer_constraint:
            assumptions.append(
                "Assumed 2-layer board (sufficient for this design)"
            )

        # Component inference
        if inferred_components:
            assumptions.append(
                f"Added {len(inferred_components)} supporting components "
                f"(decoupling, ESD protection, etc.)"
            )

        return assumptions

    def _fix_validation_errors(
        self,
        clayout: ConstitutionalLayout,
        validation: Any
    ) -> ConstitutionalLayout:
        """Attempt to fix validation errors automatically."""

        # For now, just note the errors
        for error in validation.errors:
            clayout.notes.append(f"Validation error: {error}")

        return clayout


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_clayout_from_text(user_input: str) -> ConstitutionalLayout:
    """
    One-shot function to create a c_layout from natural language.

    Args:
        user_input: Natural language design description

    Returns:
        Complete ConstitutionalLayout
    """
    ai = CircuitAI()
    intent = ai.parse_user_intent(user_input)
    return ai.generate_clayout(intent)


def get_clarifying_questions(user_input: str) -> List[str]:
    """
    Get questions that need to be answered before design can proceed.

    Args:
        user_input: Natural language design description

    Returns:
        List of questions for the user
    """
    ai = CircuitAI()
    intent = ai.parse_user_intent(user_input)
    return ai.ask_clarifying_questions(intent)
