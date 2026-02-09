"""
Circuit Pattern Library
=======================

A knowledge base of common circuit patterns and their design rules.
This is the "experience" of a human expert encoded into data.

Each pattern describes:
- What components are involved
- How they should be connected
- Critical loops and nodes
- Placement rules
- Routing rules
- Common mistakes to avoid
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable
from .circuit_types import (
    CircuitFunction, NetFunction, ComponentFunction,
    Severity, DesignIssue
)


@dataclass
class PlacementRule:
    """A placement rule for a pattern."""
    description: str
    component_role: str  # e.g., 'INPUT_CAP', 'SWITCH'
    target_role: str  # e.g., 'REGULATOR'
    max_distance: float  # mm
    preferred_position: str = 'NEAR'  # 'NEAR', 'ABOVE', 'BELOW', 'LEFT', 'RIGHT'
    severity: Severity = Severity.WARNING


@dataclass
class RoutingRule:
    """A routing rule for a pattern."""
    description: str
    net_role: str  # e.g., 'SWITCH_NODE', 'FEEDBACK'
    min_width: Optional[float] = None
    max_length: Optional[float] = None
    preferred_layer: Optional[str] = None
    max_vias: Optional[int] = None
    minimize_loop: bool = False
    keep_away_from: List[str] = field(default_factory=list)
    severity: Severity = Severity.WARNING


@dataclass
class CriticalLoop:
    """A critical current loop in a pattern."""
    name: str
    description: str
    component_roles: List[str]  # In loop order
    max_area: float  # mm²
    severity: Severity = Severity.ERROR


@dataclass
class CircuitPattern:
    """
    A recognized circuit pattern with all its design knowledge.

    This is the core of the expert system - each pattern encodes
    years of engineering experience.
    """
    name: str
    description: str
    function: CircuitFunction

    # Component roles (abstract, not specific refs)
    component_roles: Dict[str, Dict] = field(default_factory=dict)
    # Example: {'REGULATOR': {'type': 'IC', 'functions': [VOLTAGE_REGULATOR]}}

    # How to detect this pattern in a design
    detection_rules: List[str] = field(default_factory=list)

    # Design rules
    placement_rules: List[PlacementRule] = field(default_factory=list)
    routing_rules: List[RoutingRule] = field(default_factory=list)
    critical_loops: List[CriticalLoop] = field(default_factory=list)

    # Common mistakes
    common_mistakes: List[str] = field(default_factory=list)

    # Thermal considerations
    hot_components: List[str] = field(default_factory=list)  # Component roles
    thermal_notes: str = ""


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

BUCK_CONVERTER = CircuitPattern(
    name='BUCK_CONVERTER',
    description='Step-down switching regulator',
    function=CircuitFunction.POWER_REGULATION,

    component_roles={
        'CONTROLLER': {
            'type': 'IC',
            'typical_parts': ['LM2596', 'TPS54331', 'MP1584'],
            'pins': ['VIN', 'SW', 'GND', 'FB', 'EN'],
        },
        'INPUT_CAP_BULK': {
            'type': 'CAPACITOR',
            'typical_values': ['100uF', '220uF', '470uF'],
            'function': 'Supplies switching current',
        },
        'INPUT_CAP_CERAMIC': {
            'type': 'CAPACITOR',
            'typical_values': ['10uF', '22uF'],
            'function': 'High-frequency bypass',
        },
        'OUTPUT_CAP': {
            'type': 'CAPACITOR',
            'typical_values': ['100uF', '220uF'],
            'function': 'Output filtering',
        },
        'INDUCTOR': {
            'type': 'INDUCTOR',
            'typical_values': ['10uH', '22uH', '47uH'],
        },
        'CATCH_DIODE': {
            'type': 'DIODE',
            'typical_parts': ['SS34', 'SS54', 'B340'],
            'function': 'Freewheeling diode (if not synchronous)',
        },
        'FEEDBACK_DIVIDER_TOP': {
            'type': 'RESISTOR',
        },
        'FEEDBACK_DIVIDER_BOTTOM': {
            'type': 'RESISTOR',
        },
    },

    detection_rules=[
        'Has IC with SW (switch) pin',
        'Has inductor connected to SW',
        'Has input and output capacitors',
        'May have catch diode (non-synchronous) or not (synchronous)',
    ],

    placement_rules=[
        PlacementRule(
            description='Input ceramic cap must be very close to VIN and GND pins',
            component_role='INPUT_CAP_CERAMIC',
            target_role='CONTROLLER',
            max_distance=3.0,
            severity=Severity.CRITICAL,
        ),
        PlacementRule(
            description='Catch diode close to SW pin and GND',
            component_role='CATCH_DIODE',
            target_role='CONTROLLER',
            max_distance=5.0,
            severity=Severity.ERROR,
        ),
        PlacementRule(
            description='Inductor close to SW pin',
            component_role='INDUCTOR',
            target_role='CONTROLLER',
            max_distance=5.0,
            severity=Severity.ERROR,
        ),
        PlacementRule(
            description='Feedback resistors near FB pin',
            component_role='FEEDBACK_DIVIDER_TOP',
            target_role='CONTROLLER',
            max_distance=5.0,
            severity=Severity.WARNING,
        ),
    ],

    routing_rules=[
        RoutingRule(
            description='Switch node: Keep SHORT and WIDE. This is HOT!',
            net_role='SWITCH_NODE',
            max_length=10.0,
            min_width=0.5,
            minimize_loop=True,
            severity=Severity.CRITICAL,
        ),
        RoutingRule(
            description='Feedback: Keep away from switch node, route away from noise',
            net_role='FEEDBACK',
            max_length=20.0,
            keep_away_from=['SWITCH_NODE'],
            severity=Severity.WARNING,
        ),
        RoutingRule(
            description='Power ground: Star point at input cap',
            net_role='POWER_GROUND',
            min_width=0.5,
            severity=Severity.ERROR,
        ),
    ],

    critical_loops=[
        CriticalLoop(
            name='HOT_LOOP',
            description='Input cap → Controller VIN → SW → Diode → Input cap GND',
            component_roles=['INPUT_CAP_CERAMIC', 'CONTROLLER', 'CATCH_DIODE'],
            max_area=50.0,
            severity=Severity.CRITICAL,
        ),
    ],

    common_mistakes=[
        'Input cap too far from controller - causes voltage spikes',
        'Switch node trace too long - EMI antenna',
        'Feedback routed under inductor - noise pickup',
        'No input ceramic cap - only electrolytic',
        'Ground loop between input and output',
    ],

    hot_components=['CONTROLLER', 'CATCH_DIODE', 'INDUCTOR'],
    thermal_notes='Controller and diode dissipate most heat. Use copper pour under diode.',
)


LDO_REGULATOR = CircuitPattern(
    name='LDO_REGULATOR',
    description='Low dropout linear regulator',
    function=CircuitFunction.POWER_REGULATION,

    component_roles={
        'REGULATOR': {
            'type': 'IC',
            'typical_parts': ['AMS1117', 'LM1117', 'MIC5219', 'AP2112'],
            'pins': ['VIN', 'VOUT', 'GND'],
        },
        'INPUT_CAP': {
            'type': 'CAPACITOR',
            'typical_values': ['10uF', '22uF'],
        },
        'OUTPUT_CAP': {
            'type': 'CAPACITOR',
            'typical_values': ['10uF', '22uF'],
            'note': 'Check datasheet for ESR requirements!',
        },
    },

    placement_rules=[
        PlacementRule(
            description='Input cap close to VIN pin',
            component_role='INPUT_CAP',
            target_role='REGULATOR',
            max_distance=3.0,
            severity=Severity.ERROR,
        ),
        PlacementRule(
            description='Output cap close to VOUT pin',
            component_role='OUTPUT_CAP',
            target_role='REGULATOR',
            max_distance=3.0,
            severity=Severity.ERROR,
        ),
    ],

    routing_rules=[
        RoutingRule(
            description='Keep input and output current paths separate - no shared ground trace',
            net_role='GROUND',
            min_width=0.3,
            severity=Severity.WARNING,
        ),
    ],

    common_mistakes=[
        'Wrong output cap ESR - many LDOs need specific ESR for stability',
        'Input cap too small - input voltage dips during load transients',
        'Shared ground trace between input and output - noise coupling',
    ],

    hot_components=['REGULATOR'],
    thermal_notes='Heat = (VIN - VOUT) × Current. Use copper pour for heat dissipation.',
)


I2C_BUS = CircuitPattern(
    name='I2C_BUS',
    description='I2C communication bus',
    function=CircuitFunction.DIGITAL_INTERFACE,

    component_roles={
        'MASTER': {
            'type': 'IC',
            'note': 'MCU or I2C master device',
        },
        'SLAVE': {
            'type': 'IC',
            'note': 'Sensor, EEPROM, or other I2C device',
        },
        'PULLUP_SDA': {
            'type': 'RESISTOR',
            'typical_values': ['2.2k', '4.7k', '10k'],
            'note': 'Value depends on bus speed and capacitance',
        },
        'PULLUP_SCL': {
            'type': 'RESISTOR',
            'typical_values': ['2.2k', '4.7k', '10k'],
        },
    },

    placement_rules=[
        PlacementRule(
            description='Pullup resistors near master device',
            component_role='PULLUP_SDA',
            target_role='MASTER',
            max_distance=10.0,
            severity=Severity.WARNING,
        ),
    ],

    routing_rules=[
        RoutingRule(
            description='SDA and SCL parallel, similar length',
            net_role='SDA',
            severity=Severity.INFO,
        ),
        RoutingRule(
            description='Keep I2C away from high-speed signals',
            net_role='SCL',
            keep_away_from=['SWITCH_NODE', 'CLOCK'],
            severity=Severity.WARNING,
        ),
    ],

    common_mistakes=[
        'Pullups on wrong end of bus (should be near master or at one end)',
        'Bus too long for speed (100kHz: 1m, 400kHz: 0.3m, 1MHz: 0.1m)',
        'Multiple pullup sets on same bus',
        'SDA/SCL routed far apart - EMI susceptibility',
    ],
)


USB_2_INTERFACE = CircuitPattern(
    name='USB_2_INTERFACE',
    description='USB 2.0 Full/High Speed interface',
    function=CircuitFunction.DIGITAL_INTERFACE,

    component_roles={
        'CONNECTOR': {
            'type': 'CONNECTOR',
            'typical_parts': ['USB-A', 'USB-B', 'USB-C', 'Micro-USB'],
        },
        'ESD_PROTECTION': {
            'type': 'IC',
            'typical_parts': ['USBLC6-2', 'TPD2E001'],
            'note': 'Critical for real-world reliability',
        },
        'SERIES_RESISTORS': {
            'type': 'RESISTOR',
            'typical_values': ['22R', '27R', '33R'],
            'note': 'Some MCUs have internal, check datasheet',
        },
    },

    placement_rules=[
        PlacementRule(
            description='ESD protection as close as possible to connector',
            component_role='ESD_PROTECTION',
            target_role='CONNECTOR',
            max_distance=5.0,
            severity=Severity.CRITICAL,
        ),
    ],

    routing_rules=[
        RoutingRule(
            description='D+ and D- must be length matched within 0.5mm',
            net_role='USB_DP',
            severity=Severity.ERROR,
        ),
        RoutingRule(
            description='90 ohm differential impedance',
            net_role='USB_DM',
            min_width=0.2,  # Depends on stackup
            severity=Severity.ERROR,
        ),
        RoutingRule(
            description='Minimize vias on USB data lines',
            net_role='USB_DP',
            max_vias=1,
            severity=Severity.WARNING,
        ),
    ],

    common_mistakes=[
        'D+/D- length mismatch > 2mm - signal integrity issues',
        'No ESD protection - fails ESD testing',
        'Routing under switching nodes - noise injection',
        'Sharp 90° bends - impedance discontinuity',
    ],
)


BYPASS_CAPACITOR = CircuitPattern(
    name='BYPASS_CAPACITOR',
    description='IC power bypass/decoupling capacitor',
    function=CircuitFunction.POWER_DISTRIBUTION,

    component_roles={
        'IC': {
            'type': 'IC',
            'note': 'Any IC requiring bypass',
        },
        'BYPASS_CAP': {
            'type': 'CAPACITOR',
            'typical_values': ['100nF'],
            'note': 'One per VCC pin, as close as possible',
        },
    },

    placement_rules=[
        PlacementRule(
            description='Bypass cap must be within 3mm of IC VCC pin',
            component_role='BYPASS_CAP',
            target_role='IC',
            max_distance=3.0,
            severity=Severity.ERROR,
        ),
    ],

    routing_rules=[
        RoutingRule(
            description='Direct trace from cap to VCC pin - no detours',
            net_role='VCC',
            max_length=5.0,
            severity=Severity.WARNING,
        ),
        RoutingRule(
            description='Short return path from cap GND to IC GND',
            net_role='GND',
            max_length=5.0,
            severity=Severity.WARNING,
        ),
    ],

    common_mistakes=[
        'Cap too far from IC - defeats the purpose',
        'Via between cap and IC pin - adds inductance',
        'Shared GND trace with other components',
    ],
)


CRYSTAL_OSCILLATOR = CircuitPattern(
    name='CRYSTAL_OSCILLATOR',
    description='Crystal oscillator circuit',
    function=CircuitFunction.DIGITAL_CORE,

    component_roles={
        'MCU': {
            'type': 'IC',
            'note': 'Microcontroller with crystal pins',
        },
        'CRYSTAL': {
            'type': 'CRYSTAL',
            'typical_parts': ['8MHz', '12MHz', '16MHz', '25MHz'],
        },
        'LOAD_CAP_1': {
            'type': 'CAPACITOR',
            'typical_values': ['12pF', '15pF', '18pF', '22pF'],
        },
        'LOAD_CAP_2': {
            'type': 'CAPACITOR',
            'typical_values': ['12pF', '15pF', '18pF', '22pF'],
        },
    },

    placement_rules=[
        PlacementRule(
            description='Crystal and load caps MUST be close to MCU',
            component_role='CRYSTAL',
            target_role='MCU',
            max_distance=5.0,
            severity=Severity.CRITICAL,
        ),
    ],

    routing_rules=[
        RoutingRule(
            description='Keep crystal traces short and away from other signals',
            net_role='XTAL_IN',
            max_length=10.0,
            keep_away_from=['SWITCH_NODE', 'PWM'],
            severity=Severity.ERROR,
        ),
        RoutingRule(
            description='Ground plane under crystal - no routing',
            net_role='XTAL_OUT',
            severity=Severity.WARNING,
        ),
    ],

    common_mistakes=[
        'Crystal too far from MCU - may not oscillate',
        'Signal traces routed under crystal - noise injection',
        'Wrong load cap values - frequency error',
        'No ground plane under crystal',
    ],
)


LED_WITH_RESISTOR = CircuitPattern(
    name='LED_WITH_RESISTOR',
    description='LED indicator with current limiting resistor',
    function=CircuitFunction.USER_INTERFACE,

    component_roles={
        'LED': {
            'type': 'LED',
        },
        'RESISTOR': {
            'type': 'RESISTOR',
            'note': 'R = (Vsupply - Vf) / If',
        },
    },

    placement_rules=[
        PlacementRule(
            description='Resistor near LED for clear schematic-to-layout correspondence',
            component_role='RESISTOR',
            target_role='LED',
            max_distance=10.0,
            severity=Severity.INFO,
        ),
    ],

    common_mistakes=[
        'No current limiting resistor - LED burns out',
        'Wrong resistor value - too dim or too bright',
    ],
)


# =============================================================================
# PATTERN LIBRARY CLASS
# =============================================================================

class PatternLibrary:
    """
    Library of all known circuit patterns.

    Usage:
        library = PatternLibrary()
        patterns = library.detect_patterns(parts_db)
        for pattern, matches in patterns:
            print(f"Found {pattern.name}: {matches}")
    """

    def __init__(self):
        self.patterns: Dict[str, CircuitPattern] = {
            'BUCK_CONVERTER': BUCK_CONVERTER,
            'LDO_REGULATOR': LDO_REGULATOR,
            'I2C_BUS': I2C_BUS,
            'USB_2_INTERFACE': USB_2_INTERFACE,
            'BYPASS_CAPACITOR': BYPASS_CAPACITOR,
            'CRYSTAL_OSCILLATOR': CRYSTAL_OSCILLATOR,
            'LED_WITH_RESISTOR': LED_WITH_RESISTOR,
        }

    def get_pattern(self, name: str) -> Optional[CircuitPattern]:
        """Get a pattern by name."""
        return self.patterns.get(name)

    def detect_patterns(self, parts_db: Dict) -> List[Tuple[CircuitPattern, Dict]]:
        """
        Detect which patterns exist in a design.

        Returns list of (pattern, role_mapping) tuples.
        role_mapping maps pattern roles to actual component refs.
        """
        detected = []

        # Detect I2C bus
        i2c_match = self._detect_i2c(parts_db)
        if i2c_match:
            detected.append((self.patterns['I2C_BUS'], i2c_match))

        # Detect bypass caps
        bypass_matches = self._detect_bypass_caps(parts_db)
        for match in bypass_matches:
            detected.append((self.patterns['BYPASS_CAPACITOR'], match))

        # Detect LED circuits
        led_matches = self._detect_leds(parts_db)
        for match in led_matches:
            detected.append((self.patterns['LED_WITH_RESISTOR'], match))

        # Detect voltage regulators
        reg_match = self._detect_regulators(parts_db)
        if reg_match:
            detected.append(reg_match)

        return detected

    def _detect_i2c(self, parts_db: Dict) -> Optional[Dict]:
        """Detect I2C bus pattern."""
        nets = parts_db.get('nets', {})

        # Look for SDA and SCL nets
        sda_net = None
        scl_net = None
        for net_name in nets.keys():
            name_upper = net_name.upper()
            if 'SDA' in name_upper:
                sda_net = net_name
            if 'SCL' in name_upper:
                scl_net = net_name

        if sda_net and scl_net:
            return {
                'SDA_NET': sda_net,
                'SCL_NET': scl_net,
            }
        return None

    def _detect_bypass_caps(self, parts_db: Dict) -> List[Dict]:
        """Detect bypass capacitor patterns."""
        matches = []
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Find all 100nF caps connected to VCC/GND
        for ref, part in parts.items():
            if not ref.startswith('C'):
                continue
            value = part.get('value', '')
            if '100n' in value.lower() or '0.1u' in value.lower():
                # Check if connected to VCC and GND
                pins = part.get('pins', [])
                has_vcc = any('VCC' in p.get('net', '').upper() or
                             'VDD' in p.get('net', '').upper()
                             for p in pins)
                has_gnd = any('GND' in p.get('net', '').upper() or
                             'VSS' in p.get('net', '').upper()
                             for p in pins)
                if has_vcc and has_gnd:
                    matches.append({'BYPASS_CAP': ref})

        return matches

    def _detect_leds(self, parts_db: Dict) -> List[Dict]:
        """Detect LED with resistor patterns."""
        matches = []
        parts = parts_db.get('parts', {})

        for ref, part in parts.items():
            if ref.startswith('D') and 'LED' in part.get('value', '').upper():
                # Found an LED, look for series resistor
                # This would need more sophisticated analysis
                matches.append({'LED': ref})

        return matches

    def _detect_regulators(self, parts_db: Dict) -> Optional[Tuple[CircuitPattern, Dict]]:
        """Detect voltage regulator patterns."""
        parts = parts_db.get('parts', {})

        for ref, part in parts.items():
            if not ref.startswith('U'):
                continue
            value = part.get('value', '').upper()

            # Check for known regulator part numbers
            if any(reg in value for reg in ['LM2596', 'TPS54', 'MP1584', 'LM2576']):
                return (self.patterns['BUCK_CONVERTER'], {'CONTROLLER': ref})

            if any(reg in value for reg in ['AMS1117', 'LM1117', 'LM7805', 'AP2112', 'MIC5219']):
                return (self.patterns['LDO_REGULATOR'], {'REGULATOR': ref})

        return None

    def get_rules_for_component(self, component_ref: str, parts_db: Dict) -> List[Dict]:
        """Get all applicable rules for a component."""
        rules = []
        detected = self.detect_patterns(parts_db)

        for pattern, mapping in detected:
            for role, ref in mapping.items():
                if ref == component_ref:
                    # This component is part of this pattern
                    for rule in pattern.placement_rules:
                        if rule.component_role == role:
                            rules.append({
                                'pattern': pattern.name,
                                'rule': rule,
                            })

        return rules
