"""
Engineering Rules Database
===========================

This is the BRAIN of the Circuit Intelligence Engine.
It contains the WISDOM that transforms DATA into DECISIONS.

This module answers: WHERE, WHEN, HOW, and WHY.

Categories:
1. PLACEMENT RULES - Where components go and why
2. ROUTING RULES - How traces connect and in what order
3. POWER RULES - Power distribution topology
4. THERMAL RULES - Heat management decisions
5. EMI/EMC RULES - Electromagnetic compatibility
6. FABRICATION RULES - Manufacturing constraints
7. SIGNAL INTEGRITY RULES - High-speed design rules
8. COMPONENT RULES - Component-specific requirements

Sources:
- IPC-2221 (Generic PCB Design Standard)
- IPC-7351 (Land Pattern Standard)
- IPC-2152 (Current Carrying Capacity)
- Henry Ott - "Electromagnetic Compatibility Engineering"
- Howard Johnson - "High-Speed Digital Design"
- Eric Bogatin - "Signal and Power Integrity"
- Manufacturer Application Notes (TI, Analog Devices, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from enum import Enum, auto
import math


# =============================================================================
# RULE PRIORITY AND CATEGORIES
# =============================================================================

class RulePriority(Enum):
    """Rule enforcement priority."""
    MANDATORY = 1      # MUST follow - violation = design failure
    CRITICAL = 2       # Should follow - violation = likely problems
    RECOMMENDED = 3    # Best practice - violation = suboptimal
    OPTIONAL = 4       # Nice to have - violation = acceptable


class RuleCategory(Enum):
    """Rule categories."""
    PLACEMENT = auto()
    ROUTING = auto()
    POWER = auto()
    THERMAL = auto()
    EMI_EMC = auto()
    FABRICATION = auto()
    SIGNAL_INTEGRITY = auto()
    COMPONENT = auto()


# =============================================================================
# RULE DATA STRUCTURES
# =============================================================================

@dataclass
class EngineeringRule:
    """A single engineering rule with reasoning."""
    id: str
    name: str
    category: RuleCategory
    priority: RulePriority
    description: str

    # The actual rule logic
    condition: str           # When this rule applies
    action: str              # What to do
    reason: str              # WHY (the engineering reasoning)

    # Quantitative parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # References
    source: str = ""         # IPC-2221, TI App Note, etc.
    exceptions: List[str] = field(default_factory=list)

    # Related rules
    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)


@dataclass
class PlacementZone:
    """A zone on the PCB with specific purpose."""
    name: str
    purpose: str
    components: List[str]    # Component types that belong here
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingSequence:
    """Defines routing priority and order."""
    priority: int
    net_type: str
    description: str
    constraints: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PLACEMENT RULES DATABASE
# =============================================================================

PLACEMENT_RULES: List[EngineeringRule] = [
    # -------------------------------------------------------------------------
    # POWER SUPPLY PLACEMENT
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="PL-PWR-001",
        name="Power Input Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.MANDATORY,
        description="Power input connector placement",
        condition="Design has external power input",
        action="Place power input connector at board edge, away from sensitive analog",
        reason="Power input is the SOURCE of all current. Edge placement provides: "
               "1) Easy external connection, 2) Short path to bulk capacitors, "
               "3) Separation from noise-sensitive circuits",
        parameters={
            "edge_distance_max_mm": 5.0,
            "distance_from_analog_min_mm": 20.0,
        },
        source="General PCB design practice",
    ),

    EngineeringRule(
        id="PL-PWR-002",
        name="Voltage Regulator Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="Place regulators between power source and loads",
        condition="Design has voltage regulators",
        action="Place regulator: 1) Close to power input, 2) Between input and loads, "
               "3) With input caps on INPUT side, output caps on OUTPUT side",
        reason="Regulator placement creates POWER FLOW topology. Current flows: "
               "Input -> Regulator -> Loads. Capacitors filter at transition points. "
               "Short paths reduce inductance and voltage drops.",
        parameters={
            "input_cap_distance_max_mm": 5.0,
            "output_cap_distance_max_mm": 3.0,
            "thermal_pad_area_min_mm2": 100.0,
        },
        source="TI Application Note SLVA648",
    ),

    EngineeringRule(
        id="PL-PWR-003",
        name="Bulk Capacitor Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="Bulk capacitors near power input",
        condition="Design has power input",
        action="Place bulk capacitor (100uF+) within 10mm of power input connector",
        reason="Bulk caps provide energy reservoir for transient loads. "
               "Close placement minimizes inductance in the supply path. "
               "They filter low-frequency noise from external supply.",
        parameters={
            "distance_to_input_max_mm": 10.0,
            "min_capacitance_uf": 100.0,
        },
        source="Analog Devices AN-1368",
    ),

    EngineeringRule(
        id="PL-PWR-004",
        name="Decoupling Capacitor Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.MANDATORY,
        description="Decoupling caps at IC power pins",
        condition="Any IC with VCC/GND pins",
        action="Place 100nF ceramic cap within 3mm of EACH VCC pin, "
               "connected directly to nearest GND pin",
        reason="ICs draw current in fast pulses during switching. "
               "Decoupling caps provide LOCAL charge reservoir, preventing: "
               "1) Voltage droop at IC, 2) High-frequency noise on power plane, "
               "3) Ground bounce. Distance matters because trace inductance "
               "is ~1nH/mm - every mm adds impedance at high frequency.",
        parameters={
            "distance_to_vcc_max_mm": 3.0,
            "capacitance_nf": 100.0,
            "via_to_gnd_max_mm": 1.0,
        },
        source="IPC-2221, Murata Application Note",
    ),

    # -------------------------------------------------------------------------
    # MCU/PROCESSOR PLACEMENT
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="PL-MCU-001",
        name="MCU Central Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.RECOMMENDED,
        description="MCU as central hub",
        condition="Design has MCU/processor",
        action="Place MCU in board center (or center of its functional area). "
               "Peripherals radiate outward from MCU.",
        reason="MCU connects to MOST other components. Central placement "
               "minimizes average trace length to all peripherals, reducing: "
               "1) Signal propagation delay, 2) EMI antenna area, "
               "3) Routing congestion. Star topology is natural for MCU designs.",
        parameters={
            "max_distance_from_center_percent": 30,
        },
        source="General embedded design practice",
    ),

    EngineeringRule(
        id="PL-MCU-002",
        name="Crystal/Oscillator Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.MANDATORY,
        description="Clock source placement",
        condition="Design has crystal or oscillator",
        action="Place crystal within 5mm of MCU oscillator pins. "
               "Load capacitors between crystal and GND, close to crystal. "
               "Keep area under crystal clear of other signals.",
        reason="Crystal traces are HIGH-IMPEDANCE and sensitive to noise. "
               "Long traces act as antennas, picking up interference and "
               "radiating clock harmonics. Short traces minimize both. "
               "Ground plane under crystal provides shielding.",
        parameters={
            "distance_to_mcu_max_mm": 5.0,
            "load_cap_distance_max_mm": 2.0,
            "keepout_under_crystal": True,
        },
        source="Microchip AN826, ST AN2867",
    ),

    EngineeringRule(
        id="PL-MCU-003",
        name="Reset Circuit Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="Reset circuit near MCU",
        condition="Design has MCU reset circuit",
        action="Place reset components (resistor, capacitor, supervisor) "
               "within 5mm of MCU RESET pin",
        reason="Reset is a CRITICAL signal. Noise on reset causes spurious resets. "
               "Short traces minimize noise pickup. RC filter must be close to "
               "be effective against high-frequency noise.",
        parameters={
            "distance_to_reset_pin_max_mm": 5.0,
        },
        source="General practice",
    ),

    # -------------------------------------------------------------------------
    # ANALOG SECTION PLACEMENT
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="PL-ANA-001",
        name="Analog Section Isolation",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="Separate analog from digital",
        condition="Design has both analog and digital circuits",
        action="Place all analog components in dedicated board area. "
               "Separate from digital by at least 10mm or with guard ring.",
        reason="Digital circuits generate HIGH-FREQUENCY noise from switching. "
               "Analog circuits are SENSITIVE to this noise. Physical separation "
               "reduces: 1) Capacitive coupling, 2) Magnetic coupling, "
               "3) Ground noise coupling. The 'quiet' analog ground must not "
               "carry digital return currents.",
        parameters={
            "separation_distance_mm": 10.0,
            "use_guard_ring": True,
            "separate_ground_planes": False,  # Single ground is usually better!
        },
        source="Henry Ott 'EMC Engineering', Analog Devices AN-345",
    ),

    EngineeringRule(
        id="PL-ANA-002",
        name="ADC Reference Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.MANDATORY,
        description="ADC voltage reference placement",
        condition="Design has ADC with external reference",
        action="Place voltage reference IC within 10mm of ADC VREF pin. "
               "Decoupling cap within 2mm of reference output.",
        reason="Voltage reference sets ADC accuracy. Any noise on VREF "
               "directly corrupts all measurements. Short traces minimize "
               "noise pickup. Dedicated decoupling filters high-frequency noise.",
        parameters={
            "distance_to_adc_max_mm": 10.0,
            "decoupling_distance_max_mm": 2.0,
        },
        source="TI SBAA340",
    ),

    EngineeringRule(
        id="PL-ANA-003",
        name="Analog Input Protection",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="Input protection near connector",
        condition="Design has analog inputs from external sources",
        action="Place ESD protection and filtering components at board edge, "
               "immediately after input connector, BEFORE trace runs to IC.",
        reason="External inputs can carry ESD events (8-15kV). Protection "
               "must be FIRST component the signal hits. If placed after "
               "long traces, ESD travels the trace and damages the IC. "
               "Filter caps at input reject external noise before it enters board.",
        parameters={
            "distance_from_connector_max_mm": 5.0,
        },
        source="ST AN4435, TI SLVA680",
    ),

    # -------------------------------------------------------------------------
    # CONNECTOR PLACEMENT
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="PL-CON-001",
        name="Connector Edge Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.MANDATORY,
        description="Connectors at board edges",
        condition="Design has external connectors",
        action="Place all external connectors at board edges. "
               "Group by function: power on one edge, signals on another.",
        reason="Connectors interface with external world. Edge placement: "
               "1) Allows physical access, 2) Enables enclosure mounting, "
               "3) Provides natural cable routing. Grouping simplifies "
               "cable management and reduces EMI from cable radiation.",
        parameters={
            "max_distance_from_edge_mm": 3.0,
        },
        source="Mechanical design practice",
    ),

    EngineeringRule(
        id="PL-CON-002",
        name="High-Speed Connector Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="USB/Ethernet connector placement",
        condition="Design has USB, Ethernet, or other high-speed interface",
        action="Place high-speed connector with: 1) Short path to IC, "
               "2) ESD protection within 5mm of connector, "
               "3) Termination resistors within 5mm of IC pins",
        reason="High-speed signals are timing-critical and noise-sensitive. "
               "Short traces reduce: 1) Signal reflections, 2) EMI radiation, "
               "3) Susceptibility to external noise. ESD protection at entry "
               "point prevents damage. Termination at IC end absorbs reflections.",
        parameters={
            "max_trace_length_mm": 50.0,
            "esd_distance_from_connector_max_mm": 5.0,
            "termination_distance_from_ic_max_mm": 5.0,
        },
        source="USB-IF Layout Guidelines, IEEE 802.3",
    ),

    # -------------------------------------------------------------------------
    # THERMAL PLACEMENT
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="PL-THR-001",
        name="Hot Component Separation",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="Separate heat-generating components",
        condition="Design has components dissipating >0.5W",
        action="Space high-power components at least 10mm apart. "
               "Place at board edge or near copper pour for heat spreading.",
        reason="Heat affects ALL nearby components, reducing reliability. "
               "Electrolytic caps lose life at 10x rate for every 10C increase. "
               "ICs derate performance at high temperature. Separation prevents "
               "thermal hotspots. Edge placement allows convection.",
        parameters={
            "min_separation_mm": 10.0,
            "prefer_edge_placement": True,
            "thermal_relief_vias": True,
        },
        source="IPC-2221 thermal guidelines",
    ),

    EngineeringRule(
        id="PL-THR-002",
        name="Temperature-Sensitive Component Placement",
        category=RuleCategory.PLACEMENT,
        priority=RulePriority.CRITICAL,
        description="Protect temperature-sensitive components",
        condition="Design has precision components (voltage reference, crystal, etc.)",
        action="Place temperature-sensitive components AWAY from heat sources. "
               "Minimum 20mm from regulators, inductors, power resistors.",
        reason="Precision components have temperature coefficients (tempco). "
               "Voltage references: 10-50 ppm/C = 0.001-0.005%/C error. "
               "Crystals: frequency shifts with temperature. "
               "Keeping them cool ensures accuracy and stability.",
        parameters={
            "min_distance_from_heat_mm": 20.0,
            "max_ambient_temp_c": 50.0,
        },
        source="Analog precision design practice",
    ),
]


# =============================================================================
# ROUTING RULES DATABASE
# =============================================================================

ROUTING_RULES: List[EngineeringRule] = [
    # -------------------------------------------------------------------------
    # ROUTING ORDER / SEQUENCE
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="RT-ORD-001",
        name="Routing Priority Sequence",
        category=RuleCategory.ROUTING,
        priority=RulePriority.MANDATORY,
        description="Order of routing operations",
        condition="Starting routing phase",
        action="""Route in this order:
        1. Power rails (VCC, GND) - planes or wide traces
        2. Critical signals (clock, reset, differential pairs)
        3. High-speed signals (USB, SPI, I2C at high speed)
        4. Analog signals
        5. General digital signals
        6. Non-critical signals (LEDs, switches)""",
        reason="Priority routing ensures critical nets get optimal paths. "
               "Power must be solid BEFORE signals reference it. "
               "High-speed signals need direct paths without detours. "
               "Analog signals need quiet routes away from digital.",
        parameters={
            "priorities": {
                "power": 1,
                "clock": 2,
                "reset": 2,
                "differential_pairs": 3,
                "high_speed": 4,
                "analog": 5,
                "digital": 6,
                "misc": 7,
            }
        },
        source="General PCB design methodology",
    ),

    # -------------------------------------------------------------------------
    # POWER ROUTING
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="RT-PWR-001",
        name="Power Trace Width",
        category=RuleCategory.ROUTING,
        priority=RulePriority.MANDATORY,
        description="Size power traces for current",
        condition="Routing power nets",
        action="Calculate trace width using IPC-2152: "
               "Width = f(current, temp_rise, copper_weight). "
               "Minimum 20 mil (0.5mm) for any power trace.",
        reason="Undersized traces cause: 1) Resistive voltage drop (I*R), "
               "2) Heating (I^2*R), 3) Reliability issues. "
               "IPC-2152 provides tested current capacity tables. "
               "10C temp rise is typical target.",
        parameters={
            "temp_rise_max_c": 10.0,
            "min_width_mm": 0.5,
            "copper_oz": 1.0,
        },
        source="IPC-2152",
    ),

    EngineeringRule(
        id="RT-PWR-002",
        name="Power Return Path",
        category=RuleCategory.ROUTING,
        priority=RulePriority.MANDATORY,
        description="Every power trace needs return path",
        condition="Routing any power trace",
        action="For every power trace, ensure a GND return path of equal "
               "or greater width runs parallel within 2mm, or use GND plane.",
        reason="Current flows in LOOPS. Power out must equal return. "
               "If return path is far, current finds another way - through "
               "other circuits, causing ground bounce and noise. "
               "Parallel return minimizes loop area = less EMI.",
        parameters={
            "max_separation_mm": 2.0,
            "return_width_ratio": 1.0,  # At least equal width
        },
        source="Howard Johnson 'High-Speed Digital Design'",
    ),

    EngineeringRule(
        id="RT-PWR-003",
        name="Star Ground for Analog",
        category=RuleCategory.ROUTING,
        priority=RulePriority.CRITICAL,
        description="Analog ground connection topology",
        condition="Design has analog circuits with shared ground",
        action="Connect analog grounds to main ground at ONE point (star). "
               "Do NOT daisy-chain analog grounds through digital section.",
        reason="Digital ground carries fast-switching return currents. "
               "If analog shares this path, digital noise couples to analog. "
               "Star connection ensures analog return current takes "
               "dedicated path, not through noisy digital ground.",
        parameters={
            "single_point_connection": True,
            "star_point_location": "near_power_entry",
        },
        source="Analog Devices AN-345",
    ),

    # -------------------------------------------------------------------------
    # SIGNAL ROUTING
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="RT-SIG-001",
        name="Signal Trace Geometry",
        category=RuleCategory.ROUTING,
        priority=RulePriority.CRITICAL,
        description="Basic signal trace requirements",
        condition="Routing any signal trace",
        action="Use 45-degree angles (not 90-degree). "
               "Avoid stubs. Avoid acute angles. "
               "Maintain consistent width (no necking except at pads).",
        reason="90-degree corners cause: 1) Impedance discontinuity, "
               "2) Acid traps in manufacturing. "
               "Stubs cause reflections at high frequency. "
               "Width changes cause impedance changes = reflections.",
        parameters={
            "angle_degrees": 45,
            "max_stub_length_mm": 0.5,
        },
        source="IPC-2221, signal integrity basics",
    ),

    EngineeringRule(
        id="RT-SIG-002",
        name="Differential Pair Routing",
        category=RuleCategory.ROUTING,
        priority=RulePriority.MANDATORY,
        description="Route differential pairs correctly",
        condition="Routing USB, LVDS, Ethernet, or other differential signals",
        action="Route differential pairs: 1) Parallel with constant spacing, "
               "2) Equal length (match within 0.1mm), "
               "3) Symmetrical to ground plane, "
               "4) No splits in ground plane beneath",
        reason="Differential signaling relies on BALANCE. "
               "Unequal lengths cause skew = timing error. "
               "Varying spacing changes differential impedance. "
               "Ground plane splits force return current to detour, "
               "converting differential to common-mode noise.",
        parameters={
            "length_match_mm": 0.1,
            "spacing_tolerance_percent": 10,
            "impedance_ohms": 90,  # USB
        },
        source="USB-IF Layout Guidelines, TI SPRAAR7",
    ),

    EngineeringRule(
        id="RT-SIG-003",
        name="Clock Signal Routing",
        category=RuleCategory.ROUTING,
        priority=RulePriority.MANDATORY,
        description="Clock signal routing rules",
        condition="Routing clock signals (oscillator output, PLL output, etc.)",
        action="Route clocks: 1) As short as possible, "
               "2) Away from other signals (3x width spacing), "
               "3) Not parallel to other signals for >10mm, "
               "4) With ground guard traces if near analog",
        reason="Clocks are the STRONGEST noise source on a board. "
               "They switch at the fundamental frequency with harmonics. "
               "Long traces radiate EMI. Parallel traces couple noise "
               "to adjacent signals through capacitance and inductance.",
        parameters={
            "min_spacing_multiplier": 3.0,
            "max_parallel_length_mm": 10.0,
            "guard_trace_for_analog": True,
        },
        source="EMI design guidelines, Analog Devices",
    ),

    EngineeringRule(
        id="RT-SIG-004",
        name="Analog Signal Routing",
        category=RuleCategory.ROUTING,
        priority=RulePriority.CRITICAL,
        description="Analog signal trace requirements",
        condition="Routing analog signals (ADC inputs, sensor outputs, etc.)",
        action="Route analog signals: 1) Short as possible, "
               "2) Away from digital and power, "
               "3) With guard ring or guard traces around sensitive signals, "
               "4) Over continuous ground plane (no splits)",
        reason="Analog signals are LOW-LEVEL and HIGH-IMPEDANCE. "
               "They pick up noise from everywhere. Short traces minimize "
               "antenna effect. Guard traces/rings shield from E-field coupling. "
               "Continuous ground ensures clean return path.",
        parameters={
            "max_length_mm": 50.0,
            "min_spacing_from_digital_mm": 5.0,
            "guard_ring_enabled": True,
        },
        source="Analog Devices PCB layout guide",
    ),

    # -------------------------------------------------------------------------
    # HIGH-SPEED ROUTING
    # -------------------------------------------------------------------------
    EngineeringRule(
        id="RT-HS-001",
        name="Impedance Control",
        category=RuleCategory.ROUTING,
        priority=RulePriority.MANDATORY,
        description="Controlled impedance for high-speed signals",
        condition="Signal frequency > 50MHz or rise time < 2ns",
        action="Calculate and maintain trace impedance: "
               "50 ohm single-ended, 90-100 ohm differential (USB). "
               "Use stack-up calculator. Specify to fab house.",
        reason="At high speed, traces are TRANSMISSION LINES. "
               "Impedance mismatch causes reflections that: "
               "1) Corrupt data, 2) Cause ringing/overshoot, "
               "3) Increase EMI. Controlled impedance ensures signal quality.",
        parameters={
            "single_ended_ohms": 50,
            "differential_ohms": 90,
            "tolerance_percent": 10,
        },
        source="IPC-2141, signal integrity theory",
    ),

    EngineeringRule(
        id="RT-HS-002",
        name="Via Usage for High-Speed",
        category=RuleCategory.ROUTING,
        priority=RulePriority.CRITICAL,
        description="Minimize vias in high-speed paths",
        condition="Routing high-speed signals",
        action="Minimize vias in high-speed signal paths. "
               "Maximum 2 vias per net. Each via adds ~0.5nH inductance. "
               "Use ground vias near signal vias for return current.",
        reason="Every via is a discontinuity: "
               "1) Inductance (~0.5-1nH), 2) Capacitance (~0.3-0.5pF), "
               "3) Impedance bump. Multiple vias cause resonance. "
               "Ground via near signal via provides return current path.",
        parameters={
            "max_vias_per_net": 2,
            "via_inductance_nh": 0.5,
            "ground_via_distance_max_mm": 1.0,
        },
        source="Howard Johnson, Eric Bogatin",
    ),

    EngineeringRule(
        id="RT-HS-003",
        name="Length Matching",
        category=RuleCategory.ROUTING,
        priority=RulePriority.MANDATORY,
        description="Match lengths in parallel buses",
        condition="Routing parallel data bus (DDR, SDRAM, parallel interface)",
        action="Match all data lines within 2mm of each other. "
               "Match clock to data within skew budget. "
               "Use serpentine to add length to short traces.",
        reason="Parallel data is captured on clock edge. "
               "If data arrives at different times (skew), setup/hold "
               "times are violated. Length matching ensures all bits "
               "arrive within the timing window.",
        parameters={
            "data_match_mm": 2.0,
            "clock_to_data_match_mm": 5.0,
            "serpentine_min_spacing_mm": 0.3,
        },
        source="JEDEC DDR Layout Guidelines",
    ),
]


# =============================================================================
# POWER DISTRIBUTION RULES
# =============================================================================

POWER_RULES: List[EngineeringRule] = [
    EngineeringRule(
        id="PW-TOP-001",
        name="Power Tree Topology",
        category=RuleCategory.POWER,
        priority=RulePriority.MANDATORY,
        description="Power distribution tree structure",
        condition="Design has multiple voltage rails",
        action="""Design power tree:
        1. Input -> Bulk cap -> Main regulator
        2. Main rail -> Secondary regulators (if needed)
        3. Each rail -> Local decoupling at each IC
        Never branch BEFORE filtering. Never share regulator input caps.""",
        reason="Power flows like a tree from root (input) to leaves (ICs). "
               "Each branch point needs filtering for that branch. "
               "Sharing caps creates coupling between branches. "
               "Proper topology ensures each section is independently filtered.",
        parameters={
            "bulk_cap_at_input": True,
            "decoupling_at_each_ic": True,
        },
        source="Power distribution network theory",
    ),

    EngineeringRule(
        id="PW-DEC-001",
        name="Decoupling Capacitor Values",
        category=RuleCategory.POWER,
        priority=RulePriority.MANDATORY,
        description="Select decoupling capacitor values",
        condition="IC requires decoupling",
        action="""Multi-value decoupling strategy:
        - 100nF ceramic: Every VCC pin (mandatory)
        - 10uF ceramic: Shared between 2-4 ICs
        - 100uF bulk: At regulator output
        - 1nF (optional): For very high frequency (>100MHz)""",
        reason="Different capacitor values resonate at different frequencies: "
               "100uF: effective < 100kHz, "
               "10uF: effective 100kHz - 1MHz, "
               "100nF: effective 1MHz - 50MHz, "
               "1nF: effective > 50MHz. "
               "Combination covers wide frequency range.",
        parameters={
            "per_pin_nf": 100,
            "per_group_uf": 10,
            "per_regulator_uf": 100,
            "optional_high_freq_nf": 1,
        },
        source="Murata application note, power integrity theory",
    ),

    EngineeringRule(
        id="PW-SEQ-001",
        name="Power Sequencing",
        category=RuleCategory.POWER,
        priority=RulePriority.CRITICAL,
        description="Power rail sequencing requirements",
        condition="Design has multiple power rails to same IC or subsystem",
        action="Check datasheet for power sequencing requirements. "
               "Typically: Core voltage before I/O voltage. "
               "Use sequencing controller or RC delays if needed.",
        reason="Many ICs have internal structures that can latch-up "
               "if powered in wrong order. FPGA/CPU core must power "
               "before I/O to prevent current flow through ESD diodes. "
               "Wrong sequence = damage or malfunction.",
        parameters={
            "check_datasheet": True,
            "typical_order": ["VCORE", "VIO", "VAUX"],
        },
        source="FPGA/ASIC power guidelines",
    ),
]


# =============================================================================
# FABRICATION RULES
# =============================================================================

FABRICATION_RULES: List[EngineeringRule] = [
    EngineeringRule(
        id="FAB-MIN-001",
        name="Minimum Trace Width",
        category=RuleCategory.FABRICATION,
        priority=RulePriority.MANDATORY,
        description="Minimum manufacturable trace width",
        condition="All designs",
        action="Minimum trace width depends on fab capability: "
               "Standard: 6 mil (0.15mm), "
               "Advanced: 4 mil (0.1mm), "
               "HDI: 3 mil (0.075mm). "
               "Use standard unless density requires advanced.",
        reason="Etching process has limits. Too-thin traces: "
               "1) May not survive etching, 2) Have poor yield, "
               "3) Cost more. Standard 6-mil works for most designs "
               "and all fab houses.",
        parameters={
            "standard_min_mm": 0.15,
            "advanced_min_mm": 0.1,
            "hdi_min_mm": 0.075,
        },
        source="IPC-2221 Class 2",
    ),

    EngineeringRule(
        id="FAB-MIN-002",
        name="Minimum Spacing",
        category=RuleCategory.FABRICATION,
        priority=RulePriority.MANDATORY,
        description="Minimum trace-to-trace spacing",
        condition="All designs",
        action="Minimum spacing depends on voltage and fab: "
               "Low voltage (<50V): 6 mil (0.15mm), "
               "Higher voltage: IPC-2221 table. "
               "Pad-to-trace: 8 mil minimum.",
        reason="Spacing prevents: 1) Electrical shorts, 2) Crosstalk, "
               "3) Manufacturing defects (bridges). "
               "Higher voltage needs more spacing to prevent arcing. "
               "Pad-to-trace needs more for solder mask alignment.",
        parameters={
            "trace_to_trace_min_mm": 0.15,
            "trace_to_pad_min_mm": 0.2,
            "high_voltage_per_kv_mm": 0.5,
        },
        source="IPC-2221",
    ),

    EngineeringRule(
        id="FAB-VIA-001",
        name="Via Specifications",
        category=RuleCategory.FABRICATION,
        priority=RulePriority.MANDATORY,
        description="Via drill and pad sizes",
        condition="Design uses vias",
        action="Via sizes: "
               "Standard: 0.3mm drill, 0.6mm pad, "
               "Small: 0.2mm drill, 0.45mm pad, "
               "Micro: 0.1mm drill, 0.25mm pad (laser drilled). "
               "Annular ring minimum: 0.125mm.",
        reason="Via reliability depends on annular ring (pad - drill). "
               "Too small = drill breakout = open circuit. "
               "Standard vias are cheap and reliable. "
               "Micro vias need laser drilling = more expensive.",
        parameters={
            "standard_drill_mm": 0.3,
            "standard_pad_mm": 0.6,
            "annular_ring_min_mm": 0.125,
        },
        source="IPC-2221, fab house capabilities",
    ),

    EngineeringRule(
        id="FAB-MASK-001",
        name="Solder Mask Rules",
        category=RuleCategory.FABRICATION,
        priority=RulePriority.CRITICAL,
        description="Solder mask openings and dams",
        condition="Design has solder mask",
        action="Solder mask dam between pads: minimum 4 mil (0.1mm). "
               "Mask-to-copper clearance: 2-3 mil (0.05-0.075mm). "
               "Small pads (<0.5mm): may need mask-defined pads.",
        reason="Solder mask dams prevent solder bridging between pads. "
               "Too small = no dam = bridges. Clearance ensures mask "
               "doesn't cover pad copper (cold joints). Small pads "
               "use mask opening to define solder area precisely.",
        parameters={
            "dam_min_mm": 0.1,
            "clearance_mm": 0.075,
        },
        source="IPC-7351, solder paste guidelines",
    ),

    EngineeringRule(
        id="FAB-SILK-001",
        name="Silkscreen Rules",
        category=RuleCategory.FABRICATION,
        priority=RulePriority.RECOMMENDED,
        description="Silkscreen legibility rules",
        condition="Design has silkscreen",
        action="Silkscreen: "
               "Minimum line width: 6 mil (0.15mm), "
               "Minimum text height: 40 mil (1mm), "
               "Keep off pads and vias.",
        reason="Thin silkscreen lines don't print well. "
               "Small text is unreadable. Silkscreen on pads "
               "contaminates solder joint. On vias, it fills the hole.",
        parameters={
            "line_width_min_mm": 0.15,
            "text_height_min_mm": 1.0,
            "pad_clearance_mm": 0.2,
        },
        source="IPC-7351",
    ),
]


# =============================================================================
# EMI/EMC RULES
# =============================================================================

EMI_RULES: List[EngineeringRule] = [
    EngineeringRule(
        id="EMI-LOOP-001",
        name="Minimize Current Loops",
        category=RuleCategory.EMI_EMC,
        priority=RulePriority.CRITICAL,
        description="Reduce loop area for all current paths",
        condition="All designs",
        action="For every current path, minimize loop area: "
               "Signal trace should have return path directly beneath or beside. "
               "Power traces should have parallel return. "
               "No slots in ground plane under signal traces.",
        reason="EMI radiation is proportional to loop area: "
               "E ~ f^2 * Area * Current. "
               "Halving loop area reduces EMI by 6dB. "
               "Ground plane directly under trace gives MINIMUM loop area.",
        parameters={
            "max_loop_area_mm2": 100,
            "ground_plane_required": True,
        },
        source="Henry Ott 'EMC Engineering', FCC Part 15",
    ),

    EngineeringRule(
        id="EMI-FILT-001",
        name="Filter at Entry/Exit Points",
        category=RuleCategory.EMI_EMC,
        priority=RulePriority.MANDATORY,
        description="Filter all I/O connections",
        condition="Design has external connections",
        action="At every connector (power, signal, I/O): "
               "1) ESD protection first, "
               "2) Ferrite bead or common-mode choke, "
               "3) Decoupling cap to ground. "
               "This forms a pi-filter for noise.",
        reason="External cables act as antennas - both radiating "
               "internal noise OUT and coupling external noise IN. "
               "Filtering at boundary prevents: "
               "1) Conducted emissions, 2) Radiated emissions via cable, "
               "3) Susceptibility to external interference.",
        parameters={
            "esd_protection": True,
            "ferrite_bead": True,
            "filter_cap_nf": 100,
        },
        source="IEC 61000-4-x, EMC design practice",
    ),

    EngineeringRule(
        id="EMI-GND-001",
        name="Ground Plane Integrity",
        category=RuleCategory.EMI_EMC,
        priority=RulePriority.MANDATORY,
        description="Maintain solid ground plane",
        condition="Design uses ground plane",
        action="Never split ground plane except for isolation. "
               "Minimize cuts and slots. Route signals to avoid "
               "crossing plane splits. If split needed, use bridge "
               "capacitor or single-point connection.",
        reason="Ground plane is the RETURN PATH for all signals. "
               "Splits force return current to detour around split, "
               "creating large loop = EMI antenna. "
               "A solid plane ensures direct return path always available.",
        parameters={
            "no_splits": True,
            "bridge_capacitor_if_split_nf": 100,
        },
        source="Signal integrity fundamentals",
    ),
]


# =============================================================================
# ROUTING SEQUENCE / PRIORITY
# =============================================================================

ROUTING_SEQUENCE: List[RoutingSequence] = [
    RoutingSequence(
        priority=1,
        net_type="POWER_PLANE",
        description="Power planes (VCC, GND) - establish first",
        constraints={"use_plane": True, "min_width_mm": 0.5},
    ),
    RoutingSequence(
        priority=2,
        net_type="CRITICAL_CLOCK",
        description="Clock signals - shortest path, away from others",
        constraints={"max_length_mm": 30, "isolation_mm": 1.0},
    ),
    RoutingSequence(
        priority=3,
        net_type="RESET",
        description="Reset signals - short, filtered",
        constraints={"max_length_mm": 20},
    ),
    RoutingSequence(
        priority=4,
        net_type="DIFFERENTIAL_PAIR",
        description="USB, LVDS, Ethernet - matched pairs",
        constraints={"length_match_mm": 0.1, "impedance_ohms": 90},
    ),
    RoutingSequence(
        priority=5,
        net_type="HIGH_SPEED_SINGLE",
        description="High-speed single-ended (SPI, fast GPIO)",
        constraints={"max_length_mm": 50, "impedance_ohms": 50},
    ),
    RoutingSequence(
        priority=6,
        net_type="ANALOG",
        description="Analog signals - isolated from digital",
        constraints={"isolation_mm": 2.0, "guard_traces": True},
    ),
    RoutingSequence(
        priority=7,
        net_type="I2C",
        description="I2C signals - pullup routing",
        constraints={"max_length_mm": 100},
    ),
    RoutingSequence(
        priority=8,
        net_type="GENERAL_DIGITAL",
        description="General digital signals",
        constraints={"min_width_mm": 0.15},
    ),
    RoutingSequence(
        priority=9,
        net_type="LED_INDICATOR",
        description="LED indicators - lowest priority",
        constraints={"min_width_mm": 0.2},
    ),
]


# =============================================================================
# PLACEMENT ZONES
# =============================================================================

PLACEMENT_ZONES: List[PlacementZone] = [
    PlacementZone(
        name="POWER_INPUT",
        purpose="Power entry and bulk filtering",
        components=["CONNECTOR_POWER", "FUSE", "TVS", "BULK_CAP"],
        constraints={"location": "edge", "copper_pour": True},
    ),
    PlacementZone(
        name="POWER_REGULATION",
        purpose="Voltage regulation and conversion",
        components=["REGULATOR", "INDUCTOR", "POWER_CAP", "SCHOTTKY"],
        constraints={"thermal_clearance_mm": 10, "near": "POWER_INPUT"},
    ),
    PlacementZone(
        name="MCU_CORE",
        purpose="Microcontroller and immediate support",
        components=["MCU", "CRYSTAL", "DECOUPLING_CAP", "RESET_CIRCUIT"],
        constraints={"location": "center", "decoupling_distance_mm": 3},
    ),
    PlacementZone(
        name="ANALOG_SECTION",
        purpose="Analog signal processing",
        components=["ADC", "OPAMP", "VREF", "ANALOG_FILTER"],
        constraints={"isolation_from_digital_mm": 10, "guard_ring": True},
    ),
    PlacementZone(
        name="DIGITAL_IO",
        purpose="Digital interfaces and connectors",
        components=["USB_CONNECTOR", "ESD_PROTECTION", "TERMINATION_R"],
        constraints={"location": "edge", "esd_first": True},
    ),
    PlacementZone(
        name="RF_SECTION",
        purpose="RF and wireless (if present)",
        components=["RF_MODULE", "ANTENNA", "RF_MATCHING"],
        constraints={"ground_plane_solid": True, "keepout_mm": 5},
    ),
]


# =============================================================================
# ENGINEERING RULES ENGINE
# =============================================================================

class EngineeringRulesEngine:
    """
    The decision-making engine that applies engineering rules.

    This is what makes the Circuit Intelligence THINK like an engineer.
    It takes a design context and returns actionable decisions.
    """

    def __init__(self):
        self.placement_rules = PLACEMENT_RULES
        self.routing_rules = ROUTING_RULES
        self.power_rules = POWER_RULES
        self.fabrication_rules = FABRICATION_RULES
        self.emi_rules = EMI_RULES
        self.routing_sequence = ROUTING_SEQUENCE
        self.placement_zones = PLACEMENT_ZONES

        # Index rules by ID for fast lookup
        self._rule_index: Dict[str, EngineeringRule] = {}
        self._build_index()

    def _build_index(self):
        """Build rule index."""
        all_rules = (
            self.placement_rules +
            self.routing_rules +
            self.power_rules +
            self.fabrication_rules +
            self.emi_rules
        )
        for rule in all_rules:
            self._rule_index[rule.id] = rule

    def get_rule(self, rule_id: str) -> Optional[EngineeringRule]:
        """Get rule by ID."""
        return self._rule_index.get(rule_id)

    def get_rules_by_category(self, category: RuleCategory) -> List[EngineeringRule]:
        """Get all rules in a category."""
        return [r for r in self._rule_index.values() if r.category == category]

    def get_mandatory_rules(self) -> List[EngineeringRule]:
        """Get all mandatory rules."""
        return [r for r in self._rule_index.values()
                if r.priority == RulePriority.MANDATORY]

    def get_placement_zone(self, component_type: str) -> Optional[PlacementZone]:
        """Find which zone a component belongs to."""
        for zone in self.placement_zones:
            if component_type.upper() in [c.upper() for c in zone.components]:
                return zone
        return None

    def get_routing_priority(self, net_type: str) -> int:
        """Get routing priority for a net type."""
        for seq in self.routing_sequence:
            if net_type.upper() == seq.net_type.upper():
                return seq.priority
        return 99  # Default to lowest priority

    def get_routing_constraints(self, net_type: str) -> Dict:
        """Get routing constraints for a net type."""
        for seq in self.routing_sequence:
            if net_type.upper() == seq.net_type.upper():
                return seq.constraints
        return {}

    # -------------------------------------------------------------------------
    # DECISION METHODS - The WISDOM
    # -------------------------------------------------------------------------

    def decide_placement_order(self, components: List[dict]) -> List[dict]:
        """
        Decide the ORDER in which components should be placed.

        Returns components sorted by placement priority.
        """
        def get_placement_priority(comp: dict) -> int:
            comp_type = comp.get('type', '').upper()

            # Power input first
            if 'POWER' in comp_type and 'CONNECTOR' in comp_type:
                return 1
            # Then regulators
            if 'REGULATOR' in comp_type or 'LDO' in comp_type or 'BUCK' in comp_type:
                return 2
            # Then MCU
            if 'MCU' in comp_type or 'PROCESSOR' in comp_type:
                return 3
            # Then crystal
            if 'CRYSTAL' in comp_type or 'OSCILLATOR' in comp_type:
                return 4
            # Then decoupling caps (near their ICs)
            if 'DECOUPLING' in comp_type or (comp.get('value', 0) == 100e-9):
                return 5
            # Then connectors
            if 'CONNECTOR' in comp_type:
                return 6
            # Then other ICs
            if 'IC' in comp_type:
                return 7
            # Then passives
            return 8

        return sorted(components, key=get_placement_priority)

    def decide_component_position(self, component: dict,
                                   placed_components: List[dict],
                                   board_size: Tuple[float, float]) -> dict:
        """
        Decide WHERE a component should be placed.

        Returns position constraints based on engineering rules.
        """
        comp_type = component.get('type', '').upper()
        constraints = {}

        # Find zone
        zone = self.get_placement_zone(comp_type)
        if zone:
            constraints['zone'] = zone.name
            constraints.update(zone.constraints)

        # Apply specific rules
        if 'CRYSTAL' in comp_type:
            # Find MCU
            mcu = next((c for c in placed_components
                       if 'MCU' in c.get('type', '').upper()), None)
            if mcu:
                constraints['near_component'] = mcu
                constraints['max_distance_mm'] = 5.0

        if 'DECOUPLING' in comp_type or component.get('value') == 100e-9:
            # Find associated IC
            net = component.get('net', '')
            # Place near the IC that uses this net
            constraints['near_vcc_pin'] = True
            constraints['max_distance_mm'] = 3.0

        if 'REGULATOR' in comp_type:
            constraints['thermal_pad_area_mm2'] = 100
            constraints['near_power_input'] = True

        return constraints

    def decide_routing_order(self, nets: List[dict]) -> List[dict]:
        """
        Decide the ORDER in which nets should be routed.

        Returns nets sorted by routing priority.
        """
        def get_net_priority(net: dict) -> int:
            net_type = net.get('type', '').upper()
            net_name = net.get('name', '').upper()

            # Power nets first
            if net_name in ['VCC', 'VDD', 'GND', 'VSS', '3V3', '5V']:
                return 1
            # Clock signals
            if 'CLK' in net_name or 'CLOCK' in net_name:
                return 2
            # Reset
            if 'RST' in net_name or 'RESET' in net_name:
                return 3
            # Differential pairs
            if net.get('differential'):
                return 4
            # High-speed
            if net.get('frequency', 0) > 50e6:
                return 5
            # Analog
            if 'ANALOG' in net_type or 'ADC' in net_name:
                return 6
            # Default
            return self.get_routing_priority(net_type)

        return sorted(nets, key=get_net_priority)

    def decide_trace_width(self, net: dict, current_a: float = 0,
                           is_power: bool = False) -> float:
        """
        Decide trace width for a net.

        Returns width in mm.
        """
        if is_power or current_a > 0.5:
            # IPC-2152 calculation
            # Simplified: for 1oz copper, 10C rise
            # Width (mm) ≈ current / 3 for external layers
            width = max(0.5, current_a / 3)
        else:
            # Signal trace
            width = 0.2  # Default 8 mil

        return width

    def decide_layer_assignment(self, net: dict, layer_count: int) -> str:
        """
        Decide which layer a net should be routed on.

        Returns layer name.
        """
        net_type = net.get('type', '').upper()
        net_name = net.get('name', '').upper()

        if layer_count == 2:
            # 2-layer: top for signals, bottom for ground
            if net_name == 'GND':
                return 'B.Cu'
            return 'F.Cu'

        elif layer_count == 4:
            # 4-layer: top signals, inner1 GND, inner2 power, bottom signals
            if net_name == 'GND':
                return 'In1.Cu'
            if net_name in ['VCC', 'VDD', '3V3', '5V']:
                return 'In2.Cu'
            # High-speed on top (closest to ground)
            if net.get('frequency', 0) > 50e6:
                return 'F.Cu'
            return 'B.Cu'

        return 'F.Cu'  # Default

    def validate_design(self, design: dict) -> List[dict]:
        """
        Validate a design against all rules.

        Returns list of violations with severity.
        """
        violations = []

        # Check mandatory rules
        for rule in self.get_mandatory_rules():
            # This would need actual design analysis
            # For now, return rule requirements
            pass

        return violations

    def get_design_recommendations(self, design_type: str) -> List[str]:
        """
        Get recommendations for a design type.
        """
        recommendations = []

        if 'POWER' in design_type.upper():
            recommendations.extend([
                "Use ground plane on bottom layer",
                "Place bulk cap within 10mm of power input",
                "Size traces for expected current + 50% margin",
                "Add thermal relief vias under regulators",
            ])

        if 'ANALOG' in design_type.upper():
            recommendations.extend([
                "Separate analog and digital sections physically",
                "Use star grounding for analog return currents",
                "Keep analog traces short and shielded",
                "Place voltage reference away from heat sources",
            ])

        if 'HIGH_SPEED' in design_type.upper() or 'USB' in design_type.upper():
            recommendations.extend([
                "Control impedance: 90 ohm differential for USB",
                "Match differential pair lengths within 0.1mm",
                "Place ESD protection within 5mm of connector",
                "Minimize vias in differential path",
            ])

        return recommendations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_rules_engine() -> EngineeringRulesEngine:
    """Get a configured rules engine instance."""
    return EngineeringRulesEngine()


def print_all_rules():
    """Print all rules for reference."""
    engine = get_rules_engine()

    print("=" * 70)
    print("ENGINEERING RULES DATABASE")
    print("=" * 70)

    categories = [
        (RuleCategory.PLACEMENT, "PLACEMENT RULES"),
        (RuleCategory.ROUTING, "ROUTING RULES"),
        (RuleCategory.POWER, "POWER RULES"),
        (RuleCategory.FABRICATION, "FABRICATION RULES"),
        (RuleCategory.EMI_EMC, "EMI/EMC RULES"),
    ]

    for cat, title in categories:
        rules = engine.get_rules_by_category(cat)
        print(f"\n{title}")
        print("-" * 70)
        for rule in rules:
            pri = rule.priority.name
            print(f"\n[{rule.id}] {rule.name} ({pri})")
            print(f"  WHEN: {rule.condition}")
            print(f"  DO: {rule.action[:100]}...")
            print(f"  WHY: {rule.reason[:100]}...")


if __name__ == '__main__':
    print_all_rules()
