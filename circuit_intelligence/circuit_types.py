"""
Circuit Intelligence Type Definitions
=====================================

Core data structures for circuit analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto


# =============================================================================
# ENUMS
# =============================================================================

class CircuitFunction(Enum):
    """What a circuit block DOES."""
    POWER_INPUT = auto()          # Connector bringing power in
    POWER_REGULATION = auto()     # LDO, buck, boost, etc.
    POWER_DISTRIBUTION = auto()   # Power rails, ferrite beads
    POWER_PROTECTION = auto()     # Fuse, TVS, polarity protection

    SIGNAL_CONDITIONING = auto()  # Filters, level shifters
    SIGNAL_AMPLIFICATION = auto() # Op-amps, instrumentation amps
    SIGNAL_CONVERSION = auto()    # ADC, DAC

    DIGITAL_CORE = auto()         # MCU, FPGA, processor
    DIGITAL_MEMORY = auto()       # Flash, RAM, EEPROM
    DIGITAL_INTERFACE = auto()    # USB, SPI, I2C buffers

    ANALOG_SENSING = auto()       # Sensors, measurement
    ANALOG_REFERENCE = auto()     # Voltage references

    COMMUNICATION = auto()        # RF, Ethernet, CAN
    USER_INTERFACE = auto()       # LEDs, buttons, displays


class NetFunction(Enum):
    """What a net DOES in the circuit."""
    POWER_RAIL = auto()           # VCC, +5V, +3V3
    GROUND = auto()               # GND, VSS
    SWITCH_NODE = auto()          # Hot switching node in converters
    GATE_DRIVE = auto()           # MOSFET/transistor gate signals

    CLOCK = auto()                # System clocks
    HIGH_SPEED_DATA = auto()      # USB, Ethernet, DDR
    LOW_SPEED_DATA = auto()       # I2C, UART, GPIO

    ANALOG_SIGNAL = auto()        # Sensor outputs, audio
    ANALOG_REFERENCE = auto()     # VREF, precision voltages

    DIFFERENTIAL_P = auto()       # Positive of diff pair
    DIFFERENTIAL_N = auto()       # Negative of diff pair


class ComponentFunction(Enum):
    """What a component DOES."""
    # Power
    CONNECTOR_POWER = auto()
    FUSE = auto()
    POLARITY_PROTECTION = auto()
    VOLTAGE_REGULATOR = auto()
    INDUCTOR_POWER = auto()
    CAPACITOR_BULK = auto()
    CAPACITOR_BYPASS = auto()

    # Active
    MICROCONTROLLER = auto()
    MEMORY = auto()
    AMPLIFIER = auto()
    COMPARATOR = auto()
    ADC = auto()
    DAC = auto()
    SWITCH_IC = auto()
    DRIVER = auto()
    MUX = auto()

    # Passive
    RESISTOR_PULLUP = auto()
    RESISTOR_PULLDOWN = auto()
    RESISTOR_CURRENT_SENSE = auto()
    RESISTOR_FEEDBACK = auto()
    RESISTOR_SERIES = auto()
    CAPACITOR_FILTER = auto()
    CAPACITOR_COUPLING = auto()
    INDUCTOR_FILTER = auto()

    # Interface
    CONNECTOR_SIGNAL = auto()
    ESD_PROTECTION = auto()
    LED_INDICATOR = auto()
    CRYSTAL = auto()

    # Unknown
    UNKNOWN = auto()


class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = auto()   # Will not work / safety issue
    ERROR = auto()      # Likely to fail
    WARNING = auto()    # May cause problems
    INFO = auto()       # Suggestion for improvement


class ThermalRating(Enum):
    """Thermal status of a component."""
    COLD = auto()       # < 40°C, no concern
    WARM = auto()       # 40-70°C, monitor
    HOT = auto()        # 70-100°C, needs attention
    CRITICAL = auto()   # > 100°C, will fail


# =============================================================================
# DATA CLASSES - Component Level
# =============================================================================

@dataclass
class ComponentAnalysis:
    """Analysis of a single component."""
    ref: str
    value: str
    footprint: str
    function: ComponentFunction
    thermal_dissipation: float = 0.0  # Watts
    temperature_estimate: float = 25.0  # °C
    thermal_rating: ThermalRating = ThermalRating.COLD

    # Placement constraints
    must_be_near: List[str] = field(default_factory=list)  # Component refs
    max_distance_from: Dict[str, float] = field(default_factory=dict)  # ref -> mm
    keep_away_from: List[str] = field(default_factory=list)
    placement_zone: Optional[str] = None

    # Special requirements
    needs_copper_pour: bool = False
    needs_thermal_vias: int = 0
    orientation_preference: Optional[float] = None  # degrees


@dataclass
class NetAnalysis:
    """Analysis of a single net."""
    name: str
    function: NetFunction
    pins: List[Tuple[str, str]]  # (ref, pin)

    # Electrical properties
    voltage: Optional[float] = None
    current: Optional[float] = None
    frequency: Optional[float] = None
    is_differential: bool = False
    diff_pair_partner: Optional[str] = None

    # Routing constraints
    min_width: float = 0.25
    recommended_width: float = 0.25
    max_length: Optional[float] = None
    length_match_group: Optional[str] = None
    length_tolerance: float = 0.1  # 10%

    # Layer constraints
    allowed_layers: List[str] = field(default_factory=lambda: ['F.Cu', 'B.Cu'])
    preferred_layer: str = 'F.Cu'
    max_vias: Optional[int] = None

    # Special requirements
    needs_guard_ring: bool = False
    needs_shielding: bool = False
    keep_away_from_nets: List[str] = field(default_factory=list)
    minimize_loop_area: bool = False


# =============================================================================
# DATA CLASSES - Circuit Level
# =============================================================================

@dataclass
class CurrentLoop:
    """A current loop in the circuit."""
    name: str
    components: List[str]  # Component refs in loop order
    nets: List[str]  # Nets forming the loop
    area: float = 0.0  # mm² (calculated after placement)
    is_critical: bool = False  # High-frequency switching loop
    max_allowed_area: float = 100.0  # mm²


@dataclass
class CircuitBlock:
    """A functional block in the circuit."""
    name: str
    function: CircuitFunction
    components: List[str]  # Component refs
    nets: List[str]  # Internal nets
    input_nets: List[str]  # Nets coming in
    output_nets: List[str]  # Nets going out

    # Placement
    zone: Optional[str] = None
    relative_position: Optional[str] = None  # 'LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'CENTER'

    # Special requirements
    isolate_from: List[str] = field(default_factory=list)  # Other block names
    shield_required: bool = False


@dataclass
class DesignIssue:
    """An issue found during design review."""
    severity: Severity
    category: str  # 'THERMAL', 'EMI', 'POWER', 'SIGNAL', 'MANUFACTURING'
    component: Optional[str]  # Component ref if applicable
    net: Optional[str]  # Net name if applicable
    message: str
    recommendation: str
    auto_fixable: bool = False


# =============================================================================
# DATA CLASSES - Analysis Results
# =============================================================================

@dataclass
class CircuitAnalysis:
    """Complete analysis of a circuit."""
    # Component-level
    components: Dict[str, ComponentAnalysis] = field(default_factory=dict)
    nets: Dict[str, NetAnalysis] = field(default_factory=dict)

    # Circuit-level
    blocks: List[CircuitBlock] = field(default_factory=list)
    current_loops: List[CurrentLoop] = field(default_factory=list)

    # Issues
    issues: List[DesignIssue] = field(default_factory=list)

    # Summary
    total_power_dissipation: float = 0.0
    max_junction_temp: float = 25.0
    critical_nets: List[str] = field(default_factory=list)
    critical_loops: List[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Design quality score 0-100."""
        score = 100
        for issue in self.issues:
            if issue.severity == Severity.CRITICAL:
                score -= 25
            elif issue.severity == Severity.ERROR:
                score -= 10
            elif issue.severity == Severity.WARNING:
                score -= 3
        return max(0, score)


@dataclass
class PlacementConstraint:
    """A constraint for the placement engine."""
    type: str  # 'PROXIMITY', 'DISTANCE', 'ZONE', 'ORIENTATION', 'AVOID'
    component: str
    target: Optional[str] = None  # Other component or zone
    value: Optional[float] = None  # Distance in mm
    priority: int = 1  # 1=must, 2=should, 3=nice-to-have


@dataclass
class RoutingConstraint:
    """A constraint for the routing engine."""
    type: str  # 'WIDTH', 'LENGTH', 'LAYER', 'VIA_LIMIT', 'CLEARANCE', 'SHIELD'
    net: str
    value: any = None
    priority: int = 1


@dataclass
class DesignConstraints:
    """All constraints generated for PCB Engine."""
    placement: List[PlacementConstraint] = field(default_factory=list)
    routing: List[RoutingConstraint] = field(default_factory=list)
    zones: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)  # name -> (x1,y1,x2,y2)
    keep_outs: List[Tuple[float, float, float, float]] = field(default_factory=list)
