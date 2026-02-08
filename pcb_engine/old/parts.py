"""
PCB Engine - Parts Module
=========================

Handles part definitions, pin specifications, and the parts database.
This is the input to Phase 0 of the algorithm.

Key concepts:
- Part: A component with pins, connections, and constraints
- Pin: A connection point with electrical and physical properties
- Pin Usage: Whether a pin is USED, UNUSED, NC, or DNP in this design
- Connection: A required trace from one pin to another
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class MountingType(Enum):
    SMD = "smd"
    THT = "tht"
    MODULE = "module"
    MECHANICAL = "mechanical"


class PinType(Enum):
    POWER = "power"
    GROUND = "ground"
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    NC = "nc"
    PASSIVE = "passive"  # For resistors, capacitors


class PinUsage(Enum):
    """Whether this pin is used in the current design"""
    USED = "used"                   # Connected, needs routing
    UNUSED = "unused"               # Not connected, but exists physically
    DNP = "dnp"                     # Part is DNP, pads exist but no routing
    NC_BY_DESIGN = "nc"             # Manufacturer says don't connect


class SignalType(Enum):
    DC = "dc"
    I2C = "i2c"
    SPI = "spi"
    UART = "uart"
    ANALOG = "analog"
    DIGITAL = "digital"


class NetType(Enum):
    POWER = "power"
    GROUND = "ground"
    SIGNAL = "signal"


class Priority(Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    OPTIONAL = "optional"


class LayerPreference(Enum):
    F_CU = "F.Cu"
    B_CU = "B.Cu"
    ANY = "any"


# =============================================================================
# PIN STRUCTURE
# =============================================================================

@dataclass
class PinPhysical:
    """Physical properties of a pin"""
    type: MountingType = MountingType.SMD
    shape: str = "rect"             # 'rect', 'round', 'oblong'
    size_x: float = 0.6             # mm
    size_y: float = 0.6             # mm
    drill: Optional[float] = None   # mm (THT only)
    offset_x: float = 0.0           # mm from component center
    offset_y: float = 0.0           # mm from component center
    side: str = "bottom"            # 'top', 'bottom', 'left', 'right'


@dataclass
class PinElectrical:
    """Electrical properties of a pin"""
    voltage: Optional[float] = None
    current_max: Optional[float] = None
    signal_type: SignalType = SignalType.DC


@dataclass
class Pin:
    """Complete pin definition"""
    number: str                     # "1", "2", etc.
    name: str                       # "VCC", "GND", "SDA"
    function: str = ""              # Human readable description
    net: Optional[str] = None       # Net name (None for NC/unused)
    type: PinType = PinType.PASSIVE
    electrical: PinElectrical = field(default_factory=PinElectrical)
    physical: PinPhysical = field(default_factory=PinPhysical)

    # === USAGE IN THIS DESIGN ===
    usage: PinUsage = PinUsage.USED
    usage_notes: str = ""           # Why unused? What could it be used for?

    def is_used(self) -> bool:
        """Returns True if this pin needs routing"""
        return self.usage == PinUsage.USED and self.net is not None

    def is_physical(self) -> bool:
        """Returns True if this pin takes physical space"""
        return self.usage != PinUsage.DNP or True  # Pads exist even for DNP

    def needs_routing(self) -> bool:
        """Returns True if this pin needs a trace"""
        return self.is_used()

    def needs_clearance(self) -> bool:
        """Returns True if other traces must avoid this pin"""
        return self.is_physical()


# =============================================================================
# CONNECTION STRUCTURE
# =============================================================================

@dataclass
class TraceRequirements:
    """Trace routing requirements"""
    width_min: float = 0.25         # mm
    width_preferred: float = 0.3    # mm
    length_max: Optional[float] = None
    length_match: Optional[str] = None  # Net to match length with
    layer_preference: LayerPreference = LayerPreference.ANY
    via_allowed: bool = True
    via_cost: int = 10              # A* via penalty
    isolation: float = 0.0          # Extra clearance from other nets


@dataclass
class Connection:
    """Required connection between pins"""
    from_pin: str
    to_component: str
    to_pin: Optional[str]
    net: str
    priority: Priority = Priority.REQUIRED
    trace: TraceRequirements = field(default_factory=TraceRequirements)


# =============================================================================
# PART STRUCTURE
# =============================================================================

@dataclass
class Physical:
    """Physical properties"""
    mounting: MountingType = MountingType.SMD
    package: str = ""
    body_length: float = 0.0
    body_width: float = 0.0
    body_height: float = 0.0
    courtyard_length: float = 0.0
    courtyard_width: float = 0.0


@dataclass
class Clearance:
    """Clearance requirements"""
    to_same_net: float = 0.2
    to_other_net: float = 0.2
    to_power: float = 0.3
    to_board_edge: float = 2.0
    to_mounting_hole: float = 3.0


@dataclass
class Placement:
    """Placement constraints"""
    fixed_position: Optional[Tuple[float, float]] = None
    fixed_orientation: Optional[int] = None
    preferred_zone: Optional[str] = None
    must_be_on_edge: bool = False
    keep_near: List[str] = field(default_factory=list)
    keep_away: List[str] = field(default_factory=list)
    thermal_zone: bool = False


@dataclass
class Orientation:
    """Orientation properties"""
    default: int = 0
    allowed: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    preferred: Optional[int] = None
    pin1_marker: str = "none"


@dataclass
class Assembly:
    """Assembly properties"""
    method: str = "reflow"
    dnp: bool = False
    polarity: bool = False
    orientation_critical: bool = False


@dataclass
class Part:
    """Complete part definition"""
    # Identification
    ref_designator: str
    part_id: str = ""
    description: str = ""
    manufacturer: str = ""
    mpn: str = ""

    # Function
    function: str = ""
    circuit_role: str = "passive"   # 'hub', 'power_entry', 'endpoint', etc.

    # Properties
    physical: Physical = field(default_factory=Physical)
    orientation: Orientation = field(default_factory=Orientation)
    clearance: Clearance = field(default_factory=Clearance)
    placement: Placement = field(default_factory=Placement)
    assembly: Assembly = field(default_factory=Assembly)

    # Footprint
    footprint: str = ""              # Generic footprint name (e.g., "0805", "SOT-23-5")
    kicad_footprint: str = ""        # Full KiCad library path (e.g., "Package_TO_SOT_SMD:SOT-23-5")
                                     # If set, load from KiCad library (has 3D model)
                                     # If empty, create custom footprint from pin data
    value: str = ""                  # Component value (e.g., "10uF", "4.7k", "3.3V LDO")

    # Pins and connections
    pins: List[Pin] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)

    # Validation
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def get_used_pins(self) -> List[Pin]:
        """Get pins that need routing"""
        return [p for p in self.pins if p.is_used()]

    def get_physical_pins(self) -> List[Pin]:
        """Get pins that take physical space"""
        return [p for p in self.pins if p.is_physical()]

    def get_pin(self, number: str) -> Optional[Pin]:
        """Get pin by number"""
        for p in self.pins:
            if p.number == number:
                return p
        return None


# =============================================================================
# PARTS COLLECTOR
# =============================================================================

class PartsCollector:
    """
    Collects and validates part information.
    This is Phase 0 of the PCB algorithm.
    """

    def __init__(self):
        self.parts: Dict[str, Part] = {}
        self.nets: Dict[str, Dict] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_part_manual(self, data: dict) -> bool:
        """Add a part from user-provided data"""
        required = ['ref_designator', 'pins']

        missing = [f for f in required if f not in data]
        if missing:
            self.errors.append(f"Missing required fields: {missing}")
            return False

        try:
            part = self._build_part(data)
            self.parts[part.ref_designator] = part
            self._extract_nets(part)
            return True
        except Exception as e:
            self.errors.append(f"Error building part: {e}")
            return False

    def validate_all(self) -> bool:
        """Validate all parts"""
        all_valid = True

        for ref, part in self.parts.items():
            errors = self._validate_part(part)
            if errors:
                part.validation_errors = errors
                part.validated = False
                all_valid = False
                self.errors.extend([f"{ref}: {e}" for e in errors])
            else:
                part.validated = True

        # Cross-validate connections
        cross_errors = self._validate_connections()
        if cross_errors:
            self.errors.extend(cross_errors)
            all_valid = False

        return all_valid

    def export(self) -> Dict:
        """Export database for next phase"""
        return {
            'parts': {ref: self._part_to_dict(p) for ref, p in self.parts.items()},
            'nets': self.nets,
            'summary': {
                'total_parts': len(self.parts),
                'total_nets': len(self.nets),
                'validated': all(p.validated for p in self.parts.values()),
            }
        }

    def get_hub(self) -> Optional[str]:
        """Return the hub component"""
        for ref, part in self.parts.items():
            if part.circuit_role == 'hub':
                return ref
        return None

    def get_connectivity_matrix(self) -> Dict[str, Dict[str, int]]:
        """Build connectivity matrix"""
        matrix = {ref: {} for ref in self.parts}

        for net_name, net_info in self.nets.items():
            if net_info['type'] == NetType.GROUND:
                continue

            pins = net_info.get('pins', [])
            for i, (comp_a, pin_a) in enumerate(pins):
                for comp_b, pin_b in pins[i+1:]:
                    matrix.setdefault(comp_a, {})
                    matrix[comp_a][comp_b] = matrix[comp_a].get(comp_b, 0) + 1
                    matrix.setdefault(comp_b, {})
                    matrix[comp_b][comp_a] = matrix[comp_b].get(comp_a, 0) + 1

        return matrix

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _build_part(self, data: dict) -> Part:
        """Build Part from dictionary"""
        pins = []
        for p in data.get('pins', []):
            # Determine usage
            usage_str = p.get('usage', 'used')
            usage_map = {
                'used': PinUsage.USED,
                'unused': PinUsage.UNUSED,
                'nc': PinUsage.NC_BY_DESIGN,
                'dnp': PinUsage.DNP,
            }
            usage = usage_map.get(usage_str, PinUsage.USED)

            # Build pin
            phys = p.get('physical', {})
            elec = p.get('electrical', {})

            # Map pin type with fallbacks for common aliases
            pin_type_str = p.get('type', 'passive')
            pin_type_map = {
                'power_in': 'power',
                'power_out': 'power',
                'pwr': 'power',
                'gnd': 'ground',
                'io': 'bidirectional',
                'bidir': 'bidirectional',
                'in': 'input',
                'out': 'output',
                'no_connect': 'nc',
                'not_connected': 'nc',
            }
            pin_type_str = pin_type_map.get(pin_type_str, pin_type_str)

            pin = Pin(
                number=p['number'],
                name=p.get('name', ''),
                function=p.get('function', ''),
                net=p.get('net'),
                type=PinType(pin_type_str),
                electrical=PinElectrical(
                    voltage=elec.get('voltage'),
                    current_max=elec.get('current_max'),
                    signal_type=SignalType(elec.get('signal_type', 'dc'))
                ),
                physical=PinPhysical(
                    type=MountingType(phys.get('type', 'smd')),
                    shape=phys.get('shape', 'rect'),
                    # Support both pad_size tuple and separate size_x/size_y
                    size_x=phys.get('pad_size', (phys.get('size_x', 0.6), 0.6))[0] if 'pad_size' in phys else phys.get('size_x', 0.6),
                    size_y=phys.get('pad_size', (0.6, phys.get('size_y', 0.6)))[1] if 'pad_size' in phys else phys.get('size_y', 0.6),
                    drill=phys.get('drill'),
                    offset_x=phys.get('offset_x', 0),
                    offset_y=phys.get('offset_y', 0),
                    side=phys.get('side', 'bottom'),
                ),
                usage=usage,
                usage_notes=p.get('usage_notes', ''),
            )
            pins.append(pin)

        # Build connections
        connections = []
        for c in data.get('connections', []):
            trace_data = c.get('trace', {})
            conn = Connection(
                from_pin=c['from_pin'],
                to_component=c['to_component'],
                to_pin=c.get('to_pin'),
                net=c['net'],
                priority=Priority(c.get('priority', 'required')),
                trace=TraceRequirements(
                    width_min=trace_data.get('width_min', 0.25),
                    width_preferred=trace_data.get('width_preferred', 0.3),
                    length_max=trace_data.get('length_max'),
                    length_match=trace_data.get('length_match'),
                    layer_preference=LayerPreference(trace_data.get('layer_preference', 'any')),
                    via_allowed=trace_data.get('via_allowed', True),
                    via_cost=trace_data.get('via_cost', 10),
                    isolation=trace_data.get('isolation', 0),
                ),
            )
            connections.append(conn)

        # Build physical
        phys_data = data.get('physical', {})

        # Support 'size' tuple as shorthand for body dimensions
        size = data.get('size')
        if size and isinstance(size, (list, tuple)) and len(size) >= 2:
            default_length = size[0]
            default_width = size[1]
        else:
            default_length = 0
            default_width = 0

        body_length = phys_data.get('body_length', default_length)
        body_width = phys_data.get('body_width', default_width)

        physical = Physical(
            mounting=MountingType(phys_data.get('mounting', 'smd')),
            package=phys_data.get('package', data.get('footprint', '')),
            body_length=body_length,
            body_width=body_width,
            body_height=phys_data.get('body_height', 1.0),
            courtyard_length=phys_data.get('courtyard_length', body_length * 1.2 if body_length else 0),
            courtyard_width=phys_data.get('courtyard_width', body_width * 1.2 if body_width else 0),
        )

        # Build part
        return Part(
            ref_designator=data['ref_designator'],
            part_id=data.get('part_id', ''),
            description=data.get('description', ''),
            manufacturer=data.get('manufacturer', ''),
            mpn=data.get('mpn', ''),
            function=data.get('function', ''),
            circuit_role=data.get('circuit_role', 'passive'),
            physical=physical,
            footprint=data.get('footprint', ''),
            kicad_footprint=data.get('kicad_footprint', ''),  # Full KiCad library path for 3D models
            value=data.get('value', ''),  # Component value for display
            pins=pins,
            connections=connections,
            assembly=Assembly(dnp=data.get('assembly', {}).get('dnp', False)),
        )

    def _validate_part(self, part: Part) -> List[str]:
        """Validate a single part"""
        errors = []

        # Check pins exist
        if not part.pins and part.circuit_role != 'mechanical':
            errors.append("No pins defined")

        # Check physical dimensions
        if part.physical.body_length <= 0:
            errors.append("Invalid body_length")

        # Check USED pins have nets
        for pin in part.pins:
            if pin.usage == PinUsage.USED and pin.net is None:
                errors.append(f"Pin {pin.number} marked USED but has no net")

        # Check connections reference valid pins
        pin_numbers = {p.number for p in part.pins}
        for conn in part.connections:
            if conn.from_pin not in pin_numbers:
                errors.append(f"Connection from invalid pin {conn.from_pin}")

        return errors

    def _validate_connections(self) -> List[str]:
        """Cross-validate connections"""
        errors = []

        for ref, part in self.parts.items():
            for conn in part.connections:
                if conn.to_component not in self.parts:
                    if conn.to_component not in ['GND_ZONE', '+3V3_TREE', '+5V_TREE']:
                        errors.append(f"{ref}: Unknown target '{conn.to_component}'")

        return errors

    def _extract_nets(self, part: Part):
        """Extract nets from pins"""
        for pin in part.pins:
            if pin.is_used() and pin.net:
                if pin.net not in self.nets:
                    self.nets[pin.net] = {
                        'type': self._classify_net(pin.net),
                        'pins': [],
                    }
                self.nets[pin.net]['pins'].append((part.ref_designator, pin.number))

    def _classify_net(self, net_name: str) -> NetType:
        """Classify net type"""
        if net_name == 'GND':
            return NetType.GROUND
        elif net_name in ['+5V', '+3V3', 'VCC', 'VIN']:
            return NetType.POWER
        return NetType.SIGNAL

    def _part_to_dict(self, part: Part) -> dict:
        """Convert Part to dictionary"""
        used = [p for p in part.pins if p.usage == PinUsage.USED]
        unused = [p for p in part.pins if p.usage == PinUsage.UNUSED]
        physical = [p for p in part.pins if p.is_physical()]

        return {
            'ref_designator': part.ref_designator,
            'circuit_role': part.circuit_role,
            'footprint': part.footprint,
            'kicad_footprint': part.kicad_footprint,  # Full KiCad library path (has 3D model)
            'value': part.value or part.description or part.mpn or part.part_id,  # For component value display
            'size': (part.physical.body_length, part.physical.body_width),
            'physical': {
                'mounting': part.physical.mounting.value,
                'package': part.physical.package,
                'body': (part.physical.body_length, part.physical.body_width),
                'courtyard': (part.physical.courtyard_length, part.physical.courtyard_width),
            },
            'pin_count': {
                'total': len(part.pins),
                'used': len(used),
                'unused': len(unused),
                'physical': len(physical),
            },
            'used_pins': [
                {'number': p.number, 'name': p.name, 'net': p.net,
                 'offset': (p.physical.offset_x, p.physical.offset_y),
                 'pad_size': (p.physical.size_x, p.physical.size_y)}
                for p in used
            ],
            'physical_pins': [
                {'number': p.number, 'offset': (p.physical.offset_x, p.physical.offset_y),
                 'size': (p.physical.size_x, p.physical.size_y)}
                for p in physical
            ],
            'connections': len(part.connections),
            'validated': part.validated,
            'dnp': part.assembly.dnp,
        }
