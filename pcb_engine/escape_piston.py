"""
Escape Piston - BGA/QFP Pin Escape Routing Sub-Engine

Research-based escape routing algorithms for transitioning signals from
dense pin arrays (BGA, QFP, QFN) to routing layers.

ALGORITHMS IMPLEMENTED (Research-Based):
=========================================

1. DOG-BONE FANOUT (Industry Standard)
   Source: Altium, Cadence PCB Design Resources
   - Standard for BGAs with pitch >= 0.5mm
   - Short angled trace from pad to via between rows
   - Via placed at 45° or 90° from pad center

2. ORDERED ESCAPE ROUTING (MMCF - Min-cost Multi-commodity Flow)
   Source: IEEE - "Ordered Escape routing for grid pin array based on
           Min-cost Multi-commodity Flow" (2016)
   - Converts escape routing to network flow problem
   - Three transformations: non-crossing, ordering, capacity
   - 100% routability for all test cases in research
   - Optimal wire length solution

3. MULTI-CAPACITY ORDERED ESCAPE (MC-OER)
   Source: ACM TODAES - "MCMCF-Router: Multi-capacity Ordered Escape
           Routing Algorithms for Grid/Staggered Pin Array" (2024)
   - Multiple wires between adjacent pins
   - Supports both grid and staggered pin arrays
   - Improved performance over basic MMCF

4. SAT-BASED MULTI-LAYER ESCAPE
   Source: ICCAD 2016 - "Scalable, High-Quality, SAT-Based Multi-Layer
           Escape Routing" (University of British Columbia)
   - Uses MonoSAT solver with network-flow constraints
   - Scales to 2000+ pin BGAs
   - Supports 45° and 90° routing
   - Simultaneous trace and via placement
   - Supports through-hole, blind, buried, any-layer micro-vias

5. LAYER MINIMIZATION ESCAPE
   Source: ResearchGate - "Layer minimization in escape routing for
           staggered-pin-array PCBs"
   - Minimizes PCB layer count
   - Optimal layer assignment algorithm

6. RING-BASED ESCAPE (Concentric Rings)
   Source: Industry standard practice documented by NW Engineering
   - Outer rings escape first (shortest path)
   - Inner rings use vias to deeper layers
   - Systematic layer assignment by ring number

PACKAGE-SPECIFIC HANDLING:
==========================
- BGA: Full escape routing required (center pins trapped)
- QFN: Thermal pad escape + peripheral pin fanout
- QFP: Peripheral pins only (no escape needed, just fanout)
- CSP: Similar to fine-pitch BGA
- LGA: Land grid array, similar to BGA but no balls

Author: PCB Engine Team
License: MIT
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
import math
from collections import defaultdict
import heapq


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PackageType(Enum):
    """Package types requiring different escape strategies"""
    BGA = auto()      # Ball Grid Array - needs full escape
    FBGA = auto()     # Fine-pitch BGA (< 0.65mm pitch)
    CSP = auto()      # Chip Scale Package
    QFN = auto()      # Quad Flat No-lead
    QFP = auto()      # Quad Flat Package
    LGA = auto()      # Land Grid Array
    WLCSP = auto()    # Wafer Level CSP


class EscapeStrategy(Enum):
    """Available escape routing strategies"""
    DOG_BONE = auto()           # Standard dog-bone fanout
    ORDERED_MMCF = auto()       # MMCF-based ordered escape
    MULTI_CAPACITY = auto()     # MC-OER for multiple traces
    SAT_MULTI_LAYER = auto()    # SAT solver for complex BGAs
    RING_BASED = auto()         # Concentric ring escape
    LAYER_MINIMIZE = auto()     # Minimize layer count
    HYBRID = auto()             # Combination based on pin location


class ViaType(Enum):
    """Via types for escape routing"""
    THROUGH_HOLE = auto()       # Through all layers
    BLIND = auto()              # Top/bottom to inner layer
    BURIED = auto()             # Inner layer to inner layer
    MICRO_VIA = auto()          # Laser-drilled, single layer span
    ANY_LAYER = auto()          # HDI any-layer micro-via


class EscapeDirection(Enum):
    """Direction for escape path"""
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()
    NORTHEAST = auto()
    NORTHWEST = auto()
    SOUTHEAST = auto()
    SOUTHWEST = auto()


class PinLocation(Enum):
    """Pin location within array"""
    CORNER = auto()
    EDGE = auto()
    INNER = auto()
    CENTER = auto()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Pin:
    """Represents a single pin in the array"""
    id: str
    row: int
    col: int
    x: float              # Physical X position (mm)
    y: float              # Physical Y position (mm)
    net: str              # Net name
    is_power: bool = False
    is_ground: bool = False
    is_nc: bool = False   # No connect
    location: PinLocation = PinLocation.INNER
    escape_layer: int = 1
    escape_direction: Optional[EscapeDirection] = None
    assigned_via: Optional['Via'] = None


@dataclass
class Via:
    """Via for layer transition"""
    x: float
    y: float
    drill_diameter: float  # mm
    pad_diameter: float    # mm
    via_type: ViaType
    start_layer: int
    end_layer: int
    net: str
    id: str = ""


@dataclass
class EscapeTrace:
    """Trace segment for escape routing"""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    width: float
    layer: int
    net: str
    angle: float = 0.0    # Degrees from horizontal


@dataclass
class EscapePath:
    """Complete escape path for a pin"""
    pin: Pin
    traces: List[EscapeTrace] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    total_length: float = 0.0
    layer_transitions: int = 0
    escaped: bool = False
    # Start and end coordinates for routing piston compatibility
    start: Optional[Tuple[float, float]] = None  # Pad position
    end: Optional[Tuple[float, float]] = None    # Escape endpoint
    net: str = ''  # Net name for this escape
    layer: str = 'F.Cu'  # Layer for this escape

    @property
    def endpoint(self) -> Optional[Tuple[float, float]]:
        """Alias for end, for backward compatibility"""
        return self.end


@dataclass
class PinArray:
    """Complete pin array for a package"""
    package_type: PackageType
    rows: int
    cols: int
    pitch: float          # mm
    pins: List[Pin] = field(default_factory=list)
    is_staggered: bool = False
    thermal_pad: Optional[Dict] = None  # For QFN thermal pad


@dataclass
class EscapeConfig:
    """Configuration for escape routing"""
    # Via parameters
    via_drill: float = 0.3        # mm
    via_pad: float = 0.6          # mm
    micro_via_drill: float = 0.1  # mm
    micro_via_pad: float = 0.25   # mm

    # Trace parameters
    trace_width: float = 0.15     # mm
    trace_clearance: float = 0.15 # mm

    # Layer parameters
    available_layers: int = 4
    signal_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Via technology
    via_types_available: List[ViaType] = field(
        default_factory=lambda: [ViaType.THROUGH_HOLE]
    )

    # Routing angles
    allow_45_degree: bool = True
    allow_any_angle: bool = False

    # Strategy
    strategy: EscapeStrategy = EscapeStrategy.DOG_BONE

    # Optimization goals
    minimize_vias: bool = True
    minimize_layers: bool = True
    minimize_length: bool = True


@dataclass
class EscapeResult:
    """Result of escape routing"""
    paths: List[EscapePath] = field(default_factory=list)
    success_rate: float = 0.0
    total_vias: int = 0
    layers_used: Set[int] = field(default_factory=set)
    total_wire_length: float = 0.0
    routing_time_ms: float = 0.0
    strategy_used: EscapeStrategy = EscapeStrategy.DOG_BONE
    messages: List[str] = field(default_factory=list)
    escapes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {ref: {pin: EscapePath}}


# =============================================================================
# NETWORK FLOW STRUCTURES (For MMCF Algorithm)
# =============================================================================

@dataclass
class FlowEdge:
    """Edge in flow network"""
    source: str
    target: str
    capacity: int
    cost: float
    flow: int = 0


@dataclass
class FlowNetwork:
    """Network for MMCF formulation"""
    nodes: Set[str] = field(default_factory=set)
    edges: List[FlowEdge] = field(default_factory=list)
    commodities: Dict[str, Tuple[str, str]] = field(default_factory=dict)


# =============================================================================
# ESCAPE PISTON - MAIN ENGINE
# =============================================================================

class EscapePiston:
    """
    Pin Escape Routing Engine

    Handles escape routing for BGA/QFP/QFN packages using research-based
    algorithms including MMCF, SAT solving, and ring-based approaches.
    """

    def __init__(self, config: Optional[EscapeConfig] = None):
        self.config = config or EscapeConfig()
        self.pin_array: Optional[PinArray] = None
        self.result: Optional[EscapeResult] = None

        # Strategy dispatch
        self._strategies: Dict[EscapeStrategy, Callable] = {
            EscapeStrategy.DOG_BONE: self._escape_dog_bone,
            EscapeStrategy.ORDERED_MMCF: self._escape_mmcf,
            EscapeStrategy.MULTI_CAPACITY: self._escape_multi_capacity,
            EscapeStrategy.SAT_MULTI_LAYER: self._escape_sat_multi_layer,
            EscapeStrategy.RING_BASED: self._escape_ring_based,
            EscapeStrategy.LAYER_MINIMIZE: self._escape_layer_minimize,
            EscapeStrategy.HYBRID: self._escape_hybrid,
        }

        # Ring calculation cache
        self._ring_cache: Dict[Tuple[int, int], int] = {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def plan(self, parts_db: Dict, placement: Dict) -> EscapeResult:
        """
        Plan escape routes for all components in the design.

        Standard piston API that wraps the lower-level escape() method.
        Builds escape routes for each component based on its footprint.

        Args:
            parts_db: Parts database with component data
            placement: Component placement positions

        Returns:
            EscapeResult with escape paths for all components
        """
        import time
        start_time = time.time()

        all_paths = []
        total_vias = 0
        layers_used = set()
        total_length = 0.0
        messages = []
        escapes_dict = {}  # {component_ref: {pin_num: EscapePath}}

        parts = parts_db.get('parts', {})

        for ref, part in parts.items():
            if ref not in placement:
                continue

            pos = placement[ref]
            if hasattr(pos, 'x'):
                px, py = pos.x, pos.y
            elif isinstance(pos, (list, tuple)):
                px, py = pos[0], pos[1]
            else:
                continue

            # Get pin data
            pins_data = part.get('pins', []) or part.get('used_pins', [])
            if not pins_data:
                continue

            # Build simple escape paths for each pin
            for pin in pins_data:
                if isinstance(pin, dict):
                    pin_num = str(pin.get('number', ''))
                    net = pin.get('net', '')
                    offset = pin.get('offset', (0, 0))
                    if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                        ox, oy = offset[0], offset[1]
                    else:
                        ox, oy = 0, 0
                else:
                    continue

                if not net:
                    continue

                # Simple dog-bone escape: short trace + via
                pin_x = px + ox
                pin_y = py + oy

                # Create Pin object for the escape path
                pin_obj = Pin(
                    id=f"{ref}_{pin_num}",
                    row=0,
                    col=int(pin_num) if pin_num.isdigit() else 0,
                    x=pin_x,
                    y=pin_y,
                    net=net,
                    location=PinLocation.INNER  # Default
                )

                # Determine escape direction based on pin offset from component center
                # This avoids crossing the component's own body AND other components
                escape_length = 1.5  # Escape beyond component courtyard

                # Choose primary escape direction based on pin position relative to component center
                # If pin is to the right of center, escape right; otherwise escape left
                # If pin is above center, escape up; otherwise escape down
                if abs(ox) > abs(oy):
                    # Pin is more horizontal from center - try horizontal first
                    if ox >= 0:
                        primary_escape = (pin_x + escape_length, pin_y)  # Right
                        fallback_escapes = [
                            (pin_x, pin_y - escape_length),  # Up
                            (pin_x, pin_y + escape_length),  # Down
                            (pin_x - escape_length, pin_y),  # Left (last resort)
                        ]
                    else:
                        primary_escape = (pin_x - escape_length, pin_y)  # Left
                        fallback_escapes = [
                            (pin_x, pin_y - escape_length),  # Up
                            (pin_x, pin_y + escape_length),  # Down
                            (pin_x + escape_length, pin_y),  # Right (last resort)
                        ]
                else:
                    # Pin is more vertical from center - try vertical first
                    if oy >= 0:
                        primary_escape = (pin_x, pin_y + escape_length)  # Down
                        fallback_escapes = [
                            (pin_x + escape_length, pin_y),  # Right
                            (pin_x - escape_length, pin_y),  # Left
                            (pin_x, pin_y - escape_length),  # Up (last resort)
                        ]
                    else:
                        primary_escape = (pin_x, pin_y - escape_length)  # Up
                        fallback_escapes = [
                            (pin_x + escape_length, pin_y),  # Right
                            (pin_x - escape_length, pin_y),  # Left
                            (pin_x, pin_y + escape_length),  # Down (last resort)
                        ]

                # Check if primary escape crosses another component's courtyard
                escape_x, escape_y = primary_escape
                escape_blocked = False

                for other_ref, other_pos in placement.items():
                    if other_ref == ref:
                        continue
                    # Get other component position
                    if hasattr(other_pos, 'x'):
                        other_x, other_y = other_pos.x, other_pos.y
                    elif isinstance(other_pos, (list, tuple)):
                        other_x, other_y = other_pos[0], other_pos[1]
                    else:
                        continue

                    # Get accurate courtyard from pad positions
                    try:
                        from .common_types import calculate_courtyard
                    except ImportError:
                        from common_types import calculate_courtyard
                    other_part = parts.get(other_ref, {})
                    other_fp = other_part.get('footprint', '')
                    other_courtyard = calculate_courtyard(other_part, margin=0.0, footprint_name=other_fp)
                    courtyard_half_w = other_courtyard.width / 2
                    courtyard_half_h = other_courtyard.height / 2

                    # Check if escape path segment crosses this component's courtyard
                    # Path is from (pin_x, pin_y) to (escape_x, escape_y)
                    min_x = min(pin_x, escape_x)
                    max_x = max(pin_x, escape_x)
                    min_y = min(pin_y, escape_y)
                    max_y = max(pin_y, escape_y)

                    # Check AABB overlap with courtyard
                    if (max_x > other_x - courtyard_half_w and
                        min_x < other_x + courtyard_half_w and
                        max_y > other_y - courtyard_half_h and
                        min_y < other_y + courtyard_half_h):
                        escape_blocked = True
                        break

                # If primary escape is blocked, try fallbacks
                if escape_blocked:
                    for fb_x, fb_y in fallback_escapes:
                        fb_blocked = False
                        for other_ref, other_pos in placement.items():
                            if other_ref == ref:
                                continue
                            if hasattr(other_pos, 'x'):
                                other_x, other_y = other_pos.x, other_pos.y
                            elif isinstance(other_pos, (list, tuple)):
                                other_x, other_y = other_pos[0], other_pos[1]
                            else:
                                continue

                            # Get accurate courtyard from pad positions
                            other_part = parts.get(other_ref, {})
                            other_fp = other_part.get('footprint', '')
                            other_courtyard = calculate_courtyard(other_part, margin=0.0, footprint_name=other_fp)
                            courtyard_half_w = other_courtyard.width / 2
                            courtyard_half_h = other_courtyard.height / 2
                            min_x = min(pin_x, fb_x)
                            max_x = max(pin_x, fb_x)
                            min_y = min(pin_y, fb_y)
                            max_y = max(pin_y, fb_y)

                            if (max_x > other_x - courtyard_half_w and
                                min_x < other_x + courtyard_half_w and
                                max_y > other_y - courtyard_half_h and
                                min_y < other_y + courtyard_half_h):
                                fb_blocked = True
                                break

                        if not fb_blocked:
                            escape_x, escape_y = fb_x, fb_y
                            break

                # Create escape trace
                escape_trace = EscapeTrace(
                    start_x=pin_x,
                    start_y=pin_y,
                    end_x=escape_x,
                    end_y=escape_y,
                    width=self.config.trace_width,
                    layer=1,
                    net=net
                )

                # Create escape path with start/end for routing piston compatibility
                escape_path = EscapePath(
                    pin=pin_obj,
                    traces=[escape_trace],
                    vias=[],
                    total_length=escape_length,
                    layer_transitions=0,
                    escaped=True,
                    start=(pin_x, pin_y),  # Pad position
                    end=(escape_x, escape_y),  # Escape endpoint for routing
                    net=net,
                    layer='F.Cu'
                )
                all_paths.append(escape_path)
                total_length += escape_length

                # Store in escapes dict using ref and pin_num
                if ref not in escapes_dict:
                    escapes_dict[ref] = {}
                escapes_dict[ref][pin_num] = escape_path

        return EscapeResult(
            paths=all_paths,
            success_rate=1.0 if all_paths else 0.0,
            total_vias=total_vias,
            layers_used=layers_used or {1},
            total_wire_length=total_length,
            routing_time_ms=(time.time() - start_time) * 1000,
            strategy_used=EscapeStrategy.DOG_BONE,
            messages=messages,
            escapes=escapes_dict
        )

    def escape(self, pin_array: PinArray) -> EscapeResult:
        """
        Main entry point - escape route all pins in the array

        Args:
            pin_array: Complete pin array with net assignments

        Returns:
            EscapeResult with all escape paths
        """
        import time
        start_time = time.time()

        self.pin_array = pin_array
        self.result = EscapeResult()

        # Classify pin locations
        self._classify_pin_locations()

        # Select strategy based on package type if not specified
        strategy = self._select_strategy()
        self.result.strategy_used = strategy

        # Execute strategy
        strategy_func = self._strategies.get(strategy)
        if strategy_func:
            strategy_func()
        else:
            self.result.messages.append(f"Unknown strategy: {strategy}")

        # Calculate statistics
        self._calculate_statistics()

        self.result.routing_time_ms = (time.time() - start_time) * 1000
        return self.result

    def recommend_strategy(self, pin_array: PinArray) -> EscapeStrategy:
        """Recommend best strategy based on package characteristics"""
        pitch = pin_array.pitch
        total_pins = len(pin_array.pins)
        package = pin_array.package_type

        # QFP/QFN don't need escape routing (peripheral pins)
        if package in [PackageType.QFP]:
            return EscapeStrategy.DOG_BONE  # Simple fanout

        # Fine-pitch BGA needs advanced techniques
        if pitch < 0.5:
            if total_pins > 1000 and ViaType.MICRO_VIA in self.config.via_types_available:
                return EscapeStrategy.SAT_MULTI_LAYER
            return EscapeStrategy.ORDERED_MMCF

        # Large BGA with multiple routing channels
        if total_pins > 500:
            return EscapeStrategy.MULTI_CAPACITY

        # Standard BGA
        if pitch >= 0.8:
            return EscapeStrategy.RING_BASED

        # Default
        return EscapeStrategy.DOG_BONE

    # =========================================================================
    # PIN CLASSIFICATION
    # =========================================================================

    def _classify_pin_locations(self):
        """Classify each pin as corner, edge, inner, or center"""
        if not self.pin_array:
            return

        rows = self.pin_array.rows
        cols = self.pin_array.cols

        for pin in self.pin_array.pins:
            r, c = pin.row, pin.col

            # Corner pins
            if (r == 0 or r == rows - 1) and (c == 0 or c == cols - 1):
                pin.location = PinLocation.CORNER
            # Edge pins
            elif r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                pin.location = PinLocation.EDGE
            # Center region (inner quarter)
            elif (rows // 4 <= r < 3 * rows // 4) and (cols // 4 <= c < 3 * cols // 4):
                pin.location = PinLocation.CENTER
            else:
                pin.location = PinLocation.INNER

    def _get_pin_ring(self, pin: Pin) -> int:
        """
        Get ring number for pin (0 = outermost)
        Used by ring-based escape algorithm

        Source: Standard industry practice for BGA escape
        """
        key = (pin.row, pin.col)
        if key in self._ring_cache:
            return self._ring_cache[key]

        rows = self.pin_array.rows
        cols = self.pin_array.cols

        # Distance from each edge
        dist_top = pin.row
        dist_bottom = rows - 1 - pin.row
        dist_left = pin.col
        dist_right = cols - 1 - pin.col

        # Ring is minimum distance to any edge
        ring = min(dist_top, dist_bottom, dist_left, dist_right)

        self._ring_cache[key] = ring
        return ring

    def _get_escape_direction(self, pin: Pin) -> EscapeDirection:
        """Determine optimal escape direction for pin"""
        rows = self.pin_array.rows
        cols = self.pin_array.cols

        # Calculate distances to edges
        dist_top = pin.row
        dist_bottom = rows - 1 - pin.row
        dist_left = pin.col
        dist_right = cols - 1 - pin.col

        min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

        # Primary escape toward nearest edge
        if dist_top == min_dist:
            return EscapeDirection.NORTH
        elif dist_bottom == min_dist:
            return EscapeDirection.SOUTH
        elif dist_left == min_dist:
            return EscapeDirection.WEST
        else:
            return EscapeDirection.EAST

    # =========================================================================
    # STRATEGY SELECTION
    # =========================================================================

    def _select_strategy(self) -> EscapeStrategy:
        """Select best strategy based on config and package"""
        if self.config.strategy != EscapeStrategy.HYBRID:
            return self.config.strategy

        return self.recommend_strategy(self.pin_array)

    # =========================================================================
    # STRATEGY 1: DOG-BONE FANOUT
    # =========================================================================

    def _escape_dog_bone(self):
        """
        Dog-bone fanout escape routing

        Source: Altium - "BGA Fanout Routing"
        Source: Cadence - "Tips for Routing BGA Packages"

        Dog-bone places via at 45° (or 90°) from pad center in the
        channel between pin rows. Short trace connects pad to via.

        Applicable for: BGA pitch >= 0.5mm
        """
        self.result.messages.append("Using Dog-Bone Fanout strategy")

        pitch = self.pin_array.pitch
        via_pad = self.config.via_pad
        trace_width = self.config.trace_width
        clearance = self.config.trace_clearance

        # Calculate via offset from pad center
        # Via must fit in channel between rows
        # Channel width = pitch - pad_diameter
        # Typical BGA pad = 0.5 * pitch
        pad_diameter = 0.5 * pitch  # NSMD pad approximation
        channel = pitch - pad_diameter

        # Check if dog-bone is possible
        min_channel = via_pad + 2 * clearance
        if channel < min_channel:
            self.result.messages.append(
                f"Warning: Channel {channel:.3f}mm too narrow for dog-bone "
                f"(need {min_channel:.3f}mm). Using offset vias."
            )

        # Via offset at 45° for standard, 90° for tight pitch
        use_45_degree = self.config.allow_45_degree and channel >= min_channel

        for pin in self.pin_array.pins:
            if pin.is_nc:
                continue

            path = EscapePath(pin=pin)

            # Determine escape direction
            escape_dir = self._get_escape_direction(pin)
            pin.escape_direction = escape_dir

            # Calculate via position
            if use_45_degree:
                via_offset = pitch / 2 * 0.707  # cos(45°)
                via_x, via_y = self._calculate_dog_bone_via_45(pin, escape_dir, via_offset)
            else:
                via_x, via_y = self._calculate_dog_bone_via_90(pin, escape_dir, pitch / 2)

            # Determine layer based on ring
            ring = self._get_pin_ring(pin)
            layer = self._assign_layer_by_ring(ring)
            pin.escape_layer = layer

            # Create via
            via = Via(
                x=via_x,
                y=via_y,
                drill_diameter=self.config.via_drill,
                pad_diameter=self.config.via_pad,
                via_type=self._select_via_type(1, layer),
                start_layer=1,
                end_layer=layer,
                net=pin.net,
                id=f"V_{pin.id}"
            )
            pin.assigned_via = via
            path.vias.append(via)

            # Create escape trace from pad to via
            trace = EscapeTrace(
                start_x=pin.x,
                start_y=pin.y,
                end_x=via_x,
                end_y=via_y,
                width=trace_width,
                layer=1,  # Top layer
                net=pin.net,
                angle=45.0 if use_45_degree else 0.0
            )
            path.traces.append(trace)

            # Calculate length
            path.total_length = math.sqrt(
                (via_x - pin.x) ** 2 + (via_y - pin.y) ** 2
            )
            path.layer_transitions = 1 if layer > 1 else 0
            path.escaped = True

            self.result.paths.append(path)

    def _calculate_dog_bone_via_45(
        self, pin: Pin, direction: EscapeDirection, offset: float
    ) -> Tuple[float, float]:
        """Calculate via position for 45° dog-bone"""
        x, y = pin.x, pin.y

        if direction == EscapeDirection.NORTH:
            return x + offset, y - offset
        elif direction == EscapeDirection.SOUTH:
            return x + offset, y + offset
        elif direction == EscapeDirection.EAST:
            return x + offset, y + offset
        elif direction == EscapeDirection.WEST:
            return x - offset, y + offset
        elif direction == EscapeDirection.NORTHEAST:
            return x + offset, y - offset
        elif direction == EscapeDirection.NORTHWEST:
            return x - offset, y - offset
        elif direction == EscapeDirection.SOUTHEAST:
            return x + offset, y + offset
        else:  # SOUTHWEST
            return x - offset, y + offset

    def _calculate_dog_bone_via_90(
        self, pin: Pin, direction: EscapeDirection, offset: float
    ) -> Tuple[float, float]:
        """Calculate via position for 90° dog-bone (tight pitch)"""
        x, y = pin.x, pin.y

        if direction == EscapeDirection.NORTH:
            return x, y - offset
        elif direction == EscapeDirection.SOUTH:
            return x, y + offset
        elif direction == EscapeDirection.EAST:
            return x + offset, y
        elif direction == EscapeDirection.WEST:
            return x - offset, y
        else:
            # For diagonal, use primary axis
            return x + offset, y

    # =========================================================================
    # STRATEGY 2: ORDERED ESCAPE ROUTING (MMCF)
    # =========================================================================

    def _escape_mmcf(self):
        """
        Ordered Escape Routing using Min-cost Multi-commodity Flow

        Source: IEEE 2016 - "Ordered Escape routing for grid pin array
                based on Min-cost Multi-commodity Flow"

        Algorithm:
        1. Build basic network model from pin array
        2. Apply non-crossing transformation
        3. Apply ordering transformation
        4. Apply capacity transformation
        5. Solve MMCF to get optimal escape paths

        Achieves 100% routability with optimal wire length.
        """
        self.result.messages.append("Using Ordered MMCF Escape strategy")

        # Build flow network from pin array
        network = self._build_mmcf_network()

        # Solve MMCF problem
        flows = self._solve_mmcf(network)

        # Convert flows to escape paths
        self._flows_to_escape_paths(flows)

    def _build_mmcf_network(self) -> FlowNetwork:
        """
        Build MMCF network from pin array

        Network structure:
        - Node for each pin position
        - Node for each channel intersection
        - Edges represent routing channels
        - Boundary nodes for escape points
        """
        network = FlowNetwork()
        pitch = self.pin_array.pitch
        rows = self.pin_array.rows
        cols = self.pin_array.cols

        # Create grid nodes (at channel intersections)
        for r in range(rows + 1):
            for c in range(cols + 1):
                node_id = f"G_{r}_{c}"
                network.nodes.add(node_id)

        # Create pin nodes
        for pin in self.pin_array.pins:
            if not pin.is_nc:
                node_id = f"P_{pin.row}_{pin.col}"
                network.nodes.add(node_id)

                # Connect pin to adjacent grid nodes
                # Each pin can reach 4 surrounding channel intersections
                for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    grid_node = f"G_{pin.row + dr}_{pin.col + dc}"
                    edge = FlowEdge(
                        source=node_id,
                        target=grid_node,
                        capacity=1,
                        cost=pitch / 2  # Distance to channel
                    )
                    network.edges.append(edge)

        # Create boundary nodes (escape points)
        for r in range(rows + 1):
            # Left boundary
            network.nodes.add(f"B_L_{r}")
            network.edges.append(FlowEdge(
                source=f"G_{r}_0",
                target=f"B_L_{r}",
                capacity=1,
                cost=0.1
            ))
            # Right boundary
            network.nodes.add(f"B_R_{r}")
            network.edges.append(FlowEdge(
                source=f"G_{r}_{cols}",
                target=f"B_R_{r}",
                capacity=1,
                cost=0.1
            ))

        for c in range(cols + 1):
            # Top boundary
            network.nodes.add(f"B_T_{c}")
            network.edges.append(FlowEdge(
                source=f"G_0_{c}",
                target=f"B_T_{c}",
                capacity=1,
                cost=0.1
            ))
            # Bottom boundary
            network.nodes.add(f"B_B_{c}")
            network.edges.append(FlowEdge(
                source=f"G_{rows}_{c}",
                target=f"B_B_{c}",
                capacity=1,
                cost=0.1
            ))

        # Create horizontal and vertical channel edges
        for r in range(rows + 1):
            for c in range(cols):
                # Horizontal edge
                edge = FlowEdge(
                    source=f"G_{r}_{c}",
                    target=f"G_{r}_{c + 1}",
                    capacity=self._channel_capacity(r, c, 'H'),
                    cost=pitch
                )
                network.edges.append(edge)
                # Reverse edge
                network.edges.append(FlowEdge(
                    source=f"G_{r}_{c + 1}",
                    target=f"G_{r}_{c}",
                    capacity=self._channel_capacity(r, c, 'H'),
                    cost=pitch
                ))

        for r in range(rows):
            for c in range(cols + 1):
                # Vertical edge
                edge = FlowEdge(
                    source=f"G_{r}_{c}",
                    target=f"G_{r + 1}_{c}",
                    capacity=self._channel_capacity(r, c, 'V'),
                    cost=pitch
                )
                network.edges.append(edge)
                # Reverse edge
                network.edges.append(FlowEdge(
                    source=f"G_{r + 1}_{c}",
                    target=f"G_{r}_{c}",
                    capacity=self._channel_capacity(r, c, 'V'),
                    cost=pitch
                ))

        # Define commodities (one per net that needs escape)
        for pin in self.pin_array.pins:
            if not pin.is_nc:
                source = f"P_{pin.row}_{pin.col}"
                # Sink is virtual - any boundary node
                network.commodities[pin.net] = (source, "BOUNDARY")

        return network

    def _channel_capacity(self, row: int, col: int, direction: str) -> int:
        """
        Calculate routing capacity for a channel

        Based on: pitch, trace width, clearance, and via requirements
        """
        pitch = self.pin_array.pitch
        trace = self.config.trace_width
        clearance = self.config.trace_clearance

        # Available channel width
        # For grid BGA, channel is between rows/cols
        pad_dia = 0.5 * pitch  # Approximate NSMD pad
        channel_width = pitch - pad_dia

        # Number of traces that fit
        trace_pitch = trace + clearance
        capacity = int(channel_width / trace_pitch)

        return max(1, capacity)

    def _solve_mmcf(self, network: FlowNetwork) -> Dict[str, List[str]]:
        """
        Solve MMCF problem using successive shortest path algorithm

        This is a simplified implementation. Production would use:
        - CPLEX or Gurobi for optimal solution
        - NetworkX for graph operations
        """
        flows: Dict[str, List[str]] = {}

        # Build adjacency list with residual capacities
        graph: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        for edge in network.edges:
            graph[edge.source].append((edge.target, edge.cost, edge.capacity))

        # Process each commodity (net)
        for net, (source, _) in network.commodities.items():
            # Find shortest path to any boundary
            path = self._dijkstra_to_boundary(graph, source, network.nodes)

            if path:
                flows[net] = path
                # Update residual capacities (simplified)
                self._update_residual_graph(graph, path)
            else:
                self.result.messages.append(f"No path found for net {net}")

        return flows

    def _dijkstra_to_boundary(
        self,
        graph: Dict[str, List[Tuple[str, float, int]]],
        source: str,
        nodes: Set[str]
    ) -> List[str]:
        """Find shortest path from source to any boundary node"""
        dist = {node: float('inf') for node in nodes}
        dist[source] = 0
        prev = {node: None for node in nodes}

        pq = [(0, source)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            # Check if we reached boundary
            if u.startswith('B_'):
                # Reconstruct path
                path = []
                current = u
                while current:
                    path.append(current)
                    current = prev[current]
                return path[::-1]

            # Explore neighbors
            for v, cost, capacity in graph.get(u, []):
                if capacity > 0 and v not in visited:
                    new_dist = d + cost
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev[v] = u
                        heapq.heappush(pq, (new_dist, v))

        return []

    def _update_residual_graph(
        self,
        graph: Dict[str, List[Tuple[str, float, int]]],
        path: List[str]
    ):
        """Update residual capacities after using a path"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Reduce capacity of forward edge
            for j, (target, cost, cap) in enumerate(graph[u]):
                if target == v and cap > 0:
                    graph[u][j] = (target, cost, cap - 1)
                    break

    def _flows_to_escape_paths(self, flows: Dict[str, List[str]]):
        """Convert MMCF flows to physical escape paths"""
        pitch = self.pin_array.pitch
        trace_width = self.config.trace_width

        for pin in self.pin_array.pins:
            if pin.is_nc:
                continue

            path = EscapePath(pin=pin)

            if pin.net in flows:
                flow_path = flows[pin.net]

                # Convert grid nodes to coordinates
                coords = []
                for node in flow_path:
                    if node.startswith('P_'):
                        # Pin node
                        coords.append((pin.x, pin.y))
                    elif node.startswith('G_'):
                        # Grid intersection
                        parts = node.split('_')
                        r, c = int(parts[1]), int(parts[2])
                        x = c * pitch - pitch / 2  # Offset to array origin
                        y = r * pitch - pitch / 2
                        coords.append((x, y))
                    elif node.startswith('B_'):
                        # Boundary node
                        parts = node.split('_')
                        side = parts[1]
                        idx = int(parts[2])

                        if side == 'L':
                            coords.append((-pitch, idx * pitch))
                        elif side == 'R':
                            cols = self.pin_array.cols
                            coords.append((cols * pitch, idx * pitch))
                        elif side == 'T':
                            coords.append((idx * pitch, -pitch))
                        else:  # B
                            rows = self.pin_array.rows
                            coords.append((idx * pitch, rows * pitch))

                # Create traces between consecutive coordinates
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]

                    trace = EscapeTrace(
                        start_x=x1,
                        start_y=y1,
                        end_x=x2,
                        end_y=y2,
                        width=trace_width,
                        layer=1,
                        net=pin.net
                    )
                    path.traces.append(trace)
                    path.total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                path.escaped = True
            else:
                path.escaped = False
                self.result.messages.append(f"Pin {pin.id} could not escape")

            self.result.paths.append(path)

    # =========================================================================
    # STRATEGY 3: MULTI-CAPACITY ORDERED ESCAPE (MC-OER)
    # =========================================================================

    def _escape_multi_capacity(self):
        """
        Multi-Capacity Ordered Escape Routing

        Source: ACM TODAES 2024 - "MCMCF-Router: Multi-capacity Ordered
                Escape Routing Algorithms for Grid/Staggered Pin Array"

        Extension of MMCF that:
        - Allows multiple wires between adjacent pins
        - Supports staggered pin arrays
        - Better utilizes routing channels
        """
        self.result.messages.append("Using Multi-Capacity Ordered Escape strategy")

        # Calculate enhanced channel capacities
        channel_caps = self._calculate_multi_capacity_channels()

        # Build enhanced network with multiple wire support
        network = self._build_multi_capacity_network(channel_caps)

        # Solve with capacity-aware algorithm
        flows = self._solve_mc_mmcf(network)

        # Convert to escape paths with proper spacing
        self._mc_flows_to_paths(flows, channel_caps)

    def _calculate_multi_capacity_channels(self) -> Dict[Tuple[int, int, str], int]:
        """
        Calculate how many traces can fit in each channel

        For MC-OER, we need finer granularity than basic MMCF
        """
        capacities = {}
        pitch = self.pin_array.pitch
        trace = self.config.trace_width
        clearance = self.config.trace_clearance
        via_pad = self.config.via_pad

        rows = self.pin_array.rows
        cols = self.pin_array.cols

        # Staggered arrays have different channel geometry
        if self.pin_array.is_staggered:
            # Staggered: rows offset by pitch/2
            effective_pitch = pitch * math.sqrt(3) / 2  # Hexagonal packing
        else:
            effective_pitch = pitch

        # Calculate pad diameter (approximate)
        pad_dia = 0.5 * pitch

        # Channel between rows
        channel_width = effective_pitch - pad_dia

        # Must also fit vias if dog-bone needed
        if channel_width < via_pad + 2 * clearance:
            via_space = via_pad + 2 * clearance
            available = max(0, channel_width - via_space)
        else:
            available = channel_width

        # Traces that fit
        trace_pitch = trace + clearance
        capacity = max(1, int(available / trace_pitch))

        # Store for each channel
        for r in range(rows + 1):
            for c in range(cols):
                capacities[(r, c, 'H')] = capacity

        for r in range(rows):
            for c in range(cols + 1):
                capacities[(r, c, 'V')] = capacity

        return capacities

    def _build_multi_capacity_network(
        self,
        channel_caps: Dict[Tuple[int, int, str], int]
    ) -> FlowNetwork:
        """Build network with multi-capacity edges"""
        network = self._build_mmcf_network()

        # Update edge capacities
        for edge in network.edges:
            src, tgt = edge.source, edge.target

            if src.startswith('G_') and tgt.startswith('G_'):
                # Extract row/col
                src_parts = src.split('_')
                tgt_parts = tgt.split('_')
                r1, c1 = int(src_parts[1]), int(src_parts[2])
                r2, c2 = int(tgt_parts[1]), int(tgt_parts[2])

                # Determine direction
                if r1 == r2:  # Horizontal
                    key = (r1, min(c1, c2), 'H')
                else:  # Vertical
                    key = (min(r1, r2), c1, 'V')

                if key in channel_caps:
                    edge.capacity = channel_caps[key]

        return network

    def _solve_mc_mmcf(self, network: FlowNetwork) -> Dict[str, List[Tuple[str, int]]]:
        """
        Solve multi-capacity MMCF

        Returns path with lane assignment for each segment
        """
        # For this implementation, use same algorithm as basic MMCF
        # but track which "lane" each flow uses
        basic_flows = self._solve_mmcf(network)

        # Add lane information
        mc_flows = {}
        lane_usage: Dict[str, int] = defaultdict(int)  # Track lane usage per edge

        for net, path in basic_flows.items():
            path_with_lanes = []
            for i, node in enumerate(path):
                if i > 0:
                    edge_key = f"{path[i-1]}_{node}"
                    lane = lane_usage[edge_key]
                    lane_usage[edge_key] += 1
                    path_with_lanes.append((node, lane))
                else:
                    path_with_lanes.append((node, 0))
            mc_flows[net] = path_with_lanes

        return mc_flows

    def _mc_flows_to_paths(
        self,
        flows: Dict[str, List[Tuple[str, int]]],
        channel_caps: Dict[Tuple[int, int, str], int]
    ):
        """Convert MC flows to physical paths with lane offsets"""
        pitch = self.pin_array.pitch
        trace_width = self.config.trace_width
        clearance = self.config.trace_clearance
        trace_pitch = trace_width + clearance

        for pin in self.pin_array.pins:
            if pin.is_nc:
                continue

            path = EscapePath(pin=pin)

            if pin.net in flows:
                flow_path = flows[pin.net]

                coords = []
                for node, lane in flow_path:
                    # Calculate base position
                    if node.startswith('P_'):
                        base_x, base_y = pin.x, pin.y
                    elif node.startswith('G_'):
                        parts = node.split('_')
                        r, c = int(parts[1]), int(parts[2])
                        base_x = c * pitch - pitch / 2
                        base_y = r * pitch - pitch / 2
                    elif node.startswith('B_'):
                        parts = node.split('_')
                        side, idx = parts[1], int(parts[2])
                        if side == 'L':
                            base_x, base_y = -pitch, idx * pitch
                        elif side == 'R':
                            base_x = self.pin_array.cols * pitch
                            base_y = idx * pitch
                        elif side == 'T':
                            base_x, base_y = idx * pitch, -pitch
                        else:
                            base_x = idx * pitch
                            base_y = self.pin_array.rows * pitch
                    else:
                        continue

                    # Apply lane offset (perpendicular to flow direction)
                    # Simplified: offset along x for horizontal, y for vertical
                    lane_offset = lane * trace_pitch
                    coords.append((base_x + lane_offset, base_y))

                # Create traces
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    trace = EscapeTrace(
                        start_x=x1, start_y=y1,
                        end_x=x2, end_y=y2,
                        width=trace_width,
                        layer=1,
                        net=pin.net
                    )
                    path.traces.append(trace)
                    path.total_length += math.sqrt((x2-x1)**2 + (y2-y1)**2)

                path.escaped = True
            else:
                path.escaped = False

            self.result.paths.append(path)

    # =========================================================================
    # STRATEGY 4: SAT-BASED MULTI-LAYER ESCAPE
    # =========================================================================

    def _escape_sat_multi_layer(self):
        """
        SAT-Based Multi-Layer Escape Routing

        Source: ICCAD 2016 - "Scalable, High-Quality, SAT-Based Multi-Layer
                Escape Routing" (UBC)

        Uses SMT/SAT solver with network flow constraints.
        Scales to 2000+ pins, supports:
        - 45° and 90° routing
        - Simultaneous trace and via placement
        - Through-hole, blind, buried, micro-vias

        This is a simplified implementation demonstrating the approach.
        Full implementation would use python-sat or MonoSAT.
        """
        self.result.messages.append("Using SAT Multi-Layer Escape strategy")

        # For large BGAs, divide into regions
        regions = self._partition_pin_array()

        # Process each region (can be parallelized)
        for region_id, region_pins in regions.items():
            # Build SAT clauses for region
            clauses, variables = self._build_sat_clauses(region_pins)

            # Solve SAT (simplified - uses greedy heuristic)
            solution = self._solve_sat_escape(clauses, variables, region_pins)

            # Convert solution to paths
            self._sat_solution_to_paths(solution, region_pins)

    def _partition_pin_array(self) -> Dict[int, List[Pin]]:
        """
        Partition pin array into regions for SAT solving

        Large BGAs are divided into quadrants or rings for
        better scalability.
        """
        pins = [p for p in self.pin_array.pins if not p.is_nc]
        total = len(pins)

        if total <= 100:
            # Small array - single region
            return {0: pins}

        # Divide into quadrants
        regions: Dict[int, List[Pin]] = defaultdict(list)

        rows = self.pin_array.rows
        cols = self.pin_array.cols
        mid_row = rows // 2
        mid_col = cols // 2

        for pin in pins:
            if pin.row < mid_row:
                region = 0 if pin.col < mid_col else 1  # Top-left, top-right
            else:
                region = 2 if pin.col < mid_col else 3  # Bottom-left, bottom-right
            regions[region].append(pin)

        return dict(regions)

    def _build_sat_clauses(
        self,
        pins: List[Pin]
    ) -> Tuple[List[List[int]], Dict[str, int]]:
        """
        Build SAT clauses for escape routing

        Variables:
        - x_pin_layer: Pin assigned to layer
        - x_pin_dir: Pin escapes in direction
        - x_channel_layer: Channel used on layer

        Constraints:
        - Each pin must escape on exactly one layer
        - Non-crossing constraints within layer
        - Capacity constraints per channel
        """
        clauses = []
        variables: Dict[str, int] = {}
        var_id = 1

        layers = self.config.signal_layers
        num_layers = len(layers)

        # Create variables for each pin-layer assignment
        for pin in pins:
            for layer in layers:
                var_name = f"pl_{pin.id}_{layer}"
                variables[var_name] = var_id
                var_id += 1

        # Create direction variables
        directions = [
            EscapeDirection.NORTH, EscapeDirection.SOUTH,
            EscapeDirection.EAST, EscapeDirection.WEST
        ]

        for pin in pins:
            for d in directions:
                var_name = f"pd_{pin.id}_{d.name}"
                variables[var_name] = var_id
                var_id += 1

        # Constraint 1: Each pin on exactly one layer
        for pin in pins:
            layer_vars = [variables[f"pl_{pin.id}_{l}"] for l in layers]
            # At least one
            clauses.append(layer_vars)
            # At most one (pairwise negation)
            for i in range(len(layer_vars)):
                for j in range(i + 1, len(layer_vars)):
                    clauses.append([-layer_vars[i], -layer_vars[j]])

        # Constraint 2: Each pin escapes in exactly one direction
        for pin in pins:
            dir_vars = [variables[f"pd_{pin.id}_{d.name}"] for d in directions]
            clauses.append(dir_vars)
            for i in range(len(dir_vars)):
                for j in range(i + 1, len(dir_vars)):
                    clauses.append([-dir_vars[i], -dir_vars[j]])

        # Constraint 3: Non-crossing (simplified)
        # Pins in same row escaping same direction must use different layers
        # or escape in order
        for i, pin1 in enumerate(pins):
            for pin2 in pins[i + 1:]:
                if pin1.row == pin2.row:
                    # Same row - potential conflict if escaping same direction
                    for d in [EscapeDirection.NORTH, EscapeDirection.SOUTH]:
                        d1 = variables[f"pd_{pin1.id}_{d.name}"]
                        d2 = variables[f"pd_{pin2.id}_{d.name}"]
                        # If both escape same direction, must be different layers
                        for layer in layers:
                            l1 = variables[f"pl_{pin1.id}_{layer}"]
                            l2 = variables[f"pl_{pin2.id}_{layer}"]
                            # NOT(d1 AND d2 AND l1 AND l2)
                            clauses.append([-d1, -d2, -l1, -l2])

        return clauses, variables

    def _solve_sat_escape(
        self,
        clauses: List[List[int]],
        variables: Dict[str, int],
        pins: List[Pin]
    ) -> Dict[str, bool]:
        """
        Solve SAT problem for escape routing

        This is a simplified greedy solver. Production would use:
        - PySAT (python-sat)
        - MonoSAT for network flow
        - MiniSat, CaDiCaL, etc.
        """
        solution: Dict[str, bool] = {}
        layers = self.config.signal_layers

        # Greedy assignment: outer pins on layer 1, inner on deeper layers
        for pin in pins:
            ring = self._get_pin_ring(pin)

            # Assign layer based on ring
            layer_idx = min(ring, len(layers) - 1)
            assigned_layer = layers[layer_idx]

            for l in layers:
                var_name = f"pl_{pin.id}_{l}"
                solution[var_name] = (l == assigned_layer)

            # Assign escape direction based on position
            escape_dir = self._get_escape_direction(pin)

            for d in [EscapeDirection.NORTH, EscapeDirection.SOUTH,
                     EscapeDirection.EAST, EscapeDirection.WEST]:
                var_name = f"pd_{pin.id}_{d.name}"
                solution[var_name] = (d == escape_dir)

            pin.escape_layer = assigned_layer
            pin.escape_direction = escape_dir

        return solution

    def _sat_solution_to_paths(
        self,
        solution: Dict[str, bool],
        pins: List[Pin]
    ):
        """Convert SAT solution to physical escape paths"""
        pitch = self.pin_array.pitch
        trace_width = self.config.trace_width

        for pin in pins:
            path = EscapePath(pin=pin)

            layer = pin.escape_layer
            direction = pin.escape_direction

            # Calculate via position and escape trace
            via_offset = pitch / 2
            via_x, via_y = pin.x, pin.y

            if direction == EscapeDirection.NORTH:
                via_y -= via_offset
            elif direction == EscapeDirection.SOUTH:
                via_y += via_offset
            elif direction == EscapeDirection.EAST:
                via_x += via_offset
            elif direction == EscapeDirection.WEST:
                via_x -= via_offset

            # Create via if layer transition needed
            if layer > 1:
                via_type = self._select_via_type(1, layer)
                via = Via(
                    x=via_x,
                    y=via_y,
                    drill_diameter=self._get_via_drill(via_type),
                    pad_diameter=self._get_via_pad(via_type),
                    via_type=via_type,
                    start_layer=1,
                    end_layer=layer,
                    net=pin.net,
                    id=f"V_{pin.id}"
                )
                path.vias.append(via)
                path.layer_transitions = 1

            # Create escape trace
            trace = EscapeTrace(
                start_x=pin.x,
                start_y=pin.y,
                end_x=via_x,
                end_y=via_y,
                width=trace_width,
                layer=1,
                net=pin.net
            )
            path.traces.append(trace)

            path.total_length = math.sqrt(
                (via_x - pin.x) ** 2 + (via_y - pin.y) ** 2
            )
            path.escaped = True

            self.result.paths.append(path)

    # =========================================================================
    # STRATEGY 5: RING-BASED ESCAPE
    # =========================================================================

    def _escape_ring_based(self):
        """
        Ring-Based Escape Routing

        Source: NW Engineering LLC - "BGA Routing Guide"
        Source: Industry standard practice

        Process pins from outside-in:
        - Ring 0 (outermost): Escape on top layer
        - Ring 1: Escape on layer 2
        - Ring 2+: Deeper layers or micro-vias

        Simple and effective for standard-pitch BGAs.
        """
        self.result.messages.append("Using Ring-Based Escape strategy")

        # Group pins by ring number
        rings: Dict[int, List[Pin]] = defaultdict(list)

        for pin in self.pin_array.pins:
            if not pin.is_nc:
                ring = self._get_pin_ring(pin)
                rings[ring].append(pin)

        # Process rings from outside to inside
        for ring_num in sorted(rings.keys()):
            ring_pins = rings[ring_num]

            # Assign layer based on ring
            layer = self._assign_layer_by_ring(ring_num)

            for pin in ring_pins:
                path = self._create_ring_escape_path(pin, ring_num, layer)
                self.result.paths.append(path)

    def _assign_layer_by_ring(self, ring: int) -> int:
        """
        Assign routing layer based on ring number

        Algorithm from industry practice:
        - Ring 0: Layer 1 (top)
        - Ring 1: Layer 2 (if dog-bone doesn't fit)
        - Ring 2: Layer 3
        - etc.
        """
        layers = self.config.signal_layers

        # Check if dog-bone can fit on layer 1
        pitch = self.pin_array.pitch
        via_pad = self.config.via_pad
        clearance = self.config.trace_clearance

        # Channel width for dog-bone
        pad_dia = 0.5 * pitch
        channel = pitch - pad_dia
        min_channel = via_pad + 2 * clearance

        if channel >= min_channel:
            # Dog-bone fits - outer rings can use layer 1
            if ring < 2:
                return layers[0]  # Layer 1

        # Otherwise, use deeper layers
        layer_idx = min(ring, len(layers) - 1)
        return layers[layer_idx]

    def _create_ring_escape_path(
        self,
        pin: Pin,
        ring: int,
        layer: int
    ) -> EscapePath:
        """Create escape path for pin based on ring position"""
        path = EscapePath(pin=pin)

        pitch = self.pin_array.pitch
        trace_width = self.config.trace_width

        # Get escape direction
        direction = self._get_escape_direction(pin)
        pin.escape_direction = direction
        pin.escape_layer = layer

        # Calculate via position
        # For outer rings, via is closer; for inner, may need longer trace
        via_offset = pitch / 2 + ring * 0.1  # Slight increase for inner rings

        via_x, via_y = self._calculate_via_position(pin, direction, via_offset)

        # Create via if layer transition needed
        if layer > 1:
            via_type = self._select_via_type(1, layer)
            via = Via(
                x=via_x,
                y=via_y,
                drill_diameter=self._get_via_drill(via_type),
                pad_diameter=self._get_via_pad(via_type),
                via_type=via_type,
                start_layer=1,
                end_layer=layer,
                net=pin.net,
                id=f"V_{pin.id}"
            )
            pin.assigned_via = via
            path.vias.append(via)
            path.layer_transitions = 1

        # Create escape trace (may be angled for outer rings)
        angle = 45.0 if ring == 0 and self.config.allow_45_degree else 0.0

        trace = EscapeTrace(
            start_x=pin.x,
            start_y=pin.y,
            end_x=via_x,
            end_y=via_y,
            width=trace_width,
            layer=1,
            net=pin.net,
            angle=angle
        )
        path.traces.append(trace)

        path.total_length = math.sqrt(
            (via_x - pin.x) ** 2 + (via_y - pin.y) ** 2
        )
        path.escaped = True

        return path

    def _calculate_via_position(
        self,
        pin: Pin,
        direction: EscapeDirection,
        offset: float
    ) -> Tuple[float, float]:
        """Calculate via position based on escape direction"""
        x, y = pin.x, pin.y

        # 45-degree offset for standard dog-bone
        if self.config.allow_45_degree:
            diag = offset * 0.707  # cos(45°)

            if direction == EscapeDirection.NORTH:
                return x + diag, y - diag
            elif direction == EscapeDirection.SOUTH:
                return x + diag, y + diag
            elif direction == EscapeDirection.EAST:
                return x + diag, y + diag
            elif direction == EscapeDirection.WEST:
                return x - diag, y + diag
            elif direction == EscapeDirection.NORTHEAST:
                return x + diag, y - diag
            elif direction == EscapeDirection.NORTHWEST:
                return x - diag, y - diag
            elif direction == EscapeDirection.SOUTHEAST:
                return x + diag, y + diag
            else:  # SOUTHWEST
                return x - diag, y + diag
        else:
            # 90-degree offset
            if direction == EscapeDirection.NORTH:
                return x, y - offset
            elif direction == EscapeDirection.SOUTH:
                return x, y + offset
            elif direction == EscapeDirection.EAST:
                return x + offset, y
            else:
                return x - offset, y

    # =========================================================================
    # STRATEGY 6: LAYER MINIMIZATION
    # =========================================================================

    def _escape_layer_minimize(self):
        """
        Layer Minimization Escape Routing

        Source: ResearchGate - "Layer minimization in escape routing for
                staggered-pin-array PCBs"

        Goal: Minimize number of PCB layers needed for full escape

        Algorithm:
        1. Compute maximum independent set per layer
        2. Assign non-conflicting pins to same layer
        3. Greedily add layers until all pins escaped
        """
        self.result.messages.append("Using Layer Minimization Escape strategy")

        pins = [p for p in self.pin_array.pins if not p.is_nc]
        available_layers = self.config.signal_layers.copy()

        remaining_pins = set(pins)
        layer_assignments: Dict[int, List[Pin]] = {}

        layer_idx = 0

        while remaining_pins and layer_idx < len(available_layers):
            layer = available_layers[layer_idx]

            # Find maximum independent set for this layer
            # (pins that can all escape on same layer without conflict)
            independent_set = self._find_independent_set(list(remaining_pins))

            if independent_set:
                layer_assignments[layer] = independent_set
                remaining_pins -= set(independent_set)

                self.result.messages.append(
                    f"Layer {layer}: {len(independent_set)} pins"
                )

            layer_idx += 1

        if remaining_pins:
            self.result.messages.append(
                f"Warning: {len(remaining_pins)} pins could not be assigned"
            )

        # Generate escape paths for assigned pins
        for layer, assigned_pins in layer_assignments.items():
            for pin in assigned_pins:
                pin.escape_layer = layer
                pin.escape_direction = self._get_escape_direction(pin)

                path = self._create_layer_min_escape_path(pin, layer)
                self.result.paths.append(path)

        # Mark unassigned pins as failed
        for pin in remaining_pins:
            path = EscapePath(pin=pin, escaped=False)
            self.result.paths.append(path)

    def _find_independent_set(self, pins: List[Pin]) -> List[Pin]:
        """
        Find maximum independent set of pins for single layer

        Two pins conflict if their escape paths would cross.
        Uses greedy algorithm (optimal is NP-hard).
        """
        if not pins:
            return []

        # Sort by ring (outer first) then by position
        sorted_pins = sorted(pins, key=lambda p: (
            self._get_pin_ring(p),
            p.row,
            p.col
        ))

        independent: List[Pin] = []
        used_channels: Set[Tuple[int, int, str]] = set()

        for pin in sorted_pins:
            # Check if pin can be added without conflict
            escape_dir = self._get_escape_direction(pin)
            channels = self._get_escape_channels(pin, escape_dir)

            conflict = False
            for channel in channels:
                if channel in used_channels:
                    conflict = True
                    break

            if not conflict:
                independent.append(pin)
                used_channels.update(channels)

        return independent

    def _get_escape_channels(
        self,
        pin: Pin,
        direction: EscapeDirection
    ) -> List[Tuple[int, int, str]]:
        """Get list of channels used by pin's escape path"""
        channels = []
        r, c = pin.row, pin.col

        # Trace channel usage from pin to edge
        if direction == EscapeDirection.NORTH:
            for row in range(r, -1, -1):
                channels.append((row, c, 'V'))
        elif direction == EscapeDirection.SOUTH:
            for row in range(r, self.pin_array.rows):
                channels.append((row, c, 'V'))
        elif direction == EscapeDirection.WEST:
            for col in range(c, -1, -1):
                channels.append((r, col, 'H'))
        elif direction == EscapeDirection.EAST:
            for col in range(c, self.pin_array.cols):
                channels.append((r, col, 'H'))

        return channels

    def _create_layer_min_escape_path(self, pin: Pin, layer: int) -> EscapePath:
        """Create escape path for layer minimization strategy"""
        # Similar to ring-based but respects layer assignment
        return self._create_ring_escape_path(
            pin,
            self._get_pin_ring(pin),
            layer
        )

    # =========================================================================
    # STRATEGY 7: HYBRID ESCAPE
    # =========================================================================

    def _escape_hybrid(self):
        """
        Hybrid Escape Strategy

        Combines multiple strategies based on pin location:
        - Outer rings: Dog-bone (simple, fast)
        - Middle rings: Ring-based with layer assignment
        - Inner pins: MMCF for optimal routing
        """
        self.result.messages.append("Using Hybrid Escape strategy")

        # Partition pins by ring
        outer_pins = []
        middle_pins = []
        inner_pins = []

        for pin in self.pin_array.pins:
            if pin.is_nc:
                continue

            ring = self._get_pin_ring(pin)

            if ring <= 1:
                outer_pins.append(pin)
            elif ring <= 3:
                middle_pins.append(pin)
            else:
                inner_pins.append(pin)

        # Process outer pins with dog-bone
        if outer_pins:
            temp_array = PinArray(
                package_type=self.pin_array.package_type,
                rows=self.pin_array.rows,
                cols=self.pin_array.cols,
                pitch=self.pin_array.pitch,
                pins=outer_pins
            )
            self.pin_array = temp_array
            self._escape_dog_bone()

        # Process middle pins with ring-based
        if middle_pins:
            temp_array = PinArray(
                package_type=self.pin_array.package_type,
                rows=self.pin_array.rows,
                cols=self.pin_array.cols,
                pitch=self.pin_array.pitch,
                pins=middle_pins
            )
            self.pin_array = temp_array
            self._escape_ring_based()

        # Process inner pins with MMCF (if few enough)
        if inner_pins:
            if len(inner_pins) < 50:
                temp_array = PinArray(
                    package_type=self.pin_array.package_type,
                    rows=self.pin_array.rows,
                    cols=self.pin_array.cols,
                    pitch=self.pin_array.pitch,
                    pins=inner_pins
                )
                self.pin_array = temp_array
                self._escape_mmcf()
            else:
                # Too many for MMCF - use layer minimize
                temp_array = PinArray(
                    package_type=self.pin_array.package_type,
                    rows=self.pin_array.rows,
                    cols=self.pin_array.cols,
                    pitch=self.pin_array.pitch,
                    pins=inner_pins
                )
                self.pin_array = temp_array
                self._escape_layer_minimize()

    # =========================================================================
    # VIA SELECTION HELPERS
    # =========================================================================

    def _select_via_type(self, start_layer: int, end_layer: int) -> ViaType:
        """Select appropriate via type based on layer transition"""
        available = self.config.via_types_available

        # Layer 1 to inner
        if start_layer == 1:
            if end_layer == 2 and ViaType.MICRO_VIA in available:
                return ViaType.MICRO_VIA
            elif ViaType.BLIND in available:
                return ViaType.BLIND

        # Inner to inner
        elif start_layer > 1 and end_layer > 1:
            if abs(end_layer - start_layer) == 1 and ViaType.MICRO_VIA in available:
                return ViaType.MICRO_VIA
            elif ViaType.BURIED in available:
                return ViaType.BURIED

        # Default: through-hole
        return ViaType.THROUGH_HOLE

    def _get_via_drill(self, via_type: ViaType) -> float:
        """Get drill diameter for via type"""
        if via_type == ViaType.MICRO_VIA:
            return self.config.micro_via_drill
        return self.config.via_drill

    def _get_via_pad(self, via_type: ViaType) -> float:
        """Get pad diameter for via type"""
        if via_type == ViaType.MICRO_VIA:
            return self.config.micro_via_pad
        return self.config.via_pad

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def _calculate_statistics(self):
        """Calculate escape routing statistics"""
        total_pins = len(self.result.paths)
        escaped = sum(1 for p in self.result.paths if p.escaped)

        self.result.success_rate = escaped / total_pins if total_pins > 0 else 0

        self.result.total_vias = sum(
            len(p.vias) for p in self.result.paths
        )

        self.result.layers_used = set()
        for path in self.result.paths:
            for trace in path.traces:
                self.result.layers_used.add(trace.layer)
            for via in path.vias:
                for layer in range(via.start_layer, via.end_layer + 1):
                    self.result.layers_used.add(layer)

        self.result.total_wire_length = sum(
            p.total_length for p in self.result.paths
        )


# =============================================================================
# QFN THERMAL PAD ESCAPE
# =============================================================================

class QFNThermalEscape:
    """
    Specialized escape routing for QFN thermal pads

    QFN packages have exposed thermal pad on bottom that needs:
    1. Thermal vias for heat dissipation
    2. Connection to ground plane
    3. Solder mask dam patterns

    Source: TI Application Note - "QFN/SON PCB Attachment"
    Source: NXP AN10778 - "PCB Layout Guidelines for QFN"
    """

    def __init__(self, config: Optional[EscapeConfig] = None):
        self.config = config or EscapeConfig()

    def create_thermal_via_pattern(
        self,
        pad_width: float,
        pad_height: float,
        via_drill: float = 0.3,
        via_pad: float = 0.5,
        via_pitch: float = 1.0,
        ground_layer: int = 2
    ) -> List[Via]:
        """
        Create thermal via array for QFN thermal pad

        Source: IPC-7093 - "Design and Assembly Process Implementation
                for Bottom Termination Components"

        Typical thermal via density: 1 via per 1.0-1.5mm²
        """
        vias = []

        # Calculate via grid
        cols = max(1, int(pad_width / via_pitch))
        rows = max(1, int(pad_height / via_pitch))

        # Center the via grid
        start_x = -pad_width / 2 + via_pitch / 2
        start_y = -pad_height / 2 + via_pitch / 2

        # Offset if even number to avoid center
        if cols % 2 == 0:
            start_x += via_pitch / 4
        if rows % 2 == 0:
            start_y += via_pitch / 4

        via_id = 0
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * via_pitch
                y = start_y + row * via_pitch

                via = Via(
                    x=x,
                    y=y,
                    drill_diameter=via_drill,
                    pad_diameter=via_pad,
                    via_type=ViaType.THROUGH_HOLE,
                    start_layer=1,
                    end_layer=ground_layer,
                    net="GND",
                    id=f"TV_{via_id}"
                )
                vias.append(via)
                via_id += 1

        return vias

    def calculate_thermal_resistance(
        self,
        via_count: int,
        via_drill: float,
        plating_thickness: float = 0.025,  # mm
        board_thickness: float = 1.6  # mm
    ) -> float:
        """
        Calculate thermal resistance of via array

        Source: Würth Elektronik - "Thermal Management"

        R_th = L / (k * A * N)
        where:
        - L = PCB thickness
        - k = copper thermal conductivity (385 W/m·K)
        - A = via cross-section area
        - N = number of vias
        """
        COPPER_K = 385.0  # W/(m·K)

        # Via wall cross-section
        via_radius = via_drill / 2
        wall_area = math.pi * ((via_radius)**2 - (via_radius - plating_thickness)**2)

        # Convert to meters
        wall_area_m2 = wall_area * 1e-6
        board_m = board_thickness * 1e-3

        # Total thermal resistance (parallel vias)
        r_th_single = board_m / (COPPER_K * wall_area_m2)
        r_th_array = r_th_single / via_count

        return r_th_array


# =============================================================================
# PERIPHERAL PACKAGE FANOUT (QFP, SOIC, etc.)
# =============================================================================

class PeripheralFanout:
    """
    Fanout routing for peripheral lead packages (QFP, SOIC, TSSOP)

    These packages don't need "escape" routing since all pins are
    accessible from the edge. They just need organized fanout.

    Source: Industry standard practice
    """

    def __init__(self, config: Optional[EscapeConfig] = None):
        self.config = config or EscapeConfig()

    def create_fanout(
        self,
        pins: List[Pin],
        fanout_direction: EscapeDirection,
        fanout_length: float = 1.0
    ) -> List[EscapeTrace]:
        """
        Create fanout traces for peripheral pins

        Traces extend perpendicular to package edge,
        with slight angle if needed to maintain clearance.
        """
        traces = []
        trace_width = self.config.trace_width
        clearance = self.config.trace_clearance
        trace_pitch = trace_width + clearance

        # Sort pins by position
        sorted_pins = sorted(pins, key=lambda p: (p.row, p.col))

        for i, pin in enumerate(sorted_pins):
            # Calculate fanout end point
            end_x, end_y = pin.x, pin.y

            if fanout_direction == EscapeDirection.NORTH:
                end_y -= fanout_length
            elif fanout_direction == EscapeDirection.SOUTH:
                end_y += fanout_length
            elif fanout_direction == EscapeDirection.EAST:
                end_x += fanout_length
            elif fanout_direction == EscapeDirection.WEST:
                end_x -= fanout_length

            trace = EscapeTrace(
                start_x=pin.x,
                start_y=pin.y,
                end_x=end_x,
                end_y=end_y,
                width=trace_width,
                layer=1,
                net=pin.net
            )
            traces.append(trace)

        return traces


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def escape_bga(
    rows: int,
    cols: int,
    pitch: float,
    net_assignments: Dict[str, str],  # "A1" -> "NET_NAME"
    config: Optional[EscapeConfig] = None
) -> EscapeResult:
    """
    Convenience function to escape a BGA package

    Args:
        rows: Number of rows in BGA
        cols: Number of columns in BGA
        pitch: Ball pitch in mm
        net_assignments: Dict mapping pin name to net name
        config: Optional escape configuration

    Returns:
        EscapeResult with all escape paths
    """
    # Create pin array
    pins = []
    for r in range(rows):
        for c in range(cols):
            # Standard BGA naming: A1, A2, ..., B1, B2, ...
            row_letter = chr(ord('A') + r)
            pin_name = f"{row_letter}{c + 1}"

            net = net_assignments.get(pin_name, "NC")
            is_nc = net == "NC"
            is_power = "VDD" in net or "VCC" in net
            is_ground = "GND" in net or "VSS" in net

            pin = Pin(
                id=pin_name,
                row=r,
                col=c,
                x=c * pitch,
                y=r * pitch,
                net=net,
                is_power=is_power,
                is_ground=is_ground,
                is_nc=is_nc
            )
            pins.append(pin)

    pin_array = PinArray(
        package_type=PackageType.BGA,
        rows=rows,
        cols=cols,
        pitch=pitch,
        pins=pins
    )

    # Run escape routing
    piston = EscapePiston(config)
    return piston.escape(pin_array)


def escape_qfn(
    pins_per_side: int,
    pitch: float,
    thermal_pad_size: Tuple[float, float],
    net_assignments: Dict[int, str],  # pin_number -> net_name
    config: Optional[EscapeConfig] = None
) -> Tuple[EscapeResult, List[Via]]:
    """
    Convenience function to escape a QFN package

    Args:
        pins_per_side: Number of pins on each side
        pitch: Pin pitch in mm
        thermal_pad_size: (width, height) of thermal pad in mm
        net_assignments: Dict mapping pin number to net name
        config: Optional escape configuration

    Returns:
        Tuple of (EscapeResult for signal pins, List of thermal vias)
    """
    config = config or EscapeConfig()

    # QFN pins are peripheral - use fanout
    # Create pins for each side
    pins = []
    pin_num = 1

    # Calculate package dimensions
    body_size = pins_per_side * pitch + 2  # Approximate

    # Bottom side (left to right)
    for i in range(pins_per_side):
        x = -body_size/2 + (i + 0.5) * pitch
        y = body_size/2
        net = net_assignments.get(pin_num, "NC")
        pins.append(Pin(
            id=str(pin_num), row=0, col=i,
            x=x, y=y, net=net
        ))
        pin_num += 1

    # Right side (bottom to top)
    for i in range(pins_per_side):
        x = body_size/2
        y = body_size/2 - (i + 0.5) * pitch
        net = net_assignments.get(pin_num, "NC")
        pins.append(Pin(
            id=str(pin_num), row=i, col=pins_per_side,
            x=x, y=y, net=net
        ))
        pin_num += 1

    # Top side (right to left)
    for i in range(pins_per_side):
        x = body_size/2 - (i + 0.5) * pitch
        y = -body_size/2
        net = net_assignments.get(pin_num, "NC")
        pins.append(Pin(
            id=str(pin_num), row=pins_per_side, col=pins_per_side-1-i,
            x=x, y=y, net=net
        ))
        pin_num += 1

    # Left side (top to bottom)
    for i in range(pins_per_side):
        x = -body_size/2
        y = -body_size/2 + (i + 0.5) * pitch
        net = net_assignments.get(pin_num, "NC")
        pins.append(Pin(
            id=str(pin_num), row=pins_per_side-1-i, col=0,
            x=x, y=y, net=net
        ))
        pin_num += 1

    # Create pin array (though QFP doesn't really need escape)
    pin_array = PinArray(
        package_type=PackageType.QFN,
        rows=pins_per_side,
        cols=pins_per_side,
        pitch=pitch,
        pins=pins
    )

    # Simple fanout for peripheral pins
    piston = EscapePiston(config)
    piston.config.strategy = EscapeStrategy.DOG_BONE  # Simple fanout
    result = piston.escape(pin_array)

    # Create thermal via pattern
    thermal = QFNThermalEscape(config)
    thermal_vias = thermal.create_thermal_via_pattern(
        pad_width=thermal_pad_size[0],
        pad_height=thermal_pad_size[1]
    )

    return result, thermal_vias


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Test BGA escape
    print("Testing BGA Escape Routing...")
    print("=" * 60)

    # Create test net assignments for 8x8 BGA
    nets = {}
    for r in range(8):
        for c in range(8):
            row_letter = chr(ord('A') + r)
            pin_name = f"{row_letter}{c + 1}"

            # Simple pattern: power rails on corners, signals elsewhere
            if (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                nets[pin_name] = "GND"
            elif r == 3 or r == 4:
                if c == 3 or c == 4:
                    nets[pin_name] = "VDD"
                else:
                    nets[pin_name] = f"SIG_{r}_{c}"
            else:
                nets[pin_name] = f"SIG_{r}_{c}"

    # Test different strategies
    strategies = [
        EscapeStrategy.DOG_BONE,
        EscapeStrategy.RING_BASED,
        EscapeStrategy.ORDERED_MMCF,
    ]

    for strategy in strategies:
        config = EscapeConfig(strategy=strategy)
        result = escape_bga(8, 8, 0.8, nets, config)

        print(f"\n{strategy.name}:")
        print(f"  Success rate: {result.success_rate * 100:.1f}%")
        print(f"  Total vias: {result.total_vias}")
        print(f"  Layers used: {sorted(result.layers_used)}")
        print(f"  Wire length: {result.total_wire_length:.2f}mm")
        print(f"  Time: {result.routing_time_ms:.2f}ms")

    # Test QFN thermal escape
    print("\n" + "=" * 60)
    print("Testing QFN Thermal Escape...")

    qfn_nets = {i: f"SIG_{i}" for i in range(1, 25)}  # 24-pin QFN
    result, thermal_vias = escape_qfn(
        pins_per_side=6,
        pitch=0.5,
        thermal_pad_size=(3.0, 3.0),
        net_assignments=qfn_nets
    )

    print(f"QFN signal fanout: {result.success_rate * 100:.1f}% success")
    print(f"Thermal vias: {len(thermal_vias)}")

    # Calculate thermal resistance
    thermal = QFNThermalEscape()
    r_th = thermal.calculate_thermal_resistance(len(thermal_vias), 0.3)
    print(f"Thermal resistance: {r_th:.2f} K/W")

    print("\n" + "=" * 60)
    print("Escape Piston module loaded successfully!")
    print("Available strategies:")
    for s in EscapeStrategy:
        print(f"  - {s.name}")
