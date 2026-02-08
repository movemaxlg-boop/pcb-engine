"""
PCB Engine - Escape Module (Enhanced)
======================================

Calculates escape vectors for multi-pin components.
This is Phase 4 of the algorithm.

CRITICAL LESSON LEARNED:
========================
The previous PCB design had escape direction = EAST (+1, 0)
But destinations (sensors) were SOUTH of the hub (larger Y values).
This created U-turns: traces went EAST, then had to turn SOUTH+WEST.
16 traces competing for the same U-turn corridor = routing failure.

THE FIX: Escape direction MUST point toward destination centroid.
If destinations are SOUTH, escapes go SOUTH. No U-turns, no congestion.

KEY PRINCIPLE:
==============
ESCAPE DIRECTION = DIRECTION_TO_DESTINATION_CENTROID
Not natural pin side, not perpendicular to header - TOWARD DESTINATIONS.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EscapeVector:
    """Complete escape route specification for a pin"""
    pin: str
    direction: Tuple[float, float]  # Unit vector (dx, dy)
    length: float                    # mm
    endpoint: Tuple[float, float]    # (x, y) of escape endpoint

    # Extended fields for the enhanced version
    pin_name: str = ''
    net: str = ''
    start: Tuple[float, float] = (0, 0)  # Pad center
    layer: str = 'F.Cu'
    width: float = 0.25
    direction_name: str = ''  # 'N', 'S', 'E', 'W', etc.
    dest_centroid: Optional[Tuple[float, float]] = None


@dataclass
class ComponentEscapes:
    """All escapes for a component"""
    ref: str
    escapes: Dict[str, EscapeVector] = field(default_factory=dict)
    destination_centroid: Optional[Tuple[float, float]] = None
    primary_direction: str = ''
    pin_count: int = 0

    def get_endpoints(self) -> List[Tuple[float, float]]:
        """Get all escape endpoints"""
        return [e.endpoint for e in self.escapes.values()]


@dataclass
class DestinationAnalysis:
    """Analysis of where a component's connections go"""
    component: str
    destinations: List[Dict]
    centroid: Tuple[float, float]
    direction_vector: Tuple[float, float]
    direction_name: str
    total_connections: int


# =============================================================================
# DIRECTION UTILITIES
# =============================================================================

class DirectionUtils:
    """Utility functions for direction calculations"""

    DIRECTIONS = {
        'E':  (1.0, 0.0),
        'SE': (0.707, 0.707),
        'S':  (0.0, 1.0),
        'SW': (-0.707, 0.707),
        'W':  (-1.0, 0.0),
        'NW': (-0.707, -0.707),
        'N':  (0.0, -1.0),
        'NE': (0.707, -0.707),
    }

    DIRECTION_ANGLES = {
        0: 'E', 45: 'SE', 90: 'S', 135: 'SW',
        180: 'W', 225: 'NW', 270: 'N', 315: 'NE',
    }

    @staticmethod
    def vector_to_direction(dx: float, dy: float) -> Tuple[Tuple[float, float], str]:
        """Convert vector to quantized 45-degree direction"""
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return ((0, 1), 'S')

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360

        quantized = round(angle_deg / 45) * 45
        if quantized >= 360:
            quantized = 0

        direction_name = DirectionUtils.DIRECTION_ANGLES.get(quantized, 'S')
        direction_vector = DirectionUtils.DIRECTIONS[direction_name]

        return (direction_vector, direction_name)

    @staticmethod
    def direction_from_name(name: str) -> Tuple[float, float]:
        """Get direction vector from name"""
        return DirectionUtils.DIRECTIONS.get(name, (0, 1))

    @staticmethod
    def natural_escape_for_side(side: str) -> str:
        """Get natural escape direction for a pin side"""
        return {'left': 'W', 'right': 'E', 'top': 'N', 'bottom': 'S'}.get(side, 'S')


# =============================================================================
# DESTINATION ANALYZER
# =============================================================================

class DestinationAnalyzer:
    """Analyzes where each component's connections go"""

    def __init__(self, parts_db: Dict, placement: Dict, graph: Dict):
        self.parts_db = parts_db
        self.placement = placement
        self.graph = graph
        self.nets = parts_db.get('nets', {})

    def analyze(self, ref: str) -> DestinationAnalysis:
        """Analyze all destinations for a component"""
        if ref not in self.placement:
            return DestinationAnalysis(ref, [], (0, 0), (0, 1), 'S', 0)

        pos = self.placement[ref]
        part = self.parts_db['parts'].get(ref, {})
        destinations = []

        for pin_info in part.get('used_pins', []):
            net_name = pin_info.get('net')
            if not net_name or net_name == 'GND':
                continue

            net_info = self.nets.get(net_name, {})
            for other_ref, other_pin in net_info.get('pins', []):
                if other_ref == ref:
                    continue
                if other_ref in self.placement:
                    other_pos = self.placement[other_ref]
                    destinations.append({
                        'ref': other_ref,
                        'pin': other_pin,
                        'position': (other_pos.x, other_pos.y),
                        'net': net_name,
                        'weight': 1,
                    })

        if not destinations:
            return DestinationAnalysis(ref, [], (pos.x, pos.y), (0, 1), 'S', 0)

        total_weight = sum(d['weight'] for d in destinations)
        cx = sum(d['position'][0] * d['weight'] for d in destinations) / total_weight
        cy = sum(d['position'][1] * d['weight'] for d in destinations) / total_weight

        dx = cx - pos.x
        dy = cy - pos.y
        direction_vector, direction_name = DirectionUtils.vector_to_direction(dx, dy)

        return DestinationAnalysis(
            ref, destinations, (cx, cy),
            direction_vector, direction_name, len(destinations)
        )

    def analyze_pin_group(self, ref: str, pins: List[Dict]) -> DestinationAnalysis:
        """Analyze destinations for a specific group of pins"""
        if ref not in self.placement:
            return DestinationAnalysis(ref, [], (0, 0), (0, 1), 'S', 0)

        pos = self.placement[ref]
        destinations = []

        for pin_info in pins:
            net_name = pin_info.get('net')
            if not net_name or net_name == 'GND':
                continue

            net_info = self.nets.get(net_name, {})
            for other_ref, other_pin in net_info.get('pins', []):
                if other_ref == ref:
                    continue
                if other_ref in self.placement:
                    other_pos = self.placement[other_ref]
                    destinations.append({
                        'ref': other_ref,
                        'pin': other_pin,
                        'position': (other_pos.x, other_pos.y),
                        'net': net_name,
                        'weight': 1,
                    })

        if not destinations:
            return DestinationAnalysis(ref, [], (pos.x, pos.y), (0, 1), 'S', 0)

        total_weight = sum(d['weight'] for d in destinations)
        cx = sum(d['position'][0] * d['weight'] for d in destinations) / total_weight
        cy = sum(d['position'][1] * d['weight'] for d in destinations) / total_weight

        # Use pin group center
        if pins:
            group_cx = sum(p.get('offset', (0, 0))[0] for p in pins) / len(pins)
            group_cy = sum(p.get('offset', (0, 0))[1] for p in pins) / len(pins)
            origin_x = pos.x + group_cx
            origin_y = pos.y + group_cy
        else:
            origin_x, origin_y = pos.x, pos.y

        dx = cx - origin_x
        dy = cy - origin_y
        direction_vector, direction_name = DirectionUtils.vector_to_direction(dx, dy)

        return DestinationAnalysis(
            ref, destinations, (cx, cy),
            direction_vector, direction_name, len(destinations)
        )


# =============================================================================
# ESCAPE LENGTH CALCULATOR
# =============================================================================

class EscapeLengthCalculator:
    """
    Calculates escape lengths for staggered fan-out.

    PRINCIPLE: Inner pins get LONGER escapes so outer pins can pass.
    """

    def __init__(self, base_length: float = 2.0, increment: float = 0.5,
                 max_length: float = 10.0, grid_size: float = 0.5):
        self.base_length = base_length
        self.increment = increment
        self.max_length = max_length
        self.grid_size = grid_size

    def calculate(self, pin_index: int, total_pins: int) -> float:
        """Calculate escape length for a pin"""
        if total_pins <= 1:
            return self.base_length

        distance_from_end = min(pin_index, total_pins - 1 - pin_index)
        length = self.base_length + distance_from_end * self.increment
        length = min(length, self.max_length)
        length = round(length / self.grid_size) * self.grid_size

        return length


# =============================================================================
# MAIN ESCAPE CALCULATOR
# =============================================================================

class EscapeCalculator:
    """
    Main escape calculator.

    ALGORITHM:
    1. For each multi-pin component:
       a. Analyze where its destinations are
       b. Group pins by physical side
       c. For each pin group:
          - Calculate direction to destination centroid
          - Use THAT as escape direction (not natural side direction)
          - Calculate staggered lengths for fan-out
       d. Generate escape vectors
    """

    def __init__(self, board, rules, base_escape: float = 2.0,
                 escape_increment: float = 0.5):
        self.board = board
        self.rules = rules
        self.escapes = {}

        self.length_calc = EscapeLengthCalculator(
            base_length=base_escape,
            increment=escape_increment,
            grid_size=board.grid_size
        )

    def calculate(self, parts_db: Dict, placement: Dict, graph: Dict) -> Dict:
        """
        Calculate escapes for ALL components with used pins.

        Previously only calculated escapes for components with >2 pins,
        but 2-pin components (resistors, LEDs, capacitors) also need escapes
        to properly connect routes to pads via grid-aligned endpoints.

        Returns:
            {ref: {pin_number: EscapeVector, ...}, ...}
        """
        self.escapes = {}
        analyzer = DestinationAnalyzer(parts_db, placement, graph)
        parts = parts_db['parts']

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            used_pins = part.get('used_pins', [])

            if len(used_pins) < 1:
                continue  # Skip components with no pins (shouldn't happen)

            dest_analysis = analyzer.analyze(ref)
            comp_escapes = self._calculate_component_escapes(
                ref, part, pos, dest_analysis, analyzer
            )

            if comp_escapes:
                self.escapes[ref] = comp_escapes

        return self.escapes

    def _calculate_component_escapes(self, ref: str, part: Dict, pos,
                                      dest_analysis: DestinationAnalysis,
                                      analyzer: DestinationAnalyzer) -> Dict[str, EscapeVector]:
        """Calculate escapes for a single component"""
        escapes = {}
        used_pins = part.get('used_pins', [])

        # For 2-pin components (resistors, capacitors, LEDs), use unified escape direction
        # Both pins should escape PERPENDICULAR to the component axis for easier routing
        if len(used_pins) == 2:
            return self._calculate_two_pin_escapes(ref, part, pos, used_pins, dest_analysis)

        # For multi-pin components, use side-based escapes
        pin_groups = self._group_pins_by_side(used_pins)

        for side, pins in pin_groups.items():
            if not pins:
                continue

            # Analyze destinations specifically for this pin group
            group_analysis = analyzer.analyze_pin_group(ref, pins)

            # Escape direction is ALWAYS away from component body (natural direction)
            natural = DirectionUtils.natural_escape_for_side(side)
            escape_direction = DirectionUtils.direction_from_name(natural)
            direction_name = natural

            pins_sorted = self._sort_pins_for_escape(pins, escape_direction)

            for i, pin_info in enumerate(pins_sorted):
                escape_length = self.length_calc.calculate(i, len(pins_sorted))

                offset = pin_info.get('offset', (0, 0))
                pin_x = pos.x + offset[0]
                pin_y = pos.y + offset[1]

                end_x = pin_x + escape_direction[0] * escape_length
                end_y = pin_y + escape_direction[1] * escape_length

                end_x = round(end_x / self.board.grid_size) * self.board.grid_size
                end_y = round(end_y / self.board.grid_size) * self.board.grid_size

                escapes[pin_info['number']] = EscapeVector(
                    pin=pin_info['number'],
                    direction=escape_direction,
                    length=escape_length,
                    endpoint=(end_x, end_y),
                    pin_name=pin_info.get('name', ''),
                    net=pin_info.get('net', ''),
                    start=(pin_x, pin_y),
                    layer='F.Cu',
                    width=self.rules.min_trace_width,
                    direction_name=direction_name,
                    dest_centroid=group_analysis.centroid,
                )

        return escapes

    def _calculate_two_pin_escapes(self, ref: str, part: Dict, pos,
                                    used_pins: List[Dict],
                                    dest_analysis: DestinationAnalysis) -> Dict[str, EscapeVector]:
        """
        Calculate escapes for 2-pin components (resistors, capacitors, LEDs).

        For 2-pin components:
        - Pins are typically on opposite sides (left/right or top/bottom)
        - Each pin escapes TOWARD its specific destination
        - If pin has no destination or destination is along component axis,
          escape perpendicular to component axis

        Example: Horizontal resistor R1 with pins connecting to U1 (north) and D1 (south)
        - R1.1 (connects to U1) escapes NORTH toward U1
        - R1.2 (connects to D1) escapes SOUTH toward D1
        """
        escapes = {}

        # Determine component axis from pin positions
        pin0 = used_pins[0]
        pin1 = used_pins[1]
        off0 = pin0.get('offset', (0, 0))
        off1 = pin1.get('offset', (0, 0))

        dx = off1[0] - off0[0]
        dy = off1[1] - off0[1]

        # Component is horizontal if pins are spread in X, vertical if in Y
        is_horizontal = abs(dx) > abs(dy)

        # Build map of net -> destination position from dest_analysis
        net_to_dest = {}
        if dest_analysis and dest_analysis.destinations:
            for dest in dest_analysis.destinations:
                net = dest.get('net', '')
                dest_pos = dest.get('position', None)
                if net and dest_pos:
                    net_to_dest[net] = dest_pos

        escape_length = self.length_calc.calculate(0, 1)  # Short escape

        for pin_info in used_pins:
            offset = pin_info.get('offset', (0, 0))
            pin_x = pos.x + offset[0]
            pin_y = pos.y + offset[1]
            net = pin_info.get('net', '')

            # Determine escape direction for this specific pin
            dest_pos = net_to_dest.get(net)

            if dest_pos:
                # Calculate direction toward destination
                dest_dx = dest_pos[0] - pin_x
                dest_dy = dest_pos[1] - pin_y

                if is_horizontal:
                    # For horizontal component, prefer vertical escape (N/S)
                    if abs(dest_dy) > 0.1:  # Destination has meaningful Y offset
                        if dest_dy > 0:
                            escape_direction = (0.0, 1.0)  # South
                            direction_name = 'S'
                        else:
                            escape_direction = (0.0, -1.0)  # North
                            direction_name = 'N'
                    else:
                        # Destination is along component axis, use default
                        escape_direction = (0.0, 1.0)  # Default South
                        direction_name = 'S'
                else:
                    # For vertical component, prefer horizontal escape (E/W)
                    if abs(dest_dx) > 0.1:
                        if dest_dx > 0:
                            escape_direction = (1.0, 0.0)  # East
                            direction_name = 'E'
                        else:
                            escape_direction = (-1.0, 0.0)  # West
                            direction_name = 'W'
                    else:
                        escape_direction = (1.0, 0.0)  # Default East
                        direction_name = 'E'
            else:
                # No destination found, use default perpendicular escape
                if is_horizontal:
                    escape_direction = (0.0, 1.0)  # South
                    direction_name = 'S'
                else:
                    escape_direction = (1.0, 0.0)  # East
                    direction_name = 'E'

            end_x = pin_x + escape_direction[0] * escape_length
            end_y = pin_y + escape_direction[1] * escape_length

            end_x = round(end_x / self.board.grid_size) * self.board.grid_size
            end_y = round(end_y / self.board.grid_size) * self.board.grid_size

            escapes[pin_info['number']] = EscapeVector(
                pin=pin_info['number'],
                direction=escape_direction,
                length=escape_length,
                endpoint=(end_x, end_y),
                pin_name=pin_info.get('name', ''),
                net=pin_info.get('net', ''),
                start=(pin_x, pin_y),
                layer='F.Cu',
                width=self.rules.min_trace_width,
                direction_name=direction_name,
                dest_centroid=dest_pos,
            )

        return escapes

    def _group_pins_by_side(self, pins: List[Dict]) -> Dict[str, List]:
        """Group pins by which side of component they're on"""
        groups = {'left': [], 'right': [], 'top': [], 'bottom': []}

        for pin in pins:
            offset = pin.get('offset', (0, 0))
            dx, dy = offset

            if abs(dx) > abs(dy):
                side = 'right' if dx > 0 else 'left'
            else:
                side = 'bottom' if dy > 0 else 'top'

            groups[side].append(pin)

        return groups

    def _sort_pins_for_escape(self, pins: List[Dict],
                               escape_direction: Tuple[float, float]) -> List[Dict]:
        """Sort pins for escape length calculation"""
        if not pins:
            return pins

        dx, dy = escape_direction

        if abs(dx) > abs(dy):
            key = lambda p: p.get('offset', (0, 0))[1]
        else:
            key = lambda p: p.get('offset', (0, 0))[0]

        return sorted(pins, key=key)

    def get_escape_report(self) -> str:
        """Generate a human-readable escape report"""
        lines = ["=" * 60, "ESCAPE CALCULATION REPORT", "=" * 60]

        for ref, pin_escapes in self.escapes.items():
            lines.append(f"\nComponent: {ref}")

            for pin_num, escape in pin_escapes.items():
                lines.append(
                    f"  Pin {pin_num} ({escape.net}): "
                    f"{escape.direction_name} {escape.length:.1f}mm -> "
                    f"({escape.endpoint[0]:.1f}, {escape.endpoint[1]:.1f})"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# VALIDATION
# =============================================================================

class EscapeValidator:
    """Validates escape configurations"""

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules

    def validate(self, escapes: Dict[str, Dict[str, EscapeVector]]) -> Dict:
        """Validate all escapes"""
        violations = []
        warnings = []

        for ref, pin_escapes in escapes.items():
            for pin_num, escape in pin_escapes.items():
                ex, ey = escape.endpoint

                if ex < self.board.origin_x or ex > self.board.origin_x + self.board.width:
                    violations.append(f"{ref}.{pin_num}: Escape endpoint outside board (X={ex})")

                if ey < self.board.origin_y or ey > self.board.origin_y + self.board.height:
                    violations.append(f"{ref}.{pin_num}: Escape endpoint outside board (Y={ey})")

            # Check for escape collisions
            endpoints = list(pin_escapes.values())
            for i, e1 in enumerate(endpoints):
                for e2 in endpoints[i+1:]:
                    if self._escapes_collide(e1, e2):
                        warnings.append(
                            f"{ref}: Escape collision between {e1.pin} and {e2.pin}"
                        )

        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
        }

    def _escapes_collide(self, e1: EscapeVector, e2: EscapeVector) -> bool:
        """Check if two escapes might collide"""
        dx = e1.endpoint[0] - e2.endpoint[0]
        dy = e1.endpoint[1] - e2.endpoint[1]
        dist = math.sqrt(dx*dx + dy*dy)
        min_spacing = self.rules.min_trace_width + 2 * self.rules.min_clearance
        return dist < min_spacing
