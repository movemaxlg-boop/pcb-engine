#!/usr/bin/env python3
"""
Design Advisor Piston - PCB Layout Intelligence

Encodes human PCB design knowledge as rules that help the routing algorithms.
Works as a "team member" with the routing piston to achieve better results.

Knowledge encoded:
1. Signal flow direction (input → processing → output)
2. Power distribution strategy (star vs bus)
3. GND plane usage decisions
4. Layer assignment heuristics
5. Net priority ordering
6. Congestion prediction
7. Via placement strategy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import math


class NetType(Enum):
    """Classification of nets by function"""
    POWER = "power"           # VIN, VCC, 5V, 3V3, etc.
    GROUND = "ground"         # GND, AGND, DGND, etc.
    SIGNAL = "signal"         # Regular signals
    HIGH_SPEED = "high_speed" # Clock, USB, etc.
    ANALOG = "analog"         # Sensitive analog signals
    DIFFERENTIAL = "diff"     # Differential pairs


@dataclass
class NetAdvice:
    """Routing advice for a specific net"""
    net_name: str
    net_type: NetType
    priority: int                    # Lower = route first
    preferred_layer: str             # F.Cu or B.Cu
    min_width: float                 # Minimum trace width
    max_length: float                # Maximum allowed length (0 = no limit)
    avoid_vias: bool                 # Try to stay on one layer
    route_strategy: str              # 'direct', 'star', 'bus', 'daisy'
    notes: str                       # Human-readable advice


@dataclass
class PlacementAdvice:
    """Advice for component placement"""
    ref: str
    preferred_region: str            # 'left', 'center', 'right', 'top', 'bottom'
    group_with: List[str]            # Components to place nearby
    orientation: float               # Suggested rotation
    notes: str


@dataclass
class RoutingStrategy:
    """Overall routing strategy for the design"""
    use_gnd_plane: bool              # Should GND use a plane/pour?
    gnd_plane_layer: str             # Which layer for GND plane
    power_layer: str                 # Preferred layer for power
    signal_layer: str                # Preferred layer for signals
    net_order: List[str]             # Order to route nets
    congestion_zones: List[Tuple[float, float, float, float]]  # (x, y, w, h)
    notes: str


class DesignAdvisorPiston:
    """
    Provides intelligent design advice to help routing algorithms.

    This piston analyzes the parts_db and placement to generate:
    1. Net classification and routing priorities
    2. Layer assignment recommendations
    3. Routing order optimization
    4. Congestion prediction
    5. Via placement hints
    """

    # Keywords for net classification
    POWER_KEYWORDS = {'vin', 'vcc', 'vdd', '5v', '3v3', '3.3v', '12v', 'vbat',
                      'vsys', 'vbus', 'vout', 'vreg', 'pwr', 'power', 'v+'}
    GROUND_KEYWORDS = {'gnd', 'ground', 'vss', 'agnd', 'dgnd', 'pgnd', 'gnd1',
                       'gnd2', 'earth', '0v', 'v-'}
    HIGH_SPEED_KEYWORDS = {'clk', 'clock', 'usb', 'spi', 'mosi', 'miso', 'sck',
                           'i2c', 'scl', 'sda', 'uart', 'tx', 'rx', 'data'}
    ANALOG_KEYWORDS = {'adc', 'dac', 'ain', 'aout', 'ref', 'vref', 'sense'}

    def __init__(self):
        self.parts_db = None
        self.placement = None
        self.board_width = 0
        self.board_height = 0

    def analyze(self, parts_db: Dict, placement: Dict,
                board_width: float, board_height: float) -> RoutingStrategy:
        """
        Analyze the design and generate routing strategy.

        Args:
            parts_db: Parts database with components and nets
            placement: Component placements {ref: (x, y) or Position}
            board_width: Board width in mm
            board_height: Board height in mm

        Returns:
            RoutingStrategy with recommendations
        """
        self.parts_db = parts_db
        self.placement = placement
        self.board_width = board_width
        self.board_height = board_height

        # Classify all nets
        net_advice = self._classify_nets()

        # Decide on GND strategy
        gnd_strategy = self._decide_gnd_strategy(net_advice)

        # Calculate routing order
        net_order = self._calculate_net_order(net_advice)

        # Predict congestion zones
        congestion = self._predict_congestion()

        # Build notes
        notes = self._generate_notes(net_advice, gnd_strategy)

        return RoutingStrategy(
            use_gnd_plane=gnd_strategy['use_plane'],
            gnd_plane_layer=gnd_strategy['plane_layer'],
            power_layer='F.Cu',
            signal_layer='F.Cu',
            net_order=net_order,
            congestion_zones=congestion,
            notes=notes
        )

    def get_net_advice(self, net_name: str) -> NetAdvice:
        """Get routing advice for a specific net."""
        net_type = self._classify_net_type(net_name)

        if net_type == NetType.GROUND:
            return NetAdvice(
                net_name=net_name,
                net_type=net_type,
                priority=1,  # Route GND first (or use plane)
                preferred_layer='B.Cu',  # GND on bottom
                min_width=0.4,
                max_length=0,
                avoid_vias=False,  # Vias to plane are OK
                route_strategy='plane',  # Use ground plane
                notes="Use ground plane on B.Cu, connect with vias"
            )
        elif net_type == NetType.POWER:
            return NetAdvice(
                net_name=net_name,
                net_type=net_type,
                priority=2,  # Route power early
                preferred_layer='F.Cu',
                min_width=0.5,  # Wider for power
                max_length=0,
                avoid_vias=True,  # Keep power on one layer
                route_strategy='star',  # Star from source
                notes="Use wider traces, star distribution from regulator"
            )
        elif net_type == NetType.HIGH_SPEED:
            return NetAdvice(
                net_name=net_name,
                net_type=net_type,
                priority=3,
                preferred_layer='F.Cu',
                min_width=0.25,
                max_length=50.0,  # Length limit
                avoid_vias=True,  # Minimize discontinuities
                route_strategy='direct',
                notes="Keep short and direct, minimize vias"
            )
        else:  # Regular signal
            return NetAdvice(
                net_name=net_name,
                net_type=net_type,
                priority=10,
                preferred_layer='F.Cu',
                min_width=0.25,
                max_length=0,
                avoid_vias=False,
                route_strategy='direct',
                notes="Standard signal routing"
            )

    def _classify_net_type(self, net_name: str) -> NetType:
        """Classify a net by its name."""
        name_lower = net_name.lower()

        # Check for ground
        for kw in self.GROUND_KEYWORDS:
            if kw in name_lower or name_lower == kw:
                return NetType.GROUND

        # Check for power
        for kw in self.POWER_KEYWORDS:
            if kw in name_lower or name_lower == kw:
                return NetType.POWER

        # Check for high-speed
        for kw in self.HIGH_SPEED_KEYWORDS:
            if kw in name_lower:
                return NetType.HIGH_SPEED

        # Check for analog
        for kw in self.ANALOG_KEYWORDS:
            if kw in name_lower:
                return NetType.ANALOG

        return NetType.SIGNAL

    def _classify_nets(self) -> Dict[str, NetAdvice]:
        """Classify all nets in the design."""
        nets = self.parts_db.get('nets', {})
        advice = {}

        for net_name in nets:
            advice[net_name] = self.get_net_advice(net_name)

        return advice

    def _decide_gnd_strategy(self, net_advice: Dict[str, NetAdvice]) -> Dict:
        """Decide whether to use a ground plane."""
        nets = self.parts_db.get('nets', {})

        # Count GND connections
        gnd_pins = 0
        for net_name, advice in net_advice.items():
            if advice.net_type == NetType.GROUND:
                pins = nets.get(net_name, {}).get('pins', [])
                gnd_pins += len(pins)

        # Heuristics for GND plane decision
        use_plane = False
        reasons = []

        # Rule 1: Many GND pins (>4) benefit from plane
        if gnd_pins > 4:
            use_plane = True
            reasons.append(f"{gnd_pins} GND pins - plane is more efficient")

        # Rule 2: 2-layer board almost always benefits from GND plane
        # (Assumes 2-layer since we only have F.Cu and B.Cu)
        use_plane = True
        reasons.append("2-layer board - GND plane on bottom is best practice")

        return {
            'use_plane': use_plane,
            'plane_layer': 'B.Cu',
            'reasons': reasons
        }

    def _calculate_net_order(self, net_advice: Dict[str, NetAdvice]) -> List[str]:
        """
        Calculate optimal net routing order.

        Priority rules:
        1. GND first (if not using plane)
        2. Power nets second
        3. High-speed/critical signals
        4. Nets with fewer pins (easier to route)
        5. Longer nets (need more room)
        6. Everything else
        """
        nets = self.parts_db.get('nets', {})

        def net_sort_key(net_name: str) -> Tuple:
            advice = net_advice.get(net_name, self.get_net_advice(net_name))
            pins = nets.get(net_name, {}).get('pins', [])
            num_pins = len(pins)

            # Calculate net span (how spread out the pins are)
            span = self._calculate_net_span(net_name)

            return (
                advice.priority,      # Primary: priority from classification
                -span,                # Secondary: longer nets first (need room)
                num_pins,             # Tertiary: fewer pins first
            )

        net_names = list(nets.keys())
        net_names.sort(key=net_sort_key)

        return net_names

    def _calculate_net_span(self, net_name: str) -> float:
        """Calculate the span (max distance) of a net's pins."""
        nets = self.parts_db.get('nets', {})
        parts = self.parts_db.get('parts', {})
        pins = nets.get(net_name, {}).get('pins', [])

        if len(pins) < 2:
            return 0

        positions = []
        for pin_info in pins:
            if isinstance(pin_info, tuple):
                ref, pin = pin_info
            else:
                ref = pin_info.get('ref', '')
                pin = pin_info.get('pin', '')

            # Get component position
            comp_pos = self.placement.get(ref)
            if comp_pos is None:
                continue

            # Get x, y from position (handle both tuple and object)
            if hasattr(comp_pos, 'x'):
                cx, cy = comp_pos.x, comp_pos.y
            else:
                cx, cy = comp_pos[0], comp_pos[1]

            # Get pin offset
            part_info = parts.get(ref, {})
            part_pins = part_info.get('pins', [])
            offset = (0, 0)
            for p in part_pins:
                if str(p.get('number')) == str(pin):
                    offset = p.get('offset', (0, 0))
                    break

            positions.append((cx + offset[0], cy + offset[1]))

        if len(positions) < 2:
            return 0

        # Calculate maximum distance between any two pins
        max_dist = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                max_dist = max(max_dist, dist)

        return max_dist

    def _predict_congestion(self) -> List[Tuple[float, float, float, float]]:
        """Predict areas likely to have routing congestion."""
        congestion_zones = []

        # Find areas with many components close together
        positions = []
        for ref, pos in self.placement.items():
            if hasattr(pos, 'x'):
                positions.append((pos.x, pos.y, ref))
            else:
                positions.append((pos[0], pos[1], ref))

        # Check each component for nearby neighbors
        for i, (x1, y1, ref1) in enumerate(positions):
            nearby_count = 0
            for j, (x2, y2, ref2) in enumerate(positions):
                if i != j:
                    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < 8.0:  # Components within 8mm
                        nearby_count += 1

            # If 3+ components nearby, mark as congestion zone
            if nearby_count >= 2:
                congestion_zones.append((x1 - 5, y1 - 5, 10, 10))

        return congestion_zones

    def _generate_notes(self, net_advice: Dict[str, NetAdvice],
                        gnd_strategy: Dict) -> str:
        """Generate human-readable routing notes."""
        notes = []

        notes.append("=== DESIGN ADVISOR RECOMMENDATIONS ===")
        notes.append("")

        # GND strategy
        if gnd_strategy['use_plane']:
            notes.append(f"GND STRATEGY: Use ground plane on {gnd_strategy['plane_layer']}")
            for reason in gnd_strategy.get('reasons', []):
                notes.append(f"  - {reason}")
        else:
            notes.append("GND STRATEGY: Route GND as traces")

        notes.append("")
        notes.append("NET CLASSIFICATION:")

        # Group nets by type
        by_type = {}
        for name, advice in net_advice.items():
            t = advice.net_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(name)

        for net_type, net_names in by_type.items():
            notes.append(f"  {net_type.upper()}: {', '.join(net_names)}")

        return '\n'.join(notes)

    def suggest_layer(self, net_name: str,
                      from_pos: Tuple[float, float],
                      to_pos: Tuple[float, float]) -> str:
        """
        Suggest which layer to use for a route segment.

        Args:
            net_name: Name of the net
            from_pos: Start position (x, y)
            to_pos: End position (x, y)

        Returns:
            'F.Cu' or 'B.Cu'
        """
        net_type = self._classify_net_type(net_name)

        # GND goes on back
        if net_type == NetType.GROUND:
            return 'B.Cu'

        # Power prefers front (wider traces, easier inspection)
        if net_type == NetType.POWER:
            return 'F.Cu'

        # High-speed prefers front (shorter vias to GND)
        if net_type == NetType.HIGH_SPEED:
            return 'F.Cu'

        # Default to front
        return 'F.Cu'

    def should_use_via(self, net_name: str,
                       current_layer: str,
                       obstacle_ahead: bool) -> Tuple[bool, str]:
        """
        Decide if we should use a via to change layers.

        Args:
            net_name: Name of the net
            current_layer: Current routing layer
            obstacle_ahead: Is there an obstacle blocking the path?

        Returns:
            (should_via, reason)
        """
        net_type = self._classify_net_type(net_name)

        # GND: always OK to via (connects to plane)
        if net_type == NetType.GROUND:
            if obstacle_ahead:
                return True, "Via to GND plane to avoid obstacle"
            return False, "No need"

        # Power: avoid vias if possible
        if net_type == NetType.POWER:
            if obstacle_ahead:
                return True, "Via to avoid obstacle (necessary)"
            return False, "Keep power on single layer"

        # High-speed: strongly avoid vias
        if net_type == NetType.HIGH_SPEED:
            if obstacle_ahead:
                return True, "Via only because obstacle (minimize these)"
            return False, "High-speed: minimize vias"

        # Regular signals: via if needed
        if obstacle_ahead:
            return True, "Via to route around obstacle"
        return False, "No obstacle"


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def get_routing_advice(parts_db: Dict, placement: Dict,
                       board_width: float, board_height: float) -> RoutingStrategy:
    """
    Convenience function to get routing advice.

    Usage:
        strategy = get_routing_advice(parts_db, placement, 50.0, 35.0)
        print(strategy.notes)
        print(f"Route order: {strategy.net_order}")
    """
    advisor = DesignAdvisorPiston()
    return advisor.analyze(parts_db, placement, board_width, board_height)


if __name__ == '__main__':
    # Test with sample data
    test_parts_db = {
        'parts': {
            'U1': {'pins': [
                {'number': '1', 'net': 'VIN'},
                {'number': '2', 'net': 'GND'},
                {'number': '3', 'net': '5V'},
            ]},
            'C1': {'pins': [
                {'number': '1', 'net': 'VIN'},
                {'number': '2', 'net': 'GND'},
            ]},
            'R1': {'pins': [
                {'number': '1', 'net': '5V'},
                {'number': '2', 'net': 'LED'},
            ]},
            'D1': {'pins': [
                {'number': '1', 'net': 'LED'},
                {'number': '2', 'net': 'GND'},
            ]},
        },
        'nets': {
            'VIN': {'pins': [('U1', '1'), ('C1', '1')]},
            'GND': {'pins': [('U1', '2'), ('C1', '2'), ('D1', '2')]},
            '5V': {'pins': [('U1', '3'), ('R1', '1')]},
            'LED': {'pins': [('R1', '2'), ('D1', '1')]},
        }
    }

    test_placement = {
        'U1': (25.0, 17.5),
        'C1': (15.0, 17.5),
        'R1': (35.0, 17.5),
        'D1': (42.0, 17.5),
    }

    strategy = get_routing_advice(test_parts_db, test_placement, 50.0, 35.0)

    print(strategy.notes)
    print()
    print(f"Use GND plane: {strategy.use_gnd_plane}")
    print(f"GND plane layer: {strategy.gnd_plane_layer}")
    print(f"Net order: {strategy.net_order}")
    print(f"Congestion zones: {strategy.congestion_zones}")
