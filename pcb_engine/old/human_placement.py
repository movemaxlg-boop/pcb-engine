"""
PCB Engine - Human-Like Placement Module
=========================================

This module implements placement following EXACTLY how a human expert thinks.

HUMAN EXPERT WORKFLOW:
======================
1. LOOK AT THE WHOLE BOARD FIRST
   - What's the board size?
   - Where are fixed components? (connectors at edges)
   - What's the general signal flow?

2. IDENTIFY FUNCTIONAL GROUPS
   - Power section (regulator + caps)
   - MCU section
   - Sensor section
   - Interface section (USB, I2C)

3. IDENTIFY THE SIGNAL FLOW
   - Power: Input → Regulator → MCU/Sensors
   - Data: MCU ↔ Sensors
   - Signal chains: MCU → R → LED → GND

4. PLACE HUB FIRST
   - MCU is usually the hub
   - Place it where it has access to all sides
   - Consider which pins connect where

5. PLACE CONNECTED COMPONENTS IN ORDER
   - Place components that connect to placed ones
   - Position so routes will be SHORT and STRAIGHT
   - Never place where routes would cross

6. LEAVE ROUTING CHANNELS
   - Components need space between them
   - More connections = more space needed

KEY INSIGHT: A human VISUALIZES the routes BEFORE placing components.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import math


@dataclass
class Position:
    """Component position"""
    x: float
    y: float
    rotation: int = 0

    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class ComponentInfo:
    """Everything a human knows about a component before placing it"""
    ref: str
    width: float
    height: float
    pin_count: int

    # Which sides have pins
    has_left_pins: bool = False
    has_right_pins: bool = False
    has_top_pins: bool = False
    has_bottom_pins: bool = False

    # Connection analysis
    connected_to: List[str] = field(default_factory=list)  # Other component refs
    nets: List[str] = field(default_factory=list)  # Net names

    # Classification
    is_connector: bool = False  # Goes at board edge
    is_power: bool = False      # Power-related (regulator, power caps)
    is_decoupling: bool = False # Decoupling cap (stays near IC)
    is_hub: bool = False        # Main IC with most connections

    # Placement constraints
    fixed_position: Optional[Tuple[float, float]] = None
    preferred_zone: str = 'center'  # 'top', 'bottom', 'left', 'right', 'center'


class HumanLikePlacer:
    """
    Places components like a human expert would.

    The key difference from algorithmic placement:
    - Human SEES the whole picture first
    - Human KNOWS where routes will go before placing
    - Human places to make routing OBVIOUS
    """

    def __init__(self, board_width: float, board_height: float,
                 origin_x: float = 0, origin_y: float = 0,
                 grid_size: float = 0.5, spacing: float = 2.0):
        """
        spacing: Minimum routing channel width between components.
                 2.0mm allows reasonable routing while fitting more components.
        """
        self.board_width = board_width
        self.board_height = board_height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.grid_size = grid_size
        self.spacing = spacing  # Minimum routing channel width

        # Board zones (where different types of components go)
        self.zones = self._define_zones()

        # Placement state
        self.placements: Dict[str, Position] = {}
        self.component_info: Dict[str, ComponentInfo] = {}

    def _define_zones(self) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Define board zones like a human would mentally divide the board.

        Human thinks: "Connectors at edges, MCU in center, caps near their ICs"
        """
        ox, oy = self.origin_x, self.origin_y
        w, h = self.board_width, self.board_height
        margin = 3.0  # Edge margin

        return {
            # Edge zones for connectors
            'top_edge': (ox + margin, oy + margin, ox + w - margin, oy + h * 0.15),
            'bottom_edge': (ox + margin, oy + h * 0.85, ox + w - margin, oy + h - margin),
            'left_edge': (ox + margin, oy + margin, ox + w * 0.15, oy + h - margin),
            'right_edge': (ox + w * 0.85, oy + margin, ox + w - margin, oy + h - margin),

            # Center zone for MCU/main ICs
            'center': (ox + w * 0.25, oy + h * 0.25, ox + w * 0.75, oy + h * 0.75),

            # Quadrants for functional groups
            'top_left': (ox + margin, oy + margin, ox + w * 0.5, oy + h * 0.5),
            'top_right': (ox + w * 0.5, oy + margin, ox + w - margin, oy + h * 0.5),
            'bottom_left': (ox + margin, oy + h * 0.5, ox + w * 0.5, oy + h - margin),
            'bottom_right': (ox + w * 0.5, oy + h * 0.5, ox + w - margin, oy + h - margin),
        }

    def analyze_design(self, parts_db: Dict, graph: Dict):
        """
        Step 1: LOOK AT THE WHOLE DESIGN FIRST

        Human expert spends time understanding the design before touching anything.
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})
        adjacency = graph.get('adjacency', {})

        # Analyze each component
        for ref, part_data in parts.items():
            info = self._analyze_component(ref, part_data, nets, adjacency)
            self.component_info[ref] = info

        # Identify the hub (most connected component)
        self._identify_hub()

        # Detect signal chains
        self.signal_chains = self._detect_signal_chains(nets, adjacency)

        # Determine placement order
        self.placement_order = self._determine_placement_order(adjacency)

    def _analyze_component(self, ref: str, part_data: Dict,
                           nets: Dict, adjacency: Dict) -> ComponentInfo:
        """Analyze a single component like a human would"""

        # Get physical size - try multiple possible locations:
        # 1. 'size' (direct) - used in test data
        # 2. 'physical.body' (exported from PartsCollector)
        # 3. 'physical.courtyard' (preferred for placement calculations)
        size = None

        # Try 'size' first (direct in test data)
        if 'size' in part_data:
            size = part_data.get('size')

        # Try 'physical.body' (from export)
        if size is None:
            physical = part_data.get('physical', {})
            if 'body' in physical:
                size = physical.get('body')
            elif 'courtyard' in physical:
                size = physical.get('courtyard')

        # Default
        if size is None or not isinstance(size, (list, tuple)) or len(size) < 2:
            width = height = 2.0
        else:
            width, height = size[0], size[1]

        # Analyze pins
        used_pins = part_data.get('used_pins', [])
        has_left = has_right = has_top = has_bottom = False
        pin_nets = []

        for pin in used_pins:
            net = pin.get('net', '')
            if net:
                pin_nets.append(net)

            offset = pin.get('offset', (0, 0))
            dx, dy = offset[0] if isinstance(offset, (list, tuple)) else 0, \
                     offset[1] if isinstance(offset, (list, tuple)) and len(offset) > 1 else 0

            if abs(dx) > abs(dy):
                if dx < 0:
                    has_left = True
                else:
                    has_right = True
            else:
                if dy < 0:
                    has_top = True
                else:
                    has_bottom = True

        # Get connected components
        connected = list(adjacency.get(ref, {}).keys())

        # Classify component
        name = part_data.get('name', '').lower()
        footprint = part_data.get('footprint', '').lower()

        is_connector = any(kw in name or kw in footprint
                          for kw in ['connector', 'usb', 'header', 'jack', 'socket'])
        is_power = any(kw in name for kw in ['regulator', 'ldo', 'ams1117', 'vreg'])
        is_decoupling = ('capacitor' in name or footprint.startswith('c0')) and \
                        any(net in ['3V3', 'VCC', 'VBUS', '5V', 'GND'] for net in pin_nets)

        return ComponentInfo(
            ref=ref,
            width=width,
            height=height,
            pin_count=len(used_pins),
            has_left_pins=has_left,
            has_right_pins=has_right,
            has_top_pins=has_top,
            has_bottom_pins=has_bottom,
            connected_to=connected,
            nets=pin_nets,
            is_connector=is_connector,
            is_power=is_power,
            is_decoupling=is_decoupling,
        )

    def _identify_hub(self):
        """
        Identify which component is the hub (most connections).

        Human principle: The hub is the component that connects to MOST other things.
        For small designs, even a component with 2 connections can be the hub.
        Prioritize components with more pins (ICs) over 2-pin passives.
        """
        best_ref = None
        best_score = 0

        for ref, info in self.component_info.items():
            if info.is_connector:
                continue  # Connectors aren't hubs

            # Score = number of connections * pin count (more pins = more central)
            conn_count = len(info.connected_to)
            pin_count = info.pin_count

            # Weight by pin count - more pins likely means IC (hub)
            score = conn_count * (1 + pin_count / 4)

            if score > best_score:
                best_score = score
                best_ref = ref

        if best_ref:
            self.component_info[best_ref].is_hub = True
            self.hub = best_ref
        else:
            self.hub = None

    def _detect_signal_chains(self, nets: Dict, adjacency: Dict) -> List[List[str]]:
        """
        Detect signal chains like U1 → R1 → D1

        Human recognizes: "This resistor is between the MCU and LED"

        Key insight: Only consider SIGNAL nets, not power nets (GND, VCC, etc.)
        """
        chains = []
        seen_chains = set()  # Avoid duplicates

        # Power nets to exclude
        power_nets = {'GND', 'VCC', '3V3', '5V', 'VBUS', 'VBAT', 'V+', 'V-'}

        # Find 2-pin components that bridge two other components via SIGNAL nets
        for ref, info in self.component_info.items():
            if info.pin_count != 2:
                continue

            # Get the two nets this component connects (excluding power)
            signal_nets = [n for n in info.nets if n not in power_nets and n != 'NC']

            if len(signal_nets) != 2:
                continue  # Not bridging two signal nets

            net1, net2 = signal_nets[0], signal_nets[1]

            # Find what else connects to these nets
            net1_components = set()
            net2_components = set()

            for net_name, net_info in nets.items():
                pins = net_info.get('pins', [])
                for comp, pin in pins:
                    if comp == ref:
                        continue
                    if net_name == net1:
                        net1_components.add(comp)
                    elif net_name == net2:
                        net2_components.add(comp)

            # If we have a clear chain A → ref → B
            if len(net1_components) == 1 and len(net2_components) == 1:
                comp_a = list(net1_components)[0]
                comp_b = list(net2_components)[0]

                # Order: put hub/MCU first
                info_a = self.component_info.get(comp_a, ComponentInfo(ref='', width=0, height=0, pin_count=0))
                info_b = self.component_info.get(comp_b, ComponentInfo(ref='', width=0, height=0, pin_count=0))

                if info_b.is_hub:
                    comp_a, comp_b = comp_b, comp_a
                elif info_a.pin_count > info_b.pin_count:
                    pass  # comp_a has more pins, keep order
                elif info_b.pin_count > info_a.pin_count:
                    comp_a, comp_b = comp_b, comp_a

                # Create chain key to avoid duplicates
                chain_key = tuple(sorted([comp_a, ref, comp_b]))
                if chain_key not in seen_chains:
                    seen_chains.add(chain_key)
                    chains.append([comp_a, ref, comp_b])

        return chains

    def _determine_placement_order(self, adjacency: Dict) -> List[str]:
        """
        Determine the order to place components.

        Human order:
        1. Fixed components (connectors at edges)
        2. Hub (MCU)
        3. Components connected to hub
        4. Signal chain components
        5. Support components (decoupling caps)
        6. Everything else
        """
        order = []
        placed = set()

        # 1. Fixed position components (connectors at edges)
        for ref, info in self.component_info.items():
            if info.is_connector:
                order.append(ref)
                placed.add(ref)

        # 2. Hub
        if self.hub and self.hub not in placed:
            order.append(self.hub)
            placed.add(self.hub)

        # 3. Signal chains (in chain order)
        for chain in self.signal_chains:
            for ref in chain:
                if ref not in placed:
                    order.append(ref)
                    placed.add(ref)

        # 4. Components connected to already-placed, by connection count
        remaining = [(ref, len(info.connected_to))
                    for ref, info in self.component_info.items()
                    if ref not in placed]
        remaining.sort(key=lambda x: -x[1])  # Most connected first

        for ref, _ in remaining:
            if ref not in placed:
                order.append(ref)
                placed.add(ref)

        return order

    def place_all(self) -> Dict[str, Position]:
        """
        Place all components following human workflow.
        """
        for ref in self.placement_order:
            info = self.component_info[ref]
            pos = self._place_component(ref, info)
            self.placements[ref] = pos

        return self.placements

    def _place_component(self, ref: str, info: ComponentInfo) -> Position:
        """
        Place a single component like a human would.

        Human thinks:
        - "Where does this need to connect?"
        - "What's already placed nearby?"
        - "Where will the routes go?"
        """

        # 1. Connectors go at board edges
        if info.is_connector:
            return self._place_at_edge(ref, info)

        # 2. Hub goes in center
        if info.is_hub:
            return self._place_hub(ref, info)

        # 3. Signal chain components follow the chain
        chain_pos = self._get_chain_position(ref, info)
        if chain_pos:
            return chain_pos

        # 4. Other components go near their connections
        return self._place_near_connections(ref, info)

    def _place_at_edge(self, ref: str, info: ComponentInfo) -> Position:
        """Place connector at board edge"""
        # Default: bottom edge, centered
        cx = self.origin_x + self.board_width / 2
        cy = self.origin_y + self.board_height - info.height / 2 - 2

        return self._snap_to_grid(Position(x=cx, y=cy), info)

    def _place_hub(self, ref: str, info: ComponentInfo) -> Position:
        """Place hub (MCU) in a central position with room for routing"""
        # Slightly off-center toward top to leave room for connectors at bottom
        cx = self.origin_x + self.board_width / 2
        cy = self.origin_y + self.board_height * 0.4

        return self._snap_to_grid(Position(x=cx, y=cy), info)

    def _get_chain_position(self, ref: str, info: ComponentInfo) -> Optional[Position]:
        """
        Get position for signal chain component.

        Human principle: For signal chains (e.g., MCU -> R -> LED -> GND),
        place components in a VERTICAL line below the hub. This is the most
        common pattern and leaves horizontal routing channels clear.

        The escape routes will go sideways, and the signal route connects them
        with a vertical trace.
        """
        for chain in self.signal_chains:
            if ref not in chain:
                continue

            idx = chain.index(ref)
            if idx == 0:
                return None  # First in chain uses normal placement

            # Get previous component in chain
            prev_ref = chain[idx - 1]
            if prev_ref not in self.placements:
                return None

            prev_pos = self.placements[prev_ref]
            prev_info = self.component_info.get(prev_ref)

            # Always place chain components BELOW (vertical chain)
            # This is the most common human pattern for LED chains etc.
            spacing = self._get_chain_spacing(prev_info, info)
            nx = prev_pos.x
            ny = prev_pos.y + spacing
            return self._snap_to_grid(Position(x=nx, y=ny), info)

        return None

    def _get_chain_spacing(self, prev_info: Optional[ComponentInfo],
                           curr_info: ComponentInfo) -> float:
        """
        Calculate spacing between chain components.

        Human principle: Leave routing channels but don't waste space.
        For large components (like ESP32), add extra spacing to allow
        routing channels around chain components.
        """
        if prev_info:
            prev_size = max(prev_info.width, prev_info.height)
        else:
            prev_size = 2.0

        curr_size = max(curr_info.width, curr_info.height)

        # Base spacing = half of each component + routing channel
        base_spacing = (prev_size + curr_size) / 2 + self.spacing

        # For large components (>10mm), add extra spacing for routing
        # This ensures chains attached to ESP32 etc. have routing room
        if prev_size > 10.0:
            base_spacing += 3.0  # Extra 3mm for routing channels

        return base_spacing

    def _place_near_connections(self, ref: str, info: ComponentInfo) -> Position:
        """
        Place component near its connections.

        Human principle: Components should be close to what they connect to.
        """
        # Find all placed components this one connects to
        placed_connections = []
        for conn_ref in info.connected_to:
            if conn_ref in self.placements:
                placed_connections.append(self.placements[conn_ref])

        if not placed_connections:
            # No connections placed yet, use center
            return self._place_hub(ref, info)

        # Calculate centroid of connections
        cx = sum(p.x for p in placed_connections) / len(placed_connections)
        cy = sum(p.y for p in placed_connections) / len(placed_connections)

        # Find closest connection
        closest = min(placed_connections, key=lambda p: math.sqrt((p.x-cx)**2 + (p.y-cy)**2))

        # Place offset from closest connection
        # Direction: away from centroid to spread out
        dx = closest.x - cx
        dy = closest.y - cy
        d = math.sqrt(dx*dx + dy*dy)

        if d > 0.1:
            # Place on opposite side of centroid from closest
            offset = max(info.width, info.height) + self.spacing
            nx = closest.x + (dx/d) * offset
            ny = closest.y + (dy/d) * offset
        else:
            # Connections are at same point, place below
            nx = closest.x
            ny = closest.y + max(info.width, info.height) + self.spacing

        # Resolve overlaps
        pos = self._snap_to_grid(Position(x=nx, y=ny), info)
        return self._resolve_overlaps(ref, info, pos)

    def _snap_to_grid(self, pos: Position, info: Optional[ComponentInfo] = None) -> Position:
        """Snap position to grid and clamp to board boundaries"""
        x = round(pos.x / self.grid_size) * self.grid_size
        y = round(pos.y / self.grid_size) * self.grid_size

        # Clamp to board boundaries with margin for component size
        margin = 3.0  # Edge margin
        if info:
            half_w = info.width / 2
            half_h = info.height / 2
        else:
            half_w = half_h = 2.0

        min_x = self.origin_x + margin + half_w
        max_x = self.origin_x + self.board_width - margin - half_w
        min_y = self.origin_y + margin + half_h
        max_y = self.origin_y + self.board_height - margin - half_h

        x = max(min_x, min(max_x, x))
        y = max(min_y, min(max_y, y))

        return Position(x=x, y=y, rotation=pos.rotation)

    def _resolve_overlaps(self, ref: str, info: ComponentInfo,
                          pos: Position, max_attempts: int = 100) -> Position:
        """
        Resolve overlaps by searching for clear position.

        Uses spiral search pattern starting from desired position.
        """
        original_pos = pos

        for attempt in range(max_attempts):
            if not self._has_overlap(ref, info, pos):
                return pos

            # Spiral search: alternate directions, increasing radius
            angle = (attempt * 30) % 360  # 30-degree increments
            radius = self.grid_size * 2 * (1 + attempt // 12)  # Grow radius every 12 attempts

            rad = math.radians(angle)
            new_x = original_pos.x + math.cos(rad) * radius
            new_y = original_pos.y + math.sin(rad) * radius
            pos = self._snap_to_grid(Position(x=new_x, y=new_y), info)

        # Last resort: find ANY clear position on the board
        # Use at least 1mm steps to avoid slow iteration with fine grids
        step = max(1, int(self.grid_size * 4))
        for y_offset in range(0, int(self.board_height), step):
            for x_offset in range(0, int(self.board_width), step):
                test_pos = Position(
                    x=self.origin_x + x_offset + info.width/2 + self.spacing,
                    y=self.origin_y + y_offset + info.height/2 + self.spacing
                )
                test_pos = self._snap_to_grid(test_pos, info)
                if not self._has_overlap(ref, info, test_pos):
                    return test_pos

        return original_pos  # Give up - return original (will cause overlap)

    def _has_overlap(self, ref: str, info: ComponentInfo, pos: Position) -> bool:
        """
        Check if position overlaps with any placed component.

        Uses courtyard (body + clearance) for proper DRC-aware spacing.
        """
        # Courtyard = component size + spacing (ensures routing channels)
        half_w = info.width / 2 + self.spacing
        half_h = info.height / 2 + self.spacing

        for other_ref, other_pos in self.placements.items():
            if other_ref == ref:
                continue

            other_info = self.component_info.get(other_ref)
            if not other_info:
                continue

            other_half_w = other_info.width / 2 + self.spacing
            other_half_h = other_info.height / 2 + self.spacing

            # Check courtyard overlap (not just body overlap)
            if (abs(pos.x - other_pos.x) < half_w + other_half_w and
                abs(pos.y - other_pos.y) < half_h + other_half_h):
                return True

        return False


def human_like_placement(board_config, parts_db: Dict, graph: Dict) -> Dict[str, Position]:
    """
    Main entry point for human-like placement.

    Returns dict of {ref: Position}
    """
    placer = HumanLikePlacer(
        board_width=board_config.width,
        board_height=board_config.height,
        origin_x=board_config.origin_x,
        origin_y=board_config.origin_y,
        grid_size=board_config.grid_size,
    )

    # Step 1: Analyze entire design first (like human does)
    placer.analyze_design(parts_db, graph)

    # Step 2: Place all components in human-determined order
    return placer.place_all()
