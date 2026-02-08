"""
PCB Engine - Placement Module
==============================

Implements the FUNDAMENTAL placement algorithm from PCB_FUNDAMENTAL_ALGORITHM.md

KEY PRINCIPLES (from the algorithm):
=====================================
1. "90% of routing success is determined by placement"
2. "Don't place ANY component before fully creating a map for everything related to it"
3. "Escape directions MUST point toward destinations"
4. "No amount of routing cleverness can fix a bad placement"

PLACEMENT PRIORITY ORDER:
=========================
1. Fixed-position components (external constraints)
2. The Hub (highest-connectivity component)
3. Hub-connected components (placed relative to hub)
4. Clusters (grouped by function)
5. Support components (decoupling caps near ICs, etc.)

ALGORITHM:
==========
For each component, BEFORE placing:
- Calculate destination vectors for all its nets
- Find the centroid of all destinations
- Determine optimal position so escapes point toward destinations
- Validate corridor capacity for this placement
- Only then commit the placement
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """Component position with rotation"""
    x: float
    y: float
    rotation: int = 0  # degrees (0, 90, 180, 270)

    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def manhattan_to(self, other: 'Position') -> float:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def vector_to(self, other: 'Position') -> Tuple[float, float]:
        """Vector from this position to other"""
        return (other.x - self.x, other.y - self.y)

    def direction_to(self, other: 'Position') -> Tuple[float, float]:
        """Unit vector toward other position"""
        dx, dy = self.vector_to(other)
        d = math.sqrt(dx*dx + dy*dy)
        if d < 0.001:
            return (0, 0)
        return (dx/d, dy/d)


@dataclass
class DestinationVector:
    """
    Vector from a pin to its destination(s).
    This is CRITICAL for escape direction calculation.
    """
    pin_ref: str  # "U1.3" format
    source_pos: Tuple[float, float]
    destination_centroid: Tuple[float, float]
    destination_count: int

    @property
    def dx(self) -> float:
        return self.destination_centroid[0] - self.source_pos[0]

    @property
    def dy(self) -> float:
        return self.destination_centroid[1] - self.source_pos[1]

    @property
    def distance(self) -> float:
        return math.sqrt(self.dx**2 + self.dy**2)

    @property
    def angle_rad(self) -> float:
        """Angle in radians (-pi to pi)"""
        return math.atan2(self.dy, self.dx)

    @property
    def angle_deg(self) -> float:
        """Angle in degrees (-180 to 180)"""
        return math.degrees(self.angle_rad)

    @property
    def cardinal_direction(self) -> str:
        """Quantize to 8 cardinal directions"""
        angle = self.angle_deg
        if -22.5 <= angle < 22.5:
            return 'E'
        elif 22.5 <= angle < 67.5:
            return 'SE'
        elif 67.5 <= angle < 112.5:
            return 'S'
        elif 112.5 <= angle < 157.5:
            return 'SW'
        elif angle >= 157.5 or angle < -157.5:
            return 'W'
        elif -157.5 <= angle < -112.5:
            return 'NW'
        elif -112.5 <= angle < -67.5:
            return 'N'
        else:  # -67.5 <= angle < -22.5
            return 'NE'

    @property
    def escape_vector(self) -> Tuple[float, float]:
        """Unit vector in escape direction (quantized to 45 degrees)"""
        directions = {
            'N': (0, -1),
            'NE': (0.707, -0.707),
            'E': (1, 0),
            'SE': (0.707, 0.707),
            'S': (0, 1),
            'SW': (-0.707, 0.707),
            'W': (-1, 0),
            'NW': (-0.707, -0.707),
        }
        return directions[self.cardinal_direction]


@dataclass
class ComponentAnalysis:
    """
    Complete analysis of a component BEFORE placement.
    This implements: "Don't place ANY component before fully creating a map"
    """
    ref: str
    footprint: str
    body_width: float
    body_height: float
    pin_count: int

    # Connectivity analysis
    connected_components: List[str]  # refs of connected components
    connection_weights: Dict[str, int]  # ref -> number of nets
    total_connections: int

    # Destination analysis (calculated when other components have positions)
    destination_vectors: Dict[str, DestinationVector] = field(default_factory=dict)
    dominant_direction: Optional[str] = None  # The direction most nets go

    # Placement constraints
    fixed_position: Optional[Tuple[float, float]] = None
    fixed_rotation: Optional[int] = None
    must_be_on_edge: bool = False
    edge_side: Optional[str] = None  # 'top', 'bottom', 'left', 'right'

    # Role classification
    role: str = 'component'  # 'hub', 'connector', 'passive', 'ic', 'power'
    cluster: Optional[str] = None  # Which cluster this belongs to

    @property
    def is_hub(self) -> bool:
        return self.role == 'hub'

    @property
    def hub_score(self) -> int:
        """Higher score = more central to the design"""
        return self.total_connections


# =============================================================================
# DESTINATION VECTOR CALCULATOR
# =============================================================================

class DestinationAnalyzer:
    """
    Analyzes where each component's traces need to go.
    This is Phase 2.4 of the algorithm: "Connection Vector Analysis"
    """

    def __init__(self):
        self.vectors = {}  # component.pin -> DestinationVector

    def analyze_component(self, ref: str, part_data: Dict,
                          nets: Dict, placements: Dict[str, Position]) -> Dict[str, DestinationVector]:
        """
        Calculate destination vectors for all pins of a component.

        Args:
            ref: Component reference (e.g., "U1")
            part_data: Component data from parts database
            nets: All nets in the design
            placements: Current placements (may be partial)

        Returns:
            Dict mapping pin names to DestinationVectors
        """
        vectors = {}
        pins = part_data.get('pins', {})

        # Get component position (may not be placed yet)
        comp_pos = placements.get(ref)
        if comp_pos is None:
            return vectors  # Can't calculate without position

        for pin_num, pin_info in pins.items():
            net_name = pin_info.get('net', '')
            if not net_name or net_name == 'GND':  # Skip GND (it's a zone)
                continue

            net_info = nets.get(net_name, {})
            net_pins = net_info.get('pins', [])

            # Find all OTHER pins on this net (destinations)
            destinations = []
            for other_comp, other_pin in net_pins:
                if other_comp == ref and str(other_pin) == str(pin_num):
                    continue  # Skip self

                # Get position of destination component
                other_pos = placements.get(other_comp)
                if other_pos:
                    destinations.append((other_pos.x, other_pos.y))

            if not destinations:
                continue

            # Calculate centroid of destinations
            centroid_x = sum(d[0] for d in destinations) / len(destinations)
            centroid_y = sum(d[1] for d in destinations) / len(destinations)

            # Get pin position (relative to component center)
            pin_pos = self._get_pin_position(part_data, pin_num, comp_pos)

            vectors[str(pin_num)] = DestinationVector(
                pin_ref=f"{ref}.{pin_num}",
                source_pos=pin_pos,
                destination_centroid=(centroid_x, centroid_y),
                destination_count=len(destinations)
            )

        return vectors

    def _get_pin_position(self, part_data: Dict, pin_num: str,
                          comp_pos: Position) -> Tuple[float, float]:
        """Get absolute position of a pin"""
        pins = part_data.get('pins', {})
        pin_info = pins.get(str(pin_num), pins.get(int(pin_num) if str(pin_num).isdigit() else pin_num, {}))

        # Get relative position from pin data
        rel_x = pin_info.get('x', 0)
        rel_y = pin_info.get('y', 0)

        # Apply rotation
        if comp_pos.rotation == 90:
            rel_x, rel_y = -rel_y, rel_x
        elif comp_pos.rotation == 180:
            rel_x, rel_y = -rel_x, -rel_y
        elif comp_pos.rotation == 270:
            rel_x, rel_y = rel_y, -rel_x

        return (comp_pos.x + rel_x, comp_pos.y + rel_y)

    def get_dominant_direction(self, vectors: Dict[str, DestinationVector]) -> Optional[str]:
        """
        Find the direction where MOST traces need to go.
        This determines optimal component orientation.
        """
        if not vectors:
            return None

        # Count destinations in each direction, weighted by destination count
        direction_weights = {}
        for vec in vectors.values():
            direction = vec.cardinal_direction
            weight = vec.destination_count
            direction_weights[direction] = direction_weights.get(direction, 0) + weight

        if not direction_weights:
            return None

        # Return direction with highest weight
        return max(direction_weights.keys(), key=lambda d: direction_weights[d])


# =============================================================================
# HUB IDENTIFIER
# =============================================================================

class HubIdentifier:
    """
    Identifies the hub component(s) in the design.
    From algorithm: "A hub is a component with high net fanout"
    """

    def identify(self, parts_db: Dict, graph: Dict) -> Tuple[str, Dict[str, int]]:
        """
        Identify the hub and calculate hub scores for all components.

        Returns:
            (hub_ref, scores_dict)
        """
        adjacency = graph.get('adjacency', {})
        scores = {}

        for ref in adjacency.keys():
            # Hub score = sum of all connections to other components
            connections = adjacency.get(ref, {})
            score = sum(connections.values())
            scores[ref] = score

        if not scores:
            return (None, scores)

        # Hub is component with highest score
        hub = max(scores.keys(), key=lambda r: scores[r])

        return (hub, scores)


# =============================================================================
# CLUSTER IDENTIFIER
# =============================================================================

class ClusterIdentifier:
    """
    Identifies component clusters.
    From algorithm: "A cluster is a group of components that primarily connect to each other"
    """

    def identify(self, parts_db: Dict, graph: Dict) -> Dict[str, List[str]]:
        """
        Identify clusters of related components.

        Returns:
            Dict mapping cluster names to list of component refs
        """
        clusters = {}
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Identify clusters by function
        # Power cluster: components on power nets
        power_cluster = set()
        for net_name, net_info in nets.items():
            net_type = net_info.get('type')
            if hasattr(net_type, 'value') and net_type.value == 'power':
                for comp, pin in net_info.get('pins', []):
                    power_cluster.add(comp)

        # Connector cluster: all connectors
        connector_cluster = set()
        for ref, part in parts.items():
            if part.get('category') == 'connector':
                connector_cluster.add(ref)

        clusters['power'] = list(power_cluster)
        clusters['connectors'] = list(connector_cluster)

        return clusters


# =============================================================================
# PLACEMENT OPTIMIZER (Main Class)
# =============================================================================

class PlacementOptimizer:
    """
    Main placement optimizer implementing the fundamental algorithm.

    ALGORITHM FLOW:
    1. Analyze all components (connectivity, destinations)
    2. Identify hub
    3. Place fixed-position components
    4. Place hub with optimal orientation
    5. Place hub-connected components with destination-aligned positions
    6. Place remaining components
    7. Validate corridor capacity
    """

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules

        # Board bounds with margin
        self.min_x = board.origin_x + 2
        self.max_x = board.origin_x + board.width - 2
        self.min_y = board.origin_y + 2
        self.max_y = board.origin_y + board.height - 2

        # Analysis tools
        self.dest_analyzer = DestinationAnalyzer()
        self.hub_identifier = HubIdentifier()
        self.cluster_identifier = ClusterIdentifier()

        # State
        self.placements = {}  # ref -> Position
        self.analyses = {}  # ref -> ComponentAnalysis
        self.errors = []
        self.warnings = []

    def optimize(self, parts_db: Dict, graph: Dict, hub: str = None) -> Dict[str, Position]:
        """
        Run the placement optimization algorithm.

        Args:
            parts_db: Parts database from Phase 0
            graph: Connectivity graph from Phase 1
            hub: Pre-identified hub (optional)

        Returns:
            Dict mapping component refs to Positions
        """
        self.placements = {}
        self.errors = []
        self.warnings = []

        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})
        adjacency = graph.get('adjacency', {})

        # Step 1: Analyze all components
        self._analyze_all_components(parts, nets, adjacency)

        # Step 2: Detect signal chains BEFORE placement
        # Signal chains like U1→R1→D1 should be placed in sequence
        self._signal_chains = self._detect_signal_chains(nets, adjacency)

        # Step 3: Identify hub if not provided
        if hub is None:
            hub, scores = self.hub_identifier.identify(parts_db, graph)

        if hub:
            if hub in self.analyses:
                self.analyses[hub].role = 'hub'

        # Step 4: Determine placement order (uses signal chains)
        placement_order = self._determine_placement_order(hub, adjacency)

        # Step 5: Place components in order
        for ref in placement_order:
            self._place_component(ref, parts, nets, adjacency)

        return self.placements

    def _analyze_all_components(self, parts: Dict, nets: Dict, adjacency: Dict):
        """Analyze all components before placement"""
        for ref, part_data in parts.items():
            phys = part_data.get('physical', {})
            body = phys.get('body', (5, 5))

            # Handle different body formats
            if isinstance(body, (list, tuple)):
                body_w = body[0] if len(body) > 0 else 5
                body_h = body[1] if len(body) > 1 else body_w
            else:
                body_w = body_h = body

            # Get connectivity
            connections = adjacency.get(ref, {})

            analysis = ComponentAnalysis(
                ref=ref,
                footprint=part_data.get('footprint', ''),
                body_width=body_w,
                body_height=body_h,
                pin_count=len(part_data.get('pins', {})),
                connected_components=list(connections.keys()),
                connection_weights=dict(connections),
                total_connections=sum(connections.values()),
                role=self._classify_role(ref, part_data),
            )

            # Check for fixed position constraints
            if part_data.get('fixed_position'):
                analysis.fixed_position = tuple(part_data['fixed_position'])
            if part_data.get('fixed_rotation') is not None:
                analysis.fixed_rotation = part_data['fixed_rotation']
            if part_data.get('edge'):
                analysis.must_be_on_edge = True
                analysis.edge_side = part_data['edge']

            self.analyses[ref] = analysis

    def _classify_role(self, ref: str, part_data: Dict) -> str:
        """Classify component role"""
        category = part_data.get('category', '')

        if category == 'connector':
            return 'connector'
        elif category == 'ic':
            return 'ic'
        elif category in ['capacitor', 'resistor', 'inductor']:
            return 'passive'
        elif category == 'diode':
            return 'diode'
        elif ref.startswith('U'):
            return 'ic'
        elif ref.startswith('J'):
            return 'connector'
        elif ref.startswith('C'):
            return 'passive'
        elif ref.startswith('R'):
            return 'passive'
        else:
            return 'component'

    def _detect_signal_chains(self, nets: Dict, adjacency: Dict) -> List[List[str]]:
        """
        Detect linear signal chains to ensure proper placement order.

        A signal chain is: A → B → C where A connects to B and B connects to C
        but A doesn't connect directly to C.

        These should be placed in sequence to avoid route crossings.
        """
        chains = []

        # Build net-to-components mapping
        for net_name, net_info in nets.items():
            if net_name == 'GND':
                continue

            pins = net_info.get('pins', [])
            if len(pins) != 2:  # Only two-pin nets form chains
                continue

            comp_a, pin_a = pins[0]
            comp_b, pin_b = pins[1]

            # Check if this net is part of a chain
            # A chain exists if comp_b connects to another component on a different net
            for other_net, other_info in nets.items():
                if other_net == net_name or other_net == 'GND':
                    continue

                other_pins = other_info.get('pins', [])
                if len(other_pins) != 2:
                    continue

                other_comp_1, _ = other_pins[0]
                other_comp_2, _ = other_pins[1]

                # Check if comp_b is in the middle (connects to both comp_a and another)
                if comp_b in [other_comp_1, other_comp_2]:
                    third_comp = other_comp_1 if other_comp_2 == comp_b else other_comp_2
                    if third_comp != comp_a:
                        # Found chain: comp_a → comp_b → third_comp
                        chain = [comp_a, comp_b, third_comp]
                        # Check if this chain (or its reverse) is already recorded
                        if chain not in chains and chain[::-1] not in chains:
                            chains.append(chain)

        return chains

    def _determine_placement_order(self, hub: str, adjacency: Dict) -> List[str]:
        """
        Determine optimal placement order.

        ORDER:
        1. Fixed-position components
        2. Hub
        3. Signal chains (placed in sequence to avoid crossings)
        4. Other hub-connected components
        5. Remaining components
        """
        order = []
        placed = set()

        # Detect signal chains
        # Note: We need nets for this, so pass through self
        signal_chains = getattr(self, '_signal_chains', [])

        # 1. Fixed position components first
        for ref, analysis in self.analyses.items():
            if analysis.fixed_position is not None:
                order.append(ref)
                placed.add(ref)

        # 2. Hub next
        if hub and hub not in placed:
            order.append(hub)
            placed.add(hub)

        # 3. Place signal chains in sequence
        for chain in signal_chains:
            for ref in chain:
                if ref not in placed:
                    order.append(ref)
                    placed.add(ref)

        # 4. Hub-connected components (sorted by connection weight)
        if hub:
            hub_connections = adjacency.get(hub, {})
            connected = [(ref, weight) for ref, weight in hub_connections.items()
                        if ref not in placed]
            connected.sort(key=lambda x: -x[1])  # Highest weight first

            for ref, _ in connected:
                order.append(ref)
                placed.add(ref)

        # 4. Remaining components (sorted by total connections)
        remaining = [(ref, analysis.total_connections)
                    for ref, analysis in self.analyses.items()
                    if ref not in placed]
        remaining.sort(key=lambda x: -x[1])

        for ref, _ in remaining:
            order.append(ref)

        return order

    def _place_component(self, ref: str, parts: Dict, nets: Dict, adjacency: Dict):
        """Place a single component using the algorithm"""
        analysis = self.analyses.get(ref)
        if not analysis:
            return

        part_data = parts.get(ref, {})

        # Case 1: Fixed position
        if analysis.fixed_position:
            pos = Position(
                x=analysis.fixed_position[0],
                y=analysis.fixed_position[1],
                rotation=analysis.fixed_rotation or 0
            )
            self.placements[ref] = pos
            return

        # Case 2: Hub placement
        if analysis.is_hub:
            pos = self._place_hub(ref, analysis, parts, nets, adjacency)
            self.placements[ref] = pos
            return

        # Case 3: Regular component - place relative to already-placed connections
        pos = self._place_connected_component(ref, analysis, parts, nets, adjacency)
        self.placements[ref] = pos

    def _place_hub(self, ref: str, analysis: ComponentAnalysis,
                   parts: Dict, nets: Dict, adjacency: Dict) -> Position:
        """
        Place the hub component.

        From algorithm:
        - Hub should be positioned so that the MAJORITY of routes flow in ONE direction
        - Calculate centroid of all connected components
        - Position hub such that escapes align with destinations
        """
        # Calculate weighted centroid of all destination positions
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0

        for connected_ref, weight in analysis.connection_weights.items():
            if connected_ref in self.placements:
                pos = self.placements[connected_ref]
                weighted_x += pos.x * weight
                weighted_y += pos.y * weight
                total_weight += weight

        if total_weight > 0:
            centroid_x = weighted_x / total_weight
            centroid_y = weighted_y / total_weight
        else:
            # No connected components placed yet - use board center
            centroid_x = self.board.origin_x + self.board.width / 2
            centroid_y = self.board.origin_y + self.board.height / 2

        # Hub should be in center of board
        center_x = self.board.origin_x + self.board.width / 2
        center_y = self.board.origin_y + self.board.height / 2

        # Position hub at center (it will be the central node)
        hub_x = center_x
        hub_y = center_y

        # Determine optimal rotation based on pin positions and destinations
        rotation = 0  # Default rotation

        return Position(x=hub_x, y=hub_y, rotation=rotation)

    def _place_connected_component(self, ref: str, analysis: ComponentAnalysis,
                                    parts: Dict, nets: Dict, adjacency: Dict) -> Position:
        """
        Place a component relative to its already-placed connections.

        PRINCIPLE: For signal chains, place components in a LINE to avoid crossings.
        For other components, position so escape direction points toward destinations.
        """
        # Find already-placed connected components
        placed_connections = []
        for connected_ref, weight in analysis.connection_weights.items():
            if connected_ref in self.placements:
                pos = self.placements[connected_ref]
                placed_connections.append((connected_ref, pos, weight))

        if not placed_connections:
            # No connections placed yet - use a default position
            return self._find_available_position(ref, analysis)

        # Find the primary connection (highest weight)
        placed_connections.sort(key=lambda x: -x[2])
        primary_ref, primary_pos, _ = placed_connections[0]

        # Check if this component is part of a signal chain
        chain_placement = self._get_chain_placement(ref, primary_ref)
        if chain_placement:
            target_x, target_y = chain_placement
        else:
            # Standard radial placement for non-chain components
            center_x = self.board.origin_x + self.board.width / 2
            center_y = self.board.origin_y + self.board.height / 2

            # Calculate angle from center to primary connection
            angle_to_primary = math.atan2(
                primary_pos.y - center_y,
                primary_pos.x - center_x
            )

            # Place this component on the OPPOSITE side of the primary connection
            opposite_angle = angle_to_primary + math.pi
            radius = min(self.board.width, self.board.height) * 0.35

            target_x = center_x + math.cos(opposite_angle) * radius
            target_y = center_y + math.sin(opposite_angle) * radius

            # Adjust for multiple connections
            if len(placed_connections) > 1:
                conn_centroid_x = sum(p.x for _, p, _ in placed_connections) / len(placed_connections)
                conn_centroid_y = sum(p.y for _, p, _ in placed_connections) / len(placed_connections)
                target_x = (target_x + conn_centroid_x) / 2
                target_y = (target_y + conn_centroid_y) / 2

        # Offset slightly from primary connection for routing room
        offset_dist = max(analysis.body_width, analysis.body_height) + 3

        # Direction from primary to target
        dx = target_x - primary_pos.x
        dy = target_y - primary_pos.y
        d = math.sqrt(dx*dx + dy*dy)

        if d > 0.1:
            # Ensure minimum distance from primary
            if d < offset_dist:
                target_x = primary_pos.x + dx / d * offset_dist
                target_y = primary_pos.y + dy / d * offset_dist

        # Clamp to board bounds
        half_w = analysis.body_width / 2
        half_h = analysis.body_height / 2
        target_x = max(self.min_x + half_w, min(self.max_x - half_w, target_x))
        target_y = max(self.min_y + half_h, min(self.max_y - half_h, target_y))

        pos = Position(x=target_x, y=target_y, rotation=0)

        # Resolve any overlaps
        pos = self._resolve_overlaps(ref, analysis, pos)

        return pos

    def _get_chain_placement(self, ref: str, primary_ref: str) -> Optional[Tuple[float, float]]:
        """
        Get linear placement for signal chain component.

        For a chain like U1→R1→D1:
        - R1 should be placed BETWEEN U1 and D1's eventual position
        - D1 should be placed in LINE with U1→R1

        This prevents routes from crossing each other.
        """
        for chain in getattr(self, '_signal_chains', []):
            if ref not in chain:
                continue

            idx = chain.index(ref)
            if idx == 0:
                # First in chain - use standard placement
                return None

            # Get previous component in chain
            prev_ref = chain[idx - 1]
            if prev_ref not in self.placements:
                return None

            prev_pos = self.placements[prev_ref]

            # Determine chain direction from first two placed components
            if idx >= 2:
                # Use existing direction
                prev_prev_ref = chain[idx - 2]
                if prev_prev_ref in self.placements:
                    prev_prev_pos = self.placements[prev_prev_ref]
                    dx = prev_pos.x - prev_prev_pos.x
                    dy = prev_pos.y - prev_prev_pos.y
                    d = math.sqrt(dx*dx + dy*dy)
                    if d > 0.1:
                        # Continue in same direction
                        spacing = 5.0  # mm between components in chain
                        return (prev_pos.x + dx/d * spacing, prev_pos.y + dy/d * spacing)

            # For second component in chain, pick a direction
            # Default: go DOWN (positive Y) from previous component
            spacing = 5.0
            return (prev_pos.x, prev_pos.y + spacing)

        return None

    def _find_available_position(self, ref: str, analysis: ComponentAnalysis) -> Position:
        """Find an available position when no connections are placed"""
        # Use a grid-based approach to find available space
        center_x = self.board.origin_x + self.board.width / 2
        center_y = self.board.origin_y + self.board.height / 2

        # Start from center and spiral outward
        pos = Position(x=center_x, y=center_y, rotation=0)
        return self._resolve_overlaps(ref, analysis, pos)

    def _resolve_overlaps(self, ref: str, analysis: ComponentAnalysis,
                          pos: Position) -> Position:
        """Move position to avoid overlaps with existing placements"""
        max_attempts = 100

        for attempt in range(max_attempts):
            if not self._has_overlap(ref, analysis, pos):
                return pos

            # Spiral outward from current position
            angle = attempt * 0.618 * 2 * math.pi  # Golden angle for good coverage
            distance = 3 + attempt * 1.5

            new_x = pos.x + math.cos(angle) * distance
            new_y = pos.y + math.sin(angle) * distance

            # Clamp to board
            half_w = analysis.body_width / 2
            half_h = analysis.body_height / 2
            new_x = max(self.min_x + half_w, min(self.max_x - half_w, new_x))
            new_y = max(self.min_y + half_h, min(self.max_y - half_h, new_y))

            pos = Position(x=new_x, y=new_y, rotation=pos.rotation)

        return pos

    def _has_overlap(self, ref: str, analysis: ComponentAnalysis, pos: Position) -> bool:
        """Check if position overlaps with existing placements"""
        margin = 1.5  # mm clearance between components

        half_w = analysis.body_width / 2 + margin
        half_h = analysis.body_height / 2 + margin

        for other_ref, other_pos in self.placements.items():
            if other_ref == ref:
                continue

            other_analysis = self.analyses.get(other_ref)
            if not other_analysis:
                continue

            other_half_w = other_analysis.body_width / 2 + margin
            other_half_h = other_analysis.body_height / 2 + margin

            # Check bounding box overlap
            if (abs(pos.x - other_pos.x) < half_w + other_half_w and
                abs(pos.y - other_pos.y) < half_h + other_half_h):
                return True

        return False

    def validate(self) -> bool:
        """Validate the placement"""
        valid = True

        for ref, pos in self.placements.items():
            analysis = self.analyses.get(ref)
            if not analysis:
                continue

            # Check board bounds
            half_w = analysis.body_width / 2
            half_h = analysis.body_height / 2

            if (pos.x - half_w < self.min_x or pos.x + half_w > self.max_x or
                pos.y - half_h < self.min_y or pos.y + half_h > self.max_y):
                self.errors.append(f"Component {ref} outside board bounds")
                valid = False

            # Check overlaps
            if self._has_overlap(ref, analysis, pos):
                self.errors.append(f"Component {ref} overlaps with another component")
                valid = False

        return valid
