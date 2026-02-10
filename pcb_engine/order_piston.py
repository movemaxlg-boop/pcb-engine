"""
PCB Engine - Order Piston
==========================

A dedicated piston (sub-engine) for all ordering and prioritization decisions in PCB design.

This module centralizes ordering logic that affects:
1. Component Placement Order - Which component to place first
2. Net Routing Order - Which net to route first
3. Layer Assignment - Which signals go on which layers
4. Via Strategy - When and where to place vias
5. Pin Escape Order - Which pin to escape first

Ordering algorithms implemented:

PLACEMENT ORDERING:
1. Hub-Spoke Order - Place hub (most connected) component first
2. Criticality Order - Place timing-critical components first
3. Signal Flow Order - Follow signal path from input to output
4. Size-Based Order - Place largest components first (stability)

NET ROUTING ORDER:
1. Short-First Order - Route shortest nets first (less blocking)
2. Long-First Order - Route longest nets first (more constrained)
3. Critical-First Order - Route timing-critical nets first
4. Bounding-Box Order - Route by bounding box area
5. Pin-Count Order - Route by number of pins
6. Congestion-Aware Order - Consider routing congestion

LAYER ASSIGNMENT:
1. Signal Integrity Based - Separate high-speed from low-speed
2. Crosstalk Minimization - Orthogonal routing on adjacent layers
3. Power/Ground Separation - Dedicated planes for power
4. Via Minimization - Minimize layer changes

VIA STRATEGIES:
1. Minimize Vias - Prefer same-layer routing
2. Via Clustering - Group vias together
3. Via Fanout - Spread vias for thermal

PIN ESCAPE ORDER:
1. Outer-First - Escape outer pins first (less crossing)
2. Inner-First - Escape inner pins first (critical signals)
3. Clockwise/Counter-Clockwise - Spiral pattern
4. Net-Priority - Escape by net criticality
5. Congestion-Aware - Consider routing congestion

Research References:
- "Net ordering for routability" (Sequential routing papers)
- "Via minimization in VLSI routing" (IEEE, various)
- "Layer assignment for multilayer routing" (ACM/IEEE)
- "Signal integrity in PCB design" (Industry best practices)
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict

# Import common types for position handling (BUG-03/BUG-06 fix)
from .common_types import Position, normalize_position, get_xy, get_pins, get_pin_net


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class PlacementOrderStrategy(Enum):
    """Strategies for component placement order"""
    HUB_SPOKE = 'hub_spoke'           # Most connected component first
    CRITICALITY = 'criticality'       # Timing-critical first
    SIGNAL_FLOW = 'signal_flow'       # Follow signal path
    SIZE_BASED = 'size_based'         # Largest first
    THERMAL = 'thermal'               # Heat-generating first
    AUTO = 'auto'                     # Automatically select


class NetOrderStrategy(Enum):
    """Strategies for net routing order"""
    SHORT_FIRST = 'short_first'       # Shortest nets first
    LONG_FIRST = 'long_first'         # Longest nets first
    CRITICAL_FIRST = 'critical_first' # Timing-critical first
    BOUNDING_BOX = 'bounding_box'     # By bounding box area
    PIN_COUNT = 'pin_count'           # By number of pins
    CONGESTION_AWARE = 'congestion'   # Consider congestion
    POWER_FIRST = 'power_first'       # Power nets first
    SIGNAL_FIRST = 'signal_first'     # Signal nets before power
    AUTO = 'auto'                     # Automatically select


class LayerAssignmentStrategy(Enum):
    """Strategies for layer assignment"""
    SIGNAL_INTEGRITY = 'signal_integrity'   # SI-based assignment
    CROSSTALK_MIN = 'crosstalk_min'         # Minimize crosstalk
    VIA_MINIMIZE = 'via_minimize'           # Minimize vias
    POWER_GROUND_SEP = 'power_ground_sep'   # Separate power/ground
    ALTERNATING = 'alternating'             # Alternate H/V routing
    AUTO = 'auto'                           # Automatically select


class ViaStrategy(Enum):
    """Strategies for via placement"""
    MINIMIZE = 'minimize'             # Minimize total vias
    CLUSTER = 'cluster'               # Group vias together
    FANOUT = 'fanout'                 # Spread vias for thermal
    BALANCED = 'balanced'             # Balance via distribution
    AUTO = 'auto'                     # Automatically select


class PinEscapeStrategy(Enum):
    """Strategies for pin escape ordering"""
    OUTER_FIRST = 'outer_first'       # Escape outer pins first (BGA/QFP)
    INNER_FIRST = 'inner_first'       # Escape inner pins first (critical)
    CLOCKWISE = 'clockwise'           # Spiral clockwise from corner
    COUNTER_CLOCKWISE = 'counter_cw'  # Spiral counter-clockwise
    NET_PRIORITY = 'net_priority'     # By net criticality
    CONGESTION_AWARE = 'congestion'   # Consider local congestion
    QUADRANT = 'quadrant'             # Group by quadrant
    SIDE_BY_SIDE = 'side_by_side'     # Escape each side sequentially
    AUTO = 'auto'                     # Automatically select


@dataclass
class OrderConfig:
    """Configuration for the order piston"""
    # Strategy selections
    placement_strategy: str = 'auto'
    net_order_strategy: str = 'auto'
    layer_strategy: str = 'auto'
    via_strategy: str = 'auto'
    pin_escape_strategy: str = 'auto'

    # Design parameters
    num_layers: int = 2
    board_width: float = 100.0
    board_height: float = 100.0

    # Net classification thresholds
    power_net_patterns: List[str] = field(default_factory=lambda: ['VCC', 'VDD', 'GND', '5V', '3V3', '3.3V', '12V', 'PWR'])
    high_speed_threshold_mhz: float = 50.0
    critical_net_patterns: List[str] = field(default_factory=lambda: ['CLK', 'CLOCK', 'DATA', 'MISO', 'MOSI', 'SCK', 'SDA', 'SCL'])

    # Layer preferences
    top_layer_name: str = 'F.Cu'
    bottom_layer_name: str = 'B.Cu'
    inner_layer_names: List[str] = field(default_factory=list)

    # Via parameters
    via_cost: float = 5.0  # Cost factor for adding a via


@dataclass
class ComponentInfo:
    """Information about a component for ordering"""
    ref: str
    pin_count: int = 0
    net_count: int = 0  # Number of unique nets
    area: float = 0.0
    is_hub: bool = False
    criticality: float = 0.0  # 0-1, higher = more critical
    power_consumption: float = 0.0  # For thermal ordering
    position: Optional[Tuple[float, float]] = None


@dataclass
class NetInfo:
    """Information about a net for ordering"""
    name: str
    pins: List[Tuple[str, str]] = field(default_factory=list)  # [(component, pin), ...]
    pin_count: int = 0
    estimated_length: float = 0.0
    bounding_box_area: float = 0.0
    is_power: bool = False
    is_ground: bool = False
    is_critical: bool = False
    is_high_speed: bool = False
    assigned_layer: Optional[str] = None
    priority: float = 0.0


@dataclass
class PinInfo:
    """Information about a pin for escape ordering"""
    component: str
    pin_number: str
    net: str = ''
    position: Tuple[float, float] = (0.0, 0.0)
    is_outer: bool = False
    quadrant: int = 0  # 0=NE, 1=SE, 2=SW, 3=NW
    side: str = ''  # 'top', 'bottom', 'left', 'right'
    priority: float = 0.0


@dataclass
class OrderResult:
    """Result from the order piston"""
    placement_order: List[str]
    net_order: List[str]
    layer_assignments: Dict[str, str]  # net_name -> layer
    via_recommendations: Dict[str, List[Tuple[float, float]]]  # net_name -> via positions
    pin_escape_order: Dict[str, List[str]] = field(default_factory=dict)  # component -> [pin_numbers]
    statistics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ORDER PISTON
# =============================================================================

class OrderPiston:
    """
    PCB Order Piston

    Centralizes all ordering and prioritization decisions for PCB design.
    Provides research-based algorithms for determining optimal order of
    operations in placement, routing, and layer assignment.

    Usage:
        config = OrderConfig(
            placement_strategy='hub_spoke',
            net_order_strategy='short_first',
            num_layers=2
        )
        piston = OrderPiston(config)

        # Get placement order
        placement_order = piston.get_placement_order(parts_db)

        # Get net routing order
        net_order = piston.get_net_order(nets, placement)

        # Get layer assignments
        layer_map = piston.get_layer_assignment(nets)
    """

    def __init__(self, config: OrderConfig):
        self.config = config
        self.component_info: Dict[str, ComponentInfo] = {}
        self.net_info: Dict[str, NetInfo] = {}

    @staticmethod
    def _parse_pin_ref(pin_ref) -> Tuple[str, str]:
        """
        Parse a pin reference into (component, pin) tuple.
        Handles both 'U1.1' string format and ('U1', '1') tuple format.
        """
        if isinstance(pin_ref, str):
            parts = pin_ref.split('.')
            if len(parts) >= 2:
                return (parts[0], parts[1])
            return (parts[0], '') if parts else ('', '')
        elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
            return (str(pin_ref[0]), str(pin_ref[1]))
        elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) == 1:
            return (str(pin_ref[0]), '')
        return ('', '')

    # =========================================================================
    # MAIN API
    # =========================================================================

    def order(self, parts_db: Dict, placement: Optional[Dict] = None) -> OrderResult:
        """
        Compute all orderings for the design.

        Standard piston API entry point. Alias for analyze().

        Args:
            parts_db: Parts database with components and nets
            placement: Optional placement positions

        Returns:
            OrderResult with all orderings
        """
        return self.analyze(parts_db, placement)

    def analyze(self, parts_db: Dict, placement: Optional[Dict] = None) -> OrderResult:
        """
        Analyze the design and compute all orderings.

        Args:
            parts_db: Parts database with components and nets
            placement: Optional placement positions for length estimation

        Returns:
            OrderResult with all orderings and assignments
        """
        # Build component and net info
        self._analyze_components(parts_db)
        self._analyze_nets(parts_db, placement)

        # Compute orderings
        placement_order = self.get_placement_order(parts_db)
        net_order = self.get_net_order(parts_db, placement)
        layer_assignments = self.get_layer_assignment(parts_db)
        via_recommendations = self.get_via_strategy(parts_db, placement)
        pin_escape_order = self.get_pin_escape_order(parts_db)

        return OrderResult(
            placement_order=placement_order,
            net_order=net_order,
            layer_assignments=layer_assignments,
            via_recommendations=via_recommendations,
            pin_escape_order=pin_escape_order,
            statistics={
                'component_count': len(self.component_info),
                'net_count': len(self.net_info),
                'power_net_count': sum(1 for n in self.net_info.values() if n.is_power or n.is_ground),
                'critical_net_count': sum(1 for n in self.net_info.values() if n.is_critical)
            }
        )

    # =========================================================================
    # COMPONENT PLACEMENT ORDER
    # =========================================================================

    def get_placement_order(self, parts_db: Dict) -> List[str]:
        """
        Determine optimal component placement order.

        Different strategies for different design goals:
        - hub_spoke: Place most connected component first, then neighbors
        - criticality: Place timing-critical components first
        - signal_flow: Follow signal path from input to output
        - size_based: Place largest components first (more stable)
        """
        strategy = self.config.placement_strategy.lower()

        if strategy == 'hub_spoke':
            return self._placement_order_hub_spoke(parts_db)
        elif strategy == 'criticality':
            return self._placement_order_criticality(parts_db)
        elif strategy == 'signal_flow':
            return self._placement_order_signal_flow(parts_db)
        elif strategy == 'size_based':
            return self._placement_order_size_based(parts_db)
        elif strategy == 'thermal':
            return self._placement_order_thermal(parts_db)
        elif strategy == 'auto':
            return self._placement_order_auto(parts_db)
        else:
            return self._placement_order_hub_spoke(parts_db)

    def _placement_order_hub_spoke(self, parts_db: Dict) -> List[str]:
        """
        Hub-Spoke ordering: Place the most connected component first,
        then its direct neighbors, then their neighbors, etc.

        This creates a radial placement pattern that minimizes
        total wirelength for designs with a clear "hub" component.
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Build adjacency: component -> set of connected components
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        connection_count: Dict[str, int] = defaultdict(int)

        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])
            components = set(p[0] for p in pins if len(p) >= 2)

            for comp in components:
                connection_count[comp] += 1
                for other in components:
                    if other != comp:
                        adjacency[comp].add(other)

        if not connection_count:
            return list(parts.keys())

        # Find hub (most connections)
        hub = max(connection_count.keys(), key=lambda c: connection_count[c])

        # BFS from hub
        order = []
        visited = set()
        queue = [hub]

        while queue:
            comp = queue.pop(0)
            if comp in visited:
                continue
            visited.add(comp)
            order.append(comp)

            # Add neighbors sorted by connection count (most connected first)
            neighbors = sorted(
                adjacency[comp] - visited,
                key=lambda c: connection_count.get(c, 0),
                reverse=True
            )
            queue.extend(neighbors)

        # Add any unconnected components at the end
        for comp in parts.keys():
            if comp not in visited:
                order.append(comp)

        return order

    def _placement_order_criticality(self, parts_db: Dict) -> List[str]:
        """
        Criticality ordering: Place timing-critical components first.

        Critical components include:
        - Oscillators/crystals
        - High-speed ICs
        - Components with critical net connections
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Calculate criticality score for each component
        criticality: Dict[str, float] = {}

        for ref, part in parts.items():
            score = 0.0

            # Check footprint for criticality indicators
            footprint = part.get('footprint', '').lower()
            if any(x in footprint for x in ['crystal', 'osc', 'xtal']):
                score += 10.0
            if any(x in footprint for x in ['qfp', 'bga', 'qfn']):
                score += 5.0  # Complex packages often critical

            # Check connected nets for criticality
            for net_name, net_data in nets.items():
                pins = net_data.get('pins', [])
                if any(p[0] == ref for p in pins if len(p) >= 2):
                    net_upper = net_name.upper()
                    if any(p in net_upper for p in self.config.critical_net_patterns):
                        score += 3.0
                    if 'CLK' in net_upper or 'CLOCK' in net_upper:
                        score += 5.0

            # More pins = potentially more critical
            pin_count = len(part.get('used_pins', part.get('physical_pins', [])))
            score += pin_count * 0.1

            criticality[ref] = score

        # Sort by criticality (highest first)
        return sorted(parts.keys(), key=lambda r: criticality.get(r, 0), reverse=True)

    def _placement_order_signal_flow(self, parts_db: Dict) -> List[str]:
        """
        Signal Flow ordering: Follow the signal path from input to output.

        This creates a left-to-right or top-to-bottom placement that
        matches the logical signal flow of the circuit.
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Identify input/output components
        # Inputs: connectors, sensors, regulators (power in)
        # Outputs: LEDs, motors, connectors (data out)

        input_score: Dict[str, float] = defaultdict(float)
        output_score: Dict[str, float] = defaultdict(float)

        for ref, part in parts.items():
            footprint = part.get('footprint', '').lower()
            value = part.get('value', '').lower()

            # Input indicators
            if any(x in footprint for x in ['connector', 'usb', 'jack', 'header']):
                input_score[ref] += 5.0
            if any(x in footprint for x in ['regulator', 'ldo', 'dcdc']):
                input_score[ref] += 3.0
            if any(x in value for x in ['sensor', 'input']):
                input_score[ref] += 3.0

            # Output indicators
            if any(x in footprint for x in ['led', 'display', 'motor']):
                output_score[ref] += 5.0
            if any(x in value for x in ['output', 'driver']):
                output_score[ref] += 3.0

        # Calculate flow position (0 = input, 1 = output)
        flow_position: Dict[str, float] = {}
        for ref in parts.keys():
            i_score = input_score.get(ref, 0)
            o_score = output_score.get(ref, 0)
            total = i_score + o_score
            if total > 0:
                flow_position[ref] = o_score / total
            else:
                flow_position[ref] = 0.5  # Middle

        # Sort by flow position (inputs first)
        return sorted(parts.keys(), key=lambda r: flow_position.get(r, 0.5))

    def _placement_order_size_based(self, parts_db: Dict) -> List[str]:
        """
        Size-Based ordering: Place largest components first.

        Large components are harder to move once placed, so placing
        them first provides a more stable foundation.
        """
        parts = parts_db.get('parts', {})

        def get_component_size(ref: str) -> float:
            part = parts.get(ref, {})

            # Try to get size from physical pins
            pins = part.get('physical_pins', part.get('used_pins', []))
            if not pins:
                return 0.0

            # Calculate bounding box from pin positions
            xs = []
            ys = []
            for pin in pins:
                offset = pin.get('offset', (0, 0))
                if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                    xs.append(offset[0])
                    ys.append(offset[1])

            if xs and ys:
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                return width * height

            # Fallback: use pin count as proxy
            return len(pins) * 0.5

        # Sort by size (largest first)
        return sorted(parts.keys(), key=get_component_size, reverse=True)

    def _placement_order_thermal(self, parts_db: Dict) -> List[str]:
        """
        Thermal ordering: Place heat-generating components first.

        This allows proper thermal consideration and spacing
        for components that dissipate significant power.
        """
        parts = parts_db.get('parts', {})

        def get_thermal_priority(ref: str) -> float:
            part = parts.get(ref, {})
            footprint = part.get('footprint', '').lower()
            value = part.get('value', '').lower()

            score = 0.0

            # Power components
            if any(x in footprint for x in ['to-220', 'to-252', 'dpak', 'd2pak']):
                score += 10.0
            if any(x in footprint for x in ['sot-223', 'sot-89']):
                score += 5.0
            if any(x in footprint for x in ['regulator', 'ldo', 'dcdc']):
                score += 8.0

            # Check value for power rating
            if any(x in value for x in ['1w', '2w', '5w', '10w']):
                score += 7.0

            # Large ICs (more pins = potentially more power)
            pins = part.get('physical_pins', part.get('used_pins', []))
            if len(pins) > 20:
                score += 3.0

            return score

        return sorted(parts.keys(), key=get_thermal_priority, reverse=True)

    def _placement_order_auto(self, parts_db: Dict) -> List[str]:
        """
        Auto ordering: Combine multiple strategies based on design analysis.

        Strategy:
        1. Identify hub component
        2. Place hub first
        3. Place critical components near hub
        4. Fill remaining by size/connectivity
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Get scores from multiple strategies
        hub_order = self._placement_order_hub_spoke(parts_db)
        critical_order = self._placement_order_criticality(parts_db)
        size_order = self._placement_order_size_based(parts_db)

        # Combine scores
        combined_score: Dict[str, float] = defaultdict(float)

        for i, ref in enumerate(hub_order):
            combined_score[ref] += (len(hub_order) - i) * 2.0  # Hub weight

        for i, ref in enumerate(critical_order):
            combined_score[ref] += (len(critical_order) - i) * 1.5  # Critical weight

        for i, ref in enumerate(size_order):
            combined_score[ref] += (len(size_order) - i) * 1.0  # Size weight

        return sorted(parts.keys(), key=lambda r: combined_score.get(r, 0), reverse=True)

    # =========================================================================
    # NET ROUTING ORDER
    # =========================================================================

    def compute_net_order(
        self,
        parts_db: Dict,
        placement: Optional[Dict] = None,
        algorithm: Optional[str] = None
    ) -> List[str]:
        """
        Compute net routing order using specified algorithm.

        This is an alias for get_net_order that allows specifying
        the algorithm directly instead of via config.

        Args:
            parts_db: Parts database with nets
            placement: Optional component placement
            algorithm: Algorithm name (short_first, long_first, critical_first,
                       bounding_box, pin_count, congestion, power_first,
                       signal_first, auto). If None, uses config default.

        Returns:
            List of net names in routing order
        """
        if algorithm:
            # Temporarily override the strategy
            old_strategy = self.config.net_order_strategy
            self.config.net_order_strategy = algorithm
            try:
                return self.get_net_order(parts_db, placement)
            finally:
                self.config.net_order_strategy = old_strategy
        return self.get_net_order(parts_db, placement)

    def get_net_order(self, parts_db: Dict, placement: Optional[Dict] = None) -> List[str]:
        """
        Determine optimal net routing order.

        Different strategies:
        - short_first: Route shortest nets first (less blocking)
        - long_first: Route longest nets first (more constrained)
        - critical_first: Route timing-critical nets first
        - power_first: Route power/ground first
        """
        strategy = self.config.net_order_strategy.lower()

        if strategy == 'short_first':
            return self._net_order_short_first(parts_db, placement)
        elif strategy == 'long_first':
            return self._net_order_long_first(parts_db, placement)
        elif strategy == 'critical_first':
            return self._net_order_critical_first(parts_db)
        elif strategy == 'bounding_box':
            return self._net_order_bounding_box(parts_db, placement)
        elif strategy == 'pin_count':
            return self._net_order_pin_count(parts_db)
        elif strategy == 'congestion':
            return self._net_order_congestion_aware(parts_db, placement)
        elif strategy == 'power_first':
            return self._net_order_power_first(parts_db)
        elif strategy == 'signal_first':
            return self._net_order_signal_first(parts_db)
        elif strategy == 'auto':
            return self._net_order_auto(parts_db, placement)
        else:
            return self._net_order_short_first(parts_db, placement)

    def _net_order_short_first(self, parts_db: Dict, placement: Optional[Dict]) -> List[str]:
        """
        Short-First ordering: Route shortest nets first.

        Rationale: Short nets are less likely to block other nets,
        so routing them first leaves more flexibility for longer nets.

        This is one of the most common strategies in sequential routing.
        """
        nets = parts_db.get('nets', {})

        def estimate_length(net_name: str) -> float:
            net_data = nets.get(net_name, {})
            pins = net_data.get('pins', [])

            if len(pins) < 2 or not placement:
                return float('inf')

            # Get pin positions
            positions = []
            for pin_ref in pins:
                comp, pin = self._parse_pin_ref(pin_ref)
                if comp in placement:
                    pos = placement[comp]
                    x = pos.x if hasattr(pos, 'x') else pos[0]
                    y = pos.y if hasattr(pos, 'y') else pos[1]
                    positions.append((x, y))

            if len(positions) < 2:
                return float('inf')

            # Calculate MST length estimate
            total = 0
            remaining = list(positions[1:])
            connected = [positions[0]]

            while remaining:
                best = float('inf')
                best_pt = None
                for r in remaining:
                    for c in connected:
                        d = abs(r[0] - c[0]) + abs(r[1] - c[1])
                        if d < best:
                            best = d
                            best_pt = r
                if best_pt:
                    total += best
                    connected.append(best_pt)
                    remaining.remove(best_pt)
                else:
                    break

            return total

        net_names = [n for n in nets.keys() if len(nets[n].get('pins', [])) >= 2]
        return sorted(net_names, key=estimate_length)

    def _net_order_long_first(self, parts_db: Dict, placement: Optional[Dict]) -> List[str]:
        """
        Long-First ordering: Route longest nets first.

        Rationale: Long nets are more constrained and harder to route,
        so giving them priority ensures they get routed successfully.
        """
        short_first = self._net_order_short_first(parts_db, placement)
        return list(reversed(short_first))

    def _net_order_critical_first(self, parts_db: Dict) -> List[str]:
        """
        Critical-First ordering: Route timing-critical nets first.

        Critical nets include clock, high-speed data, and other
        timing-sensitive signals that need optimal routing.
        """
        nets = parts_db.get('nets', {})

        def get_criticality(net_name: str) -> float:
            score = 0.0
            net_upper = net_name.upper()

            # Clock signals - highest priority
            if 'CLK' in net_upper or 'CLOCK' in net_upper:
                score += 100.0

            # High-speed data signals
            for pattern in self.config.critical_net_patterns:
                if pattern.upper() in net_upper:
                    score += 50.0
                    break

            # Differential pairs
            if net_upper.endswith('+') or net_upper.endswith('-'):
                score += 30.0
            if '_P' in net_upper or '_N' in net_upper:
                score += 30.0

            return score

        net_names = [n for n in nets.keys() if len(nets[n].get('pins', [])) >= 2]
        return sorted(net_names, key=get_criticality, reverse=True)

    def _net_order_bounding_box(self, parts_db: Dict, placement: Optional[Dict]) -> List[str]:
        """
        Bounding-Box ordering: Route by bounding box area.

        Nets with smaller bounding boxes are routed first as they
        are more localized and less likely to cause congestion.
        """
        nets = parts_db.get('nets', {})

        def get_bounding_box_area(net_name: str) -> float:
            net_data = nets.get(net_name, {})
            pins = net_data.get('pins', [])

            if len(pins) < 2 or not placement:
                return float('inf')

            xs, ys = [], []
            for pin_ref in pins:
                comp, pin = self._parse_pin_ref(pin_ref)
                if comp in placement:
                    pos = placement[comp]
                    x = pos.x if hasattr(pos, 'x') else pos[0]
                    y = pos.y if hasattr(pos, 'y') else pos[1]
                    xs.append(x)
                    ys.append(y)

            if not xs or not ys:
                return float('inf')

            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            return width * height

        net_names = [n for n in nets.keys() if len(nets[n].get('pins', [])) >= 2]
        return sorted(net_names, key=get_bounding_box_area)

    def _net_order_pin_count(self, parts_db: Dict) -> List[str]:
        """
        Pin-Count ordering: Route by number of pins.

        Nets with fewer pins are routed first (simpler to route).
        Multi-pin nets are routed later with more context.
        """
        nets = parts_db.get('nets', {})

        def get_pin_count(net_name: str) -> int:
            net_data = nets.get(net_name, {})
            return len(net_data.get('pins', []))

        net_names = [n for n in nets.keys() if len(nets[n].get('pins', [])) >= 2]
        return sorted(net_names, key=get_pin_count)

    def _net_order_congestion_aware(self, parts_db: Dict, placement: Optional[Dict]) -> List[str]:
        """
        Congestion-Aware ordering: Consider routing congestion.

        Nets that pass through congested areas are routed first
        to ensure they can find a path before the area fills up.
        """
        nets = parts_db.get('nets', {})

        if not placement:
            return self._net_order_short_first(parts_db, placement)

        # Build congestion map (simplified grid)
        grid_size = 5.0  # mm
        congestion: Dict[Tuple[int, int], int] = defaultdict(int)

        # Count how many net bounding boxes pass through each grid cell
        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])
            xs, ys = [], []

            for pin_ref in pins:
                comp, pin = self._parse_pin_ref(pin_ref)
                if comp in placement:
                    pos = placement[comp]
                    x = pos.x if hasattr(pos, 'x') else pos[0]
                    y = pos.y if hasattr(pos, 'y') else pos[1]
                    xs.append(x)
                    ys.append(y)

            if len(xs) >= 2:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                for gx in range(int(min_x / grid_size), int(max_x / grid_size) + 1):
                    for gy in range(int(min_y / grid_size), int(max_y / grid_size) + 1):
                        congestion[(gx, gy)] += 1

        # Calculate congestion score for each net
        def get_congestion_score(net_name: str) -> float:
            net_data = nets.get(net_name, {})
            pins = net_data.get('pins', [])

            if len(pins) < 2:
                return 0.0

            xs, ys = [], []
            for pin_ref in pins:
                comp, pin = self._parse_pin_ref(pin_ref)
                if comp in placement:
                    pos = placement[comp]
                    x = pos.x if hasattr(pos, 'x') else pos[0]
                    y = pos.y if hasattr(pos, 'y') else pos[1]
                    xs.append(x)
                    ys.append(y)

            if len(xs) < 2:
                return 0.0

            # Sum congestion in bounding box
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            total_congestion = 0
            for gx in range(int(min_x / grid_size), int(max_x / grid_size) + 1):
                for gy in range(int(min_y / grid_size), int(max_y / grid_size) + 1):
                    total_congestion += congestion[(gx, gy)]

            return total_congestion

        # Route high-congestion nets first
        net_names = [n for n in nets.keys() if len(nets[n].get('pins', [])) >= 2]
        return sorted(net_names, key=get_congestion_score, reverse=True)

    def _net_order_power_first(self, parts_db: Dict) -> List[str]:
        """
        Power-First ordering: Route power and ground nets first.

        Power nets often need wider traces and may use planes,
        so routing them first establishes the power distribution.
        """
        nets = parts_db.get('nets', {})

        def is_power_net(net_name: str) -> int:
            net_upper = net_name.upper()
            for pattern in self.config.power_net_patterns:
                if pattern.upper() in net_upper:
                    if 'GND' in net_upper:
                        return 2  # Ground highest priority
                    return 1  # Other power
            return 0  # Not power

        net_names = [n for n in nets.keys() if len(nets[n].get('pins', [])) >= 2]
        return sorted(net_names, key=is_power_net, reverse=True)

    def _net_order_signal_first(self, parts_db: Dict) -> List[str]:
        """
        Signal-First ordering: Route signal nets before power.

        Signal nets are more routing-constrained, so give them
        priority. Power can use planes or wider traces later.
        """
        power_first = self._net_order_power_first(parts_db)
        return list(reversed(power_first))

    def _net_order_auto(self, parts_db: Dict, placement: Optional[Dict]) -> List[str]:
        """
        Auto ordering: Combine multiple strategies.

        Strategy:
        1. Critical nets first (clock, high-speed)
        2. Short nets (easy wins)
        3. Power/ground (can use planes)
        4. Remaining by congestion
        """
        nets = parts_db.get('nets', {})

        # Get all orderings
        critical = self._net_order_critical_first(parts_db)
        short = self._net_order_short_first(parts_db, placement)
        power = self._net_order_power_first(parts_db)

        # Build combined priority
        priority: Dict[str, float] = {}

        for i, net in enumerate(critical):
            priority[net] = priority.get(net, 0) + (len(critical) - i) * 3.0

        for i, net in enumerate(short):
            priority[net] = priority.get(net, 0) + (len(short) - i) * 1.0

        # Power nets get medium priority
        for i, net in enumerate(power):
            net_upper = net.upper()
            is_power = any(p.upper() in net_upper for p in self.config.power_net_patterns)
            if is_power:
                priority[net] = priority.get(net, 0) + (len(power) - i) * 0.5

        net_names = [n for n in nets.keys() if len(nets[n].get('pins', [])) >= 2]
        return sorted(net_names, key=lambda n: priority.get(n, 0), reverse=True)

    # =========================================================================
    # LAYER ASSIGNMENT
    # =========================================================================

    def get_layer_assignment(self, parts_db: Dict) -> Dict[str, str]:
        """
        Determine layer assignment for each net.

        Returns a mapping of net_name -> preferred layer.
        """
        strategy = self.config.layer_strategy.lower()

        if strategy == 'signal_integrity':
            return self._layer_assign_signal_integrity(parts_db)
        elif strategy == 'crosstalk_min':
            return self._layer_assign_crosstalk_min(parts_db)
        elif strategy == 'via_minimize':
            return self._layer_assign_via_minimize(parts_db)
        elif strategy == 'power_ground_sep':
            return self._layer_assign_power_ground(parts_db)
        elif strategy == 'alternating':
            return self._layer_assign_alternating(parts_db)
        elif strategy == 'auto':
            return self._layer_assign_auto(parts_db)
        else:
            return self._layer_assign_auto(parts_db)

    def _layer_assign_signal_integrity(self, parts_db: Dict) -> Dict[str, str]:
        """
        Signal Integrity assignment: Separate high-speed from low-speed.

        High-speed signals go on outer layers (better impedance control).
        Low-speed signals can use inner layers.
        """
        nets = parts_db.get('nets', {})
        assignments: Dict[str, str] = {}

        for net_name in nets.keys():
            net_upper = net_name.upper()

            # High-speed signals on top layer
            if any(p in net_upper for p in ['CLK', 'CLOCK', 'USB', 'ETH', 'HDMI']):
                assignments[net_name] = self.config.top_layer_name
            # Power on bottom or inner
            elif any(p.upper() in net_upper for p in self.config.power_net_patterns):
                if self.config.num_layers > 2 and self.config.inner_layer_names:
                    assignments[net_name] = self.config.inner_layer_names[0]
                else:
                    assignments[net_name] = self.config.bottom_layer_name
            else:
                # Default to top layer
                assignments[net_name] = self.config.top_layer_name

        return assignments

    def _layer_assign_crosstalk_min(self, parts_db: Dict) -> Dict[str, str]:
        """
        Crosstalk Minimization: Orthogonal routing on adjacent layers.

        Signals on adjacent layers should run perpendicular to minimize
        capacitive coupling (crosstalk).
        """
        nets = parts_db.get('nets', {})
        assignments: Dict[str, str] = {}

        # For 2-layer: horizontal on top, vertical on bottom
        # This is a simplified approach
        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])

            if len(pins) < 2:
                assignments[net_name] = self.config.top_layer_name
                continue

            # Estimate if net runs more horizontal or vertical
            # (would need placement info for accurate estimation)
            # Default alternating assignment
            net_index = list(nets.keys()).index(net_name)
            if net_index % 2 == 0:
                assignments[net_name] = self.config.top_layer_name
            else:
                assignments[net_name] = self.config.bottom_layer_name

        return assignments

    def _layer_assign_via_minimize(self, parts_db: Dict) -> Dict[str, str]:
        """
        Via Minimization: Assign layers to minimize layer changes.

        Based on research: "Via minimization in VLSI routing" (IEEE).
        Uses graph coloring approach for conflict resolution.
        """
        nets = parts_db.get('nets', {})

        # Simple approach: keep all nets on top layer if possible
        # Only use bottom layer when conflicts arise
        assignments: Dict[str, str] = {}

        for net_name in nets.keys():
            net_upper = net_name.upper()

            # Power/ground often benefit from dedicated layer
            if 'GND' in net_upper:
                assignments[net_name] = self.config.bottom_layer_name
            elif any(p.upper() in net_upper for p in self.config.power_net_patterns):
                assignments[net_name] = self.config.bottom_layer_name
            else:
                # Signal nets on top to minimize vias
                assignments[net_name] = self.config.top_layer_name

        return assignments

    def _layer_assign_power_ground(self, parts_db: Dict) -> Dict[str, str]:
        """
        Power/Ground Separation: Dedicated layers for power.

        In multi-layer boards, use inner layers as power planes.
        Outer layers for signals.
        """
        nets = parts_db.get('nets', {})
        assignments: Dict[str, str] = {}

        for net_name in nets.keys():
            net_upper = net_name.upper()

            if 'GND' in net_upper:
                if self.config.num_layers > 2 and len(self.config.inner_layer_names) >= 1:
                    assignments[net_name] = self.config.inner_layer_names[0]  # GND plane
                else:
                    assignments[net_name] = self.config.bottom_layer_name
            elif any(p.upper() in net_upper for p in self.config.power_net_patterns):
                if self.config.num_layers > 2 and len(self.config.inner_layer_names) >= 2:
                    assignments[net_name] = self.config.inner_layer_names[1]  # Power plane
                else:
                    assignments[net_name] = self.config.bottom_layer_name
            else:
                assignments[net_name] = self.config.top_layer_name

        return assignments

    def _layer_assign_alternating(self, parts_db: Dict) -> Dict[str, str]:
        """
        Alternating assignment: Alternate nets between layers.

        Simple load balancing approach.
        """
        nets = parts_db.get('nets', {})
        assignments: Dict[str, str] = {}

        layers = [self.config.top_layer_name, self.config.bottom_layer_name]
        if self.config.inner_layer_names:
            layers.extend(self.config.inner_layer_names)

        for i, net_name in enumerate(nets.keys()):
            assignments[net_name] = layers[i % len(layers)]

        return assignments

    def _layer_assign_auto(self, parts_db: Dict) -> Dict[str, str]:
        """
        Auto assignment: Combine strategies based on net type.

        - Critical/high-speed: Top layer
        - Power: Bottom or inner
        - Ground: Dedicated if available
        - Signals: Balance between layers
        """
        nets = parts_db.get('nets', {})
        assignments: Dict[str, str] = {}

        signal_count = {'top': 0, 'bottom': 0}

        for net_name in nets.keys():
            net_upper = net_name.upper()

            # Ground
            if 'GND' in net_upper:
                if self.config.num_layers > 2 and self.config.inner_layer_names:
                    assignments[net_name] = self.config.inner_layer_names[0]
                else:
                    assignments[net_name] = self.config.bottom_layer_name
                continue

            # Power
            if any(p.upper() in net_upper for p in self.config.power_net_patterns):
                assignments[net_name] = self.config.bottom_layer_name
                continue

            # Critical signals - top layer
            if any(p in net_upper for p in ['CLK', 'CLOCK']):
                assignments[net_name] = self.config.top_layer_name
                signal_count['top'] += 1
                continue

            # Balance remaining signals
            if signal_count['top'] <= signal_count['bottom']:
                assignments[net_name] = self.config.top_layer_name
                signal_count['top'] += 1
            else:
                assignments[net_name] = self.config.bottom_layer_name
                signal_count['bottom'] += 1

        return assignments

    # =========================================================================
    # VIA STRATEGY
    # =========================================================================

    def get_via_strategy(self, parts_db: Dict, placement: Optional[Dict] = None
                         ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Determine via placement strategy for each net.

        Returns recommended via positions (if any) for each net.
        """
        # This is a simplified implementation
        # Full via optimization would require actual routing paths

        nets = parts_db.get('nets', {})
        via_recommendations: Dict[str, List[Tuple[float, float]]] = {}

        layer_assignments = self.get_layer_assignment(parts_db)

        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])

            if len(pins) < 2:
                via_recommendations[net_name] = []
                continue

            # Check if pins are on different components that might need layer change
            via_recommendations[net_name] = []

            # For power nets, recommend via fanout from central point
            net_upper = net_name.upper()
            if any(p.upper() in net_upper for p in self.config.power_net_patterns):
                if placement:
                    # Find centroid of net
                    xs, ys = [], []
                    for pin_ref in pins:
                        comp, pin = self._parse_pin_ref(pin_ref)
                        if comp in placement:
                            pos = placement[comp]
                            x = pos.x if hasattr(pos, 'x') else pos[0]
                            y = pos.y if hasattr(pos, 'y') else pos[1]
                            xs.append(x)
                            ys.append(y)

                    if xs and ys:
                        cx = sum(xs) / len(xs)
                        cy = sum(ys) / len(ys)
                        # Recommend via at centroid for power distribution
                        via_recommendations[net_name] = [(cx, cy)]

        return via_recommendations

    # =========================================================================
    # PIN ESCAPE ORDER
    # =========================================================================

    def get_pin_escape_order(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Determine pin escape order for each component.

        Returns a mapping of component_ref -> ordered list of pin numbers.

        Different strategies:
        - outer_first: Escape outer pins first (reduces crossing for BGA/QFP)
        - inner_first: Escape inner pins first (for critical signals)
        - clockwise: Spiral pattern starting from top-right
        - net_priority: Order by net criticality
        - congestion_aware: Consider local routing congestion
        """
        strategy = self.config.pin_escape_strategy.lower()

        if strategy == 'outer_first':
            return self._pin_escape_outer_first(parts_db)
        elif strategy == 'inner_first':
            return self._pin_escape_inner_first(parts_db)
        elif strategy == 'clockwise':
            return self._pin_escape_clockwise(parts_db)
        elif strategy == 'counter_cw':
            return self._pin_escape_counter_clockwise(parts_db)
        elif strategy == 'net_priority':
            return self._pin_escape_net_priority(parts_db)
        elif strategy == 'congestion':
            return self._pin_escape_congestion_aware(parts_db)
        elif strategy == 'quadrant':
            return self._pin_escape_quadrant(parts_db)
        elif strategy == 'side_by_side':
            return self._pin_escape_side_by_side(parts_db)
        elif strategy == 'auto':
            return self._pin_escape_auto(parts_db)
        else:
            return self._pin_escape_outer_first(parts_db)

    def _get_pin_positions(self, part: Dict) -> Dict[str, Tuple[float, float]]:
        """Extract pin positions from a part"""
        positions = {}
        pins = part.get('physical_pins', part.get('used_pins', []))

        for pin in pins:
            pin_num = str(pin.get('number', pin.get('pin', '')))
            offset = pin.get('offset', (0, 0))

            if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                positions[pin_num] = (float(offset[0]), float(offset[1]))
            else:
                # Try physical sub-dict
                phys = pin.get('physical', {})
                ox = phys.get('offset_x', 0)
                oy = phys.get('offset_y', 0)
                positions[pin_num] = (float(ox), float(oy))

        return positions

    def _get_pin_net_map(self, ref: str, parts_db: Dict) -> Dict[str, str]:
        """Get mapping of pin number to net name for a component"""
        nets = parts_db.get('nets', {})
        pin_net = {}

        for net_name, net_data in nets.items():
            for pin_ref in net_data.get('pins', []):
                comp, pin = self._parse_pin_ref(pin_ref)
                if comp == ref:
                    pin_net[str(pin)] = net_name

        return pin_net

    def _classify_pin_position(self, x: float, y: float, all_positions: Dict[str, Tuple[float, float]]
                               ) -> Tuple[bool, int, str]:
        """
        Classify a pin's position relative to component center.

        Returns:
            (is_outer, quadrant, side)
            - is_outer: True if pin is on outer ring
            - quadrant: 0=NE, 1=SE, 2=SW, 3=NW
            - side: 'top', 'bottom', 'left', 'right'
        """
        if not all_positions:
            return True, 0, 'top'

        # Calculate bounds
        xs = [p[0] for p in all_positions.values()]
        ys = [p[1] for p in all_positions.values()]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Determine if outer (on edge)
        tolerance = 0.1
        is_outer = (abs(x - min_x) < tolerance or abs(x - max_x) < tolerance or
                    abs(y - min_y) < tolerance or abs(y - max_y) < tolerance)

        # Determine quadrant
        if x >= center_x and y >= center_y:
            quadrant = 0  # NE (top-right)
        elif x >= center_x and y < center_y:
            quadrant = 1  # SE (bottom-right)
        elif x < center_x and y < center_y:
            quadrant = 2  # SW (bottom-left)
        else:
            quadrant = 3  # NW (top-left)

        # Determine side
        dx_left = abs(x - min_x)
        dx_right = abs(x - max_x)
        dy_top = abs(y - max_y)
        dy_bottom = abs(y - min_y)

        min_dist = min(dx_left, dx_right, dy_top, dy_bottom)
        if min_dist == dy_top:
            side = 'top'
        elif min_dist == dy_bottom:
            side = 'bottom'
        elif min_dist == dx_left:
            side = 'left'
        else:
            side = 'right'

        return is_outer, quadrant, side

    def _pin_escape_outer_first(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Outer-First ordering: Escape outer pins first.

        For BGA and QFP packages, outer pins are easier to escape
        and don't block inner pins. This is the standard approach
        for high pin-count packages.

        Reference: "Escape routing for dense pin clusters" (DAC papers)
        """
        parts = parts_db.get('parts', {})
        result = {}

        for ref, part in parts.items():
            positions = self._get_pin_positions(part)

            if not positions:
                result[ref] = []
                continue

            # Calculate distance from center for each pin
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)

            def distance_from_center(pin_num: str) -> float:
                x, y = positions[pin_num]
                return math.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Sort by distance from center (furthest = outer first)
            pin_list = sorted(positions.keys(), key=distance_from_center, reverse=True)
            result[ref] = pin_list

        return result

    def _pin_escape_inner_first(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Inner-First ordering: Escape inner pins first.

        Useful when inner pins carry critical signals that need
        priority routing. Less common but useful for specific designs.
        """
        outer_first = self._pin_escape_outer_first(parts_db)
        return {ref: list(reversed(pins)) for ref, pins in outer_first.items()}

    def _pin_escape_clockwise(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Clockwise ordering: Spiral pattern from top-right.

        Creates a consistent escape pattern that follows the
        component perimeter in a clockwise direction.
        """
        parts = parts_db.get('parts', {})
        result = {}

        for ref, part in parts.items():
            positions = self._get_pin_positions(part)

            if not positions:
                result[ref] = []
                continue

            # Calculate center
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)

            def angle_from_center(pin_num: str) -> float:
                x, y = positions[pin_num]
                # Angle from positive x-axis, measured clockwise
                angle = math.atan2(-(y - center_y), x - center_x)
                return angle

            # Sort by angle (clockwise from east)
            pin_list = sorted(positions.keys(), key=angle_from_center)
            result[ref] = pin_list

        return result

    def _pin_escape_counter_clockwise(self, parts_db: Dict) -> Dict[str, List[str]]:
        """Counter-clockwise ordering: Reverse of clockwise"""
        clockwise = self._pin_escape_clockwise(parts_db)
        return {ref: list(reversed(pins)) for ref, pins in clockwise.items()}

    def _pin_escape_net_priority(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Net-Priority ordering: Escape by net criticality.

        Pins connected to critical nets (clock, high-speed data)
        are escaped first to give them priority routing paths.
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})
        result = {}

        for ref, part in parts.items():
            positions = self._get_pin_positions(part)
            pin_net = self._get_pin_net_map(ref, parts_db)

            if not positions:
                result[ref] = []
                continue

            def get_net_priority(pin_num: str) -> float:
                net_name = pin_net.get(pin_num, '')
                net_upper = net_name.upper()
                priority = 0.0

                # Clock signals - highest priority
                if 'CLK' in net_upper or 'CLOCK' in net_upper:
                    priority += 100.0

                # Critical data signals
                for pattern in self.config.critical_net_patterns:
                    if pattern.upper() in net_upper:
                        priority += 50.0
                        break

                # Power/ground - lower priority (can use planes)
                if any(p.upper() in net_upper for p in self.config.power_net_patterns):
                    priority -= 20.0

                # Unconnected pins - lowest priority
                if not net_name:
                    priority -= 100.0

                return priority

            pin_list = sorted(positions.keys(), key=get_net_priority, reverse=True)
            result[ref] = pin_list

        return result

    def _pin_escape_congestion_aware(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Congestion-Aware ordering: Consider local routing congestion.

        Pins in less congested areas are escaped first to build up
        routing infrastructure before tackling congested areas.
        """
        # Simplified: combine outer-first with net priority
        outer = self._pin_escape_outer_first(parts_db)
        net_priority = self._pin_escape_net_priority(parts_db)

        result = {}
        for ref in outer.keys():
            outer_pins = outer.get(ref, [])
            priority_pins = net_priority.get(ref, [])

            # Interleave: outer pins from high-priority nets first
            priority_set = set(priority_pins[:len(priority_pins)//2])

            high_priority_outer = [p for p in outer_pins if p in priority_set]
            low_priority_outer = [p for p in outer_pins if p not in priority_set]

            result[ref] = high_priority_outer + low_priority_outer

        return result

    def _pin_escape_quadrant(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Quadrant ordering: Group pins by quadrant and escape each group.

        Escape order: NE -> SE -> SW -> NW (clockwise from top-right).
        Within each quadrant, outer pins first.
        """
        parts = parts_db.get('parts', {})
        result = {}

        for ref, part in parts.items():
            positions = self._get_pin_positions(part)

            if not positions:
                result[ref] = []
                continue

            # Classify each pin
            pin_quadrants: Dict[int, List[Tuple[str, float]]] = {0: [], 1: [], 2: [], 3: []}

            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)

            for pin_num, (x, y) in positions.items():
                _, quadrant, _ = self._classify_pin_position(x, y, positions)
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                pin_quadrants[quadrant].append((pin_num, dist))

            # Sort each quadrant by distance (outer first)
            ordered = []
            for q in [0, 1, 2, 3]:  # NE, SE, SW, NW
                sorted_pins = sorted(pin_quadrants[q], key=lambda x: x[1], reverse=True)
                ordered.extend([p[0] for p in sorted_pins])

            result[ref] = ordered

        return result

    def _pin_escape_side_by_side(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Side-by-Side ordering: Escape each side sequentially.

        Order: top -> right -> bottom -> left.
        Within each side, escape from outside corners toward center.
        """
        parts = parts_db.get('parts', {})
        result = {}

        for ref, part in parts.items():
            positions = self._get_pin_positions(part)

            if not positions:
                result[ref] = []
                continue

            # Classify pins by side
            sides: Dict[str, List[Tuple[str, float]]] = {
                'top': [], 'right': [], 'bottom': [], 'left': []
            }

            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            center_x = sum(xs) / len(xs)

            for pin_num, (x, y) in positions.items():
                _, _, side = self._classify_pin_position(x, y, positions)
                # Sort key: distance from center of that side
                dist_from_center = abs(x - center_x) if side in ['top', 'bottom'] else abs(y - sum(ys)/len(ys))
                sides[side].append((pin_num, dist_from_center))

            # Order: top, right, bottom, left
            # Within each side, from ends toward center
            ordered = []
            for side_name in ['top', 'right', 'bottom', 'left']:
                sorted_pins = sorted(sides[side_name], key=lambda x: x[1], reverse=True)
                ordered.extend([p[0] for p in sorted_pins])

            result[ref] = ordered

        return result

    def _pin_escape_auto(self, parts_db: Dict) -> Dict[str, List[str]]:
        """
        Auto ordering: Select strategy based on component type.

        - BGA/QFP (many pins): outer-first
        - SOT/small packages: net-priority
        - Connectors: side-by-side
        """
        parts = parts_db.get('parts', {})
        result = {}

        # Get all strategies
        outer_first = self._pin_escape_outer_first(parts_db)
        net_priority = self._pin_escape_net_priority(parts_db)
        side_by_side = self._pin_escape_side_by_side(parts_db)

        for ref, part in parts.items():
            footprint = part.get('footprint', '').lower()
            pins = part.get('physical_pins', part.get('used_pins', []))
            pin_count = len(pins)

            # Select strategy based on package type
            if any(x in footprint for x in ['bga', 'qfp', 'qfn', 'lqfp', 'tqfp']):
                # High pin-count packages: outer-first
                result[ref] = outer_first.get(ref, [])
            elif any(x in footprint for x in ['connector', 'header', 'pin_header', 'usb']):
                # Connectors: side-by-side
                result[ref] = side_by_side.get(ref, [])
            elif pin_count <= 8:
                # Small packages: net-priority
                result[ref] = net_priority.get(ref, [])
            else:
                # Default: outer-first
                result[ref] = outer_first.get(ref, [])

        return result

    # =========================================================================
    # ANALYSIS HELPERS
    # =========================================================================

    def _analyze_components(self, parts_db: Dict):
        """Analyze components and build component info"""
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        self.component_info.clear()

        for ref, part in parts.items():
            pins = part.get('used_pins', part.get('physical_pins', []))

            # Count unique nets
            net_set = set()
            for net_name, net_data in nets.items():
                net_pins = net_data.get('pins', [])
                if any(p[0] == ref for p in net_pins if len(p) >= 2):
                    net_set.add(net_name)

            # Calculate component area
            xs, ys = [], []
            for pin in pins:
                offset = pin.get('offset', (0, 0))
                if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                    xs.append(offset[0])
                    ys.append(offset[1])

            area = 0.0
            if xs and ys:
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))

            self.component_info[ref] = ComponentInfo(
                ref=ref,
                pin_count=len(pins),
                net_count=len(net_set),
                area=area
            )

    def _analyze_nets(self, parts_db: Dict, placement: Optional[Dict] = None):
        """Analyze nets and build net info"""
        nets = parts_db.get('nets', {})

        self.net_info.clear()

        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])

            # Classify net type
            net_upper = net_name.upper()
            is_power = any(p.upper() in net_upper for p in self.config.power_net_patterns)
            is_ground = 'GND' in net_upper
            is_critical = any(p in net_upper for p in ['CLK', 'CLOCK'])

            # Estimate length
            length = 0.0
            bb_area = 0.0

            if placement and len(pins) >= 2:
                xs, ys = [], []
                for pin_ref in pins:
                    comp, _ = self._parse_pin_ref(pin_ref)
                    if comp in placement:
                        pos = placement[comp]
                        x = pos.x if hasattr(pos, 'x') else pos[0]
                        y = pos.y if hasattr(pos, 'y') else pos[1]
                        xs.append(x)
                        ys.append(y)

                if len(xs) >= 2:
                    bb_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    # MST estimate
                    remaining = list(zip(xs[1:], ys[1:]))
                    connected = [(xs[0], ys[0])]
                    while remaining:
                        best = float('inf')
                        best_pt = None
                        for r in remaining:
                            for c in connected:
                                d = abs(r[0] - c[0]) + abs(r[1] - c[1])
                                if d < best:
                                    best = d
                                    best_pt = r
                        if best_pt:
                            length += best
                            connected.append(best_pt)
                            remaining.remove(best_pt)
                        else:
                            break

            self.net_info[net_name] = NetInfo(
                name=net_name,
                pins=pins,
                pin_count=len(pins),
                estimated_length=length,
                bounding_box_area=bb_area,
                is_power=is_power,
                is_ground=is_ground,
                is_critical=is_critical
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_order_piston(
    placement_strategy: str = 'auto',
    net_order_strategy: str = 'auto',
    layer_strategy: str = 'auto',
    num_layers: int = 2
) -> OrderPiston:
    """
    Factory function to create an order piston with common settings.
    """
    config = OrderConfig(
        placement_strategy=placement_strategy,
        net_order_strategy=net_order_strategy,
        layer_strategy=layer_strategy,
        num_layers=num_layers
    )
    return OrderPiston(config)


# =============================================================================
# MODULE INFO
# =============================================================================

ORDERING_STRATEGIES = {
    'placement': {
        'hub_spoke': 'Place most connected component first, then neighbors (radial pattern)',
        'criticality': 'Place timing-critical components first',
        'signal_flow': 'Follow signal path from input to output',
        'size_based': 'Place largest components first (stability)',
        'thermal': 'Place heat-generating components first',
        'auto': 'Combine strategies based on design analysis'
    },
    'net_routing': {
        'short_first': 'Route shortest nets first (less blocking)',
        'long_first': 'Route longest nets first (more constrained)',
        'critical_first': 'Route timing-critical nets first',
        'bounding_box': 'Route by bounding box area (localized first)',
        'pin_count': 'Route by number of pins (simple first)',
        'congestion': 'Consider routing congestion (busy areas first)',
        'power_first': 'Route power/ground first',
        'signal_first': 'Route signals before power',
        'auto': 'Combine: critical → short → power'
    },
    'layer_assignment': {
        'signal_integrity': 'High-speed on outer layers',
        'crosstalk_min': 'Orthogonal routing on adjacent layers',
        'via_minimize': 'Minimize layer changes',
        'power_ground_sep': 'Dedicated layers for power',
        'alternating': 'Alternate between layers',
        'auto': 'Combine based on net type'
    },
    'via': {
        'minimize': 'Minimize total via count',
        'cluster': 'Group vias together',
        'fanout': 'Spread vias for thermal',
        'balanced': 'Balance via distribution',
        'auto': 'Based on net type'
    },
    'pin_escape': {
        'outer_first': 'Escape outer pins first (BGA/QFP, reduces crossing)',
        'inner_first': 'Escape inner pins first (critical signals priority)',
        'clockwise': 'Spiral clockwise from top-right corner',
        'counter_cw': 'Spiral counter-clockwise',
        'net_priority': 'Order by net criticality (clock/high-speed first)',
        'congestion': 'Consider local routing congestion',
        'quadrant': 'Group by quadrant, escape NE→SE→SW→NW',
        'side_by_side': 'Escape each side: top→right→bottom→left',
        'auto': 'Select based on package type (BGA:outer, connector:side)'
    }
}
