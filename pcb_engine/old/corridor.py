"""
PCB Engine - Corridor Module (Enhanced)
========================================

Validates corridor capacity BEFORE routing.
This is Phase 5 of the algorithm.

CRITICAL LESSON LEARNED:
========================
If corridor capacity < trace demand, routing WILL fail.
The previous design had 16 traces trying to pass through corridors
that could only fit 8-10. This was detected AFTER routing failed,
wasting time on impossible routes.

THE FIX: Calculate corridor capacity BEFORE routing.
If any corridor is over-capacity, return to placement phase.

KEY PRINCIPLE:
==============
CAPACITY_CHECK_FIRST → ROUTE_SECOND
Never start routing if corridors are already overloaded.
"""

from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Corridor:
    """A routing corridor between components"""
    name: str
    bounds: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    direction: str  # 'horizontal' or 'vertical'
    width: float    # Available width for traces (perpendicular to flow)
    length: float   # Length of corridor (parallel to flow)
    capacity: int   # How many traces can fit
    demand: int     # How many traces need to pass through
    utilization: float = 0.0

    # Nets passing through this corridor
    nets: List[str] = field(default_factory=list)

    # Source and destination areas
    source_area: str = ''
    dest_area: str = ''

    # Analysis data
    bottleneck_severity: float = 0.0  # 0 = fine, 1+ = over capacity

    @property
    def shortfall(self) -> int:
        """How many traces over capacity"""
        return max(0, self.demand - self.capacity)

    @property
    def is_overcapacity(self) -> bool:
        """Is this corridor overloaded?"""
        return self.demand > self.capacity

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of corridor"""
        return (
            (self.bounds[0] + self.bounds[2]) / 2,
            (self.bounds[1] + self.bounds[3]) / 2
        )


@dataclass
class CorridorRegion:
    """A region that forms a corridor between components"""
    name: str
    bounds: Tuple[float, float, float, float]
    components_left: Set[str] = field(default_factory=set)
    components_right: Set[str] = field(default_factory=set)
    direction: str = 'vertical'  # 'horizontal' or 'vertical'


@dataclass
class BottleneckAnalysis:
    """Analysis of a routing bottleneck"""
    corridor: Corridor
    severity: float  # demand / capacity ratio
    suggested_fixes: List[str]
    affected_nets: List[str]


# =============================================================================
# CAPACITY CALCULATOR
# =============================================================================

class CapacityCalculator:
    """Calculates corridor capacities"""

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules

    def calculate_capacity(self, width: float, layers_available: int = 1) -> int:
        """
        Calculate how many traces can fit in a corridor.

        Args:
            width: Available width perpendicular to trace direction (mm)
            layers_available: Number of routing layers (usually 1 for F.Cu only)

        Returns:
            Number of traces that can fit
        """
        # Trace pitch = trace width + edge-to-edge clearance on both sides
        trace_pitch = self.rules.min_trace_width + 2 * self.rules.min_clearance

        # Round up to grid alignment
        pitch_gridded = math.ceil(trace_pitch / self.board.grid_size) * self.board.grid_size

        # Available width minus edge clearances
        available = width - 2 * self.rules.min_clearance

        if available <= 0:
            return 0

        # Capacity per layer
        capacity_per_layer = int(available / pitch_gridded)

        # Total capacity
        total_capacity = capacity_per_layer * layers_available

        return max(0, total_capacity)

    def calculate_min_width_for_traces(self, num_traces: int, layers: int = 1) -> float:
        """
        Calculate minimum corridor width needed for N traces.

        Args:
            num_traces: Number of traces needed
            layers: Number of routing layers

        Returns:
            Minimum width in mm
        """
        traces_per_layer = math.ceil(num_traces / layers)

        trace_pitch = self.rules.min_trace_width + 2 * self.rules.min_clearance
        pitch_gridded = math.ceil(trace_pitch / self.board.grid_size) * self.board.grid_size

        # Width needed = traces * pitch + edge clearances
        width = traces_per_layer * pitch_gridded + 2 * self.rules.min_clearance

        return width


# =============================================================================
# CORRIDOR DETECTOR
# =============================================================================

class CorridorDetector:
    """Detects routing corridors in the layout"""

    def __init__(self, board, rules, parts_db: Dict, placement: Dict):
        self.board = board
        self.rules = rules
        self.parts_db = parts_db
        self.placement = placement

    def detect_all_corridors(self, escapes: Dict, graph: Dict) -> List[Corridor]:
        """
        Detect all routing corridors.

        Corridors are detected in several ways:
        1. Escape fan-out regions
        2. Gaps between component groups
        3. Edges of the board
        """
        corridors = []

        # Method 1: Escape fan-out corridors
        escape_corridors = self._detect_escape_corridors(escapes)
        corridors.extend(escape_corridors)

        # Method 2: Inter-component corridors
        gap_corridors = self._detect_gap_corridors(graph)
        corridors.extend(gap_corridors)

        # Method 3: Board edge corridors
        edge_corridors = self._detect_edge_corridors()
        corridors.extend(edge_corridors)

        return corridors

    def _detect_escape_corridors(self, escapes: Dict) -> List[Corridor]:
        """Detect corridors from escape fan-outs"""
        corridors = []

        for ref, pin_escapes in escapes.items():
            if len(pin_escapes) < 4:
                continue  # Not a significant fan-out

            pos = self.placement.get(ref)
            if not pos:
                continue

            corridor = self._create_escape_corridor(ref, pin_escapes, pos)
            if corridor:
                corridors.append(corridor)

        return corridors

    def _create_escape_corridor(self, ref: str, escapes: Dict, pos) -> Optional[Corridor]:
        """Create a corridor from escape fan-out"""
        if not escapes:
            return None

        # Get escape endpoints
        endpoints = [(e.endpoint[0], e.endpoint[1]) for e in escapes.values()]
        starts = [(e.start[0] if hasattr(e, 'start') else pos.x,
                   e.start[1] if hasattr(e, 'start') else pos.y) for e in escapes.values()]

        # Calculate bounding box
        all_points = endpoints + starts
        x1 = min(p[0] for p in all_points)
        y1 = min(p[1] for p in all_points)
        x2 = max(p[0] for p in all_points)
        y2 = max(p[1] for p in all_points)

        # Determine escape direction
        first_escape = list(escapes.values())[0]
        dx, dy = first_escape.direction

        if abs(dx) > abs(dy):
            # Horizontal escape
            direction = 'horizontal'
            width = y2 - y1  # Perpendicular dimension
            length = x2 - x1  # Parallel dimension
        else:
            # Vertical escape
            direction = 'vertical'
            width = x2 - x1  # Perpendicular dimension
            length = y2 - y1  # Parallel dimension

        # Collect nets in this corridor
        nets = [e.net for e in escapes.values() if hasattr(e, 'net') and e.net]

        # For escape corridors, the width should be proportional to the number
        # of escapes - each escape needs enough width to fan out
        # Minimum corridor width = (num_escapes + 1) * trace_pitch
        min_escape_width = len(escapes) * 1.0  # ~1mm per trace for escape

        return Corridor(
            name=f"{ref}_escape",
            bounds=(x1, y1, x2, y2),
            direction=direction,
            width=max(width, min_escape_width, 5.0),  # Minimum 5mm corridor
            length=max(length, 2.0),
            capacity=0,  # Calculated later
            demand=len(escapes),
            nets=nets,
            source_area=ref,
            dest_area='routing_area',
        )

    def _detect_gap_corridors(self, graph: Dict) -> List[Corridor]:
        """Detect corridors in gaps between component groups"""
        corridors = []

        # Group components by Y position (roughly)
        y_groups = {}
        for ref, pos in self.placement.items():
            y_band = int(pos.y / 20) * 20  # 20mm bands
            if y_band not in y_groups:
                y_groups[y_band] = []
            y_groups[y_band].append((ref, pos))

        # Find gaps between groups
        sorted_bands = sorted(y_groups.keys())
        for i in range(len(sorted_bands) - 1):
            band_a = sorted_bands[i]
            band_b = sorted_bands[i + 1]

            gap_size = band_b - band_a
            if gap_size > 15:  # Significant gap
                # Create horizontal corridor
                components_above = [ref for ref, pos in y_groups[band_a]]
                components_below = [ref for ref, pos in y_groups[band_b]]

                # Count nets crossing this gap
                crossing_nets = self._count_crossing_nets(
                    set(components_above), set(components_below), graph
                )

                if crossing_nets:
                    # Corridor width is the gap minus some margin for component edges
                    # But ensure we have at least the gap size as width
                    corridor_width = max(gap_size - 10, gap_size * 0.5, 5.0)

                    corridor = Corridor(
                        name=f"gap_Y{band_a}_to_Y{band_b}",
                        bounds=(
                            self.board.origin_x,
                            band_a + 10,
                            self.board.origin_x + self.board.width,
                            band_b
                        ),
                        direction='horizontal',
                        width=corridor_width,
                        length=self.board.width,
                        capacity=0,
                        demand=len(crossing_nets),
                        nets=crossing_nets,
                        source_area='north_group',
                        dest_area='south_group',
                    )
                    corridors.append(corridor)

        return corridors

    def _detect_edge_corridors(self) -> List[Corridor]:
        """Detect corridors along board edges"""
        corridors = []

        # Check for components near edges that might create edge corridors
        edge_margin = 5  # mm

        for edge, bounds in [
            ('left', (self.board.origin_x, self.board.origin_y,
                      self.board.origin_x + edge_margin, self.board.origin_y + self.board.height)),
            ('right', (self.board.origin_x + self.board.width - edge_margin, self.board.origin_y,
                       self.board.origin_x + self.board.width, self.board.origin_y + self.board.height)),
        ]:
            # Count components along this edge
            comps_on_edge = []
            for ref, pos in self.placement.items():
                if bounds[0] <= pos.x <= bounds[2]:
                    comps_on_edge.append(ref)

            if len(comps_on_edge) >= 2:
                corridor = Corridor(
                    name=f"{edge}_edge",
                    bounds=bounds,
                    direction='vertical',
                    width=edge_margin,
                    length=self.board.height,
                    capacity=0,
                    demand=0,  # Will be calculated from nets
                )
                corridors.append(corridor)

        return corridors

    def _count_crossing_nets(self, group_a: Set[str], group_b: Set[str],
                              graph: Dict) -> List[str]:
        """Count nets that cross between two component groups"""
        nets = self.parts_db.get('nets', {})
        crossing = []

        for net_name, net_info in nets.items():
            if net_name == 'GND':
                continue

            pins = net_info.get('pins', [])
            has_a = any(comp in group_a for comp, pin in pins)
            has_b = any(comp in group_b for comp, pin in pins)

            if has_a and has_b:
                crossing.append(net_name)

        return crossing


# =============================================================================
# DEMAND CALCULATOR
# =============================================================================

class DemandCalculator:
    """Calculates trace demand for corridors"""

    def __init__(self, parts_db: Dict, placement: Dict, graph: Dict):
        self.parts_db = parts_db
        self.placement = placement
        self.graph = graph
        self.nets = parts_db.get('nets', {})

    def calculate_corridor_demand(self, corridor: Corridor) -> int:
        """
        Calculate how many traces need to pass through a corridor.

        For escape corridors, demand = number of pins escaping
        For gap corridors, demand = number of nets crossing the gap
        """
        if '_escape' in corridor.name:
            # Already set during corridor creation
            return corridor.demand

        if 'gap' in corridor.name:
            return len(corridor.nets)

        return corridor.demand

    def find_nets_through_corridor(self, corridor: Corridor) -> List[str]:
        """Find all nets that must pass through a corridor"""
        nets_through = []

        # For each net, check if it crosses the corridor bounds
        for net_name, net_info in self.nets.items():
            if net_name == 'GND':
                continue

            pins = net_info.get('pins', [])

            if len(pins) < 2:
                continue

            # Get positions of pins on this net
            positions = []
            for comp, pin in pins:
                if comp in self.placement:
                    pos = self.placement[comp]
                    positions.append((pos.x, pos.y))

            if len(positions) < 2:
                continue

            # Check if net crosses corridor
            if self._net_crosses_corridor(positions, corridor):
                nets_through.append(net_name)

        return nets_through

    def _net_crosses_corridor(self, positions: List[Tuple[float, float]],
                               corridor: Corridor) -> bool:
        """Check if a net's positions cross a corridor"""
        x1, y1, x2, y2 = corridor.bounds

        # Simple check: are positions on both sides of the corridor?
        if corridor.direction == 'horizontal':
            # Check if positions span across the corridor's Y range
            below_count = sum(1 for x, y in positions if y < y1)
            above_count = sum(1 for x, y in positions if y > y2)
            return below_count > 0 and above_count > 0

        else:  # vertical
            # Check if positions span across the corridor's X range
            left_count = sum(1 for x, y in positions if x < x1)
            right_count = sum(1 for x, y in positions if x > x2)
            return left_count > 0 and right_count > 0


# =============================================================================
# BOTTLENECK ANALYZER
# =============================================================================

class BottleneckAnalyzer:
    """Analyzes bottlenecks and suggests fixes"""

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules

    def analyze(self, corridors: List[Corridor]) -> List[BottleneckAnalysis]:
        """Analyze all corridors for bottlenecks"""
        bottlenecks = []

        for corridor in corridors:
            if corridor.is_overcapacity:
                analysis = self._analyze_bottleneck(corridor)
                bottlenecks.append(analysis)

        # Sort by severity (most severe first)
        bottlenecks.sort(key=lambda b: -b.severity)

        return bottlenecks

    def _analyze_bottleneck(self, corridor: Corridor) -> BottleneckAnalysis:
        """Analyze a single bottleneck"""
        severity = corridor.demand / corridor.capacity if corridor.capacity > 0 else float('inf')

        # Generate fix suggestions
        fixes = []

        shortfall = corridor.shortfall

        # Fix 1: Widen the corridor
        min_width = self._calculate_required_width(corridor.demand)
        width_increase = min_width - corridor.width
        if width_increase > 0:
            fixes.append(
                f"Increase corridor width by {width_increase:.1f}mm "
                f"(move components apart)"
            )

        # Fix 2: Use both layers
        if corridor.capacity < corridor.demand:
            capacity_with_both = corridor.capacity * 2
            if capacity_with_both >= corridor.demand:
                fixes.append(
                    f"Use both F.Cu and B.Cu layers "
                    f"(requires {shortfall} vias)"
                )

        # Fix 3: Rotate hub
        if '_escape' in corridor.name:
            fixes.append(
                f"Rotate hub to spread escapes across wider area"
            )

        # Fix 4: Move components
        fixes.append(
            f"Move source/destination components to create alternate route"
        )

        return BottleneckAnalysis(
            corridor=corridor,
            severity=severity,
            suggested_fixes=fixes,
            affected_nets=corridor.nets,
        )

    def _calculate_required_width(self, num_traces: int) -> float:
        """Calculate width required for N traces"""
        trace_pitch = self.rules.min_trace_width + 2 * self.rules.min_clearance
        pitch_gridded = math.ceil(trace_pitch / self.board.grid_size) * self.board.grid_size
        return num_traces * pitch_gridded + 2 * self.rules.min_clearance


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

class CorridorValidator:
    """
    Main corridor validator.

    ALGORITHM:
    1. Detect all routing corridors
    2. Calculate demand for each (how many traces need to pass)
    3. Calculate capacity for each (how many traces can fit)
    4. Identify bottlenecks (demand > capacity)
    5. Generate fix suggestions
    """

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules
        self.corridors = []
        self.bottlenecks = []

        # Sub-modules
        self.capacity_calc = CapacityCalculator(board, rules)
        self.bottleneck_analyzer = BottleneckAnalyzer(board, rules)

    def validate(self, placement: Dict, escapes: Dict, graph: Dict,
                 parts_db: Dict = None) -> Dict:
        """
        Validate all corridor capacities.

        Returns:
            {
                'valid': bool,
                'corridors': [Corridor, ...],
                'violations': [str, ...],
                'bottlenecks': [BottleneckAnalysis, ...],
                'summary': {...}
            }
        """
        # Get parts_db if not provided
        if parts_db is None:
            parts_db = {'parts': {}, 'nets': {}}

        # Create detector
        detector = CorridorDetector(self.board, self.rules, parts_db, placement)

        # Detect all corridors
        self.corridors = detector.detect_all_corridors(escapes, graph)

        # Calculate demand
        demand_calc = DemandCalculator(parts_db, placement, graph)
        for corridor in self.corridors:
            if not corridor.nets:
                corridor.nets = demand_calc.find_nets_through_corridor(corridor)
            corridor.demand = max(corridor.demand, len(corridor.nets))

        # Calculate capacity
        for corridor in self.corridors:
            corridor.capacity = self.capacity_calc.calculate_capacity(
                corridor.width,
                layers_available=1  # F.Cu only typically
            )

            # Calculate utilization
            if corridor.capacity > 0:
                corridor.utilization = corridor.demand / corridor.capacity
                corridor.bottleneck_severity = max(0, corridor.utilization - 1.0)
            else:
                corridor.utilization = float('inf')
                corridor.bottleneck_severity = 10.0  # Very severe

        # Analyze bottlenecks
        self.bottlenecks = self.bottleneck_analyzer.analyze(self.corridors)

        # Generate violations
        violations = []
        for corridor in self.corridors:
            if corridor.is_overcapacity:
                violations.append(
                    f"Corridor '{corridor.name}': "
                    f"demand={corridor.demand} > capacity={corridor.capacity} "
                    f"(shortfall={corridor.shortfall})"
                )

        # Summary
        total_demand = sum(c.demand for c in self.corridors)
        total_capacity = sum(c.capacity for c in self.corridors)
        overcapacity_count = sum(1 for c in self.corridors if c.is_overcapacity)

        return {
            'valid': len(violations) == 0,
            'corridors': self.corridors,
            'violations': violations,
            'bottlenecks': self.bottlenecks,
            'summary': {
                'total_corridors': len(self.corridors),
                'overcapacity_count': overcapacity_count,
                'total_demand': total_demand,
                'total_capacity': total_capacity,
                'overall_utilization': total_demand / total_capacity if total_capacity > 0 else float('inf'),
            }
        }

    def get_report(self) -> str:
        """Generate human-readable corridor report"""
        lines = [
            "=" * 60,
            "CORRIDOR CAPACITY REPORT",
            "=" * 60,
        ]

        for corridor in self.corridors:
            status = "[OK]" if not corridor.is_overcapacity else "[OVER]"
            lines.extend([
                f"\n{corridor.name}:",
                f"  Direction: {corridor.direction}",
                f"  Width: {corridor.width:.1f}mm",
                f"  Demand: {corridor.demand} traces",
                f"  Capacity: {corridor.capacity} traces",
                f"  Utilization: {corridor.utilization:.0%}",
                f"  Status: {status}",
            ])

            if corridor.is_overcapacity:
                lines.append(f"  SHORTFALL: {corridor.shortfall} traces")

        if self.bottlenecks:
            lines.extend([
                "",
                "-" * 40,
                "BOTTLENECK ANALYSIS",
                "-" * 40,
            ])

            for bottleneck in self.bottlenecks:
                lines.extend([
                    f"\n{bottleneck.corridor.name} (severity: {bottleneck.severity:.1f}x)",
                    "  Suggested fixes:",
                ])
                for fix in bottleneck.suggested_fixes:
                    lines.append(f"    - {fix}")

        lines.append("=" * 60)
        return "\n".join(lines)
