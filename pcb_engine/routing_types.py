"""
PCB Engine - Shared Routing Types
==================================

Unified data structures for routing pistons and engines.
This module provides canonical definitions to prevent type mismatches
between routing_piston.py and routing_engine.py.

All routing-related code should import from here, not define their own.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrackSegment:
    """A single track segment on the PCB."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    layer: str
    width: float
    net: str

    @property
    def length(self) -> float:
        """Calculate segment length using Euclidean distance."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx*dx + dy*dy)

    @property
    def is_horizontal(self) -> bool:
        """Check if segment is horizontal (within 0.01mm tolerance)."""
        return abs(self.end[1] - self.start[1]) < 0.01

    @property
    def is_vertical(self) -> bool:
        """Check if segment is vertical (within 0.01mm tolerance)."""
        return abs(self.end[0] - self.start[0]) < 0.01


@dataclass
class ArcSegment:
    """
    An arc track segment on the PCB.

    KiCad uses three-point arc definition:
    - start: Starting point of the arc
    - mid: A point on the arc (defines curvature)
    - end: Ending point of the arc
    """
    start: Tuple[float, float]
    mid: Tuple[float, float]  # Point on arc that defines curvature
    end: Tuple[float, float]
    layer: str
    width: float
    net: str

    @property
    def length(self) -> float:
        """Approximate arc length using the three points."""
        # Use the arc length formula: L = r * theta
        # First, calculate radius and angle from three points
        center, radius = self._calculate_center_radius()
        if radius == 0:
            return 0.0

        # Calculate angle subtended
        dx1, dy1 = self.start[0] - center[0], self.start[1] - center[1]
        dx2, dy2 = self.end[0] - center[0], self.end[1] - center[1]

        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)

        angle_diff = abs(angle2 - angle1)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        return radius * angle_diff

    def _calculate_center_radius(self) -> Tuple[Tuple[float, float], float]:
        """Calculate arc center and radius from three points."""
        x1, y1 = self.start
        x2, y2 = self.mid
        x3, y3 = self.end

        # Use perpendicular bisectors method
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-10:
            return (0, 0), 0.0

        ux = ((x1*x1 + y1*y1) * (y2 - y3) + (x2*x2 + y2*y2) * (y3 - y1) + (x3*x3 + y3*y3) * (y1 - y2)) / d
        uy = ((x1*x1 + y1*y1) * (x3 - x2) + (x2*x2 + y2*y2) * (x1 - x3) + (x3*x3 + y3*y3) * (x2 - x1)) / d

        radius = math.sqrt((x1 - ux)**2 + (y1 - uy)**2)
        return (ux, uy), radius

    @property
    def center(self) -> Tuple[float, float]:
        """Get the arc center point."""
        center, _ = self._calculate_center_radius()
        return center

    @property
    def radius(self) -> float:
        """Get the arc radius."""
        _, radius = self._calculate_center_radius()
        return radius


@dataclass
class Via:
    """A via connecting layers on the PCB."""
    position: Tuple[float, float]
    net: str
    diameter: float = 0.8
    drill: float = 0.4
    from_layer: str = 'F.Cu'
    to_layer: str = 'B.Cu'


@dataclass
class Route:
    """Complete route for a net, including all segments, arcs, and vias."""
    net: str
    segments: List[TrackSegment] = field(default_factory=list)
    arcs: List[ArcSegment] = field(default_factory=list)  # Arc segments for smooth corners
    vias: List[Via] = field(default_factory=list)
    success: bool = False
    error: str = ''
    algorithm_used: str = ''

    @property
    def total_length(self) -> float:
        """Calculate total wirelength of all segments and arcs."""
        seg_length = sum(seg.length for seg in self.segments)
        arc_length = sum(arc.length for arc in self.arcs)
        return seg_length + arc_length

    @property
    def bend_count(self) -> int:
        """Count number of bends (direction changes) in the route."""
        if len(self.segments) < 2:
            return 0
        bends = 0
        for i in range(1, len(self.segments)):
            prev = self.segments[i-1]
            curr = self.segments[i]
            if prev.is_horizontal != curr.is_horizontal:
                bends += 1
        return bends


class RoutingAlgorithm(Enum):
    """Available routing algorithms."""
    LEE = 'lee'                     # Lee wavefront (guaranteed optimal)
    HADLOCK = 'hadlock'             # Hadlock's detour-biased algorithm
    SOUKUP = 'soukup'               # Soukup's two-phase algorithm
    MIKAMI = 'mikami'               # Mikami-Tabuchi line search
    ASTAR = 'astar'                 # A* heuristic pathfinding
    PATHFINDER = 'pathfinder'       # Negotiated congestion routing
    RIPUP_REROUTE = 'ripup'         # Rip-up and reroute
    STEINER = 'steiner'             # Steiner tree (multi-terminal)
    CHANNEL = 'channel'             # Channel/greedy routing
    HYBRID = 'hybrid'               # Combination of above
    AUTO = 'auto'                   # Automatically select best


@dataclass
class RoutingConfig:
    """
    Configuration for routing pistons/engines.

    This is the canonical configuration - all routing code should use this.
    """
    algorithm: str = 'hybrid'

    # Board parameters
    board_width: float = 100.0
    board_height: float = 100.0
    origin_x: float = 0.0  # Board origin X (should match placement)
    origin_y: float = 0.0  # Board origin Y (should match placement)
    grid_size: float = 0.08  # Routing grid resolution in mm (6 cells per 0.5mm pad)

    # Design rules
    trace_width: float = 0.25
    clearance: float = 0.15
    via_diameter: float = 0.8
    via_drill: float = 0.4

    # Algorithm parameters
    max_ripup_iterations: int = 15
    lee_max_expansion: int = 100000
    astar_timeout_ms: int = 5000
    pathfinder_max_iterations: int = 50
    pathfinder_penalty_increment: float = 0.5
    steiner_heuristic: str = 'hanan'  # 'hanan' or 'mst'

    # Layer preferences
    prefer_top_layer: bool = True
    allow_layer_change: bool = True
    top_layer_name: str = 'F.Cu'
    bottom_layer_name: str = 'B.Cu'

    # Diagonal routing (45-degree angles)
    allow_45_degree: bool = True  # Enable 8-directional routing for cleaner traces

    # Via cost (for algorithms that use cost)
    via_cost: float = 5.0

    # Random seed for reproducibility
    seed: int = 42

    # Parallel routing (multi-core optimization)
    parallel_routing: bool = True  # Enable parallel routing for independent nets
    max_workers: int = 0  # 0 = auto-detect CPU cores, otherwise use specified count


@dataclass
class RoutingResult:
    """Result from routing operations."""
    routes: Dict[str, Route]
    success: bool
    routed_count: int
    total_count: int
    algorithm_used: str
    iterations: int = 1
    total_wirelength: float = 0.0
    via_count: int = 0
    statistics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_track_segment(
    start: Tuple[float, float],
    end: Tuple[float, float],
    layer: str = 'F.Cu',
    width: float = 0.25,
    net: str = ''
) -> TrackSegment:
    """Factory function to create a TrackSegment with defaults."""
    return TrackSegment(
        start=start,
        end=end,
        layer=layer,
        width=width,
        net=net
    )


def create_via(
    position: Tuple[float, float],
    net: str = '',
    diameter: float = 0.8,
    drill: float = 0.4,
    from_layer: str = 'F.Cu',
    to_layer: str = 'B.Cu'
) -> Via:
    """Factory function to create a Via with defaults."""
    return Via(
        position=position,
        net=net,
        diameter=diameter,
        drill=drill,
        from_layer=from_layer,
        to_layer=to_layer
    )
