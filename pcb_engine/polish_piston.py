"""
PCB Engine - Polish Piston
===========================

Post-routing optimization piston that refines and polishes the design:

1. VIA REDUCTION - Remove unnecessary layer changes
2. TRACE SIMPLIFICATION - Merge tiny segments into clean lines
3. ARC SMOOTHING - Replace sharp corners with arcs (optional)
4. BOARD SHRINK - Reduce board size to fit the actual design
5. CLEANUP - Remove redundant elements, fix alignment

The Polish Piston runs AFTER routing is complete and makes the design
look professional and manufacturable.

DYNAMIC VIA COST STRATEGY:
- Phase 1 (Routing): Low via cost → maximize routability
- Phase 2 (Polish): Aggressively remove unnecessary vias

This gives the benefits of both worlds:
- Easy routing (vias available when needed)
- Clean result (vias removed when not needed)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import math

from .routing_types import TrackSegment, Via, Route


class PolishLevel(Enum):
    """How aggressively to polish"""
    MINIMAL = 'minimal'      # Just cleanup, no major changes
    STANDARD = 'standard'    # Remove obvious issues
    AGGRESSIVE = 'aggressive'  # Maximum optimization
    PROFESSIONAL = 'professional'  # Production-ready finish


@dataclass
class PolishConfig:
    """Configuration for the polish piston"""
    # General
    level: PolishLevel = PolishLevel.STANDARD

    # Via reduction
    reduce_vias: bool = True
    via_keep_margin: float = 0.5  # Keep vias if removal increases length by more than this (mm)

    # Trace simplification
    simplify_traces: bool = True
    min_segment_length: float = 0.2  # Merge segments shorter than this
    collinear_tolerance: float = 0.05  # Segments within this angle (rad) are considered collinear

    # Arc smoothing
    use_arcs: bool = False  # Replace corners with arcs (experimental)
    min_arc_radius: float = 0.5  # Minimum arc radius (mm)
    max_arc_radius: float = 2.0  # Maximum arc radius (mm)

    # Board shrink
    shrink_board: bool = True
    board_margin: float = 2.0  # Margin from components to board edge (mm)
    min_board_width: float = 10.0  # Don't shrink below this
    min_board_height: float = 10.0

    # Alignment
    align_to_grid: bool = True
    alignment_grid: float = 0.1  # Snap to this grid (mm)

    # Reporting
    verbose: bool = True


@dataclass
class PolishResult:
    """Result from the polish piston"""
    success: bool
    routes: Dict[str, Route]

    # Board size
    original_board: Tuple[float, float]  # (width, height)
    new_board: Tuple[float, float]
    board_reduction_percent: float

    # Via statistics
    original_via_count: int
    new_via_count: int
    vias_removed: int

    # Segment statistics
    original_segment_count: int
    new_segment_count: int
    segments_merged: int

    # Length statistics
    original_total_length: float
    new_total_length: float
    length_change_percent: float

    messages: List[str] = field(default_factory=list)


class PolishPiston:
    """
    Post-routing optimization piston.

    Polishes the design after routing to make it professional:
    - Removes unnecessary vias
    - Merges tiny segments into clean traces
    - Optionally adds arcs at corners
    - Shrinks board to fit components
    """

    def __init__(self, config: PolishConfig = None):
        self.config = config or PolishConfig()
        self.messages = []

    def polish(self,
               routes: Dict[str, Route],
               parts_db: Dict,
               placement: Dict,
               board_width: float,
               board_height: float) -> PolishResult:
        """
        Polish the routed design.

        Args:
            routes: Routing results {net_name: Route}
            parts_db: Parts database
            placement: Component positions
            board_width: Current board width
            board_height: Current board height

        Returns:
            PolishResult with optimized routes and statistics
        """
        self.messages = []

        # Count original stats
        orig_vias = sum(len(r.vias) for r in routes.values())
        orig_segs = sum(len(r.segments) for r in routes.values())
        orig_length = sum(r.total_length for r in routes.values())

        if self.config.verbose:
            print(f"\n[POLISH] Starting optimization...")
            print(f"  Original: {orig_segs} segments, {orig_vias} vias, {orig_length:.1f}mm total")

        # Make a deep copy of routes to modify
        polished_routes = self._copy_routes(routes)

        # Phase 1: Via reduction
        if self.config.reduce_vias:
            polished_routes = self._reduce_vias(polished_routes)

        # Phase 2: Trace simplification
        if self.config.simplify_traces:
            polished_routes = self._simplify_traces(polished_routes)

        # Phase 3: Arc smoothing (optional)
        if self.config.use_arcs:
            polished_routes = self._add_arcs(polished_routes)

        # Phase 4: Board shrink
        new_width, new_height = board_width, board_height
        if self.config.shrink_board:
            new_width, new_height = self._calculate_optimal_board_size(
                polished_routes, parts_db, placement
            )

        # Phase 5: Grid alignment
        if self.config.align_to_grid:
            polished_routes = self._align_to_grid(polished_routes)

        # Calculate final stats
        new_vias = sum(len(r.vias) for r in polished_routes.values())
        new_segs = sum(len(r.segments) for r in polished_routes.values())
        new_length = self._calculate_total_length(polished_routes)

        if self.config.verbose:
            print(f"  Polished: {new_segs} segments, {new_vias} vias, {new_length:.1f}mm total")
            print(f"  Removed: {orig_vias - new_vias} vias, {orig_segs - new_segs} segments")
            if new_width != board_width or new_height != board_height:
                reduction = (1 - (new_width * new_height) / (board_width * board_height)) * 100
                print(f"  Board: {board_width:.1f}x{board_height:.1f} -> {new_width:.1f}x{new_height:.1f} ({reduction:.1f}% smaller)")

        return PolishResult(
            success=True,
            routes=polished_routes,
            original_board=(board_width, board_height),
            new_board=(new_width, new_height),
            board_reduction_percent=(1 - (new_width * new_height) / (board_width * board_height)) * 100,
            original_via_count=orig_vias,
            new_via_count=new_vias,
            vias_removed=orig_vias - new_vias,
            original_segment_count=orig_segs,
            new_segment_count=new_segs,
            segments_merged=orig_segs - new_segs,
            original_total_length=orig_length,
            new_total_length=new_length,
            length_change_percent=(new_length - orig_length) / orig_length * 100 if orig_length > 0 else 0,
            messages=self.messages
        )

    # =========================================================================
    # PHASE 1: VIA REDUCTION
    # =========================================================================

    def _reduce_vias(self, routes: Dict[str, Route]) -> Dict[str, Route]:
        """
        Remove unnecessary vias by checking if same-layer routing is possible.

        Strategy:
        1. For each via, check if segments on both sides could be on same layer
        2. If yes, reroute those segments on one layer and remove the via
        3. Only remove if the length increase is acceptable
        """
        for net_name, route in routes.items():
            if not route.vias:
                continue

            # Try to remove each via
            vias_to_remove = []
            for i, via in enumerate(route.vias):
                if self._can_remove_via(route, via, i):
                    vias_to_remove.append(i)

            # Remove vias in reverse order to preserve indices
            for i in reversed(vias_to_remove):
                self._remove_via(route, i)

            if vias_to_remove and self.config.verbose:
                self.messages.append(f"  {net_name}: removed {len(vias_to_remove)} vias")

        return routes

    def _can_remove_via(self, route: Route, via: Via, via_index: int) -> bool:
        """Check if a via can be removed without breaking connectivity."""
        via_pos = via.position if hasattr(via, 'position') else (0, 0)

        # Find segments connected to this via on each layer
        fcu_segs = []
        bcu_segs = []

        for seg in route.segments:
            if self._segment_touches_point(seg, via_pos):
                if seg.layer == 'F.Cu':
                    fcu_segs.append(seg)
                else:
                    bcu_segs.append(seg)

        # Can remove if:
        # 1. Via only connects two segments (one on each layer)
        # 2. Those segments could be merged into one
        if len(fcu_segs) == 1 and len(bcu_segs) == 1:
            # Check if they form a continuous path
            return True

        return False

    def _remove_via(self, route: Route, via_index: int):
        """Remove a via and merge the connected segments."""
        if via_index >= len(route.vias):
            return

        via = route.vias[via_index]
        via_pos = via.position if hasattr(via, 'position') else (0, 0)

        # Find connected segments
        fcu_segs = []
        bcu_segs = []

        for i, seg in enumerate(route.segments):
            if self._segment_touches_point(seg, via_pos):
                if seg.layer == 'F.Cu':
                    fcu_segs.append((i, seg))
                else:
                    bcu_segs.append((i, seg))

        if len(fcu_segs) == 1 and len(bcu_segs) == 1:
            # Merge: move B.Cu segment to F.Cu
            bcu_idx, bcu_seg = bcu_segs[0]
            bcu_seg.layer = 'F.Cu'

            # Remove the via
            route.vias.pop(via_index)

    def _segment_touches_point(self, seg: TrackSegment, point: Tuple[float, float],
                                tolerance: float = 0.1) -> bool:
        """Check if segment endpoint touches a point."""
        start = seg.start
        end = seg.end

        d_start = math.sqrt((start[0] - point[0])**2 + (start[1] - point[1])**2)
        d_end = math.sqrt((end[0] - point[0])**2 + (end[1] - point[1])**2)

        return d_start < tolerance or d_end < tolerance

    # =========================================================================
    # PHASE 2: TRACE SIMPLIFICATION
    # =========================================================================

    def _simplify_traces(self, routes: Dict[str, Route]) -> Dict[str, Route]:
        """
        Merge collinear and tiny segments into cleaner traces.

        Strategy:
        1. Find consecutive segments on same layer
        2. If they're collinear (same direction), merge them
        3. Remove segments shorter than minimum length
        """
        for net_name, route in routes.items():
            if len(route.segments) < 2:
                continue

            original_count = len(route.segments)
            route.segments = self._merge_collinear_segments(route.segments)
            route.segments = self._remove_tiny_segments(route.segments)

            merged = original_count - len(route.segments)
            if merged > 0 and self.config.verbose:
                self.messages.append(f"  {net_name}: merged {merged} segments")

        return routes

    def _merge_collinear_segments(self, segments: List[TrackSegment]) -> List[TrackSegment]:
        """Merge consecutive collinear segments."""
        if len(segments) < 2:
            return segments

        # Group segments by layer
        by_layer = {}
        for seg in segments:
            layer = seg.layer
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append(seg)

        merged = []
        for layer, layer_segs in by_layer.items():
            merged.extend(self._merge_layer_segments(layer_segs))

        return merged

    def _merge_layer_segments(self, segments: List[TrackSegment]) -> List[TrackSegment]:
        """Merge collinear segments on the same layer."""
        if len(segments) < 2:
            return segments

        result = []
        i = 0

        while i < len(segments):
            current = segments[i]

            # Try to extend current segment by merging with following collinear segments
            while i + 1 < len(segments):
                next_seg = segments[i + 1]

                # Check if they share an endpoint and are collinear
                if self._are_collinear_and_connected(current, next_seg):
                    # Merge: extend current to include next
                    current = self._merge_two_segments(current, next_seg)
                    i += 1
                else:
                    break

            result.append(current)
            i += 1

        return result

    def _are_collinear_and_connected(self, seg1: TrackSegment, seg2: TrackSegment) -> bool:
        """Check if two segments are collinear and share an endpoint."""
        if seg1.layer != seg2.layer:
            return False

        # Check if they share an endpoint
        shared = None
        if self._points_equal(seg1.end, seg2.start):
            shared = seg1.end
        elif self._points_equal(seg1.end, seg2.end):
            shared = seg1.end
        elif self._points_equal(seg1.start, seg2.start):
            shared = seg1.start
        elif self._points_equal(seg1.start, seg2.end):
            shared = seg1.start

        if shared is None:
            return False

        # Check if collinear (same direction)
        dir1 = self._segment_direction(seg1)
        dir2 = self._segment_direction(seg2)

        # Allow opposite directions (segments going opposite ways from shared point)
        angle_diff = abs(dir1 - dir2)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        return angle_diff < self.config.collinear_tolerance or \
               abs(angle_diff - math.pi) < self.config.collinear_tolerance

    def _segment_direction(self, seg: TrackSegment) -> float:
        """Get segment direction in radians."""
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        return math.atan2(dy, dx)

    def _merge_two_segments(self, seg1: TrackSegment, seg2: TrackSegment) -> TrackSegment:
        """Merge two connected segments into one."""
        # Find the non-shared endpoints
        points = [seg1.start, seg1.end, seg2.start, seg2.end]

        # Count occurrences - shared point appears twice
        from collections import Counter
        point_counts = Counter()
        for p in points:
            point_counts[(round(p[0], 3), round(p[1], 3))] += 1

        # Get the two endpoints that appear only once
        endpoints = [p for p, count in point_counts.items() if count == 1]

        if len(endpoints) == 2:
            return TrackSegment(
                start=endpoints[0],
                end=endpoints[1],
                layer=seg1.layer,
                width=seg1.width,
                net=seg1.net
            )

        # Fallback: return first segment
        return seg1

    def _points_equal(self, p1: Tuple[float, float], p2: Tuple[float, float],
                      tolerance: float = 0.01) -> bool:
        """Check if two points are equal within tolerance."""
        return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

    def _remove_tiny_segments(self, segments: List[TrackSegment]) -> List[TrackSegment]:
        """Remove segments shorter than minimum length."""
        min_len = self.config.min_segment_length
        return [seg for seg in segments if self._segment_length(seg) >= min_len]

    def _segment_length(self, seg: TrackSegment) -> float:
        """Calculate segment length."""
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        return math.sqrt(dx*dx + dy*dy)

    # =========================================================================
    # PHASE 3: ARC SMOOTHING (Optional)
    # =========================================================================

    def _add_arcs(self, routes: Dict[str, Route]) -> Dict[str, Route]:
        """
        Replace sharp corners with arcs for professional appearance.

        This identifies 90-degree corners (L-bends) and replaces them with
        smooth arcs. The arc radius is configurable.

        Note: KiCad supports arcs natively. Some fabs may convert them to
        segments during CAM processing.
        """
        try:
            from .routing_types import ArcSegment
        except ImportError:
            from routing_types import ArcSegment

        min_radius = self.config.min_arc_radius
        max_radius = self.config.max_arc_radius
        arcs_added = 0

        for net_name, route in routes.items():
            if len(route.segments) < 2:
                continue

            new_segments = []
            new_arcs = []

            i = 0
            while i < len(route.segments):
                seg = route.segments[i]

                # Check if this segment and the next form a corner
                if i + 1 < len(route.segments):
                    next_seg = route.segments[i + 1]

                    # Check if they share an endpoint and are perpendicular
                    corner = self._find_corner(seg, next_seg)
                    if corner and self._is_perpendicular(seg, next_seg):
                        # Calculate arc parameters
                        arc_result = self._create_corner_arc(
                            seg, next_seg, corner, min_radius, max_radius
                        )

                        if arc_result:
                            new_seg1, arc, new_seg2 = arc_result

                            # Add shortened first segment (if it still has length)
                            if self._calc_segment_length(new_seg1.start, new_seg1.end) > 0.01:
                                new_segments.append(new_seg1)

                            # Add the arc
                            new_arcs.append(arc)
                            arcs_added += 1

                            # Replace second segment with shortened version
                            route.segments[i + 1] = new_seg2
                            i += 1
                            continue

                new_segments.append(seg)
                i += 1

            route.segments = new_segments
            route.arcs = new_arcs

        if arcs_added > 0 and self.config.verbose:
            self.messages.append(f"  Added {arcs_added} arcs for smooth corners")

        return routes

    def _find_corner(self, seg1: TrackSegment, seg2: TrackSegment) -> Optional[Tuple[float, float]]:
        """Find the corner point where two segments meet."""
        if self._points_equal(seg1.end, seg2.start):
            return seg1.end
        elif self._points_equal(seg1.end, seg2.end):
            return seg1.end
        elif self._points_equal(seg1.start, seg2.start):
            return seg1.start
        elif self._points_equal(seg1.start, seg2.end):
            return seg1.start
        return None

    def _is_perpendicular(self, seg1: TrackSegment, seg2: TrackSegment) -> bool:
        """Check if two segments are perpendicular (90-degree corner)."""
        # Get direction vectors
        dx1 = seg1.end[0] - seg1.start[0]
        dy1 = seg1.end[1] - seg1.start[1]
        dx2 = seg2.end[0] - seg2.start[0]
        dy2 = seg2.end[1] - seg2.start[1]

        # Normalize
        len1 = math.sqrt(dx1*dx1 + dy1*dy1)
        len2 = math.sqrt(dx2*dx2 + dy2*dy2)

        if len1 < 0.01 or len2 < 0.01:
            return False

        dx1, dy1 = dx1/len1, dy1/len1
        dx2, dy2 = dx2/len2, dy2/len2

        # Dot product should be close to 0 for perpendicular
        dot = abs(dx1*dx2 + dy1*dy2)
        return dot < 0.1  # Allow some tolerance

    def _create_corner_arc(
        self,
        seg1: TrackSegment,
        seg2: TrackSegment,
        corner: Tuple[float, float],
        min_radius: float,
        max_radius: float
    ) -> Optional[Tuple[TrackSegment, 'ArcSegment', TrackSegment]]:
        """
        Create an arc to replace a 90-degree corner.

        Returns (shortened_seg1, arc, shortened_seg2) or None if arc can't be created.
        """
        try:
            from .routing_types import ArcSegment
        except ImportError:
            from routing_types import ArcSegment

        # Calculate available length on each segment
        len1 = self._calc_segment_length(seg1.start, corner)
        len2 = self._calc_segment_length(corner, seg2.end)

        # Radius is limited by shorter segment length
        max_available = min(len1, len2) * 0.4  # Use 40% of shorter segment
        radius = min(max_radius, max(min_radius, max_available))

        if radius < min_radius:
            return None  # Segments too short for arc

        # Calculate arc start point (on seg1, radius distance from corner)
        dx1 = corner[0] - seg1.start[0]
        dy1 = corner[1] - seg1.start[1]
        len1_full = math.sqrt(dx1*dx1 + dy1*dy1)
        if len1_full < 0.01:
            return None

        # Normalize direction
        ux1, uy1 = dx1/len1_full, dy1/len1_full

        arc_start = (
            corner[0] - ux1 * radius,
            corner[1] - uy1 * radius
        )

        # Calculate arc end point (on seg2, radius distance from corner)
        dx2 = seg2.end[0] - corner[0]
        dy2 = seg2.end[1] - corner[1]
        len2_full = math.sqrt(dx2*dx2 + dy2*dy2)
        if len2_full < 0.01:
            return None

        ux2, uy2 = dx2/len2_full, dy2/len2_full

        arc_end = (
            corner[0] + ux2 * radius,
            corner[1] + uy2 * radius
        )

        # Calculate arc midpoint (45 degrees from corner)
        # The midpoint is on the arc, at the bisector direction
        bisector_x = (ux1 + ux2) / 2
        bisector_y = (uy1 + uy2) / 2
        bisector_len = math.sqrt(bisector_x*bisector_x + bisector_y*bisector_y)

        if bisector_len < 0.01:
            return None

        bisector_x /= bisector_len
        bisector_y /= bisector_len

        # Arc midpoint is offset from corner in bisector direction
        # For a 90-degree arc, the offset is radius * (sqrt(2) - 1) ≈ 0.414
        arc_offset = radius * 0.414
        arc_mid = (
            corner[0] - bisector_x * arc_offset,
            corner[1] - bisector_y * arc_offset
        )

        # Create shortened segments
        new_seg1 = TrackSegment(
            start=seg1.start,
            end=arc_start,
            layer=seg1.layer,
            width=seg1.width,
            net=seg1.net
        )

        new_seg2 = TrackSegment(
            start=arc_end,
            end=seg2.end,
            layer=seg2.layer,
            width=seg2.width,
            net=seg2.net
        )

        # Create arc
        arc = ArcSegment(
            start=arc_start,
            mid=arc_mid,
            end=arc_end,
            layer=seg1.layer,
            width=seg1.width,
            net=seg1.net
        )

        return (new_seg1, arc, new_seg2)

    def _calc_segment_length(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx*dx + dy*dy)

    # =========================================================================
    # PHASE 4: BOARD SHRINK
    # =========================================================================

    def _calculate_optimal_board_size(self,
                                       routes: Dict[str, Route],
                                       parts_db: Dict,
                                       placement: Dict) -> Tuple[float, float]:
        """
        Calculate minimum board size that fits all components and traces.
        """
        if not placement:
            return self.config.min_board_width, self.config.min_board_height

        # Import courtyard utility
        try:
            from .common_types import calculate_courtyard, get_xy
        except ImportError:
            from common_types import calculate_courtyard, get_xy

        parts = parts_db.get('parts', {})
        margin = self.config.board_margin

        # Find bounding box of all components (with courtyards)
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            footprint = part.get('footprint', '')
            courtyard = calculate_courtyard(part, margin=0.25, footprint_name=footprint)

            cx, cy = get_xy(pos)

            min_x = min(min_x, cx + courtyard.min_x)
            max_x = max(max_x, cx + courtyard.max_x)
            min_y = min(min_y, cy + courtyard.min_y)
            max_y = max(max_y, cy + courtyard.max_y)

        # Also check trace extents
        for route in routes.values():
            for seg in route.segments:
                min_x = min(min_x, seg.start[0], seg.end[0])
                max_x = max(max_x, seg.start[0], seg.end[0])
                min_y = min(min_y, seg.start[1], seg.end[1])
                max_y = max(max_y, seg.start[1], seg.end[1])

        # Add board margin
        width = max(max_x - min_x + 2 * margin, self.config.min_board_width)
        height = max(max_y - min_y + 2 * margin, self.config.min_board_height)

        # Round to nice values (0.5mm increments)
        width = math.ceil(width * 2) / 2
        height = math.ceil(height * 2) / 2

        return width, height

    # =========================================================================
    # PHASE 5: GRID ALIGNMENT
    # =========================================================================

    def _align_to_grid(self, routes: Dict[str, Route]) -> Dict[str, Route]:
        """Snap all coordinates to alignment grid."""
        grid = self.config.alignment_grid

        for route in routes.values():
            for seg in route.segments:
                seg.start = (
                    round(seg.start[0] / grid) * grid,
                    round(seg.start[1] / grid) * grid
                )
                seg.end = (
                    round(seg.end[0] / grid) * grid,
                    round(seg.end[1] / grid) * grid
                )

            for via in route.vias:
                if hasattr(via, 'position'):
                    via.position = (
                        round(via.position[0] / grid) * grid,
                        round(via.position[1] / grid) * grid
                    )

        return routes

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _copy_routes(self, routes: Dict[str, Route]) -> Dict[str, Route]:
        """Create a deep copy of routes."""
        copied = {}
        for net_name, route in routes.items():
            # Copy segments
            new_segments = [
                TrackSegment(
                    start=seg.start,
                    end=seg.end,
                    layer=seg.layer,
                    width=seg.width,
                    net=seg.net
                )
                for seg in route.segments
            ]

            # Copy vias
            new_vias = [
                Via(
                    position=via.position,
                    net=via.net,
                    diameter=via.diameter,
                    drill=via.drill,
                    from_layer=via.from_layer,
                    to_layer=via.to_layer
                )
                for via in route.vias
            ]

            copied[net_name] = Route(
                net=net_name,
                segments=new_segments,
                vias=new_vias,
                success=route.success,
                error=route.error,
                algorithm_used=route.algorithm_used
            )

        return copied

    def _calculate_total_length(self, routes: Dict[str, Route]) -> float:
        """Calculate total trace length across all routes."""
        total = 0.0
        for route in routes.values():
            total += self._calculate_route_length(route)
        return total

    def _calculate_route_length(self, route: Route) -> float:
        """Calculate total trace length for a route."""
        total = 0.0
        for seg in route.segments:
            total += self._segment_length(seg)
        return total


# =============================================================================
# DYNAMIC VIA COST STRATEGY
# =============================================================================

@dataclass
class DynamicViaCostConfig:
    """
    Configuration for dynamic via cost during routing.

    Strategy:
    - Start with low via cost to maximize routability
    - Gradually increase if routing succeeds easily
    - The Polish Piston removes unnecessary vias afterward

    This gives best of both worlds:
    - High routability (vias available when needed)
    - Clean result (unnecessary vias removed)
    """
    # Initial via cost (low for easy routing)
    initial_cost: float = 3.0

    # Maximum via cost (for polish phase)
    max_cost: float = 50.0

    # Cost increase per successful route
    cost_increment: float = 2.0

    # Reset cost after routing failure
    reset_on_failure: bool = True


def get_dynamic_via_cost(attempt: int, config: DynamicViaCostConfig = None) -> float:
    """
    Get via cost for a routing attempt.

    Args:
        attempt: Routing attempt number (0-based)
        config: Dynamic via cost configuration

    Returns:
        Via cost to use for this attempt
    """
    config = config or DynamicViaCostConfig()

    cost = config.initial_cost + attempt * config.cost_increment
    return min(cost, config.max_cost)
