#!/usr/bin/env python3
"""
INTELLIGENT ROUTER - My PCB Design Knowledge as an Algorithm
=============================================================

This module codifies my understanding of PCB routing into a complete,
standalone algorithm. It doesn't rely on complex grid-based pathfinding -
instead, it uses PCB design knowledge to make smart routing decisions.

KEY DESIGN PRINCIPLES:
1. Understand the CIRCUIT first (what connects to what)
2. Classify nets by type (power, ground, signal)
3. Choose routing strategy based on net type
4. Use layers intelligently (GND on bottom, signals on top)
5. Avoid crossings by planning routes ahead of time
6. Route simple nets first to establish "highways"

This is my PCB design knowledge turned into code.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class NetType(Enum):
    """Net classification based on name/purpose"""
    POWER = 'power'      # VCC, VDD, 5V, 3V3, etc.
    GROUND = 'ground'    # GND, VSS, etc.
    SIGNAL = 'signal'    # Regular signals
    HIGH_SPEED = 'high_speed'  # Fast signals (CLK, DATA, etc.)


@dataclass(frozen=True)
class Pin:
    """A pin on a component"""
    component: str
    number: str
    net: str
    x: float
    y: float


@dataclass
class Segment:
    """A track segment"""
    start: Tuple[float, float]
    end: Tuple[float, float]
    layer: str
    width: float
    net: str

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx*dx + dy*dy)


@dataclass
class Via:
    """A via between layers"""
    position: Tuple[float, float]
    net: str
    diameter: float = 0.8
    drill: float = 0.4
    from_layer: str = 'F.Cu'
    to_layer: str = 'B.Cu'


@dataclass
class Route:
    """Complete route for a net"""
    net: str
    segments: List[Segment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    success: bool = False
    error: str = ''


@dataclass
class RoutingPlan:
    """Routing plan for a net"""
    net: str
    net_type: NetType
    pins: List[Pin]
    layer_preference: str  # 'F.Cu', 'B.Cu', or 'any'
    routing_order: int  # Lower = route first


# =============================================================================
# INTELLIGENT ROUTER
# =============================================================================

class IntelligentRouter:
    """
    My PCB design knowledge as a routing algorithm.

    This router thinks about PCB design the way I do:
    1. Understand what you're routing (power? ground? signal?)
    2. Plan the route before executing it
    3. Use the right layer for each type of net
    4. Avoid crossings by thinking ahead
    5. Keep it simple - straight lines when possible
    """

    # Net classification keywords
    POWER_KEYWORDS = {'vin', 'vcc', 'vdd', '5v', '3v3', '3.3v', '12v', 'vbat',
                      'vsys', 'vbus', 'vref', 'v+', 'vpos', 'vin_raw', 'vout'}
    GROUND_KEYWORDS = {'gnd', 'ground', 'vss', 'vee', 'gnda', 'gndd', 'pgnd',
                       'agnd', 'dgnd', 'v-', 'vneg', 'com', 'common'}
    HIGH_SPEED_KEYWORDS = {'clk', 'clock', 'sck', 'sclk', 'data', 'mosi', 'miso',
                           'sdi', 'sdo', 'rx', 'tx', 'usb', 'd+', 'd-'}

    def __init__(self, board_width: float = 50.0, board_height: float = 30.0,
                 trace_width: float = 0.25, clearance: float = 0.15):
        self.board_width = board_width
        self.board_height = board_height
        self.trace_width = trace_width
        self.clearance = clearance

        # Track which areas are occupied (simple rectangle-based)
        # Key: layer, Value: list of (x1, y1, x2, y2) rectangles
        self.occupied: Dict[str, List[Tuple[float, float, float, float]]] = {
            'F.Cu': [],
            'B.Cu': []
        }

        # Track routed paths for crossing detection
        self.routed_segments: Dict[str, List[Segment]] = {'F.Cu': [], 'B.Cu': []}

        # Track pad locations to avoid routing through them
        # Key: (x, y), Value: net name (pads on F.Cu only, SMD components)
        self.pad_locations: Dict[Tuple[float, float], str] = {}

        # Track via locations (vias occupy both layers and should be avoided)
        # Key: (x, y), Value: net name
        self.via_locations: Dict[Tuple[float, float], str] = {}

    def route_all(self, parts_db: Dict, placement: Dict) -> Dict[str, Route]:
        """
        Route all nets using intelligent planning.

        Args:
            parts_db: Parts database with components and nets
            placement: Component placements {ref: (x, y) or Position}

        Returns:
            Dictionary of net name -> Route
        """
        print("\n[INTELLIGENT ROUTER] Starting...")

        # Step 1: Build pin list with actual positions
        pins_by_net = self._extract_pins(parts_db, placement)

        # Step 1.5: Register all pad locations for collision avoidance
        # This is CRITICAL - we must know where all pads are to avoid routing through them
        self.pad_locations.clear()
        for net_name, pins in pins_by_net.items():
            for pin in pins:
                # Round to avoid float precision issues
                pad_pos = (round(pin.x, 2), round(pin.y, 2))
                self.pad_locations[pad_pos] = net_name

        print(f"  Registered {len(self.pad_locations)} pad locations")

        # Step 2: Classify and plan each net
        plans = self._plan_routing(pins_by_net)

        # Step 3: Route in order
        routes = {}
        for plan in sorted(plans, key=lambda p: p.routing_order):
            print(f"  Routing {plan.net} ({plan.net_type.value}, {len(plan.pins)} pins)...")
            route = self._route_net(plan)
            routes[plan.net] = route

            # Mark routed segments and vias as occupied
            if route.success:
                for seg in route.segments:
                    self.routed_segments[seg.layer].append(seg)
                    self._mark_occupied(seg)

                # Register vias as obstacles (deduplicate and mark)
                for via in route.vias:
                    via_pos = (round(via.position[0], 2), round(via.position[1], 2))
                    self.via_locations[via_pos] = via.net

        # Summary
        routed = sum(1 for r in routes.values() if r.success)
        print(f"[INTELLIGENT ROUTER] Complete: {routed}/{len(routes)} routed")

        return routes

    def _extract_pins(self, parts_db: Dict, placement: Dict) -> Dict[str, List[Pin]]:
        """Extract all pins with their absolute positions."""
        pins_by_net: Dict[str, List[Pin]] = {}

        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Method 1: Use nets section if available (preferred)
        for net_name, net_info in nets.items():
            pins_by_net[net_name] = []
            for pin_ref in net_info.get('pins', []):
                # Parse pin reference (tuple or dict format)
                if isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
                    comp_ref, pin_num = str(pin_ref[0]), str(pin_ref[1])
                elif isinstance(pin_ref, dict):
                    comp_ref = pin_ref.get('ref', '')
                    pin_num = pin_ref.get('pin', '')
                else:
                    continue

                # Get component position
                if comp_ref not in placement:
                    continue
                pos = placement[comp_ref]
                if hasattr(pos, 'x'):
                    comp_x, comp_y = pos.x, pos.y
                elif isinstance(pos, (list, tuple)):
                    comp_x, comp_y = pos[0], pos[1]
                else:
                    continue

                # Get pin offset from parts_db
                if comp_ref not in parts:
                    continue
                part = parts[comp_ref]
                for pin_def in part.get('pins', []):
                    if str(pin_def.get('number', '')) == pin_num:
                        offset = pin_def.get('offset', (0, 0))
                        pin_x = comp_x + offset[0]
                        pin_y = comp_y + offset[1]

                        pins_by_net[net_name].append(Pin(
                            component=comp_ref,
                            number=pin_num,
                            net=net_name,
                            x=pin_x,
                            y=pin_y
                        ))
                        break

        return pins_by_net

    def _classify_net(self, net_name: str) -> NetType:
        """Classify a net by its name."""
        name_lower = net_name.lower()

        if any(kw in name_lower for kw in self.GROUND_KEYWORDS):
            return NetType.GROUND
        elif any(kw in name_lower for kw in self.POWER_KEYWORDS):
            return NetType.POWER
        elif any(kw in name_lower for kw in self.HIGH_SPEED_KEYWORDS):
            return NetType.HIGH_SPEED
        else:
            return NetType.SIGNAL

    def _plan_routing(self, pins_by_net: Dict[str, List[Pin]]) -> List[RoutingPlan]:
        """
        Create a routing plan for each net.

        ROUTING ORDER STRATEGY:
        1. Short signal nets (2 pins) - establish clean paths first
        2. Power nets - wider traces, direct paths
        3. Ground nets - can use bottom layer entirely
        4. Long/complex signal nets - route around established paths
        """
        plans = []

        for net_name, pins in pins_by_net.items():
            if len(pins) < 2:
                continue

            net_type = self._classify_net(net_name)

            # Determine layer preference based on net type
            if net_type == NetType.GROUND:
                layer_pref = 'B.Cu'  # Ground on bottom layer
            elif net_type == NetType.POWER:
                layer_pref = 'F.Cu'  # Power on top with potential via to bottom
            else:
                layer_pref = 'F.Cu'  # Signals on top by default

            # Determine routing order
            # Lower number = route first
            # CRITICAL: GND routes first because it uses B.Cu with vias
            # Other nets on F.Cu must route around GND vias
            if net_type == NetType.GROUND:
                order = 5  # Ground FIRST - creates vias that others must avoid
            elif net_type == NetType.SIGNAL and len(pins) == 2:
                # Short signals after ground
                order = 20
            elif net_type == NetType.POWER:
                order = 30
            else:
                order = 40 + len(pins)  # Longer nets last

            plans.append(RoutingPlan(
                net=net_name,
                net_type=net_type,
                pins=pins,
                layer_preference=layer_pref,
                routing_order=order
            ))

        return plans

    def _route_net(self, plan: RoutingPlan) -> Route:
        """
        Route a single net using intelligent decisions.

        MY ROUTING STRATEGY:
        1. For 2-pin nets: Try direct path, if blocked try L-route or via
        2. For multi-pin nets: Connect nearest pins first, build a tree
        3. Use appropriate layer based on net type
        4. Avoid existing routes and component bodies
        """
        route = Route(net=plan.net)

        if len(plan.pins) < 2:
            route.error = "Need at least 2 pins"
            return route

        # For multi-pin nets, use MST-style connection
        # Start with first pin, connect nearest unconnected pin each step
        connected = {plan.pins[0]}
        unconnected = list(plan.pins[1:])

        # Track all connection points (can branch from any point on the tree)
        connection_points: Set[Tuple[float, float]] = {(plan.pins[0].x, plan.pins[0].y)}

        while unconnected:
            # Find closest pair (connected point to unconnected pin)
            best_dist = float('inf')
            best_conn_pt = None
            best_uc_pin = None

            for uc_pin in unconnected:
                for conn_pt in connection_points:
                    dist = abs(uc_pin.x - conn_pt[0]) + abs(uc_pin.y - conn_pt[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_conn_pt = conn_pt
                        best_uc_pin = uc_pin

            if best_uc_pin is None:
                route.error = "Could not find connection point"
                return route

            # Route from best_conn_pt to best_uc_pin
            segments, vias, success = self._route_two_points(
                best_conn_pt,
                (best_uc_pin.x, best_uc_pin.y),
                plan.net,
                plan.layer_preference,
                plan.net_type
            )

            if not success:
                route.error = f"Failed to route from {best_conn_pt} to ({best_uc_pin.x}, {best_uc_pin.y})"
                return route

            # Add segments and vias to route
            route.segments.extend(segments)
            route.vias.extend(vias)

            # Update connection tracking
            connected.add(best_uc_pin)
            unconnected.remove(best_uc_pin)

            # Add segment endpoints as new connection points
            for seg in segments:
                connection_points.add(seg.start)
                connection_points.add(seg.end)
            for via in vias:
                connection_points.add(via.position)

        # Deduplicate vias (same position = same via)
        seen_via_positions = set()
        unique_vias = []
        for via in route.vias:
            via_key = (round(via.position[0], 2), round(via.position[1], 2))
            if via_key not in seen_via_positions:
                seen_via_positions.add(via_key)
                unique_vias.append(via)
        route.vias = unique_vias

        route.success = True
        return route

    def _route_two_points(self, start: Tuple[float, float], end: Tuple[float, float],
                          net: str, layer_pref: str, net_type: NetType
                          ) -> Tuple[List[Segment], List[Via], bool]:
        """
        Route between two points using intelligent path selection.

        MY ROUTING APPROACH:
        1. Try direct line if horizontal or vertical
        2. Try L-route (horizontal-vertical or vertical-horizontal)
        3. If blocked on preferred layer, try via + other layer
        4. If still blocked, try going around (add waypoints)
        """
        segments = []
        vias = []

        sx, sy = start
        ex, ey = end
        layer = layer_pref

        # Choose trace width based on net type
        width = self.trace_width
        if net_type == NetType.POWER:
            width = max(self.trace_width, 0.4)  # Wider for power
        elif net_type == NetType.GROUND:
            width = max(self.trace_width, 0.4)  # Wider for ground

        # For ground nets routed on B.Cu, we need vias at each endpoint to connect to F.Cu pads
        # SMD pads are always on F.Cu, so B.Cu routes need vias
        needs_via_at_start = (layer == 'B.Cu')
        needs_via_at_end = (layer == 'B.Cu')

        # STRATEGY 1: Direct line (if aligned)
        if abs(sx - ex) < 0.01 or abs(sy - ey) < 0.01:
            # Aligned - try direct route
            if not self._is_blocked(start, end, layer, net):
                if needs_via_at_start:
                    vias.append(Via(position=start, net=net, from_layer='F.Cu', to_layer='B.Cu'))
                segments.append(Segment(
                    start=start, end=end,
                    layer=layer, width=width, net=net
                ))
                if needs_via_at_end:
                    vias.append(Via(position=end, net=net, from_layer='B.Cu', to_layer='F.Cu'))
                return segments, vias, True

        # STRATEGY 2: L-route on preferred layer
        # Try horizontal-then-vertical
        mid1 = (ex, sy)
        if not self._is_blocked(start, mid1, layer, net) and not self._is_blocked(mid1, end, layer, net):
            if needs_via_at_start:
                vias.append(Via(position=start, net=net, from_layer='F.Cu', to_layer='B.Cu'))
            segments.append(Segment(start=start, end=mid1, layer=layer, width=width, net=net))
            segments.append(Segment(start=mid1, end=end, layer=layer, width=width, net=net))
            if needs_via_at_end:
                vias.append(Via(position=end, net=net, from_layer='B.Cu', to_layer='F.Cu'))
            return segments, vias, True

        # Try vertical-then-horizontal
        mid2 = (sx, ey)
        if not self._is_blocked(start, mid2, layer, net) and not self._is_blocked(mid2, end, layer, net):
            if needs_via_at_start:
                vias.append(Via(position=start, net=net, from_layer='F.Cu', to_layer='B.Cu'))
            segments.append(Segment(start=start, end=mid2, layer=layer, width=width, net=net))
            segments.append(Segment(start=mid2, end=end, layer=layer, width=width, net=net))
            if needs_via_at_end:
                vias.append(Via(position=end, net=net, from_layer='B.Cu', to_layer='F.Cu'))
            return segments, vias, True

        # STRATEGY 3: Via to other layer + route there
        other_layer = 'B.Cu' if layer == 'F.Cu' else 'F.Cu'

        # Via at start, route on other layer
        if not self._is_blocked(start, end, other_layer, net):
            # Can route on other layer
            vias.append(Via(position=start, net=net, from_layer=layer, to_layer=other_layer))
            segments.append(Segment(start=start, end=end, layer=other_layer, width=width, net=net))
            vias.append(Via(position=end, net=net, from_layer=other_layer, to_layer=layer))
            return segments, vias, True

        # Try L-route on other layer with vias
        mid1 = (ex, sy)
        if not self._is_blocked(start, mid1, other_layer, net) and not self._is_blocked(mid1, end, other_layer, net):
            vias.append(Via(position=start, net=net, from_layer=layer, to_layer=other_layer))
            segments.append(Segment(start=start, end=mid1, layer=other_layer, width=width, net=net))
            segments.append(Segment(start=mid1, end=end, layer=other_layer, width=width, net=net))
            vias.append(Via(position=end, net=net, from_layer=other_layer, to_layer=layer))
            return segments, vias, True

        mid2 = (sx, ey)
        if not self._is_blocked(start, mid2, other_layer, net) and not self._is_blocked(mid2, end, other_layer, net):
            vias.append(Via(position=start, net=net, from_layer=layer, to_layer=other_layer))
            segments.append(Segment(start=start, end=mid2, layer=other_layer, width=width, net=net))
            segments.append(Segment(start=mid2, end=end, layer=other_layer, width=width, net=net))
            vias.append(Via(position=end, net=net, from_layer=other_layer, to_layer=layer))
            return segments, vias, True

        # STRATEGY 4: Detour around obstacles
        # Try adding a waypoint above or below the direct path
        for offset in [2.0, -2.0, 4.0, -4.0, 6.0, -6.0]:
            # Try horizontal offset
            way_y = (sy + ey) / 2 + offset
            waypoint = ((sx + ex) / 2, way_y)

            if self._is_valid_point(waypoint):
                # Try routing start -> waypoint -> end
                segs1, vias1, ok1 = self._try_L_route(start, waypoint, layer, width, net)
                if ok1:
                    segs2, vias2, ok2 = self._try_L_route(waypoint, end, layer, width, net)
                    if ok2:
                        segments.extend(segs1)
                        segments.extend(segs2)
                        vias.extend(vias1)
                        vias.extend(vias2)
                        return segments, vias, True

        # Failed all strategies
        return [], [], False

    def _try_L_route(self, start: Tuple[float, float], end: Tuple[float, float],
                     layer: str, width: float, net: str
                     ) -> Tuple[List[Segment], List[Via], bool]:
        """Try L-route between two points."""
        segments = []
        vias = []
        sx, sy = start
        ex, ey = end

        # Try horizontal-then-vertical
        mid1 = (ex, sy)
        if not self._is_blocked(start, mid1, layer, net) and not self._is_blocked(mid1, end, layer, net):
            segments.append(Segment(start=start, end=mid1, layer=layer, width=width, net=net))
            segments.append(Segment(start=mid1, end=end, layer=layer, width=width, net=net))
            return segments, vias, True

        # Try vertical-then-horizontal
        mid2 = (sx, ey)
        if not self._is_blocked(start, mid2, layer, net) and not self._is_blocked(mid2, end, layer, net):
            segments.append(Segment(start=start, end=mid2, layer=layer, width=width, net=net))
            segments.append(Segment(start=mid2, end=end, layer=layer, width=width, net=net))
            return segments, vias, True

        return [], [], False

    def _is_blocked(self, start: Tuple[float, float], end: Tuple[float, float],
                    layer: str, current_net: str = '') -> bool:
        """
        Check if a path is blocked by existing routes, pads, or vias.

        Checks for:
        1. Segment crossings with existing routes on same layer
        2. Track-to-track clearance violations (parallel tracks too close)
        3. Path passing through pads belonging to other nets (F.Cu only, SMD pads)
        4. Path passing through vias of other nets (vias occupy BOTH layers)
        """
        # Calculate minimum clearance between track edges
        # Each track has width/2, plus need clearance between them
        min_track_clearance = self.trace_width + self.clearance

        # Check if this segment would cross OR be too close to any existing segment
        for existing in self.routed_segments.get(layer, []):
            # Skip if same net (can touch)
            if existing.net == current_net:
                continue

            # Check for crossing
            if self._segments_cross(start, end, existing.start, existing.end):
                return True

            # Check for track-to-track clearance violation
            # This catches parallel tracks that are too close
            if self._segments_too_close(start, end, existing.start, existing.end, min_track_clearance):
                return True

        # Check if path passes through any pads of OTHER nets (F.Cu only for SMD)
        if layer == 'F.Cu':
            for pad_pos, pad_net in self.pad_locations.items():
                if pad_net != current_net:
                    # Check if the segment passes within clearance of this pad
                    # Pad radius ~0.45mm + trace half-width + clearance ~= 0.8mm total
                    pad_avoidance = 0.45 + self.trace_width / 2 + self.clearance
                    if self._segment_near_point(start, end, pad_pos, pad_avoidance):
                        return True

        # Check if path passes through vias of OTHER nets (vias go through all layers)
        for via_pos, via_net in self.via_locations.items():
            if via_net != current_net:
                # Vias block both layers, check with via diameter clearance
                if self._segment_near_point(start, end, via_pos, 0.4 + self.clearance):
                    return True

        return False

    def _segments_too_close(self, a1: Tuple[float, float], a2: Tuple[float, float],
                            b1: Tuple[float, float], b2: Tuple[float, float],
                            min_dist: float) -> bool:
        """
        Check if two line segments are closer than min_dist at any point.
        This catches parallel tracks that would violate clearance rules.
        """
        # Sample points along segment A and check distance to segment B
        # This is a simple approximation but catches most violations
        num_samples = max(3, int(math.sqrt((a2[0]-a1[0])**2 + (a2[1]-a1[1])**2) / 0.5))

        for i in range(num_samples + 1):
            t = i / num_samples
            px = a1[0] + t * (a2[0] - a1[0])
            py = a1[1] + t * (a2[1] - a1[1])

            # Check distance from this point to segment B
            dist = self._point_to_segment_dist((px, py), b1, b2)
            if dist < min_dist:
                return True

        return False

    def _point_to_segment_dist(self, point: Tuple[float, float],
                                seg_start: Tuple[float, float],
                                seg_end: Tuple[float, float]) -> float:
        """Calculate minimum distance from point to line segment."""
        dx = seg_end[0] - seg_start[0]
        dy = seg_end[1] - seg_start[1]
        seg_len_sq = dx*dx + dy*dy

        if seg_len_sq < 0.0001:
            # Degenerate segment - just return distance to start
            return math.sqrt((point[0] - seg_start[0])**2 + (point[1] - seg_start[1])**2)

        # Project point onto line segment
        t = max(0, min(1, ((point[0] - seg_start[0]) * dx + (point[1] - seg_start[1]) * dy) / seg_len_sq))

        # Closest point on segment
        closest_x = seg_start[0] + t * dx
        closest_y = seg_start[1] + t * dy

        # Distance from point to closest point on segment
        return math.sqrt((point[0] - closest_x)**2 + (point[1] - closest_y)**2)

    def _segment_near_point(self, seg_start: Tuple[float, float], seg_end: Tuple[float, float],
                            point: Tuple[float, float], radius: float) -> bool:
        """Check if a line segment passes within radius of a point."""
        # Vector from start to end
        dx = seg_end[0] - seg_start[0]
        dy = seg_end[1] - seg_start[1]
        seg_len_sq = dx*dx + dy*dy

        if seg_len_sq < 0.0001:
            # Degenerate segment - just check distance
            dist_sq = (point[0] - seg_start[0])**2 + (point[1] - seg_start[1])**2
            return dist_sq < radius * radius

        # Project point onto line segment
        t = max(0, min(1, ((point[0] - seg_start[0]) * dx + (point[1] - seg_start[1]) * dy) / seg_len_sq))

        # Closest point on segment
        closest_x = seg_start[0] + t * dx
        closest_y = seg_start[1] + t * dy

        # Distance from point to closest point on segment
        dist_sq = (point[0] - closest_x)**2 + (point[1] - closest_y)**2
        return dist_sq < radius * radius

    def _segments_cross(self, a1: Tuple[float, float], a2: Tuple[float, float],
                        b1: Tuple[float, float], b2: Tuple[float, float]) -> bool:
        """Check if two line segments cross each other."""
        # Quick bounding box check first
        if (max(a1[0], a2[0]) < min(b1[0], b2[0]) - 0.01 or
            min(a1[0], a2[0]) > max(b1[0], b2[0]) + 0.01 or
            max(a1[1], a2[1]) < min(b1[1], b2[1]) - 0.01 or
            min(a1[1], a2[1]) > max(b1[1], b2[1]) + 0.01):
            return False

        # Cross product method for proper intersection
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        return ccw(a1,b1,b2) != ccw(a2,b1,b2) and ccw(a1,a2,b1) != ccw(a1,a2,b2)

    def _is_valid_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is within board bounds."""
        margin = 1.0
        return (margin < point[0] < self.board_width - margin and
                margin < point[1] < self.board_height - margin)

    def _mark_occupied(self, segment: Segment):
        """Mark a segment's area as occupied."""
        # Add clearance around the segment
        x1, y1 = segment.start
        x2, y2 = segment.end

        # Create bounding box with clearance
        margin = segment.width / 2 + self.clearance
        rect = (
            min(x1, x2) - margin,
            min(y1, y2) - margin,
            max(x1, x2) + margin,
            max(y1, y2) + margin
        )
        self.occupied[segment.layer].append(rect)


# =============================================================================
# TEST
# =============================================================================

def test_intelligent_router():
    """Test the intelligent router."""
    from dataclasses import dataclass

    @dataclass
    class Position:
        x: float
        y: float
        rotation: float = 0.0
        width: float = 2.5
        height: float = 1.5

    print("=" * 70)
    print("INTELLIGENT ROUTER TEST")
    print("=" * 70)

    # Simple test: 3 components, 3 nets
    parts_db = {
        'parts': {
            'R1': {
                'value': '330R',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'LED_A', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'D1': {
                'value': 'LED',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'LED_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'J1': {
                'value': 'CONN',
                'footprint': 'Header_1x02',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (0, 0), 'size': (1.7, 1.7)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.7, 1.7)},
                ]
            },
        },
        'nets': {
            'VIN': {'pins': [('J1', '1'), ('R1', '1')]},
            'GND': {'pins': [('J1', '2'), ('D1', '2')]},
            'LED_A': {'pins': [('R1', '2'), ('D1', '1')]},
        }
    }

    # Components on a line with space between them
    placement = {
        'J1': Position(x=10.0, y=15.0),
        'R1': Position(x=25.0, y=15.0),
        'D1': Position(x=40.0, y=15.0),
    }

    # Print pin positions
    print("\nPIN POSITIONS:")
    for ref, pos in placement.items():
        part = parts_db['parts'][ref]
        for pin in part['pins']:
            pad_x = pos.x + pin['offset'][0]
            pad_y = pos.y + pin['offset'][1]
            print(f"  {ref}.{pin['number']} ({pin['net']}): ({pad_x:.2f}, {pad_y:.2f})")

    # Create router and route
    router = IntelligentRouter(
        board_width=50.0,
        board_height=30.0,
        trace_width=0.25,
        clearance=0.15
    )

    routes = router.route_all(parts_db, placement)

    # Print results
    print("\nROUTING RESULTS:")
    for net_name, route in routes.items():
        status = "OK" if route.success else f"FAIL: {route.error}"
        print(f"  {net_name}: {len(route.segments)} segments, {len(route.vias)} vias [{status}]")

        if route.success:
            for i, seg in enumerate(route.segments):
                print(f"    Segment {i+1}: {seg.start} -> {seg.end} on {seg.layer}")
            for i, via in enumerate(route.vias):
                print(f"    Via {i+1}: {via.position} ({via.from_layer} -> {via.to_layer})")

    # Summary
    routed = sum(1 for r in routes.values() if r.success)
    print(f"\nSUMMARY: {routed}/{len(routes)} nets routed successfully")

    return routes


if __name__ == '__main__':
    test_intelligent_router()
