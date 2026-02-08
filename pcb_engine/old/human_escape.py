"""
PCB Engine - Human-Like Escape Calculation
==========================================

This module calculates escape routes following how a human expert thinks.

HUMAN EXPERT MENTAL MODEL:
==========================
1. "Escape = get OUT of the component body first"
2. "Go PERPENDICULAR to the component edge, then turn"
3. "Escape toward where you need to connect"
4. "Multiple pins escaping same direction? Stagger them"

ESCAPE RULES:
=============
- Pin on LEFT side → escape LEFT
- Pin on RIGHT side → escape RIGHT
- Pin on TOP side → escape UP
- Pin on BOTTOM side → escape DOWN

AFTER ESCAPE:
=============
- The signal route will handle getting to the destination
- Escape is SHORT (just clear the component body)
- Route is where the REAL pathfinding happens
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class EscapeRoute:
    """A single escape route from a pad to routing grid"""
    pin: str
    net: str
    layer: str

    # Geometry
    start: Tuple[float, float]      # Pad center
    end: Tuple[float, float]        # Grid-aligned escape endpoint
    direction: Tuple[float, float]  # Unit vector
    length: float

    # For debugging
    direction_name: str  # 'N', 'S', 'E', 'W'


class HumanLikeEscaper:
    """
    Calculates escape routes like a human would.

    Human principle: Escape is simple - just get clear of the component.
    The complexity comes in routing, not escaping.

    ROOT CAUSE FIX: Track occupied grid cells to prevent escape overlaps.
    """

    def __init__(self, grid_size: float = 0.5,
                 min_escape_length: float = 1.0,
                 max_escape_length: float = 3.0):
        self.grid_size = grid_size
        self.min_escape_length = min_escape_length
        self.max_escape_length = max_escape_length
        # Track occupied escape endpoints AND paths to prevent different nets from overlapping
        self.occupied_endpoints: Dict[Tuple[float, float], str] = {}  # {(x,y): net_name}
        self.occupied_paths: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []  # [(start, end, net)]

    def calculate_all_escapes(self, parts_db: Dict,
                               placements: Dict) -> Dict[str, Dict[str, EscapeRoute]]:
        """
        Calculate escapes for all placed components.

        Returns: {ref: {pin_number: EscapeRoute}}
        """
        escapes = {}
        parts = parts_db.get('parts', {})

        for ref, pos in placements.items():
            part = parts.get(ref, {})
            comp_escapes = self._calculate_component_escapes(ref, part, pos)
            if comp_escapes:
                escapes[ref] = comp_escapes

        return escapes

    def _calculate_component_escapes(self, ref: str, part: Dict,
                                      pos) -> Dict[str, EscapeRoute]:
        """Calculate escapes for one component"""
        escapes = {}
        used_pins = part.get('used_pins', [])

        if not used_pins:
            return escapes

        # Group pins by which side they're on
        pin_groups = self._group_pins_by_side(used_pins)

        for side, pins in pin_groups.items():
            if not pins:
                continue

            # Get escape direction for this side
            direction, direction_name = self._get_escape_direction(side)

            # Calculate escape length (stagger if multiple pins)
            for i, pin_info in enumerate(pins):
                base_escape_length = self._calculate_escape_length(i, len(pins))
                net = pin_info.get('net', '')

                # Get pin position
                offset = pin_info.get('offset', (0, 0))
                pin_x = pos.x + offset[0]
                pin_y = pos.y + offset[1]

                # ROOT CAUSE FIX: Find escape endpoint that doesn't conflict with other nets
                end_x, end_y, final_length = self._find_clear_endpoint(
                    pin_x, pin_y, direction, base_escape_length, net
                )

                # Register this endpoint AND path as occupied by this net
                self.occupied_endpoints[(end_x, end_y)] = net
                self.occupied_paths.append(((pin_x, pin_y), (end_x, end_y), net))

                escapes[pin_info['number']] = EscapeRoute(
                    pin=pin_info['number'],
                    net=net,
                    layer='F.Cu',
                    start=(pin_x, pin_y),
                    end=(end_x, end_y),
                    direction=direction,
                    length=final_length,
                    direction_name=direction_name,
                )

        return escapes

    def _group_pins_by_side(self, pins: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group pins by which side of the component they're on.

        Human looks at pin positions relative to component center:
        - Negative X offset = left side
        - Positive X offset = right side
        - Negative Y offset = top side
        - Positive Y offset = bottom side
        """
        groups = {'left': [], 'right': [], 'top': [], 'bottom': []}

        for pin in pins:
            offset = pin.get('offset', (0, 0))
            dx = offset[0] if isinstance(offset, (list, tuple)) else 0
            dy = offset[1] if isinstance(offset, (list, tuple)) and len(offset) > 1 else 0

            # Determine which side based on which offset is larger
            if abs(dx) > abs(dy):
                # Pin is more to the side than top/bottom
                side = 'right' if dx > 0 else 'left'
            elif abs(dy) > abs(dx):
                # Pin is more top/bottom than side
                side = 'bottom' if dy > 0 else 'top'
            else:
                # Equal - default to side (more common for 2-pin components)
                side = 'right' if dx >= 0 else 'left'

            groups[side].append(pin)

        return groups

    def _get_escape_direction(self, side: str) -> Tuple[Tuple[float, float], str]:
        """
        Get escape direction for a pin side.

        Human rule: ALWAYS escape AWAY from component body.
        """
        directions = {
            'left': ((-1.0, 0.0), 'W'),   # Left pins escape west
            'right': ((1.0, 0.0), 'E'),   # Right pins escape east
            'top': ((0.0, -1.0), 'N'),    # Top pins escape north
            'bottom': ((0.0, 1.0), 'S'),  # Bottom pins escape south
        }
        return directions.get(side, ((1.0, 0.0), 'E'))

    def _calculate_escape_length(self, pin_index: int, total_pins: int) -> float:
        """
        Calculate escape length, staggered for multiple pins on same side.

        Human technique: Stagger escapes so they don't overlap.
        First pin gets shortest escape, last pin gets longest.
        """
        if total_pins <= 1:
            return self.min_escape_length

        # Linear interpolation between min and max
        t = pin_index / (total_pins - 1) if total_pins > 1 else 0
        return self.min_escape_length + t * (self.max_escape_length - self.min_escape_length)

    def _find_clear_endpoint(self, pin_x: float, pin_y: float,
                             direction: Tuple[float, float],
                             base_length: float, net: str) -> Tuple[float, float, float]:
        """
        ROOT CAUSE FIX: Find an escape endpoint that doesn't conflict with other nets.

        MANHATTAN FIX: Ensures escape is purely horizontal OR vertical by keeping
        one coordinate constant (the pin's coordinate on the perpendicular axis).

        Checks both:
        1. Endpoint conflicts (two escapes ending at same point)
        2. Path crossings (one escape crossing another's path)

        Returns: (end_x, end_y, final_length)
        """
        max_attempts = 10  # Prevent infinite loops

        for attempt in range(max_attempts):
            escape_length = base_length + (attempt * self.grid_size)

            # Calculate escape endpoint - MANHATTAN: only move in ONE direction
            if direction[0] != 0:  # Horizontal escape (E or W)
                end_x = pin_x + direction[0] * escape_length
                end_x = round(end_x / self.grid_size) * self.grid_size
                end_y = pin_y  # Keep Y constant for horizontal escape
            else:  # Vertical escape (N or S)
                end_x = pin_x  # Keep X constant for vertical escape
                end_y = pin_y + direction[1] * escape_length
                end_y = round(end_y / self.grid_size) * self.grid_size

            endpoint = (end_x, end_y)
            new_path = ((pin_x, pin_y), (end_x, end_y))

            # Check 1: Is endpoint occupied by a DIFFERENT net?
            occupant = self.occupied_endpoints.get(endpoint)
            if occupant is not None and occupant != net:
                continue  # Try longer escape

            # Check 2: Does this path cross any existing escape path from a DIFFERENT net?
            has_crossing = False
            for existing_start, existing_end, existing_net in self.occupied_paths:
                if existing_net == net:
                    continue  # Same net - OK to cross

                if self._paths_cross(new_path[0], new_path[1], existing_start, existing_end):
                    has_crossing = True
                    break

            if not has_crossing:
                return end_x, end_y, escape_length

        # If all attempts in primary direction failed, try alternate directions
        # Human approach: prefer going DOWN (toward typical signal flow) over UP
        # This helps when components are stacked vertically (U1 -> R1 -> D1)
        alt_directions = [
            (0, 1),    # Down (S) - FIRST, for vertical signal flow
            (0, -1),   # Up (N)
            (-1, 0),   # Left (W)
            (1, 0),    # Right (E)
        ]

        for alt_dir in alt_directions:
            # Skip same direction and opposite direction (both likely blocked)
            if alt_dir == direction or alt_dir == (-direction[0], -direction[1]):
                continue

            for attempt in range(max_attempts):
                escape_length = base_length + (attempt * self.grid_size)

                # MANHATTAN: only move in ONE direction
                if alt_dir[0] != 0:  # Horizontal
                    end_x = pin_x + alt_dir[0] * escape_length
                    end_x = round(end_x / self.grid_size) * self.grid_size
                    end_y = pin_y
                else:  # Vertical
                    end_x = pin_x
                    end_y = pin_y + alt_dir[1] * escape_length
                    end_y = round(end_y / self.grid_size) * self.grid_size

                endpoint = (end_x, end_y)
                new_path = ((pin_x, pin_y), (end_x, end_y))

                # Check conflicts
                occupant = self.occupied_endpoints.get(endpoint)
                if occupant is not None and occupant != net:
                    continue

                has_crossing = False
                for existing_start, existing_end, existing_net in self.occupied_paths:
                    if existing_net == net:
                        continue
                    if self._paths_cross(new_path[0], new_path[1], existing_start, existing_end):
                        has_crossing = True
                        break

                if not has_crossing:
                    return end_x, end_y, escape_length

        # Ultimate fallback: use original direction furthest attempt
        # This will create a DRC error but at least we tried
        escape_length = base_length + (max_attempts * self.grid_size)
        if direction[0] != 0:  # Horizontal
            end_x = pin_x + direction[0] * escape_length
            end_x = round(end_x / self.grid_size) * self.grid_size
            end_y = pin_y
        else:  # Vertical
            end_x = pin_x
            end_y = pin_y + direction[1] * escape_length
            end_y = round(end_y / self.grid_size) * self.grid_size
        return end_x, end_y, escape_length

    def _paths_cross(self, p1: Tuple[float, float], p2: Tuple[float, float],
                     p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """
        Check if two line segments cross each other.

        Uses the cross-product method to determine intersection.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        # Cross product of directions
        cross = dx1 * dy2 - dy1 * dx2

        if abs(cross) < 1e-10:
            # Parallel lines - check for collinear overlap
            # For simplicity, check if any endpoints are on the other segment
            return self._point_on_segment(p1, p3, p4) or \
                   self._point_on_segment(p2, p3, p4) or \
                   self._point_on_segment(p3, p1, p2) or \
                   self._point_on_segment(p4, p1, p2)

        # Calculate intersection parameters
        dx3 = x1 - x3
        dy3 = y1 - y3

        t1 = (dx2 * dy3 - dy2 * dx3) / cross
        t2 = (dx1 * dy3 - dy1 * dx3) / cross

        # Intersection if both t1 and t2 are in [0, 1]
        return 0 <= t1 <= 1 and 0 <= t2 <= 1

    def _point_on_segment(self, p: Tuple[float, float],
                          seg_start: Tuple[float, float],
                          seg_end: Tuple[float, float],
                          tol: float = 0.01) -> bool:
        """Check if point p lies on segment from seg_start to seg_end"""
        px, py = p
        x1, y1 = seg_start
        x2, y2 = seg_end

        # Check if point is within bounding box
        if not (min(x1, x2) - tol <= px <= max(x1, x2) + tol and
                min(y1, y2) - tol <= py <= max(y1, y2) + tol):
            return False

        # Check if point is on the line (cross product should be ~0)
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        return abs(cross) < tol * 10  # Tolerance for floating point


def human_like_escapes(parts_db: Dict, placements: Dict,
                       grid_size: float = 0.5) -> Dict[str, Dict[str, EscapeRoute]]:
    """
    Main entry point for human-like escape calculation.

    Returns: {ref: {pin_number: EscapeRoute}}
    """
    escaper = HumanLikeEscaper(grid_size=grid_size)
    return escaper.calculate_all_escapes(parts_db, placements)
