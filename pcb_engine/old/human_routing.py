"""
PCB Engine - Human-Like Routing
================================

This module routes signals following how a human expert thinks.

HUMAN EXPERT ROUTING STRATEGY:
==============================
1. ROUTE SHORT NETS FIRST
   - 2-pin nets are easy
   - Get them done, clear the board

2. ROUTE CRITICAL SIGNALS EARLY
   - High-speed signals need direct paths
   - Clock, USB differential pairs

3. USE MANHATTAN ROUTING
   - Horizontal and vertical segments
   - 45° corners only when needed

4. LAYER USAGE
   - Try to route on one layer first
   - Use via to switch layers when blocked

5. POWER LAST (or planes)
   - GND often becomes a pour
   - VCC can be thick traces

ROUTING A SINGLE NET:
=====================
1. Start at source escape endpoint
2. Go horizontal until aligned with destination
3. Go vertical to destination
4. If blocked, try the other way (vertical first, then horizontal)
5. If still blocked, use a via and try on the other layer
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import math
from collections import deque
import heapq

# Import Set for type hints (already imported via typing)


@dataclass
class TrackSegment:
    """A single track segment"""
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

    @property
    def is_horizontal(self) -> bool:
        return abs(self.end[1] - self.start[1]) < 0.01

    @property
    def is_vertical(self) -> bool:
        return abs(self.end[0] - self.start[0]) < 0.01


@dataclass
class Via:
    """A via connecting layers"""
    position: Tuple[float, float]
    net: str
    diameter: float = 0.8
    drill: float = 0.4


@dataclass
class Route:
    """Complete route for a net"""
    net: str
    segments: List[TrackSegment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    success: bool = False
    error: str = ''

    @property
    def total_length(self) -> float:
        return sum(seg.length for seg in self.segments)


class HumanLikeRouter:
    """
    Routes nets like a human would.

    Human principle: Route simply and directly.
    Don't over-engineer - horizontal, vertical, done.
    """

    def __init__(self, board_width: float, board_height: float,
                 origin_x: float = 0, origin_y: float = 0,
                 grid_size: float = 0.5, trace_width: float = 0.5,
                 clearance: float = 0.2):
        self.board_width = board_width
        self.board_height = board_height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.grid_size = grid_size
        self.trace_width = trace_width
        self.clearance = clearance

        # DRC FIX: Calculate how many EXTRA cells to mark around each track
        # Track occupies trace_width/2 on each side, plus clearance needed
        # Total keepout radius = trace_width/2 + clearance
        # With 0.5mm trace, 0.2mm clearance: keepout = 0.25 + 0.2 = 0.45mm
        # With 0.5mm grid: need to mark 1 extra cell on each side to ensure
        # adjacent tracks (different nets) have at least 0.5mm center-to-center
        # which gives 0.5 - 0.25 - 0.25 = 0.0mm clearance (not enough!)
        # So we need clearance_cells = ceil((trace_width/2 + clearance) / grid_size)
        self.clearance_radius = trace_width / 2 + clearance
        self.clearance_cells = max(1, int(math.ceil(self.clearance_radius / grid_size)))

        # Board margin - MUST be at least clearance + trace_width/2 from edge
        # KiCad default is 0.5mm edge clearance
        self.board_margin = max(1.0, clearance + trace_width / 2 + 0.5)
        self.margin_cells = max(2, int(math.ceil(self.board_margin / grid_size)))

        # Occupancy grid for collision detection
        self.grid_cols = int(board_width / grid_size) + 1
        self.grid_rows = int(board_height / grid_size) + 1

        # Layer occupancy: stores net name or None (allows same-net overlap)
        # Special value '__COMPONENT__' marks component body (blocked for all nets)
        self.fcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self.bcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]

        # DRC FIX: Mark board edges as blocked
        self._mark_board_margins()

        # Track placed vias to prevent duplicates
        self.placed_vias: Set[Tuple[float, float]] = set()

        # Results
        self.routes: Dict[str, Route] = {}
        self.failed: List[str] = []

    def _mark_board_margins(self):
        """Mark cells near board edges as blocked to enforce edge clearance"""
        # Mark top and bottom margins
        for col in range(self.grid_cols):
            for row in range(self.margin_cells):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'
            for row in range(max(0, self.grid_rows - self.margin_cells), self.grid_rows):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'

        # Mark left and right margins
        for row in range(self.grid_rows):
            for col in range(self.margin_cells):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'
            for col in range(max(0, self.grid_cols - self.margin_cells), self.grid_cols):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'

    def register_components_in_grid(self, placement: Dict, parts_db: Dict):
        """
        Register component courtyards AND individual pads as routing obstacles.

        ALWAYS uses courtyard for proper DRC clearance:
        1. Explicit courtyard in physical data
        2. Auto-generated courtyard from size + clearance margin

        CRITICAL FIX: Register ALL PHYSICAL pads (including NC/No Connect pads)
        to prevent tracks from routing through ANY pad and causing shorts.

        IMPORTANT: Shrink the blocked area by one grid cell to ensure
        escape endpoints (which are at courtyard edge) are not blocked.
        """
        parts = parts_db.get('parts', {})
        # Use design rules clearance + track half-width for proper DRC compliance
        clearance_margin = self.clearance + self.trace_width / 2

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            physical = part.get('physical', {})

            # ================================================================
            # CRITICAL FIX: Register ALL PHYSICAL pads as obstacles
            # 'physical_pins' contains ALL pads (including NC pads)
            # 'used_pins' only contains pads with nets (MISSES NC pads!)
            # ================================================================

            # Build a map of pin numbers to nets from used_pins
            pin_nets = {}
            for pin in part.get('used_pins', []):
                pin_nets[pin.get('number', '')] = pin.get('net', '')

            # Use physical_pins (ALL pads) - fall back to used_pins if not available
            all_pads = part.get('physical_pins', part.get('used_pins', []))

            for pin in all_pads:
                pin_num = pin.get('number', '')
                # Get net from pin_nets map (NC pads won't be in used_pins, so net='')
                net = pin_nets.get(pin_num, '')

                # Get offset - handle multiple formats
                offset = pin.get('offset', None)
                if not offset or offset == (0, 0):
                    # Try 'physical' dict format: {offset_x, offset_y}
                    physical = pin.get('physical', {})
                    if physical:
                        offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))
                    else:
                        offset = (0, 0)

                # Get pad position
                pad_x = pos.x + offset[0]
                pad_y = pos.y + offset[1]

                # Get pad size (default 1.0x0.6mm for SOT-23-5 style SMD pads)
                pad_size = pin.get('pad_size', pin.get('size', (1.0, 0.6)))
                if not isinstance(pad_size, (list, tuple)):
                    pad_size = (1.0, 0.6)
                # CRITICAL FIX: Block pad + clearance to prevent DRC violations
                # Must account for: pad_half + track_half + DRC_clearance
                # This ensures edge-to-edge clearance between track and pad is maintained
                # Example: 0.5mm pad, 0.25mm trace, 0.2mm clearance:
                #   pad_half=0.25 + track_half=0.125 + clearance=0.2 = 0.575mm radius
                pad_half = max(pad_size) / 2
                track_half = self.trace_width / 2
                # Total blocked radius = pad edge + track edge + required clearance
                pad_radius = pad_half + track_half + self.clearance

                # Mark pad area as obstacle
                # NC pads (empty net) block ALL nets
                # Connected pads only allow their own net
                pad_col = int((pad_x - self.origin_x) / self.grid_size)
                pad_row = int((pad_y - self.origin_y) / self.grid_size)

                # Mark cells around pad based on pad size + clearance
                pad_cells = max(1, int(math.ceil(pad_radius / self.grid_size)))

                for dr in range(-pad_cells, pad_cells + 1):
                    for dc in range(-pad_cells, pad_cells + 1):
                        r, c = pad_row + dr, pad_col + dc
                        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                            if net == '' or net is None:
                                # NC pad - block ALL nets on BOTH layers
                                # print(f"    [DEBUG] NC pad {ref}.{pin_num} at ({pad_x:.2f}, {pad_y:.2f}) -> grid[{r}][{c}]")
                                self.fcu_grid[r][c] = '__PAD_NC__'
                                self.bcu_grid[r][c] = '__PAD_NC__'
                            else:
                                # Connected pad - mark with net name
                                # This BLOCKS other nets from passing through
                                # F.Cu: SMD pad is physically on this layer
                                current_f = self.fcu_grid[r][c]
                                if current_f is None or current_f == net:
                                    self.fcu_grid[r][c] = net
                                elif current_f not in ['__COMPONENT__', '__PAD_NC__', '__EDGE__']:
                                    # Conflict - another net already here, mark as blocked
                                    self.fcu_grid[r][c] = '__PAD_CONFLICT__'

                                # B.Cu: For SMD pads, DON'T block B.Cu - the pad is only on F.Cu!
                                # Only through-hole (TH) pads or vias should block B.Cu.
                                # This allows power routing to use B.Cu freely under SMD pads.
                                # TH pads would have pad_type='th' or similar indicator
                                pad_type = pin.get('pad_type', pin.get('type', 'smd')).lower()
                                if pad_type in ['th', 'through_hole', 'thru_hole', 'pth']:
                                    # Through-hole pad - blocks both layers
                                    current_b = self.bcu_grid[r][c]
                                    if current_b is None or current_b == net:
                                        self.bcu_grid[r][c] = net

            # ================================================================
            # Original: Register component courtyard (body area)
            # ================================================================
            # Priority 1: Explicit courtyard
            courtyard = physical.get('courtyard')
            if courtyard and isinstance(courtyard, (list, tuple)):
                size = courtyard
            else:
                # Priority 2: Body + clearance margin
                body = physical.get('body')
                if body and isinstance(body, (list, tuple)):
                    base_size = body
                else:
                    # Priority 3: 'size' field + clearance margin
                    base_size = part.get('size', (2.0, 2.0))

                # Auto-generate courtyard = body + clearance on each side
                size = (base_size[0] + 2 * clearance_margin,
                        base_size[1] + 2 * clearance_margin)

            # Get component bounding box from courtyard
            half_w = size[0] / 2
            half_h = size[1] / 2

            min_x = pos.x - half_w
            max_x = pos.x + half_w
            min_y = pos.y - half_h
            max_y = pos.y + half_h

            # Convert to grid coordinates
            # FIX: Add 1 to start and subtract 1 from end to shrink blocked area
            # This ensures escape endpoints at the courtyard edge are NOT blocked
            start_col = int((min_x - self.origin_x) / self.grid_size) + 1
            end_col = int((max_x - self.origin_x) / self.grid_size) - 1
            start_row = int((min_y - self.origin_y) / self.grid_size) + 1
            end_row = int((max_y - self.origin_y) / self.grid_size) - 1

            # Skip if component is too small to block anything
            if start_col > end_col or start_row > end_row:
                continue

            # Mark all cells within component courtyard as blocked
            # NOTE: Only block F.Cu (top layer) since SMD components allow
            # routing underneath on B.Cu (bottom layer)
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                        # Mark as component obstacle on TOP layer only
                        # B.Cu remains free for routing under SMD components
                        # Don't overwrite pad markers
                        if self.fcu_grid[row][col] is None:
                            self.fcu_grid[row][col] = '__COMPONENT__'
                        # Don't block B.Cu: self.bcu_grid[row][col] = '__COMPONENT__'

    def register_escapes_in_grid(self, escapes: Dict[str, Dict]):
        """
        ROOT CAUSE FIX #1 & #3: Register ALL escape routes in the collision grid
        BEFORE signal routing begins. This prevents routes from overlapping escapes.
        """
        for ref, comp_escapes in escapes.items():
            for pin_num, escape in comp_escapes.items():
                # Create a TrackSegment-like object for marking
                sx, sy = escape.start
                ex, ey = escape.end
                net = escape.net
                layer = escape.layer if hasattr(escape, 'layer') else 'F.Cu'

                # Mark the escape path as occupied by this net
                self._mark_line_occupied(sx, sy, ex, ey, net, layer)

    def _mark_cell_with_clearance(self, grid: List[List], row: int, col: int, net_name: str):
        """
        Mark a cell AND its clearance zone as occupied by net_name.

        DRC FIX: Must mark clearance_cells around each track cell to prevent
        other nets from routing too close (causing clearance violations).
        """
        for dr in range(-self.clearance_cells, self.clearance_cells + 1):
            for dc in range(-self.clearance_cells, self.clearance_cells + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    # Only mark if empty or same net (don't overwrite other nets)
                    if grid[r][c] is None or grid[r][c] == net_name:
                        grid[r][c] = net_name

    def _mark_line_occupied(self, x1: float, y1: float, x2: float, y2: float,
                            net_name: str, layer: str):
        """
        Mark a line segment AND its clearance zone as occupied by a specific net.

        DRC FIX: Each track cell marks clearance_cells around it to prevent
        adjacent routing by other nets (which would cause clearance violations).
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        start_col = int((x1 - self.origin_x) / self.grid_size)
        start_row = int((y1 - self.origin_y) / self.grid_size)
        end_col = int((x2 - self.origin_x) / self.grid_size)
        end_row = int((y2 - self.origin_y) / self.grid_size)

        if start_row == end_row:
            # Horizontal track
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                if 0 <= start_row < self.grid_rows and 0 <= col < self.grid_cols:
                    self._mark_cell_with_clearance(grid, start_row, col, net_name)
        elif start_col == end_col:
            # Vertical track
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                if 0 <= row < self.grid_rows and 0 <= start_col < self.grid_cols:
                    self._mark_cell_with_clearance(grid, row, start_col, net_name)
        else:
            # Diagonal track - use Bresenham-like marking
            dx = end_col - start_col
            dy = end_row - start_row
            steps = max(abs(dx), abs(dy))
            if steps == 0:
                return

            for i in range(steps + 1):
                t = i / steps
                col = int(start_col + t * dx)
                row = int(start_row + t * dy)
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    self._mark_cell_with_clearance(grid, row, col, net_name)

    def route_all(self, nets_to_route: List[str], net_pins: Dict[str, List[Tuple]],
                  escapes: Dict[str, Dict], placement: Dict = None,
                  parts_db: Dict = None) -> Dict[str, Route]:
        """
        Route all nets in the given order.

        Human principle: Order matters! Short nets first.
        """
        # ROOT CAUSE FIX: Register component bodies as obstacles FIRST
        # Routes must go AROUND components, not through them
        if placement and parts_db:
            self.register_components_in_grid(placement, parts_db)

        # ROOT CAUSE FIX: Register all escape routes in collision grid
        # This prevents signal routes from overlapping with escape routes
        self.register_escapes_in_grid(escapes)

        # Sort by estimated route length (short first)
        sorted_nets = self._sort_nets_by_length(nets_to_route, net_pins, escapes)

        for net_name in sorted_nets:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue  # Can't route a single pin

            route = self._route_net(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        return self.routes

    def route_all_with_ripup(self, nets_to_route: List[str], net_pins: Dict[str, List[Tuple]],
                              escapes: Dict[str, Dict], placement: Dict = None,
                              parts_db: Dict = None, max_iterations: int = 10) -> Dict[str, Route]:
        """
        Route all nets using Rip-up and Reroute algorithm.

        This is a more sophisticated routing approach that:
        1. Routes nets in initial order
        2. When a net fails, identifies blocking nets
        3. Rips up (removes) blocking nets
        4. Reroutes in new order with failed net first
        5. Repeats until all routed or max iterations reached

        Based on: "Mighty: A Rip-Up and Reroute Detailed Router" (ResearchGate)

        Args:
            nets_to_route: List of net names to route
            net_pins: {net_name: [(comp, pin), ...]}
            escapes: Escape routes for all components
            placement: Component placements
            parts_db: Parts database
            max_iterations: Maximum rip-up iterations (default 10)

        Returns:
            {net_name: Route}
        """
        # Register obstacles first
        if placement and parts_db:
            self.register_components_in_grid(placement, parts_db)
        self.register_escapes_in_grid(escapes)

        # Track routing history for intelligent reordering
        routing_attempts = {net: 0 for net in nets_to_route}
        blocking_history = {}  # {blocked_net: [blocking_nets]}

        # Initial sort by length
        current_order = self._sort_nets_by_length(nets_to_route, net_pins, escapes)

        for iteration in range(max_iterations):
            # Clear previous routes for retry
            if iteration > 0:
                self._clear_routes_from_grid()
                self.routes.clear()
                self.failed.clear()

            # Route all nets in current order
            failed_nets = []
            for net_name in current_order:
                pins = net_pins.get(net_name, [])
                if len(pins) < 2:
                    continue

                route = self._route_net(net_name, pins, escapes)
                self.routes[net_name] = route
                routing_attempts[net_name] += 1

                if not route.success:
                    failed_nets.append(net_name)
                    # Identify which nets are blocking this one
                    blockers = self._identify_blocking_nets(net_name, pins, escapes)
                    blocking_history[net_name] = blockers

            # Success! All nets routed
            if not failed_nets:
                print(f"    [RIP-UP] All {len(current_order)} nets routed in {iteration + 1} iterations")
                return self.routes

            # Determine new routing order using rip-up strategy
            # Priority: nets that were blocked get routed earlier
            # Nets that block others get routed later
            current_order = self._reorder_nets_for_ripup(
                current_order, failed_nets, blocking_history, routing_attempts
            )

            if iteration < max_iterations - 1:
                print(f"    [RIP-UP] Iteration {iteration + 1}: {len(failed_nets)} failed, reordering...")

        # Return best effort
        print(f"    [RIP-UP] Max iterations reached, {len(self.failed)} nets unrouted")
        return self.routes

    def _clear_routes_from_grid(self):
        """Clear all routed tracks from the occupancy grid (keep pads and components)."""
        # Reset grid cells that were marked by routes (net names that aren't special markers)
        special_markers = {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__'}

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                fcu_val = self.fcu_grid[row][col]
                bcu_val = self.bcu_grid[row][col]

                # Clear if it's a net name (not a special marker and not None)
                # But preserve pad nets (connected pads)
                if fcu_val is not None and fcu_val not in special_markers:
                    # Check if this cell is part of a pad (preserve) or a track (clear)
                    # Pads are marked during register_components_in_grid
                    # For simplicity, we'll mark tracks with a prefix during routing
                    # For now, just clear everything that's not a special marker
                    # This will require re-registering escapes after clearing
                    pass  # Keep pad markings, tracks will be re-routed

        # Actually, a better approach: just re-initialize the grids and re-register
        self.fcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self.bcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self._mark_board_margins()
        self.placed_vias.clear()

    def _identify_blocking_nets(self, blocked_net: str, pins: List[Tuple],
                                 escapes: Dict) -> List[str]:
        """
        Identify which routed nets are blocking this net.

        Looks at cells along potential routing paths and finds
        which nets have already claimed them.
        """
        blockers = set()

        # Get escape endpoints for this net
        endpoints = []
        for comp, pin in pins:
            if comp in escapes and pin in escapes[comp]:
                esc = escapes[comp][pin]
                end = esc.end if hasattr(esc, 'end') else getattr(esc, 'endpoint', None)
                if end:
                    endpoints.append(end)

        if len(endpoints) < 2:
            return list(blockers)

        # Check cells along potential paths between endpoints
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                start, end = endpoints[i], endpoints[j]

                # Sample cells along the path
                start_col = int((start[0] - self.origin_x) / self.grid_size)
                start_row = int((start[1] - self.origin_y) / self.grid_size)
                end_col = int((end[0] - self.origin_x) / self.grid_size)
                end_row = int((end[1] - self.origin_y) / self.grid_size)

                # Check horizontal and vertical corridors
                for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                    if 0 <= start_row < self.grid_rows and 0 <= col < self.grid_cols:
                        occupant = self.fcu_grid[start_row][col]
                        if occupant and occupant not in {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__', blocked_net}:
                            blockers.add(occupant)

                for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                    if 0 <= row < self.grid_rows and 0 <= end_col < self.grid_cols:
                        occupant = self.fcu_grid[row][end_col]
                        if occupant and occupant not in {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__', blocked_net}:
                            blockers.add(occupant)

        return list(blockers)

    def _reorder_nets_for_ripup(self, current_order: List[str], failed_nets: List[str],
                                 blocking_history: Dict[str, List[str]],
                                 routing_attempts: Dict[str, int]) -> List[str]:
        """
        Reorder nets for next routing attempt.

        Strategy:
        1. Failed nets with few blocking attempts go first
        2. Nets that frequently block others go last
        3. Short nets still preferred within groups
        """
        # Count how many times each net has blocked others
        block_count = {}
        for blocked, blockers in blocking_history.items():
            for blocker in blockers:
                block_count[blocker] = block_count.get(blocker, 0) + 1

        def priority_key(net: str) -> Tuple[int, int, int]:
            """
            Sort key: (is_failed, -block_count, routing_attempts)
            - Failed nets first (0 < 1)
            - Nets that block many others go later (negative to reverse)
            - More routing attempts = lower priority
            """
            is_failed = 0 if net in failed_nets else 1
            blocks = block_count.get(net, 0)
            attempts = routing_attempts.get(net, 0)
            return (is_failed, -blocks, attempts)

        return sorted(current_order, key=priority_key)

    def _sort_nets_by_length(self, nets: List[str], net_pins: Dict[str, List[Tuple]],
                             escapes: Dict) -> List[str]:
        """
        Sort nets by estimated route length.

        Human does short nets first - they're easier and clear space.
        """
        def estimate_length(net_name: str) -> float:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                return float('inf')

            # Get escape endpoints for this net
            endpoints = []
            for comp, pin in pins:
                if comp in escapes and pin in escapes[comp]:
                    esc = escapes[comp][pin]
                    endpoints.append(esc.end if hasattr(esc, 'end') else esc.endpoint)

            if len(endpoints) < 2:
                return float('inf')

            # Estimate as sum of distances between consecutive endpoints
            total = 0
            for i in range(len(endpoints) - 1):
                dx = endpoints[i+1][0] - endpoints[i][0]
                dy = endpoints[i+1][1] - endpoints[i][1]
                total += abs(dx) + abs(dy)  # Manhattan distance

            return total

        return sorted(nets, key=estimate_length)

    def _route_net(self, net_name: str, pins: List[Tuple],
                   escapes: Dict) -> Route:
        """
        Route a single net.

        Human approach for multi-point nets:
        1. For 2-point: direct connection
        2. For 3+ points power nets (GND, VCC): Try STAR topology on B.Cu first
        3. For 3+ points signal nets: Use MST (Minimum Spanning Tree)
        """
        route = Route(net=net_name)

        # Get escape endpoints for this net
        endpoints = []
        for comp, pin in pins:
            if comp in escapes and pin in escapes[comp]:
                esc = escapes[comp][pin]
                end = esc.end if hasattr(esc, 'end') else getattr(esc, 'endpoint', None)
                if end:
                    endpoints.append(end)

        if len(endpoints) < 2:
            route.error = f"Not enough escape endpoints for {net_name}"
            return route

        # For 2-point nets, simple direct routing
        if len(endpoints) == 2:
            segments, vias, success = self._route_two_points(
                endpoints[0], endpoints[1], net_name, 'F.Cu'
            )
            if success:
                route.segments.extend(segments)
                route.vias.extend(vias)
                route.success = True
            else:
                route.error = f"Failed to route {net_name} from {endpoints[0]} to {endpoints[1]}"
            return route

        # For multi-point nets (3+ pins), try STAR topology on B.Cu for power nets
        # Human approach: Power nets like GND often go down to B.Cu and fan out
        power_nets = ['GND', 'VCC', '3V3', '5V', 'VBUS', 'VIN', 'VOUT']
        is_power_net = net_name.upper() in [n.upper() for n in power_nets]

        if is_power_net and len(endpoints) >= 3:
            # Try 1: Star topology on B.Cu
            star_route = self._route_star_topology(net_name, endpoints)
            if star_route.success:
                return star_route

            # Try 2: Daisy-chain on B.Cu (simpler, might work when star fails)
            daisy_route = self._route_daisy_chain_bcu(net_name, endpoints)
            if daisy_route.success:
                return daisy_route

        # For multi-point nets, use MST-style routing with retry logic
        # Start with first endpoint as "connected"
        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])
        failed_points = []  # Points that failed on first attempt
        max_retries = 3  # Retry failed points after more connections made

        for retry_round in range(max_retries + 1):
            # If this is a retry round, move failed points back to unconnected
            if retry_round > 0 and failed_points:
                unconnected = failed_points[:]
                failed_points = []

            while unconnected:
                # Find closest (unconnected -> connected) pair
                best_dist = float('inf')
                best_pair = None
                best_uc = None

                for uc in unconnected:
                    for c in connected:
                        dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])  # Manhattan distance
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (uc, c)
                            best_uc = uc

                if not best_pair:
                    break

                uc_point, c_point = best_pair

                # Try to route from unconnected point to any connected point
                segments, vias, success = self._route_two_points(
                    uc_point, c_point, net_name, 'F.Cu'
                )

                if success:
                    route.segments.extend(segments)
                    route.vias.extend(vias)
                    connected.add(uc_point)
                    unconnected.remove(uc_point)
                else:
                    # Try routing to other connected points
                    routed = False
                    for alt_c in sorted(connected, key=lambda c: abs(uc_point[0] - c[0]) + abs(uc_point[1] - c[1])):
                        if alt_c == c_point:
                            continue
                        segments, vias, success = self._route_two_points(
                            uc_point, alt_c, net_name, 'F.Cu'
                        )
                        if success:
                            route.segments.extend(segments)
                            route.vias.extend(vias)
                            connected.add(uc_point)
                            unconnected.remove(uc_point)
                            routed = True
                            break

                    if not routed:
                        # Don't fail immediately - defer this point to retry later
                        # More connections might open up new routing paths
                        unconnected.remove(uc_point)
                        failed_points.append(uc_point)

            # If no points failed this round, we're done
            if not failed_points:
                break

        # Final check - any points still failed?
        # Last-ditch effort: try EVERY possible connection combination
        if failed_points:
            for uc_point in failed_points[:]:  # Copy list to modify
                routed = False
                # Try connecting to ANY connected point using ANY method
                for c_point in sorted(connected, key=lambda c: abs(uc_point[0] - c[0]) + abs(uc_point[1] - c[1])):
                    # Try all routing methods more aggressively
                    for layer_to_try in ['F.Cu', 'B.Cu']:
                        # Direct Manhattan
                        segments, vias, success = self._route_two_points(
                            uc_point, c_point, net_name, layer_to_try
                        )
                        if success:
                            route.segments.extend(segments)
                            route.vias.extend(vias)
                            connected.add(uc_point)
                            failed_points.remove(uc_point)
                            routed = True
                            break
                    if routed:
                        break

        if failed_points:
            route.error = f"Failed to connect {failed_points[0]} to net {net_name}"
            return route

        route.success = True
        return route

    def _route_star_topology(self, net_name: str, endpoints: List[Tuple[float, float]]) -> Route:
        """
        Route a multi-pin net using STAR topology on B.Cu layer.

        Human approach for GND/power nets:
        1. Find the centroid of all endpoints
        2. Place a via at (or near) centroid
        3. Route each endpoint to the centroid on B.Cu
        4. This separates power routing from signal routing

        Benefits:
        - Power doesn't block signal routes on F.Cu
        - All connections fan out from one central point
        - Clean, professional-looking power distribution
        """
        route = Route(net=net_name)

        # Calculate centroid of all endpoints
        centroid_x = sum(p[0] for p in endpoints) / len(endpoints)
        centroid_y = sum(p[1] for p in endpoints) / len(endpoints)

        # Snap centroid to grid
        centroid = (
            round(centroid_x / self.grid_size) * self.grid_size,
            round(centroid_y / self.grid_size) * self.grid_size
        )

        # Check if centroid is free for this net
        col = int((centroid[0] - self.origin_x) / self.grid_size)
        row = int((centroid[1] - self.origin_y) / self.grid_size)

        # If centroid is blocked, try finding a nearby free location
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            route.error = "Centroid outside board"
            return route

        cell_value = self.bcu_grid[row][col]
        if cell_value is not None and cell_value != net_name:
            # Try nearby locations in a spiral pattern
            found_center = False
            for radius in range(1, 10):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue  # Only check perimeter
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                            if self.bcu_grid[r][c] is None or self.bcu_grid[r][c] == net_name:
                                centroid = (
                                    self.origin_x + c * self.grid_size,
                                    self.origin_y + r * self.grid_size
                                )
                                found_center = True
                                break
                    if found_center:
                        break
                if found_center:
                    break

            if not found_center:
                route.error = "No free location for star center"
                return route

        # DEFER via creation until routing succeeds to avoid blocking paths

        # Route each endpoint to the centroid on B.Cu
        all_segments = []
        all_success = True
        for endpoint in endpoints:
            # Route on B.Cu from endpoint to centroid
            segments, success = self._astar_route(endpoint, centroid, net_name, 'B.Cu')

            if not success:
                # Try Lee wavefront (more thorough)
                segments, success = self._lee_wavefront_route(endpoint, centroid, net_name, 'B.Cu')

            if not success:
                # Try diagonal A*
                segments, success = self._astar_route(endpoint, centroid, net_name, 'B.Cu', allow_diagonal=True)

            if success:
                all_segments.extend(segments)
            else:
                all_success = False
                break

        if all_success:
            # Mark segments as occupied (so next power net routes AROUND, not THROUGH)
            for seg in all_segments:
                self._mark_occupied([seg], 'B.Cu')
            route.segments = all_segments

            # NOTE: Don't create vias here - they mark grids which blocks other power nets
            # Vias will be created later by the caller after ALL power nets route
            route._pending_via_endpoints = endpoints
            route._pending_via_centroid = centroid  # For star topology
            route.success = True
        else:
            route.error = f"Star topology failed for {net_name}"

        return route

    def _route_daisy_chain_bcu(self, net_name: str, endpoints: List[Tuple[float, float]]) -> Route:
        """
        Route a multi-pin net using daisy-chain topology on B.Cu layer.

        Human approach: Connect pins in a chain (1->2->3->4) entirely on B.Cu.
        This is simpler than star and might succeed when star fails.

        Each pin gets a via, then B.Cu routes connect them sequentially.
        """
        route = Route(net=net_name)

        # Sort endpoints by position for a logical chain (left-to-right, then top-to-bottom)
        sorted_endpoints = sorted(endpoints, key=lambda p: (p[0], p[1]))

        # DEFER via creation until routing succeeds
        # Creating vias early can block routing paths due to clearance zones

        # Route chain on B.Cu: endpoint[0] -> endpoint[1] -> endpoint[2] -> ...
        all_segments = []
        for i in range(len(sorted_endpoints) - 1):
            start = sorted_endpoints[i]
            end = sorted_endpoints[i + 1]

            # Try A* on B.Cu
            segments, success = self._astar_route(start, end, net_name, 'B.Cu')

            if success:
                all_segments.extend(segments)
            else:
                # Try Lee wavefront
                segments, success = self._lee_wavefront_route(start, end, net_name, 'B.Cu')
                if success:
                    all_segments.extend(segments)
                else:
                    # Try simple Manhattan on B.Cu
                    segments, success = self._manhattan_route(start, end, net_name, 'B.Cu', 'h_first')
                    if not success:
                        segments, success = self._manhattan_route(start, end, net_name, 'B.Cu', 'v_first')

                    if success:
                        all_segments.extend(segments)
                    else:
                        # Try diagonal A*
                        segments, success = self._astar_route(start, end, net_name, 'B.Cu', allow_diagonal=True)
                        if success:
                            all_segments.extend(segments)
                        else:
                            route.error = f"Daisy chain failed at {start} -> {end}"
                            return route

        # All segments routed successfully
        # Mark segments as occupied (so next power net routes AROUND, not THROUGH)
        for seg in all_segments:
            self._mark_occupied([seg], 'B.Cu')
        route.segments = all_segments

        # NOTE: Don't create vias here - they mark grids which blocks other power nets
        # Vias will be created later by the caller after ALL power nets route
        route._pending_via_endpoints = sorted_endpoints  # Store for later via creation
        route.success = True
        return route

    def _route_net_with_layer_preference(self, net_name: str, pins: List[Tuple],
                                          escapes: Dict,
                                          preferred_layer: str = 'F.Cu') -> Route:
        """
        Route a net with a specific layer preference.

        This is used by the DRC feedback loop to force nets to specific layers
        when clearance violations occur.

        Args:
            net_name: Name of the net
            pins: List of (component, pin) tuples
            escapes: Escape routes
            preferred_layer: Layer to prefer ('F.Cu' or 'B.Cu')
        """
        route = Route(net=net_name)

        # Get escape endpoints
        endpoints = []
        for comp, pin in pins:
            if comp in escapes and pin in escapes[comp]:
                esc = escapes[comp][pin]
                end = esc.end if hasattr(esc, 'end') else getattr(esc, 'endpoint', None)
                if end:
                    endpoints.append(end)

        if len(endpoints) < 2:
            route.error = f"Not enough escape endpoints for {net_name}"
            return route

        # For 2-point nets
        if len(endpoints) == 2:
            segments, vias, success = self._route_two_points_with_preference(
                endpoints[0], endpoints[1], net_name, preferred_layer
            )
            if success:
                route.segments.extend(segments)
                route.vias.extend(vias)
                route.success = True
            else:
                route.error = f"Failed to route {net_name}"
            return route

        # For multi-point nets, use MST approach with layer preference
        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        while unconnected:
            best_dist = float('inf')
            best_pair = None

            for uc in unconnected:
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (uc, c)

            if not best_pair:
                break

            uc_point, c_point = best_pair
            segments, vias, success = self._route_two_points_with_preference(
                uc_point, c_point, net_name, preferred_layer
            )

            if success:
                route.segments.extend(segments)
                route.vias.extend(vias)
                connected.add(uc_point)
                unconnected.remove(uc_point)
            else:
                route.error = f"Failed to connect {uc_point}"
                return route

        route.success = True
        return route

    def _route_two_points_with_preference(self, start: Tuple[float, float],
                                           end: Tuple[float, float],
                                           net_name: str,
                                           preferred_layer: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Route between two points with a layer preference.

        If preferred_layer is B.Cu, we TRY B.Cu FIRST (with vias at endpoints).
        This helps separate conflicting nets onto different layers.
        """
        other_layer = 'B.Cu' if preferred_layer == 'F.Cu' else 'F.Cu'

        # If preferred layer is B.Cu, try it FIRST (not as fallback)
        if preferred_layer == 'B.Cu':
            # Try B.Cu with vias at both endpoints first
            segments, success = self._astar_route(start, end, net_name, 'B.Cu')
            if success:
                self._mark_occupied(segments, 'B.Cu')
                vias = []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True

            # Try Lee wavefront on B.Cu
            segments, success = self._lee_wavefront_route(start, end, net_name, 'B.Cu')
            if success:
                self._mark_occupied(segments, 'B.Cu')
                vias = []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True

        # Fall back to normal routing (tries F.Cu first, then B.Cu)
        return self._route_two_points(start, end, net_name, preferred_layer)

    def _route_two_points(self, start: Tuple[float, float],
                          end: Tuple[float, float],
                          net_name: str,
                          layer: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Route between two points using Manhattan routing.

        Human approach:
        1. Try horizontal-first (go right/left, then up/down)
        2. If blocked, try vertical-first
        3. If blocked, try 3-segment L-routes around obstacles
        4. If still blocked, use via and try other layer

        IMPORTANT: Escapes are ALWAYS on F.Cu. If routing on B.Cu, we MUST add
        vias at endpoints to connect F.Cu escapes to B.Cu routes.
        """
        # CRITICAL FIX: If layer is B.Cu, we need vias at both endpoints
        # because escapes are on F.Cu. Route on B.Cu with vias.
        need_vias_at_endpoints = (layer == 'B.Cu')

        # Try 1: Horizontal first, then vertical
        segments, success = self._manhattan_route(start, end, net_name, layer, 'h_first')
        if success:
            self._mark_occupied(segments, layer)
            if need_vias_at_endpoints:
                vias = []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True
            return segments, [], True

        # Try 2: Vertical first, then horizontal
        segments, success = self._manhattan_route(start, end, net_name, layer, 'v_first')
        if success:
            self._mark_occupied(segments, layer)
            if need_vias_at_endpoints:
                vias = []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True
            return segments, [], True

        # Try 3: 3-segment L-routes around obstacles
        # Try going wide around on each side
        # ROOT CAUSE FIX: Use MUCH larger offsets to go around big components like ESP32
        # ESP32-C3-MINI is 13x16mm, so we need offsets up to 20mm to go around
        for offset in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]:  # Try larger offsets
            # Try going left then up/down
            segments, success = self._three_segment_route(start, end, net_name, layer, 'left', offset)
            if success:
                self._mark_occupied(segments, layer)
                if need_vias_at_endpoints:
                    vias = []
                    via_start = self._create_via(start, net_name)
                    if via_start:
                        vias.append(via_start)
                    via_end = self._create_via(end, net_name)
                    if via_end:
                        vias.append(via_end)
                    return segments, vias, True
                return segments, [], True

            # Try going right then up/down
            segments, success = self._three_segment_route(start, end, net_name, layer, 'right', offset)
            if success:
                self._mark_occupied(segments, layer)
                if need_vias_at_endpoints:
                    vias = []
                    via_start = self._create_via(start, net_name)
                    if via_start:
                        vias.append(via_start)
                    via_end = self._create_via(end, net_name)
                    if via_end:
                        vias.append(via_end)
                    return segments, vias, True
                return segments, [], True

            # Try going up then left/right
            segments, success = self._three_segment_route(start, end, net_name, layer, 'up', offset)
            if success:
                self._mark_occupied(segments, layer)
                if need_vias_at_endpoints:
                    vias = []
                    via_start = self._create_via(start, net_name)
                    if via_start:
                        vias.append(via_start)
                    via_end = self._create_via(end, net_name)
                    if via_end:
                        vias.append(via_end)
                    return segments, vias, True
                return segments, [], True

            # Try going down then left/right
            segments, success = self._three_segment_route(start, end, net_name, layer, 'down', offset)
            if success:
                self._mark_occupied(segments, layer)
                if need_vias_at_endpoints:
                    vias = []
                    via_start = self._create_via(start, net_name)
                    if via_start:
                        vias.append(via_start)
                    via_end = self._create_via(end, net_name)
                    if via_end:
                        vias.append(via_end)
                    return segments, vias, True
                return segments, [], True

        # Try 4: A* pathfinding on same layer (can navigate complex obstacles)
        segments, success = self._astar_route(start, end, net_name, layer)
        if success:
            self._mark_occupied(segments, layer)
            if need_vias_at_endpoints:
                vias = []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True
            return segments, [], True

        # =====================================================================
        # REORDERED: Try ALL F.Cu algorithms BEFORE using vias/B.Cu
        # User requirement: Exhaust all single-layer options first
        # =====================================================================

        # Try 5: Diagonal A* on same layer (allows 45-degree turns)
        # Human approach: "twist" the route with angular turns to find a path
        segments, success = self._astar_route(start, end, net_name, layer, allow_diagonal=True)
        if success:
            self._mark_occupied(segments, layer)
            if need_vias_at_endpoints:
                vias = []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True
            return segments, [], True

        # Try 6: Bidirectional A* (searches from both ends - faster for long routes)
        segments, vias_result, success = self._bidirectional_astar_route(start, end, net_name, layer)
        if success:
            if need_vias_at_endpoints:
                vias = vias_result if vias_result else []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True
            return segments, vias_result or [], True

        # Try 7: Lee wavefront algorithm (guaranteed to find path if one exists)
        segments, success = self._lee_wavefront_route(start, end, net_name, layer)
        if success:
            self._mark_occupied(segments, layer)
            if need_vias_at_endpoints:
                vias = []
                via_start = self._create_via(start, net_name)
                if via_start:
                    vias.append(via_start)
                via_end = self._create_via(end, net_name)
                if via_end:
                    vias.append(via_end)
                return segments, vias, True
            return segments, [], True

        # =====================================================================
        # LAST RESORT: Use vias and B.Cu layer
        # Only reached if ALL F.Cu algorithms failed
        # =====================================================================
        other_layer = 'B.Cu' if layer == 'F.Cu' else 'F.Cu'

        # Try 8: Via to other layer with simple Manhattan
        # Try via at BOTH ends (since escapes are on F.Cu, we need vias to connect to B.Cu route)
        segments, vias, success = self._route_with_via(
            start, end, net_name, layer, other_layer, 'via_at_both'
        )
        if success:
            return segments, vias, True

        # Try via at start only
        segments, vias, success = self._route_with_via(
            start, end, net_name, layer, other_layer, 'via_at_start'
        )
        if success:
            return segments, vias, True

        # Try via at midpoint
        segments, vias, success = self._route_with_via(
            start, end, net_name, layer, other_layer, 'via_at_mid'
        )
        if success:
            return segments, vias, True

        # Try 9: A* pathfinding on other layer with vias at both ends
        segments, success = self._astar_route(start, end, net_name, other_layer)
        if success:
            self._mark_occupied(segments, other_layer)
            vias = []
            via_start = self._create_via(start, net_name)
            if via_start:
                vias.append(via_start)
            via_end = self._create_via(end, net_name)
            if via_end:
                vias.append(via_end)
            return segments, vias, True

        # Try 10: Diagonal A* on other layer with vias
        segments, success = self._astar_route(start, end, net_name, other_layer, allow_diagonal=True)
        if success:
            self._mark_occupied(segments, other_layer)
            vias = []
            via_start = self._create_via(start, net_name)
            if via_start:
                vias.append(via_start)
            via_end = self._create_via(end, net_name)
            if via_end:
                vias.append(via_end)
            return segments, vias, True

        # Try 11: Bidirectional A* on other layer with vias
        segments, vias_result, success = self._bidirectional_astar_route(start, end, net_name, other_layer)
        if success:
            # Add vias at endpoints to connect F.Cu escapes to B.Cu route
            vias = vias_result if vias_result else []
            via_start = self._create_via(start, net_name)
            if via_start:
                vias.append(via_start)
            via_end = self._create_via(end, net_name)
            if via_end:
                vias.append(via_end)
            return segments, vias, True

        # Try 12: Lee wavefront on other layer with vias
        segments, success = self._lee_wavefront_route(start, end, net_name, other_layer)
        if success:
            self._mark_occupied(segments, other_layer)
            vias = []
            via_start = self._create_via(start, net_name)
            if via_start:
                vias.append(via_start)
            via_end = self._create_via(end, net_name)
            if via_end:
                vias.append(via_end)
            return segments, vias, True

        # Failed - all strategies exhausted
        return [], [], False

    def _three_segment_route(self, start: Tuple[float, float],
                              end: Tuple[float, float],
                              net_name: str, layer: str,
                              direction: str, offset: float) -> Tuple[List[TrackSegment], bool]:
        """
        Route with 3 segments to go around obstacles.

        Human approach: If direct path blocked, go AROUND the obstacle.
        """
        sx, sy = start
        ex, ey = end

        # Snap offset to grid
        offset = round(offset / self.grid_size) * self.grid_size

        if direction == 'left':
            # Go left, then vertical, then right to destination
            mid_x = min(sx, ex) - offset
            mid1 = (mid_x, sy)
            mid2 = (mid_x, ey)
        elif direction == 'right':
            # Go right, then vertical, then left to destination
            mid_x = max(sx, ex) + offset
            mid1 = (mid_x, sy)
            mid2 = (mid_x, ey)
        elif direction == 'up':
            # Go up, then horizontal, then down to destination
            mid_y = min(sy, ey) - offset
            mid1 = (sx, mid_y)
            mid2 = (ex, mid_y)
        elif direction == 'down':
            # Go down, then horizontal, then up to destination
            mid_y = max(sy, ey) + offset
            mid1 = (sx, mid_y)
            mid2 = (ex, mid_y)
        else:
            return [], False

        segments = []

        # Segment 1: start to mid1
        if start != mid1:
            if not self._is_path_clear(start, mid1, layer, net_name):
                return [], False
            segments.append(TrackSegment(
                start=start, end=mid1, layer=layer,
                width=self.trace_width, net=net_name
            ))

        # Segment 2: mid1 to mid2
        if mid1 != mid2:
            if not self._is_path_clear(mid1, mid2, layer, net_name):
                return [], False
            segments.append(TrackSegment(
                start=mid1, end=mid2, layer=layer,
                width=self.trace_width, net=net_name
            ))

        # Segment 3: mid2 to end
        if mid2 != end:
            if not self._is_path_clear(mid2, end, layer, net_name):
                return [], False
            segments.append(TrackSegment(
                start=mid2, end=end, layer=layer,
                width=self.trace_width, net=net_name
            ))

        return segments, True

    def _manhattan_route(self, start: Tuple[float, float],
                         end: Tuple[float, float],
                         net_name: str, layer: str,
                         strategy: str) -> Tuple[List[TrackSegment], bool]:
        """
        Simple Manhattan routing: horizontal + vertical or vice versa.
        """
        sx, sy = start
        ex, ey = end

        if strategy == 'h_first':
            # Go horizontal first, then vertical
            mid = (ex, sy)
        else:
            # Go vertical first, then horizontal
            mid = (sx, ey)

        segments = []

        # First segment
        if start != mid:
            seg1 = TrackSegment(
                start=start, end=mid, layer=layer,
                width=self.trace_width, net=net_name
            )
            # Pass net_name for net-aware collision detection
            if not self._is_path_clear(start, mid, layer, net_name):
                return [], False
            segments.append(seg1)

        # Second segment
        if mid != end:
            seg2 = TrackSegment(
                start=mid, end=end, layer=layer,
                width=self.trace_width, net=net_name
            )
            # Pass net_name for net-aware collision detection
            if not self._is_path_clear(mid, end, layer, net_name):
                return [], False
            segments.append(seg2)

        return segments, True

    def _create_via(self, position: Tuple[float, float], net_name: str,
                    diameter: float = 0.8, drill: float = 0.4,
                    min_via_spacing: float = 0.5) -> Optional[Via]:
        """
        Create a via at position, avoiding duplicates AND enforcing spacing.

        DRC FIX:
        1. Tracks duplicate vias to prevent 'holes_co_located' errors
        2. Checks via-to-via spacing to prevent 'drilled hole too close' errors
        3. Checks via-to-pad clearance to prevent 'clearance violation' errors
        4. Marks via clearance zone in BOTH layer grids (via connects both layers)
        5. Via clearance is larger than trace clearance (via diameter > trace width)

        Args:
            position: Via position
            net_name: Net for the via
            diameter: Via pad diameter (default 0.8mm)
            drill: Via drill diameter (default 0.4mm)
            min_via_spacing: Minimum edge-to-edge spacing between vias (default 0.5mm)
        """
        # Round position to grid for consistent comparison
        pos = (round(position[0] / self.grid_size) * self.grid_size,
               round(position[1] / self.grid_size) * self.grid_size)

        # Check if via already exists at this position
        if pos in self.placed_vias:
            return None  # Skip duplicate

        # Calculate via grid position
        col = int((pos[0] - self.origin_x) / self.grid_size)
        row = int((pos[1] - self.origin_y) / self.grid_size)

        # DRC FIX: Check via-to-pad clearance
        # Via clearance = diameter/2 + clearance
        via_clearance_radius = diameter / 2 + self.clearance
        via_clearance_cells = max(2, int(math.ceil(via_clearance_radius / self.grid_size)))

        # Markers that block via placement
        blocked_markers = {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__'}

        # Check if via position is clear in both grids
        for dr in range(-via_clearance_cells, via_clearance_cells + 1):
            for dc in range(-via_clearance_cells, via_clearance_cells + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    fcu_cell = self.fcu_grid[r][c]
                    bcu_cell = self.bcu_grid[r][c]
                    # Block if cell has a blocking marker
                    if fcu_cell in blocked_markers or bcu_cell in blocked_markers:
                        # Try to find alternate position
                        alternate_pos = self._find_alternate_via_position_for_pad(
                            pos, net_name, diameter, via_clearance_cells
                        )
                        if alternate_pos:
                            pos = alternate_pos
                            col = int((pos[0] - self.origin_x) / self.grid_size)
                            row = int((pos[1] - self.origin_y) / self.grid_size)
                            break
                        else:
                            return None  # Cannot place via
                    # Block if occupied by a DIFFERENT net
                    if fcu_cell is not None and fcu_cell != net_name and fcu_cell not in blocked_markers:
                        alternate_pos = self._find_alternate_via_position_for_pad(
                            pos, net_name, diameter, via_clearance_cells
                        )
                        if alternate_pos:
                            pos = alternate_pos
                            col = int((pos[0] - self.origin_x) / self.grid_size)
                            row = int((pos[1] - self.origin_y) / self.grid_size)
                            break
                        else:
                            return None
            else:
                continue
            break

        # DRC FIX: Check via-to-via spacing
        # Minimum distance = diameter + min_via_spacing (edge-to-edge)
        min_center_distance = diameter + min_via_spacing

        for existing_via_pos in self.placed_vias:
            dist = math.sqrt((pos[0] - existing_via_pos[0])**2 +
                           (pos[1] - existing_via_pos[1])**2)
            if dist < min_center_distance:
                # Via too close to existing via - try to find alternate position
                alternate_pos = self._find_alternate_via_position(
                    pos, existing_via_pos, min_center_distance, net_name
                )
                if alternate_pos:
                    pos = alternate_pos
                else:
                    # Cannot place via with proper spacing - skip
                    return None

        # Mark as placed
        self.placed_vias.add(pos)

        # Mark via location AND clearance in BOTH layer grids
        # Via clearance = diameter/2 + clearance (larger than trace)
        via_clearance_radius = diameter / 2 + self.clearance
        via_clearance_cells = max(1, int(math.ceil(via_clearance_radius / self.grid_size)))

        col = int((pos[0] - self.origin_x) / self.grid_size)
        row = int((pos[1] - self.origin_y) / self.grid_size)

        # Mark clearance zone in both layers
        for dr in range(-via_clearance_cells, via_clearance_cells + 1):
            for dc in range(-via_clearance_cells, via_clearance_cells + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    # Mark in both layers (via connects both)
                    if self.fcu_grid[r][c] is None or self.fcu_grid[r][c] == net_name:
                        self.fcu_grid[r][c] = net_name
                    if self.bcu_grid[r][c] is None or self.bcu_grid[r][c] == net_name:
                        self.bcu_grid[r][c] = net_name

        return Via(position=pos, net=net_name, diameter=diameter, drill=drill)

    def _find_alternate_via_position(self, pos: Tuple[float, float],
                                      conflict_pos: Tuple[float, float],
                                      min_distance: float,
                                      net_name: str) -> Optional[Tuple[float, float]]:
        """
        Find an alternate via position that maintains minimum spacing.

        Tries positions in a spiral pattern around the original position
        until finding one that is far enough from all existing vias.
        """
        # Calculate direction away from conflict
        dx = pos[0] - conflict_pos[0]
        dy = pos[1] - conflict_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.001:
            # Same position - pick arbitrary direction
            dx, dy = 1, 0
        else:
            # Normalize
            dx, dy = dx / dist, dy / dist

        # Try offsets in the direction away from conflict
        for offset_mult in [1, 2, 3, 4]:
            offset = min_distance * offset_mult / 2

            # Try perpendicular directions too
            for angle_offset in [0, 90, -90, 45, -45, 135, -135, 180]:
                rad = math.radians(angle_offset)
                # Rotate direction vector
                new_dx = dx * math.cos(rad) - dy * math.sin(rad)
                new_dy = dx * math.sin(rad) + dy * math.cos(rad)

                new_x = pos[0] + new_dx * offset
                new_y = pos[1] + new_dy * offset

                # Snap to grid
                new_pos = (
                    round(new_x / self.grid_size) * self.grid_size,
                    round(new_y / self.grid_size) * self.grid_size
                )

                # Check if this position is valid
                if new_pos in self.placed_vias:
                    continue

                # Check distance to ALL existing vias
                valid = True
                for existing in self.placed_vias:
                    d = math.sqrt((new_pos[0] - existing[0])**2 +
                                 (new_pos[1] - existing[1])**2)
                    if d < min_distance:
                        valid = False
                        break

                if valid:
                    # Check grid bounds
                    col = int((new_pos[0] - self.origin_x) / self.grid_size)
                    row = int((new_pos[1] - self.origin_y) / self.grid_size)
                    if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                        return new_pos

        return None

    def _find_alternate_via_position_for_pad(self, pos: Tuple[float, float],
                                              net_name: str, diameter: float,
                                              clearance_cells: int) -> Optional[Tuple[float, float]]:
        """
        Find an alternate via position that avoids pads and blocked areas.

        Tries positions in a spiral pattern around the original position
        until finding one that is clear of pads and other blocked markers.
        """
        blocked_markers = {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__'}

        # Try offsets in a spiral pattern
        for radius_mult in range(1, 8):
            radius = self.grid_size * radius_mult * 2

            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                new_x = pos[0] + radius * math.cos(rad)
                new_y = pos[1] + radius * math.sin(rad)

                # Snap to grid
                new_pos = (
                    round(new_x / self.grid_size) * self.grid_size,
                    round(new_y / self.grid_size) * self.grid_size
                )

                # Check if already placed
                if new_pos in self.placed_vias:
                    continue

                # Check grid bounds
                col = int((new_pos[0] - self.origin_x) / self.grid_size)
                row = int((new_pos[1] - self.origin_y) / self.grid_size)

                if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
                    continue

                # Check if position is clear in both grids
                is_clear = True
                for dr in range(-clearance_cells, clearance_cells + 1):
                    for dc in range(-clearance_cells, clearance_cells + 1):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                            fcu_cell = self.fcu_grid[r][c]
                            bcu_cell = self.bcu_grid[r][c]
                            if fcu_cell in blocked_markers or bcu_cell in blocked_markers:
                                is_clear = False
                                break
                            if fcu_cell is not None and fcu_cell != net_name and fcu_cell not in blocked_markers:
                                is_clear = False
                                break
                    if not is_clear:
                        break

                if is_clear:
                    return new_pos

        return None

    def _route_with_via(self, start: Tuple[float, float],
                        end: Tuple[float, float],
                        net_name: str, start_layer: str,
                        end_layer: str,
                        strategy: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Route using a via to switch layers.

        Important: Since escapes are on F.Cu, when routing on B.Cu we need
        vias at BOTH escape endpoints to properly connect F.Cu escapes to B.Cu route.

        DRC FIX: Uses _create_via to prevent duplicate vias at same location.
        """
        sx, sy = start
        ex, ey = end

        if strategy == 'via_at_both':
            # Via at BOTH start and end - needed when escapes are on F.Cu but route is on B.Cu
            vias = []
            via_start = self._create_via(start, net_name)
            if via_start:
                vias.append(via_start)
            via_end = self._create_via(end, net_name)
            if via_end:
                vias.append(via_end)

            # Route entirely on end_layer (B.Cu), with vias at both endpoints
            segments, success = self._manhattan_route(start, end, net_name, end_layer, 'h_first')
            if not success:
                segments, success = self._manhattan_route(start, end, net_name, end_layer, 'v_first')

            if success:
                self._mark_occupied(segments, end_layer)
                return segments, vias, True

        elif strategy == 'via_at_start':
            via_pos = start
            # Via at start, route on end_layer
            via = self._create_via(via_pos, net_name)
            vias = [via] if via else []
            segments, success = self._manhattan_route(start, end, net_name, end_layer, 'h_first')
            if success:
                self._mark_occupied(segments, end_layer)
                return segments, vias, True

        elif strategy == 'via_at_mid':
            # Via at midpoint
            mid_x = (sx + ex) / 2
            mid_y = (sy + ey) / 2
            # Snap to grid
            mid_x = round(mid_x / self.grid_size) * self.grid_size
            mid_y = round(mid_y / self.grid_size) * self.grid_size
            via_pos = (mid_x, mid_y)

            via = self._create_via(via_pos, net_name)
            vias = [via] if via else []

            # Route start to via on start_layer
            seg1_list, success1 = self._manhattan_route(start, via_pos, net_name, start_layer, 'h_first')
            if not success1:
                return [], [], False

            # Route via to end on end_layer
            seg2_list, success2 = self._manhattan_route(via_pos, end, net_name, end_layer, 'h_first')
            if not success2:
                return [], [], False

            self._mark_occupied(seg1_list, start_layer)
            self._mark_occupied(seg2_list, end_layer)
            return seg1_list + seg2_list, vias, True

        return [], [], False

    def _is_cell_clear(self, grid: List[List], row: int, col: int, net_name: str) -> bool:
        """
        Check if a cell AND its clearance zone are clear for this net.

        DRC FIX: Must check clearance_cells around each path cell to ensure
        the new track won't violate clearance with existing tracks.

        BLOCKED markers (block ALL nets):
        - '__PAD_NC__': No-Connect pad - tracks must NEVER pass through
        - '__COMPONENT__': Component body
        - '__EDGE__': Board edge margin
        - '__PAD_CONFLICT__': Multiple nets claim this pad location

        PERFORMANCE: Check center cell first (most likely to fail), then clearance zone.
        """
        # Markers that block ALL nets (even same net for NC pads)
        blocked_markers = {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__'}

        # Check center cell first (fastest common case)
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            center = grid[row][col]
            # Block if it's a special marker that blocks all nets
            if center in blocked_markers:
                return False
            # Block if occupied by a DIFFERENT net
            if center is not None and center != net_name:
                return False

        # Check clearance zone
        cc = self.clearance_cells
        for dr in range(-cc, cc + 1):
            for dc in range(-cc, cc + 1):
                if dr == 0 and dc == 0:
                    continue  # Already checked center
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                    occupant = grid[r][c]
                    # Block if it's a special marker
                    if occupant in blocked_markers:
                        return False
                    # Blocked if occupied by a DIFFERENT net
                    if occupant is not None and occupant != net_name:
                        return False
        return True

    def _is_path_clear(self, start: Tuple[float, float],
                       end: Tuple[float, float], layer: str,
                       net_name: str = None) -> bool:
        """
        Check if a path is clear of obstacles WITH CLEARANCE.

        Net-aware: Allows routing through cells occupied by the SAME net
        (since same-net connections are allowed to touch).

        DRC FIX: Checks clearance_cells around each path cell to prevent
        routing too close to existing tracks (clearance violations).
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        # Walk along the path checking each grid cell
        sx, sy = start
        ex, ey = end

        # Convert to grid coordinates
        start_col = int((sx - self.origin_x) / self.grid_size)
        start_row = int((sy - self.origin_y) / self.grid_size)
        end_col = int((ex - self.origin_x) / self.grid_size)
        end_row = int((ey - self.origin_y) / self.grid_size)

        # Check all cells along path (with clearance)
        if start_row == end_row:
            # Horizontal path
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                if 0 <= start_row < self.grid_rows and 0 <= col < self.grid_cols:
                    if not self._is_cell_clear(grid, start_row, col, net_name):
                        return False
        elif start_col == end_col:
            # Vertical path
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                if 0 <= row < self.grid_rows and 0 <= start_col < self.grid_cols:
                    if not self._is_cell_clear(grid, row, start_col, net_name):
                        return False
        else:
            # Diagonal path - check cells along line
            dx = end_col - start_col
            dy = end_row - start_row
            steps = max(abs(dx), abs(dy))
            if steps == 0:
                return True

            for i in range(steps + 1):
                t = i / steps
                col = int(start_col + t * dx)
                row = int(start_row + t * dy)
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    if not self._is_cell_clear(grid, row, col, net_name):
                        return False

        return True

    def _mark_occupied(self, segments: List[TrackSegment], layer: str):
        """Mark grid cells as occupied by segments (stores net name for net-aware routing)"""
        for seg in segments:
            self._mark_line_occupied(
                seg.start[0], seg.start[1],
                seg.end[0], seg.end[1],
                seg.net, layer
            )

    def _astar_route(self, start: Tuple[float, float], end: Tuple[float, float],
                     net_name: str, layer: str,
                     allow_diagonal: bool = False) -> Tuple[List[TrackSegment], bool]:
        """
        A* pathfinding for complex routes around obstacles.

        Human approach: When simple paths fail, find ANY valid path.
        A* guarantees shortest path if one exists.

        Args:
            allow_diagonal: If True, allows 45-degree diagonal moves.
                           This helps when Manhattan routing is blocked.
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        # Convert to grid coordinates
        start_col = int((start[0] - self.origin_x) / self.grid_size)
        start_row = int((start[1] - self.origin_y) / self.grid_size)
        end_col = int((end[0] - self.origin_x) / self.grid_size)
        end_row = int((end[1] - self.origin_y) / self.grid_size)

        # Check bounds
        if not (0 <= start_row < self.grid_rows and 0 <= start_col < self.grid_cols):
            return [], False
        if not (0 <= end_row < self.grid_rows and 0 <= end_col < self.grid_cols):
            return [], False

        # PERFORMANCE: Limit iterations to prevent infinite loops
        max_iterations = self.grid_rows * self.grid_cols * 2  # 2x grid size should be enough

        # A* algorithm
        # State: (row, col)
        # Priority: f = g + h where g = cost so far, h = heuristic

        import math
        if allow_diagonal:
            # Octile distance for 8-directional
            def heuristic(r, c):
                dr = abs(r - end_row)
                dc = abs(c - end_col)
                return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
        else:
            # Manhattan distance for 4-directional
            def heuristic(r, c):
                return abs(r - end_row) + abs(c - end_col)

        # Priority queue: (f_score, g_score, row, col, path)
        # Using g_score as tiebreaker for consistent ordering
        start_state = (start_row, start_col)
        end_state = (end_row, end_col)

        open_set = [(heuristic(start_row, start_col), 0, start_row, start_col)]
        came_from = {}  # {(row,col): (prev_row, prev_col)}
        g_score = {start_state: 0}

        # Movement directions
        if allow_diagonal:
            # 8-directional: orthogonal + 45-degree diagonals
            # Diagonals allow routes to "cut corners" when needed
            import math
            directions = [
                (0, 1, 1.0),     # right
                (0, -1, 1.0),    # left
                (1, 0, 1.0),     # down
                (-1, 0, 1.0),    # up
                (1, 1, math.sqrt(2)),    # down-right (45°)
                (1, -1, math.sqrt(2)),   # down-left
                (-1, 1, math.sqrt(2)),   # up-right
                (-1, -1, math.sqrt(2)),  # up-left
            ]
        else:
            # 4-directional (Manhattan routing)
            directions = [
                (0, 1, 1.0),     # right
                (0, -1, 1.0),    # left
                (1, 0, 1.0),     # down
                (-1, 0, 1.0),    # up
            ]

        iterations = 0
        while open_set:
            iterations += 1
            if iterations > max_iterations:
                return [], False  # Timeout - no path found in reasonable time

            f, g, row, col = heapq.heappop(open_set)
            current = (row, col)

            if current == end_state:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                # Convert grid path to segments
                return self._path_to_segments(path, net_name, layer, allow_diagonal)

            # Skip if we've found a better path
            if g > g_score.get(current, float('inf')):
                continue

            for direction in directions:
                dr, dc, cost = direction
                nr, nc = row + dr, col + dc
                neighbor = (nr, nc)

                # Check bounds
                if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                    continue

                # DRC FIX: Check if passable WITH CLEARANCE
                # Must check clearance_cells around the cell to avoid clearance violations
                if not self._is_cell_clear(grid, nr, nc, net_name):
                    continue

                # For diagonals, also check the adjacent orthogonal cells
                # to avoid cutting through corners
                if allow_diagonal and dr != 0 and dc != 0:
                    if not self._is_cell_clear(grid, row + dr, col, net_name):
                        continue
                    if not self._is_cell_clear(grid, row, col + dc, net_name):
                        continue

                # Cost: step cost + turn penalty
                # Prefer straight paths by penalizing turns
                turn_penalty = 0
                if current in came_from:
                    prev = came_from[current]
                    prev_dir = (row - prev[0], col - prev[1])
                    curr_dir = (dr, dc)
                    if prev_dir != curr_dir:
                        turn_penalty = 0.1

                tentative_g = g + cost + turn_penalty

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        # No path found
        return [], False

    # =========================================================================
    # ADVANCED ALGORITHMS (Bidirectional A*, Lee Wavefront)
    # These provide ADDITIONAL routing strategies alongside the originals
    # =========================================================================

    def _bidirectional_astar_route(self, start: Tuple[float, float], end: Tuple[float, float],
                                    net_name: str, layer: str,
                                    via_cost: float = 5.0) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Bidirectional A* search - searches from BOTH ends simultaneously.

        This is typically 2x faster than unidirectional A* because the
        search space is roughly halved (two smaller search circles vs one big circle).

        Also supports layer switching with via penalty.
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid
        other_layer = 'B.Cu' if layer == 'F.Cu' else 'F.Cu'
        other_grid = self.bcu_grid if layer == 'F.Cu' else self.fcu_grid

        # Convert to grid coordinates
        start_col = int((start[0] - self.origin_x) / self.grid_size)
        start_row = int((start[1] - self.origin_y) / self.grid_size)
        end_col = int((end[0] - self.origin_x) / self.grid_size)
        end_row = int((end[1] - self.origin_y) / self.grid_size)

        # Check bounds
        if not (0 <= start_row < self.grid_rows and 0 <= start_col < self.grid_cols):
            return [], [], False
        if not (0 <= end_row < self.grid_rows and 0 <= end_col < self.grid_cols):
            return [], [], False

        # Check if start/end are accessible
        if not self._is_cell_clear(grid, start_row, start_col, net_name):
            return [], [], False
        if not self._is_cell_clear(grid, end_row, end_col, net_name):
            return [], [], False

        def heuristic(r1, c1, r2, c2):
            return abs(r1 - r2) + abs(c1 - c2)

        # State: (row, col, layer_index) where layer_index 0=preferred, 1=other
        # Forward search from start
        forward_open = [(heuristic(start_row, start_col, end_row, end_col),
                         0, start_row, start_col, 0)]  # layer_index 0 = preferred
        forward_came_from = {}
        forward_g = {(start_row, start_col, 0): 0}
        forward_closed = set()

        # Backward search from end
        backward_open = [(heuristic(end_row, end_col, start_row, start_col),
                          0, end_row, end_col, 0)]
        backward_came_from = {}
        backward_g = {(end_row, end_col, 0): 0}
        backward_closed = set()

        # Best meeting point found
        best_cost = float('inf')
        meeting_point = None

        # Directions: (dr, dc, cost)
        directions = [
            (0, 1, 1.0),   # right
            (0, -1, 1.0),  # left
            (1, 0, 1.0),   # down
            (-1, 0, 1.0),  # up
        ]

        max_iterations = self.grid_rows * self.grid_cols * 4
        iterations = 0

        def get_grid_for_layer(layer_idx):
            return grid if layer_idx == 0 else other_grid

        def get_layer_name(layer_idx):
            return layer if layer_idx == 0 else other_layer

        while (forward_open or backward_open) and iterations < max_iterations:
            iterations += 1

            # Expand forward search
            if forward_open:
                _, g, row, col, layer_idx = heapq.heappop(forward_open)
                state = (row, col, layer_idx)

                if state in forward_closed:
                    continue
                forward_closed.add(state)

                # Check if we've met the backward search
                if state in backward_closed:
                    total_cost = forward_g[state] + backward_g.get(state, float('inf'))
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_point = state

                # Expand neighbors on same layer
                current_grid = get_grid_for_layer(layer_idx)
                for dr, dc, cost in directions:
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                        continue
                    if not self._is_cell_clear(current_grid, nr, nc, net_name):
                        continue

                    new_state = (nr, nc, layer_idx)
                    new_g = g + cost

                    if new_state not in forward_g or new_g < forward_g[new_state]:
                        forward_g[new_state] = new_g
                        forward_came_from[new_state] = state
                        f = new_g + heuristic(nr, nc, end_row, end_col)
                        heapq.heappush(forward_open, (f, new_g, nr, nc, layer_idx))

                # Try layer switch (via) - switch to the other layer
                other_layer_idx = 1 - layer_idx
                other_layer_grid = get_grid_for_layer(other_layer_idx)
                if self._is_cell_clear(other_layer_grid, row, col, net_name):
                    new_state = (row, col, other_layer_idx)
                    new_g = g + via_cost

                    if new_state not in forward_g or new_g < forward_g[new_state]:
                        forward_g[new_state] = new_g
                        forward_came_from[new_state] = state
                        f = new_g + heuristic(row, col, end_row, end_col)
                        heapq.heappush(forward_open, (f, new_g, row, col, other_layer_idx))

            # Expand backward search (similar logic)
            if backward_open:
                _, g, row, col, layer_idx = heapq.heappop(backward_open)
                state = (row, col, layer_idx)

                if state in backward_closed:
                    continue
                backward_closed.add(state)

                # Check if we've met the forward search
                if state in forward_closed:
                    total_cost = forward_g.get(state, float('inf')) + backward_g[state]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_point = state

                # Expand neighbors
                current_grid = get_grid_for_layer(layer_idx)
                for dr, dc, cost in directions:
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                        continue
                    if not self._is_cell_clear(current_grid, nr, nc, net_name):
                        continue

                    new_state = (nr, nc, layer_idx)
                    new_g = g + cost

                    if new_state not in backward_g or new_g < backward_g[new_state]:
                        backward_g[new_state] = new_g
                        backward_came_from[new_state] = state
                        f = new_g + heuristic(nr, nc, start_row, start_col)
                        heapq.heappush(backward_open, (f, new_g, nr, nc, layer_idx))

                # Try layer switch
                other_layer_idx = 1 - layer_idx
                other_layer_grid = get_grid_for_layer(other_layer_idx)
                if self._is_cell_clear(other_layer_grid, row, col, net_name):
                    new_state = (row, col, other_layer_idx)
                    new_g = g + via_cost

                    if new_state not in backward_g or new_g < backward_g[new_state]:
                        backward_g[new_state] = new_g
                        backward_came_from[new_state] = state
                        f = new_g + heuristic(row, col, start_row, start_col)
                        heapq.heappush(backward_open, (f, new_g, row, col, other_layer_idx))

            # Early termination check
            if meeting_point and best_cost < float('inf'):
                min_forward = forward_open[0][0] if forward_open else float('inf')
                min_backward = backward_open[0][0] if backward_open else float('inf')
                if min_forward + min_backward >= best_cost:
                    break

        if meeting_point is None:
            return [], [], False

        # Reconstruct path with layer info
        # Forward part (start to meeting)
        current = meeting_point
        forward_path = []
        while current in forward_came_from:
            forward_path.append(current)
            current = forward_came_from[current]
        forward_path.append((start_row, start_col, 0))
        forward_path.reverse()

        # Backward part (meeting to end)
        current = meeting_point
        backward_path = []
        while current in backward_came_from:
            prev = backward_came_from[current]
            backward_path.append(prev)
            current = prev

        full_path = forward_path + backward_path

        # Convert to segments and vias
        return self._path_with_layers_to_segments(full_path, net_name, layer, other_layer)

    def _path_with_layers_to_segments(self, path: List[Tuple[int, int, int]], net_name: str,
                                       layer: str, other_layer: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Convert a path with layer info to track segments and vias.

        path: List of (row, col, layer_index) where layer_index 0 = layer, 1 = other_layer
        """
        if len(path) < 2:
            return [], [], False

        def get_layer_name(layer_idx):
            return layer if layer_idx == 0 else other_layer

        segments = []
        vias = []

        # Group consecutive same-layer, same-direction points
        seg_start_idx = 0

        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]

            # Check for layer change (via needed)
            if prev[2] != curr[2]:
                # End current segment
                start_pt = path[seg_start_idx]
                end_pt = prev

                sx = self.origin_x + start_pt[1] * self.grid_size
                sy = self.origin_y + start_pt[0] * self.grid_size
                ex = self.origin_x + end_pt[1] * self.grid_size
                ey = self.origin_y + end_pt[0] * self.grid_size

                if (sx, sy) != (ex, ey):
                    segments.append(TrackSegment(
                        start=(sx, sy), end=(ex, ey),
                        layer=get_layer_name(start_pt[2]),
                        width=self.trace_width, net=net_name
                    ))

                # Create via at layer change point
                vx = self.origin_x + prev[1] * self.grid_size
                vy = self.origin_y + prev[0] * self.grid_size
                via = self._create_via((vx, vy), net_name)
                if via:
                    vias.append(via)

                seg_start_idx = i

        # Add final segment
        start_pt = path[seg_start_idx]
        end_pt = path[-1]

        sx = self.origin_x + start_pt[1] * self.grid_size
        sy = self.origin_y + start_pt[0] * self.grid_size
        ex = self.origin_x + end_pt[1] * self.grid_size
        ey = self.origin_y + end_pt[0] * self.grid_size

        if (sx, sy) != (ex, ey):
            segments.append(TrackSegment(
                start=(sx, sy), end=(ex, ey),
                layer=get_layer_name(start_pt[2]),
                width=self.trace_width, net=net_name
            ))

        # Mark all segments as occupied
        for seg in segments:
            self._mark_line_occupied(seg.start[0], seg.start[1],
                                     seg.end[0], seg.end[1],
                                     net_name, seg.layer)

        return segments, vias, True

    def _lee_wavefront_route(self, start: Tuple[float, float], end: Tuple[float, float],
                              net_name: str, layer: str) -> Tuple[List[TrackSegment], bool]:
        """
        Lee's algorithm - wavefront propagation with GUARANTEED optimal path.

        This is slower than A* but guarantees finding a path if one exists.
        Uses BFS to explore all cells at distance d before exploring d+1.

        The wavefront expands like ripples in water until it reaches the destination.
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        # Convert to grid coordinates
        start_col = int((start[0] - self.origin_x) / self.grid_size)
        start_row = int((start[1] - self.origin_y) / self.grid_size)
        end_col = int((end[0] - self.origin_x) / self.grid_size)
        end_row = int((end[1] - self.origin_y) / self.grid_size)

        # Check bounds
        if not (0 <= start_row < self.grid_rows and 0 <= start_col < self.grid_cols):
            return [], False
        if not (0 <= end_row < self.grid_rows and 0 <= end_col < self.grid_cols):
            return [], False

        # Wave expansion - distance grid
        distance = {}
        distance[(start_row, start_col)] = 0

        # BFS queue
        queue = deque([(start_row, start_col)])

        # Parent tracking for backtrace
        parent = {}

        # Directions: 4-connected (Manhattan routing)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        found = False
        max_iterations = self.grid_rows * self.grid_cols

        iterations = 0
        while queue and not found and iterations < max_iterations:
            iterations += 1
            row, col = queue.popleft()
            current_dist = distance[(row, col)]

            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                # Skip if out of bounds
                if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                    continue
                # Skip if already visited
                if (nr, nc) in distance:
                    continue
                # Skip if blocked (with clearance check)
                if not self._is_cell_clear(grid, nr, nc, net_name):
                    continue

                # Mark distance and parent
                distance[(nr, nc)] = current_dist + 1
                parent[(nr, nc)] = (row, col)
                queue.append((nr, nc))

                # Check if we reached the destination
                if (nr, nc) == (end_row, end_col):
                    found = True
                    break

        if not found:
            return [], False

        # Backtrace to reconstruct path
        path = []
        current = (end_row, end_col)
        while current != (start_row, start_col):
            path.append(current)
            current = parent[current]
        path.append((start_row, start_col))
        path.reverse()

        # Convert grid path to track segments
        return self._path_to_segments(path, net_name, layer, allow_diagonal=False)

    def _path_to_segments(self, path: List[Tuple[int, int]], net_name: str,
                          layer: str, allow_diagonal: bool = False) -> Tuple[List[TrackSegment], bool]:
        """
        Convert A* grid path to track segments.

        Human approach: Merge consecutive cells in same direction into single segments.
        This produces clean, minimal segment counts.

        Supports both Manhattan (4-dir) and diagonal (8-dir) paths.
        """
        if len(path) < 2:
            return [], False

        segments = []
        segment_start = path[0]
        current_dir = None

        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]
            direction = (curr[0] - prev[0], curr[1] - prev[1])

            if current_dir is None:
                current_dir = direction
            elif direction != current_dir:
                # Direction changed - end current segment, start new one
                start_x = self.origin_x + segment_start[1] * self.grid_size
                start_y = self.origin_y + segment_start[0] * self.grid_size
                end_x = self.origin_x + prev[1] * self.grid_size
                end_y = self.origin_y + prev[0] * self.grid_size

                segments.append(TrackSegment(
                    start=(start_x, start_y),
                    end=(end_x, end_y),
                    layer=layer,
                    width=self.trace_width,
                    net=net_name
                ))

                segment_start = prev
                current_dir = direction

        # Add final segment
        start_x = self.origin_x + segment_start[1] * self.grid_size
        start_y = self.origin_y + segment_start[0] * self.grid_size
        end_x = self.origin_x + path[-1][1] * self.grid_size
        end_y = self.origin_y + path[-1][0] * self.grid_size

        segments.append(TrackSegment(
            start=(start_x, start_y),
            end=(end_x, end_y),
            layer=layer,
            width=self.trace_width,
            net=net_name
        ))

        return segments, True


def human_like_routing(board_config, parts_db: Dict, escapes: Dict,
                       route_order: List[str], placement: Dict = None,
                       design_rules = None) -> Dict[str, Route]:
    """
    Main entry point for human-like routing.

    Uses AGGRESSIVE multi-strategy approach: tries MANY different routing
    strategies and returns the best result. More strategies = better chance
    of finding a complete solution.

    Args:
        board_config: Board configuration (width, height, origin, grid)
        parts_db: Parts database with nets and component info
        escapes: Escape routes for all components
        route_order: Order of nets to route
        placement: Component placements (for obstacle avoidance)
        design_rules: Optional DesignRules with trace_width, clearance, etc.

    Returns: {net_name: Route}
    """
    import random

    nets = parts_db.get('nets', {})

    # Extract trace width and clearance from design rules (with defaults)
    trace_width = 0.5  # Default 0.5mm as user requested
    clearance = 0.2    # Default 0.2mm (standard)
    if design_rules is not None:
        trace_width = getattr(design_rules, 'min_trace_width', 0.5)
        clearance = getattr(design_rules, 'min_clearance', 0.2)

    # Build net_pins: {net_name: [(comp, pin), ...]}
    net_pins = {}
    for net_name, net_info in nets.items():
        pins = net_info.get('pins', [])
        net_pins[net_name] = pins

    def try_strategy(order: List[str], strategy_name: str) -> Tuple[Dict[str, Route], int]:
        """Try a routing strategy and return result with success count"""
        print(f"    Trying strategy: {strategy_name}...", end=" ", flush=True)
        router = HumanLikeRouter(
            board_width=board_config.width,
            board_height=board_config.height,
            origin_x=board_config.origin_x,
            origin_y=board_config.origin_y,
            grid_size=board_config.grid_size,
            trace_width=trace_width,
            clearance=clearance,
        )
        result = router.route_all(order, net_pins, escapes, placement, parts_db)
        success = sum(1 for r in result.values() if r.success)
        print(f"{success}/{total_nets} nets")
        return result, success

    best_result = None
    best_success = -1
    total_nets = len(route_order)

    # Define power/ground nets
    power_nets = {'GND', '3V3', 'VCC', 'VBUS', '5V', 'VIN', 'VBAT'}

    # =========================================================================
    # STRATEGY 1: Default order (short nets first)
    # =========================================================================
    result, success = try_strategy(route_order, "default")
    if success > best_success:
        best_result, best_success = result, success
    if best_success == total_nets:
        return best_result

    # Track if strategies are making progress
    last_success = success
    no_improvement_count = 0

    # =========================================================================
    # STRATEGY 2: Signal nets first, power nets last
    # =========================================================================
    signal_first = [n for n in route_order if n not in power_nets]
    power_last = [n for n in route_order if n in power_nets]
    result, success = try_strategy(signal_first + power_last, "signal_first")
    if success > best_success:
        best_result, best_success = result, success
        no_improvement_count = 0
    elif success == last_success:
        no_improvement_count += 1
    last_success = success
    if best_success == total_nets:
        return best_result

    # =========================================================================
    # STRATEGY 3: Power nets first (reserve space early)
    # =========================================================================
    result, success = try_strategy(power_last + signal_first, "power_first")
    if success > best_success:
        best_result, best_success = result, success
        no_improvement_count = 0
    elif success == last_success:
        no_improvement_count += 1
    last_success = success
    if best_success == total_nets:
        return best_result

    # =========================================================================
    # STRATEGY 4: By pin count ascending (2-pin first)
    # =========================================================================
    by_pin_asc = sorted(route_order, key=lambda n: len(net_pins.get(n, [])))
    result, success = try_strategy(by_pin_asc, "pin_count_asc")
    if success > best_success:
        best_result, best_success = result, success
    if best_success == total_nets:
        return best_result

    # =========================================================================
    # STRATEGY 5: By pin count descending (complex nets first)
    # =========================================================================
    by_pin_desc = sorted(route_order, key=lambda n: -len(net_pins.get(n, [])))
    result, success = try_strategy(by_pin_desc, "pin_count_desc")
    if success > best_success:
        best_result, best_success = result, success
    if best_success == total_nets:
        return best_result

    # =========================================================================
    # STRATEGY 6: GND first (it's usually the biggest net)
    # =========================================================================
    gnd_first = ['GND'] if 'GND' in route_order else []
    others = [n for n in route_order if n != 'GND']
    result, success = try_strategy(gnd_first + others, "gnd_first")
    if success > best_success:
        best_result, best_success = result, success
    if best_success == total_nets:
        return best_result

    # =========================================================================
    # EARLY EXIT: If first 6 strategies all give same result, skip to GND B.Cu
    # This saves time when simple reordering won't help
    # =========================================================================
    if best_success < total_nets and best_success == last_success:
        # Check if power net is the problem
        failed_nets = [n for n, r in best_result.items() if not r.success]
        power_failed = [n for n in failed_nets if n in power_nets]
        if power_failed:
            print(f"    [SKIP] Standard strategies stuck at {best_success}/{total_nets}, jumping to GND B.Cu strategy...")
            # Jump directly to GND B.Cu exclusive strategy (defined at end)
            goto_gnd_bcu = True
        else:
            goto_gnd_bcu = False
    else:
        goto_gnd_bcu = False

    if not goto_gnd_bcu:
        # =========================================================================
        # STRATEGY 7: GND last (save it for when other routes are done)
        # =========================================================================
        result, success = try_strategy(others + gnd_first, "gnd_last")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGY 8: Alphabetical (deterministic, sometimes helps)
        # =========================================================================
        alphabetical = sorted(route_order)
        result, success = try_strategy(alphabetical, "alphabetical")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGY 9: Reverse alphabetical
        # =========================================================================
        result, success = try_strategy(list(reversed(alphabetical)), "reverse_alpha")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGY 10: Critical signals first (USB, I2C, SPI)
        # =========================================================================
        critical = []
        normal = []
        for n in route_order:
            if any(sig in n.upper() for sig in ['USB', 'SDA', 'SCL', 'SPI', 'MOSI', 'MISO', 'CLK', 'CS']):
                critical.append(n)
            else:
                normal.append(n)
        result, success = try_strategy(critical + normal, "critical_first")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGIES 11-15: Random shuffles (sometimes randomness finds solutions)
        # =========================================================================
        random.seed(42)  # Reproducible randomness
        for i in range(5):
            shuffled = route_order.copy()
            random.shuffle(shuffled)
            result, success = try_strategy(shuffled, f"random_{i+1}")
            if success > best_success:
                best_result, best_success = result, success
            if best_success == total_nets:
                return best_result

        # =========================================================================
        # STRATEGY 16: Interleave power and signal
        # =========================================================================
        interleaved = []
        sig_iter = iter(signal_first)
        pwr_iter = iter(power_last)
        while True:
            try:
                interleaved.append(next(sig_iter))
            except StopIteration:
                break
            try:
                interleaved.append(next(pwr_iter))
            except StopIteration:
                pass
        # Add remaining power nets
        for p in pwr_iter:
            interleaved.append(p)
        result, success = try_strategy(interleaved, "interleaved")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGY 17: By estimated route length (shortest routes first)
        # =========================================================================
        def estimate_route_length(net_name: str) -> float:
            """Estimate total route length for a net"""
            endpoints = []
            for comp, pin in net_pins.get(net_name, []):
                if comp in escapes and pin in escapes[comp]:
                    esc = escapes[comp][pin]
                    end = esc.end if hasattr(esc, 'end') else getattr(esc, 'endpoint', None)
                    if end:
                        endpoints.append(end)
            if len(endpoints) < 2:
                return float('inf')
            # Sum of Manhattan distances between consecutive endpoints
            total = 0
            for i in range(len(endpoints) - 1):
                total += abs(endpoints[i+1][0] - endpoints[i][0]) + abs(endpoints[i+1][1] - endpoints[i][1])
            return total

        by_length = sorted(route_order, key=estimate_route_length)
        result, success = try_strategy(by_length, "by_length_asc")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGY 18: Longest routes first (get hard ones done early)
        # =========================================================================
        result, success = try_strategy(list(reversed(by_length)), "by_length_desc")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGY 19: 2-pin nets first, then by length
        # =========================================================================
        two_pin = [n for n in route_order if len(net_pins.get(n, [])) == 2]
        multi_pin = [n for n in route_order if len(net_pins.get(n, [])) > 2]
        two_pin_sorted = sorted(two_pin, key=estimate_route_length)
        multi_pin_sorted = sorted(multi_pin, key=estimate_route_length)
        result, success = try_strategy(two_pin_sorted + multi_pin_sorted, "2pin_then_multi")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

        # =========================================================================
        # STRATEGY 20: Multi-pin nets first, then 2-pin
        # =========================================================================
        result, success = try_strategy(multi_pin_sorted + two_pin_sorted, "multi_then_2pin")
        if success > best_success:
            best_result, best_success = result, success
        if best_success == total_nets:
            return best_result

    # =========================================================================
    # STRATEGY 21: Force GND/power nets to B.Cu exclusively
    # When all strategies give same result, GND routing is likely the issue.
    # This strategy pre-routes signal nets on F.Cu, then routes GND on B.Cu.
    # =========================================================================
    print(f"    Trying strategy: gnd_bcu_exclusive...", end=" ", flush=True)

    # Create router for hybrid approach
    router = HumanLikeRouter(
        board_width=board_config.width,
        board_height=board_config.height,
        origin_x=board_config.origin_x,
        origin_y=board_config.origin_y,
        grid_size=board_config.grid_size,
        trace_width=trace_width,
        clearance=clearance,
    )

    # Register obstacles
    if placement is not None:
        router.register_components_in_grid(placement, parts_db)
    router.register_escapes_in_grid(escapes)

    hybrid_result = {}
    hybrid_success = 0

    # Route signal nets FIRST on F.Cu ONLY (preserving B.Cu for power)
    # CRITICAL: Must NOT use vias here - they would block B.Cu for power nets
    signal_nets = [n for n in route_order if n not in power_nets]
    for net_name in signal_nets:
        pins = net_pins.get(net_name, [])
        if len(pins) < 2:
            continue

        # Get escape endpoints
        endpoints = []
        for comp, pin in pins:
            if comp in escapes and pin in escapes[comp]:
                esc = escapes[comp][pin]
                end = esc.end if hasattr(esc, 'end') else getattr(esc, 'endpoint', None)
                if end:
                    endpoints.append(end)

        if len(endpoints) >= 2:
            # Route on F.Cu only (NO vias allowed in this strategy)
            route = Route(net=net_name)
            all_connected = True

            # Connect endpoints pairwise on F.Cu
            for i in range(len(endpoints) - 1):
                start_pt = endpoints[i]
                end_pt = endpoints[i + 1]

                # Try F.Cu only algorithms (no B.Cu fallback)
                segments, success = router._manhattan_route(start_pt, end_pt, net_name, 'F.Cu', 'h_first')
                if not success:
                    segments, success = router._manhattan_route(start_pt, end_pt, net_name, 'F.Cu', 'v_first')
                if not success:
                    segments, success = router._astar_route(start_pt, end_pt, net_name, 'F.Cu')
                if not success:
                    segments, success = router._astar_route(start_pt, end_pt, net_name, 'F.Cu', allow_diagonal=True)
                if not success:
                    segments, success = router._lee_wavefront_route(start_pt, end_pt, net_name, 'F.Cu')

                if success:
                    router._mark_occupied(segments, 'F.Cu')
                    route.segments.extend(segments)
                else:
                    all_connected = False
                    route.error = f"F.Cu-only routing failed for {net_name}"
                    break

            if all_connected:
                route.success = True

            hybrid_result[net_name] = route
            if route.success:
                hybrid_success += 1
        else:
            hybrid_result[net_name] = Route(net=net_name, error=f"Only {len(endpoints)} endpoints")

    # Now route power nets on B.Cu using daisy-chain
    # IMPORTANT: Route GND FIRST since it typically has the most connections
    # and needs the most routing flexibility. Then route other power nets.
    gnd_nets = [n for n in route_order if n in power_nets]
    # Sort to put 'GND' first, then other power nets by pin count (descending)
    gnd_nets = sorted(gnd_nets, key=lambda n: (0 if n == 'GND' else 1, -len(net_pins.get(n, []))))
    power_routes = {}  # {net_name: Route}
    all_power_segments = []  # Collect all segments to mark at end

    for net_name in gnd_nets:
        pins = net_pins.get(net_name, [])
        if len(pins) < 2:
            continue

        # Get escape endpoints
        endpoints = []
        for comp, pin in pins:
            if comp in escapes and pin in escapes[comp]:
                esc = escapes[comp][pin]
                end = esc.end if hasattr(esc, 'end') else getattr(esc, 'endpoint', None)
                if end:
                    endpoints.append(end)

        if len(endpoints) >= 2:
            # Try daisy-chain on B.Cu
            route = router._route_daisy_chain_bcu(net_name, endpoints)
            if not route.success:
                # Fallback to star topology
                route = router._route_star_topology(net_name, endpoints)
            if not route.success:
                # Ultimate fallback to normal routing
                route = router._route_net(net_name, pins, escapes)

            power_routes[net_name] = route
            if route.success:
                all_power_segments.extend(route.segments)

    # Power segments are already marked during routing
    # (So each net routes AROUND previous nets, not THROUGH them)

    # Create vias for all successful power routes (after all routing done)
    for net_name, route in power_routes.items():
        if route.success:
            # Create vias at pending endpoints
            if hasattr(route, '_pending_via_endpoints'):
                for endpoint in route._pending_via_endpoints:
                    via = router._create_via(endpoint, net_name)
                    if via:
                        route.vias.append(via)
            # Create centroid via for star topology
            if hasattr(route, '_pending_via_centroid'):
                via = router._create_via(route._pending_via_centroid, net_name)
                if via:
                    route.vias.append(via)

    # Add power routes to hybrid result
    for net_name, route in power_routes.items():
        hybrid_result[net_name] = route
        if route.success:
            hybrid_success += 1

    print(f"{hybrid_success}/{total_nets} nets")
    if hybrid_success > best_success:
        best_result, best_success = hybrid_result, hybrid_success

    return best_result
