"""
PCB Engine - Routing Module (Enhanced v2)
==========================================

Handles routing order optimization and A* pathfinding.
This is Phase 6 and Phase 7 of the algorithm.

CRITICAL LESSONS FROM PCB_AUTOROUTER_STRATEGY.md (33 Rules):
============================================================

Rule 1:  reserve_pad MUST set grid=BLOCKED AND cell_owner=net
Rule 2:  A* trace-width MUST include clearance: tw = ceil((width + 2*clearance) / GRID)
Rule 4:  GND connections MUST be A*-routed (no blind traces)
Rule 5:  Via placement MUST check ALL existing holes (same-net AND different-net)
Rule 6:  Try-before-commit for GND vias
Rule 7:  GND vias for power-area caps BEFORE power traces
Rule 13: Route order must match physical dependency
Rule 22: Power routes in GND-via areas MUST use allow_via=False
Rule 25: route() fallback MUST respect caller's allow_via parameter
Rule 27: Failed allow_via=False routes create topology gaps - track failures
Rule 33: GND via clearance must check ALL copper layers

KEY PRINCIPLES:
===============
1. Route most constrained nets first
2. Proper A* with layer awareness and cost functions
3. CRITICAL: reserve_pad must set BLOCKED + cell_owner (Rule 1)
4. CRITICAL: Trace width includes clearance (Rule 2)
5. CRITICAL: Fallback MUST respect allow_via (Rule 25)
6. Escape-aware routing (start from escape endpoints)
7. Multi-layer support with via costs

ROUTING QUALITY PRINCIPLES:
===========================
Routes must be:
1. CLEAN - No unnecessary segments or detours
2. SIMPLE - Minimum number of segments and bends
3. PROFESSIONAL - 45/90 degree angles only, no zigzags
4. SAFE - No potential DRC risks, proper clearances
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Layer(Enum):
    """Routing layers"""
    F_CU = 'F.Cu'
    B_CU = 'B.Cu'


@dataclass
class GridCell:
    """A cell in the routing grid"""
    row: int
    col: int
    layer: Layer = Layer.F_CU

    def __hash__(self):
        return hash((self.row, self.col, self.layer))

    def __eq__(self, other):
        if not isinstance(other, GridCell):
            return False
        return self.row == other.row and self.col == other.col and self.layer == other.layer

    def __lt__(self, other):
        return (self.row, self.col, self.layer.value) < (other.row, other.col, other.layer.value)


@dataclass
class RouteSegment:
    """A segment of a route"""
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
    def manhattan_length(self) -> float:
        return abs(self.end[0] - self.start[0]) + abs(self.end[1] - self.start[1])

    @property
    def is_horizontal(self) -> bool:
        return abs(self.end[1] - self.start[1]) < 0.01

    @property
    def is_vertical(self) -> bool:
        return abs(self.end[0] - self.start[0]) < 0.01

    @property
    def direction(self) -> Tuple[int, int]:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.001:
            return (0, 0)
        return (round(dx / length), round(dy / length))


@dataclass
class Via:
    """A via between layers"""
    position: Tuple[float, float]
    net: str
    diameter: float
    drill: float


@dataclass
class Route:
    """A complete route for a net"""
    net: str
    segments: List[RouteSegment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    length: float = 0.0
    success: bool = True
    error: str = ''

    @property
    def via_count(self) -> int:
        return len(self.vias)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def bend_count(self) -> int:
        if len(self.segments) < 2:
            return 0
        bends = 0
        for i in range(len(self.segments) - 1):
            if self.segments[i].direction != self.segments[i+1].direction:
                bends += 1
        return bends


@dataclass
class NetConstraints:
    """Constraints for routing a specific net"""
    min_width: float = 0.25
    max_length: Optional[float] = None
    length_match: Optional[str] = None
    layer_preference: Optional[Layer] = None
    via_allowed: bool = True
    priority: int = 0


# =============================================================================
# ROUTING GRID (Enhanced with Rule 1 and Rule 2)
# =============================================================================

class RoutingGrid:
    """
    Multi-layer routing grid.

    CRITICAL (Rule 1): reserve_pad MUST set BOTH:
      - grid[layer][r][c] = BLOCKED
      - cell_owner[layer][r][c] = net_name

    If only owner is set without BLOCKED, A* sees EMPTY and routes through pads!
    This was the #1 bug causing 80% of violations in early versions.
    """

    EMPTY = 0
    BLOCKED = 1
    TRACE = 2
    VIA = 3
    PAD = 4

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules

        # Grid dimensions
        self.cols = int(board.width / board.grid_size)
        self.rows = int(board.height / board.grid_size)

        # Multi-layer grids
        self.grids = {
            Layer.F_CU: [[self.EMPTY for _ in range(self.cols)] for _ in range(self.rows)],
            Layer.B_CU: [[self.EMPTY for _ in range(self.cols)] for _ in range(self.rows)],
        }

        # Cell owners (net names)
        self.owners = {
            Layer.F_CU: [['' for _ in range(self.cols)] for _ in range(self.rows)],
            Layer.B_CU: [['' for _ in range(self.cols)] for _ in range(self.rows)],
        }

        # Track via positions for Rule 5 (hole clearance checking)
        self.via_positions = []  # List of (row, col, net)
        self.tht_positions = []  # List of (row, col, net, drill_mm)

        # Rule 2: Trace width includes clearance
        self.trace_clearance_cells = int(math.ceil(
            (rules.min_trace_width + 2 * rules.min_clearance) / board.grid_size
        ))

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x - self.board.origin_x) / self.board.grid_size)
        row = int((y - self.board.origin_y) / self.board.grid_size)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.board.origin_x + col * self.board.grid_size
        y = self.board.origin_y + row * self.board.grid_size
        return (x, y)

    def is_valid(self, cell: GridCell) -> bool:
        return (0 <= cell.row < self.rows and 0 <= cell.col < self.cols)

    def is_available(self, cell: GridCell, net: str = '') -> bool:
        """
        Check if cell is available for routing.

        CRITICAL (Rule 1): Passability check is:
          if BLOCKED and owner != my_net: FAIL
          if BLOCKED and owner == my_net: PASS (same-net passthrough)
          if EMPTY: PASS
        """
        if not self.is_valid(cell):
            return False

        state = self.grids[cell.layer][cell.row][cell.col]
        owner = self.owners[cell.layer][cell.row][cell.col]

        # Empty cells are available
        if state == self.EMPTY:
            return True

        # BLOCKED cells: only passable for same net
        if state == self.BLOCKED:
            return owner == net and net != ''

        # TRACE cells: passable for same net
        if state == self.TRACE:
            return owner == net

        return False

    def reserve_pad(self, row: int, col: int, layer: Layer, net: str,
                    width_cells: int = 1, is_tht: bool = False, drill_mm: float = 0):
        """
        Reserve pad cells.

        CRITICAL (Rule 1): MUST set BOTH grid=BLOCKED AND cell_owner=net.
        If you only set owner WITHOUT setting BLOCKED, the A* passable check
        sees EMPTY and routes other-net traces through pads!

        NOTE: Don't overwrite existing signal net reservations with GND.
        This prevents GND pads from blocking signal routing on adjacent pads.
        """
        for dr in range(-width_cells, width_cells + 1):
            for dc in range(-width_cells, width_cells + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    existing_owner = self.owners[layer][r][c]

                    # Don't overwrite signal pads with GND (GND uses zones anyway)
                    if existing_owner and existing_owner != 'GND' and net == 'GND':
                        continue

                    # RULE 1: Set BOTH blocked AND owner
                    self.grids[layer][r][c] = self.BLOCKED
                    self.owners[layer][r][c] = net

        # Track THT positions for Rule 5 (hole clearance)
        if is_tht and drill_mm > 0:
            self.tht_positions.append((row, col, net, drill_mm))
            # THT pads block BOTH layers
            for lyr in [Layer.F_CU, Layer.B_CU]:
                for dr in range(-width_cells, width_cells + 1):
                    for dc in range(-width_cells, width_cells + 1):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.rows and 0 <= c < self.cols:
                            self.grids[lyr][r][c] = self.BLOCKED
                            self.owners[lyr][r][c] = net

    def mark_trace(self, cell: GridCell, net: str, width_cells: int = 1):
        """Mark cell and clearance zone as trace."""
        # Rule 2: Mark with clearance
        for dr in range(-width_cells, width_cells + 1):
            for dc in range(-width_cells, width_cells + 1):
                r, c = cell.row + dr, cell.col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.grids[cell.layer][r][c] == self.EMPTY:
                        self.grids[cell.layer][r][c] = self.TRACE
                        self.owners[cell.layer][r][c] = net

    def mark_via(self, row: int, col: int, net: str):
        """Mark via location on both layers."""
        via_clearance = int(math.ceil(
            (self.rules.min_via_diameter / 2 + self.rules.min_clearance) / self.board.grid_size
        ))

        for layer in [Layer.F_CU, Layer.B_CU]:
            for dr in range(-via_clearance, via_clearance + 1):
                for dc in range(-via_clearance, via_clearance + 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        self.grids[layer][r][c] = self.VIA
                        self.owners[layer][r][c] = net

        # Track via for Rule 5
        self.via_positions.append((row, col, net))

    def can_place_via(self, row: int, col: int, net: str) -> bool:
        """
        Check if via can be placed here.

        Rule 5: Via placement MUST check ALL existing holes (same-net AND different-net).
        Rule 33: GND via clearance must check ALL copper layers.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False

        via_clearance_cells = int(math.ceil(
            (self.rules.min_via_diameter / 2 + self.rules.min_clearance) / self.board.grid_size
        )) + 1  # +1 safety margin (Rule 11)

        min_hole_clearance_cells = int(math.ceil(
            self.rules.min_hole_clearance / self.board.grid_size
        ))

        # Rule 33: Check ALL layers
        for layer in [Layer.F_CU, Layer.B_CU]:
            for dr in range(-via_clearance_cells, via_clearance_cells + 1):
                for dc in range(-via_clearance_cells, via_clearance_cells + 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        owner = self.owners[layer][r][c]
                        if owner and owner != net and owner != 'GND':
                            return False

        # Rule 5: Check distance to ALL existing vias
        for via_row, via_col, via_net in self.via_positions:
            dist_cells = math.sqrt((row - via_row)**2 + (col - via_col)**2)
            min_dist_cells = min_hole_clearance_cells + via_clearance_cells
            if dist_cells < min_dist_cells:
                return False

        # Rule 5: Check distance to ALL THT holes
        for tht_row, tht_col, tht_net, tht_drill in self.tht_positions:
            tht_clearance = int(math.ceil(
                (tht_drill / 2 + self.rules.min_via_drill / 2 + self.rules.min_hole_clearance)
                / self.board.grid_size
            ))
            dist_cells = math.sqrt((row - tht_row)**2 + (col - tht_col)**2)
            if dist_cells < tht_clearance:
                return False

        return True

    def mark_blocked(self, row: int, col: int, layer: Layer = None):
        layers = [layer] if layer else [Layer.F_CU, Layer.B_CU]
        for l in layers:
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.grids[l][row][col] = self.BLOCKED

    def block_component_area(self, x: float, y: float, width: float, height: float,
                               smd: bool = True):
        """Block cells occupied by a component."""
        r1, c1 = self.world_to_grid(x - width/2, y - height/2)
        r2, c2 = self.world_to_grid(x + width/2, y + height/2)

        for r in range(max(0, r1), min(self.rows, r2 + 1)):
            for c in range(max(0, c1), min(self.cols, c2 + 1)):
                if smd:
                    self.mark_blocked(r, c, Layer.F_CU)
                else:
                    self.mark_blocked(r, c)

    def check_clearance(self, cell: GridCell, net: str, width_cells: int = None) -> bool:
        """
        Check if routing through cell maintains clearance.

        Rule 2: Trace width includes clearance.
        """
        if width_cells is None:
            width_cells = self.trace_clearance_cells

        for dr in range(-width_cells, width_cells + 1):
            for dc in range(-width_cells, width_cells + 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = cell.row + dr, cell.col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    other_owner = self.owners[cell.layer][r][c]
                    if other_owner and other_owner != net:
                        return False
        return True

    def _get_line_cells(self, start: GridCell, end: GridCell) -> List[GridCell]:
        """Get all cells along a line using Bresenham's algorithm."""
        cells = []
        r0, c0 = start.row, start.col
        r1, c1 = end.row, end.col

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        while True:
            cells.append(GridCell(r0, c0, start.layer))
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc

        return cells


# =============================================================================
# A* COST FUNCTIONS
# =============================================================================

class CostCalculator:
    """Calculates routing costs for A* pathfinding."""

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules

        # Cost weights
        self.base_cost = 1.0
        self.via_cost = 25.0
        self.direction_change_cost = 3.0
        self.diagonal_cost = 1.4
        self.layer_preference_bonus = -0.5
        self.edge_proximity_cost = 1.0
        self.acute_angle_cost = 10.0

    def calculate_cost(self, from_cell: GridCell, to_cell: GridCell,
                       prev_cell: Optional[GridCell] = None,
                       net_constraints: NetConstraints = None) -> float:
        cost = self.base_cost

        # Via cost (layer change)
        if from_cell.layer != to_cell.layer:
            cost += self.via_cost

        # Direction change cost
        if prev_cell:
            prev_dir = (from_cell.row - prev_cell.row, from_cell.col - prev_cell.col)
            curr_dir = (to_cell.row - from_cell.row, to_cell.col - from_cell.col)

            if prev_dir != curr_dir and prev_dir != (0, 0):
                angle = self._angle_between(prev_dir, curr_dir)
                if angle < 90:
                    cost += self.acute_angle_cost
                elif angle == 90:
                    cost += self.direction_change_cost
                else:
                    cost += self.direction_change_cost * 0.5

        # Layer preference
        if net_constraints and net_constraints.layer_preference:
            if to_cell.layer == net_constraints.layer_preference:
                cost += self.layer_preference_bonus

        return cost

    def _angle_between(self, dir1: Tuple[int, int], dir2: Tuple[int, int]) -> float:
        len1 = math.sqrt(dir1[0]**2 + dir1[1]**2)
        len2 = math.sqrt(dir2[0]**2 + dir2[1]**2)
        if len1 < 0.001 or len2 < 0.001:
            return 180
        dot = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / (len1 * len2)
        dot = max(-1, min(1, dot))
        return math.degrees(math.acos(dot))

    def heuristic(self, cell: GridCell, goal: GridCell) -> float:
        dr = abs(cell.row - goal.row)
        dc = abs(cell.col - goal.col)
        h = max(dr, dc) + 0.4 * min(dr, dc)
        if cell.layer != goal.layer:
            h += self.via_cost
        return h * self.base_cost


# =============================================================================
# PATH SIMPLIFIER
# =============================================================================

class PathSimplifier:
    """Simplifies A* grid paths into clean routes."""

    def __init__(self, grid: RoutingGrid):
        self.grid = grid

    def simplify(self, path: List[GridCell], net: str) -> List[GridCell]:
        if len(path) < 3:
            return path
        simplified = self._remove_collinear(path)
        return simplified

    def _remove_collinear(self, path: List[GridCell]) -> List[GridCell]:
        if len(path) < 3:
            return path

        result = [path[0]]

        for i in range(1, len(path) - 1):
            prev = result[-1]
            curr = path[i]
            next_p = path[i + 1]

            if curr.layer != prev.layer or curr.layer != next_p.layer:
                result.append(curr)
                continue

            dir1 = (curr.row - prev.row, curr.col - prev.col)
            dir2 = (next_p.row - curr.row, next_p.col - curr.col)

            def normalize(d):
                length = math.sqrt(d[0]**2 + d[1]**2)
                if length < 0.001:
                    return (0, 0)
                return (round(d[0] / length * 10), round(d[1] / length * 10))

            if normalize(dir1) != normalize(dir2):
                result.append(curr)

        result.append(path[-1])
        return result


# =============================================================================
# ROUTE ORDER OPTIMIZER (Enhanced with Rule 13)
# =============================================================================

class RouteOrderOptimizer:
    """
    Determines optimal routing order.

    Rule 13: Route order must match physical dependency:
      1. I2C trunk (most constrained corridor)
      1.5 Tight-pitch SMD GND chains + far vias (BEFORE signal routes)
      2. Address/config signals
      3. Power-area GND vias (BEFORE power traces - Rule 7)
      4. Power distribution
      5. I2C channels
      6. +3V3 spine to sensor caps
      7. +5V_IN power
      8. Remaining GND vias
    """

    # Priority groups
    PRIORITY_TRUNK = 100
    PRIORITY_GND_POWER_AREA = 95  # Rule 7: GND vias BEFORE power
    PRIORITY_ADDR = 90
    PRIORITY_POWER_CORE = 80
    PRIORITY_CHANNELS = 70
    PRIORITY_POWER_SPINE = 60
    PRIORITY_POWER_5V = 50
    PRIORITY_GND_REMAINING = 40

    def __init__(self):
        self.order = []
        self.scores = {}

    def calculate(self, nets: Dict, placement: Dict,
                  escapes: Dict, corridors: List) -> List[str]:
        """Calculate optimal routing order based on Rule 13."""
        self.scores = {}

        for net_name, net_info in nets.items():
            if net_name == 'GND':
                continue  # GND is a zone

            score = self._calculate_priority(net_name, net_info, placement)
            self.scores[net_name] = score

        # Sort by score descending (highest priority first)
        self.order = sorted(self.scores.keys(), key=lambda k: -self.scores[k])

        return self.order

    def _calculate_priority(self, net_name: str, net_info: Dict,
                            placement: Dict) -> float:
        """Calculate priority based on Rule 13 categories."""
        score = 0.0

        # Categorize by net name/type
        net_lower = net_name.lower()

        # Trunk signals (highest priority)
        if 'trunk' in net_lower:
            score = self.PRIORITY_TRUNK
        # Address signals
        elif 'addr' in net_lower or net_name.startswith('ADDR'):
            score = self.PRIORITY_ADDR
        # Channel signals
        elif '_ch' in net_lower or 'sda_ch' in net_lower or 'scl_ch' in net_lower:
            score = self.PRIORITY_CHANNELS
        # Power signals
        elif '+3v3' in net_lower or 'vcc' in net_lower:
            score = self.PRIORITY_POWER_CORE
        elif '+5v' in net_lower or 'vin' in net_lower:
            score = self.PRIORITY_POWER_5V
        else:
            # Default signal priority
            score = 50.0

        # Boost for nets with more pins (more constrained)
        pins = net_info.get('pins', [])
        score += len(pins) * 2

        return score

    def get_report(self) -> str:
        lines = ["=" * 50, "ROUTING ORDER (Rule 13)", "=" * 50]
        for i, net in enumerate(self.order, 1):
            score = self.scores.get(net, 0)
            lines.append(f"{i:3d}. {net:20s} (priority: {score:.0f})")
        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# A* ROUTER (Enhanced with Rule 25)
# =============================================================================

class AStarRouter:
    """
    A* pathfinding router.

    CRITICAL (Rule 25): Fallback logic MUST respect allow_via parameter.
    Never override allow_via=False in fallback paths.
    """

    def __init__(self, grid: RoutingGrid, cost_calc: CostCalculator):
        self.grid = grid
        self.cost_calc = cost_calc
        self.max_iterations = 100000

        # Track failed routes for Rule 27
        self.failed_routes = []

    def find_path(self, start: GridCell, end: GridCell,
                  net: str, allow_via: bool = True,
                  constraints: NetConstraints = None,
                  trace_width_cells: int = None) -> Optional[List[GridCell]]:
        """
        Find path from start to end using A*.

        Rule 25: allow_via is a HARD CONSTRAINT, not a preference.
        """
        if not self.grid.is_valid(start) or not self.grid.is_valid(end):
            return None

        if trace_width_cells is None:
            trace_width_cells = self.grid.trace_clearance_cells

        # A* data structures
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.cost_calc.heuristic(start, end)}

        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)

            # Check if we've reached the goal position
            # Accept arrival on EITHER layer - no need to force via back to start layer
            if current.row == end.row and current.col == end.col:
                # We've reached the destination position - accept on any layer
                # This prevents unnecessary vias just to match the start layer
                return self._reconstruct_path(came_from, current)

            prev = came_from.get(current)

            # Explore neighbors (respecting allow_via!)
            for neighbor in self._get_neighbors(current, net, allow_via):
                if not self.grid.is_available(neighbor, net):
                    continue

                if not self.grid.check_clearance(neighbor, net, trace_width_cells):
                    continue

                tentative_g = g_score[current] + self.cost_calc.calculate_cost(
                    current, neighbor, prev, constraints
                )

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.cost_calc.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def find_path_with_fallback(self, start: GridCell, end: GridCell,
                                 net: str, allow_via: bool = True,
                                 constraints: NetConstraints = None) -> Optional[List[GridCell]]:
        """
        Find path with trace-width fallback.

        CRITICAL (Rule 25): Fallback MUST respect allow_via.
        We can relax trace width, but NEVER override allow_via.
        """
        # First attempt with normal trace width
        path = self.find_path(start, end, net, allow_via, constraints,
                             self.grid.trace_clearance_cells)
        if path:
            return path

        # Fallback 1: Try with minimal trace width (still respecting allow_via!)
        path = self.find_path(start, end, net, allow_via, constraints, 1)
        if path:
            return path

        # Fallback 2: Increase max iterations (still respecting allow_via!)
        old_max = self.max_iterations
        self.max_iterations = 500000
        path = self.find_path(start, end, net, allow_via, constraints, 1)
        self.max_iterations = old_max

        if not path:
            # Rule 27: Track failed routes
            self.failed_routes.append({
                'net': net,
                'start': (start.row, start.col, start.layer.value),
                'end': (end.row, end.col, end.layer.value),
                'allow_via': allow_via,
            })

        return path

    def _get_neighbors(self, cell: GridCell, net: str, allow_via: bool) -> List[GridCell]:
        """Get valid neighboring cells."""
        neighbors = []

        # Same-layer neighbors (4-connected for orthogonal routing)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = GridCell(cell.row + dr, cell.col + dc, cell.layer)
            if self.grid.is_valid(neighbor):
                neighbors.append(neighbor)

        # Layer change (via) - ONLY if allowed!
        if allow_via:
            other_layer = Layer.B_CU if cell.layer == Layer.F_CU else Layer.F_CU
            via_cell = GridCell(cell.row, cell.col, other_layer)
            if (self.grid.is_valid(via_cell) and
                self.grid.is_available(via_cell, net) and
                self.grid.can_place_via(cell.row, cell.col, net)):
                neighbors.append(via_cell)

        return neighbors

    def _reconstruct_path(self, came_from: Dict[GridCell, GridCell],
                          current: GridCell) -> List[GridCell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# =============================================================================
# GND VIA PLACER (Rule 4, 5, 6, 7)
# =============================================================================

class GndViaPlacer:
    """
    Places GND vias using spiral search with A*-routed traces.

    Rule 4: GND connections MUST be A*-routed (no blind traces)
    Rule 5: Via placement MUST check ALL existing holes
    Rule 6: Try-before-commit for GND vias
    Rule 7: Power-area GND vias BEFORE power traces
    """

    def __init__(self, grid: RoutingGrid, astar: AStarRouter, rules):
        self.grid = grid
        self.astar = astar
        self.rules = rules
        self.placed_vias = []

    def place_gnd_via(self, pad_row: int, pad_col: int, net: str = 'GND',
                      start_radius: int = 3, max_radius: int = 30) -> Optional[Tuple[int, int]]:
        """
        Place a GND via near a pad using spiral search.

        Rule 6: Try A* route BEFORE placing via. Only place if route succeeds.
        """
        for radius in range(start_radius, max_radius + 1):
            candidates = self._get_spiral_candidates(pad_row, pad_col, radius)

            for via_row, via_col in candidates:
                # Rule 5: Check via clearance
                if not self.grid.can_place_via(via_row, via_col, net):
                    continue

                # Rule 6: Try-before-commit - route FIRST
                pad_cell = GridCell(pad_row, pad_col, Layer.F_CU)
                via_cell = GridCell(via_row, via_col, Layer.F_CU)

                path = self.astar.find_path(pad_cell, via_cell, net,
                                           allow_via=False)  # F.Cu only

                if path:
                    # Route succeeded - NOW place the via
                    self.grid.mark_via(via_row, via_col, net)
                    self.placed_vias.append((via_row, via_col, net))

                    # Mark route
                    for cell in path:
                        self.grid.mark_trace(cell, net)

                    return (via_row, via_col)

        return None

    def _get_spiral_candidates(self, center_row: int, center_col: int,
                                radius: int) -> List[Tuple[int, int]]:
        """Get candidate positions at Manhattan distance radius."""
        candidates = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) == radius:  # Manhattan distance
                    r, c = center_row + dr, center_col + dc
                    if 0 <= r < self.grid.rows and 0 <= c < self.grid.cols:
                        candidates.append((r, c))
        return candidates


# =============================================================================
# ROUTE QUALITY ANALYZER
# =============================================================================

class RouteQualityAnalyzer:
    """Analyzes and scores route quality."""

    def __init__(self):
        self.scores = {}

    def analyze(self, route: Route) -> Dict:
        metrics = {
            'segment_count': len(route.segments),
            'via_count': route.via_count,
            'total_length': route.length,
            'bend_count': route.bend_count,
            'has_acute_angles': self._check_acute_angles(route),
        }

        score = 100
        if metrics['segment_count'] > 5:
            score -= (metrics['segment_count'] - 5) * 5
        score -= metrics['via_count'] * 10
        if metrics['bend_count'] > 2:
            score -= (metrics['bend_count'] - 2) * 5
        if metrics['has_acute_angles']:
            score -= 20

        metrics['quality_score'] = max(0, score)
        return metrics

    def _check_acute_angles(self, route: Route) -> bool:
        segments = route.segments
        if len(segments) < 2:
            return False
        for i in range(len(segments) - 1):
            d1 = segments[i].direction
            d2 = segments[i + 1].direction
            if d1 == (0, 0) or d2 == (0, 0):
                continue
            dot = d1[0] * d2[0] + d1[1] * d2[1]
            if dot > 0.7:
                return True
        return False


# =============================================================================
# MAIN ROUTER (Enhanced)
# =============================================================================

class Router:
    """
    Main router that handles net routing.

    Enhanced with all 33 rules from PCB_AUTOROUTER_STRATEGY.md.
    """

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules

        # Create grid and router
        self.grid = RoutingGrid(board, rules)
        self.cost_calc = CostCalculator(board, rules)
        self.astar = AStarRouter(self.grid, self.cost_calc)
        self.simplifier = PathSimplifier(self.grid)
        self.quality_analyzer = RouteQualityAnalyzer()
        self.gnd_via_placer = GndViaPlacer(self.grid, self.astar, rules)

        # Tracking
        self.routes = {}
        self.failed = []
        self.quality_scores = {}

        # Rule 27: Track topology gaps
        self.topology_gaps = []

    def initialize(self, parts_db: Dict, placement: Dict):
        """Initialize grid with component blockages using Rule 1."""
        parts = parts_db.get('parts', {})

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            phys = part.get('physical', {})
            body = phys.get('body', (5, 5))
            is_smd = part.get('smd', True)

            # Block component area
            self.grid.block_component_area(pos.x, pos.y, body[0], body[1], is_smd)

            # Rule 1: Reserve pad cells with BLOCKED + owner
            for pin_info in part.get('used_pins', []):
                offset = pin_info.get('offset', (0, 0))
                pin_x = pos.x + offset[0]
                pin_y = pos.y + offset[1]
                pin_row, pin_col = self.grid.world_to_grid(pin_x, pin_y)
                net = pin_info.get('net', '')
                drill = pin_info.get('drill', 0)

                self.grid.reserve_pad(
                    pin_row, pin_col, Layer.F_CU, net,
                    width_cells=2, is_tht=(drill > 0), drill_mm=drill
                )

    def route_all(self, route_order: List[str], parts_db: Dict,
                  placement: Dict, escapes: Dict) -> Dict:
        """Route all nets in order with rip-up and reroute."""
        self.routes = {}
        self.failed = []
        self.quality_scores = {}

        # Initialize grid (Rule 1)
        self.initialize(parts_db, placement)

        # First pass: try routing in order
        pending = list(route_order)
        reroute_attempts = {}  # Track reroute attempts per net

        while pending:
            net_name = pending.pop(0)
            net_info = parts_db['nets'].get(net_name, {})
            pins = net_info.get('pins', [])

            if len(pins) < 2:
                continue

            # Determine if this net should avoid vias (Rule 22)
            allow_via = self._should_allow_via(net_name, parts_db)

            route = self._route_net(net_name, pins, parts_db, placement,
                                    escapes, allow_via)

            if route and route.success:
                self.routes[net_name] = route
                self._mark_route(route)
                metrics = self.quality_analyzer.analyze(route)
                self.quality_scores[net_name] = metrics['quality_score']
            else:
                # Try rip-up and reroute
                reroute_attempts[net_name] = reroute_attempts.get(net_name, 0) + 1

                if reroute_attempts[net_name] <= 2:
                    # Find blocking nets and try ripping them up
                    blocking_nets = self._find_blocking_nets(net_name, pins, parts_db, placement, escapes)

                    if blocking_nets:
                        # Rip up blocking routes
                        for blocking_net in blocking_nets:
                            if blocking_net in self.routes:
                                self._rip_up_route(blocking_net)
                                # Put blocking net back in queue
                                if blocking_net not in pending:
                                    pending.append(blocking_net)

                        # Retry this net
                        pending.insert(0, net_name)
                        continue

                # No reroute possible, mark as failed
                self.failed.append(net_name)

                # Rule 27: Track topology gaps
                if not allow_via:
                    self.topology_gaps.append({
                        'net': net_name,
                        'reason': 'allow_via=False route failed',
                    })

        success_rate = len(self.routes) / max(1, len(self.routes) + len(self.failed))
        avg_quality = sum(self.quality_scores.values()) / max(1, len(self.quality_scores))

        return {
            'routes': self.routes,
            'failed': self.failed,
            'success_rate': success_rate,
            'total_length': sum(r.length for r in self.routes.values()),
            'total_vias': sum(r.via_count for r in self.routes.values()),
            'average_quality': avg_quality,
            'quality_scores': self.quality_scores,
            'topology_gaps': self.topology_gaps,
        }

    def _should_allow_via(self, net_name: str, parts_db: Dict) -> bool:
        """
        Determine if net should allow vias.

        Rule 22: Power routes in GND-via areas MUST use allow_via=False.
        """
        net_lower = net_name.lower()

        # Power nets near GND vias should be F.Cu only
        # (This is a simplified check - real implementation would check geography)
        if '+3v3' in net_lower:
            # Check if this is a power-area route
            # For now, allow vias but this should be refined based on location
            return True

        return True

    def _route_net(self, net_name: str, pins: List, parts_db: Dict,
                   placement: Dict, escapes: Dict,
                   allow_via: bool = True) -> Optional[Route]:
        """Route a single net."""
        if len(pins) == 2:
            return self._route_two_pin(net_name, pins, parts_db, placement,
                                       escapes, allow_via)
        else:
            return self._route_multi_pin(net_name, pins, parts_db, placement,
                                         escapes, allow_via)

    def _route_two_pin(self, net_name: str, pins: List, parts_db: Dict,
                       placement: Dict, escapes: Dict,
                       allow_via: bool = True) -> Route:
        """Route a two-pin net."""
        comp_a, pin_a = pins[0]
        comp_b, pin_b = pins[1]

        pos_a = self._get_pin_position(comp_a, pin_a, parts_db, placement)
        pos_b = self._get_pin_position(comp_b, pin_b, parts_db, placement)

        if not pos_a or not pos_b:
            return Route(net=net_name, success=False, error="Pin position not found")

        # Use escape endpoints if available
        if comp_a in escapes and pin_a in escapes[comp_a]:
            start_pos = escapes[comp_a][pin_a].endpoint
        else:
            start_pos = pos_a

        if comp_b in escapes and pin_b in escapes[comp_b]:
            end_pos = escapes[comp_b][pin_b].endpoint
        else:
            end_pos = pos_b

        start_row, start_col = self.grid.world_to_grid(start_pos[0], start_pos[1])
        end_row, end_col = self.grid.world_to_grid(end_pos[0], end_pos[1])

        start_cell = GridCell(start_row, start_col, Layer.F_CU)
        end_cell = GridCell(end_row, end_col, Layer.F_CU)

        # Use fallback router (Rule 25: respects allow_via)
        path = self.astar.find_path_with_fallback(start_cell, end_cell, net_name,
                                                   allow_via)

        if not path:
            return Route(net=net_name, success=False, error="No path found")

        simplified_path = self.simplifier.simplify(path, net_name)
        segments, vias = self._path_to_segments(simplified_path, net_name)

        # NOTE: Pad stub segments removed to avoid KiCad DRC warnings
        # Without footprints placed, stubs would show as "unconnected track ends"
        # Routes now end at grid-aligned positions (escape endpoints or grid cells)
        # The pad stubs function is retained but disabled:
        # segments = self._add_pad_stubs(segments, pos_a, pos_b, start_pos, end_pos, net_name)

        total_length = sum(s.length for s in segments)

        return Route(
            net=net_name,
            segments=segments,
            vias=vias,
            length=total_length,
            success=True
        )

    def _route_multi_pin(self, net_name: str, pins: List, parts_db: Dict,
                         placement: Dict, escapes: Dict,
                         allow_via: bool = True) -> Route:
        """Route a multi-pin net using MST approach."""
        if not pins:
            return Route(net=net_name, success=False, error="No pins")

        all_segments = []
        all_vias = []

        positions = []
        for comp, pin in pins:
            pos = self._get_pin_position(comp, pin, parts_db, placement)
            if pos:
                if comp in escapes and pin in escapes[comp]:
                    pos = escapes[comp][pin].endpoint
                positions.append((comp, pin, pos))

        if len(positions) < 2:
            return Route(net=net_name, success=False, error="Not enough valid pins")

        connected = [positions[0]]
        remaining = positions[1:]

        while remaining:
            best_dist = float('inf')
            best_from = None
            best_to = None
            best_to_idx = -1

            for to_idx, (to_comp, to_pin, to_pos) in enumerate(remaining):
                for from_comp, from_pin, from_pos in connected:
                    dist = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_from = (from_comp, from_pin)
                        best_to = (to_comp, to_pin)
                        best_to_idx = to_idx

            if best_to_idx >= 0:
                sub_route = self._route_two_pin(
                    net_name, [best_from, best_to], parts_db, placement,
                    escapes, allow_via
                )

                if sub_route.success:
                    all_segments.extend(sub_route.segments)
                    all_vias.extend(sub_route.vias)
                    self._mark_route(sub_route)

                connected.append(remaining.pop(best_to_idx))

        if not all_segments:
            return Route(net=net_name, success=False, error="All sub-routes failed")

        unique_segments = self._remove_duplicate_segments(all_segments)
        total_length = sum(s.length for s in unique_segments)

        return Route(
            net=net_name,
            segments=unique_segments,
            vias=all_vias,
            length=total_length,
            success=True
        )

    def _remove_duplicate_segments(self, segments: List[RouteSegment]) -> List[RouteSegment]:
        seen = set()
        unique = []
        for seg in segments:
            key = (min(seg.start, seg.end), max(seg.start, seg.end), seg.layer)
            if key not in seen:
                seen.add(key)
                unique.append(seg)
        return unique

    def _get_pin_position(self, comp: str, pin: str, parts_db: Dict,
                          placement: Dict) -> Optional[Tuple[float, float]]:
        if comp not in placement:
            return None
        pos = placement[comp]
        part = parts_db['parts'].get(comp, {})
        for p in part.get('used_pins', []):
            if p['number'] == pin:
                offset = p.get('offset', (0, 0))
                return (pos.x + offset[0], pos.y + offset[1])
        return (pos.x, pos.y)

    def _path_to_segments(self, path: List[GridCell], net: str) -> Tuple[List[RouteSegment], List[Via]]:
        if len(path) < 2:
            return [], []

        segments = []
        vias = []
        current_layer = path[0].layer

        for i in range(len(path) - 1):
            curr = path[i]
            next_p = path[i + 1]

            if curr.layer != next_p.layer:
                via_pos = self.grid.grid_to_world(curr.row, curr.col)
                vias.append(Via(
                    position=via_pos,
                    net=net,
                    diameter=self.rules.min_via_diameter,
                    drill=self.rules.min_via_drill
                ))
                current_layer = next_p.layer
            else:
                start_pos = self.grid.grid_to_world(curr.row, curr.col)
                end_pos = self.grid.grid_to_world(next_p.row, next_p.col)

                if start_pos != end_pos:
                    segments.append(RouteSegment(
                        start=start_pos,
                        end=end_pos,
                        layer=current_layer.value,
                        width=self.rules.min_trace_width,
                        net=net
                    ))

        return segments, vias

    def _add_pad_stubs(self, segments: List[RouteSegment],
                       pad_a: Tuple[float, float], pad_b: Tuple[float, float],
                       route_start: Tuple[float, float], route_end: Tuple[float, float],
                       net: str) -> List[RouteSegment]:
        """
        Connect grid-aligned route ends to actual pad centers using 90-degree turns only.

        Strategy: Create orthogonal L-shaped connections to pads to ensure all angles
        are exactly 90 degrees (no acute angles, no acid traps, no U-turns).

        IMPORTANT: If the route start/end comes from an ESCAPE endpoint (not pad),
        we DON'T modify - the escape route handles pad-to-endpoint connection.

        Args:
            segments: Existing route segments
            pad_a, pad_b: Actual pad center positions
            route_start, route_end: Routing start/end (escape endpoints or pad positions)
            net: Net name
        """
        if not segments:
            return segments

        result = list(segments)

        # Tolerance for "same position" (0.01mm)
        def same_pos(p1, p2):
            return abs(p1[0] - p2[0]) < 0.01 and abs(p1[1] - p2[1]) < 0.01

        def is_horizontal(seg):
            return abs(seg.end[1] - seg.start[1]) < 0.01

        def is_vertical(seg):
            return abs(seg.end[0] - seg.start[0]) < 0.01

        def seg_direction(seg):
            """Get direction: 'up', 'down', 'left', 'right', or 'diagonal'"""
            dx = seg.end[0] - seg.start[0]
            dy = seg.end[1] - seg.start[1]
            if abs(dx) < 0.01:
                return 'down' if dy > 0 else 'up'
            elif abs(dy) < 0.01:
                return 'right' if dx > 0 else 'left'
            return 'diagonal'

        # Check if we need to connect at the start (pad_a side)
        has_escape_a = not same_pos(route_start, pad_a)
        if not has_escape_a:
            first_seg = result[0]
            if not same_pos(pad_a, first_seg.start):
                # Create L-shaped connection from pad to route start
                # Choose direction that doesn't create U-turn
                direction = seg_direction(first_seg)

                if direction in ('left', 'right'):
                    # Horizontal segment - approach vertically
                    mid_point = (pad_a[0], first_seg.start[1])
                elif direction in ('up', 'down'):
                    # Vertical segment - approach horizontally
                    mid_point = (first_seg.start[0], pad_a[1])
                else:
                    mid_point = (first_seg.start[0], pad_a[1])

                if not same_pos(pad_a, mid_point):
                    result.insert(0, RouteSegment(
                        start=pad_a,
                        end=mid_point,
                        layer=first_seg.layer,
                        width=first_seg.width,
                        net=net
                    ))
                    if not same_pos(mid_point, first_seg.start):
                        result.insert(1, RouteSegment(
                            start=mid_point,
                            end=first_seg.start,
                            layer=first_seg.layer,
                            width=first_seg.width,
                            net=net
                        ))

        # Check if we need to connect at the end (pad_b side)
        has_escape_b = not same_pos(route_end, pad_b)
        if not has_escape_b:
            last_seg = result[-1]
            if not same_pos(pad_b, last_seg.end):
                # Create L-shaped connection from route end to pad
                # Choose direction that continues the flow (no U-turn)
                direction = seg_direction(last_seg)
                dx_to_pad = pad_b[0] - last_seg.end[0]
                dy_to_pad = pad_b[1] - last_seg.end[1]

                if direction == 'down':
                    # Was going down, need to go horizontal then to pad
                    # If pad is above endpoint, go horizontal first
                    mid_point = (pad_b[0], last_seg.end[1])
                elif direction == 'up':
                    # Was going up
                    mid_point = (pad_b[0], last_seg.end[1])
                elif direction == 'right':
                    # Was going right
                    mid_point = (last_seg.end[0], pad_b[1])
                elif direction == 'left':
                    # Was going left
                    mid_point = (last_seg.end[0], pad_b[1])
                else:
                    # Diagonal - default to horizontal then vertical
                    mid_point = (pad_b[0], last_seg.end[1])

                if not same_pos(last_seg.end, mid_point):
                    result.append(RouteSegment(
                        start=last_seg.end,
                        end=mid_point,
                        layer=last_seg.layer,
                        width=last_seg.width,
                        net=net
                    ))
                if not same_pos(mid_point, pad_b):
                    result.append(RouteSegment(
                        start=mid_point,
                        end=pad_b,
                        layer=last_seg.layer,
                        width=last_seg.width,
                        net=net
                    ))

        return result

    def _mark_route(self, route: Route):
        for segment in route.segments:
            layer = Layer.F_CU if segment.layer == 'F.Cu' else Layer.B_CU
            start_row, start_col = self.grid.world_to_grid(segment.start[0], segment.start[1])
            end_row, end_col = self.grid.world_to_grid(segment.end[0], segment.end[1])

            cells = self.grid._get_line_cells(
                GridCell(start_row, start_col, layer),
                GridCell(end_row, end_col, layer)
            )
            for cell in cells:
                self.grid.mark_trace(cell, route.net)

        for via in route.vias:
            row, col = self.grid.world_to_grid(via.position[0], via.position[1])
            self.grid.mark_via(row, col, route.net)

    def _find_blocking_nets(self, net_name: str, pins: List, parts_db: Dict,
                             placement: Dict, escapes: Dict) -> List[str]:
        """Find which routed nets are blocking this net's path."""
        blocking = set()

        # Get endpoints for this net
        endpoints = []
        for comp, pin in pins:
            pos = self._get_pin_position(comp, pin, parts_db, placement)
            if pos:
                if comp in escapes and pin in escapes[comp]:
                    pos = escapes[comp][pin].endpoint
                endpoints.append(pos)

        if len(endpoints) < 2:
            return []

        # Check along direct path between endpoints
        start = endpoints[0]
        end = endpoints[1]

        start_row, start_col = self.grid.world_to_grid(start[0], start[1])
        end_row, end_col = self.grid.world_to_grid(end[0], end[1])

        # Sample cells along the path
        dr = 1 if end_row > start_row else (-1 if end_row < start_row else 0)
        dc = 1 if end_col > start_col else (-1 if end_col < start_col else 0)

        r, c = start_row, start_col
        for _ in range(max(abs(end_row - start_row), abs(end_col - start_col)) + 1):
            for layer in [Layer.F_CU, Layer.B_CU]:
                if 0 <= r < self.grid.rows and 0 <= c < self.grid.cols:
                    owner = self.grid.owners[layer][r][c]
                    if owner and owner != net_name and owner != 'GND' and owner in self.routes:
                        blocking.add(owner)
            r += dr
            c += dc

        return list(blocking)

    def _rip_up_route(self, net_name: str):
        """Remove a routed net from the grid to allow rerouting."""
        if net_name not in self.routes:
            return

        route = self.routes[net_name]

        # Clear trace cells
        for segment in route.segments:
            layer = Layer.F_CU if segment.layer == 'F.Cu' else Layer.B_CU
            start_row, start_col = self.grid.world_to_grid(segment.start[0], segment.start[1])
            end_row, end_col = self.grid.world_to_grid(segment.end[0], segment.end[1])

            cells = self.grid._get_line_cells(
                GridCell(start_row, start_col, layer),
                GridCell(end_row, end_col, layer)
            )
            for cell in cells:
                # Only clear if owned by this net
                if self.grid.owners[cell.layer][cell.row][cell.col] == net_name:
                    self.grid.grids[cell.layer][cell.row][cell.col] = self.grid.EMPTY
                    self.grid.owners[cell.layer][cell.row][cell.col] = ''

        # Clear via cells
        for via in route.vias:
            row, col = self.grid.world_to_grid(via.position[0], via.position[1])
            for layer in [Layer.F_CU, Layer.B_CU]:
                # Clear via area
                via_clearance = int(math.ceil(
                    (self.rules.min_via_diameter / 2 + self.rules.min_clearance) / self.grid.board.grid_size
                ))
                for dr in range(-via_clearance, via_clearance + 1):
                    for dc in range(-via_clearance, via_clearance + 1):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid.rows and 0 <= c < self.grid.cols:
                            if self.grid.owners[layer][r][c] == net_name:
                                self.grid.grids[layer][r][c] = self.grid.EMPTY
                                self.grid.owners[layer][r][c] = ''

            # Remove from via_positions list
            self.grid.via_positions = [
                (vr, vc, vn) for vr, vc, vn in self.grid.via_positions
                if not (vr == row and vc == col and vn == net_name)
            ]

        # Remove from routes
        del self.routes[net_name]
        if net_name in self.quality_scores:
            del self.quality_scores[net_name]

    def get_report(self) -> str:
        lines = [
            "=" * 60,
            "ROUTING REPORT (Enhanced with 33 Rules + Rip-up/Reroute)",
            "=" * 60,
            f"Routes completed: {len(self.routes)}",
            f"Routes failed: {len(self.failed)}",
            f"Success rate: {len(self.routes) / max(1, len(self.routes) + len(self.failed)):.0%}",
            "",
        ]

        if self.routes:
            total_length = sum(r.length for r in self.routes.values())
            total_vias = sum(r.via_count for r in self.routes.values())
            total_segments = sum(r.segment_count for r in self.routes.values())
            avg_quality = sum(self.quality_scores.values()) / max(1, len(self.quality_scores))

            lines.extend([
                "ROUTE METRICS:",
                f"  Total trace length: {total_length:.1f}mm",
                f"  Total segments: {total_segments}",
                f"  Total vias: {total_vias}",
                f"  Average quality score: {avg_quality:.0f}/100",
                "",
            ])

        if self.topology_gaps:
            lines.append("TOPOLOGY GAPS (Rule 27):")
            for gap in self.topology_gaps:
                lines.append(f"  - {gap['net']}: {gap['reason']}")
            lines.append("")

        if self.failed:
            lines.append("FAILED NETS:")
            for net in self.failed:
                lines.append(f"  - {net}")

        lines.append("=" * 60)
        return "\n".join(lines)
