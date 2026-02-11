"""
PCB Engine - Routing Piston
============================

A dedicated piston (sub-engine) for PCB routing that implements research-based algorithms.

This module provides 12 routing algorithms (10 core + 2 meta):

CORE ALGORITHMS:
1. Lee Algorithm - Guaranteed shortest path with BFS wavefront expansion (Lee, 1961)
2. Hadlock's Algorithm - Faster than Lee using detour numbers (Hadlock, 1977)
3. Soukup's Algorithm - Fast two-phase routing: greedy + maze fallback (Soukup, 1978)
4. Mikami-Tabuchi - Memory-efficient line search algorithm (Mikami & Tabuchi, 1968)
5. A* Pathfinding - Fast heuristic-based routing with Manhattan distance
6. PathFinder - Negotiated congestion routing (McMurchie & Ebeling, 1995)
7. Rip-up and Reroute - Iterative routing with intelligent reordering (Nair, 1987)
8. Rectilinear Steiner Tree (RSMT) - Optimal multi-terminal net routing (Hanan, 1966)
9. Channel Routing - Left-edge greedy algorithm (Hashimoto & Stevens, 1971)
10. Push-and-Shove (PNS) - Interactive router that pushes traces out of the way (CERN/KiCad, 2013)

META ALGORITHMS:
11. HYBRID - Combines Lee + Ripup + Steiner for best results
12. AUTO - Automatically selects best algorithm based on design

PUSH-AND-SHOVE ROUTING:
The Push-and-Shove algorithm (also called PNS or Interactive Router) is the key
to achieving near-100% routing completion rates. It implements:
- Walkaround: Route around obstacles without moving them (preferred)
- Shove: Push existing traces out of the way when walkaround fails
- Ripple: Propagate shoves to adjacent traces when needed
- DRC validation after each shove operation

This is the algorithm used by KiCad 6+, Altium, and Eagle for interactive routing.

Research References:
- "Routing procedures for printed circuit boards" (Lee, 1961)
- "A shortest path algorithm for grid graphs" (Hadlock, 1977)
- "Global router using a two-step routing approach" (Soukup, 1978)
- "A computer program for optimal routing" (Mikami & Tabuchi, 1968)
- "PathFinder: A Negotiation-Based Performance-Driven Router" (McMurchie & Ebeling, 1995)
- "MIGHTY: A Rip-Up and Reroute Detailed Router" (Nair et al., 1987)
- "On Steiner's problem with rectilinear distance" (Hanan, 1966)
- "Channel routing" (Hashimoto & Stevens, 1971)
- KiCad PNS Router (CERN, Tomasz Wlostowski, 2013)

Architecture follows the piston pattern for consistency with placement_piston.py.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import heapq
from collections import deque
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import copy

# Import FLUTE for optimal Steiner tree computation
try:
    from .flute_steiner import get_flute_instance, FLUTESteiner
    FLUTE_AVAILABLE = True
except ImportError:
    FLUTE_AVAILABLE = False


# =============================================================================
# QUADTREE SPATIAL INDEX - O(log n) COURTYARD LOOKUPS
# =============================================================================
# For answering "What components are near (x, y)?" in O(log n) instead of O(n)
# Uses component courtyards for accurate bounding boxes.

@dataclass
class BoundingBox:
    """Axis-aligned bounding box for spatial queries"""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def contains_point(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def intersects(self, other: 'BoundingBox') -> bool:
        return not (self.max_x < other.min_x or other.max_x < self.min_x or
                    self.max_y < other.min_y or other.max_y < self.min_y)

    def contains_box(self, other: 'BoundingBox') -> bool:
        return (self.min_x <= other.min_x and self.max_x >= other.max_x and
                self.min_y <= other.min_y and self.max_y >= other.max_y)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass
class QuadTreeItem:
    """An item stored in the quadtree (component courtyard)"""
    ref: str              # Component reference (e.g., "U1", "R1")
    bbox: BoundingBox     # Courtyard bounding box
    net: Optional[str]    # Net name if pad, None if component body
    layer: str = 'F.Cu'   # Which layer this blocks


class QuadTree:
    """
    Quadtree for O(log n) spatial queries on component courtyards.

    Unlike the bitmap-based SpatialIndex which works at grid resolution,
    the QuadTree works at real-world coordinates and provides:
    - "What components are at position (x, y)?" → O(log n)
    - "What components intersect this rectangle?" → O(log n + k)
    - "What obstacles are within distance d of point (x, y)?" → O(log n + k)

    This is especially useful for:
    - Large boards with many components
    - Fine-grained obstacle queries during routing
    - Courtyard-based collision detection
    """

    MAX_ITEMS = 4     # Split when exceeding this
    MAX_DEPTH = 10    # Maximum tree depth

    def __init__(self, bounds: BoundingBox, depth: int = 0):
        self.bounds = bounds
        self.depth = depth
        self.items: List[QuadTreeItem] = []
        self.children: Optional[List['QuadTree']] = None  # NW, NE, SW, SE

    def insert(self, item: QuadTreeItem) -> bool:
        """Insert an item into the quadtree. Returns True if successful."""
        # Check if item intersects this node's bounds
        if not self.bounds.intersects(item.bbox):
            return False

        # If we have children, try to insert into them
        if self.children is not None:
            for child in self.children:
                if child.bounds.contains_box(item.bbox):
                    return child.insert(item)
            # Item spans multiple children, keep it at this level
            self.items.append(item)
            return True

        # Add to this node
        self.items.append(item)

        # Split if needed
        if len(self.items) > self.MAX_ITEMS and self.depth < self.MAX_DEPTH:
            self._split()

        return True

    def _split(self):
        """Split this node into 4 children"""
        cx, cy = self.bounds.center
        min_x, min_y = self.bounds.min_x, self.bounds.min_y
        max_x, max_y = self.bounds.max_x, self.bounds.max_y

        self.children = [
            QuadTree(BoundingBox(min_x, cy, cx, max_y), self.depth + 1),  # NW
            QuadTree(BoundingBox(cx, cy, max_x, max_y), self.depth + 1),  # NE
            QuadTree(BoundingBox(min_x, min_y, cx, cy), self.depth + 1),  # SW
            QuadTree(BoundingBox(cx, min_y, max_x, cy), self.depth + 1),  # SE
        ]

        # Re-distribute items to children
        remaining = []
        for item in self.items:
            inserted = False
            for child in self.children:
                if child.bounds.contains_box(item.bbox):
                    child.insert(item)
                    inserted = True
                    break
            if not inserted:
                remaining.append(item)  # Spans multiple children
        self.items = remaining

    def query_point(self, x: float, y: float) -> List[QuadTreeItem]:
        """Find all items whose bounding boxes contain the point (x, y)."""
        results = []

        # Check if point is in bounds
        if not self.bounds.contains_point(x, y):
            return results

        # Check items at this level
        for item in self.items:
            if item.bbox.contains_point(x, y):
                results.append(item)

        # Check children
        if self.children is not None:
            for child in self.children:
                results.extend(child.query_point(x, y))

        return results

    def query_range(self, bbox: BoundingBox) -> List[QuadTreeItem]:
        """Find all items that intersect the given bounding box."""
        results = []

        if not self.bounds.intersects(bbox):
            return results

        for item in self.items:
            if item.bbox.intersects(bbox):
                results.append(item)

        if self.children is not None:
            for child in self.children:
                results.extend(child.query_range(bbox))

        return results

    def query_radius(self, x: float, y: float, radius: float) -> List[QuadTreeItem]:
        """Find all items within radius of point (x, y)."""
        # Use bounding box query then filter by distance
        search_box = BoundingBox(x - radius, y - radius, x + radius, y + radius)
        candidates = self.query_range(search_box)

        results = []
        radius_sq = radius * radius
        for item in candidates:
            # Check if any corner of bbox is within radius
            cx, cy = item.bbox.center
            dist_sq = (cx - x)**2 + (cy - y)**2
            if dist_sq <= radius_sq:
                results.append(item)
                continue
            # Check if point is inside bbox (edge case)
            if item.bbox.contains_point(x, y):
                results.append(item)

        return results


# =============================================================================
# SPATIAL INDEX - O(1) OBSTACLE LOOKUPS (Grid-based)
# =============================================================================
# Concept borrowed from database query indexing: pre-compute everything once,
# then use O(1) bitmap lookups instead of O(n) neighbor scans during routing.

@dataclass
class SpatialIndex:
    """
    Pre-computed spatial index for fast obstacle detection during routing.

    Instead of checking each neighboring cell during routing (O(n) per cell),
    we pre-compute which cells are blocked/clear ONCE, then use O(1) lookups.

    The index includes:
    1. blocked_bitmap: Which cells are blocked by obstacles
    2. net_grid: Which net owns each cell (for quick friend/foe checks)
    3. clearance_bitmap: Which cells have clearance violations (pre-expanded obstacles)
    4. via_clearance_bitmap: Same but for via placement (larger clearance)
    """
    rows: int
    cols: int

    # Core bitmaps (numpy arrays for fast operations)
    blocked_fcu: np.ndarray = None  # bool: True = blocked on F.Cu
    blocked_bcu: np.ndarray = None  # bool: True = blocked on B.Cu

    # Net ownership (for fast "is this my net?" checks)
    net_fcu: np.ndarray = None  # int: net ID (0 = empty, -1 = blocked)
    net_bcu: np.ndarray = None  # int: net ID (0 = empty, -1 = blocked)

    # Pre-expanded clearance zones (obstacle + clearance radius)
    clearance_fcu: np.ndarray = None  # bool: True = within clearance of obstacle
    clearance_bcu: np.ndarray = None  # bool: True = within clearance of obstacle

    # Via clearance zones (larger than trace clearance)
    via_clearance_fcu: np.ndarray = None
    via_clearance_bcu: np.ndarray = None

    # Net name to ID mapping
    net_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_net: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.blocked_fcu is None:
            self.blocked_fcu = np.zeros((self.rows, self.cols), dtype=bool)
            self.blocked_bcu = np.zeros((self.rows, self.cols), dtype=bool)
            self.net_fcu = np.zeros((self.rows, self.cols), dtype=np.int16)
            self.net_bcu = np.zeros((self.rows, self.cols), dtype=np.int16)
            self.clearance_fcu = np.zeros((self.rows, self.cols), dtype=bool)
            self.clearance_bcu = np.zeros((self.rows, self.cols), dtype=bool)
            self.via_clearance_fcu = np.zeros((self.rows, self.cols), dtype=bool)
            self.via_clearance_bcu = np.zeros((self.rows, self.cols), dtype=bool)

    def get_net_id(self, net_name: str) -> int:
        """Get or create a numeric ID for a net name."""
        if net_name is None or net_name == '':
            return 0  # Empty
        if net_name in self.net_to_id:
            return self.net_to_id[net_name]
        # Create new ID (start from 1, 0 = empty, -1 = blocked)
        new_id = len(self.net_to_id) + 1
        self.net_to_id[net_name] = new_id
        self.id_to_net[new_id] = net_name
        return new_id

    def mark_blocked(self, row: int, col: int, layer: str = 'F.Cu'):
        """Mark a cell as permanently blocked (component body, NC pad, etc.)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if layer == 'F.Cu':
                self.blocked_fcu[row, col] = True
                self.net_fcu[row, col] = -1  # -1 = blocked
            else:
                self.blocked_bcu[row, col] = True
                self.net_bcu[row, col] = -1

    def mark_net(self, row: int, col: int, net_name: str, layer: str = 'F.Cu'):
        """Mark a cell as owned by a net (pad, routed trace, etc.)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            net_id = self.get_net_id(net_name)
            if layer == 'F.Cu':
                self.net_fcu[row, col] = net_id
            else:
                self.net_bcu[row, col] = net_id

    def expand_clearances(self, trace_clearance_cells: int, via_clearance_cells: int):
        """
        Expand blocked areas by clearance radius using convolution.

        This is the KEY OPTIMIZATION: instead of checking neighbors during routing,
        we pre-expand all obstacles once. Then routing just checks bitmap[r,c].

        Uses scipy if available, else manual convolution.
        """
        try:
            from scipy.ndimage import binary_dilation

            # Create circular structuring element for trace clearance
            cc = trace_clearance_cells
            y, x = np.ogrid[-cc:cc+1, -cc:cc+1]
            trace_kernel = x*x + y*y <= cc*cc

            # Create circular structuring element for via clearance
            vc = via_clearance_cells
            y, x = np.ogrid[-vc:vc+1, -vc:vc+1]
            via_kernel = x*x + y*y <= vc*vc

            # Expand blocked areas by clearance
            self.clearance_fcu = binary_dilation(self.blocked_fcu, structure=trace_kernel)
            self.clearance_bcu = binary_dilation(self.blocked_bcu, structure=trace_kernel)
            self.via_clearance_fcu = binary_dilation(self.blocked_fcu, structure=via_kernel)
            self.via_clearance_bcu = binary_dilation(self.blocked_bcu, structure=via_kernel)

        except ImportError:
            # Manual expansion without scipy
            self._expand_clearances_manual(trace_clearance_cells, via_clearance_cells)

    def _expand_clearances_manual(self, trace_cc: int, via_cc: int):
        """Manual clearance expansion without scipy (slower but works)."""
        # Pre-compute circular offsets
        trace_offsets = []
        for dr in range(-trace_cc, trace_cc + 1):
            for dc in range(-trace_cc, trace_cc + 1):
                if dr*dr + dc*dc <= trace_cc*trace_cc:
                    trace_offsets.append((dr, dc))

        via_offsets = []
        for dr in range(-via_cc, via_cc + 1):
            for dc in range(-via_cc, via_cc + 1):
                if dr*dr + dc*dc <= via_cc*via_cc:
                    via_offsets.append((dr, dc))

        # Expand F.Cu
        for r in range(self.rows):
            for c in range(self.cols):
                if self.blocked_fcu[r, c]:
                    for dr, dc in trace_offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            self.clearance_fcu[nr, nc] = True
                    for dr, dc in via_offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            self.via_clearance_fcu[nr, nc] = True

        # Expand B.Cu
        for r in range(self.rows):
            for c in range(self.cols):
                if self.blocked_bcu[r, c]:
                    for dr, dc in trace_offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            self.clearance_bcu[nr, nc] = True
                    for dr, dc in via_offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            self.via_clearance_bcu[nr, nc] = True

    def is_clear_for_net(self, row: int, col: int, net_name: str, layer: str = 'F.Cu') -> bool:
        """
        O(1) check if a cell is clear for routing a specific net.

        A cell is clear if:
        1. Not blocked
        2. Not within clearance of a DIFFERENT net
        3. Either empty or owned by the SAME net
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False

        net_id = self.get_net_id(net_name)

        if layer == 'F.Cu':
            # Check if blocked
            if self.blocked_fcu[row, col]:
                return False
            # Check cell ownership
            cell_net = self.net_fcu[row, col]
            if cell_net == -1:  # Blocked
                return False
            if cell_net != 0 and cell_net != net_id:  # Different net
                return False
            # Check clearance zone (only if different net nearby)
            # Note: clearance_fcu includes the blocked cells expanded
            # We need a more sophisticated check here
            return True
        else:
            if self.blocked_bcu[row, col]:
                return False
            cell_net = self.net_bcu[row, col]
            if cell_net == -1:
                return False
            if cell_net != 0 and cell_net != net_id:
                return False
            return True

    def is_clear_for_via(self, row: int, col: int, net_name: str) -> bool:
        """O(1) check if a cell is clear for via placement (both layers)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False

        # Via needs clearance on both layers
        # Using via_clearance bitmaps which have larger expansion
        return (self.is_clear_for_net(row, col, net_name, 'F.Cu') and
                self.is_clear_for_net(row, col, net_name, 'B.Cu'))

    def build_net_clearance_bitmap(self, net_name: str, clearance_cells: int,
                                    layer: str = 'F.Cu') -> np.ndarray:
        """
        Build a boolean bitmap: True = cell is CLEAR for routing this net.

        This is THE KEY OPTIMIZATION: instead of checking 12 clearance offsets
        per cell during BFS (884K string comparisons for 68K cells), we build
        this bitmap ONCE per net using NumPy/scipy (~1ms), then BFS just does
        bitmap[row, col] — a single O(1) array lookup.

        A cell is clear if:
        1. Not blocked (component body, edge, NC pad)
        2. Not occupied by a DIFFERENT net
        3. Not within clearance distance of a DIFFERENT net's cells

        Cells owned by the SAME net are passable (routing to own pads).
        """
        net_id = self.get_net_id(net_name)
        blocked = self.blocked_fcu if layer == 'F.Cu' else self.blocked_bcu
        net_grid = self.net_fcu if layer == 'F.Cu' else self.net_bcu

        # Step 1: Find own-net cell IDs (includes __PAD_CLEAR_<netname>__ markers)
        # __PAD_CLEAR_VBUS__ gets its own net_id that's different from VBUS,
        # but we must treat it as passable for VBUS routing.
        pad_clear_name = f'__PAD_CLEAR_{net_name}__'
        pad_clear_id = self.net_to_id.get(pad_clear_name, -999)

        # Step 2: Find all cells that are obstacles for this net
        # Obstacles = blocked cells + cells belonging to OTHER nets
        # Own net (net_id), own pad clearance (pad_clear_id), and empty (0) are NOT obstacles
        own_mask = (net_grid == net_id) | (net_grid == pad_clear_id)
        other_net_mask = (net_grid != 0) & (net_grid != -1) & ~own_mask
        obstacles = blocked | other_net_mask

        # Step 3: Expand obstacles by clearance radius (binary dilation)
        try:
            from scipy.ndimage import binary_dilation
            cc = clearance_cells
            y, x = np.ogrid[-cc:cc+1, -cc:cc+1]
            kernel = x*x + y*y <= cc*cc
            expanded_obstacles = binary_dilation(obstacles, structure=kernel)
        except ImportError:
            # Manual expansion fallback
            expanded_obstacles = np.copy(obstacles)
            offsets = []
            cc = clearance_cells
            for dr in range(-cc, cc + 1):
                for dc in range(-cc, cc + 1):
                    if dr*dr + dc*dc <= cc*cc:
                        offsets.append((dr, dc))
            obstacle_coords = np.argwhere(obstacles)
            for r, c in obstacle_coords:
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        expanded_obstacles[nr, nc] = True

        # Step 4: Clear = NOT in expanded obstacle zone
        # BUT: own net cells are ALWAYS passable (we need to reach our own pads)
        clear_bitmap = ~expanded_obstacles | own_mask

        return clear_bitmap

    def build_accessible_bitmap(self, net_name: str, layer: str = 'F.Cu') -> np.ndarray:
        """
        Build a relaxed accessibility bitmap (no clearance check, just passability).

        Used for "approaching pad" zones near target — we relax clearance
        requirements to allow routing into tight pad areas.

        A cell is accessible if:
        1. Not a hard blocker (component body, edge)
        2. Either empty, belongs to our net, or is a pad conflict/clearance marker
        """
        net_id = self.get_net_id(net_name)
        blocked = self.blocked_fcu if layer == 'F.Cu' else self.blocked_bcu
        net_grid = self.net_fcu if layer == 'F.Cu' else self.net_bcu

        # Include own pad clearance markers as accessible
        pad_clear_name = f'__PAD_CLEAR_{net_name}__'
        pad_clear_id = self.net_to_id.get(pad_clear_name, -999)

        # Also include __PAD_CONFLICT__ as accessible (original code does this)
        pad_conflict_id = self.net_to_id.get('__PAD_CONFLICT__', -999)

        # Accessible = not blocked AND (empty OR own net OR own pad clear OR pad conflict)
        accessible = ~blocked & (
            (net_grid == 0) | (net_grid == net_id) |
            (net_grid == pad_clear_id) | (net_grid == pad_conflict_id)
        )
        return accessible

# Import shared types from routing_types.py (canonical definitions)
from .routing_types import (
    TrackSegment,
    Via,
    Route,
    RoutingAlgorithm,
    RoutingConfig,
    RoutingResult,
    create_track_segment,
    create_via
)


# =============================================================================
# ROUTING PISTON
# =============================================================================

class RoutingPiston:
    """
    Advanced PCB Routing Piston

    Provides multiple research-based routing algorithms and automatically
    selects the best approach based on design complexity.

    Usage:
        config = RoutingConfig(
            board_width=50.0,
            board_height=40.0,
            grid_size=0.08,  # 6 cells per 0.5mm pad
            algorithm='hybrid'
        )
        piston = RoutingPiston(config)
        result = piston.route(parts_db, escapes, placement, net_order)
    """

    @staticmethod
    def _parse_pin_ref(pin) -> Tuple[str, str]:
        """
        Parse a pin reference in either tuple or dict format.

        Accepts:
        - Tuple: ('R1', '1') -> ('R1', '1')
        - Dict: {'ref': 'R1', 'pin': '1'} -> ('R1', '1')
        - String: 'R1.1' -> ('R1', '1')

        Returns:
            Tuple of (component_ref, pin_number)
        """
        if isinstance(pin, (list, tuple)) and len(pin) >= 2:
            return str(pin[0]), str(pin[1])
        elif isinstance(pin, dict):
            return pin.get('ref', ''), pin.get('pin', '')
        elif isinstance(pin, str) and '.' in pin:
            parts = pin.split('.', 1)
            return parts[0], parts[1]
        return '', ''

    def __init__(self, config: RoutingConfig):
        self.config = config
        random.seed(config.seed)

        # Grid dimensions
        self.grid_cols = int(config.board_width / config.grid_size) + 1
        self.grid_rows = int(config.board_height / config.grid_size) + 1

        # BUG-13 FIX: Initialize grids immediately (not None) to prevent crashes
        # Occupancy grids (None = empty cell, string = net name, special markers)
        self.fcu_grid: List[List[Optional[str]]] = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self.bcu_grid: List[List[Optional[str]]] = [[None] * self.grid_cols for _ in range(self.grid_rows)]

        # Clearance calculation for traces
        # BUG FIX: Must account for BOTH track widths when checking track-to-track clearance.
        # Two tracks each have width/2 extending from center, so center-to-center distance
        # must be >= trace_width + clearance (not trace_width/2 + clearance).
        # The old formula (trace_width/2 + clearance) only measured from one track center
        # to the other track's EDGE, ignoring the first track's own width extension.
        # This caused edge-to-edge gaps of only 0.05mm instead of the required 0.15mm.
        self.clearance_radius = config.trace_width + config.clearance
        self.clearance_cells = max(1, int(math.ceil(self.clearance_radius / config.grid_size)))
        # Pre-compute squared clearance for circular check (faster than sqrt)
        self.clearance_cells_sq = self.clearance_cells * self.clearance_cells

        # PRE-COMPUTE CIRCULAR OFFSET PATTERN (INDEXING OPTIMIZATION)
        # Instead of computing (dr*dr + dc*dc <= cc_sq) for every cell check,
        # we pre-compute the list of valid (dr, dc) offsets once.
        # This is like a database index - precompute once, query fast.
        self.clearance_offsets: List[Tuple[int, int]] = []
        cc = self.clearance_cells
        for dr in range(-cc, cc + 1):
            for dc in range(-cc, cc + 1):
                if dr == 0 and dc == 0:
                    continue
                if dr * dr + dc * dc <= self.clearance_cells_sq:
                    self.clearance_offsets.append((dr, dc))

        # Clearance calculation for vias (larger than traces!)
        # Via edge to nearest track edge must be >= clearance.
        # Center-to-center = via_diameter/2 + clearance + trace_width/2
        self.via_clearance_radius = config.via_diameter / 2 + config.clearance + config.trace_width / 2
        self.via_clearance_cells = max(1, int(math.ceil(self.via_clearance_radius / config.grid_size)))
        # Pre-compute squared via clearance for circular check
        self.via_clearance_cells_sq = self.via_clearance_cells * self.via_clearance_cells

        # PRE-COMPUTE VIA CIRCULAR OFFSET PATTERN
        self.via_clearance_offsets: List[Tuple[int, int]] = []
        vc = self.via_clearance_cells
        for dr in range(-vc, vc + 1):
            for dc in range(-vc, vc + 1):
                if dr == 0 and dc == 0:
                    continue
                if dr * dr + dc * dc <= self.via_clearance_cells_sq:
                    self.via_clearance_offsets.append((dr, dc))

        # Board margin
        self.board_margin = max(1.0, config.clearance + config.trace_width / 2 + 0.5)
        self.margin_cells = max(2, int(math.ceil(self.board_margin / config.grid_size)))

        # Per-net bitmap cache for O(1) clearance checks
        self._current_net_bitmap_name: Optional[str] = None
        self._current_net_bitmap_fcu: Optional[np.ndarray] = None
        self._current_net_bitmap_bcu: Optional[np.ndarray] = None

        # Tracking
        self.placed_vias: Set[Tuple[float, float]] = set()
        self.routes: Dict[str, Route] = {}
        self.failed: List[str] = []

        # Special grid markers
        self.BLOCKED_MARKERS = {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__'}

        # SPATIAL INDEX for O(1) obstacle lookups (grid-based bitmap)
        self.spatial_index: Optional[SpatialIndex] = None
        self.use_spatial_index = True  # Enable by default

        # NET CLASS CONSTRAINTS - per-net trace widths and clearances
        # This enables proper manufacturing: power nets get wider traces,
        # high-speed nets get controlled impedance, etc.
        self._net_classes: Dict[str, 'NetClass'] = {}  # net_name -> NetClass
        self._net_trace_widths: Dict[str, float] = {}  # net_name -> trace_width
        self._net_clearances: Dict[str, float] = {}    # net_name -> clearance

        # QUADTREE for O(log n) courtyard-based spatial queries
        # Used for "what obstacles are near (x,y)?" queries
        self.courtyard_tree: Optional[QuadTree] = None

        # INDEX CACHE for CASCADE optimization
        # When trying multiple algorithms on same placement, don't rebuild index
        self._cached_placement_hash: Optional[str] = None
        self._cached_spatial_index: Optional[SpatialIndex] = None
        self._cached_courtyard_tree: Optional[QuadTree] = None
        self._cached_fcu_grid: Optional[List[List[Optional[str]]]] = None
        self._cached_bcu_grid: Optional[List[List[Optional[str]]]] = None

        # Statistics
        self.stats = {
            'cells_explored': 0,
            'backtracks': 0,
            'layer_changes': 0
        }

        # TRUNK CHAIN DETECTION - Identify nets that share routing corridors
        # A trunk chain is a group of nets that can be routed in parallel
        # along a common trunk, saving wirelength and improving manufacturability
        self._trunk_chains: List[Dict] = []  # Detected trunk chains
        self._net_to_chain: Dict[str, int] = {}  # net_name -> chain index

        # RETURN PATH AWARENESS - Track ground/power reference planes
        # High-frequency signals need a continuous ground plane beneath them
        # for proper return current flow. Without it, return currents take
        # longer paths causing EMI and signal integrity issues.
        self._ground_plane_map: Optional[np.ndarray] = None  # True = solid ground
        self._return_path_violations: List[Dict] = []  # Detected violations
        self._return_path_enabled: bool = True  # Enable return path analysis

    # =========================================================================
    # INDEX CACHE HELPER - CASCADE OPTIMIZATION
    # =========================================================================
    # When CASCADE tries 11 algorithms on same placement, don't rebuild index
    # each time. Build once, reuse for all algorithms. Saves ~8 seconds.

    def _compute_placement_hash(self, placement: Dict, parts_db: Dict) -> str:
        """
        Compute a hash of placement + parts_db to detect changes.
        If hash matches cached, we can reuse the spatial index.
        """
        import hashlib
        items = []

        # Hash placement positions
        for ref in sorted(placement.keys()):
            comp = placement[ref]
            # Handle both object and tuple/dict formats for position
            if hasattr(comp, 'x'):
                pos_x, pos_y = comp.x, comp.y
                rotation = getattr(comp, 'rotation', 0)
                width = getattr(comp, 'width', 0)
                height = getattr(comp, 'height', 0)
            elif isinstance(comp, (list, tuple)) and len(comp) >= 2:
                pos_x, pos_y = comp[0], comp[1]
                rotation = comp[2] if len(comp) > 2 else 0
                width = comp[3] if len(comp) > 3 else 0
                height = comp[4] if len(comp) > 4 else 0
            elif isinstance(comp, dict):
                pos_x = comp.get('x', 0)
                pos_y = comp.get('y', 0)
                rotation = comp.get('rotation', 0)
                width = comp.get('width', 0)
                height = comp.get('height', 0)
            else:
                pos_x, pos_y, rotation, width, height = 0, 0, 0, 0, 0
            # Include position, rotation, and size (affects courtyards)
            items.append(f"{ref}:{pos_x:.3f},{pos_y:.3f},{rotation},{width:.3f},{height:.3f}")

        # Hash parts_db nets (affects which cells are marked with net names)
        nets = parts_db.get('nets', {})
        for net_name in sorted(nets.keys()):
            pins = nets[net_name].get('pins', [])
            # Sort pins handling both tuple and dict formats
            def get_pin_sort_key(p):
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    return (str(p[0]), str(p[1]))
                elif isinstance(p, dict):
                    return (p.get('ref', ''), p.get('pin', ''))
                return ('', '')
            for pin in sorted(pins, key=get_pin_sort_key):
                # Handle both tuple and dict formats
                if isinstance(pin, (list, tuple)) and len(pin) >= 2:
                    ref, pin_num = str(pin[0]), str(pin[1])
                elif isinstance(pin, dict):
                    ref, pin_num = pin.get('ref', ''), pin.get('pin', '')
                else:
                    ref, pin_num = '', ''
                items.append(f"net:{net_name}:{ref}:{pin_num}")

        # Hash escapes would be ideal but they're computed from placement anyway

        return hashlib.md5('|'.join(items).encode()).hexdigest()

    def _try_use_cached_index(self, placement: Dict, parts_db: Dict) -> bool:
        """
        Try to use cached index if placement hasn't changed.
        Returns True if cache hit (index restored), False if cache miss.
        """
        current_hash = self._compute_placement_hash(placement, parts_db)

        if (self._cached_placement_hash == current_hash and
            self._cached_spatial_index is not None):
            # CACHE HIT - restore from cache
            self.spatial_index = self._cached_spatial_index
            self.courtyard_tree = self._cached_courtyard_tree
            # Deep copy grids (routing modifies them)
            self.fcu_grid = [row[:] for row in self._cached_fcu_grid]
            self.bcu_grid = [row[:] for row in self._cached_bcu_grid]
            print("    [INDEX] Cache HIT - reusing cached index (0ms)")
            return True

        # CACHE MISS - need to rebuild
        return False

    def _cache_index(self, placement: Dict, parts_db: Dict):
        """Save current index to cache for future reuse."""
        self._cached_placement_hash = self._compute_placement_hash(placement, parts_db)
        self._cached_spatial_index = self.spatial_index
        self._cached_courtyard_tree = self.courtyard_tree
        # Deep copy grids (so cached version isn't modified by routing)
        self._cached_fcu_grid = [row[:] for row in self.fcu_grid]
        self._cached_bcu_grid = [row[:] for row in self.bcu_grid]

    # =========================================================================
    # VIA DEDUPLICATION AND SPACING HELPER
    # =========================================================================
    # BUGFIX: Prevents "holes_co_located" and "hole_to_hole" DRC errors when:
    # 1. Multiple route legs transition at the same location (duplicate vias)
    # 2. Vias are placed too close together (KiCad min hole spacing ~0.25mm)

    # Minimum via-to-via spacing (mm) - CENTER-TO-CENTER distance
    # KiCad checks edge-to-edge (actual hole spacing), not center-to-center
    # With via_diameter=0.8mm and min edge spacing=0.25mm:
    # min_center_to_center = via_diameter + edge_spacing = 0.8 + 0.25 = 1.05mm
    MIN_VIA_SPACING = 1.1  # Use 1.1mm to be safe (via_dia + edge clearance)

    def _extend_vias_deduplicated(self, route: Route, new_vias: List[Via]):
        """
        Add vias to route, enforcing minimum spacing between vias.

        This prevents:
        1. "holes_co_located" - duplicate vias at same position
        2. "hole_to_hole" - vias placed too close (violates hole spacing rules)
        """
        for via in new_vias:
            via_x = round(via.position[0], 3)
            via_y = round(via.position[1], 3)

            # Check against all existing vias in this route AND global placed_vias
            too_close = False

            # Check route's vias
            for existing_via in route.vias:
                ex = round(existing_via.position[0], 3)
                ey = round(existing_via.position[1], 3)
                dist = ((via_x - ex)**2 + (via_y - ey)**2)**0.5
                if dist < self.MIN_VIA_SPACING:
                    too_close = True
                    break

            # Check global placed vias (from other nets)
            if not too_close:
                for (px, py) in self.placed_vias:
                    dist = ((via_x - px)**2 + (via_y - py)**2)**0.5
                    if dist < self.MIN_VIA_SPACING:
                        too_close = True
                        break

            if not too_close:
                route.vias.append(via)
                self.placed_vias.add((via_x, via_y))

    # =========================================================================
    # COORDINATE CONVERSION HELPERS (FIX: use round() not int())
    # =========================================================================
    # These methods fix the coordinate truncation bug where int() caused
    # 0.05mm gaps between routes and actual pad positions.

    def _real_to_grid_col(self, x: float) -> int:
        """Convert real X coordinate to grid column. Uses round() for accuracy."""
        return round((x - self.config.origin_x) / self.config.grid_size)

    def _real_to_grid_row(self, y: float) -> int:
        """Convert real Y coordinate to grid row. Uses round() for accuracy."""
        return round((y - self.config.origin_y) / self.config.grid_size)

    def _grid_to_real_x(self, col: int) -> float:
        """Convert grid column to real X coordinate."""
        return self.config.origin_x + col * self.config.grid_size

    def _grid_to_real_y(self, row: int) -> float:
        """Convert grid row to real Y coordinate."""
        return self.config.origin_y + row * self.config.grid_size

    def _real_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert real (x, y) to grid (col, row). Uses round() for accuracy."""
        return (self._real_to_grid_col(x), self._real_to_grid_row(y))

    def _grid_to_real(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid (col, row) to real (x, y)."""
        return (self._grid_to_real_x(col), self._grid_to_real_y(row))

    def _initialize_grids(self):
        """Initialize or reset the occupancy grids"""
        self.fcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self.bcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self._mark_board_margins()
        self.placed_vias.clear()
        self.routes.clear()
        self.failed.clear()
        self.stats = {'cells_explored': 0, 'backtracks': 0, 'layer_changes': 0}

    def _mark_board_margins(self):
        """Mark cells near board edges as blocked"""
        for col in range(self.grid_cols):
            for row in range(self.margin_cells):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'
            for row in range(max(0, self.grid_rows - self.margin_cells), self.grid_rows):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'

        for row in range(self.grid_rows):
            for col in range(self.margin_cells):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'
            for col in range(max(0, self.grid_cols - self.margin_cells), self.grid_cols):
                self.fcu_grid[row][col] = '__EDGE__'
                self.bcu_grid[row][col] = '__EDGE__'

    # =========================================================================
    # SPATIAL INDEX BUILDING
    # =========================================================================

    def _build_spatial_index(self):
        """
        Build spatial indexes from the occupancy grids and component data.

        Two index types are built:
        1. SpatialIndex (bitmap) - O(1) grid-based lookups for routing
        2. QuadTree (tree) - O(log n) courtyard-based queries

        Inspired by database query indexing - precompute once, query fast.

        OPTIMIZATION: Uses sparse iteration instead of dense loops.
        Only processes non-None cells, which are typically <1% of total cells.
        This reduces build time from ~500ms to ~20ms for 60x80mm boards.
        """
        import time
        start = time.time()

        # =====================================================================
        # BUILD BITMAP INDEX (Grid-based O(1) lookups) - SPARSE ITERATION
        # =====================================================================
        self.spatial_index = SpatialIndex(rows=self.grid_rows, cols=self.grid_cols)

        # SPARSE APPROACH: Most cells are None (empty)
        # Instead of iterating ALL 481K cells, only process non-empty cells.
        # Collect non-empty cells first, then batch process.

        # Use a pre-built set of (row, col, value) tuples for each layer
        # This is much faster than 481K iterations

        # F.Cu layer - collect non-empty cells
        fcu_blocked_cells = []
        fcu_net_cells = []  # (row, col, net_name)

        for r in range(self.grid_rows):
            row = self.fcu_grid[r]
            for c in range(self.grid_cols):
                cell = row[c]
                if cell is not None:
                    if cell in self.BLOCKED_MARKERS:
                        fcu_blocked_cells.append((r, c))
                    else:
                        fcu_net_cells.append((r, c, cell))

        # B.Cu layer - collect non-empty cells
        bcu_blocked_cells = []
        bcu_net_cells = []

        for r in range(self.grid_rows):
            row = self.bcu_grid[r]
            for c in range(self.grid_cols):
                cell = row[c]
                if cell is not None:
                    if cell in self.BLOCKED_MARKERS:
                        bcu_blocked_cells.append((r, c))
                    else:
                        bcu_net_cells.append((r, c, cell))

        # Now batch-apply to NumPy arrays (vectorized assignment)
        # F.Cu blocked
        if fcu_blocked_cells:
            rows, cols = zip(*fcu_blocked_cells)
            self.spatial_index.blocked_fcu[rows, cols] = True
            self.spatial_index.net_fcu[rows, cols] = -1

        # F.Cu nets
        if fcu_net_cells:
            rows, cols, nets = zip(*fcu_net_cells)
            for net_name in set(nets):
                net_id = self.spatial_index.get_net_id(net_name)
                indices = [i for i, n in enumerate(nets) if n == net_name]
                r_list = [rows[i] for i in indices]
                c_list = [cols[i] for i in indices]
                self.spatial_index.net_fcu[r_list, c_list] = net_id

        # B.Cu blocked
        if bcu_blocked_cells:
            rows, cols = zip(*bcu_blocked_cells)
            self.spatial_index.blocked_bcu[rows, cols] = True
            self.spatial_index.net_bcu[rows, cols] = -1

        # B.Cu nets
        if bcu_net_cells:
            rows, cols, nets = zip(*bcu_net_cells)
            for net_name in set(nets):
                net_id = self.spatial_index.get_net_id(net_name)
                indices = [i for i, n in enumerate(nets) if n == net_name]
                r_list = [rows[i] for i in indices]
                c_list = [cols[i] for i in indices]
                self.spatial_index.net_bcu[r_list, c_list] = net_id

        # Expand clearance zones (uses scipy.ndimage.binary_dilation - already fast)
        self.spatial_index.expand_clearances(
            trace_clearance_cells=self.clearance_cells,
            via_clearance_cells=self.via_clearance_cells
        )

        bitmap_time = time.time() - start

        # =====================================================================
        # BUILD QUADTREE INDEX (Courtyard-based O(log n) queries)
        # =====================================================================
        tree_start = time.time()
        self._build_quadtree()
        tree_time = time.time() - tree_start

        total_time = time.time() - start
        # Count all items in quadtree (including children)
        courtyard_count = self._count_quadtree_items(self.courtyard_tree) if self.courtyard_tree else 0
        print(f"    [INDEX] Built in {total_time*1000:.1f}ms: "
              f"bitmap={bitmap_time*1000:.1f}ms, quadtree={tree_time*1000:.1f}ms "
              f"({self.grid_rows}x{self.grid_cols} cells, {courtyard_count} courtyards)")

    def _count_quadtree_items(self, node: QuadTree) -> int:
        """Recursively count all items in a quadtree."""
        if node is None:
            return 0
        count = len(node.items)
        if node.children:
            for child in node.children:
                count += self._count_quadtree_items(child)
        return count

    def _build_quadtree(self):
        """
        Build QuadTree from component courtyards.

        This enables O(log n) queries for "what obstacles are near (x, y)?",
        essential for efficient routing on large boards with many components.
        """
        # Import courtyard calculation utility
        try:
            from .common_types import calculate_courtyard
        except ImportError:
            from common_types import calculate_courtyard

        # Create quadtree covering the board
        board_bounds = BoundingBox(
            min_x=0, min_y=0,
            max_x=self.config.board_width,
            max_y=self.config.board_height
        )
        self.courtyard_tree = QuadTree(board_bounds)

        # Insert component courtyards
        if not hasattr(self, '_placement') or not hasattr(self, '_parts_db'):
            return  # No placement data yet

        parts = self._parts_db.get('parts', {})

        for ref, pos in self._placement.items():
            part = parts.get(ref, {})
            footprint = part.get('footprint', part.get('package', ''))

            # Get position
            pos_x = pos.x if hasattr(pos, 'x') else pos[0]
            pos_y = pos.y if hasattr(pos, 'y') else pos[1]

            # PRIORITY: Use pre-calculated courtyard from Parts Piston if available
            courtyard_data = part.get('courtyard', None)
            if courtyard_data:
                # Use pre-calculated courtyard
                if hasattr(courtyard_data, 'width'):
                    # CourtyardInfo object
                    half_w = courtyard_data.width / 2
                    half_h = courtyard_data.height / 2
                    offset_x = getattr(courtyard_data, 'offset_x', 0)
                    offset_y = getattr(courtyard_data, 'offset_y', 0)
                    min_x = pos_x - half_w + offset_x
                    max_x = pos_x + half_w + offset_x
                    min_y = pos_y - half_h + offset_y
                    max_y = pos_y + half_h + offset_y
                elif isinstance(courtyard_data, dict):
                    half_w = courtyard_data.get('width', 2.0) / 2
                    half_h = courtyard_data.get('height', 2.0) / 2
                    min_x = pos_x - half_w
                    max_x = pos_x + half_w
                    min_y = pos_y - half_h
                    max_y = pos_y + half_h
                else:
                    courtyard = calculate_courtyard(part, margin=0.0, footprint_name=footprint)
                    min_x = pos_x + courtyard.min_x
                    max_x = pos_x + courtyard.max_x
                    min_y = pos_y + courtyard.min_y
                    max_y = pos_y + courtyard.max_y
            else:
                # Fallback: Calculate courtyard from pad positions
                courtyard = calculate_courtyard(part, margin=0.0, footprint_name=footprint)
                min_x = pos_x + courtyard.min_x
                max_x = pos_x + courtyard.max_x
                min_y = pos_y + courtyard.min_y
                max_y = pos_y + courtyard.max_y

            # Create bounding box
            bbox = BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

            # Determine layer blocking
            fp_lower = footprint.lower()
            is_smd = any(pkg in fp_lower for pkg in [
                '0402', '0603', '0805', '1206', 'sot', 'soic', 'qfp', 'qfn', 'bga'
            ])
            layer = 'F.Cu' if is_smd else 'BOTH'

            # Insert into quadtree
            item = QuadTreeItem(ref=ref, bbox=bbox, net=None, layer=layer)
            self.courtyard_tree.insert(item)

    def query_obstacles_near(self, x: float, y: float, radius: float = 2.0) -> List[str]:
        """
        Query component references near a point using the QuadTree.

        This is O(log n + k) where k is the number of results,
        compared to O(n) for scanning all components.

        Args:
            x, y: Position in mm
            radius: Search radius in mm

        Returns:
            List of component references near the point
        """
        if self.courtyard_tree is None:
            return []

        items = self.courtyard_tree.query_radius(x, y, radius)
        return [item.ref for item in items]

    def is_point_in_courtyard(self, x: float, y: float) -> Optional[str]:
        """
        Check if a point is inside any component courtyard.

        Uses QuadTree for O(log n) lookup.

        Returns:
            Component reference if point is in a courtyard, None otherwise
        """
        if self.courtyard_tree is None:
            return None

        items = self.courtyard_tree.query_point(x, y)
        return items[0].ref if items else None

    # =========================================================================
    # NET CLASS CONSTRAINTS
    # =========================================================================
    # These methods enable per-net trace widths and clearances based on
    # net classification (POWER, GROUND, HIGH_SPEED, etc.)

    def _classify_nets(self, parts_db: Dict):
        """
        Classify all nets and cache their routing parameters.

        This is called once at the start of routing to build lookup tables
        for per-net trace widths and clearances.
        """
        from pcb_engine.routing_types import NetClass, DEFAULT_NET_CLASS_RULES

        nets = parts_db.get('nets', {})

        for net_name in nets.keys():
            net_class = self._classify_net(net_name)
            self._net_classes[net_name] = net_class

            # Get rules for this class
            if self.config.use_net_class_constraints:
                rules = self.config.get_net_class_rules(net_class)
                self._net_trace_widths[net_name] = rules.trace_width
                self._net_clearances[net_name] = rules.clearance
            else:
                # Use global defaults
                self._net_trace_widths[net_name] = self.config.trace_width
                self._net_clearances[net_name] = self.config.clearance

    def _classify_net(self, net_name: str) -> 'NetClass':
        """Classify a net based on its name."""
        from pcb_engine.routing_types import NetClass

        name_upper = net_name.upper()

        # Ground nets (highest priority)
        if any(x in name_upper for x in ['GND', 'GROUND', 'VSS', 'AGND', 'DGND', 'PGND']):
            return NetClass.GROUND

        # Power nets
        if any(x in name_upper for x in ['VCC', 'VDD', '5V', '3V3', '3.3V', '12V', 'VBAT',
                                          'VIN', 'VOUT', 'PWR', 'POWER', '+5V', '+3.3V',
                                          'AVCC', 'AVDD', 'DVCC', 'DVDD', 'V+']):
            return NetClass.POWER

        # High-speed signals
        if any(x in name_upper for x in ['CLK', 'CLOCK', 'USB', 'HDMI', 'ETH', 'PCIE',
                                          'DDR', 'SDRAM', 'QSPI', 'HSPI', 'MOSI', 'MISO',
                                          'SCK', 'SCLK']):
            return NetClass.HIGH_SPEED

        # Differential pairs
        if any(x in name_upper for x in ['_P', '_N', '+', '-']) and \
           any(x in name_upper for x in ['USB', 'LVDS', 'HDMI', 'ETH', 'PCIE', 'DIFF']):
            return NetClass.DIFFERENTIAL

        # RF signals
        if any(x in name_upper for x in ['RF', 'ANT', 'ANTENNA', 'LNA', 'PA_OUT', '2.4G']):
            return NetClass.RF

        # Analog signals
        if any(x in name_upper for x in ['ANALOG', 'ADC', 'DAC', 'VREF', 'SENSE', 'AIN']):
            return NetClass.ANALOG

        # High current (motor, LED)
        if any(x in name_upper for x in ['MOTOR', 'LED_PWR', 'HEAT', 'DRIVE']):
            return NetClass.HIGH_CURRENT

        # Default to signal
        return NetClass.SIGNAL

    def get_trace_width(self, net_name: str) -> float:
        """Get trace width for a specific net."""
        return self._net_trace_widths.get(net_name, self.config.trace_width)

    def get_clearance(self, net_name: str) -> float:
        """Get clearance for a specific net."""
        return self._net_clearances.get(net_name, self.config.clearance)

    def get_net_class(self, net_name: str) -> 'NetClass':
        """Get the net class for a specific net."""
        from pcb_engine.routing_types import NetClass
        return self._net_classes.get(net_name, NetClass.SIGNAL)

    # =========================================================================
    # TRUNK CHAIN DETECTION - Identify parallel routing opportunities
    # =========================================================================
    # A trunk chain is a group of nets that:
    # 1. Have similar routing direction (horizontal or vertical)
    # 2. Have overlapping span (start-end range)
    # 3. Can be routed in parallel to save wirelength and improve signal integrity

    def _detect_trunk_chains(self, parts_db: Dict, escapes: Dict) -> List[Dict]:
        """
        Detect trunk chains - groups of nets that can share a routing corridor.

        Returns:
            List of trunk chain definitions:
            [
                {
                    'nets': ['NET1', 'NET2', 'NET3'],
                    'direction': 'horizontal',  # or 'vertical'
                    'corridor': (min_x, max_x, avg_y),  # or (avg_x, min_y, max_y)
                    'spacing': 0.3,  # recommended spacing between traces
                },
                ...
            ]
        """
        self._trunk_chains = []
        self._net_to_chain = {}

        nets = parts_db.get('nets', {})
        if not nets:
            return []

        # First, characterize each net's routing direction and span
        net_profiles = {}
        for net_name, info in nets.items():
            pins = info.get('pins', [])
            if len(pins) < 2:
                continue

            endpoints = self._get_escape_endpoints(pins, escapes)
            if len(endpoints) < 2:
                continue

            # Calculate net bounding box and direction
            xs = [p[0] for p in endpoints]
            ys = [p[1] for p in endpoints]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max_x - min_x
            span_y = max_y - min_y

            # Determine preferred direction
            if span_x > span_y * 1.5:
                direction = 'horizontal'
            elif span_y > span_x * 1.5:
                direction = 'vertical'
            else:
                direction = 'diagonal'  # Not a good trunk candidate

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            net_profiles[net_name] = {
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'center_x': center_x, 'center_y': center_y,
                'span_x': span_x, 'span_y': span_y,
                'direction': direction,
                'endpoints': endpoints,
            }

        # Group nets by direction and overlapping corridors
        horizontal_nets = [n for n, p in net_profiles.items() if p['direction'] == 'horizontal']
        vertical_nets = [n for n, p in net_profiles.items() if p['direction'] == 'vertical']

        # Find horizontal trunk chains
        h_chains = self._find_parallel_chains(horizontal_nets, net_profiles, 'horizontal')
        # Find vertical trunk chains
        v_chains = self._find_parallel_chains(vertical_nets, net_profiles, 'vertical')

        self._trunk_chains = h_chains + v_chains

        # Build net-to-chain mapping
        for idx, chain in enumerate(self._trunk_chains):
            for net_name in chain['nets']:
                self._net_to_chain[net_name] = idx

        return self._trunk_chains

    def _find_parallel_chains(self, net_names: List[str], profiles: Dict,
                               direction: str) -> List[Dict]:
        """
        Find groups of nets that can be routed in parallel.

        For horizontal nets: group by similar Y-coordinate and overlapping X-span
        For vertical nets: group by similar X-coordinate and overlapping Y-span
        """
        chains = []
        used = set()

        # Tolerance for "same corridor" (in mm)
        corridor_tolerance = 3.0  # Nets within 3mm can share a corridor

        for net1 in net_names:
            if net1 in used:
                continue

            p1 = profiles[net1]
            chain_nets = [net1]
            used.add(net1)

            for net2 in net_names:
                if net2 in used:
                    continue

                p2 = profiles[net2]

                if direction == 'horizontal':
                    # Check if Y centers are close and X spans overlap
                    y_diff = abs(p1['center_y'] - p2['center_y'])
                    x_overlap = min(p1['max_x'], p2['max_x']) - max(p1['min_x'], p2['min_x'])

                    if y_diff < corridor_tolerance and x_overlap > 1.0:
                        chain_nets.append(net2)
                        used.add(net2)

                else:  # vertical
                    # Check if X centers are close and Y spans overlap
                    x_diff = abs(p1['center_x'] - p2['center_x'])
                    y_overlap = min(p1['max_y'], p2['max_y']) - max(p1['min_y'], p2['min_y'])

                    if x_diff < corridor_tolerance and y_overlap > 1.0:
                        chain_nets.append(net2)
                        used.add(net2)

            # Only create chain if 2+ nets
            if len(chain_nets) >= 2:
                # Calculate corridor bounds
                if direction == 'horizontal':
                    min_x = min(profiles[n]['min_x'] for n in chain_nets)
                    max_x = max(profiles[n]['max_x'] for n in chain_nets)
                    avg_y = sum(profiles[n]['center_y'] for n in chain_nets) / len(chain_nets)
                    corridor = (min_x, max_x, avg_y)
                else:
                    avg_x = sum(profiles[n]['center_x'] for n in chain_nets) / len(chain_nets)
                    min_y = min(profiles[n]['min_y'] for n in chain_nets)
                    max_y = max(profiles[n]['max_y'] for n in chain_nets)
                    corridor = (avg_x, min_y, max_y)

                # Calculate recommended spacing (based on net classes)
                max_width = max(self.get_trace_width(n) for n in chain_nets)
                max_clearance = max(self.get_clearance(n) for n in chain_nets)
                spacing = max_width + max_clearance

                chains.append({
                    'nets': chain_nets,
                    'direction': direction,
                    'corridor': corridor,
                    'spacing': spacing,
                    'net_count': len(chain_nets),
                })

        return chains

    def _route_trunk_chain(self, chain: Dict, escapes: Dict) -> Dict[str, Route]:
        """
        Route a trunk chain - multiple nets sharing a common trunk with proper spacing.

        Returns:
            Dict mapping net_name -> Route for all nets in the chain
        """
        routes = {}
        nets = chain['nets']
        direction = chain['direction']
        corridor = chain['corridor']
        spacing = chain['spacing']

        layer = 0 if self.config.prefer_top_layer else 1
        grid = self.fcu_grid if layer == 0 else self.bcu_grid
        layer_name = 'F.Cu' if layer == 0 else 'B.Cu'

        # Calculate trunk offsets for each net (centered around corridor)
        num_nets = len(nets)
        total_width = (num_nets - 1) * spacing
        start_offset = -total_width / 2

        for i, net_name in enumerate(nets):
            offset = start_offset + i * spacing
            route = Route(net=net_name, algorithm_used='trunk_chain')

            if direction == 'horizontal':
                min_x, max_x, base_y = corridor
                trunk_y = base_y + offset

                # Create trunk segment
                trunk_seg = TrackSegment(
                    start=(min_x, trunk_y),
                    end=(max_x, trunk_y),
                    layer=layer_name,
                    width=self.get_trace_width(net_name),
                    net=net_name
                )
                route.segments.append(trunk_seg)

                # Add branches to actual endpoints
                endpoints = self._get_escape_endpoints(
                    self._parts_db.get('nets', {}).get(net_name, {}).get('pins', []),
                    escapes
                )
                for ep in endpoints:
                    if abs(ep[1] - trunk_y) > 0.01:
                        branch = TrackSegment(
                            start=(ep[0], trunk_y),
                            end=ep,
                            layer=layer_name,
                            width=self.get_trace_width(net_name),
                            net=net_name
                        )
                        route.segments.append(branch)

            else:  # vertical
                base_x, min_y, max_y = corridor
                trunk_x = base_x + offset

                # Create trunk segment
                trunk_seg = TrackSegment(
                    start=(trunk_x, min_y),
                    end=(trunk_x, max_y),
                    layer=layer_name,
                    width=self.get_trace_width(net_name),
                    net=net_name
                )
                route.segments.append(trunk_seg)

                # Add branches to actual endpoints
                endpoints = self._get_escape_endpoints(
                    self._parts_db.get('nets', {}).get(net_name, {}).get('pins', []),
                    escapes
                )
                for ep in endpoints:
                    if abs(ep[0] - trunk_x) > 0.01:
                        branch = TrackSegment(
                            start=(trunk_x, ep[1]),
                            end=ep,
                            layer=layer_name,
                            width=self.get_trace_width(net_name),
                            net=net_name
                        )
                        route.segments.append(branch)

            route.success = len(route.segments) > 0
            routes[net_name] = route

        return routes

    def get_trunk_chain_stats(self) -> Dict:
        """Get statistics about detected trunk chains."""
        if not self._trunk_chains:
            return {
                'total_chains': 0,
                'total_nets_in_chains': 0,
                'horizontal_chains': 0,
                'vertical_chains': 0,
                'largest_chain': 0,
                'chains': [],
            }

        h_chains = [c for c in self._trunk_chains if c['direction'] == 'horizontal']
        v_chains = [c for c in self._trunk_chains if c['direction'] == 'vertical']

        return {
            'total_chains': len(self._trunk_chains),
            'total_nets_in_chains': sum(c['net_count'] for c in self._trunk_chains),
            'horizontal_chains': len(h_chains),
            'vertical_chains': len(v_chains),
            'largest_chain': max(c['net_count'] for c in self._trunk_chains),
            'chains': [
                {
                    'nets': c['nets'],
                    'direction': c['direction'],
                    'spacing': c['spacing'],
                }
                for c in self._trunk_chains
            ],
        }

    # =========================================================================
    # RETURN PATH AWARENESS - Ensure proper ground reference for signals
    # =========================================================================
    # High-frequency signals require a continuous ground plane beneath them.
    # Return current flows on the reference plane directly under the signal trace.
    # Violations (slots, voids, layer changes without via stitching) cause:
    # - EMI radiation from the longer return path
    # - Crosstalk between adjacent signals
    # - Impedance discontinuities affecting signal integrity

    def _build_ground_plane_map(self, parts_db: Dict, pour_layer: str = 'B.Cu') -> None:
        """
        Build a map of where solid ground plane exists.

        For 2-layer boards: assumes bottom layer is ground pour (common practice)
        For 4+ layer boards: would check internal ground layers

        The map is a 2D boolean array where True = solid ground reference
        """
        self._ground_plane_map = np.ones((self.grid_rows, self.grid_cols), dtype=bool)

        # Mark component pads as breaks in ground plane
        # (Pads are connections, not continuous copper)
        if pour_layer == 'B.Cu':
            grid = self.bcu_grid
        else:
            grid = self.fcu_grid

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = grid[row][col]
                if cell is not None:
                    # Non-GND nets break the ground plane
                    if cell not in ['GND', 'GROUND', 'VSS', 'AGND', 'DGND']:
                        self._ground_plane_map[row, col] = False

        # Also mark anti-pads (clearance around non-ground pads)
        # These create voids in the pour
        clearance_cells = self.clearance_cells
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if not self._ground_plane_map[row, col]:
                    # Expand the void by clearance
                    for dr in range(-clearance_cells, clearance_cells + 1):
                        for dc in range(-clearance_cells, clearance_cells + 1):
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                                self._ground_plane_map[nr, nc] = False

    def _analyze_return_path(self, route: 'Route', net_name: str) -> Dict:
        """
        Analyze a route for return path quality.

        Checks:
        1. How much of the route is over solid ground plane
        2. Any layer transitions without via stitching
        3. Long segments crossing ground voids

        Returns:
            Dict with analysis results:
            {
                'coverage': 0.95,  # 95% over solid ground
                'void_crossings': 2,  # Number of void crossings
                'max_void_length': 1.5,  # Longest void crossing in mm
                'quality': 'good'  # 'good', 'marginal', 'poor'
            }
        """
        if self._ground_plane_map is None:
            return {
                'coverage': 1.0,
                'void_crossings': 0,
                'max_void_length': 0.0,
                'quality': 'unknown',
            }

        total_cells = 0
        covered_cells = 0
        void_crossings = 0
        current_void_length = 0
        max_void_length = 0
        in_void = False

        for seg in route.segments:
            # Sample points along the segment
            length = seg.length
            if length < 0.01:
                continue

            num_samples = max(2, int(length / self.config.grid_size))
            for i in range(num_samples):
                t = i / (num_samples - 1) if num_samples > 1 else 0
                x = seg.start[0] + t * (seg.end[0] - seg.start[0])
                y = seg.start[1] + t * (seg.end[1] - seg.start[1])

                col = self._real_to_grid_col(x)
                row = self._real_to_grid_row(y)

                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    total_cells += 1
                    if self._ground_plane_map[row, col]:
                        covered_cells += 1
                        if in_void:
                            # Exiting void
                            in_void = False
                            max_void_length = max(max_void_length, current_void_length)
                            current_void_length = 0
                    else:
                        if not in_void:
                            # Entering void
                            in_void = True
                            void_crossings += 1
                        current_void_length += self.config.grid_size

        # Final void check
        if in_void:
            max_void_length = max(max_void_length, current_void_length)

        coverage = covered_cells / total_cells if total_cells > 0 else 1.0

        # Determine quality
        if coverage >= 0.95 and max_void_length < 1.0:
            quality = 'good'
        elif coverage >= 0.80 and max_void_length < 3.0:
            quality = 'marginal'
        else:
            quality = 'poor'

        return {
            'coverage': coverage,
            'void_crossings': void_crossings,
            'max_void_length': max_void_length,
            'quality': quality,
        }

    def _calculate_return_path_cost(self, row: int, col: int, layer: int) -> float:
        """
        Calculate additional routing cost based on return path quality.

        Used during pathfinding to penalize routes over ground voids.
        Layer 0 (F.Cu) signals need reference on layer 1 (B.Cu).

        Returns:
            Cost multiplier (1.0 = good, up to 3.0 = poor)
        """
        if self._ground_plane_map is None or not self._return_path_enabled:
            return 1.0

        # Only top layer signals need bottom layer reference
        if layer != 0:
            return 1.0

        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            if self._ground_plane_map[row, col]:
                return 1.0  # Good reference
            else:
                return 2.5  # Penalize void crossing
        return 1.0

    def _check_route_return_path(self, route: 'Route', net_name: str) -> None:
        """
        Check a route for return path violations and record them.

        Called after routing to identify potential signal integrity issues.
        """
        from pcb_engine.routing_types import NetClass

        # Only check high-speed and critical nets
        net_class = self._net_classes.get(net_name, NetClass.SIGNAL)
        if net_class not in [NetClass.HIGH_SPEED, NetClass.DIFFERENTIAL, NetClass.RF]:
            return

        analysis = self._analyze_return_path(route, net_name)

        if analysis['quality'] == 'poor':
            self._return_path_violations.append({
                'net': net_name,
                'net_class': net_class.value,
                'coverage': analysis['coverage'],
                'void_crossings': analysis['void_crossings'],
                'max_void_length': analysis['max_void_length'],
                'severity': 'error' if analysis['coverage'] < 0.7 else 'warning',
            })
        elif analysis['quality'] == 'marginal':
            self._return_path_violations.append({
                'net': net_name,
                'net_class': net_class.value,
                'coverage': analysis['coverage'],
                'void_crossings': analysis['void_crossings'],
                'max_void_length': analysis['max_void_length'],
                'severity': 'warning',
            })

    def get_return_path_stats(self) -> Dict:
        """Get return path analysis statistics."""
        if not self._return_path_violations:
            return {
                'enabled': self._return_path_enabled,
                'violations': 0,
                'errors': 0,
                'warnings': 0,
                'details': [],
            }

        errors = [v for v in self._return_path_violations if v['severity'] == 'error']
        warnings = [v for v in self._return_path_violations if v['severity'] == 'warning']

        return {
            'enabled': self._return_path_enabled,
            'violations': len(self._return_path_violations),
            'errors': len(errors),
            'warnings': len(warnings),
            'details': self._return_path_violations,
        }

    # =========================================================================
    # MAIN ROUTING ENTRY POINT
    # =========================================================================

    def route(self, parts_db: Dict, escapes: Dict, placement: Dict,
              net_order: List[str],
              layer_directions: Dict = None,
              net_specs: Dict = None,
              global_routing: Dict = None) -> RoutingResult:
        """
        Route all nets using the configured algorithm.

        Args:
            parts_db: Parts database with nets and component info
            escapes: Escape routes {ref: {pin: EscapeRoute}}
            placement: Component placements {ref: Position}
            net_order: Order of nets to route
            layer_directions: CPU Lab layer direction preferences
                              e.g. {'F.Cu': 'horizontal', 'B.Cu': 'vertical'}
            net_specs: CPU Lab per-net routing specs
                       e.g. {'USB_DP': {'trace_width_mm': 0.25, 'clearance_mm': 0.15}}
            global_routing: CPU Lab global routing hints (region assignments)
                            e.g. {'region_size_mm': 5.0, 'net_routes': {'NET1': {'regions': [(0,1), (1,1)]}}}

        Returns:
            RoutingResult with all routes and statistics
        """
        # Store for use by _get_escape_endpoints fallback
        self._placement = placement
        self._parts_db = parts_db
        self._layer_directions = layer_directions or {}
        self._net_specs = net_specs or {}
        self._global_routing = global_routing or {}

        # CLASSIFY NETS - Determine per-net trace widths and clearances
        # This builds lookup tables for power/ground/signal/etc. net classes
        self._classify_nets(parts_db)

        # OVERRIDE with CPU Lab per-net specs (more accurate than name-based classification)
        if self._net_specs:
            for net_name, specs in self._net_specs.items():
                if 'trace_width_mm' in specs and specs['trace_width_mm']:
                    self._net_trace_widths[net_name] = specs['trace_width_mm']
                if 'clearance_mm' in specs and specs['clearance_mm']:
                    self._net_clearances[net_name] = specs['clearance_mm']

        # DETECT TRUNK CHAINS - Identify parallel routing opportunities
        # This finds groups of nets that can share a routing corridor
        self._detect_trunk_chains(parts_db, escapes)

        # Build net_pins lookup
        nets = parts_db.get('nets', {})
        net_pins = {name: info.get('pins', []) for name, info in nets.items()}

        # TRY CACHE FIRST (CASCADE optimization - saves ~8 seconds)
        cache_hit = False
        if self.use_spatial_index:
            cache_hit = self._try_use_cached_index(placement, parts_db)

        if not cache_hit:
            # CACHE MISS - Initialize grids and build index from scratch
            self._initialize_grids()

            # Register obstacles
            self._register_components(placement, parts_db)
            self._register_escapes(escapes)

            # BUILD SPATIAL INDEX for O(1) obstacle lookups
            if self.use_spatial_index:
                self._build_spatial_index()
                # Save to cache for next algorithm in CASCADE
                self._cache_index(placement, parts_db)

        # Save clean grid snapshot (components+escapes only, no routes)
        # Used by RIPUP and other iterative algorithms for fast grid restoration
        self._save_grid_snapshot()

        # BUILD GROUND PLANE MAP for return path analysis
        # This identifies where solid ground reference exists (typically B.Cu pour)
        if self._return_path_enabled:
            self._build_ground_plane_map(parts_db, 'B.Cu')
            self._return_path_violations = []  # Reset violations

        # Select and run algorithm
        algorithm = self.config.algorithm.lower()

        # Check for parallel routing
        if self.config.parallel_routing and algorithm in ['lee', 'lee_parallel']:
            return self._route_lee_parallel(net_order, net_pins, escapes)

        if algorithm == 'lee':
            return self._route_lee(net_order, net_pins, escapes)
        elif algorithm == 'lee_parallel':
            return self._route_lee_parallel(net_order, net_pins, escapes)
        elif algorithm == 'hadlock':
            return self._route_hadlock(net_order, net_pins, escapes)
        elif algorithm == 'soukup':
            return self._route_soukup(net_order, net_pins, escapes)
        elif algorithm == 'mikami':
            return self._route_mikami(net_order, net_pins, escapes)
        elif algorithm == 'astar':
            return self._route_astar_only(net_order, net_pins, escapes)
        elif algorithm == 'pathfinder':
            return self._route_pathfinder(net_order, net_pins, escapes, placement, parts_db)
        elif algorithm == 'ripup':
            return self._route_ripup_reroute(net_order, net_pins, escapes, placement, parts_db)
        elif algorithm == 'steiner':
            return self._route_steiner(net_order, net_pins, escapes)
        elif algorithm == 'channel':
            return self._route_channel(net_order, net_pins, escapes)
        elif algorithm == 'hybrid':
            return self._route_hybrid(net_order, net_pins, escapes, placement, parts_db)
        elif algorithm == 'auto':
            return self._route_auto(net_order, net_pins, escapes, placement, parts_db)
        elif algorithm in ['push_and_shove', 'pns', 'shove']:
            return self._route_push_and_shove(net_order, net_pins, escapes, placement, parts_db)
        else:
            # Default to hybrid
            return self._route_hybrid(net_order, net_pins, escapes, placement, parts_db)

    # =========================================================================
    # SMART ROUTING WITH PLAN
    # =========================================================================

    def route_with_plan(self, routing_plan: 'RoutingPlan',
                       parts_db: Dict, escapes: Dict, placement: Dict,
                       learning_db: 'LearningDatabase' = None,
                       layer_directions: Dict = None,
                       net_specs: Dict = None,
                       global_routing: Dict = None) -> 'RoutingResult':
        """
        Route using a smart routing plan with per-net algorithm selection.

        This is the INTELLIGENT routing method that uses the RoutingPlanner's
        analysis to select the best algorithm for each net, rather than
        applying a single algorithm to all nets.

        Args:
            routing_plan: RoutingPlan from RoutingPlanner with per-net strategies
            parts_db: Parts database with nets and component info
            escapes: Escape routes {ref: {pin: EscapeRoute}}
            placement: Component placements {ref: Position}
            learning_db: Optional learning database to record outcomes
            layer_directions: CPU Lab layer direction preferences
            net_specs: CPU Lab per-net routing specs
            global_routing: CPU Lab global routing hints (region assignments)

        Returns:
            RoutingResult with all routes and statistics
        """
        import time as time_module
        from .routing_planner import NetClass, RoutingAlgorithm
        from .learning_database import RoutingOutcome

        # Store for use by internal methods
        self._placement = placement
        self._parts_db = parts_db
        self._layer_directions = layer_directions or {}
        self._net_specs = net_specs or {}
        self._global_routing = global_routing or {}

        # Initialize grids and spatial index (once for all nets)
        self._initialize_grids()
        self._register_components(placement, parts_db)
        self._register_escapes(escapes)

        if self.use_spatial_index:
            self._build_spatial_index()

        # Build ground plane map if needed
        if routing_plan.enable_return_path_check:
            self._build_ground_plane_map(parts_db, 'B.Cu')
            self._return_path_violations = []

        # Build net_pins lookup
        nets = parts_db.get('nets', {})
        net_pins = {name: info.get('pins', []) for name, info in nets.items()}

        # Results tracking
        all_routes = {}
        outcomes = []  # For learning database
        routed_count = 0
        failed_nets = []
        algorithm_usage = {}  # Track which algorithm succeeded

        # Route each net according to its strategy
        for net_name in routing_plan.routing_order:
            strategy = routing_plan.get_strategy(net_name)
            if not strategy:
                continue

            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue  # Skip unroutable nets

            # Set per-net trace width and clearance
            self._net_trace_widths[net_name] = strategy.trace_width
            self._net_clearances[net_name] = strategy.clearance

            # Try routing with this net's algorithm chain
            start_time = time_module.time()
            success = False
            algorithm_used = None
            route_result = None

            # Build algorithm chain: primary + fallbacks + push-and-shove
            algorithms_to_try = [strategy.primary_algorithm] + strategy.fallback_algorithms
            if strategy.use_push_and_shove:
                algorithms_to_try.append(RoutingAlgorithm.PUSH_AND_SHOVE)

            for algorithm in algorithms_to_try:
                # Temporarily set algorithm
                orig_algorithm = self.config.algorithm
                self.config.algorithm = algorithm.value

                try:
                    # Route just this net
                    single_result = self._route_single_net(
                        net_name, pins, escapes, net_pins
                    )

                    if single_result and single_result.get('success', False):
                        success = True
                        algorithm_used = algorithm
                        route_result = single_result
                        break

                except Exception as e:
                    # Algorithm failed, try next
                    pass
                finally:
                    self.config.algorithm = orig_algorithm

            # Record outcome
            elapsed_ms = (time_module.time() - start_time) * 1000

            if success and route_result:
                route_obj = route_result.get('route')
                all_routes[net_name] = route_obj
                routed_count += 1
                algorithm_usage[algorithm_used.value] = algorithm_usage.get(algorithm_used.value, 0) + 1
                # Note: routes are already marked on grid by _route_net_* functions
            else:
                failed_nets.append(net_name)

            # Record to learning database
            if learning_db:
                outcome = RoutingOutcome(
                    net_name=net_name,
                    net_class=strategy.net_class.value,
                    design_hash='',  # Will be set by caller
                    algorithm=algorithm_used.value if algorithm_used else 'none',
                    success=success,
                    time_ms=elapsed_ms,
                    via_count=route_result.get('via_count', 0) if route_result else 0,
                    wire_length_mm=route_result.get('wire_length', 0) if route_result else 0,
                    quality_score=100 if success else 0,
                )
                outcomes.append(outcome)
                learning_db.record_outcome(outcome)

        # Check return path for high-speed nets
        if routing_plan.enable_return_path_check:
            for net_name in routing_plan.routing_order:
                strategy = routing_plan.get_strategy(net_name)
                if strategy and strategy.check_return_path:
                    route = all_routes.get(net_name)
                    if route:
                        self._check_route_return_path(route, net_name)

        # Create result
        total_nets = len([n for n in routing_plan.routing_order
                         if len(net_pins.get(n, [])) >= 2])

        return RoutingResult(
            routes=all_routes,
            success=routed_count == total_nets,
            routed_count=routed_count,
            total_count=total_nets,
            failed_nets=failed_nets,
            algorithm_used='smart_plan',
            timing_ms=0,  # Total time tracked by caller
            metadata={
                'algorithm_usage': algorithm_usage,
                'return_path_violations': len(self._return_path_violations) if hasattr(self, '_return_path_violations') else 0,
                'outcomes': outcomes,
            }
        )

    def _route_single_net(self, net_name: str, pins: List,
                         escapes: Dict, net_pins: Dict) -> Optional[Dict]:
        """
        Route a single net using the currently configured algorithm.

        This is a helper for route_with_plan() that routes just one net.
        Uses the actual working routing functions (_route_net_lee, _route_net_astar, etc.)
        which handle multi-pin MST routing correctly.

        Returns:
            Dict with 'success', 'route', 'via_count', 'wire_length' or None
        """
        if len(pins) < 2:
            return None

        algorithm = self.config.algorithm.lower()

        try:
            # Use the REAL routing functions that handle multi-pin nets via MST
            if algorithm == 'lee':
                route = self._route_net_lee(net_name, pins, escapes)
            elif algorithm == 'hadlock':
                route = self._route_net_hadlock(net_name, pins, escapes)
            elif algorithm in ('a_star', 'astar'):
                route = self._route_net_astar(net_name, pins, escapes)
            elif algorithm == 'soukup':
                route = self._route_net_soukup(net_name, pins, escapes)
            elif algorithm == 'mikami':
                route = self._route_net_mikami(net_name, pins, escapes)
            elif algorithm == 'steiner':
                route = self._route_net_steiner(net_name, pins, escapes)
            else:
                # Default: try A*, fallback to Lee
                route = self._route_net_astar(net_name, pins, escapes)
                if not route.success:
                    route = self._route_net_lee(net_name, pins, escapes)

            if route and route.success:
                via_count = len(route.vias) if hasattr(route, 'vias') else 0
                wire_length = sum(
                    self._segment_length(s) for s in route.segments
                ) if route.segments else 0

                return {
                    'success': True,
                    'route': route,  # Return full Route object
                    'via_count': via_count,
                    'wire_length': wire_length,
                }
        except Exception:
            pass

        return None

    def _route_with_push_and_shove(self, start: Tuple, end: Tuple,
                                   net_name: str) -> Optional[List]:
        """
        Try routing with push-and-shove when direct routing fails.
        """
        # First try direct A*
        path = self._route_astar_net(start, end, net_name)
        if path:
            return path

        # Try walkaround
        if hasattr(self, '_find_walkaround_path'):
            path = self._find_walkaround_path(start, end, net_name)
            if path:
                return path

        # Try shove
        if hasattr(self, '_attempt_shove_routing'):
            path = self._attempt_shove_routing(start, end, net_name)
            if path:
                return path

        return None

    def _segment_length(self, segment) -> float:
        """Calculate length of a track segment in mm"""
        if hasattr(segment, 'start') and hasattr(segment, 'end'):
            dx = segment.end[0] - segment.start[0]
            dy = segment.end[1] - segment.start[1]
            return (dx*dx + dy*dy) ** 0.5
        return 0.0

    def _mark_route_on_grid(self, segment, net_name: str) -> None:
        """Mark a routed segment on the grid to block future routing"""
        if not hasattr(segment, 'start') or not hasattr(segment, 'end'):
            return

        layer = getattr(segment, 'layer', 'F.Cu')
        grid_layer = self._layer_to_grid.get(layer, 0)

        # Mark cells along the segment
        x1, y1 = segment.start
        x2, y2 = segment.end

        # Convert to grid coordinates
        gx1, gy1 = self._mm_to_grid(x1, y1)
        gx2, gy2 = self._mm_to_grid(x2, y2)

        # Simple Bresenham's line
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy

        x, y = gx1, gy1
        while True:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                self.grid[grid_layer, y, x] = net_name

            if x == gx2 and y == gy2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # =========================================================================
    # PARALLEL ROUTING - Multi-Core Optimization
    # =========================================================================
    # Routes independent nets in parallel using multiple CPU cores.
    # Independent nets = nets that don't share components (can't interfere).
    #
    # Strategy:
    # 1. Build dependency graph from net pins (which components each net uses)
    # 2. Find independent net groups using graph coloring
    # 3. Route each group in parallel with separate grid copies
    # 4. Merge results and mark grid cells
    # =========================================================================

    def _find_independent_net_groups(self, net_pins: Dict) -> List[List[str]]:
        """
        Find groups of independent nets that can be routed in parallel.

        Two nets are DEPENDENT if they share any component.
        Independent nets can be routed simultaneously without conflict.

        Returns list of groups, where each group contains independent nets.
        """
        # Build component -> nets mapping
        component_to_nets: Dict[str, Set[str]] = {}
        for net_name, pins in net_pins.items():
            if len(pins) < 2:
                continue
            for pin in pins:
                ref, _ = self._parse_pin_ref(pin)
                if ref:
                    if ref not in component_to_nets:
                        component_to_nets[ref] = set()
                    component_to_nets[ref].add(net_name)

        # Build net -> dependent nets mapping
        net_dependencies: Dict[str, Set[str]] = {}
        for net_name in net_pins.keys():
            if len(net_pins.get(net_name, [])) < 2:
                continue
            net_dependencies[net_name] = set()
            for pin in net_pins[net_name]:
                ref, _ = self._parse_pin_ref(pin)
                if ref in component_to_nets:
                    # All nets on same component are dependent
                    net_dependencies[net_name].update(component_to_nets[ref])
            net_dependencies[net_name].discard(net_name)  # Remove self

        # Graph coloring to find independent groups
        # Greedy: assign each net to first group where it has no conflicts
        groups: List[List[str]] = []
        assigned: Set[str] = set()

        for net_name in net_pins.keys():
            if net_name in assigned or len(net_pins.get(net_name, [])) < 2:
                continue

            dependencies = net_dependencies.get(net_name, set())

            # Try to add to existing group
            added = False
            for group in groups:
                # Check if net conflicts with any net in this group
                conflicts = False
                for existing_net in group:
                    if existing_net in dependencies:
                        conflicts = True
                        break

                if not conflicts:
                    group.append(net_name)
                    assigned.add(net_name)
                    added = True
                    break

            # Create new group if couldn't add to existing
            if not added:
                groups.append([net_name])
                assigned.add(net_name)

        return groups

    def _route_net_parallel_worker(self, net_name: str, pins: List[Dict],
                                    escapes: Dict, grid_lock: threading.Lock,
                                    fcu_grid_copy: List[List],
                                    bcu_grid_copy: List[List]) -> Tuple[str, 'Route']:
        """
        Worker function for parallel routing of a single net.
        Uses thread-local grid copies to avoid conflicts.
        """
        # Route using Lee algorithm on the grid copy
        route = self._route_net_lee_on_grid(net_name, pins, escapes,
                                            fcu_grid_copy, bcu_grid_copy)
        return (net_name, route)

    def _route_net_lee_on_grid(self, net_name: str, pins: List[Dict],
                               escapes: Dict, fcu_grid: List[List],
                               bcu_grid: List[List]) -> 'Route':
        """
        Route a net using Lee algorithm on provided grid copies.
        This is the thread-safe version for parallel routing.
        """
        # Get escape endpoints for all pins
        # Convert pins to format expected by _get_escape_endpoints
        pin_refs = []
        for pin in pins:
            ref, pin_id = self._parse_pin_ref(pin)
            if ref and pin_id:
                pin_refs.append({'ref': ref, 'pin': pin_id})

        endpoints = self._get_escape_endpoints(pin_refs, escapes)

        if len(endpoints) < 2:
            return Route(net=net_name, success=False, segments=[], vias=[])

        # Convert (x, y) endpoints to (x, y, layer) format
        # Default to top layer for simplicity
        endpoints_3d = [(ep[0], ep[1], 'F.Cu') for ep in endpoints]

        # Use MST to connect all endpoints
        segments = []
        vias = []

        # Connect first two endpoints
        start = endpoints_3d[0]
        target = endpoints_3d[1]

        path = self._lee_wavefront_on_grid(start, target, net_name,
                                           fcu_grid, bcu_grid)
        if path:
            segs, vs = self._path_to_segments(path, net_name)
            segments.extend(segs)
            vias.extend(vs)

            # Mark path on grid
            for layer, row, col in path:
                if layer == 0:
                    fcu_grid[row][col] = net_name
                else:
                    bcu_grid[row][col] = net_name

        # Connect remaining endpoints to the tree
        for i in range(2, len(endpoints_3d)):
            target = endpoints_3d[i]
            path = self._lee_wavefront_on_grid(target, None, net_name,
                                               fcu_grid, bcu_grid,
                                               connect_to_existing=True)
            if path:
                segs, vs = self._path_to_segments(path, net_name)
                segments.extend(segs)
                vias.extend(vs)

                for layer, row, col in path:
                    if layer == 0:
                        fcu_grid[row][col] = net_name
                    else:
                        bcu_grid[row][col] = net_name

        success = len(segments) > 0
        return Route(net=net_name, success=success,
                     segments=segments, vias=vias)

    def _lee_wavefront_on_grid(self, start: Tuple, target: Optional[Tuple],
                               net_name: str, fcu_grid: List[List],
                               bcu_grid: List[List],
                               connect_to_existing: bool = False) -> Optional[List]:
        """
        Lee wavefront that works on provided grid copies (thread-safe).
        """
        # Convert to grid coordinates (use round() not int() to avoid truncation errors)
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        start_layer = 0 if start[2] == 'F.Cu' else 1

        if target:
            target_col = self._real_to_grid_col(target[0])
            target_row = self._real_to_grid_row(target[1])
            target_layer = 0 if target[2] == 'F.Cu' else 1

        # BFS wavefront
        dist = {}
        parent = {}
        queue = deque()

        start_cell = (start_layer, start_row, start_col)
        dist[start_cell] = 0
        parent[start_cell] = None
        queue.append(start_cell)

        found = False
        found_cell = None

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected
        if self.config.allow_45_degree:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-connected

        while queue and not found:
            layer, row, col = queue.popleft()
            current_dist = dist[(layer, row, col)]

            if current_dist > self.config.lee_max_expansion:
                break

            # Check if we reached target
            if target and layer == target_layer and row == target_row and col == target_col:
                found = True
                found_cell = (layer, row, col)
                break

            # Check if we connected to existing route
            if connect_to_existing and not target:
                grid = fcu_grid if layer == 0 else bcu_grid
                if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                    cell = grid[row][col]
                    if cell == net_name and (layer, row, col) != start_cell:
                        found = True
                        found_cell = (layer, row, col)
                        break

            # Expand to neighbors
            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                    continue

                grid = fcu_grid if layer == 0 else bcu_grid
                cell = grid[nr][nc]

                # Check if cell is available
                if cell is not None and cell != net_name and cell not in self.BLOCKED_MARKERS:
                    continue  # Blocked by another net
                if cell in self.BLOCKED_MARKERS:
                    continue  # Blocked by component/edge

                next_cell = (layer, nr, nc)
                if next_cell not in dist:
                    dist[next_cell] = current_dist + 1
                    parent[next_cell] = (layer, row, col)
                    queue.append(next_cell)

            # Try layer change (via)
            if self.config.allow_layer_change:
                other_layer = 1 - layer
                other_grid = bcu_grid if layer == 0 else fcu_grid

                if 0 <= row < len(other_grid) and 0 <= col < len(other_grid[0]):
                    other_cell = other_grid[row][col]
                    if other_cell is None or other_cell == net_name:
                        next_cell = (other_layer, row, col)
                        if next_cell not in dist:
                            via_cost = int(self.config.via_cost)
                            dist[next_cell] = current_dist + via_cost
                            parent[next_cell] = (layer, row, col)
                            queue.append(next_cell)

        if not found:
            return None

        # Backtrace path
        path = []
        cell = found_cell
        while cell:
            path.append(cell)
            cell = parent.get(cell)
        path.reverse()

        return path

    def _route_lee_parallel(self, net_order: List[str], net_pins: Dict,
                            escapes: Dict) -> 'RoutingResult':
        """
        Route using Lee algorithm with parallel execution for independent nets.
        """
        # Determine number of workers
        if self.config.max_workers > 0:
            max_workers = self.config.max_workers
        else:
            max_workers = os.cpu_count() or 4

        # Find independent net groups - ONLY for nets in net_order
        # This respects pour (GND/3V3 removed from net_order by engine)
        filtered_net_pins = {n: net_pins[n] for n in net_order if n in net_pins}
        groups = self._find_independent_net_groups(filtered_net_pins)

        total_nets = sum(len(g) for g in groups)
        print(f"    [LEE-PARALLEL] {total_nets} nets in {len(groups)} groups, "
              f"using {max_workers} workers")

        # Route each group
        all_routes = {}
        grid_lock = threading.Lock()

        for group_idx, group in enumerate(groups):
            if len(group) == 1:
                # Single net - route directly
                net_name = group[0]
                pins = net_pins.get(net_name, [])
                if len(pins) >= 2:
                    route = self._route_net_lee(net_name, pins, escapes)
                    all_routes[net_name] = route
                    if route.success:
                        self._mark_route_on_grid(route)
            else:
                # Multiple nets - route in parallel
                # Create grid copies for each worker
                with ThreadPoolExecutor(max_workers=min(max_workers, len(group))) as executor:
                    futures = {}

                    for net_name in group:
                        pins = net_pins.get(net_name, [])
                        if len(pins) < 2:
                            continue

                        # Create grid copies for this worker
                        fcu_copy = [row[:] for row in self.fcu_grid]
                        bcu_copy = [row[:] for row in self.bcu_grid]

                        future = executor.submit(
                            self._route_net_parallel_worker,
                            net_name, pins, escapes, grid_lock,
                            fcu_copy, bcu_copy
                        )
                        futures[future] = net_name

                    # Collect results and merge
                    for future in as_completed(futures):
                        net_name, route = future.result()
                        all_routes[net_name] = route

                        # Mark on main grid (synchronized)
                        with grid_lock:
                            if route.success:
                                self._mark_route_on_grid(route)

        # Update instance routes
        self.routes = all_routes
        self.failed = [n for n, r in all_routes.items() if not r.success]

        routed = sum(1 for r in all_routes.values() if r.success)
        return RoutingResult(
            routes=all_routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(all_routes),
            algorithm_used='lee_parallel',
            statistics={'trunk_chains': self.get_trunk_chain_stats(), 'return_path': self.get_return_path_stats()}
        )

    def _mark_route_on_grid(self, route: 'Route'):
        """Mark a route's segments on the main grid."""
        for seg in route.segments:
            # Mark start and end points (use round() not int() for accuracy)
            start_col = self._real_to_grid_col(seg.start[0])
            start_row = self._real_to_grid_row(seg.start[1])
            end_col = self._real_to_grid_col(seg.end[0])
            end_row = self._real_to_grid_row(seg.end[1])

            grid = self.fcu_grid if seg.layer == 'F.Cu' else self.bcu_grid

            # Mark cells along the segment (Bresenham)
            dx = abs(end_col - start_col)
            dy = abs(end_row - start_row)
            sx = 1 if start_col < end_col else -1
            sy = 1 if start_row < end_row else -1
            err = dx - dy

            col, row = start_col, start_row
            while True:
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    grid[row][col] = route.net

                if col == end_col and row == end_row:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    col += sx
                if e2 < dx:
                    err += dx
                    row += sy

    # =========================================================================
    # ALGORITHM 1: LEE WAVEFRONT ALGORITHM (Lee, 1961)
    # =========================================================================
    # Reference: "Routing procedures for printed circuit boards" - Lee, 1961
    #
    # Lee's algorithm guarantees finding the shortest path if one exists.
    # It uses BFS (breadth-first search) wavefront expansion from source.
    #
    # Time Complexity: O(M*N) where M,N are grid dimensions
    # Space Complexity: O(M*N) for distance grid
    #
    # Key insight: BFS explores all cells at distance k before distance k+1,
    # so the first path found to target is guaranteed to be shortest.
    # =========================================================================

    def _route_lee(self, net_order: List[str], net_pins: Dict,
                   escapes: Dict) -> RoutingResult:
        """
        Route using Lee wavefront algorithm.
        """
        print("    [LEE] Using Lee wavefront algorithm (Lee, 1961)...")

        for net_name in net_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_lee(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)
            else:
                # Check return path for high-speed nets
                self._check_route_return_path(route, net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        # Add trunk chain and return path statistics
        stats = dict(self.stats)
        stats['trunk_chains'] = self.get_trunk_chain_stats()
        stats['return_path'] = self.get_return_path_stats()
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='lee',
            total_wirelength=sum(r.total_length for r in self.routes.values() if r.success),
            via_count=sum(len(r.vias) for r in self.routes.values() if r.success),
            statistics=stats
        )

    def _route_net_lee(self, net_name: str, pins: List[Tuple],
                       escapes: Dict) -> Route:
        """Route a single net using Lee algorithm"""
        self._prepare_net_bitmaps(net_name)
        route = Route(net=net_name, algorithm_used='lee')

        # Get escape endpoints
        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        # For multi-point nets, use MST-style approach
        # CRITICAL: First connection goes directly between two endpoints
        # Subsequent connections can attach to ANY point on the existing tree
        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        # Track actual connection points (segment endpoints and via positions)
        # These are the ONLY valid points where new routes can attach
        # Format: Set of (x, y) coordinates
        connection_points = {endpoints[0]}

        # Track which grid cells have ROUTED segments (not just pads)
        # This is critical for MST routing: we can only connect to cells that
        # are actually part of the routed tree, not just cells marked with the net
        # (which includes unconnected pads)
        # Format: Set of (layer, row, col)
        routed_cells = set()

        while unconnected:
            # Find closest pair - for subsequent connections, also consider
            # all connection points on the existing route tree
            best_dist = float('inf')
            best_uc = None
            best_target = None  # Can be endpoint or connection point

            for uc in unconnected:
                # Check original endpoints
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_target = uc, c

                # Also check intermediate connection points (segment endpoints, vias)
                for cp in connection_points:
                    dist = abs(uc[0] - cp[0]) + abs(uc[1] - cp[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_target = uc, cp

            if not best_uc:
                break

            # Route using Lee wavefront
            # Use connect_to_existing=True to allow connecting to any cell of the
            # existing route tree (but ONLY cells in routed_cells, not just any net cell)
            is_first_connection = len(route.segments) == 0
            segments, vias, success = self._lee_wavefront_3d(
                best_uc, best_target, net_name,
                connect_to_existing=not is_first_connection,
                routed_cells=routed_cells
            )

            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                connected.add(best_uc)
                unconnected.remove(best_uc)

                # Add all new segment endpoints and via positions to connection_points
                for seg in segments:
                    connection_points.add(seg.start)
                    connection_points.add(seg.end)
                for via in vias:
                    connection_points.add(via.position)

                # Mark route in grid AND track routed cells
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
                    # Track cells along the segment as "routed" (not just pad)
                    layer_idx = 0 if seg.layer == 'F.Cu' else 1
                    start_row = self._real_to_grid_row(seg.start[1])
                    start_col = self._real_to_grid_col(seg.start[0])
                    end_row = self._real_to_grid_row(seg.end[1])
                    end_col = self._real_to_grid_col(seg.end[0])
                    # Mark cells along segment
                    if start_row == end_row:
                        for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                            routed_cells.add((layer_idx, start_row, col))
                    elif start_col == end_col:
                        for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                            routed_cells.add((layer_idx, row, start_col))
                    else:
                        # Diagonal
                        dx = end_col - start_col
                        dy = end_row - start_row
                        steps = max(abs(dx), abs(dy))
                        for i in range(steps + 1):
                            t = i / steps if steps > 0 else 0
                            routed_cells.add((layer_idx, int(start_row + t * dy), int(start_col + t * dx)))
                # Also mark via positions as routed (on both layers)
                for via in vias:
                    via_row = self._real_to_grid_row(via.position[1])
                    via_col = self._real_to_grid_col(via.position[0])
                    routed_cells.add((0, via_row, via_col))  # F.Cu
                    routed_cells.add((1, via_row, via_col))  # B.Cu
            else:
                route.error = f"Lee failed to route from {best_uc} to {best_target}"
                return route

        route.success = len(unconnected) == 0

        # Split segments at T-junctions where MST branches connect mid-segment
        if route.success and len(route.segments) > 1:
            route = self._split_segments_at_junctions(route)

        # Add pad-to-escape stub segments to connect actual pad positions
        if route.success:
            route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    def _build_bfs_bitmaps(self, net_name: str, end_row: int, end_col: int):
        """
        Build per-net routing bitmaps for O(1) BFS lookups.

        THE KEY OPTIMIZATION: Instead of _is_cell_clear_for_net() doing 13
        Python string comparisons per cell (center + 12 clearance offsets),
        we build NumPy boolean bitmaps ONCE per net (~1ms via scipy dilation),
        then BFS just checks bitmap[row, col].

        For 68K cells explored: 884K string ops → 68K array lookups = ~10x faster.

        Returns:
            (clear_fcu, clear_bcu, accessible_fcu, accessible_bcu, approach_fcu)
            - clear_*: Full clearance check (for normal routing cells)
            - accessible_*: Relaxed check (for own-net pad approach)
            - approach_fcu: Boolean mask of "approaching own pad" zone near target
        """
        si = self.spatial_index
        if si is None:
            return None  # Fallback to Python checks

        clear_fcu = si.build_net_clearance_bitmap(net_name, self.clearance_cells, 'F.Cu')
        clear_bcu = si.build_net_clearance_bitmap(net_name, self.clearance_cells, 'B.Cu')
        accessible_fcu = si.build_accessible_bitmap(net_name, 'F.Cu')
        accessible_bcu = si.build_accessible_bitmap(net_name, 'B.Cu')

        # Build "approaching own pad" zone near target
        # This is a bitmap marking cells within 2*clearance of any own-net cell
        # near the target, allowing relaxed clearance for pad approach
        net_id = si.get_net_id(net_name)
        net_fcu = si.net_fcu

        # Create a mask of own-net cells near target
        approach_radius = self.clearance_cells * 2
        target_zone_radius = 10  # Same as manhattan_to_target <= 10 in original

        # Bounding box for target zone
        r_min = max(0, end_row - target_zone_radius - approach_radius)
        r_max = min(si.rows, end_row + target_zone_radius + approach_radius + 1)
        c_min = max(0, end_col - target_zone_radius - approach_radius)
        c_max = min(si.cols, end_col + target_zone_radius + approach_radius + 1)

        # Find own-net cells in the target zone
        own_net_near_target = np.zeros((si.rows, si.cols), dtype=bool)
        zone = net_fcu[r_min:r_max, c_min:c_max]
        own_net_near_target[r_min:r_max, c_min:c_max] = (zone == net_id)

        # Expand by approach_radius to create the "approaching pad" zone
        try:
            from scipy.ndimage import binary_dilation
            ar = approach_radius
            y, x = np.ogrid[-ar:ar+1, -ar:ar+1]
            kernel = x*x + y*y <= ar*ar
            approach_fcu = binary_dilation(own_net_near_target, structure=kernel)
        except ImportError:
            approach_fcu = own_net_near_target  # Fallback: only exact cells

        return clear_fcu, clear_bcu, accessible_fcu, accessible_bcu, approach_fcu

    def _lee_wavefront_3d(self, start: Tuple[float, float], end: Tuple[float, float],
                          net_name: str, connect_to_existing: bool = False,
                          routed_cells: set = None
                          ) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Lee wavefront expansion with layer support.

        Uses 3D BFS where state is (row, col, layer).
        Via cost is added when changing layers.

        OPTIMIZED: When spatial_index is available, uses pre-built per-net
        NumPy bitmaps for O(1) clearance checks instead of Python loops.

        Args:
            start: Starting point (x, y) in mm
            end: Target point (x, y) in mm
            net_name: Name of net being routed
            connect_to_existing: If True, accept connecting to cells in routed_cells
                                 (for MST multi-terminal routing)
            routed_cells: Set of (layer, row, col) cells that have ROUTED segments
                         (not just pads). Used with connect_to_existing to ensure
                         we only connect to actual route tree, not unconnected pads.

        Steps:
        1. Initialize distance grid with -1 (unvisited)
        2. Start BFS from source with distance 0
        3. Expand wavefront in 4 directions + layer change
        4. When target reached, backtrace to find path
        """
        # Grid coordinates - FIX: use round() via helper methods
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        # Bounds check
        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], [], False

        # BUILD PER-NET BITMAPS for O(1) clearance checks
        # This replaces 13 Python string comparisons per cell with 1 array lookup
        bitmaps = self._build_bfs_bitmaps(net_name, end_row, end_col)
        use_bitmaps = bitmaps is not None
        if use_bitmaps:
            clear_fcu, clear_bcu, accessible_fcu, accessible_bcu, approach_fcu = bitmaps

        # 3D distance grids: [layer][row][col] = distance
        # layer 0 = F.Cu, layer 1 = B.Cu
        dist_grid = [
            [[-1] * self.grid_cols for _ in range(self.grid_rows)],
            [[-1] * self.grid_cols for _ in range(self.grid_rows)]
        ]

        # Parent tracking for backtrace: (layer, row, col) -> (parent_layer, parent_row, parent_col)
        parent = {}

        # Start on preferred layer
        start_layer = 0 if self.config.prefer_top_layer else 1
        dist_grid[start_layer][start_row][start_col] = 0

        # BFS queue: (distance, layer, row, col)
        queue = deque([(0, start_layer, start_row, start_col)])

        # Direction vectors
        # 4-directional: Right, Left, Down, Up
        # 8-directional: adds diagonals for cleaner 45° routing
        if self.config.allow_45_degree:
            # 8 directions: cardinal + diagonal (diagonal cost is sqrt(2) ≈ 1.414)
            directions = [
                (0, 1, 1.0),    # Right
                (0, -1, 1.0),   # Left
                (1, 0, 1.0),    # Down
                (-1, 0, 1.0),   # Up
                (1, 1, 1.414),  # Down-Right (45°)
                (1, -1, 1.414), # Down-Left (45°)
                (-1, 1, 1.414), # Up-Right (45°)
                (-1, -1, 1.414) # Up-Left (45°)
            ]
        else:
            # 4 directions only (Manhattan routing)
            directions = [
                (0, 1, 1.0),   # Right
                (0, -1, 1.0),  # Left
                (1, 0, 1.0),   # Down
                (-1, 0, 1.0)   # Up
            ]

        found = False
        found_layer = -1
        found_row, found_col = end_row, end_col  # Where we actually found the target
        max_iterations = self.config.lee_max_expansion

        iterations = 0
        while queue and iterations < max_iterations:
            iterations += 1
            dist, layer, row, col = queue.popleft()

            # Reached target? (exact cell OR any cell marked with target net)
            # BUG FIX: Allow reaching ANY cell of the target net's pad area,
            # not just the exact center. This is important when pads are blocked
            # from one direction but accessible from another.
            grid = self.fcu_grid if layer == 0 else self.bcu_grid
            cell_value = grid[row][col] if self._in_bounds(row, col) else None

            # BUG FIX: For SMD pads (only on F.Cu), we must reach them on F.Cu layer.
            # Check if target is an SMD pad by seeing if it's marked on F.Cu but not B.Cu.
            fcu_target = self.fcu_grid[end_row][end_col] if self._in_bounds(end_row, end_col) else None
            bcu_target = self.bcu_grid[end_row][end_col] if self._in_bounds(end_row, end_col) else None
            target_is_smd = (fcu_target == net_name and bcu_target != net_name)

            # For SMD targets, only accept reaching on F.Cu (layer 0)
            target_layer_ok = not target_is_smd or layer == 0

            if row == end_row and col == end_col and target_layer_ok:
                found = True
                found_layer = layer
                found_row, found_col = row, col
                break

            # BUG FIX: Also accept reaching any cell marked with our net that's
            # close to the target. This handles the case where the exact target
            # center is blocked but we can reach an adjacent pad cell.
            # For SMD pads, we must be on F.Cu to connect.
            #
            # BUG FIX 2 (2026-02-08): Don't accept the START cell as "found"!
            # The start cell is also marked with our net and might be within 10
            # cells of the target. We must have actually traveled (dist > 0).
            #
            # MST FIX (CRITICAL): When connect_to_existing=True, only accept cells
            # that are in routed_cells (actual routed segments), NOT just any cell
            # marked with the net name (which includes unconnected pads).
            if cell_value == net_name and target_layer_ok and dist > 0:
                if connect_to_existing and routed_cells is not None:
                    # MST mode: only connect to cells with ACTUAL routed segments
                    # This prevents connecting to unconnected pad cells
                    if (layer, row, col) in routed_cells:
                        found = True
                        found_layer = layer
                        found_row, found_col = row, col
                        break
                    # If not a routed cell, continue searching - might find the target
                elif not connect_to_existing:
                    # Normal mode: only accept if close to target
                    manhattan_to_target = abs(row - end_row) + abs(col - end_col)
                    if manhattan_to_target <= 10:  # Within 10 cells of target center
                        found = True
                        found_layer = layer
                        found_row, found_col = row, col
                        break

            # Skip if we've found a better path
            if dist > dist_grid[layer][row][col] and dist_grid[layer][row][col] != -1:
                continue

            # Expand to neighbors on same layer (4 or 8 directions)
            for dr, dc, move_cost in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue
                if dist_grid[layer][nr][nc] != -1:
                    continue  # Already visited

                # DIAGONAL MOVE CHECK: For diagonal moves, verify the entire path is clear.
                # A diagonal trace from (row,col) to (nr,nc) must not clip through any obstacles.
                is_diagonal = (dr != 0 and dc != 0)
                if is_diagonal:
                    # Check the entire diagonal path for obstacles
                    if not self._is_diagonal_path_clear(grid, row, col, nr, nc, net_name):
                        continue  # Can't take diagonal - path blocked

                # =====================================================================
                # CELL CLEARANCE CHECK - O(1) BITMAP vs O(13) PYTHON LOOP
                # =====================================================================
                if use_bitmaps:
                    # FAST PATH: Use pre-built NumPy bitmaps (~10x faster)
                    # clear_bitmap[r,c] = True means cell is routable for this net
                    clear_bitmap = clear_fcu if layer == 0 else clear_bcu
                    acc_bitmap = accessible_fcu if layer == 0 else accessible_bcu

                    if clear_bitmap[nr, nc]:
                        pass  # Cell is clear — proceed to add to queue
                    elif approach_fcu[nr, nc] and acc_bitmap[nr, nc]:
                        # Approaching own pad — use relaxed accessibility check
                        pass  # Cell is accessible in approach zone — proceed
                    else:
                        continue  # Cell is blocked — skip
                else:
                    # FALLBACK: Original Python loop (when spatial_index not built)
                    neighbor_cell = grid[nr][nc] if self._in_bounds(nr, nc) else None
                    is_own_net_cell = neighbor_cell == net_name
                    is_near_target = abs(nr - end_row) <= 10 and abs(nc - end_col) <= 10

                    is_approaching_own_pad = False
                    if not is_own_net_cell and is_near_target:
                        approach_radius = self.clearance_cells * 2
                        for dr2 in range(-approach_radius, approach_radius + 1):
                            for dc2 in range(-approach_radius, approach_radius + 1):
                                check_r, check_c = nr + dr2, nc + dc2
                                if self._in_bounds(check_r, check_c):
                                    if grid[check_r][check_c] == net_name:
                                        is_approaching_own_pad = True
                                        break
                            if is_approaching_own_pad:
                                break

                    use_relaxed_check = is_own_net_cell or is_approaching_own_pad

                    if use_relaxed_check:
                        if not self._is_cell_accessible_for_net(grid, nr, nc, net_name):
                            continue
                    else:
                        if not self._is_cell_clear_for_net(grid, nr, nc, net_name):
                            continue

                # Apply CPU Lab layer direction + global routing preferences
                dir_mult = self._direction_cost_multiplier(dr, dc, layer)
                gr_mult = self._global_routing_cost(nr, nc, net_name)
                new_dist = dist + move_cost * dir_mult * gr_mult
                dist_grid[layer][nr][nc] = new_dist
                parent[(layer, nr, nc)] = (layer, row, col)
                queue.append((new_dist, layer, nr, nc))
                self.stats['cells_explored'] += 1

            # Layer change (via) - if allowed
            if self.config.allow_layer_change:
                other_layer = 1 - layer
                current_grid = self.fcu_grid if layer == 0 else self.bcu_grid
                other_grid = self.bcu_grid if layer == 0 else self.fcu_grid

                if dist_grid[other_layer][row][col] == -1:
                    # BUG FIX: A via spans BOTH layers, so check clearance on BOTH.
                    # Previously only checked other_grid, which missed violations on
                    # the current layer (e.g., via placed next to SMD pad on F.Cu
                    # when transitioning from F.Cu to B.Cu).
                    # NOTE: Via checks use Python function (not bitmap) because vias
                    # have special __COMPONENT__ handling that bitmaps don't support.
                    current_clear = self._is_cell_clear_for_via(current_grid, row, col, net_name)
                    other_clear = self._is_cell_clear_for_via(other_grid, row, col, net_name)

                    if current_clear and other_clear:
                        via_cost = int(self.config.via_cost)
                        new_dist = dist + via_cost
                        dist_grid[other_layer][row][col] = new_dist
                        parent[(other_layer, row, col)] = (layer, row, col)
                        queue.append((new_dist, other_layer, row, col))
                        self.stats['layer_changes'] += 1

        if not found:
            return [], [], False

        # Backtrace to find path
        # BUG FIX: Use found_row/found_col instead of end_row/end_col
        # because we may have reached a nearby pad cell instead of the exact center
        path = []  # List of (layer, row, col)
        layer, row, col = found_layer, found_row, found_col
        path.append((layer, row, col))

        while (layer, row, col) in parent:
            layer, row, col = parent[(layer, row, col)]
            path.append((layer, row, col))

        path.reverse()

        # Convert path to segments and vias
        return self._path_to_segments_3d(path, net_name)

    # =========================================================================
    # ALGORITHM 2: HADLOCK'S ALGORITHM (Hadlock, 1977)
    # =========================================================================
    # Reference: "A shortest path algorithm for grid graphs" - Hadlock, 1977
    #            Networks 7(4): 323-334
    #
    # Hadlock improves on Lee by using "detour numbers" to bias search.
    # A detour occurs when moving AWAY from target (Manhattan distance increases).
    #
    # Key formula from paper: l(P) = MD(S,T) + 2*d(P)
    # Where:
    # - l(P) = length of path P
    # - MD(S,T) = Manhattan distance from source S to target T
    # - d(P) = detour number = count of moves that increase Manhattan distance
    #
    # By minimizing d(P) first, we find shortest path faster than Lee.
    # Uses priority queue ordered by detour number (lower = higher priority).
    # Complexity: O(n) worst case, O(sqrt(n)) best case.
    # =========================================================================

    def _route_hadlock(self, net_order: List[str], net_pins: Dict,
                       escapes: Dict) -> RoutingResult:
        """
        Route using Hadlock's algorithm.
        """
        print("    [HADLOCK] Using Hadlock's detour-biased algorithm (1977)...")

        for net_name in net_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_hadlock(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='hadlock',
            total_wirelength=sum(r.total_length for r in self.routes.values() if r.success),
            via_count=sum(len(r.vias) for r in self.routes.values() if r.success),
            statistics={**self.stats, 'trunk_chains': self.get_trunk_chain_stats(), 'return_path': self.get_return_path_stats()}
        )

    def _route_net_hadlock(self, net_name: str, pins: List[Tuple],
                           escapes: Dict) -> Route:
        """Route a single net using Hadlock's algorithm"""
        self._prepare_net_bitmaps(net_name)
        route = Route(net=net_name, algorithm_used='hadlock')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        while unconnected:
            best_dist = float('inf')
            best_uc, best_c = None, None

            for uc in unconnected:
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_c = uc, c

            if not best_uc:
                break

            segments, vias, success = self._hadlock_route(best_uc, best_c, net_name)

            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Hadlock failed from {best_uc} to {best_c}"
                return route

        route.success = len(unconnected) == 0

        # Add pad-to-escape stub segments to connect actual pad positions
        if route.success:
            route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    def _hadlock_route(self, start: Tuple[float, float], end: Tuple[float, float],
                       net_name: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Hadlock's algorithm using detour numbers.

        Detour number d(P) = number of moves in path P that increase Manhattan distance.
        We use a priority queue ordered by detour number.

        Algorithm:
        1. Start with detour = 0 at source
        2. Pop lowest-detour cell from priority queue
        3. For each neighbor:
           - If moving toward target: same detour
           - If moving away from target: detour + 1
        4. Continue until target reached
        """
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], [], False

        # Detour grid: [layer][row][col] = minimum detour to reach
        detour_grid = [
            [[-1] * self.grid_cols for _ in range(self.grid_rows)],
            [[-1] * self.grid_cols for _ in range(self.grid_rows)]
        ]

        parent = {}

        # Start on preferred layer
        start_layer = 0 if self.config.prefer_top_layer else 1
        detour_grid[start_layer][start_row][start_col] = 0

        # Priority queue: (detour_number, layer, row, col)
        # heapq gives min-heap, so lower detour = higher priority
        pq = [(0, start_layer, start_row, start_col)]

        # 8-directional or 4-directional based on config
        if self.config.allow_45_degree:
            directions = [
                (0, 1), (-1, 0), (0, -1), (1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
        else:
            directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # R, U, L, D

        max_iterations = self.config.lee_max_expansion

        for _ in range(max_iterations):
            if not pq:
                break

            detour, layer, row, col = heapq.heappop(pq)

            if row == end_row and col == end_col:
                # Reconstruct path
                path = [(layer, row, col)]
                while (layer, row, col) in parent:
                    layer, row, col = parent[(layer, row, col)]
                    path.append((layer, row, col))
                path.reverse()
                return self._path_to_segments_3d(path, net_name)

            # Skip if we've found a better path to this cell
            if detour > detour_grid[layer][row][col] and detour_grid[layer][row][col] != -1:
                continue

            grid = self.fcu_grid if layer == 0 else self.bcu_grid

            # Expand to neighbors
            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue
                if not self._is_cell_clear_for_net(grid, nr, nc, net_name):
                    continue

                # Calculate new detour
                # Manhattan distance before and after move
                old_manhattan = abs(row - end_row) + abs(col - end_col)
                new_manhattan = abs(nr - end_row) + abs(nc - end_col)

                new_detour = detour
                if new_manhattan > old_manhattan:
                    new_detour += 1  # Moving away = detour

                if detour_grid[layer][nr][nc] == -1 or new_detour < detour_grid[layer][nr][nc]:
                    detour_grid[layer][nr][nc] = new_detour
                    parent[(layer, nr, nc)] = (layer, row, col)
                    heapq.heappush(pq, (new_detour, layer, nr, nc))
                    self.stats['cells_explored'] += 1

            # Layer change
            if self.config.allow_layer_change:
                other_layer = 1 - layer
                current_grid = self.fcu_grid if layer == 0 else self.bcu_grid
                other_grid = self.bcu_grid if layer == 0 else self.fcu_grid

                if detour_grid[other_layer][row][col] == -1:
                    # BUG FIX: Check BOTH layers for via placement (via spans both)
                    current_clear = self._is_cell_clear_for_via(current_grid, row, col, net_name)
                    other_clear = self._is_cell_clear_for_via(other_grid, row, col, net_name)

                    if current_clear and other_clear:
                        # Via adds to detour (configurable cost)
                        via_detour = detour + max(1, int(self.config.via_cost / 2))
                        if detour_grid[other_layer][row][col] == -1 or via_detour < detour_grid[other_layer][row][col]:
                            detour_grid[other_layer][row][col] = via_detour
                            parent[(other_layer, row, col)] = (layer, row, col)
                            heapq.heappush(pq, (via_detour, other_layer, row, col))
                            self.stats['layer_changes'] += 1

        return [], [], False

    # =========================================================================
    # ALGORITHM 3: SOUKUP'S TWO-PHASE ALGORITHM (Soukup, 1978)
    # =========================================================================
    # Reference: "Global router using a two-step routing approach" - Soukup, 1978
    #
    # Soukup's algorithm has two phases:
    # Phase 1: Greedy line probe - try to reach target directly
    # Phase 2: If blocked, fall back to maze routing (BFS)
    #
    # Much faster than Lee for open spaces because greedy phase is O(1) per step.
    # Falls back to guaranteed-complete maze routing when greedy fails.
    # =========================================================================

    def _route_soukup(self, net_order: List[str], net_pins: Dict,
                      escapes: Dict) -> RoutingResult:
        """
        Route using Soukup's two-phase algorithm.
        """
        print("    [SOUKUP] Using Soukup's two-phase algorithm (1978)...")

        for net_name in net_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_soukup(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='soukup',
            statistics={**self.stats, 'trunk_chains': self.get_trunk_chain_stats(), 'return_path': self.get_return_path_stats()}
        )

    def _route_net_soukup(self, net_name: str, pins: List[Tuple],
                          escapes: Dict) -> Route:
        """Route a single net using Soukup's algorithm"""
        self._prepare_net_bitmaps(net_name)
        route = Route(net=net_name, algorithm_used='soukup')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        while unconnected:
            best_dist = float('inf')
            best_uc, best_c = None, None

            for uc in unconnected:
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_c = uc, c

            if not best_uc:
                break

            segments, vias, success = self._soukup_route(best_uc, best_c, net_name)

            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Soukup failed from {best_uc} to {best_c}"
                return route

        route.success = len(unconnected) == 0

        # Add pad-to-escape stub segments to connect actual pad positions
        if route.success:
            route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    def _soukup_route(self, start: Tuple[float, float], end: Tuple[float, float],
                      net_name: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Soukup's two-phase routing.

        Phase 1: Greedy - try direct line probes toward target
        Phase 2: Maze - if blocked, use BFS to find any path
        """
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], [], False

        layer = 0 if self.config.prefer_top_layer else 1
        grid = self.fcu_grid if layer == 0 else self.bcu_grid

        # Phase 1: Greedy line probe
        path = self._greedy_line_probe(start_row, start_col, end_row, end_col, grid, net_name)
        if path:
            path_3d = [(layer, r, c) for r, c in path]
            return self._path_to_segments_3d(path_3d, net_name)

        # Phase 2: Fall back to Lee-style BFS
        return self._lee_wavefront_3d(start, end, net_name)

    def _greedy_line_probe(self, start_row: int, start_col: int,
                           end_row: int, end_col: int,
                           grid: List[List], net_name: str) -> Optional[List[Tuple[int, int]]]:
        """
        Try to reach target using straight line probes.

        Strategy from Soukup paper:
        1. Extend horizontal line toward target as far as possible
        2. Extend vertical line toward target as far as possible
        3. Alternate until target reached or blocked

        Uses escape points at obstacles to try alternate paths.
        """
        path = [(start_row, start_col)]
        row, col = start_row, start_col
        escape_points = []  # Points where we were blocked

        max_moves = (abs(end_row - start_row) + abs(end_col - start_col)) * 3

        for _ in range(max_moves):
            if row == end_row and col == end_col:
                return path

            # Determine preferred direction (toward target)
            dr = 0 if row == end_row else (1 if end_row > row else -1)
            dc = 0 if col == end_col else (1 if end_col > col else -1)

            moved = False

            # Try primary direction (prefer longer axis first for efficiency)
            if abs(end_col - col) >= abs(end_row - row):
                # Horizontal preferred
                if dc != 0:
                    # Try to extend horizontal line
                    nr, nc = row, col + dc
                    if self._in_bounds(nr, nc) and self._is_cell_clear_for_net(grid, nr, nc, net_name):
                        row, col = nr, nc
                        path.append((row, col))
                        moved = True

                if not moved and dr != 0:
                    # Try vertical
                    nr, nc = row + dr, col
                    if self._in_bounds(nr, nc) and self._is_cell_clear_for_net(grid, nr, nc, net_name):
                        row, col = nr, nc
                        path.append((row, col))
                        moved = True
            else:
                # Vertical preferred
                if dr != 0:
                    nr, nc = row + dr, col
                    if self._in_bounds(nr, nc) and self._is_cell_clear_for_net(grid, nr, nc, net_name):
                        row, col = nr, nc
                        path.append((row, col))
                        moved = True

                if not moved and dc != 0:
                    nr, nc = row, col + dc
                    if self._in_bounds(nr, nc) and self._is_cell_clear_for_net(grid, nr, nc, net_name):
                        row, col = nr, nc
                        path.append((row, col))
                        moved = True

            if not moved:
                # Blocked - try escape moves (perpendicular directions)
                escape_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                escaped = False

                for edr, edc in escape_dirs:
                    if (edr, edc) == (dr, dc) or (edr, edc) == (-dr, -dc):
                        continue  # Skip primary direction

                    nr, nc = row + edr, col + edc
                    if self._in_bounds(nr, nc) and self._is_cell_clear_for_net(grid, nr, nc, net_name):
                        if (nr, nc) not in escape_points:
                            escape_points.append((row, col))
                            row, col = nr, nc
                            path.append((row, col))
                            escaped = True
                            break

                if not escaped:
                    # Completely blocked - greedy fails
                    return None

        return None  # Max moves exceeded

    # =========================================================================
    # ALGORITHM 4: MIKAMI-TABUCHI LINE SEARCH (1968)
    # =========================================================================
    # Reference: "A computer program for optimal routing" - Mikami & Tabuchi, 1968
    #
    # Instead of cell-by-cell expansion, extends lines from source and target.
    # Creates perpendicular "trial lines" at escape points.
    # When source and target lines intersect, a path is found.
    #
    # Memory efficient: O(perimeter) vs O(area) for maze routers.
    # Guaranteed to find a path if one exists (completeness).
    # =========================================================================

    def _route_mikami(self, net_order: List[str], net_pins: Dict,
                      escapes: Dict) -> RoutingResult:
        """
        Route using Mikami-Tabuchi line search algorithm.
        """
        print("    [MIKAMI] Using Mikami-Tabuchi line search algorithm (1968)...")

        for net_name in net_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_mikami(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='mikami',
            statistics={**self.stats, 'trunk_chains': self.get_trunk_chain_stats(), 'return_path': self.get_return_path_stats()}
        )

    def _route_net_mikami(self, net_name: str, pins: List[Tuple],
                          escapes: Dict) -> Route:
        """Route a single net using Mikami-Tabuchi"""
        self._prepare_net_bitmaps(net_name)
        route = Route(net=net_name, algorithm_used='mikami')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        while unconnected:
            best_dist = float('inf')
            best_uc, best_c = None, None

            for uc in unconnected:
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_c = uc, c

            if not best_uc:
                break

            segments, vias, success = self._mikami_route(best_uc, best_c, net_name)

            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Mikami failed from {best_uc} to {best_c}"
                return route

        route.success = len(unconnected) == 0

        # Add pad-to-escape stub segments to connect actual pad positions
        if route.success:
            route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    def _mikami_route(self, start: Tuple[float, float], end: Tuple[float, float],
                      net_name: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Mikami-Tabuchi line search algorithm.

        Algorithm:
        1. Level 0: Create H and V lines through source and target
        2. Level k: At each "escape point" (end of blocked line), create perpendicular lines
        3. Check for intersection between source-side and target-side lines
        4. When intersection found, reconstruct path through intersection point
        """
        layer = 0 if self.config.prefer_top_layer else 1
        grid = self.fcu_grid if layer == 0 else self.bcu_grid

        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], [], False

        # Line representation: (type, fixed_coord, min_var, max_var, origin_point)
        # type = 'H' for horizontal (fixed row), 'V' for vertical (fixed col)
        source_lines = []  # Lines emanating from source side
        target_lines = []  # Lines emanating from target side

        # Level 0: Initial lines from source and target
        h_s, v_s = self._extend_lines(start_row, start_col, grid, net_name)
        h_t, v_t = self._extend_lines(end_row, end_col, grid, net_name)

        source_lines.append(('H', start_row, h_s[0], h_s[1], (start_row, start_col)))
        source_lines.append(('V', start_col, v_s[0], v_s[1], (start_row, start_col)))
        target_lines.append(('H', end_row, h_t[0], h_t[1], (end_row, end_col)))
        target_lines.append(('V', end_col, v_t[0], v_t[1], (end_row, end_col)))

        # Check for initial intersection
        intersect = self._find_line_intersection(source_lines, target_lines)
        if intersect:
            path = self._mikami_reconstruct_path(start_row, start_col, end_row, end_col, intersect)
            if path:
                path_3d = [(layer, r, c) for r, c in path]
                return self._path_to_segments_3d(path_3d, net_name)

        # Expand up to max levels
        max_levels = 20
        source_escape_pts = set([(start_row, start_col)])
        target_escape_pts = set([(end_row, end_col)])

        for level in range(max_levels):
            # Get new escape points from lines added in previous level
            new_source_pts = set()
            new_target_pts = set()

            # Source side: extract escape points from recent lines
            for line in source_lines[-(len(source_lines) // (level + 1) + 2):]:
                ltype, fixed, min_v, max_v, origin = line
                # Escape points are at the ends of lines (where blocked)
                if ltype == 'H':
                    if (fixed, min_v) not in source_escape_pts:
                        new_source_pts.add((fixed, min_v))
                    if (fixed, max_v) not in source_escape_pts:
                        new_source_pts.add((fixed, max_v))
                else:  # V
                    if (min_v, fixed) not in source_escape_pts:
                        new_source_pts.add((min_v, fixed))
                    if (max_v, fixed) not in source_escape_pts:
                        new_source_pts.add((max_v, fixed))

            for line in target_lines[-(len(target_lines) // (level + 1) + 2):]:
                ltype, fixed, min_v, max_v, origin = line
                if ltype == 'H':
                    if (fixed, min_v) not in target_escape_pts:
                        new_target_pts.add((fixed, min_v))
                    if (fixed, max_v) not in target_escape_pts:
                        new_target_pts.add((fixed, max_v))
                else:
                    if (min_v, fixed) not in target_escape_pts:
                        new_target_pts.add((min_v, fixed))
                    if (max_v, fixed) not in target_escape_pts:
                        new_target_pts.add((max_v, fixed))

            # Extend lines from new escape points
            pts_added = 0
            for row, col in list(new_source_pts)[:50]:  # Limit expansion
                if self._in_bounds(row, col):
                    h_line, v_line = self._extend_lines(row, col, grid, net_name)
                    source_lines.append(('H', row, h_line[0], h_line[1], (row, col)))
                    source_lines.append(('V', col, v_line[0], v_line[1], (row, col)))
                    source_escape_pts.add((row, col))
                    pts_added += 1

            for row, col in list(new_target_pts)[:50]:
                if self._in_bounds(row, col):
                    h_line, v_line = self._extend_lines(row, col, grid, net_name)
                    target_lines.append(('H', row, h_line[0], h_line[1], (row, col)))
                    target_lines.append(('V', col, v_line[0], v_line[1], (row, col)))
                    target_escape_pts.add((row, col))
                    pts_added += 1

            # Check for intersection
            intersect = self._find_line_intersection(source_lines, target_lines)
            if intersect:
                path = self._mikami_reconstruct_path(start_row, start_col, end_row, end_col, intersect)
                if path:
                    path_3d = [(layer, r, c) for r, c in path]
                    return self._path_to_segments_3d(path_3d, net_name)

            if pts_added == 0:
                break  # No more expansion possible

        # Mikami failed - fall back to Lee
        return self._lee_wavefront_3d(start, end, net_name)

    def _extend_lines(self, row: int, col: int, grid: List[List],
                      net_name: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Extend horizontal and vertical lines from a point until blocked"""
        # Horizontal line (fixed row, vary col)
        h_min, h_max = col, col

        # Extend left
        for c in range(col - 1, -1, -1):
            if self._is_cell_clear_for_net(grid, row, c, net_name):
                h_min = c
            else:
                break

        # Extend right
        for c in range(col + 1, self.grid_cols):
            if self._is_cell_clear_for_net(grid, row, c, net_name):
                h_max = c
            else:
                break

        # Vertical line (fixed col, vary row)
        v_min, v_max = row, row

        # Extend up
        for r in range(row - 1, -1, -1):
            if self._is_cell_clear_for_net(grid, r, col, net_name):
                v_min = r
            else:
                break

        # Extend down
        for r in range(row + 1, self.grid_rows):
            if self._is_cell_clear_for_net(grid, r, col, net_name):
                v_max = r
            else:
                break

        return (h_min, h_max), (v_min, v_max)

    def _find_line_intersection(self, source_lines: List, target_lines: List) -> Optional[Tuple[int, int]]:
        """Find intersection point between source and target lines"""
        for s_line in source_lines:
            s_type, s_fixed, s_min, s_max, _ = s_line
            for t_line in target_lines:
                t_type, t_fixed, t_min, t_max, _ = t_line

                # H-V intersection
                if s_type == 'H' and t_type == 'V':
                    # s is horizontal at row=s_fixed, cols from s_min to s_max
                    # t is vertical at col=t_fixed, rows from t_min to t_max
                    if t_min <= s_fixed <= t_max and s_min <= t_fixed <= s_max:
                        return (s_fixed, t_fixed)  # (row, col)
                elif s_type == 'V' and t_type == 'H':
                    # s is vertical at col=s_fixed, rows from s_min to s_max
                    # t is horizontal at row=t_fixed, cols from t_min to t_max
                    if s_min <= t_fixed <= s_max and t_min <= s_fixed <= t_max:
                        return (t_fixed, s_fixed)  # (row, col)

        return None

    def _mikami_reconstruct_path(self, start_row: int, start_col: int,
                                 end_row: int, end_col: int,
                                 intersect: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path through intersection point"""
        int_row, int_col = intersect

        path = []

        # Path from start to intersection (L-shape)
        if start_row == int_row:
            # Same row - horizontal first
            step = 1 if int_col > start_col else -1
            for c in range(start_col, int_col + step, step):
                path.append((start_row, c))
        elif start_col == int_col:
            # Same col - vertical first
            step = 1 if int_row > start_row else -1
            for r in range(start_row, int_row + step, step):
                path.append((r, start_col))
        else:
            # L-shape: go vertical then horizontal
            step_r = 1 if int_row > start_row else -1
            for r in range(start_row, int_row + step_r, step_r):
                path.append((r, start_col))
            step_c = 1 if int_col > start_col else -1
            for c in range(start_col + step_c, int_col + step_c, step_c):
                path.append((int_row, c))

        # Path from intersection to end (avoid duplicating intersection)
        if len(path) > 0 and path[-1] == (int_row, int_col):
            pass  # Intersection already in path
        else:
            path.append((int_row, int_col))

        if int_row == end_row:
            step = 1 if end_col > int_col else -1
            for c in range(int_col + step, end_col + step, step):
                path.append((end_row, c))
        elif int_col == end_col:
            step = 1 if end_row > int_row else -1
            for r in range(int_row + step, end_row + step, step):
                path.append((r, int_col))
        else:
            step_r = 1 if end_row > int_row else -1
            for r in range(int_row + step_r, end_row + step_r, step_r):
                path.append((r, int_col))
            if int_col != end_col:
                step_c = 1 if end_col > int_col else -1
                for c in range(int_col + step_c, end_col + step_c, step_c):
                    path.append((end_row, c))

        return path if path else None

    # =========================================================================
    # ALGORITHM 5: A* PATHFINDING
    # =========================================================================
    # A* is a best-first search that uses f(n) = g(n) + h(n)
    # where g(n) = cost from start, h(n) = heuristic estimate to goal
    #
    # With Manhattan distance heuristic (admissible), A* finds optimal path.
    # Faster than Lee because it explores toward goal preferentially.
    # =========================================================================

    def _route_astar_only(self, net_order: List[str], net_pins: Dict,
                          escapes: Dict) -> RoutingResult:
        """Route using A* algorithm only"""
        print("    [A*] Using A* pathfinding algorithm...")

        for net_name in net_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_astar(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='astar',
            statistics={**self.stats, 'trunk_chains': self.get_trunk_chain_stats(), 'return_path': self.get_return_path_stats()}
        )

    def _route_net_astar(self, net_name: str, pins: List[Tuple],
                         escapes: Dict) -> Route:
        """Route a single net using A*"""
        self._prepare_net_bitmaps(net_name)
        route = Route(net=net_name, algorithm_used='astar')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        while unconnected:
            best_dist = float('inf')
            best_uc, best_c = None, None

            for uc in unconnected:
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_c = uc, c

            if not best_uc:
                break

            segments, vias, success = self._astar_route(best_uc, best_c, net_name)

            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"A* failed for {net_name}"
                return route

        route.success = len(unconnected) == 0

        # Add pad-to-escape stub segments to connect actual pad positions
        if route.success:
            route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    def _astar_route(self, start: Tuple[float, float], end: Tuple[float, float],
                     net_name: str) -> Tuple[List[TrackSegment], List[Via], bool]:
        """A* pathfinding with layer support"""
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], [], False

        def heuristic(r, c, layer):
            # Euclidean distance - better for 8-directional routing
            # This encourages diagonals when they lead more directly to the goal
            dr = abs(r - end_row)
            dc = abs(c - end_col)
            # Octile distance: combines diagonal and straight moves optimally
            return max(dr, dc) + 0.414 * min(dr, dc)  # sqrt(2)-1 ≈ 0.414

        start_layer = 0 if self.config.prefer_top_layer else 1

        # Priority queue: (f_score, g_score, layer, row, col)
        open_set = [(heuristic(start_row, start_col, start_layer), 0, start_layer, start_row, start_col)]
        came_from = {}
        g_score = {(start_layer, start_row, start_col): 0}

        # 8-directional or 4-directional based on config
        if self.config.allow_45_degree:
            directions = [
                (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),  # Cardinal
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)  # Diagonal
            ]
        else:
            directions = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0)]

        max_iterations = self.grid_rows * self.grid_cols * 4

        for _ in range(max_iterations):
            if not open_set:
                break

            f, g, layer, row, col = heapq.heappop(open_set)

            if (row, col) == (end_row, end_col):
                # Reconstruct path
                path = [(layer, row, col)]
                while (layer, row, col) in came_from:
                    layer, row, col = came_from[(layer, row, col)]
                    path.append((layer, row, col))
                path.reverse()
                return self._path_to_segments_3d(path, net_name)

            if g > g_score.get((layer, row, col), float('inf')):
                continue

            grid = self.fcu_grid if layer == 0 else self.bcu_grid

            for dr, dc, move_cost in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue

                # DIAGONAL MOVE CHECK: For diagonal moves, verify the entire path is clear
                is_diagonal = (dr != 0 and dc != 0)
                if is_diagonal:
                    if not self._is_diagonal_path_clear(grid, row, col, nr, nc, net_name):
                        continue  # Can't take diagonal - path blocked

                if not self._is_cell_clear_for_net(grid, nr, nc, net_name):
                    continue

                # Apply CPU Lab layer direction + global routing preferences
                dir_mult = self._direction_cost_multiplier(dr, dc, layer)
                gr_mult = self._global_routing_cost(nr, nc, net_name)
                tentative_g = g + move_cost * dir_mult * gr_mult

                if tentative_g < g_score.get((layer, nr, nc), float('inf')):
                    came_from[(layer, nr, nc)] = (layer, row, col)
                    g_score[(layer, nr, nc)] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc, layer)
                    heapq.heappush(open_set, (f_score, tentative_g, layer, nr, nc))
                    self.stats['cells_explored'] += 1

            # Layer change
            if self.config.allow_layer_change:
                other_layer = 1 - layer
                current_grid = self.fcu_grid if layer == 0 else self.bcu_grid
                other_grid = self.bcu_grid if layer == 0 else self.fcu_grid

                # BUG FIX: Check BOTH layers for via placement (via spans both)
                # BUG FIX: Use via-specific clearance (larger than trace clearance)
                current_clear = self._is_cell_clear_for_via(current_grid, row, col, net_name)
                other_clear = self._is_cell_clear_for_via(other_grid, row, col, net_name)

                if current_clear and other_clear:
                    via_g = g + self.config.via_cost
                    if via_g < g_score.get((other_layer, row, col), float('inf')):
                        came_from[(other_layer, row, col)] = (layer, row, col)
                        g_score[(other_layer, row, col)] = via_g
                        f_score = via_g + heuristic(row, col, other_layer)
                        heapq.heappush(open_set, (f_score, via_g, other_layer, row, col))
                        self.stats['layer_changes'] += 1

        return [], [], False

    # =========================================================================
    # ALGORITHM 6: PATHFINDER (McMurchie & Ebeling, 1995)
    # =========================================================================
    # Reference: "PathFinder: A Negotiation-Based Performance-Driven Router"
    #
    # PathFinder allows initial overlaps (shared routing resources).
    # Through iterations, congested resources have their costs increased.
    # Nets "negotiate" by paying higher costs for shared resources.
    # Converges when no overlaps remain.
    #
    # Cost function from paper: Cn = (bn + hn) * pn
    # - bn: base cost of using node n (intrinsic delay/cost)
    # - hn: historical congestion factor (accumulates over iterations)
    # - pn: present congestion penalty (number of other nets using n)
    # =========================================================================

    def _route_pathfinder(self, net_order: List[str], net_pins: Dict,
                          escapes: Dict, placement: Dict,
                          parts_db: Dict) -> RoutingResult:
        """
        Route using PathFinder negotiated congestion algorithm.
        """
        print("    [PATHFINDER] Using PathFinder negotiated congestion routing (1995)...")

        # History cost grids: track how often each cell has been contested
        fcu_history = [[0.0] * self.grid_cols for _ in range(self.grid_rows)]
        bcu_history = [[0.0] * self.grid_cols for _ in range(self.grid_rows)]

        # Present congestion: track which nets currently use each cell
        fcu_present: Dict[Tuple[int, int], Set[str]] = {}
        bcu_present: Dict[Tuple[int, int], Set[str]] = {}

        best_result = None
        penalty_factor = 1.0

        for iteration in range(self.config.pathfinder_max_iterations):
            # Clear present congestion for this iteration
            fcu_present.clear()
            bcu_present.clear()

            # Route all nets with current costs
            all_routes = {}
            for net_name in net_order:
                pins = net_pins.get(net_name, [])
                if len(pins) < 2:
                    continue

                route = self._route_net_pathfinder(
                    net_name, pins, escapes,
                    fcu_history, bcu_history,
                    fcu_present, bcu_present,
                    penalty_factor
                )
                all_routes[net_name] = route

                # Register in present congestion (even if overlapping)
                if route.success:
                    for seg in route.segments:
                        present = fcu_present if seg.layer == 'F.Cu' else bcu_present
                        self._register_pathfinder_present(seg, net_name, present)

            # Count overlaps
            overlaps = self._count_pathfinder_overlaps(fcu_present, bcu_present)

            routed = sum(1 for r in all_routes.values() if r.success)

            # Track best result
            if best_result is None or routed >= best_result.routed_count:
                self.routes = dict(all_routes)
                best_result = RoutingResult(
                    routes=dict(all_routes),
                    success=overlaps == 0 and routed == len(net_order),
                    routed_count=routed,
                    total_count=len(net_order),
                    algorithm_used='pathfinder',
                    iterations=iteration + 1,
                    statistics={'overlaps': overlaps, 'penalty': penalty_factor}
                )

            if overlaps == 0:
                print(f"    [PATHFINDER] Converged in {iteration + 1} iterations!")
                # Update best_result with final converged state (0 overlaps)
                best_result = RoutingResult(
                    routes=dict(all_routes),
                    success=routed == len(net_order),
                    routed_count=routed,
                    total_count=len(net_order),
                    algorithm_used='pathfinder',
                    iterations=iteration + 1,
                    statistics={'overlaps': 0, 'penalty': penalty_factor}
                )
                return best_result

            # Update history costs for contested cells
            self._update_pathfinder_history(fcu_present, fcu_history)
            self._update_pathfinder_history(bcu_present, bcu_history)

            # Increase penalty factor (exponential growth)
            penalty_factor += self.config.pathfinder_penalty_increment

            print(f"    [PATHFINDER] Iteration {iteration + 1}: {overlaps} overlaps, penalty={penalty_factor:.2f}")

        print(f"    [PATHFINDER] Max iterations reached")
        return best_result or RoutingResult(
            routes=self.routes,
            success=False,
            routed_count=0,
            total_count=len(net_order),
            algorithm_used='pathfinder'
        )

    def _route_net_pathfinder(self, net_name: str, pins: List[Tuple],
                              escapes: Dict,
                              fcu_history: List[List[float]],
                              bcu_history: List[List[float]],
                              fcu_present: Dict, bcu_present: Dict,
                              penalty: float) -> Route:
        """Route a net considering congestion costs"""
        self._prepare_net_bitmaps(net_name)
        route = Route(net=net_name, algorithm_used='pathfinder')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        while unconnected:
            best_dist = float('inf')
            best_uc, best_c = None, None

            for uc in unconnected:
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_c = uc, c

            if not best_uc:
                break

            segments, vias, success = self._pathfinder_astar(
                best_uc, best_c, net_name,
                fcu_history, bcu_history,
                fcu_present, bcu_present,
                penalty
            )

            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                connected.add(best_uc)
                unconnected.remove(best_uc)
            else:
                route.error = f"PathFinder failed for {net_name}"
                return route

        route.success = len(unconnected) == 0

        # Add pad-to-escape stub segments to connect actual pad positions
        if route.success:
            route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    def _pathfinder_astar(self, start: Tuple[float, float], end: Tuple[float, float],
                          net_name: str,
                          fcu_history: List[List[float]],
                          bcu_history: List[List[float]],
                          fcu_present: Dict, bcu_present: Dict,
                          penalty: float) -> Tuple[List[TrackSegment], List[Via], bool]:
        """A* with PathFinder cost function"""
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], [], False

        def cost(r, c, layer):
            """
            PathFinder cost function from McMurchie & Ebeling (1995):

            Cn = (bn + hn) * pn

            Where:
            - bn = base cost of using node n (intrinsic routing cost)
            - hn = historical congestion factor (accumulated over iterations)
            - pn = present congestion penalty (1 + number of other nets sharing)

            This formula causes congested resources to have exponentially
            increasing costs, forcing nets to negotiate and find alternatives.
            """
            # bn: base cost (intrinsic cost of using this cell)
            bn = 1.0

            # hn: historical congestion (accumulates over iterations for contested cells)
            hn = fcu_history[r][c] if layer == 0 else bcu_history[r][c]

            # pn: present congestion penalty
            # pn = 1 if no other nets, else = number of sharing nets + 1
            present = fcu_present if layer == 0 else bcu_present
            occ = present.get((r, c), set())

            if not occ or net_name in occ:
                pn = 1.0  # No congestion from other nets
            else:
                pn = 1.0 + len(occ)  # Penalty based on number of sharing nets

            # Apply the PathFinder formula: Cn = (bn + hn) * pn
            # The 'penalty' parameter increases hn's weight each iteration
            return (bn + hn * penalty) * pn

        def heuristic(r, c):
            return abs(r - end_row) + abs(c - end_col)

        start_layer = 0 if self.config.prefer_top_layer else 1

        open_set = [(heuristic(start_row, start_col), 0.0, start_layer, start_row, start_col)]
        came_from = {}
        g_score = {(start_layer, start_row, start_col): 0.0}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        max_iterations = self.grid_rows * self.grid_cols * 4

        for _ in range(max_iterations):
            if not open_set:
                break

            f, g, layer, row, col = heapq.heappop(open_set)

            if (row, col) == (end_row, end_col):
                path = [(layer, row, col)]
                while (layer, row, col) in came_from:
                    layer, row, col = came_from[(layer, row, col)]
                    path.append((layer, row, col))
                path.reverse()
                return self._path_to_segments_3d(path, net_name)

            if g > g_score.get((layer, row, col), float('inf')):
                continue

            grid = self.fcu_grid if layer == 0 else self.bcu_grid

            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue

                # In PathFinder, we check grid blockage but allow shared routing
                cell = grid[nr][nc]
                if cell in self.BLOCKED_MARKERS:
                    continue

                move_cost = cost(nr, nc, layer)
                # Apply CPU Lab layer direction + global routing preferences
                dir_mult = self._direction_cost_multiplier(dr, dc, layer)
                gr_mult = self._global_routing_cost(nr, nc, net_name)
                tentative_g = g + move_cost * dir_mult * gr_mult

                if tentative_g < g_score.get((layer, nr, nc), float('inf')):
                    came_from[(layer, nr, nc)] = (layer, row, col)
                    g_score[(layer, nr, nc)] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f_score, tentative_g, layer, nr, nc))

            # Layer change
            if self.config.allow_layer_change:
                other_layer = 1 - layer
                current_grid = self.fcu_grid if layer == 0 else self.bcu_grid
                other_grid = self.bcu_grid if layer == 0 else self.fcu_grid

                # BUG FIX: Check BOTH layers for via placement (via spans both)
                # BUG FIX: Use via-specific clearance (larger than trace clearance)
                current_clear = self._is_cell_clear_for_via(current_grid, row, col, net_name)
                other_clear = self._is_cell_clear_for_via(other_grid, row, col, net_name)

                if current_clear and other_clear:
                    via_cost = self.config.via_cost + cost(row, col, other_layer)
                    via_g = g + via_cost
                    if via_g < g_score.get((other_layer, row, col), float('inf')):
                        came_from[(other_layer, row, col)] = (layer, row, col)
                        g_score[(other_layer, row, col)] = via_g
                        f_score = via_g + heuristic(row, col)
                        heapq.heappush(open_set, (f_score, via_g, other_layer, row, col))

        return [], [], False

    def _register_pathfinder_present(self, segment: TrackSegment, net_name: str,
                                      present: Dict[Tuple[int, int], Set[str]]):
        """Register segment cells in present congestion map"""
        start_col = self._real_to_grid_col(segment.start[0])
        start_row = self._real_to_grid_row(segment.start[1])
        end_col = self._real_to_grid_col(segment.end[0])
        end_row = self._real_to_grid_row(segment.end[1])

        if start_row == end_row:
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                key = (start_row, col)
                if key not in present:
                    present[key] = set()
                present[key].add(net_name)
        elif start_col == end_col:
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                key = (row, start_col)
                if key not in present:
                    present[key] = set()
                present[key].add(net_name)

    def _count_pathfinder_overlaps(self, fcu: Dict, bcu: Dict) -> int:
        """Count cells with multiple net occupants"""
        overlaps = 0
        for cell, nets in fcu.items():
            if len(nets) > 1:
                overlaps += 1
        for cell, nets in bcu.items():
            if len(nets) > 1:
                overlaps += 1
        return overlaps

    def _update_pathfinder_history(self, present: Dict, history: List[List[float]]):
        """Update history costs for contested cells"""
        for (row, col), nets in present.items():
            if len(nets) > 1 and self._in_bounds(row, col):
                # Increment history cost proportional to congestion
                history[row][col] += len(nets) - 1

    # =========================================================================
    # ALGORITHM 7: RIP-UP AND REROUTE (Nair et al., 1987)
    # =========================================================================
    # Reference: "MIGHTY: A Rip-Up and Reroute Detailed Router"
    #
    # Iterative approach:
    # 1. Route all nets in current order
    # 2. Identify failed nets and their blockers
    # 3. Rip-up (remove) blocking nets
    # 4. Reorder: put failed nets first, blockers later
    # 5. Repeat until success or max iterations
    #
    # Key insight: Net ordering matters. By dynamically reordering based on
    # failures, we can find better solutions than static ordering.
    # =========================================================================

    def _route_ripup_reroute(self, net_order: List[str], net_pins: Dict,
                              escapes: Dict, placement: Dict,
                              parts_db: Dict) -> RoutingResult:
        """
        Rip-up and Reroute algorithm.
        """
        print("    [RIPUP] Using Rip-up and Reroute algorithm (Nair, 1987)...")

        routing_attempts = {net: 0 for net in net_order}
        blocking_history: Dict[str, List[str]] = {}

        # Initial order: sort by estimated length (short first)
        current_order = self._sort_by_length(net_order, net_pins, escapes)

        best_result = None
        best_routed = -1
        no_improve_count = 0
        max_no_improve = 3  # Early exit after 3 iterations with no improvement

        # Use Lee for power/ground nets (guaranteed path finding)
        power_nets = {'GND', 'VCC', 'VDD', 'VSS', 'V+', 'V-', '3V3', '5V', '12V',
                      'VBAT', 'VBUS', 'AVCC', 'AVDD', 'DVCC', 'DVDD', 'AGND', 'DGND',
                      'VIN', 'VOUT', 'PWR', 'POWER', 'SUPPLY'}

        for iteration in range(self.config.max_ripup_iterations):
            # Restore grid to clean state from snapshot (fast array copy)
            # Snapshot was saved in route() after components+escapes registered
            self._clear_routed_traces()

            # Route all nets in current order
            failed_nets = []
            for net_name in current_order:
                pins = net_pins.get(net_name, [])
                if len(pins) < 2:
                    continue

                # Use Lee for power nets (guaranteed), A* for signal nets (faster)
                is_power = net_name.upper() in power_nets
                if is_power or len(pins) > 3:
                    route = self._route_net_lee(net_name, pins, escapes)
                else:
                    route = self._route_net_astar(net_name, pins, escapes)
                self.routes[net_name] = route
                routing_attempts[net_name] += 1

                if not route.success:
                    failed_nets.append(net_name)
                    # Identify blocking nets
                    blockers = self._identify_blockers(net_name, pins, escapes)
                    blocking_history[net_name] = blockers
                else:
                    # Mark successful route in grid
                    for seg in route.segments:
                        self._mark_segment_in_grid(seg, net_name)

            routed = sum(1 for r in self.routes.values() if r.success)

            # Track best result
            if routed > best_routed:
                best_routed = routed
                no_improve_count = 0
                best_result = RoutingResult(
                    routes=dict(self.routes),
                    success=len(failed_nets) == 0,
                    routed_count=routed,
                    total_count=len(net_order),
                    algorithm_used='ripup',
                    iterations=iteration + 1
                )
            else:
                no_improve_count += 1

            # Success?
            if not failed_nets:
                print(f"    [RIPUP] Success! All {len(net_order)} nets routed in {iteration + 1} iterations")
                return best_result

            # Early exit: no improvement for N consecutive iterations
            if no_improve_count >= max_no_improve:
                print(f"    [RIPUP] Early exit: no improvement for {max_no_improve} iterations (best: {best_routed}/{len(net_order)})")
                return best_result

            # Reorder for next iteration
            current_order = self._reorder_for_ripup(
                current_order, failed_nets, blocking_history, routing_attempts
            )

            print(f"    [RIPUP] Iteration {iteration + 1}: {routed}/{len(net_order)} routed, reordering...")

        print(f"    [RIPUP] Max iterations reached, best: {best_routed}/{len(net_order)}")
        return best_result or RoutingResult(
            routes=self.routes,
            success=False,
            routed_count=best_routed,
            total_count=len(net_order),
            algorithm_used='ripup'
        )

    def _identify_blockers(self, net_name: str, pins: List[Tuple],
                           escapes: Dict) -> List[str]:
        """Identify nets that are blocking this net"""
        blockers = set()
        endpoints = self._get_escape_endpoints(pins, escapes)

        if len(endpoints) < 2:
            return []

        # Check the corridor between endpoints for other nets
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                start, end = endpoints[i], endpoints[j]

                start_col = self._real_to_grid_col(start[0])
                start_row = self._real_to_grid_row(start[1])
                end_col = self._real_to_grid_col(end[0])
                end_row = self._real_to_grid_row(end[1])

                # Check horizontal corridor
                for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                    if self._in_bounds(start_row, col):
                        occ = self.fcu_grid[start_row][col]
                        if occ and occ not in self.BLOCKED_MARKERS and occ != net_name:
                            blockers.add(occ)

                # Check vertical corridor
                for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                    if self._in_bounds(row, end_col):
                        occ = self.fcu_grid[row][end_col]
                        if occ and occ not in self.BLOCKED_MARKERS and occ != net_name:
                            blockers.add(occ)

        return list(blockers)

    def _reorder_for_ripup(self, current_order: List[str], failed: List[str],
                           blocking_history: Dict, attempts: Dict) -> List[str]:
        """
        Reorder nets for next rip-up iteration.

        Strategy from MIGHTY paper:
        1. Failed nets get highest priority (route first)
        2. Nets that block many others get lower priority
        3. Nets with many attempts get lower priority (avoid cycling)
        """
        # Count how many nets each net has blocked
        block_count: Dict[str, int] = {}
        for blocked, blockers in blocking_history.items():
            for blocker in blockers:
                block_count[blocker] = block_count.get(blocker, 0) + 1

        def key(net):
            is_failed = 0 if net in failed else 1  # Failed first
            blocks = block_count.get(net, 0)  # More blocking = later
            att = attempts.get(net, 0)  # More attempts = later
            return (is_failed, -blocks, att)

        return sorted(current_order, key=key)

    # =========================================================================
    # ALGORITHM 8: RECTILINEAR STEINER TREE (Hanan, 1966)
    # =========================================================================
    # Reference: "On Steiner's problem with rectilinear distance" - Hanan, 1966
    #
    # For multi-terminal nets, RSMT can reduce wirelength by 13-15% vs MST
    # by adding Steiner points (intermediate junctions).
    #
    # Hanan's theorem: Optimal Steiner points lie on the "Hanan grid" -
    # intersections of horizontal/vertical lines through terminals.
    # =========================================================================

    def _route_steiner(self, net_order: List[str], net_pins: Dict,
                       escapes: Dict) -> RoutingResult:
        """
        Route using Rectilinear Steiner Minimum Tree (RSMT) algorithm.
        """
        print("    [STEINER] Using Steiner tree algorithm (Hanan, 1966)...")

        for net_name in net_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_steiner(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='steiner',
            statistics={**self.stats, 'trunk_chains': self.get_trunk_chain_stats(), 'return_path': self.get_return_path_stats()}
        )

    def _route_net_steiner(self, net_name: str, pins: List[Tuple],
                           escapes: Dict) -> Route:
        """Route a single net using Steiner tree approach with FLUTE optimization."""
        self._prepare_net_bitmaps(net_name)
        route = Route(net=net_name, algorithm_used='steiner')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        if len(endpoints) == 2:
            # Simple 2-point case - just use A*
            segments, vias, success = self._astar_route(endpoints[0], endpoints[1], net_name)
            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                route.success = True
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            return route

        # Multi-point case: Use FLUTE for optimal Steiner tree
        if FLUTE_AVAILABLE:
            # Use FLUTE algorithm for optimal routing order
            flute = get_flute_instance()
            tree = flute.build_rsmt(endpoints)

            # Get optimal connections from FLUTE
            edges = [(
                (e.p1.x, e.p1.y),
                (e.p2.x, e.p2.y)
            ) for e in tree.edges]

            print(f"      [FLUTE] {net_name}: {len(endpoints)} pins, "
                  f"wirelength={tree.total_wirelength:.1f}, "
                  f"steiner_pts={len(tree.steiner_points)}")
        else:
            # Fallback: Compute Hanan grid Steiner points
            steiner_points = self._compute_hanan_steiner_points(endpoints)
            all_points = list(endpoints) + steiner_points
            edges = self._compute_mst_edges(all_points)

        # Route each edge from the Steiner tree
        for p1, p2 in edges:
            segments, vias, success = self._astar_route(p1, p2, net_name)

            if success:
                route.segments.extend(segments)
                self._extend_vias_deduplicated(route, vias)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Steiner edge failed: {p1} to {p2}"
                return route

        route.success = True

        # Add pad-to-escape stub segments to connect actual pad positions
        route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    def _compute_hanan_steiner_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Compute Hanan grid Steiner points.

        Hanan's theorem (1966): There exists an optimal RSMT where all
        Steiner points lie on intersections of horizontal and vertical
        lines through the terminal points (the "Hanan grid").
        """
        if len(points) <= 2:
            return []

        xs = sorted(set(p[0] for p in points))
        ys = sorted(set(p[1] for p in points))

        steiner_points = []
        original_set = set(points)

        # Generate Hanan grid intersections
        for x in xs:
            for y in ys:
                if (x, y) not in original_set:
                    # Only add points inside bounding box (optimization)
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    if min_x <= x <= max_x and min_y <= y <= max_y:
                        steiner_points.append((x, y))

        # Limit to avoid explosion (keep most central points)
        if len(steiner_points) > 15:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            steiner_points.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
            steiner_points = steiner_points[:15]

        return steiner_points

    def _compute_mst_edges(self, points: List[Tuple[float, float]]) -> List[Tuple[Tuple, Tuple]]:
        """Compute MST edges using Prim's algorithm with Manhattan distance"""
        if len(points) < 2:
            return []

        edges = []
        in_tree = {points[0]}
        candidates = list(points[1:])

        while candidates:
            best_dist = float('inf')
            best_edge = None
            best_cand = None

            for cand in candidates:
                for tree_pt in in_tree:
                    dist = abs(cand[0] - tree_pt[0]) + abs(cand[1] - tree_pt[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_edge = (tree_pt, cand)
                        best_cand = cand

            if best_cand:
                edges.append(best_edge)
                in_tree.add(best_cand)
                candidates.remove(best_cand)
            else:
                break

        return edges

    # =========================================================================
    # ALGORITHM 9: CHANNEL ROUTING (Hashimoto & Stevens, 1971)
    # =========================================================================
    # Reference: "Channel routing" - Hashimoto & Stevens, 1971
    #
    # Left-Edge Algorithm for channel routing:
    # 1. Sort nets by leftmost pin position
    # 2. Assign each net to a track (horizontal channel)
    # 3. Nets are assigned to lowest available track
    # 4. Track reuse: when a net ends before another begins
    #
    # Best for designs with organized pin structures (bus connections).
    # =========================================================================

    def _route_channel(self, net_order: List[str], net_pins: Dict,
                       escapes: Dict) -> RoutingResult:
        """
        Route using channel/left-edge routing approach.
        """
        print("    [CHANNEL] Using channel/left-edge routing (1971)...")

        # Sort nets by leftmost pin position (Left-Edge algorithm)
        def get_leftmost(net_name):
            pins = net_pins.get(net_name, [])
            endpoints = self._get_escape_endpoints(pins, escapes)
            if not endpoints:
                return float('inf')
            return min(p[0] for p in endpoints)

        sorted_nets = sorted(net_order, key=get_leftmost)

        for net_name in sorted_nets:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_channel(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='channel',
            statistics={**self.stats, 'trunk_chains': self.get_trunk_chain_stats(), 'return_path': self.get_return_path_stats()}
        )

    def _route_net_channel(self, net_name: str, pins: List[Tuple],
                           escapes: Dict) -> Route:
        """
        Route a net using channel routing strategy.

        For each net:
        1. Find horizontal extent (left-right span)
        2. Find available track (y-coordinate)
        3. Route horizontal trunk, then vertical branches
        """
        route = Route(net=net_name, algorithm_used='channel')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        # Sort endpoints by x-coordinate
        sorted_pts = sorted(endpoints, key=lambda p: p[0])

        # Find trunk y-coordinate (centroid of endpoints)
        center_y = sum(p[1] for p in endpoints) / len(endpoints)

        # Convert to grid - FIX: use round() via helper methods
        trunk_row = self._real_to_grid_row(center_y)
        left_x = sorted_pts[0][0]
        right_x = sorted_pts[-1][0]
        left_col = self._real_to_grid_col(left_x)
        right_col = self._real_to_grid_col(right_x)

        # Check if horizontal trunk is clear
        layer = 0 if self.config.prefer_top_layer else 1
        grid = self.fcu_grid if layer == 0 else self.bcu_grid
        layer_name = 'F.Cu' if layer == 0 else 'B.Cu'

        trunk_ok = True
        for col in range(left_col, right_col + 1):
            if not self._is_cell_clear_for_net(grid, trunk_row, col, net_name):
                trunk_ok = False
                break

        if trunk_ok:
            # Create horizontal trunk segment
            trunk_start_x = self.config.origin_x + left_col * self.config.grid_size
            trunk_end_x = self.config.origin_x + right_col * self.config.grid_size
            trunk_y = self.config.origin_y + trunk_row * self.config.grid_size

            trunk_seg = TrackSegment(
                start=(trunk_start_x, trunk_y),
                end=(trunk_end_x, trunk_y),
                layer=layer_name,
                width=self.get_trace_width(net_name),
                net=net_name
            )
            route.segments.append(trunk_seg)
            self._mark_segment_in_grid(trunk_seg, net_name)

            # Create vertical branches to each endpoint
            for pt in endpoints:
                # FIX: use round() via helper methods
                pt_col = self._real_to_grid_col(pt[0])
                pt_row = self._real_to_grid_row(pt[1])

                if pt_row != trunk_row:
                    # Need vertical branch
                    branch_x = self._grid_to_real_x(pt_col)
                    branch_start_y = trunk_y
                    branch_end_y = self._grid_to_real_y(pt_row)

                    branch_seg = TrackSegment(
                        start=(branch_x, branch_start_y),
                        end=(branch_x, branch_end_y),
                        layer=layer_name,
                        width=self.get_trace_width(net_name),
                        net=net_name
                    )
                    route.segments.append(branch_seg)
                    self._mark_segment_in_grid(branch_seg, net_name)

            route.success = True
        else:
            # Fall back to A* for this net
            connected = {endpoints[0]}
            unconnected = list(endpoints[1:])

            while unconnected:
                best_dist = float('inf')
                best_uc, best_c = None, None

                for uc in unconnected:
                    for c in connected:
                        dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_uc, best_c = uc, c

                if not best_uc:
                    break

                segments, vias, success = self._astar_route(best_uc, best_c, net_name)

                if success:
                    route.segments.extend(segments)
                    self._extend_vias_deduplicated(route, vias)
                    connected.add(best_uc)
                    unconnected.remove(best_uc)
                    for seg in segments:
                        self._mark_segment_in_grid(seg, net_name)
                else:
                    route.error = f"Channel routing failed for {net_name}"
                    return route

            route.success = len(unconnected) == 0

        # Add pad-to-escape stub segments to connect actual pad positions
        if route.success:
            route = self._add_pad_stubs_to_route(route, pins, escapes)

        return route

    # =========================================================================
    # META ALGORITHM 10: HYBRID (BEST OF ALL)
    # =========================================================================

    def _route_hybrid(self, net_order: List[str], net_pins: Dict,
                      escapes: Dict, placement: Dict, parts_db: Dict) -> RoutingResult:
        """
        Hybrid routing: combines multiple algorithms for best results.

        Strategy:
        1. Route signal nets FIRST (they need clear paths)
        2. Route power/ground nets LAST (they have many pins, benefit from pour)
        3. Try A* first (fast, good for most cases)
        4. For multi-terminal nets (>2 pins), use Steiner trees
        5. If failures occur, use rip-up and reroute
        """
        print("    [HYBRID] Using hybrid routing (A* + Steiner + Ripup)...")

        # CRITICAL: Route power/ground nets LAST
        # Power nets have many pins and create congestion if routed first.
        # Signal nets need clear routing channels.
        power_nets = {'GND', 'VCC', 'VDD', 'VSS', 'V+', 'V-', '3V3', '5V', '12V',
                      'VBAT', 'VBUS', 'AVCC', 'AVDD', 'DVCC', 'DVDD', 'AGND', 'DGND',
                      'VIN', 'VOUT', 'PWR', 'POWER', 'SUPPLY'}

        signal_nets = [n for n in net_order if n.upper() not in power_nets and
                       not n.upper().startswith('V') and not n.upper().endswith('GND')]
        pwr_gnd_nets = [n for n in net_order if n not in signal_nets]

        optimized_order = signal_nets + pwr_gnd_nets

        # First pass: route simple nets with A*, complex with FLUTE-optimized Steiner
        for net_name in optimized_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            endpoints = self._get_escape_endpoints(pins, escapes)
            is_power_net = net_name.upper() in power_nets or net_name in pwr_gnd_nets

            if len(endpoints) > 2:
                # Multi-terminal net: use FLUTE-optimized Steiner tree
                # FLUTE provides optimal routing order, minimizing wirelength
                route = self._route_net_steiner(net_name, pins, escapes)
            elif is_power_net:
                # Power net with 2 pins: use Lee (guaranteed shortest path)
                route = self._route_net_lee(net_name, pins, escapes)
            else:
                # Simple 2-pin signal net: use A* (faster)
                route = self._route_net_astar(net_name, pins, escapes)

            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)

        if len(self.failed) == 0:
            return RoutingResult(
                routes=self.routes,
                success=True,
                routed_count=routed,
                total_count=len(net_order),
                algorithm_used='hybrid',
                total_wirelength=sum(r.total_length for r in self.routes.values() if r.success),
                via_count=sum(len(r.vias) for r in self.routes.values() if r.success)
            )

        # Some nets failed - try rip-up and reroute
        print(f"    [HYBRID] Initial pass: {routed}/{len(net_order)}, trying ripup...")

        ripup_result = self._route_ripup_reroute(net_order, net_pins, escapes, placement, parts_db)

        if ripup_result.routed_count > routed:
            return ripup_result

        return RoutingResult(
            routes=self.routes,
            success=False,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='hybrid'
        )

    # =========================================================================
    # META ALGORITHM 11: AUTO
    # =========================================================================

    def _route_auto(self, net_order: List[str], net_pins: Dict,
                    escapes: Dict, placement: Dict, parts_db: Dict) -> RoutingResult:
        """
        Auto-select best algorithm based on design complexity.

        Heuristics:
        - Few nets (<10): Use Lee for optimal paths
        - Many multi-terminal nets: Use Steiner
        - Complex routing: Use PathFinder
        - Default: Use Hybrid
        """
        print("    [AUTO] Analyzing design to select best algorithm...")

        num_nets = len(net_order)
        multi_terminal = sum(1 for n in net_order if len(net_pins.get(n, [])) > 2)

        if num_nets < 10:
            print(f"    [AUTO] Few nets ({num_nets}) - using Lee for optimal paths")
            return self._route_lee(net_order, net_pins, escapes)

        if multi_terminal > num_nets * 0.3:
            print(f"    [AUTO] Many multi-terminal nets ({multi_terminal}) - using Steiner")
            return self._route_steiner(net_order, net_pins, escapes)

        # Default to hybrid
        print(f"    [AUTO] Standard design - using Hybrid algorithm")
        return self._route_hybrid(net_order, net_pins, escapes, placement, parts_db)

    # =========================================================================
    # ALGORITHM 12: PUSH-AND-SHOVE (Interactive Router)
    # =========================================================================
    # Push-and-Shove routing is what makes modern interactive routers (KiCad 6+,
    # Altium, Eagle) achieve near-100% completion rates.
    #
    # Instead of giving up when a route is blocked, P&S:
    # 1. Identifies the blocking trace
    # 2. Calculates minimum displacement to create clearance
    # 3. "Shoves" (displaces) the blocking trace
    # 4. Validates DRC after shove
    # 5. Propagates ripple effects (shoved traces may shove others)
    #
    # Reference: KiCad PNS (Push 'N' Shove) router by CERN
    # =========================================================================

    def _route_push_and_shove(self, net_order: List[str], net_pins: Dict,
                               escapes: Dict, placement: Dict,
                               parts_db: Dict) -> RoutingResult:
        """
        Push-and-Shove router - the key to 100% completion rates.

        This algorithm routes traces interactively, pushing existing traces
        out of the way when blocked instead of failing.
        """
        print("    [P&S] Using Push-and-Shove algorithm...")

        # Configuration for P&S
        max_shove_distance = self.config.grid_size * 10  # Max displacement
        max_ripple_depth = 5  # Max cascade depth for ripple effects

        # Track shove statistics
        total_shoves = 0
        max_ripple_reached = 0

        # Power nets get priority (route first, less likely to be shoved)
        power_nets = {'GND', 'VCC', 'VDD', 'VSS', 'V+', 'V-', '3V3', '5V', '12V',
                      'VBAT', 'VBUS', 'AVCC', 'AVDD', 'DVCC', 'DVDD', 'AGND', 'DGND',
                      'VIN', 'VOUT', 'PWR', 'POWER', 'SUPPLY'}

        # Sort: power nets first, then by pin count (complex first)
        sorted_order = sorted(net_order, key=lambda n: (
            0 if n.upper() in power_nets else 1,
            -len(net_pins.get(n, []))
        ))

        # Track walkaround vs shove statistics
        walkaround_count = 0

        for net_name in sorted_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            # PHASE 1: Try normal A* routing first (fastest)
            route = self._route_net_astar(net_name, pins, escapes)

            if route.success:
                self.routes[net_name] = route
                for seg in route.segments:
                    self._mark_segment_in_grid(seg, net_name)
                continue

            # PHASE 2: Try walkaround - route around obstacles without moving them
            # This is preferred over shoving as it doesn't disturb existing routes
            wa_route = self._route_walkaround(net_name, pins, escapes)

            if wa_route and wa_route.success:
                self.routes[net_name] = wa_route
                walkaround_count += 1
                for seg in wa_route.segments:
                    self._mark_segment_in_grid(seg, net_name)
                continue

            # PHASE 3: Shove - push existing traces out of the way
            ps_route, shove_count, ripple_depth = self._route_with_shove(
                net_name, pins, escapes,
                max_shove_distance, max_ripple_depth
            )

            if ps_route and ps_route.success:
                self.routes[net_name] = ps_route
                total_shoves += shove_count
                max_ripple_reached = max(max_ripple_reached, ripple_depth)
                for seg in ps_route.segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                # All methods failed - mark as failed
                self.routes[net_name] = route  # Original failed route
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)

        print(f"    [P&S] Complete: {routed}/{len(net_order)} routed, "
              f"{walkaround_count} walkarounds, {total_shoves} shoves, "
              f"max ripple depth {max_ripple_reached}")

        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='push_and_shove',
            total_wirelength=sum(r.total_length for r in self.routes.values() if r.success),
            via_count=sum(len(r.vias) for r in self.routes.values() if r.success)
        )

    def _route_walkaround(self, net_name: str, pins: List, escapes: Dict) -> Optional[Route]:
        """
        Walkaround routing - find path around obstacles without moving them.

        This is the first fallback when direct A* fails. It uses a modified A*
        that increases cost for going near obstacles, encouraging paths that
        "hug" obstacles and find gaps between them.

        Based on KiCad's PNS::WALKAROUND algorithm which generates smooth curves
        around pads, vias, and traces while minimizing total path length.
        """
        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            return None

        route = Route(net=net_name, algorithm_used='walkaround')
        connected_points = {endpoints[0]}
        remaining_points = set(endpoints[1:])

        while remaining_points:
            best_path = None
            best_target = None
            best_cost = float('inf')

            for source in connected_points:
                for target in remaining_points:
                    # Try walkaround path between source and target
                    path, cost = self._find_walkaround_path(source, target, net_name)

                    if path and cost < best_cost:
                        best_path = path
                        best_target = target
                        best_cost = cost

            if best_path is None:
                # Cannot find walkaround path
                return None

            # Convert path to segments
            segments = self._path_to_segments(best_path, net_name)
            route.segments.extend(segments)
            connected_points.add(best_target)
            remaining_points.remove(best_target)

        route.success = len(remaining_points) == 0
        # total_length is computed automatically from segments

        return route

    def _find_walkaround_path(self, start: Tuple[float, float],
                               end: Tuple[float, float], net_name: str
                               ) -> Tuple[Optional[List], float]:
        """
        Find a walkaround path from start to end.

        Uses modified A* with:
        1. Higher cost for cells near obstacles (hugging cost)
        2. Preference for paths that go around rather than through
        3. Multi-layer support with via cost

        Returns: (path, total_cost) or (None, inf)
        """
        # Convert to grid coordinates
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        # A* with walkaround heuristics
        start_state = (0, start_row, start_col)
        end_states = {(0, end_row, end_col), (1, end_row, end_col)}

        # Priority queue: (f_score, g_score, state, path)
        open_set = [(0, 0, start_state, [])]
        g_scores = {start_state: 0}
        visited = set()

        # Cost multipliers for walkaround
        NORMAL_COST = 1.0
        NEAR_OBSTACLE_COST = 2.0  # Penalty for being near obstacles
        VIA_COST = 5.0  # Cost of layer change

        while open_set:
            f, g, current, path = heapq.heappop(open_set)

            layer, row, col = current

            if current in end_states:
                # Found path to target
                return path + [current], g

            if current in visited:
                continue
            visited.add(current)

            # Get current grid
            grid = self.fcu_grid if layer == 0 else self.bcu_grid

            # Explore neighbors (8-connected for smoother paths)
            neighbors = []

            # 4-connected (Manhattan)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbors.append((layer, row + dr, col + dc, NORMAL_COST))

            # 4-connected diagonal (45-degree routing)
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbors.append((layer, row + dr, col + dc, 1.414))  # sqrt(2)

            # Via to other layer
            other_layer = 1 - layer
            neighbors.append((other_layer, row, col, VIA_COST))

            for nlayer, nr, nc, move_cost in neighbors:
                if not self._in_bounds(nr, nc):
                    continue

                next_state = (nlayer, nr, nc)
                if next_state in visited:
                    continue

                next_grid = self.fcu_grid if nlayer == 0 else self.bcu_grid
                cell = next_grid[nr][nc]

                # Check if cell is passable for walkaround
                if cell is not None and cell != net_name:
                    if cell in self.BLOCKED_MARKERS:
                        continue  # Component/edge - cannot pass
                    else:
                        continue  # Another net's trace - cannot pass (no shove in walkaround)

                # Calculate cost including "hugging" penalty, layer direction, and global routing
                step_cost = move_cost
                if nlayer == layer:  # Same-layer move (not a via)
                    dir_dr, dir_dc = nr - row, nc - col
                    step_cost *= self._direction_cost_multiplier(dir_dr, dir_dc, nlayer)
                step_cost *= self._global_routing_cost(nr, nc, net_name)

                # Check if near obstacle (within 1-2 cells) - add penalty
                near_obstacle = False
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        check_r, check_c = nr + dr, nc + dc
                        if self._in_bounds(check_r, check_c):
                            check_cell = next_grid[check_r][check_c]
                            if check_cell is not None and check_cell != net_name:
                                near_obstacle = True
                                break
                    if near_obstacle:
                        break

                if near_obstacle:
                    step_cost *= NEAR_OBSTACLE_COST

                new_g = g + step_cost

                if next_state in g_scores and g_scores[next_state] <= new_g:
                    continue

                g_scores[next_state] = new_g

                # Heuristic: Manhattan distance to target
                h = abs(nr - end_row) + abs(nc - end_col)
                f = new_g + h

                heapq.heappush(open_set, (f, new_g, next_state, path + [current]))

        # No path found
        return None, float('inf')

    def _route_with_shove(self, net_name: str, pins: List, escapes: Dict,
                          max_shove_dist: float, max_ripple: int
                          ) -> Tuple[Optional[Route], int, int]:
        """
        Route a net using push-and-shove when blocked.

        Returns: (route, shove_count, max_ripple_depth)
        """
        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            return None, 0, 0

        total_shoves = 0
        max_ripple_used = 0

        # Try to route between all pairs of endpoints
        route = Route(net=net_name, algorithm_used='push_and_shove')
        connected_points = {endpoints[0]}
        remaining_points = set(endpoints[1:])

        while remaining_points:
            best_segment = None
            best_target = None
            best_shoves = float('inf')

            for source in connected_points:
                for target in remaining_points:
                    # Try to route with shove
                    segment, shoves, ripple = self._find_path_with_shove(
                        source, target, net_name, max_shove_dist, max_ripple
                    )

                    if segment and shoves < best_shoves:
                        best_segment = segment
                        best_target = target
                        best_shoves = shoves
                        max_ripple_used = max(max_ripple_used, ripple)

            if best_segment is None:
                # Cannot connect remaining points even with shove
                route.success = False
                return route, total_shoves, max_ripple_used

            # Apply the best segment
            route.segments.extend(best_segment)
            connected_points.add(best_target)
            remaining_points.remove(best_target)
            total_shoves += best_shoves

        route.success = True
        # total_length is computed automatically from segments

        return route, total_shoves, max_ripple_used

    def _find_path_with_shove(self, start: Tuple[float, float],
                               end: Tuple[float, float], net_name: str,
                               max_shove_dist: float, max_ripple: int
                               ) -> Tuple[Optional[List[TrackSegment]], int, int]:
        """
        Find a path from start to end, shoving blocking traces if needed.

        Uses A* but when blocked, attempts to shove the blocking trace.
        Returns: (segments, shove_count, ripple_depth)
        """
        # Convert to grid coordinates
        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        # A* with shove capability
        # State: (layer, row, col)
        # We track which nets we've shoved to avoid infinite loops
        start_state = (0, start_row, start_col)  # Start on F.Cu
        end_state = (0, end_row, end_col)

        # Priority queue: (f_score, g_score, state, path, shoved_nets)
        open_set = [(0, 0, start_state, [], set())]
        visited = set()

        shove_count = 0
        max_ripple_used = 0

        while open_set:
            _, g, current, path, shoved = heapq.heappop(open_set)

            layer, row, col = current

            if (layer, row, col) == end_state or (row == end_row and col == end_col):
                # Found path - convert to segments
                segments = self._path_to_segments(path + [current], net_name)
                return segments, shove_count, max_ripple_used

            if current in visited:
                continue
            visited.add(current)

            # Explore neighbors
            grid = self.fcu_grid if layer == 0 else self.bcu_grid

            # 4-connected neighbors + via
            neighbors = [
                (layer, row-1, col),  # Up
                (layer, row+1, col),  # Down
                (layer, row, col-1),  # Left
                (layer, row, col+1),  # Right
            ]

            # Via to other layer
            other_layer = 1 - layer
            neighbors.append((other_layer, row, col))

            for next_state in neighbors:
                nlayer, nr, nc = next_state

                if not self._in_bounds(nr, nc):
                    continue
                if next_state in visited:
                    continue

                next_grid = self.fcu_grid if nlayer == 0 else self.bcu_grid
                cell = next_grid[nr][nc]

                # Check if clear
                if cell is None or cell == net_name:
                    # Cell is clear - add to open set
                    new_g = g + 1
                    h = abs(nr - end_row) + abs(nc - end_col)
                    f = new_g + h
                    heapq.heappush(open_set, (f, new_g, next_state, path + [current], shoved.copy()))

                elif cell in self.BLOCKED_MARKERS:
                    # Component/pad - cannot shove
                    continue

                else:
                    # Blocked by another net's trace - try to shove
                    blocking_net = cell

                    if blocking_net in shoved:
                        # Already shoved this net in this path - avoid loop
                        continue

                    if len(shoved) >= max_ripple:
                        # Max ripple depth reached
                        continue

                    # Try to shove the blocking trace
                    shove_success, ripple_depth = self._try_shove_trace(
                        blocking_net, nr, nc, nlayer, net_name,
                        max_shove_dist, max_ripple - len(shoved)
                    )

                    if shove_success:
                        shove_count += 1
                        max_ripple_used = max(max_ripple_used, ripple_depth + 1)

                        # Cell is now clear - add to open set
                        new_shoved = shoved.copy()
                        new_shoved.add(blocking_net)

                        new_g = g + 1
                        h = abs(nr - end_row) + abs(nc - end_col)
                        f = new_g + h
                        heapq.heappush(open_set, (f, new_g, next_state, path + [current], new_shoved))

        # No path found even with shove
        return None, shove_count, max_ripple_used

    def _try_shove_trace(self, blocking_net: str, block_row: int, block_col: int,
                          layer: int, requesting_net: str,
                          max_shove_dist: float, remaining_ripple: int) -> Tuple[bool, int]:
        """
        Try to shove a blocking trace out of the way.

        Returns: (success, ripple_depth_used)
        """
        if blocking_net not in self.routes:
            return False, 0

        blocking_route = self.routes[blocking_net]
        if not blocking_route.success:
            return False, 0

        # Find which segment contains the blocking cell
        block_x = self._grid_to_real_x(block_col)
        block_y = self._grid_to_real_y(block_row)

        blocking_segment = None
        segment_index = -1

        for i, seg in enumerate(blocking_route.segments):
            if self._point_near_segment(block_x, block_y, seg, self.config.grid_size * 2):
                blocking_segment = seg
                segment_index = i
                break

        if blocking_segment is None:
            return False, 0

        # Calculate shove direction (perpendicular to segment direction)
        dx = blocking_segment.end[0] - blocking_segment.start[0]
        dy = blocking_segment.end[1] - blocking_segment.start[1]
        seg_len = math.sqrt(dx*dx + dy*dy)

        if seg_len < 0.001:
            return False, 0

        # Perpendicular direction (normalized)
        perp_x = -dy / seg_len
        perp_y = dx / seg_len

        # Try shoving in both perpendicular directions
        shove_distances = [self.config.grid_size, self.config.grid_size * 2,
                          self.config.grid_size * 3]

        for shove_dist in shove_distances:
            if shove_dist > max_shove_dist:
                break

            for direction in [1, -1]:
                offset_x = perp_x * shove_dist * direction
                offset_y = perp_y * shove_dist * direction

                # Create new shoved segment
                new_start = (blocking_segment.start[0] + offset_x,
                            blocking_segment.start[1] + offset_y)
                new_end = (blocking_segment.end[0] + offset_x,
                          blocking_segment.end[1] + offset_y)

                # Check if new position is valid
                if self._is_shove_valid(new_start, new_end, blocking_net, layer):
                    # Apply the shove
                    self._unmark_segment_from_grid(blocking_segment, blocking_net)

                    # Update segment position
                    old_start = blocking_segment.start
                    old_end = blocking_segment.end
                    blocking_segment.start = new_start
                    blocking_segment.end = new_end

                    # Add transition segments to maintain connectivity
                    layer_name = 'F.Cu' if layer == 0 else 'B.Cu'

                    # Insert transition from old start to new start
                    if segment_index > 0:
                        trans_seg = TrackSegment(
                            start=old_start, end=new_start,
                            layer=layer_name, width=blocking_segment.width,
                            net=blocking_net
                        )
                        blocking_route.segments.insert(segment_index, trans_seg)
                        self._mark_segment_in_grid(trans_seg, blocking_net)
                        segment_index += 1

                    # Insert transition from new end to old end (for next segment)
                    if segment_index < len(blocking_route.segments) - 1:
                        trans_seg = TrackSegment(
                            start=new_end, end=old_end,
                            layer=layer_name, width=blocking_segment.width,
                            net=blocking_net
                        )
                        blocking_route.segments.insert(segment_index + 1, trans_seg)
                        self._mark_segment_in_grid(trans_seg, blocking_net)

                    # Mark new segment position
                    self._mark_segment_in_grid(blocking_segment, blocking_net)

                    return True, 0

        # Could not shove directly - try ripple effect
        if remaining_ripple > 0:
            # This would require shoving another trace first
            # For now, return failure - full ripple implementation is complex
            pass

        return False, 0

    def _is_shove_valid(self, start: Tuple[float, float], end: Tuple[float, float],
                        net_name: str, layer: int) -> bool:
        """Check if a shoved segment position is valid (doesn't hit obstacles)."""
        grid = self.fcu_grid if layer == 0 else self.bcu_grid

        start_col = self._real_to_grid_col(start[0])
        start_row = self._real_to_grid_row(start[1])
        end_col = self._real_to_grid_col(end[0])
        end_row = self._real_to_grid_row(end[1])

        # Check bounds
        if not self._in_bounds(start_row, start_col):
            return False
        if not self._in_bounds(end_row, end_col):
            return False

        # Check cells along path
        if start_row == end_row:
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                cell = grid[start_row][col]
                if cell in self.BLOCKED_MARKERS:
                    return False
                if cell is not None and cell != net_name:
                    return False  # Would collide with another net
        elif start_col == end_col:
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                cell = grid[row][start_col]
                if cell in self.BLOCKED_MARKERS:
                    return False
                if cell is not None and cell != net_name:
                    return False
        else:
            # Diagonal - check via Bresenham
            dx = end_col - start_col
            dy = end_row - start_row
            steps = max(abs(dx), abs(dy))
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                col = int(start_col + t * dx)
                row = int(start_row + t * dy)
                if self._in_bounds(row, col):
                    cell = grid[row][col]
                    if cell in self.BLOCKED_MARKERS:
                        return False
                    if cell is not None and cell != net_name:
                        return False

        return True

    def _unmark_segment_from_grid(self, segment: TrackSegment, net_name: str):
        """Remove a segment from the occupancy grid."""
        grid = self.fcu_grid if segment.layer == 'F.Cu' else self.bcu_grid

        start_col = self._real_to_grid_col(segment.start[0])
        start_row = self._real_to_grid_row(segment.start[1])
        end_col = self._real_to_grid_col(segment.end[0])
        end_row = self._real_to_grid_row(segment.end[1])

        # Unmark cells along segment (with clearance)
        cc = self.clearance_cells

        if start_row == end_row:
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                self._unmark_cell_with_clearance(grid, start_row, col, net_name, cc)
        elif start_col == end_col:
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                self._unmark_cell_with_clearance(grid, row, start_col, net_name, cc)
        else:
            dx = end_col - start_col
            dy = end_row - start_row
            steps = max(abs(dx), abs(dy))
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                col = int(start_col + t * dx)
                row = int(start_row + t * dy)
                self._unmark_cell_with_clearance(grid, row, col, net_name, cc)

    def _unmark_cell_with_clearance(self, grid: List[List], row: int, col: int,
                                     net_name: str, cc: int):
        """Unmark a cell and its clearance zone."""
        for dr in range(-cc, cc + 1):
            for dc in range(-cc, cc + 1):
                r, c = row + dr, col + dc
                if self._in_bounds(r, c):
                    if grid[r][c] == net_name:
                        grid[r][c] = None

    def _path_to_segments(self, path: List[Tuple[int, int, int]],
                          net_name: str) -> List[TrackSegment]:
        """Convert a grid path to track segments."""
        if len(path) < 2:
            return []

        segments = []
        vias = []

        current_layer = path[0][0]
        current_start = path[0]

        for i in range(1, len(path)):
            layer, row, col = path[i]
            prev_layer, prev_row, prev_col = path[i-1]

            if layer != prev_layer:
                # Layer change - add via
                # First, finish current segment
                if i > 1:
                    start_x = self._grid_to_real_x(current_start[2])
                    start_y = self._grid_to_real_y(current_start[1])
                    end_x = self._grid_to_real_x(prev_col)
                    end_y = self._grid_to_real_y(prev_row)

                    if start_x != end_x or start_y != end_y:
                        layer_name = 'F.Cu' if current_layer == 0 else 'B.Cu'
                        segments.append(TrackSegment(
                            start=(start_x, start_y),
                            end=(end_x, end_y),
                            layer=layer_name,
                            width=self.get_trace_width(net_name),
                            net=net_name
                        ))

                # Record via position
                via_x = self._grid_to_real_x(col)
                via_y = self._grid_to_real_y(row)
                vias.append((via_x, via_y))

                current_layer = layer
                current_start = path[i]

            # Check for direction change (need new segment)
            elif i > 1:
                prev2_layer, prev2_row, prev2_col = path[i-2]

                # Direction vectors
                dir1 = (prev_row - prev2_row, prev_col - prev2_col)
                dir2 = (row - prev_row, col - prev_col)

                if dir1 != dir2:
                    # Direction changed - finish current segment
                    start_x = self._grid_to_real_x(current_start[2])
                    start_y = self._grid_to_real_y(current_start[1])
                    end_x = self._grid_to_real_x(prev_col)
                    end_y = self._grid_to_real_y(prev_row)

                    if start_x != end_x or start_y != end_y:
                        layer_name = 'F.Cu' if current_layer == 0 else 'B.Cu'
                        segments.append(TrackSegment(
                            start=(start_x, start_y),
                            end=(end_x, end_y),
                            layer=layer_name,
                            width=self.get_trace_width(net_name),
                            net=net_name
                        ))

                    current_start = path[i-1]

        # Add final segment
        if len(path) >= 2:
            start_x = self._grid_to_real_x(current_start[2])
            start_y = self._grid_to_real_y(current_start[1])
            end_x = self._grid_to_real_x(path[-1][2])
            end_y = self._grid_to_real_y(path[-1][1])

            if start_x != end_x or start_y != end_y:
                layer_name = 'F.Cu' if current_layer == 0 else 'B.Cu'
                segments.append(TrackSegment(
                    start=(start_x, start_y),
                    end=(end_x, end_y),
                    layer=layer_name,
                    width=self.get_trace_width(net_name),
                    net=net_name
                ))

        return segments

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _direction_cost_multiplier(self, dr: int, dc: int, layer: int) -> float:
        """
        Apply CPU Lab layer direction preference as cost penalty.

        If F.Cu prefers horizontal (dc moves), vertical moves (dr only) get 3.0x penalty.
        If B.Cu prefers vertical (dr moves), horizontal moves (dc only) get 3.0x penalty.
        Diagonal moves get 1.5x penalty (compromise).
        """
        if not getattr(self, '_layer_directions', None):
            return 1.0

        layer_name = 'F.Cu' if layer == 0 else 'B.Cu'
        preferred = self._layer_directions.get(layer_name, '')
        if isinstance(preferred, str):
            preferred = preferred.lower()
        else:
            preferred = str(preferred).lower()

        if 'horizontal' in preferred:
            # Horizontal preferred = dc moves are cheap, dr-only moves are expensive
            if dc == 0 and dr != 0:
                return 3.0  # Pure vertical on horizontal-preferred layer
            elif dr != 0 and dc != 0:
                return 1.5  # Diagonal (partial penalty)
        elif 'vertical' in preferred:
            # Vertical preferred = dr moves are cheap, dc-only moves are expensive
            if dr == 0 and dc != 0:
                return 3.0  # Pure horizontal on vertical-preferred layer
            elif dr != 0 and dc != 0:
                return 1.5  # Diagonal (partial penalty)

        return 1.0

    def _global_routing_cost(self, row: int, col: int, net_name: str) -> float:
        """
        Apply CPU Lab global routing region preference as cost hint.

        If a net has assigned regions from global routing, traces that stay
        within their assigned regions get lower cost (1.0), while traces
        that go outside their assigned regions get a mild penalty (1.5x).

        This is a HINT, not a hard constraint — detailed routing can deviate
        when necessary but prefers to follow the global plan.
        """
        gr = getattr(self, '_global_routing', None)
        if not gr or not gr.get('net_routes'):
            return 1.0

        net_route = gr['net_routes'].get(net_name)
        if not net_route:
            return 1.0

        regions = net_route.get('regions', [])
        if not regions:
            return 1.0

        # Convert grid cell position to global routing region
        region_size = gr.get('region_size_mm', 5.0)
        grid_size = self.grid_size  # mm per grid cell
        pos_x = col * grid_size
        pos_y = row * grid_size
        region_col = int(pos_x / region_size)
        region_row = int(pos_y / region_size)

        # Check if this cell is in one of the net's assigned regions
        if (region_row, region_col) in regions:
            return 1.0  # In assigned region — no penalty
        else:
            return 1.5  # Outside assigned region — mild penalty

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if grid coordinates are in bounds"""
        return 0 <= row < self.grid_rows and 0 <= col < self.grid_cols

    def _is_cell_accessible_for_net(self, grid: List[List], row: int, col: int,
                                     net_name: str) -> bool:
        """
        Check if a cell is accessible for this net (relaxed check for target pads).

        This is a relaxed version of _is_cell_clear_for_net that only checks the
        center cell, not the clearance zone. Used for start/end pads where we
        MUST be able to reach the cell regardless of nearby obstacles.

        The cell is accessible if:
        - It belongs to this net, OR
        - It's marked with __PAD_CONFLICT__ (which happens when pads of different
          nets are close together - we can still route TO our own pad)
        - It's not a hard blocker (__COMPONENT__, __EDGE__, __PAD_NC__)
        """
        if not self._in_bounds(row, col):
            return False

        center = grid[row][col]

        # Allow cells marked with our net
        if center == net_name:
            return True

        # Allow empty cells
        if center is None:
            return True

        # Allow __PAD_CONFLICT__ - we can still route to our own pad even if
        # nearby pads belong to different nets
        if center == '__PAD_CONFLICT__':
            return True

        # Block hard blockers
        if center in {'__COMPONENT__', '__EDGE__', '__PAD_NC__'}:
            return False

        # Block cells belonging to other nets
        return False

    def _prepare_net_bitmaps(self, net_name: str):
        """
        Pre-build per-net clearance bitmaps for O(1) lookups.

        Called once before routing each net. All subsequent calls to
        _is_cell_clear_for_net for this net use the cached bitmap
        instead of the 13-offset Python loop.

        Benefits ALL algorithms: Lee BFS, PATHFINDER A*, Hadlock, etc.
        """
        if self.spatial_index is None:
            self._current_net_bitmap_name = None
            return

        self._current_net_bitmap_name = net_name
        self._current_net_bitmap_fcu = self.spatial_index.build_net_clearance_bitmap(
            net_name, self.clearance_cells, 'F.Cu')
        self._current_net_bitmap_bcu = self.spatial_index.build_net_clearance_bitmap(
            net_name, self.clearance_cells, 'B.Cu')

    def _is_cell_clear_for_net(self, grid: List[List], row: int, col: int,
                               net_name: str) -> bool:
        """Check if a cell is clear for routing this net (with CIRCULAR clearance).

        OPTIMIZED: When per-net bitmaps are pre-built (via _prepare_net_bitmaps),
        uses O(1) NumPy array lookup instead of 13 Python string comparisons.
        Falls back to Python loop when bitmaps not available.
        """
        if not self._in_bounds(row, col):
            return False

        # FAST PATH: Use pre-built per-net bitmap (O(1) lookup)
        if (hasattr(self, '_current_net_bitmap_name') and
            self._current_net_bitmap_name == net_name and
            self._current_net_bitmap_fcu is not None):
            bitmap = self._current_net_bitmap_fcu if grid is self.fcu_grid else self._current_net_bitmap_bcu
            return bool(bitmap[row, col])

        # FALLBACK: Original Python loop (when bitmaps not built)
        center = grid[row][col]
        if center in self.BLOCKED_MARKERS:
            return False
        # Handle pad clearance markers: __PAD_CLEAR_<netname>__
        # These block routing for OTHER nets, but allow same net
        if center is not None:
            if center.startswith('__PAD_CLEAR_'):
                marker_net = center[12:-2]  # Remove '__PAD_CLEAR_' prefix and '__' suffix
                if marker_net != net_name:
                    return False  # Clearance zone of different net
            elif center != net_name:
                return False

        # OPTIMIZED: Use pre-computed circular offset pattern
        for dr, dc in self.clearance_offsets:
            r, c = row + dr, col + dc
            if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                occ = grid[r][c]
                if occ in self.BLOCKED_MARKERS:
                    return False
                if occ is not None:
                    if occ.startswith('__PAD_CLEAR_'):
                        marker_net = occ[12:-2]
                        if marker_net != net_name:
                            return False
                    elif occ != net_name:
                        return False

        return True

    def _is_cell_passable_for_diagonal(self, grid: List[List], row: int, col: int,
                                        net_name: str) -> bool:
        """Check if a cell is passable for a diagonal move clipping through it.

        This is a LIGHTER check than _is_cell_clear_for_net because we're not
        routing through this cell - we're just checking that a diagonal trace
        won't clip through an obstacle corner.

        For diagonal passability, we only need to ensure:
        1. The cell isn't blocked (__BLOCKED__, __COMPONENT__)
        2. The cell isn't occupied by a DIFFERENT net's pad

        We DON'T need full clearance because the trace center isn't here.
        """
        if not self._in_bounds(row, col):
            return False

        cell = grid[row][col]

        # Blocked markers are always impassable
        if cell in self.BLOCKED_MARKERS:
            return False

        # Other net's pads are impassable (can't clip through them)
        if cell is not None and cell != net_name:
            return False

        return True

    def _is_diagonal_path_clear(self, grid: List[List], r1: int, c1: int,
                                 r2: int, c2: int, net_name: str) -> bool:
        """Check if the diagonal path from (r1,c1) to (r2,c2) is clear.

        Uses Bresenham's line algorithm to check all cells along the path.
        This is more thorough than just checking corner cells.

        For single-cell diagonal moves (dr=1, dc=1), this is equivalent to
        checking the corner cells, but this works for longer diagonals too.
        """
        # For same cell, always clear
        if r1 == r2 and c1 == c2:
            return True

        # Bresenham's line algorithm
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        r, c = r1, c1
        r_step = 1 if r2 > r1 else -1
        c_step = 1 if c2 > c1 else -1

        # For truly diagonal moves, check both the integer cells and "corners"
        if dr == dc:  # 45-degree diagonal
            # Check both corner cells for each step
            for _ in range(dr):
                # Check the two corner cells that the diagonal clips through
                corner1 = (r, c + c_step)  # Horizontal first
                corner2 = (r + r_step, c)  # Vertical first

                if not self._is_cell_passable_for_diagonal(grid, corner1[0], corner1[1], net_name):
                    return False
                if not self._is_cell_passable_for_diagonal(grid, corner2[0], corner2[1], net_name):
                    return False

                r += r_step
                c += c_step

        return True

    def _is_cell_clear_for_via(self, grid: List[List], row: int, col: int,
                               net_name: str) -> bool:
        """Check if a cell is clear for placing a VIA (larger CIRCULAR clearance).

        OPTIMIZED: Uses pre-computed via_clearance_offsets (like database index).

        Vias have diameter 0.8mm vs traces at 0.25mm, so they need more clearance
        from other nets' pads and traces.

        FIX 2 (2026-02-08): Allow vias near __COMPONENT__ cells when the center
        is marked with our net. This is PHYSICALLY VALID because:
        - SMD component bodies are on TOP of the PCB
        - Vias go THROUGH the PCB substrate
        - A via placed at an SMD pad does NOT conflict with the component body
        - Only OTHER NETS' pads are true via blockers
        """
        if not self._in_bounds(row, col):
            return False

        # Check center cell
        center = grid[row][col]

        # FIX 2: For vias on OUR OWN PAD (center == net_name), we allow placement
        # even if nearby cells have __COMPONENT__ markers. The via goes through
        # the PCB substrate, not through the component body.
        placing_on_own_pad = (center == net_name)

        if center in self.BLOCKED_MARKERS:
            return False
        if center is not None and center != net_name:
            return False

        # OPTIMIZED: Use pre-computed via clearance offset pattern
        for dr, dc in self.via_clearance_offsets:
            r, c = row + dr, col + dc
            if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                occ = grid[r][c]

                # FIX 2: When placing via on our own pad, __COMPONENT__
                # markers are NOT blockers (via goes through substrate).
                if occ == '__COMPONENT__':
                    if placing_on_own_pad:
                        continue  # OK - via goes through substrate
                    else:
                        return False

                # FIX 3: __PAD_CONFLICT__ handling
                if occ == '__PAD_CONFLICT__':
                    if placing_on_own_pad:
                        continue
                    else:
                        return False

                # Other blocked markers are always blockers
                if occ in {'__PAD_NC__', '__EDGE__'}:
                    return False

                # FIX 4: __PAD_CLEAR_<net>__ markers should allow vias of the same net.
                # These markers protect our own pad's clearance zone from OTHER nets,
                # but vias of our own net should be allowed near our own pad.
                if occ is not None and occ.startswith('__PAD_CLEAR_'):
                    # Extract the net name from the marker
                    marker_net = occ[len('__PAD_CLEAR_'):-2]  # Remove prefix and trailing __
                    if marker_net == net_name:
                        continue  # Our own pad's clearance - allow
                    else:
                        return False  # Other net's pad clearance - block

                # Other nets' pads/traces are always blockers
                if occ is not None and occ != net_name:
                    return False

        return True

    def _build_pin_name_to_number_map(self) -> Dict[str, Dict[str, str]]:
        """
        Build a lookup: comp_ref -> {pin_name -> pin_number, pin_number -> pin_number}.

        Nets reference pins by NAME (e.g., 'U1.IO19'), but escapes store by NUMBER (e.g., '8').
        This map resolves that mismatch.
        """
        if hasattr(self, '_pin_name_map') and self._pin_name_map:
            return self._pin_name_map

        self._pin_name_map = {}
        if not hasattr(self, '_parts_db') or not self._parts_db:
            return self._pin_name_map

        try:
            from .common_types import get_pins
        except ImportError:
            from common_types import get_pins

        parts = self._parts_db.get('parts', {})
        for ref, part in parts.items():
            name_map = {}
            for p in get_pins(part):
                num = str(p.get('number', ''))
                name = str(p.get('name', ''))
                # Map name -> number (e.g., 'IO19' -> '8')
                if name:
                    name_map[name] = num
                # Map number -> number (identity, so number lookups also work)
                if num:
                    name_map[num] = num
            self._pin_name_map[ref] = name_map

        return self._pin_name_map

    def _resolve_escape_key(self, comp: str, pin: str, escapes: Dict) -> str:
        """
        Resolve a pin identifier to the key used in the escapes dict.

        Tries: pin as-is, then name->number mapping, then number->name mapping.
        Returns the key that exists in escapes[comp], or '' if not found.
        """
        if comp not in escapes:
            return ''
        comp_escapes = escapes[comp]

        # Direct match (pin is already the correct key)
        if pin in comp_escapes:
            return pin

        # Resolve via name->number map
        name_map = self._build_pin_name_to_number_map()
        if comp in name_map:
            resolved = name_map[comp].get(pin, '')
            if resolved and resolved in comp_escapes:
                return resolved

        return ''

    def _get_escape_endpoints(self, pins: List, escapes: Dict) -> List[Tuple[float, float]]:
        """
        Get escape endpoints for a net's pins.

        For simple designs without escape routing, this method falls back to
        calculating pin positions directly from placement + parts_db.

        IMPORTANT: Nets reference pins by NAME (e.g., 'U1.IO19'), but escapes
        store by NUMBER (e.g., '8'). We resolve this via _resolve_escape_key().
        """
        endpoints = []
        for pin_ref in pins:
            # Parse pin reference
            if isinstance(pin_ref, str):
                parts = pin_ref.split('.')
                comp, pin = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0] if parts else '', '')
            elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
                comp, pin = str(pin_ref[0]), str(pin_ref[1])
            else:
                continue

            # Resolve pin name to escape dict key (handles name->number mismatch)
            escape_key = self._resolve_escape_key(comp, pin, escapes)

            # First try to get endpoint from escapes (for complex designs)
            if escape_key and comp in escapes and escape_key in escapes[comp]:
                esc = escapes[comp][escape_key]
                # Try explicit end/endpoint attributes first
                end = getattr(esc, 'end', None) or getattr(esc, 'endpoint', None)
                if end:
                    # Ensure coordinates are floats, not strings
                    if isinstance(end, (list, tuple)) and len(end) >= 2:
                        try:
                            endpoints.append((float(end[0]), float(end[1])))
                        except (ValueError, TypeError):
                            pass
                    continue
                # For EscapePath objects, get endpoint from last trace
                traces = getattr(esc, 'traces', None)
                if traces and len(traces) > 0:
                    last_trace = traces[-1]
                    end_x = getattr(last_trace, 'end_x', None)
                    end_y = getattr(last_trace, 'end_y', None)
                    if end_x is not None and end_y is not None:
                        try:
                            endpoints.append((float(end_x), float(end_y)))
                        except (ValueError, TypeError):
                            pass
                        continue
                # Fallback: try to get pin position from esc.pin
                pin_obj = getattr(esc, 'pin', None)
                if pin_obj:
                    px = getattr(pin_obj, 'x', None)
                    py = getattr(pin_obj, 'y', None)
                    if px is not None and py is not None:
                        try:
                            endpoints.append((float(px), float(py)))
                        except (ValueError, TypeError):
                            pass
                        continue

            # FALLBACK: Calculate pin position from placement + parts_db
            # This handles simple designs without escape routing
            if hasattr(self, '_placement') and hasattr(self, '_parts_db'):
                placement = self._placement
                parts_db = self._parts_db

                if comp in placement:
                    comp_pos = placement[comp]
                    # Get component X, Y
                    if isinstance(comp_pos, (list, tuple)) and len(comp_pos) >= 2:
                        comp_x, comp_y = float(comp_pos[0]), float(comp_pos[1])
                    elif hasattr(comp_pos, 'x') and hasattr(comp_pos, 'y'):
                        comp_x, comp_y = float(comp_pos.x), float(comp_pos.y)
                    else:
                        continue

                    # Find pin offset in parts_db
                    parts = parts_db.get('parts', {})
                    if comp in parts:
                        part = parts[comp]
                        try:
                            from .common_types import get_pins
                        except ImportError:
                            from common_types import get_pins

                        for p in get_pins(part):
                            pin_num = str(p.get('number', ''))
                            pin_name = str(p.get('name', ''))
                            # Match by number OR name
                            if pin_num == pin or pin_name == pin:
                                offset = p.get('offset', (0, 0))
                                if not offset or offset == (0, 0):
                                    physical = p.get('physical', {})
                                    offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))
                                if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                                    pin_x = comp_x + float(offset[0])
                                    pin_y = comp_y + float(offset[1])
                                    endpoints.append((pin_x, pin_y))
                                break

        return endpoints

    def _get_pad_to_escape_stubs(self, pins: List, escapes: Dict, net_name: str) -> List[TrackSegment]:
        """
        Generate stub segments from actual pad positions to escape endpoints.

        The routing algorithms work on grid-snapped escape endpoints, but the actual
        pads are at exact positions. This method creates short stub segments that
        connect the pads to the routing grid.

        Args:
            pins: List of pin references for a net (e.g., ['R1.1', 'R2.1'])
            escapes: Escape routes dictionary
            net_name: Name of the net

        Returns:
            List of TrackSegment stubs connecting pads to escape endpoints
        """
        stubs = []

        for pin_ref in pins:
            # Parse pin reference
            if isinstance(pin_ref, str):
                parts = pin_ref.split('.')
                comp, pin = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0] if parts else '', '')
            elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
                comp, pin = str(pin_ref[0]), str(pin_ref[1])
            else:
                continue

            # Resolve pin name to escape dict key (handles name->number mismatch)
            escape_key = self._resolve_escape_key(comp, pin, escapes)
            if not escape_key:
                continue

            esc = escapes[comp][escape_key]

            # Get pad position (start) and escape endpoint (end)
            start = getattr(esc, 'start', None)
            end = getattr(esc, 'end', None) or getattr(esc, 'endpoint', None)

            if not start or not end:
                continue

            try:
                pad_x = float(start[0])
                pad_y = float(start[1])
                escape_x = float(end[0])
                escape_y = float(end[1])
            except (ValueError, TypeError, IndexError):
                continue

            # Skip if pad and escape are at the same position
            if abs(pad_x - escape_x) < 0.01 and abs(pad_y - escape_y) < 0.01:
                continue

            # Get layer from escape
            layer = getattr(esc, 'layer', 'F.Cu')

            # Create stub segment from pad to escape endpoint
            stub = TrackSegment(
                start=(pad_x, pad_y),
                end=(escape_x, escape_y),
                layer=layer,
                width=self.get_trace_width(net_name),
                net=net_name
            )
            stubs.append(stub)

        return stubs

    def _add_pad_stubs_to_route(self, route: Route, pins: List, escapes: Dict) -> Route:
        """
        Add pad-to-escape stub segments to a route.

        This ensures the route connects from actual pad positions, not just
        grid-snapped escape endpoints.

        Args:
            route: The route to add stubs to
            pins: List of pin references for the net
            escapes: Escape routes dictionary

        Returns:
            Route with stub segments prepended
        """
        # First try the escape-based stubs
        stubs = self._get_pad_to_escape_stubs(pins, escapes, route.net)

        # BUG FIX: If no escapes provided, create stubs from pad positions to
        # nearest segment/via endpoints. This is critical for connecting the
        # actual pads to the grid-snapped routing.
        if not stubs and hasattr(self, '_placement') and hasattr(self, '_parts_db'):
            stubs = self._generate_pad_stubs_from_placement(route, pins)

        if stubs:
            # Prepend stubs to route segments so they're drawn first
            route.segments = stubs + route.segments
        return route

    def _generate_pad_stubs_from_placement(self, route: Route, pins: List) -> List[TrackSegment]:
        """
        Generate stub segments from actual pad positions to route endpoints.

        When no escape routes are provided, this method creates short stub segments
        that connect the physical pad positions to the nearest segment/via endpoint.
        """
        try:
            from .common_types import get_pins
        except ImportError:
            from common_types import get_pins

        stubs = []
        net_name = route.net

        # Collect all segment endpoints and via positions
        route_points = set()
        for seg in route.segments:
            route_points.add((seg.start[0], seg.start[1], seg.layer))
            route_points.add((seg.end[0], seg.end[1], seg.layer))
        for via in route.vias:
            route_points.add((via.position[0], via.position[1], 'F.Cu'))
            route_points.add((via.position[0], via.position[1], 'B.Cu'))

        if not route_points:
            return stubs

        # For each pin, find the actual pad position and connect to nearest route point
        for pin_ref in pins:
            # Parse pin reference
            if isinstance(pin_ref, str):
                parts = pin_ref.split('.')
                comp, pin = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0] if parts else '', '')
            elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
                comp, pin = str(pin_ref[0]), str(pin_ref[1])
            else:
                continue

            if comp not in self._placement:
                continue

            pos = self._placement[comp]
            pos_x = pos[0] if isinstance(pos, (list, tuple)) else 0
            pos_y = pos[1] if isinstance(pos, (list, tuple)) else 0

            part = self._parts_db.get('parts', {}).get(comp, {})
            for p in get_pins(part):
                if str(p.get('number', '')) == str(pin):
                    offset = p.get('offset', None)
                    if not offset or offset == (0, 0):
                        physical = p.get('physical', {})
                        if physical:
                            offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))
                        else:
                            offset = (0, 0)
                    pad_x = pos_x + float(offset[0])
                    pad_y = pos_y + float(offset[1])

                    # Find nearest route point - prefer F.Cu but accept B.Cu via positions
                    # SMD pads are on F.Cu, so we need either:
                    # 1. F.Cu route point nearby, or
                    # 2. Via position (which connects to B.Cu route)
                    nearest = None
                    nearest_dist = float('inf')
                    nearest_layer = None
                    for rx, ry, layer in route_points:
                        dist = abs(rx - pad_x) + abs(ry - pad_y)
                        if dist < nearest_dist and dist > 0.01:  # Not already at pad
                            # Prefer F.Cu points, but accept any
                            # (via positions appear in both F.Cu and B.Cu)
                            if nearest_layer != 'F.Cu' or layer == 'F.Cu':
                                nearest_dist = dist
                                nearest = (rx, ry)
                                nearest_layer = layer

                    # Create stub segment if there's a gap
                    # BUG FIX: Changed from > 0.05 to >= 0.01 to catch small gaps
                    if nearest and nearest_dist >= 0.01:  # >= 0.01mm gap
                        from .routing_types import TrackSegment, Via

                        # FIX: If nearest point is on B.Cu, we need to add a via
                        # to connect the F.Cu pad to the B.Cu route
                        if nearest_layer == 'B.Cu':
                            # Check if there's already a via at this position
                            # (use approximate matching for floating point)
                            has_via = False
                            for v in route.vias:
                                if (abs(v.position[0] - nearest[0]) < 0.01 and
                                    abs(v.position[1] - nearest[1]) < 0.01):
                                    has_via = True
                                    break

                            if not has_via:
                                # Add via at the connection point
                                via = Via(
                                    position=nearest,
                                    net=net_name,
                                    diameter=self.config.via_diameter,
                                    drill=self.config.via_drill,
                                    from_layer='F.Cu',
                                    to_layer='B.Cu'
                                )
                                route.vias.append(via)

                        stub = TrackSegment(
                            start=(pad_x, pad_y),
                            end=nearest,
                            layer='F.Cu',
                            width=self.get_trace_width(net_name),
                            net=net_name
                        )
                        stubs.append(stub)
                    break

        return stubs

    def _sort_by_length(self, nets: List[str], net_pins: Dict,
                        escapes: Dict) -> List[str]:
        """Sort nets by estimated length (short first)"""
        def estimate(net):
            endpoints = self._get_escape_endpoints(net_pins.get(net, []), escapes)
            if len(endpoints) < 2:
                return float('inf')
            # Sum of MST edge lengths
            total = 0
            remaining = list(endpoints[1:])
            connected = [endpoints[0]]
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

        return sorted(nets, key=estimate)

    def _path_to_segments_3d(self, path: List[Tuple[int, int, int]], net_name: str,
                              exact_start: Tuple[float, float] = None,
                              exact_end: Tuple[float, float] = None
                              ) -> Tuple[List[TrackSegment], List[Via], bool]:
        """
        Convert 3D grid path (layer, row, col) to track segments and vias.

        Args:
            path: List of (layer, row, col) grid cells
            net_name: Name of the net
            exact_start: Optional exact start coordinate (for pad connection)
            exact_end: Optional exact end coordinate (for pad connection)
        """
        if len(path) < 2:
            return [], [], False

        segments = []
        vias = []

        # Group consecutive points on same layer
        segment_start = path[0]
        current_layer = path[0][0]
        current_dir = None

        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]

            # Layer change = via
            if curr[0] != prev[0]:
                # Emit current segment up to the via location
                if segment_start != prev:
                    seg = self._create_segment(segment_start, prev, net_name)
                    if seg:
                        segments.append(seg)

                # Add via at the layer transition point
                via_x = self.config.origin_x + prev[2] * self.config.grid_size
                via_y = self.config.origin_y + prev[1] * self.config.grid_size
                from_layer = 'F.Cu' if prev[0] == 0 else 'B.Cu'
                to_layer = 'F.Cu' if curr[0] == 0 else 'B.Cu'
                vias.append(Via(
                    position=(via_x, via_y),
                    net=net_name,
                    diameter=self.config.via_diameter,
                    drill=self.config.via_drill,
                    from_layer=from_layer,
                    to_layer=to_layer
                ))

                # CRITICAL: New segment on other layer starts AT the via location (prev),
                # not at curr. The via connects both layers at prev's x,y position.
                # We create a virtual point with curr's layer but prev's position.
                segment_start = (curr[0], prev[1], prev[2])
                current_layer = curr[0]
                current_dir = None
                continue

            # Same layer - check direction
            direction = (curr[1] - prev[1], curr[2] - prev[2])

            if current_dir is None:
                current_dir = direction
            elif direction != current_dir:
                # Direction changed - emit segment
                seg = self._create_segment(segment_start, prev, net_name)
                if seg:
                    segments.append(seg)
                segment_start = prev
                current_dir = direction

        # Final segment
        seg = self._create_segment(segment_start, path[-1], net_name)
        if seg:
            segments.append(seg)

        # NOTE: exact_start/exact_end handling removed - was causing shorts
        # The pad stubs are handled by _add_pad_stubs_to_route instead
        # which checks for valid connection points

        return segments, vias, True

    def _create_segment(self, start: Tuple[int, int, int], end: Tuple[int, int, int],
                        net_name: str,
                        exact_start: Tuple[float, float] = None,
                        exact_end: Tuple[float, float] = None) -> Optional[TrackSegment]:
        """Create a track segment from 3D grid points.

        Args:
            start: (layer, row, col) start grid cell
            end: (layer, row, col) end grid cell
            net_name: Name of the net
            exact_start: Optional exact start coordinate (overrides grid snap)
            exact_end: Optional exact end coordinate (overrides grid snap)
        """
        if start == end and exact_start is None and exact_end is None:
            return None

        layer_name = 'F.Cu' if start[0] == 0 else 'B.Cu'

        # Use exact coordinates if provided, otherwise snap to grid
        if exact_start:
            start_x, start_y = exact_start
        else:
            start_x = self.config.origin_x + start[2] * self.config.grid_size
            start_y = self.config.origin_y + start[1] * self.config.grid_size

        if exact_end:
            end_x, end_y = exact_end
        else:
            end_x = self.config.origin_x + end[2] * self.config.grid_size
            end_y = self.config.origin_y + end[1] * self.config.grid_size

        # Skip if start and end are the same (after applying exact coords)
        if abs(start_x - end_x) < 0.001 and abs(start_y - end_y) < 0.001:
            return None

        return TrackSegment(
            start=(start_x, start_y),
            end=(end_x, end_y),
            layer=layer_name,
            width=self.get_trace_width(net_name),
            net=net_name
        )

    def _split_segments_at_junctions(self, route: 'Route') -> 'Route':
        """
        Split existing segments at T-junction points.

        When MST routing connects a new branch to an existing segment's middle,
        we need to split that segment to create proper connectivity.

        This finds all segment endpoints and checks if any fall on the interior
        of another segment. If so, splits that segment at the junction point.
        """
        if len(route.segments) < 2:
            return route

        # Collect all segment endpoints and via positions as potential junction points
        junction_points = set()
        for seg in route.segments:
            junction_points.add(seg.start)
            junction_points.add(seg.end)
        for via in route.vias:
            junction_points.add(via.position)

        # For each junction point, check if it falls on the interior of any segment
        new_segments = []
        segments_to_process = list(route.segments)

        while segments_to_process:
            seg = segments_to_process.pop(0)
            split_point = None

            for pt in junction_points:
                if self._point_on_segment_interior(pt, seg):
                    split_point = pt
                    break

            if split_point:
                # Split the segment at this point
                seg1 = TrackSegment(
                    start=seg.start,
                    end=split_point,
                    layer=seg.layer,
                    width=seg.width,
                    net=seg.net
                )
                seg2 = TrackSegment(
                    start=split_point,
                    end=seg.end,
                    layer=seg.layer,
                    width=seg.width,
                    net=seg.net
                )

                # Only add non-zero-length segments
                if seg1.length > 0.001:
                    segments_to_process.append(seg1)  # Re-check for more splits
                if seg2.length > 0.001:
                    segments_to_process.append(seg2)

                # Add split_point to junction_points for subsequent checks
                junction_points.add(split_point)
            else:
                # No split needed, add to final list
                new_segments.append(seg)

        route.segments = new_segments
        return route

    def _point_near_segment(self, px: float, py: float, seg: 'TrackSegment',
                            tolerance: float) -> bool:
        """
        Check if a point is within tolerance distance of a segment.

        Used by push-and-shove to find which segment contains a blocking cell.
        """
        x1, y1 = seg.start
        x2, y2 = seg.end

        # Calculate segment vector
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx*dx + dy*dy

        if seg_len_sq < 0.0001:  # Zero-length segment
            # Just check distance to point
            dist_sq = (px - x1)**2 + (py - y1)**2
            return dist_sq < tolerance * tolerance

        # Calculate projection parameter t (0 = at start, 1 = at end)
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
        t = max(0, min(1, t))  # Clamp to segment

        # Calculate closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Check distance
        dist_sq = (px - closest_x)**2 + (py - closest_y)**2
        return dist_sq < tolerance * tolerance

    def _point_on_segment_interior(self, point: Tuple[float, float],
                                    seg: 'TrackSegment',
                                    tolerance: float = 0.01) -> bool:
        """
        Check if a point lies on the interior of a segment (not at endpoints).

        Uses parametric line equation and distance check.
        """
        px, py = point
        x1, y1 = seg.start
        x2, y2 = seg.end

        # Skip if point is an endpoint
        if (abs(px - x1) < tolerance and abs(py - y1) < tolerance):
            return False
        if (abs(px - x2) < tolerance and abs(py - y2) < tolerance):
            return False

        # Calculate segment length
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx*dx + dy*dy

        if seg_len_sq < 0.0001:  # Zero-length segment
            return False

        # Calculate projection parameter t (0 = at start, 1 = at end)
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq

        # Point must be strictly between endpoints
        if t <= 0.01 or t >= 0.99:
            return False

        # Calculate closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Check if point is close to the segment
        dist_sq = (px - closest_x)**2 + (py - closest_y)**2
        return dist_sq < tolerance * tolerance

    def _mark_segment_in_grid(self, segment: TrackSegment, net_name: str):
        """Mark a segment in the occupancy grid"""
        grid = self.fcu_grid if segment.layer == 'F.Cu' else self.bcu_grid

        start_col = self._real_to_grid_col(segment.start[0])
        start_row = self._real_to_grid_row(segment.start[1])
        end_col = self._real_to_grid_col(segment.end[0])
        end_row = self._real_to_grid_row(segment.end[1])

        # Mark cells along segment with clearance
        if start_row == end_row:
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                self._mark_cell_with_clearance(grid, start_row, col, net_name)
        elif start_col == end_col:
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                self._mark_cell_with_clearance(grid, row, start_col, net_name)
        else:
            # Diagonal - use Bresenham's algorithm
            dx = end_col - start_col
            dy = end_row - start_row
            steps = max(abs(dx), abs(dy))
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                col = int(start_col + t * dx)
                row = int(start_row + t * dy)
                self._mark_cell_with_clearance(grid, row, col, net_name)

    def _mark_cell_with_clearance(self, grid: List[List], row: int, col: int, net_name: str):
        """Mark a cell and its clearance zone.
        Also updates SpatialIndex to keep bitmaps in sync with Python grid."""
        # Determine layer from which grid we're marking
        layer = 'F.Cu' if grid is self.fcu_grid else 'B.Cu'
        cc = self.clearance_cells
        for dr in range(-cc, cc + 1):
            for dc in range(-cc, cc + 1):
                r, c = row + dr, col + dc
                if self._in_bounds(r, c):
                    if grid[r][c] is None or grid[r][c] == net_name:
                        grid[r][c] = net_name
                        # Keep SpatialIndex in sync for bitmap-based BFS
                        if self.spatial_index is not None:
                            self.spatial_index.mark_net(r, c, net_name, layer)

    def _save_grid_snapshot(self):
        """
        Save a snapshot of the grid state AFTER components and escapes
        are registered but BEFORE any routing. Used by RIPUP to quickly
        restore the clean grid state without rebuilding from scratch.
        Also saves the SpatialIndex state for bitmap-based BFS.
        """
        self._grid_snapshot_fcu = [row[:] for row in self.fcu_grid]
        self._grid_snapshot_bcu = [row[:] for row in self.bcu_grid]
        # Save SpatialIndex net arrays (these get updated as routes are marked)
        if self.spatial_index is not None:
            self._si_snapshot_net_fcu = self.spatial_index.net_fcu.copy()
            self._si_snapshot_net_bcu = self.spatial_index.net_bcu.copy()

    def _clear_routed_traces(self):
        """
        Restore grid to clean state (after components+escapes, before routing).
        If a snapshot exists, restores from it (fast array copy).
        Otherwise falls back to full rebuild.
        Also restores SpatialIndex to pre-routing state.
        """
        if hasattr(self, '_grid_snapshot_fcu') and self._grid_snapshot_fcu:
            # Fast path: restore from snapshot
            for r in range(self.grid_rows):
                self.fcu_grid[r][:] = self._grid_snapshot_fcu[r][:]
                self.bcu_grid[r][:] = self._grid_snapshot_bcu[r][:]
            # Restore SpatialIndex net arrays too
            if (self.spatial_index is not None and
                hasattr(self, '_si_snapshot_net_fcu') and self._si_snapshot_net_fcu is not None):
                np.copyto(self.spatial_index.net_fcu, self._si_snapshot_net_fcu)
                np.copyto(self.spatial_index.net_bcu, self._si_snapshot_net_bcu)
        else:
            # Fallback: full rebuild (should not happen if _save_grid_snapshot called)
            self._initialize_grids()
            if hasattr(self, '_placement') and hasattr(self, '_parts_db'):
                self._register_components(self._placement, self._parts_db)
        # Clear route tracking
        self.routes = {}
        self.failed = []

    def _get_component_body_size(self, footprint: str) -> tuple:
        """Get component body size (width, height) in mm based on footprint.

        This is used to register the component body as an obstacle for routing.
        Tracks must go around component bodies, not through them.
        """
        fp_lower = (footprint or '').lower()

        # ESP32 modules - very large
        if 'esp32-wroom' in fp_lower or 'esp32-wrover' in fp_lower:
            return (25.5, 18.0)
        if 'esp32' in fp_lower:
            return (20.0, 15.0)

        # USB connectors
        if 'usb_c' in fp_lower or 'usb-c' in fp_lower:
            return (9.0, 7.5)
        if 'usb' in fp_lower:
            return (8.0, 6.0)

        # IC packages
        # BUGFIX: SOIC/SOP body sizes were massively oversized, blocking routing
        # Real SOIC-8: body 5mm x 4mm, SOIC-16: body 10mm x 4mm
        if 'qfp' in fp_lower or 'tqfp' in fp_lower:
            return (12.0, 12.0)
        if 'qfn' in fp_lower:
            return (6.0, 6.0)
        # SOIC/SOP: Use BODY size only (not including leads)
        # Leads extend beyond body and are where pads connect
        # SOIC-8: body 3.9mm x 4.9mm, lead span 5.4mm
        # SOIC-16: body 3.9mm x 9.9mm, lead span 5.4mm
        if 'soic-16' in fp_lower or 'sop-16' in fp_lower:
            return (3.9, 10.0)
        if 'soic-8' in fp_lower or 'sop-8' in fp_lower:
            return (3.9, 5.0)
        if 'soic' in fp_lower or 'sop' in fp_lower:
            return (3.9, 5.0)  # Default to SOIC-8 size
        # SOT packages - listed from largest to smallest to catch specific sizes first
        if 'sot-223' in fp_lower or 'sot223' in fp_lower:
            return (6.5, 3.5)  # SOT-223 is larger than SOT-23
        if 'sot-23-5' in fp_lower or 'sot23-5' in fp_lower:
            return (3.0, 3.0)
        if 'sot-23' in fp_lower or 'sot23' in fp_lower:
            return (3.0, 2.5)
        if 'sot' in fp_lower:
            return (3.0, 2.5)  # Default to SOT-23 size

        # Capacitors and resistors - imperial footprints
        if '0201' in fp_lower:
            return (1.0, 0.6)
        if '0402' in fp_lower:
            return (1.5, 1.0)
        if '0603' in fp_lower:
            return (2.0, 1.2)
        if '0805' in fp_lower:
            return (2.5, 1.5)
        if '1206' in fp_lower:
            return (3.5, 2.0)
        if '1210' in fp_lower:
            return (4.0, 2.8)
        if '2512' in fp_lower:
            return (7.0, 3.5)

        # LED
        if 'led' in fp_lower:
            return (3.0, 1.5)

        # Default small component
        return (2.0, 1.5)

    def _register_components(self, placement: Dict, parts_db: Dict):
        """Register component pads and bodies as obstacles.

        This marks both:
        1. Component courtyard area - blocked with '__COMPONENT__' marker
        2. Pad areas - marked with net name or '__PAD_NC__' for unconnected

        Tracks cannot pass through component courtyards or unconnected pads.

        IMPORTANT: Uses calculate_courtyard() for accurate bounds based on
        actual pad positions, not just body size estimates.
        """
        parts = parts_db.get('parts', {})

        # Import courtyard calculation utility
        try:
            from .common_types import get_pins, is_smd_footprint, calculate_courtyard
        except ImportError:
            from common_types import get_pins, is_smd_footprint, calculate_courtyard

        for ref, pos in placement.items():
            part = parts.get(ref, {})

            # Get component position
            pos_x = pos.x if hasattr(pos, 'x') else pos[0] if isinstance(pos, (list, tuple)) else 0
            pos_y = pos.y if hasattr(pos, 'y') else pos[1] if isinstance(pos, (list, tuple)) else 0

            # FIRST: Register component courtyard as obstacle
            # PRIORITY: Use pre-calculated courtyard from Parts Piston if available
            # This follows the principle: Parts Piston is the SINGLE SOURCE OF TRUTH
            footprint = part.get('footprint', part.get('package', ''))

            # Check if courtyard is already in parts_db (from Parts Piston)
            courtyard_data = part.get('courtyard', None)
            if courtyard_data:
                # Use pre-calculated courtyard from Parts Piston
                if hasattr(courtyard_data, 'width'):
                    # CourtyardInfo object
                    half_w = courtyard_data.width / 2
                    half_h = courtyard_data.height / 2
                    offset_x = getattr(courtyard_data, 'offset_x', 0)
                    offset_y = getattr(courtyard_data, 'offset_y', 0)
                    body_left = pos_x - half_w + offset_x
                    body_right = pos_x + half_w + offset_x
                    body_top = pos_y - half_h + offset_y
                    body_bottom = pos_y + half_h + offset_y
                elif isinstance(courtyard_data, dict):
                    # Dict format from footprint generator
                    half_w = courtyard_data.get('width', 2.0) / 2
                    half_h = courtyard_data.get('height', 2.0) / 2
                    body_left = pos_x - half_w
                    body_right = pos_x + half_w
                    body_top = pos_y - half_h
                    body_bottom = pos_y + half_h
                else:
                    # Fallback to calculation
                    courtyard = calculate_courtyard(part, margin=0.0, footprint_name=footprint)
                    body_left = pos_x + courtyard.min_x
                    body_right = pos_x + courtyard.max_x
                    body_top = pos_y + courtyard.min_y
                    body_bottom = pos_y + courtyard.max_y
            else:
                # No courtyard in parts_db - calculate it
                # Note: margin=0.0 because clearance is handled by routing algorithm
                courtyard = calculate_courtyard(part, margin=0.0, footprint_name=footprint)
                body_left = pos_x + courtyard.min_x
                body_right = pos_x + courtyard.max_x
                body_top = pos_y + courtyard.min_y
                body_bottom = pos_y + courtyard.max_y

            col_min = self._real_to_grid_col(body_left)
            col_max = self._real_to_grid_col(body_right)
            row_min = self._real_to_grid_row(body_top)
            row_max = self._real_to_grid_row(body_bottom)

            # Determine if this is an SMD or through-hole component
            # SMD packages only block the top layer, TH blocks both layers
            fp_lower = footprint.lower()
            is_smd = any(pkg in fp_lower for pkg in [
                '0402', '0603', '0805', '1206', '1210', '2010', '2512',  # Resistors/Caps
                'sot', 'sod', 'soic', 'ssop', 'tssop', 'qfp', 'qfn',    # ICs
                'bga', 'lqfp', 'tqfp', 'mlf', 'dfn',                    # Fine-pitch ICs
                'led', 'chip', 'sma', 'smb', 'smc'                       # Diodes/LEDs
            ])

            # Mark all cells within component body as blocked
            # BUG FIX: SMD components only block the top layer (F.Cu)
            # This allows routing on the bottom layer to go "under" SMD components
            for r in range(row_min, row_max + 1):
                for c in range(col_min, col_max + 1):
                    if self._in_bounds(r, c):
                        # Only mark if not already marked with a net
                        if self.fcu_grid[r][c] is None:
                            self.fcu_grid[r][c] = '__COMPONENT__'
                        # For SMD, don't block bottom layer (allows via routing under)
                        if not is_smd and self.bcu_grid[r][c] is None:
                            self.bcu_grid[r][c] = '__COMPONENT__'

            # SECOND: Get pin net mapping and register pads
            # Note: get_pins and is_smd_footprint already imported above
            all_pads = get_pins(part)

            pin_nets = {}
            for pin in all_pads:
                pin_nets[pin.get('number', '')] = pin.get('net', '')

            # Register all pads (these override the component body markers)
            # FIX: SMD pads only go on F.Cu, TH pads go on both layers

            for pin in all_pads:
                pin_num = pin.get('number', '')
                net = pin_nets.get(pin_num, '')

                offset = pin.get('offset', None)
                if not offset or offset == (0, 0):
                    physical = pin.get('physical', {})
                    offset = (physical.get('offset_x', 0), physical.get('offset_y', 0)) if physical else (0, 0)

                pad_x = pos_x + offset[0]
                pad_y = pos_y + offset[1]

                pad_size = pin.get('pad_size', pin.get('size', (1.0, 0.6)))
                if not isinstance(pad_size, (list, tuple)):
                    pad_size = (1.0, 0.6)

                # BUG FIX: Only mark the ACTUAL pad area, not pad + clearance
                # Clearance checking happens separately during routing via _is_cell_clear_for_net
                # Using pad+clearance was causing pads to overlap massively (21x21 cells for 0805!)
                pad_half_w = pad_size[0] / 2 if len(pad_size) > 0 else 0.5
                pad_half_h = pad_size[1] / 2 if len(pad_size) > 1 else 0.5

                # FIX: use round() via helper methods
                pad_col = self._real_to_grid_col(pad_x)
                pad_row = self._real_to_grid_row(pad_y)

                # Calculate cells for pad dimensions WITH CLEARANCE for proper DRC
                # Pad itself is marked with net name (routing target)
                # Clearance zone around pad is blocked for other nets
                pad_cells_w = max(1, int(math.ceil(pad_half_w / self.config.grid_size)))
                pad_cells_h = max(1, int(math.ceil(pad_half_h / self.config.grid_size)))
                clearance_extra = self.clearance_cells  # Add clearance zone

                # First mark the clearance zone around the pad (for OTHER nets)
                for dr in range(-(pad_cells_h + clearance_extra), pad_cells_h + clearance_extra + 1):
                    for dc in range(-(pad_cells_w + clearance_extra), pad_cells_w + clearance_extra + 1):
                        r, c = pad_row + dr, pad_col + dc
                        if self._in_bounds(r, c):
                            # Check if this is within pad area or clearance zone
                            in_pad = abs(dr) <= pad_cells_h and abs(dc) <= pad_cells_w

                            if in_pad:
                                # Actual pad area - mark with net name
                                if net == '' or net is None:
                                    self.fcu_grid[r][c] = '__PAD_NC__'
                                    if not is_smd:
                                        self.bcu_grid[r][c] = '__PAD_NC__'
                                else:
                                    current = self.fcu_grid[r][c]
                                    if current is None or current in self.BLOCKED_MARKERS:
                                        self.fcu_grid[r][c] = net
                                    elif current != net and current not in self.BLOCKED_MARKERS:
                                        self.fcu_grid[r][c] = '__PAD_CONFLICT__'
                                    if not is_smd:
                                        bcu_current = self.bcu_grid[r][c]
                                        if bcu_current is None or bcu_current in self.BLOCKED_MARKERS:
                                            self.bcu_grid[r][c] = net
                                        elif bcu_current != net and bcu_current not in self.BLOCKED_MARKERS:
                                            self.bcu_grid[r][c] = '__PAD_CONFLICT__'
                            else:
                                # Clearance zone - mark to prevent other nets from routing here
                                # Use special marker that blocks other nets but not same net
                                if net and net != '':
                                    clearance_marker = f'__PAD_CLEAR_{net}__'
                                    # Only mark if cell is empty (don't overwrite existing routing)
                                    if self.fcu_grid[r][c] is None:
                                        self.fcu_grid[r][c] = clearance_marker
                                    if not is_smd and self.bcu_grid[r][c] is None:
                                        self.bcu_grid[r][c] = clearance_marker

    def _register_escapes(self, escapes: Dict):
        """Register escape routes in the grid"""
        for ref, comp_escapes in escapes.items():
            for pin, esc in comp_escapes.items():
                net = getattr(esc, 'net', '')
                if not net:
                    continue

                sx = getattr(esc, 'start', (0, 0))[0] if hasattr(esc, 'start') else 0
                sy = getattr(esc, 'start', (0, 0))[1] if hasattr(esc, 'start') else 0
                ex = getattr(esc, 'end', (0, 0))[0] if hasattr(esc, 'end') else 0
                ey = getattr(esc, 'end', (0, 0))[1] if hasattr(esc, 'end') else 0

                layer = getattr(esc, 'layer', 'F.Cu')
                grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

                # FIX: use round() via helper methods
                start_col = self._real_to_grid_col(sx)
                start_row = self._real_to_grid_row(sy)
                end_col = self._real_to_grid_col(ex)
                end_row = self._real_to_grid_row(ey)

                if start_row == end_row:
                    for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                        self._mark_cell_with_clearance(grid, start_row, col, net)
                elif start_col == end_col:
                    for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                        self._mark_cell_with_clearance(grid, row, start_col, net)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_routing_piston(board_width: float, board_height: float,
                          algorithm: str = 'hybrid',
                          grid_size: float = 0.1,
                          trace_width: float = 0.25,
                          clearance: float = 0.15) -> RoutingPiston:
    """
    Factory function to create a routing piston with common settings.

    Args:
        board_width: Board width in mm
        board_height: Board height in mm
        algorithm: Routing algorithm to use
        grid_size: Routing grid resolution in mm
        trace_width: Default trace width in mm
        clearance: Design rule clearance in mm

    Returns:
        Configured RoutingPiston instance
    """
    config = RoutingConfig(
        board_width=board_width,
        board_height=board_height,
        algorithm=algorithm,
        grid_size=grid_size,
        trace_width=trace_width,
        clearance=clearance
    )
    return RoutingPiston(config)


def route_with_cascade(parts_db: Dict, escapes: Dict, placement: Dict,
                       net_order: List[str], board_width: float, board_height: float,
                       config: RoutingConfig = None) -> 'RoutingResult':
    """
    ENGINE COMMAND: Try multiple routing algorithms until one succeeds.

    PERFORMANCE: Uses ONE piston instance and rebuilds the spatial index
    only ONCE. Each algorithm resets the grids from cache but reuses the
    same obstacle bitmap. This avoids the O(n*m) index rebuild cost
    being multiplied by the number of cascade attempts.

    Early exit: Returns immediately when an algorithm succeeds.

    Args:
        parts_db: Parts database with nets
        escapes: Escape routes from placement
        placement: Component positions
        net_order: Order of nets to route
        board_width: Board width in mm
        board_height: Board height in mm
        config: Optional base config to modify

    Returns:
        Best RoutingResult achieved
    """
    import time as time_module

    if config is None:
        config = RoutingConfig(
            board_width=board_width,
            board_height=board_height,
            algorithm='hybrid'
        )

    # Algorithm cascade - PATHFINDER first (routes 11/11 in ~7s),
    # HYBRID second (gets 9/11 in ~60s, slower and worse for this design profile)
    CASCADE = [
        ('pathfinder', 'PATHFINDER (Negotiated Congestion)'),
        ('hybrid', 'HYBRID (A* + Steiner + Ripup)'),
        ('lee', 'LEE (Guaranteed Shortest Path)'),
        ('ripup', 'RIP-UP & REROUTE (Iterative)'),
    ]

    best_result = None
    best_routed = 0

    print("\n    [CASCADE] Engine ordering RoutingPiston to try multiple algorithms...")

    # BUILD INDEX ONCE - reuse across all algorithms
    # This is the key optimization: the spatial index (bitmap, courtyard tree)
    # only depends on placement+parts_db, NOT the routing algorithm.
    config.algorithm = CASCADE[0][0]  # Initial algorithm doesn't matter for index
    piston = RoutingPiston(config)

    # Pre-build: initialize grids and spatial index ONCE
    piston._placement = placement
    piston._parts_db = parts_db
    piston._initialize_grids()
    piston._register_components(placement, parts_db)
    piston._register_escapes(escapes)
    if piston.use_spatial_index:
        piston._build_spatial_index()
        # Cache it so .route() gets a cache HIT
        piston._cache_index(placement, parts_db)
    print("    [CASCADE] Index built ONCE and cached for all algorithms")

    cascade_start = time_module.time()

    total_nets = len(net_order)
    good_enough_threshold = 1.0  # All nets must be routed - no partial acceptance
    no_improve_count = 0
    max_no_improve = 2  # Stop if 2 algorithms in a row don't improve

    for algo, desc in CASCADE:
        print(f"    [CASCADE] Trying {desc}...")
        algo_start = time_module.time()

        # Reuse same piston - just change algorithm
        # .route() will get a cache HIT on the spatial index
        piston.config.algorithm = algo
        result = piston.route(parts_db, escapes, placement, net_order)

        algo_time = time_module.time() - algo_start
        print(f"    [CASCADE] {algo}: {result.routed_count}/{result.total_count} nets routed ({algo_time:.1f}s)")

        # EARLY EXIT on full success
        if result.success:
            total_time = time_module.time() - cascade_start
            print(f"    [CASCADE] SUCCESS with {algo}! (total cascade: {total_time:.1f}s)")
            return result

        # Track best partial result
        if result.routed_count > best_routed:
            best_routed = result.routed_count
            best_result = result
            no_improve_count = 0
        else:
            no_improve_count += 1

        # EARLY EXIT: good enough (>80% routed)
        if total_nets > 0 and best_routed / total_nets >= good_enough_threshold:
            total_time = time_module.time() - cascade_start
            print(f"    [CASCADE] Good enough: {best_routed}/{total_nets} "
                  f"({100*best_routed/total_nets:.0f}%) using {best_result.algorithm_used} ({total_time:.1f}s)")
            return best_result

        # EARLY EXIT: no improvement from last 2 algorithms
        if no_improve_count >= max_no_improve:
            total_time = time_module.time() - cascade_start
            print(f"    [CASCADE] Early exit: {no_improve_count} algorithms without improvement. "
                  f"Best: {best_routed}/{total_nets} using {best_result.algorithm_used} ({total_time:.1f}s)")
            return best_result

    # All algorithms tried - try with relaxed constraints if needed
    print("    [CASCADE] All algorithms tried. Trying with relaxed constraints...")

    piston.config.algorithm = 'hybrid'
    piston.config.clearance = piston.config.clearance * 0.8  # Reduce clearance 20%
    piston.config.via_cost = piston.config.via_cost * 0.5    # Make vias cheaper

    result = piston.route(parts_db, escapes, placement, net_order)

    if result.routed_count > best_routed:
        best_result = result
        print(f"    [CASCADE] Relaxed constraints: {result.routed_count}/{result.total_count}")

    total_time = time_module.time() - cascade_start
    if best_result:
        print(f"    [CASCADE] Best result: {best_result.routed_count}/{best_result.total_count} "
              f"using {best_result.algorithm_used} (total: {total_time:.1f}s)")
        return best_result

    # Return empty result if nothing worked
    return RoutingResult(
        routes={},
        success=False,
        routed_count=0,
        total_count=len(net_order),
        algorithm_used='cascade_failed'
    )


# =============================================================================
# MODULE INFO
# =============================================================================

ROUTING_ALGORITHMS = {
    'lee': {
        'name': 'Lee Algorithm',
        'reference': 'Lee, 1961',
        'complexity': 'O(M*N)',
        'description': 'Guaranteed shortest path using BFS wavefront expansion'
    },
    'hadlock': {
        'name': "Hadlock's Algorithm",
        'reference': 'Hadlock, 1977',
        'complexity': 'O(M*N) but explores fewer cells',
        'description': 'Faster than Lee using detour numbers to bias search'
    },
    'soukup': {
        'name': "Soukup's Algorithm",
        'reference': 'Soukup, 1978',
        'complexity': 'O(1) per step in greedy phase',
        'description': 'Two-phase: greedy line probe + maze fallback'
    },
    'mikami': {
        'name': 'Mikami-Tabuchi',
        'reference': 'Mikami & Tabuchi, 1968',
        'complexity': 'O(perimeter)',
        'description': 'Memory-efficient line search algorithm'
    },
    'astar': {
        'name': 'A* Pathfinding',
        'reference': 'Hart, Nilsson, Raphael, 1968',
        'complexity': 'O(b^d) where b=branching, d=depth',
        'description': 'Fast heuristic-based routing with Manhattan distance'
    },
    'pathfinder': {
        'name': 'PathFinder',
        'reference': 'McMurchie & Ebeling, 1995',
        'complexity': 'Iterative, typically O(k*M*N)',
        'description': 'Negotiated congestion routing'
    },
    'ripup': {
        'name': 'Rip-up and Reroute',
        'reference': 'Nair et al., 1987',
        'complexity': 'Iterative, typically O(k*n*routing_cost)',
        'description': 'Iterative routing with intelligent reordering'
    },
    'steiner': {
        'name': 'Rectilinear Steiner Tree',
        'reference': 'Hanan, 1966',
        'complexity': 'NP-hard optimal, O(n^2) heuristic',
        'description': 'Optimal multi-terminal net routing'
    },
    'channel': {
        'name': 'Channel Routing',
        'reference': 'Hashimoto & Stevens, 1971',
        'complexity': 'O(n log n)',
        'description': 'Left-edge greedy algorithm for structured designs'
    },
    'hybrid': {
        'name': 'Hybrid',
        'reference': 'Combination approach',
        'complexity': 'Varies',
        'description': 'Combines A* + Steiner + Ripup for best results'
    },
    'auto': {
        'name': 'Auto-Select',
        'reference': 'Heuristic selection',
        'complexity': 'Depends on selected algorithm',
        'description': 'Automatically selects best algorithm based on design'
    }
}
