"""
PCB Engine - Advanced Routing Engine
=====================================

A dedicated sub-engine for PCB routing that implements state-of-the-art algorithms.

This module provides:
1. Lee Algorithm - Guaranteed shortest path with BFS wavefront expansion
2. Hadlock's Algorithm - Faster than Lee using detour numbers (biased search)
3. Soukup's Algorithm - Fast two-phase routing (greedy + maze fallback)
4. Mikami-Tabuchi - Memory-efficient line search algorithm
5. A* Pathfinding - Fast heuristic-based routing
6. PathFinder - Negotiated congestion routing (iterative cost adjustment)
7. Rip-up and Reroute - Iterative routing with intelligent reordering
8. Rectilinear Steiner Tree - Optimal multi-terminal net routing
9. Channel Routing - Left-edge greedy algorithm for structured regions

Based on research from:
- "A shortest path algorithm for grid graphs" (Hadlock, 1977)
- "PathFinder: A Negotiation-Based Performance-Driven Router" (McMurchie & Ebeling, 1995)
- "Mikami-Tabuchi: A computer program for optimal routing" (IFIP, 1968)
- "Soukup's Fast Maze Algorithm for Routing" (IEEE)
- "Mighty: A Rip-Up and Reroute Detailed Router" (ResearchGate)
- "Steiner Tree Construction in VLSI" (IEEE)
- "Left-Edge Algorithm for Channel Routing" (VLSI CAD)

Architecture follows the same pattern as placement_engine.py for consistency.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import heapq
from collections import deque
import random

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
# ROUTING ENGINE
# =============================================================================

class RoutingEngine:
    """
    Advanced PCB Routing Engine

    Provides multiple routing algorithms and automatically selects
    the best approach based on design complexity.

    Usage:
        config = RoutingConfig(
            board_width=50.0,
            board_height=40.0,
            grid_size=0.1,
            algorithm='hybrid'
        )
        engine = RoutingEngine(config)
        result = engine.route(parts_db, escapes, placement, net_order)
    """

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

        # Clearance calculation
        self.clearance_radius = config.trace_width / 2 + config.clearance
        self.clearance_cells = max(1, int(math.ceil(self.clearance_radius / config.grid_size)))

        # Board margin
        self.board_margin = max(1.0, config.clearance + config.trace_width / 2 + 0.5)
        self.margin_cells = max(2, int(math.ceil(self.board_margin / config.grid_size)))

        # Tracking
        self.placed_vias: Set[Tuple[float, float]] = set()
        self.routes: Dict[str, Route] = {}
        self.failed: List[str] = []

        # Special grid markers
        self.BLOCKED_MARKERS = {'__PAD_NC__', '__COMPONENT__', '__EDGE__', '__PAD_CONFLICT__'}

    def _initialize_grids(self):
        """Initialize or reset the occupancy grids"""
        self.fcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self.bcu_grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        self._mark_board_margins()
        self.placed_vias.clear()
        self.routes.clear()
        self.failed.clear()

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
    # MAIN ROUTING ENTRY POINT
    # =========================================================================

    def route(self, parts_db: Dict, escapes: Dict, placement: Dict,
              net_order: List[str]) -> RoutingResult:
        """
        Route all nets using the configured algorithm.

        Args:
            parts_db: Parts database with nets and component info
            escapes: Escape routes {ref: {pin: EscapeRoute}}
            placement: Component placements {ref: Position}
            net_order: Order of nets to route

        Returns:
            RoutingResult with all routes and statistics
        """
        # Initialize
        self._initialize_grids()

        # Build net_pins lookup
        nets = parts_db.get('nets', {})
        net_pins = {name: info.get('pins', []) for name, info in nets.items()}

        # Register obstacles
        self._register_components(placement, parts_db)
        self._register_escapes(escapes)

        # Select and run algorithm
        algorithm = self.config.algorithm.lower()

        if algorithm == 'lee':
            return self._route_lee(net_order, net_pins, escapes)
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
        else:
            # Default to hybrid
            return self._route_hybrid(net_order, net_pins, escapes, placement, parts_db)

    # =========================================================================
    # ALGORITHM 1: LEE WAVEFRONT ALGORITHM
    # =========================================================================

    def _route_lee(self, net_order: List[str], net_pins: Dict,
                   escapes: Dict) -> RoutingResult:
        """
        Route using Lee wavefront algorithm.

        Lee algorithm guarantees finding the shortest path if one exists.
        Uses BFS (breadth-first search) wavefront expansion.

        Complexity: O(N^2) where N is grid size
        """
        print("    [LEE] Using Lee wavefront algorithm...")

        for net_name in net_order:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = self._route_net_lee(net_name, pins, escapes)
            self.routes[net_name] = route

            if not route.success:
                self.failed.append(net_name)

        routed = sum(1 for r in self.routes.values() if r.success)
        return RoutingResult(
            routes=self.routes,
            success=len(self.failed) == 0,
            routed_count=routed,
            total_count=len(net_order),
            algorithm_used='lee',
            total_wirelength=sum(r.total_length for r in self.routes.values() if r.success),
            via_count=sum(len(r.vias) for r in self.routes.values() if r.success)
        )

    def _route_net_lee(self, net_name: str, pins: List[Tuple],
                       escapes: Dict) -> Route:
        """Route a single net using Lee algorithm"""
        route = Route(net=net_name, algorithm_used='lee')

        # Get escape endpoints
        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        # For multi-point nets, use Steiner-like approach with Lee for each segment
        connected = {endpoints[0]}
        unconnected = list(endpoints[1:])

        while unconnected:
            # Find closest pair
            best_dist = float('inf')
            best_uc = None
            best_c = None

            for uc in unconnected:
                for c in connected:
                    dist = abs(uc[0] - c[0]) + abs(uc[1] - c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_uc, best_c = uc, c

            if not best_uc:
                break

            # Route using Lee wavefront
            segments, success = self._lee_wavefront(best_uc, best_c, net_name, 'F.Cu')

            if not success:
                # Try bottom layer
                segments, success = self._lee_wavefront(best_uc, best_c, net_name, 'B.Cu')

            if success:
                route.segments.extend(segments)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                # Mark route in grid
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Lee failed to route from {best_uc} to {best_c}"
                return route

        route.success = len(unconnected) == 0
        return route

    def _lee_wavefront(self, start: Tuple[float, float], end: Tuple[float, float],
                       net_name: str, layer: str) -> Tuple[List[TrackSegment], bool]:
        """
        Lee wavefront expansion algorithm.

        1. Start from source, expand wavefront in all directions
        2. Label each cell with distance from source
        3. When target reached, backtrace to find path
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        # Convert to grid coordinates
        start_col = int((start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((end[1] - self.config.origin_y) / self.config.grid_size)

        # Bounds check
        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], False

        # Distance grid (-1 = unvisited)
        dist_grid = [[-1] * self.grid_cols for _ in range(self.grid_rows)]
        dist_grid[start_row][start_col] = 0

        # BFS wavefront expansion
        queue = deque([(start_row, start_col, 0)])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

        found = False
        iterations = 0
        max_iterations = self.config.lee_max_expansion

        while queue and iterations < max_iterations:
            iterations += 1
            row, col, dist = queue.popleft()

            # Reached target?
            if row == end_row and col == end_col:
                found = True
                break

            # Expand to neighbors
            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue
                if dist_grid[nr][nc] != -1:
                    continue  # Already visited
                if not self._is_cell_clear_for_net(grid, nr, nc, net_name):
                    continue

                dist_grid[nr][nc] = dist + 1
                queue.append((nr, nc, dist + 1))

        if not found:
            return [], False

        # Backtrace to find path
        path = [(end_row, end_col)]
        row, col = end_row, end_col

        while (row, col) != (start_row, start_col):
            current_dist = dist_grid[row][col]

            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if self._in_bounds(nr, nc) and dist_grid[nr][nc] == current_dist - 1:
                    path.append((nr, nc))
                    row, col = nr, nc
                    break

        path.reverse()

        # Convert path to segments
        return self._path_to_segments(path, net_name, layer)

    # =========================================================================
    # ALGORITHM 2: HADLOCK'S ALGORITHM (Detour-biased search)
    # =========================================================================

    def _route_hadlock(self, net_order: List[str], net_pins: Dict,
                       escapes: Dict) -> RoutingResult:
        """
        Route using Hadlock's algorithm.

        Hadlock improves on Lee by using "detour numbers" to bias the search
        toward the target. The detour number d(P) = number of cells directed
        away from target. Path length = Manhattan_distance + 2*detour_number.

        Minimizing detours finds shortest path faster than uniform BFS.
        Complexity: O(MN) but explores fewer cells than Lee.

        Reference: "A shortest path algorithm for grid graphs" (Hadlock, 1977)
        """
        print("    [HADLOCK] Using Hadlock's detour-biased algorithm...")

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
            via_count=sum(len(r.vias) for r in self.routes.values() if r.success)
        )

    def _route_net_hadlock(self, net_name: str, pins: List[Tuple],
                           escapes: Dict) -> Route:
        """Route a single net using Hadlock's algorithm"""
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

            segments, success = self._hadlock_route(best_uc, best_c, net_name, 'F.Cu')
            if not success:
                segments, success = self._hadlock_route(best_uc, best_c, net_name, 'B.Cu')

            if success:
                route.segments.extend(segments)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Hadlock failed from {best_uc} to {best_c}"
                return route

        route.success = len(unconnected) == 0
        return route

    def _hadlock_route(self, start: Tuple[float, float], end: Tuple[float, float],
                       net_name: str, layer: str) -> Tuple[List[TrackSegment], bool]:
        """
        Hadlock's algorithm using detour numbers.

        Key insight: For any cell, label it with detour number (moves away from target).
        Use priority queue ordered by detour number - explore low-detour cells first.
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        start_col = int((start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((end[1] - self.config.origin_y) / self.config.grid_size)

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], False

        # Detour grid (-1 = unvisited)
        detour_grid = [[-1] * self.grid_cols for _ in range(self.grid_rows)]
        detour_grid[start_row][start_col] = 0

        # Priority queue: (detour_number, row, col)
        # Lower detour = higher priority
        pq = [(0, start_row, start_col)]
        came_from = {}

        # Direction vectors: 0=Right, 1=Up, 2=Left, 3=Down
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]

        max_iterations = self.config.lee_max_expansion

        for _ in range(max_iterations):
            if not pq:
                break

            detour, row, col = heapq.heappop(pq)

            if row == end_row and col == end_col:
                # Reconstruct path
                path = [(row, col)]
                while (row, col) in came_from:
                    row, col = came_from[(row, col)]
                    path.append((row, col))
                path.reverse()
                return self._path_to_segments(path, net_name, layer)

            # Skip if we've found a better path to this cell
            if detour > detour_grid[row][col] and detour_grid[row][col] != -1:
                continue

            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue
                if not self._is_cell_clear_for_net(grid, nr, nc, net_name):
                    continue

                # Calculate detour: is this move away from target?
                # If Manhattan distance increases, it's a detour
                old_manhattan = abs(row - end_row) + abs(col - end_col)
                new_manhattan = abs(nr - end_row) + abs(nc - end_col)

                new_detour = detour
                if new_manhattan > old_manhattan:
                    new_detour += 1  # Moving away = detour

                if detour_grid[nr][nc] == -1 or new_detour < detour_grid[nr][nc]:
                    detour_grid[nr][nc] = new_detour
                    came_from[(nr, nc)] = (row, col)
                    heapq.heappush(pq, (new_detour, nr, nc))

        return [], False

    # =========================================================================
    # ALGORITHM 3: SOUKUP'S TWO-PHASE ALGORITHM
    # =========================================================================

    def _route_soukup(self, net_order: List[str], net_pins: Dict,
                      escapes: Dict) -> RoutingResult:
        """
        Route using Soukup's two-phase algorithm.

        Phase 1: Greedy line probe - try to reach target directly
        Phase 2: If blocked, fall back to maze routing (BFS)

        Much faster than Lee for open spaces, same worst-case complexity.
        """
        print("    [SOUKUP] Using Soukup's two-phase algorithm...")

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
            algorithm_used='soukup'
        )

    def _route_net_soukup(self, net_name: str, pins: List[Tuple],
                          escapes: Dict) -> Route:
        """Route a single net using Soukup's algorithm"""
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

            segments, success = self._soukup_route(best_uc, best_c, net_name, 'F.Cu')
            if not success:
                segments, success = self._soukup_route(best_uc, best_c, net_name, 'B.Cu')

            if success:
                route.segments.extend(segments)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Soukup failed from {best_uc} to {best_c}"
                return route

        route.success = len(unconnected) == 0
        return route

    def _soukup_route(self, start: Tuple[float, float], end: Tuple[float, float],
                      net_name: str, layer: str) -> Tuple[List[TrackSegment], bool]:
        """
        Soukup's two-phase routing.

        Phase 1: Try direct line probes (greedy)
        Phase 2: Fall back to maze routing if blocked
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        start_col = int((start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((end[1] - self.config.origin_y) / self.config.grid_size)

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], False

        # Phase 1: Try greedy line probe
        path = self._greedy_line_probe(start_row, start_col, end_row, end_col, grid, net_name)
        if path:
            return self._path_to_segments(path, net_name, layer)

        # Phase 2: Fall back to BFS (Lee-style) from all reached cells
        return self._lee_wavefront(start, end, net_name, layer)

    def _greedy_line_probe(self, start_row: int, start_col: int,
                            end_row: int, end_col: int,
                            grid: List[List], net_name: str) -> Optional[List[Tuple[int, int]]]:
        """
        Try to reach target using straight line probes.

        Strategy: Alternate between horizontal and vertical moves toward target.
        """
        path = [(start_row, start_col)]
        row, col = start_row, start_col

        max_moves = (abs(end_row - start_row) + abs(end_col - start_col)) * 2

        for _ in range(max_moves):
            if row == end_row and col == end_col:
                return path

            # Try to move toward target
            dr = 0 if row == end_row else (1 if end_row > row else -1)
            dc = 0 if col == end_col else (1 if end_col > col else -1)

            moved = False

            # Prefer horizontal if both needed
            if dc != 0:
                nr, nc = row, col + dc
                if self._in_bounds(nr, nc) and self._is_cell_clear_for_net(grid, nr, nc, net_name):
                    row, col = nr, nc
                    path.append((row, col))
                    moved = True

            if not moved and dr != 0:
                nr, nc = row + dr, col
                if self._in_bounds(nr, nc) and self._is_cell_clear_for_net(grid, nr, nc, net_name):
                    row, col = nr, nc
                    path.append((row, col))
                    moved = True

            if not moved:
                # Blocked - greedy fails
                return None

        return None

    # =========================================================================
    # ALGORITHM 4: MIKAMI-TABUCHI LINE SEARCH
    # =========================================================================

    def _route_mikami(self, net_order: List[str], net_pins: Dict,
                      escapes: Dict) -> RoutingResult:
        """
        Route using Mikami-Tabuchi line search algorithm.

        Instead of cell-by-cell expansion, extends lines from source and target.
        Creates perpendicular lines at escape points. When lines from source
        and target intersect, a path is found.

        Memory efficient: O(perimeter) vs O(area) for maze routers.
        Guaranteed to find a path if one exists.

        Reference: "A computer program for optimal routing of PCB connectors" (1968)
        """
        print("    [MIKAMI] Using Mikami-Tabuchi line search algorithm...")

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
            algorithm_used='mikami'
        )

    def _route_net_mikami(self, net_name: str, pins: List[Tuple],
                          escapes: Dict) -> Route:
        """Route a single net using Mikami-Tabuchi"""
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

            segments, success = self._mikami_route(best_uc, best_c, net_name, 'F.Cu')
            if not success:
                segments, success = self._mikami_route(best_uc, best_c, net_name, 'B.Cu')

            if success:
                route.segments.extend(segments)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Mikami failed from {best_uc} to {best_c}"
                return route

        route.success = len(unconnected) == 0
        return route

    def _mikami_route(self, start: Tuple[float, float], end: Tuple[float, float],
                      net_name: str, layer: str) -> Tuple[List[TrackSegment], bool]:
        """
        Mikami-Tabuchi line search.

        1. Create horizontal and vertical lines from source
        2. Create horizontal and vertical lines from target
        3. At each escape point on a line, create perpendicular lines
        4. When source and target lines intersect, reconstruct path
        """
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        start_col = int((start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((end[1] - self.config.origin_y) / self.config.grid_size)

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], False

        # Track lines: (level, is_horizontal, fixed_coord, min_var, max_var, from_source)
        source_lines = []  # Lines from source
        target_lines = []  # Lines from target

        # Level 0: Initial lines from source and target
        h_line_s, v_line_s = self._extend_lines(start_row, start_col, grid, net_name)
        h_line_t, v_line_t = self._extend_lines(end_row, end_col, grid, net_name)

        source_lines.append(('H', start_row, h_line_s[0], h_line_s[1]))
        source_lines.append(('V', start_col, v_line_s[0], v_line_s[1]))
        target_lines.append(('H', end_row, h_line_t[0], h_line_t[1]))
        target_lines.append(('V', end_col, v_line_t[0], v_line_t[1]))

        # Check for initial intersection
        intersect = self._find_line_intersection(source_lines, target_lines)
        if intersect:
            path = self._mikami_reconstruct_path(start_row, start_col, end_row, end_col, intersect)
            if path:
                return self._path_to_segments(path, net_name, layer)

        # Expand up to max levels
        max_levels = 20
        source_escape_pts = [(start_row, start_col)]
        target_escape_pts = [(end_row, end_col)]

        for level in range(max_levels):
            # Get new escape points from last level's lines
            new_source_pts = []
            new_target_pts = []

            for line in source_lines[-4:]:  # Last 4 lines (2H + 2V from last expansion)
                ltype, fixed, min_v, max_v = line
                if ltype == 'H':
                    for c in range(min_v, max_v + 1):
                        if (fixed, c) not in source_escape_pts:
                            new_source_pts.append((fixed, c))
                else:
                    for r in range(min_v, max_v + 1):
                        if (r, fixed) not in source_escape_pts:
                            new_source_pts.append((r, fixed))

            for line in target_lines[-4:]:
                ltype, fixed, min_v, max_v = line
                if ltype == 'H':
                    for c in range(min_v, max_v + 1):
                        if (fixed, c) not in target_escape_pts:
                            new_target_pts.append((fixed, c))
                else:
                    for r in range(min_v, max_v + 1):
                        if (r, fixed) not in target_escape_pts:
                            new_target_pts.append((r, fixed))

            # Extend lines from new escape points
            for row, col in new_source_pts[:50]:  # Limit expansion
                h_line, v_line = self._extend_lines(row, col, grid, net_name)
                source_lines.append(('H', row, h_line[0], h_line[1]))
                source_lines.append(('V', col, v_line[0], v_line[1]))
                source_escape_pts.append((row, col))

            for row, col in new_target_pts[:50]:
                h_line, v_line = self._extend_lines(row, col, grid, net_name)
                target_lines.append(('H', row, h_line[0], h_line[1]))
                target_lines.append(('V', col, v_line[0], v_line[1]))
                target_escape_pts.append((row, col))

            # Check for intersection
            intersect = self._find_line_intersection(source_lines, target_lines)
            if intersect:
                path = self._mikami_reconstruct_path(start_row, start_col, end_row, end_col, intersect)
                if path:
                    return self._path_to_segments(path, net_name, layer)

            if not new_source_pts and not new_target_pts:
                break  # No more expansion possible

        return [], False

    def _extend_lines(self, row: int, col: int, grid: List[List],
                      net_name: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Extend horizontal and vertical lines from a point until blocked"""
        # Horizontal line
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

        # Vertical line
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
            s_type, s_fixed, s_min, s_max = s_line
            for t_line in target_lines:
                t_type, t_fixed, t_min, t_max = t_line

                # H-V intersection
                if s_type == 'H' and t_type == 'V':
                    if t_min <= s_fixed <= t_max and s_min <= t_fixed <= s_max:
                        return (s_fixed, t_fixed)  # row, col
                elif s_type == 'V' and t_type == 'H':
                    if s_min <= t_fixed <= s_max and t_min <= s_fixed <= t_max:
                        return (t_fixed, s_fixed)  # row, col

        return None

    def _mikami_reconstruct_path(self, start_row: int, start_col: int,
                                  end_row: int, end_col: int,
                                  intersect: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path through intersection point"""
        int_row, int_col = intersect

        # Path: start -> intersect -> end (L-shape or Z-shape)
        path = []

        # Start to intersection
        if start_row == int_row:
            # Same row - horizontal first
            step = 1 if int_col > start_col else -1
            for c in range(start_col, int_col + step, step):
                path.append((start_row, c))
        else:
            # Different row - go vertical then horizontal
            step_r = 1 if int_row > start_row else -1
            for r in range(start_row, int_row + step_r, step_r):
                path.append((r, start_col))
            step_c = 1 if int_col > start_col else -1
            for c in range(start_col + step_c, int_col + step_c, step_c):
                path.append((int_row, c))

        # Intersection to end (avoid duplicating intersection)
        if int_row == end_row:
            # Same row - horizontal
            step = 1 if end_col > int_col else -1
            for c in range(int_col + step, end_col + step, step):
                path.append((end_row, c))
        else:
            # Go vertical then horizontal
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
            algorithm_used='astar'
        )

    def _route_net_astar(self, net_name: str, pins: List[Tuple],
                         escapes: Dict) -> Route:
        """Route a single net using A*"""
        route = Route(net=net_name, algorithm_used='astar')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        # MST-style routing with A*
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

            # Try A* on both layers
            segments, success = self._astar_route(best_uc, best_c, net_name, 'F.Cu')
            if not success:
                segments, success = self._astar_route(best_uc, best_c, net_name, 'B.Cu')

            if success:
                route.segments.extend(segments)
                connected.add(best_uc)
                unconnected.remove(best_uc)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"A* failed for {net_name}"
                return route

        route.success = len(unconnected) == 0
        return route

    def _astar_route(self, start: Tuple[float, float], end: Tuple[float, float],
                     net_name: str, layer: str) -> Tuple[List[TrackSegment], bool]:
        """A* pathfinding between two points"""
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        start_col = int((start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((end[1] - self.config.origin_y) / self.config.grid_size)

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], False

        def heuristic(r, c):
            return abs(r - end_row) + abs(c - end_col)

        open_set = [(heuristic(start_row, start_col), 0, start_row, start_col)]
        came_from = {}
        g_score = {(start_row, start_col): 0}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        max_iterations = self.grid_rows * self.grid_cols * 2

        for _ in range(max_iterations):
            if not open_set:
                break

            f, g, row, col = heapq.heappop(open_set)

            if (row, col) == (end_row, end_col):
                # Reconstruct path
                path = [(row, col)]
                while (row, col) in came_from:
                    row, col = came_from[(row, col)]
                    path.append((row, col))
                path.reverse()
                return self._path_to_segments(path, net_name, layer)

            if g > g_score.get((row, col), float('inf')):
                continue

            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue
                if not self._is_cell_clear_for_net(grid, nr, nc, net_name):
                    continue

                tentative_g = g + 1
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = (row, col)
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        return [], False

    # =========================================================================
    # ALGORITHM 3: RIP-UP AND REROUTE
    # =========================================================================

    def _route_ripup_reroute(self, net_order: List[str], net_pins: Dict,
                              escapes: Dict, placement: Dict,
                              parts_db: Dict) -> RoutingResult:
        """
        Rip-up and Reroute algorithm.

        Iteratively:
        1. Route all nets in current order
        2. Identify failed nets and their blockers
        3. Rip-up (remove) blocking nets
        4. Reorder with failed nets first
        5. Repeat until success or max iterations
        """
        print("    [RIPUP] Using Rip-up and Reroute algorithm...")

        routing_attempts = {net: 0 for net in net_order}
        blocking_history = {}
        current_order = self._sort_by_length(net_order, net_pins, escapes)

        best_result = None
        best_routed = -1

        for iteration in range(self.config.max_ripup_iterations):
            # Reset grids (keep obstacles)
            self._initialize_grids()
            self._register_components(placement, parts_db)
            self._register_escapes(escapes)

            # Route all nets
            failed_nets = []
            for net_name in current_order:
                pins = net_pins.get(net_name, [])
                if len(pins) < 2:
                    continue

                route = self._route_net_astar(net_name, pins, escapes)
                self.routes[net_name] = route
                routing_attempts[net_name] += 1

                if not route.success:
                    failed_nets.append(net_name)
                    blockers = self._identify_blockers(net_name, pins, escapes)
                    blocking_history[net_name] = blockers

            routed = sum(1 for r in self.routes.values() if r.success)

            # Track best result
            if routed > best_routed:
                best_routed = routed
                best_result = RoutingResult(
                    routes=dict(self.routes),
                    success=len(failed_nets) == 0,
                    routed_count=routed,
                    total_count=len(net_order),
                    algorithm_used='ripup',
                    iterations=iteration + 1
                )

            # Success?
            if not failed_nets:
                print(f"    [RIPUP] Success! All {len(net_order)} nets routed in {iteration + 1} iterations")
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
        """Identify nets blocking this net"""
        blockers = set()
        endpoints = self._get_escape_endpoints(pins, escapes)

        if len(endpoints) < 2:
            return []

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                start, end = endpoints[i], endpoints[j]

                start_col = int((start[0] - self.config.origin_x) / self.config.grid_size)
                start_row = int((start[1] - self.config.origin_y) / self.config.grid_size)
                end_col = int((end[0] - self.config.origin_x) / self.config.grid_size)
                end_row = int((end[1] - self.config.origin_y) / self.config.grid_size)

                # Check corridor
                for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                    if self._in_bounds(start_row, col):
                        occ = self.fcu_grid[start_row][col]
                        if occ and occ not in self.BLOCKED_MARKERS and occ != net_name:
                            blockers.add(occ)

                for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                    if self._in_bounds(row, end_col):
                        occ = self.fcu_grid[row][end_col]
                        if occ and occ not in self.BLOCKED_MARKERS and occ != net_name:
                            blockers.add(occ)

        return list(blockers)

    def _reorder_for_ripup(self, current_order: List[str], failed: List[str],
                           blocking_history: Dict, attempts: Dict) -> List[str]:
        """Reorder nets for next rip-up iteration"""
        block_count = {}
        for blocked, blockers in blocking_history.items():
            for blocker in blockers:
                block_count[blocker] = block_count.get(blocker, 0) + 1

        def key(net):
            is_failed = 0 if net in failed else 1
            blocks = block_count.get(net, 0)
            att = attempts.get(net, 0)
            return (is_failed, -blocks, att)

        return sorted(current_order, key=key)

    # =========================================================================
    # ALGORITHM 4: RECTILINEAR STEINER TREE
    # =========================================================================

    def _route_steiner(self, net_order: List[str], net_pins: Dict,
                       escapes: Dict) -> RoutingResult:
        """
        Route using Rectilinear Steiner Minimum Tree (RSMT) algorithm.

        For multi-terminal nets, RSMT can reduce wirelength by 13-15% vs MST
        by adding Steiner points (intermediate junctions).
        """
        print("    [STEINER] Using Steiner tree algorithm...")

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
            algorithm_used='steiner'
        )

    def _route_net_steiner(self, net_name: str, pins: List[Tuple],
                           escapes: Dict) -> Route:
        """Route a single net using Steiner tree approach"""
        route = Route(net=net_name, algorithm_used='steiner')

        endpoints = self._get_escape_endpoints(pins, escapes)
        if len(endpoints) < 2:
            route.error = "Not enough endpoints"
            return route

        if len(endpoints) == 2:
            # Simple 2-point case - just use A*
            segments, success = self._astar_route(endpoints[0], endpoints[1], net_name, 'F.Cu')
            if success:
                route.segments.extend(segments)
                route.success = True
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            return route

        # Multi-point case: Compute RSMT
        steiner_points = self._compute_steiner_points(endpoints)

        # Build graph with original endpoints + Steiner points
        all_points = endpoints + steiner_points

        # Use MST on this expanded point set
        edges = self._compute_mst_edges(all_points)

        # Route each edge
        for p1, p2 in edges:
            segments, success = self._astar_route(p1, p2, net_name, 'F.Cu')
            if not success:
                segments, success = self._astar_route(p1, p2, net_name, 'B.Cu')

            if success:
                route.segments.extend(segments)
                for seg in segments:
                    self._mark_segment_in_grid(seg, net_name)
            else:
                route.error = f"Steiner edge failed: {p1} to {p2}"
                return route

        route.success = True
        return route

    def _compute_steiner_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Compute Steiner points for rectilinear Steiner tree.

        Uses Hanan grid approach: potential Steiner points are at intersections
        of horizontal and vertical lines through terminal points.
        """
        if len(points) <= 2:
            return []

        steiner_points = []
        xs = sorted(set(p[0] for p in points))
        ys = sorted(set(p[1] for p in points))

        # Hanan grid intersections (excluding original points)
        original_set = set(points)
        for x in xs:
            for y in ys:
                if (x, y) not in original_set:
                    # Only add if it helps (simple heuristic: inside bounding box)
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    if min_x < x < max_x and min_y < y < max_y:
                        steiner_points.append((x, y))

        # Limit to avoid explosion
        if len(steiner_points) > 10:
            # Keep only the most central ones
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            steiner_points.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
            steiner_points = steiner_points[:10]

        return steiner_points

    def _compute_mst_edges(self, points: List[Tuple[float, float]]) -> List[Tuple[Tuple, Tuple]]:
        """Compute MST edges using Prim's algorithm"""
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
    # ALGORITHM 6: PATHFINDER (Negotiated Congestion Routing)
    # =========================================================================

    def _route_pathfinder(self, net_order: List[str], net_pins: Dict,
                          escapes: Dict, placement: Dict,
                          parts_db: Dict) -> RoutingResult:
        """
        Route using PathFinder negotiated congestion algorithm.

        PathFinder iteratively routes all nets, allowing initial overlaps.
        Congested resources have their costs increased, forcing nets to
        negotiate and find alternative paths. Converges when no overlaps remain.

        Key innovation: Cost = base + history_cost * penalty_factor
        - base: basic routing cost
        - history_cost: how often this cell has been contested
        - penalty_factor: increases each iteration

        Reference: "PathFinder: A Negotiation-Based Performance-Driven Router" (1995)
        """
        print("    [PATHFINDER] Using PathFinder negotiated congestion routing...")

        # Cost grids: track contention history
        fcu_history = [[0] * self.grid_cols for _ in range(self.grid_rows)]
        bcu_history = [[0] * self.grid_cols for _ in range(self.grid_rows)]

        # Current occupancy (can overlap initially)
        fcu_occupants: Dict[Tuple[int, int], Set[str]] = {}
        bcu_occupants: Dict[Tuple[int, int], Set[str]] = {}

        best_result = None
        penalty_factor = 1.0
        penalty_increment = 0.5

        for iteration in range(self.config.max_ripup_iterations):
            # Clear occupancy for this iteration
            fcu_occupants.clear()
            bcu_occupants.clear()

            # Route all nets with current costs
            all_routes = {}
            for net_name in net_order:
                pins = net_pins.get(net_name, [])
                if len(pins) < 2:
                    continue

                route = self._route_net_pathfinder(
                    net_name, pins, escapes, 'F.Cu',
                    fcu_history, fcu_occupants, penalty_factor
                )
                all_routes[net_name] = route

                # Register occupancy (even if overlapping)
                if route.success:
                    for seg in route.segments:
                        self._register_pathfinder_occupancy(
                            seg, net_name,
                            fcu_occupants if seg.layer == 'F.Cu' else bcu_occupants
                        )

            # Check for overlaps
            overlaps = self._count_pathfinder_overlaps(fcu_occupants, bcu_occupants)

            routed = sum(1 for r in all_routes.values() if r.success)

            if best_result is None or routed > best_result.routed_count:
                best_result = RoutingResult(
                    routes=dict(all_routes),
                    success=overlaps == 0 and routed == len(net_order),
                    routed_count=routed,
                    total_count=len(net_order),
                    algorithm_used='pathfinder',
                    iterations=iteration + 1
                )

            if overlaps == 0:
                print(f"    [PATHFINDER] Converged in {iteration + 1} iterations")
                self.routes = all_routes
                return best_result

            # Update history costs for contested cells
            self._update_pathfinder_history(fcu_occupants, fcu_history)
            self._update_pathfinder_history(bcu_occupants, bcu_history)

            # Increase penalty
            penalty_factor += penalty_increment

            print(f"    [PATHFINDER] Iteration {iteration + 1}: {overlaps} overlaps, penalty={penalty_factor:.1f}")

        print(f"    [PATHFINDER] Max iterations reached")
        return best_result or RoutingResult(
            routes=self.routes,
            success=False,
            routed_count=0,
            total_count=len(net_order),
            algorithm_used='pathfinder'
        )

    def _route_net_pathfinder(self, net_name: str, pins: List[Tuple],
                               escapes: Dict, layer: str,
                               history: List[List[int]],
                               occupants: Dict, penalty: float) -> Route:
        """Route a net considering congestion costs"""
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

            segments, success = self._pathfinder_astar(
                best_uc, best_c, net_name, layer, history, occupants, penalty
            )

            if success:
                route.segments.extend(segments)
                connected.add(best_uc)
                unconnected.remove(best_uc)
            else:
                route.error = f"PathFinder failed for {net_name}"
                return route

        route.success = len(unconnected) == 0
        return route

    def _pathfinder_astar(self, start: Tuple[float, float], end: Tuple[float, float],
                          net_name: str, layer: str,
                          history: List[List[int]], occupants: Dict,
                          penalty: float) -> Tuple[List[TrackSegment], bool]:
        """A* with PathFinder cost function"""
        grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

        start_col = int((start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((end[1] - self.config.origin_y) / self.config.grid_size)

        if not self._in_bounds(start_row, start_col) or not self._in_bounds(end_row, end_col):
            return [], False

        def cost(r, c):
            """PathFinder cost: base + history * penalty"""
            base = 1.0
            h = history[r][c] if self._in_bounds(r, c) else 0
            # Also add cost for current occupancy
            occ = occupants.get((r, c), set())
            occ_cost = len(occ) * 5 if occ and net_name not in occ else 0
            return base + h * penalty + occ_cost

        def heuristic(r, c):
            return abs(r - end_row) + abs(c - end_col)

        open_set = [(heuristic(start_row, start_col), 0, start_row, start_col)]
        came_from = {}
        g_score = {(start_row, start_col): 0}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        max_iterations = self.grid_rows * self.grid_cols * 2

        for _ in range(max_iterations):
            if not open_set:
                break

            f, g, row, col = heapq.heappop(open_set)

            if (row, col) == (end_row, end_col):
                path = [(row, col)]
                while (row, col) in came_from:
                    row, col = came_from[(row, col)]
                    path.append((row, col))
                path.reverse()
                return self._path_to_segments(path, net_name, layer)

            if g > g_score.get((row, col), float('inf')):
                continue

            for dr, dc in directions:
                nr, nc = row + dr, col + dc

                if not self._in_bounds(nr, nc):
                    continue

                # In PathFinder, we allow routing through occupied cells (with cost)
                cell = grid[nr][nc]
                if cell in self.BLOCKED_MARKERS:
                    continue

                move_cost = cost(nr, nc)
                tentative_g = g + move_cost

                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = (row, col)
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        return [], False

    def _register_pathfinder_occupancy(self, segment: TrackSegment, net_name: str,
                                        occupants: Dict[Tuple[int, int], Set[str]]):
        """Register segment cells in occupancy map"""
        start_col = int((segment.start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((segment.start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((segment.end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((segment.end[1] - self.config.origin_y) / self.config.grid_size)

        if start_row == end_row:
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                key = (start_row, col)
                if key not in occupants:
                    occupants[key] = set()
                occupants[key].add(net_name)
        elif start_col == end_col:
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                key = (row, start_col)
                if key not in occupants:
                    occupants[key] = set()
                occupants[key].add(net_name)

    def _count_pathfinder_overlaps(self, fcu_occ: Dict, bcu_occ: Dict) -> int:
        """Count cells with multiple net occupants"""
        overlaps = 0
        for cell, nets in fcu_occ.items():
            if len(nets) > 1:
                overlaps += 1
        for cell, nets in bcu_occ.items():
            if len(nets) > 1:
                overlaps += 1
        return overlaps

    def _update_pathfinder_history(self, occupants: Dict, history: List[List[int]]):
        """Update history costs for contested cells"""
        for (row, col), nets in occupants.items():
            if len(nets) > 1 and self._in_bounds(row, col):
                history[row][col] += 1

    # =========================================================================
    # ALGORITHM 7: CHANNEL ROUTING (Left-Edge Greedy)
    # =========================================================================

    def _route_channel(self, net_order: List[str], net_pins: Dict,
                       escapes: Dict) -> RoutingResult:
        """
        Route using channel/greedy routing approach.

        Left-Edge Algorithm:
        1. Sort nets by leftmost pin position
        2. Assign nets to tracks (horizontal channels)
        3. Route horizontally first, then vertically

        Best for designs with organized pin structures (like bus connections).
        """
        print("    [CHANNEL] Using channel/left-edge routing...")

        # Sort nets by leftmost pin position
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
            algorithm_used='channel'
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

        # Find trunk y-coordinate (try centroid first)
        center_y = sum(p[1] for p in endpoints) / len(endpoints)

        # Try to create horizontal trunk
        trunk_row = int((center_y - self.config.origin_y) / self.config.grid_size)
        left_x = sorted_pts[0][0]
        right_x = sorted_pts[-1][0]

        # Check if trunk is clear
        trunk_ok = True
        left_col = int((left_x - self.config.origin_x) / self.config.grid_size)
        right_col = int((right_x - self.config.origin_x) / self.config.grid_size)

        for col in range(left_col, right_col + 1):
            if not self._is_cell_clear_for_net(self.fcu_grid, trunk_row, col, net_name):
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
                layer='F.Cu',
                width=self.config.trace_width,
                net=net_name
            )
            route.segments.append(trunk_seg)
            self._mark_segment_in_grid(trunk_seg, net_name)

            # Create vertical branches to each endpoint
            for pt in endpoints:
                pt_col = int((pt[0] - self.config.origin_x) / self.config.grid_size)
                pt_row = int((pt[1] - self.config.origin_y) / self.config.grid_size)

                if pt_row != trunk_row:
                    # Need vertical segment
                    branch_x = self.config.origin_x + pt_col * self.config.grid_size
                    branch_start_y = trunk_y
                    branch_end_y = self.config.origin_y + pt_row * self.config.grid_size

                    branch_seg = TrackSegment(
                        start=(branch_x, branch_start_y),
                        end=(branch_x, branch_end_y),
                        layer='F.Cu',
                        width=self.config.trace_width,
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

                segments, success = self._astar_route(best_uc, best_c, net_name, 'F.Cu')
                if not success:
                    segments, success = self._astar_route(best_uc, best_c, net_name, 'B.Cu')

                if success:
                    route.segments.extend(segments)
                    connected.add(best_uc)
                    unconnected.remove(best_uc)
                    for seg in segments:
                        self._mark_segment_in_grid(seg, net_name)
                else:
                    route.error = f"Channel routing failed for {net_name}"
                    return route

            route.success = len(unconnected) == 0

        return route

    # =========================================================================
    # ALGORITHM 8: HYBRID (BEST OF ALL)
    # =========================================================================

    def _route_hybrid(self, net_order: List[str], net_pins: Dict,
                      escapes: Dict, placement: Dict, parts_db: Dict) -> RoutingResult:
        """
        Hybrid routing: combines multiple algorithms for best results.

        Strategy:
        1. Try Lee algorithm first (guarantees optimal paths)
        2. If that fails, use rip-up and reroute
        3. For multi-terminal nets, use Steiner trees
        """
        print("    [HYBRID] Using hybrid routing (Lee + Ripup + Steiner)...")

        # First pass: Lee algorithm
        result = self._route_lee(net_order, net_pins, escapes)

        if result.success:
            return result

        # Not all routed - try rip-up and reroute
        print(f"    [HYBRID] Lee routed {result.routed_count}/{result.total_count}, trying ripup...")

        ripup_result = self._route_ripup_reroute(net_order, net_pins, escapes, placement, parts_db)

        if ripup_result.routed_count > result.routed_count:
            return ripup_result

        return result

    def _route_auto(self, net_order: List[str], net_pins: Dict,
                    escapes: Dict, placement: Dict, parts_db: Dict) -> RoutingResult:
        """Auto-select best algorithm based on design complexity"""
        # Simple heuristic: use hybrid for most cases
        return self._route_hybrid(net_order, net_pins, escapes, placement, parts_db)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if grid coordinates are in bounds"""
        return 0 <= row < self.grid_rows and 0 <= col < self.grid_cols

    def _is_cell_clear_for_net(self, grid: List[List], row: int, col: int,
                                net_name: str) -> bool:
        """Check if a cell is clear for routing this net (with clearance)"""
        # Check center
        if self._in_bounds(row, col):
            center = grid[row][col]
            if center in self.BLOCKED_MARKERS:
                return False
            if center is not None and center != net_name:
                return False

        # Check clearance zone
        cc = self.clearance_cells
        for dr in range(-cc, cc + 1):
            for dc in range(-cc, cc + 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if self._in_bounds(r, c):
                    occ = grid[r][c]
                    if occ in self.BLOCKED_MARKERS:
                        return False
                    if occ is not None and occ != net_name:
                        return False

        return True

    def _get_escape_endpoints(self, pins: List[Tuple], escapes: Dict) -> List[Tuple[float, float]]:
        """Get escape endpoints for a net's pins"""
        endpoints = []
        for comp, pin in pins:
            if comp in escapes and pin in escapes[comp]:
                esc = escapes[comp][pin]
                end = esc.end if hasattr(esc, 'end') else getattr(esc, 'endpoint', None)
                if end:
                    endpoints.append(end)
        return endpoints

    def _sort_by_length(self, nets: List[str], net_pins: Dict,
                        escapes: Dict) -> List[str]:
        """Sort nets by estimated length (short first)"""
        def estimate(net):
            endpoints = self._get_escape_endpoints(net_pins.get(net, []), escapes)
            if len(endpoints) < 2:
                return float('inf')
            total = sum(
                abs(endpoints[i+1][0] - endpoints[i][0]) + abs(endpoints[i+1][1] - endpoints[i][1])
                for i in range(len(endpoints) - 1)
            )
            return total
        return sorted(nets, key=estimate)

    def _path_to_segments(self, path: List[Tuple[int, int]], net_name: str,
                          layer: str) -> Tuple[List[TrackSegment], bool]:
        """Convert grid path to track segments"""
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
                # Direction changed - emit segment
                start_x = self.config.origin_x + segment_start[1] * self.config.grid_size
                start_y = self.config.origin_y + segment_start[0] * self.config.grid_size
                end_x = self.config.origin_x + prev[1] * self.config.grid_size
                end_y = self.config.origin_y + prev[0] * self.config.grid_size

                segments.append(TrackSegment(
                    start=(start_x, start_y),
                    end=(end_x, end_y),
                    layer=layer,
                    width=self.config.trace_width,
                    net=net_name
                ))

                segment_start = prev
                current_dir = direction

        # Final segment
        start_x = self.config.origin_x + segment_start[1] * self.config.grid_size
        start_y = self.config.origin_y + segment_start[0] * self.config.grid_size
        end_x = self.config.origin_x + path[-1][1] * self.config.grid_size
        end_y = self.config.origin_y + path[-1][0] * self.config.grid_size

        segments.append(TrackSegment(
            start=(start_x, start_y),
            end=(end_x, end_y),
            layer=layer,
            width=self.config.trace_width,
            net=net_name
        ))

        return segments, True

    def _mark_segment_in_grid(self, segment: TrackSegment, net_name: str):
        """Mark a segment in the occupancy grid"""
        grid = self.fcu_grid if segment.layer == 'F.Cu' else self.bcu_grid

        start_col = int((segment.start[0] - self.config.origin_x) / self.config.grid_size)
        start_row = int((segment.start[1] - self.config.origin_y) / self.config.grid_size)
        end_col = int((segment.end[0] - self.config.origin_x) / self.config.grid_size)
        end_row = int((segment.end[1] - self.config.origin_y) / self.config.grid_size)

        # Mark cells along segment with clearance
        if start_row == end_row:
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                self._mark_cell_with_clearance(grid, start_row, col, net_name)
        elif start_col == end_col:
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                self._mark_cell_with_clearance(grid, row, start_col, net_name)
        else:
            # Diagonal - use Bresenham
            dx = end_col - start_col
            dy = end_row - start_row
            steps = max(abs(dx), abs(dy))
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                col = int(start_col + t * dx)
                row = int(start_row + t * dy)
                self._mark_cell_with_clearance(grid, row, col, net_name)

    def _mark_cell_with_clearance(self, grid: List[List], row: int, col: int, net_name: str):
        """Mark a cell and its clearance zone"""
        cc = self.clearance_cells
        for dr in range(-cc, cc + 1):
            for dc in range(-cc, cc + 1):
                r, c = row + dr, col + dc
                if self._in_bounds(r, c):
                    if grid[r][c] is None or grid[r][c] == net_name:
                        grid[r][c] = net_name

    def _register_components(self, placement: Dict, parts_db: Dict):
        """Register component pads and bodies as obstacles"""
        parts = parts_db.get('parts', {})
        clearance_margin = self.config.clearance + self.config.trace_width / 2

        for ref, pos in placement.items():
            part = parts.get(ref, {})

            # Get pin net mapping
            pin_nets = {}
            for pin in part.get('used_pins', []):
                pin_nets[pin.get('number', '')] = pin.get('net', '')

            # Register all pads
            all_pads = part.get('physical_pins', part.get('used_pins', []))

            for pin in all_pads:
                pin_num = pin.get('number', '')
                net = pin_nets.get(pin_num, '')

                offset = pin.get('offset', None)
                if not offset or offset == (0, 0):
                    physical = pin.get('physical', {})
                    offset = (physical.get('offset_x', 0), physical.get('offset_y', 0)) if physical else (0, 0)

                pad_x = pos.x + offset[0]
                pad_y = pos.y + offset[1]

                pad_size = pin.get('pad_size', pin.get('size', (1.0, 0.6)))
                if not isinstance(pad_size, (list, tuple)):
                    pad_size = (1.0, 0.6)

                pad_half = max(pad_size) / 2
                track_half = self.config.trace_width / 2
                pad_radius = pad_half + track_half + self.config.clearance

                pad_col = int((pad_x - self.config.origin_x) / self.config.grid_size)
                pad_row = int((pad_y - self.config.origin_y) / self.config.grid_size)
                pad_cells = max(1, int(math.ceil(pad_radius / self.config.grid_size)))

                for dr in range(-pad_cells, pad_cells + 1):
                    for dc in range(-pad_cells, pad_cells + 1):
                        r, c = pad_row + dr, pad_col + dc
                        if self._in_bounds(r, c):
                            if net == '' or net is None:
                                self.fcu_grid[r][c] = '__PAD_NC__'
                                self.bcu_grid[r][c] = '__PAD_NC__'
                            else:
                                current = self.fcu_grid[r][c]
                                if current is None or current == net:
                                    self.fcu_grid[r][c] = net
                                elif current not in self.BLOCKED_MARKERS:
                                    self.fcu_grid[r][c] = '__PAD_CONFLICT__'

    def _register_escapes(self, escapes: Dict):
        """Register escape routes in the grid"""
        for ref, comp_escapes in escapes.items():
            for pin, esc in comp_escapes.items():
                net = esc.net if hasattr(esc, 'net') else ''
                if not net:
                    continue

                sx, sy = esc.start if hasattr(esc, 'start') else (0, 0)
                ex, ey = esc.end if hasattr(esc, 'end') else (0, 0)

                layer = esc.layer if hasattr(esc, 'layer') else 'F.Cu'
                grid = self.fcu_grid if layer == 'F.Cu' else self.bcu_grid

                start_col = int((sx - self.config.origin_x) / self.config.grid_size)
                start_row = int((sy - self.config.origin_y) / self.config.grid_size)
                end_col = int((ex - self.config.origin_x) / self.config.grid_size)
                end_row = int((ey - self.config.origin_y) / self.config.grid_size)

                if start_row == end_row:
                    for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                        self._mark_cell_with_clearance(grid, start_row, col, net)
                elif start_col == end_col:
                    for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                        self._mark_cell_with_clearance(grid, row, start_col, net)
