"""Debug the wavefront expansion"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque

# Simple voltage divider - 3 components
parts_db = {
    'parts': {
        'R1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'VDIV', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'R2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'C1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VIN': {'pins': [('R1', '1')]},
        'VDIV': {'pins': [('R1', '2'), ('R2', '1'), ('C1', '1')]},
        'GND': {'pins': [('R2', '2'), ('C1', '2')]},
    }
}

print("=" * 70)
print("DEBUG: Wavefront Expansion")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions
print(f"Placement: {placement}")

# Routing setup
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))
rp._initialize_grids()
rp._placement = placement
rp._parts_db = parts_db
rp._register_components(placement, parts_db)

# Get endpoints for VDIV
endpoints = rp._get_escape_endpoints(parts_db['nets']['VDIV']['pins'], {})
print(f"\nVDIV endpoints: {endpoints}")

if len(endpoints) >= 2:
    start = endpoints[0]
    end = endpoints[1]
    net_name = 'VDIV'

    start_col = int((start[0] - 0) / 0.1)
    start_row = int((start[1] - 0) / 0.1)
    end_col = int((end[0] - 0) / 0.1)
    end_row = int((end[1] - 0) / 0.1)

    print(f"\nStart: ({start[0]}, {start[1]}) -> grid[{start_row}][{start_col}]")
    print(f"End: ({end[0]}, {end[1]}) -> grid[{end_row}][{end_col}]")

    # Check start cell accessibility
    print(f"\nStart cell accessibility: {rp._is_cell_accessible_for_net(rp.fcu_grid, start_row, start_col, net_name)}")
    print(f"End cell accessibility: {rp._is_cell_accessible_for_net(rp.fcu_grid, end_row, end_col, net_name)}")

    # Manual wavefront expansion
    print("\n--- Manual wavefront expansion ---")
    target_cells = {(start_row, start_col), (end_row, end_col)}

    dist_grid = [[-1] * rp.grid_cols for _ in range(rp.grid_rows)]
    dist_grid[start_row][start_col] = 0

    queue = deque([(0, start_row, start_col)])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    max_iterations = 100000
    iterations = 0
    found = False

    while queue and iterations < max_iterations:
        iterations += 1
        dist, row, col = queue.popleft()

        if row == end_row and col == end_col:
            found = True
            print(f"FOUND target at iteration {iterations}, distance {dist}")
            break

        if dist > dist_grid[row][col] and dist_grid[row][col] != -1:
            continue

        # Try to expand
        for dr, dc in directions:
            nr, nc = row + dr, col + dc

            if not rp._in_bounds(nr, nc):
                continue
            if dist_grid[nr][nc] != -1:
                continue

            # Check accessibility
            if (nr, nc) in target_cells:
                accessible = rp._is_cell_accessible_for_net(rp.fcu_grid, nr, nc, net_name)
            else:
                accessible = rp._is_cell_clear_for_net(rp.fcu_grid, nr, nc, net_name)

            if accessible:
                dist_grid[nr][nc] = dist + 1
                queue.append((dist + 1, nr, nc))

    if not found:
        print(f"NOT FOUND after {iterations} iterations")
        print(f"Queue exhausted: {len(queue) == 0}")

        # How many cells were explored?
        explored = sum(1 for r in range(rp.grid_rows) for c in range(rp.grid_cols) if dist_grid[r][c] >= 0)
        print(f"Cells explored: {explored}")

        # What's blocking at the frontier?
        print("\nFrontier analysis (cells with neighbors that couldn't expand):")
        frontiers = []
        for r in range(rp.grid_rows):
            for c in range(rp.grid_cols):
                if dist_grid[r][c] >= 0:  # This cell was reached
                    # Check if any neighbor is unreachable
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if rp._in_bounds(nr, nc) and dist_grid[nr][nc] == -1:
                            # This neighbor wasn't reached
                            cell = rp.fcu_grid[nr][nc]
                            if cell is not None:
                                frontiers.append((nr, nc, cell, dist_grid[r][c]))

        # Show unique blockers
        blockers = {}
        for r, c, cell, dist in frontiers:
            if cell not in blockers:
                blockers[cell] = []
            blockers[cell].append((r, c, dist))

        for blocker, positions in blockers.items():
            print(f"  {blocker!r}: {len(positions)} cells blocked")
            for r, c, d in positions[:3]:
                print(f"    example: [{r}][{c}] at frontier dist {d}")

print("=" * 70)
