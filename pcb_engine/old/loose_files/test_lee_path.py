"""Trace the actual Lee path for GND"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque

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
print("Lee Path Debug for GND")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
placement = pe.place(parts_db, {})

# Routing setup
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.15,
    via_diameter=0.6,
    via_drill=0.3
))

rp._placement = placement.positions
rp._parts_db = parts_db
rp._register_components(placement.positions, parts_db)

# GND endpoints
c1 = placement.positions['C1']
r2 = placement.positions['R2']
c1_x = c1[0] if isinstance(c1, (list, tuple)) else c1.x
r2_x = r2[0] if isinstance(r2, (list, tuple)) else r2.x

start = (r2_x + 0.95, 14.5)  # R2.GND
end = (c1_x + 0.95, 14.5)    # C1.GND

print(f"\nStart: {start}")
print(f"End: {end}")

start_col = int(start[0] / 0.1)
start_row = int(start[1] / 0.1)
end_col = int(end[0] / 0.1)
end_row = int(end[1] / 0.1)

print(f"Start grid: row={start_row}, col={start_col}")
print(f"End grid: row={end_row}, col={end_col}")

# Manual Lee BFS for GND (simplified, only F.Cu)
print("\n" + "=" * 40)
print("Manual Lee BFS on F.Cu only:")
print("=" * 40)

grid = rp.fcu_grid
net_name = 'GND'

# 2D distance grid
dist_grid = [[-1] * rp.grid_cols for _ in range(rp.grid_rows)]
parent = {}

dist_grid[start_row][start_col] = 0
queue = deque([(0, start_row, start_col)])
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

found = False
iterations = 0
max_iterations = 50000

while queue and iterations < max_iterations:
    iterations += 1
    dist, row, col = queue.popleft()

    if row == end_row and col == end_col:
        found = True
        print(f"Found target at [{row}][{col}] after {iterations} iterations")
        break

    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if not rp._in_bounds(nr, nc):
            continue
        if dist_grid[nr][nc] != -1:
            continue

        # Check if clear for GND
        clear = rp._is_cell_clear_for_net(grid, nr, nc, net_name)
        if clear:
            dist_grid[nr][nc] = dist + 1
            parent[(nr, nc)] = (row, col)
            queue.append((dist + 1, nr, nc))

if not found:
    print(f"NOT FOUND after {iterations} iterations")

    # Find closest point reached
    closest = None
    closest_dist = float('inf')
    for r in range(rp.grid_rows):
        for c in range(rp.grid_cols):
            if dist_grid[r][c] >= 0:
                manhattan = abs(r - end_row) + abs(c - end_col)
                if manhattan < closest_dist:
                    closest_dist = manhattan
                    closest = (r, c)

    if closest:
        print(f"Closest reached: [{closest[0]}][{closest[1]}], manhattan={closest_dist}")
else:
    # Backtrace
    path = [(end_row, end_col)]
    current = (end_row, end_col)
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()

    print(f"Path length: {len(path)} cells")
    print(f"First 10 cells: {path[:10]}")
    print(f"Last 10 cells: {path[-10:]}")

    # Check if path goes through VDIV
    vdiv_cells = []
    for r, c in path:
        cell = grid[r][c]
        if cell == 'VDIV':
            vdiv_cells.append((r, c))

    if vdiv_cells:
        print(f"\n*** PATH GOES THROUGH VDIV! ***")
        print(f"VDIV cells in path: {vdiv_cells[:5]}...")
    else:
        print("\nPath does NOT go through VDIV cells - must go around")

print("=" * 70)
