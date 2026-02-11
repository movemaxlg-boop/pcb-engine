"""Debug GND path options"""
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
print("DEBUG: GND Path Analysis")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

# Routing setup (route VDIV first like the test does)
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

routeable_nets = ['VDIV', 'GND']
result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=placement,
    net_order=routeable_nets
)

# GND endpoints
gnd_endpoints = [(20.95, 14.5), (17.95, 14.5)]
start = gnd_endpoints[0]
end = gnd_endpoints[1]

start_col = int(start[0] / 0.1)
start_row = int(start[1] / 0.1)
end_col = int(end[0] / 0.1)
end_row = int(end[1] / 0.1)

print(f"GND start: grid[{start_row}][{start_col}]")
print(f"GND end: grid[{end_row}][{end_col}]")

# Check paths ABOVE and BELOW
print("\n--- Checking path options ---")

# Check row 130 (above components)
print(f"\nRow 130 (above):")
for c in range(end_col - 5, start_col + 5):
    if rp._in_bounds(130, c):
        clear = rp._is_cell_clear_for_net(rp.fcu_grid, 130, c, 'GND')
        cell = rp.fcu_grid[130][c]
        if not clear:
            print(f"  col {c}: BLOCKED by {cell!r}")

# Check row 160 (below components)
print(f"\nRow 160 (below):")
for c in range(end_col - 5, start_col + 5):
    if rp._in_bounds(160, c):
        clear = rp._is_cell_clear_for_net(rp.fcu_grid, 160, c, 'GND')
        cell = rp.fcu_grid[160][c]
        if not clear:
            print(f"  col {c}: BLOCKED by {cell!r}")

# Manual wavefront to find ANY path
print("\n--- Manual BFS from GND start to end ---")
target_cells = {(start_row, start_col), (end_row, end_col)}
dist_grid = [[-1] * rp.grid_cols for _ in range(rp.grid_rows)]
dist_grid[start_row][start_col] = 0

queue = deque([(0, start_row, start_col)])
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

max_iterations = 500000
iterations = 0
found = False

while queue and iterations < max_iterations:
    iterations += 1
    dist, row, col = queue.popleft()

    if row == end_row and col == end_col:
        found = True
        print(f"FOUND! Distance: {dist}, iterations: {iterations}")
        break

    if dist > dist_grid[row][col] and dist_grid[row][col] != -1:
        continue

    for dr, dc in directions:
        nr, nc = row + dr, col + dc

        if not rp._in_bounds(nr, nc):
            continue
        if dist_grid[nr][nc] != -1:
            continue

        # Use target-aware check
        if (nr, nc) in target_cells:
            accessible = rp._is_cell_accessible_for_net(rp.fcu_grid, nr, nc, 'GND')
        else:
            accessible = rp._is_cell_clear_for_net(rp.fcu_grid, nr, nc, 'GND')

        if accessible:
            dist_grid[nr][nc] = dist + 1
            queue.append((dist + 1, nr, nc))

if not found:
    print(f"NOT FOUND after {iterations} iterations")
    explored = sum(1 for r in range(rp.grid_rows) for c in range(rp.grid_cols) if dist_grid[r][c] >= 0)
    print(f"Cells explored: {explored}")

    # Find the closest cell we reached to the end
    closest = None
    closest_dist = float('inf')
    for r in range(rp.grid_rows):
        for c in range(rp.grid_cols):
            if dist_grid[r][c] >= 0:
                manhattan = abs(r - end_row) + abs(c - end_col)
                if manhattan < closest_dist:
                    closest_dist = manhattan
                    closest = (r, c, dist_grid[r][c])

    if closest:
        print(f"Closest to end: grid[{closest[0]}][{closest[1]}], manhattan dist to end: {closest_dist}")

print("=" * 70)
