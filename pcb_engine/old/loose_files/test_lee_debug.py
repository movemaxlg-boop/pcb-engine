"""Debug Lee wavefront for GND routing"""
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
print("DEBUG: Lee Wavefront for GND")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

# Setup routing (route VDIV first)
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

# Route both nets
result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=placement,
    net_order=['VDIV', 'GND']
)

print(f"Result: {result.routed_count}/{result.total_count}")

# Now manually trace what the Lee algorithm is doing for GND
print("\n--- Manual trace of Lee algorithm for GND ---")

# After routing, get the grid state
net_name = 'GND'
grid = rp.fcu_grid

# GND endpoints
start = (20.95, 14.5)  # R2.2
end = (17.95, 14.5)    # C1.2

start_col = int(start[0] / 0.1)
start_row = int(start[1] / 0.1)
end_col = int(end[0] / 0.1)
end_row = int(end[1] / 0.1)

print(f"Start: [{start_row}][{start_col}]")
print(f"End: [{end_row}][{end_col}]")

# Check if end is accessible
print(f"\nEnd cell value: {grid[end_row][end_col]!r}")
print(f"End accessible: {rp._is_cell_accessible_for_net(grid, end_row, end_col, 'GND')}")

# Run BFS with the approach zone logic
dist_grid = [[-1] * rp.grid_cols for _ in range(rp.grid_rows)]
dist_grid[start_row][start_col] = 0

queue = deque([(0, start_row, start_col)])
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
cc = rp.clearance_cells

max_iterations = 100000
iterations = 0
found = False
found_row, found_col = 0, 0

# Count how many cells use relaxed check
relaxed_count = 0
normal_count = 0

while queue and iterations < max_iterations:
    iterations += 1
    dist, row, col = queue.popleft()

    # Check if reached target
    cell_value = grid[row][col] if rp._in_bounds(row, col) else None
    if row == end_row and col == end_col:
        found = True
        found_row, found_col = row, col
        break

    if cell_value == net_name:
        manhattan = abs(row - end_row) + abs(col - end_col)
        if manhattan <= 10:
            found = True
            found_row, found_col = row, col
            break

    if dist > dist_grid[row][col] and dist_grid[row][col] != -1:
        continue

    for dr, dc in directions:
        nr, nc = row + dr, col + dc

        if not rp._in_bounds(nr, nc):
            continue
        if dist_grid[nr][nc] != -1:
            continue

        neighbor_cell = grid[nr][nc] if rp._in_bounds(nr, nc) else None

        is_own_net_cell = neighbor_cell == net_name
        is_near_target = abs(nr - end_row) <= 10 and abs(nc - end_col) <= 10

        is_approaching_own_pad = False
        if not is_own_net_cell and is_near_target:
            for dr2 in range(-cc, cc + 1):
                for dc2 in range(-cc, cc + 1):
                    check_r, check_c = nr + dr2, nc + dc2
                    if rp._in_bounds(check_r, check_c):
                        if grid[check_r][check_c] == net_name:
                            is_approaching_own_pad = True
                            break
                if is_approaching_own_pad:
                    break

        use_relaxed = is_own_net_cell or is_approaching_own_pad

        if use_relaxed:
            accessible = rp._is_cell_accessible_for_net(grid, nr, nc, net_name)
            if accessible:
                relaxed_count += 1
                dist_grid[nr][nc] = dist + 1
                queue.append((dist + 1, nr, nc))
        else:
            clear = rp._is_cell_clear_for_net(grid, nr, nc, net_name)
            if clear:
                normal_count += 1
                dist_grid[nr][nc] = dist + 1
                queue.append((dist + 1, nr, nc))

print(f"\nAfter {iterations} iterations:")
print(f"  Found: {found}")
if found:
    print(f"  Found at: [{found_row}][{found_col}]")
print(f"  Relaxed checks passed: {relaxed_count}")
print(f"  Normal checks passed: {normal_count}")

explored = sum(1 for r in range(rp.grid_rows) for c in range(rp.grid_cols) if dist_grid[r][c] >= 0)
print(f"  Total cells explored: {explored}")

# Find closest to target
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
    print(f"  Closest to end: [{closest[0]}][{closest[1]}], manhattan={closest_dist}")

print("=" * 70)
