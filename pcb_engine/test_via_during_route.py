#!/usr/bin/env python3
"""Debug via placement DURING Lee routing."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque
from pcb_engine.routing_piston import RoutingPiston
from pcb_engine.routing_types import RoutingConfig
from pcb_engine.test_complex_esp32 import create_esp32_sensor_parts_db

def test_via_during_route():
    """Simulate the Lee expansion and check via placement."""

    parts_db = create_esp32_sensor_parts_db()

    placement = {
        'U1': (15.0, 20.0),
        'U2': (35.0, 25.0),
        'U3': (35.0, 15.0),
        'C1': (42.0, 30.0),
        'C2': (42.0, 20.0),
        'C3': (8.0, 20.0),
        'R1': (25.0, 12.0),
        'R2': (25.0, 8.0),
        'R3': (8.0, 28.0),
        'D1': (5.0, 28.0),
        'R4': (8.0, 32.0),
        'D2': (5.0, 32.0),
    }

    config = RoutingConfig(
        board_width=50.0,
        board_height=40.0,
        origin_x=0.0,
        origin_y=0.0,
        grid_size=0.25,
        trace_width=0.25,
        clearance=0.15,
        via_diameter=0.6,
        via_drill=0.3,
        algorithm='lee'
    )

    piston = RoutingPiston(config)
    piston._initialize_grids()
    piston._register_components(placement, parts_db)

    net_name = 'GND'

    # Simulate Lee from U3.5 (35.95, 14.45) to U3.2 (34.05, 15.0)
    start = (35.95, 14.45)
    end = (34.05, 15.0)

    start_col = piston._real_to_grid_col(start[0])
    start_row = piston._real_to_grid_row(start[1])
    end_col = piston._real_to_grid_col(end[0])
    end_row = piston._real_to_grid_row(end[1])

    print(f"Start: ({start_row}, {start_col}) = {piston.fcu_grid[start_row][start_col]}")
    print(f"End: ({end_row}, {end_col}) = {piston.fcu_grid[end_row][end_col]}")

    # 3D distance grids
    dist_grid = [
        [[-1] * piston.grid_cols for _ in range(piston.grid_rows)],
        [[-1] * piston.grid_cols for _ in range(piston.grid_rows)]
    ]
    parent = {}

    start_layer = 0  # F.Cu
    dist_grid[start_layer][start_row][start_col] = 0
    queue = deque([(0, start_layer, start_row, start_col)])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    via_attempts = []
    layer_changes = 0
    max_iterations = 5000

    for iteration in range(max_iterations):
        if not queue:
            break

        dist, layer, row, col = queue.popleft()
        grid = piston.fcu_grid if layer == 0 else piston.bcu_grid

        # Check if reached target
        if row == end_row and col == end_col:
            print(f"\n*** FOUND TARGET at iteration {iteration} ***")
            print(f"Layer: {'F.Cu' if layer == 0 else 'B.Cu'}")
            break

        # Check if reached any cell of target net near target
        cell_value = grid[row][col]
        if cell_value == net_name:
            manhattan = abs(row - end_row) + abs(col - end_col)
            if manhattan <= 10:
                print(f"\n*** FOUND NET CELL NEAR TARGET at iteration {iteration} ***")
                print(f"Position: ({row}, {col}), distance to target: {manhattan}")
                print(f"Layer: {'F.Cu' if layer == 0 else 'B.Cu'}")
                break

        # Try 4 neighbors
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if not piston._in_bounds(nr, nc):
                continue
            if dist_grid[layer][nr][nc] != -1:
                continue

            neighbor_cell = grid[nr][nc]
            is_own_net_cell = neighbor_cell == net_name
            is_near_target = abs(nr - end_row) <= 10 and abs(nc - end_col) <= 10

            is_approaching = False
            if not is_own_net_cell and is_near_target:
                cc = piston.clearance_cells * 2
                for dr2 in range(-cc, cc + 1):
                    for dc2 in range(-cc, cc + 1):
                        check_r, check_c = nr + dr2, nc + dc2
                        if piston._in_bounds(check_r, check_c):
                            if grid[check_r][check_c] == net_name:
                                is_approaching = True
                                break
                    if is_approaching:
                        break

            use_relaxed = is_own_net_cell or is_approaching

            if use_relaxed:
                ok = piston._is_cell_accessible_for_net(grid, nr, nc, net_name)
            else:
                ok = piston._is_cell_clear_for_net(grid, nr, nc, net_name)

            if ok:
                dist_grid[layer][nr][nc] = dist + 1
                parent[(layer, nr, nc)] = (layer, row, col)
                queue.append((dist + 1, layer, nr, nc))

        # Try layer change (via)
        if config.allow_layer_change:
            other_layer = 1 - layer
            if dist_grid[other_layer][row][col] == -1:
                current_grid = piston.fcu_grid if layer == 0 else piston.bcu_grid
                other_grid = piston.bcu_grid if layer == 0 else piston.fcu_grid

                current_clear = piston._is_cell_clear_for_via(current_grid, row, col, net_name)
                other_clear = piston._is_cell_clear_for_via(other_grid, row, col, net_name)

                via_attempts.append({
                    'pos': (row, col),
                    'layer': 'F.Cu' if layer == 0 else 'B.Cu',
                    'current_clear': current_clear,
                    'other_clear': other_clear,
                    'cell': current_grid[row][col]
                })

                if current_clear and other_clear:
                    via_cost = int(config.via_cost)
                    dist_grid[other_layer][row][col] = dist + via_cost
                    parent[(other_layer, row, col)] = (layer, row, col)
                    queue.append((dist + via_cost, other_layer, row, col))
                    layer_changes += 1

    else:
        print(f"\n*** MAX ITERATIONS REACHED ({max_iterations}) ***")

    print(f"\nLayer changes (vias): {layer_changes}")
    print(f"Total via attempts: {len(via_attempts)}")

    # Show successful via positions
    successful_vias = [v for v in via_attempts if v['current_clear'] and v['other_clear']]
    print(f"Successful via placements: {len(successful_vias)}")

    if successful_vias:
        print("\nFirst 5 successful vias:")
        for v in successful_vias[:5]:
            print(f"  ({v['pos'][0]}, {v['pos'][1]}): {v['layer']} -> other, cell={v['cell']}")

    # Show failed via attempts on GND cells
    failed_on_gnd = [v for v in via_attempts if v['cell'] == 'GND' and not (v['current_clear'] and v['other_clear'])]
    if failed_on_gnd:
        print(f"\nFailed via attempts on GND cells: {len(failed_on_gnd)}")
        for v in failed_on_gnd[:5]:
            print(f"  ({v['pos'][0]}, {v['pos'][1]}): current={v['current_clear']}, other={v['other_clear']}")

if __name__ == '__main__':
    test_via_during_route()
