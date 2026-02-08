#!/usr/bin/env python3
"""Debug GND via placement options."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.routing_piston import RoutingPiston
from pcb_engine.routing_types import RoutingConfig
from pcb_engine.test_complex_esp32 import create_esp32_sensor_parts_db

def debug_via():
    """Find where vias can be placed."""

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

    # Check cells around the GND pads for via placement
    p2_x, p2_y = 34.05, 15.0  # U3.2 (GND)
    p5_x, p5_y = 35.95, 14.45  # U3.5 (GND)

    p2_col = int((p2_x - config.origin_x) / config.grid_size)
    p2_row = int((p2_y - config.origin_y) / config.grid_size)
    p5_col = int((p5_x - config.origin_x) / config.grid_size)
    p5_row = int((p5_y - config.origin_y) / config.grid_size)

    print(f"Clearance cells: {piston.clearance_cells}")
    print(f"Via clearance cells: {piston.via_clearance_cells}")
    print()

    # Check if via can be placed at different cells
    print("Via placement options around U3.5 (col=143, row=57):")
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            r, c = p5_row + dr, p5_col + dc
            if not piston._in_bounds(r, c):
                continue

            fcu_clear = piston._is_cell_clear_for_via(piston.fcu_grid, r, c, 'GND')
            bcu_clear = piston._is_cell_clear_for_via(piston.bcu_grid, r, c, 'GND')

            if fcu_clear and bcu_clear:
                cell = piston.fcu_grid[r][c]
                cell_str = '.' if cell is None else ('G' if cell == 'GND' else '?')
                print(f"  ({r}, {c}): OK (F.Cu cell={cell_str})")

    print("\nVia placement options around U3.2 (col=136, row=60):")
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            r, c = p2_row + dr, p2_col + dc
            if not piston._in_bounds(r, c):
                continue

            fcu_clear = piston._is_cell_clear_for_via(piston.fcu_grid, r, c, 'GND')
            bcu_clear = piston._is_cell_clear_for_via(piston.bcu_grid, r, c, 'GND')

            if fcu_clear and bcu_clear:
                cell = piston.fcu_grid[r][c]
                cell_str = '.' if cell is None else ('G' if cell == 'GND' else '?')
                print(f"  ({r}, {c}): OK (F.Cu cell={cell_str})")

def debug_why_blocked():
    """Debug why via can't be placed at specific cells."""

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

    # Check U3.5 GND pad (row=57, col=143)
    for test_name, test_row, test_col in [("U3.5", 57, 143), ("U3.2", 60, 136)]:
        net_name = 'GND'
        cc = piston.via_clearance_cells

        print(f"\nChecking via clearance at ({test_row}, {test_col}) for {test_name} ({net_name}):")

        center = piston.fcu_grid[test_row][test_col]
        print(f"Center cell value: {center}")

        can_place = piston._is_cell_clear_for_via(piston.fcu_grid, test_row, test_col, net_name)
        print(f"Via allowed: {can_place}")

        if not can_place:
            # Check each cell in the clearance zone
            blockers = []
            for dr in range(-cc, cc + 1):
                for dc in range(-cc, cc + 1):
                    if dr == 0 and dc == 0:
                        continue
                    dist_sq = dr * dr + dc * dc
                    if dist_sq > cc * cc:
                        continue

                    r, c = test_row + dr, test_col + dc
                    if piston._in_bounds(r, c):
                        occ = piston.fcu_grid[r][c]
                        if occ is not None and occ != net_name:
                            blockers.append((r, c, occ, dr, dc))

            if blockers:
                print("Blockers in via clearance zone:")
                for r, c, occ, dr, dc in blockers:
                    print(f"  ({r}, {c}) [dr={dr}, dc={dc}]: {occ}")

if __name__ == '__main__':
    debug_via()
    print("\n" + "="*60 + "\n")
    debug_why_blocked()
