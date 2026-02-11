#!/usr/bin/env python3
"""Debug GND routing between U3 pins."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.routing_piston import RoutingPiston
from pcb_engine.routing_types import RoutingConfig
from pcb_engine.test_complex_esp32 import create_esp32_sensor_parts_db

def debug_gnd():
    """Debug GND net routing failure."""

    parts_db = create_esp32_sensor_parts_db()

    placement = {
        'U1': (15.0, 20.0),
        'U2': (35.0, 25.0),
        'U3': (35.0, 15.0),  # BME280 - problem component
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
        algorithm='lee'
    )

    piston = RoutingPiston(config)

    # Initialize grids and register components
    piston._initialize_grids()
    piston._register_components(placement, parts_db)

    # The failing route: from (35.95, 14.45) to (34.05, 15.0)
    # These are U3.5 (GND) and U3.2 (GND)
    u3_x, u3_y = 35.0, 15.0

    # U3 is SOT-23-5 with pins:
    # Pin 2 (GND): offset (-0.95, 0.0)   -> (34.05, 15.0)
    # Pin 5 (GND): offset (0.95, -0.55)  -> (35.95, 14.45)

    p2_x, p2_y = 34.05, 15.0
    p5_x, p5_y = 35.95, 14.45

    print(f"U3 (BME280) at ({u3_x}, {u3_y})")
    print(f"U3.2 (GND) at ({p2_x}, {p2_y})")
    print(f"U3.5 (GND) at ({p5_x}, {p5_y})")
    print(f"Distance: {((p2_x-p5_x)**2 + (p2_y-p5_y)**2)**0.5:.2f}mm")
    print()

    # Get grid cells
    p2_col = int((p2_x - config.origin_x) / config.grid_size)
    p2_row = int((p2_y - config.origin_y) / config.grid_size)
    p5_col = int((p5_x - config.origin_x) / config.grid_size)
    p5_row = int((p5_y - config.origin_y) / config.grid_size)

    print(f"U3.2 grid cell: row={p2_row}, col={p2_col}")
    print(f"U3.5 grid cell: row={p5_row}, col={p5_col}")
    print()

    # Check component body size for SOT-23-5
    body_w, body_h = piston._get_component_body_size('SOT-23-5')
    print(f"SOT-23-5 body size: {body_w}x{body_h}mm")

    # Show grid around U3
    print("\nF.Cu grid around U3:")
    u3_col = int((u3_x - config.origin_x) / config.grid_size)
    u3_row = int((u3_y - config.origin_y) / config.grid_size)

    min_row = max(0, min(p2_row, p5_row) - 10)
    max_row = min(piston.grid_rows - 1, max(p2_row, p5_row) + 10)
    min_col = max(0, min(p2_col, p5_col) - 10)
    max_col = min(piston.grid_cols - 1, max(p2_col, p5_col) + 10)

    print(f"Showing rows {min_row}-{max_row}, cols {min_col}-{max_col}")
    print(f"Legend: . = empty, X = COMPONENT, # = PAD_NC, G = GND net, ? = other")
    print()

    for r in range(min_row, max_row + 1):
        row_str = f"{r:3d}: "
        for c in range(min_col, max_col + 1):
            cell = piston.fcu_grid[r][c]
            if cell is None:
                row_str += '.'
            elif cell == '__COMPONENT__':
                row_str += 'X'
            elif cell == '__PAD_NC__':
                row_str += '#'
            elif cell == 'GND':
                row_str += 'G'
            else:
                row_str += '?'
        # Mark special cells
        if r == p2_row:
            row_str += f" <- U3.2 (col {p2_col})"
        if r == p5_row:
            row_str += f" <- U3.5 (col {p5_col})"
        if r == u3_row:
            row_str += f" <- U3 center (col {u3_col})"
        print(row_str)

    print("\n\nB.Cu grid around U3 (should be clear for SMD routing under):")
    for r in range(min_row, max_row + 1):
        row_str = f"{r:3d}: "
        for c in range(min_col, max_col + 1):
            cell = piston.bcu_grid[r][c]
            if cell is None:
                row_str += '.'
            elif cell == '__COMPONENT__':
                row_str += 'X'
            elif cell == '__PAD_NC__':
                row_str += '#'
            elif cell == 'GND':
                row_str += 'G'
            else:
                row_str += '?'
        print(row_str)

if __name__ == '__main__':
    debug_gnd()
