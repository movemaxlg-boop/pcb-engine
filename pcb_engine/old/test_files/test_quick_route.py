#!/usr/bin/env python3
"""Quick routing diagnostic test - identify what's blocking paths."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.routing_piston import RoutingPiston
from pcb_engine.routing_types import RoutingConfig

def test_simple_2pin():
    """Test routing between two simple components."""

    # Very simple test: two 0805 resistors, 10mm apart, 1 net
    parts_db = {
        'parts': {
            'R1': {
                'value': '100R',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'NET1', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': '', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            },
            'R2': {
                'value': '100R',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'NET1', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': '', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            }
        },
        'nets': {
            'NET1': {'pins': [('R1', '1'), ('R2', '1')]}
        }
    }

    # Place R1 at (10, 15), R2 at (25, 15) - same Y, 15mm apart
    placement = {
        'R1': (10.0, 15.0),
        'R2': (25.0, 15.0)
    }

    config = RoutingConfig(
        board_width=35.0,
        board_height=30.0,
        origin_x=0.0,
        origin_y=0.0,
        grid_size=0.25,  # Use larger grid for faster test
        trace_width=0.25,
        clearance=0.15,
        via_diameter=0.6,
        via_drill=0.3,
        algorithm='lee'
    )

    print(f"Board: {config.board_width}x{config.board_height}mm")
    print(f"Grid size: {config.grid_size}mm")
    print(f"Grid dimensions: {int(config.board_width/config.grid_size)}x{int(config.board_height/config.grid_size)}")
    print(f"R1 at (10, 15), R2 at (25, 15)")
    print(f"R1.1 at ({10 - 0.95}, 15) = (9.05, 15)")
    print(f"R2.1 at ({25 - 0.95}, 15) = (24.05, 15)")
    print(f"Distance: 15mm (150 cells at 0.1mm, 60 cells at 0.25mm)")
    print()

    piston = RoutingPiston(config)

    # Route
    result = piston.route(parts_db, {}, placement, ['NET1'])

    print(f"Success: {result.success}")
    print(f"Routed: {result.routed_count}/{result.total_count}")

    if result.success:
        route = result.routes.get('NET1')
        if route:
            print(f"Segments: {len(route.segments)}")
            for seg in route.segments:
                print(f"  {seg.start} -> {seg.end} (layer={seg.layer})")
    else:
        route = result.routes.get('NET1')
        if route:
            print(f"Error: {route.error}")

        # Debug: show grid around the path
        print("\nGrid debug (F.Cu around R1.1 and R2.1):")
        r1_col = int((9.05 - config.origin_x) / config.grid_size)
        r2_col = int((24.05 - config.origin_x) / config.grid_size)
        center_row = int((15.0 - config.origin_y) / config.grid_size)

        print(f"R1.1 grid cell: row={center_row}, col={r1_col}")
        print(f"R2.1 grid cell: row={center_row}, col={r2_col}")

        # Show a row of the grid
        print(f"\nF.Cu row {center_row} from col {max(0, r1_col-2)} to {min(piston.grid_cols-1, r2_col+2)}:")
        row_cells = []
        for c in range(max(0, r1_col-2), min(piston.grid_cols-1, r2_col+2)+1):
            cell = piston.fcu_grid[center_row][c]
            if cell is None:
                row_cells.append('.')
            elif cell == '__COMPONENT__':
                row_cells.append('X')
            elif cell == '__PAD_NC__':
                row_cells.append('#')
            elif cell == 'NET1':
                row_cells.append('1')
            else:
                row_cells.append('?')
        print(' '.join(row_cells))

    return result.success

if __name__ == '__main__':
    success = test_simple_2pin()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
