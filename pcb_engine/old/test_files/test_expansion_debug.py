#!/usr/bin/env python3
"""Debug Lee algorithm expansion from U3.5 GND pad."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.routing_piston import RoutingPiston
from pcb_engine.routing_types import RoutingConfig
from pcb_engine.test_complex_esp32 import create_esp32_sensor_parts_db

def debug_expansion():
    """Debug expansion from U3.5 GND pad."""

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

    # Start: U3.5 GND (row=57, col=143)
    # Target: U3.2 GND (row=60, col=136)
    start_row, start_col = 57, 143
    end_row, end_col = 60, 136
    net_name = 'GND'

    print(f"Start: ({start_row}, {start_col}) - cell value: {piston.fcu_grid[start_row][start_col]}")
    print(f"End: ({end_row}, {end_col}) - cell value: {piston.fcu_grid[end_row][end_col]}")
    print()

    # Check expansion options from start
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    dir_names = ['Right', 'Left', 'Down', 'Up']

    print("Expansion from start (F.Cu):")
    for (dr, dc), name in zip(directions, dir_names):
        nr, nc = start_row + dr, start_col + dc
        cell = piston.fcu_grid[nr][nc] if piston._in_bounds(nr, nc) else "OUT"

        # Check if accessible for net (relaxed)
        accessible = piston._is_cell_accessible_for_net(piston.fcu_grid, nr, nc, net_name) if piston._in_bounds(nr, nc) else False

        # Check if clear for net (normal)
        clear = piston._is_cell_clear_for_net(piston.fcu_grid, nr, nc, net_name) if piston._in_bounds(nr, nc) else False

        # Check if we're near target
        near_target = abs(nr - end_row) <= 10 and abs(nc - end_col) <= 10

        print(f"  {name}: ({nr}, {nc}) = {cell}")
        print(f"    accessible={accessible}, clear={clear}, near_target={near_target}")

    # Check via options from start
    print("\nVia option from start:")
    via_f = piston._is_cell_clear_for_via(piston.fcu_grid, start_row, start_col, net_name)
    via_b = piston._is_cell_clear_for_via(piston.bcu_grid, start_row, start_col, net_name)
    print(f"  F.Cu clear for via: {via_f}")
    print(f"  B.Cu clear for via: {via_b}")

    # If via works, check B.Cu expansion
    if via_f and via_b:
        print("\nExpansion from start on B.Cu:")
        for (dr, dc), name in zip(directions, dir_names):
            nr, nc = start_row + dr, start_col + dc
            cell = piston.bcu_grid[nr][nc] if piston._in_bounds(nr, nc) else "OUT"
            clear = piston._is_cell_clear_for_net(piston.bcu_grid, nr, nc, net_name) if piston._in_bounds(nr, nc) else False
            print(f"  {name}: ({nr}, {nc}) = {cell}, clear={clear}")

def test_lee_route():
    """Actually run Lee algorithm and see what happens."""

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

    # Route just the GND net to see what happens
    print("\n\nRouting GND net...")
    result = piston.route(parts_db, {}, placement, ['GND'])

    print(f"Success: {result.success}")
    print(f"Routed: {result.routed_count}/{result.total_count}")

    gnd_route = result.routes.get('GND')
    if gnd_route:
        print(f"GND route success: {gnd_route.success}")
        print(f"GND route error: {gnd_route.error}")
        print(f"GND segments: {len(gnd_route.segments)}")
        print(f"GND vias: {len(gnd_route.vias)}")

        if gnd_route.segments:
            print("First 10 segments:")
            for seg in gnd_route.segments[:10]:
                print(f"  {seg.start} -> {seg.end} ({seg.layer})")

if __name__ == '__main__':
    debug_expansion()
    test_lee_route()
