#!/usr/bin/env python3
"""
Debug via placement - why can't LED_R use B.Cu to avoid LED_CTRL?
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import PCBEngine, BoardConfig, DesignRules
from pcb_engine.engine import EnginePhase
from pcb_engine.routing import Layer, GridCell


def create_design():
    return {
        'U1': {
            'name': 'MCU', 'footprint': 'QFN-16', 'value': 'MCU',
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -1.5, 'offset_y': -2.0}},
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 1.5, 'offset_y': -2.0}},
                {'number': '3', 'name': 'GPIO1', 'type': 'output', 'net': 'LED_CTRL',
                 'physical': {'offset_x': -1.5, 'offset_y': 2.0}},
                {'number': '4', 'name': 'GPIO2', 'type': 'input', 'net': 'NC',
                 'physical': {'offset_x': 1.5, 'offset_y': 2.0}},
            ],
            'size': (4.0, 4.0),
        },
        'D1': {
            'name': 'LED', 'footprint': '0603', 'value': 'Green',
            'pins': [
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': -0.7, 'offset_y': 0}},
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.7, 'offset_y': 0}},
            ],
            'size': (1.6, 0.8),
        },
        'R1': {
            'name': 'Resistor', 'footprint': '0402', 'value': '1K',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED_CTRL',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },
    }


def run():
    board = BoardConfig(
        origin_x=100.0, origin_y=100.0,
        width=30.0, height=25.0,
        layers=2, grid_size=0.5
    )

    rules = DesignRules(
        min_trace_width=0.25,
        min_clearance=0.2,
        min_via_diameter=0.8,
        min_via_drill=0.4,
    )

    engine = PCBEngine(board, rules)
    engine.load_parts_from_dict(create_design())

    # Run phases 0-6
    for i in range(7):
        engine.run_phase(EnginePhase(i))

    # Route LED_CTRL first
    engine.state.route_order = ['LED_CTRL']
    engine.run_phase(EnginePhase.PHASE_7_ROUTE)

    router = engine._router
    grid = router.grid

    print("After routing LED_CTRL:")
    print(f"  Routed: {list(router.routes.keys())}")

    # Now check why LED_R can't route
    # LED_R: D1.1 -> R1.2

    # Get pin positions
    d1_pos = engine.state.placement['D1']
    r1_pos = engine.state.placement['R1']

    d1_1_x = d1_pos.x - 0.7  # D1.1 offset
    d1_1_y = d1_pos.y
    r1_2_x = r1_pos.x + 0.45  # R1.2 offset
    r1_2_y = r1_pos.y

    print(f"\nLED_R route:")
    print(f"  D1.1: ({d1_1_x:.1f}, {d1_1_y:.1f})")
    print(f"  R1.2: ({r1_2_x:.1f}, {r1_2_y:.1f})")

    start_row, start_col = grid.world_to_grid(d1_1_x, d1_1_y)
    end_row, end_col = grid.world_to_grid(r1_2_x, r1_2_y)

    print(f"  Start grid: ({start_row}, {start_col})")
    print(f"  End grid: ({end_row}, {end_col})")

    # Check if F.Cu path is blocked
    print(f"\nF.Cu availability check along path:")
    blocked_count = 0
    for c in range(min(start_col, end_col), max(start_col, end_col) + 1):
        cell = GridCell(start_row, c, Layer.F_CU)
        avail = grid.is_available(cell, 'LED_R')
        owner = grid.owners[Layer.F_CU][start_row][c]
        if not avail:
            blocked_count += 1
            wx, wy = grid.grid_to_world(start_row, c)
            print(f"  BLOCKED at ({start_row}, {c}) = ({wx:.1f}, {wy:.1f}), owner={owner}")

    print(f"\n  Total blocked on F.Cu: {blocked_count}")

    # Check via placement options
    print(f"\nVia placement check (for LED_R to escape to B.Cu):")
    via_options = 0
    for r in range(max(0, start_row - 5), min(grid.rows, start_row + 6)):
        for c in range(max(0, start_col - 5), min(grid.cols, start_col + 6)):
            if grid.can_place_via(r, c, 'LED_R'):
                via_options += 1

    print(f"  Via placement options within 5 cells: {via_options}")

    if via_options == 0:
        print("\n  Checking WHY via placement fails...")
        # Check one specific location
        r, c = start_row, start_col + 2
        wx, wy = grid.grid_to_world(r, c)
        print(f"  Testing via at ({r}, {c}) = ({wx:.1f}, {wy:.1f}):")

        via_clearance = int(
            (rules.min_via_diameter / 2 + rules.min_clearance) / board.grid_size
        ) + 1

        for layer in [Layer.F_CU, Layer.B_CU]:
            for dr in range(-via_clearance, via_clearance + 1):
                for dc in range(-via_clearance, via_clearance + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid.rows and 0 <= cc < grid.cols:
                        owner = grid.owners[layer][rr][cc]
                        if owner and owner != 'LED_R' and owner != 'GND':
                            print(f"    Blocked by {owner} at ({rr},{cc}) on {layer.value}")


if __name__ == '__main__':
    run()
