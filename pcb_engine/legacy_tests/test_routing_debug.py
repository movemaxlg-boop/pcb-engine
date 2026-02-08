#!/usr/bin/env python3
"""
Debug routing failures - why can't LED_CTRL route?
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import PCBEngine, BoardConfig, DesignRules
from pcb_engine.engine import EnginePhase


def create_design():
    """Same design as test_basic"""
    return {
        'U1': {
            'name': 'MCU',
            'footprint': 'QFN-16',
            'value': 'MCU',
            'description': 'Simple MCU',
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
            'name': 'LED',
            'footprint': '0603',
            'value': 'Green',
            'description': 'Status LED',
            'pins': [
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': -0.7, 'offset_y': 0}},
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.7, 'offset_y': 0}},
            ],
            'size': (1.6, 0.8),
        },
        'R1': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '1K',
            'description': 'LED Resistor',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED_CTRL',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },
    }


def run_debug():
    print("=" * 60)
    print("ROUTING DEBUG")
    print("=" * 60)

    board = BoardConfig(
        origin_x=100.0, origin_y=100.0,
        width=20.0, height=15.0,
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

    # Run phases 0-7 (including routing)
    for i in range(8):
        result = engine.run_phase(EnginePhase(i))
        print(f"Phase {i}: {'OK' if result else 'FAILED'}")

    # Now analyze routing state
    print("\n--- PLACEMENT ---")
    for ref, pos in engine.state.placement.items():
        print(f"  {ref}: ({pos.x:.1f}, {pos.y:.1f})")

    print("\n--- PIN POSITIONS ---")
    for ref, pos in engine.state.placement.items():
        part = engine.state.parts_db['parts'].get(ref, {})
        for pin in part.get('used_pins', []):
            offset = pin.get('offset', (0, 0))
            px, py = pos.x + offset[0], pos.y + offset[1]
            print(f"  {ref}.{pin['number']} ({pin.get('net', 'NC')}): ({px:.1f}, {py:.1f})")

    print("\n--- NETS TO ROUTE ---")
    for net_name in engine.state.route_order:
        net_info = engine.state.parts_db['nets'].get(net_name, {})
        pins = net_info.get('pins', [])
        print(f"  {net_name}: {pins}")

    print("\n--- ESCAPE ENDPOINTS ---")
    for ref, escapes in engine.state.escapes.items():
        for pin_num, escape in escapes.items():
            print(f"  {ref}.{pin_num}: start=({escape.start[0]:.1f}, {escape.start[1]:.1f}) -> end=({escape.endpoint[0]:.1f}, {escape.endpoint[1]:.1f}), dir={escape.direction_name}")

    # Manually test routing LED_CTRL
    print("\n--- MANUAL ROUTE TEST: LED_CTRL ---")

    # LED_CTRL: U1.3 -> R1.1
    # U1.3 is at (110 - 1.5, 107.5 + 2.0) = (108.5, 109.5)
    # R1.1 is at (108.2 - 0.45, 102.2) = (107.75, 102.2)

    # Check if there's an escape for U1.3
    if 'U1' in engine.state.escapes and '3' in engine.state.escapes['U1']:
        escape = engine.state.escapes['U1']['3']
        print(f"  U1.3 escape: start=({escape.start[0]:.1f}, {escape.start[1]:.1f}) -> end=({escape.endpoint[0]:.1f}, {escape.endpoint[1]:.1f})")
    else:
        print("  U1.3 has NO escape (2-pin components skip escapes)")

    # Get router grid info
    router = engine._router if hasattr(engine, '_router') else None
    if router:
        grid = router.grid
        print(f"\n  Grid size: {grid.rows}x{grid.cols} cells")

        # Check what actually failed
        print(f"\n  Router results:")
        print(f"    Routed: {list(router.routes.keys())}")
        print(f"    Failed: {router.failed}")

        # Check if start/end points are blocked
        # LED_CTRL start: U1.3 escape endpoint or pad
        # LED_CTRL end: R1.1 pad

        u1_pos = engine.state.placement['U1']
        r1_pos = engine.state.placement['R1']

        start_x, start_y = u1_pos.x - 1.5, u1_pos.y + 2.0  # U1.3 pad
        end_x, end_y = r1_pos.x - 0.45, r1_pos.y  # R1.1 pad

        start_row, start_col = grid.world_to_grid(start_x, start_y)
        end_row, end_col = grid.world_to_grid(end_x, end_y)

        print(f"\n  LED_CTRL routing:")
        print(f"    Start (U1.3): ({start_x:.1f}, {start_y:.1f}) -> grid ({start_row}, {start_col})")
        print(f"    End (R1.1): ({end_x:.1f}, {end_y:.1f}) -> grid ({end_row}, {end_col})")

        # Check cell status
        from pcb_engine.routing import Layer, GridCell

        start_cell = GridCell(start_row, start_col, Layer.F_CU)
        end_cell = GridCell(end_row, end_col, Layer.F_CU)

        print(f"\n  Cell status:")
        print(f"    Start blocked: {grid.is_blocked(start_cell)}")
        print(f"    End blocked: {grid.is_blocked(end_cell)}")

        # Try to find path
        print("\n  Attempting A* path...")
        path = router.astar.find_path(start_cell, end_cell, 'LED_CTRL', allow_via=True)
        if path:
            print(f"    SUCCESS! Path length: {len(path)} cells")
        else:
            print("    FAILED - no path found")

            # Check what's blocking
            print("\n  Checking blocked cells in area...")
            for r in range(min(start_row, end_row) - 2, max(start_row, end_row) + 3):
                for c in range(min(start_col, end_col) - 2, max(start_col, end_col) + 3):
                    if 0 <= r < grid.rows and 0 <= c < grid.cols:
                        cell = GridCell(r, c, Layer.F_CU)
                        if grid.is_blocked(cell):
                            wx, wy = grid.grid_to_world(r, c)
                            owner = grid.cell_owner.get((r, c, Layer.F_CU), 'unknown')
                            print(f"      BLOCKED: ({r},{c}) = ({wx:.1f}, {wy:.1f}) owner={owner}")


if __name__ == '__main__':
    run_debug()
