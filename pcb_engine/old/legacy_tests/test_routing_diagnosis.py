#!/usr/bin/env python3
"""
Routing Diagnosis Test
======================

This test diagnoses WHY routes are failing by:
1. Printing the grid occupancy after component registration
2. Showing exactly which cells are blocked
3. Tracing through each routing attempt

This is the ENGINEER debugging the MACHINE.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.engine import BoardConfig, DesignRules
from pcb_engine.human_engine import HumanPCBEngine
from pcb_engine.human_routing import HumanLikeRouter
from pcb_engine.test_human_complex import create_complex_design


def diagnose_routing():
    """Diagnose routing failures"""

    print("=" * 70)
    print("ROUTING DIAGNOSIS - Finding why routes fail")
    print("=" * 70)
    print()

    # Board configuration
    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=60.0,
        height=45.0,
        layers=2,
        grid_size=0.5
    )

    rules = DesignRules(
        min_trace_width=0.25,
        min_clearance=0.2,
        min_via_diameter=0.8,
        min_via_drill=0.4,
    )

    # Create engine and run phases up to routing
    engine = HumanPCBEngine(board, rules)
    parts = create_complex_design()
    engine.load_parts_from_dict(parts)

    # Run up to placement and escapes
    engine._phase_1_parts()
    engine._phase_2_graph()
    engine._phase_3_placement()
    engine._phase_4_escapes()

    print("\n" + "=" * 70)
    print("PLACEMENT SUMMARY")
    print("=" * 70)
    for ref, pos in engine.state.placement.items():
        print(f"  {ref}: ({pos.x}, {pos.y})")

    print("\n" + "=" * 70)
    print("ESCAPE ENDPOINTS (where signal routes start/end)")
    print("=" * 70)

    # Group escapes by net
    net_escapes = {}
    for ref, comp_escapes in engine.state.escapes.items():
        for pin_num, escape in comp_escapes.items():
            net = escape.net
            if net not in net_escapes:
                net_escapes[net] = []
            net_escapes[net].append({
                'ref': ref,
                'pin': pin_num,
                'start': escape.start,
                'end': escape.end,
                'direction': escape.direction_name
            })

    for net, escapes in sorted(net_escapes.items()):
        print(f"\nNet: {net}")
        for esc in escapes:
            print(f"  {esc['ref']}.{esc['pin']}: start={esc['start']} -> end={esc['end']} dir={esc['direction']}")

    print("\n" + "=" * 70)
    print("GRID OCCUPANCY ANALYSIS")
    print("=" * 70)

    # Create router to inspect grid
    router = HumanLikeRouter(
        board_width=board.width,
        board_height=board.height,
        origin_x=board.origin_x,
        origin_y=board.origin_y,
        grid_size=board.grid_size,
    )

    # Register components
    router.register_components_in_grid(engine.state.placement, engine.state.parts_db)

    # Count blocked cells
    fcu_blocked = sum(1 for row in router.fcu_grid for cell in row if cell == '__COMPONENT__')
    fcu_total = router.grid_rows * router.grid_cols
    print(f"\nF.Cu grid: {router.grid_cols}x{router.grid_rows} = {fcu_total} cells")
    print(f"Component-blocked cells: {fcu_blocked} ({100*fcu_blocked/fcu_total:.1f}%)")

    # Register escapes
    router.register_escapes_in_grid(engine.state.escapes)

    # Count escape-occupied cells
    fcu_escape = sum(1 for row in router.fcu_grid for cell in row if cell is not None and cell != '__COMPONENT__')
    print(f"Escape-occupied cells: {fcu_escape}")

    print("\n" + "=" * 70)
    print("ROUTING ATTEMPT DIAGNOSIS - LED net")
    print("=" * 70)

    # Try routing LED net manually with debug
    led_escapes = net_escapes.get('LED', [])
    if len(led_escapes) >= 2:
        start = led_escapes[0]['end']
        end = led_escapes[1]['end']

        print(f"\nLED route: {start} -> {end}")

        # Check if start/end are blocked
        start_col = int((start[0] - board.origin_x) / board.grid_size)
        start_row = int((start[1] - board.origin_y) / board.grid_size)
        end_col = int((end[0] - board.origin_x) / board.grid_size)
        end_row = int((end[1] - board.origin_y) / board.grid_size)

        print(f"  Start grid: ({start_col}, {start_row}) = {router.fcu_grid[start_row][start_col]}")
        print(f"  End grid: ({end_col}, {end_row}) = {router.fcu_grid[end_row][end_col]}")

        # Check horizontal path (h_first strategy)
        print(f"\n  H-first strategy: go from x={start[0]} to x={end[0]} at y={start[1]}")
        blocked_cells = []
        for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
            cell = router.fcu_grid[start_row][col]
            if cell is not None and cell != 'LED':
                blocked_cells.append((col, start_row, cell))

        if blocked_cells:
            print(f"    BLOCKED by: {blocked_cells[:5]}{'...' if len(blocked_cells) > 5 else ''}")
        else:
            print(f"    Horizontal segment clear!")

        # Check vertical path
        print(f"\n  Then vertical: go from y={start[1]} to y={end[1]} at x={end[0]}")
        blocked_cells = []
        for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
            cell = router.fcu_grid[row][end_col]
            if cell is not None and cell != 'LED':
                blocked_cells.append((end_col, row, cell))

        if blocked_cells:
            print(f"    BLOCKED by: {blocked_cells[:5]}{'...' if len(blocked_cells) > 5 else ''}")
        else:
            print(f"    Vertical segment clear!")

    print("\n" + "=" * 70)
    print("ROUTING ATTEMPT DIAGNOSIS - SDA net")
    print("=" * 70)

    sda_escapes = net_escapes.get('SDA', [])
    if len(sda_escapes) >= 2:
        # SDA has multiple pins, find U1 and U2/U3
        u1_esc = next((e for e in sda_escapes if e['ref'] == 'U1'), None)
        u2_esc = next((e for e in sda_escapes if e['ref'] == 'U2'), None)

        if u1_esc and u2_esc:
            start = u1_esc['end']
            end = u2_esc['end']

            print(f"\nSDA route: {start} -> {end}")

            start_col = int((start[0] - board.origin_x) / board.grid_size)
            start_row = int((start[1] - board.origin_y) / board.grid_size)
            end_col = int((end[0] - board.origin_x) / board.grid_size)
            end_row = int((end[1] - board.origin_y) / board.grid_size)

            print(f"  Start grid: ({start_col}, {start_row}) = {router.fcu_grid[start_row][start_col]}")
            print(f"  End grid: ({end_col}, {end_row}) = {router.fcu_grid[end_row][end_col]}")

            # Check horizontal path (h_first strategy)
            print(f"\n  H-first: go from x={start[0]} to x={end[0]} at y={start[1]}")
            blocked_cells = []
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                cell = router.fcu_grid[start_row][col]
                if cell is not None and cell != 'SDA':
                    blocked_cells.append((col, start_row, cell))

            if blocked_cells:
                print(f"    BLOCKED by: {blocked_cells[:10]}{'...' if len(blocked_cells) > 10 else ''}")
            else:
                print(f"    Horizontal segment clear!")

            # Check vertical path
            print(f"\n  Then vertical: go from y={start[1]} to y={end[1]} at x={end[0]}")
            blocked_cells = []
            for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
                cell = router.fcu_grid[row][end_col]
                if cell is not None and cell != 'SDA':
                    blocked_cells.append((end_col, row, cell))

            if blocked_cells:
                print(f"    BLOCKED by: {blocked_cells[:10]}{'...' if len(blocked_cells) > 10 else ''}")
            else:
                print(f"    Vertical segment clear!")

    # Print visual grid map for a region
    print("\n" + "=" * 70)
    print("VISUAL GRID MAP (rows 15-45, cols 20-50)")
    print("=" * 70)
    print("Legend: . = empty, C = component, letter = net first char")
    print()

    # Print header with column numbers
    print("    ", end="")
    for col in range(20, 51):
        print(f"{col%10}", end="")
    print()

    for row in range(15, 46):
        print(f"{row:3} ", end="")
        for col in range(20, 51):
            if row < router.grid_rows and col < router.grid_cols:
                cell = router.fcu_grid[row][col]
                if cell is None:
                    print(".", end="")
                elif cell == '__COMPONENT__':
                    print("C", end="")
                else:
                    # First letter of net name
                    print(cell[0].upper() if cell else "?", end="")
            else:
                print(" ", end="")
        print()


if __name__ == '__main__':
    diagnose_routing()
