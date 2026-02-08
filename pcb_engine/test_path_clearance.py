"""Test clearance along the path between escape endpoints"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple 2-resistor circuit
parts_db = {
    'parts': {
        'R1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'SIG', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'R2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'SIG', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VCC': {'pins': [('R1', '1')]},
        'SIG': {'pins': [('R1', '2'), ('R2', '1')]},
        'GND': {'pins': [('R2', '2')]},
    }
}

placement = {'R1': (5.0, 22.0), 'R2': (8.0, 22.0)}

print("=" * 60)
print("TEST: Path Clearance")
print("=" * 60)

from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

config = RoutingConfig(
    board_width=30.0,
    board_height=25.0,
    algorithm='lee',
    grid_size=0.1,
    trace_width=0.25,
    clearance=0.2
)

piston = RoutingPiston(config)
piston._initialize_grids()
piston._register_components(placement, parts_db)

print(f"Clearance cells: {piston.clearance_cells}")

# Path from (5.95, 23.0) to (7.05, 23.0) at y=23
start_col = int(5.95 / config.grid_size)  # 59
end_col = int(7.05 / config.grid_size)    # 70
row = int(23.0 / config.grid_size)         # 230

print(f"\nPath from col {start_col} to col {end_col} at row {row} (y=23.0):")
blocked_cols = []
for col in range(start_col, end_col + 1):
    is_clear = piston._is_cell_clear_for_net(piston.fcu_grid, row, col, 'SIG')
    cell_value = piston.fcu_grid[row][col]
    if not is_clear:
        blocked_cols.append((col, cell_value))
        print(f"  col {col} (x={col*0.1:.2f}): BLOCKED, cell={cell_value}")
    else:
        pass  # print(f"  col {col} (x={col*0.1:.2f}): CLEAR, cell={cell_value}")

if blocked_cols:
    print(f"\n{len(blocked_cols)} cells blocked on the path")
    # Find what's blocking
    for col, cell_value in blocked_cols[:3]:  # First 3
        print(f"\nAnalyzing blocked col {col}:")
        cc = piston.clearance_cells
        for dr in range(-cc, cc + 1):
            for dc in range(-cc, cc + 1):
                r, c = row + dr, col + dc
                if piston._in_bounds(r, c):
                    occ = piston.fcu_grid[r][c]
                    if occ is not None and occ != 'SIG' and occ not in piston.BLOCKED_MARKERS:
                        print(f"  grid[{r}][{c}] = {occ} (at offset {dr},{dc})")
                    if occ in piston.BLOCKED_MARKERS:
                        print(f"  grid[{r}][{c}] = {occ} (BLOCKED_MARKER at offset {dr},{dc})")
else:
    print("\nAll cells on path are CLEAR!")

print("=" * 60)
