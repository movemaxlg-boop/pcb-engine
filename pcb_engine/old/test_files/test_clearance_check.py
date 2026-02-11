"""Test clearance checking around escape endpoints"""
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
print("TEST: Clearance Checking")
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
print(f"Clearance radius: {piston.clearance_radius}mm")

# Check escape endpoint at (5.95, 23.0)
escape_x, escape_y = 5.95, 23.0
escape_col = int(escape_x / config.grid_size)
escape_row = int(escape_y / config.grid_size)

print(f"\nEscape endpoint R1.2 at ({escape_x}, {escape_y}) -> grid[{escape_row}][{escape_col}]")
print(f"Center cell value: {piston.fcu_grid[escape_row][escape_col]}")

# Check the clearance zone
cc = piston.clearance_cells
blocked_by = []
for dr in range(-cc, cc + 1):
    for dc in range(-cc, cc + 1):
        r, c = escape_row + dr, escape_col + dc
        if piston._in_bounds(r, c):
            occ = piston.fcu_grid[r][c]
            if occ is not None and occ != 'SIG':
                blocked_by.append((r, c, occ, dr, dc))

print(f"\nCells blocking SIG routing in clearance zone:")
if blocked_by:
    for r, c, occ, dr, dc in blocked_by:
        real_x = c * config.grid_size
        real_y = r * config.grid_size
        print(f"  grid[{r}][{c}] = {occ} at ({real_x:.2f}, {real_y:.2f}), offset ({dr}, {dc})")
else:
    print("  None!")

# Now check if _is_cell_clear_for_net returns True
is_clear = piston._is_cell_clear_for_net(piston.fcu_grid, escape_row, escape_col, 'SIG')
print(f"\n_is_cell_clear_for_net(SIG) at escape endpoint: {is_clear}")

print("=" * 60)
