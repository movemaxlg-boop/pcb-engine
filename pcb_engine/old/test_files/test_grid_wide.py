"""Visualize wider grid area to see if there's a path"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parts_db = {
    'parts': {
        'R1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'VDIV', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'R2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'C1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VIN': {'pins': [('R1', '1')]},
        'VDIV': {'pins': [('R1', '2'), ('R2', '1'), ('C1', '1')]},
        'GND': {'pins': [('R2', '2'), ('C1', '2')]},
    }
}

print("=" * 70)
print("DEBUG: Wide Grid View")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions
print(f"Placement: {placement}")

# Routing setup
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))
rp._initialize_grids()
rp._placement = placement
rp._parts_db = parts_db
rp._register_components(placement, parts_db)

# Visualize rows 130-160 (vertically), cols 145-200 (horizontally)
print("\nWide grid view (rows 130-160, cols 145-200):")
print("Legend: V=VDIV, G=GND, I=VIN, C=COMPONENT, .=empty")

for r in range(130, 161):
    row_str = f"[{r}] "
    for c in range(145, 201):
        cell = rp.fcu_grid[r][c]
        if cell is None:
            row_str += "."
        elif cell == 'VDIV':
            row_str += "V"
        elif cell == 'GND':
            row_str += "G"
        elif cell == 'VIN':
            row_str += "I"
        elif cell == '__COMPONENT__':
            row_str += "C"
        elif cell == '__PAD_CONFLICT__':
            row_str += "X"
        else:
            row_str += "?"
    print(row_str)

# Check if there's a path of empty cells around the components
print("\n\nChecking for clear path above components (row 130):")
clear_cols = []
for c in range(145, 201):
    if rp._is_cell_clear_for_net(rp.fcu_grid, 130, c, 'VDIV'):
        clear_cols.append(c)
print(f"Clear cells at row 130: {len(clear_cols)} cells")
if clear_cols:
    print(f"  Range: col {clear_cols[0]} to {clear_cols[-1]}")

print("\nChecking for clear path below components (row 160):")
clear_cols = []
for c in range(145, 201):
    if rp._is_cell_clear_for_net(rp.fcu_grid, 160, c, 'VDIV'):
        clear_cols.append(c)
print(f"Clear cells at row 160: {len(clear_cols)} cells")
if clear_cols:
    print(f"  Range: col {clear_cols[0]} to {clear_cols[-1]}")

print("=" * 70)
