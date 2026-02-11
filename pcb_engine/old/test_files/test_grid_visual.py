"""Visualize the grid around the routing path"""
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
print("DEBUG: Grid Visualization")
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

# Visualize grid around row 145 (where components are)
# R1 at col 140 (center 14.0), R2 at col 200 (center 20.0), C1 at col 170 (center 17.0)
print("\nGrid visualization at row 145 (col 130 to 220):")
print("Legend: V=VDIV, G=GND, I=VIN, C=COMPONENT, X=PAD_CONFLICT, .=empty")

row = 145
row_str = ""
for c in range(130, 220):
    cell = rp.fcu_grid[row][c]
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

print(f"Row {row}: {row_str}")

# Show column markers
markers = ""
for c in range(130, 220):
    if c % 10 == 0:
        markers += str((c // 10) % 10)
    else:
        markers += " "
print(f"Cols:   {markers}")

# Visualize rows 140-150
print("\nGrid visualization (rows 140-150, cols 130-220):")
for r in range(140, 151):
    row_str = f"[{r}] "
    for c in range(130, 220):
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

# Check what's between R1.2 (col ~149) and R2.1 (col ~190)
print("\n\nWhat's between R1.2 (VDIV at col 149) and R2.1 (VDIV at col 190)?")
blocked_by = {}
for c in range(149, 191):
    cell = rp.fcu_grid[145][c]
    if cell and cell != 'VDIV':
        if cell not in blocked_by:
            blocked_by[cell] = []
        blocked_by[cell].append(c)

for blocker, cols in blocked_by.items():
    print(f"  {blocker}: columns {cols[0]} to {cols[-1]} ({len(cols)} cells)")

# Check clearance issue
print("\n\nClearance check analysis:")
print(f"Clearance cells: {rp.clearance_cells}")
print("For a cell to be 'clear', ALL cells in the clearance zone must be empty or same net")
print("\nChecking col 155 (between R1 and C1):")
c = 155
for r in range(141, 150):
    clear = rp._is_cell_clear_for_net(rp.fcu_grid, r, c, 'VDIV')
    cell = rp.fcu_grid[r][c]
    print(f"  [{r}][{c}] = {cell!r}, clear for VDIV: {clear}")

print("=" * 70)
