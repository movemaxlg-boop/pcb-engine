"""Debug why via clearance check fails to prevent shorting with GND pad"""
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
print("DEBUG: Via Clearance Check")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

print("\nPlacement:")
for ref, pos in placement.items():
    print(f"  {ref}: ({pos[0]:.2f}, {pos[1]:.2f})")

# Routing setup
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

# Register components (this marks pads)
rp._placement = placement
rp._parts_db = parts_db
rp._register_components(placement, parts_db)

print(f"\nClearance cells: {rp.clearance_cells}")
print(f"Clearance radius: {rp.clearance_radius:.3f}mm")

# C1 position
c1_pos = placement['C1']
c1_x = c1_pos[0] if isinstance(c1_pos, (list, tuple)) else c1_pos.x
c1_y = c1_pos[1] if isinstance(c1_pos, (list, tuple)) else c1_pos.y

# GND pad position (C1 pin 2)
gnd_pad_x = c1_x + 0.95
gnd_pad_y = c1_y

print(f"\nC1 position: ({c1_x:.2f}, {c1_y:.2f})")
print(f"GND pad (C1.2): ({gnd_pad_x:.2f}, {gnd_pad_y:.2f})")

# Convert to grid
gnd_col = int(gnd_pad_x / 0.1)
gnd_row = int(gnd_pad_y / 0.1)

print(f"GND pad grid: row={gnd_row}, col={gnd_col}")

# Check what's in the grid around GND pad
print("\nGrid around GND pad (cols 175-190, row at GND pad):")
for c in range(175, 191):
    cell = rp.fcu_grid[gnd_row][c] if rp._in_bounds(gnd_row, c) else "OOB"
    x = c * 0.1
    print(f"  col {c} (x={x:.1f}): {cell!r}")

# Now check: would a via at (18.5, gnd_y) be allowed for VDIV?
via_x = 18.5
via_col = int(via_x / 0.1)  # 185

print(f"\n" + "=" * 50)
print(f"CRITICAL CHECK: Can VDIV via be placed at col {via_col} (x={via_x})?")
print("=" * 50)

# Manual clearance check
cc = rp.clearance_cells
print(f"Checking clearance zone: cols {via_col - cc} to {via_col + cc}")
blocking_cells = []
for dc in range(-cc, cc + 1):
    c = via_col + dc
    if rp._in_bounds(gnd_row, c):
        cell = rp.fcu_grid[gnd_row][c]
        if cell is not None and cell != 'VDIV':
            blocking_cells.append((c, cell))
            print(f"  col {c}: {cell!r} <- SHOULD BLOCK!")

print(f"\nBlocking cells found: {len(blocking_cells)}")

# Now use the actual method
clear = rp._is_cell_clear_for_net(rp.fcu_grid, gnd_row, via_col, 'VDIV')
print(f"\n_is_cell_clear_for_net(grid, {gnd_row}, {via_col}, 'VDIV') = {clear}")

if clear and blocking_cells:
    print("\n*** BUG: Method returned True but there are blocking cells! ***")
else:
    print("\nMethod returned correctly.")

print("=" * 70)
