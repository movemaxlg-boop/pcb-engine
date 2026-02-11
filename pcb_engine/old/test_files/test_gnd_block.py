"""Debug what's blocking GND at the final approach"""
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
print("DEBUG: GND Final Approach Block Analysis")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

# Routing setup
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=placement,
    net_order=['VDIV', 'GND']
)

# GND end is at grid[145][179]
end_row, end_col = 145, 179

print(f"GND target: grid[{end_row}][{end_col}]")
print(f"Clearance cells: {rp.clearance_cells}")

# Check what's at and around the target
print("\nGrid around GND end (rows 130-152, col 179):")
for r in range(130, 153):
    cell = rp.fcu_grid[r][end_col]
    clear = rp._is_cell_clear_for_net(rp.fcu_grid, r, end_col, 'GND')
    accessible = rp._is_cell_accessible_for_net(rp.fcu_grid, r, end_col, 'GND')
    print(f"  [{r}][{end_col}]: {cell!r:15s} clear={clear}, accessible={accessible}")

# Check clearance zone at row 137 (just above component body at 137)
print(f"\nClearance zone analysis at row 137, col {end_col}:")
cc = rp.clearance_cells
for dr in range(-cc, cc + 1):
    for dc in range(-cc, cc + 1):
        r, c = 137 + dr, end_col + dc
        if rp._in_bounds(r, c):
            cell = rp.fcu_grid[r][c]
            if cell and cell != 'GND':
                print(f"  [{r}][{c}]: {cell!r}")

print("=" * 70)
