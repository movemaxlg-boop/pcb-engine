"""Test approach detection with 2x radius"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parts_db = {
    'parts': {
        'R1': {'footprint': '0805', 'pins': [
            {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
            {'number': '2', 'net': 'VDIV', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
        ]},
        'R2': {'footprint': '0805', 'pins': [
            {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
            {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
        ]},
        'C1': {'footprint': '0805', 'pins': [
            {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
            {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
        ]},
    },
    'nets': {
        'VIN': {'pins': [('R1', '1')]},
        'VDIV': {'pins': [('R1', '2'), ('R2', '1'), ('C1', '1')]},
        'GND': {'pins': [('R2', '2'), ('C1', '2')]},
    }
}

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

result = rp.route(parts_db=parts_db, escapes={}, placement=placement, net_order=['VDIV', 'GND'])

grid = rp.fcu_grid
net_name = 'GND'
cc = rp.clearance_cells
approach_radius = cc * 2  # This is what the code uses now
end_row, end_col = 145, 179

print(f"Clearance cells: {cc}")
print(f"Approach radius: {approach_radius}")
print(f"\nColumn {end_col} analysis (using approach_radius={approach_radius}):")

for r in range(130, 146):
    cell = grid[r][end_col]
    accessible = rp._is_cell_accessible_for_net(grid, r, end_col, net_name)

    # Check with 2x radius
    is_approaching = False
    for dr2 in range(-approach_radius, approach_radius + 1):
        for dc2 in range(-approach_radius, approach_radius + 1):
            check_r, check_c = r + dr2, end_col + dc2
            if rp._in_bounds(check_r, check_c) and grid[check_r][check_c] == net_name:
                is_approaching = True
                break
        if is_approaching:
            break

    print(f"  [{r}][{end_col}]: {cell!r:15s} accessible={accessible}, approaching={is_approaching}")
