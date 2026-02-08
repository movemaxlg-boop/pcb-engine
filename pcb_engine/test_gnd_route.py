"""Debug GND routing issue"""
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
print("DEBUG: GND Routing")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions
print(f"Placement: {placement}")

# Do routing
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

routeable_nets = ['VDIV', 'GND']
result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=placement,
    net_order=routeable_nets
)

print(f"\nRouting result: {result.routed_count}/{result.total_count}")
for net_name, route in result.routes.items():
    status = "OK" if route.success else "FAIL"
    error = f" - {route.error}" if route.error else ""
    print(f"  {net_name}: {status} - {len(route.segments)} segments{error}")

# Debug GND
if not result.routes.get('GND', None) or not result.routes['GND'].success:
    print("\n--- GND DEBUG ---")
    gnd_endpoints = rp._get_escape_endpoints(parts_db['nets']['GND']['pins'], {})
    print(f"GND endpoints: {gnd_endpoints}")

    if len(gnd_endpoints) >= 2:
        start = gnd_endpoints[0]
        end = gnd_endpoints[1]

        start_col = int(start[0] / 0.1)
        start_row = int(start[1] / 0.1)
        end_col = int(end[0] / 0.1)
        end_row = int(end[1] / 0.1)

        print(f"Start: ({start[0]}, {start[1]}) = grid[{start_row}][{start_col}]")
        print(f"End: ({end[0]}, {end[1]}) = grid[{end_row}][{end_col}]")

        # Check start/end accessibility
        start_cell = rp.fcu_grid[start_row][start_col] if rp._in_bounds(start_row, start_col) else 'OOB'
        end_cell = rp.fcu_grid[end_row][end_col] if rp._in_bounds(end_row, end_col) else 'OOB'
        print(f"Start cell: {start_cell!r}")
        print(f"End cell: {end_cell!r}")

        # Check accessibility
        print(f"Start accessible for GND: {rp._is_cell_accessible_for_net(rp.fcu_grid, start_row, start_col, 'GND')}")
        print(f"End accessible for GND: {rp._is_cell_accessible_for_net(rp.fcu_grid, end_row, end_col, 'GND')}")

        # Show what's between start and end
        print(f"\nGrid row {start_row} from col {min(start_col, end_col)} to {max(start_col, end_col)}:")
        blockers = []
        for c in range(min(start_col, end_col), max(start_col, end_col) + 1):
            cell = rp.fcu_grid[start_row][c]
            if cell and cell != 'GND':
                blockers.append((c, cell))
        print(f"  Non-GND cells: {len(blockers)}")
        for c, cell in blockers[:10]:
            print(f"    col {c}: {cell!r}")

print("=" * 70)
