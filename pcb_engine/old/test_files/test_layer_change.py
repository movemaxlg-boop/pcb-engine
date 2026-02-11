"""Test if layer change (via) works for GND routing"""
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

print(f"Placement: {placement}")

# Check routing config
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

print(f"Allow layer change: {rp.config.allow_layer_change}")
print(f"Via cost: {rp.config.via_cost}")

# Route
result = rp.route(parts_db=parts_db, escapes={}, placement=placement, net_order=['VDIV', 'GND'])

print(f"\nRouting result: {result.routed_count}/{result.total_count}")
for name, route in result.routes.items():
    print(f"  {name}: {'OK' if route.success else 'FAIL'} - {len(route.segments)} segments, {len(route.vias)} vias")

# Check bottom layer grid around row 137
print("\n--- Bottom layer around component body ---")
end_col = 179
for r in range(135, 140):
    cell_f = rp.fcu_grid[r][end_col] if rp._in_bounds(r, end_col) else 'OOB'
    cell_b = rp.bcu_grid[r][end_col] if rp._in_bounds(r, end_col) else 'OOB'
    print(f"  [{r}][{end_col}]: F.Cu={cell_f!r:15s} B.Cu={cell_b!r}")
