"""Debug GND routing in detail"""
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

print("=" * 60)
print("GND Routing Debug")
print("=" * 60)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
placement = pe.place(parts_db, {})

# GND pad positions
c1 = placement.positions['C1']
r2 = placement.positions['R2']
c1_x = c1[0] if isinstance(c1, (list, tuple)) else c1.x
r2_x = r2[0] if isinstance(r2, (list, tuple)) else r2.x

c1_gnd = (c1_x + 0.95, 14.5)
r2_gnd = (r2_x + 0.95, 14.5)

print(f"\nC1.GND: {c1_gnd}")
print(f"R2.GND: {r2_gnd}")
print(f"Distance: {abs(r2_gnd[0] - c1_gnd[0]):.2f}mm")

# This is a simple case - both pads on same Y, direct route should be possible
print("\nThis is a simple case - both pads on same row (Y=14.5)")
print("A direct F.Cu trace from R2.GND to C1.GND should work!")

# Routing - check what happens
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.15,
    via_diameter=0.6,
    via_drill=0.3
))

rp._placement = placement.positions
rp._parts_db = parts_db
rp._register_components(placement.positions, parts_db)

# Check the grid at GND pad positions
r2_col = int(r2_gnd[0] / 0.1)
c1_col = int(c1_gnd[0] / 0.1)
row = int(14.5 / 0.1)

print(f"\nGrid check at Y=14.5 (row {row}):")
print(f"  R2.GND position col {r2_col}")
print(f"  C1.GND position col {c1_col}")
print()

# Check what's between them on F.Cu
print("F.Cu from R2.GND to C1.GND:")
blocked = []
for col in range(c1_col, r2_col + 1):
    cell = rp.fcu_grid[row][col]
    if cell is not None and cell != 'GND':
        blocked.append((col, cell))

if blocked:
    print(f"  BLOCKED by: {blocked}")
else:
    print("  PATH IS CLEAR on F.Cu!")

# Now actually route and see what happens
result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=placement.positions,
    net_order=['GND']
)

print(f"\nRouting result: {result.routed_count}/{result.total_count}")

if 'GND' in result.routes:
    gnd = result.routes['GND']
    print(f"Segments: {len(gnd.segments)}")
    print(f"Vias: {len(gnd.vias)}")
    for seg in gnd.segments:
        print(f"  {seg.layer}: ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f})")
    for via in gnd.vias:
        print(f"  VIA: ({via.position[0]:.2f}, {via.position[1]:.2f})")

print("=" * 60)
