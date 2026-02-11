"""Debug GND routing segments"""
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
print("GND Routing Segment Analysis")
print("=" * 60)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
placement = pe.place(parts_db, {})

# Routing
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.15,
    via_diameter=0.6,
    via_drill=0.3
))

result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=placement.positions,
    net_order=['GND', 'VDIV']
)

print(f"\nRouted: {result.routed_count}/{result.total_count}")

# Analyze GND
print("\n" + "=" * 40)
print("GND ROUTE:")
print("=" * 40)
if 'GND' in result.routes:
    gnd = result.routes['GND']
    print(f"Segments: {len(gnd.segments)}")
    print(f"Vias: {len(gnd.vias)}")
    # print(f"Complete: {gnd.complete}")

    print("\nSegments:")
    for i, seg in enumerate(gnd.segments):
        print(f"  {i}: ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f}) on {seg.layer}")

    print("\nVias:")
    for i, via in enumerate(gnd.vias):
        print(f"  {i}: ({via.position[0]:.2f}, {via.position[1]:.2f})")

    # Expected endpoints
    print("\nExpected connections:")
    c1 = placement.positions['C1']
    r2 = placement.positions['R2']
    c1_x = c1[0] if isinstance(c1, (list, tuple)) else c1.x
    r2_x = r2[0] if isinstance(r2, (list, tuple)) else r2.x
    c1_gnd = (c1_x + 0.95, 14.5)
    r2_gnd = (r2_x + 0.95, 14.5)
    print(f"  C1.GND: {c1_gnd}")
    print(f"  R2.GND: {r2_gnd}")
else:
    print("GND not routed!")

# Analyze VDIV
print("\n" + "=" * 40)
print("VDIV ROUTE:")
print("=" * 40)
if 'VDIV' in result.routes:
    vdiv = result.routes['VDIV']
    print(f"Segments: {len(vdiv.segments)}")
    print(f"Vias: {len(vdiv.vias)}")
    # print(f"Complete: {vdiv.complete}")

    print("\nSegments:")
    for i, seg in enumerate(vdiv.segments):
        print(f"  {i}: ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f}) on {seg.layer}")

print("=" * 60)
