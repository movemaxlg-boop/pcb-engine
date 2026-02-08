"""Debug output piston GND segment handling"""
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
print("Output Piston Debug")
print("=" * 70)

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

print(f"\nRouting result: {result.routed_count}/{result.total_count}")

# Check GND route
print("\nGND route details:")
if 'GND' in result.routes:
    gnd = result.routes['GND']
    print(f"  Total segments: {len(gnd.segments)}")
    print(f"  Vias: {len(gnd.vias)}")

    fcu_segments = [seg for seg in gnd.segments if seg.layer == 'F.Cu']
    bcu_segments = [seg for seg in gnd.segments if seg.layer == 'B.Cu']

    print(f"  F.Cu segments: {len(fcu_segments)}")
    for seg in fcu_segments:
        print(f"    ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f})")

    print(f"  B.Cu segments: {len(bcu_segments)}")
    for seg in bcu_segments:
        print(f"    ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f})")
else:
    print("  GND not routed!")

# Now manually call the output piston to see what happens
print("\n" + "=" * 40)
print("Output piston processing:")
print("=" * 40)

from pcb_engine.output_piston import OutputPiston, OutputConfig

op = OutputPiston(OutputConfig(generate_gnd_pour=True))

# Check what segments will be output
for net_name, route in result.routes.items():
    if net_name == 'GND' and op.config.generate_gnd_pour:
        fcu_segs = [seg for seg in route.segments if seg.layer == 'F.Cu']
        print(f"\nGND (with pour): {len(fcu_segs)} F.Cu segments will be output")
        for seg in fcu_segs:
            print(f"  ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f})")
    else:
        print(f"\n{net_name}: {len(route.segments)} segments will be output")

print("=" * 70)
