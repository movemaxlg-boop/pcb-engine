"""Debug routes dictionary passed to output piston"""
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

print(f"Routes in result.routes:")
for net_name, route in result.routes.items():
    print(f"  {net_name}: {len(route.segments)} segments, {len(route.vias)} vias")

print()
print("result.routes type:", type(result.routes))
print("result.routes keys:", list(result.routes.keys()))

# Now replicate what happens in output piston
routes = result.routes
for net_name, route in routes.items():
    print(f"\nProcessing {net_name}:")
    print(f"  generate_gnd_pour: True")
    print(f"  net_name == 'GND': {net_name == 'GND'}")

    if net_name == 'GND':
        fcu_segments = [seg for seg in route.segments if seg.layer == 'F.Cu']
        print(f"  F.Cu segments: {len(fcu_segments)}")
        if fcu_segments:
            print(f"  Will output {len(fcu_segments)} GND F.Cu segments")
    else:
        print(f"  Will output all {len(route.segments)} segments")

print("=" * 70)
