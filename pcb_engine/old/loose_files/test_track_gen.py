"""Debug track generation"""
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

# Test _generate_tracks with GND route
from pcb_engine.output_piston import OutputPiston, OutputConfig
from pcb_engine.routing_types import Route

op = OutputPiston(OutputConfig(generate_gnd_pour=True))

gnd_route = result.routes['GND']
fcu_segments = [seg for seg in gnd_route.segments if seg.layer == 'F.Cu']

print("=" * 70)
print("Track Generation Debug")
print("=" * 70)

print(f"\nGND F.Cu segments: {len(fcu_segments)}")
for seg in fcu_segments:
    print(f"  {seg}")

# Create temporary route
fcu_route = Route(
    net='GND',
    segments=fcu_segments,
    vias=gnd_route.vias,
    success=gnd_route.success
)

print(f"\nfcu_route.segments: {len(fcu_route.segments)}")
for seg in fcu_route.segments:
    print(f"  layer={seg.layer}, start={seg.start}, end={seg.end}")

# Generate tracks
net_id = 3  # GND would be net 3
tracks = op._generate_tracks(fcu_route, net_id)

print(f"\nGenerated tracks content:")
print(tracks[:500] if len(tracks) > 500 else tracks)

print("=" * 70)
