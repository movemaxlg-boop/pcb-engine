"""Trace what path the actual routing returns for GND"""
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
print("Actual Routing Path for GND")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
placement = pe.place(parts_db, {})

# Routing setup
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.15,
    via_diameter=0.6,
    via_drill=0.3
))

# GND endpoints
c1 = placement.positions['C1']
r2 = placement.positions['R2']
c1_x = c1[0] if isinstance(c1, (list, tuple)) else c1.x
r2_x = r2[0] if isinstance(r2, (list, tuple)) else r2.x

start = (r2_x + 0.95, 14.5)  # R2.GND
end = (c1_x + 0.95, 14.5)    # C1.GND

print(f"Start (R2.GND): {start}")
print(f"End (C1.GND): {end}")

# Setup and call Lee directly
rp._placement = placement.positions
rp._parts_db = parts_db
rp._register_components(placement.positions, parts_db)

# Manually call _lee_wavefront_3d
segments, vias, success = rp._lee_wavefront_3d(start, end, 'GND')

print(f"\nSuccess: {success}")
print(f"Segments: {len(segments)}")
print(f"Vias: {len(vias)}")

print("\nSegments:")
for i, seg in enumerate(segments):
    print(f"  {i}: ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f}) on {seg.layer}")

print("\nVias:")
for i, via in enumerate(vias):
    print(f"  {i}: ({via.position[0]:.2f}, {via.position[1]:.2f})")

# Check if any segment appears to go through blocked area
print("\n" + "=" * 40)
print("Segment analysis:")
print("=" * 40)

for i, seg in enumerate(segments):
    # For horizontal segments at Y=14.5
    if abs(seg.start[1] - 14.5) < 0.1 and abs(seg.end[1] - 14.5) < 0.1:
        start_x = min(seg.start[0], seg.end[0])
        end_x = max(seg.start[0], seg.end[0])

        # Check if it crosses VDIV area (18.5 to 19.5)
        if start_x < 18.5 and end_x > 19.5:
            print(f"  Segment {i} CROSSES VDIV area at Y=14.5!")
            print(f"    From X={start_x:.2f} to X={end_x:.2f}")
            print(f"    VDIV is at X=18.5-19.5")
        else:
            print(f"  Segment {i}: X={start_x:.2f} to {end_x:.2f} at Y=14.5 - OK")
    else:
        print(f"  Segment {i}: Not at Y=14.5")

print("=" * 70)
