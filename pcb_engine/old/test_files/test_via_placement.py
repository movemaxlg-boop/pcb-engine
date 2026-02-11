"""Debug where the VDIV via actually gets placed during routing"""
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
print("DEBUG: Where does VDIV via actually get placed?")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

print("\nPlacement:")
for ref, pos in placement.items():
    print(f"  {ref}: ({pos[0]:.2f}, {pos[1]:.2f})")

# Route
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

print(f"\nRouting result: {result.routed_count}/{result.total_count}")

# Analyze VDIV route
print("\nVDIV route analysis:")
if 'VDIV' in result.routes:
    vdiv_route = result.routes['VDIV']
    print(f"  Segments: {len(vdiv_route.segments)}")
    print(f"  Vias: {len(vdiv_route.vias)}")

    for i, seg in enumerate(vdiv_route.segments):
        print(f"    Segment {i}: ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f}) on {seg.layer}")

    for i, via in enumerate(vdiv_route.vias):
        print(f"    Via {i}: ({via.position[0]:.2f}, {via.position[1]:.2f}) {via.from_layer} -> {via.to_layer}")

        # Check if this via is near GND pad
        c1_pos = placement['C1']
        c1_x = c1_pos[0] if isinstance(c1_pos, (list, tuple)) else c1_pos.x
        gnd_pad_x = c1_x + 0.95  # GND pad is at C1 + offset
        gnd_pad_y = c1_pos[1] if isinstance(c1_pos, (list, tuple)) else c1_pos.y

        dist = abs(via.position[0] - gnd_pad_x)
        print(f"        Distance from GND pad center: {dist:.3f}mm")

        if dist < 0.6:
            print(f"        *** WARNING: Via too close to GND pad! ***")
            print(f"        GND pad center: ({gnd_pad_x:.2f}, {gnd_pad_y:.2f})")
            print(f"        GND pad extends ~0.45mm to each side")
            print(f"        Required clearance: {rp.clearance_radius:.3f}mm")

# Check GND route
print("\nGND route analysis:")
if 'GND' in result.routes:
    gnd_route = result.routes['GND']
    print(f"  Segments: {len(gnd_route.segments)}")
    print(f"  Vias: {len(gnd_route.vias)}")

    for i, seg in enumerate(gnd_route.segments):
        print(f"    Segment {i}: ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f}) on {seg.layer}")

    for i, via in enumerate(gnd_route.vias):
        print(f"    Via {i}: ({via.position[0]:.2f}, {via.position[1]:.2f})")
else:
    print("  GND not routed!")

print("=" * 70)
