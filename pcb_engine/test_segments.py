"""Debug the actual segments generated"""
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
print(f"  R1 center: (14.0, 14.5)")
print(f"  R1.2 pad (VDIV): (14.0 + 0.95 = 14.95, 14.5)")
print(f"  R2 center: (20.0, 14.5)")
print(f"  R2.1 pad (VDIV): (20.0 - 0.95 = 19.05, 14.5)")
print(f"  C1 center: (17.0, 14.5)")
print(f"  C1.1 pad (VDIV): (17.0 - 0.95 = 16.05, 14.5)")

rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

result = rp.route(parts_db=parts_db, escapes={}, placement=placement, net_order=['VDIV', 'GND'])

print(f"\n--- VDIV Route ---")
if 'VDIV' in result.routes:
    vdiv = result.routes['VDIV']
    print(f"Success: {vdiv.success}")
    print(f"Segments: {len(vdiv.segments)}")
    for i, seg in enumerate(vdiv.segments):
        print(f"  [{i}] ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f}) on {seg.layer}")
    print(f"Vias: {len(vdiv.vias)}")
    for i, via in enumerate(vdiv.vias):
        print(f"  [{i}] at ({via.position[0]:.2f}, {via.position[1]:.2f})")

print(f"\n--- GND Route ---")
if 'GND' in result.routes:
    gnd = result.routes['GND']
    print(f"Success: {gnd.success}")
    print(f"Segments: {len(gnd.segments)}")
    for i, seg in enumerate(gnd.segments):
        print(f"  [{i}] ({seg.start[0]:.2f}, {seg.start[1]:.2f}) -> ({seg.end[0]:.2f}, {seg.end[1]:.2f}) on {seg.layer}")
    print(f"Vias: {len(gnd.vias)}")
    for i, via in enumerate(gnd.vias):
        print(f"  [{i}] at ({via.position[0]:.2f}, {via.position[1]:.2f})")

# Check if segments connect to pads
print("\n--- GAP ANALYSIS ---")
vdiv_pads = [(14.95, 14.5), (19.05, 14.5), (16.05, 14.5)]  # R1.2, R2.1, C1.1
print("VDIV pads:", vdiv_pads)
if 'VDIV' in result.routes:
    seg_ends = []
    for seg in result.routes['VDIV'].segments:
        seg_ends.append(seg.start)
        seg_ends.append(seg.end)
    for via in result.routes['VDIV'].vias:
        seg_ends.append(via.position)

    for pad in vdiv_pads:
        closest = min(seg_ends, key=lambda p: abs(p[0]-pad[0]) + abs(p[1]-pad[1]))
        dist = abs(closest[0]-pad[0]) + abs(closest[1]-pad[1])
        print(f"  Pad {pad} - closest segment end: {closest}, distance: {dist:.3f}mm")
