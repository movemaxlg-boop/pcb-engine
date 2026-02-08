"""Test medium complexity PCB - 3 components, 2 nets"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple voltage divider circuit
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
print("MEDIUM PCB TEST: Voltage Divider with Filter Cap")
print("=" * 60)

routeable_nets = [name for name, info in parts_db['nets'].items()
                  if len(info['pins']) >= 2]
print(f"Components: {len(parts_db['parts'])}")
print(f"Routeable nets: {routeable_nets}")

# Placement
print("\n1. Placement...")
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig

pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
placement = pe.place(parts_db, {})
print(f"   OK - {len(placement.positions)} components placed")

# Routing
print("\n2. Routing...")
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',  # Use simpler algorithm
    trace_width=0.25,
    clearance=0.15,  # Match KiCad default
    via_diameter=0.6,  # Smaller via for tighter layouts
    via_drill=0.3
))

# Disable GND pour for testing - use actual traces instead
# (Pour requires zone filling which isn't done automatically)
from pcb_engine.output_piston import OutputConfig
output_config = OutputConfig(generate_gnd_pour=False)

# Route GND first (simpler 2-pin net), then VDIV can go around
net_order = ['GND', 'VDIV']
result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=placement.positions,
    net_order=net_order
)

print(f"   Routed: {result.routed_count}/{result.total_count}")

for net, route in result.routes.items():
    status = "OK" if route.success else "FAIL"
    print(f"   {net}: {status} ({len(route.segments)} segments)")

# Output
print("\n3. Output...")
from pcb_engine.output_piston import OutputPiston

op = OutputPiston(output_config)
outfile = op.generate(
    parts_db=parts_db,
    placement=placement.positions,
    routes=result.routes
)
print(f"   Generated: {outfile}")

# KiCad DRC
print("\n4. KiCad DRC...")
import subprocess
import json

if outfile:
    drc_file = outfile.replace('.kicad_pcb', '_drc.json')
    try:
        subprocess.run([
            'kicad-cli', 'pcb', 'drc',
            '--output', drc_file,
            '--format', 'json',
            '--severity-all',
            outfile
        ], capture_output=True, timeout=30)

        if os.path.exists(drc_file):
            with open(drc_file) as f:
                drc = json.load(f)
            v = len(drc.get('violations', []))
            u = len(drc.get('unconnected_items', []))
            print(f"   Violations: {v}, Unconnected: {u}")
            print(f"   DRC: {'PASS' if v == 0 and u == 0 else 'FAIL'}")
    except Exception as e:
        print(f"   DRC skipped: {e}")

print("\n" + "=" * 60)
if result.routed_count == len(routeable_nets):
    print("SUCCESS: All nets routed!")
    sys.exit(0)
else:
    print(f"PARTIAL: {result.routed_count}/{len(routeable_nets)} nets")
    sys.exit(1)
