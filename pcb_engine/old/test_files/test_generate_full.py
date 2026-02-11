"""Debug full generate with print statements"""
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
from pcb_engine.output_piston import OutputPiston

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

# Generate output
op = OutputPiston()
outfile = op.generate(
    parts_db=parts_db,
    placement=placement.positions,
    routes=result.routes
)

print(f"\nGenerated: {outfile}")

# Check the output file for GND segments
if hasattr(outfile, 'files_generated'):
    pcb_file = outfile.files_generated[0]
else:
    pcb_file = outfile

print(f"\nChecking {pcb_file} for segments...")

with open(pcb_file, 'r') as f:
    content = f.read()

# Count segments by net
import re
segments = re.findall(r'\(segment.*?\(net (\d+)\).*?\)', content, re.DOTALL)
print(f"Segment counts by net ID:")
from collections import Counter
for net_id, count in Counter(segments).items():
    print(f"  net {net_id}: {count} segments")

# Find if there are any net 3 (GND) segments
if '(net 3)' in content:
    print("\nGND segments (net 3) found in file!")
else:
    print("\nNO GND segments (net 3) in file!")
