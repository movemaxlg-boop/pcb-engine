"""Trace the actual path the router is taking for GND"""
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
print("Path Trace for GND")
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

c1_gnd = (c1_x + 0.95, 14.5)  # C1 GND pad
r2_vdiv = (r2_x - 0.95, 14.5)  # R2 VDIV pad
r2_gnd = (r2_x + 0.95, 14.5)  # R2 GND pad

print(f"\nR2 position: ({r2_x:.2f}, 14.50)")
print(f"R2.GND pad: ({r2_gnd[0]:.2f}, {r2_gnd[1]:.2f})")
print(f"R2.VDIV pad: ({r2_vdiv[0]:.2f}, {r2_vdiv[1]:.2f})")
print(f"C1.GND pad: ({c1_gnd[0]:.2f}, {c1_gnd[1]:.2f})")

# Routing
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

# Check what's in the grid on F.Cu at row 145 (Y=14.5)
row = 145
print(f"\nF.Cu grid at row {row} (Y=14.5):")
print("Cols 175 to 215 (X=17.5 to 21.5):")

# Group by net
current_net = None
start_col = None
for col in range(175, 216):
    cell = rp.fcu_grid[row][col]
    if cell != current_net:
        if current_net is not None:
            x_start = start_col * 0.1
            x_end = (col - 1) * 0.1
            print(f"  {current_net}: cols {start_col}-{col-1} (x={x_start:.1f}-{x_end:.1f})")
        current_net = cell
        start_col = col

# Print last segment
if current_net is not None:
    x_start = start_col * 0.1
    x_end = 215 * 0.1
    print(f"  {current_net}: cols {start_col}-215 (x={x_start:.1f}-{x_end:.1f})")

# Now let's see: from R2.GND at col 209 to C1.GND at col 179
# The path has to cross through cols 185-195 (VDIV) - not possible on F.Cu!
# So it must go around

print()
print("=" * 40)
print("Analysis:")
print("=" * 40)
print(f"R2.GND at col 209 (x=20.9)")
print(f"C1.GND at col 179 (x=17.9)")
print(f"VDIV pad blocks cols 185-195 (x=18.5-19.5)")
print()
print("Direct path on F.Cu is IMPOSSIBLE!")
print("Route must go around (above/below) or use via to B.Cu")

print("=" * 60)
