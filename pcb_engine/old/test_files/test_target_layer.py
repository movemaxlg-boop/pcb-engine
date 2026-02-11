"""Check what's at the target location on each layer"""
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
print("Target Layer Check")
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

rp._placement = placement.positions
rp._parts_db = parts_db
rp._register_components(placement.positions, parts_db)

# GND target (C1.GND)
c1 = placement.positions['C1']
c1_x = c1[0] if isinstance(c1, (list, tuple)) else c1.x
target = (c1_x + 0.95, 14.5)
target_col = int(target[0] / 0.1)
target_row = int(target[1] / 0.1)

print(f"\nTarget (C1.GND): {target}")
print(f"Target grid: row={target_row}, col={target_col}")

print("\nGrid values at target:")
print(f"  F.Cu: {rp.fcu_grid[target_row][target_col]!r}")
print(f"  B.Cu: {rp.bcu_grid[target_row][target_col]!r}")

print("\n" + "=" * 40)
print("PROBLEM:")
print("=" * 40)
print("The Lee algorithm accepts reaching (target_row, target_col)")
print("on ANY layer, even if the target pad is only on F.Cu.")
print()
print("When routing from R2.GND on F.Cu:")
print("1. It can't go directly on F.Cu (blocked by VDIV pad)")
print("2. It takes a via to B.Cu at (20.90, 14.50)")
print("3. It routes on B.Cu to (17.90, 14.50) [near target]")
print("4. It thinks it reached the target at row 145, col 179 on B.Cu!")
print("5. But the actual C1.GND pad is on F.Cu, not B.Cu!")
print()
print("FIX NEEDED:")
print("The target detection must consider the layer.")
print("For SMD pads (F.Cu only), the route must END on F.Cu.")

print("=" * 70)
