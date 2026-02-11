"""Debug: Via clearance only checks one layer, not both!"""
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
print("DEBUG: Via checks only one layer, but via spans BOTH layers!")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

# Setup routing (just register components)
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))

rp._placement = placement
rp._parts_db = parts_db
rp._register_components(placement, parts_db)

# Check via position
via_row = 145
via_col = 185  # x=18.5

print(f"\nVia position: row={via_row}, col={via_col} (x=18.5)")
print()

# Check both layers
fcu_clear = rp._is_cell_clear_for_net(rp.fcu_grid, via_row, via_col, 'VDIV')
bcu_clear = rp._is_cell_clear_for_net(rp.bcu_grid, via_row, via_col, 'VDIV')

print(f"F.Cu clearance check: {fcu_clear}")
print(f"B.Cu clearance check: {bcu_clear}")
print()

# Show what's in each layer
print("F.Cu around via position:")
cc = rp.clearance_cells
for dc in range(-cc, cc + 1):
    c = via_col + dc
    if rp._in_bounds(via_row, c):
        cell = rp.fcu_grid[via_row][c]
        print(f"  col {c}: {cell!r}")

print()
print("B.Cu around via position:")
for dc in range(-cc, cc + 1):
    c = via_col + dc
    if rp._in_bounds(via_row, c):
        cell = rp.bcu_grid[via_row][c]
        print(f"  col {c}: {cell!r}")

print()
print("=" * 50)
print("PROBLEM: Lee algorithm line 446 only checks 'other_grid':")
print("  if self._is_cell_clear_for_net(other_grid, row, col, net_name):")
print()
print("When transitioning from F.Cu to B.Cu, it only checks B.Cu!")
print("But the via ALSO touches F.Cu where the GND pad is!")
print("=" * 50)

# What the fix should be:
print()
print("FIX: Check BOTH layers for via placement:")
print(f"  F.Cu check: {fcu_clear}")
print(f"  B.Cu check: {bcu_clear}")
print(f"  BOTH must be True: {fcu_clear and bcu_clear}")
print("=" * 70)
