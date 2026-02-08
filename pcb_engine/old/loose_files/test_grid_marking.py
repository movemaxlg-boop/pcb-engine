"""Test that grid marking works correctly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple 2-resistor circuit
parts_db = {
    'parts': {
        'R1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'SIG', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'R2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'SIG', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VCC': {'pins': [('R1', '1')]},
        'SIG': {'pins': [('R1', '2'), ('R2', '1')]},
        'GND': {'pins': [('R2', '2')]},
    }
}

placement = {'R1': (5.0, 22.0), 'R2': (8.0, 22.0)}

print("=" * 60)
print("TEST: Grid Marking")
print("=" * 60)

from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

config = RoutingConfig(
    board_width=30.0,
    board_height=25.0,
    algorithm='lee',
    grid_size=0.1,  # Same as default
    trace_width=0.25,
    clearance=0.2
)

piston = RoutingPiston(config)
piston._initialize_grids()
piston._register_components(placement, parts_db)

# Check the cells around the escape endpoints
# R1 pin 2 (SIG): component at (5, 22), offset (0.95, 0) -> pad at (5.95, 22)
# R2 pin 1 (SIG): component at (8, 22), offset (-0.95, 0) -> pad at (7.05, 22)

# The escape endpoints from _build_simple_escapes go 1mm further
# R1.2 escape ends at (5.95, 23) - escapes upward
# R2.1 escape ends at (7.05, 23) - escapes upward

for name, pos in [
    ("R1.2 pad", (5.95, 22.0)),
    ("R1.2 escape end", (5.95, 23.0)),
    ("R2.1 pad", (7.05, 22.0)),
    ("R2.1 escape end", (7.05, 23.0)),
    ("Between components", (6.5, 22.0)),
    ("Between escape ends", (6.5, 23.0)),
]:
    col = int(pos[0] / config.grid_size)
    row = int(pos[1] / config.grid_size)
    cell_value = piston.fcu_grid[row][col] if row < len(piston.fcu_grid) and col < len(piston.fcu_grid[0]) else 'OOB'
    print(f"{name:25} at ({pos[0]:5.2f}, {pos[1]:5.2f}) -> grid[{row:3d}][{col:3d}] = {cell_value}")

print("=" * 60)
