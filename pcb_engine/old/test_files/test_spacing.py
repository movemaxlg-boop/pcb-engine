"""Check component spacing for via placement"""
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
result = pe.place(parts_db, {})
placement = result.positions

print('=' * 60)
print('Component Spacing Analysis')
print('=' * 60)

print('\nPlacement:')
for ref, pos in placement.items():
    print(f'  {ref}: ({pos[0]:.2f}, {pos[1]:.2f})')

# Calculate distances
c1_x = placement['C1'][0]
r2_x = placement['R2'][0]
r1_x = placement['R1'][0]

c1_gnd_x = c1_x + 0.95  # C1 GND pad
c1_vdiv_x = c1_x - 0.95  # C1 VDIV pad
r2_vdiv_x = r2_x - 0.95  # R2 VDIV pad
r2_gnd_x = r2_x + 0.95  # R2 GND pad
r1_vdiv_x = r1_x + 0.95  # R1 VDIV pad

print()
print('Pad positions:')
print(f'  R1 VDIV pad: {r1_vdiv_x:.2f}')
print(f'  C1 VDIV pad: {c1_vdiv_x:.2f}')
print(f'  C1 GND pad: {c1_gnd_x:.2f}')
print(f'  R2 VDIV pad: {r2_vdiv_x:.2f}')
print(f'  R2 GND pad: {r2_gnd_x:.2f}')

print()
print('Gap analysis (center to center):')
print(f'  C1.GND to R2.VDIV: {r2_vdiv_x - c1_gnd_x:.2f}mm')

pad_half = 0.45  # 0.9mm / 2
via_diameter = 0.8
via_clearance = 0.2
via_total_radius = via_diameter / 2 + via_clearance

print()
print(f'Pad half-size: {pad_half}mm')
print(f'Via diameter: {via_diameter}mm')
print(f'Via clearance: {via_clearance}mm')
print(f'Via total radius (center to edge + clearance): {via_total_radius}mm')
print()
print('Space needed for via between C1.GND and R2.VDIV:')
print(f'  From C1.GND edge: {pad_half + via_total_radius:.2f}mm')
print(f'  To R2.VDIV edge: {pad_half + via_total_radius:.2f}mm')
print(f'  Total: {2 * (pad_half + via_total_radius):.2f}mm')
print()
actual_gap = r2_vdiv_x - c1_gnd_x
space_needed = 2 * (pad_half + via_total_radius)
print(f'Actual gap: {actual_gap:.2f}mm')
print(f'Space needed: {space_needed:.2f}mm')
print(f'Difference: {actual_gap - space_needed:.2f}mm')

if actual_gap < space_needed:
    print()
    print('*** PROBLEM: Not enough space for via between components! ***')
    print()
    print('Solutions:')
    print('1. Increase component spacing in placement')
    print('2. Route around (use F.Cu only path that goes around)')
    print('3. Use smaller via (reduce via_diameter)')

print('=' * 60)
