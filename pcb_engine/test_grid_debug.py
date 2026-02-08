"""Debug grid marking issue"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

print("=" * 60)
print("DEBUG: Grid Marking Issue")
print("=" * 60)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig
from pcb_engine.common_types import get_pins

# Get placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions
print(f"Placement: {placement}")

# Setup routing piston
rp = RoutingPiston(RoutingConfig(board_width=30, board_height=25, grid_size=0.1))
rp._initialize_grids()

# Now trace through _register_components manually
print("\n--- Tracing _register_components ---")
parts = parts_db.get('parts', {})

for ref, pos in placement.items():
    part = parts.get(ref, {})

    pos_x = pos[0] if isinstance(pos, (list, tuple)) else 0
    pos_y = pos[1] if isinstance(pos, (list, tuple)) else 0
    print(f"\nComponent {ref} at ({pos_x}, {pos_y})")

    # Get pins using get_pins
    all_pads = get_pins(part)
    print(f"  Pads from get_pins: {len(all_pads)}")

    # Build pin_nets mapping
    pin_nets = {}
    for pin in all_pads:
        pin_nets[pin.get('number', '')] = pin.get('net', '')
    print(f"  pin_nets: {pin_nets}")

    # For each pad
    for pin in all_pads:
        pin_num = pin.get('number', '')
        net = pin_nets.get(pin_num, '')

        offset = pin.get('offset', None)
        if not offset:
            physical = pin.get('physical', {})
            offset = (physical.get('offset_x', 0), physical.get('offset_y', 0)) if physical else (0, 0)

        pad_x = pos_x + offset[0]
        pad_y = pos_y + offset[1]

        print(f"  Pin {pin_num}: offset={offset}, pad=({pad_x}, {pad_y}), net={net}")

        # Grid cell
        pad_col = int((pad_x - 0) / 0.1)  # origin_x = 0
        pad_row = int((pad_y - 0) / 0.1)  # origin_y = 0
        print(f"    Grid: [{pad_row}][{pad_col}]")

        # Check what's there before
        current = rp.fcu_grid[pad_row][pad_col] if rp._in_bounds(pad_row, pad_col) else 'OOB'
        print(f"    Current value: {current}")

# Now actually call _register_components
print("\n--- Calling _register_components ---")
rp._register_components(placement, parts_db)

# Check the cells
for name, (x, y) in [("R1.2 pad", (5.95, 22.0)), ("R2.1 pad", (7.05, 22.0))]:
    col = int(x / 0.1)
    row = int(y / 0.1)
    cell = rp.fcu_grid[row][col] if rp._in_bounds(row, col) else 'OOB'
    print(f"{name}: grid[{row}][{col}] = {cell}")

print("=" * 60)
