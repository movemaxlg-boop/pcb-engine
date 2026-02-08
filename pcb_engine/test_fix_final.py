"""Final test to verify the routing fix works end-to-end"""
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

print("=" * 60)
print("FINAL TEST: Verifying Routing Fixes")
print("=" * 60)

# Test 1: get_pins works for 'pins' format
print("\n1. Testing get_pins:")
from pcb_engine.common_types import get_pins
for ref, part in parts_db['parts'].items():
    pins = get_pins(part)
    print(f"   {ref}: {len(pins)} pins - OK")

# Test 2: PlacementEngine works
print("\n2. Testing PlacementEngine:")
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
print(f"   Components: {len(result.positions)} - OK")

# Test 3: Grid marking with get_pins
print("\n3. Testing grid marking:")
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig
rp = RoutingPiston(RoutingConfig(board_width=30, board_height=25))
rp._initialize_grids()
rp._register_components(result.positions, parts_db)

# Check that pads are marked with net names, not __COMPONENT__
# R1.2 is at R1 position + offset (0.95, 0)
r1_pos = result.positions['R1']
r1_x = r1_pos[0] if isinstance(r1_pos, (list, tuple)) else r1_pos.x
r1_y = r1_pos[1] if isinstance(r1_pos, (list, tuple)) else r1_pos.y
pad_x = r1_x + 0.95  # R1.2 offset
pad_y = r1_y
pad_col = int(pad_x / 0.1)
pad_row = int(pad_y / 0.1)
cell = rp.fcu_grid[pad_row][pad_col]
if cell == 'SIG':
    print(f"   Pad at ({pad_x:.2f}, {pad_y:.2f}) marked with net: {cell} - OK")
else:
    print(f"   ERROR: Pad at ({pad_x:.2f}, {pad_y:.2f}) marked as: {cell}")
    sys.exit(1)

# Test 4: Escape endpoints found
print("\n4. Testing escape endpoint extraction:")
sig_pins = [('R1', '2'), ('R2', '1')]
rp._placement = result.positions
rp._parts_db = parts_db
endpoints = rp._get_escape_endpoints(sig_pins, {})
print(f"   Endpoints: {endpoints}")
if len(endpoints) == 2:
    print(f"   Found 2 endpoints - OK")
else:
    print(f"   ERROR: Expected 2 endpoints")
    sys.exit(1)

# Test 5: Routing works
print("\n5. Testing routing:")
routing_result = rp.route(
    parts_db=parts_db,
    escapes={},
    placement=result.positions,
    net_order=['SIG']
)
print(f"   Routed: {routing_result.routed_count}/{routing_result.total_count}")
if routing_result.routed_count == 1:
    print(f"   Routing successful - OK")
else:
    print(f"   ERROR: Routing failed")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
sys.exit(0)
