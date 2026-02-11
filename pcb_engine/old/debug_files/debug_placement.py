"""Debug placement piston issue"""
import sys
sys.path.insert(0, '.')

# Test parts database with 'pins' format (same as test_via_fix.py)
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
print("DEBUG: Testing PlacementEngine")
print("=" * 60)

# Test 1: Check common_types.get_pins
print("\n1. Testing get_pins from common_types:")
from common_types import get_pins
for ref, part in parts_db['parts'].items():
    pins = get_pins(part)
    print(f"   {ref}: {len(pins)} pins -> {pins}")

# Test 2: Check PlacementEngine directly
print("\n2. Testing PlacementEngine.place():")
from placement_engine import PlacementEngine, PlacementConfig

config = PlacementConfig(
    board_width=30.0,
    board_height=25.0,
    algorithm='hybrid'
)
engine = PlacementEngine(config)

result = engine.place(parts_db, {})

print(f"   Components loaded: {len(engine.components)}")
for ref, comp in engine.components.items():
    print(f"     {ref}: ({comp.x:.2f}, {comp.y:.2f})")

print(f"   Result positions: {len(result.positions)}")
for ref, pos in result.positions.items():
    print(f"     {ref}: {pos}")

print(f"   Success: {result.success}")
print("=" * 60)
