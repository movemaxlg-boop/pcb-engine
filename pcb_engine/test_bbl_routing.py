"""Test BBL routing flow directly"""
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
print("TEST: BBL Routing Flow")
print("=" * 60)

from pcb_engine import PCBEngine, EngineConfig
from pcb_engine.pcb_engine import EngineState
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Create engine
config = EngineConfig(
    board_name='test_bbl_routing',
    board_width=30.0,
    board_height=25.0
)
engine = PCBEngine(config)

# Initialize state
engine.state = EngineState()
engine.state.parts_db = parts_db

# Do placement
print("\n1. Placement:")
placement_engine = PlacementEngine(PlacementConfig(board_width=30.0, board_height=25.0, algorithm='hybrid'))
placement_result = placement_engine.place(parts_db, {})
engine.state.placement = placement_result.positions
print(f"   Placement: {engine.state.placement}")

# Build net order (from order piston)
engine.state.net_order = ['SIG']  # Only SIG has 2+ pins

# Build escapes
print("\n2. Build escapes:")
escapes = engine._build_simple_escapes()
print(f"   Escapes: {list(escapes.keys())}")
for ref, comp_esc in escapes.items():
    for pin, esc in comp_esc.items():
        print(f"   {ref}.{pin}: end={esc.end}")

# Create routing piston like BBL does
print("\n3. Routing:")
routing_config = RoutingConfig(
    algorithm='lee',
    board_width=30.0,
    board_height=25.0,
    grid_size=0.1,
    trace_width=0.25,
    clearance=0.2
)
routing_piston = RoutingPiston(routing_config)

# Test _get_escape_endpoints with actual escapes
sig_pins = parts_db['nets']['SIG']['pins']
print(f"   SIG pins: {sig_pins}")

# Set internal state for fallback
routing_piston._placement = engine.state.placement
routing_piston._parts_db = parts_db

endpoints = routing_piston._get_escape_endpoints(sig_pins, escapes)
print(f"   Endpoints: {endpoints}")

# Do routing
result = routing_piston.route(
    parts_db=parts_db,
    escapes=escapes,
    placement=engine.state.placement,
    net_order=['SIG']
)

print(f"\n4. Results:")
print(f"   routed_count: {result.routed_count}")
print(f"   total_count: {result.total_count}")
print(f"   success: {result.success}")

if 'SIG' in result.routes:
    route = result.routes['SIG']
    print(f"   SIG segments: {len(route.segments)}")
    if route.segments:
        print(f"   First segment: {route.segments[0]}")

print("=" * 60)
