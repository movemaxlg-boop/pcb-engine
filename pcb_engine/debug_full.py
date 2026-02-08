"""Debug full BBL flow with simple circuit - more details"""
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
print("DEBUG: Full BBL flow with detailed output")
print("=" * 60)

from pcb_engine import PCBEngine, EngineConfig

# Create engine
config = EngineConfig(
    board_name='debug_full',
    board_width=30.0,
    board_height=25.0,
    verbose=True
)
engine = PCBEngine(config)

# Step 1: Initialize and do placement
print("\n1. Running placement...")
from pcb_engine.pcb_engine import EngineState
engine.state = EngineState()
engine.state.parts_db = parts_db

# Do placement manually
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig

placement_config = PlacementConfig(
    board_width=30.0,
    board_height=25.0,
    algorithm='hybrid'
)
placement_engine = PlacementEngine(placement_config)
placement_result = placement_engine.place(parts_db, {})

engine.state.placement = placement_result.positions
print(f"   Placed: {engine.state.placement}")

# Step 2: Build escapes
print("\n2. Building escapes...")
escapes = engine._build_simple_escapes()
print(f"   Escapes keys: {list(escapes.keys())}")
for ref, comp_escapes in escapes.items():
    print(f"   {ref}: {list(comp_escapes.keys())}")
    for pin, esc in comp_escapes.items():
        print(f"      {pin}: start={esc.start}, end={esc.end}, net={esc.net}")

# Step 3: Set net order
engine.state.net_order = ['SIG']  # Only route SIG net

# Step 4: Routing
print("\n3. Running routing...")
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

routing_config = RoutingConfig(
    board_width=30.0,
    board_height=25.0,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2
)
routing_piston = RoutingPiston(routing_config)

# Get pins for SIG net
sig_pins = parts_db['nets']['SIG']['pins']
print(f"   SIG pins: {sig_pins}")

# Check what _get_escape_endpoints returns WITH escapes
routing_piston._placement = engine.state.placement
routing_piston._parts_db = parts_db
endpoints = routing_piston._get_escape_endpoints(sig_pins, escapes)
print(f"   Endpoints from escapes: {endpoints}")

# Now route
result = routing_piston.route(
    parts_db=parts_db,
    escapes=escapes,
    placement=engine.state.placement,
    net_order=['SIG']
)

print(f"\n4. Routing Results:")
print(f"   routed_count: {result.routed_count}")
print(f"   total_count: {result.total_count}")
print(f"   success: {result.success}")

if result.routed_count == 0 and result.routes.get('SIG'):
    route = result.routes['SIG']
    print(f"   SIG route error: {route.error if hasattr(route, 'error') else 'N/A'}")

print("=" * 60)
