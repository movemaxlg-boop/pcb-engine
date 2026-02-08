"""Test escape endpoints extraction"""
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
print("TEST: Escape Endpoints Extraction")
print("=" * 60)

# First do placement
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
placement_engine = PlacementEngine(PlacementConfig(board_width=30.0, board_height=25.0, algorithm='hybrid'))
placement_result = placement_engine.place(parts_db, {})
placement = placement_result.positions
print(f"Placement: {placement}")

# Build escapes using the same code as PCBEngine._build_simple_escapes
from pcb_engine.pcb_engine import EngineConfig, PCBEngine
from pcb_engine.common_types import get_xy, get_pins

config = EngineConfig(board_width=30.0, board_height=25.0)
engine = PCBEngine(config)

# Manually set state to avoid running full init
from pcb_engine.pcb_engine import EngineState
engine.state = EngineState()
engine.state.parts_db = parts_db
engine.state.placement = placement

# Build escapes
escapes = engine._build_simple_escapes()

print(f"\nEscapes:")
for ref, comp_escapes in escapes.items():
    for pin, esc in comp_escapes.items():
        print(f"  {ref}.{pin}: start={esc.start}, end={esc.end}, endpoint={esc.endpoint}, net={esc.net}")

# Test _get_escape_endpoints
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

routing_config = RoutingConfig(board_width=30.0, board_height=25.0)
piston = RoutingPiston(routing_config)
piston._placement = placement
piston._parts_db = parts_db

sig_pins = [('R1', '2'), ('R2', '1')]

print(f"\nSIG pins: {sig_pins}")

# Test with empty escapes (fallback to placement + parts_db)
endpoints_no_escapes = piston._get_escape_endpoints(sig_pins, {})
print(f"Endpoints (no escapes): {endpoints_no_escapes}")

# Test with escapes
endpoints_with_escapes = piston._get_escape_endpoints(sig_pins, escapes)
print(f"Endpoints (with escapes): {endpoints_with_escapes}")

print("=" * 60)
