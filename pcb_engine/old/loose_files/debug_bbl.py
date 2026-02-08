"""Debug BBL run - simpler version"""
import sys
import os
# Add parent directory to path for package imports
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
print("DEBUG: Testing BBL with simple circuit")
print("=" * 60)

from pcb_engine import PCBEngine, EngineConfig

config = EngineConfig(
    board_name='debug_bbl',
    board_width=30.0,
    board_height=25.0,
    verbose=True
)

print("\n1. Creating PCB Engine...")
engine = PCBEngine(config)

print("\n2. Running BBL...")
result = engine.run_bbl(parts_db)

print(f"\n3. Results:")
print(f"   success: {result.success}")
print(f"   routing: {result.routed_count}/{result.total_nets}")
print(f"   drc_passed: {result.drc_passed}")
print(f"   kicad_drc_passed: {result.kicad_drc_passed}")

if result.output_file:
    print(f"   output: {result.output_file}")

print("=" * 60)
