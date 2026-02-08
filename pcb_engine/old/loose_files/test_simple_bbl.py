"""Simple BBL test to verify fixes"""
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
print("SIMPLE BBL TEST")
print("=" * 60)

from pcb_engine import PCBEngine, EngineConfig

config = EngineConfig(
    board_name='simple_test',
    board_width=30.0,
    board_height=25.0,
    verbose=False  # Quiet mode
)

print("Creating engine...")
engine = PCBEngine(config)

print("Running BBL (this may take a minute)...")
import time
start = time.time()
result = engine.run_bbl(parts_db)
elapsed = time.time() - start

print(f"\nResults (took {elapsed:.1f}s):")
print(f"  Success: {result.success}")
print(f"  Routing: {result.routed_count}/{result.total_nets}")
print(f"  Internal DRC: {'PASS' if result.drc_passed else 'FAIL'}")
print(f"  KiCad DRC: {'PASS' if result.kicad_drc_passed else 'FAIL'}")

if result.output_file:
    print(f"  Output: {result.output_file}")

print("=" * 60)
sys.exit(0 if result.success else 1)
