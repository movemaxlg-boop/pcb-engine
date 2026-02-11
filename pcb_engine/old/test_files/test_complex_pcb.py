"""Test complex PCB with multiple components and nets"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LDO voltage regulator circuit with input/output capacitors
# This is a realistic power supply subcircuit
complex_parts_db = {
    'parts': {
        # LDO Regulator (SOT-223)
        'U1': {
            'footprint': 'SOT-223',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-2.3, 0), 'size': (1.0, 1.8)},
                {'number': '2', 'net': 'GND', 'offset': (0, 0), 'size': (1.0, 1.8)},
                {'number': '3', 'net': 'VOUT', 'offset': (2.3, 0), 'size': (1.0, 1.8)},
                {'number': '4', 'net': 'VOUT', 'offset': (0, 3.25), 'size': (3.0, 1.5)},  # Tab
            ]
        },
        # Input capacitor (0805)
        'C1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        # Output capacitor (0805)
        'C2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VOUT', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        # LED with current limiting resistor
        'R1': {
            'footprint': '0603',
            'pins': [
                {'number': '1', 'net': 'VOUT', 'offset': (-0.75, 0), 'size': (0.6, 0.9)},
                {'number': '2', 'net': 'LED_A', 'offset': (0.75, 0), 'size': (0.6, 0.9)},
            ]
        },
        # LED (0805 LED package)
        'D1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'LED_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},  # Anode
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},     # Cathode
            ]
        },
        # Input protection TVS diode
        'D2': {
            'footprint': 'SOD-123',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-1.35, 0), 'size': (0.9, 1.2)},
                {'number': '2', 'net': 'GND', 'offset': (1.35, 0), 'size': (0.9, 1.2)},
            ]
        },
    },
    'nets': {
        'VIN': {'pins': [('U1', '1'), ('C1', '1'), ('D2', '1')]},
        'VOUT': {'pins': [('U1', '3'), ('U1', '4'), ('C2', '1'), ('R1', '1')]},
        'GND': {'pins': [('U1', '2'), ('C1', '2'), ('C2', '2'), ('D1', '2'), ('D2', '2')]},
        'LED_A': {'pins': [('R1', '2'), ('D1', '1')]},
    }
}

print("=" * 70)
print("COMPLEX PCB TEST: LDO Power Supply with LED Indicator")
print("=" * 70)
print(f"Components: {len(complex_parts_db['parts'])}")
print(f"Nets: {len(complex_parts_db['nets'])}")

# Count routeable nets (2+ pins)
routeable = sum(1 for net, info in complex_parts_db['nets'].items()
                if len(info['pins']) >= 2)
print(f"Routeable nets: {routeable}")

from pcb_engine import PCBEngine, EngineConfig

config = EngineConfig(
    board_name='ldo_power_supply',
    board_width=40.0,
    board_height=30.0,
    verbose=False
)

print("\nCreating PCB Engine...")
engine = PCBEngine(config)

print("Running BBL (this may take a moment)...")
import time
start = time.time()
result = engine.run_bbl(complex_parts_db)
elapsed = time.time() - start

print(f"\n{'=' * 70}")
print("RESULTS")
print('=' * 70)
print(f"Time: {elapsed:.1f}s")
print(f"Success: {result.success}")
print(f"Routing: {result.routed_count}/{result.total_nets}")
print(f"Internal DRC: {'PASS' if result.drc_passed else 'FAIL'}")
print(f"KiCad DRC: {'PASS' if result.kicad_drc_passed else 'FAIL'}")

if hasattr(result, 'output_files') and result.output_files:
    print(f"Output files: {len(result.output_files)}")
    for f in result.output_files[:3]:
        print(f"  - {f}")

# Check routing details
if hasattr(result, 'errors') and result.errors:
    print(f"\nErrors ({len(result.errors)}):")
    for e in result.errors[:5]:
        print(f"  - {e}")

if hasattr(result, 'warnings') and result.warnings:
    print(f"\nWarnings ({len(result.warnings)}):")
    for w in result.warnings[:5]:
        print(f"  - {w}")

print('=' * 70)

# Exit with appropriate code
if result.success and result.kicad_drc_passed:
    print("TEST PASSED!")
    sys.exit(0)
else:
    print("TEST FAILED!")
    sys.exit(1)
