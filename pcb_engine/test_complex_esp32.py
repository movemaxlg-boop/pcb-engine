#!/usr/bin/env python3
"""
Comprehensive PCB Engine Test - ESP32 Sensor Module
=====================================================

This test exercises the FULL PCB Engine with BBL pipeline:
- 8 components (ESP32-WROOM-32, LDO, sensors, passives)
- Multiple nets (power, I2C, GPIO, GND)
- Tests all core pistons: Parts, Order, Placement, Routing, DRC, Output
- Uses BBL with progress callbacks and escalation handling

Board: 50mm x 40mm, 2-layer
Target: Zero DRC errors, zero warnings
"""

import os
import sys
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_esp32_sensor_parts_db():
    """
    Create a realistic ESP32 sensor module parts database.

    Circuit:
    - ESP32 module representation (simplified for routing test)
    - AMS1117-3.3 LDO (5V to 3.3V power)
    - BME280 sensor (I2C temperature/humidity/pressure)
    - 4x decoupling capacitors
    - 2x I2C pull-up resistors
    - 2x LEDs with resistors

    This tests:
    - Multiple component types and sizes (10 components)
    - Power distribution (5V, 3V3, GND)
    - Signal routing (I2C bus)
    - Multiple nets (7 unique nets)
    - Ground pour connectivity
    """

    parts_db = {
        'parts': {
            # ESP32 Module - represented as SOIC-8 equivalent for simpler routing
            # (Real ESP32 has 38 pins but we simplify to key pins for testing)
            'U1': {
                'value': 'ESP32',
                'footprint': 'SOIC-8',
                'description': 'ESP32 Module (simplified)',
                'pins': [
                    # Left side (pin 1-4 from top)
                    {'number': '1', 'net': '3V3', 'name': 'VCC', 'offset': (-2.7, -1.905), 'size': (0.6, 1.78)},
                    {'number': '2', 'net': 'GND', 'name': 'GND', 'offset': (-2.7, -0.635), 'size': (0.6, 1.78)},
                    {'number': '3', 'net': 'SDA', 'name': 'SDA', 'offset': (-2.7, 0.635), 'size': (0.6, 1.78)},
                    {'number': '4', 'net': 'SCL', 'name': 'SCL', 'offset': (-2.7, 1.905), 'size': (0.6, 1.78)},
                    # Right side (pin 5-8 from bottom)
                    {'number': '5', 'net': 'LED1', 'name': 'GPIO2', 'offset': (2.7, 1.905), 'size': (0.6, 1.78)},
                    {'number': '6', 'net': 'LED2', 'name': 'GPIO4', 'offset': (2.7, 0.635), 'size': (0.6, 1.78)},
                    {'number': '7', 'net': 'EN', 'name': 'EN', 'offset': (2.7, -0.635), 'size': (0.6, 1.78)},
                    {'number': '8', 'net': 'GND', 'name': 'GND2', 'offset': (2.7, -1.905), 'size': (0.6, 1.78)},
                ]
            },

            # AMS1117-3.3 LDO - SOT-223
            'U2': {
                'value': 'AMS1117',
                'footprint': 'SOT-223',
                'description': '3.3V LDO Regulator',
                'pins': [
                    {'number': '1', 'net': 'GND', 'name': 'GND', 'offset': (-2.3, -3.0), 'size': (1.0, 1.5)},
                    {'number': '2', 'net': '3V3', 'name': 'VOUT', 'offset': (0.0, -3.0), 'size': (1.0, 1.5)},
                    {'number': '3', 'net': '5V', 'name': 'VIN', 'offset': (2.3, -3.0), 'size': (1.0, 1.5)},
                    {'number': '4', 'net': '3V3', 'name': 'TAB', 'offset': (0.0, 3.0), 'size': (3.0, 2.0)},
                ]
            },

            # BME280 Sensor - simplified as SOIC-4 equivalent
            'U3': {
                'value': 'BME280',
                'footprint': 'SOT-23-5',
                'description': 'Temp/Humidity Sensor',
                'pins': [
                    {'number': '1', 'net': 'SDA', 'name': 'SDA', 'offset': (-0.95, -1.1), 'size': (0.6, 0.7)},
                    {'number': '2', 'net': 'GND', 'name': 'GND', 'offset': (-0.95, 0.0), 'size': (0.6, 0.7)},
                    {'number': '3', 'net': 'SCL', 'name': 'SCL', 'offset': (-0.95, 1.1), 'size': (0.6, 0.7)},
                    {'number': '4', 'net': '3V3', 'name': 'VCC', 'offset': (0.95, 0.55), 'size': (0.6, 0.7)},
                    {'number': '5', 'net': 'GND', 'name': 'GND2', 'offset': (0.95, -0.55), 'size': (0.6, 0.7)},
                ]
            },

            # Input capacitor for LDO (5V side)
            'C1': {
                'value': '10uF',
                'footprint': '0805',
                'description': 'LDO Input Cap',
                'pins': [
                    {'number': '1', 'net': '5V', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },

            # Output capacitor for LDO (3.3V side)
            'C2': {
                'value': '10uF',
                'footprint': '0805',
                'description': 'LDO Output Cap',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },

            # ESP32 decoupling capacitor
            'C3': {
                'value': '100nF',
                'footprint': '0603',
                'description': 'ESP32 Decoupling',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'GND', 'offset': (0.775, 0), 'size': (0.75, 0.9)},
                ]
            },

            # I2C SDA pull-up resistor
            'R1': {
                'value': '4.7k',
                'footprint': '0603',
                'description': 'I2C SDA Pull-up',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'SDA', 'offset': (0.775, 0), 'size': (0.75, 0.9)},
                ]
            },

            # I2C SCL pull-up resistor
            'R2': {
                'value': '4.7k',
                'footprint': '0603',
                'description': 'I2C SCL Pull-up',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'SCL', 'offset': (0.775, 0), 'size': (0.75, 0.9)},
                ]
            },

            # LED1 current limiting resistor
            'R3': {
                'value': '330R',
                'footprint': '0603',
                'description': 'LED1 Resistor',
                'pins': [
                    {'number': '1', 'net': 'LED1', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'LED1_A', 'offset': (0.775, 0), 'size': (0.75, 0.9)},
                ]
            },

            # LED1
            'D1': {
                'value': 'GREEN',
                'footprint': '0805',
                'description': 'Status LED 1',
                'pins': [
                    {'number': '1', 'net': 'LED1_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },

            # LED2 current limiting resistor
            'R4': {
                'value': '330R',
                'footprint': '0603',
                'description': 'LED2 Resistor',
                'pins': [
                    {'number': '1', 'net': 'LED2', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'LED2_A', 'offset': (0.775, 0), 'size': (0.75, 0.9)},
                ]
            },

            # LED2
            'D2': {
                'value': 'RED',
                'footprint': '0805',
                'description': 'Status LED 2',
                'pins': [
                    {'number': '1', 'net': 'LED2_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
        },

        'nets': {
            # Power nets
            '5V': {'pins': [('U2', '3'), ('C1', '1')]},
            '3V3': {'pins': [('U1', '1'), ('U2', '2'), ('U2', '4'), ('U3', '4'),
                            ('C2', '1'), ('C3', '1'), ('R1', '1'), ('R2', '1')]},
            'GND': {'pins': [('U1', '2'), ('U1', '8'), ('U2', '1'), ('U3', '2'), ('U3', '5'),
                            ('C1', '2'), ('C2', '2'), ('C3', '2'), ('D1', '2'), ('D2', '2')]},

            # I2C Signal nets
            'SDA': {'pins': [('U1', '3'), ('U3', '1'), ('R1', '2')]},
            'SCL': {'pins': [('U1', '4'), ('U3', '3'), ('R2', '2')]},

            # LED nets
            'LED1': {'pins': [('U1', '5'), ('R3', '1')]},
            'LED1_A': {'pins': [('R3', '2'), ('D1', '1')]},
            'LED2': {'pins': [('U1', '6'), ('R4', '1')]},
            'LED2_A': {'pins': [('R4', '2'), ('D2', '1')]},

            # Control nets (single pin - external connection)
            'EN': {'pins': [('U1', '7')]},
        }
    }

    return parts_db


def run_full_bbl_test():
    """Run the complete BBL pipeline test."""

    print("=" * 80)
    print("PCB ENGINE COMPREHENSIVE TEST - ESP32 SENSOR MODULE")
    print("=" * 80)
    print()

    # Import PCB Engine
    from pcb_engine import PCBEngine, EngineConfig

    # Create parts database
    parts_db = create_esp32_sensor_parts_db()

    print(f"Components: {len(parts_db['parts'])}")
    print(f"Nets: {len(parts_db['nets'])}")
    print()

    # List components
    print("Components:")
    for ref, part in parts_db['parts'].items():
        print(f"  {ref}: {part['value']} ({part['footprint']}) - {part.get('description', '')}")
    print()

    # List nets
    print("Nets:")
    for net, data in parts_db['nets'].items():
        pins = data.get('pins', [])
        print(f"  {net}: {len(pins)} pins")
    print()

    # Configure engine
    config = EngineConfig(
        board_name='esp32_sensor',
        board_width=50.0,
        board_height=40.0,
        layer_count=2,
        trace_width=0.25,
        clearance=0.15,
        via_diameter=0.6,
        via_drill=0.3,
        verbose=True,
        output_dir='./output'
    )

    print("-" * 80)
    print("ENGINE CONFIGURATION")
    print("-" * 80)
    print(f"  Board size: {config.board_width}mm x {config.board_height}mm")
    print(f"  Layers: {config.layer_count}")
    print(f"  Trace width: {config.trace_width}mm")
    print(f"  Clearance: {config.clearance}mm")
    print(f"  Via: {config.via_diameter}mm / {config.via_drill}mm drill")
    print()

    # Create engine
    engine = PCBEngine(config)

    # Progress tracking
    phase_times = {}
    current_phase = [None]

    def on_progress(progress):
        """Progress callback for BBL."""
        phase = str(progress.phase)
        if current_phase[0] != phase:
            if current_phase[0]:
                phase_times[current_phase[0]] = time.time() - phase_times.get(current_phase[0] + '_start', time.time())
            current_phase[0] = phase
            phase_times[phase + '_start'] = time.time()

        # Show progress bar
        bar_len = 40
        filled = int(bar_len * progress.percentage / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r  [{bar}] {progress.percentage:5.1f}% - {progress.message[:40]:<40}", end='', flush=True)

    escalation_count = [0]

    def on_escalation(escalation):
        """Escalation callback for BBL."""
        escalation_count[0] += 1
        print(f"\n  [ESCALATION #{escalation_count[0]}] {escalation.level}: {escalation.reason}")

        # Auto-resolve with defaults for testing
        escalation.resolved = True
        escalation.response = "Continue with relaxed constraints"
        return escalation

    # Run BBL
    print("=" * 80)
    print("RUNNING BIG BEAUTIFUL LOOP (BBL)")
    print("=" * 80)
    print()

    start_time = time.time()

    try:
        result = engine.run_bbl(
            parts_db,
            progress_callback=on_progress,
            escalation_callback=on_escalation
        )
    except Exception as e:
        print(f"\n\n[ERROR] BBL failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    total_time = time.time() - start_time
    print()  # New line after progress bar
    print()

    # Results
    print("=" * 80)
    print("BBL RESULTS")
    print("=" * 80)
    print()

    print(f"  Success: {'YES' if result.success else 'NO'}")
    print(f"  Total time: {total_time:.2f}s")
    print()

    # DRC Results
    print("  DRC Status:")
    print(f"    Internal DRC: {'PASS' if result.drc_passed else 'FAIL'}")
    print(f"    KiCad DRC: {'PASS' if result.kicad_drc_passed else 'FAIL'}")
    print()

    # Routing Results
    print("  Routing:")
    print(f"    Completion: {result.routing_completion * 100:.1f}%")
    print(f"    Routed: {result.routed_count}/{result.total_nets} nets")
    print()

    # Output Files
    print("  Output Files:")
    for f in result.output_files:
        print(f"    - {os.path.basename(f)}")
    print()

    # Errors
    if result.errors:
        print("  Errors:")
        for err in result.errors[:10]:
            print(f"    - {err}")
        if len(result.errors) > 10:
            print(f"    ... and {len(result.errors) - 10} more")
        print()

    # Escalations
    print(f"  Escalations: {escalation_count[0]}")
    print(f"  Rollbacks: {getattr(result, 'rollback_count', 0)}")
    print()

    # Check KiCad DRC report if available
    pcb_file = None
    for f in result.output_files:
        if f.endswith('.kicad_pcb'):
            pcb_file = f
            break

    if pcb_file:
        drc_file = pcb_file.replace('.kicad_pcb', '_drc.json')
        if os.path.exists(drc_file):
            with open(drc_file, 'r') as f:
                drc_report = json.load(f)

            violations = drc_report.get('violations', [])
            unconnected = drc_report.get('unconnected_items', [])
            errors = [v for v in violations if v.get('severity') == 'error']
            warnings = [v for v in violations if v.get('severity') == 'warning']

            print("  KiCad DRC Details:")
            print(f"    Errors: {len(errors)}")
            print(f"    Warnings: {len(warnings)}")
            print(f"    Unconnected: {len(unconnected)}")

            if errors:
                print("\n    Error types:")
                error_types = {}
                for e in errors:
                    t = e.get('type', 'unknown')
                    error_types[t] = error_types.get(t, 0) + 1
                for t, count in error_types.items():
                    print(f"      - {t}: {count}")

            if warnings:
                print("\n    Warning types:")
                warning_types = {}
                for w in warnings:
                    t = w.get('type', 'unknown')
                    warning_types[t] = warning_types.get(t, 0) + 1
                for t, count in warning_types.items():
                    print(f"      - {t}: {count}")

            print()

    # Final verdict
    print("=" * 80)
    if result.success and result.kicad_drc_passed:
        print("[PASS] TEST PASSED - Board generated successfully with clean DRC")
    elif result.success:
        print("[WARN] TEST PARTIAL - Board generated but has DRC issues")
    else:
        print("[FAIL] TEST FAILED - Board generation failed")
    print("=" * 80)

    return result.success and result.kicad_drc_passed


if __name__ == '__main__':
    success = run_full_bbl_test()
    sys.exit(0 if success else 1)
