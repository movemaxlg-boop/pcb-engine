#!/usr/bin/env python3
"""
PCB Engine Test Script
======================

This script tests the PCB Engine with a sample sensor module design.

Design: Simple Sensor Module
- ESP32-C3 (main MCU - the hub)
- BME280 (temperature/humidity/pressure sensor)
- BH1750 (light sensor)
- 3.3V regulator (AMS1117-3.3)
- Decoupling capacitors
- I2C pull-up resistors
- LED indicator
- USB-C connector

Run this to verify the engine works correctly.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import (
    PCBEngine, BoardConfig, DesignRules,
    print_library_status, libs
)


def create_sample_design():
    """
    Create a sample sensor module design.

    This represents a realistic small PCB with:
    - MCU as hub (most connections)
    - I2C sensors
    - Power regulation
    - Decoupling caps
    """

    parts = {
        # =====================================================================
        # MCU - ESP32-C3 (The Hub - most connections)
        # =====================================================================
        'U1': {
            'name': 'ESP32-C3-MINI-1',
            'footprint': 'ESP32-C3-MINI-1',
            'value': 'ESP32-C3',
            'description': 'WiFi/BLE MCU',
            'pins': [
                # Power
                {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND'},
                {'number': '2', 'name': '3V3', 'type': 'power_in', 'net': '3V3'},
                # I2C
                {'number': '5', 'name': 'GPIO4_SDA', 'type': 'bidirectional', 'net': 'SDA'},
                {'number': '6', 'name': 'GPIO5_SCL', 'type': 'output', 'net': 'SCL'},
                # GPIO
                {'number': '7', 'name': 'GPIO6', 'type': 'bidirectional', 'net': 'LED'},
                {'number': '8', 'name': 'GPIO7', 'type': 'bidirectional', 'net': 'BTN'},
                # USB
                {'number': '12', 'name': 'GPIO18_D-', 'type': 'bidirectional', 'net': 'USB_DN'},
                {'number': '13', 'name': 'GPIO19_D+', 'type': 'bidirectional', 'net': 'USB_DP'},
                # EN and Boot
                {'number': '3', 'name': 'EN', 'type': 'input', 'net': 'EN'},
                {'number': '4', 'name': 'GPIO9_BOOT', 'type': 'input', 'net': 'BOOT'},
            ],
            'size': (13.2, 16.6),  # mm
        },

        # =====================================================================
        # Sensors (I2C)
        # =====================================================================
        'U2': {
            'name': 'BME280',
            'footprint': 'LGA-8_2.5x2.5mm',
            'value': 'BME280',
            'description': 'Temperature/Humidity/Pressure Sensor',
            'pins': [
                {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND'},
                {'number': '2', 'name': 'CSB', 'type': 'input', 'net': '3V3'},  # Pulled high for I2C
                {'number': '3', 'name': 'SDI', 'type': 'bidirectional', 'net': 'SDA'},
                {'number': '4', 'name': 'SCK', 'type': 'input', 'net': 'SCL'},
                {'number': '5', 'name': 'SDO', 'type': 'output', 'net': 'GND'},  # Address select
                {'number': '6', 'name': 'VDDIO', 'type': 'power_in', 'net': '3V3'},
                {'number': '7', 'name': 'GND2', 'type': 'power_in', 'net': 'GND'},
                {'number': '8', 'name': 'VDD', 'type': 'power_in', 'net': '3V3'},
            ],
            'size': (2.5, 2.5),
        },

        'U3': {
            'name': 'BH1750',
            'footprint': 'WSOF6I',
            'value': 'BH1750',
            'description': 'Ambient Light Sensor',
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': '3V3'},
                {'number': '2', 'name': 'ADDR', 'type': 'input', 'net': 'GND'},  # Address select
                {'number': '3', 'name': 'GND', 'type': 'power_in', 'net': 'GND'},
                {'number': '4', 'name': 'SDA', 'type': 'bidirectional', 'net': 'SDA'},
                {'number': '5', 'name': 'DVI', 'type': 'input', 'net': 'GND'},  # Not used
                {'number': '6', 'name': 'SCL', 'type': 'input', 'net': 'SCL'},
            ],
            'size': (3.0, 1.6),
        },

        # =====================================================================
        # Power Regulation
        # =====================================================================
        'U4': {
            'name': 'AMS1117-3.3',
            'footprint': 'SOT-223',
            'value': 'AMS1117-3.3',
            'description': '3.3V LDO Regulator',
            'pins': [
                {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND'},
                {'number': '2', 'name': 'VOUT', 'type': 'power_out', 'net': '3V3'},
                {'number': '3', 'name': 'VIN', 'type': 'power_in', 'net': 'VBUS'},
                {'number': '4', 'name': 'VOUT2', 'type': 'power_out', 'net': '3V3'},  # Tab
            ],
            'size': (6.5, 3.5),
        },

        # =====================================================================
        # USB Connector
        # =====================================================================
        'J1': {
            'name': 'USB-C',
            'footprint': 'USB_C_Receptacle_GCT_USB4105',
            'value': 'USB-C',
            'description': 'USB Type-C Connector',
            'pins': [
                {'number': 'A1', 'name': 'GND', 'type': 'power_in', 'net': 'GND'},
                {'number': 'A4', 'name': 'VBUS', 'type': 'power_in', 'net': 'VBUS'},
                {'number': 'A6', 'name': 'D+', 'type': 'bidirectional', 'net': 'USB_DP'},
                {'number': 'A7', 'name': 'D-', 'type': 'bidirectional', 'net': 'USB_DN'},
                {'number': 'B1', 'name': 'GND2', 'type': 'power_in', 'net': 'GND'},
                {'number': 'B4', 'name': 'VBUS2', 'type': 'power_in', 'net': 'VBUS'},
                {'number': 'A5', 'name': 'CC1', 'type': 'bidirectional', 'net': 'CC1'},
                {'number': 'B5', 'name': 'CC2', 'type': 'bidirectional', 'net': 'CC2'},
            ],
            'size': (8.9, 7.3),
        },

        # =====================================================================
        # Decoupling Capacitors
        # =====================================================================
        'C1': {
            'name': 'Capacitor',
            'footprint': '0402',
            'value': '100nF',
            'description': 'MCU Decoupling',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3'},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (1.0, 0.5),
        },

        'C2': {
            'name': 'Capacitor',
            'footprint': '0402',
            'value': '100nF',
            'description': 'BME280 Decoupling',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3'},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (1.0, 0.5),
        },

        'C3': {
            'name': 'Capacitor',
            'footprint': '0402',
            'value': '100nF',
            'description': 'BH1750 Decoupling',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3'},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (1.0, 0.5),
        },

        'C4': {
            'name': 'Capacitor',
            'footprint': '0805',
            'value': '10uF',
            'description': 'Regulator Input',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': 'VBUS'},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (2.0, 1.25),
        },

        'C5': {
            'name': 'Capacitor',
            'footprint': '0805',
            'value': '22uF',
            'description': 'Regulator Output',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3'},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (2.0, 1.25),
        },

        # =====================================================================
        # I2C Pull-up Resistors
        # =====================================================================
        'R1': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '4.7K',
            'description': 'SDA Pull-up',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'SDA'},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': '3V3'},
            ],
            'size': (1.0, 0.5),
        },

        'R2': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '4.7K',
            'description': 'SCL Pull-up',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'SCL'},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': '3V3'},
            ],
            'size': (1.0, 0.5),
        },

        # =====================================================================
        # USB CC Resistors (for USB-C)
        # =====================================================================
        'R3': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '5.1K',
            'description': 'CC1 Pulldown',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'CC1'},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (1.0, 0.5),
        },

        'R4': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '5.1K',
            'description': 'CC2 Pulldown',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'CC2'},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (1.0, 0.5),
        },

        # =====================================================================
        # LED Indicator
        # =====================================================================
        'D1': {
            'name': 'LED',
            'footprint': '0603',
            'value': 'Green',
            'description': 'Status LED',
            'pins': [
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED'},
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'LED_R'},
            ],
            'size': (1.6, 0.8),
        },

        'R5': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '1K',
            'description': 'LED Resistor',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED_R'},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (1.0, 0.5),
        },

        # =====================================================================
        # EN/BOOT Circuit
        # =====================================================================
        'R6': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '10K',
            'description': 'EN Pull-up',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'EN'},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': '3V3'},
            ],
            'size': (1.0, 0.5),
        },

        'C6': {
            'name': 'Capacitor',
            'footprint': '0402',
            'value': '100nF',
            'description': 'EN Filter',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': 'EN'},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND'},
            ],
            'size': (1.0, 0.5),
        },
    }

    return parts


def run_test():
    """Run the complete engine test"""

    print("=" * 70)
    print("PCB ENGINE TEST - Sensor Module Design")
    print("=" * 70)
    print()

    # Check available libraries
    print("LIBRARY STATUS:")
    print("-" * 40)
    print(f"  NumPy:      {'Available' if libs.has_numpy else 'Not installed'}")
    print(f"  Matplotlib: {'Available' if libs.has_matplotlib else 'Not installed'}")
    print(f"  NetworkX:   {'Available' if libs.has_networkx else 'Not installed'}")
    print(f"  SciPy:      {'Available' if libs.has_scipy else 'Not installed'}")
    print(f"  Shapely:    {'Available' if libs.has_shapely else 'Not installed'}")
    print()

    # Create engine with board configuration
    print("CONFIGURING ENGINE:")
    print("-" * 40)

    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=60.0,     # 60mm x 45mm board (sized for ESP32 + sensors)
        height=45.0,
        layers=2,       # 2-layer board
        grid_size=0.5   # 0.5mm grid (Rule: grid_size = min_pitch / 4)
    )

    rules = DesignRules(
        min_trace_width=0.2,      # 0.2mm min trace
        min_clearance=0.15,       # 0.15mm clearance
        min_via_diameter=0.6,     # 0.6mm via
        min_via_drill=0.3,        # 0.3mm drill
        min_hole_clearance=0.2,
        min_annular_ring=0.125,
    )

    engine = PCBEngine(board, rules)

    print(f"  Board: {board.width}mm x {board.height}mm")
    print(f"  Layers: {board.layers}")
    print(f"  Grid: {board.grid_size}mm")
    print(f"  Min trace: {rules.min_trace_width}mm")
    print(f"  Min clearance: {rules.min_clearance}mm")
    print()

    # Load sample design
    print("LOADING DESIGN:")
    print("-" * 40)

    parts = create_sample_design()
    engine.load_parts_from_dict(parts)

    print(f"  Parts loaded: {len(parts)}")
    print(f"  ICs: U1 (ESP32-C3), U2 (BME280), U3 (BH1750), U4 (AMS1117)")
    print(f"  Connectors: J1 (USB-C)")
    print(f"  Passives: C1-C6, R1-R6, D1")
    print()

    # Run all phases
    print("RUNNING ENGINE PHASES:")
    print("-" * 40)

    phase_names = [
        "Phase 0: Parts Validation",
        "Phase 1: Graph Building",
        "Phase 2: Hub Identification",
        "Phase 3: Placement Optimization",
        "Phase 4: Escape Calculation",
        "Phase 5: Corridor Validation",
        "Phase 6: Route Ordering",
        "Phase 7: Routing",
        "Phase 8: DRC Validation",
    ]

    success = True
    for i, phase_name in enumerate(phase_names):
        from pcb_engine.engine import EnginePhase
        phase = EnginePhase(i)

        print(f"\n  {phase_name}...")
        result = engine.run_phase(phase)

        if result:
            print(f"    [OK] Success")

            # Print phase-specific info
            if i == 1:  # Graph
                print(f"      Nets found: {len(engine.state.parts_db.get('nets', {}))}")
            elif i == 2:  # Hub
                print(f"      Hub component: {engine.state.hub}")
            elif i == 3:  # Placement
                print(f"      Components placed: {len(engine.state.placement)}")
            elif i == 4:  # Escape
                print(f"      Escape vectors: {len(engine.state.escapes)}")
            elif i == 5:  # Corridor
                print(f"      Corridors validated: {len(engine.state.corridors)}")
            elif i == 6:  # Route order
                print(f"      Nets to route: {len(engine.state.route_order)}")
            elif i == 7:  # Routing
                print(f"      Routes completed: {len(engine.state.routes)}")
            elif i == 8:  # Validation
                valid = engine.state.validation.get('valid', False)
                print(f"      DRC passed: {valid}")
        else:
            print(f"    [FAIL] Failed")
            for error in engine.state.errors[-3:]:  # Last 3 errors
                print(f"      Error: {error}")
            success = False
            break

    print()

    # Generate output
    if success:
        print("GENERATING OUTPUT:")
        print("-" * 40)

        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, 'test_output.py')

        if engine.generate_kicad_script(output_path):
            print(f"  [OK] KiCad script generated: test_output.py")
            print(f"    Path: {output_path}")
        else:
            print(f"  [FAIL] Failed to generate script")

    # Print final report
    print()
    print(engine.get_report())

    return success


def main():
    """Main entry point"""
    try:
        success = run_test()

        if success:
            print("\n" + "=" * 70)
            print("TEST PASSED - Engine completed successfully!")
            print("=" * 70)
            print("\nNext steps:")
            print("  1. Open KiCad")
            print("  2. Go to Tools > Scripting Console")
            print("  3. Run: exec(open('test_output.py').read())")
            print("  4. Review the generated PCB design")
            return 0
        else:
            print("\n" + "=" * 70)
            print("TEST FAILED - See errors above")
            print("=" * 70)
            return 1

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
