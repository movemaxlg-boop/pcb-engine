#!/usr/bin/env python3
"""
Human-Like PCB Engine Test - Complex 18 Components
===================================================

Tests the human-workflow engine with a realistic sensor module design:
- ESP32-C3 MCU (hub)
- 2 I2C sensors (BME280, BH1750)
- USB-C connector
- Power regulation (AMS1117)
- 6 decoupling capacitors
- 6 resistors (I2C pull-ups, USB CC, LED, EN)
- 1 LED indicator

This is a REAL design that should produce DRC-free results.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.engine import BoardConfig, DesignRules
from pcb_engine.human_engine import HumanPCBEngine


def create_complex_design():
    """
    Create a realistic sensor module design.

    This represents a real PCB with:
    - MCU as hub (most connections)
    - I2C sensors
    - Power regulation
    - Decoupling caps
    - USB interface
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
                {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': -6.0, 'offset_y': -8.0}},
                {'number': '2', 'name': '3V3', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -6.0, 'offset_y': -6.0}},
                # I2C
                {'number': '5', 'name': 'GPIO4_SDA', 'type': 'bidirectional', 'net': 'SDA',
                 'physical': {'offset_x': -6.0, 'offset_y': 0}},
                {'number': '6', 'name': 'GPIO5_SCL', 'type': 'output', 'net': 'SCL',
                 'physical': {'offset_x': -6.0, 'offset_y': 2.0}},
                # GPIO
                {'number': '7', 'name': 'GPIO6', 'type': 'bidirectional', 'net': 'LED',
                 'physical': {'offset_x': -6.0, 'offset_y': 4.0}},
                {'number': '8', 'name': 'GPIO7', 'type': 'bidirectional', 'net': 'BTN',
                 'physical': {'offset_x': -6.0, 'offset_y': 6.0}},
                # USB
                {'number': '12', 'name': 'GPIO18_D-', 'type': 'bidirectional', 'net': 'USB_DN',
                 'physical': {'offset_x': 6.0, 'offset_y': -4.0}},
                {'number': '13', 'name': 'GPIO19_D+', 'type': 'bidirectional', 'net': 'USB_DP',
                 'physical': {'offset_x': 6.0, 'offset_y': -2.0}},
                # EN and Boot
                {'number': '3', 'name': 'EN', 'type': 'input', 'net': 'EN',
                 'physical': {'offset_x': 6.0, 'offset_y': 2.0}},
                {'number': '4', 'name': 'GPIO9_BOOT', 'type': 'input', 'net': 'BOOT',
                 'physical': {'offset_x': 6.0, 'offset_y': 4.0}},
            ],
            'size': (13.2, 16.6),
            # Courtyard = body + clearance for routing
            'physical': {
                'body': (13.2, 16.6),
                'courtyard': (14.2, 17.6),  # 0.5mm clearance each side
            },
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
                {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': -1.0, 'offset_y': -1.0}},
                {'number': '3', 'name': 'SDI', 'type': 'bidirectional', 'net': 'SDA',
                 'physical': {'offset_x': 1.0, 'offset_y': -1.0}},
                {'number': '4', 'name': 'SCK', 'type': 'input', 'net': 'SCL',
                 'physical': {'offset_x': 1.0, 'offset_y': 1.0}},
                {'number': '8', 'name': 'VDD', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -1.0, 'offset_y': 1.0}},
            ],
            'size': (2.5, 2.5),
        },

        'U3': {
            'name': 'BH1750',
            'footprint': 'WSOF6I',
            'value': 'BH1750',
            'description': 'Ambient Light Sensor',
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -1.0, 'offset_y': -0.5}},
                {'number': '3', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 1.0, 'offset_y': -0.5}},
                {'number': '4', 'name': 'SDA', 'type': 'bidirectional', 'net': 'SDA',
                 'physical': {'offset_x': -1.0, 'offset_y': 0.5}},
                {'number': '6', 'name': 'SCL', 'type': 'input', 'net': 'SCL',
                 'physical': {'offset_x': 1.0, 'offset_y': 0.5}},
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
                {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': -2.3, 'offset_y': 0}},
                {'number': '2', 'name': 'VOUT', 'type': 'power_out', 'net': '3V3',
                 'physical': {'offset_x': 0, 'offset_y': 0}},
                {'number': '3', 'name': 'VIN', 'type': 'power_in', 'net': 'VBUS',
                 'physical': {'offset_x': 2.3, 'offset_y': 0}},
            ],
            'size': (6.5, 3.5),
        },

        # =====================================================================
        # USB Connector
        # =====================================================================
        'J1': {
            'name': 'USB-C',
            'footprint': 'USB_C_Receptacle',
            'value': 'USB-C',
            'description': 'USB Type-C Connector',
            'pins': [
                {'number': 'A1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': -3.0, 'offset_y': 0}},
                {'number': 'A4', 'name': 'VBUS', 'type': 'power_in', 'net': 'VBUS',
                 'physical': {'offset_x': -1.0, 'offset_y': 0}},
                {'number': 'A6', 'name': 'D+', 'type': 'bidirectional', 'net': 'USB_DP',
                 'physical': {'offset_x': 1.0, 'offset_y': 0}},
                {'number': 'A7', 'name': 'D-', 'type': 'bidirectional', 'net': 'USB_DN',
                 'physical': {'offset_x': 3.0, 'offset_y': 0}},
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
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },

        'C2': {
            'name': 'Capacitor',
            'footprint': '0402',
            'value': '100nF',
            'description': 'BME280 Decoupling',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },

        'C3': {
            'name': 'Capacitor',
            'footprint': '0402',
            'value': '100nF',
            'description': 'BH1750 Decoupling',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },

        'C4': {
            'name': 'Capacitor',
            'footprint': '0805',
            'value': '10uF',
            'description': 'Regulator Input',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': 'VBUS',
                 'physical': {'offset_x': -1.0, 'offset_y': 0}},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 1.0, 'offset_y': 0}},
            ],
            'size': (2.0, 1.25),
        },

        'C5': {
            'name': 'Capacitor',
            'footprint': '0805',
            'value': '22uF',
            'description': 'Regulator Output',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': -1.0, 'offset_y': 0}},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 1.0, 'offset_y': 0}},
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
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'SDA',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },

        'R2': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '4.7K',
            'description': 'SCL Pull-up',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'SCL',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
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
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED',
                 'physical': {'offset_x': -0.7, 'offset_y': 0}},
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': 0.7, 'offset_y': 0}},
            ],
            'size': (1.6, 0.8),
        },

        'R5': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '1K',
            'description': 'LED Resistor',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },

        # =====================================================================
        # EN Circuit
        # =====================================================================
        'R6': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '10K',
            'description': 'EN Pull-up',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'EN',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },

        'C6': {
            'name': 'Capacitor',
            'footprint': '0402',
            'value': '100nF',
            'description': 'EN Filter',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': 'EN',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },
    }

    return parts


def run_human_complex_test():
    """Run the human-like engine with complex design"""

    print("=" * 70)
    print("HUMAN-LIKE PCB ENGINE TEST - 15 Components Sensor Module")
    print("=" * 70)
    print()

    # Board configuration - 60mm x 45mm for ESP32 + sensors
    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=60.0,
        height=45.0,
        layers=2,
        grid_size=0.5
    )

    rules = DesignRules(
        min_trace_width=0.5,   # Minimum 0.5mm traces as required
        min_clearance=0.2,
        min_via_diameter=0.8,
        min_via_drill=0.4,
    )

    print(f"Board: {board.width}mm x {board.height}mm")
    print(f"Grid: {board.grid_size}mm")
    print()

    # Create engine
    engine = HumanPCBEngine(board, rules)

    # Load design
    parts = create_complex_design()
    engine.load_parts_from_dict(parts)

    print(f"Parts loaded: {len(parts)}")
    print("  ICs: U1 (ESP32), U2 (BME280), U3 (BH1750), U4 (AMS1117)")
    print("  Connectors: J1 (USB-C)")
    print("  Passives: C1-C6, R1-R2, R5-R6, D1")
    print()

    # Run all phases
    print("RUNNING HUMAN-LIKE WORKFLOW:")
    print("-" * 40)

    success = engine.run()

    print()
    print(engine.get_report())

    # Generate script if successful
    if success:
        output_path = os.path.join(os.path.dirname(__file__), 'test_human_complex_output.py')
        if engine.generate_kicad_script(output_path, include_routes=True):
            print(f"\nKiCad script: {output_path}")
            print("\nTo use in KiCad:")
            print("  1. Create a new PCB project")
            print("  2. Add all footprints from the script")
            print("  3. Open Tools -> Scripting Console")
            print(f'  4. Run: exec(open(r"{output_path}").read())')
            print("  5. Run DRC to verify")

    # Return both success flag and engine for inspection
    routed_count = sum(1 for r in engine.state.routes.values() if r.success)
    return {
        'success': success,
        'routes': engine.state.routes,
        'placement': engine.state.placement,
        'escapes': engine.state.escapes,
        'total_nets': 12,  # Known from design
        'routed_nets': routed_count,
    }


if __name__ == '__main__':
    result = run_human_complex_test()
    sys.exit(0 if result['success'] else 1)
