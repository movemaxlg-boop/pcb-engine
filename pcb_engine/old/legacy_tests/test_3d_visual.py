#!/usr/bin/env python3
"""
Test 3D Visualization
=====================

This test creates a PCB using REAL KiCad library footprints with 3D models.
When opened in KiCad's 3D viewer, you will see actual component shapes.

Components used:
- SOT-23-5 (small IC package) - has 3D model
- 0805 Resistors - has 3D model
- 0805 Capacitor - has 3D model
- 0805 LED - has 3D model
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.engine import BoardConfig, DesignRules
from pcb_engine.human_engine import HumanPCBEngine


def create_test_design_with_real_footprints():
    """
    Create a test design using REAL KiCad footprints.

    IMPORTANT: We STILL need physical pin positions for the routing algorithm.
    The 'kicad_footprint' field tells the generator to load from KiCad library
    (which includes 3D models) instead of creating a custom footprint.
    """

    # SOT-23-5 pin positions (from KiCad library)
    # Pins 1,2,3 on bottom (y=1.1), pins 4,5 on top (y=-1.1)
    sot23_5_pins = {
        '1': (-0.95, 1.1),   # VIN
        '2': (0.0, 1.1),     # GND
        '3': (0.95, 1.1),    # EN
        '4': (0.95, -1.1),   # NC
        '5': (-0.95, -1.1),  # VOUT
    }

    # 0805 pin positions (2-pin SMD)
    # Pads at +/- 0.95mm from center
    smd_0805_pins = {
        '1': (-0.95, 0),
        '2': (0.95, 0),
    }

    parts = {
        # Voltage regulator - SOT-23-5 (has 3D model in KiCad)
        'U1': {
            'name': 'AP2112K-3.3',
            'footprint': 'SOT-23-5',
            'kicad_footprint': 'Package_TO_SOT_SMD:SOT-23-5',
            'value': '3.3V LDO',
            'pins': [
                {'number': '1', 'name': 'VIN', 'type': 'power_in', 'net': '5V',
                 'physical': {'offset_x': sot23_5_pins['1'][0], 'offset_y': sot23_5_pins['1'][1]}},
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': sot23_5_pins['2'][0], 'offset_y': sot23_5_pins['2'][1]}},
                {'number': '3', 'name': 'EN', 'type': 'input', 'net': '5V',
                 'physical': {'offset_x': sot23_5_pins['3'][0], 'offset_y': sot23_5_pins['3'][1]}},
                {'number': '4', 'name': 'NC', 'type': 'no_connect', 'net': '',
                 'physical': {'offset_x': sot23_5_pins['4'][0], 'offset_y': sot23_5_pins['4'][1]}},
                {'number': '5', 'name': 'VOUT', 'type': 'power_out', 'net': '3V3',
                 'physical': {'offset_x': sot23_5_pins['5'][0], 'offset_y': sot23_5_pins['5'][1]}},
            ],
            'size': (3.0, 3.0),
        },
        # Input capacitor - 0805 (has 3D model)
        'C1': {
            'name': 'Input Cap',
            'footprint': '0805',
            'kicad_footprint': 'Capacitor_SMD:C_0805_2012Metric',
            'value': '10uF',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': '5V',
                 'physical': {'offset_x': smd_0805_pins['1'][0], 'offset_y': smd_0805_pins['1'][1]}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': smd_0805_pins['2'][0], 'offset_y': smd_0805_pins['2'][1]}},
            ],
            'size': (2.0, 1.25),
        },
        # Output capacitor - 0805 (has 3D model)
        'C2': {
            'name': 'Output Cap',
            'footprint': '0805',
            'kicad_footprint': 'Capacitor_SMD:C_0805_2012Metric',
            'value': '10uF',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': smd_0805_pins['1'][0], 'offset_y': smd_0805_pins['1'][1]}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': smd_0805_pins['2'][0], 'offset_y': smd_0805_pins['2'][1]}},
            ],
            'size': (2.0, 1.25),
        },
        # LED current limiting resistor - 0805 (has 3D model)
        'R1': {
            'name': 'LED Resistor',
            'footprint': '0805',
            'kicad_footprint': 'Resistor_SMD:R_0805_2012Metric',
            'value': '330R',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': smd_0805_pins['1'][0], 'offset_y': smd_0805_pins['1'][1]}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED',
                 'physical': {'offset_x': smd_0805_pins['2'][0], 'offset_y': smd_0805_pins['2'][1]}},
            ],
            'size': (2.0, 1.25),
        },
        # Power LED - 0805 (has 3D model)
        'D1': {
            'name': 'Power LED',
            'footprint': '0805',
            'kicad_footprint': 'LED_SMD:LED_0805_2012Metric',
            'value': 'Green',
            'pins': [
                {'number': '1', 'name': 'K', 'type': 'passive', 'net': 'LED',
                 'physical': {'offset_x': smd_0805_pins['1'][0], 'offset_y': smd_0805_pins['1'][1]}},
                {'number': '2', 'name': 'A', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': smd_0805_pins['2'][0], 'offset_y': smd_0805_pins['2'][1]}},
            ],
            'size': (2.0, 1.25),
        },
    }
    return parts


def test_3d_visual():
    """Test with real KiCad footprints for 3D visualization"""
    print("=" * 70)
    print("3D VISUALIZATION TEST")
    print("=" * 70)
    print()
    print("This test generates a PCB with REAL KiCad library footprints.")
    print("When you open it in KiCad and press Alt+3, you'll see 3D components.")
    print()

    # Board configuration
    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=25.0,
        height=20.0,
        layers=2,
        grid_size=0.5
    )

    rules = DesignRules(
        min_trace_width=0.3,
        min_clearance=0.2,
        min_via_diameter=0.8,
        min_via_drill=0.4,
    )

    # Create engine
    engine = HumanPCBEngine(board, rules)

    # Load design with real footprints
    parts = create_test_design_with_real_footprints()
    engine.load_parts_from_dict(parts)

    print(f"Board: {board.width}mm x {board.height}mm")
    print(f"Components: {len(parts)}")
    print()
    print("Footprints (with 3D models):")
    for ref, part in parts.items():
        kicad_fp = part.get('kicad_footprint', part.get('footprint', 'unknown'))
        print(f"  {ref}: {kicad_fp}")
    print()

    # Run with DRC feedback loop
    output_path = os.path.join(os.path.dirname(__file__), 'test_3d_visual_output.py')
    result = engine.run_with_drc_loop(max_iterations=5, output_path=output_path)

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  Success: {result['success']}")
    print(f"  Routed: {result['routed_nets']}/{result['total_nets']} nets")
    print()

    if result['success']:
        print("TO VIEW IN 3D:")
        print("  1. Open KiCad 8 -> PCB Editor")
        print("  2. Tools -> Scripting Console (F4)")
        print(f'  3. Run: exec(open(r"{output_path}").read())')
        print("  4. Press Alt+3 for 3D view")
        print()
        print("You should see actual 3D component shapes!")

    return result


if __name__ == '__main__':
    result = test_3d_visual()
    sys.exit(0 if result['success'] else 1)
