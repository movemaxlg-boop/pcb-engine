#!/usr/bin/env python3
"""
Test DRC Feedback Loop
======================

This test demonstrates the closed-loop DRC correction system.
The engine will:
1. Generate initial layout
2. Run BUILT-IN DRC checks (ValidationGate)
3. Identify violations
4. Apply fixes
5. Repeat until DRC passes

This is the "automatic loop" that keeps refining until standards are met.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.engine import BoardConfig, DesignRules
from pcb_engine.human_engine import HumanPCBEngine


def create_test_design():
    """Create a simple test design with potential DRC issues"""
    parts = {
        # MCU
        'U1': {
            'name': 'MCU',
            'footprint': 'TQFP-32',
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -3.5, 'offset_y': 0}},
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 3.5, 'offset_y': 0}},
                {'number': '3', 'name': 'SDA', 'type': 'bidirectional', 'net': 'SDA',
                 'physical': {'offset_x': 0, 'offset_y': -3.5}},
                {'number': '4', 'name': 'SCL', 'type': 'output', 'net': 'SCL',
                 'physical': {'offset_x': 0, 'offset_y': 3.5}},
            ],
            'size': (7, 7),
        },
        # Sensor
        'U2': {
            'name': 'Sensor',
            'footprint': 'LGA-8',
            'pins': [
                {'number': '1', 'name': 'VDD', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -1.0, 'offset_y': -1.0}},
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 1.0, 'offset_y': -1.0}},
                {'number': '3', 'name': 'SDA', 'type': 'bidirectional', 'net': 'SDA',
                 'physical': {'offset_x': -1.0, 'offset_y': 1.0}},
                {'number': '4', 'name': 'SCL', 'type': 'input', 'net': 'SCL',
                 'physical': {'offset_x': 1.0, 'offset_y': 1.0}},
            ],
            'size': (2.5, 2.5),
        },
        # Decoupling cap for MCU
        'C1': {
            'name': 'Capacitor',
            'footprint': '0402',
            'pins': [
                {'number': '1', 'name': '+', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '-', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },
        # I2C pull-up
        'R1': {
            'name': 'Resistor',
            'footprint': '0402',
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
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'SCL',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': '3V3',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },
    }
    return parts


def test_drc_feedback():
    """Test the DRC feedback loop"""
    print("=" * 70)
    print("DRC FEEDBACK LOOP TEST")
    print("=" * 70)
    print()
    print("This test demonstrates automatic DRC correction:")
    print("  1. Engine generates initial layout")
    print("  2. Built-in DRC (ValidationGate) checks for violations")
    print("  3. Violations are analyzed and fix instructions generated")
    print("  4. Engine applies fixes and regenerates")
    print("  5. Loop continues until DRC passes or max iterations")
    print()

    # Board configuration
    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=30.0,
        height=25.0,
        layers=2,
        grid_size=0.5
    )

    rules = DesignRules(
        min_trace_width=0.5,
        min_clearance=0.2,
        min_via_diameter=0.8,
        min_via_drill=0.4,
    )

    # Create engine
    engine = HumanPCBEngine(board, rules)

    # Load design
    parts = create_test_design()
    engine.load_parts_from_dict(parts)

    print(f"Board: {board.width}mm x {board.height}mm")
    print(f"Components: {len(parts)}")
    print()

    # Run with DRC feedback loop (max 5 iterations)
    output_path = os.path.join(os.path.dirname(__file__), 'test_drc_feedback_output.py')
    result = engine.run_with_drc_loop(max_iterations=5, output_path=output_path)

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Routed: {result['routed_nets']}/{result['total_nets']} nets")
    print(f"  Final violations: {len(result['final_violations'])}")

    if result['success']:
        print()
        print(f"KiCad script: {output_path}")

    return result


if __name__ == '__main__':
    result = test_drc_feedback()
    sys.exit(0 if result['success'] else 1)
