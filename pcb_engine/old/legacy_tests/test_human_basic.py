#!/usr/bin/env python3
"""
Human-Like PCB Engine Test - Basic 3 Components
================================================

Tests the human-workflow engine with a simple design:
  - MCU (U1): 4 pins
  - Resistor (R1): 2 pins
  - LED (D1): 2 pins

Signal chain: U1.GPIO → R1 → D1 → GND

This should produce CLEAN, DRC-FREE results because
the human-like engine:
1. Places components in signal flow order
2. Escapes perpendicular to component body
3. Routes with Manhattan paths (no weird angles)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.engine import BoardConfig, DesignRules
from pcb_engine.human_engine import HumanPCBEngine


def create_basic_design():
    """
    Minimal 3-component design for testing.

    Circuit: MCU GPIO -> R1 -> LED -> GND
             MCU VCC <- 3V3
             MCU GND -> GND
    """

    parts = {
        # MCU - 4 pins, the hub (QFN-16 like footprint, 4x4mm)
        'U1': {
            'name': 'MCU',
            'footprint': 'QFN-16',
            'value': 'MCU',
            'description': 'Simple MCU',
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -1.5, 'offset_y': -2.0}},  # Top-left
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 1.5, 'offset_y': -2.0}},   # Top-right
                {'number': '3', 'name': 'GPIO1', 'type': 'output', 'net': 'LED_CTRL',
                 'physical': {'offset_x': -1.5, 'offset_y': 2.0}},   # Bottom-left
                {'number': '4', 'name': 'GPIO2', 'type': 'input', 'net': 'NC',
                 'physical': {'offset_x': 1.5, 'offset_y': 2.0}},    # Bottom-right
            ],
            'size': (4.0, 4.0),
        },

        # LED - 0603 footprint (1.6 x 0.8mm)
        'D1': {
            'name': 'LED',
            'footprint': '0603',
            'value': 'Green',
            'description': 'Status LED',
            'pins': [
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': -0.7, 'offset_y': 0}},     # Left pad (Anode)
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.7, 'offset_y': 0}},      # Right pad (Cathode)
            ],
            'size': (1.6, 0.8),
        },

        # Resistor - 0402 footprint (1.0 x 0.5mm)
        'R1': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '1K',
            'description': 'LED Resistor',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED_CTRL',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},    # Left pad
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},     # Right pad
            ],
            'size': (1.0, 0.5),
        },
    }

    return parts


def run_human_test():
    """Run the human-like engine test"""

    print("=" * 60)
    print("HUMAN-LIKE PCB ENGINE TEST - 3 Components")
    print("=" * 60)
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
        min_trace_width=0.25,
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
    parts = create_basic_design()
    engine.load_parts_from_dict(parts)

    print(f"Parts loaded: {len(parts)}")
    print(f"  U1: MCU (4 pins)")
    print(f"  D1: LED (2 pins)")
    print(f"  R1: Resistor (2 pins)")
    print()

    # Run all phases
    print("RUNNING HUMAN-LIKE WORKFLOW:")
    print("-" * 40)

    success = engine.run()

    print()
    print(engine.get_report())

    # Generate script if successful
    if success:
        output_path = os.path.join(os.path.dirname(__file__), 'test_human_basic_output.py')
        if engine.generate_kicad_script(output_path, include_routes=True):
            print(f"\nKiCad script: {output_path}")
            print("\nNEXT STEPS:")
            print("  1. Open KiCad and create a new PCB")
            print("  2. Add footprints: U1 (QFN-16), R1 (0402), D1 (0603)")
            print("  3. Open Tools -> Scripting Console")
            print(f'  4. Run: exec(open(r"{output_path}").read())')
            print("  5. Run DRC - should have ZERO errors!")

    return success


if __name__ == '__main__':
    success = run_human_test()
    sys.exit(0 if success else 1)
