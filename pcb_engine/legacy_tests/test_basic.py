#!/usr/bin/env python3
"""
PCB Engine Basic Test
=====================

MINIMAL test: 3 components, 2 signal nets, 2 power nets.
This is the simplest possible test to validate the engine works.

Design:
  - MCU (4 pins: VCC, GND, GPIO, GPIO)
  - LED (2 pins: A, K)
  - Resistor (2 pins)

Nets:
  - 3V3 (power)
  - GND (power)
  - LED_CTRL (MCU -> Resistor -> LED)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import PCBEngine, BoardConfig, DesignRules


def create_basic_design():
    """
    Minimal 3-component design for testing.

    Circuit: MCU GPIO -> R1 -> LED -> GND
             MCU VCC <- 3V3
             MCU GND -> GND
    """

    parts = {
        # MCU - 4 pins, the hub (QFN-16 like footprint, 4x4mm)
        # Pin 1.27mm pitch, pins on edges
        'U1': {
            'name': 'MCU',
            'footprint': 'QFN-16',
            'value': 'MCU',
            'description': 'Simple MCU',
            'pins': [
                # Pin offsets in physical subdictionary
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


def run_basic_test():
    """Run the basic engine test"""

    print("=" * 60)
    print("PCB ENGINE BASIC TEST - 3 Components")
    print("=" * 60)
    print()

    # Larger board for 3 components - more routing room
    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=30.0,     # 30mm x 25mm - more room for routing
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

    engine = PCBEngine(board, rules)

    print(f"Board: {board.width}mm x {board.height}mm")
    print(f"Grid: {board.grid_size}mm")
    print()

    # Load design
    parts = create_basic_design()
    engine.load_parts_from_dict(parts)
    print(f"Parts loaded: {len(parts)}")
    print(f"  U1: MCU (4 pins)")
    print(f"  D1: LED (2 pins)")
    print(f"  R1: Resistor (2 pins)")
    print()

    # Run all phases
    print("RUNNING PHASES:")
    print("-" * 40)

    phase_names = [
        "Phase 0: Parts Validation",
        "Phase 1: Graph Building",
        "Phase 2: Hub Identification",
        "Phase 3: Placement",
        "Phase 4: Escape Calculation",
        "Phase 5: Corridor Validation",
        "Phase 6: Route Ordering",
        "Phase 7: Routing",
        "Phase 8: Validation",
    ]

    from pcb_engine.engine import EnginePhase

    for i, phase_name in enumerate(phase_names):
        phase = EnginePhase(i)
        print(f"\n{phase_name}...")

        result = engine.run_phase(phase)

        if result:
            print(f"  [OK]")

            # Phase-specific output
            if i == 1:  # Graph
                nets = engine.state.parts_db.get('nets', {})
                print(f"  Nets: {list(nets.keys())}")
            elif i == 2:  # Hub
                print(f"  Hub: {engine.state.hub}")
            elif i == 3:  # Placement
                for ref, pos in engine.state.placement.items():
                    print(f"  {ref}: ({pos.x:.1f}, {pos.y:.1f})")
            elif i == 6:  # Route order
                print(f"  Order: {engine.state.route_order}")
            elif i == 7:  # Routing
                print(f"  Routed: {list(engine.state.routes.keys())}")
        else:
            print(f"  [FAIL]")
            for error in engine.state.errors[-5:]:
                print(f"    ERROR: {error}")
            break

    print()
    print(engine.get_report())

    # Generate script if successful
    if engine.state.phase.value >= EnginePhase.PHASE_8_VALIDATE.value:
        output_path = os.path.join(os.path.dirname(__file__), 'test_basic_output.py')
        # Generate with routes included - use with KiCad projects that have footprints placed
        if engine.generate_kicad_script(output_path, include_routes=True):
            print(f"\nKiCad script: {output_path}")

    return engine.state.phase == EnginePhase.COMPLETE


if __name__ == '__main__':
    success = run_basic_test()
    sys.exit(0 if success else 1)
