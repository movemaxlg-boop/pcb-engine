#!/usr/bin/env python3
"""
PCB Engine MINIMAL Test - 2 Components Only
============================================

ULTRA-MINIMAL: Just a connector and resistor on a single net.
No multi-pin routing, no vias needed.

Design:
  - J1: 2-pin connector (SIG, GND)
  - R1: Resistor (connects SIG to ROUT)

This tests the absolute minimum:
  - 2 components
  - 1 signal net (SIG connects J1.1 to R1.1)
  - 1 power net (GND connects J1.2 to R1.2)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import PCBEngine, BoardConfig, DesignRules


def create_minimal_design():
    """
    Ultra-minimal 2-component design.

    Circuit: J1.1 (SIG) --[trace]--> R1.1 (SIG)
             J1.2 (GND) --[zone]---> R1.2 (GND)
    """

    parts = {
        # 2-pin connector - vertical pin header 2.54mm pitch
        'J1': {
            'name': 'Connector',
            'footprint': 'PinHeader_1x02_P2.54mm',
            'value': '2-pin',
            'description': 'Input connector',
            'pins': [
                # Pins are vertically stacked, 2.54mm apart
                {'number': '1', 'name': 'SIG', 'type': 'passive', 'net': 'SIG',
                 'physical': {'offset_x': 0, 'offset_y': -1.27}},   # Top pin
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 0, 'offset_y': 1.27}},    # Bottom pin
            ],
            'size': (2.54, 5.08),  # Standard header size
        },

        # Resistor - 0805 footprint (2.0 x 1.25mm)
        'R1': {
            'name': 'Resistor',
            'footprint': '0805',
            'value': '10K',
            'description': 'Pull-up resistor',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'SIG',
                 'physical': {'offset_x': -1.0, 'offset_y': 0}},    # Left pad
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 1.0, 'offset_y': 0}},     # Right pad
            ],
            'size': (2.0, 1.25),
        },
    }

    return parts


def run_minimal_test():
    """Run the minimal engine test"""

    print("=" * 60)
    print("PCB ENGINE MINIMAL TEST - 2 Components")
    print("=" * 60)
    print()

    # Small board - plenty of room for 2 small components
    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=20.0,     # 20mm x 15mm - compact
        height=15.0,
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
    parts = create_minimal_design()
    engine.load_parts_from_dict(parts)
    print(f"Parts loaded: {len(parts)}")
    print(f"  J1: 2-pin Connector")
    print(f"  R1: 0805 Resistor")
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
        output_path = os.path.join(os.path.dirname(__file__), 'test_minimal_output.py')
        if engine.generate_kicad_script(output_path):
            print(f"\nKiCad script: {output_path}")

    return engine.state.phase == EnginePhase.COMPLETE


if __name__ == '__main__':
    success = run_minimal_test()
    sys.exit(0 if success else 1)
