#!/usr/bin/env python3
"""
Debug test to understand routing failures.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import PCBEngine, BoardConfig, DesignRules
from pcb_engine.engine import EnginePhase


def create_simple_design():
    """Even simpler: MCU + Resistor only"""
    return {
        'U1': {
            'name': 'MCU',
            'footprint': 'QFN-4',
            'value': 'MCU',
            'description': 'Simple MCU',
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -1.5, 'offset_y': 0}},
                {'number': '2', 'name': 'GPIO', 'type': 'output', 'net': 'SIG',
                 'physical': {'offset_x': 1.5, 'offset_y': 0}},
            ],
            'size': (3.0, 3.0),
        },

        'R1': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '1K',
            'description': 'Resistor',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'SIG',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },
    }


def run_debug():
    print("=" * 60)
    print("DEBUG TEST - 2 Components, 1 Signal Net")
    print("=" * 60)
    print()

    board = BoardConfig(
        origin_x=100.0,
        origin_y=100.0,
        width=20.0,
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
    parts = create_simple_design()
    engine.load_parts_from_dict(parts)

    # Run phases one by one with debug info
    for i in range(9):
        phase = EnginePhase(i)
        print(f"\n--- {phase.name} ---")
        result = engine.run_phase(phase)
        print(f"Result: {'OK' if result else 'FAILED'}")

        if i == 1:  # Graph
            nets = engine.state.parts_db.get('nets', {})
            print(f"Nets: {list(nets.keys())}")
            for net_name, net_info in nets.items():
                pins = net_info.get('pins', [])
                print(f"  {net_name}: {pins}")

        elif i == 3:  # Placement
            print("Placement:")
            for ref, pos in engine.state.placement.items():
                print(f"  {ref}: ({pos.x:.1f}, {pos.y:.1f})")

            # Calculate actual pin positions
            print("\nPin Positions:")
            for ref, pos in engine.state.placement.items():
                part = engine.state.parts_db['parts'].get(ref, {})
                for pin in part.get('used_pins', []):
                    offset = pin.get('offset', (0, 0))
                    px = pos.x + offset[0]
                    py = pos.y + offset[1]
                    print(f"  {ref}.{pin['number']} ({pin.get('net', 'NC')}): ({px:.1f}, {py:.1f})")

        elif i == 6:  # Route order
            print(f"Route order: {engine.state.route_order}")

        elif i == 7:  # Routing
            print(f"Routed: {list(engine.state.routes.keys())}")
            if hasattr(engine, '_router') and engine._router:
                print(f"Failed: {engine._router.failed}")

        if not result:
            print("Errors:", engine.state.errors[-5:])
            break

    print()
    print(engine.get_report())


if __name__ == '__main__':
    run_debug()
