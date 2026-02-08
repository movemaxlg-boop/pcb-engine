#!/usr/bin/env python3
"""
Test routing order impact - does LED_R succeed if routed first?
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import PCBEngine, BoardConfig, DesignRules
from pcb_engine.engine import EnginePhase


def create_design():
    return {
        'U1': {
            'name': 'MCU',
            'footprint': 'QFN-16',
            'value': 'MCU',
            'description': 'Simple MCU',
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                 'physical': {'offset_x': -1.5, 'offset_y': -2.0}},
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 1.5, 'offset_y': -2.0}},
                {'number': '3', 'name': 'GPIO1', 'type': 'output', 'net': 'LED_CTRL',
                 'physical': {'offset_x': -1.5, 'offset_y': 2.0}},
                {'number': '4', 'name': 'GPIO2', 'type': 'input', 'net': 'NC',
                 'physical': {'offset_x': 1.5, 'offset_y': 2.0}},
            ],
            'size': (4.0, 4.0),
        },
        'D1': {
            'name': 'LED',
            'footprint': '0603',
            'value': 'Green',
            'description': 'Status LED',
            'pins': [
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': -0.7, 'offset_y': 0}},
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.7, 'offset_y': 0}},
            ],
            'size': (1.6, 0.8),
        },
        'R1': {
            'name': 'Resistor',
            'footprint': '0402',
            'value': '1K',
            'description': 'LED Resistor',
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED_CTRL',
                 'physical': {'offset_x': -0.45, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED_R',
                 'physical': {'offset_x': 0.45, 'offset_y': 0}},
            ],
            'size': (1.0, 0.5),
        },
    }


def test_with_order(order_name, route_order):
    """Test with specific route order"""
    print(f"\n{'='*60}")
    print(f"TEST: {order_name}")
    print(f"Route order: {route_order}")
    print(f"{'='*60}")

    board = BoardConfig(
        origin_x=100.0, origin_y=100.0,
        width=30.0, height=25.0,
        layers=2, grid_size=0.5
    )

    rules = DesignRules(
        min_trace_width=0.25,
        min_clearance=0.2,
        min_via_diameter=0.8,
        min_via_drill=0.4,
    )

    engine = PCBEngine(board, rules)
    engine.load_parts_from_dict(create_design())

    # Run phases 0-5
    for i in range(6):
        engine.run_phase(EnginePhase(i))

    # Override route order
    engine.state.route_order = route_order

    # Run routing (phase 7)
    engine.run_phase(EnginePhase.PHASE_7_ROUTE)

    # Check results
    router = engine._router
    print(f"\nResults:")
    print(f"  Routed: {list(router.routes.keys())}")
    print(f"  Failed: {router.failed}")

    return len(router.routes)


def main():
    # Test 1: LED_CTRL first
    r1 = test_with_order("LED_CTRL first", ['LED_CTRL', 'LED_R'])

    # Test 2: LED_R first
    r2 = test_with_order("LED_R first", ['LED_R', 'LED_CTRL'])

    # Test 3: Just LED_CTRL
    r3 = test_with_order("LED_CTRL only", ['LED_CTRL'])

    # Test 4: Just LED_R
    r4 = test_with_order("LED_R only", ['LED_R'])

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  LED_CTRL first: {r1} routes")
    print(f"  LED_R first: {r2} routes")
    print(f"  LED_CTRL only: {r3} routes")
    print(f"  LED_R only: {r4} routes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
