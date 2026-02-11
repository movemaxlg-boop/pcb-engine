#!/usr/bin/env python3
"""
MINIMAL Auto-Routing Test

The simplest possible test: 3 components, 2 nets, lots of space.
This is to verify the routing algorithms work at all.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Position:
    """Component position"""
    x: float
    y: float
    rotation: float = 0.0
    width: float = 2.5
    height: float = 1.5
    layer: str = 'F.Cu'


def generate_minimal():
    """Generate the simplest possible auto-routed PCB"""
    print("=" * 70)
    print("MINIMAL AUTO-ROUTING TEST")
    print("=" * 70)

    from pcb_engine import OutputPiston, OutputConfig
    from pcb_engine import SilkscreenPiston, SilkscreenConfig
    from pcb_engine.routing_piston import RoutingPiston
    from pcb_engine.routing_types import RoutingConfig

    # ==========================================================================
    # PARTS DATABASE - Just 3 components
    # ==========================================================================
    # Circuit: VIN -> R1 -> LED -> GND
    # Simplest possible: R1 and D1, plus connector for VIN/GND

    parts_db = {
        'parts': {
            'R1': {
                'value': '330R',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'LED_A', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'D1': {
                'value': 'LED',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'LED_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'J1': {
                'value': 'CONN',
                'footprint': 'Header_1x02',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (0, 0), 'size': (1.7, 1.7)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.7, 1.7)},
                ]
            },
        },
        'nets': {
            # VIN: 2 pins (J1.1 -> R1.1)
            'VIN': {'pins': [('J1', '1'), ('R1', '1')]},
            # GND: 2 pins (J1.2 -> D1.2)
            'GND': {'pins': [('J1', '2'), ('D1', '2')]},
            # LED_A: 2 pins (R1.2 -> D1.1)
            'LED_A': {'pins': [('R1', '2'), ('D1', '1')]},
        }
    }

    # ==========================================================================
    # PLACEMENT - Spread components far apart
    # ==========================================================================
    # Board: 50mm x 30mm with LOTS of space
    # J1 far left, R1 center, D1 right - all on same horizontal line
    placement = {
        'J1': Position(x=10.0, y=15.0, width=5.0, height=3.0),   # Connector left
        'R1': Position(x=25.0, y=15.0, width=2.5, height=1.5),   # Resistor center
        'D1': Position(x=40.0, y=15.0, width=2.5, height=1.5),   # LED right
    }

    placement_tuples = {ref: (pos.x, pos.y) for ref, pos in placement.items()}

    # ==========================================================================
    # AUTO-ROUTING
    # ==========================================================================
    print("\n1. Running AUTO-ROUTING...")
    print(f"   Board: 50mm x 30mm")
    print(f"   Components: {list(placement.keys())}")
    print(f"   Nets: {list(parts_db['nets'].keys())}")

    # Print pad positions
    print("\n   PAD POSITIONS:")
    for ref, pos in placement.items():
        part = parts_db['parts'][ref]
        for pin in part['pins']:
            pad_x = pos.x + pin['offset'][0]
            pad_y = pos.y + pin['offset'][1]
            print(f"   - {ref}.{pin['number']} ({pin['net']}): ({pad_x:.2f}, {pad_y:.2f})")

    routing_config = RoutingConfig(
        board_width=50.0,
        board_height=30.0,
        grid_size=0.5,  # Coarse grid for speed
        trace_width=0.3,
        clearance=0.2,
        algorithm='lee',  # Simple Lee algorithm
        allow_layer_change=True,
        via_cost=5.0,
    )

    routing_piston = RoutingPiston(routing_config)

    # Route smallest nets first
    net_order = ['LED_A', 'VIN', 'GND']

    routing_result = routing_piston.route(parts_db, {}, placement, net_order)

    print(f"\n   ROUTING RESULT:")
    print(f"   - Algorithm: {routing_config.algorithm}")
    print(f"   - Routed: {routing_result.routed_count}/{routing_result.total_count}")
    print(f"   - Success: {routing_result.success}")

    for net_name, route in routing_result.routes.items():
        status = "OK" if route.success else f"FAIL: {route.error}"
        print(f"   - {net_name}: {len(route.segments)} segments, {len(route.vias)} vias [{status}]")
        if route.success and route.segments:
            print(f"     First segment: {route.segments[0].start} -> {route.segments[0].end}")

    # ==========================================================================
    # GENERATE OUTPUT
    # ==========================================================================
    if routing_result.routed_count > 0:
        print("\n2. Generating KiCad PCB file...")

        silk_config = SilkscreenConfig()
        silk_piston = SilkscreenPiston(silk_config)
        silkscreen = silk_piston.generate(parts_db, placement_tuples)

        output_config = OutputConfig(
            board_name='minimal_auto',
            board_width=50.0,
            board_height=30.0,
            trace_width=0.3,
            clearance=0.2,
        )

        output_piston = OutputPiston(output_config)

        all_vias = []
        for route in routing_result.routes.values():
            all_vias.extend(route.vias)

        gen_result = output_piston.generate(
            parts_db,
            placement_tuples,
            routing_result.routes,
            all_vias,
            silkscreen
        )

        if gen_result.success:
            pcb_path = next((f for f in gen_result.files_generated if f.endswith('.kicad_pcb')), None)
            print(f"   Generated: {pcb_path}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MINIMAL AUTO-ROUTING TEST COMPLETE")
    print("=" * 70)
    print(f"Components: 3 (J1, R1, D1)")
    print(f"Nets: 3 (VIN, GND, LED_A)")
    print(f"Routed: {routing_result.routed_count}/{routing_result.total_count}")

    return routing_result.success


if __name__ == '__main__':
    generate_minimal()
