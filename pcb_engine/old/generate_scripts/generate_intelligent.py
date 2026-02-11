#!/usr/bin/env python3
"""
Test the Intelligent Router with KiCad output and DRC validation.

This generates a real PCB file using my knowledge-based routing algorithm.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Position:
    x: float
    y: float
    rotation: float = 0.0
    width: float = 2.5
    height: float = 1.5


def generate_intelligent():
    """Generate PCB using the intelligent router."""
    print("=" * 70)
    print("INTELLIGENT ROUTER - FULL PCB GENERATION")
    print("=" * 70)

    from pcb_engine import OutputPiston, OutputConfig
    from pcb_engine import SilkscreenPiston, SilkscreenConfig
    from pcb_engine.intelligent_router import IntelligentRouter
    from pcb_engine.routing_types import Route as RouteType, TrackSegment, Via as ViaType

    # ==========================================================================
    # PARTS DATABASE - 5 components (LED driver circuit)
    # ==========================================================================
    parts_db = {
        'parts': {
            'J1': {
                'value': 'CONN_2P',
                'footprint': 'Header_1x02',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (0, 0), 'size': (1.7, 1.7)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.7, 1.7)},
                ]
            },
            'C1': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'R1': {
                'value': '330R',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'LED_A', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'D1': {
                'value': 'LED_RED',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'LED_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C2': {
                'value': '10uF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
        },
        'nets': {
            'VIN': {'pins': [('J1', '1'), ('C1', '1'), ('R1', '1'), ('C2', '1')]},
            'GND': {'pins': [('J1', '2'), ('C1', '2'), ('D1', '2'), ('C2', '2')]},
            'LED_A': {'pins': [('R1', '2'), ('D1', '1')]},
        }
    }

    # ==========================================================================
    # PLACEMENT - Smart layout with routing channels
    # ==========================================================================
    # Board: 50mm x 35mm (spacious for clean routing)
    # Layout strategy:
    # - J1 (connector) on far left
    # - C1 and C2 (bypass caps) stacked vertically, offset from center
    # - R1 and D1 on right side (LED circuit)
    # - Wide routing channels between components
    placement = {
        'J1': Position(x=8.0, y=17.5, width=5.0, height=3.0),    # Connector far left
        'C1': Position(x=20.0, y=26.0, width=2.5, height=1.5),   # Cap top (lots of room)
        'C2': Position(x=20.0, y=9.0, width=2.5, height=1.5),    # Cap bottom (lots of room)
        'R1': Position(x=32.0, y=17.5, width=2.5, height=1.5),   # Resistor center
        'D1': Position(x=42.0, y=17.5, width=2.5, height=1.5),   # LED right
    }

    placement_tuples = {ref: (pos.x, pos.y) for ref, pos in placement.items()}

    # ==========================================================================
    # INTELLIGENT ROUTING
    # ==========================================================================
    print("\n1. Running INTELLIGENT ROUTER...")
    print(f"   Board: 50mm x 35mm")
    print(f"   Components: {len(parts_db['parts'])}")
    print(f"   Nets: {len(parts_db['nets'])}")

    # Print expected connections
    print("\n   NET CONNECTIONS:")
    for net_name, net_info in parts_db['nets'].items():
        pins = ', '.join(f"{p[0]}.{p[1]}" for p in net_info['pins'])
        print(f"   - {net_name}: {pins}")

    router = IntelligentRouter(
        board_width=50.0,
        board_height=35.0,
        trace_width=0.25,
        clearance=0.15
    )

    routes = router.route_all(parts_db, placement)

    # Convert routes to routing_types format
    converted_routes = {}
    all_vias = []

    for net_name, route in routes.items():
        converted = RouteType(
            net=net_name,
            segments=[
                TrackSegment(
                    start=seg.start,
                    end=seg.end,
                    layer=seg.layer,
                    width=seg.width,
                    net=seg.net
                ) for seg in route.segments
            ],
            vias=[
                ViaType(
                    position=via.position,
                    net=via.net,
                    diameter=via.diameter,
                    drill=via.drill,
                    from_layer=via.from_layer,
                    to_layer=via.to_layer
                ) for via in route.vias
            ],
            success=route.success,
            error=route.error
        )
        converted_routes[net_name] = converted
        all_vias.extend(converted.vias)

    # Print routing results
    print("\n   ROUTING RESULTS:")
    for net_name, route in routes.items():
        status = "OK" if route.success else f"FAIL: {route.error}"
        print(f"   - {net_name}: {len(route.segments)} segments, {len(route.vias)} vias [{status}]")

    routed = sum(1 for r in routes.values() if r.success)
    print(f"\n   Total: {routed}/{len(routes)} nets routed")

    # ==========================================================================
    # GENERATE KICAD FILES
    # ==========================================================================
    print("\n2. Generating KiCad PCB file...")

    silk_config = SilkscreenConfig()
    silk_piston = SilkscreenPiston(silk_config)
    silkscreen = silk_piston.generate(parts_db, placement_tuples)

    output_config = OutputConfig(
        board_name='intelligent_router',
        board_width=50.0,
        board_height=35.0,
        trace_width=0.25,
        clearance=0.15,
    )

    output_piston = OutputPiston(output_config)

    gen_result = output_piston.generate(
        parts_db,
        placement_tuples,
        converted_routes,
        all_vias,
        silkscreen
    )

    if not gen_result.success:
        print(f"   ERROR: {gen_result.errors}")
        return None

    pcb_path = next((f for f in gen_result.files_generated if f.endswith('.kicad_pcb')), None)
    print(f"   Generated: {pcb_path}")

    # ==========================================================================
    # RUN KICAD DRC
    # ==========================================================================
    import subprocess
    import json

    kicad_cli = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    if os.path.exists(kicad_cli) and pcb_path:
        print("\n3. Running KiCad DRC...")
        pcb_dir = os.path.dirname(pcb_path)
        report_path = os.path.join(pcb_dir, 'intelligent_router_drc.json')

        subprocess.run(
            [kicad_cli, 'pcb', 'drc', '--format', 'json', '--severity-all',
             '--output', report_path, pcb_path],
            capture_output=True, text=True
        )

        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)

            errors = [v for v in report.get('violations', []) if v.get('severity') == 'error']
            warnings = [v for v in report.get('violations', []) if v.get('severity') == 'warning']
            unconnected = report.get('unconnected_items', [])

            print(f"\n   DRC RESULTS:")
            print(f"   - Errors: {len(errors)}")
            print(f"   - Warnings: {len(warnings)}")
            print(f"   - Unconnected: {len(unconnected)}")

            if errors:
                print("\n   ERRORS:")
                for e in errors[:10]:
                    print(f"   - {e.get('type')}: {e.get('description', '')[:70]}")

            if unconnected:
                print(f"\n   UNCONNECTED ITEMS:")
                for u in unconnected[:5]:
                    print(f"   - {u}")

            if len(errors) == 0 and len(unconnected) == 0:
                print("\n   *** DRC PASSED! ***")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("INTELLIGENT ROUTER TEST COMPLETE")
    print("=" * 70)
    print(f"Components: {len(parts_db['parts'])}")
    print(f"Nets: {len(parts_db['nets'])}")
    print(f"Routed: {routed}/{len(routes)}")
    if pcb_path:
        print(f"File: {pcb_path}")

    return pcb_path


if __name__ == '__main__':
    generate_intelligent()
