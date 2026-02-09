#!/usr/bin/env python3
"""
Auto-Routed PCB Test

This test uses the ACTUAL routing_piston algorithms (Lee, A*, etc.)
to automatically route - NO manual routes.

Only provides:
1. Parts database (components + nets)
2. Placement (component positions)

The routing_piston figures out the paths automatically.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Position:
    """Component position with required attributes for routing piston"""
    x: float
    y: float
    rotation: float = 0.0
    width: float = 2.5  # default 0805 body width
    height: float = 1.5  # default 0805 body height
    layer: str = 'F.Cu'


def generate_auto_routed():
    """Generate a PCB using automatic routing algorithms"""
    print("=" * 70)
    print("PCB ENGINE - AUTO-ROUTED TEST (NO MANUAL ROUTES)")
    print("=" * 70)

    from pcb_engine import OutputPiston, OutputConfig
    from pcb_engine import SilkscreenPiston, SilkscreenConfig
    from pcb_engine.routing_piston import RoutingPiston
    from pcb_engine.routing_types import RoutingConfig

    # ==========================================================================
    # PARTS DATABASE - Simple 5-component circuit
    # ==========================================================================
    # Circuit: VIN -> R1 -> LED -> GND
    #          VIN -> C1 -> GND
    #          VIN -> C2 -> GND

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
            'C1': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
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
            # Format: pins as list of tuples (component_ref, pin_number)
            'VIN': {'pins': [('J1', '1'), ('R1', '1'), ('C1', '1'), ('C2', '1')]},
            'GND': {'pins': [('J1', '2'), ('D1', '2'), ('C1', '2'), ('C2', '2')]},
            'LED_A': {'pins': [('R1', '2'), ('D1', '1')]},
        }
    }

    # ==========================================================================
    # PLACEMENT - Component positions with Position objects
    # ==========================================================================
    # Board: 40mm x 30mm (bigger to allow routing channels)
    # Layout: J1 on left, C1/C2 stacked vertically, R1 -> D1 on right
    # Key: Leave routing channels between components (at least 3mm gaps)
    placement = {
        'J1': Position(x=8.0, y=15.0, width=5.0, height=3.0),    # Connector on left
        'C1': Position(x=18.0, y=22.0, width=2.5, height=1.5),   # Cap top (VIN bypass)
        'C2': Position(x=18.0, y=8.0, width=2.5, height=1.5),    # Cap bottom (VIN bypass)
        'R1': Position(x=28.0, y=15.0, width=2.5, height=1.5),   # Resistor (center)
        'D1': Position(x=35.0, y=15.0, width=2.5, height=1.5),   # LED on right
    }

    # Also create tuple format for output piston (some pistons use different formats)
    placement_tuples = {ref: (pos.x, pos.y) for ref, pos in placement.items()}

    # ==========================================================================
    # AUTO-ROUTING - Let the algorithm do the work!
    # ==========================================================================
    print("\n1. Running AUTO-ROUTING (Lee algorithm)...")

    routing_config = RoutingConfig(
        board_width=40.0,
        board_height=30.0,
        grid_size=0.25,  # 0.25mm grid
        trace_width=0.25,  # 0.25mm trace width
        clearance=0.15,  # 0.15mm clearance
        algorithm='hybrid',  # Use hybrid for best results
        allow_layer_change=True,  # Allow vias
        via_cost=2.0,  # Lower via penalty to encourage layer changes
    )

    routing_piston = RoutingPiston(routing_config)

    # Build net order (route smaller nets first)
    net_order = ['LED_A', 'VIN', 'GND']  # 2-pin first, then 4-pin

    # No escapes needed for simple through-hole/SMD (escapes are for BGA)
    escapes = {}

    # THIS IS THE KEY - routing_piston.route() figures out all paths!
    routing_result = routing_piston.route(parts_db, escapes, placement, net_order)

    print(f"   Algorithm: {routing_config.algorithm}")
    print(f"   Nets to route: {len(parts_db['nets'])}")
    print(f"   Routes found: {len(routing_result.routes)}")
    print(f"   Routed: {routing_result.routed_count}/{routing_result.total_count}")
    print(f"   Success: {routing_result.success}")

    if not routing_result.success:
        print(f"   FAILED nets: {[n for n, r in routing_result.routes.items() if not r.success]}")

    # Show what the algorithm found
    for net_name, route in routing_result.routes.items():
        num_segments = len(route.segments) if hasattr(route, 'segments') else 0
        num_vias = len(route.vias) if hasattr(route, 'vias') else 0
        status = "OK" if route.success else f"FAIL: {route.error}"
        print(f"   - {net_name}: {num_segments} segments, {num_vias} vias [{status}]")

    # ==========================================================================
    # GENERATE OUTPUT
    # ==========================================================================
    print("\n2. Generating silkscreen...")
    silk_config = SilkscreenConfig()
    silk_piston = SilkscreenPiston(silk_config)
    silkscreen = silk_piston.generate(parts_db, placement_tuples)

    print("\n3. Generating KiCad PCB file...")
    output_config = OutputConfig(
        board_name='auto_routed',
        board_width=40.0,
        board_height=30.0,
        trace_width=0.25,
        clearance=0.15,
    )

    output_piston = OutputPiston(output_config)

    # Collect all vias from routes
    all_vias = []
    for route in routing_result.routes.values():
        if hasattr(route, 'vias'):
            all_vias.extend(route.vias)

    gen_result = output_piston.generate(
        parts_db,
        placement_tuples,  # Output piston uses tuple format
        routing_result.routes,
        all_vias,
        silkscreen
    )

    if not gen_result.success:
        print(f"   ERROR: {gen_result.errors}")
        return None

    pcb_path = None
    for f in gen_result.files_generated:
        if f.endswith('.kicad_pcb'):
            pcb_path = f
            break

    print(f"   Generated: {pcb_path}")

    # ==========================================================================
    # RUN DRC
    # ==========================================================================
    import subprocess
    import json

    kicad_cli = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    if os.path.exists(kicad_cli) and pcb_path:
        print("\n4. Running KiCad DRC...")
        pcb_dir = os.path.dirname(pcb_path)
        report_path = os.path.join(pcb_dir, 'auto_routed_drc.json')

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

            print(f"\n   DRC: {len(errors)} errors, {len(warnings)} warnings, {len(unconnected)} unconnected")

            if errors:
                print("\n   ERRORS:")
                for e in errors[:5]:
                    print(f"   - {e.get('type')}: {e.get('description', '')[:60]}")

            if unconnected:
                print(f"\n   UNCONNECTED: {len(unconnected)} items")
                for u in unconnected[:3]:
                    print(f"   - {u}")

            if len(errors) == 0 and len(unconnected) == 0:
                print("\n   *** DRC PASSED! ***")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("AUTO-ROUTING TEST COMPLETE")
    print("=" * 70)
    print(f"Components: 5")
    print(f"Nets: 3 (VIN, GND, LED_A)")
    print(f"Routing algorithm: {routing_config.algorithm}")
    print(f"Routing success: {routing_result.success}")
    print(f"Routed: {routing_result.routed_count}/{routing_result.total_count}")
    if pcb_path:
        print(f"File: {pcb_path}")

    return pcb_path


if __name__ == '__main__':
    generate_auto_routed()
