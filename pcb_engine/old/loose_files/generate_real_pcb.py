#!/usr/bin/env python3
"""
Generate a Real, Functional PCB Design

This creates a simple but complete LED driver circuit:
- Resistor + LED on 2-layer PCB
- Clean, DRC-passing design

Board: 20mm x 15mm, 2-layer
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_led_driver():
    """Generate a simple LED driver PCB"""
    print("=" * 70)
    print("PCB ENGINE - SIMPLE LED DRIVER MODULE")
    print("=" * 70)

    from pcb_engine import (
        OutputPiston, OutputConfig,
        SilkscreenPiston, SilkscreenConfig
    )
    from pcb_engine.routing_types import TrackSegment, Route, Via

    # ==========================================================================
    # PARTS DATABASE - Simple LED Driver
    # ==========================================================================
    # Simple circuit: VCC -> R1 -> LED (D1) -> GND
    # With decoupling cap C1 between VCC and GND

    parts_db = {
        'parts': {
            # Current limiting resistor - 330 ohm
            'R1': {
                'value': '330R',
                'footprint': '0805',
                'description': 'LED Current Limiting Resistor',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'LED_A', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            },

            # LED - Red
            'D1': {
                'value': 'RED',
                'footprint': '0805',
                'description': 'Power Indicator LED',
                'pins': [
                    {'number': '1', 'net': 'LED_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},  # Anode
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)}      # Cathode
                ]
            },

            # Decoupling capacitor - 100nF
            'C1': {
                'value': '100nF',
                'footprint': '0603',
                'description': 'Decoupling Capacitor',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'GND', 'offset': (0.775, 0), 'size': (0.75, 0.9)}
                ]
            },
        },
        'nets': {
            'VCC': {'pins': [('R1', '1'), ('C1', '1')]},
            'GND': {'pins': [('D1', '2'), ('C1', '2')]},
            'LED_A': {'pins': [('R1', '2'), ('D1', '1')]},
        }
    }

    # ==========================================================================
    # COMPONENT PLACEMENT
    # ==========================================================================
    # Board is 20mm x 15mm, origin at (0,0)
    # R1 and D1 in a row, C1 below
    # Leave 2mm margin from edges for manufacturability

    # Component centers (must be away from board edge)
    placement = {
        'R1': (6.0, 7.5),       # Resistor on left
        'D1': (14.0, 7.5),      # LED on right (8mm spacing)
        'C1': (10.0, 12.0),     # Decoupling cap below, centered
    }

    # ==========================================================================
    # ROUTING - Calculated from actual pad positions
    # ==========================================================================
    # R1 at (6.0, 7.5): pad1 at (6.0-0.95, 7.5) = (5.05, 7.5), pad2 at (6.95, 7.5)
    # D1 at (14.0, 7.5): pad1 at (13.05, 7.5), pad2 at (14.95, 7.5)
    # C1 at (10.0, 12.0): pad1 at (9.225, 12.0), pad2 at (10.775, 12.0)

    routes = {}

    # VCC net - R1.1 to C1.1
    # From R1 pad1 (5.05, 7.5) to C1 pad1 (9.225, 12.0)
    routes['VCC'] = Route(
        net='VCC',
        segments=[
            # R1.1 down then across to C1.1
            TrackSegment(start=(5.05, 7.5), end=(5.05, 12.0), layer='F.Cu', width=0.3, net='VCC'),
            TrackSegment(start=(5.05, 12.0), end=(9.225, 12.0), layer='F.Cu', width=0.3, net='VCC'),
        ],
        success=True
    )

    # LED_A net - R1.2 to D1.1
    # From R1 pad2 (6.95, 7.5) to D1 pad1 (13.05, 7.5) - direct horizontal
    routes['LED_A'] = Route(
        net='LED_A',
        segments=[
            TrackSegment(start=(6.95, 7.5), end=(13.05, 7.5), layer='F.Cu', width=0.3, net='LED_A'),
        ],
        success=True
    )

    # GND net - D1.2 to C1.2
    # From D1 pad2 (14.95, 7.5) to C1 pad2 (10.775, 12.0)
    routes['GND'] = Route(
        net='GND',
        segments=[
            # D1.2 down then across to C1.2
            TrackSegment(start=(14.95, 7.5), end=(14.95, 12.0), layer='F.Cu', width=0.3, net='GND'),
            TrackSegment(start=(14.95, 12.0), end=(10.775, 12.0), layer='F.Cu', width=0.3, net='GND'),
        ],
        success=True
    )

    # No vias needed - single layer design
    vias = []

    # ==========================================================================
    # GENERATE OUTPUT
    # ==========================================================================
    print("\n1. Generating silkscreen...")
    silk_config = SilkscreenConfig()
    silk_piston = SilkscreenPiston(silk_config)
    silkscreen = silk_piston.generate(parts_db, placement)

    # Use default output directory: D:\Anas\tmp\output\<board_name>\
    config = OutputConfig(
        board_name='led_driver',
        board_width=20.0,
        board_height=15.0,
        board_origin_x=0.0,
        board_origin_y=0.0,
        trace_width=0.25,
        via_diameter=0.8,
        via_drill=0.4,
        clearance=0.15
    )

    print("\n2. Generating KiCad PCB file...")
    output_piston = OutputPiston(config)
    gen_result = output_piston.generate(parts_db, placement, routes, vias, silkscreen)

    if not gen_result.success:
        print(f"   ERROR: {gen_result.errors}")
        return None

    pcb_path = None
    for f in gen_result.files_generated:
        if f.endswith('.kicad_pcb'):
            pcb_path = f
            break

    if not pcb_path:
        print("   ERROR: No KiCad PCB file generated!")
        return None

    print(f"   Generated: {pcb_path}")

    # ==========================================================================
    # RUN KICAD DRC
    # ==========================================================================
    import subprocess
    import json

    kicad_cli = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    if os.path.exists(kicad_cli):
        print("\n3. Running KiCad 9 DRC...")
        # DRC report goes in same folder as PCB file
        pcb_dir = os.path.dirname(pcb_path)
        report_path = os.path.join(pcb_dir, 'led_driver_drc.json')

        result = subprocess.run(
            [kicad_cli, 'pcb', 'drc', '--format', 'json', '--severity-all',
             '--output', report_path, pcb_path],
            capture_output=True,
            text=True
        )

        print(f"   {result.stdout.strip()}" if result.stdout else "   DRC completed")

        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)

            violations = report.get('violations', [])
            unconnected = report.get('unconnected_items', [])
            errors = [v for v in violations if v.get('severity') == 'error']
            warnings = [v for v in violations if v.get('severity') == 'warning']

            print(f"\n   DRC Summary:")
            print(f"   - Errors:      {len(errors)}")
            print(f"   - Warnings:    {len(warnings)}")
            print(f"   - Unconnected: {len(unconnected)}")

            if errors:
                print("\n   ERRORS:")
                for e in errors[:5]:
                    print(f"   - {e.get('type')}: {e.get('description', '')[:60]}")

            if len(errors) == 0:
                print("\n   *** DRC PASSED! ***")
    else:
        print("\n3. KiCad CLI not found, skipping DRC")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PCB GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nBoard: Simple LED Driver Module")
    print(f"Size:  20mm x 15mm (2-layer)")
    print(f"\nComponents:")
    print(f"  - R1: 330R Current Limiting Resistor (0805)")
    print(f"  - D1: Red LED (0805)")
    print(f"  - C1: 100nF Decoupling Capacitor (0603)")
    print(f"\nCircuit: VCC -> R1 -> D1 -> GND")
    print(f"         VCC -> C1 -> GND (decoupling)")
    print(f"\nFile: {pcb_path}")
    print("\nOpen in KiCad to view and verify the design!")

    return pcb_path


if __name__ == '__main__':
    generate_led_driver()
