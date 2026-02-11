#!/usr/bin/env python3
"""
Test script to generate a PCB file and run KiCad DRC on it.
"""

import os
import json
import subprocess
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_generate_and_drc():
    """Generate a PCB file and run KiCad DRC to verify fixes"""
    print("=" * 70)
    print("PCB ENGINE - KICAD DRC VALIDATION TEST")
    print("=" * 70)

    from pcb_engine import (
        OutputPiston, OutputConfig,
        SilkscreenPiston, SilkscreenConfig
    )
    from pcb_engine.routing_types import TrackSegment, Route, Via

    # Test data - simple design with proper net assignments
    parts_db = {
        'parts': {
            'C1': {
                'value': '100nF',
                'footprint': '0603',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-1.0, 0), 'size': (1.0, 1.5)},
                    {'number': '2', 'net': 'GND', 'offset': (1.0, 0), 'size': (1.0, 1.5)}
                ]
            },
            'R1': {
                'value': '10K',
                'footprint': '0603',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-1.0, 0), 'size': (1.0, 1.5)},
                    {'number': '2', 'net': 'SIGNAL', 'offset': (1.0, 0), 'size': (1.0, 1.5)}
                ]
            }
        },
        'nets': {
            'GND': {'pins': [('C1', '2')]},
            '3V3': {'pins': [('C1', '1'), ('R1', '1')]},
            'SIGNAL': {'pins': [('R1', '2')]}
        }
    }

    # Placement - components in center of board
    placement = {
        'C1': (15.0, 20.0),  # Center-ish
        'R1': (25.0, 20.0),  # Next to C1
    }

    # Routes - connect C1.1 to R1.1 (both on 3V3 net)
    # C1 at (15, 20), pad1 at offset -0.775, so pad1 center = 14.225
    # R1 at (25, 20), pad1 at offset -0.775, so pad1 center = 24.225
    # Route should go around C1.pad2 (at 15.775), not through it!
    routes = {
        '3V3': Route(
            net='3V3',
            segments=[
                # From C1.pad1 go up
                TrackSegment(
                    start=(14.225, 20.0),  # C1 pad 1 center
                    end=(14.225, 18.0),    # Go up
                    layer='F.Cu',
                    width=0.25,
                    net='3V3'
                ),
                # Horizontal across top
                TrackSegment(
                    start=(14.225, 18.0),
                    end=(24.225, 18.0),
                    layer='F.Cu',
                    width=0.25,
                    net='3V3'
                ),
                # Down to R1.pad1
                TrackSegment(
                    start=(24.225, 18.0),
                    end=(24.225, 20.0),  # R1 pad 1 center
                    layer='F.Cu',
                    width=0.25,
                    net='3V3'
                ),
            ],
            success=True
        )
    }

    # No vias for this simple test - they need tracks connected to them
    vias = []

    # Generate silkscreen
    silk_config = SilkscreenConfig()
    silk_piston = SilkscreenPiston(silk_config)
    silkscreen = silk_piston.generate(parts_db, placement)

    # Output configuration
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    config = OutputConfig(
        output_dir=output_dir,
        board_name='drc_test',
        board_width=50.0,
        board_height=40.0,
        board_origin_x=0.0,
        board_origin_y=0.0
    )

    # Generate PCB
    print("\n1. Generating KiCad PCB file...")
    output_piston = OutputPiston(config)
    gen_result = output_piston.generate(parts_db, placement, routes, vias, silkscreen)

    if not gen_result.success:
        print(f"   ERROR: {gen_result.errors}")
        return False

    # Find the generated PCB file
    pcb_path = None
    for f in gen_result.files_generated:
        if f.endswith('.kicad_pcb'):
            pcb_path = f
            break

    if not pcb_path:
        print("   ERROR: No KiCad PCB file generated!")
        return False

    print(f"   Generated: {pcb_path}")

    # Check if KiCad CLI is available
    kicad_cli = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    if not os.path.exists(kicad_cli):
        print("\n2. KiCad CLI not found, skipping DRC check")
        return True

    # Run KiCad DRC
    print("\n2. Running KiCad 9 DRC...")
    report_path = os.path.join(output_dir, 'drc_test_report.json')

    result = subprocess.run(
        [kicad_cli, 'pcb', 'drc', '--format', 'json', '--severity-all',
         '--output', report_path, pcb_path],
        capture_output=True,
        text=True
    )

    print(f"   {result.stdout.strip()}" if result.stdout else "   DRC completed")
    if result.stderr:
        print(f"   stderr: {result.stderr.strip()}")

    # Parse DRC report
    print("\n3. Analyzing DRC results...")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)

        violations = report.get('violations', [])
        unconnected = report.get('unconnected_items', [])

        errors = [v for v in violations if v.get('severity') == 'error']
        warnings = [v for v in violations if v.get('severity') == 'warning']

        print(f"\n   DRC Report Summary:")
        print(f"   - Errors:      {len(errors)}")
        print(f"   - Warnings:    {len(warnings)}")
        print(f"   - Unconnected: {len(unconnected)}")

        if errors:
            print("\n   ERRORS:")
            for e in errors:
                print(f"   - {e.get('type', 'unknown')}: {e.get('description', '')[:60]}")

        if warnings:
            print("\n   WARNINGS:")
            for w in warnings[:10]:  # Show first 10
                print(f"   - {w.get('type', 'unknown')}: {w.get('description', '')[:60]}")
            if len(warnings) > 10:
                print(f"   - ... and {len(warnings) - 10} more")

        # Determine pass/fail
        # We expect no errors. Some warnings may be acceptable.
        if len(errors) == 0:
            print("\n" + "=" * 70)
            print("RESULT: PASS - No DRC errors!")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print(f"RESULT: FAIL - {len(errors)} DRC errors found")
            print("=" * 70)
            return False
    else:
        print("   DRC report not found!")
        return False


if __name__ == '__main__':
    success = test_generate_and_drc()
    sys.exit(0 if success else 1)
