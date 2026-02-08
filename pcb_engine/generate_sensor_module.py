#!/usr/bin/env python3
"""
Generate a Moderate Complexity PCB - Power Supply with LED

Components:
- Voltage regulator (SOT-23-5)
- Input/output capacitors (0805)
- LED with resistor (0603)

Board: 25mm x 20mm, 2-layer
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_sensor_module():
    """Generate a moderate complexity PCB"""
    print("=" * 70)
    print("PCB ENGINE - POWER SUPPLY MODULE")
    print("=" * 70)

    from pcb_engine import (
        OutputPiston, OutputConfig,
        SilkscreenPiston, SilkscreenConfig
    )
    from pcb_engine.routing_types import TrackSegment, Route, Via

    # ==========================================================================
    # PARTS DATABASE
    # ==========================================================================
    # Circuit: 5V -> U1 (regulator) -> 3V3 -> R1 -> D1 -> GND
    #          C1 on input, C2 on output

    parts_db = {
        'parts': {
            # Voltage regulator - AP2112K-3.3 (SOT-23-5)
            'U1': {
                'value': 'AP2112K',
                'footprint': 'SOT-23-5',
                'description': '3.3V LDO',
                'pins': [
                    {'number': '1', 'net': '5V', 'offset': (-0.95, 1.1), 'size': (0.6, 0.7)},
                    {'number': '2', 'net': 'GND', 'offset': (0, 1.1), 'size': (0.6, 0.7)},
                    {'number': '3', 'net': '5V', 'offset': (0.95, 1.1), 'size': (0.6, 0.7)},  # EN tied to VIN
                    {'number': '4', 'net': 'NC', 'offset': (0.95, -1.1), 'size': (0.6, 0.7)},
                    {'number': '5', 'net': '3V3', 'offset': (-0.95, -1.1), 'size': (0.6, 0.7)},
                ]
            },

            # Input capacitor - 10uF (0805)
            'C1': {
                'value': '10uF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': '5V', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            },

            # Output capacitor - 10uF (0805)
            'C2': {
                'value': '10uF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            },

            # LED resistor - 1K (0603)
            'R1': {
                'value': '1K',
                'footprint': '0603',
                'pins': [
                    {'number': '1', 'net': '3V3', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'LED', 'offset': (0.775, 0), 'size': (0.75, 0.9)}
                ]
            },

            # Power LED (0603)
            'D1': {
                'value': 'GREEN',
                'footprint': '0603',
                'pins': [
                    {'number': '1', 'net': 'LED', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'GND', 'offset': (0.775, 0), 'size': (0.75, 0.9)}
                ]
            },
        },
        'nets': {
            '5V': {'pins': [('U1', '1'), ('U1', '3'), ('C1', '1')]},
            'GND': {'pins': [('U1', '2'), ('C1', '2'), ('C2', '2'), ('D1', '2')]},
            '3V3': {'pins': [('U1', '5'), ('C2', '1'), ('R1', '1')]},
            'LED': {'pins': [('R1', '2'), ('D1', '1')]},
            'NC': {'pins': [('U1', '4')]},
        }
    }

    # ==========================================================================
    # PLACEMENT
    # ==========================================================================
    # Board 25mm x 20mm
    placement = {
        'C1': (6.0, 10.0),      # Input cap left
        'U1': (12.5, 10.0),     # Regulator center
        'C2': (19.0, 10.0),     # Output cap right
        'R1': (12.5, 16.0),     # LED resistor below
        'D1': (19.0, 16.0),     # LED
    }

    # ==========================================================================
    # PAD POSITIONS
    # ==========================================================================
    def pad(comp, offset):
        cx, cy = placement[comp]
        return (cx + offset[0], cy + offset[1])

    # U1 (SOT-23-5)
    u1_vin = pad('U1', (-0.95, 1.1))    # pin 1: 5V
    u1_gnd = pad('U1', (0, 1.1))        # pin 2: GND
    u1_en = pad('U1', (0.95, 1.1))      # pin 3: EN (tied to 5V)
    u1_vout = pad('U1', (-0.95, -1.1))  # pin 5: 3V3

    # C1 (0805)
    c1_5v = pad('C1', (-0.95, 0))
    c1_gnd = pad('C1', (0.95, 0))

    # C2 (0805)
    c2_3v3 = pad('C2', (-0.95, 0))
    c2_gnd = pad('C2', (0.95, 0))

    # R1 (0603)
    r1_3v3 = pad('R1', (-0.775, 0))
    r1_led = pad('R1', (0.775, 0))

    # D1 (0603)
    d1_led = pad('D1', (-0.775, 0))
    d1_gnd = pad('D1', (0.775, 0))

    # ==========================================================================
    # ROUTING - Carefully planned to avoid crossings
    # ==========================================================================
    routes = {}

    # 5V: C1.1 -> U1.1 (top), U1.1 -> U1.3 (EN)
    # Route goes UP from C1, then RIGHT to U1
    routes['5V'] = Route(
        net='5V',
        segments=[
            # C1.1 up then right to U1.1
            TrackSegment(start=c1_5v, end=(c1_5v[0], 13.0), layer='F.Cu', width=0.4, net='5V'),
            TrackSegment(start=(c1_5v[0], 13.0), end=(u1_vin[0], 13.0), layer='F.Cu', width=0.4, net='5V'),
            TrackSegment(start=(u1_vin[0], 13.0), end=u1_vin, layer='F.Cu', width=0.4, net='5V'),
            # U1.1 to U1.3 (connect VIN to EN) - straight across top of chip
            TrackSegment(start=u1_vin, end=u1_en, layer='F.Cu', width=0.3, net='5V'),
        ],
        success=True
    )

    # 3V3: U1.5 -> C2.1, then separate branch to R1.1
    # Route goes DOWN from U1.5, then RIGHT to C2
    routes['3V3'] = Route(
        net='3V3',
        segments=[
            # U1.5 down then right to C2.1
            TrackSegment(start=u1_vout, end=(u1_vout[0], 7.5), layer='F.Cu', width=0.4, net='3V3'),
            TrackSegment(start=(u1_vout[0], 7.5), end=(c2_3v3[0], 7.5), layer='F.Cu', width=0.4, net='3V3'),
            TrackSegment(start=(c2_3v3[0], 7.5), end=c2_3v3, layer='F.Cu', width=0.4, net='3V3'),
            # Branch from C2 area up to R1.1 (going around, not through)
            TrackSegment(start=c2_3v3, end=(c2_3v3[0], 14.0), layer='F.Cu', width=0.3, net='3V3'),
            TrackSegment(start=(c2_3v3[0], 14.0), end=(r1_3v3[0], 14.0), layer='F.Cu', width=0.3, net='3V3'),
            TrackSegment(start=(r1_3v3[0], 14.0), end=r1_3v3, layer='F.Cu', width=0.3, net='3V3'),
        ],
        success=True
    )

    # LED: R1.2 -> D1.1 (simple horizontal)
    routes['LED'] = Route(
        net='LED',
        segments=[
            TrackSegment(start=r1_led, end=d1_led, layer='F.Cu', width=0.25, net='LED'),
        ],
        success=True
    )

    # GND: All ground pads connect to a bus at y=5.0 (bottom of board)
    gnd_y = 5.0
    routes['GND'] = Route(
        net='GND',
        segments=[
            # GND bus across bottom
            TrackSegment(start=(c1_gnd[0], gnd_y), end=(d1_gnd[0], gnd_y), layer='F.Cu', width=0.5, net='GND'),
            # C1.2 down to bus
            TrackSegment(start=c1_gnd, end=(c1_gnd[0], gnd_y), layer='F.Cu', width=0.4, net='GND'),
            # U1.2 down to bus (go down from pin)
            TrackSegment(start=u1_gnd, end=(u1_gnd[0], gnd_y), layer='F.Cu', width=0.4, net='GND'),
            # C2.2 down to bus
            TrackSegment(start=c2_gnd, end=(c2_gnd[0], gnd_y), layer='F.Cu', width=0.4, net='GND'),
            # D1.2 down to bus
            TrackSegment(start=d1_gnd, end=(d1_gnd[0], gnd_y), layer='F.Cu', width=0.3, net='GND'),
        ],
        success=True
    )

    # No vias needed - all routing on F.Cu
    vias = []

    # ==========================================================================
    # GENERATE
    # ==========================================================================
    print("\n1. Generating silkscreen...")
    silk_config = SilkscreenConfig()
    silk_piston = SilkscreenPiston(silk_config)
    silkscreen = silk_piston.generate(parts_db, placement)

    # Use default output directory: D:\Anas\tmp\output\<board_name>\
    config = OutputConfig(
        board_name='sensor_module',
        board_width=25.0,
        board_height=20.0,
        board_origin_x=0.0,
        board_origin_y=0.0,
        trace_width=0.25,
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

    print(f"   Generated: {pcb_path}")

    # ==========================================================================
    # RUN DRC
    # ==========================================================================
    import subprocess
    import json

    kicad_cli = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    if os.path.exists(kicad_cli):
        print("\n3. Running KiCad 9 DRC...")
        # DRC report goes in same folder as PCB file
        pcb_dir = os.path.dirname(pcb_path)
        report_path = os.path.join(pcb_dir, 'sensor_module_drc.json')

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
                for e in errors[:5]:
                    print(f"   - {e.get('type')}: {e.get('description', '')[:60]}")

            if len(errors) == 0 and len(unconnected) == 0:
                print("\n   *** DRC PASSED! ***")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE: 5 components, 5 nets, 2-layer board")
    print("=" * 70)
    print(f"File: {pcb_path}")

    return pcb_path


if __name__ == '__main__':
    generate_sensor_module()
