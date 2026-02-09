#!/usr/bin/env python3
"""
Generate a 10-Component PCB - Protected Power Supply

Components:
- U1: LM7805 voltage regulator (TO-220 footprint simplified as SIP-3)
- D1: 1N4007 reverse protection diode (DO-41 simplified as 2-pin)
- D2: Power LED (0603)
- R1: LED resistor 1K (0603)
- C1: Input electrolytic 100uF (radial 8mm)
- C2: Input ceramic 100nF (0805)
- C3: Output electrolytic 100uF (radial 8mm)
- C4: Output ceramic 100nF (0805)
- F1: Polyfuse 500mA (1812)
- J1: 2-pin power input header (2.54mm pitch)

Circuit:
  J1 -> F1 -> D1 -> C1/C2 -> U1 -> C3/C4 -> Output
                                    |
                                    +-> R1 -> D2 -> GND

Board: 50mm x 35mm, 2-layer
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_10comp_power():
    """Generate a 10-component protected power supply PCB"""
    print("=" * 70)
    print("PCB ENGINE - 10 COMPONENT PROTECTED POWER SUPPLY")
    print("=" * 70)

    from pcb_engine import (
        OutputPiston, OutputConfig,
        SilkscreenPiston, SilkscreenConfig
    )
    from pcb_engine.routing_types import TrackSegment, Route, Via

    # ==========================================================================
    # PARTS DATABASE - 10 Components
    # ==========================================================================
    parts_db = {
        'parts': {
            # J1: 2-pin power input header (2.54mm pitch)
            'J1': {
                'value': 'CONN_01X02',
                'footprint': 'Header_1x02_P2.54mm',
                'description': 'Power input',
                'pins': [
                    {'number': '1', 'net': 'VIN_RAW', 'offset': (0, 0), 'size': (1.7, 1.7)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.7, 1.7)},
                ]
            },

            # F1: Polyfuse 500mA (1812 package)
            'F1': {
                'value': '500mA',
                'footprint': '1812',
                'description': 'Resettable fuse',
                'pins': [
                    {'number': '1', 'net': 'VIN_RAW', 'offset': (-2.0, 0), 'size': (1.0, 1.5)},
                    {'number': '2', 'net': 'VIN_FUSED', 'offset': (2.0, 0), 'size': (1.0, 1.5)},
                ]
            },

            # D1: 1N4007 reverse protection diode (simplified 2-pin axial)
            'D1': {
                'value': '1N4007',
                'footprint': 'DO-41',
                'description': 'Reverse protection',
                'pins': [
                    {'number': '1', 'net': 'VIN_FUSED', 'offset': (-2.5, 0), 'size': (1.0, 1.5)},  # Anode
                    {'number': '2', 'net': 'VIN', 'offset': (2.5, 0), 'size': (1.0, 1.5)},  # Cathode
                ]
            },

            # U1: LM7805 voltage regulator (TO-220 as 3-pin inline)
            'U1': {
                'value': 'LM7805',
                'footprint': 'TO-220-3',
                'description': '5V regulator',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-2.54, 0), 'size': (1.5, 2.0)},    # Input
                    {'number': '2', 'net': 'GND', 'offset': (0, 0), 'size': (1.5, 2.0)},        # Ground
                    {'number': '3', 'net': '5V', 'offset': (2.54, 0), 'size': (1.5, 2.0)},      # Output
                ]
            },

            # C1: Input electrolytic 100uF (radial 8mm diameter)
            'C1': {
                'value': '100uF',
                'footprint': 'CP_Radial_D8.0mm_P3.5mm',
                'description': 'Input bulk cap',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-1.75, 0), 'size': (1.3, 1.3)},   # +
                    {'number': '2', 'net': 'GND', 'offset': (1.75, 0), 'size': (1.3, 1.3)},    # -
                ]
            },

            # C2: Input ceramic 100nF (0805)
            'C2': {
                'value': '100nF',
                'footprint': '0805',
                'description': 'Input decoupling',
                'pins': [
                    {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },

            # C3: Output electrolytic 100uF (radial 8mm diameter)
            'C3': {
                'value': '100uF',
                'footprint': 'CP_Radial_D8.0mm_P3.5mm',
                'description': 'Output bulk cap',
                'pins': [
                    {'number': '1', 'net': '5V', 'offset': (-1.75, 0), 'size': (1.3, 1.3)},    # +
                    {'number': '2', 'net': 'GND', 'offset': (1.75, 0), 'size': (1.3, 1.3)},    # -
                ]
            },

            # C4: Output ceramic 100nF (0805)
            'C4': {
                'value': '100nF',
                'footprint': '0805',
                'description': 'Output decoupling',
                'pins': [
                    {'number': '1', 'net': '5V', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },

            # R1: LED resistor 1K (0603)
            'R1': {
                'value': '1K',
                'footprint': '0603',
                'description': 'LED current limiter',
                'pins': [
                    {'number': '1', 'net': '5V', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                    {'number': '2', 'net': 'LED_A', 'offset': (0.775, 0), 'size': (0.75, 0.9)},
                ]
            },

            # D2: Power LED (0603)
            'D2': {
                'value': 'GREEN',
                'footprint': '0603',
                'description': 'Power indicator',
                'pins': [
                    {'number': '1', 'net': 'LED_A', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},  # Anode
                    {'number': '2', 'net': 'GND', 'offset': (0.775, 0), 'size': (0.75, 0.9)},     # Cathode
                ]
            },
        },
        'nets': {
            'VIN_RAW': {'pins': [('J1', '1'), ('F1', '1')]},
            'VIN_FUSED': {'pins': [('F1', '2'), ('D1', '1')]},
            'VIN': {'pins': [('D1', '2'), ('U1', '1'), ('C1', '1'), ('C2', '1')]},
            'GND': {'pins': [('J1', '2'), ('U1', '2'), ('C1', '2'), ('C2', '2'), ('C3', '2'), ('C4', '2'), ('D2', '2')]},
            '5V': {'pins': [('U1', '3'), ('C3', '1'), ('C4', '1'), ('R1', '1')]},
            'LED_A': {'pins': [('R1', '2'), ('D2', '1')]},
        }
    }

    # ==========================================================================
    # PLACEMENT - Logical flow left to right
    # ==========================================================================
    # Board: 50mm x 35mm
    # Layout: Input (left) -> Protection -> Regulator -> Output (right)
    #         LED indicator at bottom right

    placement = {
        # Input section (left side)
        'J1': (5.0, 17.5),          # Power connector, left edge
        'F1': (14.0, 17.5),         # Fuse after connector
        'D1': (24.0, 17.5),         # Diode after fuse

        # Input capacitors (near regulator input)
        'C1': (32.0, 24.0),         # Bulk cap above
        'C2': (32.0, 11.0),         # Ceramic cap below

        # Regulator (center)
        'U1': (32.0, 17.5),         # Main regulator

        # Output capacitors (near regulator output)
        'C3': (42.0, 24.0),         # Bulk cap above
        'C4': (42.0, 11.0),         # Ceramic cap below

        # LED indicator (bottom right)
        'R1': (42.0, 5.0),          # LED resistor
        'D2': (46.0, 5.0),          # LED
    }

    # ==========================================================================
    # PAD POSITIONS - Calculate absolute positions
    # ==========================================================================
    def pad(comp, offset):
        cx, cy = placement[comp]
        return (cx + offset[0], cy + offset[1])

    # J1 pads
    j1_vin = pad('J1', (0, 0))
    j1_gnd = pad('J1', (2.54, 0))

    # F1 pads
    f1_in = pad('F1', (-2.0, 0))
    f1_out = pad('F1', (2.0, 0))

    # D1 pads
    d1_a = pad('D1', (-2.5, 0))     # Anode (VIN_FUSED)
    d1_k = pad('D1', (2.5, 0))      # Cathode (VIN)

    # U1 pads (TO-220)
    u1_in = pad('U1', (-2.54, 0))   # Input
    u1_gnd = pad('U1', (0, 0))      # Ground
    u1_out = pad('U1', (2.54, 0))   # Output

    # C1 pads (electrolytic)
    c1_pos = pad('C1', (-1.75, 0))
    c1_neg = pad('C1', (1.75, 0))

    # C2 pads (ceramic)
    c2_pos = pad('C2', (-0.95, 0))
    c2_neg = pad('C2', (0.95, 0))

    # C3 pads (electrolytic)
    c3_pos = pad('C3', (-1.75, 0))
    c3_neg = pad('C3', (1.75, 0))

    # C4 pads (ceramic)
    c4_pos = pad('C4', (-0.95, 0))
    c4_neg = pad('C4', (0.95, 0))

    # R1 pads
    r1_5v = pad('R1', (-0.775, 0))
    r1_led = pad('R1', (0.775, 0))

    # D2 pads
    d2_a = pad('D2', (-0.775, 0))   # Anode
    d2_k = pad('D2', (0.775, 0))    # Cathode

    # ==========================================================================
    # ROUTING - Carefully planned for no crossings
    # ==========================================================================
    routes = {}

    # VIN_RAW: J1.1 -> F1.1
    # Must go AROUND J1.2 (GND pad at 7.54, 17.5)
    # Route: J1.1 down -> right past GND pad -> up -> to F1.1
    routes['VIN_RAW'] = Route(
        net='VIN_RAW',
        segments=[
            # Down from J1.1 to avoid GND pad
            TrackSegment(start=j1_vin, end=(j1_vin[0], 14.0), layer='F.Cu', width=0.5, net='VIN_RAW'),
            # Right past the GND pad
            TrackSegment(start=(j1_vin[0], 14.0), end=(f1_in[0], 14.0), layer='F.Cu', width=0.5, net='VIN_RAW'),
            # Up to F1.1
            TrackSegment(start=(f1_in[0], 14.0), end=f1_in, layer='F.Cu', width=0.5, net='VIN_RAW'),
        ],
        success=True
    )

    # VIN_FUSED: F1.2 -> D1.1 (simple horizontal on F.Cu)
    routes['VIN_FUSED'] = Route(
        net='VIN_FUSED',
        segments=[
            TrackSegment(start=f1_out, end=d1_a, layer='F.Cu', width=0.5, net='VIN_FUSED'),
        ],
        success=True
    )

    # VIN: D1.2 -> U1.1, C1.1, C2.1
    # Route on F.Cu with branches
    routes['VIN'] = Route(
        net='VIN',
        segments=[
            # D1 cathode to U1 input
            TrackSegment(start=d1_k, end=u1_in, layer='F.Cu', width=0.5, net='VIN'),
            # Branch up to C1
            TrackSegment(start=u1_in, end=(u1_in[0], 24.0), layer='F.Cu', width=0.4, net='VIN'),
            TrackSegment(start=(u1_in[0], 24.0), end=c1_pos, layer='F.Cu', width=0.4, net='VIN'),
            # Branch down to C2
            TrackSegment(start=u1_in, end=(u1_in[0], 11.0), layer='F.Cu', width=0.4, net='VIN'),
            TrackSegment(start=(u1_in[0], 11.0), end=c2_pos, layer='F.Cu', width=0.4, net='VIN'),
        ],
        success=True
    )

    # 5V: U1.3 -> C3.1, C4.1, R1.1
    # Route on F.Cu, go right from U1 to output caps
    routes['5V'] = Route(
        net='5V',
        segments=[
            # U1 output up to C3
            TrackSegment(start=u1_out, end=(u1_out[0], 20.0), layer='F.Cu', width=0.5, net='5V'),
            TrackSegment(start=(u1_out[0], 20.0), end=(c3_pos[0], 20.0), layer='F.Cu', width=0.5, net='5V'),
            TrackSegment(start=(c3_pos[0], 20.0), end=c3_pos, layer='F.Cu', width=0.5, net='5V'),
            # U1 output down to C4
            TrackSegment(start=u1_out, end=(u1_out[0], 15.0), layer='F.Cu', width=0.5, net='5V'),
            TrackSegment(start=(u1_out[0], 15.0), end=(c4_pos[0], 15.0), layer='F.Cu', width=0.4, net='5V'),
            TrackSegment(start=(c4_pos[0], 15.0), end=c4_pos, layer='F.Cu', width=0.4, net='5V'),
            # C4 down to R1 (LED)
            TrackSegment(start=c4_pos, end=(c4_pos[0], 5.0), layer='F.Cu', width=0.3, net='5V'),
            TrackSegment(start=(c4_pos[0], 5.0), end=r1_5v, layer='F.Cu', width=0.3, net='5V'),
        ],
        success=True
    )

    # LED_A: R1.2 -> D2.1 (simple horizontal)
    routes['LED_A'] = Route(
        net='LED_A',
        segments=[
            TrackSegment(start=r1_led, end=d2_a, layer='F.Cu', width=0.25, net='LED_A'),
        ],
        success=True
    )

    # GND: All ground pads connected via B.Cu ground plane
    # Use B.Cu for GND bus to avoid crossing power traces
    gnd_bus_y = 30.0  # GND bus near top on B.Cu
    routes['GND'] = Route(
        net='GND',
        segments=[
            # Main GND bus on B.Cu
            TrackSegment(start=(j1_gnd[0], gnd_bus_y), end=(d2_k[0], gnd_bus_y), layer='B.Cu', width=0.8, net='GND'),

            # J1.2 up to bus
            TrackSegment(start=j1_gnd, end=(j1_gnd[0], gnd_bus_y), layer='B.Cu', width=0.5, net='GND'),

            # U1.2 up to bus
            TrackSegment(start=u1_gnd, end=(u1_gnd[0], gnd_bus_y), layer='B.Cu', width=0.5, net='GND'),

            # C1.2 up to bus
            TrackSegment(start=c1_neg, end=(c1_neg[0], gnd_bus_y), layer='B.Cu', width=0.4, net='GND'),

            # C2.2 up to bus (longer route)
            TrackSegment(start=c2_neg, end=(c2_neg[0], gnd_bus_y), layer='B.Cu', width=0.4, net='GND'),

            # C3.2 up to bus
            TrackSegment(start=c3_neg, end=(c3_neg[0], gnd_bus_y), layer='B.Cu', width=0.4, net='GND'),

            # C4.2 up to bus (longer route)
            TrackSegment(start=c4_neg, end=(c4_neg[0], gnd_bus_y), layer='B.Cu', width=0.4, net='GND'),

            # D2.2 up to bus
            TrackSegment(start=d2_k, end=(d2_k[0], gnd_bus_y), layer='B.Cu', width=0.3, net='GND'),
        ],
        vias=[
            Via(position=j1_gnd, net='GND'),
            Via(position=u1_gnd, net='GND'),
            Via(position=c1_neg, net='GND'),
            Via(position=c2_neg, net='GND'),
            Via(position=c3_neg, net='GND'),
            Via(position=c4_neg, net='GND'),
            Via(position=d2_k, net='GND'),
        ],
        success=True
    )

    # ==========================================================================
    # GENERATE
    # ==========================================================================
    print("\n1. Generating silkscreen...")
    silk_config = SilkscreenConfig()
    silk_piston = SilkscreenPiston(silk_config)
    silkscreen = silk_piston.generate(parts_db, placement)

    config = OutputConfig(
        board_name='10comp_power',
        board_width=50.0,
        board_height=35.0,
        board_origin_x=0.0,
        board_origin_y=0.0,
        trace_width=0.3,
        clearance=0.2
    )

    print("\n2. Generating KiCad PCB file...")
    output_piston = OutputPiston(config)
    gen_result = output_piston.generate(parts_db, placement, routes, [], silkscreen)

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
        pcb_dir = os.path.dirname(pcb_path)
        report_path = os.path.join(pcb_dir, '10comp_power_drc.json')

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
                for e in errors[:10]:
                    print(f"   - {e.get('type')}: {e.get('description', '')[:70]}")

            if unconnected:
                print(f"\n   UNCONNECTED: {len(unconnected)} items")
                for u in unconnected[:5]:
                    print(f"   - {u.get('description', '')[:70]}")

            if len(errors) == 0 and len(unconnected) == 0:
                print("\n   *** DRC PASSED! ***")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE: 10 components, 6 nets, 2-layer board")
    print("=" * 70)
    print(f"Components: J1, F1, D1, U1, C1, C2, C3, C4, R1, D2")
    print(f"Board size: 50mm x 35mm")
    print(f"File: {pcb_path}")

    return pcb_path


if __name__ == '__main__':
    generate_10comp_power()
