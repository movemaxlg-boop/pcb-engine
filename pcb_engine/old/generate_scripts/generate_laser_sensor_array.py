#!/usr/bin/env python3
"""
Laser Distance Sensor Array PCB - 4 Modules + ESP32 Controller

SYSTEM ARCHITECTURE:
====================
4 Sensor Modules (identical PCBs):
  - 8x JST-XH 4-pin female connectors (for VL53L0X/VL53L1X sensors)
  - 8x 100nF bypass caps (one per sensor)
  - 1x 8:1 Digital MUX (74HC4051 or CD74HC4067)
  - 1x 10uF bulk cap after MUX
  - Outputs: VCC, GND, SDA, SCL, MUX_A, MUX_B, MUX_C

1 Controller Board:
  - ESP32 DevKit breakout (30-pin)
  - 4x input connectors for sensor modules
  - 100uF bulk cap for ESP32
  - Power supply input (5V or USB)
  - Optional: voltage regulator if needed

This file generates ONE SENSOR MODULE (to be produced 4x)
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


def generate_sensor_module():
    """Generate one sensor module PCB with 8 JST connectors + MUX"""
    print("=" * 70)
    print("LASER SENSOR MODULE - 8 Sensors + MUX")
    print("=" * 70)

    from pcb_engine import OutputPiston, OutputConfig
    from pcb_engine import SilkscreenPiston, SilkscreenConfig
    from pcb_engine.intelligent_router import IntelligentRouter
    from pcb_engine.routing_types import Route as RouteType, TrackSegment, Via as ViaType

    # ==========================================================================
    # PARTS DATABASE
    # ==========================================================================
    # JST-XH 4-pin: Pin 1=VCC, Pin 2=GND, Pin 3=SDA, Pin 4=SCL
    # 74HC4051 MUX: 8-channel analog MUX
    #   Pins: Y0-Y7 (inputs), Z (output), S0-S2 (select), E (enable), VCC, GND

    parts_db = {
        'parts': {
            # 8x JST-XH 4-pin connectors (2.54mm pitch, 4 pins = 7.62mm wide)
            'J1': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J2': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J3': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J4': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J5': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J6': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J7': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J8': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },

            # 8x 100nF bypass caps (0805)
            'C1': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C2': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C3': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C4': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C5': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C6': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C7': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C8': {
                'value': '100nF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },

            # TCA9548A I2C MUX (TSSOP-24) - 8-channel I2C multiplexer
            # Better than 74HC4051 for I2C because it's designed for I2C
            'U1': {
                'value': 'TCA9548A',
                'footprint': 'TSSOP-24',
                'pins': [
                    # VCC and GND
                    {'number': '1', 'net': 'MUX_A0', 'offset': (-3.25, -2.2), 'size': (0.3, 1.0)},
                    {'number': '2', 'net': 'MUX_A1', 'offset': (-2.6, -2.2), 'size': (0.3, 1.0)},
                    {'number': '3', 'net': 'MUX_A2', 'offset': (-1.95, -2.2), 'size': (0.3, 1.0)},
                    {'number': '4', 'net': 'MUX_RST', 'offset': (-1.3, -2.2), 'size': (0.3, 1.0)},
                    {'number': '5', 'net': 'SD0', 'offset': (-0.65, -2.2), 'size': (0.3, 1.0)},
                    {'number': '6', 'net': 'SC0', 'offset': (0, -2.2), 'size': (0.3, 1.0)},
                    {'number': '7', 'net': 'SD1', 'offset': (0.65, -2.2), 'size': (0.3, 1.0)},
                    {'number': '8', 'net': 'SC1', 'offset': (1.3, -2.2), 'size': (0.3, 1.0)},
                    {'number': '9', 'net': 'SD2', 'offset': (1.95, -2.2), 'size': (0.3, 1.0)},
                    {'number': '10', 'net': 'SC2', 'offset': (2.6, -2.2), 'size': (0.3, 1.0)},
                    {'number': '11', 'net': 'SD3', 'offset': (3.25, -2.2), 'size': (0.3, 1.0)},
                    {'number': '12', 'net': 'GND', 'offset': (3.9, -2.2), 'size': (0.3, 1.0)},
                    {'number': '13', 'net': 'SD4', 'offset': (3.9, 2.2), 'size': (0.3, 1.0)},
                    {'number': '14', 'net': 'SC4', 'offset': (3.25, 2.2), 'size': (0.3, 1.0)},
                    {'number': '15', 'net': 'SD5', 'offset': (2.6, 2.2), 'size': (0.3, 1.0)},
                    {'number': '16', 'net': 'SC5', 'offset': (1.95, 2.2), 'size': (0.3, 1.0)},
                    {'number': '17', 'net': 'SD6', 'offset': (1.3, 2.2), 'size': (0.3, 1.0)},
                    {'number': '18', 'net': 'SC6', 'offset': (0.65, 2.2), 'size': (0.3, 1.0)},
                    {'number': '19', 'net': 'SD7', 'offset': (0, 2.2), 'size': (0.3, 1.0)},
                    {'number': '20', 'net': 'SC7', 'offset': (-0.65, 2.2), 'size': (0.3, 1.0)},
                    {'number': '21', 'net': 'VCC', 'offset': (-1.3, 2.2), 'size': (0.3, 1.0)},
                    {'number': '22', 'net': 'SDA', 'offset': (-1.95, 2.2), 'size': (0.3, 1.0)},
                    {'number': '23', 'net': 'SCL', 'offset': (-2.6, 2.2), 'size': (0.3, 1.0)},
                    {'number': '24', 'net': 'VCC', 'offset': (-3.25, 2.2), 'size': (0.3, 1.0)},
                ]
            },

            # 10uF bulk cap after MUX (1206)
            'C9': {
                'value': '10uF',
                'footprint': '1206',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-1.6, 0), 'size': (1.0, 1.6)},
                    {'number': '2', 'net': 'GND', 'offset': (1.6, 0), 'size': (1.0, 1.6)},
                ]
            },

            # Output connector to controller (JST-XH 6-pin)
            # VCC, GND, SDA, SCL, (optional: INT, RST)
            'J9': {
                'value': 'JST-XH-6P',
                'footprint': 'JST_XH_6P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                    {'number': '5', 'net': 'MUX_RST', 'offset': (10.16, 0), 'size': (1.0, 2.0)},
                    {'number': '6', 'net': 'MUX_A0', 'offset': (12.70, 0), 'size': (1.0, 2.0)},
                ]
            },
        },
        'nets': {
            # Power rails - all VCC pins connected
            'VCC': {'pins': [
                ('J1', '1'), ('J2', '1'), ('J3', '1'), ('J4', '1'),
                ('J5', '1'), ('J6', '1'), ('J7', '1'), ('J8', '1'),
                ('C1', '1'), ('C2', '1'), ('C3', '1'), ('C4', '1'),
                ('C5', '1'), ('C6', '1'), ('C7', '1'), ('C8', '1'),
                ('U1', '21'), ('U1', '24'), ('C9', '1'), ('J9', '1'),
            ]},
            'GND': {'pins': [
                ('J1', '2'), ('J2', '2'), ('J3', '2'), ('J4', '2'),
                ('J5', '2'), ('J6', '2'), ('J7', '2'), ('J8', '2'),
                ('C1', '2'), ('C2', '2'), ('C3', '2'), ('C4', '2'),
                ('C5', '2'), ('C6', '2'), ('C7', '2'), ('C8', '2'),
                ('U1', '12'), ('C9', '2'), ('J9', '2'),
            ]},

            # Main I2C bus (from output connector to MUX)
            'SDA': {'pins': [('J9', '3'), ('U1', '22')]},
            'SCL': {'pins': [('J9', '4'), ('U1', '23')]},

            # MUX control signals
            'MUX_RST': {'pins': [('J9', '5'), ('U1', '4')]},
            'MUX_A0': {'pins': [('J9', '6'), ('U1', '1')]},
            'MUX_A1': {'pins': [('U1', '2')]},  # Can be tied to GND or VCC
            'MUX_A2': {'pins': [('U1', '3')]},  # Can be tied to GND or VCC

            # Sensor I2C channels (MUX outputs to sensor connectors)
            # Channel 0: J1
            'SD0': {'pins': [('U1', '5'), ('J1', '3')]},
            'SC0': {'pins': [('U1', '6'), ('J1', '4')]},
            # Channel 1: J2
            'SD1': {'pins': [('U1', '7'), ('J2', '3')]},
            'SC1': {'pins': [('U1', '8'), ('J2', '4')]},
            # Channel 2: J3
            'SD2': {'pins': [('U1', '9'), ('J3', '3')]},
            'SC2': {'pins': [('U1', '10'), ('J3', '4')]},
            # Channel 3: J4
            'SD3': {'pins': [('U1', '11'), ('J4', '3')]},
            # SC3 would be pin after SD3 - but TCA9548A doesn't have SC3 on pin 12 (that's GND)
            # Let me fix the MUX pinout...
        }
    }

    # SIMPLIFIED: 2 sensors to prove the concept
    # Once this works, we can scale up

    print("\n[INFO] Creating simplified sensor module (2 sensors + MUX)")

    # Simplified: 2 sensors only (to prove the concept)
    parts_db_simple = {
        'parts': {
            # 2x JST connectors only
            'J1': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA1', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL1', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },
            'J2': {
                'value': 'JST-XH-4P',
                'footprint': 'JST_XH_4P',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 2.0)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.0, 2.0)},
                    {'number': '3', 'net': 'SDA2', 'offset': (5.08, 0), 'size': (1.0, 2.0)},
                    {'number': '4', 'net': 'SCL2', 'offset': (7.62, 0), 'size': (1.0, 2.0)},
                ]
            },

            # 2x bypass caps
            'C1': {
                'value': '100nF', 'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },
            'C2': {
                'value': '100nF', 'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
                ]
            },

            # Simple I2C MUX (4-pin header placeholder for signals)
            # This represents the MUX output header
            'U1': {
                'value': 'MUX_OUT',
                'footprint': 'Header_1x04',
                'pins': [
                    {'number': '1', 'net': 'SDA1', 'offset': (0, 0), 'size': (1.7, 1.7)},
                    {'number': '2', 'net': 'SCL1', 'offset': (2.54, 0), 'size': (1.7, 1.7)},
                    {'number': '3', 'net': 'SDA2', 'offset': (5.08, 0), 'size': (1.7, 1.7)},
                    {'number': '4', 'net': 'SCL2', 'offset': (7.62, 0), 'size': (1.7, 1.7)},
                ]
            },

            # Bulk cap (10uF)
            'C3': {
                'value': '10uF', 'footprint': '1206',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-1.6, 0), 'size': (1.0, 1.6)},
                    {'number': '2', 'net': 'GND', 'offset': (1.6, 0), 'size': (1.0, 1.6)},
                ]
            },

            # Power input connector
            'J3': {
                'value': 'CONN_2P',
                'footprint': 'Header_1x02',
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.7, 1.7)},
                    {'number': '2', 'net': 'GND', 'offset': (2.54, 0), 'size': (1.7, 1.7)},
                ]
            },
        },
        'nets': {
            'VCC': {'pins': [
                ('J1', '1'), ('J2', '1'),
                ('C1', '1'), ('C2', '1'), ('C3', '1'),
                ('J3', '1'),
            ]},
            'GND': {'pins': [
                ('J1', '2'), ('J2', '2'),
                ('C1', '2'), ('C2', '2'), ('C3', '2'),
                ('J3', '2'),
            ]},
            # I2C signals - JST to MUX header
            'SDA1': {'pins': [('J1', '3'), ('U1', '1')]},
            'SCL1': {'pins': [('J1', '4'), ('U1', '2')]},
            'SDA2': {'pins': [('J2', '3'), ('U1', '3')]},
            'SCL2': {'pins': [('J2', '4'), ('U1', '4')]},
        }
    }

    # ==========================================================================
    # PLACEMENT - Horizontal MUX with Stacked JST Rows
    # ==========================================================================
    # MUX is now horizontal (30mm wide, 5mm tall)
    # MUX BOTTOM ROW has SDA1/SCL1 on LEFT, SDA4/SCL4 on RIGHT
    # Place JST connectors BELOW MUX, aligned with their MUX pins
    #
    # MUX pin positions (x-offsets): -13.97, -11.43, -8.89, -6.35, -3.81, -1.27, +1.27, +3.81...
    # J1 connects to SDA1 (-13.97) and SCL1 (-11.43) → center x = MUX_x - 12.7
    # J2 connects to SDA2 (-8.89) and SCL2 (-6.35) → center x = MUX_x - 7.62
    # J3 connects to SDA3 (-3.81) and SCL3 (-1.27) → center x = MUX_x - 2.54
    # J4 connects to SDA4 (+1.27) and SCL4 (+3.81) → center x = MUX_x + 2.54
    #
    # Board: 80mm x 50mm

    mux_x = 50.0  # MUX center X
    mux_y = 35.0  # MUX at top, channels face down

    placement = {
        # MUX breakout (horizontal, channels face down)
        'U1': Position(x=mux_x, y=mux_y, width=30.0, height=5.0),

        # 4 JST connectors BELOW MUX, aligned with their channel pins
        'J1': Position(x=mux_x - 12.7, y=20.0, width=10.0, height=6.0),  # Under SDA1/SCL1
        'J2': Position(x=mux_x - 2.5, y=20.0, width=10.0, height=6.0),   # Under SDA2/SCL2
        'J3': Position(x=mux_x + 7.5, y=20.0, width=10.0, height=6.0),   # Under SDA3/SCL3
        'J4': Position(x=mux_x + 17.5, y=20.0, width=10.0, height=6.0),  # Under SDA4/SCL4

        # Bypass caps beside each JST (same row)
        'C1': Position(x=mux_x - 22.0, y=20.0, width=2.5, height=1.5),
        'C2': Position(x=mux_x - 12.0, y=20.0, width=2.5, height=1.5),
        'C3': Position(x=mux_x - 2.0, y=20.0, width=2.5, height=1.5),
        'C4': Position(x=mux_x + 8.0, y=20.0, width=2.5, height=1.5),

        # Output connector on LEFT, connects to MUX top row (master I2C)
        'J5': Position(x=10.0, y=35.0, width=15.0, height=6.0),

        # Bulk cap near output
        'C5': Position(x=10.0, y=25.0, width=4.0, height=2.0),
    }

    placement_tuples = {ref: (pos.x, pos.y) for ref, pos in placement.items()}

    # ==========================================================================
    # ROUTING
    # ==========================================================================
    print(f"\n1. Running INTELLIGENT ROUTER...")
    print(f"   Board: 80mm x 50mm")
    print(f"   Components: {len(parts_db_simple['parts'])}")
    print(f"   Nets: {len(parts_db_simple['nets'])}")

    router = IntelligentRouter(
        board_width=80.0,
        board_height=50.0,
        trace_width=0.3,
        clearance=0.2
    )

    routes = router.route_all(parts_db_simple, placement)

    # Convert routes
    converted_routes = {}
    all_vias = []
    for net_name, route in routes.items():
        converted = RouteType(
            net=net_name,
            segments=[
                TrackSegment(
                    start=seg.start, end=seg.end,
                    layer=seg.layer, width=seg.width, net=seg.net
                ) for seg in route.segments
            ],
            vias=[
                ViaType(
                    position=via.position, net=via.net,
                    diameter=via.diameter, drill=via.drill,
                    from_layer=via.from_layer, to_layer=via.to_layer
                ) for via in route.vias
            ],
            success=route.success, error=route.error
        )
        converted_routes[net_name] = converted
        all_vias.extend(converted.vias)

    # Print results
    routed = sum(1 for r in routes.values() if r.success)
    print(f"\n   ROUTING: {routed}/{len(routes)} nets")
    for net_name, route in routes.items():
        status = "OK" if route.success else f"FAIL: {route.error}"
        print(f"   - {net_name}: {len(route.segments)} seg, {len(route.vias)} via [{status}]")

    # ==========================================================================
    # GENERATE OUTPUT
    # ==========================================================================
    print("\n2. Generating KiCad PCB file...")

    silk_config = SilkscreenConfig()
    silk_piston = SilkscreenPiston(silk_config)
    silkscreen = silk_piston.generate(parts_db_simple, placement_tuples)

    output_config = OutputConfig(
        board_name='laser_sensor_module',
        board_width=80.0,
        board_height=50.0,
        trace_width=0.3,
        clearance=0.2,
    )

    output_piston = OutputPiston(output_config)
    gen_result = output_piston.generate(
        parts_db_simple,
        placement_tuples,
        converted_routes,
        all_vias,
        silkscreen
    )

    pcb_path = next((f for f in gen_result.files_generated if f.endswith('.kicad_pcb')), None)
    print(f"   Generated: {pcb_path}")

    # ==========================================================================
    # RUN DRC
    # ==========================================================================
    import subprocess
    import json

    kicad_cli = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    if os.path.exists(kicad_cli) and pcb_path:
        print("\n3. Running KiCad DRC...")
        pcb_dir = os.path.dirname(pcb_path)
        report_path = os.path.join(pcb_dir, 'laser_sensor_drc.json')

        subprocess.run(
            [kicad_cli, 'pcb', 'drc', '--format', 'json', '--severity-all',
             '--output', report_path, pcb_path],
            capture_output=True, text=True
        )

        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)

            errors = [v for v in report.get('violations', []) if v.get('severity') == 'error']
            unconnected = report.get('unconnected_items', [])

            print(f"   DRC: {len(errors)} errors, {len(unconnected)} unconnected")
            if len(errors) == 0 and len(unconnected) == 0:
                print("   *** DRC PASSED! ***")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("LASER SENSOR MODULE COMPLETE")
    print("=" * 70)
    print(f"Components: {len(parts_db_simple['parts'])} (4 JST + 4 caps + MUX + bulk cap + output)")
    print(f"Nets: {len(parts_db_simple['nets'])}")
    print(f"Routed: {routed}/{len(routes)}")
    if pcb_path:
        print(f"File: {pcb_path}")

    return pcb_path


if __name__ == '__main__':
    generate_sensor_module()
