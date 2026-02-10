"""
REAL-WORLD 20 COMPONENT PCB TEST
================================

Tests the PCB Engine with a realistic 2-layer board:
- ESP32-WROOM-32 module
- USB-C connector with ESD protection
- 3.3V LDO regulator
- Multiple sensors (I2C, analog)
- Status LEDs
- Decoupling capacitors
- Pull-up resistors

This is a stress test to find the engine's real limits.

Run with:
    cd D:/Anas/projects/pcb-engine
    python test_real_20_component_board.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcb_engine.pcb_engine import PCBEngine, EngineConfig
from pcb_engine.routing_planner import RoutingPlanner, create_routing_plan


def create_20_component_parts_db():
    """
    Create a realistic 20-component ESP32 sensor board.
    Uses the correct parts_db format with full pin specifications.

    Components:
    - U1: ESP32-WROOM-32 (simplified to 16 pins for test)
    - U2: AMS1117-3.3 (LDO regulator)
    - U3: USBLC6-2SC6 (USB ESD protection)
    - U4: BME280 (temp/humidity/pressure sensor)
    - J1: USB-C connector (simplified to 8 pins)
    - C1-C6: Decoupling capacitors
    - R1-R6: Pull-up and LED resistors
    - LED1-LED2: Status LEDs
    """

    parts_db = {
        'parts': {
            # Main MCU - ESP32 (simplified 16-pin representation for QFN-style)
            'U1': {
                'name': 'ESP32',
                'footprint': 'QFN-32',
                'value': 'ESP32-WROOM-32',
                'description': 'ESP32 WiFi/BT Module',
                'size': (18.0, 25.5),  # Real ESP32-WROOM size
                'pins': [
                    # Left side pins
                    {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': -8.5, 'offset_y': -10.0}},
                    {'number': '2', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                     'physical': {'offset_x': -8.5, 'offset_y': -7.5}},
                    {'number': '3', 'name': 'EN', 'type': 'input', 'net': 'EN',
                     'physical': {'offset_x': -8.5, 'offset_y': -5.0}},
                    {'number': '4', 'name': 'IO0', 'type': 'bidirectional', 'net': 'BOOT',
                     'physical': {'offset_x': -8.5, 'offset_y': -2.5}},
                    {'number': '5', 'name': 'IO2', 'type': 'bidirectional', 'net': 'LED1_CTRL',
                     'physical': {'offset_x': -8.5, 'offset_y': 0.0}},
                    {'number': '6', 'name': 'IO4', 'type': 'bidirectional', 'net': 'LED2_CTRL',
                     'physical': {'offset_x': -8.5, 'offset_y': 2.5}},
                    {'number': '7', 'name': 'IO18', 'type': 'bidirectional', 'net': 'USB_DN',
                     'physical': {'offset_x': -8.5, 'offset_y': 5.0}},
                    {'number': '8', 'name': 'IO19', 'type': 'bidirectional', 'net': 'USB_DP',
                     'physical': {'offset_x': -8.5, 'offset_y': 7.5}},
                    # Right side pins
                    {'number': '9', 'name': 'IO21', 'type': 'bidirectional', 'net': 'I2C_SDA',
                     'physical': {'offset_x': 8.5, 'offset_y': 7.5}},
                    {'number': '10', 'name': 'IO22', 'type': 'bidirectional', 'net': 'I2C_SCL',
                     'physical': {'offset_x': 8.5, 'offset_y': 5.0}},
                    {'number': '11', 'name': 'RXD0', 'type': 'input', 'net': 'UART_RX',
                     'physical': {'offset_x': 8.5, 'offset_y': 2.5}},
                    {'number': '12', 'name': 'TXD0', 'type': 'output', 'net': 'UART_TX',
                     'physical': {'offset_x': 8.5, 'offset_y': 0.0}},
                    {'number': '13', 'name': 'GND2', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': 8.5, 'offset_y': -2.5}},
                    {'number': '14', 'name': 'IO25', 'type': 'bidirectional', 'net': 'NC1',
                     'physical': {'offset_x': 8.5, 'offset_y': -5.0}},
                    {'number': '15', 'name': 'IO26', 'type': 'bidirectional', 'net': 'NC2',
                     'physical': {'offset_x': 8.5, 'offset_y': -7.5}},
                    {'number': '16', 'name': 'IO27', 'type': 'bidirectional', 'net': 'NC3',
                     'physical': {'offset_x': 8.5, 'offset_y': -10.0}},
                ],
            },

            # LDO Regulator - SOT-223
            'U2': {
                'name': 'LDO',
                'footprint': 'SOT-223',
                'value': 'AMS1117-3.3',
                'description': '3.3V LDO Regulator',
                'size': (6.5, 3.5),
                'pins': [
                    {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': -2.3, 'offset_y': -1.5}},
                    {'number': '2', 'name': 'VOUT', 'type': 'power_out', 'net': '3V3',
                     'physical': {'offset_x': 0.0, 'offset_y': -1.5}},
                    {'number': '3', 'name': 'VIN', 'type': 'power_in', 'net': 'VBUS',
                     'physical': {'offset_x': 2.3, 'offset_y': -1.5}},
                    {'number': '4', 'name': 'VOUT2', 'type': 'power_out', 'net': '3V3',
                     'physical': {'offset_x': 0.0, 'offset_y': 1.5}},
                ],
            },

            # USB ESD Protection - SOT-23-6
            'U3': {
                'name': 'ESD',
                'footprint': 'SOT-23-6',
                'value': 'USBLC6-2SC6',
                'description': 'USB ESD Protection',
                'size': (3.0, 1.75),
                'pins': [
                    {'number': '1', 'name': 'IO1', 'type': 'bidirectional', 'net': 'USB_DP',
                     'physical': {'offset_x': -1.0, 'offset_y': -0.95}},
                    {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': 0.0, 'offset_y': -0.95}},
                    {'number': '3', 'name': 'IO2', 'type': 'bidirectional', 'net': 'USB_DN',
                     'physical': {'offset_x': 1.0, 'offset_y': -0.95}},
                    {'number': '4', 'name': 'IO3', 'type': 'bidirectional', 'net': 'NC_ESD1',
                     'physical': {'offset_x': 1.0, 'offset_y': 0.95}},
                    {'number': '5', 'name': 'VBUS', 'type': 'power_in', 'net': 'VBUS',
                     'physical': {'offset_x': 0.0, 'offset_y': 0.95}},
                    {'number': '6', 'name': 'IO4', 'type': 'bidirectional', 'net': 'NC_ESD2',
                     'physical': {'offset_x': -1.0, 'offset_y': 0.95}},
                ],
            },

            # BME280 Sensor - LGA-8
            'U4': {
                'name': 'BME280',
                'footprint': 'LGA-8',
                'value': 'BME280',
                'description': 'Temp/Humidity/Pressure Sensor',
                'size': (2.5, 2.5),
                'pins': [
                    {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': -0.975, 'offset_y': -0.65}},
                    {'number': '2', 'name': 'CSB', 'type': 'input', 'net': 'BME_CS',
                     'physical': {'offset_x': -0.325, 'offset_y': -0.65}},
                    {'number': '3', 'name': 'SDI', 'type': 'bidirectional', 'net': 'I2C_SDA',
                     'physical': {'offset_x': 0.325, 'offset_y': -0.65}},
                    {'number': '4', 'name': 'SCK', 'type': 'input', 'net': 'I2C_SCL',
                     'physical': {'offset_x': 0.975, 'offset_y': -0.65}},
                    {'number': '5', 'name': 'SDO', 'type': 'output', 'net': 'BME_SDO',
                     'physical': {'offset_x': 0.975, 'offset_y': 0.65}},
                    {'number': '6', 'name': 'VDDIO', 'type': 'power_in', 'net': '3V3',
                     'physical': {'offset_x': 0.325, 'offset_y': 0.65}},
                    {'number': '7', 'name': 'GND2', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': -0.325, 'offset_y': 0.65}},
                    {'number': '8', 'name': 'VDD', 'type': 'power_in', 'net': '3V3',
                     'physical': {'offset_x': -0.975, 'offset_y': 0.65}},
                ],
            },

            # USB-C Connector (simplified 8-pin)
            'J1': {
                'name': 'USB-C',
                'footprint': 'USB-C-16P',
                'value': 'USB-C',
                'description': 'USB Type-C Connector',
                'size': (9.0, 7.5),
                'pins': [
                    {'number': '1', 'name': 'GND1', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': -3.5, 'offset_y': -3.0}},
                    {'number': '2', 'name': 'VBUS1', 'type': 'power_in', 'net': 'VBUS',
                     'physical': {'offset_x': -2.5, 'offset_y': -3.0}},
                    {'number': '3', 'name': 'CC1', 'type': 'bidirectional', 'net': 'CC1',
                     'physical': {'offset_x': -1.5, 'offset_y': -3.0}},
                    {'number': '4', 'name': 'DP1', 'type': 'bidirectional', 'net': 'USB_DP',
                     'physical': {'offset_x': -0.5, 'offset_y': -3.0}},
                    {'number': '5', 'name': 'DN1', 'type': 'bidirectional', 'net': 'USB_DN',
                     'physical': {'offset_x': 0.5, 'offset_y': -3.0}},
                    {'number': '6', 'name': 'GND2', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': 1.5, 'offset_y': -3.0}},
                    {'number': '7', 'name': 'SHIELD1', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 3.5, 'offset_y': 0.0}},
                    {'number': '8', 'name': 'SHIELD2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': -3.5, 'offset_y': 0.0}},
                ],
            },

            # Decoupling Capacitors - 0402
            'C1': {
                'name': 'C1',
                'footprint': '0402',
                'value': '100nF',
                'description': 'ESP32 Decoupling',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },
            'C2': {
                'name': 'C2',
                'footprint': '0402',
                'value': '100nF',
                'description': 'LDO Output Decoupling',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },
            'C3': {
                'name': 'C3',
                'footprint': '0402',
                'value': '100nF',
                'description': 'BME280 Decoupling',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },

            # Bulk Capacitors - 0805
            'C4': {
                'name': 'C4',
                'footprint': '0805',
                'value': '10uF',
                'description': 'LDO Input Bulk',
                'size': (2.0, 1.25),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': 'VBUS',
                     'physical': {'offset_x': -0.9, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.9, 'offset_y': 0.0}},
                ],
            },
            'C5': {
                'name': 'C5',
                'footprint': '0805',
                'value': '10uF',
                'description': 'LDO Output Bulk',
                'size': (2.0, 1.25),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                     'physical': {'offset_x': -0.9, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.9, 'offset_y': 0.0}},
                ],
            },
            'C6': {
                'name': 'C6',
                'footprint': '0402',
                'value': '100nF',
                'description': 'BME CS Decoupling',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': 'BME_CS',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },

            # I2C Pull-up Resistors - 0402
            'R1': {
                'name': 'R1',
                'footprint': '0402',
                'value': '10k',
                'description': 'I2C SDA Pull-up',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'I2C_SDA',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },
            'R2': {
                'name': 'R2',
                'footprint': '0402',
                'value': '10k',
                'description': 'I2C SCL Pull-up',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': '3V3',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'I2C_SCL',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },

            # USB CC Resistors - 0402
            'R3': {
                'name': 'R3',
                'footprint': '0402',
                'value': '5.1k',
                'description': 'USB CC1 Pulldown',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': 'CC1',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },

            # LED Current Limiting Resistors - 0402
            'R5': {
                'name': 'R5',
                'footprint': '0402',
                'value': '330',
                'description': 'LED1 Current Limit',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED1_CTRL',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED1_A',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },
            'R6': {
                'name': 'R6',
                'footprint': '0402',
                'value': '330',
                'description': 'LED2 Current Limit',
                'size': (1.0, 0.5),
                'pins': [
                    {'number': '1', 'name': '1', 'type': 'passive', 'net': 'LED2_CTRL',
                     'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                    {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED2_A',
                     'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
                ],
            },

            # LEDs - 0603
            'LED1': {
                'name': 'LED1',
                'footprint': '0603',
                'value': 'RED',
                'description': 'Status LED',
                'size': (1.6, 0.8),
                'pins': [
                    {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED1_A',
                     'physical': {'offset_x': -0.7, 'offset_y': 0.0}},
                    {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.7, 'offset_y': 0.0}},
                ],
            },
            'LED2': {
                'name': 'LED2',
                'footprint': '0603',
                'value': 'GREEN',
                'description': 'Activity LED',
                'size': (1.6, 0.8),
                'pins': [
                    {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED2_A',
                     'physical': {'offset_x': -0.7, 'offset_y': 0.0}},
                    {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                     'physical': {'offset_x': 0.7, 'offset_y': 0.0}},
                ],
            },
        },

        # Nets definition (for routing planner analysis)
        'nets': {
            # Power Rails
            'VBUS': {
                'pins': ['J1.VBUS1', 'U2.VIN', 'C4.1', 'U3.VBUS'],
                'class': 'power'
            },
            '3V3': {
                'pins': ['U2.VOUT', 'U2.VOUT2', 'U1.VCC', 'C1.1', 'C2.1', 'C3.1',
                         'C5.1', 'U4.VDD', 'U4.VDDIO', 'R1.1', 'R2.1'],
                'class': 'power'
            },
            'GND': {
                'pins': ['J1.GND1', 'J1.GND2', 'J1.SHIELD1', 'J1.SHIELD2',
                         'U1.GND', 'U1.GND2', 'U2.GND', 'U3.GND', 'U4.GND', 'U4.GND2',
                         'C1.2', 'C2.2', 'C3.2', 'C4.2', 'C5.2', 'C6.2', 'R3.2',
                         'LED1.K', 'LED2.K'],
                'class': 'ground'
            },
            # USB Data Lines (Differential Pair)
            'USB_DP': {
                'pins': ['J1.DP1', 'U3.IO1', 'U1.IO19'],
                'class': 'differential'
            },
            'USB_DN': {
                'pins': ['J1.DN1', 'U3.IO2', 'U1.IO18'],
                'class': 'differential'
            },
            # USB CC Line
            'CC1': {
                'pins': ['J1.CC1', 'R3.1'],
                'class': 'signal'
            },
            # I2C Bus
            'I2C_SDA': {
                'pins': ['U1.IO21', 'U4.SDI', 'R1.2'],
                'class': 'i2c'
            },
            'I2C_SCL': {
                'pins': ['U1.IO22', 'U4.SCK', 'R2.2'],
                'class': 'i2c'
            },
            # BME280 Control
            'BME_CS': {
                'pins': ['U4.CSB', 'C6.1'],
                'class': 'signal'
            },
            'BME_SDO': {
                'pins': ['U4.SDO'],
                'class': 'signal'
            },
            # LED Control
            'LED1_CTRL': {
                'pins': ['U1.IO2', 'R5.1'],
                'class': 'signal'
            },
            'LED2_CTRL': {
                'pins': ['U1.IO4', 'R6.1'],
                'class': 'signal'
            },
            'LED1_A': {
                'pins': ['R5.2', 'LED1.A'],
                'class': 'signal'
            },
            'LED2_A': {
                'pins': ['R6.2', 'LED2.A'],
                'class': 'signal'
            },
            # ESP32 Control
            'EN': {
                'pins': ['U1.EN'],
                'class': 'signal'
            },
            'BOOT': {
                'pins': ['U1.IO0'],
                'class': 'signal'
            },
            # UART (for programming/debug)
            'UART_TX': {
                'pins': ['U1.TXD0'],
                'class': 'signal'
            },
            'UART_RX': {
                'pins': ['U1.RXD0'],
                'class': 'signal'
            },
            # NC pins
            'NC1': {'pins': ['U1.IO25'], 'class': 'signal'},
            'NC2': {'pins': ['U1.IO26'], 'class': 'signal'},
            'NC3': {'pins': ['U1.IO27'], 'class': 'signal'},
            'NC_ESD1': {'pins': ['U3.IO3'], 'class': 'signal'},
            'NC_ESD2': {'pins': ['U3.IO4'], 'class': 'signal'},
        }
    }

    return parts_db


def run_test():
    """Run the 20-component board test."""

    print("=" * 70)
    print("REAL-WORLD 20 COMPONENT PCB TEST")
    print("=" * 70)

    parts_db = create_20_component_parts_db()

    # Count components and nets
    num_parts = len(parts_db['parts'])
    num_nets = len(parts_db['nets'])
    num_pins = sum(len(p.get('pins', [])) for p in parts_db['parts'].values())
    num_connections = sum(len(n.get('pins', [])) for n in parts_db['nets'].values())

    print(f"\nDESIGN STATISTICS:")
    print(f"  Components: {num_parts}")
    print(f"  Total pins: {num_pins}")
    print(f"  Nets: {num_nets}")
    print(f"  Total connections: {num_connections}")

    # Board configuration
    board_config = {
        'board_width': 50.0,  # 50mm
        'board_height': 40.0,  # 40mm
        'layers': 2,
    }

    print(f"\nBOARD CONFIGURATION:")
    print(f"  Size: {board_config['board_width']}x{board_config['board_height']}mm")
    print(f"  Layers: {board_config['layers']}")

    # Test 1: Routing Planner Analysis
    print("\n" + "-" * 50)
    print("TEST 1: ROUTING PLANNER ANALYSIS")
    print("-" * 50)

    start_time = time.time()
    plan = create_routing_plan(parts_db, board_config)
    plan_time = time.time() - start_time

    print(f"\n  Analysis time: {plan_time*1000:.1f}ms")
    print(f"  Density: {plan.design_profile.density.value}")
    print(f"  Area utilization: {plan.design_profile.area_utilization:.1f}%")
    print(f"  Has QFN: {plan.design_profile.has_qfn}")
    print(f"  Needs ground pour: {plan.design_profile.needs_ground_pour}")
    print(f"  Success prediction: {plan.overall_success_prediction*100:.0f}%")

    print(f"\n  NET CLASSIFICATION:")
    class_counts = {}
    for strategy in plan.net_strategies.values():
        cls = strategy.net_class.value
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls:15}: {count} nets")

    print(f"\n  ALGORITHM DISTRIBUTION:")
    algo_dist = plan.get_algorithm_distribution()
    for algo, count in sorted(algo_dist.items(), key=lambda x: -x[1]):
        print(f"    {algo:15}: {count} nets")

    # Test 2: Full PCB Engine Run
    print("\n" + "-" * 50)
    print("TEST 2: FULL PCB ENGINE RUN")
    print("-" * 50)

    try:
        config = EngineConfig(
            board_width=board_config['board_width'],
            board_height=board_config['board_height'],
            layer_count=board_config['layers'],
            trace_width=0.25,
            clearance=0.15,
            via_drill=0.3,
            via_diameter=0.6,
            grid_size=0.15,       # Coarser grid for speed (vs 0.08 default)
            min_grid_size=0.15,
            max_grid_size=0.25,
        )

        engine = PCBEngine(config)

        print("\n  Starting PCB Engine...")
        start_time = time.time()

        # Run the engine
        result = engine.run(parts_db)

        total_time = time.time() - start_time

        print(f"\n  RESULTS:")
        print(f"  Total time: {total_time:.2f}s")

        if hasattr(result, 'success'):
            print(f"  Success: {result.success}")

        if hasattr(result, 'routed_count') and hasattr(result, 'total_nets'):
            pct = result.routed_count / result.total_nets * 100 if result.total_nets > 0 else 0
            print(f"  Routed: {result.routed_count}/{result.total_nets} ({pct:.1f}%)")

        if hasattr(result, 'drc_passed'):
            print(f"  DRC passed: {result.drc_passed}")

        if hasattr(result, 'output_path'):
            print(f"  Output: {result.output_path}")

        # Check for unrouted nets
        if hasattr(result, 'unrouted_nets') and result.unrouted_nets:
            print(f"\n  UNROUTED NETS ({len(result.unrouted_nets)}):")
            for net in result.unrouted_nets[:10]:
                print(f"    - {net}")
            if len(result.unrouted_nets) > 10:
                print(f"    ... and {len(result.unrouted_nets) - 10} more")

    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Stress test with smaller board
    print("\n" + "-" * 50)
    print("TEST 3: STRESS TEST - SMALLER BOARD (40x30mm)")
    print("-" * 50)

    try:
        config_small = EngineConfig(
            board_width=40.0,
            board_height=30.0,
            layer_count=2,
            trace_width=0.2,
            clearance=0.15,
            grid_size=0.15,
            min_grid_size=0.15,
            max_grid_size=0.25,
        )

        engine_small = PCBEngine(config_small)

        print("\n  Starting PCB Engine on smaller board...")
        start_time = time.time()

        result_small = engine_small.run(parts_db)

        total_time = time.time() - start_time

        print(f"\n  RESULTS:")
        print(f"  Total time: {total_time:.2f}s")

        if hasattr(result_small, 'routed_count') and hasattr(result_small, 'total_nets'):
            pct = result_small.routed_count / result_small.total_nets * 100 if result_small.total_nets > 0 else 0
            print(f"  Routed: {result_small.routed_count}/{result_small.total_nets} ({pct:.1f}%)")

    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"""
Components tested: {num_parts}
Nets tested: {num_nets}
Board sizes tested: 50x40mm, 40x30mm

The PCB Engine's real-world limits depend on:
1. Board size vs component count
2. Net complexity (differential pairs, buses)
3. Pin density of components
4. Routing congestion

For a 20-component ESP32 board:
- 50x40mm: Should work well
- 40x30mm: May struggle with routing
""")


if __name__ == '__main__':
    run_test()
