"""
================================================================================
Complex PCB Test - MCU Sensor Module
================================================================================

This test creates a more complex PCB with:
- Microcontroller (8-pin ATtiny or similar)
- 2 LEDs (status + activity)
- 2 Current-limiting resistors
- Temperature sensor (3-pin)
- Decoupling capacitor
- Power connector (2-pin)

Total: 8 components, ~15 nets

Signal chains:
1. POWER: VIN -> MCU_VCC, SENSOR_VCC, CAP
2. LED1: MCU_GPIO1 -> R1 -> LED1 -> GND
3. LED2: MCU_GPIO2 -> R2 -> LED2 -> GND
4. SENSOR: MCU_SDA -> SENSOR_SDA, MCU_SCL -> SENSOR_SCL

This tests:
- Multiple signal chains
- Component grouping
- Escape conflict resolution
- Via routing on blocked paths
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.engine import BoardConfig, DesignRules
from pcb_engine.human_engine import HumanPCBEngine


def create_complex_sensor_module():
    """Create a complex sensor module PCB"""

    # Board configuration - 40mm x 35mm
    board = BoardConfig(
        width=40.0,
        height=35.0,
        origin_x=100.0,
        origin_y=100.0,
        grid_size=0.5,
    )

    # Design rules
    rules = DesignRules(
        min_trace_width=0.25,
        min_clearance=0.2,
        min_via_diameter=0.8,
        min_via_drill=0.4,
    )

    # Parts definition - using same format as test_human_basic.py
    # ATtiny-like 8-pin MCU (SOIC-8)
    parts = {
        'U1': {
            'name': 'MCU',
            'footprint': 'SOIC-8',
            'value': 'ATtiny85',
            'description': 'ATtiny85 Microcontroller',
            'size': (5.0, 4.0),
            'pins': [
                {'number': '1', 'name': 'RESET', 'type': 'input', 'net': 'RESET',
                 'physical': {'offset_x': -2.0, 'offset_y': -1.27}},
                {'number': '2', 'name': 'GPIO1', 'type': 'output', 'net': 'GPIO1',
                 'physical': {'offset_x': -2.0, 'offset_y': 0.0}},
                {'number': '3', 'name': 'GPIO2', 'type': 'output', 'net': 'GPIO2',
                 'physical': {'offset_x': -2.0, 'offset_y': 1.27}},
                {'number': '4', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': -2.0, 'offset_y': 2.54}},
                {'number': '5', 'name': 'SDA', 'type': 'bidirectional', 'net': 'SDA',
                 'physical': {'offset_x': 2.0, 'offset_y': 2.54}},
                {'number': '6', 'name': 'SCL', 'type': 'output', 'net': 'SCL',
                 'physical': {'offset_x': 2.0, 'offset_y': 1.27}},
                {'number': '7', 'name': 'NC', 'type': 'passive', 'net': 'NC',
                 'physical': {'offset_x': 2.0, 'offset_y': 0.0}},
                {'number': '8', 'name': 'VCC', 'type': 'power_in', 'net': 'VCC',
                 'physical': {'offset_x': 2.0, 'offset_y': -1.27}},
            ],
        },

        # Temperature sensor (SOT-23, 3 pins)
        'U2': {
            'name': 'TempSensor',
            'footprint': 'SOT-23',
            'value': 'TMP36',
            'description': 'Temperature Sensor',
            'size': (3.0, 2.5),
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': 'VCC',
                 'physical': {'offset_x': -1.0, 'offset_y': -0.95}},
                {'number': '2', 'name': 'VOUT', 'type': 'output', 'net': 'TEMP',
                 'physical': {'offset_x': -1.0, 'offset_y': 0.95}},
                {'number': '3', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 1.0, 'offset_y': 0.0}},
            ],
        },

        # LED1 - Status (0603)
        'D1': {
            'name': 'LED1',
            'footprint': 'LED_0603',
            'value': 'GREEN',
            'description': 'Status LED',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED1_A',
                 'physical': {'offset_x': -0.7, 'offset_y': 0}},
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.7, 'offset_y': 0}},
            ],
        },

        # LED2 - Activity (0603)
        'D2': {
            'name': 'LED2',
            'footprint': 'LED_0603',
            'value': 'RED',
            'description': 'Activity LED',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'name': 'A', 'type': 'passive', 'net': 'LED2_A',
                 'physical': {'offset_x': -0.7, 'offset_y': 0}},
                {'number': '2', 'name': 'K', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.7, 'offset_y': 0}},
            ],
        },

        # R1 - LED1 current limit (0603, 330 ohm)
        'R1': {
            'name': 'R1',
            'footprint': '0603',
            'value': '330R',
            'description': 'LED1 Current Limit',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'GPIO1',
                 'physical': {'offset_x': -0.75, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED1_A',
                 'physical': {'offset_x': 0.75, 'offset_y': 0}},
            ],
        },

        # R2 - LED2 current limit (0603, 330 ohm)
        'R2': {
            'name': 'R2',
            'footprint': '0603',
            'value': '330R',
            'description': 'LED2 Current Limit',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'GPIO2',
                 'physical': {'offset_x': -0.75, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'LED2_A',
                 'physical': {'offset_x': 0.75, 'offset_y': 0}},
            ],
        },

        # C1 - Decoupling cap (0603, 100nF)
        'C1': {
            'name': 'C1',
            'footprint': 'C0603',
            'value': '100nF',
            'description': 'Decoupling Capacitor',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'name': '1', 'type': 'passive', 'net': 'VCC',
                 'physical': {'offset_x': -0.75, 'offset_y': 0}},
                {'number': '2', 'name': '2', 'type': 'passive', 'net': 'GND',
                 'physical': {'offset_x': 0.75, 'offset_y': 0}},
            ],
        },

        # J1 - Power connector (2-pin header)
        'J1': {
            'name': 'Power',
            'footprint': 'Header_2Pin',
            'value': 'POWER',
            'description': 'Power Connector',
            'size': (5.08, 2.54),
            'pins': [
                {'number': '1', 'name': 'VCC', 'type': 'power_in', 'net': 'VCC',
                 'physical': {'offset_x': -1.27, 'offset_y': 0}},
                {'number': '2', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                 'physical': {'offset_x': 1.27, 'offset_y': 0}},
            ],
        },
    }

    # Create engine
    engine = HumanPCBEngine(board=board, rules=rules)

    # Load parts
    print("=" * 60)
    print("COMPLEX PCB TEST - MCU SENSOR MODULE")
    print("=" * 60)
    print("\nComponents:")
    for ref, part in parts.items():
        pins = len(part.get('used_pins', []))
        print(f"  {ref}: {part.get('value', 'Unknown')} ({pins} pins)")

    if not engine.load_parts_from_dict(parts):
        print("\nERROR: Failed to load parts!")
        for e in engine.state.errors:
            print(f"  - {e}")
        return None

    # Run the engine
    print("\nRunning Human-Like PCB Engine...")
    print("-" * 40)

    if engine.run():
        print("-" * 40)
        print("Engine completed successfully!")

        # Print placement results
        print("\nPlacement Results:")
        for ref, pos in engine.state.placement.items():
            print(f"  {ref}: ({pos.x:.1f}, {pos.y:.1f})")

        # Print routing results
        print("\nRouting Results:")
        for net_name, route in engine.state.routes.items():
            if route.success:
                segs = len(route.segments)
                vias = len(route.vias)
                print(f"  {net_name}: OK ({segs} segments, {vias} vias)")
            else:
                print(f"  {net_name}: FAILED - {route.error}")

        return engine
    else:
        print("\nERROR: Engine failed!")
        for e in engine.state.errors:
            print(f"  - {e}")
        return None


def main():
    # Create the complex PCB
    engine = create_complex_sensor_module()

    if engine is None:
        print("\nTest FAILED - could not create PCB")
        return 1

    # Generate KiCad script
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "test_complex_output.py")

    if engine.generate_kicad_script(output_file):
        print(f"\nKiCad script generated: {output_file}")

        # Print summary
        print("\n" + engine.get_report())

        print("\n" + "=" * 60)
        print("TO TEST IN KICAD:")
        print("=" * 60)
        print("1. Create new board: File -> New -> Board")
        print("2. Open Python console: Tools -> Scripting Console")
        print("3. Run command:")
        print(f'   exec(open(r"{output_file}").read())')
        print("4. Run DRC: Inspect -> Design Rules Checker")
        print("=" * 60)

        return 0
    else:
        print("\nFailed to generate KiCad script")
        return 1


if __name__ == '__main__':
    sys.exit(main())
