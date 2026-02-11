"""
Test: 10-component ATmega328P board
- U1: ATmega328P (SOIC-28) main MCU
- Y1: 16MHz crystal (HC49)
- R1, R2: 4.7k I2C pull-ups (0603)
- R3: 330 ohm LED resistor (0603)
- R4: 10k reset pull-up (0603)
- LED1: Green LED (0805)
- C1: 100nF bypass (0402)
- C2, C3: 22pF crystal load caps (0402)
- 9 nets, 35x30mm board, 2 layers
"""
import sys, time
sys.path.insert(0, '.')

from pcb_engine.pcb_engine import PCBEngine, EngineConfig

# SOIC-28: 7 pins per side, 1.27mm pitch, body 17.9x7.5mm
# Pin 1 top-left, pins go down left side then up right side
def _soic28_pins():
    """Generate SOIC-28 pin positions (actual footprint geometry)."""
    pins = []
    pitch = 1.27
    # Left side: pins 1-14 going down, x = -3.81 (half of 7.62mm lead span)
    for i in range(14):
        pins.append({
            'number': str(i + 1),
            'physical': {'offset_x': -3.81, 'offset_y': -8.255 + i * pitch}
        })
    # Right side: pins 15-28 going up, x = +3.81
    for i in range(14):
        pins.append({
            'number': str(28 - i),
            'physical': {'offset_x': 3.81, 'offset_y': -8.255 + i * pitch}
        })
    return pins

# Build SOIC-28 pins with net assignments
soic28_pins = _soic28_pins()
pin_nets = {
    '1': 'RESET',     # PC6/RESET
    '7': '5V',        # VCC
    '8': 'GND',       # GND
    '9': 'XTAL1',     # PB6/XTAL1
    '10': 'XTAL2',    # PB7/XTAL2
    '19': 'LED_OUT',  # PB5/SCK
    '20': '5V',       # AVCC
    '22': 'GND',      # GND
    '27': 'SDA',      # PC4/SDA
    '28': 'SCL',      # PC5/SCL
}
for pin in soic28_pins:
    pin['net'] = pin_nets.get(pin['number'], '')

parts_db = {
    'parts': {
        'U1': {
            'name': 'ATmega328P', 'footprint': 'SOIC-28', 'value': 'ATmega328P',
            'size': (17.9, 7.5),
            'pins': soic28_pins,
        },
        'Y1': {
            'name': '16MHz_Crystal', 'footprint': 'HC49', 'value': '16MHz',
            'size': (11.4, 4.7),
            'pins': [
                {'number': '1', 'net': 'XTAL1', 'physical': {'offset_x': -2.44, 'offset_y': 0}},
                {'number': '2', 'net': 'XTAL2', 'physical': {'offset_x': 2.44, 'offset_y': 0}},
            ]
        },
        'R1': {
            'name': 'R_4K7_SDA', 'footprint': '0603', 'value': '4.7k',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'net': '5V', 'physical': {'offset_x': -0.775, 'offset_y': 0}},
                {'number': '2', 'net': 'SDA', 'physical': {'offset_x': 0.775, 'offset_y': 0}},
            ]
        },
        'R2': {
            'name': 'R_4K7_SCL', 'footprint': '0603', 'value': '4.7k',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'net': '5V', 'physical': {'offset_x': -0.775, 'offset_y': 0}},
                {'number': '2', 'net': 'SCL', 'physical': {'offset_x': 0.775, 'offset_y': 0}},
            ]
        },
        'R3': {
            'name': 'R_330_LED', 'footprint': '0603', 'value': '330',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'net': 'LED_OUT', 'physical': {'offset_x': -0.775, 'offset_y': 0}},
                {'number': '2', 'net': 'LED_A', 'physical': {'offset_x': 0.775, 'offset_y': 0}},
            ]
        },
        'R4': {
            'name': 'R_10K_RESET', 'footprint': '0603', 'value': '10k',
            'size': (1.6, 0.8),
            'pins': [
                {'number': '1', 'net': '5V', 'physical': {'offset_x': -0.775, 'offset_y': 0}},
                {'number': '2', 'net': 'RESET', 'physical': {'offset_x': 0.775, 'offset_y': 0}},
            ]
        },
        'LED1': {
            'name': 'LED_GREEN', 'footprint': '0805', 'value': 'Green',
            'size': (2.0, 1.25),
            'pins': [
                {'number': '1', 'net': 'LED_A', 'physical': {'offset_x': -0.95, 'offset_y': 0}},
                {'number': '2', 'net': 'GND', 'physical': {'offset_x': 0.95, 'offset_y': 0}},
            ]
        },
        'C1': {
            'name': 'C_100nF', 'footprint': '0402', 'value': '100nF',
            'size': (1.0, 0.5),
            'pins': [
                {'number': '1', 'net': '5V', 'physical': {'offset_x': -0.48, 'offset_y': 0}},
                {'number': '2', 'net': 'GND', 'physical': {'offset_x': 0.48, 'offset_y': 0}},
            ]
        },
        'C2': {
            'name': 'C_22pF_XTAL1', 'footprint': '0402', 'value': '22pF',
            'size': (1.0, 0.5),
            'pins': [
                {'number': '1', 'net': 'XTAL1', 'physical': {'offset_x': -0.48, 'offset_y': 0}},
                {'number': '2', 'net': 'GND', 'physical': {'offset_x': 0.48, 'offset_y': 0}},
            ]
        },
        'C3': {
            'name': 'C_22pF_XTAL2', 'footprint': '0402', 'value': '22pF',
            'size': (1.0, 0.5),
            'pins': [
                {'number': '1', 'net': 'XTAL2', 'physical': {'offset_x': -0.48, 'offset_y': 0}},
                {'number': '2', 'net': 'GND', 'physical': {'offset_x': 0.48, 'offset_y': 0}},
            ]
        },
    },
    'nets': {
        'GND':     {'type': 'power', 'pins': ['U1.8', 'U1.22', 'LED1.2', 'C1.2', 'C2.2', 'C3.2']},
        '5V':      {'type': 'power', 'pins': ['U1.7', 'U1.20', 'R1.1', 'R2.1', 'R4.1', 'C1.1']},
        'XTAL1':   {'type': 'signal', 'pins': ['U1.9', 'Y1.1', 'C2.1']},
        'XTAL2':   {'type': 'signal', 'pins': ['U1.10', 'Y1.2', 'C3.1']},
        'SDA':     {'type': 'signal', 'pins': ['U1.27', 'R1.2']},
        'SCL':     {'type': 'signal', 'pins': ['U1.28', 'R2.2']},
        'LED_OUT': {'type': 'signal', 'pins': ['U1.19', 'R3.1']},
        'LED_A':   {'type': 'signal', 'pins': ['R3.2', 'LED1.1']},
        'RESET':   {'type': 'signal', 'pins': ['U1.1', 'R4.2']},
    },
    'board': {'width': 35, 'height': 30, 'layers': 2},
}

def main():
    print("=" * 60)
    print("10-COMPONENT ATmega328P BOARD TEST")
    print("=" * 60)
    print(f"Components: {len(parts_db['parts'])}")
    print(f"Nets: {len(parts_db['nets'])}")
    print(f"Board: 35x30mm, 2 layers")
    print()

    config = EngineConfig(
        board_width=35.0,
        board_height=30.0,
        layer_count=2,
        trace_width=0.25,
        clearance=0.2,
        via_drill=0.3,
        via_diameter=0.6,
        board_name="ATmega328P_10comp",
    )

    engine = PCBEngine(config)

    t0 = time.time()
    try:
        result = engine.run_orchestrated(parts_db)
        elapsed = time.time() - t0

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Stage reached: {result.stage_reached}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Routing: {result.routed_count}/{result.total_nets} nets")
        print(f"DRC passed: {result.drc_passed}")
        print(f"Output files: {result.output_files}")
        if result.errors:
            print(f"Errors: {result.errors[:5]}")
        if result.warnings:
            print(f"Warnings: {result.warnings[:5]}")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\nFAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
