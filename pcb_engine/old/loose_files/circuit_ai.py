"""
PCB Engine - Circuit AI (THE ENGINEER)
=======================================

COMMAND HIERARCHY:
==================

    ┌─────────────────────────────────────────────────────────────┐
    │                         USER                                 │
    │                       (THE BOSS)                             │
    │   - Describes what they want in natural language             │
    │   - Provides constraints (size, power, features)             │
    │   - Has final say on all decisions                           │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      CIRCUIT AI          ◄── YOU ARE HERE    │
    │                    (THE ENGINEER)                            │
    │                                                              │
    │   RESPONSIBILITIES:                                          │
    │   - Understand USER requirements (natural language parsing)  │
    │   - Design circuit topology and component selection          │
    │   - Make INTELLIGENT decisions (brain, not just machine)     │
    │   - Choose algorithms/strategies for Foreman                 │
    │   - Solve problems that need engineering knowledge           │
    │   - Escalate to USER when approval needed                    │
    │                                                              │
    │   DECISION MAKING:                                           │
    │   - Which routing algorithm for this design density?         │
    │   - Should we increase board size or add layers?             │
    │   - Can we accept these DRC violations?                      │
    │   - Is this component substitution acceptable?               │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      PCB ENGINE                              │
    │                    (THE FOREMAN)                             │
    │   - Receives complete design package from Engineer           │
    │   - Coordinates all piston workers                           │
    │   - Follows Engineer's algorithm recommendations             │
    │   - Reports challenges back to Engineer                      │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        PISTONS                               │
    │                      (THE WORKERS)                           │
    │   Parts | Order | Placement | Escape | Routing | ...         │
    │   - Execute specific tasks as directed                       │
    │   - Report results back to Foreman                           │
    └─────────────────────────────────────────────────────────────┘


ENGINEER WORKFLOW:
==================
1. USER INTERACTION
   - Gather circuit requirements through conversation
   - Ask clarifying questions
   - Understand the application context

2. REQUIREMENT ANALYSIS (Engineering Knowledge)
   - Parse user requirements into structured format
   - Identify circuit blocks (power, MCU, sensors, etc.)
   - Determine electrical constraints

3. CIRCUIT TOPOLOGY
   - Generate circuit block diagram
   - Define signal flow
   - Identify power distribution

4. COMPONENT SELECTION
   - Select components that fit electrically
   - Verify physical compatibility
   - Check availability/sourcing

5. HANDOFF TO FOREMAN
   - Generate complete parts_db dictionary
   - Include design constraints and rules
   - Provide algorithm recommendations

6. DECISION SUPPORT (During Execution)
   - Foreman asks: "Which algorithm for routing?"
   - Engineer decides based on design context
   - Engineer can override defaults with reasoning

7. CHALLENGE RESOLUTION
   - Foreman reports: "Routing failed"
   - Engineer analyzes and decides:
     * Try different algorithm
     * Add layers
     * Increase board size
     * Escalate to USER

SUPPORTED CIRCUIT TYPES:
========================
- Power supplies (LDO, DC-DC, Battery charging)
- Microcontroller systems (ESP32, STM32, Arduino)
- Sensor interfaces (I2C, SPI, Analog)
- Communication modules (WiFi, BLE, LoRa, USB)
- Motor drivers
- LED drivers
- Audio circuits
- Custom analog circuits

Research References:
- EDASolver: https://edasolver.com/ (Automatic component selection)
- AI-Assisted PCB Design: ResearchGate 2024
- Component selection guidelines: EMA Design Automation
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import math
from collections import defaultdict


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CircuitBlockType(Enum):
    """Types of circuit blocks"""
    POWER_INPUT = 'power_input'           # Power input (USB, barrel jack, battery)
    POWER_REGULATOR = 'power_regulator'   # Voltage regulators (LDO, DC-DC)
    MCU = 'mcu'                           # Microcontroller
    SENSOR = 'sensor'                     # Sensors (temp, humidity, motion, etc.)
    COMMUNICATION = 'communication'       # Comm modules (WiFi, BLE, LoRa)
    DISPLAY = 'display'                   # Displays (LCD, OLED, LED matrix)
    MOTOR_DRIVER = 'motor_driver'         # Motor drivers
    AUDIO = 'audio'                       # Audio (amp, DAC, microphone)
    INTERFACE = 'interface'               # User interface (buttons, LEDs, buzzer)
    PROTECTION = 'protection'             # Protection circuits (ESD, fuse, TVS)
    CONNECTOR = 'connector'               # Connectors
    PASSIVE_NETWORK = 'passive_network'   # RC filters, voltage dividers
    CUSTOM = 'custom'                     # Custom/other


class PowerType(Enum):
    """Power supply types"""
    USB_5V = 'usb_5v'
    BATTERY_LIPO = 'battery_lipo'
    BATTERY_18650 = 'battery_18650'
    BATTERY_AA = 'battery_aa'
    DC_BARREL = 'dc_barrel'
    SOLAR = 'solar'
    POE = 'poe'
    EXTERNAL = 'external'


class MCUFamily(Enum):
    """Microcontroller families"""
    ESP32 = 'esp32'
    ESP8266 = 'esp8266'
    STM32 = 'stm32'
    ATMEGA = 'atmega'
    ATTINY = 'attiny'
    RP2040 = 'rp2040'
    NRF52 = 'nrf52'
    PIC = 'pic'
    CUSTOM = 'custom'


class InterfaceType(Enum):
    """Communication interfaces"""
    I2C = 'i2c'
    SPI = 'spi'
    UART = 'uart'
    USB = 'usb'
    ANALOG = 'analog'
    DIGITAL = 'digital'
    PWM = 'pwm'
    ONEWIRE = 'onewire'
    CAN = 'can'
    RS485 = 'rs485'


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PowerRequirement:
    """Power requirements for a circuit block"""
    voltage: float                    # Required voltage (V)
    voltage_tolerance: float = 0.05   # Voltage tolerance (±%)
    current_typical: float = 0.0      # Typical current (A)
    current_max: float = 0.0          # Maximum current (A)
    is_analog: bool = False           # Requires clean analog power
    ripple_max_mv: float = 50.0       # Max ripple (mV)


@dataclass
class PinRequirement:
    """Pin requirement for a circuit block"""
    name: str                         # Pin name
    interface: InterfaceType          # Interface type
    direction: str = 'bidirectional'  # input, output, bidirectional
    voltage: float = 3.3              # Signal voltage level
    is_critical: bool = False         # Timing-critical
    pull_up: bool = False             # Needs pull-up
    pull_down: bool = False           # Needs pull-down


@dataclass
class CircuitBlock:
    """A functional circuit block"""
    name: str
    block_type: CircuitBlockType
    description: str = ''

    # Power requirements
    power: PowerRequirement = None

    # Pin requirements
    pins: List[PinRequirement] = field(default_factory=list)

    # Physical constraints
    footprint_preference: str = ''    # SMD, THT, or specific
    max_height_mm: float = 0.0        # Height restriction
    keep_out_mm: float = 0.0          # Keep-out radius

    # Component preferences
    preferred_manufacturers: List[str] = field(default_factory=list)
    preferred_packages: List[str] = field(default_factory=list)
    avoid_components: List[str] = field(default_factory=list)

    # Computed
    selected_components: List[Dict] = field(default_factory=list)


@dataclass
class CircuitRequirements:
    """Complete circuit requirements from user"""
    # Project info
    project_name: str = 'untitled'
    description: str = ''
    application: str = ''              # IoT, industrial, consumer, etc.

    # Power specifications
    input_power: PowerType = PowerType.USB_5V
    input_voltage_range: Tuple[float, float] = (4.5, 5.5)
    output_voltages: List[float] = field(default_factory=lambda: [3.3])
    battery_capacity_mah: float = 0.0
    power_budget_mw: float = 0.0

    # MCU requirements
    mcu_family: MCUFamily = MCUFamily.ESP32
    mcu_features: List[str] = field(default_factory=list)  # WiFi, BLE, etc.
    gpio_count: int = 0
    adc_count: int = 0
    pwm_count: int = 0
    uart_count: int = 1
    i2c_count: int = 1
    spi_count: int = 1

    # Circuit blocks
    blocks: List[CircuitBlock] = field(default_factory=list)

    # Board constraints
    board_size_mm: Tuple[float, float] = (50.0, 50.0)
    layers: int = 2
    smd_only: bool = True

    # Environment
    temperature_min: float = 0.0      # °C
    temperature_max: float = 70.0     # °C
    humidity_max: float = 85.0        # %
    outdoor_use: bool = False

    # Cost/sourcing
    budget_usd: float = 0.0
    preferred_suppliers: List[str] = field(default_factory=list)
    production_quantity: int = 1


@dataclass
class UserQuestion:
    """A question to ask the user"""
    question: str
    field: str                        # Field to populate
    options: List[str] = field(default_factory=list)  # Multiple choice options
    default: Any = None
    required: bool = True
    validator: str = ''               # Validation regex


@dataclass
class CircuitAIResult:
    """Result from Circuit AI processing"""
    requirements: CircuitRequirements
    parts_db: Dict                    # Ready for Parts Piston
    topology: Dict                    # Circuit topology/block diagram
    power_tree: Dict                  # Power distribution tree
    bom_preview: List[Dict]           # Preliminary BOM
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# =============================================================================
# COMPONENT DATABASE (Built-in knowledge)
# =============================================================================

# Common MCUs with specifications
MCU_DATABASE = {
    'ESP32-WROOM-32': {
        'family': 'esp32',
        'voltage': 3.3,
        'current_typical': 0.08,  # 80mA
        'current_max': 0.5,       # 500mA WiFi TX
        'gpio': 34,
        'adc': 18,
        'pwm': 16,
        'uart': 3,
        'i2c': 2,
        'spi': 4,
        'features': ['wifi', 'ble', 'adc', 'dac', 'touch'],
        'footprint': 'Module:ESP32-WROOM-32',
        'pins': [
            {'num': 1, 'name': 'GND', 'type': 'power'},
            {'num': 2, 'name': '3V3', 'type': 'power'},
            {'num': 3, 'name': 'EN', 'type': 'input'},
            # ... more pins
        ]
    },
    'ESP32-S3-WROOM-1': {
        'family': 'esp32',
        'voltage': 3.3,
        'current_typical': 0.08,
        'current_max': 0.5,
        'gpio': 45,
        'adc': 20,
        'pwm': 8,
        'uart': 3,
        'i2c': 2,
        'spi': 4,
        'features': ['wifi', 'ble5', 'adc', 'usb_otg', 'lcd'],
        'footprint': 'Module:ESP32-S3-WROOM-1',
    },
    'STM32F103C8T6': {
        'family': 'stm32',
        'voltage': 3.3,
        'current_typical': 0.036,
        'current_max': 0.15,
        'gpio': 37,
        'adc': 10,
        'pwm': 15,
        'uart': 3,
        'i2c': 2,
        'spi': 2,
        'features': ['usb', 'can', 'rtc'],
        'footprint': 'Package_QFP:LQFP-48_7x7mm_P0.5mm',
    },
    'ATMEGA328P': {
        'family': 'atmega',
        'voltage': 5.0,
        'current_typical': 0.012,
        'current_max': 0.2,
        'gpio': 23,
        'adc': 6,
        'pwm': 6,
        'uart': 1,
        'i2c': 1,
        'spi': 1,
        'features': ['eeprom', 'watchdog'],
        'footprint': 'Package_DIP:DIP-28_W7.62mm',
    },
    'RP2040': {
        'family': 'rp2040',
        'voltage': 3.3,
        'current_typical': 0.025,
        'current_max': 0.1,
        'gpio': 30,
        'adc': 4,
        'pwm': 16,
        'uart': 2,
        'i2c': 2,
        'spi': 2,
        'features': ['pio', 'usb'],
        'footprint': 'Package_DFN_QFN:QFN-56-1EP_7x7mm_P0.4mm_EP3.2x3.2mm',
    },
}

# Common voltage regulators
REGULATOR_DATABASE = {
    'AMS1117-3.3': {
        'type': 'ldo',
        'vin_min': 4.5,
        'vin_max': 15.0,
        'vout': 3.3,
        'current_max': 1.0,
        'dropout': 1.0,
        'quiescent_ua': 5000,
        'footprint': 'Package_TO_SOT_SMD:SOT-223-3_TabPin2',
    },
    'AP2112K-3.3': {
        'type': 'ldo',
        'vin_min': 2.5,
        'vin_max': 6.0,
        'vout': 3.3,
        'current_max': 0.6,
        'dropout': 0.25,
        'quiescent_ua': 55,
        'footprint': 'Package_TO_SOT_SMD:SOT-23-5',
    },
    'MP1584': {
        'type': 'dcdc_buck',
        'vin_min': 4.5,
        'vin_max': 28.0,
        'vout_adj': True,
        'current_max': 3.0,
        'efficiency': 0.92,
        'footprint': 'Package_TO_SOT_SMD:SOT-23-8',
    },
    'MT3608': {
        'type': 'dcdc_boost',
        'vin_min': 2.0,
        'vin_max': 24.0,
        'vout_max': 28.0,
        'current_max': 2.0,
        'efficiency': 0.93,
        'footprint': 'Package_TO_SOT_SMD:SOT-23-6',
    },
}

# Common sensors
SENSOR_DATABASE = {
    'BME280': {
        'type': 'environmental',
        'measures': ['temperature', 'humidity', 'pressure'],
        'interface': ['i2c', 'spi'],
        'voltage': 3.3,
        'current_ua': 3.6,
        'footprint': 'Package_LGA:Bosch_LGA-8_2.5x2.5mm_P0.65mm_ClockwisePinNumbering',
    },
    'MPU6050': {
        'type': 'imu',
        'measures': ['accelerometer', 'gyroscope'],
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ma': 3.9,
        'footprint': 'Package_DFN_QFN:QFN-24-1EP_4x4mm_P0.5mm_EP2.7x2.7mm',
    },
    'DS18B20': {
        'type': 'temperature',
        'measures': ['temperature'],
        'interface': ['onewire'],
        'voltage': 3.3,
        'current_ma': 1.5,
        'footprint': 'Package_TO_SOT_THT:TO-92_Inline',
    },
    'HC-SR04': {
        'type': 'distance',
        'measures': ['distance'],
        'interface': ['digital'],
        'voltage': 5.0,
        'current_ma': 15,
        'footprint': 'Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical',
    },
    'AHT20': {
        'type': 'environmental',
        'measures': ['temperature', 'humidity'],
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ua': 23,
        'address': 0x38,
        'footprint': 'Package_DFN_QFN:DFN-6-1EP_3x3mm_P1mm_EP1.5x2.4mm',
    },
    'SHT31': {
        'type': 'environmental',
        'measures': ['temperature', 'humidity'],
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ua': 600,
        'accuracy': {'temp': 0.2, 'humidity': 2.0},
        'footprint': 'Package_DFN_QFN:DFN-8-1EP_2.5x2.5mm_P0.5mm_EP1.1x1.7mm',
    },
    'BMP280': {
        'type': 'pressure',
        'measures': ['temperature', 'pressure'],
        'interface': ['i2c', 'spi'],
        'voltage': 3.3,
        'current_ua': 2.7,
        'footprint': 'Package_LGA:Bosch_LGA-8_2x2.5mm_P0.65mm_ClockwisePinNumbering',
    },
    'MAX30102': {
        'type': 'biometric',
        'measures': ['heart_rate', 'spo2'],
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ma': 0.6,
        'footprint': 'Package_LGA:OLGA-14_3.3x5.6mm_P0.8mm',
    },
    'VEML7700': {
        'type': 'light',
        'measures': ['lux', 'uv'],
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ua': 45,
        'footprint': 'Package_LGA:LGA-4_2x2mm_P1mm',
    },
    'VL53L0X': {
        'type': 'distance',
        'measures': ['distance'],
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ma': 19,
        'range_mm': 2000,
        'footprint': 'Package_LGA:ST_LGA-12_2.4x4.4mm_P0.65mm',
    },
    'LSM6DS3': {
        'type': 'imu',
        'measures': ['accelerometer', 'gyroscope'],
        'interface': ['i2c', 'spi'],
        'voltage': 3.3,
        'current_ma': 0.9,
        'footprint': 'Package_LGA:LGA-14_2.5x3mm_P0.5mm',
    },
    'ICM-20948': {
        'type': 'imu',
        'measures': ['accelerometer', 'gyroscope', 'magnetometer'],
        'interface': ['i2c', 'spi'],
        'voltage': 3.3,
        'current_ma': 3.0,
        'footprint': 'Package_LGA:InvenSense_LGA-24_3x4mm_P0.4mm',
    },
    'NEO-6M': {
        'type': 'gps',
        'measures': ['position', 'time', 'velocity'],
        'interface': ['uart'],
        'voltage': 3.3,
        'current_ma': 45,
        'footprint': 'Module:u-blox_NEO-6M',
    },
    'HX711': {
        'type': 'adc',
        'measures': ['weight'],
        'interface': ['digital'],
        'voltage': 5.0,
        'current_ma': 1.5,
        'bits': 24,
        'footprint': 'Package_SO:SOIC-16_3.9x9.9mm_P1.27mm',
    },
    'INA219': {
        'type': 'power_monitor',
        'measures': ['current', 'voltage', 'power'],
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ma': 1.0,
        'footprint': 'Package_SO:MSOP-8_3x3mm_P0.65mm',
    },
    'INMP441': {
        'type': 'audio',
        'measures': ['sound'],
        'interface': ['i2s'],
        'voltage': 3.3,
        'current_ma': 1.4,
        'footprint': 'Package_LGA:LGA-6_2.5x3.2mm_P0.8mm',
    },
}


# =============================================================================
# EXPANDED COMPONENT DATABASES
# =============================================================================

# Display modules
DISPLAY_DATABASE = {
    'SSD1306_128x64': {
        'type': 'oled',
        'resolution': (128, 64),
        'interface': ['i2c', 'spi'],
        'voltage': 3.3,
        'current_ma': 20,
        'diagonal': 0.96,
        'footprint': 'Display_OLED:OLED_0.96in_128x64',
    },
    'SSD1306_128x32': {
        'type': 'oled',
        'resolution': (128, 32),
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ma': 10,
        'diagonal': 0.91,
        'footprint': 'Display_OLED:OLED_0.91in_128x32',
    },
    'SH1106_128x64': {
        'type': 'oled',
        'resolution': (128, 64),
        'interface': ['i2c', 'spi'],
        'voltage': 3.3,
        'current_ma': 25,
        'diagonal': 1.3,
        'footprint': 'Display_OLED:OLED_1.3in_128x64',
    },
    'ST7735': {
        'type': 'tft',
        'resolution': (128, 160),
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 50,
        'diagonal': 1.8,
        'color': 'rgb565',
        'footprint': 'Display_TFT:TFT_1.8in_ST7735',
    },
    'ILI9341': {
        'type': 'tft',
        'resolution': (240, 320),
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 100,
        'diagonal': 2.4,
        'color': 'rgb565',
        'footprint': 'Display_TFT:TFT_2.4in_ILI9341',
    },
    'ST7789': {
        'type': 'tft',
        'resolution': (240, 240),
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 80,
        'diagonal': 1.54,
        'color': 'rgb565',
        'footprint': 'Display_TFT:TFT_1.54in_ST7789',
    },
    'HD44780_16x2': {
        'type': 'lcd_char',
        'characters': (16, 2),
        'interface': ['parallel', 'i2c'],
        'voltage': 5.0,
        'current_ma': 120,
        'footprint': 'Display_LCD:LCD_16x2_HD44780',
    },
    'MAX7219': {
        'type': 'led_matrix',
        'resolution': (8, 8),
        'interface': ['spi'],
        'voltage': 5.0,
        'current_ma': 330,
        'cascadable': True,
        'footprint': 'Package_DIP:DIP-24_W15.24mm',
    },
}

# Communication modules
COMMUNICATION_DATABASE = {
    'ESP-01': {
        'type': 'wifi',
        'protocol': '802.11b/g/n',
        'interface': ['uart'],
        'voltage': 3.3,
        'current_ma': 170,
        'tx_power_dbm': 20,
        'footprint': 'Module:ESP-01',
    },
    'NRF24L01': {
        'type': 'rf',
        'frequency': 2.4,
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 13.5,
        'range_m': 100,
        'footprint': 'Module:nRF24L01',
    },
    'HC-05': {
        'type': 'bluetooth_classic',
        'interface': ['uart'],
        'voltage': 3.3,
        'current_ma': 40,
        'baudrate': 38400,
        'footprint': 'Module:HC-05',
    },
    'HM-10': {
        'type': 'ble',
        'interface': ['uart'],
        'voltage': 3.3,
        'current_ma': 8.5,
        'footprint': 'Module:HM-10',
    },
    'SX1278': {
        'type': 'lora',
        'frequency': 433,
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 120,
        'range_km': 10,
        'footprint': 'Module:SX1278_Ra-02',
    },
    'RFM95W': {
        'type': 'lora',
        'frequency': 915,
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 120,
        'range_km': 15,
        'footprint': 'Module:RFM95W',
    },
    'W5500': {
        'type': 'ethernet',
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 132,
        'speed_mbps': 100,
        'footprint': 'Package_QFP:LQFP-48_7x7mm_P0.5mm',
    },
    'ENC28J60': {
        'type': 'ethernet',
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 160,
        'speed_mbps': 10,
        'footprint': 'Package_SO:SOIC-28W_7.5x17.9mm_P1.27mm',
    },
    'MCP2515': {
        'type': 'can',
        'interface': ['spi'],
        'voltage': 5.0,
        'current_ma': 5,
        'requires': ['TJA1050'],
        'footprint': 'Package_SO:SOIC-18W_7.5x11.6mm_P1.27mm',
    },
    'TJA1050': {
        'type': 'can_transceiver',
        'interface': ['can'],
        'voltage': 5.0,
        'current_ma': 70,
        'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
    },
    'MAX485': {
        'type': 'rs485',
        'interface': ['uart'],
        'voltage': 5.0,
        'current_ma': 300,
        'footprint': 'Package_DIP:DIP-8_W7.62mm',
    },
    'CH340G': {
        'type': 'usb_uart',
        'interface': ['usb', 'uart'],
        'voltage': 3.3,
        'current_ma': 30,
        'footprint': 'Package_SO:SOIC-16_3.9x9.9mm_P1.27mm',
    },
    'CP2102': {
        'type': 'usb_uart',
        'interface': ['usb', 'uart'],
        'voltage': 3.3,
        'current_ma': 20,
        'footprint': 'Package_DFN_QFN:QFN-28-1EP_5x5mm_P0.5mm_EP3.35x3.35mm',
    },
    'FT232RL': {
        'type': 'usb_uart',
        'interface': ['usb', 'uart'],
        'voltage': 3.3,
        'current_ma': 25,
        'footprint': 'Package_SO:SSOP-28_5.3x10.2mm_P0.65mm',
    },
}

# Motor drivers
MOTOR_DRIVER_DATABASE = {
    'L298N': {
        'type': 'h_bridge_dual',
        'voltage_max': 46,
        'current_max': 2.0,
        'channels': 2,
        'interface': ['digital', 'pwm'],
        'footprint': 'Module:L298N',
    },
    'DRV8833': {
        'type': 'h_bridge_dual',
        'voltage_max': 10.8,
        'current_max': 1.5,
        'channels': 2,
        'interface': ['pwm'],
        'voltage': 3.3,
        'footprint': 'Package_SON:WSON-10-1EP_2.5x2.5mm_P0.5mm_EP1.2x1.4mm',
    },
    'TB6612FNG': {
        'type': 'h_bridge_dual',
        'voltage_max': 15,
        'current_max': 1.2,
        'channels': 2,
        'interface': ['pwm'],
        'voltage': 3.3,
        'footprint': 'Package_SO:SSOP-24_5.3x8.2mm_P0.65mm',
    },
    'A4988': {
        'type': 'stepper',
        'voltage_max': 35,
        'current_max': 2.0,
        'microstep': 16,
        'interface': ['step_dir'],
        'footprint': 'Package_QFP:TQFP-28_4x4mm_P0.4mm',
    },
    'TMC2209': {
        'type': 'stepper',
        'voltage_max': 29,
        'current_max': 2.0,
        'microstep': 256,
        'interface': ['step_dir', 'uart'],
        'silent': True,
        'footprint': 'Package_QFP:TQFP-28_5x5mm_P0.5mm',
    },
    'DRV8825': {
        'type': 'stepper',
        'voltage_max': 45,
        'current_max': 2.5,
        'microstep': 32,
        'interface': ['step_dir'],
        'footprint': 'Package_QFP:HTQFP-28-1EP_5x5mm_P0.5mm_EP3.45x3.45mm',
    },
    'L9110S': {
        'type': 'h_bridge_dual',
        'voltage_max': 12,
        'current_max': 0.8,
        'channels': 2,
        'interface': ['pwm'],
        'footprint': 'Package_SO:SOP-8_3.9x4.9mm_P1.27mm',
    },
    'BTS7960': {
        'type': 'h_bridge',
        'voltage_max': 27,
        'current_max': 43,
        'channels': 1,
        'interface': ['pwm'],
        'footprint': 'Module:BTS7960',
    },
    'VNH5019': {
        'type': 'h_bridge',
        'voltage_max': 41,
        'current_max': 30,
        'channels': 1,
        'interface': ['pwm'],
        'footprint': 'Module:VNH5019',
    },
}

# LED drivers
LED_DRIVER_DATABASE = {
    'WS2812B': {
        'type': 'addressable',
        'protocol': 'single_wire',
        'voltage': 5.0,
        'current_ma': 60,
        'channels': 3,
        'footprint': 'LED_SMD:LED_WS2812B_PLCC4_5.0x5.0mm',
    },
    'SK6812': {
        'type': 'addressable',
        'protocol': 'single_wire',
        'voltage': 5.0,
        'current_ma': 80,
        'channels': 4,
        'footprint': 'LED_SMD:LED_SK6812_PLCC4_5.0x5.0mm',
    },
    'APA102': {
        'type': 'addressable',
        'protocol': 'spi',
        'voltage': 5.0,
        'current_ma': 60,
        'channels': 3,
        'footprint': 'LED_SMD:LED_APA102_5.0x5.0mm',
    },
    'PCA9685': {
        'type': 'pwm_driver',
        'channels': 16,
        'resolution_bits': 12,
        'interface': ['i2c'],
        'voltage': 3.3,
        'footprint': 'Package_SO:TSSOP-28_4.4x9.7mm_P0.65mm',
    },
    'TLC5940': {
        'type': 'pwm_driver',
        'channels': 16,
        'resolution_bits': 12,
        'interface': ['spi'],
        'voltage': 5.0,
        'footprint': 'Package_SO:HTSSOP-28-1EP_4.4x9.7mm_P0.635mm_EP2.85x5.4mm',
    },
    'CAT4104': {
        'type': 'constant_current',
        'channels': 4,
        'current_max_ma': 75,
        'interface': ['pwm'],
        'voltage': 5.0,
        'footprint': 'Package_DFN_QFN:DFN-16-1EP_3x3mm_P0.45mm_EP1.65x2.35mm',
    },
}

# Audio components
AUDIO_DATABASE = {
    'MAX98357A': {
        'type': 'i2s_amp',
        'power_w': 3.2,
        'interface': ['i2s'],
        'voltage': 5.0,
        'current_ma': 10,
        'footprint': 'Package_DFN_QFN:TQFN-16-1EP_3x3mm_P0.5mm_EP1.23x1.23mm',
    },
    'PCM5102A': {
        'type': 'i2s_dac',
        'bits': 32,
        'sample_rate': 384000,
        'interface': ['i2s'],
        'voltage': 3.3,
        'current_ma': 15,
        'footprint': 'Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm',
    },
    'PAM8403': {
        'type': 'class_d_amp',
        'power_w': 3,
        'channels': 2,
        'interface': ['analog'],
        'voltage': 5.0,
        'current_ma': 50,
        'footprint': 'Package_SO:SOP-16_4.4x10.4mm_P1.27mm',
    },
    'LM386': {
        'type': 'audio_amp',
        'power_mw': 325,
        'gain': 200,
        'interface': ['analog'],
        'voltage': 9.0,
        'current_ma': 4,
        'footprint': 'Package_DIP:DIP-8_W7.62mm',
    },
    'SPH0645LM4H': {
        'type': 'mems_mic',
        'interface': ['i2s'],
        'voltage': 3.3,
        'current_ma': 0.6,
        'snr_db': 65,
        'footprint': 'Package_LGA:LGA-6_3.5x2.65mm_P1.27mm',
    },
}

# Power management ICs
POWER_IC_DATABASE = {
    'TP4056': {
        'type': 'battery_charger',
        'chemistry': 'li-ion',
        'charge_current_ma': 1000,
        'voltage': 5.0,
        'footprint': 'Package_SO:SOP-8_3.9x4.9mm_P1.27mm',
    },
    'MCP73831': {
        'type': 'battery_charger',
        'chemistry': 'li-ion',
        'charge_current_ma': 500,
        'voltage': 6.0,
        'footprint': 'Package_TO_SOT_SMD:SOT-23-5',
    },
    'DW01A': {
        'type': 'battery_protection',
        'chemistry': 'li-ion',
        'features': ['overcharge', 'overdischarge', 'overcurrent'],
        'footprint': 'Package_TO_SOT_SMD:SOT-23-6',
    },
    'FS8205A': {
        'type': 'load_switch',
        'rds_on_mohm': 25,
        'current_max': 6,
        'footprint': 'Package_TO_SOT_SMD:SOT-23-6',
    },
    'IP5306': {
        'type': 'powerbank_ic',
        'features': ['charging', 'boost', 'led_indicator'],
        'current_max': 2.1,
        'footprint': 'Package_SO:SOP-8_3.9x4.9mm_P1.27mm',
    },
    'TPS63020': {
        'type': 'buck_boost',
        'vin_range': (1.8, 5.5),
        'vout': 3.3,
        'current_max': 2.0,
        'efficiency': 0.96,
        'footprint': 'Package_DFN_QFN:QFN-14_3x3mm_P0.5mm',
    },
    'TPS61200': {
        'type': 'boost',
        'vin_min': 0.3,
        'vout_max': 5.5,
        'current_max': 0.5,
        'footprint': 'Package_SON:WSON-10_3x3mm_P0.5mm',
    },
    'LM2596': {
        'type': 'buck',
        'vin_max': 40,
        'vout_adj': True,
        'current_max': 3.0,
        'footprint': 'Package_TO_SOT_SMD:TO-263-5_TabPin3',
    },
    'XL6009': {
        'type': 'boost',
        'vin_max': 32,
        'vout_max': 38,
        'current_max': 4.0,
        'footprint': 'Package_TO_SOT_SMD:TO-263-5_TabPin3',
    },
}

# Connectors
CONNECTOR_DATABASE = {
    'USB_C_Receptacle': {
        'type': 'usb_c',
        'power_delivery': False,
        'pins': 16,
        'footprint': 'Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A',
    },
    'USB_Micro_B': {
        'type': 'usb_micro',
        'pins': 5,
        'footprint': 'Connector_USB:USB_Micro-B_Molex-105017-0001',
    },
    'JST_PH_2pin': {
        'type': 'battery',
        'pitch_mm': 2.0,
        'pins': 2,
        'footprint': 'Connector_JST:JST_PH_B2B-PH-K_1x02_P2.00mm_Vertical',
    },
    'JST_SH_4pin': {
        'type': 'i2c',
        'pitch_mm': 1.0,
        'pins': 4,
        'footprint': 'Connector_JST:JST_SH_BM04B-SRSS-TB_1x04-1MP_P1.00mm_Vertical',
    },
    'SD_Card_Slot': {
        'type': 'sd_card',
        'interface': 'sdio',
        'footprint': 'Connector_Card:SD_Card_Holder',
    },
    'Barrel_Jack_2.1mm': {
        'type': 'power',
        'voltage_max': 24,
        'footprint': 'Connector_BarrelJack:BarrelJack_Horizontal',
    },
    'Header_2x20': {
        'type': 'gpio',
        'pitch_mm': 2.54,
        'pins': 40,
        'footprint': 'Connector_PinHeader_2.54mm:PinHeader_2x20_P2.54mm_Vertical',
    },
    'RJ45_MagJack': {
        'type': 'ethernet',
        'integrated_magnetics': True,
        'footprint': 'Connector_RJ:RJ45_Amphenol_RJHSE5380',
    },
    'SMA_Edge': {
        'type': 'rf',
        'impedance': 50,
        'footprint': 'Connector_Coaxial:SMA_Amphenol_132289_EdgeMount',
    },
    'Screw_Terminal_2pin': {
        'type': 'power',
        'pitch_mm': 5.08,
        'current_max': 15,
        'footprint': 'TerminalBlock:TerminalBlock_bornier-2_P5.08mm',
    },
}

# Storage
STORAGE_DATABASE = {
    'W25Q32': {
        'type': 'flash',
        'capacity_mbit': 32,
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 15,
        'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
    },
    'W25Q128': {
        'type': 'flash',
        'capacity_mbit': 128,
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 15,
        'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
    },
    'AT24C256': {
        'type': 'eeprom',
        'capacity_kbit': 256,
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ma': 3,
        'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
    },
    '23LC1024': {
        'type': 'sram',
        'capacity_kbit': 1024,
        'interface': ['spi'],
        'voltage': 3.3,
        'current_ma': 3,
        'footprint': 'Package_DIP:DIP-8_W7.62mm',
    },
}

# RTC (Real-Time Clock)
RTC_DATABASE = {
    'DS3231': {
        'type': 'rtc',
        'accuracy_ppm': 2,
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ua': 200,
        'features': ['temperature_compensation', 'alarm', 'battery_backup'],
        'footprint': 'Package_SO:SOIC-16W_7.5x10.3mm_P1.27mm',
    },
    'DS1307': {
        'type': 'rtc',
        'accuracy_ppm': 20,
        'interface': ['i2c'],
        'voltage': 5.0,
        'current_ua': 500,
        'features': ['battery_backup'],
        'footprint': 'Package_DIP:DIP-8_W7.62mm',
    },
    'PCF8563': {
        'type': 'rtc',
        'accuracy_ppm': 20,
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_ua': 250,
        'features': ['alarm', 'timer'],
        'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
    },
    'RV-3028-C7': {
        'type': 'rtc',
        'accuracy_ppm': 1,
        'interface': ['i2c'],
        'voltage': 3.3,
        'current_na': 40,
        'features': ['temperature_compensation', 'unix_time'],
        'footprint': 'Package_LGA:LGA-8_2.5x2.5mm_P0.55mm',
    },
}


# =============================================================================
# ENHANCED NLP PATTERNS
# =============================================================================

# Semantic patterns for natural language parsing
NLP_PATTERNS = {
    # MCU detection patterns
    'mcu': {
        'esp32': [r'\besp32\b', r'\besp-32\b', r'esp32[-_]?s3', r'esp32[-_]?c3', r'esp32[-_]?wroom'],
        'esp8266': [r'\besp8266\b', r'esp[-_]?01', r'nodemcu'],
        'stm32': [r'\bstm32\b', r'stm32f\d', r'blue\s*pill', r'black\s*pill'],
        'atmega': [r'\batmega\b', r'\barduino\b', r'nano\b', r'uno\b', r'mega\b'],
        'rp2040': [r'\brp2040\b', r'\bpico\b', r'raspberry\s*pi\s*pico'],
        'nrf52': [r'\bnrf52\b', r'\bnrf\b', r'nordic'],
        'attiny': [r'\battiny\b', r'attiny85', r'attiny13', r'digispark'],
    },
    # Power source patterns
    'power': {
        'usb': [r'\busb\b', r'usb[-_]?c', r'micro[-_]?usb', r'5\s*v\s*usb'],
        'battery_lipo': [r'\blipo\b', r'li[-_]?po', r'lithium\s*polymer', r'3\.7\s*v\s*battery'],
        'battery_18650': [r'18650', r'lithium\s*ion', r'li[-_]?ion'],
        'battery_aa': [r'\baa\b\s*batter', r'alkaline', r'1\.5\s*v\s*batter'],
        'solar': [r'\bsolar\b', r'photovoltaic', r'pv\s*panel'],
        'poe': [r'\bpoe\b', r'power\s*over\s*ethernet'],
        'barrel': [r'barrel\s*jack', r'dc\s*jack', r'12\s*v\s*adapter', r'9\s*v\s*adapter'],
    },
    # Sensor patterns
    'sensors': {
        'temperature': [r'\btemperature\b', r'\btemp\b', r'thermometer', r'thermal'],
        'humidity': [r'\bhumidity\b', r'humid', r'moisture'],
        'pressure': [r'\bpressure\b', r'barometer', r'barometric', r'altimeter'],
        'imu': [r'\bimu\b', r'accelerometer', r'gyroscope', r'motion', r'orientation'],
        'magnetometer': [r'magnetometer', r'compass', r'magnetic'],
        'distance': [r'\bdistance\b', r'ultrasonic', r'lidar', r'tof', r'ranging'],
        'light': [r'\blight\b', r'\blux\b', r'ambient\s*light', r'brightness'],
        'color': [r'\bcolor\b', r'rgb\s*sensor', r'colour'],
        'gas': [r'\bgas\b', r'air\s*quality', r'co2', r'voc', r'smoke'],
        'current': [r'current\s*sens', r'power\s*monitor', r'ina219', r'acs712'],
        'weight': [r'\bweight\b', r'load\s*cell', r'scale', r'hx711'],
        'gps': [r'\bgps\b', r'gnss', r'location', r'position'],
        'heart_rate': [r'heart\s*rate', r'pulse', r'spo2', r'oximeter'],
        'sound': [r'\bmicrophone\b', r'\bmic\b', r'audio\s*input', r'sound\s*sensor'],
    },
    # Display patterns
    'display': {
        'oled': [r'\boled\b', r'ssd1306', r'sh1106', r'128x64', r'128x32'],
        'tft': [r'\btft\b', r'st7735', r'ili9341', r'st7789', r'color\s*display'],
        'lcd': [r'\blcd\b', r'16x2', r'20x4', r'character\s*display', r'hd44780'],
        'epaper': [r'e[-_]?paper', r'e[-_]?ink', r'epd'],
        'led_matrix': [r'led\s*matrix', r'max7219', r'8x8\s*led'],
        'seven_segment': [r'7[-_]?segment', r'seven\s*segment', r'digit\s*display'],
    },
    # Communication patterns
    'communication': {
        'wifi': [r'\bwifi\b', r'wi[-_]?fi', r'wireless\s*lan', r'802\.11'],
        'bluetooth': [r'\bbluetooth\b', r'\bble\b', r'bt\s*classic'],
        'lora': [r'\blora\b', r'lorawan', r'sx127', r'rfm9'],
        'rf': [r'\brf\b', r'nrf24', r'433\s*mhz', r'315\s*mhz', r'radio'],
        'zigbee': [r'\bzigbee\b', r'xbee', r'mesh\s*network'],
        'ethernet': [r'\bethernet\b', r'lan\b', r'rj45', r'w5500', r'enc28j60'],
        'can': [r'\bcan\b', r'can\s*bus', r'obd', r'automotive'],
        'rs485': [r'rs[-_]?485', r'modbus', r'industrial'],
        'infrared': [r'\bir\b', r'infrared', r'remote\s*control'],
    },
    # Motor patterns
    'motors': {
        'dc': [r'\bdc\s*motor\b', r'brushed\s*motor'],
        'stepper': [r'\bstepper\b', r'step\s*motor', r'nema\s*\d+'],
        'servo': [r'\bservo\b', r'rc\s*servo', r'pwm\s*servo'],
        'brushless': [r'brushless', r'bldc', r'esc'],
    },
    # LED patterns
    'leds': {
        'addressable': [r'addressable', r'neopixel', r'ws2812', r'sk6812', r'apa102', r'rgb\s*led'],
        'pwm': [r'pwm\s*led', r'dimmable', r'led\s*strip', r'led\s*channel'],
        'indicator': [r'status\s*led', r'indicator', r'signal\s*led'],
    },
    # Audio patterns
    'audio': {
        'speaker': [r'\bspeaker\b', r'\bamp\b', r'amplifier', r'audio\s*out'],
        'microphone': [r'\bmic\b', r'microphone', r'mems\s*mic', r'audio\s*in'],
        'buzzer': [r'\bbuzzer\b', r'piezo', r'beep'],
        'dac': [r'\bdac\b', r'audio\s*dac', r'i2s\s*out'],
    },
    # Quantity patterns
    'quantities': {
        'channels': [r'(\d+)\s*(?:ch|channel|channels)'],
        'count': [r'(\d+)\s*(?:x|pcs|pieces)'],
    },
}


# =============================================================================
# SCHEMATIC DATA STRUCTURES
# =============================================================================

@dataclass
class SchematicSymbol:
    """A symbol in the schematic"""
    reference: str
    value: str
    library: str
    symbol: str
    x: float
    y: float
    rotation: int = 0
    mirror: bool = False
    properties: Dict = field(default_factory=dict)


@dataclass
class SchematicWire:
    """A wire connection in the schematic"""
    start: Tuple[float, float]
    end: Tuple[float, float]
    net_name: str = ''


@dataclass
class SchematicLabel:
    """A net label in the schematic"""
    name: str
    x: float
    y: float
    rotation: int = 0
    style: str = 'local'  # local, global, hierarchical


@dataclass
class SchematicSheet:
    """A schematic sheet/page"""
    name: str
    width: float = 297.0  # A4 landscape
    height: float = 210.0
    symbols: List[SchematicSymbol] = field(default_factory=list)
    wires: List[SchematicWire] = field(default_factory=list)
    labels: List[SchematicLabel] = field(default_factory=list)
    title_block: Dict = field(default_factory=dict)


@dataclass
class Schematic:
    """Complete schematic document"""
    sheets: List[SchematicSheet] = field(default_factory=list)
    net_classes: Dict = field(default_factory=dict)
    design_rules: Dict = field(default_factory=dict)

    def to_kicad_sch(self) -> str:
        """Export to KiCad schematic format"""
        return generate_kicad_schematic(self)


# =============================================================================
# CIRCUIT AI
# =============================================================================

class CircuitAI:
    """
    AI-powered circuit design assistant.

    Interacts with users to gather requirements, selects components,
    and generates a complete parts database for the Parts Piston.

    Usage:
        ai = CircuitAI()

        # Interactive mode
        requirements = ai.interview_user()

        # Or programmatic mode
        requirements = CircuitRequirements(
            project_name='my_sensor',
            mcu_family=MCUFamily.ESP32,
            input_power=PowerType.USB_5V
        )
        ai.add_sensor_block('temperature', 'BME280')

        # Generate parts database
        result = ai.generate_parts_db(requirements)

        # Pass to Parts Piston
        parts_piston.build_from_dict(result.parts_db)
    """

    def __init__(self):
        self.requirements = CircuitRequirements()
        self.conversation_history: List[Dict] = []
        self.current_questions: List[UserQuestion] = []

    # =========================================================================
    # STANDARD PISTON API
    # =========================================================================

    def suggest(self, parts_db: Dict) -> Dict[str, Any]:
        """
        Standard piston API - suggest improvements for the design.

        Args:
            parts_db: Parts database to analyze

        Returns:
            Dictionary with suggestions
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        suggestions = []

        # Check for decoupling capacitors
        has_ic = any('U' in ref for ref in parts.keys())
        cap_count = sum(1 for ref in parts.keys() if ref.startswith('C'))
        ic_count = sum(1 for ref in parts.keys() if ref.startswith('U'))

        if has_ic and cap_count < ic_count:
            suggestions.append({
                'type': 'decoupling',
                'message': f'Consider adding more decoupling capacitors. Found {ic_count} ICs but only {cap_count} capacitors.',
                'severity': 'warning'
            })

        # Check for pull-up resistors on I2C
        for net_name in nets.keys():
            if 'SDA' in net_name.upper() or 'SCL' in net_name.upper():
                # Check if there's a pull-up resistor
                has_pullup = any(
                    ref.startswith('R') and net_name in str(parts.get(ref, {}))
                    for ref in parts.keys()
                )
                if not has_pullup:
                    suggestions.append({
                        'type': 'pullup',
                        'message': f'I2C net {net_name} may need a pull-up resistor',
                        'severity': 'info'
                    })

        return {
            'suggestions': suggestions,
            'component_count': len(parts),
            'net_count': len(nets),
            'analysis_complete': True
        }

    # =========================================================================
    # USER INTERACTION
    # =========================================================================

    def get_initial_questions(self) -> List[UserQuestion]:
        """Get initial questions to ask the user"""
        return [
            UserQuestion(
                question="What is your project name?",
                field="project_name",
                default="my_project"
            ),
            UserQuestion(
                question="Describe your project in a few sentences:",
                field="description",
                required=False
            ),
            UserQuestion(
                question="What is the main application?",
                field="application",
                options=["IoT", "Consumer Electronics", "Industrial", "Wearable",
                        "Automotive", "Medical", "Education/Hobby", "Other"]
            ),
            UserQuestion(
                question="What power source will you use?",
                field="input_power",
                options=["USB 5V", "LiPo Battery", "18650 Battery", "AA Batteries",
                        "DC Barrel Jack", "Solar", "External Supply"]
            ),
            UserQuestion(
                question="Which microcontroller family do you prefer?",
                field="mcu_family",
                options=["ESP32 (WiFi+BLE)", "STM32", "ATmega/Arduino",
                        "RP2040 (Raspberry Pi)", "nRF52 (BLE)", "No MCU needed"]
            ),
        ]

    def get_follow_up_questions(self, requirements: CircuitRequirements) -> List[UserQuestion]:
        """Get follow-up questions based on initial answers"""
        questions = []

        # Power-related questions
        if requirements.input_power in [PowerType.BATTERY_LIPO, PowerType.BATTERY_18650]:
            questions.append(UserQuestion(
                question="What battery capacity do you need (mAh)?",
                field="battery_capacity_mah",
                options=["500", "1000", "2000", "3000", "5000"],
                default="1000"
            ))
            questions.append(UserQuestion(
                question="Do you need battery charging on the board?",
                field="needs_charging",
                options=["Yes", "No"],
                default="Yes"
            ))

        # MCU-related questions
        if requirements.mcu_family != MCUFamily.CUSTOM:
            questions.append(UserQuestion(
                question="What features does your MCU need?",
                field="mcu_features",
                options=["WiFi", "Bluetooth/BLE", "USB", "CAN Bus",
                        "Ethernet", "LCD/Display", "Camera"],
                default=[]
            ))
            questions.append(UserQuestion(
                question="How many GPIO pins do you need approximately?",
                field="gpio_count",
                options=["< 10", "10-20", "20-30", "30+"],
                default="10-20"
            ))

        # Peripheral questions
        questions.append(UserQuestion(
            question="What peripherals/sensors do you need?",
            field="peripherals",
            options=["Temperature/Humidity", "Motion/IMU", "Distance/Proximity",
                    "Light/Color", "GPS", "Display (OLED/LCD)", "Motor Driver",
                    "Audio", "SD Card", "RTC (Real-Time Clock)", "None"],
            default=[]
        ))

        # Physical constraints
        questions.append(UserQuestion(
            question="What is your target board size?",
            field="board_size",
            options=["Tiny (<25x25mm)", "Small (25-50mm)", "Medium (50-100mm)",
                    "Large (>100mm)", "No constraint"],
            default="Small (25-50mm)"
        ))

        return questions

    def parse_user_response(self, response: str, question: UserQuestion) -> Any:
        """Parse user response and update requirements"""
        field = question.field

        # Handle different field types
        if field == 'project_name':
            return response.strip() or question.default

        elif field == 'input_power':
            power_map = {
                'usb': PowerType.USB_5V,
                'lipo': PowerType.BATTERY_LIPO,
                '18650': PowerType.BATTERY_18650,
                'aa': PowerType.BATTERY_AA,
                'barrel': PowerType.DC_BARREL,
                'solar': PowerType.SOLAR,
                'external': PowerType.EXTERNAL,
            }
            resp_lower = response.lower()
            for key, ptype in power_map.items():
                if key in resp_lower:
                    return ptype
            return PowerType.USB_5V

        elif field == 'mcu_family':
            mcu_map = {
                'esp32': MCUFamily.ESP32,
                'esp8266': MCUFamily.ESP8266,
                'stm32': MCUFamily.STM32,
                'atmega': MCUFamily.ATMEGA,
                'arduino': MCUFamily.ATMEGA,
                'rp2040': MCUFamily.RP2040,
                'pico': MCUFamily.RP2040,
                'nrf': MCUFamily.NRF52,
            }
            resp_lower = response.lower()
            for key, mcu in mcu_map.items():
                if key in resp_lower:
                    return mcu
            return MCUFamily.ESP32

        elif field == 'board_size':
            size_map = {
                'tiny': (25.0, 25.0),
                'small': (50.0, 50.0),
                'medium': (100.0, 100.0),
                'large': (150.0, 150.0),
            }
            resp_lower = response.lower()
            for key, size in size_map.items():
                if key in resp_lower:
                    return size
            return (50.0, 50.0)

        return response

    def parse_natural_language(self, text: str) -> CircuitRequirements:
        """
        Parse natural language description into requirements.

        Uses the EnhancedNLPParser for advanced pattern matching
        and semantic analysis.

        Example inputs:
        - "I need an ESP32-based temperature logger with WiFi"
        - "Battery-powered motion sensor with BLE"
        - "USB-powered LED controller with 8 channels"
        - "STM32 motor controller for 2 stepper motors with CAN bus"
        - "ESP32-S3 with BME280, OLED display, and LoRa communication"
        """
        # Use the enhanced NLP parser
        parser = EnhancedNLPParser()
        return parser.parse(text)

    def parse_natural_language_legacy(self, text: str) -> CircuitRequirements:
        """
        Legacy parser - kept for compatibility.
        Use parse_natural_language() for enhanced parsing.
        """
        text_lower = text.lower()
        requirements = CircuitRequirements()

        # Detect MCU
        if 'esp32' in text_lower:
            requirements.mcu_family = MCUFamily.ESP32
            if 's3' in text_lower:
                requirements.mcu_features.append('usb_otg')
        elif 'esp8266' in text_lower:
            requirements.mcu_family = MCUFamily.ESP8266
        elif 'stm32' in text_lower:
            requirements.mcu_family = MCUFamily.STM32
        elif 'arduino' in text_lower or 'atmega' in text_lower:
            requirements.mcu_family = MCUFamily.ATMEGA
        elif 'pico' in text_lower or 'rp2040' in text_lower:
            requirements.mcu_family = MCUFamily.RP2040

        # Detect power source
        if 'battery' in text_lower or 'lipo' in text_lower:
            requirements.input_power = PowerType.BATTERY_LIPO
        elif 'usb' in text_lower:
            requirements.input_power = PowerType.USB_5V
        elif 'solar' in text_lower:
            requirements.input_power = PowerType.SOLAR

        # Detect features
        if 'wifi' in text_lower:
            requirements.mcu_features.append('wifi')
        if 'ble' in text_lower or 'bluetooth' in text_lower:
            requirements.mcu_features.append('ble')

        # Detect sensors
        if 'temperature' in text_lower:
            block = self._create_sensor_block('temperature')
            requirements.blocks.append(block)
        if 'humidity' in text_lower:
            block = self._create_sensor_block('humidity')
            requirements.blocks.append(block)
        if 'motion' in text_lower or 'imu' in text_lower or 'accelerometer' in text_lower:
            block = self._create_sensor_block('imu')
            requirements.blocks.append(block)
        if 'distance' in text_lower or 'ultrasonic' in text_lower:
            block = self._create_sensor_block('distance')
            requirements.blocks.append(block)

        # Detect peripherals
        if 'display' in text_lower or 'oled' in text_lower or 'lcd' in text_lower:
            block = CircuitBlock(
                name='display',
                block_type=CircuitBlockType.DISPLAY,
                power=PowerRequirement(voltage=3.3, current_max=0.05)
            )
            requirements.blocks.append(block)

        if 'motor' in text_lower:
            block = CircuitBlock(
                name='motor_driver',
                block_type=CircuitBlockType.MOTOR_DRIVER,
                power=PowerRequirement(voltage=12.0, current_max=2.0)
            )
            requirements.blocks.append(block)

        if 'led' in text_lower:
            # Detect LED count
            led_count = 1
            match = re.search(r'(\d+)\s*(?:ch|channel|led)', text_lower)
            if match:
                led_count = int(match.group(1))

            block = CircuitBlock(
                name='led_driver',
                block_type=CircuitBlockType.INTERFACE,
                description=f'{led_count} channel LED driver'
            )
            requirements.blocks.append(block)

        return requirements

    def generate_schematic(self, parts_db: Dict = None,
                          requirements: CircuitRequirements = None) -> Schematic:
        """
        Generate a schematic from parts database.

        Returns a Schematic object that can be exported to KiCad format.
        """
        if parts_db is None:
            result = self.generate_parts_db(requirements or self.requirements)
            parts_db = result.parts_db

        generator = SchematicGenerator()
        return generator.generate(parts_db, requirements or self.requirements)

    def export_schematic_kicad(self, parts_db: Dict = None,
                               requirements: CircuitRequirements = None,
                               output_path: str = None) -> str:
        """
        Export schematic to KiCad format.

        Args:
            parts_db: Parts database (optional, will generate if not provided)
            requirements: Circuit requirements
            output_path: Output file path (optional)

        Returns:
            KiCad schematic content as string
        """
        schematic = self.generate_schematic(parts_db, requirements)
        content = schematic.to_kicad_sch()

        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)

        return content

    def export_netlist(self, parts_db: Dict = None,
                       format: str = 'kicad') -> str:
        """
        Export netlist in various formats.

        Args:
            parts_db: Parts database
            format: 'kicad', 'spice', or 'json'

        Returns:
            Netlist content as string
        """
        if parts_db is None:
            result = self.generate_parts_db()
            parts_db = result.parts_db

        if format == 'kicad':
            return generate_netlist_kicad(parts_db)
        elif format == 'spice':
            return generate_spice_netlist(parts_db, self.requirements.project_name)
        elif format == 'json':
            import json
            return json.dumps(parts_db, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

    # =========================================================================
    # CIRCUIT BLOCK CREATION
    # =========================================================================

    def _create_sensor_block(self, sensor_type: str) -> CircuitBlock:
        """Create a sensor circuit block"""
        sensor_map = {
            'temperature': ('BME280', 'Temperature/Humidity/Pressure sensor'),
            'humidity': ('BME280', 'Temperature/Humidity/Pressure sensor'),
            'pressure': ('BME280', 'Temperature/Humidity/Pressure sensor'),
            'imu': ('MPU6050', '6-axis IMU (Accelerometer + Gyroscope)'),
            'distance': ('HC-SR04', 'Ultrasonic distance sensor'),
            'light': ('BH1750', 'Ambient light sensor'),
        }

        component, description = sensor_map.get(sensor_type, ('Generic', 'Sensor'))
        sensor_data = SENSOR_DATABASE.get(component, {})

        return CircuitBlock(
            name=f'sensor_{sensor_type}',
            block_type=CircuitBlockType.SENSOR,
            description=description,
            power=PowerRequirement(
                voltage=sensor_data.get('voltage', 3.3),
                current_max=sensor_data.get('current_ma', 10) / 1000
            ),
            pins=[PinRequirement(
                name=iface,
                interface=InterfaceType(iface) if iface in ['i2c', 'spi'] else InterfaceType.DIGITAL
            ) for iface in sensor_data.get('interface', ['i2c'])]
        )

    def create_power_block(self, input_type: PowerType, output_voltages: List[float]) -> List[CircuitBlock]:
        """Create power supply circuit blocks"""
        blocks = []

        # Input protection
        blocks.append(CircuitBlock(
            name='power_input',
            block_type=CircuitBlockType.POWER_INPUT,
            description=f'{input_type.value} power input',
            power=PowerRequirement(voltage=5.0, current_max=2.0)
        ))

        # Add regulators for each output voltage
        for vout in output_voltages:
            if vout == 3.3:
                regulator = 'AP2112K-3.3'
            elif vout == 5.0:
                regulator = 'AMS1117-5.0'
            else:
                regulator = 'adjustable'

            blocks.append(CircuitBlock(
                name=f'regulator_{vout}V',
                block_type=CircuitBlockType.POWER_REGULATOR,
                description=f'{vout}V voltage regulator ({regulator})',
                power=PowerRequirement(voltage=vout, current_max=1.0)
            ))

        return blocks

    def create_mcu_block(self, mcu_family: MCUFamily, features: List[str]) -> CircuitBlock:
        """Create MCU circuit block"""
        # Select best MCU based on requirements
        mcu_name = self._select_mcu(mcu_family, features)
        mcu_data = MCU_DATABASE.get(mcu_name, {})

        pins = []
        # Add power pins
        pins.append(PinRequirement(name='VCC', interface=InterfaceType.ANALOG, direction='input'))
        pins.append(PinRequirement(name='GND', interface=InterfaceType.ANALOG, direction='input'))

        # Add I2C if needed
        if mcu_data.get('i2c', 0) > 0:
            pins.append(PinRequirement(name='SDA', interface=InterfaceType.I2C, pull_up=True))
            pins.append(PinRequirement(name='SCL', interface=InterfaceType.I2C, pull_up=True))

        # Add SPI if needed
        if mcu_data.get('spi', 0) > 0:
            pins.extend([
                PinRequirement(name='MOSI', interface=InterfaceType.SPI),
                PinRequirement(name='MISO', interface=InterfaceType.SPI),
                PinRequirement(name='SCK', interface=InterfaceType.SPI, is_critical=True),
                PinRequirement(name='CS', interface=InterfaceType.SPI),
            ])

        # Add UART
        if mcu_data.get('uart', 0) > 0:
            pins.extend([
                PinRequirement(name='TX', interface=InterfaceType.UART, direction='output'),
                PinRequirement(name='RX', interface=InterfaceType.UART, direction='input'),
            ])

        return CircuitBlock(
            name='mcu',
            block_type=CircuitBlockType.MCU,
            description=f'{mcu_name} microcontroller',
            power=PowerRequirement(
                voltage=mcu_data.get('voltage', 3.3),
                current_typical=mcu_data.get('current_typical', 0.05),
                current_max=mcu_data.get('current_max', 0.5)
            ),
            pins=pins,
            selected_components=[{'name': mcu_name, 'footprint': mcu_data.get('footprint', '')}]
        )

    def _select_mcu(self, family: MCUFamily, features: List[str]) -> str:
        """Select best MCU for requirements"""
        if family == MCUFamily.ESP32:
            if 'usb_otg' in features or 'camera' in features:
                return 'ESP32-S3-WROOM-1'
            return 'ESP32-WROOM-32'
        elif family == MCUFamily.STM32:
            return 'STM32F103C8T6'
        elif family == MCUFamily.ATMEGA:
            return 'ATMEGA328P'
        elif family == MCUFamily.RP2040:
            return 'RP2040'
        return 'ESP32-WROOM-32'

    # =========================================================================
    # PARTS DATABASE GENERATION
    # =========================================================================

    def generate_parts_db(self, requirements: CircuitRequirements = None) -> CircuitAIResult:
        """
        Generate complete parts database from requirements.

        This is the main output that feeds into the Parts Piston.
        """
        if requirements is None:
            requirements = self.requirements

        parts_db = {
            'parts': {},
            'nets': {},
            'metadata': {
                'project_name': requirements.project_name,
                'description': requirements.description,
                'generated_by': 'CircuitAI'
            }
        }

        warnings = []
        suggestions = []
        bom_preview = []
        power_tree = {}
        topology = {'blocks': [], 'connections': []}

        ref_counters = defaultdict(int)

        # Generate power supply components
        power_blocks = self.create_power_block(
            requirements.input_power,
            requirements.output_voltages
        )

        for block in power_blocks:
            components = self._generate_block_components(block, ref_counters)
            parts_db['parts'].update(components)
            topology['blocks'].append({
                'name': block.name,
                'type': block.block_type.value,
                'components': list(components.keys())
            })

        # Generate MCU components
        mcu_block = self.create_mcu_block(
            requirements.mcu_family,
            requirements.mcu_features
        )
        mcu_components = self._generate_block_components(mcu_block, ref_counters)
        parts_db['parts'].update(mcu_components)
        topology['blocks'].append({
            'name': 'mcu',
            'type': 'mcu',
            'components': list(mcu_components.keys())
        })

        # Generate peripheral components
        for block in requirements.blocks:
            components = self._generate_block_components(block, ref_counters)
            parts_db['parts'].update(components)
            topology['blocks'].append({
                'name': block.name,
                'type': block.block_type.value,
                'components': list(components.keys())
            })

        # Generate nets (connections)
        parts_db['nets'] = self._generate_nets(parts_db['parts'], requirements)

        # Generate power tree
        power_tree = self._generate_power_tree(requirements)

        # Generate BOM preview
        bom_preview = self._generate_bom_preview(parts_db['parts'])

        # Add suggestions
        suggestions.extend(self._generate_suggestions(requirements))

        return CircuitAIResult(
            requirements=requirements,
            parts_db=parts_db,
            topology=topology,
            power_tree=power_tree,
            bom_preview=bom_preview,
            warnings=warnings,
            suggestions=suggestions
        )

    def _generate_block_components(self, block: CircuitBlock,
                                   ref_counters: Dict[str, int]) -> Dict[str, Dict]:
        """Generate component entries for a circuit block"""
        components = {}

        if block.block_type == CircuitBlockType.POWER_REGULATOR:
            # Add voltage regulator
            ref_counters['U'] += 1
            ref = f"U{ref_counters['U']}"

            # Select regulator
            vout = block.power.voltage if block.power else 3.3
            if vout == 3.3:
                reg_data = REGULATOR_DATABASE.get('AP2112K-3.3', {})
                value = 'AP2112K-3.3'
            else:
                reg_data = REGULATOR_DATABASE.get('AMS1117-3.3', {})
                value = f'LDO {vout}V'

            components[ref] = {
                'value': value,
                'footprint': reg_data.get('footprint', 'Package_TO_SOT_SMD:SOT-23-5'),
                'description': block.description,
                'group': 'power',
                'pins': [
                    {'number': '1', 'name': 'VIN', 'net': 'VIN'},
                    {'number': '2', 'name': 'GND', 'net': 'GND'},
                    {'number': '3', 'name': 'EN', 'net': 'VIN'},
                    {'number': '4', 'name': 'NC', 'net': ''},
                    {'number': '5', 'name': 'VOUT', 'net': f'V{vout}'.replace('.', 'V')},
                ]
            }

            # Add input capacitor
            ref_counters['C'] += 1
            components[f"C{ref_counters['C']}"] = {
                'value': '10uF',
                'footprint': 'Capacitor_SMD:C_0805_2012Metric',
                'description': 'Input capacitor',
                'group': 'power',
                'pins': [
                    {'number': '1', 'net': 'VIN'},
                    {'number': '2', 'net': 'GND'},
                ]
            }

            # Add output capacitor
            ref_counters['C'] += 1
            components[f"C{ref_counters['C']}"] = {
                'value': '10uF',
                'footprint': 'Capacitor_SMD:C_0805_2012Metric',
                'description': 'Output capacitor',
                'group': 'power',
                'pins': [
                    {'number': '1', 'net': f'V{vout}'.replace('.', 'V')},
                    {'number': '2', 'net': 'GND'},
                ]
            }

        elif block.block_type == CircuitBlockType.MCU:
            ref_counters['U'] += 1
            ref = f"U{ref_counters['U']}"

            if block.selected_components:
                comp = block.selected_components[0]
                mcu_data = MCU_DATABASE.get(comp['name'], {})

                components[ref] = {
                    'value': comp['name'],
                    'footprint': comp.get('footprint', mcu_data.get('footprint', '')),
                    'description': block.description,
                    'group': 'mcu',
                    'electrical': {
                        'max_current': mcu_data.get('current_max', 0.5),
                        'max_voltage': mcu_data.get('voltage', 3.3),
                    },
                    'pins': self._generate_mcu_pins(comp['name'])
                }

            # Add decoupling capacitors
            for i in range(2):
                ref_counters['C'] += 1
                components[f"C{ref_counters['C']}"] = {
                    'value': '100nF',
                    'footprint': 'Capacitor_SMD:C_0402_1005Metric',
                    'description': 'MCU decoupling capacitor',
                    'group': 'mcu',
                    'pins': [
                        {'number': '1', 'net': 'V3V3'},
                        {'number': '2', 'net': 'GND'},
                    ]
                }

        elif block.block_type == CircuitBlockType.SENSOR:
            ref_counters['U'] += 1
            ref = f"U{ref_counters['U']}"

            # Get sensor from database
            sensor_name = block.description.split()[0] if block.description else 'BME280'
            for name, data in SENSOR_DATABASE.items():
                if name in block.description or block.name:
                    sensor_name = name
                    break

            sensor_data = SENSOR_DATABASE.get(sensor_name, {})

            components[ref] = {
                'value': sensor_name,
                'footprint': sensor_data.get('footprint', ''),
                'description': block.description,
                'group': 'sensor',
                'pins': self._generate_sensor_pins(sensor_name)
            }

            # Add I2C pull-ups if needed
            if 'i2c' in sensor_data.get('interface', []):
                for pin_name in ['SDA', 'SCL']:
                    ref_counters['R'] += 1
                    components[f"R{ref_counters['R']}"] = {
                        'value': '4.7k',
                        'footprint': 'Resistor_SMD:R_0402_1005Metric',
                        'description': f'I2C {pin_name} pull-up',
                        'group': 'sensor',
                        'pins': [
                            {'number': '1', 'net': 'V3V3'},
                            {'number': '2', 'net': pin_name},
                        ]
                    }

        elif block.block_type == CircuitBlockType.POWER_INPUT:
            # Add input connector
            ref_counters['J'] += 1
            ref = f"J{ref_counters['J']}"

            if 'usb' in block.description.lower():
                components[ref] = {
                    'value': 'USB_C',
                    'footprint': 'Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A',
                    'description': 'USB-C power input',
                    'group': 'power',
                    'pins': [
                        {'number': 'A1', 'name': 'GND', 'net': 'GND'},
                        {'number': 'A4', 'name': 'VBUS', 'net': 'VIN'},
                        {'number': 'B1', 'name': 'GND', 'net': 'GND'},
                        {'number': 'B4', 'name': 'VBUS', 'net': 'VIN'},
                    ]
                }
            else:
                components[ref] = {
                    'value': 'Barrel_Jack',
                    'footprint': 'Connector_BarrelJack:BarrelJack_Horizontal',
                    'description': 'DC power input',
                    'group': 'power',
                    'pins': [
                        {'number': '1', 'name': 'TIP', 'net': 'VIN'},
                        {'number': '2', 'name': 'SLEEVE', 'net': 'GND'},
                    ]
                }

        return components

    def _generate_mcu_pins(self, mcu_name: str) -> List[Dict]:
        """Generate pin list for an MCU"""
        mcu_data = MCU_DATABASE.get(mcu_name, {})
        pins = []

        # Basic power pins
        pins.extend([
            {'number': '1', 'name': 'GND', 'net': 'GND', 'type': 'power'},
            {'number': '2', 'name': '3V3', 'net': 'V3V3', 'type': 'power'},
        ])

        # Add GPIO pins
        gpio_count = mcu_data.get('gpio', 20)
        for i in range(min(gpio_count, 10)):  # Limit for brevity
            pins.append({
                'number': str(i + 3),
                'name': f'GPIO{i}',
                'net': '',
                'type': 'io'
            })

        return pins

    def _generate_sensor_pins(self, sensor_name: str) -> List[Dict]:
        """Generate pin list for a sensor"""
        sensor_data = SENSOR_DATABASE.get(sensor_name, {})
        pins = [
            {'number': '1', 'name': 'VCC', 'net': 'V3V3'},
            {'number': '2', 'name': 'GND', 'net': 'GND'},
        ]

        if 'i2c' in sensor_data.get('interface', []):
            pins.extend([
                {'number': '3', 'name': 'SDA', 'net': 'SDA'},
                {'number': '4', 'name': 'SCL', 'net': 'SCL'},
            ])
        elif 'spi' in sensor_data.get('interface', []):
            pins.extend([
                {'number': '3', 'name': 'MOSI', 'net': 'MOSI'},
                {'number': '4', 'name': 'MISO', 'net': 'MISO'},
                {'number': '5', 'name': 'SCK', 'net': 'SCK'},
                {'number': '6', 'name': 'CS', 'net': 'CS_SENSOR'},
            ])

        return pins

    def _generate_nets(self, parts: Dict, requirements: CircuitRequirements) -> Dict:
        """Generate net connections from parts"""
        nets = defaultdict(lambda: {'pins': []})

        for ref, part in parts.items():
            for pin in part.get('pins', []):
                net_name = pin.get('net', '')
                if net_name:
                    pin_num = pin.get('number', '')
                    nets[net_name]['pins'].append((ref, pin_num))

        # Add net properties
        for net_name in nets:
            net_upper = net_name.upper()
            if 'GND' in net_upper:
                nets[net_name]['class'] = 'ground'
            elif any(x in net_upper for x in ['VCC', 'VDD', 'VIN', 'V3V3', 'V5V']):
                nets[net_name]['class'] = 'power'
            elif any(x in net_upper for x in ['CLK', 'SCK']):
                nets[net_name]['class'] = 'high_speed'
            else:
                nets[net_name]['class'] = 'signal'

        return dict(nets)

    def _generate_power_tree(self, requirements: CircuitRequirements) -> Dict:
        """Generate power distribution tree"""
        tree = {
            'input': {
                'type': requirements.input_power.value,
                'voltage': requirements.input_voltage_range,
            },
            'rails': []
        }

        for vout in requirements.output_voltages:
            rail = {
                'voltage': vout,
                'consumers': []
            }

            # Add MCU
            mcu_data = MCU_DATABASE.get(self._select_mcu(requirements.mcu_family, []), {})
            if mcu_data.get('voltage', 3.3) == vout:
                rail['consumers'].append({
                    'name': 'MCU',
                    'current_typ': mcu_data.get('current_typical', 0.05),
                    'current_max': mcu_data.get('current_max', 0.5)
                })

            # Add peripherals
            for block in requirements.blocks:
                if block.power and block.power.voltage == vout:
                    rail['consumers'].append({
                        'name': block.name,
                        'current_typ': block.power.current_typical,
                        'current_max': block.power.current_max
                    })

            tree['rails'].append(rail)

        return tree

    def _generate_bom_preview(self, parts: Dict) -> List[Dict]:
        """Generate preliminary BOM"""
        bom = []
        groups = defaultdict(list)

        for ref, part in parts.items():
            key = (part.get('value', ''), part.get('footprint', ''))
            groups[key].append(ref)

        for (value, footprint), refs in groups.items():
            bom.append({
                'refs': ', '.join(sorted(refs)),
                'value': value,
                'footprint': footprint.split(':')[-1] if ':' in footprint else footprint,
                'quantity': len(refs)
            })

        return sorted(bom, key=lambda x: x['refs'])

    def _generate_suggestions(self, requirements: CircuitRequirements) -> List[str]:
        """Generate design suggestions"""
        suggestions = []

        # Power suggestions
        if requirements.input_power == PowerType.BATTERY_LIPO:
            suggestions.append("Consider adding battery protection (DW01A + FS8205A)")
            suggestions.append("Add battery charging IC (TP4056 or MCP73831)")

        if requirements.mcu_family == MCUFamily.ESP32:
            suggestions.append("Add 10uF + 100nF decoupling on 3.3V rail for WiFi stability")
            suggestions.append("Consider adding EN/BOOT control circuit with auto-reset")

        # Size suggestions
        if requirements.board_size_mm[0] < 30:
            suggestions.append("For small board: Use 0402 passives and QFN packages")

        return suggestions

    # =========================================================================
    # CONVERSATION INTERFACE
    # =========================================================================

    def get_conversation_prompt(self) -> str:
        """Get a prompt for conversation with user"""
        return """
I'm your Circuit Design AI assistant. I'll help you design your circuit by asking a few questions.

Tell me about your project! You can describe it naturally, for example:
- "I need an ESP32-based temperature and humidity logger"
- "Battery-powered motion sensor with Bluetooth"
- "USB-powered RGB LED controller with 16 channels"

Or just tell me what you're trying to build, and I'll ask follow-up questions.
"""

    def format_result_summary(self, result: CircuitAIResult) -> str:
        """Format result for user display"""
        lines = [
            f"\n{'='*60}",
            f"CIRCUIT DESIGN SUMMARY: {result.requirements.project_name}",
            f"{'='*60}",
            f"\nDescription: {result.requirements.description}",
            f"\nPower Input: {result.requirements.input_power.value}",
            f"Output Voltages: {result.requirements.output_voltages}",
            f"MCU: {result.requirements.mcu_family.value}",
            f"\nCircuit Blocks: {len(result.topology['blocks'])}",
        ]

        for block in result.topology['blocks']:
            lines.append(f"  - {block['name']} ({block['type']}): {len(block['components'])} components")

        lines.append(f"\nTotal Components: {len(result.parts_db['parts'])}")
        lines.append(f"Total Nets: {len(result.parts_db['nets'])}")

        lines.append("\nBOM Preview:")
        for item in result.bom_preview[:10]:
            lines.append(f"  {item['refs']}: {item['value']} x{item['quantity']}")

        if result.suggestions:
            lines.append("\nSuggestions:")
            for s in result.suggestions:
                lines.append(f"  - {s}")

        if result.warnings:
            lines.append("\nWarnings:")
            for w in result.warnings:
                lines.append(f"  ! {w}")

        lines.append(f"\n{'='*60}")
        lines.append("Parts database ready for Parts Piston!")

        return '\n'.join(lines)


# =============================================================================
# ENGINE HANDOFF
# =============================================================================

@dataclass
class EngineHandoff:
    """
    Complete data package for handoff to PCB Engine.

    Contains everything the engine needs to design the PCB:
    - Parts database with all components and nets
    - Circuit requirements and constraints
    - Board specifications
    - AI context for decision making
    """
    # Core data
    parts_db: Dict
    requirements: CircuitRequirements

    # Computed analysis
    topology: Dict
    power_tree: Dict
    bom_preview: List[Dict]

    # Constraints for engine
    board_constraints: Dict = field(default_factory=dict)
    design_rules: Dict = field(default_factory=dict)

    # AI context
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    user_preferences: Dict = field(default_factory=dict)

    # Metadata
    project_name: str = ''
    description: str = ''
    generated_at: str = ''


class CircuitAIEngineInterface:
    """
    Interface between Circuit AI and PCB Engine.

    Handles the complete flow:
    1. Circuit AI gathers requirements
    2. Creates complete handoff package
    3. Passes to PCB Engine
    4. Receives challenges from engine
    5. Makes decisions or asks user
    6. Approves final output

    Usage:
        interface = CircuitAIEngineInterface()

        # Natural language input
        result = interface.design_from_description(
            "ESP32 temperature logger with WiFi and OLED display"
        )

        # Or with interactive requirements
        result = interface.design_interactive()
    """

    def __init__(self):
        self.circuit_ai = CircuitAI()
        self.current_handoff: EngineHandoff = None
        self.engine_result = None

    def create_handoff(self, result: CircuitAIResult) -> EngineHandoff:
        """
        Create a complete handoff package from CircuitAI result.

        This packages ALL information the PCB Engine needs.
        """
        import time

        # Extract board constraints from requirements
        board_constraints = {
            'width': result.requirements.board_size_mm[0],
            'height': result.requirements.board_size_mm[1],
            'layers': result.requirements.layers,
            'smd_only': result.requirements.smd_only,
        }

        # Design rules based on requirements
        design_rules = {
            'trace_width': 0.25,  # Default, could be computed
            'clearance': 0.15,
            'via_diameter': 0.8,
            'via_drill': 0.4,
        }

        # Adjust for small boards
        if result.requirements.board_size_mm[0] < 30:
            design_rules['trace_width'] = 0.15
            design_rules['clearance'] = 0.1
            design_rules['via_diameter'] = 0.6
            design_rules['via_drill'] = 0.3

        handoff = EngineHandoff(
            parts_db=result.parts_db,
            requirements=result.requirements,
            topology=result.topology,
            power_tree=result.power_tree,
            bom_preview=result.bom_preview,
            board_constraints=board_constraints,
            design_rules=design_rules,
            warnings=result.warnings,
            suggestions=result.suggestions,
            project_name=result.requirements.project_name,
            description=result.requirements.description,
            generated_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        self.current_handoff = handoff
        return handoff

    def design_from_description(self, description: str,
                                 ai_callback=None) -> 'EngineResult':
        """
        Complete design flow from natural language description.

        Args:
            description: Natural language description of the circuit
            ai_callback: Optional callback for AI Agent decisions

        Returns:
            EngineResult from PCB Engine
        """
        # Step 1: Parse requirements
        print("=" * 60)
        print("CIRCUIT AI - Parsing Requirements")
        print("=" * 60)
        print(f"Input: {description}")

        requirements = self.circuit_ai.parse_natural_language(description)
        print(f"  MCU: {requirements.mcu_family.value}")
        print(f"  Power: {requirements.input_power.value}")
        print(f"  Blocks: {len(requirements.blocks)}")

        # Step 2: Generate parts database
        print("\n" + "=" * 60)
        print("CIRCUIT AI - Generating Parts Database")
        print("=" * 60)

        result = self.circuit_ai.generate_parts_db(requirements)
        print(f"  Components: {len(result.parts_db['parts'])}")
        print(f"  Nets: {len(result.parts_db['nets'])}")

        # Step 3: Create handoff package
        print("\n" + "=" * 60)
        print("CIRCUIT AI - Creating Engine Handoff")
        print("=" * 60)

        handoff = self.create_handoff(result)
        print(f"  Board: {handoff.board_constraints['width']}x{handoff.board_constraints['height']}mm")
        print(f"  Layers: {handoff.board_constraints['layers']}")
        print(f"  Warnings: {len(handoff.warnings)}")
        print(f"  Suggestions: {len(handoff.suggestions)}")

        # Step 4: Hand off to PCB Engine
        print("\n" + "=" * 60)
        print("HANDOFF TO PCB ENGINE")
        print("=" * 60)

        return self._run_engine(handoff, ai_callback)

    def _run_engine(self, handoff: EngineHandoff, ai_callback=None):
        """
        Run the PCB Engine with the handoff package.
        """
        try:
            from .pcb_engine import PCBEngine, EngineConfig, AIRequest, AIResponse, AIDecision
        except ImportError:
            print("ERROR: PCB Engine not available")
            return None

        # Create engine config from handoff
        config = EngineConfig(
            board_name=handoff.project_name or 'circuit_ai_design',
            board_width=handoff.board_constraints['width'],
            board_height=handoff.board_constraints['height'],
            layer_count=handoff.board_constraints['layers'],
            trace_width=handoff.design_rules['trace_width'],
            clearance=handoff.design_rules['clearance'],
            via_diameter=handoff.design_rules['via_diameter'],
            via_drill=handoff.design_rules['via_drill'],
            ai_agent_callback=ai_callback or self._default_ai_callback,
            verbose=True
        )

        # Create and run engine
        engine = PCBEngine(config)
        self.engine_result = engine.run_orchestrated(
            handoff.parts_db,
            circuit_ai_result=handoff
        )

        return self.engine_result

    def _default_ai_callback(self, request) -> 'AIResponse':
        """
        Default AI Agent callback for decision making.

        HIERARCHY:
        ===========
        USER      = Boss (gives requirements, final approval)
        Circuit AI = Engineer (intelligent decisions, algorithm selection)
        PCB Engine = Foreman (coordinates pistons, monitors DRC)
        Pistons   = Workers (execute specific tasks)

        This is the ENGINEER's brain - making intelligent decisions about:
        - Algorithm selection based on design context
        - Challenge resolution
        - When to escalate to USER (Boss)
        - Output approval
        """
        try:
            from .pcb_engine import AIResponse, AIDecision
        except ImportError:
            try:
                from pcb_engine import AIResponse, AIDecision
            except ImportError:
                return None

        print(f"\n[AI AGENT] Request: {request.question}")
        print(f"           Stage: {request.stage.value if hasattr(request.stage, 'value') else request.stage}")

        # ===================================================================
        # ALGORITHM SELECTION (Brain Logic)
        # ===================================================================
        if hasattr(request, 'algorithm_choices') and request.algorithm_choices:
            return self._intelligent_algorithm_selection(request)

        # ===================================================================
        # CHALLENGE HANDLING
        # ===================================================================
        if request.challenge:
            return self._intelligent_challenge_handling(request)

        # ===================================================================
        # OUTPUT APPROVAL
        # ===================================================================
        if 'Approve file generation' in request.question:
            state = request.current_state
            if state.get('errors', 0) == 0:
                print("           Decision: DELIVER")
                return AIResponse(decision=AIDecision.DELIVER)
            else:
                print(f"           Decision: ASK_USER ({state.get('errors')} errors)")
                return AIResponse(
                    decision=AIDecision.ASK_USER,
                    message_to_user=f"Design has {state.get('errors')} errors. Continue anyway?"
                )

        # Default: approve
        print("           Decision: APPROVE (default)")
        return AIResponse(decision=AIDecision.APPROVE)

    def _intelligent_algorithm_selection(self, request) -> 'AIResponse':
        """
        Intelligent algorithm selection based on design context.

        This is where the ENGINEER (Circuit AI) "thinks" about which
        algorithm is best for the current design situation.

        HIERARCHY:
        - USER = Boss (final approval)
        - Circuit AI = Engineer (intelligent decisions)
        - PCB Engine = Foreman (coordinates work)
        - Pistons = Workers (execute tasks)
        """
        try:
            from .pcb_engine import AIResponse, AIDecision
        except ImportError:
            try:
                from pcb_engine import AIResponse, AIDecision
            except ImportError:
                return None

        piston = request.piston_name
        choices = request.algorithm_choices
        context = request.context_hints
        previous = request.previous_attempts

        print(f"           Piston: {piston}")
        print(f"           Context: {context}")
        print(f"           Choices: {[c.name for c in choices]}")

        # Extract context
        net_count = context.get('net_count', 0)
        comp_count = context.get('component_count', 0)
        board_w = float(context.get('board_size', '50x50').split('x')[0])
        board_h = float(context.get('board_size', '50x50').split('x')[1].replace('mm', ''))
        board_area = board_w * board_h
        has_bga = context.get('has_bga', False)
        failures = context.get('previous_failures', 0)
        density = net_count / board_area if board_area > 0 else 0

        # Failed algorithms (don't repeat)
        failed_algos = [p['algorithm'] for p in previous if not p.get('success', True)]

        selected = None
        reasoning = ""

        # ===================================================================
        # ROUTING ALGORITHM SELECTION
        # ===================================================================
        if piston == 'routing':
            if failures > 2:
                # Multiple failures - use the big gun
                selected = next((c for c in choices if c.name == 'pathfinder' and c.name not in failed_algos), None)
                reasoning = f"Multiple failures ({failures}) - escalating to PathFinder congestion negotiation"

            elif density > 0.1:  # High density
                selected = next((c for c in choices if c.name == 'pathfinder' and c.name not in failed_algos), None)
                if not selected:
                    selected = next((c for c in choices if c.name == 'hadlock' and c.name not in failed_algos), None)
                reasoning = f"High density ({density:.3f} nets/mm²) - using congestion-aware algorithm"

            elif net_count < 15:
                # Small design - quality matters
                selected = next((c for c in choices if c.name == 'lee' and c.name not in failed_algos), None)
                reasoning = f"Small design ({net_count} nets) - using Lee for optimal paths"

            elif net_count < 50:
                # Medium design - balance speed and quality
                selected = next((c for c in choices if c.name == 'astar' and c.name not in failed_algos), None)
                reasoning = f"Medium design ({net_count} nets) - A* for speed/quality balance"

            else:
                # Large design - speed matters
                selected = next((c for c in choices if c.name == 'hadlock' and c.name not in failed_algos), None)
                reasoning = f"Large design ({net_count} nets) - Hadlock for speed"

        # ===================================================================
        # PLACEMENT ALGORITHM SELECTION
        # ===================================================================
        elif piston == 'placement':
            if failures > 1:
                # Try SA after failures
                selected = next((c for c in choices if c.name == 'simulated_annealing' and c.name not in failed_algos), None)
                reasoning = f"Placement failed {failures}x - trying SA for global optimization"

            elif comp_count < 10:
                selected = next((c for c in choices if c.name == 'force_directed' and c.name not in failed_algos), None)
                reasoning = f"Small component count ({comp_count}) - force-directed for quick result"

            elif comp_count > 50:
                selected = next((c for c in choices if c.name == 'analytical' and c.name not in failed_algos), None)
                reasoning = f"Large design ({comp_count} parts) - analytical for scalability"

            else:
                # Check if hierarchical design
                if self.current_handoff and len(self.current_handoff.topology.get('blocks', [])) > 3:
                    selected = next((c for c in choices if c.name == 'partition' and c.name not in failed_algos), None)
                    reasoning = "Hierarchical design detected - using partition-based placement"
                else:
                    selected = next((c for c in choices if c.name == 'simulated_annealing' and c.name not in failed_algos), None)
                    reasoning = f"Medium design ({comp_count} parts) - SA for quality"

        # ===================================================================
        # ESCAPE ALGORITHM SELECTION
        # ===================================================================
        elif piston == 'escape':
            if has_bga:
                # BGA needs careful escape
                selected = next((c for c in choices if c.name == 'mmcf' and c.name not in failed_algos), None)
                reasoning = "BGA detected - using MMCF for optimal escape"
            else:
                selected = next((c for c in choices if c.name == 'ring' and c.name not in failed_algos), None)
                reasoning = "Standard package - using ring escape"

        # ===================================================================
        # TOPOLOGICAL ROUTING SELECTION
        # ===================================================================
        elif piston == 'topological_routing':
            # Check if design has differential pairs or RF traces
            has_differential = context.get('has_differential', False)
            has_rf = context.get('has_rf_traces', False)

            if has_differential or has_rf:
                selected = next((c for c in choices if c.name == 'force_directed' and c.name not in failed_algos), None)
                reasoning = "Differential/RF traces detected - using force-directed for even spacing"
            else:
                selected = next((c for c in choices if c.name == 'delaunay_rubberband' and c.name not in failed_algos), None)
                reasoning = "Standard design - using Delaunay rubber-band for smooth routing"

        # ===================================================================
        # 3D VISUALIZATION SELECTION
        # ===================================================================
        elif piston == 'visualization_3d':
            output_purpose = context.get('output_purpose', 'preview')

            if output_purpose == 'mechanical_cad':
                selected = next((c for c in choices if c.name == 'step_cad' and c.name not in failed_algos), None)
                reasoning = "Mechanical CAD integration - using STEP format"
            elif output_purpose == '3d_printing':
                selected = next((c for c in choices if c.name == 'stl_mesh' and c.name not in failed_algos), None)
                reasoning = "3D printing output - using STL mesh format"
            elif output_purpose == 'web_preview':
                selected = next((c for c in choices if c.name == 'gltf_web' and c.name not in failed_algos), None)
                reasoning = "Web visualization - using glTF format"
            else:
                selected = next((c for c in choices if c.name == 'stl_mesh' and c.name not in failed_algos), None)
                reasoning = "Default visualization - using STL format"

        # ===================================================================
        # BOM OPTIMIZATION SELECTION
        # ===================================================================
        elif piston == 'bom_optimization':
            production_qty = context.get('production_quantity', 1)
            urgency = context.get('urgency', 'normal')
            budget = context.get('budget_priority', 'balanced')

            if urgency == 'urgent':
                selected = next((c for c in choices if c.name == 'fastest_delivery' and c.name not in failed_algos), None)
                reasoning = "Urgent build - prioritizing fastest delivery"
            elif budget == 'minimum':
                selected = next((c for c in choices if c.name == 'lowest_cost' and c.name not in failed_algos), None)
                reasoning = "Budget priority - optimizing for lowest cost"
            elif production_qty < 5:
                selected = next((c for c in choices if c.name == 'single_supplier' and c.name not in failed_algos), None)
                reasoning = f"Small quantity ({production_qty}) - minimizing supplier count"
            else:
                selected = next((c for c in choices if c.name == 'balanced' and c.name not in failed_algos), None)
                reasoning = "Production build - balanced optimization"

        # ===================================================================
        # LEARNING ALGORITHM SELECTION
        # ===================================================================
        elif piston == 'learning':
            training_samples = context.get('training_samples', 0)
            has_labels = context.get('has_labels', False)
            learning_goal = context.get('learning_goal', 'patterns')

            if has_labels and training_samples >= 10:
                selected = next((c for c in choices if c.name == 'supervised' and c.name not in failed_algos), None)
                reasoning = f"Labeled data available ({training_samples} samples) - using supervised learning"
            elif learning_goal == 'routing_improvement':
                selected = next((c for c in choices if c.name == 'reinforcement' and c.name not in failed_algos), None)
                reasoning = "Routing improvement goal - using reinforcement learning"
            elif training_samples < 5 and context.get('has_similar_designs', False):
                selected = next((c for c in choices if c.name == 'transfer' and c.name not in failed_algos), None)
                reasoning = "Limited samples but similar designs exist - using transfer learning"
            else:
                selected = next((c for c in choices if c.name == 'unsupervised' and c.name not in failed_algos), None)
                reasoning = "No labels available - using unsupervised pattern discovery"

        # Fallback
        if not selected:
            available = [c for c in choices if c.name not in failed_algos]
            if available:
                selected = available[0]
                reasoning = f"Fallback to {selected.name} (previous attempts exhausted)"
            else:
                # All failed - try hybrid or first choice
                selected = next((c for c in choices if c.name == 'hybrid'), choices[0])
                reasoning = "All algorithms failed - trying hybrid approach"

        print(f"           Selected: {selected.name}")
        print(f"           Reasoning: {reasoning}")

        return AIResponse(
            decision=AIDecision.APPROVE,
            selected_algorithm=selected.name,
            reasoning=reasoning,
            algorithm_parameters={}
        )

    def _intelligent_challenge_handling(self, request) -> 'AIResponse':
        """
        Intelligent challenge handling with reasoning.

        The ENGINEER (Circuit AI) evaluates challenges and decides
        the best course of action, or escalates to USER (Boss).
        """
        try:
            from .pcb_engine import AIResponse, AIDecision
        except ImportError:
            try:
                from pcb_engine import AIResponse, AIDecision
            except ImportError:
                return None

        challenge = request.challenge
        print(f"           Challenge: {challenge.type.value}")
        print(f"           Severity: {challenge.severity}")

        # Analyze the challenge
        ch_type = challenge.type.value
        severity = challenge.severity
        context = challenge.context

        # ===================================================================
        # MINOR ISSUES - Auto-approve
        # ===================================================================
        if severity == 'minor':
            print("           Decision: APPROVE (minor issue - auto-handled)")
            return AIResponse(decision=AIDecision.APPROVE)

        # ===================================================================
        # ROUTING IMPOSSIBLE
        # ===================================================================
        if ch_type == 'routing_impossible':
            # Check how many nets failed
            unrouted = sum(1 for e in context.get('errors', []) if 'unrouted' in str(e))
            total_nets = request.current_state.get('nets', 1)
            failure_rate = unrouted / total_nets if total_nets > 0 else 1

            if failure_rate < 0.1:  # Less than 10% failed
                print(f"           Decision: APPROVE ({unrouted} nets failed, {failure_rate:.0%} - acceptable)")
                return AIResponse(
                    decision=AIDecision.APPROVE,
                    message_to_user=f"{unrouted} nets could not be routed ({failure_rate:.0%})"
                )
            elif failure_rate < 0.3:  # Try adding layers
                print("           Decision: ADD_LAYERS (moderate routing failure)")
                return AIResponse(
                    decision=AIDecision.ADD_LAYERS,
                    reasoning="Adding 2 layers to improve routability"
                )
            else:  # Major routing problem
                print("           Decision: ASK_USER (major routing failure)")
                return AIResponse(
                    decision=AIDecision.ASK_USER,
                    message_to_user=f"{failure_rate:.0%} of nets failed to route. Consider redesign."
                )

        # ===================================================================
        # BOARD TOO SMALL
        # ===================================================================
        if ch_type == 'board_too_small':
            if self.current_handoff:
                current_w = self.current_handoff.board_constraints['width']
                current_h = self.current_handoff.board_constraints['height']
                # Increase by 25%
                new_w = current_w * 1.25
                new_h = current_h * 1.25
                print(f"           Decision: MODIFY_BOARD ({current_w}x{current_h} -> {new_w}x{new_h}mm)")
                return AIResponse(
                    decision=AIDecision.MODIFY_BOARD,
                    parameters={'width': new_w, 'height': new_h},
                    reasoning=f"Board expanded from {current_w}x{current_h} to {new_w}x{new_h}mm"
                )
            else:
                print("           Decision: RETRY_HARDER")
                return AIResponse(decision=AIDecision.RETRY_HARDER)

        # ===================================================================
        # PLACEMENT CONFLICT
        # ===================================================================
        if ch_type == 'placement_conflict':
            retries = context.get('retries', 0)
            if retries < 2:
                print("           Decision: RETRY_HARDER (placement)")
                return AIResponse(decision=AIDecision.RETRY_HARDER)
            else:
                print("           Decision: ASK_USER (placement failed repeatedly)")
                return AIResponse(
                    decision=AIDecision.ASK_USER,
                    message_to_user="Components overlap after multiple attempts. Manual adjustment may be needed."
                )

        # ===================================================================
        # DRC PERSISTENT
        # ===================================================================
        if ch_type == 'drc_persistent':
            error_count = context.get('violations', {}).get('error_count', 0)
            if error_count < 3:
                print(f"           Decision: APPROVE ({error_count} DRC errors - acceptable)")
                return AIResponse(
                    decision=AIDecision.APPROVE,
                    message_to_user=f"Design has {error_count} minor DRC violations"
                )
            else:
                print("           Decision: ASK_USER (multiple DRC errors)")
                return AIResponse(
                    decision=AIDecision.ASK_USER,
                    message_to_user=f"Design has {error_count} DRC errors requiring review"
                )

        # ===================================================================
        # CRITICAL - Always ask user
        # ===================================================================
        if severity == 'critical':
            print("           Decision: ASK_USER (critical issue)")
            return AIResponse(
                decision=AIDecision.ASK_USER,
                message_to_user=f"Critical issue: {challenge.description}"
            )

        # Default for major issues
        print("           Decision: RETRY_HARDER (default for major)")
        return AIResponse(decision=AIDecision.RETRY_HARDER)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def quick_design(description: str) -> CircuitAIResult:
    """
    Quick design from natural language description.

    Example:
        result = quick_design("ESP32 temperature logger with WiFi")
        parts_piston.build_from_dict(result.parts_db)
    """
    ai = CircuitAI()
    requirements = ai.parse_natural_language(description)
    return ai.generate_parts_db(requirements)


def full_design(description: str, ai_callback=None):
    """
    Full design flow from description to output files.

    This is the main entry point for the complete AI-driven
    PCB design pipeline.

    Example:
        result = full_design("ESP32 temperature logger with WiFi and OLED")
        print(f"Output files: {result.output_files}")
    """
    interface = CircuitAIEngineInterface()
    return interface.design_from_description(description, ai_callback)


# =============================================================================
# SCHEMATIC GENERATION
# =============================================================================

def generate_kicad_schematic(schematic: Schematic) -> str:
    """
    Generate KiCad 7+ schematic file content (.kicad_sch).

    This creates a complete schematic file that can be opened
    directly in KiCad.
    """
    import time

    lines = [
        '(kicad_sch (version 20230121) (generator "circuit_ai")',
        '',
        '  (uuid "' + generate_uuid() + '")',
        '',
        '  (paper "A4")',
        '',
    ]

    for sheet in schematic.sheets:
        # Title block
        lines.append('  (title_block')
        lines.append(f'    (title "{sheet.title_block.get("title", "Circuit AI Design")}")')
        lines.append(f'    (date "{time.strftime("%Y-%m-%d")}")')
        lines.append(f'    (rev "{sheet.title_block.get("rev", "1.0")}")')
        lines.append(f'    (company "{sheet.title_block.get("company", "")}")')
        lines.append('  )')
        lines.append('')

        # Net labels
        for label in sheet.labels:
            lines.append(f'  (label "{label.name}" (at {label.x} {label.y} {label.rotation})')
            lines.append('    (effects (font (size 1.27 1.27)))')
            lines.append('    (uuid "' + generate_uuid() + '")')
            lines.append('  )')

        # Global labels for power
        power_nets = ['GND', 'VCC', 'V3V3', 'V5V', 'VIN']
        for label in sheet.labels:
            if label.name.upper() in power_nets or label.style == 'global':
                lines.append(f'  (global_label "{label.name}" (shape input) (at {label.x} {label.y} {label.rotation})')
                lines.append('    (effects (font (size 1.27 1.27)))')
                lines.append('    (uuid "' + generate_uuid() + '")')
                lines.append('  )')

        # Symbols
        for symbol in sheet.symbols:
            lines.append(f'  (symbol (lib_id "{symbol.library}:{symbol.symbol}")')
            lines.append(f'    (at {symbol.x} {symbol.y} {symbol.rotation})')
            if symbol.mirror:
                lines.append('    (mirror y)')
            lines.append('    (unit 1)')
            lines.append('    (in_bom yes) (on_board yes) (dnp no)')
            lines.append('    (uuid "' + generate_uuid() + '")')

            # Reference property
            lines.append(f'    (property "Reference" "{symbol.reference}" (at {symbol.x} {symbol.y - 5} 0)')
            lines.append('      (effects (font (size 1.27 1.27)))')
            lines.append('    )')

            # Value property
            lines.append(f'    (property "Value" "{symbol.value}" (at {symbol.x} {symbol.y + 5} 0)')
            lines.append('      (effects (font (size 1.27 1.27)))')
            lines.append('    )')

            # Footprint property
            fp = symbol.properties.get('footprint', '')
            lines.append(f'    (property "Footprint" "{fp}" (at {symbol.x} {symbol.y + 7.5} 0)')
            lines.append('      (effects (font (size 1.27 1.27)) hide)')
            lines.append('    )')

            # Pin connections
            for pin_name, net_name in symbol.properties.get('pin_nets', {}).items():
                lines.append(f'    (pin "{pin_name}" (uuid "' + generate_uuid() + '"))')

            lines.append('  )')
            lines.append('')

        # Wires
        for wire in sheet.wires:
            lines.append(f'  (wire (pts (xy {wire.start[0]} {wire.start[1]}) (xy {wire.end[0]} {wire.end[1]}))')
            lines.append('    (stroke (width 0) (type default))')
            lines.append('    (uuid "' + generate_uuid() + '")')
            lines.append('  )')

    # Power symbols
    lines.append('')
    lines.append('  (symbol (lib_id "power:GND") (at 50 150 0)')
    lines.append('    (in_bom no) (on_board no)')
    lines.append('    (uuid "' + generate_uuid() + '")')
    lines.append('    (property "Reference" "#PWR01" (at 50 156.21 0) (effects (font (size 1.27 1.27)) hide))')
    lines.append('    (property "Value" "GND" (at 50 153.67 0) (effects (font (size 1.27 1.27))))')
    lines.append('  )')

    lines.append('')
    lines.append(')')

    return '\n'.join(lines)


def generate_uuid() -> str:
    """Generate a random UUID for KiCad objects"""
    import random
    hex_chars = '0123456789abcdef'
    uuid = ''.join(random.choice(hex_chars) for _ in range(8))
    uuid += '-'
    uuid += ''.join(random.choice(hex_chars) for _ in range(4))
    uuid += '-'
    uuid += ''.join(random.choice(hex_chars) for _ in range(4))
    uuid += '-'
    uuid += ''.join(random.choice(hex_chars) for _ in range(4))
    uuid += '-'
    uuid += ''.join(random.choice(hex_chars) for _ in range(12))
    return uuid


def generate_netlist_kicad(parts_db: Dict) -> str:
    """
    Generate KiCad netlist from parts database.

    This creates a .net file that can be imported into KiCad.
    """
    import time

    lines = [
        '(export (version D)',
        f'  (design',
        f'    (source "circuit_ai_generated")',
        f'    (date "{time.strftime("%Y-%m-%d %H:%M:%S")}")',
        f'    (tool "CircuitAI 1.0")',
        '  )',
        '',
        '  (components',
    ]

    # Add components
    for ref, part in parts_db.get('parts', {}).items():
        lines.append(f'    (comp (ref "{ref}")')
        lines.append(f'      (value "{part.get("value", "")}")')
        lines.append(f'      (footprint "{part.get("footprint", "")}")')
        lines.append(f'      (description "{part.get("description", "")}")')

        # Add fields
        fields = part.get('fields', {})
        for fname, fval in fields.items():
            lines.append(f'      (field (name "{fname}") "{fval}")')

        lines.append('    )')

    lines.append('  )')
    lines.append('')

    # Add net classes
    lines.append('  (net_class "Default" ""')
    lines.append('    (clearance 0.15)')
    lines.append('    (trace_width 0.25)')
    lines.append('    (via_dia 0.8)')
    lines.append('    (via_drill 0.4)')
    lines.append('  )')
    lines.append('')
    lines.append('  (net_class "Power" "Power nets"')
    lines.append('    (clearance 0.2)')
    lines.append('    (trace_width 0.5)')
    lines.append('    (via_dia 1.0)')
    lines.append('    (via_drill 0.6)')
    lines.append('  )')
    lines.append('')

    # Add nets
    lines.append('  (nets')
    net_id = 0
    for net_name, net_data in parts_db.get('nets', {}).items():
        net_id += 1
        lines.append(f'    (net (code "{net_id}") (name "{net_name}")')

        for ref, pin in net_data.get('pins', []):
            lines.append(f'      (node (ref "{ref}") (pin "{pin}"))')

        lines.append('    )')

    lines.append('  )')
    lines.append(')')

    return '\n'.join(lines)


def generate_spice_netlist(parts_db: Dict, project_name: str = 'circuit') -> str:
    """
    Generate SPICE netlist for simulation.

    This creates a .cir file for LTSpice, ngspice, etc.
    """
    lines = [
        f'* {project_name}',
        f'* Generated by Circuit AI',
        f'* SPICE Netlist',
        '',
    ]

    # Build node map
    node_map = {'GND': '0', 'V3V3': '3V3', 'V5V': '5V', 'VIN': 'VIN'}
    node_counter = 1

    nets = parts_db.get('nets', {})
    for net_name in nets:
        if net_name not in node_map:
            node_map[net_name] = f'N{node_counter:03d}'
            node_counter += 1

    # Generate component lines
    for ref, part in parts_db.get('parts', {}).items():
        value = part.get('value', '')
        pins = part.get('pins', [])

        if ref.startswith('R'):
            # Resistor: R1 node1 node2 value
            if len(pins) >= 2:
                n1 = node_map.get(pins[0].get('net', ''), '0')
                n2 = node_map.get(pins[1].get('net', ''), '0')
                lines.append(f'{ref} {n1} {n2} {value}')

        elif ref.startswith('C'):
            # Capacitor: C1 node1 node2 value
            if len(pins) >= 2:
                n1 = node_map.get(pins[0].get('net', ''), '0')
                n2 = node_map.get(pins[1].get('net', ''), '0')
                lines.append(f'{ref} {n1} {n2} {value}')

        elif ref.startswith('L'):
            # Inductor: L1 node1 node2 value
            if len(pins) >= 2:
                n1 = node_map.get(pins[0].get('net', ''), '0')
                n2 = node_map.get(pins[1].get('net', ''), '0')
                lines.append(f'{ref} {n1} {n2} {value}')

        elif ref.startswith('D'):
            # Diode: D1 anode cathode model
            if len(pins) >= 2:
                n1 = node_map.get(pins[0].get('net', ''), '0')
                n2 = node_map.get(pins[1].get('net', ''), '0')
                lines.append(f'{ref} {n1} {n2} D_GENERIC')

        elif ref.startswith('Q'):
            # BJT: Q1 collector base emitter model
            if len(pins) >= 3:
                nc = node_map.get(pins[0].get('net', ''), '0')
                nb = node_map.get(pins[1].get('net', ''), '0')
                ne = node_map.get(pins[2].get('net', ''), '0')
                lines.append(f'{ref} {nc} {nb} {ne} NPN_GENERIC')

        elif ref.startswith('U') or ref.startswith('IC'):
            # Subcircuit: X1 pins... model
            pin_nodes = ' '.join(node_map.get(p.get('net', ''), '0') for p in pins)
            model = value.replace('-', '_').replace('.', '_')
            lines.append(f'X{ref[1:]} {pin_nodes} {model}')

    # Add power sources
    lines.append('')
    lines.append('* Power Sources')
    lines.append('V1 VIN 0 DC 5V')
    lines.append('V2 3V3 0 DC 3.3V')
    lines.append('')

    # Add simulation commands
    lines.append('* Simulation Commands')
    lines.append('.tran 0.1m 10m')
    lines.append('.end')

    return '\n'.join(lines)


class SchematicGenerator:
    """
    Generates schematic diagrams from circuit data.

    Creates properly laid out schematics with:
    - Power section at top
    - MCU in center
    - Peripherals grouped by function
    - Proper wire routing
    """

    def __init__(self):
        self.grid_size = 2.54  # KiCad standard grid
        self.symbol_spacing_x = 30
        self.symbol_spacing_y = 25
        self.current_x = 50
        self.current_y = 50

    def generate(self, parts_db: Dict, requirements: CircuitRequirements = None) -> Schematic:
        """Generate complete schematic from parts database"""
        schematic = Schematic()
        sheet = SchematicSheet(
            name='Main',
            title_block={
                'title': requirements.project_name if requirements else 'Circuit Design',
                'rev': '1.0',
            }
        )

        symbols = []
        wires = []
        labels = []

        # Group components by type
        groups = self._group_components(parts_db)

        # Place power components at top
        y_offset = 30
        x_offset = 30

        if 'power' in groups:
            for ref in groups['power']:
                part = parts_db['parts'][ref]
                symbol = SchematicSymbol(
                    reference=ref,
                    value=part['value'],
                    library='Device',
                    symbol=self._get_symbol_name(ref, part),
                    x=x_offset,
                    y=y_offset,
                    properties={'footprint': part.get('footprint', '')}
                )
                symbols.append(symbol)
                x_offset += self.symbol_spacing_x

        # Place MCU in center
        y_offset += self.symbol_spacing_y * 2
        x_offset = 100

        if 'mcu' in groups:
            for ref in groups['mcu']:
                part = parts_db['parts'][ref]
                symbol = SchematicSymbol(
                    reference=ref,
                    value=part['value'],
                    library='MCU_Module',
                    symbol=self._get_symbol_name(ref, part),
                    x=x_offset,
                    y=y_offset,
                    properties={'footprint': part.get('footprint', '')}
                )
                symbols.append(symbol)

        # Place sensors on left
        y_offset += self.symbol_spacing_y
        x_offset = 30

        if 'sensor' in groups:
            for ref in groups['sensor']:
                part = parts_db['parts'][ref]
                symbol = SchematicSymbol(
                    reference=ref,
                    value=part['value'],
                    library='Sensor',
                    symbol=self._get_symbol_name(ref, part),
                    x=x_offset,
                    y=y_offset,
                    properties={'footprint': part.get('footprint', '')}
                )
                symbols.append(symbol)
                y_offset += self.symbol_spacing_y

        # Place connectors on right
        y_offset = 80
        x_offset = 200

        for group_name in ['connector', 'interface']:
            if group_name in groups:
                for ref in groups[group_name]:
                    part = parts_db['parts'][ref]
                    symbol = SchematicSymbol(
                        reference=ref,
                        value=part['value'],
                        library='Connector',
                        symbol=self._get_symbol_name(ref, part),
                        x=x_offset,
                        y=y_offset,
                        properties={'footprint': part.get('footprint', '')}
                    )
                    symbols.append(symbol)
                    y_offset += self.symbol_spacing_y

        # Add net labels
        for net_name, net_data in parts_db.get('nets', {}).items():
            if net_data.get('pins'):
                ref, pin = net_data['pins'][0]
                # Find symbol position
                for sym in symbols:
                    if sym.reference == ref:
                        labels.append(SchematicLabel(
                            name=net_name,
                            x=sym.x + 15,
                            y=sym.y,
                            style='global' if net_name.upper() in ['GND', 'VCC', 'V3V3', 'V5V', 'VIN'] else 'local'
                        ))
                        break

        sheet.symbols = symbols
        sheet.wires = wires
        sheet.labels = labels
        schematic.sheets = [sheet]

        return schematic

    def _group_components(self, parts_db: Dict) -> Dict[str, List[str]]:
        """Group components by their function"""
        groups = defaultdict(list)

        for ref, part in parts_db.get('parts', {}).items():
            group = part.get('group', 'other')
            groups[group].append(ref)

        return dict(groups)

    def _get_symbol_name(self, ref: str, part: Dict) -> str:
        """Get KiCad symbol name for a component"""
        value = part.get('value', '')

        # Passives
        if ref.startswith('R'):
            return 'R'
        elif ref.startswith('C'):
            return 'C'
        elif ref.startswith('L'):
            return 'L'
        elif ref.startswith('D'):
            return 'D'

        # Connectors
        elif ref.startswith('J'):
            if 'USB' in value.upper():
                return 'USB_C_Receptacle'
            return 'Conn_01x02'

        # MCUs/ICs
        elif ref.startswith('U'):
            if 'ESP32' in value.upper():
                return 'ESP32-WROOM-32'
            elif 'STM32' in value.upper():
                return 'STM32F103C8Tx'
            elif 'ATMEGA' in value.upper():
                return 'ATmega328P-PU'
            return 'IC_Generic'

        return 'Generic'


# =============================================================================
# ENHANCED NLP ENGINE
# =============================================================================

class EnhancedNLPParser:
    """
    Advanced natural language parser for circuit descriptions.

    Uses pattern matching and semantic analysis to extract:
    - MCU requirements
    - Power specifications
    - Sensor needs
    - Communication protocols
    - Display requirements
    - Motor control needs
    - Quantity specifications
    """

    def __init__(self):
        self.patterns = NLP_PATTERNS

    def parse(self, text: str) -> CircuitRequirements:
        """
        Parse natural language description into structured requirements.

        Example inputs:
        - "I need an ESP32-S3 with BME280 sensor and OLED display, USB powered"
        - "Battery-powered LoRa gateway with GPS and 4 channel relay"
        - "STM32 motor controller for 2 stepper motors with CAN bus"
        """
        text_lower = text.lower()
        requirements = CircuitRequirements()

        # Extract project context
        requirements.description = text
        requirements.project_name = self._extract_project_name(text)

        # Detect MCU
        mcu_result = self._detect_mcu(text_lower)
        requirements.mcu_family = mcu_result['family']
        requirements.mcu_features.extend(mcu_result['features'])

        # Detect power source
        requirements.input_power = self._detect_power(text_lower)

        # Detect sensors
        sensors = self._detect_sensors(text_lower)
        for sensor in sensors:
            block = self._create_sensor_block(sensor)
            requirements.blocks.append(block)

        # Detect displays
        displays = self._detect_displays(text_lower)
        for display in displays:
            block = self._create_display_block(display)
            requirements.blocks.append(block)

        # Detect communication modules
        comms = self._detect_communication(text_lower)
        for comm in comms:
            block = self._create_comm_block(comm)
            requirements.blocks.append(block)
            # Add to MCU features if applicable
            if comm in ['wifi', 'bluetooth', 'ble']:
                if comm not in requirements.mcu_features:
                    requirements.mcu_features.append(comm)

        # Detect motors
        motors = self._detect_motors(text_lower)
        for motor in motors:
            block = self._create_motor_block(motor)
            requirements.blocks.append(block)

        # Detect LEDs
        leds = self._detect_leds(text_lower)
        for led in leds:
            block = self._create_led_block(led)
            requirements.blocks.append(block)

        # Detect audio
        audio = self._detect_audio(text_lower)
        for audio_type in audio:
            block = self._create_audio_block(audio_type)
            requirements.blocks.append(block)

        # Detect quantities
        quantities = self._extract_quantities(text_lower)
        for block in requirements.blocks:
            if block.name in quantities:
                block.description += f" x{quantities[block.name]}"

        # Estimate board size
        requirements.board_size_mm = self._estimate_board_size(requirements)

        # Set layer count
        requirements.layers = self._estimate_layers(requirements)

        return requirements

    def _extract_project_name(self, text: str) -> str:
        """Extract or generate project name"""
        # Look for explicit naming
        patterns = [
            r'(?:project|device|system|module|board)\s*(?:named?|called?|:)\s*["\']?(\w+)',
            r'(?:build|make|create|design)\s+(?:a|an)\s+(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # Generate from content
        keywords = []
        if 'esp32' in text.lower():
            keywords.append('esp32')
        if any(x in text.lower() for x in ['sensor', 'temperature', 'humidity']):
            keywords.append('sensor')
        if 'motor' in text.lower():
            keywords.append('motor')
        if 'led' in text.lower():
            keywords.append('led')

        return '_'.join(keywords) if keywords else 'circuit_design'

    def _detect_mcu(self, text: str) -> Dict:
        """Detect MCU family and specific requirements"""
        result = {'family': MCUFamily.ESP32, 'features': []}

        for family, patterns in self.patterns['mcu'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    result['family'] = MCUFamily(family) if family in [e.value for e in MCUFamily] else MCUFamily.ESP32

                    # Detect specific variants
                    if family == 'esp32':
                        if re.search(r's3', text):
                            result['features'].append('usb_otg')
                        if re.search(r'c3', text):
                            result['features'].append('risc_v')
                    break

        return result

    def _detect_power(self, text: str) -> PowerType:
        """Detect power source from text"""
        for power_type, patterns in self.patterns['power'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return PowerType(power_type) if power_type in [e.value for e in PowerType] else PowerType.USB_5V

        return PowerType.USB_5V

    def _detect_sensors(self, text: str) -> List[str]:
        """Detect sensor requirements"""
        found = []
        for sensor_type, patterns in self.patterns['sensors'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found.append(sensor_type)
                    break
        return list(set(found))

    def _detect_displays(self, text: str) -> List[str]:
        """Detect display requirements"""
        found = []
        for display_type, patterns in self.patterns['display'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found.append(display_type)
                    break
        return list(set(found))

    def _detect_communication(self, text: str) -> List[str]:
        """Detect communication protocols"""
        found = []
        for comm_type, patterns in self.patterns['communication'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found.append(comm_type)
                    break
        return list(set(found))

    def _detect_motors(self, text: str) -> List[str]:
        """Detect motor types"""
        found = []
        for motor_type, patterns in self.patterns['motors'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found.append(motor_type)
                    break
        return list(set(found))

    def _detect_leds(self, text: str) -> List[str]:
        """Detect LED types"""
        found = []
        for led_type, patterns in self.patterns['leds'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found.append(led_type)
                    break
        return list(set(found))

    def _detect_audio(self, text: str) -> List[str]:
        """Detect audio requirements"""
        found = []
        for audio_type, patterns in self.patterns['audio'].items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found.append(audio_type)
                    break
        return list(set(found))

    def _extract_quantities(self, text: str) -> Dict[str, int]:
        """Extract quantities from text"""
        quantities = {}

        # Channel counts
        match = re.search(r'(\d+)\s*(?:ch|channel)', text)
        if match:
            quantities['channel'] = int(match.group(1))

        # Motor counts
        match = re.search(r'(\d+)\s*(?:motor|stepper|servo)', text)
        if match:
            quantities['motor'] = int(match.group(1))

        # LED counts
        match = re.search(r'(\d+)\s*(?:led|rgb)', text)
        if match:
            quantities['led'] = int(match.group(1))

        # Relay counts
        match = re.search(r'(\d+)\s*(?:relay|channel)', text)
        if match:
            quantities['relay'] = int(match.group(1))

        return quantities

    def _create_sensor_block(self, sensor_type: str) -> CircuitBlock:
        """Create a sensor circuit block"""
        sensor_mapping = {
            'temperature': ('BME280', SENSOR_DATABASE.get('BME280', {})),
            'humidity': ('BME280', SENSOR_DATABASE.get('BME280', {})),
            'pressure': ('BMP280', SENSOR_DATABASE.get('BMP280', {})),
            'imu': ('MPU6050', SENSOR_DATABASE.get('MPU6050', {})),
            'magnetometer': ('ICM-20948', SENSOR_DATABASE.get('ICM-20948', {})),
            'distance': ('VL53L0X', SENSOR_DATABASE.get('VL53L0X', {})),
            'light': ('VEML7700', SENSOR_DATABASE.get('VEML7700', {})),
            'gps': ('NEO-6M', SENSOR_DATABASE.get('NEO-6M', {})),
            'heart_rate': ('MAX30102', SENSOR_DATABASE.get('MAX30102', {})),
            'current': ('INA219', SENSOR_DATABASE.get('INA219', {})),
            'weight': ('HX711', SENSOR_DATABASE.get('HX711', {})),
            'sound': ('INMP441', SENSOR_DATABASE.get('INMP441', {})),
        }

        name, data = sensor_mapping.get(sensor_type, ('Generic', {}))

        return CircuitBlock(
            name=f'sensor_{sensor_type}',
            block_type=CircuitBlockType.SENSOR,
            description=f'{name} {sensor_type} sensor',
            power=PowerRequirement(
                voltage=data.get('voltage', 3.3),
                current_max=data.get('current_ma', 10) / 1000
            )
        )

    def _create_display_block(self, display_type: str) -> CircuitBlock:
        """Create a display circuit block"""
        display_mapping = {
            'oled': ('SSD1306_128x64', DISPLAY_DATABASE.get('SSD1306_128x64', {})),
            'tft': ('ST7789', DISPLAY_DATABASE.get('ST7789', {})),
            'lcd': ('HD44780_16x2', DISPLAY_DATABASE.get('HD44780_16x2', {})),
            'led_matrix': ('MAX7219', DISPLAY_DATABASE.get('MAX7219', {})),
        }

        name, data = display_mapping.get(display_type, ('Generic_Display', {}))

        return CircuitBlock(
            name=f'display_{display_type}',
            block_type=CircuitBlockType.DISPLAY,
            description=f'{name} {display_type} display',
            power=PowerRequirement(
                voltage=data.get('voltage', 3.3),
                current_max=data.get('current_ma', 50) / 1000
            )
        )

    def _create_comm_block(self, comm_type: str) -> CircuitBlock:
        """Create a communication circuit block"""
        comm_mapping = {
            'lora': ('SX1278', COMMUNICATION_DATABASE.get('SX1278', {})),
            'rf': ('NRF24L01', COMMUNICATION_DATABASE.get('NRF24L01', {})),
            'ethernet': ('W5500', COMMUNICATION_DATABASE.get('W5500', {})),
            'can': ('MCP2515', COMMUNICATION_DATABASE.get('MCP2515', {})),
            'rs485': ('MAX485', COMMUNICATION_DATABASE.get('MAX485', {})),
        }

        name, data = comm_mapping.get(comm_type, ('Generic_Comm', {}))

        return CircuitBlock(
            name=f'comm_{comm_type}',
            block_type=CircuitBlockType.COMMUNICATION,
            description=f'{name} {comm_type} module',
            power=PowerRequirement(
                voltage=data.get('voltage', 3.3),
                current_max=data.get('current_ma', 100) / 1000
            )
        )

    def _create_motor_block(self, motor_type: str) -> CircuitBlock:
        """Create a motor driver circuit block"""
        motor_mapping = {
            'dc': ('DRV8833', MOTOR_DRIVER_DATABASE.get('DRV8833', {})),
            'stepper': ('TMC2209', MOTOR_DRIVER_DATABASE.get('TMC2209', {})),
            'servo': ('PCA9685', LED_DRIVER_DATABASE.get('PCA9685', {})),
            'brushless': ('BTS7960', MOTOR_DRIVER_DATABASE.get('BTS7960', {})),
        }

        name, data = motor_mapping.get(motor_type, ('L298N', MOTOR_DRIVER_DATABASE.get('L298N', {})))

        return CircuitBlock(
            name=f'motor_{motor_type}',
            block_type=CircuitBlockType.MOTOR_DRIVER,
            description=f'{name} {motor_type} motor driver',
            power=PowerRequirement(
                voltage=data.get('voltage_max', 12),
                current_max=data.get('current_max', 2)
            )
        )

    def _create_led_block(self, led_type: str) -> CircuitBlock:
        """Create an LED driver circuit block"""
        led_mapping = {
            'addressable': ('WS2812B', LED_DRIVER_DATABASE.get('WS2812B', {})),
            'pwm': ('PCA9685', LED_DRIVER_DATABASE.get('PCA9685', {})),
        }

        name, data = led_mapping.get(led_type, ('WS2812B', {}))

        return CircuitBlock(
            name=f'led_{led_type}',
            block_type=CircuitBlockType.INTERFACE,
            description=f'{name} {led_type} LED driver',
            power=PowerRequirement(
                voltage=data.get('voltage', 5.0),
                current_max=data.get('current_ma', 60) / 1000
            )
        )

    def _create_audio_block(self, audio_type: str) -> CircuitBlock:
        """Create an audio circuit block"""
        audio_mapping = {
            'speaker': ('MAX98357A', AUDIO_DATABASE.get('MAX98357A', {})),
            'microphone': ('INMP441', AUDIO_DATABASE.get('INMP441', {})),
            'buzzer': ('Buzzer_Piezo', {}),
            'dac': ('PCM5102A', AUDIO_DATABASE.get('PCM5102A', {})),
        }

        name, data = audio_mapping.get(audio_type, ('Generic_Audio', {}))

        return CircuitBlock(
            name=f'audio_{audio_type}',
            block_type=CircuitBlockType.AUDIO,
            description=f'{name} {audio_type}',
            power=PowerRequirement(
                voltage=data.get('voltage', 5.0),
                current_max=data.get('current_ma', 50) / 1000
            )
        )

    def _estimate_board_size(self, requirements: CircuitRequirements) -> Tuple[float, float]:
        """Estimate appropriate board size based on components"""
        # Base size
        base = 30.0

        # MCU contribution
        if requirements.mcu_family == MCUFamily.ESP32:
            base += 15
        elif requirements.mcu_family == MCUFamily.STM32:
            base += 10
        elif requirements.mcu_family == MCUFamily.ATMEGA:
            base += 12

        # Block contribution
        block_sizes = {
            CircuitBlockType.DISPLAY: 15,
            CircuitBlockType.MOTOR_DRIVER: 10,
            CircuitBlockType.COMMUNICATION: 8,
            CircuitBlockType.SENSOR: 5,
            CircuitBlockType.AUDIO: 8,
        }

        for block in requirements.blocks:
            base += block_sizes.get(block.block_type, 3)

        # Cap at reasonable sizes
        base = min(base, 150)
        base = max(base, 25)

        return (base, base * 0.8)

    def _estimate_layers(self, requirements: CircuitRequirements) -> int:
        """Estimate required PCB layers"""
        # Complex designs need more layers
        complexity = len(requirements.blocks)

        if complexity < 3:
            return 2
        elif complexity < 8:
            return 2
        elif complexity < 15:
            return 4
        else:
            return 4


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Circuit AI Self-Test")
    print("=" * 60)

    # Test enhanced NLP parsing
    print("\n--- Enhanced NLP Parser Test ---")
    nlp = EnhancedNLPParser()

    test_inputs = [
        "ESP32-S3 with BME280 temperature sensor, OLED display, and LoRa communication",
        "Battery-powered GPS tracker with BLE and accelerometer",
        "STM32 based motor controller for 2 stepper motors with CAN bus interface",
        "Arduino Nano clone with 4 channel relay and current sensing",
        "USB powered addressable LED controller with 8 channels",
    ]

    for text in test_inputs:
        print(f"\nInput: {text}")
        req = nlp.parse(text)
        print(f"  Project: {req.project_name}")
        print(f"  MCU: {req.mcu_family.value}")
        print(f"  Power: {req.input_power.value}")
        print(f"  Features: {req.mcu_features}")
        print(f"  Blocks: {len(req.blocks)}")
        for block in req.blocks:
            print(f"    - {block.name}: {block.description}")
        print(f"  Board: {req.board_size_mm[0]:.0f}x{req.board_size_mm[1]:.0f}mm, {req.layers} layers")

    # Test original CircuitAI
    print("\n--- Original CircuitAI Test ---")
    ai = CircuitAI()

    test_inputs = [
        "ESP32-based temperature and humidity logger with WiFi",
        "Battery-powered motion sensor with Bluetooth",
        "USB-powered 8 channel LED controller",
    ]

    for text in test_inputs:
        print(f"\nInput: {text}")
        requirements = ai.parse_natural_language(text)
        print(f"  MCU: {requirements.mcu_family.value}")
        print(f"  Power: {requirements.input_power.value}")
        print(f"  Features: {requirements.mcu_features}")
        print(f"  Blocks: {len(requirements.blocks)}")

    # Test full parts_db generation
    print("\n" + "=" * 60)
    print("Testing parts_db generation")
    print("=" * 60)

    result = quick_design("ESP32 temperature logger with WiFi")
    print(f"Components: {len(result.parts_db['parts'])}")
    print(f"Nets: {len(result.parts_db['nets'])}")
    print(f"Suggestions: {result.suggestions}")

    # Test handoff creation
    print("\n" + "=" * 60)
    print("Testing Engine Handoff")
    print("=" * 60)

    interface = CircuitAIEngineInterface()
    handoff = interface.create_handoff(result)
    print(f"Handoff created: {handoff.project_name}")
    print(f"Board: {handoff.board_constraints['width']}x{handoff.board_constraints['height']}mm")
    print(f"Design rules: {handoff.design_rules}")

    print("\n" + "=" * 60)
    print("Circuit AI Self-Test PASSED")
    print("=" * 60)
