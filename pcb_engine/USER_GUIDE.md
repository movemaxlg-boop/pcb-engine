# PCB ENGINE - User Interaction Guide

## Quick Start

```bash
# Install (no dependencies required - pure Python!)
cd kicad_sensor_module/pcb_engine

# Design a PCB with one command
python main.py "ESP32 temperature sensor with OLED display"
```

## 5 Ways to Interact with PCB Engine

### 1. Natural Language (Easiest - Recommended)

Just describe what you want in plain English:

```bash
# Simple sensor board
python main.py "ESP32 temperature logger with WiFi"

# Complex IoT device
python main.py "ESP32-S3 with BME280, OLED display, LoRa, and GPS"

# Motor controller
python main.py "STM32 motor controller for 2 stepper motors with CAN bus"

# Battery-powered device
python main.py "Battery-powered BLE beacon with accelerometer"

# LED controller
python main.py "USB powered 8 channel addressable LED controller"
```

**Output:** Generates KiCad schematic, netlist, BOM, and SPICE files automatically.

---

### 2. Interactive Wizard

Step-by-step guided design:

```bash
python main.py --wizard
# or
python main.py -w
```

The wizard asks you:
1. Project name
2. Description
3. MCU selection (ESP32, STM32, Arduino, etc.)
4. Power source (USB, Battery, Solar, etc.)
5. Sensors needed
6. Peripherals (display, motors, LEDs, etc.)
7. Board size

---

### 3. Quick Design (Parts Only)

Fast mode - just generates parts list without full PCB:

```bash
python main.py --quick "ESP32 sensor board"
# or
python main.py -q "GPS tracker"
```

Useful for:
- Checking component selection
- Getting BOM estimate
- Verifying requirements parsing

---

### 4. From Requirements File

Load design from JSON or YAML:

```bash
python main.py --from-file requirements.json
python main.py -f my_design.yaml
```

Example `requirements.json`:
```json
{
  "project_name": "my_sensor",
  "description": "Temperature monitoring system",
  "mcu": "esp32",
  "power": "usb_5v",
  "board_size": [50, 40],
  "layers": 2,
  "components": [
    "temperature sensor",
    "OLED display",
    "LoRa communication"
  ]
}
```

---

### 5. Python API (For Developers)

Use PCB Engine in your own Python code:

```python
from main import design_pcb, quick_parts
from circuit_ai import CircuitAI, quick_design

# Method A: One-liner design
result = design_pcb("ESP32 temperature logger with WiFi")
print(f"Files: {result['files']}")

# Method B: Quick parts only
parts = quick_parts("GPS tracker with BLE")
print(f"Components: {parts.parts_db['parts'].keys()}")

# Method C: Full control with CircuitAI
ai = CircuitAI()
requirements = ai.parse_natural_language("STM32 motor controller")
result = ai.generate_parts_db(requirements)

# Export to different formats
schematic = ai.export_schematic_kicad(result.parts_db, requirements)
netlist = ai.export_netlist(result.parts_db, format='spice')
```

---

## Command Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-w, --wizard` | Interactive wizard |
| `-f, --from-file FILE` | Load from JSON/YAML |
| `-q, --quick` | Quick design (parts only) |
| `--full` | Full PCB design with routing |
| `-o, --output DIR` | Output directory (default: ./output) |
| `--format FORMAT` | Output format: kicad, gerber, both |
| `--layers N` | Force layer count (2 or 4) |
| `--size WxH` | Force board size in mm |
| `--no-color` | Disable colored output |
| `-v, --verbose` | Verbose output |

---

## Supported Components

### Microcontrollers
- **ESP32** (WiFi + Bluetooth)
- **ESP32-S3** (USB OTG)
- **ESP8266** (WiFi)
- **STM32** (High performance)
- **ATmega/Arduino** (Simple)
- **RP2040/Pico** (Dual-core)
- **nRF52** (Low power BLE)
- **ATtiny** (Tiny)

### Sensors
| Type | Components |
|------|------------|
| Temperature | BME280, BMP280, AHT20, SHT31, DS18B20 |
| Humidity | BME280, SHT31, AHT20 |
| Pressure | BME280, BMP280 |
| IMU | MPU6050, LSM6DS3, ICM-20948 |
| Distance | VL53L0X, HC-SR04 |
| Light | VEML7700, BH1750 |
| GPS | NEO-6M |
| Current | INA219 |
| Weight | HX711 |
| Heart Rate | MAX30102 |

### Displays
| Type | Components |
|------|------------|
| OLED | SSD1306 (128x64, 128x32), SH1106 |
| TFT | ST7735, ST7789, ILI9341 |
| LCD | HD44780 (16x2, 20x4) |
| LED Matrix | MAX7219 |

### Communication
| Protocol | Components |
|----------|------------|
| WiFi | ESP-01 (built-in for ESP32) |
| Bluetooth | HC-05, HM-10 (built-in for ESP32) |
| LoRa | SX1278, RFM95W |
| RF | NRF24L01 |
| Ethernet | W5500, ENC28J60 |
| CAN | MCP2515 + TJA1050 |
| RS485 | MAX485 |
| USB-UART | CH340G, CP2102, FT232RL |

### Motor Drivers
| Type | Components |
|------|------------|
| DC Motor | DRV8833, TB6612FNG, L298N, L9110S |
| Stepper | TMC2209, A4988, DRV8825 |
| High Power | BTS7960, VNH5019 |

### Power
| Type | Components |
|------|------------|
| LDO | AMS1117, AP2112K |
| Buck | LM2596, MP1584 |
| Boost | MT3608, XL6009 |
| Buck-Boost | TPS63020 |
| Battery Charger | TP4056, MCP73831 |
| Battery Protection | DW01A + FS8205A |

---

## Output Files

After running a design, you get:

| File | Description |
|------|-------------|
| `project.kicad_sch` | KiCad 7+ schematic file |
| `project.net` | KiCad netlist |
| `project.cir` | SPICE netlist (for simulation) |
| `project_bom.csv` | Bill of Materials |
| `project_parts.json` | Full parts database |

---

## Examples

### IoT Weather Station
```bash
python main.py "ESP32 weather station with BME280, OLED display, and LoRa for remote monitoring" -o ./weather_station
```

### Robot Controller
```bash
python main.py "STM32 robot controller with 4 DC motors, 2 encoders, IMU, and Bluetooth" -o ./robot
```

### Home Automation Node
```bash
python main.py "ESP32 home automation node with 4 relay outputs, temperature sensor, and WiFi" -o ./home_node
```

### Wearable Device
```bash
python main.py "Battery-powered wearable with heart rate sensor, accelerometer, OLED, and BLE" -o ./wearable
```

---

## Architecture

```
USER (You - The Boss)
    |
    | Natural language description
    v
CIRCUIT AI (The Engineer)
    |
    | Intelligent decisions
    v
PCB ENGINE (The Foreman)
    |
    | Work orders
    v
18 PISTONS (The Workers)
    |
    v
OUTPUT FILES (KiCad, Gerber, BOM)
```

### The 18 Pistons:

**Core Flow (9):**
- Parts, Order, Placement, Escape, Routing, Optimize, Silkscreen, DRC, Output

**Analysis (5):**
- Stackup, Thermal, PDN, Signal Integrity, Netlist

**Advanced (3):**
- Topological Router, 3D Visualization, BOM Optimizer

**Machine Learning (1):**
- Learning (reverse engineers PCB files to improve algorithms)

---

## Need Help?

```bash
python main.py --help
```

Or check the source code in `circuit_ai.py` and `pcb_engine.py` for full API documentation.
