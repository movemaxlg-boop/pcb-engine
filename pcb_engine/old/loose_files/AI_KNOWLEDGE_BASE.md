# PCB ENGINE - AI KNOWLEDGE BASE
## Complete System Documentation for AI Agent

---

## 1. SYSTEM ARCHITECTURE

```
USER (The Boss)
    |
    | Natural language: "ESP32 with temperature sensor"
    v
CIRCUIT AI (The Engineer) - circuit_ai.py
    |
    | Intelligent decisions, component selection
    v
PCB ENGINE (The Foreman) - pcb_engine.py
    |
    | Work orders to pistons
    v
18 PISTONS (The Workers)
    |
    v
OUTPUT FILES (KiCad, Gerber, BOM, SPICE)
```

---

## 2. THE 18 PISTONS

### CORE PISTONS (Always Required)

| Piston | File | Purpose |
|--------|------|---------|
| **Parts** | parts_piston.py | Converts requirements to component database with footprints |
| **Order** | order_piston.py | Determines placement order, net priority, layer assignment |
| **Placement** | placement_piston.py | Places components using force-directed or simulated annealing |
| **Routing** | routing_piston.py | Routes traces using A*, Lee, Hadlock, or hybrid algorithms |
| **DRC** | drc_piston.py | Design Rule Check - validates clearance, width, overlap |
| **Output** | output_piston.py | Generates KiCad schematic, PCB, Gerber, BOM files |

### CONDITIONAL PISTONS (Auto-enabled)

| Piston | File | When Enabled |
|--------|------|--------------|
| **Escape** | escape_piston.py | BGA/QFN packages with dense pin patterns |
| **Optimize** | optimization_piston.py | After routing >80% complete |
| **Silkscreen** | silkscreen_piston.py | When reference designators needed |
| **Stackup** | stackup_piston.py | 4+ layer boards |
| **Netlist** | netlist_piston.py | Always (for simulation) |

### OPTIONAL PISTONS (User Preference)

| Piston | File | Purpose |
|--------|------|---------|
| **Thermal** | thermal_piston.py | Heat analysis for >2W designs |
| **PDN** | pdn_piston.py | Power delivery network analysis |
| **Signal Integrity** | signal_integrity_piston.py | High-speed signals (>100MHz) |
| **Topological Router** | topological_router_piston.py | Rubber-band routing for complex boards |
| **3D Visualization** | visualization_3d_piston.py | STL/STEP/GLTF export |
| **BOM Optimizer** | bom_optimizer_piston.py | Cost/availability optimization |
| **Learning** | learning_piston.py | ML-based improvement |

---

## 3. COMPONENT DATABASE (700+ Parts)

### Microcontrollers (MCU_DATABASE)
```python
'ESP32-WROOM-32': {
    'footprint': 'ESP32-WROOM-32',
    'pins': ['EN', 'IO0', 'IO1', ..., 'GND', 'VCC'],
    'voltage': 3.3,
    'interfaces': ['SPI', 'I2C', 'UART', 'WiFi', 'BLE'],
    'flash': '4MB',
    'ram': '520KB'
}
```

**Available MCUs:**
- ESP32-WROOM-32, ESP32-S3-WROOM-1, ESP32-C3-MINI-1
- STM32F103C8T6, STM32F411CEU6, STM32F407VGT6
- ATMEGA328P-AU, ATMEGA2560-16AU
- RP2040
- NRF52840
- ATTINY85-20SU

### Sensors (SENSOR_DATABASE)
```python
'BME280': {
    'footprint': 'Bosch_LGA-8_2.5x2.5mm',
    'interface': 'I2C/SPI',
    'i2c_address': '0x76 or 0x77',
    'voltage': '1.8-3.6V',
    'measurements': ['temperature', 'humidity', 'pressure']
}
```

**Available Sensors:**
- Temperature: BME280, BMP280, SHT31, DS18B20, AHT20
- Motion: MPU6050, ICM-20948, LSM6DS3
- Distance: VL53L0X, HC-SR04
- Light: VEML7700, BH1750
- Current: INA219
- GPS: NEO-6M, NEO-M8N
- Heart: MAX30102
- Weight: HX711

### Displays (DISPLAY_DATABASE)
```python
'SSD1306': {
    'footprint': 'OLED_128x64_I2C',
    'interface': 'I2C',
    'resolution': '128x64',
    'voltage': '3.3V',
    'i2c_address': '0x3C or 0x3D'
}
```

**Available Displays:**
- OLED: SSD1306, SH1106
- TFT: ST7735, ST7789, ILI9341
- LCD: HD44780
- LED: MAX7219, WS2812B

### Communication (COMMUNICATION_DATABASE)
```python
'SX1278': {
    'footprint': 'SX1278_Module',
    'interface': 'SPI',
    'frequency': '433/868/915MHz',
    'range': '10+ km',
    'protocol': 'LoRa'
}
```

**Available Modules:**
- WiFi: ESP-01, ESP8266
- Bluetooth: HC-05, HC-06, HM-10
- LoRa: SX1278, RFM95W
- RF: NRF24L01
- Ethernet: W5500, ENC28J60
- CAN: MCP2515 + TJA1050
- RS485: MAX485
- USB: CH340G, CP2102

### Motor Drivers (MOTOR_DRIVER_DATABASE)
```python
'DRV8833': {
    'footprint': 'HTSSOP-16',
    'channels': 2,
    'current_per_channel': '1.5A',
    'voltage': '2.7-10.8V',
    'interface': 'PWM'
}
```

**Available Drivers:**
- DC: DRV8833, TB6612FNG, L298N, L9110S, BTS7960
- Stepper: A4988, DRV8825, TMC2209

### Power Supply (POWER_IC_DATABASE)
```python
'AP2112K-3.3': {
    'footprint': 'SOT-23-5',
    'type': 'LDO',
    'output': '3.3V',
    'current': '600mA',
    'quiescent': '55uA',
    'dropout': '250mV'
}
```

**Available Power ICs:**
- LDO: AMS1117-3.3, AP2112K-3.3, MCP1700
- Buck: LM2596, MP1584, TPS62200
- Boost: MT3608, TPS61200
- Buck-Boost: TPS63020
- Charger: TP4056, MCP73831, BQ24074
- Protection: DW01A, FS8205A

### Connectors (CONNECTOR_DATABASE)
```python
'USB_C': {
    'footprint': 'USB_C_Receptacle_GCT_USB4105',
    'pins': ['VBUS', 'GND', 'CC1', 'CC2', 'D+', 'D-'],
    'voltage': '5V',
    'current': '3A max'
}
```

**Available Connectors:**
- USB: USB_C, USB_Micro, USB_Mini
- Headers: Pin_Header_1x04, Pin_Header_2x10
- JST: JST_PH_2, JST_XH_4
- Barrel: Barrel_Jack_5.5x2.1
- Card: SD_Card, MicroSD

---

## 4. CIRCUIT AI (circuit_ai.py)

### Natural Language Parser
```python
# Parses user input like:
"ESP32 temperature sensor with OLED display, battery powered"

# Extracts:
- MCU: ESP32
- Sensors: temperature (BME280)
- Display: OLED (SSD1306)
- Power: Battery (TP4056 + AP2112K)
```

### Block Types
```python
class BlockType(Enum):
    MCU = 'mcu'
    SENSOR = 'sensor'
    DISPLAY = 'display'
    COMMUNICATION = 'communication'
    POWER = 'power'
    MOTOR = 'motor'
    STORAGE = 'storage'
    INTERFACE = 'interface'
    PROTECTION = 'protection'
```

### Power Types
```python
class PowerType(Enum):
    USB_5V = 'usb_5v'
    BATTERY_LIPO = 'battery_lipo'
    BATTERY_18650 = 'battery_18650'
    EXTERNAL_12V = 'external_12v'
    EXTERNAL_24V = 'external_24v'
    SOLAR = 'solar'
```

---

## 5. PCB ENGINE (pcb_engine.py)

### Configuration
```python
EngineConfig(
    board_width=100.0,      # mm
    board_height=100.0,     # mm
    layer_count=2,          # 2 or 4
    trace_width=0.25,       # mm
    clearance=0.15,         # mm
    via_diameter=0.8,       # mm
    via_drill=0.4,          # mm
    routing_algorithm='hybrid',  # 'astar', 'lee', 'hadlock', 'hybrid'
)
```

### Effort Levels
```python
class PistonEffort(Enum):
    NORMAL = 'normal'     # Standard processing
    HARDER = 'harder'     # More iterations
    DEEPER = 'deeper'     # Try more algorithms
    LONGER = 'longer'     # Extended timeout
    MAXIMUM = 'maximum'   # All of the above
```

### Challenge Types
```python
class ChallengeType(Enum):
    BOARD_TOO_SMALL = 'board_too_small'
    LAYERS_NEEDED = 'layers_needed'
    ROUTING_IMPOSSIBLE = 'routing_impossible'
    DRC_PERSISTENT = 'drc_persistent'
    THERMAL_ISSUE = 'thermal_issue'
    SIGNAL_INTEGRITY = 'signal_integrity'
```

---

## 6. PISTON ORCHESTRATOR (piston_orchestrator.py)

### Auto-Detection Rules
```python
# Escape piston enabled when:
- BGA package detected
- QFN package detected
- Fine pitch (<0.5mm) components

# Thermal piston enabled when:
- Motor driver detected
- Total power > 5W
- High-power components (>2W single)

# Signal Integrity enabled when:
- USB 3.0 detected
- PCIe detected
- DDR memory detected
- Signals > 100MHz

# Stackup piston enabled when:
- Layer count >= 4
```

---

## 7. OUTPUT FILES

### KiCad Schematic (.kicad_sch)
- KiCad 7+ format
- Symbols with pin assignments
- Net connections
- Power symbols

### KiCad Netlist (.net)
```
(export
  (components
    (comp (ref U1) (value ESP32-WROOM-32) (footprint ESP32-WROOM-32))
    (comp (ref U2) (value BME280) (footprint LGA-8))
  )
  (nets
    (net (code 1) (name VCC) (node (ref U1) (pin VCC)) (node (ref U2) (pin VDD)))
  )
)
```

### SPICE Netlist (.cir)
```spice
* ESP32 Sensor Board
.SUBCKT ESP32_SENSOR VCC GND SDA SCL
R1 SDA VCC 4.7k
R2 SCL VCC 4.7k
C1 VCC GND 100n
.ENDS
```

### Bill of Materials (.csv)
```csv
Reference,Value,Footprint,Quantity,Description
U1,ESP32-WROOM-32,ESP32-WROOM-32,1,WiFi+BLE MCU
U2,BME280,LGA-8,1,Temp/Humidity/Pressure
R1-R2,4.7k,0402,2,I2C Pull-ups
C1-C4,100nF,0402,4,Decoupling
```

---

## 8. DESIGN RULES

### Trace Width
| Current | Width (1oz Cu) |
|---------|----------------|
| < 0.5A  | 0.25mm |
| 0.5-1A  | 0.3mm |
| 1-2A    | 0.5mm |
| 2-3A    | 0.8mm |
| > 3A    | 1.0mm+ |

### Clearance
| Voltage | Clearance |
|---------|-----------|
| < 50V   | 0.15mm |
| 50-100V | 0.5mm |
| > 100V  | Calculate |

### Via Sizing
| Type | Pad | Drill |
|------|-----|-------|
| Signal | 0.8mm | 0.4mm |
| Power | 1.0mm | 0.5mm |
| Thermal | 1.2mm | 0.6mm |

### Layer Stackup (4-layer)
```
Layer 1: Signal (Top)
Layer 2: Ground Plane
Layer 3: Power Plane
Layer 4: Signal (Bottom)
```

---

## 9. COMMON ISSUES & WARNINGS

### ESP32 GPIO Restrictions
- GPIO 6-11: Connected to internal flash - DO NOT USE
- GPIO 34-39: Input only, no internal pull-up/down
- GPIO 0, 2, 15: Boot strapping pins - be careful

### I2C Address Conflicts
Check these common addresses:
- BME280: 0x76 or 0x77
- SSD1306: 0x3C or 0x3D
- MPU6050: 0x68 or 0x69
- INA219: 0x40-0x4F

### Power Supply Issues
- LiPo direct to 3.3V MCU: Need regulator (4.2V too high!)
- 5V sensors with 3.3V MCU: Need level shifter
- Motors on same rail as MCU: Separate with ferrite bead

### EMI/Antenna
- No traces under WiFi/BLE antenna
- Ground plane under RF section
- Keep crystal away from noisy signals

---

## 10. API USAGE

### Quick Design
```python
from circuit_ai import quick_design

result = quick_design("ESP32 temperature sensor with OLED")
print(result.parts_db)
print(result.bom_preview)
```

### Full Design
```python
from main import design_pcb

result = design_pcb(
    "ESP32 weather station with BME280, OLED, and LoRa",
    output_dir="./my_project",
    verbose=True
)
```

### Using Pistons Directly
```python
from pcb_engine import PCBEngine, EngineConfig

config = EngineConfig(
    board_width=50,
    board_height=40,
    layer_count=2
)
engine = PCBEngine(config)
result = engine.run_smart(parts_db, user_preferences={
    'generate_3d': True,
    'run_thermal_analysis': True
})
```

---

## 11. WEB API ENDPOINTS

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /api/chat | POST | AI conversation |
| /api/design | POST | Full design from description |
| /api/parse | POST | Parse requirements only |
| /api/generate | POST | Generate parts database |
| /api/export | POST | Export files (KiCad, SPICE, BOM) |
| /api/custom | POST | Add custom component |

---

## 12. FILE STRUCTURE

```
pcb_engine/
├── circuit_ai.py          # THE ENGINEER - NLP + Component Selection
├── pcb_engine.py          # THE FOREMAN - Orchestrates pistons
├── piston_orchestrator.py # Smart piston selection
├── ai_connector.py        # External AI integration (Groq/Gemini)
├── main.py                # CLI entry point
│
├── *_piston.py            # 18 specialized workers
│   ├── parts_piston.py
│   ├── order_piston.py
│   ├── placement_piston.py
│   ├── routing_piston.py
│   ├── escape_piston.py
│   ├── optimization_piston.py
│   ├── silkscreen_piston.py
│   ├── drc_piston.py
│   ├── output_piston.py
│   ├── stackup_piston.py
│   ├── thermal_piston.py
│   ├── pdn_piston.py
│   ├── signal_integrity_piston.py
│   ├── netlist_piston.py
│   ├── topological_router_piston.py
│   ├── visualization_3d_piston.py
│   ├── bom_optimizer_piston.py
│   └── learning_piston.py
│
├── web/
│   ├── server.py          # Web server with REST API
│   ├── index.html         # Chat interface
│   └── footprint_form.html # Custom component designer
│
├── output/                # Generated files go here
├── .env                   # API keys
└── AI_KNOWLEDGE_BASE.md   # This file
```

---

*This document is the complete reference for the PCB Engine AI system.*
