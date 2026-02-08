#!/usr/bin/env python3
"""
PCB ENGINE - Main Entry Point
==============================

USER INTERACTION SYSTEM
=======================

This is the MAIN entry point for users to interact with the PCB Engine.

INTERACTION MODES:
==================

1. NATURAL LANGUAGE (Easiest)
   python main.py "ESP32 temperature logger with WiFi and OLED display"

2. INTERACTIVE WIZARD
   python main.py --wizard

3. FROM FILE (JSON/YAML requirements)
   python main.py --from-file requirements.json

4. PYTHON API (For developers)
   from main import design_pcb
   result = design_pcb("ESP32 sensor board")

COMMAND HIERARCHY:
==================
    USER (You - The Boss)
        ↓ Natural language / requirements
    CIRCUIT AI (The Engineer)
        ↓ Intelligent decisions
    PCB ENGINE (The Foreman)
        ↓ Work orders
    18 PISTONS (The Workers)
        ↓ Execute tasks
    OUTPUT FILES (KiCad, Gerber, BOM)
"""

import sys
import os
import argparse
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from circuit_ai import (
        CircuitAI, CircuitAIEngineInterface, EnhancedNLPParser,
        CircuitRequirements, CircuitAIResult, MCUFamily, PowerType,
        quick_design, full_design, generate_netlist_kicad, generate_spice_netlist
    )
    from pcb_engine import PCBEngine, EngineConfig, EngineResult
    from piston_orchestrator import PistonOrchestrator, select_pistons_for_design
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all piston files are in the same directory.")
    sys.exit(1)


# =============================================================================
# COLORS FOR TERMINAL OUTPUT
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)"""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.ENDC = cls.BOLD = ''


# =============================================================================
# USER INTERFACE
# =============================================================================

def print_banner():
    """Print welcome banner"""
    print(f"""
{Colors.CYAN}+====================================================================+
|                                                                    |
|   {Colors.BOLD}PPPP   CCCC  BBBB      EEEEE  N   N   GGGG  III  N   N  EEEEE{Colors.CYAN}   |
|   {Colors.BOLD}P   P C      B   B     E      NN  N  G       I   NN  N  E    {Colors.CYAN}   |
|   {Colors.BOLD}PPPP  C      BBBB      EEEE   N N N  G  GG   I   N N N  EEEE {Colors.CYAN}   |
|   {Colors.BOLD}P     C      B   B     E      N  NN  G   G   I   N  NN  E    {Colors.CYAN}   |
|   {Colors.BOLD}P      CCCC  BBBB      EEEEE  N   N   GGGG  III  N   N  EEEEE{Colors.CYAN}   |
|                                                                    |
|   {Colors.GREEN}AI-Powered PCB Design System{Colors.CYAN}                                  |
|   {Colors.YELLOW}18 Specialized Pistons - Intelligent Routing - ML Learning{Colors.CYAN}   |
|                                                                    |
+====================================================================+{Colors.ENDC}
""")


def print_help():
    """Print detailed help"""
    print(f"""
{Colors.BOLD}PCB ENGINE - AI-Powered PCB Design System{Colors.ENDC}

{Colors.CYAN}USAGE:{Colors.ENDC}
  python main.py [OPTIONS] [DESCRIPTION]

{Colors.CYAN}MODES:{Colors.ENDC}

  {Colors.GREEN}1. Natural Language (Quickest){Colors.ENDC}
     python main.py "ESP32 temperature logger with WiFi"
     python main.py "Battery-powered GPS tracker with LoRa"
     python main.py "STM32 motor controller for 2 steppers"

  {Colors.GREEN}2. Interactive Wizard{Colors.ENDC}
     python main.py --wizard
     python main.py -w

  {Colors.GREEN}3. From Requirements File{Colors.ENDC}
     python main.py --from-file requirements.json
     python main.py -f my_design.yaml

  {Colors.GREEN}4. Quick Design (Parts Only){Colors.ENDC}
     python main.py --quick "ESP32 sensor board"

  {Colors.GREEN}5. Full Design (Complete PCB){Colors.ENDC}
     python main.py --full "ESP32 IoT device"

{Colors.CYAN}OPTIONS:{Colors.ENDC}
  -h, --help          Show this help message
  -w, --wizard        Start interactive wizard
  -f, --from-file     Load requirements from JSON/YAML file
  -q, --quick         Quick design (parts database only)
  --full              Full design with PCB layout
  -o, --output DIR    Output directory (default: ./output)
  --format FORMAT     Output format: kicad, gerber, both (default: both)
  --layers N          Force layer count (2 or 4)
  --size WxH          Force board size in mm (e.g., 50x50)
  --no-color          Disable colored output
  -v, --verbose       Verbose output

{Colors.CYAN}EXAMPLES:{Colors.ENDC}
  {Colors.YELLOW}# Simple sensor board{Colors.ENDC}
  python main.py "ESP32 with BME280 temperature sensor"

  {Colors.YELLOW}# Complex IoT device{Colors.ENDC}
  python main.py "ESP32-S3 with OLED display, BME280, LoRa, and GPS"

  {Colors.YELLOW}# Motor controller{Colors.ENDC}
  python main.py "STM32 motor controller with 2 stepper drivers and CAN bus"

  {Colors.YELLOW}# Battery-powered device{Colors.ENDC}
  python main.py "Battery-powered BLE beacon with accelerometer"

  {Colors.YELLOW}# Custom output{Colors.ENDC}
  python main.py --output ./my_pcb --format kicad "Arduino shield"

{Colors.CYAN}SUPPORTED COMPONENTS:{Colors.ENDC}
  MCUs:       ESP32, ESP8266, STM32, ATmega, RP2040, nRF52, ATtiny
  Sensors:    Temperature, Humidity, Pressure, IMU, GPS, Distance, Light
  Displays:   OLED, TFT, LCD, LED Matrix, E-Paper
  Comms:      WiFi, BLE, LoRa, Ethernet, CAN, RS485, Zigbee
  Motors:     DC, Stepper, Servo, Brushless
  Audio:      Speaker, Microphone, DAC, Amplifier
  Power:      USB, LiPo, 18650, Solar, DC Jack
""")


# =============================================================================
# INTERACTIVE WIZARD
# =============================================================================

def run_wizard() -> CircuitRequirements:
    """Run interactive design wizard"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== PCB Design Wizard ==={Colors.ENDC}\n")
    print("I'll help you design your PCB step by step.\n")

    requirements = CircuitRequirements()

    # Step 1: Project Name
    print(f"{Colors.GREEN}Step 1/7: Project Name{Colors.ENDC}")
    name = input("  What's your project name? [my_project]: ").strip()
    requirements.project_name = name if name else "my_project"

    # Step 2: Description
    print(f"\n{Colors.GREEN}Step 2/7: Description{Colors.ENDC}")
    print("  Describe your project (or press Enter to skip):")
    desc = input("  > ").strip()
    requirements.description = desc

    # Step 3: MCU Selection
    print(f"\n{Colors.GREEN}Step 3/7: Microcontroller{Colors.ENDC}")
    print("  Select your MCU:")
    print("    1. ESP32 (WiFi + Bluetooth)")
    print("    2. ESP32-S3 (WiFi + BLE + USB OTG)")
    print("    3. STM32 (High performance)")
    print("    4. ATmega/Arduino (Simple, reliable)")
    print("    5. RP2040/Pico (Dual-core, PIO)")
    print("    6. nRF52 (Low power BLE)")
    print("    7. ATtiny (Tiny, low power)")

    mcu_choice = input("  Choice [1]: ").strip() or "1"
    mcu_map = {
        "1": MCUFamily.ESP32,
        "2": MCUFamily.ESP32,
        "3": MCUFamily.STM32,
        "4": MCUFamily.ATMEGA,
        "5": MCUFamily.RP2040,
        "6": MCUFamily.NRF52,
        "7": MCUFamily.ATTINY,
    }
    requirements.mcu_family = mcu_map.get(mcu_choice, MCUFamily.ESP32)
    if mcu_choice == "2":
        requirements.mcu_features.append("usb_otg")

    # Step 4: Power Source
    print(f"\n{Colors.GREEN}Step 4/7: Power Source{Colors.ENDC}")
    print("  How will your device be powered?")
    print("    1. USB (5V)")
    print("    2. LiPo Battery (3.7V)")
    print("    3. 18650 Battery")
    print("    4. AA Batteries")
    print("    5. DC Barrel Jack (9-12V)")
    print("    6. Solar Panel")

    power_choice = input("  Choice [1]: ").strip() or "1"
    power_map = {
        "1": PowerType.USB_5V,
        "2": PowerType.BATTERY_LIPO,
        "3": PowerType.BATTERY_18650,
        "4": PowerType.BATTERY_AA,
        "5": PowerType.DC_BARREL,
        "6": PowerType.SOLAR,
    }
    requirements.input_power = power_map.get(power_choice, PowerType.USB_5V)

    # Step 5: Sensors
    print(f"\n{Colors.GREEN}Step 5/7: Sensors{Colors.ENDC}")
    print("  What sensors do you need? (comma-separated, or Enter for none)")
    print("  Options: temperature, humidity, pressure, imu, gps, distance, light, current")
    sensors = input("  > ").strip()

    if sensors:
        nlp = EnhancedNLPParser()
        for sensor in sensors.split(","):
            sensor = sensor.strip().lower()
            if sensor:
                block = nlp._create_sensor_block(sensor)
                requirements.blocks.append(block)

    # Step 6: Peripherals
    print(f"\n{Colors.GREEN}Step 6/7: Peripherals{Colors.ENDC}")
    print("  What peripherals do you need? (comma-separated, or Enter for none)")
    print("  Options: oled, tft, lcd, lora, ethernet, can, motor, stepper, led, speaker")
    peripherals = input("  > ").strip()

    if peripherals:
        nlp = EnhancedNLPParser()
        for periph in peripherals.split(","):
            periph = periph.strip().lower()
            if periph in ["oled", "tft", "lcd"]:
                block = nlp._create_display_block(periph)
                requirements.blocks.append(block)
            elif periph in ["lora", "ethernet", "can", "rs485"]:
                block = nlp._create_comm_block(periph)
                requirements.blocks.append(block)
            elif periph in ["motor", "stepper", "servo"]:
                block = nlp._create_motor_block(periph if periph != "motor" else "dc")
                requirements.blocks.append(block)
            elif periph in ["led", "neopixel"]:
                block = nlp._create_led_block("addressable")
                requirements.blocks.append(block)
            elif periph in ["speaker", "audio"]:
                block = nlp._create_audio_block("speaker")
                requirements.blocks.append(block)

    # Step 7: Board Size
    print(f"\n{Colors.GREEN}Step 7/7: Board Size{Colors.ENDC}")
    print("  Target board size:")
    print("    1. Tiny (25x25mm)")
    print("    2. Small (50x40mm)")
    print("    3. Medium (80x60mm)")
    print("    4. Large (100x80mm)")
    print("    5. Auto (let AI decide)")

    size_choice = input("  Choice [5]: ").strip() or "5"
    size_map = {
        "1": (25.0, 25.0),
        "2": (50.0, 40.0),
        "3": (80.0, 60.0),
        "4": (100.0, 80.0),
    }
    if size_choice in size_map:
        requirements.board_size_mm = size_map[size_choice]
    else:
        # Auto-calculate based on components
        nlp = EnhancedNLPParser()
        requirements.board_size_mm = nlp._estimate_board_size(requirements)

    print(f"\n{Colors.CYAN}=== Configuration Complete ==={Colors.ENDC}")
    print(f"  Project: {requirements.project_name}")
    print(f"  MCU: {requirements.mcu_family.value}")
    print(f"  Power: {requirements.input_power.value}")
    print(f"  Blocks: {len(requirements.blocks)}")
    print(f"  Board: {requirements.board_size_mm[0]:.0f}x{requirements.board_size_mm[1]:.0f}mm")

    return requirements


# =============================================================================
# DESIGN FUNCTIONS
# =============================================================================

def design_pcb(description: str,
               output_dir: str = "./output",
               output_format: str = "both",
               verbose: bool = True) -> Dict:
    """
    Main function to design a PCB from natural language description.

    Args:
        description: Natural language description of the circuit
        output_dir: Directory for output files
        output_format: "kicad", "gerber", or "both"
        verbose: Print progress messages

    Returns:
        Dictionary with design results and file paths

    Example:
        result = design_pcb("ESP32 temperature logger with WiFi and OLED")
        print(f"Files: {result['files']}")
    """
    if verbose:
        print(f"\n{Colors.CYAN}=== Starting PCB Design ==={Colors.ENDC}")
        print(f"  Description: {description}")

    # Step 1: Parse requirements
    if verbose:
        print(f"\n{Colors.GREEN}[1/5] Parsing Requirements...{Colors.ENDC}")

    ai = CircuitAI()
    requirements = ai.parse_natural_language(description)

    if verbose:
        print(f"  MCU: {requirements.mcu_family.value}")
        print(f"  Power: {requirements.input_power.value}")
        print(f"  Blocks: {len(requirements.blocks)}")
        for block in requirements.blocks:
            print(f"    - {block.name}")

    # Step 2: Generate parts database
    if verbose:
        print(f"\n{Colors.GREEN}[2/5] Generating Parts Database...{Colors.ENDC}")

    result = ai.generate_parts_db(requirements)

    if verbose:
        print(f"  Components: {len(result.parts_db['parts'])}")
        print(f"  Nets: {len(result.parts_db['nets'])}")

    # Step 3: Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Generate outputs
    if verbose:
        print(f"\n{Colors.GREEN}[3/5] Generating Schematic...{Colors.ENDC}")

    files = []

    # Generate KiCad schematic
    schematic_content = ai.export_schematic_kicad(result.parts_db, requirements)
    sch_path = os.path.join(output_dir, f"{requirements.project_name}.kicad_sch")
    with open(sch_path, 'w') as f:
        f.write(schematic_content)
    files.append(sch_path)
    if verbose:
        print(f"  Schematic: {sch_path}")

    # Generate netlist
    netlist_content = generate_netlist_kicad(result.parts_db)
    net_path = os.path.join(output_dir, f"{requirements.project_name}.net")
    with open(net_path, 'w') as f:
        f.write(netlist_content)
    files.append(net_path)
    if verbose:
        print(f"  Netlist: {net_path}")

    # Generate SPICE netlist
    spice_content = generate_spice_netlist(result.parts_db, requirements.project_name)
    spice_path = os.path.join(output_dir, f"{requirements.project_name}.cir")
    with open(spice_path, 'w') as f:
        f.write(spice_content)
    files.append(spice_path)
    if verbose:
        print(f"  SPICE: {spice_path}")

    # Generate BOM
    if verbose:
        print(f"\n{Colors.GREEN}[4/5] Generating BOM...{Colors.ENDC}")

    bom_path = os.path.join(output_dir, f"{requirements.project_name}_bom.csv")
    with open(bom_path, 'w') as f:
        f.write("Reference,Value,Footprint,Quantity,Description\n")
        for item in result.bom_preview:
            f.write(f"{item['refs']},{item['value']},{item['footprint']},{item['quantity']},\n")
    files.append(bom_path)
    if verbose:
        print(f"  BOM: {bom_path}")

    # Generate JSON parts database
    parts_path = os.path.join(output_dir, f"{requirements.project_name}_parts.json")
    with open(parts_path, 'w') as f:
        json.dump(result.parts_db, f, indent=2)
    files.append(parts_path)
    if verbose:
        print(f"  Parts DB: {parts_path}")

    # Step 5: Summary
    if verbose:
        print(f"\n{Colors.GREEN}[5/5] Design Complete!{Colors.ENDC}")
        print(f"\n{Colors.CYAN}=== Summary ==={Colors.ENDC}")
        print(f"  Project: {requirements.project_name}")
        print(f"  Components: {len(result.parts_db['parts'])}")
        print(f"  Nets: {len(result.parts_db['nets'])}")
        print(f"  Board Size: {requirements.board_size_mm[0]:.0f}x{requirements.board_size_mm[1]:.0f}mm")
        print(f"  Layers: {requirements.layers}")
        print(f"  Files Generated: {len(files)}")

        if result.suggestions:
            print(f"\n{Colors.YELLOW}Suggestions:{Colors.ENDC}")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")

    return {
        "success": True,
        "requirements": requirements,
        "parts_db": result.parts_db,
        "bom": result.bom_preview,
        "files": files,
        "suggestions": result.suggestions,
    }


def quick_parts(description: str, verbose: bool = True) -> CircuitAIResult:
    """
    Quick design - generates parts database only (no PCB layout).

    Faster than full design, useful for:
    - Checking component selection
    - Getting BOM estimate
    - Verifying requirements parsing
    """
    if verbose:
        print(f"\n{Colors.CYAN}=== Quick Design ==={Colors.ENDC}")
        print(f"  Description: {description}")

    result = quick_design(description)

    if verbose:
        print(f"\n  MCU: {result.requirements.mcu_family.value}")
        print(f"  Power: {result.requirements.input_power.value}")
        print(f"  Components: {len(result.parts_db['parts'])}")
        print(f"  Nets: {len(result.parts_db['nets'])}")

        print(f"\n{Colors.GREEN}Components:{Colors.ENDC}")
        for ref, part in result.parts_db['parts'].items():
            print(f"  {ref}: {part['value']}")

        print(f"\n{Colors.GREEN}BOM Preview:{Colors.ENDC}")
        for item in result.bom_preview[:10]:
            print(f"  {item['refs']}: {item['value']} x{item['quantity']}")

    return result


def run_full_design(description: str, output_dir: str = "./output") -> EngineResult:
    """
    Full design - runs complete PCB layout with all pistons.

    This is the most comprehensive mode:
    - Parses requirements
    - Selects components
    - Places components
    - Routes traces
    - Runs DRC
    - Generates output files
    """
    print(f"\n{Colors.CYAN}=== Full PCB Design ==={Colors.ENDC}")
    print(f"  Description: {description}")
    print(f"  Output: {output_dir}")
    print()

    # Run the complete design flow
    result = full_design(description)

    if result and result.success:
        print(f"\n{Colors.GREEN}Design completed successfully!{Colors.ENDC}")
        print(f"  Files: {result.output_files}")
    else:
        print(f"\n{Colors.RED}Design failed.{Colors.ENDC}")
        if result:
            print(f"  Stage: {result.stage_reached}")
            for error in result.errors:
                print(f"  Error: {error}")

    return result


# =============================================================================
# FILE LOADING
# =============================================================================

def load_requirements_file(filepath: str) -> CircuitRequirements:
    """Load requirements from JSON or YAML file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML files: pip install pyyaml")
        else:
            data = json.load(f)

    # Convert to CircuitRequirements
    requirements = CircuitRequirements()

    if 'project_name' in data:
        requirements.project_name = data['project_name']
    if 'description' in data:
        requirements.description = data['description']
    if 'mcu' in data:
        requirements.mcu_family = MCUFamily(data['mcu'])
    if 'power' in data:
        requirements.input_power = PowerType(data['power'])
    if 'board_size' in data:
        requirements.board_size_mm = tuple(data['board_size'])
    if 'layers' in data:
        requirements.layers = data['layers']

    # Parse blocks from natural language if provided
    if 'components' in data:
        nlp = EnhancedNLPParser()
        for comp in data['components']:
            if isinstance(comp, str):
                # Parse component string
                req = nlp.parse(comp)
                requirements.blocks.extend(req.blocks)
            elif isinstance(comp, dict):
                # Direct block specification
                pass  # TODO: parse dict format

    return requirements


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PCB Engine - AI-Powered PCB Design",
        add_help=False
    )

    parser.add_argument('description', nargs='?', help='Circuit description')
    parser.add_argument('-h', '--help', action='store_true', help='Show help')
    parser.add_argument('-w', '--wizard', action='store_true', help='Interactive wizard')
    parser.add_argument('-f', '--from-file', type=str, help='Load from file')
    parser.add_argument('-q', '--quick', action='store_true', help='Quick design (parts only)')
    parser.add_argument('--full', action='store_true', help='Full PCB design')
    parser.add_argument('-o', '--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--format', type=str, default='both', help='Output format')
    parser.add_argument('--layers', type=int, help='Force layer count')
    parser.add_argument('--size', type=str, help='Force board size (WxH)')
    parser.add_argument('--no-color', action='store_true', help='Disable colors')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    # Piston selection options
    parser.add_argument('--show-pistons', action='store_true', help='Preview which pistons will run')
    parser.add_argument('--thermal', action='store_true', help='Enable thermal analysis')
    parser.add_argument('--signal-integrity', action='store_true', help='Enable signal integrity analysis')
    parser.add_argument('--3d', dest='gen_3d', action='store_true', help='Generate 3D visualization')
    parser.add_argument('--optimize-bom', action='store_true', help='Enable BOM cost optimization')
    parser.add_argument('--all-pistons', action='store_true', help='Run all 18 pistons')
    parser.add_argument('--minimal', action='store_true', help='Run only essential pistons')

    args = parser.parse_args()

    # Disable colors if requested or not a terminal
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Show help
    if args.help or (not args.description and not args.wizard and not args.from_file):
        print_banner()
        print_help()
        return 0

    # Interactive wizard
    if args.wizard:
        print_banner()
        requirements = run_wizard()

        # Generate design from wizard results
        ai = CircuitAI()
        ai.requirements = requirements
        result = ai.generate_parts_db(requirements)

        print(f"\n{Colors.GREEN}Generating output files...{Colors.ENDC}")
        design_result = design_pcb(
            requirements.description or f"{requirements.mcu_family.value} project",
            output_dir=args.output,
            verbose=True
        )
        return 0

    # Load from file
    if args.from_file:
        print_banner()
        print(f"Loading requirements from: {args.from_file}")
        requirements = load_requirements_file(args.from_file)

        ai = CircuitAI()
        ai.requirements = requirements
        result = ai.generate_parts_db(requirements)

        # Generate outputs
        design_result = design_pcb(
            requirements.description,
            output_dir=args.output,
            verbose=True
        )
        return 0

    # Natural language design
    if args.description:
        print_banner()

        if args.quick:
            # Quick design (parts only)
            quick_parts(args.description, verbose=True)
        elif args.full:
            # Full PCB design
            run_full_design(args.description, output_dir=args.output)
        else:
            # Standard design
            design_pcb(
                args.description,
                output_dir=args.output,
                output_format=args.format,
                verbose=True
            )
        return 0

    # No valid input
    print_help()
    return 1


# =============================================================================
# PYTHON API EXAMPLES
# =============================================================================

def example_usage():
    """
    Example usage of PCB Engine from Python code.

    This shows how developers can integrate PCB Engine into their own applications.
    """
    print("=" * 60)
    print("PCB Engine - Python API Examples")
    print("=" * 60)

    # Example 1: Quick design from natural language
    print("\n--- Example 1: Quick Design ---")
    result = quick_design("ESP32 temperature logger with WiFi")
    print(f"Components: {len(result.parts_db['parts'])}")

    # Example 2: Full design with custom output
    print("\n--- Example 2: Full Design ---")
    design_result = design_pcb(
        "Battery-powered GPS tracker with BLE",
        output_dir="./my_gps_tracker",
        verbose=True
    )
    print(f"Files generated: {design_result['files']}")

    # Example 3: Using CircuitAI directly
    print("\n--- Example 3: CircuitAI Direct Usage ---")
    ai = CircuitAI()

    # Parse requirements
    requirements = ai.parse_natural_language("STM32 motor controller with CAN bus")
    print(f"MCU: {requirements.mcu_family.value}")
    print(f"Blocks: {[b.name for b in requirements.blocks]}")

    # Generate parts database
    result = ai.generate_parts_db(requirements)
    print(f"Parts: {list(result.parts_db['parts'].keys())}")

    # Export schematic
    schematic = ai.generate_schematic(result.parts_db, requirements)
    print(f"Schematic sheets: {len(schematic.sheets)}")

    # Export to KiCad
    kicad_content = ai.export_schematic_kicad(result.parts_db, requirements)
    print(f"KiCad schematic size: {len(kicad_content)} bytes")

    # Export netlist
    netlist = ai.export_netlist(result.parts_db, format='kicad')
    print(f"Netlist size: {len(netlist)} bytes")


if __name__ == '__main__':
    sys.exit(main())
