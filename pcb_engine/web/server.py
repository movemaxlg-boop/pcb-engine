#!/usr/bin/env python3
"""
PCB Engine Web Server
=====================

A simple web server that provides:
1. Static file serving for the web frontend
2. REST API endpoints for the PCB Engine
3. WebSocket support for real-time updates (optional)

Usage:
    python server.py
    # Then open http://localhost:8080

API Endpoints:
    POST /api/design     - Process a design request
    POST /api/parse      - Parse natural language to requirements
    POST /api/generate   - Generate parts database
    POST /api/export     - Export design files
    POST /api/custom     - Add custom component
"""

import http.server
import socketserver
import json
import os
import sys
import urllib.parse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from circuit_ai import (
        CircuitAI, EnhancedNLPParser, CircuitRequirements,
        quick_design, generate_netlist_kicad, generate_spice_netlist
    )
    from pcb_engine import PCBEngine, EngineConfig
    from output_piston import OutputPiston, OutputConfig
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import PCB Engine: {e}")
    ENGINE_AVAILABLE = False
    PCBEngine = None
    OutputPiston = None

try:
    from ai_connector import AIConnector, AIProvider, create_ai_connector
    AI_CONNECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import AI Connector: {e}")
    AI_CONNECTOR_AVAILABLE = False
    AIConnector = None
    AIProvider = None

# Global AI connector instance
_ai_connector = None

def _load_env_file():
    """Load .env file from parent directory"""
    import os
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"Loaded environment from {env_path}")

# Load .env on import
_load_env_file()

def get_ai_connector():
    """Get or create the AI connector"""
    global _ai_connector
    if _ai_connector is None and AI_CONNECTOR_AVAILABLE:
        # Try providers in order of preference
        # 1. Check for API keys in environment
        import os
        if os.environ.get('GROQ_API_KEY'):
            _ai_connector = create_ai_connector('groq', os.environ['GROQ_API_KEY'])
            print("Using Groq AI (free)")
        elif os.environ.get('GEMINI_API_KEY'):
            _ai_connector = create_ai_connector('gemini', os.environ['GEMINI_API_KEY'])
            print("Using Google Gemini AI (free)")
        else:
            # Try local Ollama
            _ai_connector = create_ai_connector('ollama')
            # Test if Ollama is running
            test = _ai_connector.chat("test")
            if not test.success:
                # Fall back to mock
                _ai_connector = create_ai_connector('mock')
                print("Using Mock AI (no API key set)")
            else:
                print("Using local Ollama AI (free)")
    return _ai_connector

PORT = 8080
WEB_DIR = Path(__file__).parent


class PCBEngineHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler with API endpoints"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        """Handle POST requests to API endpoints"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON")
            return

        # Route to appropriate handler
        if path == '/api/design':
            self.handle_design(data)
        elif path == '/api/parse':
            self.handle_parse(data)
        elif path == '/api/generate':
            self.handle_generate(data)
        elif path == '/api/export':
            self.handle_export(data)
        elif path == '/api/custom':
            self.handle_custom_part(data)
        elif path == '/api/chat':
            self.handle_chat(data)
        else:
            self.send_error_response(404, "Endpoint not found")

    def send_cors_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # Prevent caching for development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')

    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_error_response(self, status, message):
        """Send error response"""
        self.send_json_response({'error': message}, status)

    def handle_design(self, data):
        """Handle full design request"""
        description = data.get('description', '')

        if not description:
            self.send_error_response(400, "Description is required")
            return

        if not ENGINE_AVAILABLE:
            self.send_error_response(500, "PCB Engine not available")
            return

        try:
            # Process with CircuitAI
            ai = CircuitAI()
            requirements = ai.parse_natural_language(description)
            result = ai.generate_parts_db(requirements)

            # Build response
            response = {
                'success': True,
                'project_name': requirements.project_name,
                'mcu': requirements.mcu_family.value,
                'power': requirements.input_power.value,
                'board_size': list(requirements.board_size_mm),
                'layers': requirements.layers,
                'components': len(result.parts_db.get('parts', {})),
                'nets': len(result.parts_db.get('nets', {})),
                'parts_db': result.parts_db,
                'bom': result.bom_preview,
                'topology': result.topology,
                'suggestions': result.suggestions,
                'blocks': [
                    {'name': b.name, 'type': b.block_type.value, 'description': b.description}
                    for b in requirements.blocks
                ]
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_error_response(500, str(e))

    def handle_parse(self, data):
        """Handle requirement parsing only"""
        description = data.get('description', '')

        if not description:
            self.send_error_response(400, "Description is required")
            return

        if not ENGINE_AVAILABLE:
            self.send_error_response(500, "PCB Engine not available")
            return

        try:
            parser = EnhancedNLPParser()
            requirements = parser.parse(description)

            response = {
                'success': True,
                'project_name': requirements.project_name,
                'mcu': requirements.mcu_family.value,
                'power': requirements.input_power.value,
                'features': requirements.mcu_features,
                'board_size': list(requirements.board_size_mm),
                'layers': requirements.layers,
                'blocks': [
                    {'name': b.name, 'type': b.block_type.value, 'description': b.description}
                    for b in requirements.blocks
                ]
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_error_response(500, str(e))

    def handle_generate(self, data):
        """Handle parts database generation"""
        description = data.get('description', '')

        if not description:
            self.send_error_response(400, "Description is required")
            return

        if not ENGINE_AVAILABLE:
            self.send_error_response(500, "PCB Engine not available")
            return

        try:
            result = quick_design(description)

            response = {
                'success': True,
                'parts_db': result.parts_db,
                'bom': result.bom_preview,
                'topology': result.topology,
                'suggestions': result.suggestions
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_error_response(500, str(e))

    def handle_export(self, data):
        """Handle file export"""
        parts_db = data.get('parts_db')
        format_type = data.get('format', 'json')
        project_name = data.get('project_name', 'design')

        if not parts_db:
            self.send_error_response(400, "parts_db is required")
            return

        try:
            if format_type == 'kicad_netlist':
                content = generate_netlist_kicad(parts_db)
                filename = f"{project_name}.net"
            elif format_type == 'spice':
                content = generate_spice_netlist(parts_db, project_name)
                filename = f"{project_name}.cir"
            elif format_type == 'json':
                content = json.dumps(parts_db, indent=2)
                filename = f"{project_name}.json"
            elif format_type == 'csv':
                # Generate BOM CSV
                lines = ['Reference,Value,Footprint,Quantity']
                for ref, part in parts_db.get('parts', {}).items():
                    lines.append(f"{ref},{part.get('value', '')},{part.get('footprint', '')},1")
                content = '\n'.join(lines)
                filename = f"{project_name}_bom.csv"
            else:
                self.send_error_response(400, f"Unknown format: {format_type}")
                return

            response = {
                'success': True,
                'filename': filename,
                'content': content
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_error_response(500, str(e))

    def handle_custom_part(self, data):
        """Handle custom component addition"""
        part_data = data.get('part')

        if not part_data:
            self.send_error_response(400, "part data is required")
            return

        try:
            # Validate part data
            required_fields = ['reference', 'name', 'footprint', 'pins']
            for field in required_fields:
                if field not in part_data:
                    self.send_error_response(400, f"Missing field: {field}")
                    return

            # Format as parts_db entry
            ref = part_data['reference']
            entry = {
                'value': part_data.get('name', ''),
                'footprint': part_data.get('footprint', ''),
                'description': part_data.get('description', ''),
                'group': 'custom',
                'pins': part_data.get('pins', [])
            }

            response = {
                'success': True,
                'reference': ref,
                'entry': entry,
                'message': f"Custom component {ref} added successfully"
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_error_response(500, str(e))

    def handle_chat(self, data):
        """
        Handle chat message - this is the main AI interaction endpoint.
        Provides conversational responses about the design.
        """
        message = data.get('message', '')
        context = data.get('context', {})

        if not message:
            self.send_error_response(400, "Message is required")
            return

        try:
            response = self.process_chat_message(message, context)

            # Log conversation to file for debugging
            log_path = Path(__file__).parent.parent / 'chat_log.txt'
            with open(log_path, 'a', encoding='utf-8') as f:
                import datetime
                f.write(f"\n{'='*60}\n")
                f.write(f"[{datetime.datetime.now()}]\n")
                f.write(f"USER: {message}\n")
                f.write(f"AI: {response.get('text', 'No text')[:500]}...\n")
                f.write(f"Provider: {response.get('ai_provider', 'unknown')}\n")

            self.send_json_response(response)

        except Exception as e:
            self.send_error_response(500, str(e))

    def process_chat_message(self, message: str, context: dict) -> dict:
        """
        Process a chat message and generate an intelligent response.

        Uses real AI if available, falls back to rule-based system.
        """
        message_lower = message.lower()

        # Check for custom component keywords
        custom_keywords = ['custom', 'special', 'unique', 'my own', 'proprietary', 'unusual']
        needs_custom = any(kw in message_lower for kw in custom_keywords)

        # Try to use real AI connector
        ai = get_ai_connector()
        if ai and AI_CONNECTOR_AVAILABLE:
            ai_response = ai.chat(message, context)
            if ai_response.success:
                # Parse AI response and enhance with PCB Engine data
                return self._enhance_ai_response(message, ai_response.text, needs_custom)

        # Fall back to rule-based system

        # Detect components from message
        component_patterns = {
            'ESP32': ['esp32', 'esp-32'],
            'ESP32-S3': ['esp32-s3', 'esp32s3'],
            'STM32': ['stm32'],
            'Arduino': ['arduino', 'atmega'],
            'RP2040': ['rp2040', 'pico'],
            'BME280': ['bme280', 'temperature humidity'],
            'BMP280': ['bmp280'],
            'OLED': ['oled', 'ssd1306'],
            'TFT': ['tft', 'display', 'screen'],
            'LoRa': ['lora', 'lorawan', 'sx127'],
            'GPS': ['gps', 'gnss', 'neo-6'],
            'IMU': ['imu', 'accelerometer', 'mpu6050'],
            'Motor': ['motor', 'stepper'],
            'CAN': ['can bus', 'can interface'],
            'WiFi': ['wifi', 'wi-fi'],
            'Bluetooth': ['bluetooth', 'ble'],
            'Battery': ['battery', 'lipo'],
            'USB-C': ['usb-c', 'usb c', 'type-c'],
            'LED': ['led', 'neopixel', 'ws2812'],
        }

        detected = []
        for comp, patterns in component_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    detected.append(comp)
                    break

        # Generate response based on detected intent
        if needs_custom:
            return {
                'text': "I understand you want to use a custom component. Let me help you define its specifications.\n\nYou can use the **Custom Component Designer** to:\n- Define pin layout and positions\n- Set pad sizes and shapes\n- Specify electrical properties\n- Generate footprint data\n\nClick the button below to open the designer.",
                'needsCustomPart': True,
                'customPartMessage': "You mentioned wanting to use a custom or proprietary component.",
                'components': detected,
                'progress': None,
                'design': None
            }

        elif detected:
            # Process with actual engine if available
            if ENGINE_AVAILABLE:
                try:
                    ai = CircuitAI()
                    requirements = ai.parse_natural_language(message)
                    result = ai.generate_parts_db(requirements)

                    # Build design data
                    design = {
                        'boardSize': f"{requirements.board_size_mm[0]:.0f}x{requirements.board_size_mm[1]:.0f}mm",
                        'layers': requirements.layers,
                        'components': len(result.parts_db.get('parts', {})),
                        'nets': len(result.parts_db.get('nets', {})),
                        'bom': [
                            {'ref': item['refs'], 'value': item['value'], 'qty': item['quantity']}
                            for item in result.bom_preview[:10]
                        ]
                    }

                    response_text = f"I've analyzed your requirements and detected these components:\n\n"
                    response_text += '\n'.join(f"- **{c}**" for c in detected)
                    response_text += "\n\nI'm generating the design now. Here's what I'm doing:\n\n"
                    response_text += "1. Selecting optimal component variants\n"
                    response_text += "2. Creating power supply circuit\n"
                    response_text += "3. Generating net connections\n"
                    response_text += "4. Calculating board size\n\n"
                    response_text += "The design preview is now available in the right panel."

                    if result.suggestions:
                        response_text += "\n\n**Suggestions:**\n"
                        response_text += '\n'.join(f"- {s}" for s in result.suggestions)

                    return {
                        'text': response_text,
                        'components': detected,
                        'needsCustomPart': False,
                        'progress': [
                            {'name': 'Parse Requirements', 'status': 'done'},
                            {'name': 'Select Components', 'status': 'done'},
                            {'name': 'Generate Schematic', 'status': 'done'},
                            {'name': 'Create Layout', 'status': 'current'},
                        ],
                        'design': design
                    }

                except Exception as e:
                    # Fallback to simple response
                    pass

            # Fallback response without engine
            return {
                'text': f"I've detected these components in your description:\n\n" + '\n'.join(f"- **{c}**" for c in detected) + "\n\nI'll help you design a PCB with these components. Let me process the requirements...",
                'components': detected,
                'needsCustomPart': False,
                'progress': [
                    {'name': 'Parse Requirements', 'status': 'done'},
                    {'name': 'Select Components', 'status': 'current'},
                    {'name': 'Generate Schematic', 'status': 'pending'},
                    {'name': 'Create Layout', 'status': 'pending'},
                ],
                'design': None
            }

        else:
            # No components detected - ask for more info
            return {
                'text': "I'd be happy to help you design a PCB! Could you tell me more about:\n\n- **What microcontroller** do you want to use? (ESP32, STM32, Arduino, etc.)\n- **What sensors or peripherals** do you need?\n- **How will it be powered?** (USB, battery, external supply)\n- **Any size constraints?**\n\nFor example: \"ESP32 with temperature sensor and OLED display, USB powered\"",
                'components': [],
                'needsCustomPart': False,
                'progress': None,
                'design': None
            }

    def _enhance_ai_response(self, original_message: str, ai_text: str, needs_custom: bool) -> dict:
        """
        Enhance AI response with PCB Engine data.

        Takes the raw AI response and adds component detection,
        design preview, and progress indicators.
        """
        message_lower = original_message.lower()

        # Detect components from both message and AI response
        component_patterns = {
            'ESP32': ['esp32', 'esp-32'],
            'ESP32-S3': ['esp32-s3', 'esp32s3'],
            'STM32': ['stm32'],
            'Arduino': ['arduino', 'atmega'],
            'RP2040': ['rp2040', 'pico'],
            'BME280': ['bme280'],
            'BMP280': ['bmp280'],
            'OLED': ['oled', 'ssd1306'],
            'TFT': ['tft', 'st7789'],
            'LoRa': ['lora', 'sx127'],
            'GPS': ['gps', 'neo-6'],
            'Motor Driver': ['drv8', 'motor driver', 'a4988', 'tmc'],
            'WiFi': ['wifi', 'wi-fi'],
            'Bluetooth': ['bluetooth', 'ble'],
        }

        detected = []
        combined_text = (message_lower + ' ' + ai_text.lower())
        for comp, patterns in component_patterns.items():
            for pattern in patterns:
                if pattern in combined_text:
                    detected.append(comp)
                    break

        # Generate design AND save files if components detected
        design = None
        files_generated = []
        if detected and ENGINE_AVAILABLE:
            try:
                # 1. Circuit AI parses natural language and generates parts database
                circuit_ai = CircuitAI()
                requirements = circuit_ai.parse_natural_language(original_message)
                result = circuit_ai.generate_parts_db(requirements)

                project_name = requirements.project_name or 'pcb_design'

                # Create output directory - each design in its own folder
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                base_output_dir = Path('D:/Anas/tmp/output')
                output_dir = base_output_dir / f"{project_name}_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # 2. PCB Engine runs the full pipeline with all pistons
                engine_config = EngineConfig()
                engine_config.board_name = project_name
                engine_config.board_width = requirements.board_size_mm[0]
                engine_config.board_height = requirements.board_size_mm[1]
                engine_config.board_origin_x = 100
                engine_config.board_origin_y = 100
                engine_config.trace_width = 0.25
                engine_config.clearance = 0.15
                engine_config.via_diameter = 0.8
                engine_config.via_drill = 0.4

                engine = PCBEngine(engine_config)

                # Run the full engine pipeline (all 18 pistons)
                engine_result = engine.run(result.parts_db)

                # 3. Output Piston generates all manufacturing files
                output_config = OutputConfig(
                    output_dir=str(output_dir),
                    board_name=project_name,
                    board_width=requirements.board_size_mm[0],
                    board_height=requirements.board_size_mm[1],
                    generate_kicad=True,
                    generate_gerbers=True,
                    generate_drill=True,
                    generate_bom=True,
                    generate_pnp=True
                )
                output_piston = OutputPiston(output_config)
                output_result = output_piston.generate(
                    parts_db=result.parts_db,
                    placement=engine.state.placement if engine.state else {},
                    routes=engine.state.routes if engine.state else {},
                    vias=engine.state.vias if engine.state else [],
                    silkscreen=engine.state.silkscreen if engine.state else None
                )

                files_generated = output_result.files_generated

                # Also generate schematic and netlist from Circuit AI
                schematic = circuit_ai.export_schematic_kicad(result.parts_db, requirements)
                sch_path = output_dir / f"{project_name}.kicad_sch"
                with open(sch_path, 'w') as f:
                    f.write(schematic)
                files_generated.append(str(sch_path))

                netlist = generate_netlist_kicad(result.parts_db)
                net_path = output_dir / f"{project_name}.net"
                with open(net_path, 'w') as f:
                    f.write(netlist)
                files_generated.append(str(net_path))

                design = {
                    'boardSize': f"{requirements.board_size_mm[0]:.0f}x{requirements.board_size_mm[1]:.0f}mm",
                    'layers': requirements.layers,
                    'components': len(result.parts_db.get('parts', {})),
                    'nets': len(result.parts_db.get('nets', {})),
                    'bom': [
                        {'ref': item['refs'], 'value': item['value'], 'qty': item['quantity']}
                        for item in result.bom_preview[:10]
                    ],
                    'engine_status': getattr(engine_result, 'status', 'unknown')
                }

                # Add file generation info to AI text
                ai_text += f"\n\n**PCB Engine Generated Files:**\n"
                for f in files_generated:
                    ai_text += f"- `{Path(f).name}`\n"
                ai_text += f"\nFiles saved to: `{output_dir}`"

                if output_result.errors:
                    ai_text += f"\n\n*Warnings: {', '.join(output_result.errors)}*"

            except Exception as e:
                import traceback
                ai_text += f"\n\n*Note: Could not auto-generate files: {str(e)}*"
                print(f"Engine error: {traceback.format_exc()}")

        return {
            'text': ai_text,
            'components': detected,
            'needsCustomPart': needs_custom,
            'customPartMessage': "You mentioned a custom component." if needs_custom else None,
            'progress': [
                {'name': 'AI Analysis', 'status': 'done'},
                {'name': 'Component Selection', 'status': 'done' if detected else 'pending'},
                {'name': 'Generate Schematic', 'status': 'done' if files_generated else 'current' if detected else 'pending'},
                {'name': 'Create Layout', 'status': 'current' if files_generated else 'pending'},
            ] if detected else None,
            'design': design,
            'files_generated': files_generated,
            'ai_provider': get_ai_connector().config.provider.value if get_ai_connector() else 'rule-based'
        }


def run_server():
    """Run the web server"""
    with socketserver.TCPServer(("", PORT), PCBEngineHandler) as httpd:
        print(f"""
+====================================================================+
|                                                                    |
|   PCB ENGINE WEB SERVER                                            |
|                                                                    |
|   Server running at: http://localhost:{PORT}                        |
|                                                                    |
|   API Endpoints:                                                   |
|     POST /api/design  - Full design from description               |
|     POST /api/parse   - Parse requirements only                    |
|     POST /api/generate - Generate parts database                   |
|     POST /api/export  - Export design files                        |
|     POST /api/chat    - Chat interaction                           |
|     POST /api/custom  - Add custom component                       |
|                                                                    |
|   Press Ctrl+C to stop                                             |
|                                                                    |
+====================================================================+
""")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == '__main__':
    run_server()
