"""
AI AGENT - Connected to PCB Engine
===================================

This is the REAL AI Agent that can:
1. Understand user requests
2. Call PCB Engine functions directly
3. Generate actual designs
4. Export real files

The AI is not just giving advice - it EXECUTES actions!
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Import PCB Engine components
try:
    from circuit_ai import (
        CircuitAI, EnhancedNLPParser, CircuitRequirements,
        quick_design, generate_netlist_kicad, generate_spice_netlist
    )
    from pcb_engine import PCBEngine, EngineConfig
    from piston_orchestrator import PistonOrchestrator, select_pistons_for_design
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PCB Engine not fully available: {e}")
    ENGINE_AVAILABLE = False

# Import AI connector
try:
    from ai_connector import AIConnector, AIProvider, create_ai_connector, AIConfig
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


@dataclass
class AgentAction:
    """An action the agent can take"""
    name: str
    description: str
    parameters: Dict = field(default_factory=dict)
    result: Any = None
    success: bool = False
    error: str = ''


@dataclass
class AgentResponse:
    """Complete response from the agent"""
    text: str                           # AI's response text
    actions_taken: List[AgentAction]    # What the agent did
    design_generated: bool = False      # Was a design created?
    files_created: List[str] = field(default_factory=list)
    parts_db: Dict = field(default_factory=dict)
    bom: List[Dict] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    pistons_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PCBAgent:
    """
    AI Agent that is CONNECTED to PCB Engine.

    This agent can:
    - Parse natural language into circuit requirements
    - Generate parts databases
    - Create schematics and netlists
    - Export files to disk
    - Run specific pistons

    Usage:
        agent = PCBAgent(api_key="your-groq-key")
        response = agent.process("Design an ESP32 board with BME280 sensor")

        print(response.text)  # AI explanation
        print(response.files_created)  # Actual files generated
    """

    def __init__(
        self,
        api_key: str = None,
        provider: str = 'groq',
        output_dir: str = './output'
    ):
        """
        Initialize the PCB Agent.

        Args:
            api_key: API key for AI provider (or set GROQ_API_KEY env var)
            provider: AI provider ('groq', 'gemini', 'ollama', 'mock')
            output_dir: Where to save generated files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize AI connector
        if api_key is None:
            api_key = os.environ.get('GROQ_API_KEY', '')

        if AI_AVAILABLE and api_key:
            self.ai = create_ai_connector(provider, api_key)
            print(f"AI Agent initialized with {provider}")
        else:
            self.ai = None
            print("AI Agent running without external AI (rule-based mode)")

        # Initialize PCB Engine components
        if ENGINE_AVAILABLE:
            self.circuit_ai = CircuitAI()
            self.orchestrator = PistonOrchestrator()
            print("PCB Engine connected")
        else:
            self.circuit_ai = None
            self.orchestrator = None
            print("PCB Engine not available")

        # State
        self.current_requirements = None
        self.current_parts_db = None
        self.conversation_history = []

    def process(self, user_message: str, auto_generate: bool = True) -> AgentResponse:
        """
        Process a user message and take appropriate actions.

        Args:
            user_message: What the user wants
            auto_generate: Automatically generate design if components detected

        Returns:
            AgentResponse with AI text and actions taken
        """
        actions = []
        files_created = []
        errors = []

        # Step 1: Get AI analysis
        ai_text = self._get_ai_response(user_message)

        # Step 2: Parse requirements from user message
        if ENGINE_AVAILABLE:
            parse_action = self._parse_requirements(user_message)
            actions.append(parse_action)

            if parse_action.success and auto_generate:
                # Step 3: Generate parts database
                generate_action = self._generate_parts()
                actions.append(generate_action)

                if generate_action.success:
                    # Step 4: Select pistons
                    piston_action = self._select_pistons()
                    actions.append(piston_action)

                    # Step 5: Export files
                    export_action = self._export_files()
                    actions.append(export_action)

                    if export_action.success:
                        files_created = export_action.result.get('files', [])

        # Collect errors
        for action in actions:
            if not action.success and action.error:
                errors.append(f"{action.name}: {action.error}")

        # Build response
        return AgentResponse(
            text=ai_text,
            actions_taken=actions,
            design_generated=any(a.name == 'generate_parts' and a.success for a in actions),
            files_created=files_created,
            parts_db=self.current_parts_db or {},
            bom=self._get_bom(),
            suggestions=self._get_suggestions(),
            pistons_used=self._get_pistons_used(actions),
            errors=errors
        )

    def _get_ai_response(self, message: str) -> str:
        """Get response from AI model"""
        if self.ai:
            response = self.ai.chat(message)
            if response.success:
                return response.text
            else:
                return f"AI Error: {response.error}\n\nFalling back to rule-based analysis..."

        # Fallback response
        return self._generate_fallback_response(message)

    def _generate_fallback_response(self, message: str) -> str:
        """Generate response without AI"""
        message_lower = message.lower()

        components = []
        if 'esp32' in message_lower:
            components.append('ESP32-WROOM-32')
        if 'stm32' in message_lower:
            components.append('STM32F103C8T6')
        if 'temperature' in message_lower or 'bme280' in message_lower:
            components.append('BME280')
        if 'oled' in message_lower or 'display' in message_lower:
            components.append('SSD1306 OLED')
        if 'motor' in message_lower:
            components.append('DRV8833 Motor Driver')
        if 'lora' in message_lower:
            components.append('SX1278 LoRa Module')

        if components:
            text = "**Detected Components:**\n"
            for c in components:
                text += f"- {c}\n"
            text += "\n**Generating design...**"
            return text
        else:
            return "Please describe your circuit. What MCU, sensors, and peripherals do you need?"

    def _parse_requirements(self, description: str) -> AgentAction:
        """Parse natural language into requirements"""
        action = AgentAction(
            name='parse_requirements',
            description='Parse user description into circuit requirements'
        )

        try:
            self.current_requirements = self.circuit_ai.parse_natural_language(description)
            action.success = True
            action.result = {
                'project_name': self.current_requirements.project_name,
                'mcu': self.current_requirements.mcu_family.value,
                'power': self.current_requirements.input_power.value,
                'blocks': len(self.current_requirements.blocks),
                'board_size': self.current_requirements.board_size_mm,
                'layers': self.current_requirements.layers
            }
        except Exception as e:
            action.success = False
            action.error = str(e)

        return action

    def _generate_parts(self) -> AgentAction:
        """Generate parts database from requirements"""
        action = AgentAction(
            name='generate_parts',
            description='Generate component database with footprints and nets'
        )

        if not self.current_requirements:
            action.success = False
            action.error = "No requirements parsed"
            return action

        try:
            result = self.circuit_ai.generate_parts_db(self.current_requirements)
            self.current_parts_db = result.parts_db

            action.success = True
            action.result = {
                'parts_count': len(result.parts_db.get('parts', {})),
                'nets_count': len(result.parts_db.get('nets', {})),
                'bom_items': len(result.bom_preview),
                'suggestions': result.suggestions
            }
        except Exception as e:
            action.success = False
            action.error = str(e)

        return action

    def _select_pistons(self) -> AgentAction:
        """Select which pistons to run"""
        action = AgentAction(
            name='select_pistons',
            description='Determine which of 18 pistons are needed'
        )

        if not self.current_parts_db:
            action.success = False
            action.error = "No parts database"
            return action

        try:
            selection = self.orchestrator.select_pistons(
                parts_db=self.current_parts_db,
                requirements=self.current_requirements
            )

            action.success = True
            action.result = {
                'required': selection.required_pistons,
                'optional': selection.optional_pistons,
                'skipped': selection.skipped_pistons,
                'execution_order': selection.execution_order,
                'reasons': selection.selection_reasons
            }
        except Exception as e:
            action.success = False
            action.error = str(e)

        return action

    def _export_files(self) -> AgentAction:
        """Export design files"""
        action = AgentAction(
            name='export_files',
            description='Generate KiCad, SPICE, and BOM files'
        )

        if not self.current_parts_db or not self.current_requirements:
            action.success = False
            action.error = "No design to export"
            return action

        try:
            files = []
            project_name = self.current_requirements.project_name

            # Export KiCad schematic
            schematic = self.circuit_ai.export_schematic_kicad(
                self.current_parts_db,
                self.current_requirements
            )
            sch_path = self.output_dir / f"{project_name}.kicad_sch"
            with open(sch_path, 'w') as f:
                f.write(schematic)
            files.append(str(sch_path))

            # Export KiCad netlist
            netlist = generate_netlist_kicad(self.current_parts_db)
            net_path = self.output_dir / f"{project_name}.net"
            with open(net_path, 'w') as f:
                f.write(netlist)
            files.append(str(net_path))

            # Export SPICE netlist
            spice = generate_spice_netlist(self.current_parts_db, project_name)
            spice_path = self.output_dir / f"{project_name}.cir"
            with open(spice_path, 'w') as f:
                f.write(spice)
            files.append(str(spice_path))

            # Export BOM CSV
            bom_lines = ['Reference,Value,Footprint,Quantity']
            for ref, part in self.current_parts_db.get('parts', {}).items():
                bom_lines.append(f"{ref},{part.get('value', '')},{part.get('footprint', '')},1")
            bom_path = self.output_dir / f"{project_name}_bom.csv"
            with open(bom_path, 'w') as f:
                f.write('\n'.join(bom_lines))
            files.append(str(bom_path))

            # Export parts JSON
            json_path = self.output_dir / f"{project_name}_parts.json"
            with open(json_path, 'w') as f:
                json.dump(self.current_parts_db, f, indent=2)
            files.append(str(json_path))

            action.success = True
            action.result = {'files': files}

        except Exception as e:
            action.success = False
            action.error = str(e)

        return action

    def _get_bom(self) -> List[Dict]:
        """Get Bill of Materials"""
        if not self.current_parts_db:
            return []

        bom = []
        for ref, part in self.current_parts_db.get('parts', {}).items():
            bom.append({
                'reference': ref,
                'value': part.get('value', ''),
                'footprint': part.get('footprint', ''),
                'description': part.get('description', '')
            })
        return bom

    def _get_suggestions(self) -> List[str]:
        """Get design suggestions"""
        # This would come from the CircuitAI result
        return []

    def _get_pistons_used(self, actions: List[AgentAction]) -> List[str]:
        """Get list of pistons that were selected"""
        for action in actions:
            if action.name == 'select_pistons' and action.success:
                return action.result.get('execution_order', [])
        return []

    # === DIRECT ACTION METHODS ===

    def design(self, description: str) -> Dict:
        """
        Design a PCB from description.

        Args:
            description: Natural language description

        Returns:
            Dict with design results
        """
        response = self.process(description, auto_generate=True)
        return {
            'success': response.design_generated,
            'files': response.files_created,
            'parts_count': len(response.parts_db.get('parts', {})),
            'bom': response.bom,
            'pistons': response.pistons_used,
            'errors': response.errors
        }

    def get_parts_for(self, description: str) -> Dict:
        """Quick parts lookup without full design"""
        if ENGINE_AVAILABLE:
            result = quick_design(description)
            return {
                'parts': result.parts_db.get('parts', {}),
                'bom': result.bom_preview,
                'suggestions': result.suggestions
            }
        return {'error': 'Engine not available'}

    def export_current_design(self, format: str = 'all') -> List[str]:
        """Export current design to files"""
        if not self.current_parts_db:
            return []

        action = self._export_files()
        if action.success:
            return action.result.get('files', [])
        return []

    def list_available_components(self, category: str = None) -> List[str]:
        """List available components in database"""
        if not ENGINE_AVAILABLE:
            return []

        from circuit_ai import (
            MCU_DATABASE, SENSOR_DATABASE, DISPLAY_DATABASE,
            COMMUNICATION_DATABASE, MOTOR_DRIVER_DATABASE, POWER_IC_DATABASE
        )

        databases = {
            'mcu': MCU_DATABASE,
            'sensor': SENSOR_DATABASE,
            'display': DISPLAY_DATABASE,
            'communication': COMMUNICATION_DATABASE,
            'motor': MOTOR_DRIVER_DATABASE,
            'power': POWER_IC_DATABASE
        }

        if category and category.lower() in databases:
            return list(databases[category.lower()].keys())

        # Return all
        all_components = []
        for db in databases.values():
            all_components.extend(db.keys())
        return all_components

    def get_component_info(self, component_name: str) -> Dict:
        """Get detailed info about a component"""
        if not ENGINE_AVAILABLE:
            return {}

        from circuit_ai import (
            MCU_DATABASE, SENSOR_DATABASE, DISPLAY_DATABASE,
            COMMUNICATION_DATABASE, MOTOR_DRIVER_DATABASE, POWER_IC_DATABASE
        )

        for db in [MCU_DATABASE, SENSOR_DATABASE, DISPLAY_DATABASE,
                   COMMUNICATION_DATABASE, MOTOR_DRIVER_DATABASE, POWER_IC_DATABASE]:
            if component_name in db:
                return db[component_name]

        return {}


# === CONVENIENCE FUNCTIONS ===

def create_agent(api_key: str = None) -> PCBAgent:
    """Create a PCB Agent"""
    return PCBAgent(api_key=api_key)


def quick_agent_design(description: str, api_key: str = None) -> Dict:
    """
    One-liner to design a PCB.

    Example:
        result = quick_agent_design("ESP32 with temperature sensor and OLED")
        print(result['files'])
    """
    agent = PCBAgent(api_key=api_key)
    return agent.design(description)


# === SELF TEST ===

if __name__ == '__main__':
    print("=" * 60)
    print("PCB AGENT TEST")
    print("=" * 60)

    # Load API key from .env
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    agent = PCBAgent()

    print("\n[Test 1] Design ESP32 sensor board")
    response = agent.process("ESP32 with BME280 temperature sensor and OLED display, USB powered")

    print(f"\nAI Response:\n{response.text[:500]}...")
    print(f"\nDesign Generated: {response.design_generated}")
    print(f"Files Created: {response.files_created}")
    print(f"Parts Count: {len(response.parts_db.get('parts', {}))}")
    print(f"Pistons Used: {response.pistons_used}")

    if response.errors:
        print(f"Errors: {response.errors}")

    print("\n[Test 2] List available MCUs")
    mcus = agent.list_available_components('mcu')
    print(f"Available MCUs: {mcus[:5]}...")

    print("\n[DONE]")
