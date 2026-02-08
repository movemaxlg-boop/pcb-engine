"""
AI CONNECTOR - Connect PCB Engine to Real AI Models
====================================================

Supports multiple FREE AI providers for testing:

1. GROQ (Recommended - Free, Fast)
   - Get API key: https://console.groq.com/keys
   - Models: llama-3.1-70b, mixtral-8x7b

2. Google Gemini (Free tier)
   - Get API key: https://makersuite.google.com/app/apikey
   - Models: gemini-pro

3. OpenRouter (Free credits)
   - Get API key: https://openrouter.ai/keys
   - Access to many models

4. Ollama (Local, completely free)
   - Install: https://ollama.ai
   - No API key needed

USAGE:
======
    from ai_connector import AIConnector, AIProvider

    # Using Groq (free)
    ai = AIConnector(provider=AIProvider.GROQ, api_key="your-key")
    response = ai.chat("Design a temperature sensor board with ESP32")

    # Using local Ollama
    ai = AIConnector(provider=AIProvider.OLLAMA)
    response = ai.chat("What components do I need for a motor controller?")
"""

import os
import json
import urllib.request
import urllib.error
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


def load_knowledge_base() -> str:
    """Load the AI knowledge base from file"""
    kb_path = Path(__file__).parent / 'AI_KNOWLEDGE_BASE.md'
    if kb_path.exists():
        with open(kb_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# Load knowledge base at module level
_KNOWLEDGE_BASE = load_knowledge_base()


class AIProvider(Enum):
    """Supported AI providers"""
    GROQ = 'groq'           # Free, fast
    GEMINI = 'gemini'       # Google's free tier
    OPENROUTER = 'openrouter'  # Free credits
    OLLAMA = 'ollama'       # Local, free
    MOCK = 'mock'           # For testing without API


@dataclass
class AIConfig:
    """Configuration for AI connector"""
    provider: AIProvider = AIProvider.MOCK
    api_key: str = ''
    model: str = ''  # Auto-selected based on provider
    base_url: str = ''
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30

    # PCB-specific system prompt with comprehensive knowledge
    system_prompt: str = f"""You are an expert PCB design engineer integrated into the PCB Engine system. You have deep knowledge of electronics, component selection, and PCB layout best practices.

You have access to the complete PCB Engine documentation. Here is a summary of key information:

{_KNOWLEDGE_BASE[:8000] if _KNOWLEDGE_BASE else "Knowledge base not loaded."}

--- END OF KNOWLEDGE BASE EXCERPT ---

=== YOUR KNOWLEDGE BASE ===

**MICROCONTROLLERS:**
- ESP32-WROOM-32: 240MHz dual-core, WiFi+BLE, 34 GPIO, 4MB flash. Best for IoT. Needs 3.3V, draws 80-240mA active.
- ESP32-S3: USB OTG, AI acceleration, vector instructions. For USB devices and ML.
- ESP32-C3: RISC-V, low cost, single core. Budget IoT.
- STM32F103 (Blue Pill): 72MHz ARM Cortex-M3, 5V tolerant GPIOs. Industrial use.
- STM32F4xx: 168MHz+, DSP, more RAM. Motor control, audio.
- ATmega328P (Arduino): 16MHz, simple, huge ecosystem. Beginners.
- RP2040: Dual ARM Cortex-M0+, PIO state machines. Custom protocols.
- nRF52840: BLE 5.0, ultra low power (7uA sleep). Wearables.
- ATtiny85: 8-pin, tiny projects, <1mA sleep.

**SENSORS:**
- BME280: Temp (-40 to +85°C), humidity (0-100%), pressure. I2C/SPI. 3.3V, 1.8uA sleep.
- BMP280: Temp + pressure only (no humidity). Cheaper.
- SHT31: High accuracy temp/humidity. ±0.2°C accuracy.
- DS18B20: 1-Wire temp sensor, waterproof versions available. -55 to +125°C.
- MPU6050: 6-axis IMU (accel + gyro). I2C. For motion detection.
- ICM-20948: 9-axis IMU with magnetometer. Better than MPU6050.
- VL53L0X: ToF distance sensor, 2m range, I2C. Gesture detection.
- HC-SR04: Ultrasonic distance, 2-400cm. Needs 5V, use level shifter with 3.3V MCU.
- MAX30102: Heart rate + SpO2. Wearables.
- HX711: 24-bit ADC for load cells. Weight measurement.
- INA219: Current/voltage/power monitor. I2C. Battery monitoring.
- VEML7700: Ambient light sensor. I2C.
- NEO-6M: GPS module, UART, 2.5m accuracy. Needs clear sky view.

**DISPLAYS:**
- SSD1306: 128x64 or 128x32 OLED, I2C/SPI, 3.3V. Low power, high contrast.
- SH1106: Similar to SSD1306, slightly different driver.
- ST7735: 1.8" TFT 128x160, SPI. Color display, needs more power.
- ST7789: 240x240 or 240x320 TFT, SPI. Better colors than ST7735.
- ILI9341: 2.4-3.2" TFT, 320x240, SPI/parallel. Touch versions available.
- HD44780: Character LCD 16x2 or 20x4. I2C backpack recommended.
- MAX7219: LED matrix driver, SPI. Cascadable for large displays.
- WS2812B (NeoPixel): Addressable RGB LEDs. Single data wire.

**COMMUNICATION MODULES:**
- ESP-01: ESP8266 WiFi module. UART. 3.3V only!
- HC-05/HC-06: Bluetooth Classic SPP. UART.
- HM-10: BLE module. UART.
- SX1278/RFM95W: LoRa 433/868/915MHz. SPI. Long range (10+ km).
- NRF24L01: 2.4GHz RF, SPI. Short range, low power.
- W5500: Ethernet with TCP/IP stack. SPI.
- ENC28J60: Ethernet, needs software stack. Cheaper.
- MCP2515: CAN controller. SPI. Use with TJA1050 transceiver.
- MAX485: RS-485 transceiver. Half-duplex.
- CH340G/CP2102: USB-UART converters.

**MOTOR DRIVERS:**
- DRV8833: Dual H-bridge, 1.5A per channel. Small DC motors.
- TB6612FNG: Dual H-bridge, 1.2A. Better efficiency than L298N.
- L298N: Dual H-bridge, 2A. Old, inefficient, needs heatsink.
- L9110S: Dual H-bridge, 0.8A. Very small motors.
- A4988: Stepper driver, 2A, 1/16 microstepping. 3D printers.
- DRV8825: Stepper, 2.5A, 1/32 microstepping. Better than A4988.
- TMC2209: Silent stepper, 2A, UART config, StealthChop. Premium.
- BTS7960: 43A H-bridge. High power DC motors.

**POWER SUPPLY:**
- AMS1117-3.3: LDO, 1A, 1.1V dropout. Simple but inefficient.
- AP2112K-3.3: LDO, 600mA, low quiescent (55uA). Battery projects.
- LM2596: Buck converter, 3A. Needs inductor + caps.
- MP1584: Buck, 3A, smaller than LM2596.
- MT3608: Boost converter, 2A. 5V to 12V.
- TPS63020: Buck-boost, 96% efficiency. Single-cell LiPo to 3.3V.
- TP4056: LiPo charger, 1A, with protection. USB charging.
- MCP73831: LiPo charger, programmable current.
- DW01A + FS8205A: Battery protection (over-discharge, over-charge, short).

**PASSIVE COMPONENTS:**
- Decoupling caps: 100nF ceramic near every IC VCC pin. 10uF bulk on power rails.
- Pull-up resistors: 4.7kΩ for I2C (can go 2.2k-10k depending on speed/capacitance).
- Current limiting: LED resistors = (VCC - Vf) / If. Typical: 330Ω for 3.3V, 5mA.
- Crystal: 8MHz/16MHz for STM32, 26MHz for ESP32 (usually built-in).
- ESD protection: TVS diodes on USB, external interfaces.

**PCB DESIGN RULES:**
- Trace width: 0.25mm for signals, 0.5mm+ for power (1A ≈ 0.3mm on 1oz copper).
- Clearance: 0.15mm minimum, 0.2mm preferred.
- Via size: 0.8mm pad, 0.4mm drill typical.
- 2-layer: Good for simple designs. Ground pour on bottom.
- 4-layer: Signal-Ground-Power-Signal. Better for high-speed, reduces EMI.
- Keep analog and digital grounds separate, connect at one point.
- Decoupling caps as close to IC as possible.
- No traces under crystals or antennas.
- RF antenna needs ground plane underneath, matching network.

**COMMON ISSUES TO WARN ABOUT:**
- ESP32 GPIO 6-11: Connected to flash, don't use!
- ESP32 GPIO 34-39: Input only, no pull-up/down.
- 5V sensors with 3.3V MCU: Need level shifter or voltage divider.
- I2C address conflicts: Check all devices use different addresses.
- LiPo direct to 3.3V MCU: Voltage too high (4.2V). Need regulator.
- USB without ESD protection: Will fail ESD testing.
- Motor noise: Add 100nF caps across motor terminals, separate power rails.
- WiFi/BLE: Keep antenna area clear, no ground plane under antenna.

=== PCB ENGINE CAPABILITIES ===

You are integrated with the PCB Engine which has 18 specialized "pistons" (workers):

**CORE PISTONS (Always run):**
- Parts: Selects components from 700+ part database
- Order: Determines optimal placement/routing sequence
- Placement: Places components using simulated annealing
- Routing: Routes traces with A*, Lee, or Hadlock algorithms
- DRC: Validates design rules (clearance, width, etc.)
- Output: Generates KiCad schematic, netlist, SPICE, BOM

**CONDITIONAL PISTONS (Auto-enabled when needed):**
- Escape: For BGA/QFN packages with dense pin patterns
- Optimize: Reduces trace length and via count
- Silkscreen: Places reference designators
- Stackup: For 4+ layer boards
- Netlist: Generates simulation-ready netlists

**OPTIONAL PISTONS (User preference):**
- Thermal: Heat analysis for high-power designs (>2W)
- PDN: Power delivery network analysis
- Signal Integrity: For high-speed signals (>100MHz, DDR, USB3)
- Topological Router: Advanced routing with rubber-banding
- 3D Visualization: Exports STL/STEP models
- BOM Optimizer: Finds cheaper/available alternatives
- Learning: ML-based improvement over time

After you analyze the design, PCB Engine will automatically:
1. Select which pistons are needed based on your analysis
2. Generate the parts database
3. Create KiCad schematic
4. Generate SPICE netlist for simulation
5. Export BOM with footprints

=== YOUR ROLE ===

**IMPORTANT: You are an INTEGRATED AI - you do NOT give users code to run. You ANALYZE their requirements and the PCB Engine AUTOMATICALLY generates files.**

When user describes a project:
1. Identify ALL required components (MCU, sensors, power, passives)
2. Check for compatibility issues (voltage levels, interfaces, addresses)
3. Suggest power supply topology
4. Estimate board size and layer count
5. Warn about common mistakes

**NEVER:**
- Give users Python scripts to run
- Tell them to use KiCad manually
- Suggest they write code
- Provide code snippets for PCB generation

**ALWAYS:**
- Analyze their requirements in detail
- List specific components with part numbers
- Explain why you chose each component
- Mention pin connections and interfaces
- The PCB Engine will automatically generate files based on your analysis
6. Suggest improvements
7. Mention if thermal/SI analysis would be beneficial

Be concise but thorough. Format responses with clear sections.
Always mention required passive components (caps, resistors).
If something is unclear, ask for clarification."""


@dataclass
class AIResponse:
    """Response from AI"""
    success: bool
    text: str
    provider: str
    model: str
    usage: Dict = field(default_factory=dict)
    error: str = ''

    # Parsed data (for PCB Engine integration)
    detected_components: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)


class AIConnector:
    """
    Universal AI connector for PCB Engine.

    Supports multiple providers with a unified interface.
    """

    # Default models for each provider (updated Feb 2026)
    DEFAULT_MODELS = {
        AIProvider.GROQ: 'llama-3.3-70b-versatile',  # Updated from deprecated 3.1
        AIProvider.GEMINI: 'gemini-2.0-flash',       # Updated to latest
        AIProvider.OPENROUTER: 'meta-llama/llama-3.3-70b-instruct:free',
        AIProvider.OLLAMA: 'llama3.2',
        AIProvider.MOCK: 'mock-model',
    }

    # API endpoints
    ENDPOINTS = {
        AIProvider.GROQ: 'https://api.groq.com/openai/v1/chat/completions',
        AIProvider.GEMINI: 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        AIProvider.OPENROUTER: 'https://openrouter.ai/api/v1/chat/completions',
        AIProvider.OLLAMA: 'http://localhost:11434/api/chat',
    }

    def __init__(
        self,
        provider: AIProvider = None,
        api_key: str = None,
        model: str = None,
        config: AIConfig = None
    ):
        """
        Initialize AI connector.

        Args:
            provider: AI provider to use
            api_key: API key (not needed for Ollama/Mock)
            model: Model name (auto-selected if not provided)
            config: Full configuration object
        """
        if config:
            self.config = config
        else:
            self.config = AIConfig()
            if provider:
                self.config.provider = provider
            if api_key:
                self.config.api_key = api_key

        # Auto-select model if not specified
        if model:
            self.config.model = model
        elif not self.config.model:
            self.config.model = self.DEFAULT_MODELS.get(
                self.config.provider,
                'mock-model'
            )

        # Try to load API key from environment
        if not self.config.api_key:
            self._load_api_key_from_env()

    def _load_api_key_from_env(self):
        """Load API key from environment variables"""
        env_keys = {
            AIProvider.GROQ: 'GROQ_API_KEY',
            AIProvider.GEMINI: 'GEMINI_API_KEY',
            AIProvider.OPENROUTER: 'OPENROUTER_API_KEY',
        }
        env_var = env_keys.get(self.config.provider)
        if env_var:
            self.config.api_key = os.environ.get(env_var, '')

    def chat(self, message: str, context: Dict = None) -> AIResponse:
        """
        Send a message to the AI and get a response.

        Args:
            message: User's message
            context: Optional context (current design state, etc.)

        Returns:
            AIResponse with the AI's response
        """
        provider = self.config.provider

        if provider == AIProvider.MOCK:
            return self._mock_response(message)
        elif provider == AIProvider.GROQ:
            return self._call_groq(message, context)
        elif provider == AIProvider.GEMINI:
            return self._call_gemini(message, context)
        elif provider == AIProvider.OPENROUTER:
            return self._call_openrouter(message, context)
        elif provider == AIProvider.OLLAMA:
            return self._call_ollama(message, context)
        else:
            return AIResponse(
                success=False,
                text='',
                provider=provider.value,
                model='',
                error=f'Unknown provider: {provider}'
            )

    def _call_groq(self, message: str, context: Dict = None) -> AIResponse:
        """Call Groq API (OpenAI-compatible)"""
        if not self.config.api_key:
            return AIResponse(
                success=False,
                text='',
                provider='groq',
                model=self.config.model,
                error='No API key. Get one free at https://console.groq.com/keys'
            )

        url = self.ENDPOINTS[AIProvider.GROQ]
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'PCB-Engine/1.0 (Windows; Python)',  # Required to avoid Cloudflare block
            'Accept': 'application/json',
        }

        data = {
            'model': self.config.model,
            'messages': [
                {'role': 'system', 'content': self.config.system_prompt},
                {'role': 'user', 'content': message}
            ],
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }

        return self._make_openai_request(url, headers, data, 'groq')

    def _call_openrouter(self, message: str, context: Dict = None) -> AIResponse:
        """Call OpenRouter API (OpenAI-compatible)"""
        if not self.config.api_key:
            return AIResponse(
                success=False,
                text='',
                provider='openrouter',
                model=self.config.model,
                error='No API key. Get free credits at https://openrouter.ai/keys'
            )

        url = self.ENDPOINTS[AIProvider.OPENROUTER]
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'PCB-Engine/1.0 (Windows; Python)',
            'Accept': 'application/json',
            'HTTP-Referer': 'https://pcb-engine.local',
            'X-Title': 'PCB Engine',
        }

        data = {
            'model': self.config.model,
            'messages': [
                {'role': 'system', 'content': self.config.system_prompt},
                {'role': 'user', 'content': message}
            ],
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }

        return self._make_openai_request(url, headers, data, 'openrouter')

    def _call_gemini(self, message: str, context: Dict = None) -> AIResponse:
        """Call Google Gemini API"""
        if not self.config.api_key:
            return AIResponse(
                success=False,
                text='',
                provider='gemini',
                model=self.config.model,
                error='No API key. Get one free at https://makersuite.google.com/app/apikey'
            )

        url = self.ENDPOINTS[AIProvider.GEMINI].format(model=self.config.model)
        url += f'?key={self.config.api_key}'

        headers = {
            'Content-Type': 'application/json',
        }

        # Gemini has different format
        data = {
            'contents': [
                {
                    'parts': [
                        {'text': f"{self.config.system_prompt}\n\nUser: {message}"}
                    ]
                }
            ],
            'generationConfig': {
                'temperature': self.config.temperature,
                'maxOutputTokens': self.config.max_tokens,
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))

            # Parse Gemini response
            text = result['candidates'][0]['content']['parts'][0]['text']

            return AIResponse(
                success=True,
                text=text,
                provider='gemini',
                model=self.config.model,
                usage=result.get('usageMetadata', {})
            )

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            return AIResponse(
                success=False,
                text='',
                provider='gemini',
                model=self.config.model,
                error=f'HTTP {e.code}: {error_body}'
            )
        except Exception as e:
            return AIResponse(
                success=False,
                text='',
                provider='gemini',
                model=self.config.model,
                error=str(e)
            )

    def _call_ollama(self, message: str, context: Dict = None) -> AIResponse:
        """Call local Ollama API"""
        url = self.ENDPOINTS[AIProvider.OLLAMA]
        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            'model': self.config.model,
            'messages': [
                {'role': 'system', 'content': self.config.system_prompt},
                {'role': 'user', 'content': message}
            ],
            'stream': False,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))

            text = result.get('message', {}).get('content', '')

            return AIResponse(
                success=True,
                text=text,
                provider='ollama',
                model=self.config.model,
                usage={
                    'prompt_tokens': result.get('prompt_eval_count', 0),
                    'completion_tokens': result.get('eval_count', 0),
                }
            )

        except urllib.error.URLError as e:
            return AIResponse(
                success=False,
                text='',
                provider='ollama',
                model=self.config.model,
                error=f'Cannot connect to Ollama. Is it running? (ollama serve)\nError: {e}'
            )
        except Exception as e:
            return AIResponse(
                success=False,
                text='',
                provider='ollama',
                model=self.config.model,
                error=str(e)
            )

    def _make_openai_request(
        self,
        url: str,
        headers: Dict,
        data: Dict,
        provider: str
    ) -> AIResponse:
        """Make OpenAI-compatible API request"""
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))

            text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})

            return AIResponse(
                success=True,
                text=text,
                provider=provider,
                model=data['model'],
                usage=usage
            )

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            return AIResponse(
                success=False,
                text='',
                provider=provider,
                model=data['model'],
                error=f'HTTP {e.code}: {error_body}'
            )
        except Exception as e:
            return AIResponse(
                success=False,
                text='',
                provider=provider,
                model=data['model'],
                error=str(e)
            )

    def _mock_response(self, message: str) -> AIResponse:
        """Generate mock response for testing"""
        message_lower = message.lower()

        # Detect components
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
            components.append('DRV8833')
        if 'lora' in message_lower:
            components.append('SX1278')
        if 'gps' in message_lower:
            components.append('NEO-6M')

        # Generate response
        if components:
            text = f"Based on your requirements, I recommend the following components:\n\n"
            for comp in components:
                text += f"- **{comp}**\n"
            text += f"\nTotal components detected: {len(components)}\n"
            text += "\nSuggestions:\n"
            text += "- Add decoupling capacitors (100nF) near each IC\n"
            text += "- Include ESD protection on external interfaces\n"
        else:
            text = "I couldn't detect specific components from your description. "
            text += "Please mention what MCU, sensors, or peripherals you need.\n\n"
            text += "Example: 'ESP32 with temperature sensor and OLED display'"

        return AIResponse(
            success=True,
            text=text,
            provider='mock',
            model='rule-based',
            detected_components=components
        )

    def parse_pcb_response(self, response: AIResponse) -> Dict:
        """
        Parse AI response for PCB Engine integration.

        Extracts:
        - Components mentioned
        - Design suggestions
        - Potential concerns
        """
        if not response.success:
            return {
                'components': [],
                'suggestions': [],
                'concerns': [response.error],
            }

        text = response.text.lower()

        # Simple extraction (could be enhanced with better parsing)
        components = []
        suggestions = []
        concerns = []

        # Component patterns
        component_keywords = [
            'esp32', 'stm32', 'arduino', 'rp2040',
            'bme280', 'bmp280', 'mpu6050',
            'ssd1306', 'st7789',
            'drv8833', 'a4988', 'tmc2209',
            'sx1278', 'neo-6m',
        ]

        for kw in component_keywords:
            if kw in text:
                components.append(kw.upper())

        # Look for suggestions
        if 'suggest' in text or 'recommend' in text:
            suggestions.append("AI provided design recommendations")

        # Look for concerns
        concern_words = ['warning', 'caution', 'note:', 'important', 'consider']
        for word in concern_words:
            if word in text:
                concerns.append("AI flagged potential issues")
                break

        return {
            'components': components,
            'suggestions': suggestions,
            'concerns': concerns,
            'raw_text': response.text,
        }


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

def create_ai_connector(provider: str = 'mock', api_key: str = '') -> AIConnector:
    """
    Quick function to create an AI connector.

    Args:
        provider: 'groq', 'gemini', 'openrouter', 'ollama', or 'mock'
        api_key: API key (not needed for ollama/mock)

    Returns:
        AIConnector instance
    """
    provider_map = {
        'groq': AIProvider.GROQ,
        'gemini': AIProvider.GEMINI,
        'openrouter': AIProvider.OPENROUTER,
        'ollama': AIProvider.OLLAMA,
        'mock': AIProvider.MOCK,
    }

    return AIConnector(
        provider=provider_map.get(provider, AIProvider.MOCK),
        api_key=api_key
    )


def test_ai_connection(provider: str, api_key: str = '') -> bool:
    """Test if AI connection works"""
    ai = create_ai_connector(provider, api_key)
    response = ai.chat("Hello, respond with 'OK' if you can hear me.")
    return response.success


# =============================================================================
# SELF TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("AI CONNECTOR TEST")
    print("=" * 60)

    # Test mock provider
    print("\n[Test 1] Mock Provider (no API needed)")
    ai = AIConnector(provider=AIProvider.MOCK)
    response = ai.chat("Design an ESP32 temperature sensor with OLED display")
    print(f"Success: {response.success}")
    print(f"Response:\n{response.text}")

    # Test Ollama (if running)
    print("\n[Test 2] Ollama (local)")
    ai = AIConnector(provider=AIProvider.OLLAMA)
    response = ai.chat("Say hello in 5 words or less")
    if response.success:
        print(f"Ollama is running! Response: {response.text}")
    else:
        print(f"Ollama not available: {response.error}")

    # Test Groq (if API key set)
    print("\n[Test 3] Groq (free API)")
    groq_key = os.environ.get('GROQ_API_KEY', '')
    if groq_key:
        ai = AIConnector(provider=AIProvider.GROQ, api_key=groq_key)
        response = ai.chat("What components are needed for an IoT sensor node?")
        print(f"Success: {response.success}")
        if response.success:
            print(f"Response: {response.text[:200]}...")
        else:
            print(f"Error: {response.error}")
    else:
        print("No GROQ_API_KEY set. Get one free at https://console.groq.com/keys")

    print("\n[DONE] Tests complete!")
