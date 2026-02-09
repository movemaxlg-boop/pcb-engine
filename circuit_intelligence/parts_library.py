"""
Comprehensive Parts Library with Indexing
==========================================

A searchable, indexed library of electronic components.
Optimized for fast lookups and comprehensive coverage.

**CRITICAL: ALL DATA MUST BE FROM VERIFIED MANUFACTURER DATASHEETS**

DO NOT add parts without:
1. Official manufacturer datasheet URL
2. Verified specifications (not estimates or guesses)
3. Proper attribution to data source

Data Sources for Expansion:
- DigiKey API (parametric search)
- Mouser API (component data)
- Octopart API (cross-reference)
- Manufacturer direct datasheets

Features:
- Fast lookup by part number, category, or property
- Cross-reference between manufacturers
- Complete specifications (electrical, thermal, mechanical)
- Design guidelines for each part
- Alternative parts suggestions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
import json
import os
from pathlib import Path
from collections import defaultdict


# =============================================================================
# PART SPECIFICATIONS - Data classes for component parameters
# =============================================================================

@dataclass
class ThermalSpec:
    """Thermal specifications - MUST be from datasheet."""
    theta_ja: float = 0.0       # °C/W (junction to ambient) - 0 = unknown
    theta_jc: float = 0.0       # °C/W (junction to case) - 0 = unknown
    max_temp: float = 125.0     # °C (max junction/operating temp)
    min_temp: float = -40.0     # °C (min operating temp)
    power_max: float = 0.0      # Max power dissipation in W


@dataclass
class ElectricalSpec:
    """Electrical specifications - MUST be from datasheet."""
    # Voltage (V)
    voltage_min: float = 0.0
    voltage_max: float = 0.0
    voltage_typical: float = 0.0

    # Current (A)
    current_min: float = 0.0
    current_max: float = 0.0
    current_typical: float = 0.0

    # Power (W)
    power_max: float = 0.0

    # For passives
    value: float = 0.0           # Capacitance (F), Resistance (Ω), Inductance (H)
    tolerance: float = 0.0       # ±% (e.g., 10 for ±10%)
    temp_coeff: str = ""         # X7R, X5R, NP0, etc. for caps

    # For ICs
    supply_voltage_range: Tuple[float, float] = (0.0, 0.0)  # (min, max)
    quiescent_current: float = 0.0   # mA
    output_current: float = 0.0      # A or mA depending on device


@dataclass
class MechanicalSpec:
    """Mechanical specifications - MUST be from datasheet."""
    package: str = ""            # Package name (e.g., "0805", "SOT-23-5")
    length: float = 0.0          # mm
    width: float = 0.0           # mm
    height: float = 0.0          # mm
    weight: float = 0.0          # grams
    pin_count: int = 0
    pin_pitch: float = 0.0       # mm


@dataclass
class Part:
    """
    Complete part specification.

    ALL fields should be populated from official datasheets only.
    If data is unknown, leave as default (0.0 or empty string).
    """
    # Identification - REQUIRED
    part_number: str
    manufacturer: str
    description: str
    category: str                # REGULATOR, CAPACITOR, RESISTOR, MCU, etc.

    # Sub-classification
    subcategory: str = ""        # BUCK, LDO, CERAMIC, etc.

    # Specifications - populate only what's verified
    thermal: ThermalSpec = field(default_factory=ThermalSpec)
    electrical: ElectricalSpec = field(default_factory=ElectricalSpec)
    mechanical: MechanicalSpec = field(default_factory=MechanicalSpec)

    # Cross-references
    alternatives: List[str] = field(default_factory=list)
    pin_compatible: List[str] = field(default_factory=list)

    # Design info
    typical_applications: List[str] = field(default_factory=list)
    design_notes: List[str] = field(default_factory=list)

    # KiCad integration
    schematic_symbol: str = ""
    footprint: str = ""

    # Lifecycle
    status: str = "ACTIVE"       # ACTIVE, NRND, OBSOLETE
    rohs_compliant: bool = True

    # Data source - REQUIRED for verification
    datasheet_url: str = ""      # Official datasheet link
    data_source: str = ""        # Where this data came from


# =============================================================================
# INDEX STRUCTURES FOR FAST LOOKUP
# =============================================================================

class PartsIndex:
    """
    Fast index for part lookups.

    Index types:
    - by_part_number: Direct part number lookup
    - by_category: Parts grouped by category
    - by_manufacturer: Parts by manufacturer
    - by_value: Passive components by value
    - by_package: Parts by package type
    """

    def __init__(self):
        self.by_part_number: Dict[str, Part] = {}
        self.by_category: Dict[str, List[str]] = defaultdict(list)
        self.by_manufacturer: Dict[str, List[str]] = defaultdict(list)
        self.by_value: Dict[str, List[str]] = defaultdict(list)
        self.by_package: Dict[str, List[str]] = defaultdict(list)

        # Full-text search index
        self.search_index: Dict[str, Set[str]] = defaultdict(set)

    def add_part(self, part: Part):
        """Add a part to all indexes."""
        pn = part.part_number.upper()

        # Primary index
        self.by_part_number[pn] = part

        # Category index
        self.by_category[part.category.upper()].append(pn)
        if part.subcategory:
            self.by_category[f"{part.category}/{part.subcategory}".upper()].append(pn)

        # Manufacturer index
        self.by_manufacturer[part.manufacturer.upper()].append(pn)

        # Value index (for passives)
        if part.electrical.value > 0:
            value_key = self._normalize_value(part.electrical.value, part.category)
            self.by_value[value_key].append(pn)

        # Package index
        if part.mechanical.package:
            self.by_package[part.mechanical.package.upper()].append(pn)

        # Build search index
        self._add_to_search_index(part)

    def _normalize_value(self, value: float, category: str) -> str:
        """Normalize value to standard notation."""
        category = category.upper()

        if category == 'CAPACITOR':
            if value >= 1e-6:
                return f"{value * 1e6:.1f}uF"
            elif value >= 1e-9:
                return f"{value * 1e9:.0f}nF"
            else:
                return f"{value * 1e12:.0f}pF"
        elif category == 'RESISTOR':
            if value >= 1e6:
                return f"{value / 1e6:.1f}M"
            elif value >= 1e3:
                return f"{value / 1e3:.1f}k"
            else:
                return f"{value:.1f}R"
        elif category == 'INDUCTOR':
            if value >= 1e-3:
                return f"{value * 1e3:.1f}mH"
            else:
                return f"{value * 1e6:.1f}uH"

        return str(value)

    def _add_to_search_index(self, part: Part):
        """Add part to full-text search index."""
        pn = part.part_number.upper()

        # Index part number fragments
        for i in range(len(pn)):
            for j in range(i + 2, min(i + 10, len(pn) + 1)):  # Limit fragment length
                self.search_index[pn[i:j]].add(pn)

        # Index description words
        for word in part.description.upper().split():
            if len(word) >= 2:
                self.search_index[word].add(pn)

    def search(self, query: str, limit: int = 20) -> List[Part]:
        """Search for parts matching query."""
        query_upper = query.upper()
        matches = set()

        # Exact part number match
        if query_upper in self.by_part_number:
            return [self.by_part_number[query_upper]]

        # Search index
        if query_upper in self.search_index:
            matches.update(self.search_index[query_upper])

        # Category match
        for cat, parts in self.by_category.items():
            if query_upper in cat:
                matches.update(parts[:limit])

        # Return parts
        result = [self.by_part_number[pn] for pn in list(matches)[:limit]
                  if pn in self.by_part_number]

        return result


# =============================================================================
# PARTS LIBRARY - Main interface
# =============================================================================

class PartsLibrary:
    """
    Comprehensive parts library with fast indexing.

    IMPORTANT: This library only contains VERIFIED data from official sources.

    Usage:
        lib = PartsLibrary()
        lib.load_defaults()

        # Lookup by part number
        part = lib.get('LM2596')

        # Search
        results = lib.search('3.3V LDO')

        # Find by category
        caps = lib.find_by_category('CAPACITOR')

        # Find alternatives
        alts = lib.find_alternatives('LM2596')
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.index = PartsIndex()
        self.data_dir = data_dir or str(Path(__file__).parent / 'parts_data')
        self._data_loaded = False

    def load_defaults(self):
        """Load verified parts from built-in database."""
        if self._data_loaded:
            return

        self._load_verified_regulators()
        self._load_verified_mcus()
        self._load_verified_capacitors()
        self._load_verified_diodes()
        self._load_verified_protection()

        self._data_loaded = True

    def load_from_json(self, filepath: str):
        """
        Load additional parts from a JSON file.

        JSON format:
        {
            "parts": [
                {
                    "part_number": "...",
                    "manufacturer": "...",
                    "datasheet_url": "...",  // REQUIRED
                    ...
                }
            ]
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        for part_data in data.get('parts', []):
            # Require datasheet URL for verification
            if not part_data.get('datasheet_url'):
                print(f"WARNING: Skipping {part_data.get('part_number')} - no datasheet URL")
                continue

            part = self._dict_to_part(part_data)
            self.index.add_part(part)

    def _dict_to_part(self, d: dict) -> Part:
        """Convert dictionary to Part object."""
        thermal = ThermalSpec(
            theta_ja=d.get('thermal', {}).get('theta_ja', 0.0),
            theta_jc=d.get('thermal', {}).get('theta_jc', 0.0),
            max_temp=d.get('thermal', {}).get('max_temp', 125.0),
        )

        electrical = ElectricalSpec(
            voltage_max=d.get('electrical', {}).get('voltage_max', 0.0),
            current_max=d.get('electrical', {}).get('current_max', 0.0),
            value=d.get('electrical', {}).get('value', 0.0),
            tolerance=d.get('electrical', {}).get('tolerance', 0.0),
            temp_coeff=d.get('electrical', {}).get('temp_coeff', ''),
            supply_voltage_range=tuple(d.get('electrical', {}).get('supply_voltage_range', (0.0, 0.0))),
            output_current=d.get('electrical', {}).get('output_current', 0.0),
        )

        mechanical = MechanicalSpec(
            package=d.get('mechanical', {}).get('package', ''),
            length=d.get('mechanical', {}).get('length', 0.0),
            width=d.get('mechanical', {}).get('width', 0.0),
            height=d.get('mechanical', {}).get('height', 0.0),
            pin_count=d.get('mechanical', {}).get('pin_count', 0),
            pin_pitch=d.get('mechanical', {}).get('pin_pitch', 0.0),
        )

        return Part(
            part_number=d['part_number'],
            manufacturer=d['manufacturer'],
            description=d.get('description', ''),
            category=d.get('category', ''),
            subcategory=d.get('subcategory', ''),
            thermal=thermal,
            electrical=electrical,
            mechanical=mechanical,
            alternatives=d.get('alternatives', []),
            pin_compatible=d.get('pin_compatible', []),
            typical_applications=d.get('typical_applications', []),
            design_notes=d.get('design_notes', []),
            footprint=d.get('footprint', ''),
            datasheet_url=d.get('datasheet_url', ''),
            data_source=d.get('data_source', ''),
        )

    # =========================================================================
    # VERIFIED PARTS - All data from official datasheets
    # =========================================================================

    def _load_verified_regulators(self):
        """
        Load voltage regulators with VERIFIED specifications.
        Sources: Texas Instruments, Diodes Inc, AMS datasheets
        """
        regulators = [
            # LM2596 - Source: TI datasheet SNVS124C (April 2013)
            Part(
                part_number='LM2596S-5.0/NOPB',
                manufacturer='Texas Instruments',
                description='SIMPLE SWITCHER 150kHz 3A Step-Down Voltage Regulator 5V',
                category='REGULATOR',
                subcategory='BUCK',
                thermal=ThermalSpec(
                    theta_ja=70.0,   # TO-263 with 0.5 in² copper (per SNVS124C Table 7.5)
                    theta_jc=5.0,    # Junction to case
                    max_temp=125.0,
                    power_max=2.0,   # Based on thermal limits
                ),
                electrical=ElectricalSpec(
                    supply_voltage_range=(4.5, 40.0),
                    output_current=3.0,
                    quiescent_current=5.0,  # mA typ
                ),
                mechanical=MechanicalSpec(
                    package='TO-263 (D2PAK)',
                    pin_count=5,
                ),
                alternatives=['LM2596S-ADJ', 'LM2596T-5.0', 'TPS54331'],
                typical_applications=['5V DC-DC step-down', 'USB power'],
                design_notes=[
                    'Fsw = 150kHz fixed',
                    'Inductor: 33µH (L = Vout×(Vin-Vout)/(ΔIL×f×Vin))',
                    'Output cap: 220µF electrolytic ESR < 100mΩ',
                    'Catch diode: 3A Schottky (SS34)',
                ],
                footprint='TO-263-5',
                datasheet_url='https://www.ti.com/lit/ds/symlink/lm2596.pdf',
                data_source='TI datasheet SNVS124C',
            ),

            # AMS1117-3.3 - Source: AMS datasheet ds1117.pdf
            Part(
                part_number='AMS1117-3.3',
                manufacturer='Advanced Monolithic Systems',
                description='1A Low Dropout Regulator 3.3V Fixed',
                category='REGULATOR',
                subcategory='LDO',
                thermal=ThermalSpec(
                    # NOTE: 15°C/W is θJC (junction-to-tab), NOT θJA!
                    theta_ja=90.0,   # SOT-223 θJA = 90°C/W (minimal copper)
                    theta_jc=15.0,   # Junction to tab
                    max_temp=125.0,
                ),
                electrical=ElectricalSpec(
                    supply_voltage_range=(4.5, 15.0),
                    output_current=1.0,  # Guaranteed to 0.8A
                    quiescent_current=5.0,  # mA typ
                ),
                mechanical=MechanicalSpec(
                    package='SOT-223',
                    pin_count=3,
                    pin_pitch=2.3,
                ),
                alternatives=['LM1117-3.3', 'LD1117S33CTR'],
                pin_compatible=['LM1117-3.3', 'LD1117S33CTR'],
                typical_applications=['3.3V MCU rail', 'USB powered'],
                design_notes=[
                    'Dropout: 1.3V max @ 1A',
                    'Input cap: 10µF tantalum or ceramic',
                    'Output cap: 22µF tantalum (ESR 0.3-22Ω)',
                    'Ceramic caps need series R for stability',
                ],
                footprint='SOT-223-3',
                datasheet_url='http://www.advanced-monolithic.com/pdf/ds1117.pdf',
                data_source='AMS datasheet ds1117',
            ),

            # AP2112K-3.3 - Source: Diodes Inc datasheet
            Part(
                part_number='AP2112K-3.3TRG1',
                manufacturer='Diodes Incorporated',
                description='600mA Low IQ LDO 3.3V with Enable',
                category='REGULATOR',
                subcategory='LDO',
                thermal=ThermalSpec(
                    theta_ja=180.0,  # SOT-25 package
                    max_temp=125.0,
                ),
                electrical=ElectricalSpec(
                    supply_voltage_range=(2.5, 6.0),
                    output_current=0.6,  # 600mA
                    quiescent_current=0.055,  # 55µA typ
                ),
                mechanical=MechanicalSpec(
                    package='SOT-25 (SOT-23-5)',
                    pin_count=5,
                    pin_pitch=0.95,
                ),
                alternatives=['MIC5219-3.3', 'XC6206P332MR'],
                typical_applications=['Battery devices', 'ESP32/ESP8266'],
                design_notes=[
                    'Ultra-low Iq: 55µA',
                    'Dropout: 250mV @ 600mA',
                    'Stable with ceramic caps',
                    'Input/Output cap: 1µF ceramic',
                ],
                footprint='SOT-23-5',
                datasheet_url='https://www.diodes.com/assets/Datasheets/AP2112.pdf',
                data_source='Diodes Inc AP2112 datasheet',
            ),
        ]

        for part in regulators:
            self.index.add_part(part)

    def _load_verified_mcus(self):
        """
        Load MCUs with VERIFIED specifications.
        Sources: Espressif, Microchip datasheets
        """
        mcus = [
            # ESP32-WROOM-32E - Source: Espressif datasheet v1.9
            Part(
                part_number='ESP32-WROOM-32E',
                manufacturer='Espressif Systems',
                description='WiFi+BT Module Dual-Core 240MHz',
                category='MCU',
                subcategory='WIFI_MODULE',
                thermal=ThermalSpec(
                    max_temp=105.0,  # Operating max
                    min_temp=-40.0,
                ),
                electrical=ElectricalSpec(
                    supply_voltage_range=(3.0, 3.6),
                    current_typical=0.080,  # 80mA active
                    current_max=0.500,      # 500mA peak TX
                ),
                mechanical=MechanicalSpec(
                    package='Module',
                    length=25.5,
                    width=18.0,
                    height=3.1,
                    pin_count=38,
                    pin_pitch=1.27,
                ),
                typical_applications=['IoT', 'WiFi/BT', 'Smart home'],
                design_notes=[
                    'VDD: 100nF ceramic + 10µF bulk',
                    'EN: 10kΩ pullup + 0.1µF RC',
                    'Keep antenna area clear (last 5.5mm)',
                    'TX 802.11n HT40: 190mA avg',
                ],
                footprint='ESP32-WROOM-32',
                datasheet_url='https://www.espressif.com/sites/default/files/documentation/esp32-wroom-32e_esp32-wroom-32ue_datasheet_en.pdf',
                data_source='Espressif datasheet v1.9',
            ),

            # ATmega328P - Source: Microchip DS40002061B
            Part(
                part_number='ATMEGA328P-AU',
                manufacturer='Microchip',
                description='8-bit AVR MCU 32KB Flash (Arduino Uno)',
                category='MCU',
                subcategory='8BIT',
                thermal=ThermalSpec(
                    theta_ja=73.0,  # TQFP-32
                    max_temp=85.0,  # Industrial grade
                ),
                electrical=ElectricalSpec(
                    supply_voltage_range=(1.8, 5.5),
                    current_typical=0.0002,  # 0.2mA @ 1MHz 1.8V
                    current_max=0.012,       # 12mA @ 16MHz 5V
                ),
                mechanical=MechanicalSpec(
                    package='TQFP-32',
                    length=9.0,
                    width=9.0,
                    pin_count=32,
                    pin_pitch=0.8,
                ),
                typical_applications=['Arduino', 'Embedded', 'Hobby'],
                design_notes=[
                    'VCC + AVCC: 100nF each',
                    'Crystal: 16MHz + 12-22pF caps',
                    'RESET: 10kΩ pullup + 100nF',
                    'Power-down: 0.1µA typ',
                ],
                footprint='TQFP-32',
                datasheet_url='https://ww1.microchip.com/downloads/en/DeviceDoc/ATmega48A-PA-88A-PA-168A-PA-328-P-DS-DS40002061B.pdf',
                data_source='Microchip DS40002061B',
            ),
        ]

        for part in mcus:
            self.index.add_part(part)

    def _load_verified_capacitors(self):
        """
        Load capacitors with VERIFIED specifications.
        Sources: Murata, Samsung datasheets
        """
        capacitors = [
            # 100nF 0805 X7R - Source: Murata GRM series
            Part(
                part_number='GRM21BR71H104KA01L',
                manufacturer='Murata',
                description='0.1µF 50V X7R 0805 MLCC',
                category='CAPACITOR',
                subcategory='CERAMIC',
                electrical=ElectricalSpec(
                    voltage_max=50.0,
                    value=100e-9,  # 100nF
                    tolerance=10.0,
                    temp_coeff='X7R',
                ),
                mechanical=MechanicalSpec(
                    package='0805',
                    length=2.0,
                    width=1.25,
                    height=1.25,
                ),
                typical_applications=['IC bypass', 'Decoupling'],
                design_notes=[
                    'X7R: ±15% over -55 to +125°C',
                    'DC bias derating: ~10% @ 25V',
                    'Place within 3mm of IC VCC',
                ],
                footprint='0805',
                datasheet_url='https://www.murata.com/en-us/products/productdetail?partno=GRM21BR71H104KA01',
                data_source='Murata GRM series datasheet',
            ),

            # 10µF 0805 X5R - Source: Murata GRM series
            Part(
                part_number='GRM21BR61E106KA73L',
                manufacturer='Murata',
                description='10µF 25V X5R 0805 MLCC',
                category='CAPACITOR',
                subcategory='CERAMIC',
                electrical=ElectricalSpec(
                    voltage_max=25.0,
                    value=10e-6,  # 10µF
                    tolerance=10.0,
                    temp_coeff='X5R',
                ),
                mechanical=MechanicalSpec(
                    package='0805',
                    length=2.0,
                    width=1.25,
                    height=1.25,
                ),
                typical_applications=['LDO I/O', 'Bulk decoupling'],
                design_notes=[
                    'X5R: ±15% over -55 to +85°C',
                    'DC bias derating: 40-50% @ 12V',
                    'Effective ~5µF at 12V DC bias',
                ],
                footprint='0805',
                datasheet_url='https://www.murata.com/en-us/products/productdetail?partno=GRM21BR61E106KA73',
                data_source='Murata GRM series datasheet',
            ),
        ]

        for part in capacitors:
            self.index.add_part(part)

    def _load_verified_diodes(self):
        """
        Load diodes with VERIFIED specifications.
        Sources: Vishay, ON Semi datasheets
        """
        diodes = [
            # SS34 Schottky - Source: Vishay 88751
            Part(
                part_number='SS34',
                manufacturer='Vishay',
                description='3A 40V Schottky Rectifier DO-214AB',
                category='DIODE',
                subcategory='SCHOTTKY',
                thermal=ThermalSpec(
                    theta_ja=75.0,
                    max_temp=150.0,  # Tj max
                ),
                electrical=ElectricalSpec(
                    voltage_max=40.0,  # VRRM
                    current_max=3.0,   # IF(AV)
                    # VF = 0.5V typ, 0.53V max @ 3A
                ),
                mechanical=MechanicalSpec(
                    package='DO-214AB (SMC)',
                    length=7.11,
                    width=6.22,
                    height=2.62,
                ),
                alternatives=['SS34A', 'SK34', 'B340A', 'MBR340'],
                typical_applications=['Buck catch diode', 'Freewheeling'],
                design_notes=[
                    'VF = 0.5V typ @ 3A',
                    'IFSM = 100A surge',
                    'IR = 0.5mA @ 40V',
                ],
                footprint='DO-214AB',
                datasheet_url='https://www.vishay.com/doc/?88751',
                data_source='Vishay datasheet 88751',
            ),
        ]

        for part in diodes:
            self.index.add_part(part)

    def _load_verified_protection(self):
        """
        Load protection devices with VERIFIED specifications.
        Sources: ST Microelectronics datasheets
        """
        protection = [
            # USBLC6-2 - Source: ST datasheet
            Part(
                part_number='USBLC6-2SC6',
                manufacturer='STMicroelectronics',
                description='USB ESD Protection Low Cap',
                category='PROTECTION',
                subcategory='ESD',
                electrical=ElectricalSpec(
                    voltage_max=5.5,  # VRWM
                ),
                mechanical=MechanicalSpec(
                    package='SOT-23-6L',
                    length=2.9,
                    width=1.6,
                    pin_count=6,
                    pin_pitch=0.95,
                ),
                typical_applications=['USB 2.0/3.0', 'HDMI'],
                design_notes=[
                    'Capacitance: 3.5pF per line',
                    'ESD: ±15kV air, ±8kV contact',
                    'Place within 10mm of connector',
                ],
                footprint='SOT-23-6',
                datasheet_url='https://www.st.com/resource/en/datasheet/usblc6-2.pdf',
                data_source='ST USBLC6-2 datasheet',
            ),
        ]

        for part in protection:
            self.index.add_part(part)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get(self, part_number: str) -> Optional[Part]:
        """Get part by part number."""
        return self.index.by_part_number.get(part_number.upper())

    def search(self, query: str, limit: int = 20) -> List[Part]:
        """Search for parts."""
        return self.index.search(query, limit)

    def find_by_category(self, category: str) -> List[Part]:
        """Find parts by category."""
        parts = []
        for pn in self.index.by_category.get(category.upper(), []):
            part = self.index.by_part_number.get(pn)
            if part:
                parts.append(part)
        return parts

    def find_alternatives(self, part_number: str) -> List[Part]:
        """Find alternative parts."""
        part = self.get(part_number)
        if not part:
            return []

        alternatives = []
        for alt_pn in part.alternatives + part.pin_compatible:
            alt = self.get(alt_pn)
            if alt:
                alternatives.append(alt)

        return alternatives

    def get_design_notes(self, part_number: str) -> List[str]:
        """Get design notes for a part."""
        part = self.get(part_number)
        return part.design_notes if part else []

    def get_stats(self) -> Dict:
        """Get library statistics."""
        return {
            'total_parts': len(self.index.by_part_number),
            'categories': dict([(k, len(v)) for k, v in self.index.by_category.items()]),
            'manufacturers': list(self.index.by_manufacturer.keys()),
            'verified': True,  # All parts have datasheet URLs
        }

    def export_to_json(self, filepath: str):
        """Export library to JSON for backup/sharing."""
        parts = []
        for pn, part in self.index.by_part_number.items():
            parts.append({
                'part_number': part.part_number,
                'manufacturer': part.manufacturer,
                'description': part.description,
                'category': part.category,
                'subcategory': part.subcategory,
                'datasheet_url': part.datasheet_url,
                'data_source': part.data_source,
                'thermal': {
                    'theta_ja': part.thermal.theta_ja,
                    'theta_jc': part.thermal.theta_jc,
                    'max_temp': part.thermal.max_temp,
                },
                'electrical': {
                    'voltage_max': part.electrical.voltage_max,
                    'current_max': part.electrical.current_max,
                    'value': part.electrical.value,
                    'tolerance': part.electrical.tolerance,
                    'supply_voltage_range': part.electrical.supply_voltage_range,
                    'output_current': part.electrical.output_current,
                },
                'mechanical': {
                    'package': part.mechanical.package,
                    'length': part.mechanical.length,
                    'width': part.mechanical.width,
                    'pin_count': part.mechanical.pin_count,
                },
                'design_notes': part.design_notes,
                'alternatives': part.alternatives,
                'footprint': part.footprint,
            })

        with open(filepath, 'w') as f:
            json.dump({'parts': parts, 'version': '1.0'}, f, indent=2)


# =============================================================================
# LIBRARY STATISTICS
# =============================================================================
# Current verified parts: 8
#
# Regulators: 3 (LM2596, AMS1117-3.3, AP2112K-3.3)
# MCUs: 2 (ESP32-WROOM-32E, ATMEGA328P)
# Capacitors: 2 (100nF X7R, 10µF X5R)
# Diodes: 1 (SS34 Schottky)
# Protection: 1 (USBLC6-2)
#
# TO EXPAND THE LIBRARY:
# 1. Use load_from_json() with parts verified from official datasheets
# 2. Each part MUST have datasheet_url for verification
# 3. DO NOT add parts with estimated or guessed specifications
#
