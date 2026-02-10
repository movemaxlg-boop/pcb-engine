"""
PCB Engine - Parts Piston
==========================

A comprehensive piston (sub-engine) for ALL component-related operations in PCB design.

This is the foundation piston that handles:
1. Component Selection - Choose optimal parts from libraries
2. Component Sourcing - KiCad, custom, manufacturer libraries
3. Naming & Designators - Reference designator assignment (R1, C1, U1...)
4. Parameters - Values, tolerances, ratings
5. Footprint Resolution - Match symbol to physical footprint
6. Courtyard Calculation - Component boundaries for placement
7. User Input Parsing - Natural language to component spec
8. Electrical Requirements - Current, voltage, power constraints
9. Thermal Requirements - Power dissipation, thermal resistance
10. Pin Information - Net connectivity, function, electrical type
11. 3D Model Association - Link to STEP/WRL models
12. BOM Data - Manufacturer, MPN, supplier, price

COMPONENT SOURCING HIERARCHY:
1. User Custom Library (highest priority)
2. Project Local Library
3. KiCad Official Library
4. Manufacturer Libraries (TI, Analog, etc.)
5. Community Libraries
6. Auto-generated (parametric)

ELECTRICAL REQUIREMENTS FOR ROUTING:
- Net class assignment (power, signal, high-speed, analog)
- Trace width requirements (based on current)
- Clearance requirements (based on voltage)
- Via requirements (thermal, current)
- Length matching groups
- Differential pair identification

ELECTRICAL REQUIREMENTS FOR PLACEMENT:
- Keep-out zones (high voltage isolation)
- Thermal zones (heat-generating components)
- Grouping requirements (functional blocks)
- Orientation constraints (connectors, displays)
- Height restrictions (enclosure fit)

LIBRARY MANAGEMENT:
===================
The Parts Piston has its own library system with multi-source access:

1. INTERNAL LIBRARY (built-in):
   - Common passives (R, C, L) with parametric generation
   - Standard packages (0402, 0603, 0805, etc.)
   - Generic connectors, test points, mounting holes

2. KICAD LIBRARIES:
   - Official KiCad symbol libraries
   - Official KiCad footprint libraries
   - KiCad 3D model libraries

3. MANUFACTURER LIBRARIES:
   - Texas Instruments (ti.kicad.sym)
   - Analog Devices
   - STMicroelectronics
   - Espressif (ESP32, ESP8266)
   - Nordic Semiconductor
   - Microchip/Atmel

4. COMMUNITY LIBRARIES:
   - DigiKey KiCad Library
   - SnapEDA
   - Ultra Librarian
   - ComponentSearchEngine

5. USER CUSTOM LIBRARIES:
   - Project-local library
   - User global library

Library File Formats Supported:
- KiCad 6/7/8 .kicad_sym (symbols)
- KiCad 6/7/8 .kicad_mod (footprints)
- KiCad legacy .lib/.dcm (symbols)
- KiCad legacy .mod (footprints)
- STEP/WRL (3D models)

Research References:
- IPC-7351B: Generic Requirements for Surface Mount Design
- IPC-2221: Generic Standard on Printed Board Design
- Saturn PCB Toolkit: Trace width calculations
- Thermal design guidelines from TI, Analog Devices
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import math
import re
import os
import json
from collections import defaultdict


# =============================================================================
# LIBRARY SYSTEM
# =============================================================================

class LibraryType(Enum):
    """Types of component libraries"""
    INTERNAL = 'internal'           # Built-in parametric library
    KICAD_SYMBOL = 'kicad_sym'      # KiCad .kicad_sym files
    KICAD_FOOTPRINT = 'kicad_mod'   # KiCad .kicad_mod files
    KICAD_LEGACY_SYM = 'lib'        # Legacy .lib files
    KICAD_LEGACY_FP = 'mod'         # Legacy .mod files
    USER_CUSTOM = 'user'            # User custom library
    MANUFACTURER = 'mfr'            # Manufacturer library


@dataclass
class LibraryInfo:
    """Information about a component library"""
    name: str
    lib_type: LibraryType
    path: str
    description: str = ''
    version: str = ''
    priority: int = 0               # Higher = search first
    enabled: bool = True
    manufacturer: str = ''          # For manufacturer libs


@dataclass
class LibraryComponent:
    """A component from a library"""
    lib_id: str                     # Library:ComponentName
    name: str
    description: str = ''
    keywords: List[str] = field(default_factory=list)
    footprint_filters: List[str] = field(default_factory=list)
    datasheet: str = ''
    symbol_data: Dict = field(default_factory=dict)
    footprint_data: Dict = field(default_factory=dict)

    # Parsed pin/pad info
    pins: List[Dict] = field(default_factory=list)
    pads: List[Dict] = field(default_factory=list)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ComponentSource(Enum):
    """Where the component definition comes from"""
    USER_CUSTOM = 'user_custom'           # User's custom library
    PROJECT_LOCAL = 'project_local'       # Project-specific library
    KICAD_OFFICIAL = 'kicad_official'     # KiCad official symbols/footprints
    MANUFACTURER = 'manufacturer'         # Manufacturer library (TI, AD, etc.)
    COMMUNITY = 'community'               # Community/third-party libraries
    AUTO_GENERATED = 'auto_generated'     # Parametrically generated
    UNKNOWN = 'unknown'


class NetClass(Enum):
    """Classification of nets for routing"""
    POWER = 'power'                 # Power supply nets (VCC, VDD, 5V, etc.)
    GROUND = 'ground'               # Ground nets (GND, AGND, DGND, etc.)
    HIGH_SPEED = 'high_speed'       # High-speed digital (>50MHz)
    DIFFERENTIAL = 'differential'   # Differential pairs (USB, LVDS, etc.)
    ANALOG = 'analog'               # Sensitive analog signals
    RF = 'rf'                       # RF/microwave signals
    HIGH_CURRENT = 'high_current'   # High current (motor, LED, etc.)
    HIGH_VOLTAGE = 'high_voltage'   # High voltage isolation required
    SIGNAL = 'signal'               # Standard digital signals
    LOW_PRIORITY = 'low_priority'   # Non-critical connections


class PinType(Enum):
    """Electrical type of a pin"""
    INPUT = 'input'
    OUTPUT = 'output'
    BIDIRECTIONAL = 'bidirectional'
    TRISTATE = 'tristate'
    PASSIVE = 'passive'
    POWER_IN = 'power_in'
    POWER_OUT = 'power_out'
    OPEN_COLLECTOR = 'open_collector'
    OPEN_EMITTER = 'open_emitter'
    NO_CONNECT = 'no_connect'
    UNSPECIFIED = 'unspecified'


class PadShape(Enum):
    """Shape of component pads"""
    CIRCLE = 'circle'
    RECT = 'rect'
    ROUNDRECT = 'roundrect'
    OVAL = 'oval'
    TRAPEZOID = 'trapezoid'
    CUSTOM = 'custom'


class MountType(Enum):
    """Component mounting type"""
    SMD = 'smd'                     # Surface mount
    THT = 'tht'                     # Through-hole
    VIRTUAL = 'virtual'             # Virtual (no footprint)


# =============================================================================
# IPC-2221 TRACE WIDTH CALCULATION CONSTANTS
# =============================================================================

# IPC-2221 constants for trace width calculation
IPC2221_INTERNAL_K = 0.024   # Internal layer constant
IPC2221_EXTERNAL_K = 0.048   # External layer constant
IPC2221_B = 0.44             # Exponent for cross-section
IPC2221_C = 0.725            # Exponent for temperature rise


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PadInfo:
    """Information about a component pad"""
    number: str
    name: str = ''
    shape: PadShape = PadShape.RECT
    size_x: float = 1.0          # mm
    size_y: float = 1.0          # mm
    offset_x: float = 0.0        # mm from component center
    offset_y: float = 0.0        # mm from component center
    drill: float = 0.0           # mm, 0 for SMD
    layers: List[str] = field(default_factory=lambda: ['F.Cu', 'F.Paste', 'F.Mask'])
    thermal_relief: bool = True
    zone_connect: int = 1        # 0=none, 1=thermal, 2=solid


@dataclass
class PinInfo:
    """Complete information about a component pin"""
    number: str                  # Physical pin number
    name: str = ''               # Pin name (e.g., "VCC", "GPIO1")
    electrical_type: PinType = PinType.PASSIVE
    net: str = ''                # Connected net name

    # Physical properties
    pad: Optional[PadInfo] = None

    # Electrical properties
    max_current: float = 0.5     # Amps
    max_voltage: float = 50.0    # Volts
    io_standard: str = ''        # e.g., "LVCMOS33", "LVDS"

    # Functional properties
    function: str = ''           # e.g., "CLOCK", "DATA", "RESET"
    is_critical: bool = False
    length_match_group: str = '' # For matched-length routing


@dataclass
class CourtyardInfo:
    """Component courtyard (keep-out zone)"""
    width: float                 # Total width in mm
    height: float                # Total height in mm
    offset_x: float = 0.0        # Offset from center
    offset_y: float = 0.0
    clearance: float = 0.25      # IPC courtyard excess (CY)
    layer: str = 'F.CrtYd'


@dataclass
class ThermalInfo:
    """Thermal properties of a component"""
    power_dissipation: float = 0.0       # Watts
    theta_ja: float = 0.0                # Junction-to-ambient thermal resistance (°C/W)
    theta_jc: float = 0.0                # Junction-to-case thermal resistance (°C/W)
    max_junction_temp: float = 125.0     # Maximum junction temperature (°C)
    requires_heatsink: bool = False
    thermal_pad: bool = False            # Has exposed thermal pad
    thermal_via_count: int = 0           # Recommended thermal vias


@dataclass
class ElectricalRequirements:
    """Electrical requirements for routing and placement"""
    # Current requirements
    max_current: float = 0.5             # Amps
    rms_current: float = 0.0             # RMS current for AC

    # Voltage requirements
    max_voltage: float = 50.0            # Volts
    working_voltage: float = 0.0         # Normal operating voltage

    # Routing requirements
    net_class: NetClass = NetClass.SIGNAL
    min_trace_width: float = 0.0         # Calculated based on current
    min_clearance: float = 0.0           # Calculated based on voltage

    # Special requirements
    is_differential: bool = False
    diff_pair_name: str = ''             # Name of differential pair
    length_match_group: str = ''         # Length matching group name
    max_length: float = 0.0              # Maximum trace length (mm)
    max_vias: int = -1                   # Maximum via count (-1 = no limit)

    # Placement requirements
    keep_out_radius: float = 0.0         # Keep other components away (mm)
    height_restriction: float = 0.0      # Maximum component height above (mm)
    requires_isolation: bool = False     # High voltage isolation


@dataclass
class SourceInfo:
    """Component sourcing information"""
    source: ComponentSource = ComponentSource.UNKNOWN
    library_path: str = ''
    symbol_name: str = ''
    footprint_name: str = ''

    # Manufacturer info
    manufacturer: str = ''
    mpn: str = ''                        # Manufacturer Part Number

    # Supplier info
    supplier: str = ''                   # Digi-Key, Mouser, etc.
    supplier_pn: str = ''                # Supplier part number
    unit_price: float = 0.0
    moq: int = 1                         # Minimum order quantity
    lead_time_days: int = 0

    # 3D model
    model_3d_path: str = ''
    model_3d_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    model_3d_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Component:
    """Complete component definition"""
    # Identity
    ref: str                             # Reference designator (R1, U1, etc.)
    value: str                           # Component value ("10k", "100nF", "ESP32")

    # Physical
    footprint: str                       # Footprint name
    mount_type: MountType = MountType.SMD
    pins: List[PinInfo] = field(default_factory=list)
    courtyard: Optional[CourtyardInfo] = None

    # Electrical
    electrical: ElectricalRequirements = field(default_factory=ElectricalRequirements)
    thermal: ThermalInfo = field(default_factory=ThermalInfo)

    # Sourcing
    source: SourceInfo = field(default_factory=SourceInfo)

    # User notes
    description: str = ''
    datasheet: str = ''
    notes: str = ''

    # Placement hints
    placement_group: str = ''            # Functional group (e.g., "power", "mcu")
    fixed_position: Optional[Tuple[float, float]] = None  # Fixed placement
    fixed_rotation: Optional[float] = None
    prefer_top: bool = True              # Prefer top layer

    @property
    def pin_count(self) -> int:
        return len(self.pins)

    @property
    def is_power_component(self) -> bool:
        """Check if this is a power-related component"""
        val = self.value.upper()
        ref = self.ref.upper()
        return (ref.startswith('U') and any(x in val for x in ['REG', 'LDO', 'DCDC', 'BUCK', 'BOOST'])) or \
               ref.startswith('L') or \
               (ref.startswith('D') and 'LED' not in val)


@dataclass
class PartsConfig:
    """Configuration for the parts piston"""
    # Library paths
    kicad_symbol_path: str = ''
    kicad_footprint_path: str = ''
    custom_library_path: str = ''

    # Sourcing preferences
    source_priority: List[ComponentSource] = field(default_factory=lambda: [
        ComponentSource.USER_CUSTOM,
        ComponentSource.PROJECT_LOCAL,
        ComponentSource.KICAD_OFFICIAL,
        ComponentSource.MANUFACTURER,
        ComponentSource.COMMUNITY
    ])

    # Naming rules
    ref_designator_rules: Dict[str, str] = field(default_factory=lambda: {
        'resistor': 'R',
        'capacitor': 'C',
        'inductor': 'L',
        'diode': 'D',
        'led': 'D',
        'transistor': 'Q',
        'ic': 'U',
        'mcu': 'U',
        'connector': 'J',
        'switch': 'SW',
        'fuse': 'F',
        'crystal': 'Y',
        'relay': 'K',
        'transformer': 'T',
        'test_point': 'TP',
    })

    # Electrical defaults
    default_trace_width: float = 0.25    # mm
    default_clearance: float = 0.15      # mm
    ambient_temperature: float = 25.0    # °C
    max_temp_rise: float = 10.0          # °C (for trace width calculation)
    copper_weight_oz: float = 1.0        # oz/ft² (1oz = 35µm)

    # Courtyard settings (IPC-7351B)
    courtyard_excess: float = 0.25       # mm (Nominal density)
    courtyard_grid: float = 0.05         # mm (round to this grid)


@dataclass
class PartsResult:
    """Result from parts piston operations"""
    components: Dict[str, Component]     # ref -> Component
    nets: Dict[str, Dict]                # net_name -> net info
    net_classes: Dict[str, NetClass]     # net_name -> class

    # Statistics
    component_count: int = 0
    net_count: int = 0
    power_net_count: int = 0
    high_speed_net_count: int = 0

    # Warnings and errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # BOM info
    unique_parts: int = 0
    total_cost: float = 0.0


# =============================================================================
# LIBRARY MANAGER
# =============================================================================

class LibraryManager:
    """
    Comprehensive Library Management System

    Provides access to multiple component library sources:
    - Internal parametric library (built-in)
    - KiCad official libraries
    - Manufacturer libraries (TI, STM, Espressif, etc.)
    - Community libraries (SnapEDA, Ultra Librarian)
    - User custom libraries

    Uses kiutils-compatible parsing for KiCad 6/7/8 files.

    Research References:
    - KiCad Library Conventions (KLC): https://klc.kicad.org/
    - kiutils Python library: https://github.com/mvnmgrx/kiutils
    - kicad-footprint-generator: https://github.com/pointhi/kicad-footprint-generator
    """

    # Common KiCad library paths by platform
    KICAD_PATHS = {
        'win32': [
            'C:/Program Files/KiCad/8.0/share/kicad',
            'C:/Program Files/KiCad/7.0/share/kicad',
            'C:/Program Files/KiCad/6.0/share/kicad',
            os.path.expanduser('~/Documents/KiCad/8.0'),
        ],
        'linux': [
            '/usr/share/kicad',
            '/usr/local/share/kicad',
            os.path.expanduser('~/.local/share/kicad'),
        ],
        'darwin': [
            '/Applications/KiCad/KiCad.app/Contents/SharedSupport',
            os.path.expanduser('~/Library/Preferences/kicad'),
        ]
    }

    # Built-in parametric component database
    INTERNAL_LIBRARY = {
        'R': {  # Resistors
            'packages': ['0201', '0402', '0603', '0805', '1206', '1210', '2010', '2512'],
            'values': ['0', '1', '10', '100', '1k', '10k', '100k', '1M'],
            'tolerances': ['1%', '5%', '10%'],
            'power': [0.03125, 0.0625, 0.1, 0.125, 0.25, 0.5, 1.0],  # Watts
        },
        'C': {  # Capacitors
            'packages': ['0201', '0402', '0603', '0805', '1206', '1210'],
            'values': ['1pF', '10pF', '100pF', '1nF', '10nF', '100nF', '1uF', '10uF', '100uF'],
            'tolerances': ['5%', '10%', '20%'],
            'voltages': [6.3, 10, 16, 25, 50, 100],  # Volts
        },
        'L': {  # Inductors
            'packages': ['0402', '0603', '0805', '1008', '1210'],
            'values': ['1nH', '10nH', '100nH', '1uH', '10uH', '100uH', '1mH'],
            'currents': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],  # Amps
        },
        'D': {  # Diodes/LEDs
            'packages': ['0402', '0603', '0805', 'SOD-123', 'SOD-323', 'SOD-523'],
            'types': ['LED', 'Schottky', 'Zener', 'TVS', 'Standard'],
        },
    }

    # IPC-7351B Package dimensions (mm) for footprint generation
    # Format: package -> (pad_width, pad_height, pad_spacing, courtyard_excess)
    IPC7351B_PACKAGES = {
        '0201': (0.30, 0.30, 0.60, 0.10),   # 0.6mm x 0.3mm
        '0402': (0.50, 0.50, 1.00, 0.15),   # 1.0mm x 0.5mm
        '0603': (0.80, 0.75, 1.60, 0.20),   # 1.6mm x 0.8mm
        '0805': (1.00, 1.20, 2.00, 0.25),   # 2.0mm x 1.25mm
        '1206': (1.60, 1.60, 3.20, 0.25),   # 3.2mm x 1.6mm
        '1210': (1.60, 2.50, 3.20, 0.25),   # 3.2mm x 2.5mm
        '2010': (2.50, 2.50, 5.00, 0.30),   # 5.0mm x 2.5mm
        '2512': (2.50, 3.20, 6.40, 0.30),   # 6.4mm x 3.2mm
    }

    def __init__(self, config: 'PartsConfig' = None):
        self.config = config
        self.libraries: Dict[str, LibraryInfo] = {}
        self.component_cache: Dict[str, LibraryComponent] = {}
        self._symbol_cache: Dict[str, Dict] = {}
        self._footprint_cache: Dict[str, Dict] = {}

        # Auto-detect and register libraries
        self._auto_detect_libraries()

    def _auto_detect_libraries(self):
        """Auto-detect available libraries on the system"""
        import sys
        platform = sys.platform

        # Add internal library
        self.libraries['internal'] = LibraryInfo(
            name='Internal Parametric',
            lib_type=LibraryType.INTERNAL,
            path='',
            description='Built-in parametric component library',
            priority=100
        )

        # Detect KiCad installation
        kicad_paths = self.KICAD_PATHS.get(platform, [])
        for kicad_base in kicad_paths:
            if os.path.exists(kicad_base):
                # Symbol libraries
                sym_path = os.path.join(kicad_base, 'symbols')
                if os.path.exists(sym_path):
                    self.libraries['kicad_symbols'] = LibraryInfo(
                        name='KiCad Official Symbols',
                        lib_type=LibraryType.KICAD_SYMBOL,
                        path=sym_path,
                        description='Official KiCad symbol libraries',
                        priority=50
                    )

                # Footprint libraries
                fp_path = os.path.join(kicad_base, 'footprints')
                if os.path.exists(fp_path):
                    self.libraries['kicad_footprints'] = LibraryInfo(
                        name='KiCad Official Footprints',
                        lib_type=LibraryType.KICAD_FOOTPRINT,
                        path=fp_path,
                        description='Official KiCad footprint libraries',
                        priority=50
                    )
                break

        # Add user library paths from config
        if self.config:
            if self.config.custom_library_path and os.path.exists(self.config.custom_library_path):
                self.libraries['user_custom'] = LibraryInfo(
                    name='User Custom',
                    lib_type=LibraryType.USER_CUSTOM,
                    path=self.config.custom_library_path,
                    description='User custom component library',
                    priority=90
                )

    def add_library(self, lib_info: LibraryInfo):
        """Add a library to the manager"""
        self.libraries[lib_info.name] = lib_info

    def search(self, query: str, lib_type: LibraryType = None,
               limit: int = 50) -> List[LibraryComponent]:
        """
        Search for components across all libraries.

        Args:
            query: Search query (part name, value, description)
            lib_type: Optional filter by library type
            limit: Maximum results to return

        Returns:
            List of matching LibraryComponent objects
        """
        results = []
        query_lower = query.lower()

        # Search internal library first
        if lib_type is None or lib_type == LibraryType.INTERNAL:
            results.extend(self._search_internal(query_lower))

        # Search file-based libraries
        for lib_name, lib_info in self.libraries.items():
            if not lib_info.enabled:
                continue
            if lib_type and lib_info.lib_type != lib_type:
                continue
            if lib_info.lib_type == LibraryType.INTERNAL:
                continue  # Already searched

            if lib_info.lib_type == LibraryType.KICAD_SYMBOL:
                results.extend(self._search_kicad_symbols(lib_info, query_lower))
            elif lib_info.lib_type == LibraryType.KICAD_FOOTPRINT:
                results.extend(self._search_kicad_footprints(lib_info, query_lower))

        # Sort by relevance and limit
        results.sort(key=lambda x: self._relevance_score(x, query_lower), reverse=True)
        return results[:limit]

    def _search_internal(self, query: str) -> List[LibraryComponent]:
        """Search the internal parametric library"""
        results = []

        # Detect component type from query
        if any(x in query for x in ['resistor', 'res', 'ohm', 'r ']):
            comp_type = 'R'
        elif any(x in query for x in ['capacitor', 'cap', 'farad', 'c ']):
            comp_type = 'C'
        elif any(x in query for x in ['inductor', 'ind', 'henry', 'l ']):
            comp_type = 'L'
        elif any(x in query for x in ['led', 'diode']):
            comp_type = 'D'
        else:
            return results

        lib_data = self.INTERNAL_LIBRARY.get(comp_type, {})
        packages = lib_data.get('packages', [])

        # Match package size in query
        for pkg in packages:
            if pkg in query:
                # Generate component for this package
                results.append(LibraryComponent(
                    lib_id=f"internal:{comp_type}_{pkg}",
                    name=f"{comp_type}_{pkg}",
                    description=f"Generic {comp_type} component, {pkg} package",
                    keywords=[comp_type.lower(), pkg, 'smd', 'generic'],
                    footprint_filters=[f"*{pkg}*"]
                ))

        return results

    def _search_kicad_symbols(self, lib_info: LibraryInfo, query: str) -> List[LibraryComponent]:
        """Search KiCad symbol libraries"""
        results = []

        if not os.path.exists(lib_info.path):
            return results

        # Scan .kicad_sym files
        for filename in os.listdir(lib_info.path):
            if not filename.endswith('.kicad_sym'):
                continue

            lib_name = filename[:-10]  # Remove .kicad_sym

            # Check if library name matches query
            if query in lib_name.lower():
                results.append(LibraryComponent(
                    lib_id=f"{lib_name}:*",
                    name=lib_name,
                    description=f"KiCad library: {lib_name}",
                    keywords=[lib_name.lower()]
                ))

        return results

    def _search_kicad_footprints(self, lib_info: LibraryInfo, query: str) -> List[LibraryComponent]:
        """Search KiCad footprint libraries"""
        results = []

        if not os.path.exists(lib_info.path):
            return results

        # Scan .pretty directories
        for dirname in os.listdir(lib_info.path):
            if not dirname.endswith('.pretty'):
                continue

            lib_name = dirname[:-7]  # Remove .pretty

            if query in lib_name.lower():
                fp_dir = os.path.join(lib_info.path, dirname)
                # Count footprints in library
                fp_count = len([f for f in os.listdir(fp_dir) if f.endswith('.kicad_mod')])

                results.append(LibraryComponent(
                    lib_id=f"{lib_name}:*",
                    name=lib_name,
                    description=f"KiCad footprint library: {lib_name} ({fp_count} footprints)",
                    keywords=[lib_name.lower(), 'footprint']
                ))

        return results

    def _relevance_score(self, component: LibraryComponent, query: str) -> float:
        """Calculate relevance score for search ranking"""
        score = 0.0

        # Exact name match
        if query == component.name.lower():
            score += 100.0

        # Name contains query
        if query in component.name.lower():
            score += 50.0

        # Description contains query
        if query in component.description.lower():
            score += 20.0

        # Keywords match
        for kw in component.keywords:
            if query in kw.lower():
                score += 10.0

        return score

    def get_component(self, lib_id: str) -> Optional[LibraryComponent]:
        """
        Get a specific component by library ID.

        Args:
            lib_id: Format "LibraryName:ComponentName"

        Returns:
            LibraryComponent or None if not found
        """
        if lib_id in self.component_cache:
            return self.component_cache[lib_id]

        parts = lib_id.split(':')
        if len(parts) != 2:
            return None

        lib_name, comp_name = parts

        # Check internal library
        if lib_name == 'internal':
            return self._get_internal_component(comp_name)

        # Check file-based libraries
        # TODO: Implement full kiutils parsing

        return None

    def _get_internal_component(self, comp_name: str) -> Optional[LibraryComponent]:
        """Get component from internal library"""
        # Parse component name like "R_0603" or "C_0402"
        if '_' not in comp_name:
            return None

        comp_type, package = comp_name.split('_', 1)

        if comp_type not in self.INTERNAL_LIBRARY:
            return None

        if package not in self.IPC7351B_PACKAGES:
            return None

        # Generate component with IPC-7351B footprint
        pkg_dims = self.IPC7351B_PACKAGES[package]
        pad_w, pad_h, pad_spacing, cy_excess = pkg_dims

        pins = [
            {'number': '1', 'name': '1', 'x': -pad_spacing/2, 'y': 0},
            {'number': '2', 'name': '2', 'x': pad_spacing/2, 'y': 0},
        ]

        pads = [
            {'number': '1', 'x': -pad_spacing/2, 'y': 0, 'width': pad_w, 'height': pad_h},
            {'number': '2', 'x': pad_spacing/2, 'y': 0, 'width': pad_w, 'height': pad_h},
        ]

        return LibraryComponent(
            lib_id=f"internal:{comp_name}",
            name=comp_name,
            description=f"Generic {comp_type} {package} (IPC-7351B)",
            keywords=[comp_type.lower(), package, 'smd', 'ipc-7351b'],
            footprint_filters=[f"*{package}*"],
            pins=pins,
            pads=pads
        )

    def generate_ipc7351b_footprint(self, package: str, comp_type: str = 'R') -> Dict:
        """
        Generate IPC-7351B compliant footprint.

        Based on research:
        - IPC-7351B mathematical algorithms calculate optimal pad sizes
        - Three density levels: Most (M), Nominal (N), Least (L)
        - Uses solder fillet goals (toe, heel, side)

        Reference: https://pcbsync.com/ipc-7351/

        Args:
            package: Package size (e.g., '0603', '0805')
            comp_type: Component type ('R', 'C', 'L')

        Returns:
            Dict with footprint data
        """
        if package not in self.IPC7351B_PACKAGES:
            return {}

        pad_w, pad_h, pad_spacing, cy_excess = self.IPC7351B_PACKAGES[package]

        # IPC-7351B naming convention
        # Example: RESC1608X55N (Resistor, Chip, 1.6x0.8mm, 0.55mm height, Nominal)
        pkg_mm = {
            '0201': '0603', '0402': '1005', '0603': '1608', '0805': '2012',
            '1206': '3216', '1210': '3225', '2010': '5025', '2512': '6432'
        }

        type_codes = {'R': 'RES', 'C': 'CAP', 'L': 'IND'}
        type_code = type_codes.get(comp_type, 'RES')

        footprint = {
            'name': f"{type_code}C{pkg_mm.get(package, package)}X55N",
            'description': f"IPC-7351B {comp_type} {package} Nominal density",
            'pads': [
                {
                    'number': '1',
                    'type': 'smd',
                    'shape': 'roundrect',
                    'at': {'x': -pad_spacing/2, 'y': 0},
                    'size': {'x': pad_w, 'y': pad_h},
                    'layers': ['F.Cu', 'F.Paste', 'F.Mask'],
                    'roundrect_rratio': 0.25
                },
                {
                    'number': '2',
                    'type': 'smd',
                    'shape': 'roundrect',
                    'at': {'x': pad_spacing/2, 'y': 0},
                    'size': {'x': pad_w, 'y': pad_h},
                    'layers': ['F.Cu', 'F.Paste', 'F.Mask'],
                    'roundrect_rratio': 0.25
                }
            ],
            'courtyard': {
                'width': pad_spacing + pad_w + 2*cy_excess,
                'height': pad_h + 2*cy_excess,
                'layer': 'F.CrtYd'
            },
            'fabrication': {
                'width': pad_spacing,
                'height': pad_h,
                'layer': 'F.Fab'
            }
        }

        return footprint

    def list_libraries(self) -> List[LibraryInfo]:
        """List all registered libraries"""
        return sorted(self.libraries.values(), key=lambda x: -x.priority)


# =============================================================================
# THERMAL VIA CALCULATOR
# =============================================================================

class ThermalViaCalculator:
    """
    Calculate thermal via requirements based on power dissipation.

    Research References:
    - TI PCB Thermal Calculator: https://www.ti.com/design-resources/design-tools-simulation/models-simulators/pcb-thermal-calculator.html
    - Würth Elektronik thermal via study: https://www.we-online.com/
    - IPC thermal design guidelines

    Key Formulas:
    - Single via thermal resistance: R_th = L / (k * A)
      where L = PCB thickness, k = copper conductivity, A = via area
    - Via array: R_th_array = R_th_single / N
    - Temperature rise: ΔT = P * R_th
    """

    # Thermal conductivity constants
    COPPER_K = 385.0          # W/(m·K) thermal conductivity of copper
    FR4_K = 0.3               # W/(m·K) thermal conductivity of FR4

    def __init__(self, pcb_thickness: float = 1.6, copper_thickness: float = 0.035):
        """
        Args:
            pcb_thickness: PCB thickness in mm (default 1.6mm)
            copper_thickness: Copper plating thickness in mm (default 35µm = 1oz)
        """
        self.pcb_thickness = pcb_thickness
        self.copper_thickness = copper_thickness

    def calculate_via_thermal_resistance(self, via_diameter: float, plating_thickness: float = None) -> float:
        """
        Calculate thermal resistance of a single via.

        Formula: R_th = L / (k * A_copper)
        where A_copper = π * d * t (via wall area)

        Args:
            via_diameter: Via hole diameter in mm
            plating_thickness: Copper plating in mm (default from config)

        Returns:
            Thermal resistance in °C/W (K/W)
        """
        if plating_thickness is None:
            plating_thickness = self.copper_thickness

        # Via wall area (cylinder surface)
        # A = π * diameter * plating_thickness
        via_wall_area = math.pi * via_diameter * plating_thickness  # mm²
        via_wall_area_m2 = via_wall_area * 1e-6  # Convert to m²

        pcb_thickness_m = self.pcb_thickness * 1e-3  # Convert to m

        # Thermal resistance: R = L / (k * A)
        r_th = pcb_thickness_m / (self.COPPER_K * via_wall_area_m2)

        return r_th

    def calculate_via_array_resistance(self, via_diameter: float, via_count: int,
                                       plating_thickness: float = None) -> float:
        """
        Calculate thermal resistance of a via array.

        Formula: R_th_array = R_th_single / N

        Args:
            via_diameter: Via hole diameter in mm
            via_count: Number of vias in array
            plating_thickness: Copper plating in mm

        Returns:
            Array thermal resistance in °C/W
        """
        single_r = self.calculate_via_thermal_resistance(via_diameter, plating_thickness)
        return single_r / via_count

    def calculate_required_vias(self, power_dissipation: float, max_temp_rise: float,
                                via_diameter: float = 0.3, plating_thickness: float = None) -> int:
        """
        Calculate number of thermal vias needed for given power.

        Formula: N = P * R_th_single / ΔT_max

        Args:
            power_dissipation: Power in Watts
            max_temp_rise: Maximum allowed temperature rise in °C
            via_diameter: Via diameter in mm
            plating_thickness: Copper plating in mm

        Returns:
            Minimum number of vias needed
        """
        if power_dissipation <= 0 or max_temp_rise <= 0:
            return 0

        single_r = self.calculate_via_thermal_resistance(via_diameter, plating_thickness)

        # ΔT = P * R_th, so R_required = ΔT_max / P
        r_required = max_temp_rise / power_dissipation

        # N = R_single / R_required
        via_count = math.ceil(single_r / r_required)

        return max(1, via_count)

    def recommend_via_pattern(self, power_dissipation: float, pad_size: Tuple[float, float],
                              max_temp_rise: float = 20.0, via_diameter: float = 0.3) -> Dict:
        """
        Recommend thermal via pattern for a thermal pad.

        Args:
            power_dissipation: Power in Watts
            pad_size: (width, height) of thermal pad in mm
            max_temp_rise: Maximum temperature rise in °C
            via_diameter: Via diameter in mm

        Returns:
            Dict with via pattern recommendation
        """
        via_count = self.calculate_required_vias(
            power_dissipation, max_temp_rise, via_diameter
        )

        pad_w, pad_h = pad_size

        # Optimal via spacing: 1.0-1.5mm (from research)
        via_pitch = 1.0  # mm

        # Calculate grid dimensions
        cols = max(1, int(pad_w / via_pitch))
        rows = max(1, int(pad_h / via_pitch))

        # Ensure enough vias
        while cols * rows < via_count:
            if cols <= rows:
                cols += 1
            else:
                rows += 1

        # Generate via positions
        positions = []
        start_x = -(cols - 1) * via_pitch / 2
        start_y = -(rows - 1) * via_pitch / 2

        for r in range(rows):
            for c in range(cols):
                positions.append((
                    start_x + c * via_pitch,
                    start_y + r * via_pitch
                ))

        actual_r_th = self.calculate_via_array_resistance(via_diameter, len(positions))
        temp_rise = power_dissipation * actual_r_th

        return {
            'via_count': len(positions),
            'via_diameter': via_diameter,
            'grid': (cols, rows),
            'pitch': via_pitch,
            'positions': positions,
            'thermal_resistance': actual_r_th,
            'estimated_temp_rise': temp_rise
        }


# =============================================================================
# PARTS PISTON
# =============================================================================

class PartsPiston:
    """
    Comprehensive Parts Management Piston

    Handles ALL component-related operations for PCB design:
    - Component selection and sourcing
    - Parameter parsing and validation
    - Footprint resolution
    - Courtyard calculation
    - Electrical requirement extraction
    - Net classification
    - BOM generation

    Usage:
        config = PartsConfig(
            kicad_symbol_path='/usr/share/kicad/symbols',
            custom_library_path='./my_libs'
        )
        piston = PartsPiston(config)

        # Parse user input
        component = piston.parse_component("10k resistor 0603")

        # Build from schematic data
        result = piston.build_from_dict(schematic_data)

        # Calculate trace requirements
        width = piston.calculate_trace_width(current=2.0, temp_rise=10.0)
    """

    def __init__(self, config: PartsConfig = None):
        self.config = config or PartsConfig()
        self.components: Dict[str, Component] = {}
        self.nets: Dict[str, Dict] = {}
        self.net_classes: Dict[str, NetClass] = {}
        self.ref_counters: Dict[str, int] = defaultdict(int)

        # Initialize library manager
        self.library_manager = LibraryManager(config)

        # Initialize thermal via calculator
        self.thermal_calculator = ThermalViaCalculator()

    # =========================================================================
    # LIBRARY ACCESS
    # =========================================================================

    def search_library(self, query: str, limit: int = 50) -> List[LibraryComponent]:
        """
        Search all component libraries.

        Args:
            query: Search string (part name, value, etc.)
            limit: Max results

        Returns:
            List of matching components
        """
        return self.library_manager.search(query, limit=limit)

    def get_from_library(self, lib_id: str) -> Optional[LibraryComponent]:
        """
        Get specific component from library.

        Args:
            lib_id: Library ID in format "LibraryName:ComponentName"

        Returns:
            LibraryComponent or None
        """
        return self.library_manager.get_component(lib_id)

    def list_available_libraries(self) -> List[LibraryInfo]:
        """List all available component libraries"""
        return self.library_manager.list_libraries()

    def add_custom_library(self, path: str, name: str = None):
        """
        Add a custom library path.

        Args:
            path: Path to library directory
            name: Optional library name
        """
        if name is None:
            name = os.path.basename(path)

        self.library_manager.add_library(LibraryInfo(
            name=name,
            lib_type=LibraryType.USER_CUSTOM,
            path=path,
            priority=80
        ))

    # =========================================================================
    # MAIN API
    # =========================================================================

    def analyze(self, parts_db: Dict) -> PartsResult:
        """
        Analyze parts database and build complete component information.

        This is the standard piston API entry point.
        Alias for build_from_dict() for consistent piston interface.

        Args:
            parts_db: Raw parts database with 'parts' and 'nets' keys

        Returns:
            PartsResult with processed components and net information
        """
        return self.build_from_dict(parts_db)

    def build_from_dict(self, parts_db: Dict) -> PartsResult:
        """
        Build complete component database from raw parts dictionary.

        This is the main entry point that processes user input and creates
        fully-specified components with all electrical requirements.

        Args:
            parts_db: Raw parts database with 'parts' and 'nets' keys

        Returns:
            PartsResult with processed components and net information
        """
        self.components.clear()
        self.nets.clear()
        self.net_classes.clear()
        self.ref_counters.clear()

        errors = []
        warnings = []

        # Process parts
        raw_parts = parts_db.get('parts', {})
        for ref, part_data in raw_parts.items():
            try:
                component = self._build_component(ref, part_data)
                self.components[ref] = component
            except Exception as e:
                errors.append(f"Error processing {ref}: {e}")

        # Process nets
        raw_nets = parts_db.get('nets', {})
        for net_name, net_data in raw_nets.items():
            try:
                self._process_net(net_name, net_data)
            except Exception as e:
                errors.append(f"Error processing net {net_name}: {e}")

        # Classify nets
        self._classify_all_nets()

        # Calculate electrical requirements
        self._calculate_electrical_requirements()

        # Build result
        return PartsResult(
            components=self.components.copy(),
            nets=self.nets.copy(),
            net_classes=self.net_classes.copy(),
            component_count=len(self.components),
            net_count=len(self.nets),
            power_net_count=sum(1 for nc in self.net_classes.values()
                               if nc in [NetClass.POWER, NetClass.GROUND]),
            high_speed_net_count=sum(1 for nc in self.net_classes.values()
                                    if nc == NetClass.HIGH_SPEED),
            errors=errors,
            warnings=warnings,
            unique_parts=self._count_unique_parts(),
            total_cost=self._calculate_total_cost()
        )

    def parse_component(self, description: str) -> Component:
        """
        Parse natural language component description.

        Examples:
            "10k resistor 0603"
            "100nF capacitor C0402"
            "ESP32-WROOM-32 module"
            "USB-C connector"
            "LED red 0805"

        Args:
            description: Natural language component description

        Returns:
            Component object with parsed values
        """
        desc_lower = description.lower()

        # Detect component type
        comp_type = self._detect_component_type(desc_lower)

        # Parse value
        value = self._parse_value(description, comp_type)

        # Parse footprint
        footprint = self._parse_footprint(description, comp_type)

        # Generate reference designator
        ref = self._generate_ref_designator(comp_type)

        # Create component
        component = Component(
            ref=ref,
            value=value,
            footprint=footprint,
            description=description
        )

        # Set defaults based on type
        self._set_component_defaults(component, comp_type)

        return component

    def calculate_trace_width(self, current: float, temp_rise: float = 10.0,
                             copper_oz: float = 1.0, is_internal: bool = False) -> float:
        """
        Calculate minimum trace width for given current.

        Uses IPC-2221 formula:
        I = k * (ΔT)^0.44 * (A)^0.725

        Where:
        - I = current (A)
        - k = 0.024 internal, 0.048 external
        - ΔT = temperature rise (°C)
        - A = cross-sectional area (mils²)

        Args:
            current: Current in Amps
            temp_rise: Allowed temperature rise in °C
            copper_oz: Copper weight in oz/ft²
            is_internal: True for internal layers

        Returns:
            Minimum trace width in mm
        """
        if current <= 0:
            return self.config.default_trace_width

        k = IPC2221_INTERNAL_K if is_internal else IPC2221_EXTERNAL_K

        # Solve for area: A = (I / (k * ΔT^0.44))^(1/0.725)
        area_mils2 = (current / (k * (temp_rise ** IPC2221_C))) ** (1 / IPC2221_B)

        # Convert to mm: copper thickness in mils
        # 1 oz/ft² = 1.37 mils = 0.035 mm
        thickness_mils = copper_oz * 1.37

        # Width = Area / Thickness
        width_mils = area_mils2 / thickness_mils

        # Convert mils to mm (1 mil = 0.0254 mm)
        width_mm = width_mils * 0.0254

        return max(width_mm, self.config.default_trace_width)

    def calculate_clearance(self, voltage: float, is_internal: bool = False) -> float:
        """
        Calculate minimum clearance for given voltage.

        Uses IPC-2221 voltage-to-clearance table.

        Args:
            voltage: Working voltage in Volts
            is_internal: True for internal layers

        Returns:
            Minimum clearance in mm
        """
        # IPC-2221 Table 6-1 (simplified, external uncoated)
        # For internal layers, values are typically 50% of external
        clearance_table = [
            (0, 0.1),        # 0-15V
            (15, 0.1),
            (30, 0.1),
            (50, 0.13),
            (100, 0.25),
            (150, 0.4),
            (170, 0.5),
            (250, 0.8),
            (300, 0.8),
            (500, 2.5),
        ]

        clearance = self.config.default_clearance
        for v, c in clearance_table:
            if voltage >= v:
                clearance = c

        if is_internal:
            clearance *= 0.5

        return max(clearance, self.config.default_clearance)

    def calculate_courtyard(self, pins: List[PinInfo], mount_type: MountType) -> CourtyardInfo:
        """
        Calculate component courtyard based on pin positions and IPC-7351B.

        Args:
            pins: List of component pins with positions
            mount_type: SMD or THT

        Returns:
            CourtyardInfo with calculated dimensions
        """
        if not pins:
            return CourtyardInfo(width=1.0, height=1.0)

        # Find bounding box of all pads
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for pin in pins:
            if pin.pad:
                pad = pin.pad
                left = pad.offset_x - pad.size_x / 2
                right = pad.offset_x + pad.size_x / 2
                bottom = pad.offset_y - pad.size_y / 2
                top = pad.offset_y + pad.size_y / 2

                min_x = min(min_x, left)
                max_x = max(max_x, right)
                min_y = min(min_y, bottom)
                max_y = max(max_y, top)

        if min_x == float('inf'):
            return CourtyardInfo(width=1.0, height=1.0)

        # Add courtyard excess (IPC-7351B)
        excess = self.config.courtyard_excess

        width = (max_x - min_x) + 2 * excess
        height = (max_y - min_y) + 2 * excess

        # Round to grid
        grid = self.config.courtyard_grid
        width = math.ceil(width / grid) * grid
        height = math.ceil(height / grid) * grid

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        return CourtyardInfo(
            width=width,
            height=height,
            offset_x=center_x,
            offset_y=center_y,
            clearance=excess
        )

    def get_net_class(self, net_name: str) -> NetClass:
        """Get the classification of a net"""
        return self.net_classes.get(net_name, NetClass.SIGNAL)

    def get_routing_requirements(self, net_name: str) -> Dict[str, Any]:
        """
        Get complete routing requirements for a net.

        Returns dict with:
        - net_class: NetClass enum
        - min_trace_width: mm
        - min_clearance: mm
        - max_length: mm (0 = no limit)
        - is_differential: bool
        - diff_pair_name: str
        - length_match_group: str
        - max_vias: int (-1 = no limit)
        """
        net_class = self.get_net_class(net_name)
        net_data = self.nets.get(net_name, {})

        # Calculate requirements based on net class
        trace_width = self.config.default_trace_width
        clearance = self.config.default_clearance

        if net_class == NetClass.POWER:
            trace_width = self.calculate_trace_width(current=1.0)
        elif net_class == NetClass.HIGH_CURRENT:
            trace_width = self.calculate_trace_width(current=3.0)
        elif net_class == NetClass.HIGH_VOLTAGE:
            clearance = self.calculate_clearance(voltage=200.0)

        return {
            'net_class': net_class,
            'min_trace_width': trace_width,
            'min_clearance': clearance,
            'max_length': net_data.get('max_length', 0),
            'is_differential': net_data.get('is_differential', False),
            'diff_pair_name': net_data.get('diff_pair_name', ''),
            'length_match_group': net_data.get('length_match_group', ''),
            'max_vias': net_data.get('max_vias', -1)
        }

    def get_placement_requirements(self, ref: str) -> Dict[str, Any]:
        """
        Get placement requirements for a component.

        Returns dict with:
        - placement_group: str
        - keep_out_radius: mm
        - height_restriction: mm
        - fixed_position: Optional tuple
        - fixed_rotation: Optional float
        - prefer_top: bool
        - thermal_constraints: dict
        """
        component = self.components.get(ref)
        if not component:
            return {}

        return {
            'placement_group': component.placement_group,
            'keep_out_radius': component.electrical.keep_out_radius,
            'height_restriction': component.electrical.height_restriction,
            'fixed_position': component.fixed_position,
            'fixed_rotation': component.fixed_rotation,
            'prefer_top': component.prefer_top,
            'thermal_constraints': {
                'power_dissipation': component.thermal.power_dissipation,
                'requires_heatsink': component.thermal.requires_heatsink,
                'thermal_via_count': component.thermal.thermal_via_count
            }
        }

    # =========================================================================
    # COMPONENT BUILDING
    # =========================================================================

    def _build_component(self, ref: str, part_data: Dict) -> Component:
        """Build a complete Component from raw part data"""
        # Extract basic info
        value = part_data.get('value', '')
        footprint = part_data.get('footprint', '')
        description = part_data.get('description', '')

        # Determine mount type
        mount_type = MountType.SMD
        if any(x in footprint.lower() for x in ['tht', 'dip', 'to-220', 'to-92', 'sip']):
            mount_type = MountType.THT

        # Build pins
        pins = self._build_pins(part_data)

        # Calculate courtyard
        courtyard = self.calculate_courtyard(pins, mount_type)

        # Build electrical requirements
        electrical = self._build_electrical_requirements(part_data, pins)

        # Build thermal info
        thermal = self._build_thermal_info(part_data)

        # Build source info
        source = self._build_source_info(part_data)

        return Component(
            ref=ref,
            value=value,
            footprint=footprint,
            mount_type=mount_type,
            pins=pins,
            courtyard=courtyard,
            electrical=electrical,
            thermal=thermal,
            source=source,
            description=description,
            datasheet=part_data.get('datasheet', ''),
            placement_group=part_data.get('group', ''),
            prefer_top=part_data.get('prefer_top', True)
        )

    def _build_pins(self, part_data: Dict) -> List[PinInfo]:
        """Build pin list from part data"""
        pins = []

        # Try different formats
        raw_pins = part_data.get('pins', part_data.get('physical_pins',
                                 part_data.get('used_pins', [])))

        for pin_data in raw_pins:
            # Handle both dict and tuple formats
            if isinstance(pin_data, dict):
                pin = self._build_pin_from_dict(pin_data)
            elif isinstance(pin_data, (list, tuple)):
                pin = self._build_pin_from_tuple(pin_data)
            else:
                continue

            pins.append(pin)

        return pins

    def _build_pin_from_dict(self, pin_data: Dict) -> PinInfo:
        """Build PinInfo from dictionary"""
        # Extract pad info
        pad = None
        phys = pin_data.get('physical', {})
        if phys:
            pad = PadInfo(
                number=str(pin_data.get('number', '')),
                size_x=phys.get('pad_w', phys.get('size_x', 1.0)),
                size_y=phys.get('pad_h', phys.get('size_y', 1.0)),
                offset_x=phys.get('offset_x', 0.0),
                offset_y=phys.get('offset_y', 0.0),
                drill=phys.get('drill', 0.0)
            )

        # Determine electrical type
        elec_type = PinType.PASSIVE
        type_str = pin_data.get('type', '').lower()
        if 'input' in type_str:
            elec_type = PinType.INPUT
        elif 'output' in type_str:
            elec_type = PinType.OUTPUT
        elif 'power' in type_str:
            elec_type = PinType.POWER_IN
        elif 'bidir' in type_str:
            elec_type = PinType.BIDIRECTIONAL

        return PinInfo(
            number=str(pin_data.get('number', pin_data.get('pin', ''))),
            name=pin_data.get('name', pin_data.get('function', '')),
            electrical_type=elec_type,
            net=pin_data.get('net', ''),
            pad=pad,
            max_current=pin_data.get('max_current', 0.5),
            max_voltage=pin_data.get('max_voltage', 50.0),
            function=pin_data.get('function', ''),
            is_critical=pin_data.get('is_critical', False)
        )

    def _build_pin_from_tuple(self, pin_data: Tuple) -> PinInfo:
        """Build PinInfo from tuple (number, name, net)"""
        number = str(pin_data[0]) if len(pin_data) > 0 else ''
        name = str(pin_data[1]) if len(pin_data) > 1 else ''
        net = str(pin_data[2]) if len(pin_data) > 2 else ''

        return PinInfo(
            number=number,
            name=name,
            net=net
        )

    def _build_electrical_requirements(self, part_data: Dict, pins: List[PinInfo]) -> ElectricalRequirements:
        """Build electrical requirements from part data"""
        elec = part_data.get('electrical', {})

        # Calculate max current from pins
        max_current = max((p.max_current for p in pins), default=0.5)
        max_voltage = max((p.max_voltage for p in pins), default=50.0)

        # Override with explicit values
        max_current = elec.get('max_current', max_current)
        max_voltage = elec.get('max_voltage', max_voltage)

        # Calculate trace width and clearance
        min_trace = self.calculate_trace_width(max_current)
        min_clearance = self.calculate_clearance(max_voltage)

        return ElectricalRequirements(
            max_current=max_current,
            max_voltage=max_voltage,
            min_trace_width=min_trace,
            min_clearance=min_clearance,
            is_differential=elec.get('is_differential', False),
            diff_pair_name=elec.get('diff_pair', ''),
            length_match_group=elec.get('length_match', ''),
            max_length=elec.get('max_length', 0.0),
            keep_out_radius=elec.get('keep_out', 0.0),
            requires_isolation=elec.get('isolation', False)
        )

    def _build_thermal_info(self, part_data: Dict) -> ThermalInfo:
        """Build thermal info from part data"""
        therm = part_data.get('thermal', {})

        return ThermalInfo(
            power_dissipation=therm.get('power', 0.0),
            theta_ja=therm.get('theta_ja', 0.0),
            theta_jc=therm.get('theta_jc', 0.0),
            max_junction_temp=therm.get('max_temp', 125.0),
            requires_heatsink=therm.get('heatsink', False),
            thermal_pad=therm.get('thermal_pad', False),
            thermal_via_count=therm.get('thermal_vias', 0)
        )

    def _build_source_info(self, part_data: Dict) -> SourceInfo:
        """Build sourcing info from part data"""
        src = part_data.get('source', {})

        source_type = ComponentSource.UNKNOWN
        source_str = src.get('type', '').lower()
        if 'custom' in source_str:
            source_type = ComponentSource.USER_CUSTOM
        elif 'kicad' in source_str:
            source_type = ComponentSource.KICAD_OFFICIAL
        elif 'mfr' in source_str or 'manufacturer' in source_str:
            source_type = ComponentSource.MANUFACTURER

        return SourceInfo(
            source=source_type,
            library_path=src.get('library', ''),
            symbol_name=src.get('symbol', ''),
            footprint_name=src.get('footprint', part_data.get('footprint', '')),
            manufacturer=src.get('manufacturer', part_data.get('mfr', '')),
            mpn=src.get('mpn', part_data.get('mpn', '')),
            supplier=src.get('supplier', ''),
            supplier_pn=src.get('supplier_pn', ''),
            unit_price=src.get('price', 0.0),
            model_3d_path=src.get('3d_model', '')
        )

    # =========================================================================
    # NET PROCESSING
    # =========================================================================

    def _process_net(self, net_name: str, net_data: Dict):
        """Process a net and store its information"""
        self.nets[net_name] = {
            'pins': net_data.get('pins', []),
            'class': None,  # Will be set by _classify_all_nets
            'is_differential': self._is_differential_net(net_name),
            'diff_pair_name': self._get_diff_pair_name(net_name),
            'length_match_group': net_data.get('length_match', ''),
            'max_length': net_data.get('max_length', 0),
            'max_vias': net_data.get('max_vias', -1)
        }

    def _classify_all_nets(self):
        """Classify all nets into net classes"""
        for net_name in self.nets.keys():
            self.net_classes[net_name] = self._classify_net(net_name)

    def _classify_net(self, net_name: str) -> NetClass:
        """Classify a single net based on its name and connections"""
        name_upper = net_name.upper()

        # Ground nets
        if any(x in name_upper for x in ['GND', 'GROUND', 'VSS', 'AGND', 'DGND', 'PGND']):
            return NetClass.GROUND

        # Power nets
        if any(x in name_upper for x in ['VCC', 'VDD', '5V', '3V3', '3.3V', '12V', 'VBAT',
                                          'VIN', 'VOUT', 'PWR', 'POWER', '+5V', '+3.3V']):
            return NetClass.POWER

        # Differential pairs
        if self._is_differential_net(net_name):
            return NetClass.DIFFERENTIAL

        # High-speed signals
        if any(x in name_upper for x in ['CLK', 'CLOCK', 'USB', 'HDMI', 'ETH', 'PCIE',
                                          'DDR', 'SDRAM', 'QSPI', 'HSPI']):
            return NetClass.HIGH_SPEED

        # RF signals
        if any(x in name_upper for x in ['RF', 'ANT', 'ANTENNA', 'LNA', 'PA_OUT']):
            return NetClass.RF

        # Analog signals
        if any(x in name_upper for x in ['ANALOG', 'ADC', 'DAC', 'VREF', 'SENSE']):
            return NetClass.ANALOG

        return NetClass.SIGNAL

    def _is_differential_net(self, net_name: str) -> bool:
        """Check if net is part of a differential pair"""
        # Common differential pair suffixes
        diff_patterns = ['_P', '_N', '+', '-', '_DP', '_DN', '_POS', '_NEG']
        name_upper = net_name.upper()
        return any(name_upper.endswith(p) for p in diff_patterns)

    def _get_diff_pair_name(self, net_name: str) -> str:
        """Get the base name of a differential pair"""
        if not self._is_differential_net(net_name):
            return ''

        # Remove the differential suffix
        name = net_name.upper()
        for suffix in ['_P', '_N', '_DP', '_DN', '_POS', '_NEG']:
            if name.endswith(suffix):
                return net_name[:-len(suffix)]
        if name.endswith('+') or name.endswith('-'):
            return net_name[:-1]

        return net_name

    def _calculate_electrical_requirements(self):
        """Calculate electrical requirements for all components based on nets"""
        for ref, component in self.components.items():
            for pin in component.pins:
                if pin.net:
                    net_class = self.get_net_class(pin.net)
                    if net_class in [NetClass.POWER, NetClass.GROUND]:
                        pin.is_critical = True
                    elif net_class == NetClass.HIGH_SPEED:
                        pin.is_critical = True

    # =========================================================================
    # COMPONENT PARSING
    # =========================================================================

    def _detect_component_type(self, description: str) -> str:
        """Detect component type from description"""
        desc = description.lower()

        if any(x in desc for x in ['resistor', 'res', 'ohm', 'ω']):
            return 'resistor'
        elif any(x in desc for x in ['capacitor', 'cap', 'farad', 'nf', 'uf', 'pf']):
            return 'capacitor'
        elif any(x in desc for x in ['inductor', 'ind', 'henry', 'uh', 'mh', 'nh']):
            return 'inductor'
        elif any(x in desc for x in ['led']):
            return 'led'
        elif any(x in desc for x in ['diode']):
            return 'diode'
        elif any(x in desc for x in ['transistor', 'mosfet', 'bjt', 'fet']):
            return 'transistor'
        elif any(x in desc for x in ['mcu', 'microcontroller', 'esp32', 'stm32', 'atmega']):
            return 'mcu'
        elif any(x in desc for x in ['connector', 'header', 'usb', 'socket']):
            return 'connector'
        elif any(x in desc for x in ['switch', 'button']):
            return 'switch'
        elif any(x in desc for x in ['crystal', 'oscillator', 'xtal']):
            return 'crystal'
        elif any(x in desc for x in ['regulator', 'ldo', 'dcdc', 'buck', 'boost']):
            return 'ic'
        else:
            return 'ic'

    def _parse_value(self, description: str, comp_type: str) -> str:
        """Parse component value from description"""
        # Patterns for different value formats
        patterns = {
            'resistor': r'(\d+(?:\.\d+)?)\s*([kKmM]?)\s*(?:ohm|Ω|ohms?)?',
            'capacitor': r'(\d+(?:\.\d+)?)\s*([pnuμµm]?)\s*[Ff]?',
            'inductor': r'(\d+(?:\.\d+)?)\s*([nuμµm]?)\s*[Hh]?',
        }

        pattern = patterns.get(comp_type, r'(\S+)')
        match = re.search(pattern, description, re.IGNORECASE)

        if match:
            value = match.group(1)
            if len(match.groups()) > 1:
                multiplier = match.group(2)
                if multiplier:
                    if comp_type == 'resistor':
                        if multiplier.lower() == 'k':
                            return f"{value}k"
                        elif multiplier.lower() == 'm':
                            return f"{value}M"
                    elif comp_type == 'capacitor':
                        return f"{value}{multiplier}F"
                    elif comp_type == 'inductor':
                        return f"{value}{multiplier}H"
            return value

        return description

    def _parse_footprint(self, description: str, comp_type: str) -> str:
        """Parse or generate footprint name from description"""
        desc = description.lower()

        # Common SMD sizes
        smd_sizes = ['0201', '0402', '0603', '0805', '1206', '1210', '2010', '2512']
        for size in smd_sizes:
            if size in desc:
                if comp_type == 'resistor':
                    return f"Resistor_SMD:R_{size}_1005Metric"
                elif comp_type == 'capacitor':
                    return f"Capacitor_SMD:C_{size}_1005Metric"
                elif comp_type == 'inductor':
                    return f"Inductor_SMD:L_{size}"
                elif comp_type == 'led':
                    return f"LED_SMD:LED_{size}"

        # Through-hole patterns
        if any(x in desc for x in ['dip', 'through-hole', 'tht']):
            match = re.search(r'dip[- ]?(\d+)', desc)
            if match:
                return f"Package_DIP:DIP-{match.group(1)}_W7.62mm"

        return ''

    def _generate_ref_designator(self, comp_type: str) -> str:
        """Generate next reference designator for component type"""
        prefix = self.config.ref_designator_rules.get(comp_type, 'U')
        self.ref_counters[prefix] += 1
        return f"{prefix}{self.ref_counters[prefix]}"

    def _set_component_defaults(self, component: Component, comp_type: str):
        """Set default values based on component type"""
        if comp_type == 'resistor':
            component.thermal.power_dissipation = 0.1  # 100mW default
        elif comp_type == 'capacitor':
            pass
        elif comp_type == 'mcu':
            component.thermal.power_dissipation = 0.5
            component.placement_group = 'mcu'
        elif comp_type == 'connector':
            component.placement_group = 'connectors'

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _count_unique_parts(self) -> int:
        """Count unique part types (by value+footprint)"""
        unique = set()
        for comp in self.components.values():
            unique.add((comp.value, comp.footprint))
        return len(unique)

    def _calculate_total_cost(self) -> float:
        """Calculate total BOM cost"""
        return sum(c.source.unit_price for c in self.components.values())

    def generate_bom(self) -> List[Dict]:
        """
        Generate Bill of Materials.

        Returns list of dicts with:
        - ref: Reference designator
        - value: Component value
        - footprint: Footprint name
        - manufacturer: Manufacturer name
        - mpn: Manufacturer part number
        - quantity: Count (grouped by same part)
        - unit_price: Price per unit
        - total_price: quantity * unit_price
        """
        # Group by value+footprint+mpn
        groups: Dict[Tuple, List[str]] = defaultdict(list)

        for ref, comp in self.components.items():
            key = (comp.value, comp.footprint, comp.source.mpn)
            groups[key].append(ref)

        bom = []
        for (value, footprint, mpn), refs in groups.items():
            comp = self.components[refs[0]]
            qty = len(refs)
            bom.append({
                'refs': ', '.join(sorted(refs)),
                'value': value,
                'footprint': footprint,
                'manufacturer': comp.source.manufacturer,
                'mpn': mpn,
                'quantity': qty,
                'unit_price': comp.source.unit_price,
                'total_price': qty * comp.source.unit_price,
                'description': comp.description
            })

        return sorted(bom, key=lambda x: x['refs'])

    def optimize_bom(self, bom: List[Dict] = None,
                     prefer_single_supplier: bool = True,
                     allow_substitutions: bool = True) -> Dict:
        """
        Optimize BOM for cost and supply chain risk.

        Based on research from:
        - Cadence: https://resources.pcb.cadence.com/blog/2024-engineering-bom-management
        - Altium: https://resources.altium.com/p/electronics-supply-chain-management-best-practices

        Optimization strategies:
        1. Component consolidation - reduce unique part count
        2. Value substitution - use standard values
        3. Package standardization - fewer package types
        4. Supplier consolidation - single supplier benefits
        5. Alternative parts - identify drop-in replacements

        Args:
            bom: BOM to optimize (uses current if None)
            prefer_single_supplier: Try to consolidate suppliers
            allow_substitutions: Allow equivalent part substitutions

        Returns:
            Dict with optimization results and recommendations
        """
        if bom is None:
            bom = self.generate_bom()

        recommendations = []
        potential_savings = 0.0

        # Standard E-series values for consolidation
        E24_VALUES = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                      3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
        E12_VALUES = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]

        # Analyze resistors for value consolidation
        resistor_values: Dict[str, List[Dict]] = defaultdict(list)
        for item in bom:
            if item['value'].endswith('k') or item['value'].endswith('M') or \
               item['value'].endswith('Ω') or 'ohm' in item['value'].lower():
                resistor_values[item['footprint']].append(item)

        # Check for close values that could be consolidated
        for footprint, items in resistor_values.items():
            if len(items) <= 1:
                continue

            values = []
            for item in items:
                try:
                    val = parse_si_value(item['value'])
                    values.append((val, item))
                except (ValueError, KeyError, TypeError):
                    # Skip items with invalid or unparseable values
                    continue

            values.sort(key=lambda x: x[0])

            for i in range(len(values) - 1):
                v1, item1 = values[i]
                v2, item2 = values[i + 1]

                if v1 > 0 and v2 / v1 < 1.1:  # Within 10%
                    recommendations.append({
                        'type': 'consolidation',
                        'severity': 'suggestion',
                        'message': f"Consider consolidating {item1['value']} and {item2['value']} (same footprint, <10% difference)",
                        'parts': [item1['refs'], item2['refs']]
                    })

        # Analyze package diversity
        packages = defaultdict(int)
        for item in bom:
            packages[item['footprint']] += item['quantity']

        if len(packages) > 10:
            recommendations.append({
                'type': 'standardization',
                'severity': 'info',
                'message': f"High package diversity ({len(packages)} unique footprints). Consider standardizing.",
                'details': dict(packages)
            })

        # Check for missing manufacturer data
        missing_mpn = [item for item in bom if not item['mpn']]
        if missing_mpn:
            recommendations.append({
                'type': 'sourcing',
                'severity': 'warning',
                'message': f"{len(missing_mpn)} items missing manufacturer part numbers",
                'parts': [item['refs'] for item in missing_mpn]
            })

        return {
            'original_bom': bom,
            'unique_parts': len(bom),
            'total_components': sum(item['quantity'] for item in bom),
            'unique_footprints': len(packages),
            'recommendations': recommendations,
            'potential_savings': potential_savings
        }

    def suggest_alternatives(self, ref: str) -> List[Dict]:
        """
        Suggest alternative parts for a component.

        Uses parametric search to find functionally equivalent alternatives.

        Args:
            ref: Reference designator

        Returns:
            List of alternative part suggestions
        """
        component = self.components.get(ref)
        if not component:
            return []

        alternatives = []

        # Search library for similar parts
        search_terms = [component.value, component.footprint]
        for term in search_terms:
            results = self.library_manager.search(term, limit=5)
            for result in results:
                if result.lib_id != f"{component.source.library_path}:{component.value}":
                    alternatives.append({
                        'lib_id': result.lib_id,
                        'name': result.name,
                        'description': result.description,
                        'compatibility': 'check_required'
                    })

        return alternatives

    def get_component_summary(self) -> Dict[str, int]:
        """Get count of components by type"""
        summary = defaultdict(int)
        for ref in self.components.keys():
            prefix = ''.join(c for c in ref if c.isalpha())
            summary[prefix] += 1
        return dict(summary)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_si_value(value_str: str) -> float:
    """
    Parse SI-prefixed value string to float.

    Examples:
        "10k" -> 10000.0
        "4.7u" -> 0.0000047
        "100n" -> 0.0000001
    """
    multipliers = {
        'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'μ': 1e-6,
        'm': 1e-3, 'k': 1e3, 'K': 1e3, 'M': 1e6, 'G': 1e9
    }

    value_str = value_str.strip()

    # Find the multiplier character
    for suffix, mult in multipliers.items():
        if suffix in value_str:
            num_str = value_str.replace(suffix, '')
            try:
                return float(num_str) * mult
            except ValueError:
                break

    try:
        return float(value_str)
    except ValueError:
        return 0.0


def format_si_value(value: float, unit: str = '') -> str:
    """
    Format float value to SI-prefixed string.

    Examples:
        format_si_value(10000, 'Ω') -> "10kΩ"
        format_si_value(0.0000047, 'F') -> "4.7µF"
    """
    prefixes = [
        (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'),
        (1, ''), (1e-3, 'm'), (1e-6, 'µ'), (1e-9, 'n'), (1e-12, 'p')
    ]

    if value == 0:
        return f"0{unit}"

    for threshold, prefix in prefixes:
        if abs(value) >= threshold:
            formatted = value / threshold
            if formatted == int(formatted):
                return f"{int(formatted)}{prefix}{unit}"
            return f"{formatted:.2g}{prefix}{unit}"

    return f"{value}{unit}"
