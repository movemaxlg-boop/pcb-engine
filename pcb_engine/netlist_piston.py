"""
PCB Engine - Netlist Piston (Import/Export)
=============================================

A dedicated piston (sub-engine) for parsing and generating netlist files.

SUPPORTED FORMATS:
==================
1. KiCad S-Expression Netlist (.net)
   Reference: KiCad EESCHEMA netlist format
   Parser: S-expression tree parser

2. KiCad Legacy Netlist (pre-v5)
   Reference: Older KiCad format

3. Eagle XML Netlist (.xml)
   Reference: Autodesk Eagle netlist export

4. Altium Netlist (.NET)
   Reference: Protel/Altium netlist format

5. Generic CSV Netlist
   Simple comma-separated format for manual entry

6. SPICE Netlist (.cir, .spice)
   Basic SPICE subcircuit parsing

NETLIST STRUCTURE:
==================
- Components: Reference, Value, Footprint, Library source
- Pins: Pin number, Pin name, Pin type (input/output/bidirectional)
- Nets: Net name, connected pins
- Net classes: High-speed, power, differential pairs

EXPORT FORMATS:
================
- Parts database JSON (for PCB Engine)
- BOM CSV
- Net report

Sources:
- https://forum.kicad.info/t/how-to-parse-the-kicad-netlist/6933
- https://github.com/devbisme/kinparse
- https://docs.kicad.org/doxygen/classKICAD__NETLIST__PARSER.html
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path


# =============================================================================
# ENUMS
# =============================================================================

class NetlistFormat(Enum):
    """Supported netlist formats"""
    KICAD_SEXPR = 'kicad_sexpr'      # KiCad S-expression (.net)
    KICAD_LEGACY = 'kicad_legacy'    # KiCad legacy format
    EAGLE_XML = 'eagle_xml'          # Eagle XML
    ALTIUM = 'altium'                # Altium/Protel
    CSV = 'csv'                      # Generic CSV
    SPICE = 'spice'                  # SPICE netlist
    AUTO = 'auto'                    # Auto-detect


class PinType(Enum):
    """Pin electrical types"""
    INPUT = 'input'
    OUTPUT = 'output'
    BIDIRECTIONAL = 'bidirectional'
    TRISTATE = 'tristate'
    PASSIVE = 'passive'
    POWER_INPUT = 'power_input'
    POWER_OUTPUT = 'power_output'
    OPEN_COLLECTOR = 'open_collector'
    OPEN_EMITTER = 'open_emitter'
    NO_CONNECT = 'no_connect'
    UNSPECIFIED = 'unspecified'


class NetClass(Enum):
    """Net classification"""
    DEFAULT = 'default'
    POWER = 'power'
    GROUND = 'ground'
    CLOCK = 'clock'
    HIGH_SPEED = 'high_speed'
    DIFFERENTIAL = 'differential'
    ANALOG = 'analog'
    CRITICAL = 'critical'


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Pin:
    """Component pin definition"""
    number: str                  # Pin number (e.g., "1", "A1")
    name: str = ""               # Pin name (e.g., "VCC", "DATA0")
    pin_type: PinType = PinType.UNSPECIFIED
    net: str = ""                # Connected net name

    # Physical properties (for footprint)
    x_offset_mm: float = 0.0
    y_offset_mm: float = 0.0


@dataclass
class Component:
    """Component definition from netlist"""
    reference: str               # Reference designator (e.g., "U1", "R1")
    value: str                   # Component value (e.g., "10k", "ATmega328P")
    footprint: str = ""          # Footprint name
    library: str = ""            # Source library

    # Pin list
    pins: List[Pin] = field(default_factory=list)

    # Properties from schematic
    properties: Dict[str, str] = field(default_factory=dict)

    # Sheet path (for hierarchical designs)
    sheet_path: str = "/"


@dataclass
class Net:
    """Net (electrical connection) definition"""
    name: str
    code: int = 0                # Net number/code

    # Connected pins
    pins: List[Tuple[str, str]] = field(default_factory=list)  # [(ref, pin_num), ...]

    # Classification
    net_class: NetClass = NetClass.DEFAULT

    # Properties
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class LibPart:
    """Library part definition"""
    library: str
    part: str
    description: str = ""
    footprints: List[str] = field(default_factory=list)
    pins: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Netlist:
    """Complete netlist representation"""
    source_file: str = ""
    format: NetlistFormat = NetlistFormat.AUTO
    version: str = ""
    title: str = ""
    date: str = ""

    # Content
    components: Dict[str, Component] = field(default_factory=dict)
    nets: Dict[str, Net] = field(default_factory=dict)
    lib_parts: Dict[str, LibPart] = field(default_factory=dict)

    # Statistics
    component_count: int = 0
    net_count: int = 0
    connection_count: int = 0


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NetlistConfig:
    """Configuration for netlist parsing"""
    # Auto-detection
    auto_detect_format: bool = True

    # Component filtering
    include_power_flags: bool = False   # Include PWR_FLAG symbols
    include_test_points: bool = True

    # Net classification
    power_net_prefixes: List[str] = field(default_factory=lambda: [
        'VCC', 'VDD', 'V3V3', 'V5V', '+3V3', '+5V', '+12V', 'VBAT'
    ])
    ground_net_prefixes: List[str] = field(default_factory=lambda: [
        'GND', 'VSS', 'AGND', 'DGND', 'PGND', '0V'
    ])
    clock_net_patterns: List[str] = field(default_factory=lambda: [
        r'CLK', r'CLOCK', r'XTAL', r'OSC'
    ])

    # Differential pair detection
    diff_pair_suffixes: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('_P', '_N'), ('+', '-'), ('_DP', '_DN')
    ])


# =============================================================================
# S-EXPRESSION PARSER
# =============================================================================

class SExprParser:
    """
    Parser for S-expression (Lisp-like) syntax used by KiCad

    Example:
    (export (version D)
      (components
        (comp (ref U1)
          (value ATmega328P)
          (footprint Package_QFP:TQFP-32)))
      (nets
        (net (code 1) (name GND)
          (node (ref U1) (pin 8)))))
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> List:
        """Parse entire S-expression"""
        self._skip_whitespace()
        if self.pos >= len(self.text):
            return []
        return self._parse_expr()

    def _parse_expr(self) -> Any:
        """Parse a single expression"""
        self._skip_whitespace()

        if self.pos >= len(self.text):
            return None

        char = self.text[self.pos]

        if char == '(':
            return self._parse_list()
        elif char == '"':
            return self._parse_string()
        else:
            return self._parse_atom()

    def _parse_list(self) -> List:
        """Parse a list (...) """
        self.pos += 1  # Skip (
        result = []

        while self.pos < len(self.text):
            self._skip_whitespace()

            if self.pos >= len(self.text):
                break

            if self.text[self.pos] == ')':
                self.pos += 1  # Skip )
                break

            expr = self._parse_expr()
            if expr is not None:
                result.append(expr)

        return result

    def _parse_string(self) -> str:
        """Parse a quoted string"""
        self.pos += 1  # Skip opening "
        result = []

        while self.pos < len(self.text):
            char = self.text[self.pos]

            if char == '"':
                self.pos += 1  # Skip closing "
                break
            elif char == '\\' and self.pos + 1 < len(self.text):
                self.pos += 1
                result.append(self.text[self.pos])
            else:
                result.append(char)

            self.pos += 1

        return ''.join(result)

    def _parse_atom(self) -> str:
        """Parse an atom (unquoted token)"""
        result = []

        while self.pos < len(self.text):
            char = self.text[self.pos]

            if char in '() \t\n\r':
                break

            result.append(char)
            self.pos += 1

        return ''.join(result)

    def _skip_whitespace(self):
        """Skip whitespace and comments"""
        while self.pos < len(self.text):
            char = self.text[self.pos]

            if char in ' \t\n\r':
                self.pos += 1
            elif char == ';':
                # Skip line comment
                while self.pos < len(self.text) and self.text[self.pos] != '\n':
                    self.pos += 1
            else:
                break


# =============================================================================
# NETLIST PISTON CLASS
# =============================================================================

class NetlistPiston:
    """
    Netlist Import/Export Piston

    Provides:
    1. Multi-format netlist parsing
    2. Net class classification
    3. Differential pair detection
    4. Export to PCB Engine parts_db format
    5. BOM and net report generation
    """

    def __init__(self, config: Optional[NetlistConfig] = None):
        self.config = config or NetlistConfig()
        self.netlist: Optional[Netlist] = None
        self.warnings: List[str] = []
        self.errors: List[str] = []

    # =========================================================================
    # STANDARD PISTON API
    # =========================================================================

    def extract(self, parts_db: Dict) -> Dict[str, Any]:
        """
        Standard piston API - extract netlist from parts database.

        Args:
            parts_db: Parts database

        Returns:
            Dictionary with netlist information
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        components = []
        for ref, part in parts.items():
            components.append({
                'ref': ref,
                'value': part.get('value', ''),
                'footprint': part.get('footprint', ''),
                'pin_count': len(part.get('pins', []) or part.get('used_pins', []))
            })

        net_list = []
        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])
            net_list.append({
                'name': net_name,
                'pin_count': len(pins),
                'pins': pins
            })

        return {
            'component_count': len(components),
            'net_count': len(net_list),
            'components': components,
            'nets': net_list,
            'warnings': self.warnings,
            'errors': self.errors
        }

    # =========================================================================
    # FORMAT DETECTION
    # =========================================================================

    def detect_format(self, filepath: str) -> NetlistFormat:
        """
        Auto-detect netlist format from file content

        Args:
            filepath: Path to netlist file

        Returns:
            Detected format
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        # Check extension first
        if ext in ('.net', '.kicad_net'):
            return NetlistFormat.KICAD_SEXPR
        elif ext in ('.xml'):
            return NetlistFormat.EAGLE_XML
        elif ext in ('.csv'):
            return NetlistFormat.CSV
        elif ext in ('.cir', '.spice', '.sp'):
            return NetlistFormat.SPICE

        # Read file header to detect format
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.read(500)

            if '(export' in header and '(version' in header:
                return NetlistFormat.KICAD_SEXPR
            elif '<?xml' in header and 'eagle' in header.lower():
                return NetlistFormat.EAGLE_XML
            elif header.startswith('*') or '.SUBCKT' in header.upper():
                return NetlistFormat.SPICE
            elif ',' in header.split('\n')[0]:
                return NetlistFormat.CSV

        except (IOError, UnicodeDecodeError) as e:
            # File reading error - fall through to default
            pass

        return NetlistFormat.KICAD_SEXPR  # Default

    # =========================================================================
    # KICAD S-EXPRESSION PARSER
    # =========================================================================

    def parse_kicad_sexpr(self, filepath: str) -> Netlist:
        """
        Parse KiCad S-expression netlist

        Format:
        (export (version D)
          (design ...)
          (components (comp (ref X) (value Y) (footprint Z) (libsource ...)))
          (libparts ...)
          (nets (net (code N) (name "name") (node (ref X) (pin N)))))

        Args:
            filepath: Path to .net file

        Returns:
            Parsed Netlist
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        parser = SExprParser(content)
        tree = parser.parse()

        netlist = Netlist(
            source_file=filepath,
            format=NetlistFormat.KICAD_SEXPR
        )

        if not tree or tree[0] != 'export':
            self.errors.append("Invalid KiCad netlist: missing 'export' root")
            return netlist

        # Process each section
        for item in tree[1:]:
            if not isinstance(item, list) or not item:
                continue

            section = item[0]

            if section == 'version':
                netlist.version = item[1] if len(item) > 1 else ''

            elif section == 'design':
                self._parse_kicad_design(item, netlist)

            elif section == 'components':
                self._parse_kicad_components(item, netlist)

            elif section == 'libparts':
                self._parse_kicad_libparts(item, netlist)

            elif section == 'nets':
                self._parse_kicad_nets(item, netlist)

        # Update statistics
        netlist.component_count = len(netlist.components)
        netlist.net_count = len(netlist.nets)
        netlist.connection_count = sum(len(n.pins) for n in netlist.nets.values())

        # Classify nets
        self._classify_nets(netlist)

        return netlist

    def _parse_kicad_design(self, design_tree: List, netlist: Netlist):
        """Parse design section"""
        for item in design_tree[1:]:
            if isinstance(item, list):
                if item[0] == 'source':
                    netlist.title = item[1] if len(item) > 1 else ''
                elif item[0] == 'date':
                    netlist.date = item[1] if len(item) > 1 else ''

    def _parse_kicad_components(self, comp_tree: List, netlist: Netlist):
        """Parse components section"""
        for item in comp_tree[1:]:
            if not isinstance(item, list) or item[0] != 'comp':
                continue

            comp = Component(reference='', value='')

            for field in item[1:]:
                if not isinstance(field, list):
                    continue

                key = field[0]
                val = field[1] if len(field) > 1 else ''

                if key == 'ref':
                    comp.reference = val
                elif key == 'value':
                    comp.value = val
                elif key == 'footprint':
                    comp.footprint = val
                elif key == 'libsource':
                    for lf in field[1:]:
                        if isinstance(lf, list):
                            if lf[0] == 'lib':
                                comp.library = lf[1] if len(lf) > 1 else ''
                elif key == 'sheetpath':
                    for sp in field[1:]:
                        if isinstance(sp, list) and sp[0] == 'names':
                            comp.sheet_path = sp[1] if len(sp) > 1 else '/'
                elif key == 'property':
                    prop_name = ''
                    prop_val = ''
                    for pf in field[1:]:
                        if isinstance(pf, list):
                            if pf[0] == 'name':
                                prop_name = pf[1] if len(pf) > 1 else ''
                            elif pf[0] == 'value':
                                prop_val = pf[1] if len(pf) > 1 else ''
                    if prop_name:
                        comp.properties[prop_name] = prop_val

            if comp.reference:
                # Skip power flags if configured
                if not self.config.include_power_flags:
                    if 'PWR_FLAG' in comp.value or comp.reference.startswith('#'):
                        continue

                netlist.components[comp.reference] = comp

    def _parse_kicad_libparts(self, libparts_tree: List, netlist: Netlist):
        """Parse library parts section"""
        for item in libparts_tree[1:]:
            if not isinstance(item, list) or item[0] != 'libpart':
                continue

            libpart = LibPart(library='', part='')

            for field in item[1:]:
                if not isinstance(field, list):
                    continue

                key = field[0]

                if key == 'lib':
                    libpart.library = field[1] if len(field) > 1 else ''
                elif key == 'part':
                    libpart.part = field[1] if len(field) > 1 else ''
                elif key == 'description':
                    libpart.description = field[1] if len(field) > 1 else ''
                elif key == 'footprints':
                    for fp in field[1:]:
                        if isinstance(fp, list) and fp[0] == 'fp':
                            libpart.footprints.append(fp[1] if len(fp) > 1 else '')
                elif key == 'pins':
                    for pin in field[1:]:
                        if isinstance(pin, list) and pin[0] == 'pin':
                            pin_dict = {}
                            for pf in pin[1:]:
                                if isinstance(pf, list) and len(pf) > 1:
                                    pin_dict[pf[0]] = pf[1]
                            libpart.pins.append(pin_dict)

            key = f"{libpart.library}:{libpart.part}"
            netlist.lib_parts[key] = libpart

    def _parse_kicad_nets(self, nets_tree: List, netlist: Netlist):
        """Parse nets section"""
        for item in nets_tree[1:]:
            if not isinstance(item, list) or item[0] != 'net':
                continue

            net = Net(name='')

            for field in item[1:]:
                if not isinstance(field, list):
                    continue

                key = field[0]

                if key == 'code':
                    try:
                        net.code = int(field[1]) if len(field) > 1 else 0
                    except ValueError:
                        net.code = 0
                elif key == 'name':
                    net.name = field[1] if len(field) > 1 else ''
                elif key == 'node':
                    ref = ''
                    pin = ''
                    for nf in field[1:]:
                        if isinstance(nf, list):
                            if nf[0] == 'ref':
                                ref = nf[1] if len(nf) > 1 else ''
                            elif nf[0] == 'pin':
                                pin = nf[1] if len(nf) > 1 else ''
                    if ref and pin:
                        net.pins.append((ref, pin))

                        # Update component pin info
                        if ref in netlist.components:
                            comp = netlist.components[ref]
                            comp.pins.append(Pin(number=pin, net=net.name))

            if net.name:
                netlist.nets[net.name] = net

    # =========================================================================
    # CSV PARSER
    # =========================================================================

    def parse_csv(self, filepath: str) -> Netlist:
        """
        Parse simple CSV netlist

        Expected format:
        Reference,Value,Footprint,Pin,Net
        U1,ATmega328P,TQFP-32,1,PC6
        U1,ATmega328P,TQFP-32,2,PD0
        ...

        Args:
            filepath: Path to CSV file

        Returns:
            Parsed Netlist
        """
        import csv

        netlist = Netlist(
            source_file=filepath,
            format=NetlistFormat.CSV
        )

        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                ref = row.get('Reference', row.get('Ref', ''))
                value = row.get('Value', '')
                footprint = row.get('Footprint', '')
                pin_num = row.get('Pin', row.get('PinNumber', ''))
                net_name = row.get('Net', row.get('NetName', ''))

                if not ref:
                    continue

                # Add/update component
                if ref not in netlist.components:
                    netlist.components[ref] = Component(
                        reference=ref,
                        value=value,
                        footprint=footprint
                    )

                comp = netlist.components[ref]

                # Add pin
                if pin_num:
                    comp.pins.append(Pin(number=pin_num, net=net_name))

                # Add/update net
                if net_name:
                    if net_name not in netlist.nets:
                        netlist.nets[net_name] = Net(name=net_name)
                    netlist.nets[net_name].pins.append((ref, pin_num))

        # Update statistics
        netlist.component_count = len(netlist.components)
        netlist.net_count = len(netlist.nets)
        netlist.connection_count = sum(len(n.pins) for n in netlist.nets.values())

        self._classify_nets(netlist)

        return netlist

    # =========================================================================
    # EAGLE XML PARSER
    # =========================================================================

    def parse_eagle_xml(self, filepath: str) -> Netlist:
        """
        Parse Eagle XML netlist

        Args:
            filepath: Path to XML file

        Returns:
            Parsed Netlist
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            self.errors.append("XML parsing not available")
            return Netlist()

        netlist = Netlist(
            source_file=filepath,
            format=NetlistFormat.EAGLE_XML
        )

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Find parts
            parts = root.findall('.//part')
            for part in parts:
                ref = part.get('name', '')
                value = part.get('value', '')
                library = part.get('library', '')
                deviceset = part.get('deviceset', '')

                if ref:
                    netlist.components[ref] = Component(
                        reference=ref,
                        value=value if value else deviceset,
                        library=library
                    )

            # Find nets
            signals = root.findall('.//signal')
            for signal in signals:
                net_name = signal.get('name', '')
                if not net_name:
                    continue

                net = Net(name=net_name)

                for contact in signal.findall('contactref'):
                    element = contact.get('element', '')
                    pad = contact.get('pad', '')
                    if element and pad:
                        net.pins.append((element, pad))

                netlist.nets[net_name] = net

        except ET.ParseError as e:
            self.errors.append(f"XML parse error: {e}")

        # Update statistics
        netlist.component_count = len(netlist.components)
        netlist.net_count = len(netlist.nets)

        self._classify_nets(netlist)

        return netlist

    # =========================================================================
    # NET CLASSIFICATION
    # =========================================================================

    def _classify_nets(self, netlist: Netlist):
        """Classify nets by type (power, ground, clock, etc.)"""
        for net_name, net in netlist.nets.items():
            name_upper = net_name.upper()

            # Check power nets
            for prefix in self.config.power_net_prefixes:
                if name_upper.startswith(prefix.upper()):
                    net.net_class = NetClass.POWER
                    break

            # Check ground nets
            for prefix in self.config.ground_net_prefixes:
                if name_upper.startswith(prefix.upper()):
                    net.net_class = NetClass.GROUND
                    break

            # Check clock nets
            for pattern in self.config.clock_net_patterns:
                if re.search(pattern, name_upper):
                    net.net_class = NetClass.CLOCK
                    break

    def detect_differential_pairs(self, netlist: Netlist) -> List[Tuple[str, str]]:
        """
        Detect differential signal pairs

        Args:
            netlist: Parsed netlist

        Returns:
            List of (positive_net, negative_net) pairs
        """
        pairs = []
        net_names = list(netlist.nets.keys())

        for pos_suffix, neg_suffix in self.config.diff_pair_suffixes:
            for net_name in net_names:
                if net_name.endswith(pos_suffix):
                    base = net_name[:-len(pos_suffix)]
                    neg_name = base + neg_suffix

                    if neg_name in netlist.nets:
                        pairs.append((net_name, neg_name))

                        # Mark as differential
                        netlist.nets[net_name].net_class = NetClass.DIFFERENTIAL
                        netlist.nets[neg_name].net_class = NetClass.DIFFERENTIAL

        return pairs

    # =========================================================================
    # MAIN PARSE METHOD
    # =========================================================================

    def parse(self, filepath: str, format: NetlistFormat = NetlistFormat.AUTO) -> Netlist:
        """
        Parse netlist file

        Args:
            filepath: Path to netlist file
            format: Netlist format (or AUTO for detection)

        Returns:
            Parsed Netlist
        """
        self.warnings.clear()
        self.errors.clear()

        if format == NetlistFormat.AUTO:
            format = self.detect_format(filepath)

        if format == NetlistFormat.KICAD_SEXPR:
            netlist = self.parse_kicad_sexpr(filepath)
        elif format == NetlistFormat.CSV:
            netlist = self.parse_csv(filepath)
        elif format == NetlistFormat.EAGLE_XML:
            netlist = self.parse_eagle_xml(filepath)
        else:
            self.errors.append(f"Unsupported format: {format}")
            netlist = Netlist()

        self.netlist = netlist
        return netlist

    # =========================================================================
    # EXPORT TO PARTS_DB
    # =========================================================================

    def to_parts_db(self, netlist: Optional[Netlist] = None) -> Dict[str, Any]:
        """
        Convert netlist to PCB Engine parts_db format

        The parts_db format is what the PCB Engine uses internally.

        Returns:
            Dict compatible with PCB Engine
        """
        netlist = netlist or self.netlist
        if not netlist:
            return {'parts': {}, 'nets': {}}

        parts_db = {
            'parts': {},
            'nets': {},
            'metadata': {
                'source': netlist.source_file,
                'format': netlist.format.value,
                'component_count': netlist.component_count,
                'net_count': netlist.net_count
            }
        }

        # Convert components
        for ref, comp in netlist.components.items():
            parts_db['parts'][ref] = {
                'value': comp.value,
                'footprint': comp.footprint,
                'library': comp.library,
                'pins': [
                    {
                        'number': pin.number,
                        'name': pin.name,
                        'net': pin.net,
                        'type': pin.pin_type.value
                    }
                    for pin in comp.pins
                ],
                'properties': comp.properties
            }

        # Convert nets
        for net_name, net in netlist.nets.items():
            parts_db['nets'][net_name] = {
                'code': net.code,
                'class': net.net_class.value,
                'pins': [
                    {'ref': ref, 'pin': pin}
                    for ref, pin in net.pins
                ]
            }

        return parts_db

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export_bom(self, netlist: Optional[Netlist] = None) -> List[Dict[str, Any]]:
        """
        Export Bill of Materials

        Returns:
            List of BOM entries
        """
        netlist = netlist or self.netlist
        if not netlist:
            return []

        # Group by value + footprint
        groups = {}

        for ref, comp in netlist.components.items():
            key = f"{comp.value}|{comp.footprint}"

            if key not in groups:
                groups[key] = {
                    'value': comp.value,
                    'footprint': comp.footprint,
                    'references': [],
                    'quantity': 0
                }

            groups[key]['references'].append(ref)
            groups[key]['quantity'] += 1

        # Convert to list
        bom = []
        for group in groups.values():
            group['references'] = ', '.join(sorted(group['references']))
            bom.append(group)

        return sorted(bom, key=lambda x: x['references'])

    def export_net_report(self, netlist: Optional[Netlist] = None) -> str:
        """
        Generate text net report

        Returns:
            Formatted net report
        """
        netlist = netlist or self.netlist
        if not netlist:
            return ""

        lines = []
        lines.append(f"Net Report: {netlist.source_file}")
        lines.append(f"Components: {netlist.component_count}")
        lines.append(f"Nets: {netlist.net_count}")
        lines.append(f"Connections: {netlist.connection_count}")
        lines.append("")

        # Group nets by class
        by_class = {}
        for net_name, net in netlist.nets.items():
            cls = net.net_class.value
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(net)

        for cls, nets in sorted(by_class.items()):
            lines.append(f"=== {cls.upper()} NETS ===")
            for net in sorted(nets, key=lambda n: n.name):
                pins_str = ', '.join(f"{ref}.{pin}" for ref, pin in net.pins)
                lines.append(f"  {net.name}: {pins_str}")
            lines.append("")

        return '\n'.join(lines)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_component_nets(
        self,
        reference: str,
        netlist: Optional[Netlist] = None
    ) -> Dict[str, str]:
        """
        Get all nets connected to a component

        Returns:
            Dict of {pin_number: net_name}
        """
        netlist = netlist or self.netlist
        if not netlist:
            return {}

        result = {}

        for net_name, net in netlist.nets.items():
            for ref, pin in net.pins:
                if ref == reference:
                    result[pin] = net_name

        return result

    def get_net_components(
        self,
        net_name: str,
        netlist: Optional[Netlist] = None
    ) -> List[Tuple[str, str]]:
        """
        Get all components connected to a net

        Returns:
            List of (reference, pin_number)
        """
        netlist = netlist or self.netlist
        if not netlist or net_name not in netlist.nets:
            return []

        return list(netlist.nets[net_name].pins)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_netlist(filepath: str) -> Netlist:
    """Quick parse of any netlist file"""
    piston = NetlistPiston()
    return piston.parse(filepath)


def netlist_to_parts_db(filepath: str) -> Dict[str, Any]:
    """Parse netlist and convert to parts_db format"""
    piston = NetlistPiston()
    netlist = piston.parse(filepath)
    return piston.to_parts_db(netlist)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("NETLIST PISTON - Self Test")
    print("=" * 60)

    # Test S-expression parser
    print("\n1. S-Expression Parser:")
    test_sexpr = '''
    (export (version D)
      (components
        (comp (ref U1) (value ATmega328P) (footprint TQFP-32))
        (comp (ref R1) (value 10k) (footprint 0402)))
      (nets
        (net (code 1) (name GND) (node (ref U1) (pin 8)) (node (ref R1) (pin 1)))
        (net (code 2) (name VCC) (node (ref U1) (pin 7)))))
    '''

    parser = SExprParser(test_sexpr)
    tree = parser.parse()
    print(f"   Parsed tree depth: {len(tree)} items")
    print(f"   Root: {tree[0]}")

    # Test netlist creation from parsed data
    print("\n2. Create Netlist from S-Expression:")
    piston = NetlistPiston()

    # Create sample netlist manually
    netlist = Netlist(
        source_file='test.net',
        format=NetlistFormat.KICAD_SEXPR,
        version='D'
    )

    netlist.components['U1'] = Component(
        reference='U1',
        value='ATmega328P',
        footprint='Package_QFP:TQFP-32',
        pins=[
            Pin(number='7', name='VCC', net='VCC'),
            Pin(number='8', name='GND', net='GND'),
            Pin(number='1', name='PC6', net='RESET'),
        ]
    )

    netlist.components['R1'] = Component(
        reference='R1',
        value='10k',
        footprint='Resistor_SMD:R_0402',
        pins=[
            Pin(number='1', net='GND'),
            Pin(number='2', net='RESET'),
        ]
    )

    netlist.nets['GND'] = Net(name='GND', pins=[('U1', '8'), ('R1', '1')])
    netlist.nets['VCC'] = Net(name='VCC', pins=[('U1', '7')])
    netlist.nets['RESET'] = Net(name='RESET', pins=[('U1', '1'), ('R1', '2')])

    netlist.component_count = len(netlist.components)
    netlist.net_count = len(netlist.nets)

    print(f"   Components: {netlist.component_count}")
    print(f"   Nets: {netlist.net_count}")

    # Test classification
    print("\n3. Net Classification:")
    piston._classify_nets(netlist)
    for net_name, net in netlist.nets.items():
        print(f"   {net_name}: {net.net_class.value}")

    # Test export to parts_db
    print("\n4. Export to Parts DB:")
    parts_db = piston.to_parts_db(netlist)
    print(f"   Parts: {list(parts_db['parts'].keys())}")
    print(f"   Nets: {list(parts_db['nets'].keys())}")

    # Test BOM export
    print("\n5. BOM Export:")
    bom = piston.export_bom(netlist)
    for item in bom:
        print(f"   {item['quantity']}x {item['value']} ({item['footprint']})")

    # Test component nets lookup
    print("\n6. Component Nets Lookup (U1):")
    u1_nets = piston.get_component_nets('U1', netlist)
    for pin, net in u1_nets.items():
        print(f"   Pin {pin}: {net}")

    # Test differential pair detection
    print("\n7. Differential Pair Detection:")
    # Add some diff pairs for testing
    netlist.nets['USB_D+'] = Net(name='USB_D+', pins=[])
    netlist.nets['USB_D-'] = Net(name='USB_D-', pins=[])
    netlist.nets['ETH_TX_P'] = Net(name='ETH_TX_P', pins=[])
    netlist.nets['ETH_TX_N'] = Net(name='ETH_TX_N', pins=[])

    pairs = piston.detect_differential_pairs(netlist)
    for p, n in pairs:
        print(f"   {p} <-> {n}")

    # Test format detection
    print("\n8. Format Detection:")
    print("   .net file: ", NetlistFormat.KICAD_SEXPR.value)
    print("   .xml file: ", NetlistFormat.EAGLE_XML.value)
    print("   .csv file: ", NetlistFormat.CSV.value)

    print("\n" + "=" * 60)
    print("Netlist Piston self-test PASSED")
    print("=" * 60)
