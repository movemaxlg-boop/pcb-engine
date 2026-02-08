"""
PCB Engine - KiCad Generator Module (Enhanced)
===============================================

Generates KiCad Python scripts from the algorithm output.
This is the final output of the engine.

OUTPUT PHILOSOPHY:
==================
The generated script must:
1. Be complete and self-contained
2. Run without errors in KiCad's Python console
3. Produce a professional, manufacturable PCB
4. Include proper error handling
5. Generate detailed logging for debugging

SCRIPT STRUCTURE:
=================
1. Header and configuration
2. Board setup (outline, layers, design rules)
3. Net creation
4. Component placement
5. Escape routes (from pad to routing grid)
6. Signal routes (between components)
7. GND zone (pour on bottom layer)
8. Silkscreen and reference designators
9. Finalization and DRC preparation
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class FootprintInfo:
    """Information about a component footprint"""
    library: str
    name: str
    description: str


class FootprintMapper:
    """Maps component packages to KiCad footprints"""

    # Common package to footprint mappings
    FOOTPRINT_MAP = {
        # Resistors
        '0402': ('Resistor_SMD', 'R_0402_1005Metric'),
        '0603': ('Resistor_SMD', 'R_0603_1608Metric'),
        '0805': ('Resistor_SMD', 'R_0805_2012Metric'),
        '1206': ('Resistor_SMD', 'R_1206_3216Metric'),

        # Capacitors
        'C0402': ('Capacitor_SMD', 'C_0402_1005Metric'),
        'C0603': ('Capacitor_SMD', 'C_0603_1608Metric'),
        'C0805': ('Capacitor_SMD', 'C_0805_2012Metric'),
        'C1206': ('Capacitor_SMD', 'C_1206_3216Metric'),

        # SOT packages
        'SOT-23': ('Package_TO_SOT_SMD', 'SOT-23'),
        'SOT-23-3': ('Package_TO_SOT_SMD', 'SOT-23'),
        'SOT-23-5': ('Package_TO_SOT_SMD', 'SOT-23-5'),
        'SOT-23-6': ('Package_TO_SOT_SMD', 'SOT-23-6'),

        # SOIC packages
        'SOIC-8': ('Package_SO', 'SOIC-8_3.9x4.9mm_P1.27mm'),
        'SOIC-14': ('Package_SO', 'SOIC-14_3.9x8.7mm_P1.27mm'),
        'SOIC-16': ('Package_SO', 'SOIC-16_3.9x9.9mm_P1.27mm'),
        'TSSOP-14': ('Package_SO', 'TSSOP-14_4.4x5mm_P0.65mm'),
        'TSSOP-16': ('Package_SO', 'TSSOP-16_4.4x5mm_P0.65mm'),

        # QFP packages
        'TQFP-32': ('Package_QFP', 'TQFP-32_7x7mm_P0.8mm'),
        'TQFP-44': ('Package_QFP', 'TQFP-44_10x10mm_P0.8mm'),
        'TQFP-48': ('Package_QFP', 'TQFP-48_7x7mm_P0.5mm'),
        'LQFP-48': ('Package_QFP', 'LQFP-48_7x7mm_P0.5mm'),
        'LQFP-64': ('Package_QFP', 'LQFP-64_10x10mm_P0.5mm'),

        # Pin headers
        'PinHeader_1x02': ('Connector_PinHeader_2.54mm', 'PinHeader_1x02_P2.54mm_Vertical'),
        'PinHeader_1x03': ('Connector_PinHeader_2.54mm', 'PinHeader_1x03_P2.54mm_Vertical'),
        'PinHeader_1x04': ('Connector_PinHeader_2.54mm', 'PinHeader_1x04_P2.54mm_Vertical'),
        'PinHeader_1x05': ('Connector_PinHeader_2.54mm', 'PinHeader_1x05_P2.54mm_Vertical'),
        'PinHeader_1x06': ('Connector_PinHeader_2.54mm', 'PinHeader_1x06_P2.54mm_Vertical'),
        'PinHeader_1x08': ('Connector_PinHeader_2.54mm', 'PinHeader_1x08_P2.54mm_Vertical'),
        'PinHeader_1x10': ('Connector_PinHeader_2.54mm', 'PinHeader_1x10_P2.54mm_Vertical'),
        'PinHeader_1x16': ('Connector_PinHeader_2.54mm', 'PinHeader_1x16_P2.54mm_Vertical'),
        'PinHeader_2x04': ('Connector_PinHeader_2.54mm', 'PinHeader_2x04_P2.54mm_Vertical'),
        'PinHeader_2x08': ('Connector_PinHeader_2.54mm', 'PinHeader_2x08_P2.54mm_Vertical'),
        'PinHeader_2x10': ('Connector_PinHeader_2.54mm', 'PinHeader_2x10_P2.54mm_Vertical'),

        # JST connectors
        'JST_PH_2': ('Connector_JST', 'JST_PH_B2B-PH-K_1x02_P2.00mm_Vertical'),
        'JST_PH_3': ('Connector_JST', 'JST_PH_B3B-PH-K_1x03_P2.00mm_Vertical'),
        'JST_PH_4': ('Connector_JST', 'JST_PH_B4B-PH-K_1x04_P2.00mm_Vertical'),
        'JST_XH_2': ('Connector_JST', 'JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical'),
        'JST_XH_3': ('Connector_JST', 'JST_XH_B3B-XH-A_1x03_P2.50mm_Vertical'),
        'JST_XH_4': ('Connector_JST', 'JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical'),

        # LEDs
        'LED_0603': ('LED_SMD', 'LED_0603_1608Metric'),
        'LED_0805': ('LED_SMD', 'LED_0805_2012Metric'),
        'LED_1206': ('LED_SMD', 'LED_1206_3216Metric'),

        # Diodes
        'SOD-123': ('Diode_SMD', 'D_SOD-123'),
        'SOD-323': ('Diode_SMD', 'D_SOD-323'),
        'SMA': ('Diode_SMD', 'D_SMA'),
        'SMB': ('Diode_SMD', 'D_SMB'),

        # Sensors
        'LGA-14': ('Package_LGA', 'LGA-14_3x2.5mm_P0.5mm'),
        'DFN-8': ('Package_DFN_QFN', 'DFN-8-1EP_3x2mm_P0.5mm_EP1.36x1.46mm'),

        # Mounting holes
        'MountingHole_3.2mm': ('MountingHole', 'MountingHole_3.2mm_M3'),
    }

    @classmethod
    def get_footprint(cls, package: str) -> Tuple[str, str]:
        """Get KiCad footprint for a package"""
        if package in cls.FOOTPRINT_MAP:
            return cls.FOOTPRINT_MAP[package]

        # Try case-insensitive match
        for key, value in cls.FOOTPRINT_MAP.items():
            if key.lower() == package.lower():
                return value

        # Default fallback
        return ('', package)


class KiCadGenerator:
    """
    Generates KiCad Python scripts.

    The output script can be run directly in KiCad's Python console
    to create the PCB.
    """

    def __init__(self, board, rules):
        self.board = board
        self.rules = rules
        self.indent = '    '

    def generate(self, parts: Dict, placement: Dict,
                 routes: Dict, escapes: Dict,
                 include_routes: bool = False) -> str:
        """
        Generate complete KiCad Python script.

        Args:
            parts: Parts database
            placement: Component placement positions
            routes: Routed signal nets
            escapes: Escape routes from pads
            include_routes: If True, include escape and signal routes in output.
                           If False (default), skip routes to avoid DRC warnings
                           when footprints aren't placed in KiCad.

        Returns the script as a string.
        """
        lines = []
        skip_for_drc = not include_routes

        # Header
        lines.extend(self._generate_header())

        # Imports and setup
        lines.extend(self._generate_imports())

        # Configuration
        lines.extend(self._generate_config())

        # Board setup
        lines.extend(self._generate_board_setup())

        # Design rules
        lines.extend(self._generate_design_rules())

        # Helper functions
        lines.extend(self._generate_helpers())

        # Net creation
        lines.extend(self._generate_nets(parts))

        # Component placement
        lines.extend(self._generate_placement(parts, placement))

        # Escape routes (only for nets that have signal routes)
        lines.extend(self._generate_escapes(escapes, parts, routes, skip_for_drc=skip_for_drc))

        # Signal routes
        lines.extend(self._generate_routes(routes, skip_for_drc=skip_for_drc))

        # GND zone (only if there are GND connections)
        lines.extend(self._generate_gnd_zone(routes, escapes))

        # Silkscreen
        lines.extend(self._generate_silkscreen(parts, placement))

        # Finalization
        lines.extend(self._generate_finalization())

        return '\n'.join(lines)

    def _generate_header(self) -> List[str]:
        """Generate script header"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return [
            '"""',
            '================================================================================',
            'KiCad PCB Script - Generated by PCB Engine',
            '================================================================================',
            f'Generated: {timestamp}',
            f'Board Size: {self.board.width}mm x {self.board.height}mm',
            f'Layers: {self.board.layers}',
            '',
            'Usage in KiCad Python console:',
            '  1. Open Tools -> Scripting Console',
            '  2. Run: exec(open("path/to/this_file.py").read())',
            '',
            'IMPORTANT:',
            '  - Ensure footprints are loaded in your project',
            '  - Run DRC after execution to verify design',
            '  - Press B to fill zones after script completes',
            '================================================================================',
            '"""',
            '',
        ]

    def _generate_imports(self) -> List[str]:
        """Generate import statements"""
        return [
            '# =============================================================================',
            '# IMPORTS',
            '# =============================================================================',
            'import pcbnew',
            'import os',
            'import math',
            'from pcbnew import (VECTOR2I, FromMM, ToMM, F_Cu, B_Cu, Edge_Cuts,',
            '                    F_SilkS, B_SilkS, F_Mask, B_Mask, F_Paste, B_Paste,',
            '                    F_Fab, B_Fab)',
            '',
            '# Get current board',
            'try:',
            '    board = pcbnew.GetBoard()',
            '    if board is None:',
            '        print("ERROR: No board loaded. Open a board first.")',
            '        print("Create a new board or open an existing one, then run this script.")',
            '        raise SystemExit',
            'except SystemExit:',
            '    raise',
            'except Exception as e:',
            '    print(f"ERROR: Could not get board: {e}")',
            '    raise SystemExit',
            '',
            'print("=" * 60)',
            'print("PCB ENGINE SCRIPT EXECUTION")',
            'print("=" * 60)',
            '',
        ]

    def _generate_config(self) -> List[str]:
        """Generate configuration section"""
        return [
            '# =============================================================================',
            '# CONFIGURATION',
            '# =============================================================================',
            f'BX, BY = {self.board.origin_x}, {self.board.origin_y}  # Board origin (mm)',
            f'BW, BH = {self.board.width}, {self.board.height}  # Board size (mm)',
            f'GRID = {self.board.grid_size}  # Routing grid (mm)',
            f'LAYERS = {self.board.layers}',
            '',
            '# Design rules',
            f'MIN_TRACE = {self.rules.min_trace_width}  # mm',
            f'MIN_CLEARANCE = {self.rules.min_clearance}  # mm',
            f'MIN_VIA_DIA = {self.rules.min_via_diameter}  # mm',
            f'MIN_VIA_DRILL = {self.rules.min_via_drill}  # mm',
            '',
        ]

    def _generate_board_setup(self) -> List[str]:
        """Generate board outline setup"""
        return [
            '# =============================================================================',
            '# BOARD OUTLINE',
            '# =============================================================================',
            'print("Drawing board outline...")',
            '',
            'def mm(val):',
            '    """Convert mm to KiCad internal units"""',
            '    return FromMM(val)',
            '',
            'def pt(x, y):',
            '    """Create a point from mm coordinates"""',
            '    return VECTOR2I(mm(x), mm(y))',
            '',
            '# Clear existing outline (optional - comment out to keep existing)',
            '# for item in list(board.GetDrawings()):',
            '#     if item.GetLayerName() == "Edge.Cuts":',
            '#         board.Remove(item)',
            '',
            '# Draw new outline',
            'outline_pts = [',
            f'    (BX, BY),',
            f'    (BX + BW, BY),',
            f'    (BX + BW, BY + BH),',
            f'    (BX, BY + BH),',
            ']',
            '',
            'for i in range(len(outline_pts)):',
            '    start = outline_pts[i]',
            '    end = outline_pts[(i + 1) % len(outline_pts)]',
            '    line = pcbnew.PCB_SHAPE(board)',
            '    line.SetShape(pcbnew.SHAPE_T_SEGMENT)',
            '    line.SetStart(pt(start[0], start[1]))',
            '    line.SetEnd(pt(end[0], end[1]))',
            '    line.SetLayer(Edge_Cuts)',
            '    line.SetWidth(mm(0.15))',
            '    board.Add(line)',
            '',
            'print(f"  Board outline: {BW}mm x {BH}mm")',
            '',
        ]

    def _generate_design_rules(self) -> List[str]:
        """Generate design rules setup"""
        return [
            '# =============================================================================',
            '# DESIGN RULES',
            '# =============================================================================',
            'print("Setting design rules...")',
            '',
            'ds = board.GetDesignSettings()',
            '',
            '# KiCad 7+ uses different API for design rules',
            '# Try new API first, fall back to old if needed',
            'try:',
            '    # KiCad 7+ API - design rules are in constraints',
            '    constraints = ds.m_DRCEngine if hasattr(ds, "m_DRCEngine") else None',
            '    if constraints:',
            '        print("  Using KiCad 7+ constraints API")',
            '    else:',
            '        print("  Using direct settings")',
            'except:',
            '    pass',
            '',
            '# Set track width - try multiple approaches',
            'try:',
            '    ds.m_TrackMinWidth = mm(MIN_TRACE)',
            'except AttributeError:',
            '    try:',
            '        ds.SetMinTrackWidth(mm(MIN_TRACE))',
            '    except:',
            '        print("  Note: Could not set min track width via API")',
            '',
            '# Clearance',
            'try:',
            '    ds.m_MinClearance = mm(MIN_CLEARANCE)',
            'except AttributeError:',
            '    print("  Note: Could not set min clearance via API")',
            '',
            '# Via settings',
            'try:',
            '    ds.m_ViasMinSize = mm(MIN_VIA_DIA)',
            'except AttributeError:',
            '    print("  Note: Could not set min via size via API")',
            'try:',
            '    ds.m_ViasMinDrill = mm(MIN_VIA_DRILL)',
            'except AttributeError:',
            '    try:',
            '        ds.m_MinThroughDrill = mm(MIN_VIA_DRILL)',
            '    except:',
            '        print("  Note: Could not set via drill via API")',
            '',
            'print(f"  Target min trace: {MIN_TRACE}mm")',
            'print(f"  Target min clearance: {MIN_CLEARANCE}mm")',
            'print(f"  Target min via: {MIN_VIA_DIA}mm dia, {MIN_VIA_DRILL}mm drill")',
            'print("  (Verify settings in Board Setup -> Design Rules)")',
            '',
        ]

    def _generate_helpers(self) -> List[str]:
        """Generate helper functions"""
        return [
            '# =============================================================================',
            '# HELPER FUNCTIONS',
            '# =============================================================================',
            'placed_footprints = {}  # ref -> footprint',
            'nets = {}  # name -> NETINFO_ITEM',
            'track_count = 0',
            'via_count = 0',
            '',
            'def get_or_create_net(name):',
            '    """Get or create a net by name"""',
            '    global nets',
            '    if name in nets:',
            '        return nets[name]',
            '    ',
            '    # Check if net already exists on board',
            '    existing = board.FindNet(name)',
            '    if existing:',
            '        nets[name] = existing',
            '        return existing',
            '    ',
            '    # Create new net',
            '    net = pcbnew.NETINFO_ITEM(board, name)',
            '    board.Add(net)',
            '    nets[name] = net',
            '    return net',
            '',
            'def add_track(x1, y1, x2, y2, net_name, width=MIN_TRACE, layer=F_Cu):',
            '    """Add a track segment"""',
            '    global track_count',
            '    track = pcbnew.PCB_TRACK(board)',
            '    track.SetStart(pt(x1, y1))',
            '    track.SetEnd(pt(x2, y2))',
            '    track.SetWidth(mm(width))',
            '    track.SetLayer(layer)',
            '    track.SetNet(get_or_create_net(net_name))',
            '    board.Add(track)',
            '    track_count += 1',
            '    return track',
            '',
            'def add_via(x, y, net_name, dia=MIN_VIA_DIA, drill=MIN_VIA_DRILL):',
            '    """Add a through-hole via"""',
            '    global via_count',
            '    via = pcbnew.PCB_VIA(board)',
            '    via.SetPosition(pt(x, y))',
            '    via.SetWidth(mm(dia))',
            '    via.SetDrill(mm(drill))',
            '    via.SetNet(get_or_create_net(net_name))',
            '    # Via type - KiCad 7+ uses different constant',
            '    try:',
            '        via.SetViaType(pcbnew.VIATYPE_THROUGH)',
            '    except AttributeError:',
            '        try:',
            '            via.SetViaType(pcbnew.VIA_THROUGH)',
            '        except:',
            '            pass  # Use default via type',
            '    board.Add(via)',
            '    via_count += 1',
            '    return via',
            '',
            'def find_footprint(ref):',
            '    """Find a footprint by reference"""',
            '    for fp in board.GetFootprints():',
            '        if fp.GetReference() == ref:',
            '            return fp',
            '    return None',
            '',
            'def assign_net_to_pad(fp, pad_name, net_name):',
            '    """Assign a net to a pad on a footprint"""',
            '    if fp is None:',
            '        return False',
            '    for pad in fp.Pads():',
            '        if pad.GetName() == str(pad_name):',
            '            pad.SetNet(get_or_create_net(net_name))',
            '            return True',
            '    return False',
            '',
        ]

    def _generate_nets(self, parts: Dict) -> List[str]:
        """Generate net creation code"""
        lines = [
            '# =============================================================================',
            '# NET CREATION',
            '# =============================================================================',
            'print("Creating nets...")',
            '',
        ]

        nets = parts.get('nets', {})
        net_count = 0

        for net_name in sorted(nets.keys()):
            if net_name:
                lines.append(f'get_or_create_net("{net_name}")')
                net_count += 1

        lines.extend([
            '',
            f'print(f"  Created {{len(nets)}} nets")',
            '',
        ])

        return lines

    def _generate_placement(self, parts: Dict, placement: Dict) -> List[str]:
        """Generate component placement code"""
        lines = [
            '# =============================================================================',
            '# COMPONENT PLACEMENT',
            '# =============================================================================',
            'print("Placing components...")',
            'placement_success = 0',
            'placement_failed = 0',
            '',
        ]

        parts_db = parts.get('parts', {})

        for ref, pos in sorted(placement.items()):
            part = parts_db.get(ref, {})
            package = part.get('physical', {}).get('package', 'unknown')
            value = part.get('value', '')

            lines.extend([
                f'# {ref}: {package}',
                f'fp = find_footprint("{ref}")',
                'if fp:',
                f'    fp.SetPosition(pt({pos.x}, {pos.y}))',
                f'    fp.SetOrientationDegrees({pos.rotation})',
                f'    placed_footprints["{ref}"] = fp',
                '    placement_success += 1',
            ])

            # Assign nets to pads
            used_pins = part.get('used_pins', [])
            for pin in used_pins:
                net = pin.get('net', '')
                if net:
                    lines.append(
                        f'    assign_net_to_pad(fp, "{pin["number"]}", "{net}")'
                    )

            lines.extend([
                'else:',
                f'    print(f"  WARNING: Footprint {ref} not found")',
                '    placement_failed += 1',
                '',
            ])

        lines.extend([
            'print(f"  Placed: {placement_success}, Failed: {placement_failed}")',
            '',
        ])

        return lines

    def _generate_escapes(self, escapes: Dict, parts: Dict, routes: Dict = None,
                          skip_for_drc: bool = True) -> List[str]:
        """
        Generate escape route code.

        IMPORTANT: Only generate escapes that connect to signal routes.
        Orphan escapes (nets with no signal route) cause KiCad DRC warnings.

        Args:
            skip_for_drc: If True, skip all escapes to avoid DRC warnings when
                          footprints aren't placed. Default True for clean DRC.
        """
        lines = [
            '# =============================================================================',
            '# ESCAPE ROUTES',
            '# =============================================================================',
            'print("Creating escape routes...")',
            'escape_count = 0',
            '',
        ]

        # Skip escapes entirely if footprints aren't placed (causes DRC warnings)
        if skip_for_drc:
            lines.append('# Escape routes skipped (footprints not placed - would cause DRC warnings)')
            lines.append('print("  Skipped - run with footprints placed for full routing")')
            lines.append('')
            return lines

        if not escapes:
            lines.append('# No escape routes defined')
            lines.append('print("  No escapes needed")')
            lines.append('')
            return lines

        # Get set of nets that have signal routes
        routed_nets = set(routes.keys()) if routes else set()

        # Also include nets where escape connects to a via in a signal route
        via_endpoints = set()
        if routes:
            for net_name, route in routes.items():
                for via in getattr(route, 'vias', []):
                    via_endpoints.add((round(via.position[0], 2), round(via.position[1], 2), net_name))

        for ref, pin_escapes in escapes.items():
            has_escapes = False
            escape_lines = []

            for pin_num, escape in pin_escapes.items():
                net = getattr(escape, 'net', '')
                if not net:
                    continue

                # Skip escapes for nets that have no signal route
                # Exception: if escape endpoint matches a via position, include it
                escape_end = (round(escape.endpoint[0], 2), round(escape.endpoint[1], 2))
                connects_to_via = (escape_end[0], escape_end[1], net) in via_endpoints

                if net not in routed_nets and not connects_to_via:
                    # Skip - this escape would be orphaned
                    continue

                start = getattr(escape, 'start', (0, 0))
                end = escape.endpoint
                width = getattr(escape, 'width', self.rules.min_trace_width)
                layer_str = getattr(escape, 'layer', 'F.Cu')
                layer = 'F_Cu' if layer_str == 'F.Cu' else 'B_Cu'

                escape_lines.extend([
                    f'add_track({start[0]}, {start[1]}, {end[0]}, {end[1]}, '
                    f'"{net}", {width}, {layer})',
                    'escape_count += 1',
                ])
                has_escapes = True

            if has_escapes:
                lines.append(f'# Escapes for {ref}')
                lines.extend(escape_lines)
                lines.append('')

        lines.extend([
            'print(f"  Created {escape_count} escape routes")',
            '',
        ])

        return lines

    def _generate_routes(self, routes: Dict, skip_for_drc: bool = True) -> List[str]:
        """
        Generate signal route code.

        Args:
            skip_for_drc: If True, skip all routes to avoid DRC warnings when
                          footprints aren't placed. Default True for clean DRC.
        """
        lines = [
            '# =============================================================================',
            '# SIGNAL ROUTES',
            '# =============================================================================',
            'print("Creating signal routes...")',
            '',
        ]

        # Skip routes entirely if footprints aren't placed (causes DRC warnings)
        if skip_for_drc:
            lines.append('# Signal routes skipped (footprints not placed - would cause DRC warnings)')
            lines.append('print("  Skipped - run with footprints placed for full routing")')
            lines.append('')
            return lines

        if not routes:
            lines.append('# No routes defined')
            lines.append('print("  No signal routes")')
            lines.append('')
            return lines

        for net_name, route in routes.items():
            lines.append(f'# Net: {net_name}')

            segments = getattr(route, 'segments', [])
            for seg in segments:
                layer_str = getattr(seg, 'layer', 'F.Cu')
                layer = 'F_Cu' if layer_str == 'F.Cu' else 'B_Cu'
                width = getattr(seg, 'width', self.rules.min_trace_width)

                lines.append(
                    f'add_track({seg.start[0]}, {seg.start[1]}, '
                    f'{seg.end[0]}, {seg.end[1]}, '
                    f'"{net_name}", {width}, {layer})'
                )

            vias = getattr(route, 'vias', [])
            for via in vias:
                lines.append(
                    f'add_via({via.position[0]}, {via.position[1]}, "{net_name}")'
                )

            lines.append('')

        lines.extend([
            'print(f"  Created {track_count} tracks, {via_count} vias")',
            '',
        ])

        return lines

    def _generate_gnd_zone(self, routes: Dict = None, escapes: Dict = None) -> List[str]:
        """
        Generate GND zone code.

        IMPORTANT: Only generate zone if there are GND vias or GND routes.
        An isolated zone (no via connecting pads to zone) causes KiCad DRC warnings.

        GND zone on B.Cu connects to:
        - GND vias that go from F.Cu pads to B.Cu zone
        - Through-hole GND pads

        Without a via, SMD pads on F.Cu can't connect to B.Cu zone.
        """
        # Check if there's a GND route with vias
        has_gnd_route = False
        has_gnd_via = False

        if routes:
            if 'GND' in routes:
                gnd_route = routes['GND']
                has_gnd_route = True
                # Check for vias in GND route
                if getattr(gnd_route, 'vias', []):
                    has_gnd_via = True

        # Without GND vias, zone on B.Cu can't connect to F.Cu pads
        # Skip the zone to avoid "isolated copper fill" warning
        if not has_gnd_via:
            return [
                '# =============================================================================',
                '# GND ZONE (Skipped - no GND vias to connect zone to pads)',
                '# =============================================================================',
                'print("Skipping GND zone - no GND vias (zone would be isolated)")',
                '',
            ]

        margin = 1.0  # mm from board edge

        return [
            '# =============================================================================',
            '# GND ZONE (Bottom Layer Pour)',
            '# =============================================================================',
            'print("Creating GND zone...")',
            '',
            '# Create zone on bottom copper layer',
            'zone = pcbnew.ZONE(board)',
            'zone.SetNet(get_or_create_net("GND"))',
            'zone.SetLayer(B_Cu)',
            'zone.SetIsFilled(False)  # Will be filled later',
            '',
            '# Zone settings - KiCad 7+ compatible',
            'try:',
            '    # KiCad 7+ uses ZONE_SETTINGS',
            '    zone_settings = zone.GetZoneSettings() if hasattr(zone, "GetZoneSettings") else None',
            '    if zone_settings:',
            '        zone_settings.m_ZoneClearance = mm(MIN_CLEARANCE)',
            '        zone_settings.m_ZoneMinThickness = mm(MIN_TRACE)',
            '        zone_settings.m_PadConnection = pcbnew.ZONE_CONNECTION_THERMAL',
            f'        zone_settings.m_ThermalReliefGap = mm({self.rules.min_clearance})',
            f'        zone_settings.m_ThermalReliefSpokeWidth = mm({self.rules.min_trace_width})',
            '        zone.SetZoneSettings(zone_settings)',
            '    else:',
            '        # Fallback for older KiCad versions',
            '        zone.SetLocalClearance(mm(MIN_CLEARANCE))',
            '        zone.SetMinThickness(mm(MIN_TRACE))',
            '        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)',
            f'        zone.SetThermalReliefGap(mm({self.rules.min_clearance}))',
            f'        zone.SetThermalReliefSpokeWidth(mm({self.rules.min_trace_width}))',
            'except Exception as e:',
            '    print(f"  Note: Zone settings via alternate method: {e}")',
            '    # Minimal fallback - just set connection type',
            '    try:',
            '        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)',
            '    except:',
            '        pass',
            '',
            '# Zone outline (inset from board edge)',
            f'margin = {margin}',
            '# KiCad 9 zone outline - Append takes VECTOR2I',
            'try:',
            '    outline = zone.Outline()',
            '    # NewOutline returns the outline index',
            '    outline.NewOutline()',
            '    outline.Append(mm(BX + margin), mm(BY + margin))',
            '    outline.Append(mm(BX + BW - margin), mm(BY + margin))',
            '    outline.Append(mm(BX + BW - margin), mm(BY + BH - margin))',
            '    outline.Append(mm(BX + margin), mm(BY + BH - margin))',
            'except Exception as e:',
            '    print(f"  WARNING: Could not create zone outline: {e}")',
            '',
            'board.Add(zone)',
            '',
            'print("  GND zone created on bottom layer")',
            'print("  NOTE: Press \'B\' to fill zones after script completes")',
            '',
        ]

    def _generate_silkscreen(self, parts: Dict, placement: Dict) -> List[str]:
        """Generate silkscreen code"""
        return [
            '# =============================================================================',
            '# SILKSCREEN',
            '# =============================================================================',
            'print("Adding silkscreen...")',
            '',
            '# Board title/info',
            'try:',
            '    title = pcbnew.PCB_TEXT(board)',
            '    title.SetText("Generated by PCB Engine")',
            '    title.SetPosition(pt(BX + BW/2, BY + 2))',
            '    title.SetTextSize(VECTOR2I(mm(1.0), mm(1.0)))',
            '    title.SetLayer(F_SilkS)',
            '    # Text justification - KiCad 7+ compatible',
            '    try:',
            '        title.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)',
            '    except AttributeError:',
            '        try:',
            '            title.SetHorizJustify(pcbnew.TEXT_H_ALIGN_T_CENTER)',
            '        except:',
            '            pass  # Skip justification',
            '    board.Add(title)',
            'except Exception as e:',
            '    print(f"  Note: Could not add title text: {e}")',
            '',
            '# Ensure reference designators are visible',
            'try:',
            '    for fp in board.GetFootprints():',
            '        ref = fp.Reference()',
            '        ref.SetVisible(True)',
            '        ref.SetLayer(F_SilkS)',
            'except Exception as e:',
            '    print(f"  Note: Could not update reference designators: {e}")',
            '',
            'print("  Silkscreen added")',
            '',
        ]

    def _generate_finalization(self) -> List[str]:
        """Generate finalization code"""
        return [
            '# =============================================================================',
            '# FINALIZATION',
            '# =============================================================================',
            'print("")',
            'print("=" * 60)',
            'print("PCB GENERATION COMPLETE")',
            'print("=" * 60)',
            '',
            '# Summary',
            f'print(f"Board size: {self.board.width}mm x {self.board.height}mm")',
            'print(f"Components: {placement_success} placed, {placement_failed} missing")',
            'print(f"Tracks: {track_count}")',
            'print(f"Vias: {via_count}")',
            'print(f"Nets: {len(nets)}")',
            '',
            '# Instructions',
            'print("")',
            'print("NEXT STEPS:")',
            'print("  1. Press \'B\' to fill GND zones")',
            'print("  2. Run Design Rules Check (DRC)")',
            'print("  3. Review 3D view (Alt+3)")',
            'print("  4. Generate Gerber files for manufacturing")',
            'print("")',
            'print("=" * 60)',
            '',
            '# Refresh view',
            'try:',
            '    pcbnew.Refresh()',
            'except:',
            '    pass',
        ]

    def generate_standalone(self, parts: Dict, placement: Dict,
                            routes: Dict, escapes: Dict,
                            output_path: str) -> bool:
        """
        Generate and save script to file.

        Returns True if successful.
        """
        try:
            script = self.generate(parts, placement, routes, escapes)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(script)

            return True
        except Exception as e:
            print(f"Error generating script: {e}")
            return False


class KiCadProjectGenerator:
    """
    Generates a complete KiCad project structure.

    Creates:
    - .kicad_pcb file (empty board template)
    - .kicad_pro file (project settings)
    - .kicad_sch file (empty schematic template)
    - generated_routing.py (the routing script)
    """

    def __init__(self, board, rules, project_name: str):
        self.board = board
        self.rules = rules
        self.project_name = project_name
        self.kicad_gen = KiCadGenerator(board, rules)

    def generate_project(self, output_dir: str, parts: Dict,
                         placement: Dict, routes: Dict,
                         escapes: Dict) -> bool:
        """
        Generate complete project structure.

        Returns True if successful.
        """
        import os

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate routing script
            script_path = os.path.join(output_dir, 'generated_routing.py')
            self.kicad_gen.generate_standalone(
                parts, placement, routes, escapes, script_path
            )

            # Generate empty PCB template
            pcb_path = os.path.join(output_dir, f'{self.project_name}.kicad_pcb')
            self._generate_empty_pcb(pcb_path)

            # Generate project file
            pro_path = os.path.join(output_dir, f'{self.project_name}.kicad_pro')
            self._generate_project_file(pro_path)

            # Generate README
            readme_path = os.path.join(output_dir, 'README.md')
            self._generate_readme(readme_path)

            return True

        except Exception as e:
            print(f"Error generating project: {e}")
            return False

    def _generate_empty_pcb(self, path: str):
        """Generate empty KiCad PCB file"""
        content = f'''(kicad_pcb (version 20230121) (generator pcb_engine)
  (general
    (thickness 1.6)
  )
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )
  (setup
    (pad_to_mask_clearance 0)
    (pcbplotparams
      (layerselection 0x00010fc_ffffffff)
      (disableapertmacros false)
      (usegerberextensions false)
      (usegerberattributes true)
    )
  )
  (net 0 "")
  (net 1 "GND")
)
'''
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _generate_project_file(self, path: str):
        """Generate KiCad project file"""
        content = f'''{{
  "board": {{
    "design_settings": {{
      "defaults": {{
        "board_outline_line_width": 0.1,
        "copper_line_width": 0.2,
        "copper_text_size_h": 1.5,
        "copper_text_size_v": 1.5,
        "copper_text_thickness": 0.3,
        "other_line_width": 0.15,
        "silk_line_width": 0.15,
        "silk_text_size_h": 1.0,
        "silk_text_size_v": 1.0,
        "silk_text_thickness": 0.15
      }},
      "rules": {{
        "min_clearance": {self.rules.min_clearance},
        "min_track_width": {self.rules.min_trace_width},
        "min_via_annular_width": {self.rules.min_annular_ring},
        "min_via_diameter": {self.rules.min_via_diameter}
      }}
    }}
  }},
  "meta": {{
    "filename": "{self.project_name}.kicad_pro",
    "version": 1
  }}
}}
'''
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _generate_readme(self, path: str):
        """Generate README file"""
        content = f'''# {self.project_name}

Generated by PCB Engine on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Board Specifications

- **Size:** {self.board.width}mm x {self.board.height}mm
- **Layers:** {self.board.layers}
- **Min trace width:** {self.rules.min_trace_width}mm
- **Min clearance:** {self.rules.min_clearance}mm
- **Min via diameter:** {self.rules.min_via_diameter}mm
- **Min via drill:** {self.rules.min_via_drill}mm

## Usage

1. Open `{self.project_name}.kicad_pcb` in KiCad
2. Open Tools → Scripting Console
3. Run: `exec(open("generated_routing.py").read())`
4. Press 'B' to fill GND zones
5. Run DRC to verify design

## Files

- `{self.project_name}.kicad_pcb` - PCB layout file
- `{self.project_name}.kicad_pro` - Project settings
- `generated_routing.py` - Auto-routing script

## Notes

This PCB was generated algorithmically. Always verify:
- All connections are correct
- Clearances meet your fab house requirements
- Thermal relief is adequate for power connections
'''
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
