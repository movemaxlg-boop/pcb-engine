"""
KiCad .kicad_mod Footprint Parser
==================================

Parses KiCad 9.0 footprint files (.kicad_mod S-expression format) and
extracts pad positions, sizes, courtyard bounds, and body outline into
the engine's FootprintDefinition dataclass.

This is the bridge between KiCad's 15,415 footprint library and our engine.

Usage:
    from pcb_engine.kicad_footprint_parser import KiCadFootprintParser

    fp_def = KiCadFootprintParser.parse_file(
        'C:/Program Files/KiCad/9.0/share/kicad/footprints/'
        'Package_SO.pretty/SOIC-8_3.9x4.9mm_P1.27mm.kicad_mod'
    )
    print(fp_def.name)           # 'SOIC-8_3.9x4.9mm_P1.27mm'
    print(len(fp_def.pad_positions))  # 8
    print(fp_def.body_width)     # 3.9
    print(fp_def.courtyard_size) # (7.4, 4.92)
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ParsedPad:
    """A single pad extracted from a .kicad_mod file."""
    number: str          # Pad number ("1", "2", "33", "")
    pad_type: str        # "smd" or "thru_hole" or "np_thru_hole"
    shape: str           # "roundrect", "circle", "rect", "oval", "custom"
    at_x: float          # X offset from component origin (mm)
    at_y: float          # Y offset from component origin (mm)
    size_x: float        # Pad width (mm)
    size_y: float        # Pad height (mm)
    drill: float = 0.0   # Drill diameter (THT only, mm)
    layers: str = ""     # Layer string (e.g., "F.Cu" or "*.Cu")
    is_copper: bool = True  # Has copper layer (not paste-only)


class KiCadFootprintParser:
    """
    Parses .kicad_mod files into FootprintDefinition objects.

    Uses regex-based extraction — the .kicad_mod format is machine-generated
    and regular enough that targeted regex works reliably for the specific
    elements we need (pads, courtyard, body, attr).
    """

    # Regex patterns for extracting S-expression data
    # Pad: (pad "NUM" TYPE SHAPE (at X Y [rot]) (size W H) ... (layers ...) ...)
    _PAD_RE = re.compile(
        r'\(pad\s+"([^"]*)"\s+'              # pad number (may be empty for paste pads)
        r'(\w+)\s+'                           # type: smd, thru_hole, np_thru_hole
        r'(\w+)\s+'                           # shape: roundrect, circle, rect, oval, custom
        r'(.*?)\)\s*\)',                       # rest of pad content until closing
        re.DOTALL
    )

    # Pad (at X Y) or (at X Y rotation)
    _AT_RE = re.compile(r'\(at\s+(-?[\d.]+)\s+(-?[\d.]+)(?:\s+[\d.]+)?\)')

    # Pad (size W H)
    _SIZE_RE = re.compile(r'\(size\s+(-?[\d.]+)\s+(-?[\d.]+)\)')

    # Pad (drill D) or (drill oval DX DY)
    _DRILL_RE = re.compile(r'\(drill\s+(?:oval\s+)?(-?[\d.]+)')

    # Pad (layers "...")
    _LAYERS_RE = re.compile(r'\(layers\s+([^)]+)\)')

    # fp_rect on specific layer: (fp_rect (start X1 Y1) (end X2 Y2) ... (layer "LAYER"))
    _FP_RECT_RE = re.compile(
        r'\(fp_rect\s+'
        r'\(start\s+(-?[\d.]+)\s+(-?[\d.]+)\)\s+'
        r'\(end\s+(-?[\d.]+)\s+(-?[\d.]+)\)\s+'
        r'.*?\(layer\s+"([^"]+)"\)',
        re.DOTALL
    )

    # fp_line on specific layer: (fp_line (start X1 Y1) (end X2 Y2) ... (layer "LAYER"))
    _FP_LINE_RE = re.compile(
        r'\(fp_line\s+'
        r'\(start\s+(-?[\d.]+)\s+(-?[\d.]+)\)\s+'
        r'\(end\s+(-?[\d.]+)\s+(-?[\d.]+)\)\s+'
        r'.*?\(layer\s+"([^"]+)"\)',
        re.DOTALL
    )

    # fp_poly pts on specific layer
    _FP_POLY_RE = re.compile(
        r'\(fp_poly\s+\(pts\s+(.*?)\)\s+'
        r'.*?\(layer\s+"([^"]+)"\)',
        re.DOTALL
    )

    # xy coordinates within pts
    _XY_RE = re.compile(r'\(xy\s+(-?[\d.]+)\s+(-?[\d.]+)\)')

    # Attribute: (attr smd) or (attr through_hole)
    _ATTR_RE = re.compile(r'\(attr\s+(\w+)\)')

    # Footprint name from first line
    _NAME_RE = re.compile(r'\(footprint\s+"([^"]+)"')

    @classmethod
    def parse_file(cls, filepath: str) -> 'FootprintDefinition':
        """
        Parse a .kicad_mod file and return a FootprintDefinition.

        Args:
            filepath: Path to the .kicad_mod file

        Returns:
            FootprintDefinition with accurate pad positions, body, courtyard

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file can't be parsed
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return cls.parse_content(content)

    @classmethod
    def parse_content(cls, content: str) -> 'FootprintDefinition':
        """
        Parse .kicad_mod content string and return a FootprintDefinition.

        Args:
            content: The full text content of a .kicad_mod file

        Returns:
            FootprintDefinition
        """
        # Import here to avoid circular imports
        from .common_types import FootprintDefinition

        # 1. Extract footprint name
        name_match = cls._NAME_RE.search(content)
        name = name_match.group(1) if name_match else 'unknown'

        # 2. Extract attribute (smd vs through_hole)
        attr_match = cls._ATTR_RE.search(content)
        is_smd = True  # default to SMD
        if attr_match:
            is_smd = attr_match.group(1) != 'through_hole'

        # 3. Extract all pads
        pads = cls._extract_pads(content)

        # 4. Filter to copper pads with pin numbers (skip paste-only pads)
        numbered_pads = [p for p in pads if p.number and p.is_copper]

        # 5. Extract body outline from F.Fab layer
        body_bounds = cls._extract_layer_bounds(content, 'F.Fab')

        # 6. Extract courtyard from F.CrtYd layer
        courtyard_bounds = cls._extract_layer_bounds(content, 'F.CrtYd')

        # 7. Compute body dimensions
        if body_bounds:
            body_width = body_bounds[2] - body_bounds[0]   # max_x - min_x
            body_height = body_bounds[3] - body_bounds[1]  # max_y - min_y
        elif courtyard_bounds:
            # Approximate body from courtyard minus margin
            body_width = max(0.5, (courtyard_bounds[2] - courtyard_bounds[0]) - 0.5)
            body_height = max(0.5, (courtyard_bounds[3] - courtyard_bounds[1]) - 0.5)
        elif numbered_pads:
            # Compute from pad bounding box
            min_x = min(p.at_x - p.size_x / 2 for p in numbered_pads)
            max_x = max(p.at_x + p.size_x / 2 for p in numbered_pads)
            min_y = min(p.at_y - p.size_y / 2 for p in numbered_pads)
            max_y = max(p.at_y + p.size_y / 2 for p in numbered_pads)
            body_width = max_x - min_x
            body_height = max_y - min_y
        else:
            body_width = 2.0
            body_height = 1.0

        # 8. Build pad_positions list for FootprintDefinition
        # Format: [(pin_number, x_offset, y_offset, pad_width, pad_height), ...]
        pad_positions = [
            (p.number, p.at_x, p.at_y, p.size_x, p.size_y)
            for p in numbered_pads
        ]

        return FootprintDefinition(
            name=name,
            body_width=round(body_width, 4),
            body_height=round(body_height, 4),
            pad_positions=pad_positions,
            is_smd=is_smd,
        )

    @classmethod
    def _extract_pads(cls, content: str) -> List[ParsedPad]:
        """Extract all pad definitions from file content."""
        pads = []

        # Use a more robust approach: find each (pad ...) block
        # We can't use the simple regex for the full pad because of nested parens
        # Instead, find pad start positions and extract balanced parens
        pad_starts = [m.start() for m in re.finditer(r'\(pad\s+"', content)]

        for start in pad_starts:
            # Extract balanced parentheses block
            block = cls._extract_balanced(content, start)
            if not block:
                continue

            # Parse pad number
            num_match = re.match(r'\(pad\s+"([^"]*)"', block)
            if num_match is None:
                continue
            number = num_match.group(1)

            # Parse pad type and shape
            type_match = re.match(r'\(pad\s+"[^"]*"\s+(\w+)\s+(\w+)', block)
            if type_match is None:
                continue
            pad_type = type_match.group(1)
            shape = type_match.group(2)

            # Parse position
            at_match = cls._AT_RE.search(block)
            if not at_match:
                continue
            at_x = float(at_match.group(1))
            at_y = float(at_match.group(2))

            # Parse size
            size_match = cls._SIZE_RE.search(block)
            if not size_match:
                continue
            size_x = float(size_match.group(1))
            size_y = float(size_match.group(2))

            # Parse drill (optional, THT only)
            drill = 0.0
            drill_match = cls._DRILL_RE.search(block)
            if drill_match:
                drill = float(drill_match.group(1))

            # Parse layers
            layers_str = ""
            layers_match = cls._LAYERS_RE.search(block)
            if layers_match:
                layers_str = layers_match.group(1)

            # Determine if pad has copper (not paste-only)
            is_copper = ('F.Cu' in layers_str or 'B.Cu' in layers_str
                        or '*.Cu' in layers_str)

            pads.append(ParsedPad(
                number=number,
                pad_type=pad_type,
                shape=shape,
                at_x=at_x,
                at_y=at_y,
                size_x=size_x,
                size_y=size_y,
                drill=drill,
                layers=layers_str,
                is_copper=is_copper,
            ))

        return pads

    @classmethod
    def _extract_balanced(cls, text: str, start: int) -> Optional[str]:
        """Extract a balanced parentheses block starting at position `start`."""
        if start >= len(text) or text[start] != '(':
            return None

        depth = 0
        i = start
        in_string = False

        while i < len(text):
            c = text[i]
            if c == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
            elif not in_string:
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            i += 1

        return None  # Unbalanced

    @classmethod
    def _extract_layer_bounds(
        cls, content: str, layer: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Extract bounding box of geometry on a specific layer.

        Handles:
        - fp_rect: direct rectangle bounds
        - fp_line: accumulate min/max of all line endpoints
        - fp_poly: accumulate min/max of all polygon vertices

        Returns:
            (min_x, min_y, max_x, max_y) or None if no geometry found
        """
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        found = False

        # Check fp_rect elements
        for match in cls._FP_RECT_RE.finditer(content):
            if match.group(5) == layer:
                x1, y1 = float(match.group(1)), float(match.group(2))
                x2, y2 = float(match.group(3)), float(match.group(4))
                min_x = min(min_x, x1, x2)
                min_y = min(min_y, y1, y2)
                max_x = max(max_x, x1, x2)
                max_y = max(max_y, y1, y2)
                found = True

        # Check fp_line elements
        for match in cls._FP_LINE_RE.finditer(content):
            if match.group(5) == layer:
                x1, y1 = float(match.group(1)), float(match.group(2))
                x2, y2 = float(match.group(3)), float(match.group(4))
                min_x = min(min_x, x1, x2)
                min_y = min(min_y, y1, y2)
                max_x = max(max_x, x1, x2)
                max_y = max(max_y, y1, y2)
                found = True

        # Check fp_poly elements
        for match in cls._FP_POLY_RE.finditer(content):
            if match.group(2) == layer:
                pts_str = match.group(1)
                for xy_match in cls._XY_RE.finditer(pts_str):
                    x, y = float(xy_match.group(1)), float(xy_match.group(2))
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    found = True

        if found:
            return (min_x, min_y, max_x, max_y)
        return None
