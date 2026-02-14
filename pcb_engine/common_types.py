"""
PCB Engine - Common Types
==========================

Shared type definitions used across the PCB engine.
This module provides canonical types to prevent mismatches between pistons.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any, Union, Iterator


@dataclass
class Position:
    """
    Component position that supports BOTH attribute and index access.

    This class unifies two position formats that were causing bugs:
    - Tuple format: (x, y) - used by some placement code
    - Object format: pos.x, pos.y - used by other placement code

    Now both work:
        pos = Position(10.0, 20.0)
        x, y = pos           # Unpacking works
        x = pos[0]           # Index access works
        x = pos.x            # Attribute access works
    """
    x: float
    y: float
    rotation: float = 0.0
    layer: str = 'F.Cu'

    def __getitem__(self, idx: int) -> float:
        """Support tuple-style access: pos[0], pos[1]"""
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        else:
            raise IndexError(f"Position index out of range: {idx}")

    def __iter__(self) -> Iterator[float]:
        """Support unpacking: x, y = pos"""
        yield self.x
        yield self.y

    def __len__(self) -> int:
        """Support len() for compatibility."""
        return 2

    def as_tuple(self) -> Tuple[float, float]:
        """Get position as a tuple."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y}, rotation={self.rotation}, layer='{self.layer}')"


def normalize_position(pos: Any) -> Position:
    """
    Convert any position format to a Position object.

    Handles:
    - Position objects (returned as-is)
    - Tuples/lists: (x, y) or [x, y]
    - Dicts: {'x': ..., 'y': ...}
    - Objects with .x and .y attributes

    Args:
        pos: Position in any supported format

    Returns:
        Position object

    Raises:
        TypeError: If format is not recognized
    """
    if pos is None:
        raise TypeError("Cannot convert None to Position")

    if isinstance(pos, Position):
        return pos

    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return Position(
            x=float(pos[0]),
            y=float(pos[1]),
            rotation=float(pos[2]) if len(pos) > 2 else 0.0
        )

    if isinstance(pos, dict):
        return Position(
            x=float(pos.get('x', 0.0)),
            y=float(pos.get('y', 0.0)),
            rotation=float(pos.get('rotation', 0.0)),
            layer=pos.get('layer', 'F.Cu')
        )

    if hasattr(pos, 'x') and hasattr(pos, 'y'):
        return Position(
            x=float(pos.x),
            y=float(pos.y),
            rotation=float(getattr(pos, 'rotation', 0.0)),
            layer=getattr(pos, 'layer', 'F.Cu')
        )

    raise TypeError(f"Cannot convert {type(pos).__name__} to Position: {pos}")


def get_xy(pos: Any) -> Tuple[float, float]:
    """
    Extract (x, y) tuple from any position format.

    This is a convenience function for code that just needs coordinates.

    Args:
        pos: Position in any supported format

    Returns:
        Tuple of (x, y)
    """
    if pos is None:
        raise TypeError("Cannot get coordinates from None")

    if isinstance(pos, Position):
        return (pos.x, pos.y)

    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return (float(pos[0]), float(pos[1]))

    if isinstance(pos, dict):
        return (float(pos.get('x', 0.0)), float(pos.get('y', 0.0)))

    if hasattr(pos, 'x') and hasattr(pos, 'y'):
        return (float(pos.x), float(pos.y))

    raise TypeError(f"Cannot extract coordinates from {type(pos).__name__}: {pos}")


# =============================================================================
# PIN/PAD UTILITIES (BUG-06 fix)
# =============================================================================

def get_pins(part: Dict) -> List[Dict]:
    """
    Get pins from a part, handling ALL legacy formats.

    The codebase uses inconsistent naming:
    - 'pins' (sometimes list, sometimes dict)
    - 'used_pins' (list format)
    - 'physical_pins' (list format)

    This function handles all of them.

    Args:
        part: Part dictionary from parts_db

    Returns:
        List of pin dicts with 'number', 'name', 'net', 'offset' keys
    """
    if not part:
        return []

    # Try standard list format first (most common)
    pins = part.get('pins', [])
    if isinstance(pins, list) and pins:
        return pins

    # Try used_pins list format (from processed parts)
    pins = part.get('used_pins', [])
    if isinstance(pins, list) and pins:
        return pins

    # Try physical_pins format
    pins = part.get('physical_pins', [])
    if isinstance(pins, list) and pins:
        return pins

    # Handle dict format (legacy): {'1': {'net': 'VCC', ...}, '2': {...}}
    pins_dict = part.get('pins', {})
    if isinstance(pins_dict, dict) and pins_dict:
        return [
            {'number': k, **v} for k, v in pins_dict.items()
        ]

    return []


def get_pin_net(part: Dict, pin_number: Union[str, int]) -> str:
    """
    Get the net name for a specific pin number.

    Args:
        part: Part dictionary from parts_db
        pin_number: Pin number (as string or int)

    Returns:
        Net name, or empty string if not found
    """
    pin_str = str(pin_number)

    for pin in get_pins(part):
        if str(pin.get('number', '')) == pin_str:
            return pin.get('net', '')

    return ''


def get_pin_offset(part: Dict, pin_number: Union[str, int]) -> Tuple[float, float]:
    """
    Get the offset for a specific pin (relative to component center).

    Args:
        part: Part dictionary from parts_db
        pin_number: Pin number (as string or int)

    Returns:
        Tuple of (offset_x, offset_y), or (0.0, 0.0) if not found
    """
    pin_str = str(pin_number)

    for pin in get_pins(part):
        if str(pin.get('number', '')) == pin_str:
            offset = pin.get('offset', (0.0, 0.0))
            if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                return (float(offset[0]), float(offset[1]))
            elif isinstance(offset, dict):
                return (float(offset.get('x', 0.0)), float(offset.get('y', 0.0)))

    return (0.0, 0.0)


def get_pin_position(
    part: Dict,
    pin_number: Union[str, int],
    component_pos: Any
) -> Tuple[float, float]:
    """
    Get the absolute position of a pin on the board.

    Args:
        part: Part dictionary from parts_db
        pin_number: Pin number
        component_pos: Component position (any format)

    Returns:
        Absolute (x, y) position of the pin
    """
    comp_x, comp_y = get_xy(component_pos)
    offset_x, offset_y = get_pin_offset(part, pin_number)
    return (comp_x + offset_x, comp_y + offset_y)


# =============================================================================
# GRID COORDINATE SYSTEM (FIX FOR COORDINATE TRUNCATION BUG)
# =============================================================================

@dataclass
class GridCoordinate:
    """
    Unified coordinate class that properly handles real<->grid conversions.

    This fixes the fundamental bug where int() truncation caused 0.05mm gaps
    between routes and pads. Now uses round() for proper conversion.

    Usage:
        coord = GridCoordinate.from_real(17.95, 14.5, grid_size=0.1)
        # coord.grid_x = 180 (rounded from 179.5)
        # coord.grid_y = 145
        # coord.real_x = 18.0 (grid-aligned)
        # coord.real_y = 14.5
        # coord.original_x = 17.95 (preserved)
    """
    grid_x: int
    grid_y: int
    grid_size: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    # Preserve original coordinates for pad stubs
    original_x: float = 0.0
    original_y: float = 0.0

    @classmethod
    def from_real(cls, x: float, y: float, grid_size: float,
                  origin_x: float = 0.0, origin_y: float = 0.0) -> 'GridCoordinate':
        """
        Create GridCoordinate from real (mm) coordinates.

        Uses round() instead of int() to prevent truncation errors.
        """
        grid_x = round((x - origin_x) / grid_size)
        grid_y = round((y - origin_y) / grid_size)
        return cls(
            grid_x=grid_x,
            grid_y=grid_y,
            grid_size=grid_size,
            origin_x=origin_x,
            origin_y=origin_y,
            original_x=x,
            original_y=y
        )

    @classmethod
    def from_grid(cls, grid_x: int, grid_y: int, grid_size: float,
                  origin_x: float = 0.0, origin_y: float = 0.0) -> 'GridCoordinate':
        """Create GridCoordinate from grid cell indices."""
        real_x = origin_x + grid_x * grid_size
        real_y = origin_y + grid_y * grid_size
        return cls(
            grid_x=grid_x,
            grid_y=grid_y,
            grid_size=grid_size,
            origin_x=origin_x,
            origin_y=origin_y,
            original_x=real_x,
            original_y=real_y
        )

    @property
    def real_x(self) -> float:
        """Get grid-aligned real X coordinate."""
        return self.origin_x + self.grid_x * self.grid_size

    @property
    def real_y(self) -> float:
        """Get grid-aligned real Y coordinate."""
        return self.origin_y + self.grid_y * self.grid_size

    @property
    def real(self) -> Tuple[float, float]:
        """Get grid-aligned real coordinates as tuple."""
        return (self.real_x, self.real_y)

    @property
    def grid(self) -> Tuple[int, int]:
        """Get grid coordinates as tuple."""
        return (self.grid_x, self.grid_y)

    @property
    def original(self) -> Tuple[float, float]:
        """Get original (non-grid-aligned) coordinates."""
        return (self.original_x, self.original_y)

    @property
    def offset_from_grid(self) -> Tuple[float, float]:
        """Get the offset from grid-aligned position to original position."""
        return (self.original_x - self.real_x, self.original_y - self.real_y)

    def distance_to(self, other: 'GridCoordinate') -> float:
        """Manhattan distance to another coordinate (in grid cells)."""
        return abs(self.grid_x - other.grid_x) + abs(self.grid_y - other.grid_y)

    def euclidean_distance_to(self, other: 'GridCoordinate') -> float:
        """Euclidean distance in mm."""
        dx = self.real_x - other.real_x
        dy = self.real_y - other.real_y
        return (dx*dx + dy*dy) ** 0.5


# =============================================================================
# UNIFIED FOOTPRINT DEFINITIONS
# =============================================================================
# Single source of truth for footprint dimensions, used by BOTH routing and output.
# This fixes the mismatch where routing used parts_db offsets but output used hardcoded values.

@dataclass
class FootprintDefinition:
    """Definition of a footprint's physical characteristics."""
    name: str
    body_width: float  # mm
    body_height: float  # mm
    pad_positions: List[Tuple[str, float, float, float, float]]  # (pin_num, x_offset, y_offset, pad_width, pad_height)
    is_smd: bool = True

    def get_pad_offset(self, pin_number: str) -> Tuple[float, float]:
        """Get pad offset for a pin number."""
        for pin, x, y, w, h in self.pad_positions:
            if pin == pin_number:
                return (x, y)
        return (0.0, 0.0)

    def get_pad_size(self, pin_number: str) -> Tuple[float, float]:
        """Get pad size for a pin number."""
        for pin, x, y, w, h in self.pad_positions:
            if pin == pin_number:
                return (w, h)
        return (1.0, 1.0)

    @property
    def courtyard_size(self) -> Tuple[float, float]:
        """Courtyard size — encloses body AND all pads, plus margin.

        For ICs like QFN, pads extend beyond the body, so the courtyard must
        cover the full pad bounding box, not just the body.
        """
        # Start with body dimensions
        extent_w = self.body_width
        extent_h = self.body_height

        # Expand to include pad bounding box (pad center + half pad size)
        if self.pad_positions:
            max_x = max(abs(px) + pw / 2 for _, px, _, pw, _ in self.pad_positions)
            max_y = max(abs(py) + ph / 2 for _, _, py, _, ph in self.pad_positions)
            extent_w = max(extent_w, max_x * 2)
            extent_h = max(extent_h, max_y * 2)

        margin = 0.25 if max(extent_w, extent_h) < 5.0 else 0.5
        return (round(extent_w + 2 * margin, 2), round(extent_h + 2 * margin, 2))

    @property
    def pitch(self) -> float:
        """Minimum pad pitch (center-to-center distance between adjacent pads)."""
        if len(self.pad_positions) < 2:
            return self.body_width
        positions = [(x, y) for _, x, y, _, _ in self.pad_positions]
        min_pitch = float('inf')
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < min_pitch:
                    min_pitch = dist
        return min_pitch if min_pitch < float('inf') else self.body_width


# Standard footprint library - THE source of truth
FOOTPRINT_LIBRARY: Dict[str, FootprintDefinition] = {
    # 0402 - Imperial (1005 Metric)
    '0402': FootprintDefinition(
        name='0402',
        body_width=1.0,
        body_height=0.5,
        pad_positions=[
            ('1', -0.48, 0.0, 0.56, 0.62),
            ('2', 0.48, 0.0, 0.56, 0.62),
        ],
        is_smd=True
    ),
    # 0603 - Imperial (1608 Metric)
    '0603': FootprintDefinition(
        name='0603',
        body_width=1.6,
        body_height=0.8,
        pad_positions=[
            ('1', -0.775, 0.0, 0.75, 0.9),
            ('2', 0.775, 0.0, 0.75, 0.9),
        ],
        is_smd=True
    ),
    # 0805 - Imperial (2012 Metric)
    '0805': FootprintDefinition(
        name='0805',
        body_width=2.0,
        body_height=1.25,
        pad_positions=[
            ('1', -0.95, 0.0, 0.9, 1.25),
            ('2', 0.95, 0.0, 0.9, 1.25),
        ],
        is_smd=True
    ),
    # 1206 - Imperial (3216 Metric)
    '1206': FootprintDefinition(
        name='1206',
        body_width=3.2,
        body_height=1.6,
        pad_positions=[
            ('1', -1.475, 0.0, 1.05, 1.75),
            ('2', 1.475, 0.0, 1.05, 1.75),
        ],
        is_smd=True
    ),
    # SOT-23-3
    'SOT-23': FootprintDefinition(
        name='SOT-23',
        body_width=2.9,
        body_height=1.3,
        pad_positions=[
            ('1', -0.95, 1.1, 0.6, 0.7),
            ('2', 0.95, 1.1, 0.6, 0.7),
            ('3', 0.0, -1.1, 0.6, 0.7),
        ],
        is_smd=True
    ),
    # SOT-23-5
    'SOT-23-5': FootprintDefinition(
        name='SOT-23-5',
        body_width=2.9,
        body_height=1.6,
        pad_positions=[
            ('1', -0.95, 1.1, 0.6, 0.7),
            ('2', 0.0, 1.1, 0.6, 0.7),
            ('3', 0.95, 1.1, 0.6, 0.7),
            ('4', 0.95, -1.1, 0.6, 0.7),
            ('5', -0.95, -1.1, 0.6, 0.7),
        ],
        is_smd=True
    ),
    # SOT-223 (LDO regulators like LM1117, AMS1117)
    # Simplified layout: 3 pins in horizontal line + tab above
    # MUST match parts_db offsets for routing/output consistency
    'SOT-223': FootprintDefinition(
        name='SOT-223',
        body_width=6.5,
        body_height=5.0,
        pad_positions=[
            ('1', -2.3, 0.0, 1.0, 1.8),    # Left pad (VIN)
            ('2', 0.0, 0.0, 1.0, 1.8),     # Center pad (GND)
            ('3', 2.3, 0.0, 1.0, 1.8),     # Right pad (VOUT)
            ('4', 0.0, 3.25, 3.0, 1.5),    # Large tab pad (VOUT/thermal)
        ],
        is_smd=True
    ),
}


def get_footprint_definition(footprint_name: str) -> FootprintDefinition:
    """
    Get footprint definition with multi-tier resolution.

    Resolution order:
    1. In-memory cache (instant)
    2. Pre-parsed JSON cache (footprint_cache.json)
    3a. Exact KiCad .kicad_mod file (from library:footprint path)
    4. Hardcoded FOOTPRINT_LIBRARY (7 verified entries)
    3b. Fuzzy KiCad .kicad_mod search
    5. Name-based inference (parse dimensions from name string)
    6. Default 2-pad fallback

    Args:
        footprint_name: Any footprint name format:
            - Simple: '0805', 'SOT-23', 'SOIC-8'
            - Prefixed: 'R_0805', 'C_0603'
            - KiCad: 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm'

    Returns:
        FootprintDefinition (never None)
    """
    try:
        from .footprint_resolver import FootprintResolver
        return FootprintResolver.get_instance().resolve(footprint_name)
    except ImportError:
        # Fallback if resolver not available (backward compat)
        if footprint_name in FOOTPRINT_LIBRARY:
            return FOOTPRINT_LIBRARY[footprint_name]

        fp_lower = (footprint_name or '').lower()
        for size in ['0402', '0603', '0805', '1206', 'sot-23-5', 'sot-23']:
            if size in fp_lower:
                key = size.upper() if 'sot' in size else size
                if key in FOOTPRINT_LIBRARY:
                    return FOOTPRINT_LIBRARY[key]

        return FootprintDefinition(
            name='default',
            body_width=2.0,
            body_height=1.0,
            pad_positions=[
                ('1', -0.95, 0.0, 0.9, 1.25),
                ('2', 0.95, 0.0, 0.9, 1.25),
            ],
            is_smd=True
        )


def is_smd_footprint(footprint_name: str) -> bool:
    """
    Check if a footprint is SMD (Surface Mount) vs Through-Hole.

    SMD components only block F.Cu layer.
    Through-hole components block both layers.
    """
    fp_lower = (footprint_name or '').lower()

    # Explicit through-hole patterns
    th_patterns = ['dip', 'sip', 'pin_header', 'pinheader', 'conn_', 'terminal', 'thru']
    if any(p in fp_lower for p in th_patterns):
        return False

    # Explicit SMD patterns
    smd_patterns = [
        '0201', '0402', '0603', '0805', '1206', '1210', '2010', '2512',  # R/C
        'sot', 'sod', 'soic', 'ssop', 'tssop', 'qfp', 'qfn', 'bga',     # ICs
        'lqfp', 'tqfp', 'mlf', 'dfn', 'led', 'chip', 'sma', 'smb', 'smc' # Other
    ]
    if any(p in fp_lower for p in smd_patterns):
        return True

    # Default to SMD (most modern components are SMD)
    return True


# =============================================================================
# COURTYARD CALCULATION UTILITY
# =============================================================================
# Single source of truth for component courtyard calculations.
# All pistons should use this instead of their own estimates.
#
# Per IPC-7351B, courtyard = pad extent + margin for pick-and-place clearance.
# Default margin is 0.25mm for nominal density (Level B).

@dataclass
class CourtyardBounds:
    """
    Component courtyard bounds (rectangle centered at component origin).

    Courtyards define the physical boundary around a component including:
    - The component body
    - All pads/leads extending beyond the body
    - Pick-and-place clearance margin

    Pistons should use courtyards for:
    - Placement: Prevent overlapping components
    - Routing: Route tracks around component areas
    - DRC: Check for courtyard violations
    - Silkscreen: Keep silkscreen outside courtyard
    - Escape: Route escapes from courtyard boundary
    - Pour: Maintain clearance from pour to components
    """
    width: float   # Total courtyard width (mm)
    height: float  # Total courtyard height (mm)
    min_x: float   # Left edge relative to component center
    max_x: float   # Right edge relative to component center
    min_y: float   # Bottom edge relative to component center
    max_y: float   # Top edge relative to component center
    margin: float  # IPC margin used (mm)

    @property
    def half_width(self) -> float:
        """Half-width for collision detection."""
        return self.width / 2

    @property
    def half_height(self) -> float:
        """Half-height for collision detection."""
        return self.height / 2

    def contains_point(self, x: float, y: float, comp_x: float = 0, comp_y: float = 0) -> bool:
        """Check if a point is inside the courtyard (given component position)."""
        rel_x = x - comp_x
        rel_y = y - comp_y
        return self.min_x <= rel_x <= self.max_x and self.min_y <= rel_y <= self.max_y

    def overlaps(self, other: 'CourtyardBounds',
                 self_pos: Tuple[float, float],
                 other_pos: Tuple[float, float],
                 gap: float = 0.0) -> bool:
        """
        Check if two courtyards overlap (given component positions).

        Args:
            other: Other courtyard bounds
            self_pos: (x, y) position of this component
            other_pos: (x, y) position of other component
            gap: Additional gap required between courtyards
        """
        # Calculate absolute bounds
        self_left = self_pos[0] + self.min_x - gap
        self_right = self_pos[0] + self.max_x + gap
        self_bottom = self_pos[1] + self.min_y - gap
        self_top = self_pos[1] + self.max_y + gap

        other_left = other_pos[0] + other.min_x
        other_right = other_pos[0] + other.max_x
        other_bottom = other_pos[1] + other.min_y
        other_top = other_pos[1] + other.max_y

        # Check for no overlap (any separating axis)
        if self_right < other_left or self_left > other_right:
            return False
        if self_top < other_bottom or self_bottom > other_top:
            return False

        return True


# IPC-7351B courtyard margin levels
COURTYARD_MARGIN_DENSE = 0.10    # Level A - Least (high-density boards)
COURTYARD_MARGIN_NOMINAL = 0.25  # Level B - Nominal (standard)
COURTYARD_MARGIN_LOOSE = 0.50   # Level C - Most (prototype/hand solder)


def calculate_courtyard(
    part: Dict,
    margin: float = COURTYARD_MARGIN_NOMINAL,
    footprint_name: str = None,
    rotation: int = 0
) -> CourtyardBounds:
    """
    Calculate component courtyard from pad positions.

    This is the SINGLE SOURCE OF TRUTH for courtyard calculations.
    All pistons should use this function instead of their own estimates.

    The courtyard is calculated as:
    1. Find the bounding box of all pad extents (position + size)
    2. Add IPC-7351B margin for pick-and-place clearance

    Args:
        part: Part dictionary with 'pins' containing pad offsets and sizes
        margin: IPC courtyard margin (default 0.25mm for nominal density)
        footprint_name: Optional footprint name for library lookup

    Returns:
        CourtyardBounds with width, height, and edge positions

    Example:
        >>> part = {'footprint': 'SOT-223', 'pins': [...]}
        >>> bounds = calculate_courtyard(part)
        >>> print(f"Courtyard: {bounds.width:.2f} x {bounds.height:.2f} mm")
        Courtyard: 5.40 x 6.10 mm
    """
    # Start with minimum bounds at origin
    min_x, max_x = 0.0, 0.0
    min_y, max_y = 0.0, 0.0

    # Get pins from part
    pins = get_pins(part)

    # If no pins in part, try footprint library
    if not pins and footprint_name:
        fp_def = get_footprint_definition(footprint_name)
        if fp_def and fp_def.pad_positions:
            for pin_num, ox, oy, pw, ph in fp_def.pad_positions:
                half_w = pw / 2
                half_h = ph / 2
                min_x = min(min_x, ox - half_w)
                max_x = max(max_x, ox + half_w)
                min_y = min(min_y, oy - half_h)
                max_y = max(max_y, oy + half_h)

    # Calculate bounds from pins
    for pin in pins:
        # Get pad offset - handle both 'offset' tuple and 'physical' dict formats
        offset = pin.get('offset', None)
        if offset is not None and isinstance(offset, (list, tuple)) and len(offset) >= 2:
            ox, oy = float(offset[0]), float(offset[1])
        elif offset is not None and isinstance(offset, dict):
            ox = float(offset.get('x', 0))
            oy = float(offset.get('y', 0))
        else:
            # Fallback: check 'physical' dict format (offset_x, offset_y)
            physical = pin.get('physical', {})
            if physical:
                ox = float(physical.get('offset_x', 0))
                oy = float(physical.get('offset_y', 0))
            else:
                ox, oy = 0.0, 0.0

        # Get pad size - prefer pin data, then FOOTPRINT_LIBRARY, then fallback
        pad_size = pin.get('size', pin.get('pad_size', None))
        if pad_size is None:
            # Try FOOTPRINT_LIBRARY for accurate IPC pad sizes
            fp_name = footprint_name or part.get('footprint', '')
            fp_def = get_footprint_definition(fp_name)
            pin_num = str(pin.get('number', pin.get('pin', '')))
            if fp_def and fp_def.pad_positions:
                for lib_pin_num, _, _, lib_pw, lib_ph in fp_def.pad_positions:
                    if str(lib_pin_num) == pin_num:
                        pad_size = (lib_pw, lib_ph)
                        break
            if pad_size is None:
                pad_size = (1.0, 0.6)  # Last resort fallback
        if isinstance(pad_size, (list, tuple)) and len(pad_size) >= 2:
            pw, ph = float(pad_size[0]), float(pad_size[1])
        elif isinstance(pad_size, (int, float)):
            pw = ph = float(pad_size)
        else:
            pw, ph = 1.0, 0.6

        # Expand bounds to include this pad
        half_w = pw / 2
        half_h = ph / 2
        min_x = min(min_x, ox - half_w)
        max_x = max(max_x, ox + half_w)
        min_y = min(min_y, oy - half_h)
        max_y = max(max_y, oy + half_h)

    # Ensure bounds are at least as large as the body size from parts_db
    body_size = part.get('size', None)
    if body_size and isinstance(body_size, (list, tuple)) and len(body_size) >= 2:
        body_half_w = float(body_size[0]) / 2
        body_half_h = float(body_size[1]) / 2
        min_x = min(min_x, -body_half_w)
        max_x = max(max_x, body_half_w)
        min_y = min(min_y, -body_half_h)
        max_y = max(max_y, body_half_h)

    # If still no bounds, use body size from footprint library
    if min_x == 0 and max_x == 0 and min_y == 0 and max_y == 0:
        fp_name = footprint_name or part.get('footprint', '')
        fp_def = get_footprint_definition(fp_name)
        if fp_def:
            min_x = -fp_def.body_width / 2
            max_x = fp_def.body_width / 2
            min_y = -fp_def.body_height / 2
            max_y = fp_def.body_height / 2
        else:
            # Absolute fallback - 2mm x 2mm
            min_x, max_x = -1.0, 1.0
            min_y, max_y = -1.0, 1.0

    # Apply rotation BEFORE adding margin
    if rotation in (90, 270):
        # Swap X and Y axes for 90/270 degree rotation
        min_x, min_y = min_y, min_x
        max_x, max_y = max_y, max_x
    elif rotation == 180:
        # Flip both axes for 180 degree rotation
        min_x, max_x = -max_x, -min_x
        min_y, max_y = -max_y, -min_y

    # Add IPC margin
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    width = max_x - min_x
    height = max_y - min_y

    return CourtyardBounds(
        width=width,
        height=height,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        margin=margin
    )


def get_component_courtyard(
    ref: str,
    parts_db: Dict,
    margin: float = COURTYARD_MARGIN_NOMINAL
) -> CourtyardBounds:
    """
    Get courtyard bounds for a component by reference designator.

    Convenience function that extracts the part from parts_db.

    Args:
        ref: Component reference designator (e.g., 'U1', 'C1')
        parts_db: Parts database with 'parts' dict
        margin: IPC courtyard margin

    Returns:
        CourtyardBounds for the component
    """
    parts = parts_db.get('parts', {})
    part = parts.get(ref, {})
    footprint = part.get('footprint', '')
    return calculate_courtyard(part, margin, footprint)


def calculate_board_size(
    parts_db: Dict,
    utilization_target: float = 0.40,
    edge_clearance: float = 1.0,
    round_increment: float = 0.5,
    min_size: float = 10.0,
) -> Tuple[float, float]:
    """Auto-calculate board dimensions from total component courtyard area.

    Uses calculate_courtyard() as the single source of truth for component
    dimensions. Produces a rectangular board with golden-ratio aspect and
    sufficient routing space.

    Args:
        parts_db: Parts database with 'parts' dict
        utilization_target: Component fill ratio (0.40 = 40% fill, 60% routing)
        edge_clearance: Board edge clearance in mm (added to each side)
        round_increment: Round dimensions up to this increment (mm)
        min_size: Minimum board dimension in mm

    Returns:
        (board_width, board_height) in mm
    """
    parts = parts_db.get('parts', {})
    if not parts:
        return (min_size, min_size)

    # Step 1: Sum courtyard areas + inter-component spacing
    total_area = 0.0
    max_single_w = 0.0
    max_single_h = 0.0
    spacing = 0.5  # mm between components

    for ref, part in parts.items():
        footprint = part.get('footprint', '')
        courtyard = calculate_courtyard(part, margin=COURTYARD_MARGIN_NOMINAL,
                                        footprint_name=footprint)
        eff_w = courtyard.width + spacing
        eff_h = courtyard.height + spacing
        total_area += eff_w * eff_h
        max_single_w = max(max_single_w, courtyard.width)
        max_single_h = max(max_single_h, courtyard.height)

    # Step 2: Required area from utilization target
    utilization_target = max(0.15, min(utilization_target, 0.80))
    required_area = total_area / utilization_target

    # Step 3: Golden ratio dimensions (1.618:1)
    golden = 1.618
    raw_h = math.sqrt(required_area / golden)
    raw_w = raw_h * golden

    # Step 4: Add edge clearance (both sides)
    raw_w += 2 * edge_clearance
    raw_h += 2 * edge_clearance

    # Step 5: Must fit largest single component
    min_w = max_single_w + 2 * edge_clearance + 2 * spacing
    min_h = max_single_h + 2 * edge_clearance + 2 * spacing
    raw_w = max(raw_w, min_w)
    raw_h = max(raw_h, min_h)

    # Step 6: Clamp aspect ratio (0.6 – 1.67)
    ar = raw_w / raw_h if raw_h > 0 else 1.0
    if ar < 0.6:
        raw_w = raw_h * 0.6
    elif ar > 1.67:
        raw_h = raw_w / 1.67

    # Step 7: Round up
    width = math.ceil(raw_w / round_increment) * round_increment
    height = math.ceil(raw_h / round_increment) * round_increment

    # Step 8: Enforce minimum
    width = max(width, min_size)
    height = max(height, min_size)

    return (width, height)
