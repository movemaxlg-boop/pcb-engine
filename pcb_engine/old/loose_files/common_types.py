"""
PCB Engine - Common Types
==========================

Shared type definitions used across the PCB engine.
This module provides canonical types to prevent mismatches between pistons.
"""

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
    Get footprint definition from library, with fallback matching.

    Args:
        footprint_name: Footprint name (e.g., 'R_0805', '0805', 'C_0603')

    Returns:
        FootprintDefinition, or a default if not found
    """
    # Direct match
    if footprint_name in FOOTPRINT_LIBRARY:
        return FOOTPRINT_LIBRARY[footprint_name]

    # Try extracting size code from name like 'R_0805', 'C_0603'
    fp_lower = footprint_name.lower()
    for size in ['0402', '0603', '0805', '1206', 'sot-23-5', 'sot-23']:
        if size in fp_lower:
            key = size.upper() if 'sot' in size else size
            if key in FOOTPRINT_LIBRARY:
                return FOOTPRINT_LIBRARY[key]

    # Default 2-pin SMD component
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
