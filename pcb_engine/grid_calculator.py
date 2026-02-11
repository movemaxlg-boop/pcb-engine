"""
PCB Engine - Dynamic Grid Calculator
=====================================

Calculates optimal routing grid size based on component requirements.

THE PROBLEM:
============
Fixed grid sizes (like 0.5mm) can't represent tight-pitch components accurately.
For example, SOT-23-5 has 0.95mm pin pitch - with 0.5mm grid:
- Pins at 0mm, 0.95mm become grid cells 0, 2 (skipping cell 1)
- This creates gaps where routes can pass through pads!

THE SOLUTION:
=============
Calculate grid size dynamically based on:
1. Minimum pad pitch from all components (grid must be divisor of pitch)
2. Minimum clearance (grid must allow precise clearance checking)
3. Minimum trace width (grid should represent trace width accurately)
4. Performance constraints (very fine grid = slow routing)

FORMULA:
========
optimal_grid = min(
    min_pad_pitch / 4,      # At least 4 grid cells per pad pitch
    min_clearance,          # Grid <= clearance (1 cell is enough for DRC)
    min_trace_width,        # Grid <= trace width
    max_grid_size           # Performance limit (default 0.25mm)
)

Round down to nearest 0.05mm for clean values.

PERFORMANCE TIERS:
==================
- FAST:     0.25mm grid, ~32K cells for 50x40mm (quick prototyping)
- BALANCED: 0.15mm grid, ~90K cells for 50x40mm (recommended)
- ACCURATE: 0.10mm grid, ~200K cells for 50x40mm (tight-pitch)
- PRECISE:  0.05mm grid, ~800K cells for 50x40mm (fine-pitch ICs)

COMMON FOOTPRINT PITCHES:
=========================
- 0402: 0.5mm between pads
- 0603: 1.6mm between pads
- 0805: 1.9mm between pads
- SOT-23: 0.95mm pitch
- SOT-23-5: 0.95mm pitch
- SOIC-8: 1.27mm pitch
- TQFP-32: 0.8mm pitch
- QFN-16: 0.5mm pitch
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
import math

from .common_types import FOOTPRINT_LIBRARY, get_footprint_definition


class PerformanceTier(Enum):
    """Grid performance/accuracy tradeoff tiers"""
    FAST = 'fast'           # 0.25mm - quick prototyping
    BALANCED = 'balanced'   # 0.15mm - recommended default
    ACCURATE = 'accurate'   # 0.10mm - tight-pitch components
    PRECISE = 'precise'     # 0.05mm - fine-pitch ICs
    AUTO = 'auto'           # Calculate based on components


# Extended footprint pitches for packages NOT in FOOTPRINT_LIBRARY
# FOOTPRINT_LIBRARY (common_types.py) is the primary source — these are supplements
FOOTPRINT_PITCHES = {
    # Passives (2-pin)
    '0201': 0.3,
    '0402': 0.5,
    '0603': 0.8,
    '0805': 1.0,
    '1206': 1.6,
    '1210': 1.6,
    '2512': 3.2,

    # SOT packages
    'SOT-23': 0.95,
    'SOT-23-3': 0.95,
    'SOT-23-5': 0.95,
    'SOT-23-6': 0.95,
    'SOT-323': 0.65,
    'SOT-363': 0.65,
    'SOT-89': 1.5,
    'SOT-223': 2.3,

    # SOIC packages
    'SOIC-8': 1.27,
    'SOIC-14': 1.27,
    'SOIC-16': 1.27,
    'SSOP-8': 0.65,
    'SSOP-16': 0.65,
    'TSSOP-8': 0.65,
    'TSSOP-14': 0.65,
    'TSSOP-16': 0.65,
    'TSSOP-20': 0.65,

    # QFP/QFN packages
    'QFP-32': 0.8,
    'QFP-44': 0.8,
    'QFP-48': 0.5,
    'QFP-64': 0.5,
    'QFP-100': 0.5,
    'LQFP-32': 0.8,
    'LQFP-48': 0.5,
    'LQFP-64': 0.5,
    'TQFP-32': 0.8,
    'TQFP-44': 0.8,
    'TQFP-48': 0.5,
    'QFN-16': 0.5,
    'QFN-20': 0.5,
    'QFN-24': 0.5,
    'QFN-32': 0.5,
    'QFN-48': 0.4,
    'DFN-8': 0.5,

    # BGA packages (very fine pitch)
    'BGA-256': 0.8,
    'BGA-484': 0.5,
    'WLCSP': 0.4,

    # Through-hole
    'DIP-8': 2.54,
    'DIP-14': 2.54,
    'DIP-16': 2.54,
    'DIP-28': 2.54,
    'SIP-3': 2.54,
    'TO-220': 2.54,
    'TO-92': 1.27,
}

# Tier grid sizes
TIER_GRID_SIZES = {
    PerformanceTier.FAST: 0.25,
    PerformanceTier.BALANCED: 0.15,
    PerformanceTier.ACCURATE: 0.10,
    PerformanceTier.PRECISE: 0.05,
}


def get_footprint_pitch(footprint_name: str) -> Optional[float]:
    """
    Get known pitch for a footprint name.

    Priority: FOOTPRINT_LIBRARY.pitch > FOOTPRINT_PITCHES fallback

    Args:
        footprint_name: Footprint name (e.g., 'C_0603_1608Metric')

    Returns:
        Known pitch in mm, or None if unknown
    """
    if not footprint_name:
        return None

    # Priority 1: FOOTPRINT_LIBRARY (single source of truth)
    fp_def = get_footprint_definition(footprint_name)
    if fp_def.name != 'default' and len(fp_def.pad_positions) >= 2:
        return fp_def.pitch

    # Priority 2: Extended FOOTPRINT_PITCHES for packages not in FOOTPRINT_LIBRARY
    name_upper = footprint_name.upper()
    for pattern, pitch in FOOTPRINT_PITCHES.items():
        if pattern.upper() in name_upper:
            return pitch

    for size in ['0201', '0402', '0603', '0805', '1206', '1210', '2512']:
        if size in name_upper:
            return FOOTPRINT_PITCHES.get(size)

    return None


def calculate_min_pad_pitch(parts_db: Dict) -> float:
    """
    Calculate the minimum pad-to-pad pitch from all components.

    Uses multiple strategies:
    1. Check footprint names against known pitches
    2. Calculate from pin offset positions
    3. Fall back to sensible default

    Args:
        parts_db: Parts database with 'parts' dict containing component data

    Returns:
        Minimum pad pitch in mm (returns 1.0 if no valid pitch found)
    """
    min_pitch = float('inf')
    parts = parts_db.get('parts', {})

    for ref, part in parts.items():
        # Strategy 1: Check footprint name against known pitches
        footprint = part.get('footprint', '') or part.get('package', '')
        known_pitch = get_footprint_pitch(footprint)
        if known_pitch:
            min_pitch = min(min_pitch, known_pitch)
            continue  # Known pitch is reliable, skip calculation

        # Strategy 2: Calculate from pin positions
        pins = []

        # Collect pin offsets from all possible formats
        for pin_list_key in ['pins', 'used_pins', 'physical_pins']:
            for pin in part.get(pin_list_key, []):
                # Direct offset field (most common)
                offset = pin.get('offset')
                if offset and isinstance(offset, (list, tuple)) and len(offset) >= 2:
                    if offset[0] != 0 or offset[1] != 0:
                        pins.append((float(offset[0]), float(offset[1])))
                        continue

                # Physical sub-dict format
                physical = pin.get('physical', {})
                if physical:
                    ox = physical.get('offset_x', 0)
                    oy = physical.get('offset_y', 0)
                    if ox != 0 or oy != 0:
                        pins.append((float(ox), float(oy)))

        # Calculate minimum distance between pin pairs
        if len(pins) >= 2:
            for i, p1 in enumerate(pins):
                for p2 in pins[i+1:]:
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    dist = math.sqrt(dx*dx + dy*dy)

                    # Only consider valid distances (> 0.1mm)
                    if dist > 0.1:
                        min_pitch = min(min_pitch, dist)

    # Return sensible default if no valid pitch found
    return min_pitch if min_pitch != float('inf') else 1.0


def suggest_performance_tier(parts_db: Dict) -> PerformanceTier:
    """
    Suggest a performance tier based on component complexity.

    Args:
        parts_db: Parts database

    Returns:
        Recommended PerformanceTier
    """
    min_pitch = calculate_min_pad_pitch(parts_db)
    num_parts = len(parts_db.get('parts', {}))
    num_nets = len(parts_db.get('nets', {}))

    # Check for fine-pitch components
    if min_pitch < 0.5:
        return PerformanceTier.PRECISE
    elif min_pitch < 0.8:
        return PerformanceTier.ACCURATE

    # Simple designs can use faster grid
    if num_parts <= 5 and num_nets <= 10:
        return PerformanceTier.FAST

    # Default to balanced for most designs
    return PerformanceTier.BALANCED


def calculate_optimal_grid_size(
    parts_db: Dict,
    min_clearance: float = 0.15,
    min_trace_width: float = 0.2,
    max_grid_size: float = 0.25,
    min_grid_size: float = 0.05,
    tier: PerformanceTier = PerformanceTier.AUTO
) -> Tuple[float, Dict]:
    """
    Calculate optimal grid size based on component requirements.

    The grid must be fine enough to:
    1. Accurately represent pad positions (at least 4 cells per pad pitch)
    2. Check clearances precisely (grid <= clearance)
    3. Represent trace width (grid <= trace width)

    But not too fine (performance impact):
    - More cells = more memory
    - More cells = slower pathfinding
    - Typical limit: 0.05mm minimum

    Args:
        parts_db: Parts database with component data
        min_clearance: Minimum clearance in mm (from design rules)
        min_trace_width: Minimum trace width in mm (from design rules)
        max_grid_size: Maximum allowed grid size in mm
        min_grid_size: Minimum allowed grid size in mm
        tier: Performance tier (AUTO calculates dynamically)

    Returns:
        Tuple of (optimal_grid_size, analysis_dict)
    """
    # If specific tier requested, use that grid size
    if tier != PerformanceTier.AUTO:
        tier_grid = TIER_GRID_SIZES[tier]
        return tier_grid, {
            'min_pad_pitch': calculate_min_pad_pitch(parts_db),
            'grid_for_pitch': 0.0,
            'grid_for_clearance': min_clearance,
            'grid_for_trace': min_trace_width,
            'max_allowed': max_grid_size,
            'min_allowed': min_grid_size,
            'optimal_grid': tier_grid,
            'limiting_factor': f'tier_{tier.value}',
            'tier': tier.value
        }

    # AUTO: Calculate minimum pad pitch
    min_pad_pitch = calculate_min_pad_pitch(parts_db)

    # Calculate grid requirements
    grid_for_pitch = min_pad_pitch / 4      # At least 4 cells per pad pitch
    grid_for_clearance = min_clearance      # Grid <= clearance (1 cell for DRC)
    grid_for_trace = min_trace_width        # Grid should be <= trace width

    # Take the minimum of all requirements
    optimal = min(
        grid_for_pitch,
        grid_for_clearance,
        grid_for_trace,
        max_grid_size
    )

    # Enforce minimum grid size (performance)
    optimal = max(optimal, min_grid_size)

    # Round down to nearest 0.05mm for clean values
    optimal = math.floor(optimal / 0.05) * 0.05
    optimal = max(optimal, min_grid_size)  # Ensure still >= min after rounding

    # Build analysis dict
    analysis = {
        'min_pad_pitch': min_pad_pitch,
        'grid_for_pitch': grid_for_pitch,
        'grid_for_clearance': grid_for_clearance,
        'grid_for_trace': grid_for_trace,
        'max_allowed': max_grid_size,
        'min_allowed': min_grid_size,
        'optimal_grid': optimal,
        'limiting_factor': _get_limiting_factor(
            optimal, grid_for_pitch, grid_for_clearance,
            grid_for_trace, max_grid_size, min_grid_size
        ),
        'tier': 'auto',
        'suggested_tier': suggest_performance_tier(parts_db).value
    }

    return optimal, analysis


def _get_limiting_factor(
    optimal: float,
    grid_for_pitch: float,
    grid_for_clearance: float,
    grid_for_trace: float,
    max_grid: float,
    min_grid: float
) -> str:
    """Determine what factor limited the grid size"""
    if optimal == min_grid:
        return "minimum_grid_limit"
    if optimal <= grid_for_pitch and grid_for_pitch <= min(grid_for_clearance, grid_for_trace, max_grid):
        return "pad_pitch"
    if optimal <= grid_for_clearance and grid_for_clearance <= min(grid_for_pitch, grid_for_trace, max_grid):
        return "clearance"
    if optimal <= grid_for_trace and grid_for_trace <= min(grid_for_pitch, grid_for_clearance, max_grid):
        return "trace_width"
    return "max_grid_limit"


def estimate_grid_memory(
    board_width: float,
    board_height: float,
    grid_size: float,
    num_layers: int = 2
) -> Dict:
    """
    Estimate memory usage and performance metrics for the routing grid.

    Args:
        board_width: Board width in mm
        board_height: Board height in mm
        grid_size: Grid cell size in mm
        num_layers: Number of routing layers

    Returns:
        Dict with cell counts, estimated memory, and performance rating
    """
    cols = int(board_width / grid_size) + 1
    rows = int(board_height / grid_size) + 1
    total_cells = cols * rows * num_layers

    # Memory estimate: each cell = 16 bytes (pointer + overhead)
    memory_bytes = total_cells * 16

    # Performance rating based on cell count
    if total_cells < 50000:
        perf_rating = 'excellent'
        perf_note = 'Very fast routing'
    elif total_cells < 200000:
        perf_rating = 'good'
        perf_note = 'Good balance of speed and accuracy'
    elif total_cells < 500000:
        perf_rating = 'moderate'
        perf_note = 'May take a few seconds'
    elif total_cells < 1000000:
        perf_rating = 'slow'
        perf_note = 'Consider using coarser grid'
    else:
        perf_rating = 'very_slow'
        perf_note = 'Use FAST or BALANCED tier'

    return {
        'cols': cols,
        'rows': rows,
        'cells_per_layer': cols * rows,
        'total_cells': total_cells,
        'estimated_memory_mb': memory_bytes / (1024 * 1024),
        'performance_rating': perf_rating,
        'performance_note': perf_note
    }


def get_grid_for_tier(tier: PerformanceTier) -> float:
    """Get grid size for a specific performance tier"""
    return TIER_GRID_SIZES.get(tier, 0.15)


def print_grid_analysis(
    parts_db: Dict,
    board_width: float,
    board_height: float,
    min_clearance: float = 0.15,
    min_trace_width: float = 0.2,
    tier: PerformanceTier = PerformanceTier.AUTO
) -> float:
    """
    Print detailed grid analysis and return optimal grid size.

    Useful for debugging and understanding grid requirements.
    """
    optimal, analysis = calculate_optimal_grid_size(
        parts_db, min_clearance, min_trace_width, tier=tier
    )

    print("\n" + "=" * 60)
    print("DYNAMIC GRID ANALYSIS")
    print("=" * 60)

    print(f"\nComponent Analysis:")
    print(f"  Parts count: {len(parts_db.get('parts', {}))}")
    print(f"  Nets count: {len(parts_db.get('nets', {}))}")
    print(f"  Minimum pad pitch: {analysis['min_pad_pitch']:.3f} mm")

    if tier == PerformanceTier.AUTO:
        print(f"\nGrid Requirements (AUTO mode):")
        print(f"  For pad pitch (pitch/4): {analysis['grid_for_pitch']:.3f} mm")
        print(f"  For clearance: {analysis['grid_for_clearance']:.3f} mm")
        print(f"  For trace width: {analysis['grid_for_trace']:.3f} mm")
        print(f"  Max allowed: {analysis['max_allowed']:.3f} mm")
        print(f"  Min allowed: {analysis['min_allowed']:.3f} mm")
        print(f"  Suggested tier: {analysis.get('suggested_tier', 'balanced')}")
    else:
        print(f"\nUsing {tier.value.upper()} performance tier")

    print(f"\nResult:")
    print(f"  Optimal grid size: {optimal:.3f} mm")
    print(f"  Limiting factor: {analysis['limiting_factor']}")

    # Memory estimate
    mem = estimate_grid_memory(board_width, board_height, optimal)
    print(f"\nGrid Dimensions:")
    print(f"  Size: {mem['cols']} x {mem['rows']} cells")
    print(f"  Total cells (2 layers): {mem['total_cells']:,}")
    print(f"  Estimated memory: {mem['estimated_memory_mb']:.2f} MB")
    print(f"  Performance: {mem['performance_rating']} - {mem['performance_note']}")

    # Show tier comparison
    print(f"\nPerformance Tier Comparison:")
    for t in PerformanceTier:
        if t == PerformanceTier.AUTO:
            continue
        grid = TIER_GRID_SIZES[t]
        mem_t = estimate_grid_memory(board_width, board_height, grid)
        marker = " <-- current" if abs(grid - optimal) < 0.01 else ""
        print(f"  {t.value.upper():10} ({grid:.2f}mm): {mem_t['total_cells']:>10,} cells, {mem_t['performance_rating']}{marker}")

    print("=" * 60 + "\n")

    return optimal


def quick_grid_estimate(
    board_width: float = 50.0,
    board_height: float = 40.0,
    has_fine_pitch: bool = False,
    num_parts: int = 10
) -> Tuple[float, str]:
    """
    Quick grid size estimate without parts database.

    Useful for early planning before full parts list is available.

    Args:
        board_width: Board width in mm
        board_height: Board height in mm
        has_fine_pitch: True if design has QFN/QFP/BGA packages
        num_parts: Approximate number of components

    Returns:
        Tuple of (grid_size, tier_name)
    """
    if has_fine_pitch:
        tier = PerformanceTier.ACCURATE
    elif num_parts <= 5:
        tier = PerformanceTier.FAST
    else:
        tier = PerformanceTier.BALANCED

    grid = TIER_GRID_SIZES[tier]

    return grid, tier.value
