"""
PCB Engine - DRC (Design Rule Check) Piston
=============================================

Comprehensive Design Rule Check engine with research-based algorithms.

CHECKS IMPLEMENTED:
===================

ELECTRICAL CHECKS:
1. Clearance - Track-to-track, track-to-pad, pad-to-pad
2. Track Width - Minimum width for current capacity (IPC-2221/IPC-2152)
3. Voltage Spacing - IPC-2221 voltage-dependent clearance
4. Net Connectivity - Verify all nets properly connected
5. Short Circuit Detection - Different nets touching (Bentley-Ottmann)

MANUFACTURING CHECKS (DFM):
6. Via Drill - Minimum via hole size
7. Via Annular Ring - Pad-to-drill ratio
8. Hole Spacing - Minimum drill-to-drill distance
9. Acid Traps - Acute angle detection (<90°) at junctions
10. Solder Mask Slivers - Narrow mask between features
11. Silkscreen Clearance - Silk over exposed pads
12. Copper Slivers - Narrow copper features

MECHANICAL CHECKS:
13. Board Edge Clearance - Features too close to edge
14. Component Courtyard - Overlapping courtyards
15. Hole-to-Edge Distance - Structural integrity

SIGNAL INTEGRITY:
16. Impedance Discontinuity - Width changes on controlled impedance
17. Return Path - Ground plane coverage for signals
18. Differential Pair Spacing - Consistent coupling
19. Differential Pair Skew - Length mismatch (≤5mm target, ≤25mm max)

ALGORITHMS:
===========
1. Bentley-Ottmann Sweepline - O((n+k)log n) intersection detection
   Source: https://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
   Source: https://www.cse.cuhk.edu.hk/~byu/papers/C172-DAC2023-OpenDRC.pdf

2. R-Tree Spatial Indexing - Fast bounding box queries for clearance
   Source: https://en.wikipedia.org/wiki/R-tree

3. Hierarchical Interval Lists - GPU-accelerated stabbing queries (PDRC)
   Source: https://www.cse.cuhk.edu.hk/~byu/papers/C219-DAC2024-PDRC.pdf

RESEARCH REFERENCES:
====================
- IPC-2221B: Generic Standard on Printed Board Design
  Source: https://www.protoexpress.com/blog/ipc-2221-circuit-board-design/

- IPC-2152: Current-Carrying Capacity (2009, supersedes IPC-2221 for current)
  Source: https://resources.altium.com/p/using-ipc-2152-calculator-designing-standards
  Source: https://www.smps.us/pcb-calculator.html

- IPC-2221 Voltage Spacing (Table 6-1):
  Formula (>500V): 2.5 + (V-500) * 0.005 mm
  Source: https://resources.altium.com/p/using-an-ipc-2221-calculator-for-high-voltage-design
  Source: https://www.smpspowersupply.com/ipc2221pcbclearance.html

- IPC-2221 Trace Width for Current:
  I = k * ΔT^0.44 * A^0.725
  k = 0.048 (outer), 0.024 (inner)
  Source: https://resources.altium.com/p/ipc-2221-calculator-pcb-trace-current-and-heating

- Acid Trap Detection:
  Angles < 90° trap etchant during PCB manufacturing
  Source: https://www.nextpcb.com/blog/acid-traps
  Source: https://resources.pcb.cadence.com/blog/are-acid-traps-still-a-problem-for-pcbs-in-2019-2

- GPU-Accelerated DRC (OpenDRC, PDRC):
  30-50x speedup with hierarchical GPU kernels
  Source: https://dl.acm.org/doi/10.1145/3649329.3657367
  Source: https://github.com/opendrc/opendrc

- Solder Mask Minimum Sliver:
  Typical minimum: 0.1mm (4 mils)
  Source: https://www.protoexpress.com/blog/6-common-solder-mask-errors-every-pcb-designer-should-know/

- Differential Pair Length Matching:
  Intra-pair skew target: ≤5mm, max ≤25mm (≤150ps on FR-4)
  Source: https://resources.pcb.cadence.com/blog/2025-differential-pair-length-matching-guidelines

DRC is CRITICAL for ensuring the PCB can be manufactured correctly.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappush, heappop
from bisect import insort, bisect_left
import math

# Import common types for position handling (BUG-03/BUG-06 fix)
from .common_types import Position, normalize_position, get_xy, get_pins, get_pin_net


# =============================================================================
# BENTLEY-OTTMANN SWEEPLINE ALGORITHM
# =============================================================================
# Research: https://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
# Research: https://www.hackerearth.com/practice/math/geometry/line-intersection-using-bentley-ottmann-algorithm/tutorial/
# Time complexity: O((n + k) log n) where k is number of intersections
# =============================================================================

class EventType(Enum):
    """Event types for Bentley-Ottmann sweepline"""
    LEFT_ENDPOINT = 0   # Segment starts
    INTERSECTION = 1    # Two segments cross
    RIGHT_ENDPOINT = 2  # Segment ends


@dataclass
class BOSegment:
    """Segment for Bentley-Ottmann algorithm"""
    id: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    net: str = ''
    layer: str = ''

    def __post_init__(self):
        # Ensure start.x <= end.x (left-to-right sweep)
        if self.start[0] > self.end[0]:
            self.start, self.end = self.end, self.start

    def y_at_x(self, x: float) -> float:
        """Get y-coordinate at given x (for sweep line status ordering)"""
        if abs(self.end[0] - self.start[0]) < 1e-10:
            return self.start[1]  # Vertical segment
        t = (x - self.start[0]) / (self.end[0] - self.start[0])
        return self.start[1] + t * (self.end[1] - self.start[1])


@dataclass(order=True)
class BOEvent:
    """Event for the priority queue"""
    x: float
    event_type: int  # Use int for ordering (LEFT=0, INTERSECTION=1, RIGHT=2)
    y: float = 0.0
    segments: Tuple = field(default_factory=tuple, compare=False)


class BentleyOttmann:
    """
    Bentley-Ottmann Sweepline Algorithm for Line Segment Intersection

    This is the standard algorithm used in professional EDA tools for
    detecting track crossings in DRC. It's more efficient than the
    naive O(n²) pairwise comparison for large numbers of segments.

    Research sources:
    - https://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
    - https://www.hackerearth.com/practice/math/geometry/line-intersection-using-bentley-ottmann-algorithm/tutorial/
    - https://github.com/ideasman42/isect_segments-bentley_ottmann
    """

    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance
        self.event_queue: List[BOEvent] = []
        self.sweep_status: List[BOSegment] = []
        self.intersections: List[Tuple[Tuple[float, float], BOSegment, BOSegment]] = []

    def find_intersections(self, segments: List[BOSegment]) -> List[Tuple[Tuple[float, float], BOSegment, BOSegment]]:
        """
        Find all intersections between segments.

        Args:
            segments: List of BOSegment objects

        Returns:
            List of (intersection_point, segment1, segment2) tuples
        """
        self.event_queue = []
        self.sweep_status = []
        self.intersections = []
        self.current_x = float('-inf')

        # Initialize event queue with segment endpoints
        for seg in segments:
            heappush(self.event_queue, BOEvent(
                x=seg.start[0],
                event_type=EventType.LEFT_ENDPOINT.value,
                y=seg.start[1],
                segments=(seg,)
            ))
            heappush(self.event_queue, BOEvent(
                x=seg.end[0],
                event_type=EventType.RIGHT_ENDPOINT.value,
                y=seg.end[1],
                segments=(seg,)
            ))

        # Process events
        while self.event_queue:
            event = heappop(self.event_queue)
            self.current_x = event.x

            if event.event_type == EventType.LEFT_ENDPOINT.value:
                self._handle_left_endpoint(event)
            elif event.event_type == EventType.RIGHT_ENDPOINT.value:
                self._handle_right_endpoint(event)
            else:  # INTERSECTION
                self._handle_intersection(event)

        return self.intersections

    def _handle_left_endpoint(self, event: BOEvent):
        """Handle segment start event"""
        seg = event.segments[0]

        # Insert segment into sweep status (sorted by y at current x)
        self._insert_segment(seg)

        # Check for intersections with neighbors
        idx = self._find_segment_index(seg)
        if idx > 0:
            self._check_intersection(self.sweep_status[idx - 1], seg)
        if idx < len(self.sweep_status) - 1:
            self._check_intersection(seg, self.sweep_status[idx + 1])

    def _handle_right_endpoint(self, event: BOEvent):
        """Handle segment end event"""
        seg = event.segments[0]
        idx = self._find_segment_index(seg)

        # Check if neighbors will become adjacent
        if idx > 0 and idx < len(self.sweep_status) - 1:
            self._check_intersection(
                self.sweep_status[idx - 1],
                self.sweep_status[idx + 1]
            )

        # Remove segment from sweep status
        self._remove_segment(seg)

    def _handle_intersection(self, event: BOEvent):
        """Handle intersection event"""
        seg1, seg2 = event.segments

        # Record intersection
        intersection_point = (event.x, event.y)
        self.intersections.append((intersection_point, seg1, seg2))

        # Swap segments in sweep status
        self._swap_segments(seg1, seg2)

        # Check for new intersections with new neighbors
        idx1 = self._find_segment_index(seg1)
        idx2 = self._find_segment_index(seg2)

        if idx1 > 0:
            self._check_intersection(self.sweep_status[idx1 - 1], seg1)
        if idx2 < len(self.sweep_status) - 1:
            self._check_intersection(seg2, self.sweep_status[idx2 + 1])

    def _insert_segment(self, seg: BOSegment):
        """Insert segment into sweep status maintaining y-order"""
        y = seg.y_at_x(self.current_x)
        idx = 0
        for i, s in enumerate(self.sweep_status):
            if s.y_at_x(self.current_x) > y:
                break
            idx = i + 1
        self.sweep_status.insert(idx, seg)

    def _remove_segment(self, seg: BOSegment):
        """Remove segment from sweep status"""
        for i, s in enumerate(self.sweep_status):
            if s.id == seg.id:
                self.sweep_status.pop(i)
                return

    def _find_segment_index(self, seg: BOSegment) -> int:
        """Find index of segment in sweep status"""
        for i, s in enumerate(self.sweep_status):
            if s.id == seg.id:
                return i
        return -1

    def _swap_segments(self, seg1: BOSegment, seg2: BOSegment):
        """Swap two segments in sweep status"""
        idx1 = self._find_segment_index(seg1)
        idx2 = self._find_segment_index(seg2)
        if idx1 >= 0 and idx2 >= 0:
            self.sweep_status[idx1], self.sweep_status[idx2] = \
                self.sweep_status[idx2], self.sweep_status[idx1]

    def _check_intersection(self, seg1: BOSegment, seg2: BOSegment):
        """Check if two segments intersect and add event if they do"""
        intersection = self._compute_intersection(seg1, seg2)
        if intersection and intersection[0] > self.current_x + self.tolerance:
            # Add intersection event
            heappush(self.event_queue, BOEvent(
                x=intersection[0],
                event_type=EventType.INTERSECTION.value,
                y=intersection[1],
                segments=(seg1, seg2)
            ))

    def _compute_intersection(self, seg1: BOSegment, seg2: BOSegment) -> Optional[Tuple[float, float]]:
        """
        Compute intersection point of two segments.

        Uses parametric line intersection.
        """
        x1, y1 = seg1.start
        x2, y2 = seg1.end
        x3, y3 = seg2.start
        x4, y4 = seg2.end

        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        cross = dx1 * dy2 - dy1 * dx2

        if abs(cross) < self.tolerance:
            return None  # Parallel

        dx3 = x1 - x3
        dy3 = y1 - y3

        t1 = (dx2 * dy3 - dy2 * dx3) / cross
        t2 = (dx1 * dy3 - dy1 * dx3) / cross

        # Check if intersection is within both segments (not at endpoints)
        eps = 0.001
        if eps < t1 < 1 - eps and eps < t2 < 1 - eps:
            ix = x1 + t1 * dx1
            iy = y1 + t1 * dy1
            return (ix, iy)

        return None


# =============================================================================
# R-TREE SPATIAL INDEXING (Simplified)
# =============================================================================
# Research: https://en.wikipedia.org/wiki/R-tree
# Used for fast bounding box queries in clearance checking
# =============================================================================

@dataclass
class BoundingBox:
    """Axis-aligned bounding box for spatial indexing"""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if two bounding boxes overlap"""
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                    self.max_y < other.min_y or self.min_y > other.max_y)

    def expand(self, margin: float) -> 'BoundingBox':
        """Return expanded bounding box"""
        return BoundingBox(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin
        )

    @staticmethod
    def from_segment(start: Tuple[float, float], end: Tuple[float, float],
                     width: float = 0) -> 'BoundingBox':
        """Create bounding box from segment with optional width"""
        half_w = width / 2
        return BoundingBox(
            min(start[0], end[0]) - half_w,
            min(start[1], end[1]) - half_w,
            max(start[0], end[0]) + half_w,
            max(start[1], end[1]) + half_w
        )


class SpatialIndex:
    """
    Simple spatial index for DRC clearance checking.

    Uses a grid-based approach for fast neighbor queries.
    For very large designs, replace with R-tree implementation.

    Research: https://en.wikipedia.org/wiki/R-tree
    """

    def __init__(self, cell_size: float = 5.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[Tuple[BoundingBox, any]]] = {}

    def insert(self, bbox: BoundingBox, data: any):
        """Insert item with bounding box into index"""
        cells = self._get_cells(bbox)
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append((bbox, data))

    def query(self, bbox: BoundingBox) -> List[any]:
        """Find all items whose bounding boxes intersect query box"""
        results = []
        seen = set()

        cells = self._get_cells(bbox)
        for cell in cells:
            if cell in self.grid:
                for item_bbox, data in self.grid[cell]:
                    if id(data) not in seen and bbox.intersects(item_bbox):
                        seen.add(id(data))
                        results.append(data)

        return results

    def _get_cells(self, bbox: BoundingBox) -> List[Tuple[int, int]]:
        """Get all grid cells that a bounding box overlaps"""
        min_cx = int(bbox.min_x / self.cell_size)
        min_cy = int(bbox.min_y / self.cell_size)
        max_cx = int(bbox.max_x / self.cell_size)
        max_cy = int(bbox.max_y / self.cell_size)

        cells = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cells.append((cx, cy))

        return cells


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DRCViolationType(Enum):
    """Types of DRC violations"""
    # Electrical
    CLEARANCE = 'clearance'               # Objects too close
    TRACK_WIDTH = 'track_width'           # Track too narrow
    VOLTAGE_SPACING = 'voltage_spacing'   # High voltage clearance (IPC-2221)
    UNCONNECTED = 'unconnected'           # Net not fully connected
    SHORT_CIRCUIT = 'short_circuit'       # Different nets touching
    CURRENT_CAPACITY = 'current_capacity' # Trace too narrow for current

    # Manufacturing (DFM)
    VIA_DRILL = 'via_drill'               # Via drill too small
    VIA_ANNULAR = 'via_annular'           # Via annular ring too small
    PAD_ANNULAR = 'pad_annular'           # Pad annular ring too small
    HOLE_SIZE = 'hole_size'               # Hole too small
    HOLE_SPACING = 'hole_spacing'         # Holes too close
    ACID_TRAP = 'acid_trap'               # Acute angle (<90°)
    SOLDER_MASK_SLIVER = 'mask_sliver'    # Narrow solder mask
    COPPER_SLIVER = 'copper_sliver'       # Narrow copper feature
    SILK_OVER_PAD = 'silk_over_pad'       # Silkscreen over exposed pad
    SILK_CLEARANCE = 'silk_clearance'     # Silk too close to mask opening

    # Mechanical
    EDGE_CLEARANCE = 'edge_clearance'     # Too close to board edge
    COURTYARD_OVERLAP = 'courtyard'       # Component courtyards overlap
    HOLE_TO_EDGE = 'hole_to_edge'         # Hole too close to board edge
    DANGLING_TRACK = 'dangling_track'     # Track end not connected
    DANGLING_VIA = 'dangling_via'         # Via not connected or only on one layer
    HOLES_COLOCATED = 'holes_colocated'   # Multiple holes at same position
    OVERLAP = 'overlap'                   # Objects overlapping
    SILK_OVERLAP = 'silk_overlap'         # Silkscreen items overlapping
    SILK_CLIPPED = 'silk_clipped'         # Silkscreen clipped by solder mask

    # Signal Integrity
    IMPEDANCE_DISCONTINUITY = 'impedance' # Width change on controlled net
    RETURN_PATH = 'return_path'           # Missing ground reference
    DIFF_PAIR_SPACING = 'diff_pair'       # Inconsistent diff pair spacing
    DIFF_PAIR_SKEW = 'diff_skew'          # Diff pair length mismatch
    STUB = 'stub'                         # Unterminated stub on high-speed


class DRCSeverity(Enum):
    """Severity levels for DRC violations"""
    ERROR = 'error'       # Must fix before manufacturing
    WARNING = 'warning'   # Should review
    INFO = 'info'         # FYI


@dataclass
class DRCViolation:
    """A single DRC violation"""
    violation_type: DRCViolationType
    severity: DRCSeverity
    message: str
    location: Tuple[float, float]
    layer: str = ''
    net1: str = ''
    net2: str = ''
    actual_value: float = 0.0
    required_value: float = 0.0

    # References to involved objects
    object1: str = ''  # e.g., "Track from (1.0, 2.0) to (3.0, 2.0)"
    object2: str = ''  # e.g., "Pad U1.1"


@dataclass
class DRCRules:
    """
    Design rules to check against

    Based on IPC-2221B and common manufacturer capabilities.
    Source: https://www.protoexpress.com/blog/ipc-2221-circuit-board-design/
    """
    # Clearances (mm) - Basic
    min_clearance: float = 0.15         # Minimum copper-to-copper clearance
    min_track_clearance: float = 0.15   # Track-to-track clearance
    min_pad_clearance: float = 0.15     # Track-to-pad clearance
    min_via_clearance: float = 0.2      # Via-to-via clearance

    # Track widths (mm)
    min_track_width: float = 0.15       # Minimum track width (6 mil)
    max_track_width: float = 5.0        # Maximum track width

    # Via specifications (mm)
    min_via_drill: float = 0.3          # Minimum via drill diameter
    min_via_diameter: float = 0.6       # Minimum via pad diameter
    min_via_annular: float = 0.125      # Minimum via annular ring

    # Hole specifications (mm)
    min_hole_size: float = 0.3          # Minimum hole diameter
    min_hole_spacing: float = 0.5       # Minimum hole-to-hole spacing (wall-to-wall)

    # Board edge (mm)
    min_edge_clearance: float = 0.25    # Copper to board edge
    min_hole_to_edge: float = 0.4       # Hole to board edge

    # Silkscreen (mm)
    min_silk_clearance: float = 0.15    # Silkscreen clearance from pads
    min_silk_width: float = 0.15        # Minimum silkscreen line width

    # Solder mask (mm)
    min_mask_sliver: float = 0.1        # Minimum solder mask web (4 mil)
    mask_expansion: float = 0.05        # Mask opening expansion from pad

    # Manufacturing (DFM)
    min_copper_sliver: float = 0.15     # Minimum copper feature width
    min_acid_trap_angle: float = 90.0   # Minimum angle (degrees) to avoid acid traps

    # IPC-2221 Voltage Spacing (Table 6-1)
    # Applied when voltage > threshold
    voltage_spacing_enabled: bool = False
    max_working_voltage: float = 50.0   # Maximum voltage on board

    # Current capacity (IPC-2221)
    copper_weight_oz: float = 1.0       # 1 oz/ft² = 35µm
    max_temp_rise: float = 10.0         # Maximum temperature rise (°C)

    # Signal integrity
    controlled_impedance_nets: List[str] = field(default_factory=list)
    impedance_tolerance_pct: float = 10.0  # ±10% impedance tolerance
    diff_pair_nets: List[Tuple[str, str]] = field(default_factory=list)
    max_diff_pair_skew: float = 0.127   # ±5 mils

    # Component spacing
    min_courtyard_gap: float = 0.25     # Gap between component courtyards


# =============================================================================
# IPC-2221 CALCULATORS
# =============================================================================

class IPC2221Calculator:
    """
    IPC-2221B Standard Calculations

    Research References:
    - https://resources.altium.com/p/ipc-2221-calculator-pcb-trace-current-and-heating
    - https://www.smps.us/pcb-calculator.html
    - https://tracewidthcalculator.com/
    """

    # IPC-2221 Table 6-1: Minimum Conductor Spacing (mm)
    # Format: (voltage_max, bare_board, coated_internal, coated_external)
    VOLTAGE_SPACING_TABLE = [
        (15, 0.05, 0.05, 0.1),
        (30, 0.05, 0.05, 0.1),
        (50, 0.1, 0.1, 0.6),
        (100, 0.1, 0.1, 0.6),
        (150, 0.2, 0.2, 0.6),
        (170, 0.25, 0.25, 1.25),
        (250, 0.5, 0.5, 1.25),
        (300, 0.8, 0.8, 1.25),
        (500, 2.5, 2.5, 2.5),
    ]

    # Constants for trace current formula
    # I = k * ΔT^b * A^c
    K_EXTERNAL = 0.048  # Outer layer constant
    K_INTERNAL = 0.024  # Inner layer constant
    B = 0.44            # Temperature exponent
    C = 0.725           # Area exponent

    @classmethod
    def get_voltage_spacing(cls, voltage: float, is_internal: bool = False,
                            is_coated: bool = True) -> float:
        """
        Get minimum spacing for given voltage per IPC-2221 Table 6-1

        Args:
            voltage: Working voltage (peak or DC)
            is_internal: True for internal layers
            is_coated: True if conformal coated

        Returns:
            Minimum spacing in mm
        """
        if voltage <= 0:
            return 0.1  # Default minimum

        # Find applicable row
        for v_max, bare, coat_int, coat_ext in cls.VOLTAGE_SPACING_TABLE:
            if voltage <= v_max:
                if is_coated:
                    return coat_int if is_internal else coat_ext
                else:
                    return bare

        # For voltages > 500V: 2.5 + (V-500) * 0.005 mm
        if voltage > 500:
            return 2.5 + (voltage - 500) * 0.005

        return 2.5  # Default for high voltage

    @classmethod
    def calculate_trace_width(cls, current: float, temp_rise: float,
                              copper_thickness_oz: float = 1.0,
                              is_external: bool = True) -> float:
        """
        Calculate required trace width for given current (IPC-2221)

        Formula: I = k * ΔT^0.44 * A^0.725
        Rearranged: A = (I / (k * ΔT^0.44))^(1/0.725)
        Width = A / thickness

        Args:
            current: Current in Amps
            temp_rise: Allowed temperature rise in °C
            copper_thickness_oz: Copper weight in oz/ft² (1oz = 35µm)
            is_external: True for outer layers (better cooling)

        Returns:
            Required trace width in mm
        """
        if current <= 0 or temp_rise <= 0:
            return 0.15  # Minimum default

        k = cls.K_EXTERNAL if is_external else cls.K_INTERNAL

        # Calculate required cross-sectional area (square mils)
        area_sq_mils = (current / (k * (temp_rise ** cls.B))) ** (1 / cls.C)

        # Convert copper thickness: oz/ft² to mils
        # 1 oz/ft² = 1.37 mils
        thickness_mils = copper_thickness_oz * 1.37

        # Width in mils
        width_mils = area_sq_mils / thickness_mils

        # Convert to mm (1 mil = 0.0254 mm)
        width_mm = width_mils * 0.0254

        return max(0.15, width_mm)  # Enforce minimum

    @classmethod
    def calculate_current_capacity(cls, width_mm: float, temp_rise: float,
                                   copper_thickness_oz: float = 1.0,
                                   is_external: bool = True) -> float:
        """
        Calculate current capacity for given trace width (IPC-2221)

        Formula: I = k * ΔT^0.44 * A^0.725

        Args:
            width_mm: Trace width in mm
            temp_rise: Allowed temperature rise in °C
            copper_thickness_oz: Copper weight in oz/ft²
            is_external: True for outer layers

        Returns:
            Maximum current in Amps
        """
        if width_mm <= 0 or temp_rise <= 0:
            return 0

        k = cls.K_EXTERNAL if is_external else cls.K_INTERNAL

        # Convert to mils
        width_mils = width_mm / 0.0254
        thickness_mils = copper_thickness_oz * 1.37

        # Cross-sectional area in square mils
        area = width_mils * thickness_mils

        # Current capacity
        current = k * (temp_rise ** cls.B) * (area ** cls.C)

        return current

    @classmethod
    def calculate_annular_ring(cls, pad_diameter: float, drill_diameter: float) -> float:
        """
        Calculate annular ring size

        Args:
            pad_diameter: Pad diameter in mm
            drill_diameter: Drill diameter in mm

        Returns:
            Annular ring width in mm
        """
        return (pad_diameter - drill_diameter) / 2


@dataclass
class DRCConfig:
    """Configuration for the DRC piston"""
    rules: DRCRules = field(default_factory=DRCRules)

    # Electrical checks
    check_clearances: bool = True
    check_track_widths: bool = True
    check_voltage_spacing: bool = False  # Enable for high voltage
    check_connectivity: bool = True
    check_short_circuits: bool = True
    check_current_capacity: bool = False  # Enable for power nets

    # Manufacturing checks (DFM)
    check_vias: bool = True
    check_holes: bool = True
    check_acid_traps: bool = True
    check_solder_mask: bool = True
    check_silkscreen: bool = True
    check_copper_slivers: bool = True

    # Mechanical checks
    check_edge_clearance: bool = True
    check_courtyards: bool = True
    check_hole_to_edge: bool = True
    check_dangling_tracks: bool = True    # Track ends not connected
    check_dangling_vias: bool = True      # Vias not connected or single-layer
    check_colocated_holes: bool = True    # Multiple holes at same position
    check_silk_overlap: bool = True       # Silkscreen items overlapping

    # Signal integrity checks
    check_impedance: bool = False  # Enable for controlled impedance
    check_diff_pairs: bool = False  # Enable for differential pairs
    check_return_path: bool = False  # Enable for high-speed

    # Board dimensions
    board_width: float = 100.0
    board_height: float = 100.0
    board_origin_x: float = 0.0
    board_origin_y: float = 0.0
    num_layers: int = 2

    # Net current assignments (for current capacity check)
    net_currents: Dict[str, float] = field(default_factory=dict)  # net -> amps

    # Net voltage assignments (for voltage spacing)
    net_voltages: Dict[str, float] = field(default_factory=dict)  # net -> volts


@dataclass
class DRCResult:
    """Result from the DRC piston"""
    violations: List[DRCViolation]
    passed: bool
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Summary by type
    violations_by_type: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# DRC PISTON
# =============================================================================

class DRCPiston:
    """
    DRC (Design Rule Check) Piston

    Performs comprehensive design rule checking on PCB layout data.

    Usage:
        config = DRCConfig()
        piston = DRCPiston(config)
        result = piston.check(parts_db, placement, routes, vias, silkscreen)
    """

    def __init__(self, config: DRCConfig = None):
        self.config = config or DRCConfig()
        self.rules = self.config.rules
        self.violations: List[DRCViolation] = []

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def check(self, parts_db: Dict, placement: Dict, routes: Dict,
              vias: List = None, silkscreen: List = None) -> DRCResult:
        """
        Run complete DRC check.

        Args:
            parts_db: Parts database with component and net info
            placement: Component placements {ref: Position}
            routes: Routing data {net: Route}
            vias: List of via positions (optional)
            silkscreen: List of silkscreen elements (optional)

        Returns:
            DRCResult with all violations found
        """
        self.violations.clear()
        vias = vias or []
        silkscreen = silkscreen or []

        # Build data structures for checking
        pads = self._extract_pads(parts_db, placement)
        tracks = self._extract_tracks(routes)

        # Run checks
        if self.config.check_clearances:
            self._check_clearances(pads, tracks, vias)

        if self.config.check_track_widths:
            self._check_track_widths(tracks)

        if self.config.check_vias:
            self._check_vias(vias)

        if self.config.check_holes:
            self._check_holes(pads, vias)

        if self.config.check_edge_clearance:
            self._check_edge_clearance(pads, tracks, vias)

        if self.config.check_silkscreen:
            self._check_silkscreen(pads, silkscreen)

        if self.config.check_connectivity:
            self._check_connectivity(parts_db, routes)

        # New KiCad-compatible checks
        if self.config.check_dangling_tracks:
            self._check_dangling_tracks(tracks, pads, vias)

        if self.config.check_dangling_vias:
            self._check_dangling_vias(vias, tracks)

        if self.config.check_colocated_holes:
            self._check_colocated_holes(pads, vias)

        if self.config.check_silk_overlap:
            self._check_silk_overlap(silkscreen)

        # Track crossing/short detection - CRITICAL!
        # Bug #14 fix: This check was missing from the main check() method
        # It was only in check_enhanced(), but track crossings are fundamental DRC
        self._check_track_crossings(tracks)

        # Summarize results
        error_count = sum(1 for v in self.violations if v.severity == DRCSeverity.ERROR)
        warning_count = sum(1 for v in self.violations if v.severity == DRCSeverity.WARNING)
        info_count = sum(1 for v in self.violations if v.severity == DRCSeverity.INFO)

        violations_by_type = {}
        for v in self.violations:
            key = v.violation_type.value
            violations_by_type[key] = violations_by_type.get(key, 0) + 1

        return DRCResult(
            violations=self.violations.copy(),
            passed=error_count == 0,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            violations_by_type=violations_by_type
        )

    # =========================================================================
    # DATA EXTRACTION
    # =========================================================================

    def _extract_pads(self, parts_db: Dict, placement: Dict) -> List[Dict]:
        """Extract all pads with their positions"""
        pads = []
        parts = parts_db.get('parts', {})

        # Guard against empty placement
        if not placement:
            return pads

        for ref, pos in placement.items():
            part = parts.get(ref, {})

            # Use get_pins for consistent pin access (handles all formats)
            for pin in get_pins(part):
                pin_num = pin.get('number', '')
                net = pin.get('net', '')

                offset = pin.get('offset', (0, 0))
                if not offset or offset == (0, 0):
                    physical = pin.get('physical', {})
                    offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))

                # Use get_xy for consistent position access
                try:
                    pos_x, pos_y = get_xy(pos)
                except (TypeError, ValueError):
                    pos_x, pos_y = 0.0, 0.0
                pad_x = pos_x + offset[0]
                pad_y = pos_y + offset[1]

                pad_size = pin.get('pad_size', pin.get('size', (1.0, 0.6)))
                if not isinstance(pad_size, (list, tuple)):
                    pad_size = (1.0, 0.6)

                pads.append({
                    'ref': ref,
                    'pin': pin_num,
                    'net': net,
                    'x': pad_x,
                    'y': pad_y,
                    'width': pad_size[0],
                    'height': pad_size[1],
                    'layer': getattr(pos, 'layer', 'F.Cu'),
                    'shape': pin.get('shape', 'rect'),
                    'hole': pin.get('hole', 0)  # 0 for SMD
                })

        return pads

    def _extract_tracks(self, routes: Dict) -> List[Dict]:
        """Extract all track segments"""
        tracks = []

        for net_name, route in routes.items():
            if hasattr(route, 'segments'):
                for seg in route.segments:
                    tracks.append({
                        'net': net_name,
                        'start': seg.start,
                        'end': seg.end,
                        'width': seg.width,
                        'layer': seg.layer
                    })
            elif isinstance(route, dict) and 'segments' in route:
                for seg in route['segments']:
                    tracks.append({
                        'net': net_name,
                        'start': seg.get('start', (0, 0)),
                        'end': seg.get('end', (0, 0)),
                        'width': seg.get('width', 0.25),
                        'layer': seg.get('layer', 'F.Cu')
                    })

        return tracks

    # =========================================================================
    # CLEARANCE CHECKS
    # =========================================================================

    def _check_clearances(self, pads: List[Dict], tracks: List[Dict], vias: List):
        """Check all clearance rules"""
        # Track-to-track clearance
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i+1:]:
                if t1['layer'] != t2['layer']:
                    continue
                if t1['net'] == t2['net']:
                    continue

                dist = self._track_to_track_distance(t1, t2)
                if dist < self.rules.min_track_clearance:
                    self._add_violation(
                        DRCViolationType.CLEARANCE,
                        DRCSeverity.ERROR,
                        f"Track clearance violation: {dist:.3f}mm < {self.rules.min_track_clearance:.3f}mm",
                        self._midpoint(t1['start'], t1['end']),
                        t1['layer'],
                        t1['net'], t2['net'],
                        dist, self.rules.min_track_clearance,
                        f"Track {t1['net']}",
                        f"Track {t2['net']}"
                    )

        # Track-to-pad clearance
        for track in tracks:
            for pad in pads:
                if track['layer'] != pad['layer'] and pad['hole'] == 0:
                    continue  # Different layers and SMD pad
                if track['net'] == pad['net']:
                    continue  # Same net

                dist = self._track_to_pad_distance(track, pad)
                if dist < self.rules.min_pad_clearance:
                    self._add_violation(
                        DRCViolationType.CLEARANCE,
                        DRCSeverity.ERROR,
                        f"Track-to-pad clearance: {dist:.3f}mm < {self.rules.min_pad_clearance:.3f}mm",
                        (pad['x'], pad['y']),
                        track['layer'],
                        track['net'], pad['net'],
                        dist, self.rules.min_pad_clearance,
                        f"Track {track['net']}",
                        f"Pad {pad['ref']}.{pad['pin']}"
                    )

        # Pad-to-pad clearance (different nets only, different components)
        for i, p1 in enumerate(pads):
            for p2 in pads[i+1:]:
                if p1['layer'] != p2['layer'] and p1['hole'] == 0 and p2['hole'] == 0:
                    continue
                if p1['net'] == p2['net']:
                    continue
                # Skip pads on the same component - footprint defines their spacing
                if p1['ref'] == p2['ref']:
                    continue

                dist = self._pad_to_pad_distance(p1, p2)
                if dist < self.rules.min_clearance:
                    self._add_violation(
                        DRCViolationType.CLEARANCE,
                        DRCSeverity.ERROR,
                        f"Pad clearance violation: {dist:.3f}mm < {self.rules.min_clearance:.3f}mm",
                        ((p1['x'] + p2['x']) / 2, (p1['y'] + p2['y']) / 2),
                        p1['layer'],
                        p1['net'], p2['net'],
                        dist, self.rules.min_clearance,
                        f"Pad {p1['ref']}.{p1['pin']}",
                        f"Pad {p2['ref']}.{p2['pin']}"
                    )

    def _track_to_track_distance(self, t1: Dict, t2: Dict) -> float:
        """Calculate minimum distance between two track segments"""
        # Simplified: use segment-to-segment distance
        return self._segment_distance(
            t1['start'], t1['end'],
            t2['start'], t2['end']
        ) - (t1['width'] + t2['width']) / 2

    def _track_to_pad_distance(self, track: Dict, pad: Dict) -> float:
        """Calculate minimum distance from track to pad"""
        # Distance from segment to pad center, minus radii
        point_dist = self._point_to_segment_distance(
            (pad['x'], pad['y']),
            track['start'], track['end']
        )
        pad_radius = max(pad['width'], pad['height']) / 2
        track_radius = track['width'] / 2
        return point_dist - pad_radius - track_radius

    def _pad_to_pad_distance(self, p1: Dict, p2: Dict) -> float:
        """Calculate minimum distance between two pads"""
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        center_dist = math.sqrt(dx*dx + dy*dy)
        r1 = max(p1['width'], p1['height']) / 2
        r2 = max(p2['width'], p2['height']) / 2
        return center_dist - r1 - r2

    # =========================================================================
    # TRACK WIDTH CHECKS
    # =========================================================================

    def _check_track_widths(self, tracks: List[Dict]):
        """Check track width rules"""
        for track in tracks:
            width = track['width']

            if width < self.rules.min_track_width:
                self._add_violation(
                    DRCViolationType.TRACK_WIDTH,
                    DRCSeverity.ERROR,
                    f"Track too narrow: {width:.3f}mm < {self.rules.min_track_width:.3f}mm",
                    self._midpoint(track['start'], track['end']),
                    track['layer'],
                    track['net'], '',
                    width, self.rules.min_track_width,
                    f"Track {track['net']}"
                )

            if width > self.rules.max_track_width:
                self._add_violation(
                    DRCViolationType.TRACK_WIDTH,
                    DRCSeverity.WARNING,
                    f"Track too wide: {width:.3f}mm > {self.rules.max_track_width:.3f}mm",
                    self._midpoint(track['start'], track['end']),
                    track['layer'],
                    track['net'], '',
                    width, self.rules.max_track_width,
                    f"Track {track['net']}"
                )

    # =========================================================================
    # VIA CHECKS
    # =========================================================================

    def _check_vias(self, vias: List):
        """Check via specifications"""
        for via in vias:
            # Handle dict, tuple, or Via object
            if isinstance(via, dict):
                pos = via.get('position', (0, 0))
                diameter = via.get('diameter', 0.8)
                drill = via.get('drill', 0.4)
            elif isinstance(via, (list, tuple)):
                pos = (via[0], via[1])
                diameter = 0.8
                drill = 0.4
            elif hasattr(via, 'x'):  # Via object with attributes
                pos = (via.x, via.y)
                diameter = getattr(via, 'diameter', 0.8)
                drill = getattr(via, 'drill', 0.4)
            else:
                continue  # Skip unrecognized via format

            # Check drill size
            if drill < self.rules.min_via_drill:
                self._add_violation(
                    DRCViolationType.VIA_DRILL,
                    DRCSeverity.ERROR,
                    f"Via drill too small: {drill:.3f}mm < {self.rules.min_via_drill:.3f}mm",
                    pos if isinstance(pos, tuple) else (pos, pos),
                    '',
                    '', '',
                    drill, self.rules.min_via_drill,
                    f"Via at ({pos[0]:.2f}, {pos[1]:.2f})"
                )

            # Check annular ring
            annular = (diameter - drill) / 2
            if annular < self.rules.min_via_annular:
                self._add_violation(
                    DRCViolationType.VIA_ANNULAR,
                    DRCSeverity.ERROR,
                    f"Via annular ring too small: {annular:.3f}mm < {self.rules.min_via_annular:.3f}mm",
                    pos if isinstance(pos, tuple) else (pos, pos),
                    '',
                    '', '',
                    annular, self.rules.min_via_annular,
                    f"Via at ({pos[0]:.2f}, {pos[1]:.2f})"
                )

    # =========================================================================
    # HOLE CHECKS
    # =========================================================================

    def _check_holes(self, pads: List[Dict], vias: List):
        """Check hole specifications"""
        holes = []

        # Collect holes from THT pads
        for pad in pads:
            if pad['hole'] > 0:
                holes.append({
                    'x': pad['x'],
                    'y': pad['y'],
                    'size': pad['hole'],
                    'ref': f"{pad['ref']}.{pad['pin']}"
                })

        # Add vias - handle dict, tuple, or Via object using get_xy
        for via in vias:
            try:
                if isinstance(via, dict):
                    pos = via.get('position', (0, 0))
                    drill = via.get('drill', 0.4)
                    x, y = get_xy(pos)
                elif isinstance(via, (list, tuple)) and len(via) >= 2:
                    x, y = float(via[0]), float(via[1])
                    drill = 0.4
                elif hasattr(via, 'position'):
                    # Via dataclass from routing_types
                    x, y = get_xy(via.position)
                    drill = getattr(via, 'drill', 0.4)
                else:
                    continue  # Skip unrecognized via format
                holes.append({
                    'x': x,
                    'y': y,
                    'size': drill,
                    'ref': 'Via'
                })
            except (TypeError, ValueError):
                continue  # Skip invalid via data

        # Check hole sizes
        for hole in holes:
            if hole['size'] < self.rules.min_hole_size:
                self._add_violation(
                    DRCViolationType.HOLE_SIZE,
                    DRCSeverity.ERROR,
                    f"Hole too small: {hole['size']:.3f}mm < {self.rules.min_hole_size:.3f}mm",
                    (hole['x'], hole['y']),
                    '',
                    '', '',
                    hole['size'], self.rules.min_hole_size,
                    hole['ref']
                )

        # Check hole spacing
        for i, h1 in enumerate(holes):
            for h2 in holes[i+1:]:
                dx = h2['x'] - h1['x']
                dy = h2['y'] - h1['y']
                dist = math.sqrt(dx*dx + dy*dy)

                min_spacing = self.rules.min_hole_spacing + (h1['size'] + h2['size']) / 2

                if dist < min_spacing:
                    self._add_violation(
                        DRCViolationType.HOLE_SPACING,
                        DRCSeverity.ERROR,
                        f"Holes too close: {dist:.3f}mm < {min_spacing:.3f}mm",
                        ((h1['x'] + h2['x']) / 2, (h1['y'] + h2['y']) / 2),
                        '',
                        '', '',
                        dist, min_spacing,
                        h1['ref'], h2['ref']
                    )

    # =========================================================================
    # EDGE CLEARANCE CHECKS
    # =========================================================================

    def _check_edge_clearance(self, pads: List[Dict], tracks: List[Dict], vias: List):
        """Check edge clearance rules"""
        ox = self.config.board_origin_x
        oy = self.config.board_origin_y
        bw = self.config.board_width
        bh = self.config.board_height

        edges = [
            ('left', ox),
            ('right', ox + bw),
            ('bottom', oy),
            ('top', oy + bh)
        ]

        # Check pads
        for pad in pads:
            pad_radius = max(pad['width'], pad['height']) / 2

            for edge_name, edge_val in edges:
                if edge_name in ['left', 'right']:
                    dist = abs(pad['x'] - edge_val) - pad_radius
                else:
                    dist = abs(pad['y'] - edge_val) - pad_radius

                if dist < self.rules.min_edge_clearance:
                    self._add_violation(
                        DRCViolationType.EDGE_CLEARANCE,
                        DRCSeverity.ERROR,
                        f"Pad too close to {edge_name} edge: {dist:.3f}mm",
                        (pad['x'], pad['y']),
                        pad['layer'],
                        pad['net'], '',
                        dist, self.rules.min_edge_clearance,
                        f"Pad {pad['ref']}.{pad['pin']}"
                    )

        # Check track endpoints
        for track in tracks:
            for point in [track['start'], track['end']]:
                for edge_name, edge_val in edges:
                    if edge_name in ['left', 'right']:
                        dist = abs(point[0] - edge_val) - track['width'] / 2
                    else:
                        dist = abs(point[1] - edge_val) - track['width'] / 2

                    if dist < self.rules.min_edge_clearance:
                        self._add_violation(
                            DRCViolationType.EDGE_CLEARANCE,
                            DRCSeverity.WARNING,
                            f"Track too close to {edge_name} edge: {dist:.3f}mm",
                            point,
                            track['layer'],
                            track['net'], '',
                            dist, self.rules.min_edge_clearance,
                            f"Track {track['net']}"
                        )

    # =========================================================================
    # SILKSCREEN CHECKS
    # =========================================================================

    def _check_silkscreen(self, pads: List[Dict], silkscreen):
        """Check silkscreen over pads"""
        # Handle SilkscreenResult object (Bug #15 fix)
        # The silkscreen parameter might be a SilkscreenResult object, not a list
        if silkscreen is None:
            return

        # Extract list of silkscreen items from SilkscreenResult
        silk_items = []
        if hasattr(silkscreen, 'texts'):
            # It's a SilkscreenResult object
            silk_items.extend(getattr(silkscreen, 'texts', []) or [])
            silk_items.extend(getattr(silkscreen, 'polarity_markers', []) or [])
        elif isinstance(silkscreen, (list, tuple)):
            silk_items = silkscreen
        else:
            return  # Unknown format

        for silk in silk_items:
            # Handle both dict and SilkscreenText objects
            if isinstance(silk, dict):
                silk_x = silk.get('x', 0)
                silk_y = silk.get('y', 0)
                silk_layer = silk.get('layer', 'F.SilkS')
                silk_text = silk.get('text', 'element')
            elif hasattr(silk, 'x'):  # SilkscreenText dataclass
                silk_x = silk.x
                silk_y = silk.y
                silk_layer = getattr(silk, 'layer', 'F.SilkS')
                silk_text = getattr(silk, 'text', 'element')
            else:
                continue  # Skip unrecognized format

            # Check if silkscreen overlaps any SMD pad
            for pad in pads:
                if pad['hole'] > 0:
                    continue  # THT pads usually have solder mask

                # Simple distance check
                dx = silk_x - pad['x']
                dy = silk_y - pad['y']
                dist = math.sqrt(dx*dx + dy*dy)

                pad_radius = max(pad['width'], pad['height']) / 2

                if dist < pad_radius + self.rules.min_silk_clearance:
                    self._add_violation(
                        DRCViolationType.SILK_OVER_PAD,
                        DRCSeverity.WARNING,
                        f"Silkscreen too close to pad",
                        (silk_x, silk_y),
                        silk_layer,
                        '', pad['net'],
                        dist - pad_radius, self.rules.min_silk_clearance,
                        f"Silkscreen '{silk_text}'",
                        f"Pad {pad['ref']}.{pad['pin']}"
                    )

    # =========================================================================
    # CONNECTIVITY CHECKS
    # =========================================================================

    def _check_connectivity(self, parts_db: Dict, routes: Dict):
        """Check that all nets are properly connected"""
        nets = parts_db.get('nets', {})

        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])
            if len(pins) < 2:
                continue

            # Check if this net has a route
            route = routes.get(net_name)
            if not route:
                self._add_violation(
                    DRCViolationType.UNCONNECTED,
                    DRCSeverity.ERROR,
                    f"Net '{net_name}' has no route ({len(pins)} pins unconnected)",
                    (0, 0),
                    '',
                    net_name, '',
                    0, len(pins)
                )
            elif hasattr(route, 'success') and not route.success:
                self._add_violation(
                    DRCViolationType.UNCONNECTED,
                    DRCSeverity.ERROR,
                    f"Net '{net_name}' routing failed: {getattr(route, 'error', 'unknown')}",
                    (0, 0),
                    '',
                    net_name, '',
                    0, len(pins)
                )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _add_violation(self, vtype: DRCViolationType, severity: DRCSeverity,
                       message: str, location: Tuple[float, float], layer: str,
                       net1: str, net2: str, actual: float, required: float,
                       obj1: str = '', obj2: str = ''):
        """Add a DRC violation"""
        self.violations.append(DRCViolation(
            violation_type=vtype,
            severity=severity,
            message=message,
            location=location,
            layer=layer,
            net1=net1,
            net2=net2,
            actual_value=actual,
            required_value=required,
            object1=obj1,
            object2=obj2
        ))

    def _midpoint(self, p1: Tuple, p2: Tuple) -> Tuple[float, float]:
        """Calculate midpoint between two points"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _segment_distance(self, s1: Tuple, e1: Tuple, s2: Tuple, e2: Tuple) -> float:
        """Calculate minimum distance between two line segments"""
        # Simplified: check endpoints and midpoints
        points1 = [s1, e1, self._midpoint(s1, e1)]
        points2 = [s2, e2, self._midpoint(s2, e2)]

        min_dist = float('inf')
        for p1 in points1:
            dist = self._point_to_segment_distance(p1, s2, e2)
            min_dist = min(min_dist, dist)
        for p2 in points2:
            dist = self._point_to_segment_distance(p2, s1, e1)
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_to_segment_distance(self, point: Tuple, seg_start: Tuple, seg_end: Tuple) -> float:
        """Calculate distance from point to line segment"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx*dx + dy*dy

        if length_sq == 0:
            # Segment is a point
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        # Project point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    # =========================================================================
    # ACID TRAP CHECK (Manufacturing - DFM)
    # Research: https://www.nextpcb.com/blog/acid-traps
    # Angles < 90° trap etchant during PCB manufacturing, causing open circuits
    # =========================================================================

    def _check_acid_traps(self, tracks: List[Dict]):
        """
        Check for acid traps (acute angles) at track junctions.

        Acid traps occur when two trace segments meet at an angle < 90°.
        During etching, etchant gets trapped in the acute corner, over-etching
        the copper and potentially creating an open circuit.
        """
        if not self.config.check_acid_traps:
            return

        min_angle = self.rules.min_acid_trap_angle

        # Group tracks by endpoint to find junctions
        junctions = {}  # {(x, y): [track_indices]}

        for i, track in enumerate(tracks):
            for point in [track['start'], track['end']]:
                key = self._round_point(point)
                if key not in junctions:
                    junctions[key] = []
                junctions[key].append(i)

        # Check angles at each junction with 2+ tracks
        checked_pairs = set()
        for point, indices in junctions.items():
            if len(indices) < 2:
                continue

            # Check all pairs of tracks at this junction
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    pair_key = (min(idx1, idx2), max(idx1, idx2))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    t1, t2 = tracks[idx1], tracks[idx2]

                    # Skip if different layers
                    if t1['layer'] != t2['layer']:
                        continue

                    # Calculate angle between tracks
                    angle = self._angle_at_junction(t1, t2, point)

                    if angle < min_angle:
                        self._add_violation(
                            DRCViolationType.ACID_TRAP,
                            DRCSeverity.WARNING,
                            f"Acid trap: {angle:.1f}° angle < {min_angle:.1f}° minimum",
                            point,
                            t1['layer'],
                            t1['net'], t2['net'],
                            angle, min_angle,
                            f"Track junction"
                        )

    def _angle_at_junction(self, t1: Dict, t2: Dict,
                           junction: Tuple[float, float]) -> float:
        """Calculate angle between two tracks at a junction point"""
        # Get vectors pointing away from junction
        def get_vector(track, junction):
            jp = self._round_point(junction)
            sp = self._round_point(track['start'])
            ep = self._round_point(track['end'])

            if sp == jp:
                return (track['end'][0] - track['start'][0],
                        track['end'][1] - track['start'][1])
            else:
                return (track['start'][0] - track['end'][0],
                        track['start'][1] - track['end'][1])

        v1 = get_vector(t1, junction)
        v2 = get_vector(t2, junction)

        # Calculate angle using dot product
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if len1 < 0.001 or len2 < 0.001:
            return 180.0  # Degenerate case

        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
        dot = max(-1, min(1, dot))  # Clamp for acos

        return math.degrees(math.acos(dot))

    def _round_point(self, point: Tuple[float, float],
                     precision: int = 3) -> Tuple[float, float]:
        """Round point for comparison"""
        return (round(point[0], precision), round(point[1], precision))

    # =========================================================================
    # SOLDER MASK SLIVER CHECK
    # Research: https://www.protoexpress.com/blog/6-common-solder-mask-errors/
    # Minimum sliver width is typically 0.1mm (4 mils)
    # =========================================================================

    def _check_solder_mask_slivers(self, pads: List[Dict]):
        """
        Check for solder mask slivers between pads.

        When two pads are close together, the solder mask between them
        becomes a narrow "sliver" that may peel or bridge during reflow.
        """
        if not self.config.check_solder_mask:
            return

        min_sliver = self.rules.min_mask_sliver
        mask_expansion = self.rules.mask_expansion

        for i, p1 in enumerate(pads):
            for p2 in pads[i+1:]:
                # Skip if different layers (SMD pads)
                if p1['layer'] != p2['layer'] and p1['hole'] == 0 and p2['hole'] == 0:
                    continue

                # Calculate edge-to-edge distance
                dx = abs(p2['x'] - p1['x'])
                dy = abs(p2['y'] - p1['y'])
                center_dist = math.sqrt(dx*dx + dy*dy)

                # Pad edge distances (considering mask expansion)
                r1 = max(p1['width'], p1['height']) / 2 + mask_expansion
                r2 = max(p2['width'], p2['height']) / 2 + mask_expansion

                # Solder mask sliver width
                sliver = center_dist - r1 - r2

                if 0 < sliver < min_sliver:
                    self._add_violation(
                        DRCViolationType.SOLDER_MASK_SLIVER,
                        DRCSeverity.WARNING,
                        f"Solder mask sliver: {sliver:.3f}mm < {min_sliver:.3f}mm minimum",
                        ((p1['x'] + p2['x']) / 2, (p1['y'] + p2['y']) / 2),
                        p1['layer'],
                        p1['net'], p2['net'],
                        sliver, min_sliver,
                        f"Pad {p1['ref']}.{p1['pin']}",
                        f"Pad {p2['ref']}.{p2['pin']}"
                    )

    # =========================================================================
    # COPPER SLIVER CHECK
    # Narrow copper features may not etch properly or may break
    # =========================================================================

    def _check_copper_slivers(self, tracks: List[Dict]):
        """Check for narrow copper features that may cause manufacturing issues"""
        if not self.config.check_copper_slivers:
            return

        min_width = self.rules.min_copper_sliver

        # Track widths already checked in _check_track_widths
        # This is for more complex geometry - narrow copper between close tracks
        # For now, we focus on track width which is the primary source
        pass

    # =========================================================================
    # VOLTAGE SPACING CHECK (IPC-2221)
    # High voltage requires larger clearance between conductors
    # =========================================================================

    def _check_voltage_spacing(self, pads: List[Dict], tracks: List[Dict]):
        """
        Check voltage-dependent clearance per IPC-2221 Table 6-1.

        Higher voltages require larger spacing between conductors to
        prevent arcing and breakdown.
        """
        if not self.config.check_voltage_spacing:
            return

        net_voltages = self.config.net_voltages
        if not net_voltages:
            return

        # Check pad-to-pad spacing for high voltage nets
        high_voltage_pads = [p for p in pads if net_voltages.get(p['net'], 0) > 50]

        for i, p1 in enumerate(high_voltage_pads):
            v1 = net_voltages.get(p1['net'], 0)

            for p2 in pads[i+1:]:
                if p1['net'] == p2['net']:
                    continue

                v2 = net_voltages.get(p2['net'], 0)
                voltage_diff = abs(v1 - v2)

                if voltage_diff < 50:
                    continue

                # Get required spacing for this voltage difference
                is_internal = 'In' in p1['layer']
                required_spacing = IPC2221Calculator.get_voltage_spacing(
                    voltage_diff, is_internal
                )

                # Calculate actual spacing
                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                center_dist = math.sqrt(dx*dx + dy*dy)
                r1 = max(p1['width'], p1['height']) / 2
                r2 = max(p2['width'], p2['height']) / 2
                actual_spacing = center_dist - r1 - r2

                if actual_spacing < required_spacing:
                    self._add_violation(
                        DRCViolationType.VOLTAGE_SPACING,
                        DRCSeverity.ERROR,
                        f"Voltage spacing: {actual_spacing:.3f}mm < {required_spacing:.3f}mm "
                        f"for {voltage_diff:.0f}V difference (IPC-2221)",
                        ((p1['x'] + p2['x']) / 2, (p1['y'] + p2['y']) / 2),
                        p1['layer'],
                        p1['net'], p2['net'],
                        actual_spacing, required_spacing,
                        f"Pad {p1['ref']}.{p1['pin']} ({v1:.0f}V)",
                        f"Pad {p2['ref']}.{p2['pin']} ({v2:.0f}V)"
                    )

    # =========================================================================
    # CURRENT CAPACITY CHECK (IPC-2221)
    # Traces must be wide enough for the current they carry
    # =========================================================================

    def _check_current_capacity(self, tracks: List[Dict]):
        """
        Check trace width meets current capacity requirements per IPC-2221.

        Uses formula: I = k * ΔT^0.44 * A^0.725
        where k=0.048 (external) or k=0.024 (internal)
        """
        if not self.config.check_current_capacity:
            return

        net_currents = self.config.net_currents
        if not net_currents:
            return

        temp_rise = self.rules.max_temp_rise
        copper_oz = self.rules.copper_weight_oz

        for track in tracks:
            net = track['net']
            current = net_currents.get(net, 0)

            if current <= 0:
                continue

            is_external = track['layer'] in ('F.Cu', 'B.Cu')
            required_width = IPC2221Calculator.calculate_trace_width(
                current, temp_rise, copper_oz, is_external
            )

            actual_width = track['width']

            if actual_width < required_width:
                capacity = IPC2221Calculator.calculate_current_capacity(
                    actual_width, temp_rise, copper_oz, is_external
                )

                self._add_violation(
                    DRCViolationType.CURRENT_CAPACITY,
                    DRCSeverity.ERROR,
                    f"Current capacity: {actual_width:.3f}mm trace can carry {capacity:.2f}A, "
                    f"need {required_width:.3f}mm for {current:.2f}A",
                    self._midpoint(track['start'], track['end']),
                    track['layer'],
                    net, '',
                    actual_width, required_width,
                    f"Track {net} ({current:.2f}A)"
                )

    # =========================================================================
    # TRACK CROSSING/SHORT CHECK - BENTLEY-OTTMANN ALGORITHM
    # Research: https://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
    # O((n+k)log n) vs naive O(n²) - much faster for large designs
    # =========================================================================

    def _check_track_crossings(self, tracks: List[Dict]):
        """
        Check for track-to-track crossings on the same layer.

        Uses naive O(n²) method for small designs (< 100 tracks) for reliability,
        and BENTLEY-OTTMANN sweepline algorithm for larger designs.

        Bug #16 fix: Use naive method for small track counts where B-O overhead
        isn't worth it and where edge cases (vertical lines) might cause issues.

        Research sources:
        - https://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
        - https://www.cse.cuhk.edu.hk/~byu/papers/C172-DAC2023-OpenDRC.pdf
        """
        # For small track counts, use reliable naive O(n²) method
        if len(tracks) < 100:
            self._check_track_crossings_naive(tracks)
            return

        # Group tracks by layer (intersections only matter on same layer)
        tracks_by_layer: Dict[str, List[Dict]] = {}
        for i, track in enumerate(tracks):
            layer = track['layer']
            if layer not in tracks_by_layer:
                tracks_by_layer[layer] = []
            tracks_by_layer[layer].append((i, track))

        # Check each layer using Bentley-Ottmann
        for layer, layer_tracks in tracks_by_layer.items():
            if len(layer_tracks) < 2:
                continue

            # Convert to BOSegment format
            bo_segments = []
            for i, track in layer_tracks:
                bo_segments.append(BOSegment(
                    id=i,
                    start=track['start'],
                    end=track['end'],
                    net=track['net'],
                    layer=layer
                ))

            # Run Bentley-Ottmann algorithm
            bo = BentleyOttmann()
            intersections = bo.find_intersections(bo_segments)

            # Report crossings between different nets
            for intersection_point, seg1, seg2 in intersections:
                if seg1.net != seg2.net:
                    self._add_violation(
                        DRCViolationType.SHORT_CIRCUIT,
                        DRCSeverity.ERROR,
                        f"Tracks crossing/shorting: [{seg1.net}] and [{seg2.net}]",
                        intersection_point,
                        layer,
                        seg1.net, seg2.net,
                        0, 0,
                        f"Track {seg1.net}",
                        f"Track {seg2.net}"
                    )

    def _check_track_crossings_naive(self, tracks: List[Dict]):
        """
        Naive O(n²) track crossing check - fallback method.

        Used when track count is small (<100) where the overhead of
        Bentley-Ottmann isn't worthwhile.
        """
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i+1:]:
                # Skip same net
                if t1['net'] == t2['net']:
                    continue

                # Skip different layers
                if t1['layer'] != t2['layer']:
                    continue

                # Check for intersection
                intersection = self._segments_intersect(
                    t1['start'], t1['end'],
                    t2['start'], t2['end']
                )

                if intersection:
                    self._add_violation(
                        DRCViolationType.SHORT_CIRCUIT,
                        DRCSeverity.ERROR,
                        f"Tracks crossing/shorting: [{t1['net']}] and [{t2['net']}]",
                        intersection,
                        t1['layer'],
                        t1['net'], t2['net'],
                        0, 0,
                        f"Track {t1['net']}",
                        f"Track {t2['net']}"
                    )

    # =========================================================================
    # CLEARANCE CHECK WITH SPATIAL INDEX
    # Uses grid-based spatial indexing for faster neighbor queries
    # =========================================================================

    def _check_clearances_fast(self, pads: List[Dict], tracks: List[Dict], vias: List):
        """
        Fast clearance check using spatial indexing.

        Uses grid-based spatial index for O(n) average case instead of O(n²).
        Based on R-tree concepts: https://en.wikipedia.org/wiki/R-tree
        """
        clearance = self.rules.min_clearance

        # Build spatial index
        index = SpatialIndex(cell_size=max(2.0, clearance * 10))

        # Index all tracks
        for i, track in enumerate(tracks):
            bbox = BoundingBox.from_segment(
                track['start'], track['end'], track['width']
            ).expand(clearance)
            index.insert(bbox, ('track', i, track))

        # Index all pads
        for i, pad in enumerate(pads):
            half_w = pad['width'] / 2
            half_h = pad['height'] / 2
            bbox = BoundingBox(
                pad['x'] - half_w - clearance,
                pad['y'] - half_h - clearance,
                pad['x'] + half_w + clearance,
                pad['y'] + half_h + clearance
            )
            index.insert(bbox, ('pad', i, pad))

        # Check each track against nearby elements
        for i, track in enumerate(tracks):
            track_bbox = BoundingBox.from_segment(
                track['start'], track['end'], track['width']
            ).expand(clearance)

            # Query nearby elements
            nearby = index.query(track_bbox)

            for item_type, item_idx, item in nearby:
                if item_type == 'track' and item_idx > i:
                    # Track-to-track (only check once per pair)
                    t2 = item
                    if track['layer'] != t2['layer']:
                        continue
                    if track['net'] == t2['net']:
                        continue

                    dist = self._track_to_track_distance(track, t2)
                    if dist < self.rules.min_track_clearance:
                        self._add_violation(
                            DRCViolationType.CLEARANCE,
                            DRCSeverity.ERROR,
                            f"Track clearance: {dist:.3f}mm < {self.rules.min_track_clearance:.3f}mm",
                            self._midpoint(track['start'], track['end']),
                            track['layer'],
                            track['net'], t2['net'],
                            dist, self.rules.min_track_clearance,
                            f"Track {track['net']}",
                            f"Track {t2['net']}"
                        )

                elif item_type == 'pad':
                    pad = item
                    if track['layer'] != pad['layer'] and pad['hole'] == 0:
                        continue
                    if track['net'] == pad['net']:
                        continue

                    dist = self._track_to_pad_distance(track, pad)
                    if dist < self.rules.min_pad_clearance:
                        self._add_violation(
                            DRCViolationType.CLEARANCE,
                            DRCSeverity.ERROR,
                            f"Track-to-pad clearance: {dist:.3f}mm < {self.rules.min_pad_clearance:.3f}mm",
                            (pad['x'], pad['y']),
                            track['layer'],
                            track['net'], pad['net'],
                            dist, self.rules.min_pad_clearance,
                            f"Track {track['net']}",
                            f"Pad {pad['ref']}.{pad['pin']}"
                        )

    def _segments_intersect(self, p1: Tuple, p2: Tuple,
                           p3: Tuple, p4: Tuple) -> Optional[Tuple[float, float]]:
        """
        Check if two line segments intersect (crossing, not just touching).

        Uses parametric line intersection algorithm.
        Returns intersection point if they cross, None otherwise.

        For DRC track crossings, we detect when one track crosses THROUGH another,
        not when they touch at endpoints (that's handled by clearance checks).
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        cross = dx1 * dy2 - dy1 * dx2

        if abs(cross) < 1e-10:
            # Parallel lines - don't report as crossing (clearance check handles this)
            return None

        dx3 = x1 - x3
        dy3 = y1 - y3

        t1 = (dx2 * dy3 - dy2 * dx3) / cross
        t2 = (dx1 * dy3 - dy1 * dx3) / cross

        # For a TRUE crossing, the intersection must be:
        # - Strictly inside segment 1 (not at its endpoints): 0 < t1 < 1
        # - Within segment 2 (including endpoints): 0 <= t2 <= 1
        # OR
        # - Within segment 1 (including endpoints): 0 <= t1 <= 1
        # - Strictly inside segment 2 (not at its endpoints): 0 < t2 < 1
        #
        # This detects when one track passes THROUGH another track
        eps = 0.001  # 1 micron tolerance

        in_seg1 = eps < t1 < 1 - eps  # Strictly inside segment 1
        in_seg2 = eps < t2 < 1 - eps  # Strictly inside segment 2
        on_seg1 = -eps <= t1 <= 1 + eps  # On segment 1 (including endpoints)
        on_seg2 = -eps <= t2 <= 1 + eps  # On segment 2 (including endpoints)

        # Crossing occurs if intersection is strictly inside at least one segment
        # and on the other segment
        if (in_seg1 and on_seg2) or (in_seg2 and on_seg1):
            ix = x1 + t1 * dx1
            iy = y1 + t1 * dy1
            return (ix, iy)

        return None

    # =========================================================================
    # DIFFERENTIAL PAIR CHECK
    # Differential pairs need matched length and consistent spacing
    # =========================================================================

    def _check_differential_pairs(self, tracks: List[Dict]):
        """
        Check differential pair routing requirements.

        Differential pairs must have:
        - Consistent spacing between the two traces
        - Matched length (within skew tolerance)
        """
        if not self.config.check_diff_pairs:
            return

        diff_pairs = self.rules.diff_pair_nets
        if not diff_pairs:
            return

        max_skew = self.rules.max_diff_pair_skew

        for net_p, net_n in diff_pairs:
            # Find tracks for each net
            tracks_p = [t for t in tracks if t['net'] == net_p]
            tracks_n = [t for t in tracks if t['net'] == net_n]

            if not tracks_p or not tracks_n:
                continue

            # Calculate total length of each net
            len_p = sum(self._track_length(t) for t in tracks_p)
            len_n = sum(self._track_length(t) for t in tracks_n)

            skew = abs(len_p - len_n)

            if skew > max_skew:
                self._add_violation(
                    DRCViolationType.DIFF_PAIR_SKEW,
                    DRCSeverity.WARNING,
                    f"Differential pair skew: {skew:.3f}mm > {max_skew:.3f}mm max",
                    tracks_p[0]['start'] if tracks_p else (0, 0),
                    '',
                    net_p, net_n,
                    skew, max_skew,
                    f"Pair: {net_p}/{net_n}"
                )

    def _track_length(self, track: Dict) -> float:
        """Calculate length of a track segment"""
        dx = track['end'][0] - track['start'][0]
        dy = track['end'][1] - track['start'][1]
        return math.sqrt(dx*dx + dy*dy)

    # =========================================================================
    # ENHANCED MAIN CHECK METHOD
    # =========================================================================

    def check_enhanced(self, parts_db: Dict, placement: Dict, routes: Dict,
                       vias: List = None, silkscreen: List = None) -> DRCResult:
        """
        Run complete enhanced DRC check with all IPC-2221 checks.

        This is the full version with:
        - All basic checks (clearance, track width, vias, etc.)
        - Acid trap detection
        - Solder mask sliver check
        - Voltage spacing (IPC-2221)
        - Current capacity (IPC-2221)
        - Track crossing detection
        - Differential pair checks
        """
        self.violations.clear()
        vias = vias or []
        silkscreen = silkscreen or []

        # Build data structures
        pads = self._extract_pads(parts_db, placement)
        tracks = self._extract_tracks(routes)

        # === Basic checks ===
        if self.config.check_clearances:
            self._check_clearances(pads, tracks, vias)

        if self.config.check_track_widths:
            self._check_track_widths(tracks)

        if self.config.check_vias:
            self._check_vias(vias)

        if self.config.check_holes:
            self._check_holes(pads, vias)

        if self.config.check_edge_clearance:
            self._check_edge_clearance(pads, tracks, vias)

        if self.config.check_silkscreen:
            self._check_silkscreen(pads, silkscreen)

        if self.config.check_connectivity:
            self._check_connectivity(parts_db, routes)

        # === Enhanced manufacturing checks ===
        if self.config.check_acid_traps:
            self._check_acid_traps(tracks)

        if self.config.check_solder_mask:
            self._check_solder_mask_slivers(pads)

        # === Track crossing/short detection ===
        self._check_track_crossings(tracks)

        # === IPC-2221 checks ===
        if self.config.check_voltage_spacing:
            self._check_voltage_spacing(pads, tracks)

        if self.config.check_current_capacity:
            self._check_current_capacity(tracks)

        # === Signal integrity checks ===
        if self.config.check_diff_pairs:
            self._check_differential_pairs(tracks)

        # Summarize results
        error_count = sum(1 for v in self.violations if v.severity == DRCSeverity.ERROR)
        warning_count = sum(1 for v in self.violations if v.severity == DRCSeverity.WARNING)
        info_count = sum(1 for v in self.violations if v.severity == DRCSeverity.INFO)

        violations_by_type = {}
        for v in self.violations:
            key = v.violation_type.value
            violations_by_type[key] = violations_by_type.get(key, 0) + 1

        return DRCResult(
            violations=self.violations.copy(),
            passed=error_count == 0,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            violations_by_type=violations_by_type
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_report(self, result: DRCResult) -> str:
        """Generate human-readable DRC report"""
        lines = []
        lines.append("=" * 60)
        lines.append("DRC REPORT")
        lines.append("=" * 60)
        lines.append(f"\nStatus: {'PASSED' if result.passed else 'FAILED'}")
        lines.append(f"Errors: {result.error_count}")
        lines.append(f"Warnings: {result.warning_count}")
        lines.append(f"Info: {result.info_count}")

        if result.violations_by_type:
            lines.append("\nViolations by Type:")
            for vtype, count in sorted(result.violations_by_type.items()):
                lines.append(f"  {vtype}: {count}")

        if result.violations:
            lines.append("\nDetailed Violations:")
            lines.append("-" * 60)

            for i, v in enumerate(result.violations, 1):
                lines.append(f"\n{i}. [{v.severity.value.upper()}] {v.violation_type.value}")
                lines.append(f"   {v.message}")
                lines.append(f"   Location: ({v.location[0]:.3f}, {v.location[1]:.3f})")
                if v.layer:
                    lines.append(f"   Layer: {v.layer}")
                if v.net1:
                    lines.append(f"   Net: {v.net1}" + (f" / {v.net2}" if v.net2 else ""))
                if v.object1:
                    lines.append(f"   Objects: {v.object1}" + (f" <-> {v.object2}" if v.object2 else ""))

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    # =========================================================================
    # KICAD-COMPATIBLE ADDITIONAL CHECKS
    # =========================================================================

    def _check_dangling_tracks(self, tracks: List[Dict], pads: List[Dict], vias: List):
        """
        Check for track segments with unconnected ends.

        KiCad: track_dangling - "Track has unconnected end"

        A track end is dangling if it doesn't connect to:
        - Another track segment on the same net
        - A pad on the same net
        - A via on the same net
        """
        if not tracks:
            return

        # Build connection points for each net
        # {net: set of (x, y) connection points}
        net_endpoints: Dict[str, Set[Tuple[float, float]]] = {}

        # Add pad centers as connection points
        for pad in pads:
            net = pad.get('net', '')
            if net:
                if net not in net_endpoints:
                    net_endpoints[net] = set()
                net_endpoints[net].add((round(pad['x'], 3), round(pad['y'], 3)))

        # Add via positions as connection points
        for via in vias:
            if hasattr(via, 'net'):
                net = via.net
            elif isinstance(via, dict):
                net = via.get('net', '')
            else:
                continue

            if net:
                if net not in net_endpoints:
                    net_endpoints[net] = set()

                if hasattr(via, 'position'):
                    pos = via.position
                elif isinstance(via, dict):
                    pos = via.get('position', (0, 0))
                else:
                    continue

                if isinstance(pos, (list, tuple)):
                    net_endpoints[net].add((round(pos[0], 3), round(pos[1], 3)))

        # Add track endpoints as connection points
        for track in tracks:
            net = track.get('net', '')
            start = track.get('start', (0, 0))
            end = track.get('end', (0, 0))

            if net:
                if net not in net_endpoints:
                    net_endpoints[net] = set()
                net_endpoints[net].add((round(start[0], 3), round(start[1], 3)))
                net_endpoints[net].add((round(end[0], 3), round(end[1], 3)))

        # Now check each track endpoint
        for track in tracks:
            net = track.get('net', '')
            if not net:
                continue

            start = track.get('start', (0, 0))
            end = track.get('end', (0, 0))
            start_rounded = (round(start[0], 3), round(start[1], 3))
            end_rounded = (round(end[0], 3), round(end[1], 3))

            # Count how many times each endpoint appears (should be >= 2 if connected)
            start_count = sum(1 for t in tracks if t.get('net') == net and (
                (round(t['start'][0], 3), round(t['start'][1], 3)) == start_rounded or
                (round(t['end'][0], 3), round(t['end'][1], 3)) == start_rounded
            ))
            end_count = sum(1 for t in tracks if t.get('net') == net and (
                (round(t['start'][0], 3), round(t['start'][1], 3)) == end_rounded or
                (round(t['end'][0], 3), round(t['end'][1], 3)) == end_rounded
            ))

            # Check if start is dangling (only appears once in tracks AND not at a pad/via)
            if start_count == 1:
                # Check if it connects to a pad or via
                connects_to_pad = any(
                    abs(p['x'] - start[0]) < 0.1 and abs(p['y'] - start[1]) < 0.1
                    for p in pads if p.get('net') == net
                )
                connects_to_via = any(
                    self._via_at_point(v, start)
                    for v in vias if self._get_via_net(v) == net
                )

                if not connects_to_pad and not connects_to_via:
                    length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                    self.violations.append(DRCViolation(
                        violation_type=DRCViolationType.DANGLING_TRACK,
                        severity=DRCSeverity.WARNING,
                        message=f"Track has unconnected end",
                        location=start,
                        layer=track.get('layer', 'F.Cu'),
                        net1=net,
                        object1=f"Track [{net}] length {length:.4f} mm"
                    ))

            # Check if end is dangling
            if end_count == 1:
                connects_to_pad = any(
                    abs(p['x'] - end[0]) < 0.1 and abs(p['y'] - end[1]) < 0.1
                    for p in pads if p.get('net') == net
                )
                connects_to_via = any(
                    self._via_at_point(v, end)
                    for v in vias if self._get_via_net(v) == net
                )

                if not connects_to_pad and not connects_to_via:
                    length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                    self.violations.append(DRCViolation(
                        violation_type=DRCViolationType.DANGLING_TRACK,
                        severity=DRCSeverity.WARNING,
                        message=f"Track has unconnected end",
                        location=end,
                        layer=track.get('layer', 'F.Cu'),
                        net1=net,
                        object1=f"Track [{net}] length {length:.4f} mm"
                    ))

    def _check_dangling_vias(self, vias: List, tracks: List[Dict]):
        """
        Check for vias that are not connected or only connected on one layer.

        KiCad: via_dangling - "Via is not connected or connected on only one layer"
        """
        if not vias:
            return

        for via in vias:
            # Get via position and net
            if hasattr(via, 'position'):
                pos = via.position
                net = getattr(via, 'net', '')
            elif isinstance(via, dict):
                pos = via.get('position', (0, 0))
                net = via.get('net', '')
            else:
                continue

            if isinstance(pos, (list, tuple)):
                vx, vy = pos[0], pos[1]
            else:
                continue

            # Count tracks connecting to this via, per layer
            layer_connections: Dict[str, int] = {}
            for track in tracks:
                if track.get('net') != net:
                    continue

                start = track.get('start', (0, 0))
                end = track.get('end', (0, 0))
                layer = track.get('layer', 'F.Cu')

                if (abs(start[0] - vx) < 0.1 and abs(start[1] - vy) < 0.1) or \
                   (abs(end[0] - vx) < 0.1 and abs(end[1] - vy) < 0.1):
                    layer_connections[layer] = layer_connections.get(layer, 0) + 1

            total_connections = sum(layer_connections.values())
            layers_connected = len(layer_connections)

            # Dangling if no connections or only one layer connected
            if total_connections == 0 or layers_connected <= 1:
                self.violations.append(DRCViolation(
                    violation_type=DRCViolationType.DANGLING_VIA,
                    severity=DRCSeverity.WARNING,
                    message="Via is not connected or connected on only one layer",
                    location=(vx, vy),
                    layer='F.Cu - B.Cu',
                    net1=net if net else '<no net>',
                    object1=f"Via [{net if net else '<no net>'}]"
                ))

    def _check_colocated_holes(self, pads: List[Dict], vias: List):
        """
        Check for holes at the same position.

        KiCad: holes_co_located - "Drilled holes co-located"
        """
        # Collect all hole positions
        holes: List[Tuple[float, float, str]] = []  # (x, y, description)

        # Add THT pad holes
        for pad in pads:
            if pad.get('is_tht', False):
                holes.append((pad['x'], pad['y'], f"Pad {pad.get('ref', '?')}.{pad.get('number', '?')}"))

        # Add via holes
        for via in vias:
            if hasattr(via, 'position'):
                pos = via.position
                net = getattr(via, 'net', '<no net>')
            elif isinstance(via, dict):
                pos = via.get('position', (0, 0))
                net = via.get('net', '<no net>')
            else:
                continue

            if isinstance(pos, (list, tuple)):
                holes.append((pos[0], pos[1], f"Via [{net}]"))

        # Check for co-located holes
        checked = set()
        for i, (x1, y1, desc1) in enumerate(holes):
            pos1 = (round(x1, 3), round(y1, 3))
            if pos1 in checked:
                continue

            colocated = []
            for j, (x2, y2, desc2) in enumerate(holes):
                if i != j and abs(x1 - x2) < 0.01 and abs(y1 - y2) < 0.01:
                    colocated.append(desc2)

            if colocated:
                checked.add(pos1)
                self.violations.append(DRCViolation(
                    violation_type=DRCViolationType.HOLES_COLOCATED,
                    severity=DRCSeverity.WARNING,
                    message="Drilled holes co-located",
                    location=(x1, y1),
                    object1=desc1,
                    object2=colocated[0] if colocated else ''
                ))

    def _check_silk_overlap(self, silkscreen):
        """
        Check for overlapping silkscreen elements.

        KiCad: silk_overlap - "Silkscreen overlap"
        """
        if not silkscreen:
            return

        # Handle SilkscreenResult object (Bug #15 fix)
        raw_items = []
        if hasattr(silkscreen, 'texts'):
            # It's a SilkscreenResult object
            raw_items.extend(getattr(silkscreen, 'texts', []) or [])
        elif isinstance(silkscreen, (list, tuple)):
            raw_items = list(silkscreen)
        else:
            return  # Unknown format

        if len(raw_items) < 2:
            return

        # Extract silkscreen bounding boxes
        silk_items = []
        for item in raw_items:
            if isinstance(item, dict):
                x = item.get('x', 0)
                y = item.get('y', 0)
                width = item.get('width', 1)
                height = item.get('height', 0.5)
                text = item.get('text', '')
                silk_items.append({
                    'x': x, 'y': y, 'width': width, 'height': height,
                    'text': text,
                    'min_x': x - width/2, 'max_x': x + width/2,
                    'min_y': y - height/2, 'max_y': y + height/2
                })

        # Check for overlaps
        for i, item1 in enumerate(silk_items):
            for j, item2 in enumerate(silk_items):
                if i >= j:
                    continue

                # Check bounding box overlap
                if (item1['min_x'] < item2['max_x'] and item1['max_x'] > item2['min_x'] and
                    item1['min_y'] < item2['max_y'] and item1['max_y'] > item2['min_y']):
                    self.violations.append(DRCViolation(
                        violation_type=DRCViolationType.SILK_OVERLAP,
                        severity=DRCSeverity.WARNING,
                        message="Silkscreen overlap",
                        location=(item1['x'], item1['y']),
                        layer='F.Silkscreen',
                        object1=f"'{item1['text']}'" if item1['text'] else "Silkscreen item",
                        object2=f"'{item2['text']}'" if item2['text'] else "Silkscreen item"
                    ))

    def _via_at_point(self, via, point: Tuple[float, float], tolerance: float = 0.1) -> bool:
        """Check if a via is at a given point"""
        if hasattr(via, 'position'):
            pos = via.position
        elif isinstance(via, dict):
            pos = via.get('position', (0, 0))
        else:
            return False

        if isinstance(pos, (list, tuple)):
            return abs(pos[0] - point[0]) < tolerance and abs(pos[1] - point[1]) < tolerance
        return False

    def _get_via_net(self, via) -> str:
        """Get net name from via"""
        if hasattr(via, 'net'):
            return via.net
        elif isinstance(via, dict):
            return via.get('net', '')
        return ''


# =============================================================================
# GEOMETRY UTILITIES (from validation.py)
# =============================================================================

class GeometryUtils:
    """Utility functions for geometric calculations"""

    @staticmethod
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @staticmethod
    def point_to_segment_distance(point: Tuple[float, float],
                                  seg_start: Tuple[float, float],
                                  seg_end: Tuple[float, float]) -> float:
        """Calculate minimum distance from point to line segment"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx * dx + dy * dy

        if len_sq < 0.0001:  # Degenerate segment
            return GeometryUtils.distance(point, seg_start)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return GeometryUtils.distance(point, (proj_x, proj_y))

    @staticmethod
    def segment_to_segment_distance(s1_start: Tuple[float, float],
                                    s1_end: Tuple[float, float],
                                    s2_start: Tuple[float, float],
                                    s2_end: Tuple[float, float]) -> float:
        """Calculate minimum distance between two line segments"""
        d1 = GeometryUtils.point_to_segment_distance(s1_start, s2_start, s2_end)
        d2 = GeometryUtils.point_to_segment_distance(s1_end, s2_start, s2_end)
        d3 = GeometryUtils.point_to_segment_distance(s2_start, s1_start, s1_end)
        d4 = GeometryUtils.point_to_segment_distance(s2_end, s1_start, s1_end)
        return min(d1, d2, d3, d4)

    @staticmethod
    def bounding_box_overlap(box1: Tuple[float, float, float, float],
                             box2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes overlap (x1, y1, x2, y2)"""
        return not (box1[2] < box2[0] or box1[0] > box2[2] or
                    box1[3] < box2[1] or box1[1] > box2[3])

    @staticmethod
    def angle_between_vectors(v1: Tuple[float, float],
                              v2: Tuple[float, float]) -> float:
        """Calculate angle between two vectors in degrees"""
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if len1 < 0.001 or len2 < 0.001:
            return 0

        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
        dot = max(-1, min(1, dot))

        return math.degrees(math.acos(dot))


# =============================================================================
# QUICK DRC CHECK FUNCTION
# =============================================================================

def quick_drc_check(parts_db: Dict, placement: Dict, routes: Dict,
                    rules: DRCRules = None) -> DRCResult:
    """
    Quick DRC check with default rules.

    Usage:
        from drc_piston import quick_drc_check
        result = quick_drc_check(parts_db, placement, routes)
        if not result.passed:
            print(result.violations)
    """
    config = DRCConfig(rules=rules or DRCRules())
    piston = DRCPiston(config)
    return piston.check(parts_db, placement, routes)
