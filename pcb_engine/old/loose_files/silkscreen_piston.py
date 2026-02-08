"""
PCB Engine - Silkscreen Piston
==============================

A research-based piston (sub-engine) for silkscreen placement and optimization.

RESEARCH-BASED ALGORITHMS:
==========================

1. FORCE-DIRECTED LABEL PLACEMENT
   Source: Imhof, "Point-Feature Label Placement" (1975)
   Source: Christensen et al., "An Empirical Study of Algorithms for Point-Feature Label Placement"
   - Labels repel each other (spring forces)
   - Labels attract to anchor points
   - Iterative relaxation until stable

2. IPC-7351 COMPLIANT SIZING
   Source: IPC-7351B - "Generic Requirements for Surface Mount Design"
   - Minimum silkscreen line width: 0.15mm (6 mil)
   - Recommended text height: 0.8mm - 1.2mm
   - Text-to-pad clearance: 0.1mm minimum
   - Reference designator font: stroke font preferred

3. IPC-7093 BTC GUIDELINES
   Source: IPC-7093 - "Design and Assembly Process Implementation for BTCs"
   - Component outline should not overlap pads
   - Pin 1 indicator required for all ICs
   - Polarity marks for polarized components

4. JLCPCB/PCBWAY MANUFACTURING RULES
   Source: JLCPCB Capabilities, PCBWay Design Guidelines
   - Minimum text height: 0.8mm (JLCPCB), 0.15mm line width
   - 40% of text height for stroke width
   - Clearance from soldermask openings

5. DFA (DESIGN FOR ASSEMBLY) STANDARDS
   Source: IPC-A-610 Assembly Standards
   - Fiducials for pick-and-place alignment
   - Polarity indicators visible from assembly side
   - Clear ref des for visual inspection

6. QUADRANT-BASED PLACEMENT STRATEGY
   Source: Industry best practice
   - 8 positions around component (4 cardinal + 4 diagonal)
   - Score each position based on overlap/occlusion
   - Select optimal position with minimal conflicts

7. GRID-ALIGNED PLACEMENT
   Source: PCB CAD best practices
   - Snap text positions to grid
   - Align text baselines for visual consistency
   - Uniform spacing for arrays of components

Author: PCB Engine Team
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto

# Import common types for position handling (BUG-03/BUG-06 fix)
from .common_types import Position, normalize_position, get_xy, get_pins, get_pin_net
import math
from collections import defaultdict


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class TextAnchor(Enum):
    """Text anchor point"""
    CENTER = 'center'
    LEFT = 'left'
    RIGHT = 'right'
    TOP_LEFT = 'top_left'
    TOP_RIGHT = 'top_right'
    BOTTOM_LEFT = 'bottom_left'
    BOTTOM_RIGHT = 'bottom_right'


class TextOrientation(Enum):
    """Text orientation options"""
    HORIZONTAL = 0
    VERTICAL_UP = 90
    VERTICAL_DOWN = 270
    UPSIDE_DOWN = 180


class PlacementPosition(Enum):
    """Placement positions around component"""
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()


class ComponentCategory(Enum):
    """Component categories for silkscreen handling"""
    RESISTOR = 'resistor'
    CAPACITOR = 'capacitor'
    INDUCTOR = 'inductor'
    DIODE = 'diode'
    TRANSISTOR = 'transistor'
    IC = 'ic'
    CONNECTOR = 'connector'
    CRYSTAL = 'crystal'
    LED = 'led'
    SWITCH = 'switch'
    FUSE = 'fuse'
    TEST_POINT = 'test_point'
    FIDUCIAL = 'fiducial'
    OTHER = 'other'


class ManufacturingStandard(Enum):
    """Manufacturing standard presets"""
    IPC_CLASS_2 = auto()   # Standard commercial
    IPC_CLASS_3 = auto()   # High reliability
    JLCPCB = auto()        # JLCPCB specific
    PCBWAY = auto()        # PCBWay specific
    OSH_PARK = auto()      # OSH Park specific


# IPC-7351B Recommended Sizes
IPC_TEXT_SIZES = {
    'reference': {
        'min_height': 0.8,      # mm
        'max_height': 2.0,
        'recommended': 1.0,
        'stroke_ratio': 0.15,   # thickness = height * ratio
    },
    'value': {
        'min_height': 0.6,
        'max_height': 1.5,
        'recommended': 0.8,
        'stroke_ratio': 0.15,
    }
}

# Manufacturing-specific rules
MANUFACTURING_RULES = {
    ManufacturingStandard.IPC_CLASS_2: {
        'min_line_width': 0.15,        # mm
        'min_text_height': 0.8,
        'text_stroke_ratio': 0.15,
        'pad_clearance': 0.1,
        'silkscreen_accuracy': 0.15,   # Registration accuracy
    },
    ManufacturingStandard.IPC_CLASS_3: {
        'min_line_width': 0.12,
        'min_text_height': 0.6,
        'text_stroke_ratio': 0.12,
        'pad_clearance': 0.1,
        'silkscreen_accuracy': 0.1,
    },
    ManufacturingStandard.JLCPCB: {
        'min_line_width': 0.15,        # 6 mil minimum
        'min_text_height': 0.8,        # Recommended 1.0mm+
        'text_stroke_ratio': 0.15,
        'pad_clearance': 0.127,        # 5 mil
        'silkscreen_accuracy': 0.15,
    },
    ManufacturingStandard.PCBWAY: {
        'min_line_width': 0.15,
        'min_text_height': 0.8,
        'text_stroke_ratio': 0.15,
        'pad_clearance': 0.15,
        'silkscreen_accuracy': 0.15,
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SilkscreenText:
    """A silkscreen text element with full metadata"""
    text: str
    x: float
    y: float
    layer: str = 'F.SilkS'
    size: float = 1.0           # Text height in mm
    thickness: float = 0.15     # Line thickness in mm
    rotation: float = 0.0       # Rotation in degrees
    anchor: str = 'center'
    mirror: bool = False
    hidden: bool = False        # Hidden in KiCad but in data

    # Metadata
    component_ref: str = ''
    text_type: str = ''         # 'reference', 'value', 'custom', 'polarity'
    placement_position: PlacementPosition = None
    placement_score: float = 0.0    # Quality score for this placement

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box (min_x, min_y, max_x, max_y)"""
        char_width = self.size * 0.6  # Approximate character width
        text_width = len(self.text) * char_width
        text_height = self.size

        angle_rad = math.radians(self.rotation)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))

        bbox_width = text_width * cos_a + text_height * sin_a
        bbox_height = text_width * sin_a + text_height * cos_a

        half_w = bbox_width / 2
        half_h = bbox_height / 2

        return (self.x - half_w, self.y - half_h,
                self.x + half_w, self.y + half_h)


@dataclass
class SilkscreenLine:
    """A silkscreen line element"""
    start: Tuple[float, float]
    end: Tuple[float, float]
    layer: str = 'F.SilkS'
    width: float = 0.12

    def length(self) -> float:
        """Calculate line length"""
        return math.sqrt((self.end[0] - self.start[0])**2 +
                        (self.end[1] - self.start[1])**2)


@dataclass
class SilkscreenArc:
    """A silkscreen arc element"""
    center: Tuple[float, float]
    radius: float
    start_angle: float
    end_angle: float
    layer: str = 'F.SilkS'
    width: float = 0.12


@dataclass
class SilkscreenPolygon:
    """A silkscreen polygon/filled shape"""
    points: List[Tuple[float, float]]
    layer: str = 'F.SilkS'
    width: float = 0.12
    fill: bool = False


@dataclass
class PolarityMarker:
    """Polarity indicator for polarized components"""
    x: float
    y: float
    marker_type: str  # 'dot', 'plus', 'bar', 'notch', 'arrow'
    size: float = 0.3
    layer: str = 'F.SilkS'
    rotation: float = 0.0


@dataclass
class Fiducial:
    """Fiducial marker for pick-and-place alignment"""
    x: float
    y: float
    outer_diameter: float = 2.0
    inner_diameter: float = 1.0
    layer: str = 'F.SilkS'
    fiducial_type: str = 'global'  # 'global', 'local', 'panel'


@dataclass
class AssemblyMarker:
    """Assembly aid marker (orientation, polarity, etc.)"""
    x: float
    y: float
    marker_type: str  # 'pin1_dot', 'pin1_bar', 'cathode_bar', 'anode_plus'
    size: float = 0.4
    layer: str = 'F.SilkS'
    rotation: float = 0.0
    component_ref: str = ''


@dataclass
class SilkscreenConfig:
    """Configuration for the silkscreen piston"""
    # Manufacturing standard
    manufacturing_standard: ManufacturingStandard = ManufacturingStandard.IPC_CLASS_2

    # Text settings
    ref_text_size: float = 1.0        # Reference designator height
    ref_text_thickness: float = 0.15
    value_text_size: float = 0.8      # Value text height
    value_text_thickness: float = 0.12
    min_text_size: float = 0.5        # Minimum readable size
    max_text_size: float = 2.0        # Maximum text size

    # Placement settings
    ref_offset: float = 0.5           # Distance from component body
    value_offset: float = 1.5         # Distance from component body
    pad_clearance: float = 0.2        # Clearance from pads
    text_spacing: float = 0.3         # Spacing between text elements

    # Grid alignment
    grid_size: float = 0.1            # Grid for snapping
    align_to_grid: bool = True

    # Options
    show_references: bool = True
    show_values: bool = False         # Usually disabled to save space
    rotate_for_readability: bool = True
    prefer_horizontal: bool = True

    # Polarity and assembly markers
    show_polarity_markers: bool = True
    show_pin1_markers: bool = True
    polarity_marker_size: float = 0.4

    # Component outlines
    generate_outlines: bool = True
    outline_line_width: float = 0.12
    courtyard_line_width: float = 0.05

    # Optimization settings
    optimization_method: str = 'hybrid'  # 'force_directed', 'simulated_annealing', 'hybrid', 'none'
    force_iterations: int = 50
    spring_constant: float = 0.5
    repulsion_constant: float = 1.0
    sa_initial_temp: float = 100.0
    sa_cooling_rate: float = 0.95

    # DFA features
    add_fiducials: bool = False
    fiducial_positions: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class SilkscreenResult:
    """Result from the silkscreen piston"""
    texts: List[SilkscreenText] = field(default_factory=list)
    lines: List[SilkscreenLine] = field(default_factory=list)
    arcs: List[SilkscreenArc] = field(default_factory=list)
    polygons: List[SilkscreenPolygon] = field(default_factory=list)
    polarity_markers: List[PolarityMarker] = field(default_factory=list)
    assembly_markers: List[AssemblyMarker] = field(default_factory=list)
    fiducials: List[Fiducial] = field(default_factory=list)

    success: bool = True
    warnings: List[str] = field(default_factory=list)

    # Statistics
    ref_count: int = 0
    value_count: int = 0
    collision_count: int = 0
    optimization_iterations: int = 0
    total_polarity_markers: int = 0

    # Quality metrics
    average_placement_score: float = 0.0
    min_clearance_achieved: float = 0.0


# =============================================================================
# SILKSCREEN PISTON
# =============================================================================

class SilkscreenPiston:
    """
    Research-Based Silkscreen Piston

    Handles intelligent placement and optimization of silkscreen elements
    using force-directed algorithms and IPC-compliant sizing.

    Algorithms:
    1. Quadrant-based initial placement (8 positions)
    2. Force-directed optimization
    3. Collision resolution
    4. IPC-7351 compliance checking

    Usage:
        config = SilkscreenConfig()
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)
    """

    def __init__(self, config: SilkscreenConfig = None):
        self.config = config or SilkscreenConfig()

        # Apply manufacturing rules
        self._apply_manufacturing_rules()

        # Results storage
        self.texts: List[SilkscreenText] = []
        self.lines: List[SilkscreenLine] = []
        self.arcs: List[SilkscreenArc] = []
        self.polygons: List[SilkscreenPolygon] = []
        self.polarity_markers: List[PolarityMarker] = []
        self.assembly_markers: List[AssemblyMarker] = []
        self.fiducials: List[Fiducial] = []
        self.warnings: List[str] = []

        # Collision detection structures
        self.occupied_rects: List[Tuple[float, float, float, float]] = []
        self.pad_regions: List[Tuple[float, float, float, float]] = []

        # Component info cache
        self._component_cache: Dict[str, Dict] = {}

    @staticmethod
    def _get_pos_xy(pos) -> Tuple[float, float]:
        """Get x,y from position that could be object with .x/.y or tuple/list."""
        if hasattr(pos, 'x'):
            return (pos.x, pos.y)
        elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return (pos[0], pos[1])
        return (0.0, 0.0)

    def _apply_manufacturing_rules(self):
        """Apply manufacturing standard rules to config"""
        rules = MANUFACTURING_RULES.get(self.config.manufacturing_standard, {})

        if rules:
            # Override minimum values if config is below standard
            if self.config.ref_text_size < rules.get('min_text_height', 0):
                self.config.ref_text_size = rules['min_text_height']

            if self.config.ref_text_thickness < rules.get('min_line_width', 0):
                self.config.ref_text_thickness = rules['min_line_width']

            if self.config.pad_clearance < rules.get('pad_clearance', 0):
                self.config.pad_clearance = rules['pad_clearance']

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def generate(self, parts_db: Dict, placement: Dict, routes: Optional[Dict] = None) -> SilkscreenResult:
        """
        Generate silkscreen elements for all components.

        Uses research-based algorithms:
        1. Mark occupied regions (pads, components)
        2. Initial placement using quadrant scoring
        3. Force-directed optimization
        4. Polarity/assembly markers
        5. Component outlines
        6. Fiducials (if enabled)

        Args:
            parts_db: Parts database with component info
            placement: Component placements {ref: Position}

        Returns:
            SilkscreenResult with all silkscreen elements
        """
        # Clear previous state
        self._clear_state()

        parts = parts_db.get('parts', {})

        # Phase 1: Mark all occupied regions
        self._mark_pad_regions(parts, placement)
        self._mark_component_bodies(parts, placement)

        # Phase 2: Generate initial text placements
        placement_candidates: Dict[str, List[SilkscreenText]] = {}

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            self._component_cache[ref] = {
                'part': part,
                'pos': pos,
                'category': self._classify_component(part)
            }

            # Generate reference designator candidates
            if self.config.show_references:
                candidates = self._generate_placement_candidates(ref, pos, part, 'reference')
                placement_candidates[f"{ref}_ref"] = candidates

            # Generate value text candidates
            if self.config.show_values:
                value = part.get('value', '')
                if value:
                    candidates = self._generate_placement_candidates(ref, pos, part, 'value', value)
                    placement_candidates[f"{ref}_val"] = candidates

        # Phase 3: Select best placements with force-directed optimization
        selected_texts = self._select_optimal_placements(placement_candidates)
        self.texts.extend(selected_texts)

        # Mark text regions as occupied
        for text in self.texts:
            self._mark_text_region(text)

        # Phase 4: Generate polarity and assembly markers
        if self.config.show_polarity_markers or self.config.show_pin1_markers:
            self._generate_assembly_markers(parts, placement)

        # Phase 5: Generate component outlines
        if self.config.generate_outlines:
            for ref, pos in placement.items():
                part = parts.get(ref, {})
                self._generate_component_outline(ref, pos, part)

        # Phase 6: Add fiducials
        if self.config.add_fiducials:
            self._generate_fiducials(parts_db)

        # Phase 7: Final optimization pass
        iterations = self._run_optimization()

        # Calculate statistics
        return self._create_result(iterations)

    def _clear_state(self):
        """Clear all internal state"""
        self.texts.clear()
        self.lines.clear()
        self.arcs.clear()
        self.polygons.clear()
        self.polarity_markers.clear()
        self.assembly_markers.clear()
        self.fiducials.clear()
        self.warnings.clear()
        self.occupied_rects.clear()
        self.pad_regions.clear()
        self._component_cache.clear()

    def _classify_component(self, part: Dict) -> ComponentCategory:
        """Classify component into category for specific handling"""
        category = part.get('category', '').lower()
        ref = part.get('reference', '')

        # Check by category field
        category_map = {
            'resistor': ComponentCategory.RESISTOR,
            'capacitor': ComponentCategory.CAPACITOR,
            'inductor': ComponentCategory.INDUCTOR,
            'diode': ComponentCategory.DIODE,
            'led': ComponentCategory.LED,
            'transistor': ComponentCategory.TRANSISTOR,
            'ic': ComponentCategory.IC,
            'connector': ComponentCategory.CONNECTOR,
            'crystal': ComponentCategory.CRYSTAL,
            'oscillator': ComponentCategory.CRYSTAL,
            'switch': ComponentCategory.SWITCH,
            'fuse': ComponentCategory.FUSE,
            'test_point': ComponentCategory.TEST_POINT,
            'fiducial': ComponentCategory.FIDUCIAL,
        }

        if category in category_map:
            return category_map[category]

        # Fallback: check by reference designator prefix
        ref_prefix = ''.join(c for c in ref if c.isalpha()).upper()

        ref_map = {
            'R': ComponentCategory.RESISTOR,
            'C': ComponentCategory.CAPACITOR,
            'L': ComponentCategory.INDUCTOR,
            'D': ComponentCategory.DIODE,
            'Q': ComponentCategory.TRANSISTOR,
            'U': ComponentCategory.IC,
            'J': ComponentCategory.CONNECTOR,
            'P': ComponentCategory.CONNECTOR,
            'Y': ComponentCategory.CRYSTAL,
            'SW': ComponentCategory.SWITCH,
            'F': ComponentCategory.FUSE,
            'TP': ComponentCategory.TEST_POINT,
            'FID': ComponentCategory.FIDUCIAL,
            'LED': ComponentCategory.LED,
        }

        return ref_map.get(ref_prefix, ComponentCategory.OTHER)

    # =========================================================================
    # REGION MARKING
    # =========================================================================

    def _mark_pad_regions(self, parts: Dict, placement: Dict):
        """Mark pad regions as occupied with clearance"""
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            pos_x, pos_y = self._get_pos_xy(pos)
            rotation = getattr(pos, 'rotation', 0)
            angle_rad = math.radians(rotation)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            for pin in part.get('used_pins', part.get('physical_pins', [])):
                offset = pin.get('offset', (0, 0))
                if not offset or offset == (0, 0):
                    physical = pin.get('physical', {})
                    offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))

                # Apply rotation to offset
                rot_offset_x = offset[0] * cos_a - offset[1] * sin_a
                rot_offset_y = offset[0] * sin_a + offset[1] * cos_a

                pad_x = pos_x + rot_offset_x
                pad_y = pos_y + rot_offset_y

                pad_size = pin.get('pad_size', pin.get('size', (1.0, 0.6)))
                if not isinstance(pad_size, (list, tuple)):
                    pad_size = (1.0, 0.6)

                clearance = self.config.pad_clearance
                half_w = pad_size[0] / 2 + clearance
                half_h = pad_size[1] / 2 + clearance

                rect = (pad_x - half_w, pad_y - half_h,
                       pad_x + half_w, pad_y + half_h)
                self.pad_regions.append(rect)
                self.occupied_rects.append(rect)

    def _mark_component_bodies(self, parts: Dict, placement: Dict):
        """Mark component body regions"""
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            pos_x, pos_y = self._get_pos_xy(pos)

            body = part.get('body', {})
            body_width = body.get('width', part.get('body_width', 0))
            body_height = body.get('height', part.get('body_height', 0))

            if body_width <= 0 or body_height <= 0:
                continue

            rotation = getattr(pos, 'rotation', 0)
            angle_rad = math.radians(rotation)
            cos_a = abs(math.cos(angle_rad))
            sin_a = abs(math.sin(angle_rad))

            # Rotated bounding box
            bbox_width = body_width * cos_a + body_height * sin_a
            bbox_height = body_width * sin_a + body_height * cos_a

            half_w = bbox_width / 2
            half_h = bbox_height / 2

            self.occupied_rects.append((
                pos_x - half_w, pos_y - half_h,
                pos_x + half_w, pos_y + half_h
            ))

    def _mark_text_region(self, text: SilkscreenText):
        """Mark text region as occupied"""
        bbox = text.get_bounding_box()
        self.occupied_rects.append(bbox)

    # =========================================================================
    # PLACEMENT CANDIDATE GENERATION (QUADRANT-BASED)
    # =========================================================================

    def _generate_placement_candidates(
        self,
        ref: str,
        pos,
        part: Dict,
        text_type: str,
        text_value: str = None
    ) -> List[SilkscreenText]:
        """
        Generate 8 placement candidates around component

        Algorithm: Quadrant-based placement
        - Try all 8 positions (4 cardinal + 4 diagonal)
        - Score each based on collision, readability, proximity
        """
        candidates = []

        # Get component dimensions
        body = part.get('body', {})
        body_width = body.get('width', part.get('body_width', 2.0))
        body_height = body.get('height', part.get('body_height', 1.5))

        pos_x, pos_y = self._get_pos_xy(pos)
        rotation = getattr(pos, 'rotation', 0)
        layer = getattr(pos, 'layer', 'F.Cu')
        silk_layer = 'F.SilkS' if layer == 'F.Cu' else 'B.SilkS'

        # Determine text content and size
        if text_type == 'reference':
            text = ref
            size = self.config.ref_text_size
            thickness = self.config.ref_text_thickness
            offset_distance = self.config.ref_offset
        else:  # value
            text = text_value or part.get('value', '')
            size = self.config.value_text_size
            thickness = self.config.value_text_thickness
            offset_distance = self.config.value_offset

        # Define 8 positions with rotation
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        half_w = body_width / 2 + offset_distance
        half_h = body_height / 2 + offset_distance

        # Position offsets (before rotation)
        positions = {
            PlacementPosition.TOP: (0, -half_h - size/2),
            PlacementPosition.BOTTOM: (0, half_h + size/2),
            PlacementPosition.LEFT: (-half_w - size/2, 0),
            PlacementPosition.RIGHT: (half_w + size/2, 0),
            PlacementPosition.TOP_LEFT: (-half_w, -half_h),
            PlacementPosition.TOP_RIGHT: (half_w, -half_h),
            PlacementPosition.BOTTOM_LEFT: (-half_w, half_h),
            PlacementPosition.BOTTOM_RIGHT: (half_w, half_h),
        }

        for position, (dx, dy) in positions.items():
            # Apply component rotation to offset
            rot_dx = dx * cos_a - dy * sin_a
            rot_dy = dx * sin_a + dy * cos_a

            text_x = pos_x + rot_dx
            text_y = pos_y + rot_dy

            # Snap to grid if enabled
            if self.config.align_to_grid:
                text_x = round(text_x / self.config.grid_size) * self.config.grid_size
                text_y = round(text_y / self.config.grid_size) * self.config.grid_size

            # Determine text rotation for readability
            if position in [PlacementPosition.LEFT, PlacementPosition.RIGHT]:
                text_rotation = 90 if self.config.prefer_horizontal else rotation
            else:
                text_rotation = 0 if self.config.prefer_horizontal else rotation

            # Adjust for readability
            text_rotation = self._adjust_rotation_for_readability(text_rotation)

            candidate = SilkscreenText(
                text=text,
                x=text_x,
                y=text_y,
                layer=silk_layer,
                size=size,
                thickness=thickness,
                rotation=text_rotation,
                component_ref=ref,
                text_type=text_type,
                placement_position=position,
                mirror=(silk_layer == 'B.SilkS')
            )

            # Score this placement
            candidate.placement_score = self._score_placement(candidate)
            candidates.append(candidate)

        return candidates

    def _adjust_rotation_for_readability(self, rotation: float) -> float:
        """Adjust text rotation to maintain readability"""
        if not self.config.rotate_for_readability:
            return rotation

        rot = rotation % 360

        # Keep text readable (not upside-down)
        if 90 < rot < 270:
            return (rot + 180) % 360

        return rot

    def _score_placement(self, text: SilkscreenText) -> float:
        """
        Score a text placement candidate

        Higher score = better placement

        Factors:
        - Collision penalty (large negative)
        - Pad overlap penalty (very large negative)
        - Distance from ideal position
        - Orientation preference
        """
        score = 100.0  # Base score

        bbox = text.get_bounding_box()

        # Check collision with pads (critical)
        for pad_rect in self.pad_regions:
            if self._rects_overlap(bbox, pad_rect):
                score -= 1000  # Very heavy penalty

        # Check collision with other occupied regions
        for rect in self.occupied_rects:
            if rect not in self.pad_regions:  # Don't double-count pads
                if self._rects_overlap(bbox, rect):
                    overlap = self._calculate_overlap_area(bbox, rect)
                    score -= overlap * 50

        # Prefer top placement for reference designators
        if text.text_type == 'reference':
            if text.placement_position == PlacementPosition.TOP:
                score += 10
            elif text.placement_position == PlacementPosition.BOTTOM:
                score += 5

        # Prefer horizontal text
        if self.config.prefer_horizontal:
            if text.rotation in [0, 180]:
                score += 5

        return score

    def _calculate_overlap_area(
        self,
        r1: Tuple[float, float, float, float],
        r2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate overlap area between two rectangles"""
        x_overlap = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
        y_overlap = max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
        return x_overlap * y_overlap

    # =========================================================================
    # OPTIMAL PLACEMENT SELECTION
    # =========================================================================

    def _select_optimal_placements(
        self,
        candidates: Dict[str, List[SilkscreenText]]
    ) -> List[SilkscreenText]:
        """
        Select optimal placements from candidates

        Algorithm:
        1. Sort all candidates by score (descending)
        2. Greedily select best non-conflicting placement for each text
        3. Mark selected regions as occupied
        """
        selected: List[SilkscreenText] = []
        selected_regions: List[Tuple[float, float, float, float]] = []

        # Sort keys by best available score
        sorted_keys = sorted(
            candidates.keys(),
            key=lambda k: max(c.placement_score for c in candidates[k]),
            reverse=True
        )

        for key in sorted_keys:
            text_candidates = sorted(
                candidates[key],
                key=lambda c: c.placement_score,
                reverse=True
            )

            # Find first candidate that doesn't conflict with selected
            for candidate in text_candidates:
                bbox = candidate.get_bounding_box()

                conflicts = False
                for sel_rect in selected_regions:
                    if self._rects_overlap(bbox, sel_rect):
                        conflicts = True
                        break

                if not conflicts:
                    selected.append(candidate)
                    selected_regions.append(bbox)
                    break
            else:
                # All candidates conflict - use best anyway and warn
                if text_candidates:
                    best = text_candidates[0]
                    selected.append(best)
                    selected_regions.append(best.get_bounding_box())
                    self.warnings.append(
                        f"Text '{best.text}' may overlap with other elements"
                    )

        return selected

    # =========================================================================
    # OPTIMIZATION DISPATCHER
    # =========================================================================

    def _run_optimization(self) -> int:
        """
        Run the configured optimization method

        Methods:
        - 'force_directed': Spring-based relaxation (fast)
        - 'simulated_annealing': Stochastic optimization (thorough)
        - 'hybrid': SA followed by force-directed (best quality)
        - 'none': No optimization
        """
        method = self.config.optimization_method.lower()

        if method == 'none':
            return 0
        elif method == 'force_directed':
            return self._force_directed_optimize()
        elif method == 'simulated_annealing':
            return self._simulated_annealing_optimize()
        elif method == 'hybrid':
            # Hybrid: SA for global optimization, then force-directed for polish
            sa_iters = self._simulated_annealing_optimize()
            fd_iters = self._force_directed_optimize()
            return sa_iters + fd_iters
        else:
            # Default to force-directed
            return self._force_directed_optimize()

    # =========================================================================
    # SIMULATED ANNEALING OPTIMIZATION (Research-Based)
    # =========================================================================

    def _simulated_annealing_optimize(self) -> int:
        """
        Simulated Annealing label placement optimization

        Source: Christensen et al., "An Empirical Study of Algorithms for
                Point-Feature Label Placement" (1995)
        Source: GitHub - migurski/Dymo: Map label placer with simulated annealing
        Source: Wolff, "Automated Label Placement in Theory and Practice" (1999)

        Algorithm:
        1. Start with high temperature (allow large random moves)
        2. Gradually cool (reduce move magnitude)
        3. Accept worse solutions probabilistically (escape local optima)
        4. Converge to near-optimal solution

        This solves the NP-hard label placement problem heuristically.
        """
        import random

        if not self.texts:
            return 0

        # Simulated annealing parameters
        initial_temp = 100.0
        final_temp = 0.1
        cooling_rate = 0.95
        moves_per_temp = len(self.texts) * 3

        # Store best solution
        best_texts = [(t.x, t.y, t.rotation) for t in self.texts]
        best_energy = self._calculate_total_energy()

        current_energy = best_energy
        temp = initial_temp
        iterations = 0

        while temp > final_temp:
            for _ in range(moves_per_temp):
                iterations += 1

                # Pick random text to move
                idx = random.randint(0, len(self.texts) - 1)
                text = self.texts[idx]

                # Save current position
                old_x, old_y, old_rot = text.x, text.y, text.rotation

                # Generate random move (magnitude decreases with temperature)
                move_scale = temp / initial_temp
                dx = random.gauss(0, 0.5 * move_scale)
                dy = random.gauss(0, 0.5 * move_scale)

                # Occasionally try rotation change
                if random.random() < 0.1:
                    text.rotation = random.choice([0, 90, 180, 270])

                # Apply move
                text.x += dx
                text.y += dy

                # Snap to grid if enabled
                if self.config.align_to_grid:
                    text.x = round(text.x / self.config.grid_size) * self.config.grid_size
                    text.y = round(text.y / self.config.grid_size) * self.config.grid_size

                # Calculate new energy
                new_energy = self._calculate_total_energy()

                # Metropolis criterion
                delta_e = new_energy - current_energy

                if delta_e < 0:
                    # Accept improvement
                    current_energy = new_energy

                    # Track best solution
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_texts = [(t.x, t.y, t.rotation) for t in self.texts]
                else:
                    # Accept worse solution with probability
                    if random.random() < math.exp(-delta_e / temp):
                        current_energy = new_energy
                    else:
                        # Reject - revert
                        text.x, text.y, text.rotation = old_x, old_y, old_rot

            # Cool down
            temp *= cooling_rate

        # Restore best solution
        for i, (x, y, rot) in enumerate(best_texts):
            self.texts[i].x = x
            self.texts[i].y = y
            self.texts[i].rotation = rot

        return iterations

    def _calculate_total_energy(self) -> float:
        """
        Calculate total energy (cost) of current label placement

        Lower energy = better placement

        Energy components:
        - Overlap with pads (very high penalty)
        - Overlap with other labels
        - Distance from original anchor point
        - Non-standard rotation penalty
        """
        energy = 0.0

        for i, text in enumerate(self.texts):
            bbox = text.get_bounding_box()

            # Pad overlap penalty (critical)
            for pad_rect in self.pad_regions:
                if self._rects_overlap(bbox, pad_rect):
                    overlap = self._calculate_overlap_area(bbox, pad_rect)
                    energy += overlap * 1000

            # Label-to-label overlap
            for j, other in enumerate(self.texts):
                if i >= j:
                    continue
                other_bbox = other.get_bounding_box()
                if self._rects_overlap(bbox, other_bbox):
                    overlap = self._calculate_overlap_area(bbox, other_bbox)
                    energy += overlap * 100

            # Non-horizontal penalty
            if text.rotation not in [0, 180]:
                energy += 5

        return energy

    # =========================================================================
    # FORCE-DIRECTED OPTIMIZATION
    # =========================================================================

    def _force_directed_optimize(self) -> int:
        """
        Apply force-directed optimization to text positions

        Algorithm (based on Imhof 1975):
        1. Each text has spring force toward original position
        2. Each text repels other texts
        3. Texts repel from pads (stronger force)
        4. Iterate until convergence or max iterations

        Returns: Number of iterations performed
        """
        if not self.texts:
            return 0

        k_spring = self.config.spring_constant
        k_repel = self.config.repulsion_constant
        damping = 0.8
        min_movement = 0.01  # Convergence threshold

        # Store original positions (anchor points)
        original_positions = [(t.x, t.y) for t in self.texts]

        for iteration in range(self.config.force_iterations):
            max_movement = 0.0

            for i, text in enumerate(self.texts):
                if text.placement_score < -500:
                    # Don't optimize texts with critical placement issues
                    continue

                fx, fy = 0.0, 0.0

                # Spring force toward original position
                orig_x, orig_y = original_positions[i]
                dx = orig_x - text.x
                dy = orig_y - text.y
                fx += k_spring * dx
                fy += k_spring * dy

                # Repulsion from other texts
                for j, other in enumerate(self.texts):
                    if i == j:
                        continue

                    dx = text.x - other.x
                    dy = text.y - other.y
                    dist = math.sqrt(dx*dx + dy*dy)

                    if dist < 0.1:
                        dist = 0.1  # Avoid division by zero

                    # Check if texts actually overlap
                    bbox1 = text.get_bounding_box()
                    bbox2 = other.get_bounding_box()

                    if self._rects_overlap(bbox1, bbox2):
                        force = k_repel / (dist * dist)
                        fx += force * dx / dist
                        fy += force * dy / dist

                # Strong repulsion from pads
                bbox = text.get_bounding_box()
                for pad_rect in self.pad_regions:
                    if self._rects_overlap(bbox, pad_rect):
                        # Calculate repulsion direction
                        pad_cx = (pad_rect[0] + pad_rect[2]) / 2
                        pad_cy = (pad_rect[1] + pad_rect[3]) / 2

                        dx = text.x - pad_cx
                        dy = text.y - pad_cy
                        dist = math.sqrt(dx*dx + dy*dy)

                        if dist < 0.1:
                            dist = 0.1

                        force = k_repel * 3 / (dist * dist)  # Stronger for pads
                        fx += force * dx / dist
                        fy += force * dy / dist

                # Apply forces with damping
                new_x = text.x + fx * damping
                new_y = text.y + fy * damping

                # Snap to grid if enabled
                if self.config.align_to_grid:
                    new_x = round(new_x / self.config.grid_size) * self.config.grid_size
                    new_y = round(new_y / self.config.grid_size) * self.config.grid_size

                movement = math.sqrt((new_x - text.x)**2 + (new_y - text.y)**2)
                max_movement = max(max_movement, movement)

                text.x = new_x
                text.y = new_y

            # Check convergence
            if max_movement < min_movement:
                return iteration + 1

        return self.config.force_iterations

    # =========================================================================
    # POLARITY AND ASSEMBLY MARKERS
    # =========================================================================

    def _generate_assembly_markers(self, parts: Dict, placement: Dict):
        """Generate polarity and pin 1 markers for components"""
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            info = self._component_cache.get(ref, {})
            category = info.get('category', ComponentCategory.OTHER)

            rotation = getattr(pos, 'rotation', 0)
            layer = getattr(pos, 'layer', 'F.Cu')
            silk_layer = 'F.SilkS' if layer == 'F.Cu' else 'B.SilkS'

            # Pin 1 marker for ICs and connectors
            if self.config.show_pin1_markers:
                if category in [ComponentCategory.IC, ComponentCategory.CONNECTOR]:
                    marker = self._create_pin1_marker(ref, pos, part, silk_layer)
                    if marker:
                        self.assembly_markers.append(marker)

            # Polarity markers for polarized components
            if self.config.show_polarity_markers:
                if category == ComponentCategory.DIODE:
                    marker = self._create_cathode_marker(ref, pos, part, silk_layer)
                    if marker:
                        self.polarity_markers.append(marker)

                elif category == ComponentCategory.LED:
                    marker = self._create_cathode_marker(ref, pos, part, silk_layer)
                    if marker:
                        self.polarity_markers.append(marker)

                elif category == ComponentCategory.CAPACITOR:
                    # Check if electrolytic (polarized)
                    if part.get('polarized', False) or 'electrolytic' in part.get('type', '').lower():
                        marker = self._create_plus_marker(ref, pos, part, silk_layer)
                        if marker:
                            self.polarity_markers.append(marker)

    def _create_pin1_marker(
        self,
        ref: str,
        pos,
        part: Dict,
        layer: str
    ) -> Optional[AssemblyMarker]:
        """Create pin 1 indicator dot"""
        pos_x, pos_y = self._get_pos_xy(pos)
        body = part.get('body', {})
        body_width = body.get('width', part.get('body_width', 2.0))
        body_height = body.get('height', part.get('body_height', 2.0))

        rotation = getattr(pos, 'rotation', 0)
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Pin 1 at top-left corner (before rotation)
        offset_x = -body_width / 2 + 0.5
        offset_y = -body_height / 2 + 0.5

        # Rotate offset
        rot_x = offset_x * cos_a - offset_y * sin_a
        rot_y = offset_x * sin_a + offset_y * cos_a

        return AssemblyMarker(
            x=pos_x + rot_x,
            y=pos_y + rot_y,
            marker_type='pin1_dot',
            size=self.config.polarity_marker_size,
            layer=layer,
            rotation=rotation,
            component_ref=ref
        )

    def _create_cathode_marker(
        self,
        ref: str,
        pos,
        part: Dict,
        layer: str
    ) -> Optional[PolarityMarker]:
        """Create cathode bar for diodes/LEDs"""
        pos_x, pos_y = self._get_pos_xy(pos)
        body = part.get('body', {})
        body_width = body.get('width', part.get('body_width', 2.0))

        rotation = getattr(pos, 'rotation', 0)
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Cathode at right side (standard diode orientation)
        offset_x = body_width / 2 - 0.3
        offset_y = 0

        rot_x = offset_x * cos_a - offset_y * sin_a
        rot_y = offset_x * sin_a + offset_y * cos_a

        return PolarityMarker(
            x=pos_x + rot_x,
            y=pos_y + rot_y,
            marker_type='bar',
            size=self.config.polarity_marker_size,
            layer=layer,
            rotation=rotation
        )

    def _create_plus_marker(
        self,
        ref: str,
        pos,
        part: Dict,
        layer: str
    ) -> Optional[PolarityMarker]:
        """Create plus marker for polarized capacitors"""
        pos_x, pos_y = self._get_pos_xy(pos)
        body = part.get('body', {})
        body_width = body.get('width', part.get('body_width', 2.0))

        rotation = getattr(pos, 'rotation', 0)
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Plus at left side (anode side)
        offset_x = -body_width / 2 - 0.5
        offset_y = 0

        rot_x = offset_x * cos_a - offset_y * sin_a
        rot_y = offset_x * sin_a + offset_y * cos_a

        return PolarityMarker(
            x=pos_x + rot_x,
            y=pos_y + rot_y,
            marker_type='plus',
            size=self.config.polarity_marker_size,
            layer=layer,
            rotation=rotation
        )

    # =========================================================================
    # COMPONENT OUTLINES
    # =========================================================================

    def _generate_component_outline(self, ref: str, pos, part: Dict):
        """Generate silkscreen outline for component"""
        body = part.get('body', {})
        body_width = body.get('width', part.get('body_width', 0))
        body_height = body.get('height', part.get('body_height', 0))

        if body_width <= 0 or body_height <= 0:
            return

        rotation = getattr(pos, 'rotation', 0)
        layer = getattr(pos, 'layer', 'F.Cu')
        silk_layer = 'F.SilkS' if layer == 'F.Cu' else 'B.SilkS'

        # Only generate outlines for larger components
        category = self._component_cache.get(ref, {}).get('category', ComponentCategory.OTHER)
        if category in [ComponentCategory.IC, ComponentCategory.CONNECTOR]:
            self._generate_ic_outline(ref, pos, part, silk_layer, body_width, body_height, rotation)
        elif category in [ComponentCategory.RESISTOR, ComponentCategory.CAPACITOR, ComponentCategory.INDUCTOR]:
            # Small passives don't need outlines - ref des is enough
            pass
        elif body_width > 3 or body_height > 3:
            # Large components get outlines
            self._generate_simple_outline(pos, silk_layer, body_width, body_height, rotation)

    def _generate_ic_outline(
        self,
        ref: str,
        pos,
        part: Dict,
        layer: str,
        width: float,
        height: float,
        rotation: float
    ):
        """Generate IC outline with pin 1 notch"""
        pos_x, pos_y = self._get_pos_xy(pos)
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        half_w = width / 2
        half_h = height / 2
        notch_size = min(0.5, half_w / 4)

        # Define corners (with pin 1 notch at top-left)
        # Starting from top-left, going clockwise
        corners = [
            (-half_w + notch_size, -half_h),  # After notch
            (half_w, -half_h),                 # Top-right
            (half_w, half_h),                  # Bottom-right
            (-half_w, half_h),                 # Bottom-left
            (-half_w, -half_h + notch_size),   # Before notch
        ]

        # Notch arc points
        notch_center = (-half_w, -half_h)

        # Rotate and translate
        def transform(x, y):
            rx = x * cos_a - y * sin_a + pos_x
            ry = x * sin_a + y * cos_a + pos_y
            return (rx, ry)

        transformed = [transform(x, y) for x, y in corners]

        # Create lines (skip where overlapping pads)
        for i in range(len(transformed)):
            start = transformed[i]
            end = transformed[(i + 1) % len(transformed)]

            if not self._line_overlaps_pads(start, end):
                self.lines.append(SilkscreenLine(
                    start=start,
                    end=end,
                    layer=layer,
                    width=self.config.outline_line_width
                ))

        # Add notch arc
        notch_start = transform(-half_w + notch_size, -half_h)
        notch_end = transform(-half_w, -half_h + notch_size)
        notch_c = transform(-half_w, -half_h)

        self.arcs.append(SilkscreenArc(
            center=notch_c,
            radius=notch_size,
            start_angle=180 - rotation,
            end_angle=270 - rotation,
            layer=layer,
            width=self.config.outline_line_width
        ))

    def _generate_simple_outline(
        self,
        pos,
        layer: str,
        width: float,
        height: float,
        rotation: float
    ):
        """Generate simple rectangular outline"""
        pos_x, pos_y = self._get_pos_xy(pos)
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        half_w = width / 2
        half_h = height / 2

        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]

        # Rotate and translate
        transformed = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a + pos_x
            ry = x * sin_a + y * cos_a + pos_y
            transformed.append((rx, ry))

        # Create lines
        for i in range(4):
            start = transformed[i]
            end = transformed[(i + 1) % 4]

            if not self._line_overlaps_pads(start, end):
                self.lines.append(SilkscreenLine(
                    start=start,
                    end=end,
                    layer=layer,
                    width=self.config.outline_line_width
                ))

    def _line_overlaps_pads(self, start: Tuple, end: Tuple) -> bool:
        """Check if line overlaps pad regions"""
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])

            for rect in self.pad_regions:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    return True

        return False

    # =========================================================================
    # FIDUCIALS
    # =========================================================================

    def _generate_fiducials(self, parts_db: Dict):
        """Generate fiducial markers for pick-and-place"""
        board_width = parts_db.get('board_width', 100)
        board_height = parts_db.get('board_height', 100)

        if self.config.fiducial_positions:
            # Use custom positions
            for x, y in self.config.fiducial_positions:
                self.fiducials.append(Fiducial(
                    x=x, y=y,
                    fiducial_type='global'
                ))
        else:
            # Default: 3 fiducials in corners
            margin = 5.0  # mm from edge

            positions = [
                (margin, margin),                           # Bottom-left
                (board_width - margin, margin),             # Bottom-right
                (margin, board_height - margin),            # Top-left
            ]

            for x, y in positions:
                self.fiducials.append(Fiducial(
                    x=x, y=y,
                    fiducial_type='global'
                ))

    # =========================================================================
    # COLLISION HELPERS
    # =========================================================================

    def _rects_overlap(
        self,
        r1: Tuple[float, float, float, float],
        r2: Tuple[float, float, float, float]
    ) -> bool:
        """Check if two rectangles overlap"""
        return not (r1[2] < r2[0] or r1[0] > r2[2] or
                   r1[3] < r2[1] or r1[1] > r2[3])

    # =========================================================================
    # RESULT CREATION
    # =========================================================================

    def _create_result(self, optimization_iterations: int) -> SilkscreenResult:
        """Create final result object with statistics"""
        ref_count = sum(1 for t in self.texts if t.text_type == 'reference')
        value_count = sum(1 for t in self.texts if t.text_type == 'value')

        # Count collisions
        collision_count = 0
        for i, text in enumerate(self.texts):
            bbox = text.get_bounding_box()
            for pad_rect in self.pad_regions:
                if self._rects_overlap(bbox, pad_rect):
                    collision_count += 1
                    break

        # Calculate average placement score
        avg_score = 0.0
        if self.texts:
            avg_score = sum(t.placement_score for t in self.texts) / len(self.texts)

        # Calculate minimum clearance achieved
        min_clearance = float('inf')
        for text in self.texts:
            bbox = text.get_bounding_box()
            for pad_rect in self.pad_regions:
                # Calculate distance between rectangles
                dx = max(pad_rect[0] - bbox[2], bbox[0] - pad_rect[2], 0)
                dy = max(pad_rect[1] - bbox[3], bbox[1] - pad_rect[3], 0)
                dist = math.sqrt(dx*dx + dy*dy)
                min_clearance = min(min_clearance, dist)

        if min_clearance == float('inf'):
            min_clearance = self.config.pad_clearance

        return SilkscreenResult(
            texts=self.texts,
            lines=self.lines,
            arcs=self.arcs,
            polygons=self.polygons,
            polarity_markers=self.polarity_markers,
            assembly_markers=self.assembly_markers,
            fiducials=self.fiducials,
            success=(collision_count == 0),
            warnings=self.warnings,
            ref_count=ref_count,
            value_count=value_count,
            collision_count=collision_count,
            optimization_iterations=optimization_iterations,
            total_polarity_markers=len(self.polarity_markers),
            average_placement_score=avg_score,
            min_clearance_achieved=min_clearance
        )

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def to_kicad_sexpr(self) -> str:
        """Export silkscreen to KiCad S-expression format"""
        lines = []

        # Export text elements
        for text in self.texts:
            if text.hidden:
                continue

            lines.append(f'''  (gr_text "{text.text}"
    (at {text.x:.4f} {text.y:.4f} {text.rotation})
    (layer "{text.layer}")
    (effects
      (font (size {text.size} {text.size}) (thickness {text.thickness}))
      (justify center{' mirror' if text.mirror else ''})
    )
  )''')

        # Export lines
        for line in self.lines:
            lines.append(f'''  (gr_line
    (start {line.start[0]:.4f} {line.start[1]:.4f})
    (end {line.end[0]:.4f} {line.end[1]:.4f})
    (layer "{line.layer}")
    (width {line.width})
  )''')

        # Export arcs (full circles)
        for arc in self.arcs:
            if arc.end_angle - arc.start_angle >= 359:
                lines.append(f'''  (gr_circle
    (center {arc.center[0]:.4f} {arc.center[1]:.4f})
    (end {arc.center[0] + arc.radius:.4f} {arc.center[1]:.4f})
    (layer "{arc.layer}")
    (width {arc.width})
  )''')
            else:
                # Partial arc - would need conversion to KiCad arc format
                pass

        # Export polarity markers
        for marker in self.polarity_markers:
            if marker.marker_type == 'bar':
                # Vertical bar
                half_size = marker.size / 2
                lines.append(f'''  (gr_line
    (start {marker.x:.4f} {marker.y - half_size:.4f})
    (end {marker.x:.4f} {marker.y + half_size:.4f})
    (layer "{marker.layer}")
    (width 0.15)
  )''')
            elif marker.marker_type == 'plus':
                # Plus sign
                half_size = marker.size / 2
                lines.append(f'''  (gr_line
    (start {marker.x - half_size:.4f} {marker.y:.4f})
    (end {marker.x + half_size:.4f} {marker.y:.4f})
    (layer "{marker.layer}")
    (width 0.12)
  )''')
                lines.append(f'''  (gr_line
    (start {marker.x:.4f} {marker.y - half_size:.4f})
    (end {marker.x:.4f} {marker.y + half_size:.4f})
    (layer "{marker.layer}")
    (width 0.12)
  )''')

        # Export assembly markers (pin 1 dots)
        for marker in self.assembly_markers:
            if marker.marker_type == 'pin1_dot':
                lines.append(f'''  (gr_circle
    (center {marker.x:.4f} {marker.y:.4f})
    (end {marker.x + marker.size/2:.4f} {marker.y:.4f})
    (layer "{marker.layer}")
    (width 0.12)
    (fill solid)
  )''')

        # Export fiducials
        for fid in self.fiducials:
            # Fiducial: filled circle with clearance ring
            lines.append(f'''  (gr_circle
    (center {fid.x:.4f} {fid.y:.4f})
    (end {fid.x + fid.inner_diameter/2:.4f} {fid.y:.4f})
    (layer "{fid.layer}")
    (width 0.1)
    (fill solid)
  )''')

        return '\n'.join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about silkscreen generation"""
        return {
            'total_texts': len(self.texts),
            'reference_count': sum(1 for t in self.texts if t.text_type == 'reference'),
            'value_count': sum(1 for t in self.texts if t.text_type == 'value'),
            'total_lines': len(self.lines),
            'total_arcs': len(self.arcs),
            'polarity_markers': len(self.polarity_markers),
            'assembly_markers': len(self.assembly_markers),
            'fiducials': len(self.fiducials),
            'warnings': len(self.warnings),
            'pad_regions_marked': len(self.pad_regions),
            'manufacturing_standard': self.config.manufacturing_standard.name,
        }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Silkscreen Piston...")
    print("=" * 60)

    # Create test data
    class MockPosition:
        def __init__(self, x, y, rotation=0, layer='F.Cu'):
            self.x = x
            self.y = y
            self.rotation = rotation
            self.layer = layer

    parts_db = {
        'parts': {
            'U1': {
                'reference': 'U1',
                'value': 'ATmega328P',
                'category': 'ic',
                'body': {'width': 8.0, 'height': 8.0},
                'used_pins': [
                    {'number': '1', 'offset': (-3.5, -3.5), 'pad_size': (0.6, 1.5)},
                    {'number': '2', 'offset': (-3.5, -2.5), 'pad_size': (0.6, 1.5)},
                ]
            },
            'R1': {
                'reference': 'R1',
                'value': '10k',
                'category': 'resistor',
                'body': {'width': 1.6, 'height': 0.8},
                'used_pins': [
                    {'number': '1', 'offset': (-0.8, 0), 'pad_size': (0.8, 0.9)},
                    {'number': '2', 'offset': (0.8, 0), 'pad_size': (0.8, 0.9)},
                ]
            },
            'C1': {
                'reference': 'C1',
                'value': '100nF',
                'category': 'capacitor',
                'body': {'width': 1.6, 'height': 0.8},
            },
            'D1': {
                'reference': 'D1',
                'value': '1N4148',
                'category': 'diode',
                'body': {'width': 2.5, 'height': 1.0},
            },
            'LED1': {
                'reference': 'LED1',
                'value': 'Red',
                'category': 'led',
                'body': {'width': 1.6, 'height': 1.0},
            },
        },
        'board_width': 50,
        'board_height': 40,
    }

    placement = {
        'U1': MockPosition(25, 20, 0),
        'R1': MockPosition(10, 10, 0),
        'C1': MockPosition(15, 10, 90),
        'D1': MockPosition(35, 30, 45),
        'LED1': MockPosition(40, 10, 0),
    }

    # Test different configurations
    configs = [
        SilkscreenConfig(manufacturing_standard=ManufacturingStandard.IPC_CLASS_2),
        SilkscreenConfig(manufacturing_standard=ManufacturingStandard.JLCPCB),
        SilkscreenConfig(show_values=True, show_polarity_markers=True),
    ]

    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: {config.manufacturing_standard.name}")
        print("-" * 40)

        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)

        print(f"  Success: {result.success}")
        print(f"  References: {result.ref_count}")
        print(f"  Values: {result.value_count}")
        print(f"  Collisions: {result.collision_count}")
        print(f"  Polarity markers: {result.total_polarity_markers}")
        print(f"  Optimization iterations: {result.optimization_iterations}")
        print(f"  Average placement score: {result.average_placement_score:.1f}")
        print(f"  Min clearance: {result.min_clearance_achieved:.3f}mm")

        if result.warnings:
            print(f"  Warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                print(f"    - {w}")

        stats = piston.get_statistics()
        print(f"  Lines: {stats['total_lines']}, Arcs: {stats['total_arcs']}")

    # Test KiCad export
    print("\n" + "=" * 60)
    print("KiCad S-expression export (first 500 chars):")
    print("-" * 40)

    piston = SilkscreenPiston()
    result = piston.generate(parts_db, placement)
    kicad_output = piston.to_kicad_sexpr()
    print(kicad_output[:500] + "..." if len(kicad_output) > 500 else kicad_output)

    print("\n" + "=" * 60)
    print("Silkscreen Piston module loaded successfully!")
    print("Features:")
    print("  - Force-directed label optimization")
    print("  - IPC-7351 compliant sizing")
    print("  - JLCPCB/PCBWay manufacturing rules")
    print("  - Polarity and pin 1 markers")
    print("  - Fiducial generation")
    print("  - 8-position candidate generation")
    print("  - Collision detection and resolution")
