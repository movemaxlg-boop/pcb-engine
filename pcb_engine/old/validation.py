"""
PCB Engine - Validation Module (Enhanced)
==========================================

Validates the complete design against DRC rules.
This is Phase 8 of the algorithm.

VALIDATION PHILOSOPHY:
======================
The validation gate is the LAST DEFENSE before KiCad output.
It must catch ALL issues that would cause:
1. Manufacturing problems (clearance, trace width, drill)
2. Functional problems (unconnected nets, dangling traces)
3. Assembly problems (silkscreen overlap, courtyard violations)

DRC CATEGORIES:
===============
1. CONNECTIVITY - Are all nets properly connected?
2. CLEARANCE - Are traces/pads far enough apart?
3. TRACE GEOMETRY - Width, length matching, via count
4. VIA/HOLE - Diameter, drill, annular ring
5. ZONE - GND pour issues, thermal relief
6. SILKSCREEN - Overlap with pads, readability
7. MANUFACTURABILITY - Acid traps, acute angles
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class ViolationSeverity(Enum):
    """Violation severity levels"""
    ERROR = 'error'      # Must fix - will cause failure
    WARNING = 'warning'  # Should fix - may cause issues
    INFO = 'info'        # Advisory - best practice


class ViolationType(Enum):
    """Categories of DRC violations"""
    CONNECTIVITY = 'connectivity'
    CLEARANCE = 'clearance'
    TRACE_WIDTH = 'trace_width'
    VIA_DRILL = 'via_drill'
    VIA_ANNULAR = 'via_annular'
    VIA_SPACING = 'via_spacing'  # Via-to-via spacing
    HOLE_SPACING = 'hole_spacing'  # Drilled hole spacing
    DANGLING_TRACE = 'dangling_trace'
    ZONE_VIOLATION = 'zone_violation'
    SILKSCREEN = 'silkscreen'
    MANUFACTURING = 'manufacturing'
    COMPONENT_OVERLAP = 'component_overlap'
    COURTYARD = 'courtyard'


@dataclass
class Violation:
    """A DRC violation"""
    type: ViolationType
    severity: ViolationSeverity
    location: Tuple[float, float]
    message: str
    items: List[str] = field(default_factory=list)
    value: Optional[float] = None  # Actual value (e.g., 0.15mm clearance)
    limit: Optional[float] = None  # Required value (e.g., 0.2mm minimum)
    layer: str = ''

    def __str__(self):
        severity_prefix = {
            ViolationSeverity.ERROR: '[ERROR]',
            ViolationSeverity.WARNING: '[WARN]',
            ViolationSeverity.INFO: '[INFO]',
        }
        return f"{severity_prefix[self.severity]} {self.type.value}: {self.message}"


@dataclass
class ValidationResult:
    """Complete validation result"""
    valid: bool
    errors: List[Violation] = field(default_factory=list)
    warnings: List[Violation] = field(default_factory=list)
    info: List[Violation] = field(default_factory=list)

    # Summary statistics
    nets_checked: int = 0
    nets_connected: int = 0
    nets_unconnected: int = 0

    routes_checked: int = 0
    clearance_checks: int = 0
    clearance_passed: int = 0

    total_trace_length: float = 0.0
    total_vias: int = 0

    def add(self, violation: Violation):
        """Add a violation to appropriate list"""
        if violation.severity == ViolationSeverity.ERROR:
            self.errors.append(violation)
        elif violation.severity == ViolationSeverity.WARNING:
            self.warnings.append(violation)
        else:
            self.info.append(violation)

    @property
    def all_violations(self) -> List[Violation]:
        return self.errors + self.warnings + self.info

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'valid': self.valid,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'violations': [str(v) for v in self.all_violations],
            'summary': {
                'nets_checked': self.nets_checked,
                'nets_connected': self.nets_connected,
                'nets_unconnected': self.nets_unconnected,
                'routes_checked': self.routes_checked,
                'clearance_checks': self.clearance_checks,
                'clearance_passed': self.clearance_passed,
                'total_trace_length': f"{self.total_trace_length:.1f}mm",
                'total_vias': self.total_vias,
            }
        }


# =============================================================================
# GEOMETRY UTILITIES
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

        # Vector from seg_start to seg_end
        dx = x2 - x1
        dy = y2 - y1

        # Length squared
        len_sq = dx * dx + dy * dy

        if len_sq < 0.0001:  # Degenerate segment
            return GeometryUtils.distance(point, seg_start)

        # Project point onto line (t = 0..1 means on segment)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))

        # Closest point on segment
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return GeometryUtils.distance(point, (proj_x, proj_y))

    @staticmethod
    def segment_to_segment_distance(s1_start: Tuple[float, float],
                                     s1_end: Tuple[float, float],
                                     s2_start: Tuple[float, float],
                                     s2_end: Tuple[float, float]) -> float:
        """Calculate minimum distance between two line segments"""
        # Check all endpoint-to-segment distances
        d1 = GeometryUtils.point_to_segment_distance(s1_start, s2_start, s2_end)
        d2 = GeometryUtils.point_to_segment_distance(s1_end, s2_start, s2_end)
        d3 = GeometryUtils.point_to_segment_distance(s2_start, s1_start, s1_end)
        d4 = GeometryUtils.point_to_segment_distance(s2_end, s1_start, s1_end)

        return min(d1, d2, d3, d4)

    @staticmethod
    def bounding_box_overlap(box1: Tuple[float, float, float, float],
                              box2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes overlap (x1, y1, x2, y2)"""
        return not (box1[2] < box2[0] or  # box1 right < box2 left
                    box1[0] > box2[2] or  # box1 left > box2 right
                    box1[3] < box2[1] or  # box1 bottom < box2 top
                    box1[1] > box2[3])    # box1 top > box2 bottom

    @staticmethod
    def angle_between_vectors(v1: Tuple[float, float],
                               v2: Tuple[float, float]) -> float:
        """Calculate angle between two vectors in degrees"""
        # Normalize
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if len1 < 0.001 or len2 < 0.001:
            return 0

        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
        dot = max(-1, min(1, dot))  # Clamp for acos

        return math.degrees(math.acos(dot))


# =============================================================================
# CONNECTIVITY CHECKER
# =============================================================================

class ConnectivityChecker:
    """Checks that all nets are properly connected"""

    def __init__(self, rules):
        self.rules = rules

    def check(self, routes: Dict, parts_db: Dict, escapes: Dict) -> List[Violation]:
        """Check connectivity of all nets"""
        violations = []
        nets = parts_db.get('nets', {})

        for net_name, net_info in nets.items():
            # GND is now handled by copper pour/zone on B.Cu, not traces
            # Skip GND in connectivity check - zone fills will connect all GND pads
            if net_name == 'GND':
                continue

            pins = net_info.get('pins', [])
            if len(pins) < 2:
                continue

            # Check if net is routed
            if net_name not in routes:
                violations.append(Violation(
                    type=ViolationType.CONNECTIVITY,
                    severity=ViolationSeverity.ERROR,
                    location=(0, 0),
                    message=f"Net '{net_name}' is not routed ({len(pins)} pins unconnected)",
                    # IMPORTANT: Put net_name FIRST so feedback loop knows which net to reroute
                    items=[net_name] + [f"{comp}.{pin}" for comp, pin in pins],
                ))
                continue

            route = routes[net_name]

            # Check if route completed successfully
            if hasattr(route, 'success') and not route.success:
                violations.append(Violation(
                    type=ViolationType.CONNECTIVITY,
                    severity=ViolationSeverity.ERROR,
                    location=(0, 0),
                    message=f"Net '{net_name}' routing failed: {getattr(route, 'error', 'unknown')}",
                    # IMPORTANT: Put net_name FIRST so feedback loop knows which net to reroute
                    items=[net_name] + [f"{comp}.{pin}" for comp, pin in pins],
                ))
                continue

            # Check segment count vs pin count
            segments = getattr(route, 'segments', [])
            if len(segments) < len(pins) - 1:
                violations.append(Violation(
                    type=ViolationType.CONNECTIVITY,
                    severity=ViolationSeverity.WARNING,
                    location=(0, 0),
                    message=f"Net '{net_name}' may have insufficient segments ({len(segments)} for {len(pins)} pins)",
                    items=[f"{comp}.{pin}" for comp, pin in pins],
                ))

        return violations


# =============================================================================
# CLEARANCE CHECKER
# =============================================================================

class ClearanceChecker:
    """Checks clearance between traces, pads, and other elements"""

    def __init__(self, rules):
        self.rules = rules
        self.min_clearance = rules.min_clearance

    def check(self, routes: Dict, placement: Dict, parts_db: Dict) -> Tuple[List[Violation], int, int]:
        """
        Check clearance between all elements.

        Returns:
            (violations, total_checks, passed_checks)
        """
        violations = []
        total_checks = 0
        passed_checks = 0

        # Get all segments for collision checking
        all_segments = self._collect_all_segments(routes)
        all_pads = self._collect_all_pads(placement, parts_db)

        # Check segment-to-segment clearance
        seg_violations, seg_checks, seg_passed = self._check_segment_clearance(all_segments)
        violations.extend(seg_violations)
        total_checks += seg_checks
        passed_checks += seg_passed

        # Check segment-to-pad clearance
        pad_violations, pad_checks, pad_passed = self._check_segment_pad_clearance(all_segments, all_pads)
        violations.extend(pad_violations)
        total_checks += pad_checks
        passed_checks += pad_passed

        return violations, total_checks, passed_checks

    def _collect_all_segments(self, routes: Dict) -> List[Dict]:
        """Collect all segments from all routes"""
        segments = []
        for net_name, route in routes.items():
            for seg in getattr(route, 'segments', []):
                segments.append({
                    'net': net_name,
                    'start': seg.start,
                    'end': seg.end,
                    'width': getattr(seg, 'width', 0.25),
                    'layer': getattr(seg, 'layer', 'F.Cu'),
                })
        return segments

    def _collect_all_pads(self, placement: Dict, parts_db: Dict) -> List[Dict]:
        """
        Collect ALL physical pads from all placed components.

        CRITICAL FIX: Must include NC (No Connect) pads!
        'used_pins' only has pads with nets - misses NC pads.
        'physical_pins' has ALL pads including NC.

        NC pads cause "Items shorting two nets" errors in KiCad if tracks
        pass through them, even though they have no net assigned.
        """
        pads = []
        parts = parts_db.get('parts', {})

        for ref, pos in placement.items():
            part = parts.get(ref, {})

            # Build a map of pin numbers to nets from used_pins
            pin_nets = {}
            for pin in part.get('used_pins', []):
                pin_nets[pin.get('number', '')] = pin.get('net', '')

            # Use physical_pins (ALL pads) - fall back to used_pins if not available
            all_pins = part.get('physical_pins', part.get('used_pins', []))

            for pin in all_pins:
                pin_num = pin.get('number', '')

                # Get offset - handle multiple formats
                offset = pin.get('offset', None)
                if not offset or offset == (0, 0):
                    # Try 'physical' dict format: {offset_x, offset_y}
                    physical = pin.get('physical', {})
                    if physical:
                        offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))
                    else:
                        offset = (0, 0)

                # Get net from pin_nets map (NC pads won't have a net)
                net = pin_nets.get(pin_num, '')

                # Get pad size from either 'pad_size' or 'size' key
                pad_size = pin.get('pad_size', pin.get('size', (0.8, 0.8)))

                pads.append({
                    'component': ref,
                    'pin': pin_num,
                    'net': net,  # Empty string for NC pads
                    'position': (pos.x + offset[0], pos.y + offset[1]),
                    'size': pad_size,
                })

        return pads

    def _check_segment_clearance(self, segments: List[Dict]) -> Tuple[List[Violation], int, int]:
        """Check clearance between trace segments"""
        violations = []
        total_checks = 0
        passed_checks = 0

        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                # Skip same net
                if seg1['net'] == seg2['net']:
                    continue

                # Skip different layers
                if seg1['layer'] != seg2['layer']:
                    continue

                total_checks += 1

                # Calculate distance
                distance = GeometryUtils.segment_to_segment_distance(
                    seg1['start'], seg1['end'],
                    seg2['start'], seg2['end']
                )

                # Account for trace widths
                required_clearance = self.min_clearance + (seg1['width'] + seg2['width']) / 2

                if distance < required_clearance:
                    violations.append(Violation(
                        type=ViolationType.CLEARANCE,
                        severity=ViolationSeverity.ERROR,
                        location=seg1['start'],
                        message=f"Clearance violation: {distance:.3f}mm < {required_clearance:.3f}mm between {seg1['net']} and {seg2['net']}",
                        items=[seg1['net'], seg2['net']],
                        value=distance,
                        limit=required_clearance,
                        layer=seg1['layer'],
                    ))
                else:
                    passed_checks += 1

        return violations, total_checks, passed_checks

    def _check_segment_pad_clearance(self, segments: List[Dict], pads: List[Dict]) -> Tuple[List[Violation], int, int]:
        """Check clearance between trace segments and pads"""
        violations = []
        total_checks = 0
        passed_checks = 0

        for seg in segments:
            for pad in pads:
                # Skip if same net (connection allowed)
                if seg['net'] == pad['net'] and pad['net']:
                    continue

                # CRITICAL FIX: Do NOT skip NC pads (empty net)!
                # Tracks must maintain clearance from ALL pads including NC pads.
                # Routing through an NC pad causes "Items shorting two nets" in KiCad DRC.
                # NC pads are just as physical as connected pads.

                total_checks += 1

                # Distance from pad center to segment
                distance = GeometryUtils.point_to_segment_distance(
                    pad['position'], seg['start'], seg['end']
                )

                # Account for trace width and pad size
                # Use actual clearance requirement (edge-to-edge)
                pad_radius = max(pad['size']) / 2
                trace_half_width = seg['width'] / 2

                # Required: edge of trace to edge of pad
                edge_to_edge = distance - pad_radius - trace_half_width

                if edge_to_edge < self.min_clearance:
                    # Determine severity based on overlap amount
                    if edge_to_edge < -0.01:  # Actual overlap
                        severity = ViolationSeverity.ERROR
                    else:
                        severity = ViolationSeverity.WARNING

                    # Special message for NC pad violations
                    if not pad['net']:
                        msg = f"Track [{seg['net']}] too close to NC pad {pad['component']}.{pad['pin']}: {edge_to_edge:.3f}mm"
                    else:
                        msg = f"Trace-to-pad clearance: {edge_to_edge:.3f}mm (need {self.min_clearance:.3f}mm)"

                    # CRITICAL FIX: Use the pad's net name, not pad reference
                    # This allows DRC feedback to reroute the correct net
                    # For NC pads (no net), use a marker so feedback knows it's a pad
                    pad_net = pad['net'] if pad['net'] else f"__NC__{pad['component']}.{pad['pin']}"
                    violations.append(Violation(
                        type=ViolationType.CLEARANCE,
                        severity=severity,
                        location=pad['position'],
                        message=msg,
                        items=[seg['net'], pad_net],
                        value=edge_to_edge,
                        limit=self.min_clearance,
                    ))
                else:
                    passed_checks += 1

        return violations, total_checks, passed_checks


# =============================================================================
# TRACE WIDTH CHECKER
# =============================================================================

class TraceWidthChecker:
    """Checks trace width requirements"""

    def __init__(self, rules):
        self.rules = rules
        self.min_width = rules.min_trace_width

    def check(self, routes: Dict, parts_db: Dict) -> List[Violation]:
        """Check all trace widths"""
        violations = []
        nets = parts_db.get('nets', {})

        for net_name, route in routes.items():
            net_info = nets.get(net_name, {})

            # Get net-specific width requirement
            required_width = net_info.get('min_width', self.min_width)

            # Check if power net (needs wider traces)
            net_type = net_info.get('type')
            if hasattr(net_type, 'value') and net_type.value == 'power':
                required_width = max(required_width, 0.5)  # 0.5mm for power

            for segment in getattr(route, 'segments', []):
                seg_width = getattr(segment, 'width', 0.25)

                if seg_width < required_width - 0.001:  # Tolerance
                    violations.append(Violation(
                        type=ViolationType.TRACE_WIDTH,
                        severity=ViolationSeverity.ERROR,
                        location=segment.start,
                        message=f"Trace width {seg_width:.3f}mm < required {required_width:.3f}mm on net '{net_name}'",
                        items=[net_name],
                        value=seg_width,
                        limit=required_width,
                        layer=getattr(segment, 'layer', 'F.Cu'),
                    ))

        return violations


# =============================================================================
# VIA CHECKER
# =============================================================================

class ViaChecker:
    """Checks via specifications"""

    def __init__(self, rules):
        self.rules = rules

    def check(self, routes: Dict, escapes: Dict = None) -> Tuple[List[Violation], int]:
        """
        Check all vias.

        Args:
            routes: All routed nets
            escapes: Escape vectors (to check if via connects to escape on F.Cu)

        Returns:
            (violations, total_via_count)
        """
        violations = []
        total_vias = 0
        escapes = escapes or {}

        for net_name, route in routes.items():
            for via in getattr(route, 'vias', []):
                total_vias += 1

                # Check via diameter
                diameter = getattr(via, 'diameter', 0.8)
                if diameter < self.rules.min_via_diameter:
                    violations.append(Violation(
                        type=ViolationType.VIA_DRILL,
                        severity=ViolationSeverity.ERROR,
                        location=via.position,
                        message=f"Via diameter {diameter:.3f}mm < minimum {self.rules.min_via_diameter:.3f}mm",
                        items=[net_name],
                        value=diameter,
                        limit=self.rules.min_via_diameter,
                    ))

                # Check drill size
                drill = getattr(via, 'drill', 0.4)
                if drill < self.rules.min_via_drill:
                    violations.append(Violation(
                        type=ViolationType.VIA_DRILL,
                        severity=ViolationSeverity.ERROR,
                        location=via.position,
                        message=f"Via drill {drill:.3f}mm < minimum {self.rules.min_via_drill:.3f}mm",
                        items=[net_name],
                        value=drill,
                        limit=self.rules.min_via_drill,
                    ))

                # Check annular ring
                annular_ring = (diameter - drill) / 2
                if annular_ring < self.rules.min_annular_ring:
                    violations.append(Violation(
                        type=ViolationType.VIA_ANNULAR,
                        severity=ViolationSeverity.ERROR,
                        location=via.position,
                        message=f"Via annular ring {annular_ring:.3f}mm < minimum {self.rules.min_annular_ring:.3f}mm",
                        items=[net_name],
                        value=annular_ring,
                        limit=self.rules.min_annular_ring,
                    ))

                # Check via has connections on both layers
                # A via must connect tracks on F.Cu and B.Cu to be useful
                fcu_conn = False
                bcu_conn = False
                # Use larger tolerance for grid-aligned routing (0.5mm grid typical)
                tol = 0.3  # mm tolerance - allows for grid quantization

                # Check signal route segments
                for seg in getattr(route, 'segments', []):
                    seg_layer = getattr(seg, 'layer', 'F.Cu')
                    for pt in [seg.start, seg.end]:
                        if abs(pt[0] - via.position[0]) < tol and abs(pt[1] - via.position[1]) < tol:
                            if seg_layer == 'F.Cu':
                                fcu_conn = True
                            else:
                                bcu_conn = True

                # Also check escape routes (which may connect to via on F.Cu)
                # Note: Handle both EscapeVector (.endpoint) and EscapeRoute (.end)
                for ref, pin_escapes in escapes.items():
                    for pin, escape in pin_escapes.items():
                        if getattr(escape, 'net', '') == net_name:
                            # Try .endpoint first, then .end (different escape classes)
                            esc_end = getattr(escape, 'endpoint', None)
                            if esc_end is None:
                                esc_end = getattr(escape, 'end', (0, 0))
                            esc_layer = getattr(escape, 'layer', 'F.Cu')
                            if abs(esc_end[0] - via.position[0]) < tol and abs(esc_end[1] - via.position[1]) < tol:
                                if esc_layer == 'F.Cu':
                                    fcu_conn = True
                                else:
                                    bcu_conn = True

                # IMPORTANT: If via is at start or end of a B.Cu route that connects
                # to an F.Cu escape, both connections are valid even if the via
                # doesn't have a separate F.Cu track (the escape provides F.Cu connection)
                # This handles the case where: Escape (F.Cu) -> Via -> Route (B.Cu)
                if bcu_conn and not fcu_conn:
                    # Check if there's an escape endpoint near the via
                    for ref, pin_escapes in escapes.items():
                        for pin, escape in pin_escapes.items():
                            if getattr(escape, 'net', '') == net_name:
                                esc_end = getattr(escape, 'endpoint', None)
                                if esc_end is None:
                                    esc_end = getattr(escape, 'end', (0, 0))
                                # If via is near escape end, the escape provides F.Cu connection
                                if abs(esc_end[0] - via.position[0]) < tol and abs(esc_end[1] - via.position[1]) < tol:
                                    fcu_conn = True  # Escape provides F.Cu connection to via
                                    break
                        if fcu_conn:
                            break

                if not fcu_conn or not bcu_conn:
                    missing = []
                    if not fcu_conn:
                        missing.append('F.Cu')
                    if not bcu_conn:
                        missing.append('B.Cu')
                    violations.append(Violation(
                        type=ViolationType.DANGLING_TRACE,
                        severity=ViolationSeverity.ERROR,
                        location=via.position,
                        message=f"Via not connected on {', '.join(missing)} - no track connects to via on this layer",
                        items=[net_name],
                    ))

        return violations, total_vias


# =============================================================================
# VIA SPACING CHECKER (NEW - Catches "Drilled hole too close" errors)
# =============================================================================

class ViaSpacingChecker:
    """
    Checks via-to-via and via-to-hole spacing.

    This catches the KiCad DRC warning:
    "Drilled hole in U1 pad 1 too close to drilled hole in via"

    Default KiCad board setup constraint is ~0.25mm edge-to-edge.
    """

    def __init__(self, rules):
        self.rules = rules
        # Minimum edge-to-edge spacing between drilled holes
        # KiCad default is typically 0.25mm, but we use a more conservative value
        self.min_hole_spacing = getattr(rules, 'min_hole_clearance', 0.25)

    def check(self, routes: Dict, placement: Dict = None, parts_db: Dict = None) -> List[Violation]:
        """
        Check via-to-via spacing across all nets.

        Args:
            routes: All routed nets with vias
            placement: Component placement (for pad hole checking)
            parts_db: Parts database (for pad info)

        Returns:
            List of spacing violations
        """
        violations = []

        # Collect ALL vias from all nets
        all_vias = []
        for net_name, route in routes.items():
            for via in getattr(route, 'vias', []):
                all_vias.append({
                    'position': via.position,
                    'net': net_name,
                    'diameter': getattr(via, 'diameter', 0.8),
                    'drill': getattr(via, 'drill', 0.4),
                })

        # Check via-to-via spacing
        for i, via1 in enumerate(all_vias):
            for via2 in all_vias[i+1:]:
                # Calculate center-to-center distance
                dist = math.sqrt(
                    (via1['position'][0] - via2['position'][0])**2 +
                    (via1['position'][1] - via2['position'][1])**2
                )

                # Calculate edge-to-edge distance
                # Edge distance = center distance - radius1 - radius2
                edge_dist = dist - via1['diameter']/2 - via2['diameter']/2

                if edge_dist < self.min_hole_spacing:
                    # This is the EXACT error KiCad reports
                    violations.append(Violation(
                        type=ViolationType.VIA_SPACING,
                        severity=ViolationSeverity.ERROR if edge_dist < 0 else ViolationSeverity.WARNING,
                        location=via1['position'],
                        message=f"Drilled hole too close to other hole: {edge_dist:.3f}mm < {self.min_hole_spacing:.3f}mm",
                        items=[via1['net'], via2['net']],
                        value=edge_dist,
                        limit=self.min_hole_spacing,
                    ))

        # Also check via-to-pad spacing if placement info provided
        if placement and parts_db:
            pad_holes = self._collect_pad_holes(placement, parts_db)

            for via in all_vias:
                for pad in pad_holes:
                    # Skip same net (via might connect to pad)
                    if via['net'] == pad['net']:
                        continue

                    dist = math.sqrt(
                        (via['position'][0] - pad['position'][0])**2 +
                        (via['position'][1] - pad['position'][1])**2
                    )

                    # Edge-to-edge for THT pad
                    if pad.get('drill', 0) > 0:
                        edge_dist = dist - via['diameter']/2 - pad['drill']/2
                        if edge_dist < self.min_hole_spacing:
                            violations.append(Violation(
                                type=ViolationType.HOLE_SPACING,
                                severity=ViolationSeverity.ERROR if edge_dist < 0 else ViolationSeverity.WARNING,
                                location=via['position'],
                                message=f"Via too close to pad hole ({pad['ref']}.{pad['pin']}): {edge_dist:.3f}mm",
                                items=[via['net'], f"{pad['ref']}.{pad['pin']}"],
                                value=edge_dist,
                                limit=self.min_hole_spacing,
                            ))

        return violations

    def _collect_pad_holes(self, placement: Dict, parts_db: Dict) -> List[Dict]:
        """Collect all THT pad holes from placed components"""
        pad_holes = []
        parts = parts_db.get('parts', {})

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            for pin in part.get('used_pins', []):
                # Only THT pads have drill holes
                drill = pin.get('drill', 0)
                if drill > 0:
                    offset = pin.get('offset', (0, 0))
                    pad_holes.append({
                        'ref': ref,
                        'pin': pin['number'],
                        'net': pin.get('net', ''),
                        'position': (pos.x + offset[0], pos.y + offset[1]),
                        'drill': drill,
                    })

        return pad_holes


# =============================================================================
# DANGLING TRACE CHECKER
# =============================================================================

class DanglingTraceChecker:
    """Checks for dangling (unconnected) trace ends"""

    def __init__(self, rules):
        self.rules = rules

    def check(self, routes: Dict, placement: Dict, parts_db: Dict, escapes: Dict) -> List[Violation]:
        """Check for dangling trace ends"""
        violations = []

        # Build set of valid connection points
        valid_points = self._get_valid_endpoints(placement, parts_db, escapes)

        for net_name, route in routes.items():
            segments = getattr(route, 'segments', [])
            if not segments:
                continue

            # Collect all segment endpoints
            endpoints = set()
            for seg in segments:
                endpoints.add(self._round_point(seg.start))
                endpoints.add(self._round_point(seg.end))

            # Via positions count as valid connection points (they connect layers)
            via_points = set()
            for via in getattr(route, 'vias', []):
                via_points.add(self._round_point(via.position))
                endpoints.add(self._round_point(via.position))

            # Check each segment endpoint
            for seg in segments:
                for point in [seg.start, seg.end]:
                    rp = self._round_point(point)

                    # Count connections at this point
                    connections = sum(1 for s in segments
                                    if self._round_point(s.start) == rp
                                    or self._round_point(s.end) == rp)

                    # Check if it's a valid pad/via/escape connection
                    is_valid_endpoint = rp in valid_points.get(net_name, set())
                    is_via_point = rp in via_points  # Vias are valid endpoints

                    # Dangling if only 1 connection and not at a pad/via
                    if connections <= 1 and not is_valid_endpoint and not is_via_point:
                        violations.append(Violation(
                            type=ViolationType.DANGLING_TRACE,
                            severity=ViolationSeverity.WARNING,
                            location=point,
                            message=f"Potential dangling trace end on net '{net_name}' at ({point[0]:.2f}, {point[1]:.2f})",
                            items=[net_name],
                        ))

        return violations

    def _get_valid_endpoints(self, placement: Dict, parts_db: Dict, escapes: Dict) -> Dict[str, Set]:
        """Get valid endpoint positions per net"""
        valid = {}  # {net_name: set of (x, y)}
        parts = parts_db.get('parts', {})

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            for pin in part.get('used_pins', []):
                net = pin.get('net', '')
                if not net:
                    continue

                if net not in valid:
                    valid[net] = set()

                offset = pin.get('offset', (0, 0))
                pad_pos = (pos.x + offset[0], pos.y + offset[1])
                valid[net].add(self._round_point(pad_pos))

                # Also add escape endpoints
                if ref in escapes and pin['number'] in escapes[ref]:
                    escape = escapes[ref][pin['number']]
                    valid[net].add(self._round_point(escape.end))

        return valid

    def _round_point(self, point: Tuple[float, float], precision: int = 2) -> Tuple[float, float]:
        """Round point for comparison"""
        return (round(point[0], precision), round(point[1], precision))


# =============================================================================
# MANUFACTURING CHECKER
# =============================================================================

class ManufacturingChecker:
    """Checks for manufacturing issues like acid traps and acute angles"""

    def __init__(self, rules):
        self.rules = rules
        self.min_angle = 90  # Minimum angle for traces (avoid acid traps)

    def check(self, routes: Dict) -> List[Violation]:
        """Check for manufacturing issues"""
        violations = []

        for net_name, route in routes.items():
            segments = getattr(route, 'segments', [])

            # Check for acute angles between consecutive segments
            for i in range(len(segments) - 1):
                seg1 = segments[i]
                seg2 = segments[i + 1]

                # Check if segments share a point
                if self._points_equal(seg1.end, seg2.start):
                    angle = self._calculate_angle(seg1, seg2)
                    if angle < self.min_angle:
                        violations.append(Violation(
                            type=ViolationType.MANUFACTURING,
                            severity=ViolationSeverity.WARNING,
                            location=seg1.end,
                            message=f"Acute angle ({angle:.0f}°) may cause acid trap on net '{net_name}'",
                            items=[net_name],
                            value=angle,
                            limit=self.min_angle,
                        ))

        return violations

    def _points_equal(self, p1: Tuple[float, float], p2: Tuple[float, float], tol: float = 0.01) -> bool:
        """Check if two points are equal within tolerance"""
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    def _calculate_angle(self, seg1, seg2) -> float:
        """Calculate angle between two segments"""
        v1 = (seg1.start[0] - seg1.end[0], seg1.start[1] - seg1.end[1])
        v2 = (seg2.end[0] - seg2.start[0], seg2.end[1] - seg2.start[1])

        return GeometryUtils.angle_between_vectors(v1, v2)


# =============================================================================
# SILKSCREEN CHECKER (NEW - Catches "Silkscreen clipped by solder mask" warnings)
# =============================================================================

class SilkscreenChecker:
    """
    Checks silkscreen text positioning to avoid solder mask clipping.

    This catches the KiCad DRC warning:
    "Silkscreen clipped by solder mask - Reference field of U2"

    Silkscreen text must be positioned OUTSIDE the solder mask opening area.
    """

    def __init__(self, rules):
        self.rules = rules
        # Minimum clearance between silkscreen and solder mask edge
        self.min_silkscreen_clearance = 0.15  # mm

    def check(self, placement: Dict, parts_db: Dict) -> List[Violation]:
        """
        Check silkscreen text positions against solder mask areas.

        Args:
            placement: Component placement positions
            parts_db: Parts database with component info

        Returns:
            List of silkscreen violations
        """
        violations = []
        parts = parts_db.get('parts', {})

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            physical = part.get('physical', {})

            # Get component body size
            body = physical.get('body', part.get('size', (2.0, 2.0)))
            if not isinstance(body, (list, tuple)):
                body = (2.0, 2.0)

            body_half_h = body[1] / 2

            # Calculate max pad extent (pads may extend beyond body)
            max_pad_extent = 0
            for pin in part.get('used_pins', part.get('pins', [])):
                offset = pin.get('offset', pin.get('physical', {}).get('offset_y', 0))
                if isinstance(offset, (list, tuple)):
                    offset_y = abs(offset[1])
                else:
                    offset_y = abs(offset)
                pad_size = pin.get('pad_size', (0.6, 0.6))
                if isinstance(pad_size, (list, tuple)):
                    pad_h = pad_size[1] / 2
                else:
                    pad_h = 0.3
                pad_extent = offset_y + pad_h
                max_pad_extent = max(max_pad_extent, pad_extent)

            # Solder mask opening is pad area + expansion (typically 0.05mm)
            solder_mask_edge = max(body_half_h, max_pad_extent) + 0.1

            # Reference text typical position (estimated from script generation)
            # Default: 1.0mm below body center for small parts, more for larger
            ref_text_y = body_half_h + 1.0  # typical offset

            # Check if reference text would overlap solder mask
            # Text height is typically 0.8mm, so center is at ref_text_y
            # Top edge of text would be at ref_text_y - 0.4
            text_top_edge = ref_text_y - 0.4

            if text_top_edge < solder_mask_edge + self.min_silkscreen_clearance:
                violations.append(Violation(
                    type=ViolationType.SILKSCREEN,
                    severity=ViolationSeverity.WARNING,
                    location=(pos.x, pos.y),
                    message=f"Silkscreen may clip solder mask on {ref} (text at {text_top_edge:.2f}mm, mask edge at {solder_mask_edge:.2f}mm)",
                    items=[ref],
                    value=text_top_edge,
                    limit=solder_mask_edge + self.min_silkscreen_clearance,
                ))

        return violations


# =============================================================================
# ESCAPE CONNECTION CHECKER
# =============================================================================

class EscapeConnectionChecker:
    """
    Checks that escape routes connect to signal routes or other valid endpoints.

    An escape route goes from a pad to a routing grid point. If no signal route
    connects to that escape endpoint, the escape is "dangling" and will show up
    as an unconnected track in KiCad DRC.
    """

    def __init__(self, rules):
        self.rules = rules

    def check(self, escapes: Dict, routes: Dict, parts_db: Dict) -> List[Violation]:
        """
        Check that all escape routes connect to something.

        Args:
            escapes: Escape vectors per component/pin
            routes: Routed signal nets
            parts_db: Parts database with net info
        """
        violations = []
        tol = 0.01  # mm tolerance for position matching

        # Build set of all signal route endpoints and via positions
        route_endpoints = set()
        via_positions = set()

        for net_name, route in routes.items():
            for seg in getattr(route, 'segments', []):
                route_endpoints.add(self._round_point(seg.start))
                route_endpoints.add(self._round_point(seg.end))
            for via in getattr(route, 'vias', []):
                via_positions.add(self._round_point(via.position))
                route_endpoints.add(self._round_point(via.position))

        # Check each escape
        # Note: Handle both EscapeVector (.endpoint) and EscapeRoute (.end)
        for ref, pin_escapes in escapes.items():
            for pin, escape in pin_escapes.items():
                net = getattr(escape, 'net', '')
                # Try .endpoint first, then .end (different escape classes)
                endpoint = getattr(escape, 'endpoint', None)
                if endpoint is None:
                    endpoint = getattr(escape, 'end', (0, 0))
                ep_rounded = self._round_point(endpoint)

                # Skip GND escapes - they may connect to zone fill
                if net == 'GND':
                    continue

                # Check if escape endpoint connects to:
                # 1. A signal route segment endpoint
                # 2. A via position
                connects_to_route = ep_rounded in route_endpoints
                connects_to_via = ep_rounded in via_positions

                # Also check if another escape on same net connects (multi-pin nets)
                connects_to_other_escape = False
                for other_ref, other_pin_escapes in escapes.items():
                    for other_pin, other_escape in other_pin_escapes.items():
                        if (other_ref, other_pin) != (ref, pin):
                            if getattr(other_escape, 'net', '') == net:
                                # Try .endpoint first, then .end
                                other_endpoint = getattr(other_escape, 'endpoint', None)
                                if other_endpoint is None:
                                    other_endpoint = getattr(other_escape, 'end', (0, 0))
                                other_ep = self._round_point(other_endpoint)
                                if ep_rounded == other_ep:
                                    connects_to_other_escape = True

                if not connects_to_route and not connects_to_via and not connects_to_other_escape:
                    # Calculate escape length for the message
                    start = getattr(escape, 'start', (0, 0))
                    length = ((endpoint[0] - start[0])**2 + (endpoint[1] - start[1])**2)**0.5

                    violations.append(Violation(
                        type=ViolationType.DANGLING_TRACE,
                        severity=ViolationSeverity.WARNING,
                        location=endpoint,
                        message=f"Escape route [{net}] on {escape.layer} has unconnected end, length {length:.4f}mm",
                        items=[net, f"{ref}.{pin}"],
                        value=length,
                    ))

        return violations

    def _round_point(self, point: Tuple[float, float], precision: int = 2) -> Tuple[float, float]:
        """Round point coordinates for comparison"""
        return (round(point[0], precision), round(point[1], precision))


# =============================================================================
# TRACK OVERLAP CHECKER (ROOT CAUSE FIX - Detects crossing/shorting tracks)
# =============================================================================

class TrackOverlapChecker:
    """
    Checks for track-to-track overlaps and crossings BEFORE KiCad generation.

    ROOT CAUSE FIX: This catches the exact DRC errors KiCad was reporting:
    - [tracks_crossing]: Tracks crossing
    - [shorting_items]: Items shorting two nets

    By detecting these in our engine, we prevent generating bad geometry.
    """

    def __init__(self, rules):
        self.rules = rules

    def check(self, routes: Dict, escapes: Dict) -> List[Violation]:
        """Check for track overlaps between different nets"""
        violations = []

        # Collect ALL track segments (both escapes and signal routes)
        all_segments = []

        # Add escape route segments
        for ref, pin_escapes in escapes.items():
            for pin, escape in pin_escapes.items():
                start = getattr(escape, 'start', (0, 0))
                end = getattr(escape, 'end', getattr(escape, 'endpoint', (0, 0)))
                net = getattr(escape, 'net', '')
                layer = getattr(escape, 'layer', 'F.Cu')

                all_segments.append({
                    'start': start,
                    'end': end,
                    'net': net,
                    'layer': layer,
                    'type': 'escape',
                    'ref': f"{ref}.{pin}",
                })

        # Add signal route segments
        for net_name, route in routes.items():
            for seg in getattr(route, 'segments', []):
                all_segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'net': net_name,
                    'layer': getattr(seg, 'layer', 'F.Cu'),
                    'type': 'route',
                    'ref': net_name,
                })

        # Check for overlaps between segments on the same layer from different nets
        for i, seg1 in enumerate(all_segments):
            for seg2 in all_segments[i+1:]:
                # Skip same net (connections are allowed)
                if seg1['net'] == seg2['net']:
                    continue

                # Skip different layers (no collision possible)
                if seg1['layer'] != seg2['layer']:
                    continue

                # Check if segments cross or overlap
                crossing = self._segments_intersect(
                    seg1['start'], seg1['end'],
                    seg2['start'], seg2['end']
                )

                if crossing:
                    # This is the exact error KiCad reports: tracks_crossing / shorting_items
                    violations.append(Violation(
                        type=ViolationType.CLEARANCE,
                        severity=ViolationSeverity.ERROR,
                        location=crossing,  # Intersection point
                        message=f"Tracks crossing/shorting: [{seg1['net']}] and [{seg2['net']}] on {seg1['layer']}",
                        items=[seg1['ref'], seg2['ref']],
                        layer=seg1['layer'],
                    ))

        return violations

    def _segments_intersect(self, p1: Tuple, p2: Tuple,
                            p3: Tuple, p4: Tuple) -> Optional[Tuple[float, float]]:
        """
        Check if two line segments intersect.

        Returns intersection point if they cross, None otherwise.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Calculate direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        # Cross product of directions
        cross = dx1 * dy2 - dy1 * dx2

        if abs(cross) < 1e-10:
            # Parallel lines - check for collinear overlap
            return self._check_collinear_overlap(p1, p2, p3, p4)

        # Calculate parameters for intersection point
        dx3 = x1 - x3
        dy3 = y1 - y3

        t1 = (dx2 * dy3 - dy2 * dx3) / cross
        t2 = (dx1 * dy3 - dy1 * dx3) / cross

        # Check if intersection is within both segments
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            # Intersection point
            ix = x1 + t1 * dx1
            iy = y1 + t1 * dy1
            return (ix, iy)

        return None

    def _check_collinear_overlap(self, p1: Tuple, p2: Tuple,
                                  p3: Tuple, p4: Tuple) -> Optional[Tuple[float, float]]:
        """Check if collinear segments overlap"""
        # Project all points onto the line and check for overlap
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Use x-coordinate for horizontal, y for vertical
        if abs(x2 - x1) > abs(y2 - y1):
            # Mostly horizontal
            min1, max1 = min(x1, x2), max(x1, x2)
            min2, max2 = min(x3, x4), max(x3, x4)

            # Check if on same line (same y)
            if abs(y1 - y3) > 0.01:
                return None

            # Check overlap
            if max1 >= min2 and max2 >= min1:
                overlap_x = max(min1, min2)
                return (overlap_x, y1)
        else:
            # Mostly vertical
            min1, max1 = min(y1, y2), max(y1, y2)
            min2, max2 = min(y3, y4), max(y3, y4)

            # Check if on same line (same x)
            if abs(x1 - x3) > 0.01:
                return None

            # Check overlap
            if max1 >= min2 and max2 >= min1:
                overlap_y = max(min1, min2)
                return (x1, overlap_y)

        return None


# =============================================================================
# COMPONENT OVERLAP CHECKER
# =============================================================================

class ComponentOverlapChecker:
    """Checks for component overlap and courtyard violations"""

    def __init__(self, rules):
        self.rules = rules
        self.courtyard_margin = 0.25  # mm

    def check(self, placement: Dict, parts_db: Dict) -> List[Violation]:
        """Check for component overlaps"""
        violations = []
        parts = parts_db.get('parts', {})

        # Build bounding boxes for all components
        boxes = {}
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            phys = part.get('physical', {})
            body = phys.get('body', (5, 5))

            # Bounding box with courtyard margin
            margin = self.courtyard_margin
            boxes[ref] = (
                pos.x - body[0]/2 - margin,
                pos.y - body[1]/2 - margin,
                pos.x + body[0]/2 + margin,
                pos.y + body[1]/2 + margin,
            )

        # Check all pairs
        refs = list(boxes.keys())
        for i, ref1 in enumerate(refs):
            for ref2 in refs[i+1:]:
                if GeometryUtils.bounding_box_overlap(boxes[ref1], boxes[ref2]):
                    pos1 = placement[ref1]
                    pos2 = placement[ref2]

                    violations.append(Violation(
                        type=ViolationType.COMPONENT_OVERLAP,
                        severity=ViolationSeverity.ERROR,
                        location=(pos1.x, pos1.y),
                        message=f"Component overlap/courtyard violation: {ref1} and {ref2}",
                        items=[ref1, ref2],
                    ))

        return violations


# =============================================================================
# MAIN VALIDATION GATE
# =============================================================================

class ValidationGate:
    """
    Validates the complete PCB design.

    This is the final gate before output generation.
    ALL checks must pass for a valid design.

    Checks performed:
    - All nets connected
    - No dangling traces
    - Clearance rules met
    - Trace width rules met
    - Via specifications met
    - Via-to-via spacing (catches "drilled hole too close" errors)
    - Silkscreen positioning (catches "silkscreen clipped by solder mask" warnings)
    - Manufacturing checks (acid traps)
    - Component overlap checks
    """

    def __init__(self, rules):
        self.rules = rules

        # Initialize checkers
        self.connectivity_checker = ConnectivityChecker(rules)
        self.clearance_checker = ClearanceChecker(rules)
        self.trace_width_checker = TraceWidthChecker(rules)
        self.via_checker = ViaChecker(rules)
        self.via_spacing_checker = ViaSpacingChecker(rules)  # Via spacing
        self.silkscreen_checker = SilkscreenChecker(rules)  # NEW: Silkscreen clipping
        self.dangling_checker = DanglingTraceChecker(rules)
        self.escape_checker = EscapeConnectionChecker(rules)
        self.manufacturing_checker = ManufacturingChecker(rules)
        self.overlap_checker = ComponentOverlapChecker(rules)
        self.track_overlap_checker = TrackOverlapChecker(rules)  # Track crossing/shorting

    def validate(self, routes: Dict, placement: Dict, parts_db: Dict,
                 escapes: Dict = None) -> Dict:
        """
        Validate the complete design.

        Args:
            routes: All routed nets
            placement: Component placement
            parts_db: Parts database
            escapes: Escape vectors (optional)

        Returns:
            Dictionary with validation results
        """
        result = ValidationResult(valid=True)
        escapes = escapes or {}

        # 1. Connectivity check
        connectivity_violations = self.connectivity_checker.check(routes, parts_db, escapes)
        for v in connectivity_violations:
            result.add(v)

        result.nets_checked = len(parts_db.get('nets', {})) - 1  # -1 for GND
        result.nets_connected = result.nets_checked - len([v for v in connectivity_violations
                                                           if v.severity == ViolationSeverity.ERROR])
        result.nets_unconnected = len([v for v in connectivity_violations
                                       if v.severity == ViolationSeverity.ERROR])

        # 2. Clearance check
        clearance_violations, total_checks, passed = self.clearance_checker.check(
            routes, placement, parts_db
        )
        for v in clearance_violations:
            result.add(v)

        result.clearance_checks = total_checks
        result.clearance_passed = passed

        # 3. Trace width check
        width_violations = self.trace_width_checker.check(routes, parts_db)
        for v in width_violations:
            result.add(v)

        # 4. Via check (include escapes to verify F.Cu connections)
        via_violations, total_vias = self.via_checker.check(routes, escapes)
        for v in via_violations:
            result.add(v)

        result.total_vias = total_vias

        # 4b. Via spacing check (NEW - catches "drilled hole too close" errors)
        via_spacing_violations = self.via_spacing_checker.check(routes, placement, parts_db)
        for v in via_spacing_violations:
            result.add(v)

        # 5. Dangling trace check (signal routes)
        dangling_violations = self.dangling_checker.check(routes, placement, parts_db, escapes)
        for v in dangling_violations:
            result.add(v)

        # 6. Escape connection check (escape routes that don't connect to signal routes)
        escape_violations = self.escape_checker.check(escapes, routes, parts_db)
        for v in escape_violations:
            result.add(v)

        # 7. Manufacturing check
        mfg_violations = self.manufacturing_checker.check(routes)
        for v in mfg_violations:
            result.add(v)

        # 8. Component overlap check
        overlap_violations = self.overlap_checker.check(placement, parts_db)
        for v in overlap_violations:
            result.add(v)

        # 9. Track overlap check (catches crossing/shorting before KiCad)
        track_overlap_violations = self.track_overlap_checker.check(routes, escapes)
        for v in track_overlap_violations:
            result.add(v)

        # 10. Silkscreen check (NEW - catches "silkscreen clipped by solder mask")
        silkscreen_violations = self.silkscreen_checker.check(placement, parts_db)
        for v in silkscreen_violations:
            result.add(v)

        # Calculate totals
        result.routes_checked = len(routes)
        result.total_trace_length = sum(
            sum(seg.length for seg in getattr(route, 'segments', []))
            for route in routes.values()
        )

        # Final validity
        result.valid = len(result.errors) == 0

        return result.to_dict()

    def validate_detailed(self, routes: Dict, placement: Dict, parts_db: Dict,
                          escapes: Dict = None) -> ValidationResult:
        """
        Validate and return detailed result object.

        Use this for programmatic access to violations.
        """
        # Same as validate but returns ValidationResult object
        result = ValidationResult(valid=True)
        escapes = escapes or {}

        # Run all checks...
        connectivity_violations = self.connectivity_checker.check(routes, parts_db, escapes)
        for v in connectivity_violations:
            result.add(v)

        clearance_violations, total_checks, passed = self.clearance_checker.check(
            routes, placement, parts_db
        )
        for v in clearance_violations:
            result.add(v)

        width_violations = self.trace_width_checker.check(routes, parts_db)
        for v in width_violations:
            result.add(v)

        via_violations, total_vias = self.via_checker.check(routes, escapes)
        for v in via_violations:
            result.add(v)

        # Via spacing check (NEW)
        via_spacing_violations = self.via_spacing_checker.check(routes, placement, parts_db)
        for v in via_spacing_violations:
            result.add(v)

        dangling_violations = self.dangling_checker.check(routes, placement, parts_db, escapes)
        for v in dangling_violations:
            result.add(v)

        escape_violations = self.escape_checker.check(escapes, routes, parts_db)
        for v in escape_violations:
            result.add(v)

        mfg_violations = self.manufacturing_checker.check(routes)
        for v in mfg_violations:
            result.add(v)

        overlap_violations = self.overlap_checker.check(placement, parts_db)
        for v in overlap_violations:
            result.add(v)

        # Track overlap check
        track_overlap_violations = self.track_overlap_checker.check(routes, escapes)
        for v in track_overlap_violations:
            result.add(v)

        # Silkscreen check (NEW)
        silkscreen_violations = self.silkscreen_checker.check(placement, parts_db)
        for v in silkscreen_violations:
            result.add(v)

        result.valid = len(result.errors) == 0

        return result

    def get_report(self, result: ValidationResult = None) -> str:
        """Generate human-readable validation report"""
        if result is None:
            return "No validation performed"

        lines = [
            "=" * 60,
            "DRC VALIDATION REPORT",
            "=" * 60,
            "",
            f"Status: {'PASSED' if result.valid else 'FAILED'}",
            "",
            "SUMMARY:",
            f"  Errors:   {len(result.errors)}",
            f"  Warnings: {len(result.warnings)}",
            f"  Info:     {len(result.info)}",
            "",
            f"  Nets checked:     {result.nets_checked}",
            f"  Nets connected:   {result.nets_connected}",
            f"  Nets unconnected: {result.nets_unconnected}",
            "",
            f"  Clearance checks: {result.clearance_checks}",
            f"  Clearance passed: {result.clearance_passed}",
            "",
            f"  Total trace length: {result.total_trace_length:.1f}mm",
            f"  Total vias:         {result.total_vias}",
        ]

        if result.errors:
            lines.extend([
                "",
                "-" * 60,
                "ERRORS (must fix):",
                "-" * 60,
            ])
            for v in result.errors:
                lines.append(f"  • {v}")

        if result.warnings:
            lines.extend([
                "",
                "-" * 60,
                "WARNINGS (should fix):",
                "-" * 60,
            ])
            for v in result.warnings:
                lines.append(f"  • {v}")

        lines.append("=" * 60)

        return "\n".join(lines)
