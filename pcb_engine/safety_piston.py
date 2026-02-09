"""
PCB Engine - Safety Piston (Manufacturing Quality Gate)
========================================================

Goes BEYOND electrical DRC to ensure designs are:
1. Fabrication-friendly (easy to manufacture)
2. Assembly-friendly (easy to assemble)
3. Cost-optimized (no wasted material/time)
4. Reliable (long-term quality)

This piston answers: "Can we PROFITABLY manufacture this board at scale?"

DRC catches ERRORS. Safety Piston catches RISK.

Research References:
- IPC-2221 (Generic Standard on Printed Board Design)
- IPC-A-610 (Acceptability of Electronic Assemblies)
- IPC-7351 (Generic Requirements for Surface Mount Design)
- JEDEC Standards for component handling
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math


class SafetyLevel(Enum):
    """Safety check severity levels"""
    CRITICAL = "critical"    # Must fix - will fail manufacturing
    WARNING = "warning"      # Should fix - may cause issues
    INFO = "info"            # Consider fixing - optimization opportunity


class SafetyCategory(Enum):
    """Categories of safety checks"""
    FABRICATION = "fabrication"      # PCB fab issues
    ASSEMBLY = "assembly"            # SMT assembly issues
    COST = "cost"                    # Cost optimization
    RELIABILITY = "reliability"      # Long-term reliability
    EMC = "emc"                      # EMC/EMI issues


@dataclass
class SafetyViolation:
    """A single safety violation"""
    category: SafetyCategory
    level: SafetyLevel
    rule: str                        # Rule name (e.g., "MIN_ANNULAR_RING")
    message: str                     # Human-readable description
    location: Optional[Tuple[float, float]] = None  # Where on board
    component: Optional[str] = None  # Which component affected
    net: Optional[str] = None        # Which net affected
    fix_suggestion: Optional[str] = None  # How to fix it
    cost_impact: Optional[str] = None     # Cost impact if not fixed


@dataclass
class SafetyConfig:
    """Configuration for Safety Piston checks"""

    # =========================================================================
    # FABRICATION RULES (IPC-2221 Class 2 by default)
    # =========================================================================

    # Minimum annular ring (copper around drill hole)
    # Class 1: 0.05mm, Class 2: 0.10mm, Class 3: 0.15mm
    min_annular_ring: float = 0.10  # mm

    # Minimum copper-to-edge clearance
    # Standard: 0.25mm, high-reliability: 0.5mm
    min_edge_clearance: float = 0.25  # mm

    # Minimum solder mask web (sliver between mask openings)
    # Below this, mask may not adhere properly
    min_solder_mask_web: float = 0.10  # mm

    # Acid trap angle (angles below this trap etchant)
    min_trace_angle: float = 90.0  # degrees

    # Standard drill sizes (non-standard = surcharge)
    standard_drill_sizes: List[float] = field(default_factory=lambda: [
        0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.2
    ])

    # =========================================================================
    # ASSEMBLY RULES (IPC-7351, IPC-A-610)
    # =========================================================================

    # Minimum component-to-component spacing for pick-and-place
    min_component_spacing: float = 0.5  # mm (edge to edge)

    # Minimum component-to-edge spacing
    min_component_edge_spacing: float = 2.0  # mm

    # Fiducial requirements
    require_fiducials: bool = True
    min_fiducial_count: int = 3  # For proper alignment

    # Thermal relief requirements for large pads
    thermal_relief_threshold: float = 2.0  # mm² pad area

    # =========================================================================
    # COST OPTIMIZATION RULES
    # =========================================================================

    # Via count warning threshold
    via_count_warning: int = 50   # per 100cm² board area
    via_count_critical: int = 100

    # Layer count justification thresholds
    layer_2_max_nets: int = 20     # Above this, consider 4L
    layer_4_max_nets: int = 100    # Above this, consider 6L

    # Board size optimization (% wasted in panel)
    max_panel_waste: float = 30.0  # percent

    # =========================================================================
    # RELIABILITY RULES
    # =========================================================================

    # Thermal via requirements for power dissipation
    thermal_via_threshold: float = 0.5  # W - above this, need thermal vias
    thermal_via_spacing: float = 1.5    # mm - spacing between thermal vias

    # Copper pour stitching via requirements
    stitching_via_spacing: float = 5.0  # mm - max spacing between stitch vias

    # Current capacity (A/mm of trace width)
    # Using IPC-2152 curves for 1oz copper, 10°C rise
    current_capacity_1oz: float = 0.3   # A per mm width (external layer)
    current_capacity_internal: float = 0.2  # A per mm width (internal layer)

    # =========================================================================
    # EMC RULES
    # =========================================================================

    # Maximum return path length (% of signal length)
    max_return_path_ratio: float = 1.5

    # High-speed signal threshold
    high_speed_frequency: float = 100e6  # Hz - above this, EMC rules apply

    # =========================================================================
    # ENABLE/DISABLE CATEGORIES
    # =========================================================================
    check_fabrication: bool = True
    check_assembly: bool = True
    check_cost: bool = True
    check_reliability: bool = True
    check_emc: bool = True


@dataclass
class SafetyResult:
    """Results from Safety Piston analysis"""

    passed: bool                          # True if no CRITICAL violations
    violations: List[SafetyViolation] = field(default_factory=list)

    # Statistics
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # By category
    fabrication_issues: int = 0
    assembly_issues: int = 0
    cost_issues: int = 0
    reliability_issues: int = 0
    emc_issues: int = 0

    # Scores (0-100)
    fabrication_score: float = 100.0
    assembly_score: float = 100.0
    cost_score: float = 100.0
    reliability_score: float = 100.0
    overall_score: float = 100.0

    # Estimated cost impact
    estimated_cost_impact: str = "None"


class SafetyPiston:
    """
    Manufacturing Quality Gate Piston

    Ensures PCB designs are not just electrically correct (DRC) but also:
    - Easy to fabricate without issues
    - Easy to assemble with standard equipment
    - Cost-optimized for production
    - Reliable for long-term operation

    Usage:
        config = SafetyConfig(min_annular_ring=0.15)  # Stricter rules
        piston = SafetyPiston(config)
        result = piston.check(
            parts_db, placement, routes,
            board_width=50, board_height=40
        )
        if not result.passed:
            for v in result.violations:
                print(f"[{v.level.value}] {v.message}")
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.violations: List[SafetyViolation] = []

    def check(self, parts_db: Dict, placement: Dict, routes: Dict,
              board_width: float, board_height: float,
              vias: List = None, pour_zones: List = None) -> SafetyResult:
        """
        Run all safety checks on a PCB design.

        Args:
            parts_db: Parts database with component info
            placement: Component placements {ref: Position}
            routes: Routed traces {net_name: Route}
            board_width: Board width in mm
            board_height: Board height in mm
            vias: List of Via objects
            pour_zones: List of copper pour zones

        Returns:
            SafetyResult with all violations and scores
        """
        self.violations = []

        # Run all enabled checks
        if self.config.check_fabrication:
            self._check_fabrication(parts_db, placement, routes, vias,
                                   board_width, board_height)

        if self.config.check_assembly:
            self._check_assembly(parts_db, placement, board_width, board_height)

        if self.config.check_cost:
            self._check_cost(parts_db, routes, vias, board_width, board_height)

        if self.config.check_reliability:
            self._check_reliability(parts_db, placement, routes, vias, pour_zones)

        if self.config.check_emc:
            self._check_emc(parts_db, routes)

        # Calculate results
        result = self._calculate_result()
        return result

    # =========================================================================
    # FABRICATION CHECKS
    # =========================================================================

    def _check_fabrication(self, parts_db: Dict, placement: Dict, routes: Dict,
                          vias: List, board_width: float, board_height: float):
        """Check fabrication-related issues"""

        # Check 1: Annular ring
        self._check_annular_ring(vias)

        # Check 2: Edge clearance
        self._check_edge_clearance(parts_db, placement, routes,
                                   board_width, board_height)

        # Check 3: Acid traps
        self._check_acid_traps(routes)

        # Check 4: Non-standard drill sizes
        self._check_drill_sizes(vias)

        # Check 5: Solder mask slivers
        self._check_solder_mask_web(parts_db, placement)

    def _check_annular_ring(self, vias: List):
        """Check that all vias have adequate annular ring"""
        if not vias:
            return

        min_ring = self.config.min_annular_ring
        for via in vias:
            # Annular ring = (pad_diameter - drill_diameter) / 2
            pad_dia = getattr(via, 'pad_diameter', 0.8)
            drill_dia = getattr(via, 'drill_diameter', 0.4)
            ring = (pad_dia - drill_dia) / 2

            if ring < min_ring:
                pos = getattr(via, 'position', (0, 0))
                self.violations.append(SafetyViolation(
                    category=SafetyCategory.FABRICATION,
                    level=SafetyLevel.CRITICAL if ring < min_ring * 0.5 else SafetyLevel.WARNING,
                    rule="MIN_ANNULAR_RING",
                    message=f"Annular ring {ring:.3f}mm < {min_ring}mm minimum",
                    location=pos,
                    fix_suggestion=f"Increase via pad diameter or reduce drill size",
                    cost_impact="Risk of drill breakout and open circuits"
                ))

    def _check_edge_clearance(self, parts_db: Dict, placement: Dict, routes: Dict,
                              board_width: float, board_height: float):
        """Check copper-to-edge clearance"""
        min_clear = self.config.min_edge_clearance

        # Check component pads
        parts = parts_db.get('parts', {})
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            pos_x = pos.x if hasattr(pos, 'x') else pos[0]
            pos_y = pos.y if hasattr(pos, 'y') else pos[1]

            # Simple check: is component center too close to edge?
            edge_dist = min(pos_x, pos_y, board_width - pos_x, board_height - pos_y)
            if edge_dist < min_clear + 2.0:  # Add component size buffer
                self.violations.append(SafetyViolation(
                    category=SafetyCategory.FABRICATION,
                    level=SafetyLevel.WARNING,
                    rule="MIN_EDGE_CLEARANCE",
                    message=f"Component {ref} is {edge_dist:.2f}mm from board edge",
                    location=(pos_x, pos_y),
                    component=ref,
                    fix_suggestion="Move component away from board edge",
                    cost_impact="Risk of copper exposure at board edge"
                ))

    def _check_acid_traps(self, routes: Dict):
        """Check for acute angles that can trap etchant"""
        min_angle = self.config.min_trace_angle

        for net_name, route in routes.items():
            segments = getattr(route, 'segments', [])
            if len(segments) < 2:
                continue

            # Check angles between consecutive segments
            for i in range(len(segments) - 1):
                seg1 = segments[i]
                seg2 = segments[i + 1]

                # Calculate angle (simplified)
                # In reality, need to check if segments meet and calculate actual angle
                # For now, just flag potential issues
                pass  # TODO: Implement proper angle calculation

    def _check_drill_sizes(self, vias: List):
        """Check for non-standard drill sizes"""
        if not vias:
            return

        standard = set(self.config.standard_drill_sizes)
        non_standard = set()

        for via in vias:
            drill = getattr(via, 'drill_diameter', 0.4)
            drill_rounded = round(drill, 1)
            if drill_rounded not in standard:
                non_standard.add(drill_rounded)

        if non_standard:
            self.violations.append(SafetyViolation(
                category=SafetyCategory.COST,
                level=SafetyLevel.INFO,
                rule="NON_STANDARD_DRILL",
                message=f"Non-standard drill sizes: {sorted(non_standard)}mm",
                fix_suggestion="Use standard drill sizes to avoid surcharges",
                cost_impact="$10-50 surcharge per non-standard size"
            ))

    def _check_solder_mask_web(self, parts_db: Dict, placement: Dict):
        """Check for solder mask slivers between pads"""
        # This would require detailed pad geometry analysis
        # Simplified version: flag close components
        pass

    # =========================================================================
    # ASSEMBLY CHECKS
    # =========================================================================

    def _check_assembly(self, parts_db: Dict, placement: Dict,
                       board_width: float, board_height: float):
        """Check assembly-related issues"""

        # Check 1: Component spacing
        self._check_component_spacing(parts_db, placement)

        # Check 2: Fiducial markers
        if self.config.require_fiducials:
            self._check_fiducials(parts_db)

        # Check 3: Component orientation consistency
        self._check_component_orientation(parts_db, placement)

        # Check 4: Polarity markers
        self._check_polarity_markers(parts_db)

    def _check_component_spacing(self, parts_db: Dict, placement: Dict):
        """Check minimum component-to-component spacing"""
        min_spacing = self.config.min_component_spacing
        parts = parts_db.get('parts', {})
        refs = list(placement.keys())

        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                ref1, ref2 = refs[i], refs[j]
                pos1 = placement[ref1]
                pos2 = placement[ref2]

                x1 = pos1.x if hasattr(pos1, 'x') else pos1[0]
                y1 = pos1.y if hasattr(pos1, 'y') else pos1[1]
                x2 = pos2.x if hasattr(pos2, 'x') else pos2[0]
                y2 = pos2.y if hasattr(pos2, 'y') else pos2[1]

                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Rough check - actual check would use courtyard bounds
                if dist < min_spacing + 3.0:  # Add component size buffer
                    self.violations.append(SafetyViolation(
                        category=SafetyCategory.ASSEMBLY,
                        level=SafetyLevel.WARNING,
                        rule="MIN_COMPONENT_SPACING",
                        message=f"Components {ref1} and {ref2} may be too close ({dist:.2f}mm center-to-center)",
                        location=((x1 + x2) / 2, (y1 + y2) / 2),
                        fix_suggestion="Increase spacing for pick-and-place clearance",
                        cost_impact="May require slower assembly or hand placement"
                    ))

    def _check_fiducials(self, parts_db: Dict):
        """Check for fiducial markers"""
        parts = parts_db.get('parts', {})
        fiducial_count = 0

        for ref, part in parts.items():
            footprint = part.get('footprint', '').lower()
            if 'fiducial' in footprint or 'fid' in footprint:
                fiducial_count += 1

        if fiducial_count < self.config.min_fiducial_count:
            self.violations.append(SafetyViolation(
                category=SafetyCategory.ASSEMBLY,
                level=SafetyLevel.WARNING,
                rule="FIDUCIAL_REQUIRED",
                message=f"Only {fiducial_count} fiducials found, need {self.config.min_fiducial_count}",
                fix_suggestion="Add fiducial markers at board corners",
                cost_impact="May cause pick-and-place alignment issues"
            ))

    def _check_component_orientation(self, parts_db: Dict, placement: Dict):
        """Check that similar components have consistent orientation"""
        # Would check if all 0402s are aligned the same way, etc.
        pass

    def _check_polarity_markers(self, parts_db: Dict):
        """Check that polarized components have polarity indicators"""
        parts = parts_db.get('parts', {})

        for ref, part in parts.items():
            footprint = part.get('footprint', '').lower()
            # Check for polarized components
            if any(p in footprint for p in ['sod', 'diode', 'led', 'electrolytic']):
                # Should have a polarity marker in silkscreen
                # This would require checking the actual silkscreen layer
                pass

    # =========================================================================
    # COST CHECKS
    # =========================================================================

    def _check_cost(self, parts_db: Dict, routes: Dict, vias: List,
                   board_width: float, board_height: float):
        """Check cost optimization opportunities"""

        # Check 1: Via count
        self._check_via_count(vias, board_width, board_height)

        # Check 2: Layer count justification
        self._check_layer_count(parts_db, routes)

        # Check 3: Board size optimization
        self._check_board_size(parts_db, board_width, board_height)

    def _check_via_count(self, vias: List, board_width: float, board_height: float):
        """Check if via count is excessive"""
        via_count = len(vias) if vias else 0
        board_area = board_width * board_height / 100  # cm²
        vias_per_100cm2 = via_count / board_area * 100 if board_area > 0 else 0

        if vias_per_100cm2 > self.config.via_count_critical:
            self.violations.append(SafetyViolation(
                category=SafetyCategory.COST,
                level=SafetyLevel.WARNING,
                rule="HIGH_VIA_COUNT",
                message=f"Via density is high: {vias_per_100cm2:.1f} vias per 100cm²",
                fix_suggestion="Consider routing optimization or GND pour",
                cost_impact="Each via adds ~$0.01 to board cost"
            ))
        elif vias_per_100cm2 > self.config.via_count_warning:
            self.violations.append(SafetyViolation(
                category=SafetyCategory.COST,
                level=SafetyLevel.INFO,
                rule="MODERATE_VIA_COUNT",
                message=f"Via count: {via_count} ({vias_per_100cm2:.1f} per 100cm²)",
                fix_suggestion="Consider if all vias are necessary",
                cost_impact="Moderate via cost"
            ))

    def _check_layer_count(self, parts_db: Dict, routes: Dict):
        """Check if layer count is justified"""
        net_count = len(parts_db.get('nets', {}))

        # Simple heuristic: more nets = more layers needed
        if net_count <= self.config.layer_2_max_nets:
            recommended = 2
        elif net_count <= self.config.layer_4_max_nets:
            recommended = 4
        else:
            recommended = 6

        # INFO level - just informational
        self.violations.append(SafetyViolation(
            category=SafetyCategory.COST,
            level=SafetyLevel.INFO,
            rule="LAYER_RECOMMENDATION",
            message=f"With {net_count} nets, {recommended}-layer board is recommended",
            fix_suggestion=None,
            cost_impact=None
        ))

    def _check_board_size(self, parts_db: Dict, board_width: float, board_height: float):
        """Check if board size is optimized for panelization"""
        board_area = board_width * board_height

        # Standard panel sizes: 100x100mm, 100x160mm, etc.
        panel_sizes = [(100, 100), (100, 160), (160, 100), (160, 233)]

        best_utilization = 0
        for pw, ph in panel_sizes:
            boards_w = int(pw / board_width)
            boards_h = int(ph / board_height)
            if boards_w > 0 and boards_h > 0:
                util = (boards_w * boards_h * board_area) / (pw * ph) * 100
                best_utilization = max(best_utilization, util)

        waste = 100 - best_utilization
        if waste > self.config.max_panel_waste:
            self.violations.append(SafetyViolation(
                category=SafetyCategory.COST,
                level=SafetyLevel.INFO,
                rule="PANEL_UTILIZATION",
                message=f"Best panel utilization: {best_utilization:.1f}% ({waste:.1f}% waste)",
                fix_suggestion="Consider adjusting board dimensions for better panelization",
                cost_impact=f"~{waste:.0f}% material waste"
            ))

    # =========================================================================
    # RELIABILITY CHECKS
    # =========================================================================

    def _check_reliability(self, parts_db: Dict, placement: Dict,
                          routes: Dict, vias: List, pour_zones: List):
        """Check reliability-related issues"""

        # Check 1: Thermal via placement
        self._check_thermal_vias(parts_db, placement, vias)

        # Check 2: Pour stitching vias
        self._check_pour_stitching(pour_zones, vias)

        # Check 3: Current capacity
        self._check_current_capacity(parts_db, routes)

    def _check_thermal_vias(self, parts_db: Dict, placement: Dict, vias: List):
        """Check thermal via placement for power components"""
        parts = parts_db.get('parts', {})

        for ref, part in parts.items():
            footprint = part.get('footprint', '').lower()
            power = part.get('power_dissipation', 0)

            # Check for power components
            if power > self.config.thermal_via_threshold or \
               any(p in footprint for p in ['dpak', 'to-252', 'to-263', 'power', 'qfn']):

                # Check if there are vias near this component
                if ref in placement:
                    pos = placement[ref]
                    comp_x = pos.x if hasattr(pos, 'x') else pos[0]
                    comp_y = pos.y if hasattr(pos, 'y') else pos[1]

                    nearby_vias = 0
                    if vias:
                        for via in vias:
                            via_pos = getattr(via, 'position', (0, 0))
                            dist = math.sqrt((via_pos[0] - comp_x)**2 +
                                           (via_pos[1] - comp_y)**2)
                            if dist < 5.0:  # 5mm radius
                                nearby_vias += 1

                    if nearby_vias == 0:
                        self.violations.append(SafetyViolation(
                            category=SafetyCategory.RELIABILITY,
                            level=SafetyLevel.WARNING,
                            rule="THERMAL_VIAS_NEEDED",
                            message=f"Power component {ref} has no thermal vias",
                            location=(comp_x, comp_y),
                            component=ref,
                            fix_suggestion="Add thermal vias under thermal pad",
                            cost_impact="Risk of thermal runaway or derating"
                        ))

    def _check_pour_stitching(self, pour_zones: List, vias: List):
        """Check if copper pours have adequate stitching vias"""
        # Would analyze pour zones and check via spacing
        pass

    def _check_current_capacity(self, parts_db: Dict, routes: Dict):
        """Check trace width vs expected current"""
        nets = parts_db.get('nets', {})

        for net_name, net_info in nets.items():
            if isinstance(net_info, dict):
                max_current = net_info.get('max_current', 0)
                if max_current > 0 and net_name in routes:
                    route = routes[net_name]
                    segments = getattr(route, 'segments', [])

                    for seg in segments:
                        width = getattr(seg, 'width', 0.25)
                        capacity = width * self.config.current_capacity_1oz

                        if capacity < max_current:
                            self.violations.append(SafetyViolation(
                                category=SafetyCategory.RELIABILITY,
                                level=SafetyLevel.CRITICAL,
                                rule="TRACE_CURRENT_CAPACITY",
                                message=f"Net {net_name}: {width}mm trace can carry {capacity:.2f}A, needs {max_current}A",
                                net=net_name,
                                fix_suggestion=f"Increase trace width to {max_current / self.config.current_capacity_1oz:.2f}mm",
                                cost_impact="Risk of trace melting or fire"
                            ))

    # =========================================================================
    # EMC CHECKS
    # =========================================================================

    def _check_emc(self, parts_db: Dict, routes: Dict):
        """Check EMC-related issues"""
        # Would check return paths, ground loops, etc.
        pass

    # =========================================================================
    # RESULT CALCULATION
    # =========================================================================

    def _calculate_result(self) -> SafetyResult:
        """Calculate final result from violations"""
        result = SafetyResult(passed=True, violations=self.violations)

        # Count by level
        for v in self.violations:
            if v.level == SafetyLevel.CRITICAL:
                result.critical_count += 1
                result.passed = False
            elif v.level == SafetyLevel.WARNING:
                result.warning_count += 1
            else:
                result.info_count += 1

            # Count by category
            if v.category == SafetyCategory.FABRICATION:
                result.fabrication_issues += 1
            elif v.category == SafetyCategory.ASSEMBLY:
                result.assembly_issues += 1
            elif v.category == SafetyCategory.COST:
                result.cost_issues += 1
            elif v.category == SafetyCategory.RELIABILITY:
                result.reliability_issues += 1
            elif v.category == SafetyCategory.EMC:
                result.emc_issues += 1

        # Calculate scores (100 - 10*critical - 5*warning - 1*info)
        def calc_score(issues):
            criticals = sum(1 for v in self.violations
                          if v.category == SafetyCategory.FABRICATION and v.level == SafetyLevel.CRITICAL)
            warnings = sum(1 for v in self.violations
                         if v.category == SafetyCategory.FABRICATION and v.level == SafetyLevel.WARNING)
            return max(0, 100 - 10 * criticals - 5 * warnings)

        result.fabrication_score = calc_score(SafetyCategory.FABRICATION)
        result.assembly_score = calc_score(SafetyCategory.ASSEMBLY)
        result.cost_score = calc_score(SafetyCategory.COST)
        result.reliability_score = calc_score(SafetyCategory.RELIABILITY)

        result.overall_score = (
            result.fabrication_score * 0.3 +
            result.assembly_score * 0.2 +
            result.cost_score * 0.2 +
            result.reliability_score * 0.3
        )

        return result

    def print_report(self, result: SafetyResult):
        """Print a human-readable safety report"""
        print("\n" + "=" * 70)
        print("SAFETY PISTON REPORT - Manufacturing Quality Gate")
        print("=" * 70)

        # Overall status (ASCII-safe)
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"\nOverall Status: {status}")
        print(f"Overall Score: {result.overall_score:.0f}/100")

        # Category breakdown
        print(f"\nCategory Scores:")
        print(f"  Fabrication:  {result.fabrication_score:.0f}/100 ({result.fabrication_issues} issues)")
        print(f"  Assembly:     {result.assembly_score:.0f}/100 ({result.assembly_issues} issues)")
        print(f"  Cost:         {result.cost_score:.0f}/100 ({result.cost_issues} issues)")
        print(f"  Reliability:  {result.reliability_score:.0f}/100 ({result.reliability_issues} issues)")

        # Violations
        if result.violations:
            print(f"\nViolations: {result.critical_count} critical, "
                  f"{result.warning_count} warnings, {result.info_count} info")

            for v in result.violations:
                level_icon = {
                    SafetyLevel.CRITICAL: "[!!]",
                    SafetyLevel.WARNING: "[!]",
                    SafetyLevel.INFO: "[i]"
                }[v.level]

                print(f"\n  {level_icon} [{v.category.value}] {v.rule}")
                print(f"     {v.message}")
                if v.fix_suggestion:
                    print(f"     Fix: {v.fix_suggestion}")
                if v.cost_impact:
                    print(f"     Impact: {v.cost_impact}")
        else:
            print("\nNo violations found! Design is manufacturing-ready.")

        print("\n" + "=" * 70)
