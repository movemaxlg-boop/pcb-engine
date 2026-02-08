"""
FeasibilityPiston - Pre-flight Check for PCB Designs

This piston runs BEFORE all others in the BBL to determine if a design
is physically possible before wasting time on impossible layouts.

Checks performed:
1. Minimum board area calculation
2. Component density analysis
3. Layer requirement estimation
4. Net complexity analysis
5. Routing feasibility prediction

Part of the PCB Engine - The Foreman's first worker.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class FeasibilityStatus(Enum):
    """Overall feasibility determination"""
    FEASIBLE = "feasible"           # Design is achievable
    MARGINAL = "marginal"           # Achievable but tight
    INFEASIBLE = "infeasible"       # Cannot be achieved as specified
    NEEDS_CHANGES = "needs_changes" # Feasible with suggested changes


class Severity(Enum):
    """Issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class FeasibilityIssue:
    """A single feasibility issue or observation"""
    category: str
    message: str
    severity: Severity
    suggestion: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class FeasibilityResult:
    """Complete feasibility analysis result"""
    status: FeasibilityStatus
    issues: List[FeasibilityIssue] = field(default_factory=list)

    # Calculated metrics
    min_board_area_mm2: float = 0.0
    requested_board_area_mm2: float = 0.0
    component_density: float = 0.0  # 0-1 ratio

    # Recommendations
    min_board_width: float = 0.0
    min_board_height: float = 0.0
    recommended_layers: int = 2

    # Routing prediction
    estimated_routing_success: float = 0.0  # 0-1 probability
    net_complexity_score: float = 0.0

    @property
    def is_feasible(self) -> bool:
        return self.status in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.MARGINAL)

    def add_issue(self, category: str, message: str, severity: Severity,
                  suggestion: str = None, value: float = None, threshold: float = None):
        self.issues.append(FeasibilityIssue(
            category=category,
            message=message,
            severity=severity,
            suggestion=suggestion,
            value=value,
            threshold=threshold
        ))

    def get_issues_by_severity(self, severity: Severity) -> List[FeasibilityIssue]:
        return [i for i in self.issues if i.severity == severity]

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"Feasibility: {self.status.value.upper()}",
            f"Board Area: {self.requested_board_area_mm2:.1f}mm² (min required: {self.min_board_area_mm2:.1f}mm²)",
            f"Component Density: {self.component_density*100:.1f}%",
            f"Recommended Layers: {self.recommended_layers}",
            f"Estimated Routing Success: {self.estimated_routing_success*100:.0f}%",
        ]

        critical = self.get_issues_by_severity(Severity.CRITICAL)
        errors = self.get_issues_by_severity(Severity.ERROR)
        warnings = self.get_issues_by_severity(Severity.WARNING)

        if critical:
            lines.append(f"\nCRITICAL Issues ({len(critical)}):")
            for issue in critical:
                lines.append(f"  - {issue.message}")
                if issue.suggestion:
                    lines.append(f"    -> {issue.suggestion}")

        if errors:
            lines.append(f"\nErrors ({len(errors)}):")
            for issue in errors:
                lines.append(f"  - {issue.message}")
                if issue.suggestion:
                    lines.append(f"    -> {issue.suggestion}")

        if warnings:
            lines.append(f"\nWarnings ({len(warnings)}):")
            for issue in warnings[:5]:  # Show max 5 warnings
                lines.append(f"  - {issue.message}")

        return "\n".join(lines)


@dataclass
class FeasibilityConfig:
    """Configuration for feasibility analysis"""
    board_width: float = 50.0
    board_height: float = 40.0
    num_layers: int = 2

    # Spacing requirements (mm)
    component_spacing: float = 0.5      # Between component courtyards
    edge_clearance: float = 1.0         # Board edge clearance
    via_diameter: float = 0.6
    trace_width: float = 0.25
    clearance: float = 0.15

    # Thresholds
    max_density: float = 0.70           # Max 70% fill is reasonable
    critical_density: float = 0.85      # Above this is very difficult
    min_routing_channels: int = 2       # Minimum channels between components


# Standard footprint sizes (courtyard dimensions in mm)
FOOTPRINT_SIZES = {
    # Passives
    '0201': (0.8, 0.5),
    '0402': (1.5, 1.0),
    '0603': (2.1, 1.3),
    '0805': (2.5, 1.75),
    '1206': (3.7, 2.1),
    '1210': (3.7, 3.0),
    '2010': (5.7, 3.0),
    '2512': (7.0, 3.7),

    # Discretes
    'SOT-23': (3.4, 1.8),
    'SOT-23-5': (3.4, 1.8),
    'SOT-23-6': (3.4, 1.8),
    'SOT-223': (7.5, 4.0),
    'SOT-89': (5.0, 2.5),
    'TO-252': (7.5, 7.5),
    'TO-263': (11.0, 10.0),

    # ICs
    'SOIC-8': (6.0, 5.0),
    'SOIC-14': (9.0, 5.0),
    'SOIC-16': (10.5, 5.0),
    'SSOP-20': (8.0, 4.0),
    'TSSOP-14': (5.5, 3.5),
    'TSSOP-16': (6.0, 3.5),
    'TSSOP-20': (7.0, 3.5),
    'QFP-32': (9.0, 9.0),
    'QFP-44': (12.0, 12.0),
    'QFP-64': (14.0, 14.0),
    'QFP-100': (16.0, 16.0),
    'QFN-16': (4.0, 4.0),
    'QFN-20': (5.0, 5.0),
    'QFN-24': (5.0, 5.0),
    'QFN-32': (6.0, 6.0),
    'QFN-48': (8.0, 8.0),
    'QFN-64': (10.0, 10.0),

    # Connectors
    'USB-C': (10.0, 8.0),
    'USB-MICRO': (8.0, 6.0),
    'JST-SH-2': (5.0, 4.0),
    'JST-SH-4': (7.0, 4.0),
    'JST-PH-2': (6.5, 5.0),

    # Modules
    'ESP32-WROOM': (26.0, 18.5),
    'ESP32-S3-WROOM': (26.0, 18.5),
    'ESP32-C3-MINI': (14.0, 13.0),

    # Default for unknown
    'DEFAULT': (3.0, 3.0),
}


class FeasibilityPiston:
    """
    Feasibility Piston - Pre-flight Check

    Runs BEFORE Parts/Placement to determine if a design is achievable.
    Part of the PCB Engine (The Foreman).

    Usage:
        piston = FeasibilityPiston(config)
        result = piston.analyze(parts_db)

        if not result.is_feasible:
            print(result.summary())
            # Escalate to Engineer/User
    """

    def __init__(self, config: FeasibilityConfig = None):
        self.config = config or FeasibilityConfig()

    def analyze(self, parts_db: Dict) -> FeasibilityResult:
        """
        Perform complete feasibility analysis.

        Args:
            parts_db: Parts database with 'parts' and 'nets' sections

        Returns:
            FeasibilityResult with status and detailed analysis
        """
        result = FeasibilityResult(status=FeasibilityStatus.FEASIBLE)

        # Calculate board area
        result.requested_board_area_mm2 = self.config.board_width * self.config.board_height

        # Run all checks
        self._check_component_area(parts_db, result)
        self._check_component_density(parts_db, result)
        self._check_net_complexity(parts_db, result)
        self._check_layer_requirements(parts_db, result)
        self._estimate_routing_success(parts_db, result)

        # Determine final status based on issues
        result.status = self._determine_status(result)

        return result

    def _get_footprint_size(self, footprint: str) -> Tuple[float, float]:
        """Get courtyard size for a footprint"""
        # Try exact match
        if footprint in FOOTPRINT_SIZES:
            return FOOTPRINT_SIZES[footprint]

        # Try partial match (e.g., "0805" in "C_0805")
        for key, size in FOOTPRINT_SIZES.items():
            if key in footprint.upper():
                return size

        # Default
        return FOOTPRINT_SIZES['DEFAULT']

    def _check_component_area(self, parts_db: Dict, result: FeasibilityResult):
        """Calculate minimum board area needed for all components"""
        parts = parts_db.get('parts', {})

        total_area = 0.0
        component_areas = []

        for ref, part in parts.items():
            footprint = part.get('footprint', 'DEFAULT')
            w, h = self._get_footprint_size(footprint)

            # Add spacing around each component
            w += self.config.component_spacing * 2
            h += self.config.component_spacing * 2

            area = w * h
            total_area += area
            component_areas.append((ref, footprint, area))

        # Add edge clearance to effective board area
        effective_width = self.config.board_width - (self.config.edge_clearance * 2)
        effective_height = self.config.board_height - (self.config.edge_clearance * 2)
        effective_area = effective_width * effective_height

        result.min_board_area_mm2 = total_area

        # Calculate minimum board dimensions (assuming square-ish layout)
        aspect_ratio = self.config.board_width / self.config.board_height
        min_side = math.sqrt(total_area / aspect_ratio)
        result.min_board_width = min_side * math.sqrt(aspect_ratio) + self.config.edge_clearance * 2
        result.min_board_height = min_side / math.sqrt(aspect_ratio) + self.config.edge_clearance * 2

        if total_area > effective_area:
            deficit = total_area - effective_area
            result.add_issue(
                category="board_size",
                message=f"Components require {total_area:.1f}mm² but board only has {effective_area:.1f}mm² usable",
                severity=Severity.CRITICAL,
                suggestion=f"Increase board to at least {result.min_board_width:.1f}mm x {result.min_board_height:.1f}mm",
                value=total_area,
                threshold=effective_area
            )
        elif total_area > effective_area * 0.8:
            result.add_issue(
                category="board_size",
                message=f"Component area ({total_area:.1f}mm²) is close to board capacity ({effective_area:.1f}mm²)",
                severity=Severity.WARNING,
                suggestion="Consider a slightly larger board for easier routing"
            )

    def _check_component_density(self, parts_db: Dict, result: FeasibilityResult):
        """Check if component density is within acceptable limits"""
        effective_area = (self.config.board_width - self.config.edge_clearance * 2) * \
                        (self.config.board_height - self.config.edge_clearance * 2)

        if effective_area > 0:
            density = result.min_board_area_mm2 / effective_area
        else:
            density = 1.0

        result.component_density = min(density, 1.0)

        if density > self.config.critical_density:
            result.add_issue(
                category="density",
                message=f"Component density {density*100:.1f}% exceeds critical threshold {self.config.critical_density*100:.0f}%",
                severity=Severity.CRITICAL,
                suggestion="Use smaller packages (0402 instead of 0603) or increase board size",
                value=density,
                threshold=self.config.critical_density
            )
        elif density > self.config.max_density:
            result.add_issue(
                category="density",
                message=f"Component density {density*100:.1f}% is high (target: <{self.config.max_density*100:.0f}%)",
                severity=Severity.WARNING,
                suggestion="Routing may be difficult; consider increasing board size by 20%",
                value=density,
                threshold=self.config.max_density
            )
        else:
            result.add_issue(
                category="density",
                message=f"Component density {density*100:.1f}% is acceptable",
                severity=Severity.INFO
            )

    def _check_net_complexity(self, parts_db: Dict, result: FeasibilityResult):
        """Analyze net complexity and potential routing challenges"""
        nets = parts_db.get('nets', {})
        parts = parts_db.get('parts', {})

        total_pins = 0
        max_net_size = 0
        power_nets = []
        high_fanout_nets = []

        for net_name, net_info in nets.items():
            pins = net_info.get('pins', [])
            pin_count = len(pins)
            total_pins += pin_count

            if pin_count > max_net_size:
                max_net_size = pin_count

            # Identify power/ground nets (typically high fanout)
            if net_name.upper() in ('GND', 'VCC', 'VDD', '3V3', '5V', '12V', 'VBAT', 'VSYS'):
                power_nets.append((net_name, pin_count))

            # High fanout nets are harder to route
            if pin_count > 6:
                high_fanout_nets.append((net_name, pin_count))

        # Calculate complexity score (0-1)
        # Based on: total pins, max fanout, number of nets
        num_nets = len(nets)
        complexity = 0.0

        if num_nets > 0:
            avg_fanout = total_pins / num_nets
            complexity = min(1.0, (avg_fanout / 10) * 0.3 +
                           (max_net_size / 20) * 0.3 +
                           (num_nets / 50) * 0.4)

        result.net_complexity_score = complexity

        if complexity > 0.8:
            result.add_issue(
                category="complexity",
                message=f"High net complexity (score: {complexity:.2f})",
                severity=Severity.WARNING,
                suggestion="Consider using ground pour to simplify GND routing"
            )

        # Check for power nets that might benefit from a plane
        for net_name, pin_count in power_nets:
            if pin_count > 4:
                result.add_issue(
                    category="power",
                    message=f"Power net '{net_name}' has {pin_count} connections",
                    severity=Severity.INFO,
                    suggestion=f"Consider using a copper pour for {net_name}"
                )

        # Warn about high fanout nets
        for net_name, pin_count in high_fanout_nets:
            if net_name.upper() not in ('GND', 'VCC', 'VDD', '3V3', '5V'):
                result.add_issue(
                    category="fanout",
                    message=f"Signal net '{net_name}' has high fanout ({pin_count} pins)",
                    severity=Severity.WARNING,
                    suggestion="May require careful placement or multiple vias"
                )

    def _check_layer_requirements(self, parts_db: Dict, result: FeasibilityResult):
        """Estimate number of layers needed"""
        nets = parts_db.get('nets', {})
        parts = parts_db.get('parts', {})

        # Count routing requirements
        num_nets = len([n for n, info in nets.items() if len(info.get('pins', [])) >= 2])
        num_parts = len(parts)

        # Check for BGA/QFN packages that typically need more layers
        dense_packages = 0
        for ref, part in parts.items():
            footprint = part.get('footprint', '').upper()
            if any(pkg in footprint for pkg in ['BGA', 'QFN-48', 'QFN-64', 'QFP-100']):
                dense_packages += 1

        # Estimate layers needed
        # Simple heuristic: 2 layers handle ~30 nets at moderate density
        # 4 layers for >50 nets or dense packages
        # 6 layers for >100 nets or multiple BGAs

        if dense_packages > 0 or num_nets > 100:
            recommended = 6
        elif num_nets > 50 or result.component_density > 0.6:
            recommended = 4
        else:
            recommended = 2

        result.recommended_layers = recommended

        if recommended > self.config.num_layers:
            severity = Severity.ERROR if recommended > self.config.num_layers + 2 else Severity.WARNING
            result.add_issue(
                category="layers",
                message=f"Design may need {recommended} layers (configured: {self.config.num_layers})",
                severity=severity,
                suggestion=f"Consider using a {recommended}-layer stackup for this design"
            )
        else:
            result.add_issue(
                category="layers",
                message=f"{self.config.num_layers} layers should be sufficient",
                severity=Severity.INFO
            )

    def _estimate_routing_success(self, parts_db: Dict, result: FeasibilityResult):
        """Estimate probability of successful routing"""
        # Factors that affect routing success:
        # 1. Component density (higher = worse)
        # 2. Net complexity (higher = worse)
        # 3. Layer availability (more = better)
        # 4. Board area margin (more = better)

        # Start with base probability
        success_prob = 1.0

        # Density impact (exponential penalty above 60%)
        if result.component_density > 0.6:
            density_penalty = (result.component_density - 0.6) ** 2 * 2
            success_prob *= max(0.1, 1.0 - density_penalty)

        # Complexity impact
        success_prob *= max(0.3, 1.0 - result.net_complexity_score * 0.5)

        # Layer bonus
        if self.config.num_layers >= result.recommended_layers:
            success_prob *= 1.0
        elif self.config.num_layers == result.recommended_layers - 2:
            success_prob *= 0.7
        else:
            success_prob *= 0.4

        # Area margin bonus
        if result.requested_board_area_mm2 > result.min_board_area_mm2 * 1.5:
            success_prob *= 1.1  # Extra space helps
        elif result.requested_board_area_mm2 < result.min_board_area_mm2:
            success_prob *= 0.3  # Not enough space

        result.estimated_routing_success = min(1.0, max(0.0, success_prob))

        if result.estimated_routing_success < 0.3:
            result.add_issue(
                category="routing",
                message=f"Low routing success probability: {result.estimated_routing_success*100:.0f}%",
                severity=Severity.CRITICAL,
                suggestion="Design changes strongly recommended before proceeding"
            )
        elif result.estimated_routing_success < 0.6:
            result.add_issue(
                category="routing",
                message=f"Moderate routing success probability: {result.estimated_routing_success*100:.0f}%",
                severity=Severity.WARNING,
                suggestion="Routing may require multiple attempts or manual intervention"
            )
        else:
            result.add_issue(
                category="routing",
                message=f"Good routing success probability: {result.estimated_routing_success*100:.0f}%",
                severity=Severity.INFO
            )

    def _determine_status(self, result: FeasibilityResult) -> FeasibilityStatus:
        """Determine overall status based on issues found"""
        critical = len(result.get_issues_by_severity(Severity.CRITICAL))
        errors = len(result.get_issues_by_severity(Severity.ERROR))
        warnings = len(result.get_issues_by_severity(Severity.WARNING))

        if critical > 0:
            return FeasibilityStatus.INFEASIBLE
        elif errors > 0:
            return FeasibilityStatus.NEEDS_CHANGES
        elif warnings > 2:
            return FeasibilityStatus.MARGINAL
        else:
            return FeasibilityStatus.FEASIBLE


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_feasibility(parts_db: Dict, board_width: float, board_height: float,
                      num_layers: int = 2) -> FeasibilityResult:
    """
    Quick feasibility check.

    Args:
        parts_db: Parts database
        board_width: Board width in mm
        board_height: Board height in mm
        num_layers: Number of copper layers

    Returns:
        FeasibilityResult
    """
    config = FeasibilityConfig(
        board_width=board_width,
        board_height=board_height,
        num_layers=num_layers
    )
    piston = FeasibilityPiston(config)
    return piston.analyze(parts_db)


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    # Test with a sample design
    test_parts_db = {
        'parts': {
            'U1': {'footprint': 'ESP32-WROOM'},
            'U2': {'footprint': 'QFN-32'},
            'U3': {'footprint': 'SOIC-8'},
            'C1': {'footprint': '0805'},
            'C2': {'footprint': '0805'},
            'C3': {'footprint': '0402'},
            'C4': {'footprint': '0402'},
            'R1': {'footprint': '0603'},
            'R2': {'footprint': '0603'},
            'R3': {'footprint': '0402'},
            'R4': {'footprint': '0402'},
            'D1': {'footprint': '0603'},
            'D2': {'footprint': '0603'},
        },
        'nets': {
            'GND': {'pins': [('U1', '1'), ('U2', '1'), ('U3', '4'), ('C1', '2'), ('C2', '2'), ('C3', '2'), ('C4', '2')]},
            '3V3': {'pins': [('U1', '2'), ('U2', '2'), ('U3', '8'), ('C1', '1'), ('C2', '1')]},
            '5V': {'pins': [('U1', '3'), ('C3', '1'), ('C4', '1')]},
            'SDA': {'pins': [('U1', '21'), ('U2', '3'), ('R1', '1')]},
            'SCL': {'pins': [('U1', '22'), ('U2', '4'), ('R2', '1')]},
            'LED1': {'pins': [('U1', '5'), ('R3', '1')]},
            'LED2': {'pins': [('U1', '6'), ('R4', '1')]},
            'LED1_A': {'pins': [('R3', '2'), ('D1', '1')]},
            'LED2_A': {'pins': [('R4', '2'), ('D2', '1')]},
        }
    }

    print("=" * 60)
    print("FEASIBILITY PISTON TEST")
    print("=" * 60)

    # Test with a small board (should fail)
    print("\n--- Test 1: Small board (30x25mm) ---")
    result = check_feasibility(test_parts_db, 30, 25, 2)
    print(result.summary())

    # Test with adequate board
    print("\n--- Test 2: Adequate board (50x40mm) ---")
    result = check_feasibility(test_parts_db, 50, 40, 2)
    print(result.summary())

    # Test with large board
    print("\n--- Test 3: Large board (80x60mm) ---")
    result = check_feasibility(test_parts_db, 80, 60, 2)
    print(result.summary())
