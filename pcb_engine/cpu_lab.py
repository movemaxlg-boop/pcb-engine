"""
CPU LAB - VLSI-Inspired PCB Design Enhancement
================================================

The CPU Lab sits between c_layout and BBL in the design pipeline.
It applies strategies borrowed from CPU/VLSI chip design to dramatically
improve PCB routing success on 2-layer boards.

PIPELINE:
  User Order -> AI Department -> c_layout -> CPU LAB -> BBL -> PCB Files

CPU LAB STRATEGIES (6 total):
  1. Power Grid Planner     - GND pour before routing (CRITICAL)
  2. Layer Direction Assign  - H/V layer preference (HIGH IMPACT)
  3. Global Router          - Coarse path planning before detail routing
  4. Congestion Estimator   - Detect routing bottlenecks early
  5. Net Priority Ordering  - Timing-driven net ordering
  6. Component Group Templates - Pre-verified placement groups

CONTINUOUS DRC MONITOR:
  Unlike end-of-line DRC, the CPU Lab provides a FactoryInspector that
  monitors quality at EVERY stage of the BBL, not just at the end.

Research Sources:
  - McMurchie & Ebeling 1995 (PathFinder negotiated congestion)
  - Kahng et al. "VLSI Physical Design" (hierarchical routing)
  - KiCad ground plane best practices
  - Cadence NanoRoute global/detail routing flow
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from enum import Enum
import math
import time


# =============================================================================
# ENUMS
# =============================================================================

class LayerDirection(Enum):
    """Preferred routing direction for a layer."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    ANY = "any"


class GndStrategy(Enum):
    """How to handle GND net."""
    POUR = "pour"           # Copper pour on bottom layer (preferred for 2-layer)
    TRACES = "traces"       # Route as traces (only for very few GND pins)
    DEDICATED_LAYER = "dedicated_layer"  # Full GND layer (4+ layers)


class PowerStrategy(Enum):
    """How to handle power rails."""
    WIDE_TRACES = "wide_traces"     # Route as wide traces
    POUR = "pour"                   # Copper pour on a layer
    STAR_TOPOLOGY = "star"          # Star from regulator output
    TRUNK_AND_BRANCH = "trunk"      # Thick trunk, thin branches


class CongestionLevel(Enum):
    """Routing congestion level for a region."""
    LOW = "low"         # < 30% utilization
    MEDIUM = "medium"   # 30-60% utilization
    HIGH = "high"       # 60-80% utilization
    CRITICAL = "critical"  # > 80% utilization


class InspectionStage(Enum):
    """Stages where factory inspector checks quality."""
    POST_PLACEMENT = "post_placement"
    POST_POUR = "post_pour"
    DURING_ROUTING = "during_routing"
    POST_ROUTING = "post_routing"
    POST_OPTIMIZE = "post_optimize"
    FINAL = "final"


class InspectionSeverity(Enum):
    """Severity of inspection findings."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"  # Stop the line


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LayerAssignment:
    """Layer direction assignment for routing."""
    layer_name: str           # "F.Cu", "B.Cu"
    layer_index: int          # 0, 1, ...
    preferred_direction: LayerDirection
    direction_cost_penalty: float = 3.0  # Cost multiplier for wrong direction
    description: str = ""


@dataclass
class PowerGridPlan:
    """Plan for power distribution."""
    gnd_strategy: GndStrategy
    gnd_pour_layer: str = "B.Cu"
    gnd_pour_config: Dict[str, Any] = field(default_factory=dict)

    power_strategies: Dict[str, PowerStrategy] = field(default_factory=dict)
    # net_name -> strategy

    power_trace_widths: Dict[str, float] = field(default_factory=dict)
    # net_name -> trace width in mm

    nets_removed_from_routing: List[str] = field(default_factory=list)
    # Nets handled by pour (removed from router's job)

    reasoning: List[str] = field(default_factory=list)


@dataclass
class GlobalRoute:
    """A coarse routing plan for a net (region-level, not exact path)."""
    net_name: str
    regions: List[Tuple[int, int]]  # List of (region_row, region_col) the net passes through
    estimated_length_mm: float = 0.0
    estimated_vias: int = 0
    preferred_layer: Optional[str] = None
    congestion_risk: CongestionLevel = CongestionLevel.LOW


@dataclass
class GlobalRoutingPlan:
    """Complete global routing plan for all nets."""
    region_grid_rows: int
    region_grid_cols: int
    region_size_mm: float           # Size of each region in mm
    net_routes: Dict[str, GlobalRoute] = field(default_factory=dict)
    congestion_map: List[List[float]] = field(default_factory=list)
    # 2D array of congestion values per region
    hotspots: List[Tuple[int, int, float]] = field(default_factory=list)
    # (row, col, utilization) for congested regions


@dataclass
class CongestionAnalysis:
    """Result of congestion estimation."""
    overall_level: CongestionLevel
    overall_utilization: float  # 0.0 to 1.0
    region_utilization: List[List[float]] = field(default_factory=list)
    hotspot_regions: List[Tuple[int, int, float]] = field(default_factory=list)
    bottleneck_nets: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class NetPriority:
    """Priority assignment for a net."""
    net_name: str
    priority: int               # Lower = route first (0 = highest)
    net_type: str               # "power", "ground", "differential", "i2c", "signal"
    trace_width_mm: float = 0.25
    clearance_mm: float = 0.15
    preferred_layer: Optional[str] = None
    must_route_with: Optional[str] = None  # For diff pairs
    max_length_mm: Optional[float] = None
    reasoning: str = ""


@dataclass
class ComponentGroup:
    """A pre-verified placement group (like a standard cell in VLSI)."""
    name: str                   # "LDO_CIRCUIT", "I2C_PULLUPS"
    components: List[str]       # ["U2", "C4", "C5"]
    anchor: str                 # Main component ("U2")
    relative_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # component -> (dx, dy) relative to anchor
    internal_nets: List[str] = field(default_factory=list)
    # Nets that are internal to this group
    description: str = ""


@dataclass
class InspectionFinding:
    """A finding from the factory inspector."""
    stage: InspectionStage
    severity: InspectionSeverity
    code: str               # "PLACEMENT_OVERLAP", "ROUTING_CONGESTION", etc.
    message: str
    location: Optional[Tuple[float, float]] = None  # (x, y) if location-specific
    affected_nets: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    suggestion: str = ""
    timestamp: float = 0.0


@dataclass
class InspectionReport:
    """Cumulative inspection report from the factory inspector."""
    stage: InspectionStage
    passed: bool
    findings: List[InspectionFinding] = field(default_factory=list)
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    duration_ms: float = 0.0

    @property
    def stop_the_line(self) -> bool:
        """Should we stop production?"""
        return self.critical_count > 0


@dataclass
class CPULabResult:
    """Complete output from the CPU Lab stage."""
    # Power grid decisions
    power_grid: PowerGridPlan

    # Layer assignments
    layer_assignments: List[LayerAssignment]

    # Global routing plan
    global_routing: GlobalRoutingPlan

    # Congestion analysis
    congestion: CongestionAnalysis

    # Net priority ordering
    net_priorities: List[NetPriority]

    # Component groups
    component_groups: List[ComponentGroup]

    # Enhanced parts_db (with CPU Lab decisions injected)
    enhanced_parts_db: Dict[str, Any] = field(default_factory=dict)

    # Processing time
    processing_time_ms: float = 0.0

    # Summary
    summary: List[str] = field(default_factory=list)


# =============================================================================
# POWER GRID PLANNER
# =============================================================================

class PowerGridPlanner:
    """
    Strategy 1: Power Grid Planning (from VLSI power distribution)

    In CPU design, power (VDD/GND) is handled FIRST on dedicated metal layers.
    Signals are routed AFTER power is solved.

    For 2-layer PCBs:
    - GND: Copper pour on bottom layer (removes GND from routing entirely)
    - Power rails: Wide traces with star topology from regulator

    For 4+ layer PCBs:
    - Dedicated GND layer (full plane)
    - Dedicated power layer (split planes if multiple rails)
    """

    def plan(self, parts_db: Dict, board_config: Dict) -> PowerGridPlan:
        """Analyze the design and decide power distribution strategy."""

        nets = parts_db.get('nets', {})
        parts = parts_db.get('parts', {})
        layer_count = board_config.get('layers', board_config.get('layer_count', 2))

        # Count GND pins
        gnd_net = self._find_gnd_net(nets)
        gnd_pin_count = len(nets.get(gnd_net, {}).get('pins', [])) if gnd_net else 0

        # Count power nets and their pins
        power_nets = self._find_power_nets(nets)

        # Decide GND strategy
        gnd_strategy, gnd_reasoning = self._decide_gnd_strategy(
            gnd_pin_count, layer_count, len(parts)
        )

        # Decide power strategy for each rail
        power_strategies = {}
        power_widths = {}
        for net_name, net_info in power_nets.items():
            pin_count = len(net_info.get('pins', []))
            strategy, width = self._decide_power_strategy(
                net_name, pin_count, layer_count
            )
            power_strategies[net_name] = strategy
            power_widths[net_name] = width

        # Determine which nets are removed from routing
        removed = []
        reasoning = list(gnd_reasoning)

        if gnd_strategy in (GndStrategy.POUR, GndStrategy.DEDICATED_LAYER):
            removed.append(gnd_net)
            reasoning.append(
                f"GND ({gnd_pin_count} pins) handled by {'pour' if gnd_strategy == GndStrategy.POUR else 'dedicated layer'} "
                f"- removed from routing queue"
            )

        # Pour config
        pour_config = {}
        if gnd_strategy == GndStrategy.POUR:
            pour_config = {
                'net': gnd_net,
                'layer': 'B.Cu',
                'clearance': 0.3,
                'thermal_relief': 'thermal',
                'thermal_spoke_width': 0.5,
                'thermal_gap': 0.5,
                'add_stitching_vias': True,
                'stitching_via_spacing': 8.0,
            }

        return PowerGridPlan(
            gnd_strategy=gnd_strategy,
            gnd_pour_layer='B.Cu',
            gnd_pour_config=pour_config,
            power_strategies=power_strategies,
            power_trace_widths=power_widths,
            nets_removed_from_routing=removed,
            reasoning=reasoning,
        )

    def _find_gnd_net(self, nets: Dict) -> str:
        """Find the GND net name."""
        for name, info in nets.items():
            net_class = info.get('class', info.get('type', '')).lower()
            if net_class == 'ground' or name.upper() == 'GND':
                return name
        return 'GND'

    def _find_power_nets(self, nets: Dict) -> Dict:
        """Find all power nets (excluding GND)."""
        power = {}
        for name, info in nets.items():
            net_class = info.get('class', info.get('type', '')).lower()
            if net_class == 'power' or name.upper() in ('VCC', 'VDD', '3V3', '5V', 'VBUS', '3.3V', '5V0', '12V'):
                power[name] = info
        return power

    def _decide_gnd_strategy(self, gnd_pins: int, layers: int, num_parts: int
                             ) -> Tuple[GndStrategy, List[str]]:
        """Decide GND distribution strategy."""
        reasoning = []

        if layers >= 4:
            reasoning.append(f"4+ layers ({layers}): dedicated GND layer (best EMI performance)")
            return GndStrategy.DEDICATED_LAYER, reasoning

        if gnd_pins >= 5:
            reasoning.append(
                f"GND has {gnd_pins} pins on 2-layer board: "
                f"copper pour is MANDATORY (routing {gnd_pins} GND traces is impractical)"
            )
            return GndStrategy.POUR, reasoning

        if gnd_pins >= 3:
            reasoning.append(
                f"GND has {gnd_pins} pins: pour recommended for cleaner layout"
            )
            return GndStrategy.POUR, reasoning

        reasoning.append(f"GND has only {gnd_pins} pins: can route as traces")
        return GndStrategy.TRACES, reasoning

    def _decide_power_strategy(self, net_name: str, pin_count: int, layers: int
                               ) -> Tuple[PowerStrategy, float]:
        """Decide strategy and trace width for a power net."""
        if pin_count >= 8:
            return PowerStrategy.TRUNK_AND_BRANCH, 0.5  # Thick trunk
        elif pin_count >= 4:
            return PowerStrategy.STAR_TOPOLOGY, 0.4  # Star from source
        else:
            return PowerStrategy.WIDE_TRACES, 0.35  # Simple wide traces


# =============================================================================
# LAYER DIRECTION ASSIGNER
# =============================================================================

class LayerDirectionAssigner:
    """
    Strategy 2: Layer Direction Assignment (from VLSI metal layer rules)

    In CPU design, each metal layer routes in ONE direction only:
      M1: Vertical, M2: Horizontal, M3: Vertical, ...

    This eliminates crossing conflicts. For 2-layer PCBs:
      F.Cu (top):    Prefer HORIZONTAL traces
      B.Cu (bottom): Prefer VERTICAL traces

    The router adds a cost penalty for going against the preferred direction,
    encouraging orderly routing and reducing congestion.
    """

    def assign(self, layer_count: int, board_width: float, board_height: float
               ) -> List[LayerAssignment]:
        """Assign preferred directions to layers."""

        assignments = []

        if layer_count == 1:
            assignments.append(LayerAssignment(
                layer_name="F.Cu", layer_index=0,
                preferred_direction=LayerDirection.ANY,
                direction_cost_penalty=1.0,
                description="Single layer - no direction preference"
            ))

        elif layer_count == 2:
            # 2-layer: top=horizontal, bottom=vertical
            # Choose based on board aspect ratio for efficiency
            if board_width >= board_height:
                # Wider board: top=H (more horizontal runs), bottom=V
                top_dir, bot_dir = LayerDirection.HORIZONTAL, LayerDirection.VERTICAL
            else:
                # Taller board: top=V, bottom=H
                top_dir, bot_dir = LayerDirection.VERTICAL, LayerDirection.HORIZONTAL

            assignments.append(LayerAssignment(
                layer_name="F.Cu", layer_index=0,
                preferred_direction=top_dir,
                direction_cost_penalty=3.0,
                description=f"Top copper: prefer {top_dir.value} (board is {'wider' if board_width >= board_height else 'taller'})"
            ))
            assignments.append(LayerAssignment(
                layer_name="B.Cu", layer_index=1,
                preferred_direction=bot_dir,
                direction_cost_penalty=3.0,
                description=f"Bottom copper: prefer {bot_dir.value} (orthogonal to top)"
            ))

        elif layer_count == 4:
            # 4-layer: Signal-GND-Power-Signal
            assignments = [
                LayerAssignment("F.Cu", 0, LayerDirection.HORIZONTAL, 3.0,
                                "Top signal: horizontal"),
                LayerAssignment("In1.Cu", 1, LayerDirection.ANY, 1.0,
                                "GND plane (no routing)"),
                LayerAssignment("In2.Cu", 2, LayerDirection.ANY, 1.0,
                                "Power plane (no routing)"),
                LayerAssignment("B.Cu", 3, LayerDirection.VERTICAL, 3.0,
                                "Bottom signal: vertical"),
            ]

        else:
            # 6+ layers: alternate H/V
            for i in range(layer_count):
                direction = LayerDirection.HORIZONTAL if i % 2 == 0 else LayerDirection.VERTICAL
                name = "F.Cu" if i == 0 else f"In{i}.Cu" if i < layer_count - 1 else "B.Cu"
                assignments.append(LayerAssignment(
                    name, i, direction, 3.0,
                    f"Layer {i}: {direction.value}"
                ))

        return assignments


# =============================================================================
# GLOBAL ROUTER
# =============================================================================

class GlobalRouter:
    """
    Strategy 3: Global Routing (from VLSI hierarchical routing)

    In CPU design, routing happens in two stages:
      1. GLOBAL: Divide chip into regions, plan which regions each net crosses
      2. DETAILED: Route exact paths within each region

    We apply this to PCBs:
      1. Divide board into NxM regions (e.g., 5mm x 5mm each)
      2. For each net, plan which regions it must pass through
      3. Estimate congestion per region
      4. Feed this plan to the detailed router as hints

    This catches congestion problems BEFORE wasting time on detailed routing.
    """

    def __init__(self, region_size_mm: float = 5.0):
        self.region_size = region_size_mm

    def plan(self, parts_db: Dict, placement: Dict,
             board_width: float, board_height: float,
             power_plan: PowerGridPlan) -> GlobalRoutingPlan:
        """Create a global routing plan."""

        # Calculate region grid
        cols = max(1, int(math.ceil(board_width / self.region_size)))
        rows = max(1, int(math.ceil(board_height / self.region_size)))

        # Initialize congestion map
        congestion = [[0.0] * cols for _ in range(rows)]

        # Build component position lookup
        comp_positions = {}
        for ref, pos_data in placement.items():
            if isinstance(pos_data, dict):
                x = pos_data.get('x', 0)
                y = pos_data.get('y', 0)
            elif isinstance(pos_data, (list, tuple)) and len(pos_data) >= 2:
                x, y = pos_data[0], pos_data[1]
            else:
                continue
            comp_positions[ref] = (x, y)

        # Plan global routes for each net
        nets = parts_db.get('nets', {})
        parts = parts_db.get('parts', {})
        net_routes = {}

        for net_name, net_info in nets.items():
            # Skip nets removed by power grid planner
            if net_name in power_plan.nets_removed_from_routing:
                continue

            pins = net_info.get('pins', [])
            if len(pins) < 2:
                continue  # Single-pin nets don't need routing

            # Find physical positions of all pins in this net
            pin_positions = self._get_pin_positions(pins, parts, comp_positions)
            if len(pin_positions) < 2:
                continue

            # Plan the route through regions
            route = self._plan_net_route(
                net_name, pin_positions, rows, cols, board_width, board_height
            )
            net_routes[net_name] = route

            # Update congestion map
            for r, c in route.regions:
                if 0 <= r < rows and 0 <= c < cols:
                    congestion[r][c] += 1.0

        # Normalize congestion and find hotspots
        max_capacity = self._estimate_capacity(self.region_size)
        hotspots = []
        for r in range(rows):
            for c in range(cols):
                congestion[r][c] /= max_capacity
                if congestion[r][c] > 0.6:
                    hotspots.append((r, c, congestion[r][c]))

        hotspots.sort(key=lambda h: -h[2])

        return GlobalRoutingPlan(
            region_grid_rows=rows,
            region_grid_cols=cols,
            region_size_mm=self.region_size,
            net_routes=net_routes,
            congestion_map=congestion,
            hotspots=hotspots,
        )

    def _get_pin_positions(self, pins: List[str], parts: Dict,
                           comp_positions: Dict) -> List[Tuple[float, float]]:
        """Get physical positions of pins."""
        positions = []
        for pin_ref in pins:
            # pin_ref is like "U1.VCC" or "C1.1"
            parts_list = pin_ref.split('.')
            if len(parts_list) < 2:
                continue
            comp_ref = parts_list[0]
            pin_name = '.'.join(parts_list[1:])

            if comp_ref not in comp_positions:
                continue

            cx, cy = comp_positions[comp_ref]

            # Try to find pin offset from parts_db
            part = parts.get(comp_ref, {})
            pin_offset = self._get_pin_offset(part, pin_name)
            positions.append((cx + pin_offset[0], cy + pin_offset[1]))

        return positions

    def _get_pin_offset(self, part: Dict, pin_name: str) -> Tuple[float, float]:
        """Get pin offset from component center."""
        for pin in part.get('pins', []):
            if pin.get('name') == pin_name or pin.get('number') == pin_name:
                phys = pin.get('physical', {})
                return (phys.get('offset_x', 0), phys.get('offset_y', 0))
        return (0.0, 0.0)

    def _plan_net_route(self, net_name: str, positions: List[Tuple[float, float]],
                        rows: int, cols: int,
                        board_w: float, board_h: float) -> GlobalRoute:
        """Plan which regions a net passes through using bounding box."""
        # Find bounding box of all pin positions
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Convert to region coordinates
        min_col = max(0, int(min_x / self.region_size))
        max_col = min(cols - 1, int(max_x / self.region_size))
        min_row = max(0, int(min_y / self.region_size))
        max_row = min(rows - 1, int(max_y / self.region_size))

        # The net passes through all regions in its bounding box
        # (L-shaped or rectilinear Steiner tree approximation)
        regions = set()
        for pos in positions:
            r = min(rows - 1, max(0, int(pos[1] / self.region_size)))
            c = min(cols - 1, max(0, int(pos[0] / self.region_size)))
            regions.add((r, c))

        # Add regions along Manhattan path between first and last pin
        if len(positions) >= 2:
            p1, p2 = positions[0], positions[-1]
            r1 = min(rows - 1, max(0, int(p1[1] / self.region_size)))
            c1 = min(cols - 1, max(0, int(p1[0] / self.region_size)))
            r2 = min(rows - 1, max(0, int(p2[1] / self.region_size)))
            c2 = min(cols - 1, max(0, int(p2[0] / self.region_size)))

            # Horizontal then vertical (L-shape)
            for c in range(min(c1, c2), max(c1, c2) + 1):
                regions.add((r1, c))
            for r in range(min(r1, r2), max(r1, r2) + 1):
                regions.add((r, c2))

        estimated_length = abs(max_x - min_x) + abs(max_y - min_y)

        return GlobalRoute(
            net_name=net_name,
            regions=list(regions),
            estimated_length_mm=estimated_length,
            estimated_vias=1 if estimated_length > 15.0 else 0,
        )

    def _estimate_capacity(self, region_size: float) -> float:
        """Estimate how many nets can pass through a region."""
        # At 0.25mm trace + 0.15mm clearance = 0.4mm per trace
        # A 5mm region can fit ~12 parallel traces
        traces_per_region = region_size / 0.4
        # Two layers doubles capacity
        return traces_per_region * 2


# =============================================================================
# CONGESTION ESTIMATOR
# =============================================================================

class CongestionEstimator:
    """
    Strategy 4: Congestion Estimation (from VLSI placement optimization)

    Estimates routing congestion BEFORE routing starts, using:
    - Pin density per region
    - Net crossing density
    - Component body blocking

    This allows the placement to be adjusted if congestion is too high.
    """

    def analyze(self, global_plan: GlobalRoutingPlan,
                parts_db: Dict, placement: Dict,
                board_width: float, board_height: float) -> CongestionAnalysis:
        """Analyze congestion from global routing plan."""

        cmap = global_plan.congestion_map
        if not cmap or not cmap[0]:
            return CongestionAnalysis(
                overall_level=CongestionLevel.LOW,
                overall_utilization=0.0,
            )

        rows = len(cmap)
        cols = len(cmap[0])

        # Calculate overall utilization
        total_util = sum(sum(row) for row in cmap)
        cell_count = rows * cols
        avg_util = total_util / cell_count if cell_count > 0 else 0.0

        # Find hotspots
        hotspots = []
        for r in range(rows):
            for c in range(cols):
                if cmap[r][c] > 0.6:
                    hotspots.append((r, c, cmap[r][c]))
        hotspots.sort(key=lambda h: -h[2])

        # Determine overall level
        if avg_util > 0.8:
            level = CongestionLevel.CRITICAL
        elif avg_util > 0.6:
            level = CongestionLevel.HIGH
        elif avg_util > 0.3:
            level = CongestionLevel.MEDIUM
        else:
            level = CongestionLevel.LOW

        # Find bottleneck nets (nets passing through hotspot regions)
        bottleneck_nets = set()
        hotspot_regions = {(r, c) for r, c, _ in hotspots}
        for net_name, route in global_plan.net_routes.items():
            if any((r, c) in hotspot_regions for r, c in route.regions):
                bottleneck_nets.add(net_name)

        # Generate recommendations
        recommendations = []
        if level in (CongestionLevel.HIGH, CongestionLevel.CRITICAL):
            recommendations.append(
                f"Congestion is {level.value} (avg {avg_util:.0%}). "
                f"Consider increasing board size or reducing component count."
            )
        if len(hotspots) > 0:
            recommendations.append(
                f"{len(hotspots)} congested regions detected. "
                f"Worst: region ({hotspots[0][0]},{hotspots[0][1]}) at {hotspots[0][2]:.0%} utilization."
            )
        if bottleneck_nets:
            recommendations.append(
                f"{len(bottleneck_nets)} nets pass through congested regions: "
                f"{', '.join(list(bottleneck_nets)[:5])}"
            )

        return CongestionAnalysis(
            overall_level=level,
            overall_utilization=avg_util,
            region_utilization=cmap,
            hotspot_regions=hotspots,
            bottleneck_nets=list(bottleneck_nets),
            recommendations=recommendations,
        )


# =============================================================================
# NET PRIORITY ORDERING
# =============================================================================

class NetPriorityAssigner:
    """
    Strategy 5: Timing-Driven Net Ordering (from VLSI timing closure)

    In CPU design, timing-critical paths are routed FIRST with the shortest,
    most direct routes. Less critical nets get whatever's left.

    Priority order for PCBs:
      1. Differential pairs (must be routed together, matched length)
      2. High-speed signals (short paths, controlled impedance)
      3. Clock signals (short, clean routing)
      4. I2C/SPI buses (moderate priority)
      5. Power distribution (3V3, VBUS) - wide traces, star topology
      6. Simple signals (LED, GPIO) - whatever path works
      7. GND - SKIP (handled by pour)
    """

    # Priority levels (lower = route first)
    PRIORITY_DIFFERENTIAL = 10
    PRIORITY_HIGH_SPEED = 20
    PRIORITY_CLOCK = 25
    PRIORITY_I2C_SPI = 30
    PRIORITY_ANALOG = 35
    PRIORITY_POWER = 40
    PRIORITY_SIGNAL = 50
    PRIORITY_NC = 100  # No-connect, don't route

    def assign(self, parts_db: Dict, power_plan: PowerGridPlan) -> List[NetPriority]:
        """Assign routing priorities to all nets."""

        nets = parts_db.get('nets', {})
        priorities = []

        for net_name, net_info in nets.items():
            # Skip nets handled by pour
            if net_name in power_plan.nets_removed_from_routing:
                continue

            pins = net_info.get('pins', [])
            if len(pins) < 2:
                # Single-pin nets (NC) - skip
                priorities.append(NetPriority(
                    net_name=net_name,
                    priority=self.PRIORITY_NC,
                    net_type='nc',
                    reasoning=f"Single pin net - no routing needed"
                ))
                continue

            # Classify the net
            net_class = net_info.get('class', net_info.get('type', 'signal')).lower()
            priority, net_type, trace_w, clearance, reasoning = self._classify(
                net_name, net_class, net_info, power_plan
            )

            # Check for differential pair matching
            matched_with = None
            if net_type == 'differential':
                matched_with = self._find_diff_pair_partner(net_name, nets)

            priorities.append(NetPriority(
                net_name=net_name,
                priority=priority,
                net_type=net_type,
                trace_width_mm=trace_w,
                clearance_mm=clearance,
                must_route_with=matched_with,
                reasoning=reasoning,
            ))

        # Sort by priority
        priorities.sort(key=lambda p: p.priority)
        return priorities

    def _classify(self, name: str, net_class: str, info: Dict,
                  power_plan: PowerGridPlan
                  ) -> Tuple[int, str, float, float, str]:
        """Classify a net and return (priority, type, trace_width, clearance, reasoning)."""

        name_upper = name.upper()

        # Differential pairs
        if net_class == 'differential' or any(
            tag in name_upper for tag in ('_DP', '_DN', '_P', '_N', 'USB_D')
        ):
            return (self.PRIORITY_DIFFERENTIAL, 'differential', 0.25, 0.15,
                    "Differential pair - route first with matched length")

        # High speed
        if net_class == 'high_speed' or any(
            tag in name_upper for tag in ('CLK', 'CLOCK', 'MCLK')
        ):
            return (self.PRIORITY_CLOCK, 'clock', 0.25, 0.2,
                    "Clock/high-speed - short direct path needed")

        # I2C/SPI
        if net_class == 'i2c' or any(
            tag in name_upper for tag in ('SDA', 'SCL', 'I2C')
        ):
            return (self.PRIORITY_I2C_SPI, 'i2c', 0.25, 0.15,
                    "I2C bus - moderate priority, keep traces short")

        if any(tag in name_upper for tag in ('MOSI', 'MISO', 'SCLK', 'SPI', 'CS')):
            return (self.PRIORITY_I2C_SPI, 'spi', 0.25, 0.15,
                    "SPI bus - moderate priority")

        # Analog
        if net_class == 'analog' or any(
            tag in name_upper for tag in ('ANALOG', 'ADC', 'AREF')
        ):
            return (self.PRIORITY_ANALOG, 'analog', 0.25, 0.2,
                    "Analog signal - keep away from digital noise")

        # Power rails
        if net_class == 'power' or name_upper in ('VCC', 'VDD', '3V3', '5V', 'VBUS',
                                                    '3.3V', '5V0', '12V', '1V8'):
            width = power_plan.power_trace_widths.get(name, 0.4)
            return (self.PRIORITY_POWER, 'power', width, 0.2,
                    f"Power rail - wide traces ({width}mm)")

        # Default signals
        return (self.PRIORITY_SIGNAL, 'signal', 0.25, 0.15,
                "General signal - standard routing")

    def _find_diff_pair_partner(self, net_name: str, nets: Dict) -> Optional[str]:
        """Find the differential pair partner of a net."""
        name = net_name.upper()
        # USB_DP <-> USB_DN
        if 'DP' in name:
            partner = name.replace('DP', 'DN')
        elif 'DN' in name:
            partner = name.replace('DN', 'DP')
        elif '_P' in name:
            partner = name.replace('_P', '_N')
        elif '_N' in name:
            partner = name.replace('_N', '_P')
        else:
            return None

        # Find matching net (case-insensitive)
        for n in nets:
            if n.upper() == partner:
                return n
        return None


# =============================================================================
# COMPONENT GROUP TEMPLATES
# =============================================================================

class ComponentGroupBuilder:
    """
    Strategy 6: Component Group Templates (from VLSI standard cells)

    In CPU design, every gate has a pre-designed, pre-verified layout.
    We apply this to common PCB circuit blocks:
    - LDO + decoupling caps
    - I2C pull-up resistors near MCU
    - USB connector + ESD + CC resistors
    - LED + current limiting resistor

    These groups ensure proper placement proximity and internal routing.
    """

    def build_groups(self, parts_db: Dict) -> List[ComponentGroup]:
        """Detect and build component groups from the parts_db."""

        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})
        groups = []

        # Detect LDO + capacitor groups
        groups.extend(self._detect_ldo_groups(parts, nets))

        # Detect I2C pull-up groups
        groups.extend(self._detect_i2c_pullup_groups(parts, nets))

        # Detect LED + resistor groups
        groups.extend(self._detect_led_groups(parts, nets))

        # Detect decoupling capacitor groups
        groups.extend(self._detect_decoupling_groups(parts, nets))

        return groups

    def _detect_ldo_groups(self, parts: Dict, nets: Dict) -> List[ComponentGroup]:
        """Detect LDO regulator + input/output capacitor groups."""
        groups = []
        for ref, part in parts.items():
            name = part.get('name', '').upper()
            desc = part.get('description', '').upper()
            if any(tag in name + desc for tag in ('LDO', 'REGULATOR', 'AMS1117', 'LM1117', 'AP2112')):
                # Find capacitors connected to same power nets
                ldo_nets = set()
                for pin in part.get('pins', []):
                    net = pin.get('net', '')
                    if net:
                        ldo_nets.add(net)

                caps = []
                for cap_ref, cap_part in parts.items():
                    if cap_ref == ref:
                        continue
                    cap_fp = cap_part.get('footprint', '').upper()
                    if not any(tag in cap_fp for tag in ('0402', '0603', '0805', '1206', 'CAP')):
                        continue
                    for pin in cap_part.get('pins', []):
                        if pin.get('net', '') in ldo_nets:
                            caps.append(cap_ref)
                            break

                if caps:
                    group = ComponentGroup(
                        name=f"LDO_{ref}",
                        components=[ref] + list(set(caps)),
                        anchor=ref,
                        relative_positions={c: (2.0, 0.0) for i, c in enumerate(caps)},
                        internal_nets=list(ldo_nets),
                        description=f"LDO regulator {ref} with {len(caps)} decoupling caps"
                    )
                    groups.append(group)
        return groups

    def _detect_i2c_pullup_groups(self, parts: Dict, nets: Dict) -> List[ComponentGroup]:
        """Detect I2C pull-up resistor groups."""
        groups = []
        i2c_nets = set()
        for net_name, net_info in nets.items():
            if any(tag in net_name.upper() for tag in ('SDA', 'SCL', 'I2C')):
                i2c_nets.add(net_name)

        if not i2c_nets:
            return groups

        # Find resistors connected to I2C nets
        pullup_resistors = []
        for ref, part in parts.items():
            if not ref.startswith('R'):
                continue
            for pin in part.get('pins', []):
                if pin.get('net', '') in i2c_nets:
                    pullup_resistors.append(ref)
                    break

        if pullup_resistors:
            groups.append(ComponentGroup(
                name="I2C_PULLUPS",
                components=pullup_resistors,
                anchor=pullup_resistors[0],
                description=f"I2C pull-up resistors: {', '.join(pullup_resistors)}"
            ))

        return groups

    def _detect_led_groups(self, parts: Dict, nets: Dict) -> List[ComponentGroup]:
        """Detect LED + current limiting resistor groups."""
        groups = []
        for ref, part in parts.items():
            name = part.get('name', '').upper()
            fp = part.get('footprint', '').upper()
            if 'LED' not in name and 'LED' not in ref.upper():
                continue

            # Find the net connecting LED anode to a resistor
            led_nets = set()
            for pin in part.get('pins', []):
                net = pin.get('net', '')
                if net and net.upper() != 'GND':
                    led_nets.add(net)

            # Find resistor on same net
            for r_ref, r_part in parts.items():
                if not r_ref.startswith('R'):
                    continue
                for pin in r_part.get('pins', []):
                    if pin.get('net', '') in led_nets:
                        groups.append(ComponentGroup(
                            name=f"LED_{ref}",
                            components=[ref, r_ref],
                            anchor=r_ref,
                            relative_positions={ref: (2.0, 0.0)},
                            internal_nets=list(led_nets),
                            description=f"LED {ref} with current limiter {r_ref}"
                        ))
                        break

        return groups

    def _detect_decoupling_groups(self, parts: Dict, nets: Dict) -> List[ComponentGroup]:
        """Detect decoupling capacitors that should be near their IC."""
        groups = []

        # Find ICs (U-prefix components)
        ics = {ref: part for ref, part in parts.items()
               if ref.startswith('U') or ref.startswith('IC')}

        for ic_ref, ic_part in ics.items():
            # Find power nets connected to this IC
            ic_power_nets = set()
            for pin in ic_part.get('pins', []):
                ptype = pin.get('type', '')
                if ptype in ('power_in', 'power_out'):
                    net = pin.get('net', '')
                    if net and net.upper() != 'GND':
                        ic_power_nets.add(net)

            if not ic_power_nets:
                continue

            # Find capacitors on same power nets
            decoupling_caps = []
            for cap_ref, cap_part in parts.items():
                if cap_ref.startswith(('C',)):
                    cap_nets = set(pin.get('net', '') for pin in cap_part.get('pins', []))
                    if cap_nets & ic_power_nets:
                        # Check if other pin is GND
                        if any(pin.get('net', '').upper() == 'GND' for pin in cap_part.get('pins', [])):
                            decoupling_caps.append(cap_ref)

            if decoupling_caps:
                groups.append(ComponentGroup(
                    name=f"DECOUPLING_{ic_ref}",
                    components=[ic_ref] + decoupling_caps,
                    anchor=ic_ref,
                    description=f"Decoupling for {ic_ref}: {', '.join(decoupling_caps)}"
                ))

        return groups


# =============================================================================
# FACTORY INSPECTOR (Continuous DRC Monitor)
# =============================================================================

class FactoryInspector:
    """
    Continuous DRC Monitor - Quality Control on the Factory Floor

    Unlike end-of-line DRC that only checks the final product, the Factory
    Inspector monitors quality at EVERY stage of the BBL:

    1. POST_PLACEMENT: Check component overlaps, courtyard violations
    2. POST_POUR: Verify pour connectivity, island detection
    3. DURING_ROUTING: Check each net as it's routed (live monitoring)
    4. POST_ROUTING: Full routing DRC (clearance, shorts)
    5. POST_OPTIMIZE: Verify optimization didn't introduce errors
    6. FINAL: Combined internal DRC + KiCad DRC

    The inspector can STOP THE LINE if a critical violation is found,
    saving time by catching problems early instead of at the end.
    """

    def __init__(self):
        self.all_findings: List[InspectionFinding] = []
        self.reports: Dict[InspectionStage, InspectionReport] = {}
        self._callbacks: List[Callable] = []

    def add_callback(self, callback: Callable[[InspectionReport], None]):
        """Register a callback for inspection reports."""
        self._callbacks.append(callback)

    def _notify(self, report: InspectionReport):
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(report)
            except Exception:
                pass

    def inspect_placement(self, placement: Dict, parts_db: Dict,
                          board_width: float, board_height: float,
                          component_groups: List[ComponentGroup]) -> InspectionReport:
        """Inspect placement quality."""
        start = time.time()
        findings = []
        parts = parts_db.get('parts', {})

        # Check 1: Components within board boundaries
        for ref, pos_data in placement.items():
            if isinstance(pos_data, dict):
                x, y = pos_data.get('x', 0), pos_data.get('y', 0)
            elif isinstance(pos_data, (list, tuple)):
                x, y = pos_data[0], pos_data[1]
            else:
                continue

            part = parts.get(ref, {})
            size = part.get('size', (2.0, 2.0))
            if isinstance(size, (list, tuple)):
                w, h = size[0], size[1]
            else:
                w, h = 2.0, 2.0

            half_w, half_h = w / 2, h / 2

            if x - half_w < 0 or x + half_w > board_width:
                findings.append(InspectionFinding(
                    stage=InspectionStage.POST_PLACEMENT,
                    severity=InspectionSeverity.ERROR,
                    code="COMP_OUT_OF_BOUNDS_X",
                    message=f"{ref} extends beyond board edge (x={x:.1f}, width={w:.1f}, board={board_width:.1f})",
                    location=(x, y),
                    affected_components=[ref],
                    suggestion=f"Move {ref} at least {half_w}mm from board edge"
                ))

            if y - half_h < 0 or y + half_h > board_height:
                findings.append(InspectionFinding(
                    stage=InspectionStage.POST_PLACEMENT,
                    severity=InspectionSeverity.ERROR,
                    code="COMP_OUT_OF_BOUNDS_Y",
                    message=f"{ref} extends beyond board edge (y={y:.1f}, height={h:.1f}, board={board_height:.1f})",
                    location=(x, y),
                    affected_components=[ref],
                    suggestion=f"Move {ref} at least {half_h}mm from board edge"
                ))

        # Check 2: Component overlap (courtyard check)
        refs = list(placement.keys())
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                ref_a, ref_b = refs[i], refs[j]
                if self._components_overlap(ref_a, ref_b, placement, parts):
                    findings.append(InspectionFinding(
                        stage=InspectionStage.POST_PLACEMENT,
                        severity=InspectionSeverity.CRITICAL,
                        code="COURTYARD_OVERLAP",
                        message=f"Components {ref_a} and {ref_b} overlap",
                        affected_components=[ref_a, ref_b],
                        suggestion=f"Increase spacing between {ref_a} and {ref_b}"
                    ))

        # Check 3: Decoupling caps near their ICs
        for group in component_groups:
            if group.name.startswith('DECOUPLING_'):
                ic_ref = group.anchor
                if ic_ref not in placement:
                    continue
                for cap_ref in group.components:
                    if cap_ref == ic_ref or cap_ref not in placement:
                        continue
                    dist = self._distance(ic_ref, cap_ref, placement)
                    if dist > 10.0:  # More than 10mm from IC
                        findings.append(InspectionFinding(
                            stage=InspectionStage.POST_PLACEMENT,
                            severity=InspectionSeverity.WARNING,
                            code="DECOUPLING_TOO_FAR",
                            message=f"Decoupling cap {cap_ref} is {dist:.1f}mm from {ic_ref} (should be <5mm)",
                            affected_components=[ic_ref, cap_ref],
                            suggestion=f"Move {cap_ref} closer to {ic_ref} power pin"
                        ))

        report = self._create_report(InspectionStage.POST_PLACEMENT, findings, start)
        self._notify(report)
        return report

    def inspect_routing_live(self, net_name: str, route_success: bool,
                             route_length: float, via_count: int,
                             net_priority: Optional[NetPriority] = None) -> Optional[InspectionFinding]:
        """Inspect a single net as it's routed (live monitoring)."""

        if not route_success:
            severity = InspectionSeverity.ERROR
            if net_priority and net_priority.priority <= 20:
                severity = InspectionSeverity.CRITICAL  # Critical net failed
            finding = InspectionFinding(
                stage=InspectionStage.DURING_ROUTING,
                severity=severity,
                code="NET_ROUTING_FAILED",
                message=f"Failed to route net '{net_name}' (type: {net_priority.net_type if net_priority else 'unknown'})",
                affected_nets=[net_name],
                suggestion="Check for congestion or component placement blocking path",
                timestamp=time.time(),
            )
            self.all_findings.append(finding)
            return finding

        # Check for excessive length
        if net_priority and net_priority.max_length_mm:
            if route_length > net_priority.max_length_mm:
                finding = InspectionFinding(
                    stage=InspectionStage.DURING_ROUTING,
                    severity=InspectionSeverity.WARNING,
                    code="NET_TOO_LONG",
                    message=f"Net '{net_name}' route is {route_length:.1f}mm (max: {net_priority.max_length_mm:.1f}mm)",
                    affected_nets=[net_name],
                    timestamp=time.time(),
                )
                self.all_findings.append(finding)
                return finding

        return None

    def inspect_post_routing(self, routes: Dict, total_nets: int,
                             routed_count: int) -> InspectionReport:
        """Inspect overall routing results."""
        start = time.time()
        findings = []

        completion = routed_count / total_nets if total_nets > 0 else 0

        if completion < 0.5:
            findings.append(InspectionFinding(
                stage=InspectionStage.POST_ROUTING,
                severity=InspectionSeverity.CRITICAL,
                code="ROUTING_FAILURE",
                message=f"Only {routed_count}/{total_nets} nets routed ({completion:.0%})",
                suggestion="Check power grid plan and component placement"
            ))
        elif completion < 0.9:
            findings.append(InspectionFinding(
                stage=InspectionStage.POST_ROUTING,
                severity=InspectionSeverity.ERROR,
                code="ROUTING_INCOMPLETE",
                message=f"{routed_count}/{total_nets} nets routed ({completion:.0%})",
                suggestion="Try different routing algorithm or relaxed constraints"
            ))
        elif completion < 1.0:
            findings.append(InspectionFinding(
                stage=InspectionStage.POST_ROUTING,
                severity=InspectionSeverity.WARNING,
                code="ROUTING_PARTIAL",
                message=f"{routed_count}/{total_nets} nets routed ({completion:.0%})",
            ))

        report = self._create_report(InspectionStage.POST_ROUTING, findings, start)
        self._notify(report)
        return report

    def get_cumulative_report(self) -> Dict[str, Any]:
        """Get a summary of all inspections so far."""
        total_critical = sum(1 for f in self.all_findings if f.severity == InspectionSeverity.CRITICAL)
        total_errors = sum(1 for f in self.all_findings if f.severity == InspectionSeverity.ERROR)
        total_warnings = sum(1 for f in self.all_findings if f.severity == InspectionSeverity.WARNING)

        return {
            'total_findings': len(self.all_findings),
            'critical': total_critical,
            'errors': total_errors,
            'warnings': total_warnings,
            'should_stop': total_critical > 0,
            'stages_inspected': list(self.reports.keys()),
            'findings': self.all_findings,
        }

    def _components_overlap(self, ref_a: str, ref_b: str,
                            placement: Dict, parts: Dict) -> bool:
        """Check if two components overlap (simplified AABB check)."""
        pos_a = placement.get(ref_a)
        pos_b = placement.get(ref_b)
        if not pos_a or not pos_b:
            return False

        xa = pos_a.get('x', pos_a[0] if isinstance(pos_a, (list, tuple)) else 0)
        ya = pos_a.get('y', pos_a[1] if isinstance(pos_a, (list, tuple)) else 0)
        xb = pos_b.get('x', pos_b[0] if isinstance(pos_b, (list, tuple)) else 0)
        yb = pos_b.get('y', pos_b[1] if isinstance(pos_b, (list, tuple)) else 0)

        size_a = parts.get(ref_a, {}).get('size', (2.0, 2.0))
        size_b = parts.get(ref_b, {}).get('size', (2.0, 2.0))

        wa = size_a[0] / 2 if isinstance(size_a, (list, tuple)) else 1.0
        ha = size_a[1] / 2 if isinstance(size_a, (list, tuple)) else 1.0
        wb = size_b[0] / 2 if isinstance(size_b, (list, tuple)) else 1.0
        hb = size_b[1] / 2 if isinstance(size_b, (list, tuple)) else 1.0

        # Add courtyard margin (0.25mm)
        margin = 0.25
        wa += margin
        ha += margin
        wb += margin
        hb += margin

        return (abs(xa - xb) < wa + wb) and (abs(ya - yb) < ha + hb)

    def _distance(self, ref_a: str, ref_b: str, placement: Dict) -> float:
        """Calculate distance between two components."""
        pos_a = placement.get(ref_a, {})
        pos_b = placement.get(ref_b, {})

        xa = pos_a.get('x', pos_a[0] if isinstance(pos_a, (list, tuple)) else 0)
        ya = pos_a.get('y', pos_a[1] if isinstance(pos_a, (list, tuple)) else 0)
        xb = pos_b.get('x', pos_b[0] if isinstance(pos_b, (list, tuple)) else 0)
        yb = pos_b.get('y', pos_b[1] if isinstance(pos_b, (list, tuple)) else 0)

        return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

    def _create_report(self, stage: InspectionStage,
                       findings: List[InspectionFinding],
                       start_time: float) -> InspectionReport:
        """Create an inspection report from findings."""
        self.all_findings.extend(findings)

        critical = sum(1 for f in findings if f.severity == InspectionSeverity.CRITICAL)
        errors = sum(1 for f in findings if f.severity == InspectionSeverity.ERROR)
        warnings = sum(1 for f in findings if f.severity == InspectionSeverity.WARNING)
        infos = sum(1 for f in findings if f.severity == InspectionSeverity.INFO)

        report = InspectionReport(
            stage=stage,
            passed=critical == 0 and errors == 0,
            findings=findings,
            critical_count=critical,
            error_count=errors,
            warning_count=warnings,
            info_count=infos,
            duration_ms=(time.time() - start_time) * 1000,
        )

        self.reports[stage] = report
        return report


# =============================================================================
# CPU LAB - MAIN ORCHESTRATOR
# =============================================================================

class CPULab:
    """
    CPU Lab - VLSI-Inspired PCB Design Enhancement

    The main orchestrator that runs all 6 strategies and produces
    an enhanced design specification for the BBL engine.

    Usage:
        cpu_lab = CPULab()
        result = cpu_lab.enhance(parts_db, board_config)

        # result.enhanced_parts_db has all CPU Lab decisions injected
        # Pass to BBL for execution
        bbl_result = bbl_engine.run(result.enhanced_parts_db)
    """

    def __init__(self):
        self.power_planner = PowerGridPlanner()
        self.layer_assigner = LayerDirectionAssigner()
        self.global_router = GlobalRouter(region_size_mm=5.0)
        self.congestion_estimator = CongestionEstimator()
        self.priority_assigner = NetPriorityAssigner()
        self.group_builder = ComponentGroupBuilder()
        self.inspector = FactoryInspector()

    def enhance(self, parts_db: Dict, board_config: Dict,
                placement: Optional[Dict] = None) -> CPULabResult:
        """
        Run all CPU Lab strategies and produce enhanced design spec.

        Args:
            parts_db: Original parts database from c_layout
            board_config: Board configuration (width, height, layers)
            placement: Optional pre-computed placement (if available)

        Returns:
            CPULabResult with all decisions and enhanced parts_db
        """
        start = time.time()
        summary = []

        board_w = board_config.get('board_width', board_config.get('width', 50.0))
        board_h = board_config.get('board_height', board_config.get('height', 40.0))
        layers = board_config.get('layers', board_config.get('layer_count', 2))

        print(f"\n{'='*60}")
        print(f"CPU LAB - VLSI-Inspired Design Enhancement")
        print(f"{'='*60}")
        print(f"  Board: {board_w}x{board_h}mm, {layers} layers")
        print(f"  Components: {len(parts_db.get('parts', {}))}")
        print(f"  Nets: {len(parts_db.get('nets', {}))}")

        # ---- Strategy 1: Power Grid Planning ----
        print(f"\n--- Strategy 1: Power Grid Planning ---")
        power_plan = self.power_planner.plan(parts_db, board_config)
        for r in power_plan.reasoning:
            print(f"  {r}")
        summary.append(f"Power: GND={power_plan.gnd_strategy.value}, "
                       f"removed {len(power_plan.nets_removed_from_routing)} nets from routing")

        # ---- Strategy 2: Layer Direction Assignment ----
        print(f"\n--- Strategy 2: Layer Direction Assignment ---")
        layer_assignments = self.layer_assigner.assign(layers, board_w, board_h)
        for la in layer_assignments:
            print(f"  {la.layer_name}: {la.preferred_direction.value} "
                  f"(penalty {la.direction_cost_penalty}x for wrong dir)")
        summary.append(f"Layers: {', '.join(f'{la.layer_name}={la.preferred_direction.value[0].upper()}' for la in layer_assignments)}")

        # ---- Strategy 3: Global Routing ----
        print(f"\n--- Strategy 3: Global Routing ---")
        # Use provided placement or create a dummy one
        if placement is None:
            placement = self._create_dummy_placement(parts_db, board_w, board_h)

        global_plan = self.global_router.plan(
            parts_db, placement, board_w, board_h, power_plan
        )
        print(f"  Grid: {global_plan.region_grid_rows}x{global_plan.region_grid_cols} regions "
              f"({global_plan.region_size_mm}mm each)")
        print(f"  Nets planned: {len(global_plan.net_routes)}")
        print(f"  Hotspots: {len(global_plan.hotspots)}")
        summary.append(f"Global: {len(global_plan.net_routes)} nets planned, "
                       f"{len(global_plan.hotspots)} hotspots")

        # ---- Strategy 4: Congestion Estimation ----
        print(f"\n--- Strategy 4: Congestion Estimation ---")
        congestion = self.congestion_estimator.analyze(
            global_plan, parts_db, placement, board_w, board_h
        )
        print(f"  Overall: {congestion.overall_level.value} ({congestion.overall_utilization:.0%})")
        for rec in congestion.recommendations:
            print(f"  ! {rec}")
        summary.append(f"Congestion: {congestion.overall_level.value} ({congestion.overall_utilization:.0%})")

        # ---- Strategy 5: Net Priority Ordering ----
        print(f"\n--- Strategy 5: Net Priority Ordering ---")
        net_priorities = self.priority_assigner.assign(parts_db, power_plan)
        routable = [p for p in net_priorities if p.priority < 100]
        print(f"  Routable nets: {len(routable)} (ordered by priority)")
        for p in routable[:8]:
            print(f"    P{p.priority:3d} [{p.net_type:12s}] {p.net_name} "
                  f"(w={p.trace_width_mm}mm)")
        if len(routable) > 8:
            print(f"    ... and {len(routable) - 8} more")
        summary.append(f"Priorities: {len(routable)} nets ordered, "
                       f"top={routable[0].net_type if routable else 'none'}")

        # ---- Strategy 6: Component Groups ----
        print(f"\n--- Strategy 6: Component Group Templates ---")
        groups = self.group_builder.build_groups(parts_db)
        for g in groups:
            print(f"  [{g.name}] {g.description}")
        summary.append(f"Groups: {len(groups)} detected")

        # ---- Build Enhanced parts_db ----
        enhanced = self._build_enhanced_parts_db(
            parts_db, power_plan, layer_assignments, global_plan,
            congestion, net_priorities, groups, board_config
        )

        processing_time = (time.time() - start) * 1000

        print(f"\n{'='*60}")
        print(f"CPU LAB COMPLETE ({processing_time:.0f}ms)")
        print(f"{'='*60}")
        for s in summary:
            print(f"  {s}")
        print()

        return CPULabResult(
            power_grid=power_plan,
            layer_assignments=layer_assignments,
            global_routing=global_plan,
            congestion=congestion,
            net_priorities=net_priorities,
            component_groups=groups,
            enhanced_parts_db=enhanced,
            processing_time_ms=processing_time,
            summary=summary,
        )

    def _build_enhanced_parts_db(self, original: Dict,
                                 power: PowerGridPlan,
                                 layers: List[LayerAssignment],
                                 global_plan: GlobalRoutingPlan,
                                 congestion: CongestionAnalysis,
                                 priorities: List[NetPriority],
                                 groups: List[ComponentGroup],
                                 board_config: Dict) -> Dict:
        """Build the enhanced parts_db with all CPU Lab decisions."""
        import copy
        enhanced = copy.deepcopy(original)

        # Inject CPU Lab decisions
        enhanced['cpu_lab'] = {
            # Power grid decisions
            'power_grid': {
                'gnd_strategy': power.gnd_strategy.value,
                'gnd_pour_layer': power.gnd_pour_layer,
                'gnd_pour_config': power.gnd_pour_config,
                'power_strategies': {k: v.value for k, v in power.power_strategies.items()},
                'power_trace_widths': power.power_trace_widths,
                'nets_removed_from_routing': power.nets_removed_from_routing,
            },

            # Layer direction preferences
            'layer_directions': {
                la.layer_name: {
                    'preferred': la.preferred_direction.value,
                    'penalty': la.direction_cost_penalty,
                } for la in layers
            },

            # Global routing hints
            'global_routing': {
                'region_size_mm': global_plan.region_size_mm,
                'grid': f"{global_plan.region_grid_rows}x{global_plan.region_grid_cols}",
                'hotspot_count': len(global_plan.hotspots),
                'net_regions': {
                    name: route.regions
                    for name, route in global_plan.net_routes.items()
                },
            },

            # Congestion
            'congestion': {
                'level': congestion.overall_level.value,
                'utilization': congestion.overall_utilization,
                'bottleneck_nets': congestion.bottleneck_nets,
            },

            # Net routing order (priority sorted)
            'routing_order': [
                {
                    'net': p.net_name,
                    'priority': p.priority,
                    'type': p.net_type,
                    'trace_width': p.trace_width_mm,
                    'clearance': p.clearance_mm,
                    'must_route_with': p.must_route_with,
                }
                for p in priorities if p.priority < 100
            ],

            # Component groups
            'component_groups': [
                {
                    'name': g.name,
                    'components': g.components,
                    'anchor': g.anchor,
                    'description': g.description,
                }
                for g in groups
            ],

            # Board config reference
            'board': board_config,
        }

        return enhanced

    def _create_dummy_placement(self, parts_db: Dict,
                                board_w: float, board_h: float) -> Dict:
        """Create a rough placement estimate for global routing analysis."""
        parts = parts_db.get('parts', {})
        placement = {}
        n = len(parts)
        cols = max(1, int(math.ceil(math.sqrt(n * board_w / board_h))))
        rows = max(1, int(math.ceil(n / cols)))

        spacing_x = board_w / (cols + 1)
        spacing_y = board_h / (rows + 1)

        for i, ref in enumerate(parts):
            row = i // cols
            col = i % cols
            placement[ref] = {
                'x': spacing_x * (col + 1),
                'y': spacing_y * (row + 1),
            }

        return placement
