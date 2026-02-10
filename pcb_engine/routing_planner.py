"""
ROUTING PLANNER - Smart Algorithm Selection
============================================

The RoutingPlanner is the INTELLIGENCE BRIDGE between Circuit AI (Engineer)
and the Routing Piston. It analyzes the design and creates a per-net
routing strategy instead of using blind cascade.

ROLE IN COMMAND HIERARCHY:
==========================
    CIRCUIT AI (Engineer)
           │
           │ parts_db + requirements
           ▼
    ┌─────────────────────┐
    │   ROUTING PLANNER   │  ← THIS MODULE
    │                     │
    │  Analyze → Classify │
    │  Select → Plan      │
    └──────────┬──────────┘
               │
               │ RoutingPlan (per-net strategies)
               ▼
         PCB ENGINE (Foreman)
               │
               ▼
         ROUTING PISTON

RESPONSIBILITIES:
=================
1. Analyze design profile (board size, density, component types)
2. Classify every net (POWER, GROUND, HIGH_SPEED, BUS, SIGNAL, etc.)
3. Select best algorithm for each net class
4. Query learning database for past successes
5. Create complete routing plan with priorities and fallbacks

Author: PCB Engine Team
Date: 2026-02-09
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Set, Any
import re
import math
from collections import defaultdict


# =============================================================================
# ENUMS
# =============================================================================

class DesignDensity(Enum):
    """Board density classification"""
    SPARSE = 'sparse'       # <20% area utilization
    LOW = 'low'             # 20-40%
    MEDIUM = 'medium'       # 40-60%
    HIGH = 'high'           # 60-80%
    VERY_HIGH = 'very_high' # >80%


class PackageComplexity(Enum):
    """Package complexity for escape routing"""
    SIMPLE = 'simple'       # 2-pin, through-hole
    MODERATE = 'moderate'   # SOIC, SOT-23
    COMPLEX = 'complex'     # QFP, QFN
    VERY_COMPLEX = 'very_complex'  # BGA, fine-pitch


class NetClass(Enum):
    """Net classification for algorithm selection"""
    POWER = 'power'             # VCC, VDD, VBAT - wide traces
    GROUND = 'ground'           # GND - use pour when possible
    HIGH_SPEED = 'high_speed'   # CLK, USB, DDR - impedance controlled
    DIFFERENTIAL = 'differential'  # USB_D+/D-, HDMI - length matched
    BUS = 'bus'                 # DATA[0:7] - parallel routing
    I2C = 'i2c'                 # SDA, SCL - pull-ups needed
    SPI = 'spi'                 # MOSI, MISO, SCK, CS
    ANALOG = 'analog'           # ADC inputs - avoid crosstalk
    RF = 'rf'                   # Antenna, RF signals - minimal vias
    HIGH_CURRENT = 'high_current'  # Motor, LED - thick traces
    SIGNAL = 'signal'           # General GPIO - default


class RoutingAlgorithm(Enum):
    """Available routing algorithms"""
    LEE = 'lee'
    HADLOCK = 'hadlock'
    SOUKUP = 'soukup'
    MIKAMI = 'mikami'
    A_STAR = 'a_star'
    PATHFINDER = 'pathfinder'
    RIPUP = 'ripup'
    STEINER = 'steiner'
    CHANNEL = 'channel'
    HYBRID = 'hybrid'
    AUTO = 'auto'
    PUSH_AND_SHOVE = 'push_and_shove'


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DesignProfile:
    """Complete analysis of the design"""
    # Board characteristics
    board_width: float
    board_height: float
    board_area: float
    layer_count: int

    # Component analysis
    component_count: int
    ic_count: int
    passive_count: int
    connector_count: int

    # Package complexity
    has_bga: bool = False
    has_qfn: bool = False
    has_fine_pitch: bool = False
    max_pin_count: int = 0

    # Net analysis
    net_count: int = 0
    avg_net_length: float = 0.0
    max_fanout: int = 0

    # Density
    density: DesignDensity = DesignDensity.MEDIUM
    area_utilization: float = 0.0

    # Special requirements
    has_high_speed: bool = False
    has_differential: bool = False
    has_power_planes: bool = False
    needs_ground_pour: bool = False

    # Recommendations
    recommended_algorithms: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class NetRoutingStrategy:
    """Strategy for routing a single net"""
    net_name: str
    net_class: NetClass
    priority: int  # 1=highest (route first)

    # Algorithm selection
    primary_algorithm: RoutingAlgorithm
    fallback_algorithms: List[RoutingAlgorithm] = field(default_factory=list)

    # Constraints
    trace_width: float = 0.25  # mm
    clearance: float = 0.15    # mm
    via_size: float = 0.8      # mm
    via_drill: float = 0.4     # mm

    # Special handling flags
    use_push_and_shove: bool = False
    trunk_chain_id: Optional[int] = None
    check_return_path: bool = False
    length_match_group: Optional[str] = None

    # Learning database reference
    learned_algorithm: Optional[str] = None
    learned_success_rate: float = 0.0


@dataclass
class RoutingPlan:
    """Complete routing plan for a design"""
    design_profile: DesignProfile
    net_strategies: Dict[str, NetRoutingStrategy] = field(default_factory=dict)
    routing_order: List[str] = field(default_factory=list)

    # Global settings
    enable_trunk_chains: bool = False
    enable_return_path_check: bool = False
    ground_pour_recommended: bool = False

    # Trunk chains detected
    trunk_chains: List[List[str]] = field(default_factory=list)

    # Learning database info
    similar_designs: List[str] = field(default_factory=list)
    overall_success_prediction: float = 0.0

    def get_strategy(self, net_name: str) -> Optional[NetRoutingStrategy]:
        """Get strategy for a specific net"""
        return self.net_strategies.get(net_name)

    def get_nets_by_class(self, net_class: NetClass) -> List[str]:
        """Get all nets of a specific class"""
        return [name for name, strategy in self.net_strategies.items()
                if strategy.net_class == net_class]

    def get_algorithm_distribution(self) -> Dict[str, int]:
        """Get count of nets per primary algorithm"""
        dist = defaultdict(int)
        for strategy in self.net_strategies.values():
            dist[strategy.primary_algorithm.value] += 1
        return dict(dist)


# =============================================================================
# ALGORITHM SELECTION MATRIX
# =============================================================================

# Decision matrix: net_class -> (primary_algorithm, fallback_chain)
ALGORITHM_SELECTION_MATRIX: Dict[NetClass, Tuple[RoutingAlgorithm, List[RoutingAlgorithm]]] = {
    NetClass.POWER: (
        RoutingAlgorithm.LEE,  # Guaranteed shortest path for power
        [RoutingAlgorithm.PATHFINDER, RoutingAlgorithm.A_STAR]
    ),
    NetClass.GROUND: (
        RoutingAlgorithm.LEE,  # But prefer pour - handled separately
        [RoutingAlgorithm.PATHFINDER]
    ),
    NetClass.HIGH_SPEED: (
        RoutingAlgorithm.STEINER,  # Optimal multi-terminal
        [RoutingAlgorithm.LEE, RoutingAlgorithm.HADLOCK]
    ),
    NetClass.DIFFERENTIAL: (
        RoutingAlgorithm.STEINER,  # Length matching support
        [RoutingAlgorithm.LEE]
    ),
    NetClass.BUS: (
        RoutingAlgorithm.PATHFINDER,  # Congestion-aware for parallel
        [RoutingAlgorithm.CHANNEL, RoutingAlgorithm.HYBRID]
    ),
    NetClass.I2C: (
        RoutingAlgorithm.LEE,  # Short paths critical
        [RoutingAlgorithm.HADLOCK]
    ),
    NetClass.SPI: (
        RoutingAlgorithm.A_STAR,  # Fast, good enough
        [RoutingAlgorithm.LEE, RoutingAlgorithm.HADLOCK]
    ),
    NetClass.ANALOG: (
        RoutingAlgorithm.LEE,  # Avoid crosstalk
        [RoutingAlgorithm.HADLOCK]
    ),
    NetClass.RF: (
        RoutingAlgorithm.LEE,  # Minimal vias
        []
    ),
    NetClass.HIGH_CURRENT: (
        RoutingAlgorithm.LEE,  # Wide traces only
        []
    ),
    NetClass.SIGNAL: (
        RoutingAlgorithm.HYBRID,  # Best general purpose
        [RoutingAlgorithm.A_STAR, RoutingAlgorithm.PATHFINDER, RoutingAlgorithm.LEE]
    ),
}


# Net class priority order (lower = higher priority, route first)
NET_CLASS_PRIORITY: Dict[NetClass, int] = {
    NetClass.POWER: 1,
    NetClass.GROUND: 2,
    NetClass.HIGH_SPEED: 3,
    NetClass.DIFFERENTIAL: 4,
    NetClass.RF: 5,
    NetClass.ANALOG: 6,
    NetClass.BUS: 7,
    NetClass.I2C: 8,
    NetClass.SPI: 9,
    NetClass.HIGH_CURRENT: 10,
    NetClass.SIGNAL: 11,
}


# Trace widths by net class (mm)
NET_CLASS_TRACE_WIDTH: Dict[NetClass, float] = {
    NetClass.POWER: 0.5,
    NetClass.GROUND: 0.5,
    NetClass.HIGH_SPEED: 0.12,
    NetClass.DIFFERENTIAL: 0.12,
    NetClass.RF: 0.15,
    NetClass.ANALOG: 0.2,
    NetClass.BUS: 0.15,
    NetClass.I2C: 0.2,
    NetClass.SPI: 0.2,
    NetClass.HIGH_CURRENT: 1.0,
    NetClass.SIGNAL: 0.25,
}


# Clearance by net class (mm)
NET_CLASS_CLEARANCE: Dict[NetClass, float] = {
    NetClass.POWER: 0.2,
    NetClass.GROUND: 0.2,
    NetClass.HIGH_SPEED: 0.15,
    NetClass.DIFFERENTIAL: 0.15,
    NetClass.RF: 0.3,
    NetClass.ANALOG: 0.25,
    NetClass.BUS: 0.15,
    NetClass.I2C: 0.15,
    NetClass.SPI: 0.15,
    NetClass.HIGH_CURRENT: 0.3,
    NetClass.SIGNAL: 0.15,
}


# =============================================================================
# NET CLASSIFICATION PATTERNS
# =============================================================================

# Regex patterns for net name classification
NET_NAME_PATTERNS: List[Tuple[str, NetClass]] = [
    # Power nets
    (r'^VCC|^VDD|^VBAT|^VBUS|^V3V3|^V5V|^V12V|^VCORE|^VIN|^VOUT|^VREG', NetClass.POWER),
    (r'_VCC$|_VDD$|_PWR$|_POWER$', NetClass.POWER),

    # Ground nets
    (r'^GND|^AGND|^DGND|^PGND|^VSS|^EARTH', NetClass.GROUND),
    (r'_GND$|_VSS$', NetClass.GROUND),

    # High-speed nets
    (r'^CLK|^CLOCK|^XTAL|^OSC', NetClass.HIGH_SPEED),
    (r'_CLK$|_CLOCK$', NetClass.HIGH_SPEED),
    (r'^DDR|^SDRAM|^DRAM', NetClass.HIGH_SPEED),

    # Differential pairs (detect pairs)
    (r'^USB_D[+-]|^USB[+-]|^DP$|^DM$|^D[+-]$', NetClass.DIFFERENTIAL),
    (r'^HDMI|^LVDS|^ETH_TX|^ETH_RX', NetClass.DIFFERENTIAL),
    (r'_[PN]$|_DIFF[PN]$', NetClass.DIFFERENTIAL),

    # Bus signals
    (r'^DATA\[|^D\[|^ADDR\[|^A\[|^DB\[|^AB\[', NetClass.BUS),
    (r'_DATA\[|_D\[|_ADDR\[|_A\[', NetClass.BUS),
    (r'^DATA[0-9]+$|^D[0-9]+$|^ADDR[0-9]+$|^A[0-9]+$', NetClass.BUS),

    # I2C
    (r'^SDA|^SCL|^I2C', NetClass.I2C),
    (r'_SDA$|_SCL$', NetClass.I2C),

    # SPI
    (r'^MOSI|^MISO|^SCK|^SCLK|^CS$|^SS$|^SPI', NetClass.SPI),
    (r'_MOSI$|_MISO$|_SCK$|_SCLK$|_CS$|_SS$', NetClass.SPI),
    (r'^CS[0-9]$|^SS[0-9]$', NetClass.SPI),

    # Analog
    (r'^ADC|^AIN|^ANALOG|^VREF|^AREF', NetClass.ANALOG),
    (r'_ADC$|_AIN$|_ANALOG$', NetClass.ANALOG),

    # RF
    (r'^RF|^ANT|^ANTENNA', NetClass.RF),
    (r'_RF$|_ANT$', NetClass.RF),

    # High current
    (r'^MOTOR|^LED_PWR|^HEATER|^RELAY|^LOAD', NetClass.HIGH_CURRENT),
]


# =============================================================================
# ROUTING PLANNER
# =============================================================================

class RoutingPlanner:
    """
    Analyzes design and creates optimal routing plan.

    Acts as intelligence layer between Circuit AI and Routing Piston.
    Replaces blind cascade with smart per-net algorithm selection.
    """

    def __init__(self, learning_db: 'LearningDatabase' = None):
        """
        Initialize the routing planner.

        Args:
            learning_db: Optional learning database for historical data
        """
        self.learning_db = learning_db
        self._net_classes: Dict[str, NetClass] = {}
        self._trunk_chains: List[List[str]] = []

    # =========================================================================
    # MAIN API
    # =========================================================================

    def create_routing_plan(self, parts_db: Dict,
                           board_config: Dict) -> RoutingPlan:
        """
        Create complete routing plan for a design.

        This is the main entry point. It combines:
        1. Design profile analysis
        2. Net classification
        3. Algorithm selection
        4. Priority ordering
        5. Learning database queries

        Args:
            parts_db: Parts database with components and nets
            board_config: Board configuration (width, height, layers)

        Returns:
            Complete RoutingPlan with per-net strategies
        """
        # Step 1: Analyze design profile
        profile = self.analyze_design(parts_db, board_config)

        # Step 2: Classify all nets
        net_classes = self.classify_nets(parts_db)

        # Step 3: Detect trunk chains (parallel bus signals)
        trunk_chains = self._detect_trunk_chains(parts_db, net_classes)

        # Step 4: Create per-net strategies
        strategies = {}
        nets = parts_db.get('nets', {})

        for net_name in nets:
            net_class = net_classes.get(net_name, NetClass.SIGNAL)
            strategy = self._create_net_strategy(
                net_name, net_class, profile, trunk_chains
            )
            strategies[net_name] = strategy

        # Step 5: Determine routing order
        routing_order = self._calculate_routing_order(strategies)

        # Step 6: Query learning database (if available)
        if self.learning_db:
            self._apply_learning(strategies, profile)

        # Create the plan
        plan = RoutingPlan(
            design_profile=profile,
            net_strategies=strategies,
            routing_order=routing_order,
            enable_trunk_chains=len(trunk_chains) > 0,
            enable_return_path_check=profile.has_high_speed,
            ground_pour_recommended=profile.needs_ground_pour,
            trunk_chains=trunk_chains,
        )

        # Calculate success prediction
        plan.overall_success_prediction = self._predict_success(plan)

        return plan

    # =========================================================================
    # STEP 1: ANALYZE DESIGN
    # =========================================================================

    def analyze_design(self, parts_db: Dict, board_config: Dict) -> DesignProfile:
        """
        Analyze design characteristics.

        Examines:
        - Board dimensions and layer count
        - Component types and counts
        - Package complexity
        - Net characteristics
        - Special requirements
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        board_width = board_config.get('board_width', 50.0)
        board_height = board_config.get('board_height', 40.0)
        board_area = board_width * board_height
        layer_count = board_config.get('layers', 2)

        # Count component types
        ic_count = 0
        passive_count = 0
        connector_count = 0
        has_bga = False
        has_qfn = False
        has_fine_pitch = False
        max_pin_count = 0
        total_component_area = 0.0

        for ref, part_info in parts.items():
            footprint = part_info.get('footprint', '').upper()
            pins = part_info.get('pins', [])
            pin_count = len(pins)
            max_pin_count = max(max_pin_count, pin_count)

            # Classify component
            if ref.startswith('U') or ref.startswith('IC'):
                ic_count += 1
            elif ref.startswith(('R', 'C', 'L')):
                passive_count += 1
            elif ref.startswith(('J', 'P', 'CON')):
                connector_count += 1

            # Check package type
            if 'BGA' in footprint:
                has_bga = True
            if 'QFN' in footprint or 'DFN' in footprint:
                has_qfn = True
            if '0201' in footprint or '01005' in footprint:
                has_fine_pitch = True

            # Estimate component area
            comp_area = self._estimate_component_area(footprint, pin_count)
            total_component_area += comp_area

        component_count = len(parts)
        area_utilization = (total_component_area / board_area) * 100 if board_area > 0 else 0

        # Determine density
        if area_utilization < 20:
            density = DesignDensity.SPARSE
        elif area_utilization < 40:
            density = DesignDensity.LOW
        elif area_utilization < 60:
            density = DesignDensity.MEDIUM
        elif area_utilization < 80:
            density = DesignDensity.HIGH
        else:
            density = DesignDensity.VERY_HIGH

        # Analyze nets
        net_count = len(nets)
        max_fanout = 0
        has_high_speed = False
        has_differential = False

        for net_name, net_info in nets.items():
            pins = net_info.get('pins', [])
            fanout = len(pins)
            max_fanout = max(max_fanout, fanout)

            # Check for high-speed/differential from name
            net_upper = net_name.upper()
            if any(x in net_upper for x in ['CLK', 'CLOCK', 'DDR', 'USB', 'HDMI']):
                has_high_speed = True
            if any(x in net_upper for x in ['_P', '_N', 'D+', 'D-', 'DP', 'DM']):
                has_differential = True

        # Determine if ground pour is recommended
        gnd_pin_count = 0
        for net_name, net_info in nets.items():
            if net_name.upper() in ['GND', 'VSS', 'AGND', 'DGND']:
                gnd_pin_count = len(net_info.get('pins', []))
                break

        # Ground pour recommended for 2-layer with many GND pins
        needs_ground_pour = (layer_count == 2 and gnd_pin_count >= 5) or has_high_speed

        # Generate recommendations
        recommendations = []
        warnings = []

        if has_bga and layer_count < 4:
            warnings.append("BGA package may require 4+ layers for escape routing")

        if density == DesignDensity.VERY_HIGH:
            warnings.append("Very high density - consider larger board or more layers")
            recommendations.append("pathfinder")  # Congestion-aware

        if has_high_speed:
            recommendations.append("steiner")  # Optimal for high-speed
            if not needs_ground_pour:
                warnings.append("High-speed signals need solid ground reference")

        return DesignProfile(
            board_width=board_width,
            board_height=board_height,
            board_area=board_area,
            layer_count=layer_count,
            component_count=component_count,
            ic_count=ic_count,
            passive_count=passive_count,
            connector_count=connector_count,
            has_bga=has_bga,
            has_qfn=has_qfn,
            has_fine_pitch=has_fine_pitch,
            max_pin_count=max_pin_count,
            net_count=net_count,
            max_fanout=max_fanout,
            density=density,
            area_utilization=area_utilization,
            has_high_speed=has_high_speed,
            has_differential=has_differential,
            needs_ground_pour=needs_ground_pour,
            recommended_algorithms=recommendations,
            warnings=warnings,
        )

    def _estimate_component_area(self, footprint: str, pin_count: int) -> float:
        """Estimate component area in mm^2 from footprint string"""
        footprint = footprint.upper()

        # Common footprint areas (approximate with courtyard)
        if '0201' in footprint:
            return 0.5 * 0.3
        elif '0402' in footprint:
            return 1.5 * 1.0
        elif '0603' in footprint:
            return 2.1 * 1.3
        elif '0805' in footprint:
            return 2.5 * 1.75
        elif '1206' in footprint:
            return 3.7 * 2.1
        elif 'SOT-23' in footprint or 'SOT23' in footprint:
            return 3.4 * 1.8
        elif 'SOIC' in footprint or 'SOP' in footprint:
            return 6.0 * 5.0
        elif 'QFP' in footprint:
            # Estimate from pin count
            side = math.ceil(math.sqrt(pin_count)) * 0.5 + 2
            return side * side
        elif 'QFN' in footprint or 'DFN' in footprint:
            # Estimate from pin count
            side = math.ceil(math.sqrt(pin_count)) * 0.5 + 1
            return side * side
        elif 'BGA' in footprint:
            # Estimate from pin count
            side = math.ceil(math.sqrt(pin_count)) * 0.8 + 2
            return side * side
        else:
            # Default estimate based on pin count
            return max(4, pin_count * 0.5)

    # =========================================================================
    # STEP 2: CLASSIFY NETS
    # =========================================================================

    def classify_nets(self, parts_db: Dict) -> Dict[str, NetClass]:
        """
        Classify every net by its function.

        Uses:
        - Net naming conventions (VCC, GND, CLK, etc.)
        - Connected component types
        - User-provided net classes (if available)
        """
        nets = parts_db.get('nets', {})
        net_classes = {}

        for net_name in nets:
            net_class = self._classify_net(net_name, parts_db)
            net_classes[net_name] = net_class

        self._net_classes = net_classes
        return net_classes

    def _classify_net(self, net_name: str, parts_db: Dict) -> NetClass:
        """Classify a single net"""
        net_upper = net_name.upper()

        # First, check for user-defined class in parts_db
        nets = parts_db.get('nets', {})
        net_info = nets.get(net_name, {})
        if 'class' in net_info:
            try:
                return NetClass(net_info['class'])
            except ValueError:
                pass

        # Check against patterns
        for pattern, net_class in NET_NAME_PATTERNS:
            if re.search(pattern, net_upper):
                return net_class

        # Default to SIGNAL
        return NetClass.SIGNAL

    # =========================================================================
    # STEP 3: DETECT TRUNK CHAINS
    # =========================================================================

    def _detect_trunk_chains(self, parts_db: Dict,
                            net_classes: Dict[str, NetClass]) -> List[List[str]]:
        """
        Detect groups of nets that should be routed as parallel buses.

        Looks for:
        - Numbered net sequences (DATA0, DATA1, DATA2...)
        - Nets connecting same source/destination
        - Explicitly marked bus signals
        """
        nets = parts_db.get('nets', {})
        chains = []
        used_nets = set()

        # Find numbered sequences
        numbered_patterns = [
            (r'^(.+?)(\d+)$', 'suffix'),  # NAME0, NAME1, NAME2
            (r'^(.+?)\[(\d+)\]$', 'bracket'),  # NAME[0], NAME[1]
        ]

        groups = defaultdict(list)

        for net_name in nets:
            if net_name in used_nets:
                continue

            for pattern, style in numbered_patterns:
                match = re.match(pattern, net_name)
                if match:
                    base_name = match.group(1)
                    index = int(match.group(2))
                    groups[base_name].append((index, net_name))
                    break

        # Create chains from groups with 3+ members
        for base_name, members in groups.items():
            if len(members) >= 3:
                # Sort by index
                members.sort(key=lambda x: x[0])
                chain = [name for _, name in members]
                chains.append(chain)
                used_nets.update(chain)

        self._trunk_chains = chains
        return chains

    # =========================================================================
    # STEP 4: CREATE NET STRATEGIES
    # =========================================================================

    def _create_net_strategy(self, net_name: str, net_class: NetClass,
                            profile: DesignProfile,
                            trunk_chains: List[List[str]]) -> NetRoutingStrategy:
        """Create routing strategy for a single net"""

        # Get algorithm from selection matrix
        primary, fallbacks = ALGORITHM_SELECTION_MATRIX.get(
            net_class,
            (RoutingAlgorithm.HYBRID, [RoutingAlgorithm.A_STAR])
        )

        # Get constraints for this class
        trace_width = NET_CLASS_TRACE_WIDTH.get(net_class, 0.25)
        clearance = NET_CLASS_CLEARANCE.get(net_class, 0.15)

        # Get priority
        priority = NET_CLASS_PRIORITY.get(net_class, 11)

        # Check if part of trunk chain
        trunk_chain_id = None
        for idx, chain in enumerate(trunk_chains):
            if net_name in chain:
                trunk_chain_id = idx
                break

        # Enable push-and-shove as last resort for complex designs
        use_push_and_shove = profile.density in [
            DesignDensity.HIGH, DesignDensity.VERY_HIGH
        ]

        # Check return path for high-speed
        check_return_path = net_class in [
            NetClass.HIGH_SPEED, NetClass.DIFFERENTIAL, NetClass.RF
        ]

        # Length match group for differential pairs
        length_match_group = None
        if net_class == NetClass.DIFFERENTIAL:
            # Extract base name for pairing
            for suffix in ['_P', '_N', '+', '-', 'P', 'N']:
                if net_name.endswith(suffix):
                    length_match_group = net_name[:-len(suffix)]
                    break

        return NetRoutingStrategy(
            net_name=net_name,
            net_class=net_class,
            priority=priority,
            primary_algorithm=primary,
            fallback_algorithms=list(fallbacks),
            trace_width=trace_width,
            clearance=clearance,
            use_push_and_shove=use_push_and_shove,
            trunk_chain_id=trunk_chain_id,
            check_return_path=check_return_path,
            length_match_group=length_match_group,
        )

    # =========================================================================
    # STEP 5: CALCULATE ROUTING ORDER
    # =========================================================================

    def _calculate_routing_order(self,
                                strategies: Dict[str, NetRoutingStrategy]) -> List[str]:
        """
        Calculate optimal routing order.

        Order is based on:
        1. Net class priority (power first, general signals last)
        2. Criticality (high-speed before low-speed)
        3. Connectivity (route highly-connected nets early)
        """
        # Sort by priority, then by name for stability
        sorted_nets = sorted(
            strategies.keys(),
            key=lambda n: (strategies[n].priority, n)
        )
        return sorted_nets

    # =========================================================================
    # STEP 6: APPLY LEARNING
    # =========================================================================

    def _apply_learning(self, strategies: Dict[str, NetRoutingStrategy],
                       profile: DesignProfile) -> None:
        """Apply learning from past routing successes"""
        if not self.learning_db:
            return

        for net_name, strategy in strategies.items():
            # Query learning database for best algorithm
            learned = self.learning_db.get_best_algorithm(
                strategy.net_class, profile
            )

            if learned:
                strategy.learned_algorithm = learned.value
                strategy.learned_success_rate = self.learning_db.get_success_rate(
                    learned, strategy.net_class
                )

                # If learned algorithm has >90% success, use it as primary
                if strategy.learned_success_rate > 0.9:
                    old_primary = strategy.primary_algorithm
                    strategy.primary_algorithm = learned
                    # Add old primary to fallbacks
                    if old_primary not in strategy.fallback_algorithms:
                        strategy.fallback_algorithms.insert(0, old_primary)

    def _predict_success(self, plan: RoutingPlan) -> float:
        """Predict overall routing success probability"""
        if not plan.net_strategies:
            return 0.0

        # Base prediction on design profile
        base = 0.7  # Default 70%

        # Adjust for density
        density_factor = {
            DesignDensity.SPARSE: 1.0,
            DesignDensity.LOW: 0.95,
            DesignDensity.MEDIUM: 0.85,
            DesignDensity.HIGH: 0.70,
            DesignDensity.VERY_HIGH: 0.50,
        }
        base *= density_factor.get(plan.design_profile.density, 0.8)

        # Boost for learned algorithms
        learned_count = sum(
            1 for s in plan.net_strategies.values()
            if s.learned_algorithm
        )
        if plan.net_strategies:
            learned_ratio = learned_count / len(plan.net_strategies)
            base += learned_ratio * 0.15  # Up to 15% boost

        # Boost for ground pour on 2-layer
        if plan.ground_pour_recommended and plan.design_profile.layer_count == 2:
            base += 0.05

        return min(0.95, base)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_summary(self, plan: RoutingPlan) -> str:
        """Get human-readable summary of routing plan"""
        lines = []
        lines.append("=" * 60)
        lines.append("ROUTING PLAN SUMMARY")
        lines.append("=" * 60)

        # Design profile
        p = plan.design_profile
        lines.append(f"\nBoard: {p.board_width}x{p.board_height}mm, {p.layer_count} layers")
        lines.append(f"Components: {p.component_count} ({p.ic_count} ICs, {p.passive_count} passives)")
        lines.append(f"Density: {p.density.value} ({p.area_utilization:.1f}% utilization)")
        lines.append(f"Nets: {p.net_count}")

        # Net classification
        lines.append("\nNet Classification:")
        class_counts = defaultdict(int)
        for strategy in plan.net_strategies.values():
            class_counts[strategy.net_class] += 1

        for net_class in NetClass:
            count = class_counts.get(net_class, 0)
            if count > 0:
                lines.append(f"  {net_class.value:15} : {count:3} nets")

        # Algorithm distribution
        lines.append("\nAlgorithm Selection:")
        algo_dist = plan.get_algorithm_distribution()
        for algo, count in sorted(algo_dist.items(), key=lambda x: -x[1]):
            lines.append(f"  {algo:15} : {count:3} nets")

        # Special features
        lines.append("\nFeatures Enabled:")
        lines.append(f"  Trunk chains: {'Yes' if plan.enable_trunk_chains else 'No'}")
        if plan.trunk_chains:
            lines.append(f"    {len(plan.trunk_chains)} chains detected")
        lines.append(f"  Return path check: {'Yes' if plan.enable_return_path_check else 'No'}")
        lines.append(f"  Ground pour: {'Recommended' if plan.ground_pour_recommended else 'Not needed'}")

        # Success prediction
        lines.append(f"\nPredicted Success: {plan.overall_success_prediction*100:.0f}%")

        # Warnings
        if p.warnings:
            lines.append("\nWarnings:")
            for warning in p.warnings:
                lines.append(f"  ! {warning}")

        return "\n".join(lines)


# =============================================================================
# STANDALONE USAGE
# =============================================================================

def create_routing_plan(parts_db: Dict, board_config: Dict,
                       learning_db: 'LearningDatabase' = None) -> RoutingPlan:
    """
    Convenience function to create routing plan.

    Args:
        parts_db: Parts database
        board_config: Board configuration
        learning_db: Optional learning database

    Returns:
        RoutingPlan
    """
    planner = RoutingPlanner(learning_db)
    return planner.create_routing_plan(parts_db, board_config)


# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    'RoutingPlanner',
    'RoutingPlan',
    'NetRoutingStrategy',
    'DesignProfile',
    'NetClass',
    'RoutingAlgorithm',
    'DesignDensity',
    'create_routing_plan',
]
