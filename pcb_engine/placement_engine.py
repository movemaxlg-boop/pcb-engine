"""
PCB Placement Engine - Multi-Algorithm Component Placement
===========================================================

This dedicated placement engine provides multiple algorithms for optimal
component placement on PCB boards.

Algorithms:
1. Force-Directed (FD) - Fast global placement using spring physics
   - Fruchterman-Reingold with adaptive cooling
   - Attractive forces from nets (Hooke's law)
   - Repulsive forces between components (Coulomb's law)

2. Simulated Annealing (SA) - Refinement with probabilistic optimization
   - Adaptive cooling schedule
   - Multiple move operators (shift, swap, rotate, flip)
   - Reheating when stuck

3. Genetic Algorithm (GA) - Evolutionary optimization
   - Order crossover (OX) for permutation representation
   - Adaptive mutation rates
   - Diversity preservation with crowding

4. Analytical (Quadratic) - Mathematical optimization
   - Quadratic wire length minimization
   - Cell spreading to avoid overlap
   - Conjugate gradient solver

5. Human-Like - Rule-based placement mimicking human designers
   - Power distribution awareness
   - Decoupling capacitor placement rules
   - Signal flow optimization

The recommended approach is:
  FD (global) -> SA (refinement) -> Legalization

Usage:
    from placement_engine import PlacementEngine, PlacementConfig

    config = PlacementConfig(
        algorithm='hybrid',  # 'fd', 'sa', 'ga', 'analytical', 'human', 'hybrid', 'auto', 'parallel'
        board_width=50.0,
        board_height=40.0,
    )

    engine = PlacementEngine(config)
    placements = engine.place(parts_db, graph)
"""

import math
import random
import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import common helpers for consistent pin access (BUG FIX: use get_pins instead of .get('used_pins'))
try:
    from .common_types import get_pins
    from .footprint_resolver import FootprintResolver
except ImportError:
    from common_types import get_pins
    from footprint_resolver import FootprintResolver


class PlacementAlgorithm(Enum):
    FORCE_DIRECTED = 'fd'
    SIMULATED_ANNEALING = 'sa'
    GENETIC = 'ga'
    ANALYTICAL = 'analytical'
    HUMAN = 'human'
    HYBRID = 'hybrid'  # FD + SA combined
    AUTO = 'auto'      # Try multiple sequentially, pick best
    PARALLEL = 'parallel'  # Run ALL algorithms in parallel, pick best


@dataclass
class PlacementConfig:
    """Configuration for placement engine"""
    # Board parameters
    board_width: float = 50.0
    board_height: float = 40.0
    auto_board_size: bool = False  # Auto-calculate from component area
    origin_x: float = 0.0  # Board starts at origin (was 100.0 which caused out-of-bounds)
    origin_y: float = 0.0
    grid_size: float = 0.5

    # Algorithm selection
    algorithm: str = 'hybrid'

    # Force-directed parameters (Fruchterman-Reingold based)
    fd_iterations: int = 200          # More iterations for convergence
    fd_initial_temp: float = 10.0     # Initial temperature (controls movement)
    fd_cooling_factor: float = 0.95   # Temperature reduction per iteration
    fd_attraction_k: float = 1.0      # Attraction constant (spring constant)
    fd_repulsion_k: float = 10000.0   # Repulsion constant (Coulomb constant)
    fd_min_temp: float = 0.01         # Minimum temperature before stopping

    # Simulated annealing parameters (adaptive Markov chain)
    sa_initial_temp: float = 200.0    # Higher initial temp for exploration
    sa_final_temp: float = 0.01       # Lower final temp for convergence
    sa_moves_per_temp: int = 100      # Moves per temperature step
    sa_cooling_rate: float = 0.97     # Slower cooling for better results
    sa_reheat_threshold: int = 50     # Reheat after this many rejections
    sa_reheat_factor: float = 1.5     # Multiply temp by this when reheating

    # Genetic algorithm parameters (NSGA-II inspired)
    ga_population_size: int = 100     # Larger population for diversity
    ga_generations: int = 150         # More generations
    ga_elite_ratio: float = 0.1       # Top 10% always survive
    ga_crossover_rate: float = 0.8    # 80% crossover probability
    ga_mutation_rate: float = 0.15    # 15% mutation probability
    ga_tournament_size: int = 5       # Tournament selection size
    ga_diversity_threshold: float = 0.1  # Minimum diversity to maintain

    # Analytical placement parameters
    an_iterations: int = 50           # CG iterations
    an_spreading_factor: float = 0.5  # Spreading force strength
    an_anchor_weight: float = 0.1     # Weight for anchoring to center

    # Component spacing and margins
    min_spacing: float = 0.5          # Minimum spacing between components
    edge_margin: float = 2.0          # Margin from board edge
    routing_channel_width: float = 1.0  # Width reserved for routing channels

    # Random seed for reproducibility
    seed: Optional[int] = 42

    # Parallel execution
    parallel_workers: int = 6         # Number of parallel workers
    parallel_timeout: float = 120.0   # Timeout per algorithm in seconds

    # Net weights for prioritization
    power_net_weight: float = 0.05    # Power nets connected by pour, not traces — minimal attraction
    critical_net_weight: float = 3.0  # Weight for critical signal nets

    # Component Fusion — absorb functional passives into owner IC before placement
    fusion_enabled: bool = True          # Fuse caps/passives onto their owner IC
    fusion_priority_threshold: float = 1.5  # Min functional priority to fuse (DECOUPLING=3, ESD=2.5, PULLUP=1.5)
    fusion_max_per_owner: int = 6        # Max passives fused per IC (excess stay independent)


@dataclass
class ComponentState:
    """State of a component during placement"""
    ref: str
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    fixed: bool = False  # True for connectors at fixed positions

    # Physics state (for force-directed)
    vx: float = 0.0
    vy: float = 0.0
    fx: float = 0.0  # Accumulated force
    fy: float = 0.0

    # Component properties
    pin_count: int = 0
    is_decoupling_cap: bool = False
    is_power_component: bool = False

    def copy(self) -> 'ComponentState':
        """Create a deep copy of this state"""
        return ComponentState(
            ref=self.ref,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            rotation=self.rotation,
            fixed=self.fixed,
            vx=self.vx,
            vy=self.vy,
            fx=self.fx,
            fy=self.fy,
            pin_count=self.pin_count,
            is_decoupling_cap=self.is_decoupling_cap,
            is_power_component=self.is_power_component
        )


@dataclass
class PlacementResult:
    """Result of placement"""
    positions: Dict[str, Tuple[float, float]]
    rotations: Dict[str, float]
    cost: float
    algorithm_used: str
    iterations: int
    converged: bool
    wirelength: float = 0.0
    overlap_area: float = 0.0
    success: bool = True  # For compatibility with engine
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    board_width: float = 0.0   # Final board width (after auto-sizing/shrink)
    board_height: float = 0.0  # Final board height (after auto-sizing/shrink)


class OccupancyGrid:
    """
    2D boolean grid tracking which board cells are occupied by component courtyards.
    Cell size = 0.5mm for precise overlap detection.

    Every placement move is validated against this grid — overlaps are
    physically impossible, not just penalized.
    """

    def __init__(self, board_width: float, board_height: float, cell_size: float = 0.5):
        self.cell_size = cell_size
        self.cols = int(math.ceil(board_width / cell_size)) + 1
        self.rows = int(math.ceil(board_height / cell_size)) + 1
        self.grid = [[False] * self.cols for _ in range(self.rows)]
        self.owner = [[None] * self.cols for _ in range(self.rows)]

    def _rect_cells(self, cx: float, cy: float, width: float, height: float):
        """Convert component center + courtyard size to cell range.
        Uses floor for min edges and ceil for max edges to conservatively
        round OUT — ensures no sub-cell overlaps sneak through."""
        half_w, half_h = width / 2, height / 2
        c_min = max(0, int(math.floor((cx - half_w) / self.cell_size)))
        c_max = min(self.cols - 1, int(math.ceil((cx + half_w) / self.cell_size)))
        r_min = max(0, int(math.floor((cy - half_h) / self.cell_size)))
        r_max = min(self.rows - 1, int(math.ceil((cy + half_h) / self.cell_size)))
        return r_min, r_max, c_min, c_max

    def can_place(self, cx: float, cy: float, width: float, height: float,
                  ignore_ref: str = None) -> bool:
        """Check if courtyard rectangle fits entirely in free cells."""
        r_min, r_max, c_min, c_max = self._rect_cells(cx, cy, width, height)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if self.grid[r][c] and self.owner[r][c] != ignore_ref:
                    return False
        return True

    def place(self, ref: str, cx: float, cy: float, width: float, height: float):
        """Mark courtyard cells as occupied by ref."""
        r_min, r_max, c_min, c_max = self._rect_cells(cx, cy, width, height)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                self.grid[r][c] = True
                self.owner[r][c] = ref

    def remove(self, ref: str, cx: float, cy: float, width: float, height: float):
        """Free courtyard cells owned by ref."""
        r_min, r_max, c_min, c_max = self._rect_cells(cx, cy, width, height)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if self.owner[r][c] == ref:
                    self.grid[r][c] = False
                    self.owner[r][c] = None

    def clear(self):
        """Reset entire grid to empty."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = False
                self.owner[r][c] = None

    def save_state(self):
        """Snapshot for SA rollback."""
        return ([row[:] for row in self.grid], [row[:] for row in self.owner])

    def restore_state(self, state):
        """Restore from snapshot."""
        saved_grid, saved_owner = state
        for r in range(self.rows):
            self.grid[r] = saved_grid[r][:]
            self.owner[r] = saved_owner[r][:]


class PlacementEngine:
    """
    Multi-algorithm PCB placement engine.

    Provides multiple placement strategies that can be used individually
    or combined (hybrid approach).
    """

    def __init__(self, config: PlacementConfig = None):
        self.config = config or PlacementConfig()

        # Set random seed for reproducibility
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # State
        self.components: Dict[str, ComponentState] = {}
        self.nets: Dict[str, Dict] = {}  # net_name -> {pins: [(comp, pin), ...], weight: float}
        self.adjacency: Dict[str, Dict[str, int]] = {}
        self.hub: Optional[str] = None

        # Pin positions (relative to component center)
        self.pin_offsets: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Pin sizes for pad conflict detection
        self.pin_sizes: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Pin nets for conflict detection
        self.pin_nets: Dict[str, Dict[str, str]] = {}

        # Net weights for prioritization
        self.net_weights: Dict[str, float] = {}

        # Optimal distance for force-directed (computed from board size)
        self._optimal_dist = 0.0

        # Functional roles and orientation (computed during placement)
        self._functional_roles: Dict[str, Dict] = {}
        self._preferred_passive_orientation: Optional[float] = None
        self._small_passive_refs: set = set()

        # Component fusion state (set during place() if fusion_enabled)
        self._fusion = None

        # JIT acceleration (set during place() if numba available)
        self._jit = None

    @staticmethod
    def _parse_pin_ref(pin_ref) -> Tuple[str, str]:
        """Parse pin reference to (component, pin) tuple."""
        if isinstance(pin_ref, str):
            parts = pin_ref.split('.')
            return (parts[0], parts[1]) if len(parts) >= 2 else (parts[0] if parts else '', '')
        elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
            return (str(pin_ref[0]), str(pin_ref[1]))
        elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) == 1:
            return (str(pin_ref[0]), '')
        return ('', '')

    def place(self, parts_db: Dict, graph: Dict,
              placement_hints: Dict = None) -> PlacementResult:
        """
        Run placement with configured algorithm.

        Args:
            parts_db: Parts database from collector
            graph: Connectivity graph
            placement_hints: Optional hints from c_layout or CPU Lab
                             Keys: proximity_groups, edge_components, keep_apart, zones

        Returns:
            PlacementResult with positions and metrics
        """
        # Store placement hints for use by cost function
        self._placement_hints = placement_hints or {}

        # Auto-calculate board size when flagged and no explicit dims
        if self.config.auto_board_size:
            try:
                from .common_types import calculate_board_size
            except ImportError:
                from common_types import calculate_board_size
            auto_w, auto_h = calculate_board_size(parts_db)
            self.config.board_width = auto_w
            self.config.board_height = auto_h
            print(f"  [PLACEMENT] Board AUTO-SIZED: {auto_w}x{auto_h}mm")

        # Infer functional roles and build placement constraints (before placement)
        self._build_placement_constraints(parts_db)

        # Initialize state from parts database
        self._init_from_parts(parts_db, graph)

        # Create occupancy grid — makes overlaps physically impossible
        self._occ = OccupancyGrid(self.config.board_width, self.config.board_height)

        # --- FUSE: Absorb functional passives into their owner IC ---
        self._fusion = None
        if self.config.fusion_enabled and self._functional_roles:
            from .component_fusion import ComponentFusion
            self._fusion = ComponentFusion(
                self.config.fusion_priority_threshold,
                self.config.fusion_max_per_owner)
            self._fusion.fuse(self)

        # Compute optimal distance for force-directed
        n = len(self.components)
        if n > 0:
            area = self.config.board_width * self.config.board_height
            self._optimal_dist = math.sqrt(area / n) * 0.8

        # Initialize JIT acceleration (if numba available)
        self._jit = None
        try:
            from .numba_accel import accel_available, JITCostEvaluator
            if accel_available:
                self._jit = JITCostEvaluator.from_engine(self)
                print("  [JIT] Numba acceleration enabled")
        except Exception:
            pass  # Graceful fallback to pure Python

        # Select algorithm
        algo = PlacementAlgorithm(self.config.algorithm)

        if algo == PlacementAlgorithm.FORCE_DIRECTED:
            result = self._place_force_directed()
        elif algo == PlacementAlgorithm.SIMULATED_ANNEALING:
            result = self._place_simulated_annealing()
        elif algo == PlacementAlgorithm.GENETIC:
            result = self._place_genetic()
        elif algo == PlacementAlgorithm.ANALYTICAL:
            result = self._place_analytical()
        elif algo == PlacementAlgorithm.HUMAN:
            result = self._place_human_like()
        elif algo == PlacementAlgorithm.HYBRID:
            result = self._place_hybrid()
        elif algo == PlacementAlgorithm.AUTO:
            result = self._place_auto()
        elif algo == PlacementAlgorithm.PARALLEL:
            result = self._place_parallel()
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        # --- UNFUSE: Restore passives as independent components on IC perimeter ---
        if self._fusion and self._fusion.fused_components:
            self._jit = None  # Component set changes — invalidate JIT snapshot
            self._fusion.unfuse(self)
            # Post-unfuse WL recovery: jitter unfused passives to reduce wirelength
            # while keeping them within leash distance of their owner IC
            self._post_unfuse_jitter()
            result = self._create_result(result.algorithm_used, result.iterations, result.converged)

        # Post-placement: shrink board to fit when auto-sizing
        if self.config.auto_board_size and self.components:
            self._shrink_board_to_fit()
            # Rebuild result with updated positions and board dims
            result = self._create_result(result.algorithm_used, result.iterations, result.converged)

        return result

    def _build_placement_constraints(self, parts_db: Dict):
        """Infer functional roles from net topology and build placement constraints.

        General-purpose algorithm — works for ANY board by analyzing what
        each component DOES in the circuit (decoupling, pull-up, ESD, etc.)
        rather than matching hardcoded ref prefixes.
        """
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})
        groups = list(self._placement_hints.get('proximity_groups', []))
        edge = list(self._placement_hints.get('edge_components', []))

        # --- Helpers ---
        def _part_nets(part):
            return {pin.get('net', '') for pin in get_pins(part) if pin.get('net', '')}

        def _signal_nets(part):
            return {n for n in _part_nets(part) if not self._is_power_net(n) and n != 'GND'}

        def _is_ground(net):
            return net.upper() in ('GND', 'AGND', 'DGND', 'PGND', 'VSS', 'AVSS')

        def _is_power(net):
            return self._is_power_net(net) and not _is_ground(net)

        # --- Classify components ---
        ic_refs = set()       # multi-pin ICs (8+ pins)
        passive_refs = set()  # 2-pin passives (R, C, L)
        connector_refs = set()
        led_refs = set()

        for ref, part in parts.items():
            pin_count = len(get_pins(part))
            fp = part.get('footprint', '').upper()
            name = part.get('name', '').upper()

            if ref.startswith('J') or 'USB' in fp or 'CONN' in fp or 'HDR' in fp:
                connector_refs.add(ref)
            elif ref.startswith('LED') or 'LED' in name:
                led_refs.add(ref)
            elif pin_count >= 6 and ref.startswith('U'):
                ic_refs.add(ref)
            elif pin_count <= 2 and ref[0] in ('R', 'C', 'L'):
                passive_refs.add(ref)

        # --- Build net → component index ---
        net_to_refs = {}  # net_name -> set of refs on that net
        for ref, part in parts.items():
            for net in _part_nets(part):
                if net:
                    net_to_refs.setdefault(net, set()).add(ref)

        # --- Helper: compute physical minimum distance between two parts ---
        def _min_proximity(ref_a, ref_b):
            """Physical minimum center-to-center distance (courtyard-based)."""
            gap = 0.3  # mm assembly gap
            for r in (ref_a, ref_b):
                if r not in parts:
                    return 3.0  # fallback
            sizes = []
            for r in (ref_a, ref_b):
                court = parts[r].get('courtyard')
                if court:
                    if isinstance(court, dict):
                        sizes.append(max(court.get('width', 2), court.get('height', 2)))
                    elif hasattr(court, 'width'):
                        sizes.append(max(court.width, court.height))
                    else:
                        sizes.append(2.0)
                else:
                    fp_name = parts[r].get('footprint', 'unknown')
                    resolver = FootprintResolver.get_instance()
                    fp_def = resolver.resolve(fp_name)
                    cw, ch = fp_def.courtyard_size
                    sizes.append(max(cw, ch))
            return sizes[0] / 2 + gap + sizes[1] / 2

        # --- Infer functional roles for each passive ---
        roles = {}  # ref -> {role, owner, distance, priority, reason}

        for ref in sorted(passive_refs):
            part = parts[ref]
            pins = get_pins(part)
            if len(pins) < 2:
                continue

            net1 = pins[0].get('net', '')
            net2 = pins[1].get('net', '')
            if not net1 or not net2:
                continue

            # --- DECOUPLING CAP: power + GND ---
            if ref.startswith('C'):
                power_pin = None
                gnd_pin = None
                if _is_power(net1) and _is_ground(net2):
                    power_pin, gnd_pin = net1, net2
                elif _is_power(net2) and _is_ground(net1):
                    power_pin, gnd_pin = net2, net1

                if power_pin and gnd_pin:
                    # Find IC with most pins on this power net
                    best_ic = None
                    best_count = 0
                    for ic in ic_refs:
                        ic_part_nets = _part_nets(parts[ic])
                        if power_pin in ic_part_nets:
                            count = sum(1 for p in get_pins(parts[ic])
                                        if p.get('net', '') == power_pin)
                            if count > best_count:
                                best_count = count
                                best_ic = ic
                    if best_ic:
                        min_dist = _min_proximity(ref, best_ic)
                        roles[ref] = {'role': 'DECOUPLING', 'owner': best_ic,
                                      'distance': min_dist, 'priority': 3.0,
                                      'reason': f'Decoupling cap for {best_ic} ({power_pin})'}
                    continue

            # --- PULL-UP: power + signal ---
            if ref.startswith('R'):
                power_net = None
                signal_net = None
                if _is_power(net1) and not _is_ground(net2) and not _is_power(net2):
                    power_net, signal_net = net1, net2
                elif _is_power(net2) and not _is_ground(net1) and not _is_power(net1):
                    power_net, signal_net = net2, net1

                if power_net and signal_net:
                    # Find which IC uses this signal
                    owner_ic = None
                    for ic in ic_refs:
                        if signal_net in _part_nets(parts[ic]):
                            owner_ic = ic
                            break
                    if owner_ic:
                        min_dist = _min_proximity(ref, owner_ic)
                        roles[ref] = {'role': 'PULLUP', 'owner': owner_ic,
                                      'distance': max(min_dist, 5.0), 'priority': 1.5,
                                      'reason': f'Pull-up for {owner_ic}.{signal_net}'}
                    continue

                # --- PULL-DOWN: GND + signal ---
                gnd_net = None
                signal_net = None
                if _is_ground(net1) and not _is_power(net2):
                    gnd_net, signal_net = net1, net2
                elif _is_ground(net2) and not _is_power(net1):
                    gnd_net, signal_net = net2, net1

                if gnd_net and signal_net:
                    owner_ic = None
                    for ic in ic_refs:
                        if signal_net in _part_nets(parts[ic]):
                            owner_ic = ic
                            break
                    if owner_ic:
                        min_dist = _min_proximity(ref, owner_ic)
                        roles[ref] = {'role': 'PULLDOWN', 'owner': owner_ic,
                                      'distance': max(min_dist, 5.0), 'priority': 1.5,
                                      'reason': f'Pull-down for {owner_ic}.{signal_net}'}
                    continue

            # --- LED DRIVER: signal → LED ---
            if ref.startswith('R'):
                # Check if one net connects to an IC and the other to an LED
                refs_on_net1 = net_to_refs.get(net1, set())
                refs_on_net2 = net_to_refs.get(net2, set())
                ic_on_1 = refs_on_net1 & ic_refs
                ic_on_2 = refs_on_net2 & ic_refs
                led_on_1 = refs_on_net1 & led_refs
                led_on_2 = refs_on_net2 & led_refs

                if ic_on_1 and led_on_2:
                    led = sorted(led_on_2)[0]
                    roles[ref] = {'role': 'LED_DRIVER', 'owner': led,
                                  'distance': 5.0, 'priority': 2.0,
                                  'reason': f'Current limiter for {led}'}
                    continue
                elif ic_on_2 and led_on_1:
                    led = sorted(led_on_1)[0]
                    roles[ref] = {'role': 'LED_DRIVER', 'owner': led,
                                  'distance': 5.0, 'priority': 2.0,
                                  'reason': f'Current limiter for {led}'}
                    continue

        # --- ESD PROTECTION: IC sharing signal nets with connector AND another IC ---
        for ic in sorted(ic_refs):
            ic_signals = _signal_nets(parts[ic])
            if not ic_signals:
                continue

            shared_with_connector = set()
            shared_with_other_ic = set()

            for conn in connector_refs:
                shared = ic_signals & _signal_nets(parts[conn])
                if shared:
                    shared_with_connector.add(conn)

            for other_ic in ic_refs:
                if other_ic == ic:
                    continue
                shared = ic_signals & _signal_nets(parts[other_ic])
                if shared:
                    shared_with_other_ic.add(other_ic)

            # Small IC (< 8 pins) that bridges connector and main IC → ESD
            pin_count = len(get_pins(parts[ic]))
            if shared_with_connector and shared_with_other_ic and pin_count <= 8:
                conn = sorted(shared_with_connector)[0]
                roles[ic] = {'role': 'ESD_PROTECTION', 'owner': conn,
                             'distance': 5.0, 'priority': 2.5,
                             'reason': f'ESD protection near {conn}',
                             'between': (conn, sorted(shared_with_other_ic)[0])}

        # --- Convert roles to proximity_groups ---
        for ref, role in roles.items():
            owner = role['owner']
            groups.append({
                'components': [owner, ref],
                'max_distance': role['distance'],
                'priority': role['priority'],
                'reason': role['reason'],
            })

        # --- Edge components (connectors) ---
        for ref in sorted(connector_refs):
            if ref not in edge:
                edge.append(ref)

        self._placement_hints['proximity_groups'] = groups
        self._placement_hints['edge_components'] = edge
        self._functional_roles = roles

        # Build owner lookup for group-aware SA moves: passive_ref → owner_ref
        self._owner_of: Dict[str, str] = {}
        for ref, role in roles.items():
            owner = role.get('owner')
            if owner:
                self._owner_of[ref] = owner

    def _init_from_parts(self, parts_db: Dict, graph: Dict):
        """Initialize placement state from parts database"""
        parts = parts_db.get('parts', {})
        raw_nets = parts_db.get('nets', {})
        self.adjacency = graph.get('adjacency', {})

        # Process nets and assign weights
        self._process_nets(raw_nets)

        # Find hub (most connected component)
        max_connections = 0
        for ref, neighbors in self.adjacency.items():
            total = sum(neighbors.values())
            if total > max_connections:
                max_connections = total
                self.hub = ref

        # Create component states with initial positions
        cx = self.config.origin_x + self.config.board_width / 2
        cy = self.config.origin_y + self.config.board_height / 2

        for ref, part in sorted(parts.items()):  # Sorted for determinism
            # SINGLE SOURCE OF TRUTH: Read courtyard from Parts Piston (parts_db)
            # Parts Piston pre-calculates IPC-7351B courtyards, we just use them
            courtyard_data = part.get('courtyard', None)

            if courtyard_data:
                # Use pre-calculated courtyard from Parts Piston
                if isinstance(courtyard_data, dict):
                    width = courtyard_data.get('width', 2.0)
                    height = courtyard_data.get('height', 2.0)
                elif hasattr(courtyard_data, 'width'):
                    width = courtyard_data.width
                    height = courtyard_data.height
                else:
                    width, height = 2.0, 2.0
            else:
                # Use FootprintResolver courtyard — same source as output & overlap check
                fp_name = part.get('footprint', 'unknown')
                resolver = FootprintResolver.get_instance()
                fp_def = resolver.resolve(fp_name)
                width, height = fp_def.courtyard_size

            # Initial position: spread around center using golden angle
            idx = len(self.components)
            if ref == self.hub:
                x, y = cx, cy
            else:
                # Golden angle spiral for initial spread (use 45% of board for wider coverage)
                golden_angle = 137.508 * math.pi / 180
                angle = idx * golden_angle
                r = min(self.config.board_width, self.config.board_height) * 0.45 * math.sqrt(idx + 1) / math.sqrt(len(parts) + 1)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)

            # Determine component properties
            used_pins = get_pins(part)
            pin_count = len(used_pins)

            # Check if decoupling cap
            is_decap = ref.startswith('C') and any(
                self._is_power_net(pin.get('net', ''))
                for pin in used_pins
            )

            # Check if power component
            is_power = any(
                self._is_power_net(pin.get('net', ''))
                for pin in used_pins
            )

            self.components[ref] = ComponentState(
                ref=ref,
                x=x,
                y=y,
                width=width,
                height=height,
                pin_count=pin_count,
                is_decoupling_cap=is_decap,
                is_power_component=is_power
            )

            # Store pin offsets, sizes, and nets
            self.pin_offsets[ref] = {}
            self.pin_sizes[ref] = {}
            self.pin_nets[ref] = {}
            for pin in used_pins:
                pin_num = pin.get('number', '')
                offset = pin.get('offset', (0, 0))
                if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                    self.pin_offsets[ref][pin_num] = (offset[0], offset[1])
                # Store pad size (default to 1.0 x 0.6 if not specified)
                size = pin.get('size', pin.get('pad_size', (1.0, 0.6)))
                if isinstance(size, (list, tuple)) and len(size) >= 2:
                    self.pin_sizes[ref][pin_num] = (size[0], size[1])
                else:
                    self.pin_sizes[ref][pin_num] = (1.0, 0.6)
                # Store net for conflict detection
                self.pin_nets[ref][pin_num] = pin.get('net', '')

        # Apply fixed positions/rotations from placement hints
        fixed_pos = self._placement_hints.get('fixed_positions', {})
        fixed_rot = self._placement_hints.get('fixed_rotations', {})
        for ref, pos in fixed_pos.items():
            if ref in self.components:
                self.components[ref].x = pos[0]
                self.components[ref].y = pos[1]
                self.components[ref].fixed = True
        for ref, rot in fixed_rot.items():
            if ref in self.components:
                self.components[ref].rotation = rot

    def _process_nets(self, raw_nets: Dict):
        """Process nets and assign weights based on type"""
        self.nets = {}
        self.net_weights = {}

        for net_name, net_info in raw_nets.items():
            # Determine net weight based on type
            if self._is_power_net(net_name):
                weight = self.config.power_net_weight
            elif self._is_critical_net(net_name):
                weight = self.config.critical_net_weight
            else:
                weight = 1.0

            self.nets[net_name] = {
                'pins': net_info.get('pins', []),
                'weight': weight
            }
            self.net_weights[net_name] = weight

    def _is_power_net(self, net_name: str) -> bool:
        """Check if net is a power net"""
        power_names = ['GND', 'VCC', 'VDD', '3V3', '5V', 'VBUS', '12V', 'AVCC', 'AGND']
        return net_name.upper() in power_names or net_name.upper().startswith('V')

    def _is_critical_net(self, net_name: str) -> bool:
        """Check if net is a critical signal net"""
        critical_keywords = ['CLK', 'CLOCK', 'RST', 'RESET', 'CS', 'EN', 'ENABLE']
        return any(kw in net_name.upper() for kw in critical_keywords)

    # =========================================================================
    # FORCE-DIRECTED PLACEMENT (Fruchterman-Reingold Algorithm)
    # =========================================================================

    def _place_force_directed(self) -> PlacementResult:
        """
        Force-directed placement using Fruchterman-Reingold algorithm.

        This is a physics-based approach where:
        - Connected components attract each other (springs/Hooke's law)
        - All components repel each other (electrostatic/Coulomb's law)
        - Temperature controls movement magnitude and decreases over time

        The algorithm finds an equilibrium where forces balance out,
        resulting in connected components being close together while
        maintaining separation between all components.
        """
        print("  [FD] Running force-directed placement (Fruchterman-Reingold)...")

        n = len(self.components)
        if n == 0:
            return self._create_result('force_directed', 0, True)

        # Optimal distance between nodes (k in FR algorithm)
        k = self._optimal_dist if self._optimal_dist > 0 else 5.0

        # Temperature starts high (allows large movements) and cools down
        temp = self.config.fd_initial_temp * min(self.config.board_width, self.config.board_height)

        iteration = 0
        converged = False

        # Check if JIT forces are available
        _use_jit_fd = self._jit is not None
        _jit_refs = None
        if _use_jit_fd:
            _jit_refs = sorted(self.components.keys())

        for iteration in range(self.config.fd_iterations):
            # Reset forces
            for comp in self.components.values():
                comp.fx = 0.0
                comp.fy = 0.0

            refs = list(self.components.keys())

            if _use_jit_fd:
                # JIT path: compute repulsive + attractive + proximity forces in one call
                try:
                    self._jit.sync_from_engine(self)
                    jit_forces = self._jit.fd_forces()
                    # Write JIT forces back to components
                    for r, idx in self._jit.ref_to_idx.items():
                        if r in self.components:
                            self.components[r].fx += jit_forces[idx, 0]
                            self.components[r].fy += jit_forces[idx, 1]
                except Exception:
                    _use_jit_fd = False  # Disable JIT for remaining iterations

            if not _use_jit_fd:
                # Python fallback: O(n^2) repulsive + attractive forces
                # Calculate repulsive forces between ALL component pairs
                for i, ref_a in enumerate(refs):
                    for ref_b in refs[i + 1:]:
                        self._apply_repulsive_force(ref_a, ref_b, k)

                # Calculate attractive forces from net connections
                for net_name, net_info in self.nets.items():
                    pins = net_info.get('pins', [])
                    weight = net_info.get('weight', 1.0)
                    components_in_net = sorted(set(self._parse_pin_ref(p)[0] for p in pins))

                    # Apply attraction between all pairs in the net
                    for i, comp_a in enumerate(components_in_net):
                        for comp_b in components_in_net[i + 1:]:
                            if comp_a in self.components and comp_b in self.components:
                                self._apply_attractive_force(comp_a, comp_b, k, weight)

                # Extra attraction for proximity_groups (electrical hints)
                hints = getattr(self, '_placement_hints', None)
                if hints:
                    for group in hints.get('proximity_groups', []):
                        components = group.get('components', [])
                        priority = group.get('priority', 1.0)
                        for i, ref_a in enumerate(components):
                            for ref_b in components[i + 1:]:
                                if ref_a in self.components and ref_b in self.components:
                                    self._apply_attractive_force(ref_a, ref_b, k, priority * 2.0)

            # Spreading force: push components away from center of mass
            # This prevents clustering and encourages full board utilization
            self._apply_spreading_force(k)

            # Apply boundary forces to keep components inside board
            for comp in self.components.values():
                self._apply_boundary_force(comp)

            # Push connectors toward board edges
            self._apply_edge_force()

            # Update positions based on forces, limited by temperature
            # NOTE: FD does NOT use occupancy grid — physics (repulsion) resolves overlaps.
            # Occupancy enforcement happens in _legalize() after FD converges.
            max_displacement = 0.0
            for comp in self.components.values():
                if comp.fixed:
                    continue

                # Calculate displacement magnitude
                force_mag = math.sqrt(comp.fx ** 2 + comp.fy ** 2)
                if force_mag > 0:
                    # Limit displacement by temperature
                    displacement = min(force_mag, temp)

                    # Normalize force and scale by displacement
                    dx = (comp.fx / force_mag) * displacement
                    dy = (comp.fy / force_mag) * displacement

                    comp.x += dx
                    comp.y += dy
                    comp.x, comp.y = self._clamp_to_board(comp)

                    # Track maximum displacement for convergence check
                    max_displacement = max(max_displacement, abs(dx), abs(dy))

            # Clamp all components to board bounds
            for comp in self.components.values():
                comp.x, comp.y = self._clamp_to_board(comp)

            # HARD CONSTRAINT: Check for pad conflicts and add repulsive force
            # This helps FD naturally spread components that have conflicting pads
            pad_conflict = self._calculate_pad_conflict_penalty()
            if pad_conflict > 0.01:
                # Add extra repulsion between components with conflicting pads
                for i, ref_a in enumerate(refs):
                    for ref_b in refs[i + 1:]:
                        # Check if this pair has pad conflict
                        pair_conflict = self._check_pad_pair_conflict(ref_a, ref_b)
                        if pair_conflict > 0:
                            # Apply strong repulsion
                            self._apply_repulsive_force(ref_a, ref_b, k * 3)

            # Cool down temperature (adaptive based on movement)
            if max_displacement < temp * 0.1:
                # Little movement, cool faster
                temp *= self.config.fd_cooling_factor * 0.9
            else:
                temp *= self.config.fd_cooling_factor

            # Check for convergence
            if temp < self.config.fd_min_temp or max_displacement < 0.01:
                converged = True
                print(f"  [FD] Converged at iteration {iteration + 1}, temp={temp:.4f}")
                break

        # Legalize: snap to grid and resolve any remaining overlaps
        self._legalize()

        cost = self._calculate_cost()
        print(f"  [FD] Final cost: {cost:.2f}")
        return self._create_result('force_directed', iteration + 1, converged)

    def _apply_repulsive_force(self, ref_a: str, ref_b: str, k: float):
        """
        Apply repulsive force between two components (Coulomb's law).

        Treats component courtyards as indivisible blocks: if courtyard
        rectangles overlap (or nearly touch), apply very strong repulsion
        to push them apart. Force direction is axis-aligned to the smallest
        overlap axis for faster separation.
        """
        ca = self.components[ref_a]
        cb = self.components[ref_b]

        dx = ca.x - cb.x
        dy = ca.y - cb.y

        # Avoid division by zero
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            dx = random.uniform(-0.5, 0.5)
            dy = random.uniform(-0.5, 0.5)

        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 0.01:
            dist = 0.01

        # Courtyard-aware: minimum separation along each axis
        min_sep_x = (ca.width + cb.width) / 2 + self.config.min_spacing
        min_sep_y = (ca.height + cb.height) / 2 + self.config.min_spacing

        # How much do courtyards overlap (or nearly overlap) on each axis?
        overlap_x = min_sep_x - abs(dx)
        overlap_y = min_sep_y - abs(dy)

        if overlap_x > 0 and overlap_y > 0:
            # Courtyards are overlapping — STRONG repulsion proportional to overlap
            # Push along the axis with smaller overlap (easier to resolve)
            force_scale = 50.0  # Strong push when overlapping
            if overlap_x < overlap_y:
                fx = force_scale * overlap_x * (1 if dx >= 0 else -1)
                fy = force_scale * overlap_y * 0.3 * (1 if dy >= 0 else -1)
            else:
                fx = force_scale * overlap_x * 0.3 * (1 if dx >= 0 else -1)
                fy = force_scale * overlap_y * (1 if dy >= 0 else -1)
        else:
            # No overlap — standard Coulomb repulsion with courtyard-scaled k
            min_dist = max(min_sep_x, min_sep_y)
            effective_k = k * (1 + min_dist / k)
            force = (effective_k ** 2) / dist
            fx = force * dx / dist
            fy = force * dy / dist

        if not ca.fixed:
            ca.fx += fx
            ca.fy += fy
        if not cb.fixed:
            cb.fx -= fx
            cb.fy -= fy

    def _apply_attractive_force(self, ref_a: str, ref_b: str, k: float, weight: float = 1.0):
        """
        Apply attractive force between connected components (Hooke's law).

        Force magnitude: f_a = d^2 / k
        Direction: Towards each other
        """
        ca = self.components[ref_a]
        cb = self.components[ref_b]

        dx = cb.x - ca.x
        dy = cb.y - ca.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.01:
            return  # Already close enough

        # Attractive force (stronger when far)
        force = (dist ** 2) / k * weight * self.config.fd_attraction_k

        # Apply force in direction towards each other
        fx = force * dx / dist
        fy = force * dy / dist

        if not ca.fixed:
            ca.fx += fx
            ca.fy += fy
        if not cb.fixed:
            cb.fx -= fx
            cb.fy -= fy

    def _apply_boundary_force(self, comp: ComponentState):
        """Apply force to keep component within board boundaries"""
        margin = self.config.edge_margin + max(comp.width, comp.height) / 2

        min_x = self.config.origin_x + margin
        max_x = self.config.origin_x + self.config.board_width - margin
        min_y = self.config.origin_y + margin
        max_y = self.config.origin_y + self.config.board_height - margin

        boundary_force = 100.0  # Strong force at boundaries

        # Left boundary
        if comp.x < min_x:
            comp.fx += boundary_force * (min_x - comp.x)
        # Right boundary
        if comp.x > max_x:
            comp.fx -= boundary_force * (comp.x - max_x)
        # Top boundary
        if comp.y < min_y:
            comp.fy += boundary_force * (min_y - comp.y)
        # Bottom boundary
        if comp.y > max_y:
            comp.fy -= boundary_force * (comp.y - max_y)

    def _apply_edge_force(self):
        """Push connectors toward nearest board edge with strong directional force.

        Applies force only along the edge-normal axis (perpendicular to edge),
        leaving the parallel axis free for net-based optimization.
        """
        hints = getattr(self, '_placement_hints', None)
        if not hints:
            return
        edge_refs = hints.get('edge_components', [])
        if not edge_refs:
            return

        ox = self.config.origin_x
        oy = self.config.origin_y
        bw = self.config.board_width
        bh = self.config.board_height
        edge_force = 20.0  # Strong pull toward edge

        for ref in edge_refs:
            if ref not in self.components:
                continue
            comp = self.components[ref]
            if comp.fixed:
                continue

            half_w = comp.width / 2
            half_h = comp.height / 2

            # Distance to each edge (from courtyard boundary, not center)
            d_left = comp.x - half_w - ox
            d_right = (ox + bw) - (comp.x + half_w)
            d_top = comp.y - half_h - oy
            d_bottom = (oy + bh) - (comp.y + half_h)

            # Find nearest edge
            nearest = min(d_left, d_right, d_top, d_bottom)

            if nearest <= 0.5:
                continue  # Already at edge

            # Push toward nearest edge (only along normal axis)
            if nearest == d_left:
                comp.fx -= edge_force * d_left
            elif nearest == d_right:
                comp.fx += edge_force * d_right
            elif nearest == d_top:
                comp.fy -= edge_force * d_top
            else:
                comp.fy += edge_force * d_bottom

    def _compute_density_grid(self):
        """Compute NxM density grid from current component positions.

        Returns (grid, bin_w, bin_h, cols, rows, target_density, utilization, ox, oy)
        where grid[r][c] = fraction of bin area occupied by component courtyards.

        Grid resolution adapts to component count and board aspect ratio.
        All parameters derived from board geometry — no magic constants.
        """
        margin = self.config.edge_margin
        ox = self.config.origin_x + margin
        oy = self.config.origin_y + margin
        usable_w = self.config.board_width - 2 * margin
        usable_h = self.config.board_height - 2 * margin

        if usable_w <= 0 or usable_h <= 0:
            return [[0.0]], usable_w or 1, usable_h or 1, 1, 1, 0, 0, ox, oy

        # Grid resolution: sqrt(n) scaled by aspect ratio, min 3x3
        n = len(self.components)
        aspect = usable_w / max(usable_h, 0.1)
        base = max(3, int(math.sqrt(n) + 0.5))
        cols = max(3, int(base * math.sqrt(aspect) + 0.5))
        rows = max(3, int(base / math.sqrt(aspect) + 0.5))

        bin_w = usable_w / cols
        bin_h = usable_h / rows

        # Accumulate courtyard area per bin
        density = [[0.0] * cols for _ in range(rows)]
        total_area = 0.0

        for comp in self.components.values():
            area = comp.width * comp.height
            total_area += area
            c = int((comp.x - ox) / bin_w)
            r = int((comp.y - oy) / bin_h)
            c = max(0, min(cols - 1, c))
            r = max(0, min(rows - 1, r))
            density[r][c] += area

        # Normalize: density[r][c] = fraction of bin area used
        bin_area = bin_w * bin_h
        for r in range(rows):
            for c in range(cols):
                density[r][c] /= bin_area

        usable_area = usable_w * usable_h
        utilization = total_area / usable_area if usable_area > 0 else 0
        target = utilization  # uniform distribution = spread total area evenly

        return density, bin_w, bin_h, cols, rows, target, utilization, ox, oy

    def _apply_spreading_force(self, k: float):
        """Density-grid adaptive spreading — dynamic, situation-aware.

        Divides the board into NxM density bins. Each FD iteration:
        1. Measures local component density per bin (courtyard area / bin area)
        2. Compares against target density (= uniform distribution)
        3. Overcrowded bins push components down the density gradient
        4. Force strength auto-scales with board utilization

        All parameters derive from measurable board properties:
        - Grid resolution: sqrt(n_components) * aspect_ratio
        - Target density: total_component_area / usable_board_area
        - Spread strength: k * clamp(utilization * 3, 0.1, 2.0)
        """
        movable = [c for c in self.components.values() if not c.fixed]
        if len(movable) < 2:
            return

        density, bin_w, bin_h, cols, rows, target, util, ox, oy = \
            self._compute_density_grid()

        # Spread strength scales with board utilization
        # Low util (0.05) → barely spread (k*0.15). High util (0.6) → strong (k*1.8)
        strength = k * max(0.1, min(2.0, util * 3.0))

        for comp in movable:
            # Which bin is this component in?
            bc = int((comp.x - ox) / bin_w)
            br = int((comp.y - oy) / bin_h)
            bc = max(0, min(cols - 1, bc))
            br = max(0, min(rows - 1, br))

            excess = density[br][bc] - target
            if excess <= 0:
                continue  # This bin is at or below target — no spreading needed

            # Density gradient: central differences with boundary clamping
            d_left = density[br][max(0, bc - 1)]
            d_right = density[br][min(cols - 1, bc + 1)]
            d_up = density[max(0, br - 1)][bc]
            d_down = density[min(rows - 1, br + 1)][bc]

            grad_x = d_right - d_left  # positive = denser to the right
            grad_y = d_down - d_up     # positive = denser downward

            grad_mag = math.sqrt(grad_x * grad_x + grad_y * grad_y)

            if grad_mag > 1e-6:
                # Push AGAINST the gradient (toward lower density)
                force = strength * excess
                comp.fx -= force * grad_x / grad_mag
                comp.fy -= force * grad_y / grad_mag
            else:
                # Flat gradient (symmetric density around this bin)
                # Push toward board center as tiebreaker
                board_cx = self.config.origin_x + self.config.board_width / 2
                board_cy = self.config.origin_y + self.config.board_height / 2
                dx = board_cx - comp.x
                dy = board_cy - comp.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.1:
                    force = strength * excess * 0.5
                    comp.fx += force * dx / dist
                    comp.fy += force * dy / dist

    def _compute_preferred_orientation(self):
        """Determine consistent orientation for small passives.

        Groups all small 2-pin passives (body < 3mm in both dimensions),
        evaluates wirelength at 0 deg vs 90 deg for each, then picks the
        orientation preferred by the majority. This biases SA toward
        consistent orientation without forcing it.
        """
        small_passive_refs = []
        for ref, comp in self.components.items():
            if ref[0] not in ('R', 'C', 'L'):
                continue
            if comp.width >= 3.0 or comp.height >= 3.0:
                continue
            if comp.pin_count > 2:
                continue
            small_passive_refs.append(ref)

        if len(small_passive_refs) < 2:
            self._preferred_passive_orientation = None
            return

        # Count how many passives prefer 0 deg vs 90 deg based on net wirelength
        votes_0 = 0
        votes_90 = 0

        for ref in small_passive_refs:
            comp = self.components[ref]
            # Get nets this passive connects to
            pin_data = self.pin_nets.get(ref, {})
            net_names = [n for n in pin_data.values() if n]

            if len(net_names) < 2:
                votes_0 += 1  # default
                continue

            # Find center of mass of other components on each net
            net_centers = []
            for net_name in net_names[:2]:
                net_info = self.nets.get(net_name, {})
                pins = net_info.get('pins', [])
                other_refs = set()
                for p in pins:
                    if isinstance(p, str) and '.' in p:
                        r = p.split('.')[0]
                        if r != ref and r in self.components:
                            other_refs.add(r)
                if other_refs:
                    cx = sum(self.components[r].x for r in other_refs) / len(other_refs)
                    cy = sum(self.components[r].y for r in other_refs) / len(other_refs)
                    net_centers.append((cx, cy))

            if len(net_centers) < 2:
                votes_0 += 1
                continue

            # Direction of net pull
            dx = abs(net_centers[0][0] - net_centers[1][0])
            dy = abs(net_centers[0][1] - net_centers[1][1])

            # 0 deg = pads along X axis, 90 deg = pads along Y axis
            # Prefer orientation that aligns pads with the net direction
            if dx >= dy:
                votes_0 += 1  # horizontal net pull → horizontal pads
            else:
                votes_90 += 1  # vertical net pull → vertical pads

        self._preferred_passive_orientation = 0.0 if votes_0 >= votes_90 else 90.0
        self._small_passive_refs = set(small_passive_refs)

    # =========================================================================
    # SIMULATED ANNEALING PLACEMENT
    # =========================================================================

    def _run_sa_single(self, seed: int, initial_temp: float, moves_per_temp: int,
                       starting_state: Dict) -> Tuple[float, Dict, int, int]:
        """Run a single SA pass with given seed and params. Returns (cost, state, iters, accepted)."""
        rng = random.Random(seed)
        self._restore_state(starting_state)

        current_cost = self._calculate_cost()
        best_cost = current_cost
        best_state = self._save_state()

        temp = initial_temp
        rejections = 0
        total_iterations = 0
        accepted_moves = 0
        stall_count = 0  # Temp steps without improvement
        max_stall = 10   # Exit early if no improvement for 10 temp steps

        while temp > self.config.sa_final_temp:
            prev_best = best_cost
            for _ in range(moves_per_temp):
                total_iterations += 1
                old_state = self._save_state()

                r = rng.random()
                occ = self._occ if hasattr(self, '_occ') else None
                move_valid = True
                owner_map = getattr(self, '_owner_of', {})

                if r < 0.40:
                    # Shift move (random direction)
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if refs:
                        ref = rng.choice(refs)
                        comp = self.components[ref]
                        old_cx, old_cy = comp.x, comp.y
                        max_shift = (temp / initial_temp) * self.config.board_width * 0.3
                        max_shift = max(max_shift, self.config.grid_size)
                        comp.x += rng.uniform(-max_shift, max_shift)
                        comp.y += rng.uniform(-max_shift, max_shift)
                        comp.x, comp.y = self._clamp_to_board(comp)
                        # Remove from grid, check new position, re-place
                        if occ:
                            occ.remove(ref, old_cx, old_cy, comp.width, comp.height)
                            if occ.can_place(comp.x, comp.y, comp.width, comp.height):
                                occ.place(ref, comp.x, comp.y, comp.width, comp.height)
                            else:
                                occ.place(ref, old_cx, old_cy, comp.width, comp.height)
                                move_valid = False
                elif r < 0.55 and owner_map:
                    # Group-aware shift: bias a passive toward its owner IC
                    # Generalized — works for any functional role, not just caps
                    owned_refs = [ref for ref in owner_map
                                  if ref in self.components and not self.components[ref].fixed]
                    if owned_refs:
                        ref = rng.choice(owned_refs)
                        comp = self.components[ref]
                        owner_ref = owner_map[ref]
                        if owner_ref in self.components:
                            owner = self.components[owner_ref]
                            old_cx, old_cy = comp.x, comp.y
                            # Move toward owner with temperature-scaled step
                            dx = owner.x - comp.x
                            dy = owner.y - comp.y
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist > 0.1:
                                # Step fraction: at high temp, take big steps;
                                # at low temp, fine-tune position near owner
                                step_frac = rng.uniform(0.1, 0.6) * (temp / initial_temp + 0.3)
                                # Add some random jitter to avoid deterministic paths
                                jitter = rng.uniform(-1.0, 1.0)
                                comp.x += dx * step_frac + jitter
                                comp.y += dy * step_frac + jitter
                                comp.x, comp.y = self._clamp_to_board(comp)
                                if occ:
                                    occ.remove(ref, old_cx, old_cy, comp.width, comp.height)
                                    if occ.can_place(comp.x, comp.y, comp.width, comp.height):
                                        occ.place(ref, comp.x, comp.y, comp.width, comp.height)
                                    else:
                                        occ.place(ref, old_cx, old_cy, comp.width, comp.height)
                                        move_valid = False
                elif r < 0.75:
                    # Swap move
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if len(refs) >= 2:
                        a, b = rng.sample(refs, 2)
                        ca, cb = self.components[a], self.components[b]
                        # Save old positions for grid removal
                        old_ax, old_ay = ca.x, ca.y
                        old_bx, old_by = cb.x, cb.y
                        # Swap positions
                        ca.x, cb.x = cb.x, ca.x
                        ca.y, cb.y = cb.y, ca.y
                        # Remove BOTH from grid, then check BOTH at new positions
                        if occ:
                            occ.remove(a, old_ax, old_ay, ca.width, ca.height)
                            occ.remove(b, old_bx, old_by, cb.width, cb.height)
                            ok_a = occ.can_place(ca.x, ca.y, ca.width, ca.height)
                            ok_b = occ.can_place(cb.x, cb.y, cb.width, cb.height) if ok_a else False
                            if ok_a and ok_b:
                                occ.place(a, ca.x, ca.y, ca.width, ca.height)
                                occ.place(b, cb.x, cb.y, cb.width, cb.height)
                            else:
                                # Restore grid to pre-swap state
                                occ.place(a, old_ax, old_ay, ca.width, ca.height)
                                occ.place(b, old_bx, old_by, cb.width, cb.height)
                                move_valid = False
                elif r < 0.90:
                    # Rotate 90
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if refs:
                        ref = rng.choice(refs)
                        comp = self.components[ref]
                        old_cx, old_cy = comp.x, comp.y
                        old_w, old_h = comp.width, comp.height
                        comp.rotation = (comp.rotation + 90) % 360
                        comp.width, comp.height = comp.height, comp.width
                        comp.x, comp.y = self._clamp_to_board(comp)
                        if occ:
                            occ.remove(ref, old_cx, old_cy, old_w, old_h)
                            if occ.can_place(comp.x, comp.y, comp.width, comp.height):
                                occ.place(ref, comp.x, comp.y, comp.width, comp.height)
                            else:
                                occ.place(ref, old_cx, old_cy, old_w, old_h)
                                move_valid = False
                else:
                    # Rotate 180
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if refs:
                        ref = rng.choice(refs)
                        comp = self.components[ref]
                        old_cx, old_cy = comp.x, comp.y
                        comp.rotation = (comp.rotation + 180) % 360
                        comp.x, comp.y = self._clamp_to_board(comp)
                        if occ:
                            occ.remove(ref, old_cx, old_cy, comp.width, comp.height)
                            if occ.can_place(comp.x, comp.y, comp.width, comp.height):
                                occ.place(ref, comp.x, comp.y, comp.width, comp.height)
                            else:
                                occ.place(ref, old_cx, old_cy, comp.width, comp.height)
                                move_valid = False

                # HARD CONSTRAINT: reject moves that create overlaps
                if not move_valid:
                    self._restore_state(old_state)
                    rejections += 1
                    continue

                new_cost = self._calculate_cost()
                delta = new_cost - current_cost

                pad_conflict = self._calculate_pad_conflict_penalty()
                if pad_conflict > 0.01:
                    self._restore_state(old_state)
                    rejections += 1
                    continue

                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    # Move accepted — update occupancy grid
                    self._sync_occupancy_grid()
                    current_cost = new_cost
                    accepted_moves += 1
                    rejections = 0
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_state = self._save_state()
                else:
                    self._restore_state(old_state)
                    rejections += 1
                    if rejections > self.config.sa_reheat_threshold:
                        temp *= self.config.sa_reheat_factor
                        rejections = 0

            temp *= self.config.sa_cooling_rate

            # Early exit: stall detection
            if best_cost < prev_best:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= max_stall:
                    break  # No improvement for max_stall temp steps — converged

        return best_cost, best_state, total_iterations, accepted_moves

    def _place_simulated_annealing(self) -> PlacementResult:
        """
        Multi-start simulated annealing placement.

        Runs multiple SA passes IN PARALLEL with different random seeds and
        initial temperatures, each exploring a different region of the solution
        space. The best result across all workers is selected.

        Each worker gets a deep copy of the engine to avoid shared-state conflicts.

        Reference: "Optimization by Simulated Annealing" (Kirkpatrick, 1983)
        Multi-start: (Ram et al., "Parallel Simulated Annealing Algorithms", 1996)
        """
        import time as _time

        # Ensure occupancy grid reflects current state before SA starts
        self._sync_occupancy_grid()

        num_workers = min(os.cpu_count() or 4, 8)  # Cap at 8
        starting_state = self._save_state()
        base_temp = self.config.sa_initial_temp

        # Reduce moves per worker so total time stays similar
        moves_per_temp = max(self.config.sa_moves_per_temp // num_workers, 20)

        print(f"  [SA] Running multi-start simulated annealing ({num_workers} starts, parallel)...")
        sa_start = _time.time()

        def _sa_worker(worker_id, seed, temp, mpt, state):
            """Run one SA pass on a deep copy of the engine."""
            worker = copy.deepcopy(self)
            worker._restore_state(state)
            cost, best_state, iters, acc = worker._run_sa_single(seed, temp, mpt, state)
            return (cost, best_state, iters, acc, temp)

        # Build worker params
        worker_args = []
        for i in range(num_workers):
            seed = 42 + i * 1000
            temp_factor = 0.7 + (i / max(num_workers - 1, 1)) * 0.6  # 0.7x to 1.3x
            temp = base_temp * temp_factor
            worker_args.append((i, seed, temp, moves_per_temp, starting_state))

        # Run all workers in parallel
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_sa_worker, *args): args[0] for args in worker_args}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"  [SA] Worker {futures[future]} failed: {e}")

        if not results:
            # All workers failed — fall back to sequential single run
            self._restore_state(starting_state)
            cost, state, iters, acc = self._run_sa_single(42, base_temp, self.config.sa_moves_per_temp, starting_state)
            results.append((cost, state, iters, acc, base_temp))

        # Pick best
        results.sort(key=lambda r: r[0])
        best_cost, best_state, best_iters, best_acc, best_temp = results[0]

        self._restore_state(best_state)
        self._legalize()

        sa_elapsed = _time.time() - sa_start
        acceptance_rate = best_acc / best_iters if best_iters > 0 else 0
        print(f"  [SA] Final cost: {best_cost:.2f}, acceptance rate: {acceptance_rate:.2%}")
        print(f"  [SA] {num_workers} parallel starts in {sa_elapsed:.1f}s, best temp={best_temp:.1f}")

        return self._create_result('simulated_annealing', best_iters, True)

    def _move_shift(self, temp: float):
        """Shift a random component by an amount proportional to temperature"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if not refs:
            return

        ref = random.choice(refs)
        comp = self.components[ref]

        # Shift amount decreases with temperature
        max_shift = (temp / self.config.sa_initial_temp) * self.config.board_width * 0.3
        max_shift = max(max_shift, self.config.grid_size)  # Minimum shift

        comp.x += random.uniform(-max_shift, max_shift)
        comp.y += random.uniform(-max_shift, max_shift)
        comp.x, comp.y = self._clamp_to_board(comp)

    def _move_swap(self):
        """Swap positions of two random components"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if len(refs) < 2:
            return

        ref_a, ref_b = random.sample(refs, 2)
        ca, cb = self.components[ref_a], self.components[ref_b]

        # Swap positions
        ca.x, cb.x = cb.x, ca.x
        ca.y, cb.y = cb.y, ca.y

    def _move_rotate(self):
        """Rotate a random component by 90 degrees"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if not refs:
            return

        ref = random.choice(refs)
        comp = self.components[ref]

        old_rotation = comp.rotation
        comp.rotation = (comp.rotation + 90) % 360

        # Swap width and height when TRANSITIONING to/from 90/270 rotation
        # (i.e., when the parity of being rotated 90 degrees changes)
        was_swapped = old_rotation in [90, 270]
        is_swapped = comp.rotation in [90, 270]
        if was_swapped != is_swapped:
            comp.width, comp.height = comp.height, comp.width

        # Re-clamp to board
        comp.x, comp.y = self._clamp_to_board(comp)

    def _move_flip(self):
        """Flip a component (180 degree rotation)"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if not refs:
            return

        ref = random.choice(refs)
        comp = self.components[ref]
        comp.rotation = (comp.rotation + 180) % 360

    # =========================================================================
    # GENETIC ALGORITHM PLACEMENT
    # =========================================================================

    def _place_genetic(self) -> PlacementResult:
        """
        Genetic algorithm placement with diversity preservation.

        This evolutionary approach:
        - Maintains a population of candidate solutions
        - Uses tournament selection
        - Order crossover (OX) for position inheritance
        - Adaptive mutation based on diversity
        - Elitism to preserve best solutions
        """
        print("  [GA] Running genetic algorithm placement...")

        n_components = len(self.components)
        if n_components == 0:
            return self._create_result('genetic', 0, True)

        refs = sorted(self.components.keys())
        n_elite = max(1, int(self.config.ga_population_size * self.config.ga_elite_ratio))

        # Initialize population with diverse solutions
        population = self._init_ga_population(refs)

        best_individual = None
        best_fitness = float('inf')
        stagnation_count = 0

        for gen in range(self.config.ga_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                self._apply_ga_individual(individual, refs)
                fitness = self._calculate_cost()
                fitness_scores.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)
                    stagnation_count = 0
                else:
                    stagnation_count += 1

            # Calculate population diversity
            diversity = self._calculate_ga_diversity(population)

            # Adaptive mutation rate based on diversity
            effective_mutation_rate = self.config.ga_mutation_rate
            if diversity < self.config.ga_diversity_threshold:
                effective_mutation_rate *= 2  # Increase mutation when low diversity

            # Sort by fitness (lower is better)
            sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i])

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            for i in range(n_elite):
                new_population.append(copy.deepcopy(population[sorted_indices[i]]))

            # Fill rest with offspring
            while len(new_population) < self.config.ga_population_size:
                # Tournament selection
                parent1 = self._ga_tournament_select(population, fitness_scores)
                parent2 = self._ga_tournament_select(population, fitness_scores)

                # Crossover
                if random.random() < self.config.ga_crossover_rate:
                    child = self._ga_order_crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                # Mutation
                if random.random() < effective_mutation_rate:
                    self._ga_mutate(child)

                new_population.append(child)

            population = new_population

            # Progress report every 25 generations
            if (gen + 1) % 25 == 0:
                print(f"  [GA] Generation {gen + 1}: best={best_fitness:.2f}, diversity={diversity:.3f}")

        # Apply best solution
        self._apply_ga_individual(best_individual, refs)
        self._legalize()

        final_cost = self._calculate_cost()
        print(f"  [GA] Final cost: {final_cost:.2f}")
        return self._create_result('genetic', self.config.ga_generations, True)

    def _init_ga_population(self, refs: List[str]) -> List[Dict]:
        """Initialize diverse population for GA"""
        population = []

        for i in range(self.config.ga_population_size):
            individual = {}

            # Different initialization strategies for diversity
            strategy = i % 4

            for ref in refs:
                comp = self.components[ref]
                margin = self.config.edge_margin + max(comp.width, comp.height) / 2

                if strategy == 0:
                    # Random uniform
                    x = self.config.origin_x + margin + random.random() * (self.config.board_width - 2 * margin)
                    y = self.config.origin_y + margin + random.random() * (self.config.board_height - 2 * margin)
                elif strategy == 1:
                    # Gaussian around center
                    cx = self.config.origin_x + self.config.board_width / 2
                    cy = self.config.origin_y + self.config.board_height / 2
                    x = cx + random.gauss(0, self.config.board_width / 4)
                    y = cy + random.gauss(0, self.config.board_height / 4)
                elif strategy == 2:
                    # Grid-based
                    n = len(refs)
                    cols = int(math.sqrt(n)) + 1
                    idx = refs.index(ref)
                    cell_w = (self.config.board_width - 2 * margin) / cols
                    cell_h = (self.config.board_height - 2 * margin) / max(1, (n + cols - 1) // cols)
                    x = self.config.origin_x + margin + (idx % cols + 0.5) * cell_w
                    y = self.config.origin_y + margin + (idx // cols + 0.5) * cell_h
                else:
                    # Spiral from center
                    idx = refs.index(ref)
                    angle = idx * 137.508 * math.pi / 180  # Golden angle
                    r = self.config.board_width * 0.3 * math.sqrt(idx + 1) / math.sqrt(len(refs))
                    cx = self.config.origin_x + self.config.board_width / 2
                    cy = self.config.origin_y + self.config.board_height / 2
                    x = cx + r * math.cos(angle)
                    y = cy + r * math.sin(angle)

                # Clamp to bounds
                x = max(self.config.origin_x + margin,
                        min(self.config.origin_x + self.config.board_width - margin, x))
                y = max(self.config.origin_y + margin,
                        min(self.config.origin_y + self.config.board_height - margin, y))

                individual[ref] = (x, y, 0)  # x, y, rotation

            population.append(individual)

        return population

    def _apply_ga_individual(self, individual: Dict, refs: List[str]):
        """Apply GA individual's positions to components"""
        for ref, (x, y, rot) in individual.items():
            if ref in self.components:
                self.components[ref].x = x
                self.components[ref].y = y
                self.components[ref].rotation = rot

    def _ga_tournament_select(self, population: List, fitness_scores: List) -> Dict:
        """Tournament selection"""
        indices = random.sample(range(len(population)), min(self.config.ga_tournament_size, len(population)))
        best_idx = min(indices, key=lambda i: fitness_scores[i])
        return copy.deepcopy(population[best_idx])

    def _ga_order_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Order crossover (OX) - blend positions from both parents"""
        child = {}
        refs = list(parent1.keys())

        # Randomly select crossover points
        split_point = random.randint(0, len(refs))

        for i, ref in enumerate(refs):
            if i < split_point:
                # Take from parent1
                child[ref] = parent1[ref]
            else:
                # Blend: average of both parents
                x1, y1, r1 = parent1[ref]
                x2, y2, r2 = parent2[ref]

                # Weighted average with small random perturbation
                alpha = random.uniform(0.3, 0.7)
                x = alpha * x1 + (1 - alpha) * x2
                y = alpha * y1 + (1 - alpha) * y2
                r = r1 if random.random() < 0.5 else r2

                child[ref] = (x, y, r)

        return child

    def _ga_mutate(self, individual: Dict):
        """Mutate individual by perturbing positions"""
        refs = list(individual.keys())
        n_mutations = max(1, len(refs) // 5)  # Mutate ~20% of components

        for _ in range(n_mutations):
            ref = random.choice(refs)
            if ref not in self.components:
                continue

            comp = self.components[ref]
            x, y, r = individual[ref]

            # Random perturbation
            x += random.gauss(0, self.config.board_width * 0.1)
            y += random.gauss(0, self.config.board_height * 0.1)

            # Occasionally rotate
            if random.random() < 0.1:
                r = (r + 90) % 360

            # Clamp to bounds
            margin = self.config.edge_margin + max(comp.width, comp.height) / 2
            x = max(self.config.origin_x + margin,
                    min(self.config.origin_x + self.config.board_width - margin, x))
            y = max(self.config.origin_y + margin,
                    min(self.config.origin_y + self.config.board_height - margin, y))

            individual[ref] = (x, y, r)

    def _calculate_ga_diversity(self, population: List) -> float:
        """Calculate population diversity (standard deviation of positions)"""
        if len(population) < 2:
            return 1.0

        refs = list(population[0].keys())
        total_variance = 0.0

        for ref in refs:
            x_values = [ind[ref][0] for ind in population]
            y_values = [ind[ref][1] for ind in population]

            if len(x_values) > 1:
                x_mean = sum(x_values) / len(x_values)
                y_mean = sum(y_values) / len(y_values)

                x_var = sum((x - x_mean) ** 2 for x in x_values) / len(x_values)
                y_var = sum((y - y_mean) ** 2 for y in y_values) / len(y_values)

                total_variance += x_var + y_var

        # Normalize by board area
        board_area = self.config.board_width * self.config.board_height
        diversity = math.sqrt(total_variance / (len(refs) * board_area + 0.001))

        return min(1.0, diversity)

    # =========================================================================
    # ANALYTICAL PLACEMENT (Quadratic Optimization)
    # =========================================================================

    def _place_analytical(self) -> PlacementResult:
        """
        Analytical placement using quadratic wire length minimization.

        This mathematical approach:
        - Minimizes sum of squared wire lengths
        - Uses conjugate gradient solver
        - Adds spreading force to avoid overlap
        - Multi-level approach for large designs
        """
        print("  [AN] Running analytical (quadratic) placement...")

        refs = sorted(self.components.keys())
        n = len(refs)
        if n == 0:
            return self._create_result('analytical', 0, True)

        ref_to_idx = {ref: i for i, ref in enumerate(refs)}

        # Build Laplacian matrix from connectivity
        # L[i][j] = -weight if i and j are connected
        # L[i][i] = sum of weights of edges incident to i
        L = [[0.0] * n for _ in range(n)]
        bx = [0.0] * n
        by = [0.0] * n

        for net_name, net_info in self.nets.items():
            pins = net_info.get('pins', [])
            weight = net_info.get('weight', 1.0)
            components_in_net = sorted(set(self._parse_pin_ref(p)[0] for p in pins if self._parse_pin_ref(p)[0] in ref_to_idx))

            if len(components_in_net) < 2:
                continue

            # Clique model: connect all pairs in net
            # Weight inversely proportional to net degree
            net_weight = weight / (len(components_in_net) - 1)

            for i, comp_a in enumerate(components_in_net):
                for comp_b in components_in_net[i + 1:]:
                    idx_a = ref_to_idx[comp_a]
                    idx_b = ref_to_idx[comp_b]

                    L[idx_a][idx_a] += net_weight
                    L[idx_b][idx_b] += net_weight
                    L[idx_a][idx_b] -= net_weight
                    L[idx_b][idx_a] -= net_weight

        # Add anchor terms for fixed components and spreading
        cx = self.config.origin_x + self.config.board_width / 2
        cy = self.config.origin_y + self.config.board_height / 2

        for ref, comp in self.components.items():
            idx = ref_to_idx[ref]

            if comp.fixed:
                # Strong anchor to fixed position
                anchor_weight = 1000.0
                L[idx][idx] += anchor_weight
                bx[idx] += anchor_weight * comp.x
                by[idx] += anchor_weight * comp.y
            elif ref == self.hub:
                # Hub anchored to center
                anchor_weight = 10.0
                L[idx][idx] += anchor_weight
                bx[idx] += anchor_weight * cx
                by[idx] += anchor_weight * cy
            else:
                # Weak anchor to center for spreading
                anchor_weight = self.config.an_anchor_weight
                L[idx][idx] += anchor_weight
                bx[idx] += anchor_weight * cx
                by[idx] += anchor_weight * cy

        # Solve using Conjugate Gradient method
        x = [comp.x for ref, comp in sorted(self.components.items())]
        y = [comp.y for ref, comp in sorted(self.components.items())]

        x = self._conjugate_gradient(L, bx, x, self.config.an_iterations)
        y = self._conjugate_gradient(L, by, y, self.config.an_iterations)

        # Apply solution
        for i, ref in enumerate(refs):
            self.components[ref].x = x[i]
            self.components[ref].y = y[i]

        # Apply spreading to avoid overlap
        self._apply_spreading()

        # Legalize
        self._legalize()

        cost = self._calculate_cost()
        print(f"  [AN] Final cost: {cost:.2f}")
        return self._create_result('analytical', self.config.an_iterations, True)

    def _conjugate_gradient(self, A: List[List[float]], b: List[float],
                            x0: List[float], max_iter: int) -> List[float]:
        """
        Conjugate Gradient solver for Ax = b.

        This iterative method is efficient for sparse symmetric positive definite matrices.
        """
        n = len(b)
        x = x0.copy()

        # r = b - Ax
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        p = r.copy()
        rs_old = sum(r[i] ** 2 for i in range(n))

        for iteration in range(max_iter):
            # Ap
            Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]

            # alpha = rs_old / (p^T * Ap)
            pAp = sum(p[i] * Ap[i] for i in range(n))
            if abs(pAp) < 1e-10:
                break
            alpha = rs_old / pAp

            # x = x + alpha * p
            for i in range(n):
                x[i] += alpha * p[i]

            # r = r - alpha * Ap
            for i in range(n):
                r[i] -= alpha * Ap[i]

            rs_new = sum(r[i] ** 2 for i in range(n))

            # Check convergence
            if math.sqrt(rs_new) < 1e-6:
                break

            # p = r + (rs_new / rs_old) * p
            beta = rs_new / (rs_old + 1e-10)
            for i in range(n):
                p[i] = r[i] + beta * p[i]

            rs_old = rs_new

        return x

    def _apply_spreading(self):
        """Apply spreading force to reduce overlap"""
        for _ in range(20):  # Spreading iterations
            overlaps_found = False

            refs = list(self.components.keys())
            for i, ref_a in enumerate(refs):
                for ref_b in refs[i + 1:]:
                    ca = self.components[ref_a]
                    cb = self.components[ref_b]

                    dx = cb.x - ca.x
                    dy = cb.y - ca.y
                    dist = math.sqrt(dx ** 2 + dy ** 2) + 0.001

                    # Required separation
                    min_sep = (ca.width + cb.width) / 2 + self.config.min_spacing

                    if dist < min_sep:
                        overlaps_found = True
                        # Push apart
                        push = (min_sep - dist) * self.config.an_spreading_factor
                        push_x = push * dx / dist
                        push_y = push * dy / dist

                        if not ca.fixed:
                            ca.x -= push_x
                            ca.y -= push_y
                        if not cb.fixed:
                            cb.x += push_x
                            cb.y += push_y

            # Clamp to bounds
            for comp in self.components.values():
                comp.x, comp.y = self._clamp_to_board(comp)

            if not overlaps_found:
                break

    # =========================================================================
    # HUMAN-LIKE PLACEMENT
    # =========================================================================

    def _place_human_like(self) -> PlacementResult:
        """
        Human-like placement using design rules.

        Mimics how human PCB designers place components:
        - Place main IC (hub) at center
        - Place decoupling caps very close to power pins
        - Group related components
        - Leave routing channels
        - Follow signal flow
        """
        print("  [HL] Running human-like placement...")

        cx = self.config.origin_x + self.config.board_width / 2
        cy = self.config.origin_y + self.config.board_height / 2

        # Get placement order
        order = self._get_human_placement_order()
        placed = set()

        # Track occupied regions for channel planning
        occupied = []

        for ref in order:
            comp = self.components[ref]

            if ref == self.hub:
                # Hub at center
                comp.x = cx
                comp.y = cy
                print(f"    {ref}: Hub placed at center")

            elif comp.is_decoupling_cap:
                # Decoupling cap: place very close to power pin of hub
                target_pos = self._find_decap_position(ref, placed)
                comp.x, comp.y = target_pos
                print(f"    {ref}: Decoupling cap placed near power pin")

            else:
                # Find optimal position based on connectivity
                target_pos = self._find_optimal_position(ref, placed, occupied)
                comp.x, comp.y = target_pos

            # Resolve overlaps using spiral search
            self._resolve_overlap_spiral(ref, placed)

            # Update occupied regions
            half_w = comp.width / 2 + self.config.routing_channel_width
            half_h = comp.height / 2 + self.config.routing_channel_width
            occupied.append((comp.x - half_w, comp.y - half_h, comp.x + half_w, comp.y + half_h))

            placed.add(ref)

        cost = self._calculate_cost()
        print(f"  [HL] Final cost: {cost:.2f}")
        return self._create_result('human_like', len(order), True)

    def _get_human_placement_order(self) -> List[str]:
        """Determine placement order like a human designer"""
        order = []
        placed = set()

        # 1. Hub (main IC) first
        if self.hub:
            order.append(self.hub)
            placed.add(self.hub)

        # 2. Decoupling capacitors (must be close to hub)
        decaps = [ref for ref in self.components
                  if ref not in placed and self.components[ref].is_decoupling_cap]
        for ref in sorted(decaps):
            order.append(ref)
            placed.add(ref)

        # 3. Components directly connected to hub, by connection count
        hub_connected = []
        if self.hub:
            for ref in self.components:
                if ref in placed:
                    continue
                conn = self.adjacency.get(ref, {}).get(self.hub, 0)
                if conn > 0:
                    hub_connected.append((ref, conn))
            hub_connected.sort(key=lambda x: -x[1])
            for ref, _ in hub_connected:
                order.append(ref)
                placed.add(ref)

        # 4. Remaining components by total connectivity
        remaining = [(ref, sum(self.adjacency.get(ref, {}).values()))
                     for ref in self.components if ref not in placed]
        remaining.sort(key=lambda x: -x[1])
        for ref, _ in remaining:
            order.append(ref)

        return order

    def _find_decap_position(self, ref: str, placed: Set[str]) -> Tuple[float, float]:
        """Find optimal position for decoupling capacitor near power pins"""
        comp = self.components[ref]

        # Find which power net this cap is on
        power_net = None
        for net_name, net_info in self.nets.items():
            if not self._is_power_net(net_name):
                continue
            for pin_ref in net_info.get('pins', []):
                comp_ref, pin = self._parse_pin_ref(pin_ref)
                if comp_ref == ref:
                    power_net = net_name
                    break
            if power_net:
                break

        if not power_net or not self.hub:
            # Fallback: place near hub
            hub_comp = self.components.get(self.hub)
            if hub_comp:
                return (hub_comp.x + comp.width + 1, hub_comp.y)
            return (self.config.origin_x + self.config.board_width / 2,
                    self.config.origin_y + self.config.board_height / 2)

        # Find power pin position on hub
        hub_comp = self.components[self.hub]
        hub_pins = self.pin_offsets.get(self.hub, {})

        # Look for power pin on hub
        for pin_ref in self.nets[power_net].get('pins', []):
            comp_ref, pin = self._parse_pin_ref(pin_ref)
            if comp_ref == self.hub and pin in hub_pins:
                offset = hub_pins[pin]
                pin_x = hub_comp.x + offset[0]
                pin_y = hub_comp.y + offset[1]

                # Place cap adjacent to pin
                if offset[1] > 0:  # Pin on bottom of hub
                    return (pin_x, pin_y + comp.height / 2 + self.config.min_spacing)
                elif offset[1] < 0:  # Pin on top of hub
                    return (pin_x, pin_y - comp.height / 2 - self.config.min_spacing)
                elif offset[0] > 0:  # Pin on right
                    return (pin_x + comp.width / 2 + self.config.min_spacing, pin_y)
                else:  # Pin on left
                    return (pin_x - comp.width / 2 - self.config.min_spacing, pin_y)

        # Fallback: place adjacent to hub
        return (hub_comp.x + hub_comp.width / 2 + comp.width / 2 + self.config.min_spacing, hub_comp.y)

    def _find_optimal_position(self, ref: str, placed: Set[str],
                               occupied: List[Tuple]) -> Tuple[float, float]:
        """Find optimal position based on connectivity to placed components"""
        connections = self.adjacency.get(ref, {})

        if not connections or not placed:
            # No connections: random position
            return (
                self.config.origin_x + self.config.board_width / 2,
                self.config.origin_y + self.config.board_height / 2
            )

        # Weighted centroid of connected placed components
        sum_x = 0.0
        sum_y = 0.0
        total_weight = 0.0

        for other_ref, weight in connections.items():
            if other_ref in placed:
                other = self.components[other_ref]
                sum_x += other.x * weight
                sum_y += other.y * weight
                total_weight += weight

        if total_weight > 0:
            return (sum_x / total_weight, sum_y / total_weight)
        else:
            return (
                self.config.origin_x + self.config.board_width / 2,
                self.config.origin_y + self.config.board_height / 2
            )

    def _resolve_overlap_spiral(self, ref: str, placed: Set[str], max_attempts: int = 100):
        """Resolve overlap using spiral search pattern"""
        comp = self.components[ref]
        original_x, original_y = comp.x, comp.y

        for attempt in range(max_attempts):
            if not self._has_overlap(ref, placed):
                return

            # Spiral pattern: increasing radius, rotating angle
            angle_step = 30  # degrees
            angle = (attempt * angle_step) % 360
            radius = self.config.grid_size * (1 + attempt // 12)

            comp.x = original_x + radius * math.cos(math.radians(angle))
            comp.y = original_y + radius * math.sin(math.radians(angle))
            comp.x, comp.y = self._clamp_to_board(comp)

        # If still overlapping, try grid search
        for dy in range(-10, 11, 2):
            for dx in range(-10, 11, 2):
                comp.x = original_x + dx * self.config.grid_size
                comp.y = original_y + dy * self.config.grid_size
                comp.x, comp.y = self._clamp_to_board(comp)
                if not self._has_overlap(ref, placed):
                    return

    def _has_overlap(self, ref: str, placed: Set[str]) -> bool:
        """Check if component overlaps with any placed component"""
        comp = self.components[ref]
        half_w = comp.width / 2 + self.config.min_spacing
        half_h = comp.height / 2 + self.config.min_spacing

        for other_ref in placed:
            other = self.components[other_ref]
            other_half_w = other.width / 2 + self.config.min_spacing
            other_half_h = other.height / 2 + self.config.min_spacing

            if (abs(comp.x - other.x) < half_w + other_half_w and
                    abs(comp.y - other.y) < half_h + other_half_h):
                return True
        return False

    # =========================================================================
    # HYBRID PLACEMENT (FD + SA)
    # =========================================================================

    def _place_hybrid(self) -> PlacementResult:
        """
        Hybrid placement: Force-Directed global + Simulated Annealing refinement.

        This two-phase approach:
        1. FD provides good global placement quickly
        2. SA refines locally for optimal wire length
        """
        print("  [HY] Running hybrid placement (FD + SA)...")

        # Phase 1: Force-directed for global placement
        print("  [HY] Phase 1: Force-directed global placement")
        fd_result = self._place_force_directed()

        # Sync occupancy grid after FD before SA starts
        self._sync_occupancy_grid()

        # Compute preferred passive orientation from FD result
        self._compute_preferred_orientation()

        # Phase 2: Simulated annealing for local refinement
        print("  [HY] Phase 2: Simulated annealing refinement")
        # Use lower temperature since we already have a good solution
        original_temp = self.config.sa_initial_temp
        original_final = self.config.sa_final_temp
        self.config.sa_initial_temp = 30.0  # Lower temp for refinement
        self.config.sa_final_temp = 0.1

        sa_result = self._place_simulated_annealing()

        # Restore config
        self.config.sa_initial_temp = original_temp
        self.config.sa_final_temp = original_final

        return self._create_result(
            'hybrid (fd+sa)',
            fd_result.iterations + sa_result.iterations,
            True,
        )

    # =========================================================================
    # AUTO PLACEMENT (Try multiple, pick best)
    # =========================================================================

    def _place_auto(self) -> PlacementResult:
        """
        Automatic algorithm selection.

        Tries multiple algorithms and selects the best result based on:
        1. No overlaps (critical)
        2. No out-of-bounds (critical)
        3. Lowest cost (wire length)
        """
        print("  [AUTO] Automatic algorithm selection...")

        results = []
        best_result = None
        best_score = float('inf')
        best_positions = None

        # Save initial state
        initial_state = self._save_state()

        # Algorithms to try (ordered by expected quality)
        algorithms = [
            ('hybrid', self._place_hybrid),
            ('human', self._place_human_like),
            ('fd', self._place_force_directed),
            ('analytical', self._place_analytical),
        ]

        for algo_name, algo_func in algorithms:
            print(f"\n  [AUTO] Trying {algo_name}...")

            # Reset to initial state
            self._restore_state(initial_state)
            # Reset velocities and forces
            for comp in self.components.values():
                comp.vx = comp.vy = comp.fx = comp.fy = 0.0

            try:
                result = algo_func()

                # Evaluate quality
                has_overlaps = self._check_has_overlaps()
                has_oob = self._check_out_of_bounds()
                cost = result.cost

                # Score: penalize overlaps and OOB heavily
                score = cost
                if has_overlaps:
                    score += 10000
                if has_oob:
                    score += 5000

                quality = "GOOD" if not has_overlaps and not has_oob else "POOR"
                print(f"  [AUTO] {algo_name}: cost={cost:.2f}, overlaps={has_overlaps}, oob={has_oob} -> {quality}")

                results.append((algo_name, result, score))

                if score < best_score:
                    best_score = score
                    best_result = result
                    best_positions = self._save_state()

                    # If good enough, stop early
                    if not has_overlaps and not has_oob and cost < 100:
                        print(f"  [AUTO] Found good solution with {algo_name}, stopping early")
                        break

            except Exception as e:
                print(f"  [AUTO] {algo_name} failed: {e}")

        # Try GA as last resort if no good solution
        if best_result is None or best_score > 5000:
            print("\n  [AUTO] No good result yet, trying genetic algorithm...")
            self._restore_state(initial_state)
            try:
                result = self._place_genetic()
                has_overlaps = self._check_has_overlaps()
                has_oob = self._check_out_of_bounds()
                score = result.cost + (10000 if has_overlaps else 0) + (5000 if has_oob else 0)

                if score < best_score:
                    best_score = score
                    best_result = result
                    best_positions = self._save_state()
            except Exception as e:
                print(f"  [AUTO] GA failed: {e}")

        # Restore best solution
        if best_positions:
            self._restore_state(best_positions)
            print(f"\n  [AUTO] Selected: {best_result.algorithm_used} (cost={best_result.cost:.2f})")

            return self._create_result(
                f"auto ({best_result.algorithm_used})",
                sum(r.iterations for _, r, _ in results),
                True,
            )
        else:
            print("  [AUTO] WARNING: All algorithms failed")
            return self._create_result('auto (fallback)', 0, False)

    # =========================================================================
    # PARALLEL PLACEMENT (Run ALL algorithms simultaneously)
    # =========================================================================

    def _place_parallel(self) -> PlacementResult:
        """
        Run ALL placement algorithms in parallel and pick the best result.

        Uses ThreadPoolExecutor for concurrent execution.
        Each algorithm runs in isolation with its own copy of state.
        """
        print("  [PARALLEL] Running ALL algorithms in parallel...")

        algorithms = ['fd', 'sa', 'ga', 'analytical', 'human', 'hybrid']

        results = {}
        results_lock = threading.Lock()

        def run_algorithm(algo_name: str):
            """Run a single algorithm in isolation"""
            try:
                # Create isolated engine instance
                isolated_config = copy.deepcopy(self.config)
                isolated_config.algorithm = algo_name
                # Vary seed for diversity
                if isolated_config.seed is not None:
                    isolated_config.seed = isolated_config.seed + hash(algo_name) % 10000

                isolated_engine = PlacementEngine(isolated_config)
                isolated_engine._copy_state_from(self)

                # Run algorithm
                if algo_name == 'fd':
                    result = isolated_engine._place_force_directed()
                elif algo_name == 'sa':
                    result = isolated_engine._place_simulated_annealing()
                elif algo_name == 'ga':
                    result = isolated_engine._place_genetic()
                elif algo_name == 'analytical':
                    result = isolated_engine._place_analytical()
                elif algo_name == 'human':
                    result = isolated_engine._place_human_like()
                elif algo_name == 'hybrid':
                    result = isolated_engine._place_hybrid()
                else:
                    return

                # Evaluate quality
                has_overlaps = isolated_engine._check_has_overlaps()
                has_oob = isolated_engine._check_out_of_bounds()

                with results_lock:
                    results[algo_name] = {
                        'result': result,
                        'has_overlaps': has_overlaps,
                        'has_oob': has_oob,
                        'positions': {ref: (c.x, c.y) for ref, c in isolated_engine.components.items()},
                        'rotations': {ref: c.rotation for ref, c in isolated_engine.components.items()},
                    }

                quality = "GOOD" if not has_overlaps and not has_oob else "POOR"
                print(f"  [PARALLEL] {algo_name}: cost={result.cost:.2f}, "
                      f"overlaps={has_overlaps}, oob={has_oob} -> {quality}")

            except Exception as e:
                print(f"  [PARALLEL] {algo_name} failed: {e}")

        # Run all algorithms in parallel
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {executor.submit(run_algorithm, algo): algo for algo in algorithms}

            for future in as_completed(futures, timeout=self.config.parallel_timeout):
                algo = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [PARALLEL] {algo} exception: {e}")

        # Find best result
        best_algo = None
        best_score = float('inf')
        best_data = None

        for algo_name, data in results.items():
            result = data['result']
            score = result.cost
            if data['has_overlaps']:
                score += 10000
            if data['has_oob']:
                score += 5000

            if score < best_score:
                best_score = score
                best_algo = algo_name
                best_data = data

        if best_algo and best_data:
            print(f"\n  [PARALLEL] Best: {best_algo} (cost={best_data['result'].cost:.2f})")

            # Apply best positions
            for ref, (x, y) in best_data['positions'].items():
                if ref in self.components:
                    self.components[ref].x = x
                    self.components[ref].y = y
            for ref, rot in best_data['rotations'].items():
                if ref in self.components:
                    self.components[ref].rotation = rot

            return self._create_result(
                f"parallel ({best_algo})",
                sum(d['result'].iterations for d in results.values()),
                True,
            )
        else:
            print("  [PARALLEL] WARNING: All algorithms failed!")
            return self._create_result('parallel (fallback)', 0, False)

    def _copy_state_from(self, other: 'PlacementEngine'):
        """Copy state from another engine instance"""
        self.nets = copy.deepcopy(other.nets)
        self.adjacency = copy.deepcopy(other.adjacency)
        self.hub = other.hub
        self.pin_offsets = copy.deepcopy(other.pin_offsets)
        self.pin_sizes = copy.deepcopy(other.pin_sizes)
        self.pin_nets = copy.deepcopy(other.pin_nets)
        self.net_weights = copy.deepcopy(other.net_weights)
        self._optimal_dist = other._optimal_dist

        self.components = {}
        for ref, comp in other.components.items():
            self.components[ref] = comp.copy()

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================

    def _calculate_cost(self) -> float:
        """
        Calculate placement cost (lower is better).

        Uses JIT-compiled kernels when Numba is available (10-50x faster).
        Falls back to pure Python otherwise.

        Cost components (weight rationale):
        - Physical constraints (must-fix): pad_conflict*5000, overlap*1000, oob*500
        - Electrical hints (guide): keep_apart*200, proximity*150, edge*100
        - Optimization: wirelength*1, orientation*5
        """
        # Fast path: JIT-compiled cost (skips pad_conflict + orientation for speed,
        # those are checked separately in SA accept/reject)
        if self._jit is not None:
            try:
                self._jit.sync_from_engine(self)
                return self._jit.total_cost()
            except Exception:
                # JIT failed — disable and fall back
                self._jit = None

        # Slow path: pure Python
        wirelength = self._calculate_wirelength()
        overlap = self._calculate_overlap_area()
        pad_conflict = self._calculate_pad_conflict_penalty()
        oob = self._calculate_oob_penalty()
        prox = self._calculate_proximity_cost()
        edge = self._calculate_edge_cost()
        keep_apart = self._calculate_keep_apart_cost()
        orientation = self._calculate_orientation_cost()

        return (wirelength +
                overlap * 1000 +
                pad_conflict * 5000 +
                oob * 500 +
                prox * 150 +
                edge * 100 +
                keep_apart * 200 +
                orientation * 5)

    def _calculate_wirelength(self) -> float:
        """Calculate total estimated wire length using HPWL"""
        total = 0.0

        for net_name, net_info in self.nets.items():
            pins = net_info.get('pins', [])
            weight = net_info.get('weight', 1.0)

            if len(pins) < 2:
                continue

            # Get bounding box of all pins
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')

            for pin_ref in pins:
                comp_ref, pin_num = self._parse_pin_ref(pin_ref)
                if comp_ref not in self.components:
                    continue
                comp = self.components[comp_ref]

                # Get pin position
                offset = self.pin_offsets.get(comp_ref, {}).get(pin_num, (0, 0))
                px = comp.x + offset[0]
                py = comp.y + offset[1]

                min_x = min(min_x, px)
                max_x = max(max_x, px)
                min_y = min(min_y, py)
                max_y = max(max_y, py)

            # HPWL (Half Perimeter Wire Length)
            if min_x != float('inf'):
                total += ((max_x - min_x) + (max_y - min_y)) * weight

        return total

    def _calculate_overlap_area(self) -> float:
        """Calculate total overlap area between components"""
        total = 0.0
        refs = list(self.components.keys())

        for i, ref_a in enumerate(refs):
            for ref_b in refs[i + 1:]:
                ca = self.components[ref_a]
                cb = self.components[ref_b]

                half_w_a = ca.width / 2 + self.config.min_spacing / 2
                half_h_a = ca.height / 2 + self.config.min_spacing / 2
                half_w_b = cb.width / 2 + self.config.min_spacing / 2
                half_h_b = cb.height / 2 + self.config.min_spacing / 2

                overlap_x = max(0, (half_w_a + half_w_b) - abs(ca.x - cb.x))
                overlap_y = max(0, (half_h_a + half_h_b) - abs(ca.y - cb.y))

                total += overlap_x * overlap_y

        return total

    def _calculate_pad_conflict_penalty(self) -> float:
        """
        Calculate penalty for pads of DIFFERENT nets that are too close together.

        This is critical to prevent DRC failures where pads overlap (pad clearance violation).
        Components can be placed close together, but their pads from different nets must
        maintain minimum clearance.

        Returns penalty value (higher = worse placement).
        """
        total = 0.0
        min_clearance = 0.15  # Minimum clearance between pads of different nets (mm)

        refs = list(self.components.keys())

        for i, ref_a in enumerate(refs):
            comp_a = self.components[ref_a]
            pins_a = self.pin_offsets.get(ref_a, {})
            sizes_a = self.pin_sizes.get(ref_a, {})
            nets_a = self.pin_nets.get(ref_a, {})

            for ref_b in refs[i + 1:]:
                comp_b = self.components[ref_b]
                pins_b = self.pin_offsets.get(ref_b, {})
                sizes_b = self.pin_sizes.get(ref_b, {})
                nets_b = self.pin_nets.get(ref_b, {})

                # Check all pin pairs between these two components
                for pin_num_a, offset_a in pins_a.items():
                    net_a = nets_a.get(pin_num_a, '')
                    size_a = sizes_a.get(pin_num_a, (1.0, 0.6))

                    # Pin A absolute position
                    px_a = comp_a.x + offset_a[0]
                    py_a = comp_a.y + offset_a[1]
                    half_w_a = size_a[0] / 2
                    half_h_a = size_a[1] / 2

                    for pin_num_b, offset_b in pins_b.items():
                        net_b = nets_b.get(pin_num_b, '')

                        # Skip if same net (same net pads can overlap)
                        if net_a == net_b and net_a != '':
                            continue

                        size_b = sizes_b.get(pin_num_b, (1.0, 0.6))

                        # Pin B absolute position
                        px_b = comp_b.x + offset_b[0]
                        py_b = comp_b.y + offset_b[1]
                        half_w_b = size_b[0] / 2
                        half_h_b = size_b[1] / 2

                        # Check if pad bounding boxes overlap (with clearance)
                        req_gap_x = half_w_a + half_w_b + min_clearance
                        req_gap_y = half_h_a + half_h_b + min_clearance

                        gap_x = abs(px_a - px_b)
                        gap_y = abs(py_a - py_b)

                        # Both X and Y must have enough gap
                        overlap_x = max(0, req_gap_x - gap_x)
                        overlap_y = max(0, req_gap_y - gap_y)

                        if overlap_x > 0 and overlap_y > 0:
                            # Pads are too close - calculate penalty
                            conflict = overlap_x * overlap_y
                            total += conflict

        return total

    def _check_pad_pair_conflict(self, ref_a: str, ref_b: str) -> float:
        """Check if two specific components have any pad conflict."""
        min_clearance = 0.15

        comp_a = self.components.get(ref_a)
        comp_b = self.components.get(ref_b)
        if not comp_a or not comp_b:
            return 0.0

        pins_a = self.pin_offsets.get(ref_a, {})
        sizes_a = self.pin_sizes.get(ref_a, {})
        nets_a = self.pin_nets.get(ref_a, {})

        pins_b = self.pin_offsets.get(ref_b, {})
        sizes_b = self.pin_sizes.get(ref_b, {})
        nets_b = self.pin_nets.get(ref_b, {})

        conflict = 0.0
        for pin_num_a, offset_a in pins_a.items():
            net_a = nets_a.get(pin_num_a, '')
            size_a = sizes_a.get(pin_num_a, (1.0, 0.6))
            px_a = comp_a.x + offset_a[0]
            py_a = comp_a.y + offset_a[1]
            half_w_a = size_a[0] / 2
            half_h_a = size_a[1] / 2

            for pin_num_b, offset_b in pins_b.items():
                net_b = nets_b.get(pin_num_b, '')
                if net_a == net_b and net_a != '':
                    continue

                size_b = sizes_b.get(pin_num_b, (1.0, 0.6))
                px_b = comp_b.x + offset_b[0]
                py_b = comp_b.y + offset_b[1]
                half_w_b = size_b[0] / 2
                half_h_b = size_b[1] / 2

                req_gap_x = half_w_a + half_w_b + min_clearance
                req_gap_y = half_h_a + half_h_b + min_clearance
                gap_x = abs(px_a - px_b)
                gap_y = abs(py_a - py_b)

                overlap_x = max(0, req_gap_x - gap_x)
                overlap_y = max(0, req_gap_y - gap_y)

                if overlap_x > 0 and overlap_y > 0:
                    conflict += overlap_x * overlap_y

        return conflict

    def _calculate_oob_penalty(self) -> float:
        """Calculate out-of-bounds penalty"""
        total = 0.0

        for comp in self.components.values():
            margin = self.config.edge_margin
            half_w = comp.width / 2
            half_h = comp.height / 2

            min_x = self.config.origin_x + margin + half_w
            max_x = self.config.origin_x + self.config.board_width - margin - half_w
            min_y = self.config.origin_y + margin + half_h
            max_y = self.config.origin_y + self.config.board_height - margin - half_h

            if comp.x < min_x:
                total += min_x - comp.x
            if comp.x > max_x:
                total += comp.x - max_x
            if comp.y < min_y:
                total += min_y - comp.y
            if comp.y > max_y:
                total += comp.y - max_y

        return total

    def _calculate_proximity_cost(self) -> float:
        """
        Continuous proximity penalty for functionally-related components.

        Unlike a simple threshold penalty, this provides gradient at ALL
        distances — components always feel a pull toward their functional
        partner.  The penalty is quadratic beyond target distance (strong
        repulsion from bad placements) and linear below it (gentle pull
        toward optimal placement).

        Generalized: works for ANY functional role (decoupling caps,
        pull-ups, LED drivers, ESD, etc.) using the priority and
        distance fields already inferred by _build_placement_constraints().
        """
        hints = getattr(self, '_placement_hints', None)
        if not hints:
            return 0.0

        groups = hints.get('proximity_groups', [])
        if not groups:
            return 0.0

        total = 0.0
        for group in groups:
            components = group.get('components', [])
            target_dist = group.get('max_distance', 10.0)
            priority = group.get('priority', 1.0)

            for i, ref_a in enumerate(components):
                if ref_a not in self.components:
                    continue
                ca = self.components[ref_a]
                for ref_b in components[i + 1:]:
                    if ref_b not in self.components:
                        continue
                    cb = self.components[ref_b]
                    dist = math.sqrt((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2)

                    if dist <= target_dist:
                        # Within target: gentle linear pull (still want closer)
                        # Normalized so cost=0 at dist=0, cost=priority at dist=target
                        total += (dist / target_dist) * priority
                    else:
                        # Beyond target: quadratic penalty (grows fast)
                        # Continuous at boundary: priority + priority*(excess/target)^2
                        excess = dist - target_dist
                        total += priority + priority * (excess / target_dist) ** 2

        return total

    def _calculate_edge_cost(self) -> float:
        """Quadratic penalty when edge_components are far from board edges.

        Connectors should be flush with board edge. Quadratic scaling
        makes the penalty grow fast with distance, strongly discouraging
        connectors in the middle of the board.
        """
        hints = getattr(self, '_placement_hints', None)
        if not hints:
            return 0.0
        edge_refs = hints.get('edge_components', [])
        if not edge_refs:
            return 0.0
        total = 0.0
        bw, bh = self.config.board_width, self.config.board_height
        ox, oy = self.config.origin_x, self.config.origin_y
        for ref in edge_refs:
            if ref not in self.components:
                continue
            c = self.components[ref]
            half_w = c.width / 2
            half_h = c.height / 2
            # Distance from courtyard edge to board edge
            d_left = c.x - half_w - ox
            d_right = (ox + bw) - (c.x + half_w)
            d_top = c.y - half_h - oy
            d_bottom = (oy + bh) - (c.y + half_h)
            nearest = max(0, min(d_left, d_right, d_top, d_bottom))
            # Quadratic penalty — grows fast with distance
            total += nearest * nearest * 5.0
        return total

    def _calculate_keep_apart_cost(self) -> float:
        """Penalty when keep_apart pairs are too close."""
        hints = getattr(self, '_placement_hints', None)
        if not hints:
            return 0.0
        pairs = hints.get('keep_apart', [])
        total = 0.0
        for pair in pairs:
            a, b = pair.get('a'), pair.get('b')
            min_dist = pair.get('min_distance', 10.0)
            if a not in self.components or b not in self.components:
                continue
            ca, cb = self.components[a], self.components[b]
            dist = math.sqrt((ca.x - cb.x)**2 + (ca.y - cb.y)**2)
            if dist < min_dist:
                total += (min_dist - dist) * 2.0
        return total

    def _calculate_orientation_cost(self) -> float:
        """Penalty for inconsistent passive orientation.

        After FD determines the preferred orientation (0 or 90 deg),
        each small passive at a non-preferred angle adds a penalty.
        This biases SA toward consistent orientation.
        """
        preferred = getattr(self, '_preferred_passive_orientation', None)
        if preferred is None:
            return 0.0
        refs = getattr(self, '_small_passive_refs', set())
        if not refs:
            return 0.0

        count_wrong = 0
        for ref in refs:
            if ref not in self.components:
                continue
            rot = self.components[ref].rotation % 180  # 0/180 same, 90/270 same
            if abs(rot - (preferred % 180)) > 1.0:
                count_wrong += 1

        return float(count_wrong)

    def _check_has_overlaps(self) -> bool:
        """Check if any components overlap"""
        return self._calculate_overlap_area() > 0.01

    def _check_out_of_bounds(self) -> bool:
        """Check if any component is out of bounds"""
        return self._calculate_oob_penalty() > 0.01

    def _clamp_to_board(self, comp: ComponentState) -> Tuple[float, float]:
        """Clamp component position to board bounds"""
        margin = self.config.edge_margin
        half_w = comp.width / 2
        half_h = comp.height / 2

        x = max(self.config.origin_x + margin + half_w,
                min(self.config.origin_x + self.config.board_width - margin - half_w, comp.x))
        y = max(self.config.origin_y + margin + half_h,
                min(self.config.origin_y + self.config.board_height - margin - half_h, comp.y))

        return x, y

    def _sync_occupancy_grid(self):
        """Rebuild occupancy grid from current component positions."""
        if not hasattr(self, '_occ') or not self._occ:
            return
        self._occ.clear()
        for ref, comp in self.components.items():
            self._occ.place(ref, comp.x, comp.y, comp.width, comp.height)

    def _clamp_to_board_at(self, x: float, y: float,
                           width: float, height: float) -> Tuple[float, float]:
        """Clamp an arbitrary position to board bounds given component dimensions."""
        margin = self.config.edge_margin
        half_w = width / 2
        half_h = height / 2
        x = max(self.config.origin_x + margin + half_w,
                min(self.config.origin_x + self.config.board_width - margin - half_w, x))
        y = max(self.config.origin_y + margin + half_h,
                min(self.config.origin_y + self.config.board_height - margin - half_h, y))
        return x, y

    def _legalize(self):
        """Legalize placement using occupancy grid — zero overlaps guaranteed."""
        # Snap connectors to board edge FIRST (they get priority)
        edge_refs = set()
        hints = getattr(self, '_placement_hints', None)
        if hints:
            edge_refs = set(hints.get('edge_components', []))

        # Sort: edge components first (by area desc), then others (by area desc)
        all_refs = list(self.components.keys())
        edge_sorted = sorted([r for r in all_refs if r in edge_refs],
                             key=lambda r: self.components[r].width * self.components[r].height,
                             reverse=True)
        other_sorted = sorted([r for r in all_refs if r not in edge_refs],
                              key=lambda r: self.components[r].width * self.components[r].height,
                              reverse=True)
        sorted_refs = edge_sorted + other_sorted

        # Snap connectors to nearest board edge
        ox = self.config.origin_x
        oy = self.config.origin_y
        bw = self.config.board_width
        bh = self.config.board_height

        for ref in edge_refs:
            if ref not in self.components:
                continue
            comp = self.components[ref]
            if comp.fixed:
                continue

            half_w = comp.width / 2
            half_h = comp.height / 2

            # Distance to each edge
            d_left = comp.x - ox
            d_right = (ox + bw) - comp.x
            d_top = comp.y - oy
            d_bottom = (oy + bh) - comp.y

            nearest = min(d_left, d_right, d_top, d_bottom)

            # Snap to the nearest edge (courtyard flush with board boundary)
            if nearest == d_left:
                comp.x = ox + half_w
            elif nearest == d_right:
                comp.x = ox + bw - half_w
            elif nearest == d_top:
                comp.y = oy + half_h
            else:
                comp.y = oy + bh - half_h

        # Clear grid and re-place in size order
        if hasattr(self, '_occ') and self._occ:
            self._occ.clear()
        else:
            return  # No occupancy grid — skip

        for ref in sorted_refs:
            comp = self.components[ref]
            # Snap to grid
            comp.x = round(comp.x / self.config.grid_size) * self.config.grid_size
            comp.y = round(comp.y / self.config.grid_size) * self.config.grid_size
            comp.x, comp.y = self._clamp_to_board(comp)

            # If current position is free, take it
            if self._occ.can_place(comp.x, comp.y, comp.width, comp.height):
                self._occ.place(ref, comp.x, comp.y, comp.width, comp.height)
                continue

            # Find nearest free position (expanding spiral, full board radius)
            found = False
            max_radius = max(self.config.board_width, self.config.board_height)
            step = self.config.grid_size
            for radius_steps in range(1, int(max_radius / step) + 1):
                radius = radius_steps * step
                for angle_deg in range(0, 360, 15):  # 24 directions per radius
                    nx = comp.x + radius * math.cos(math.radians(angle_deg))
                    ny = comp.y + radius * math.sin(math.radians(angle_deg))
                    nx = round(nx / self.config.grid_size) * self.config.grid_size
                    ny = round(ny / self.config.grid_size) * self.config.grid_size
                    nx, ny = self._clamp_to_board_at(nx, ny, comp.width, comp.height)

                    if self._occ.can_place(nx, ny, comp.width, comp.height):
                        comp.x, comp.y = nx, ny
                        self._occ.place(ref, comp.x, comp.y, comp.width, comp.height)
                        found = True
                        break
                if found:
                    break

            if not found:
                # Last resort: place anyway (board may be too small for all components)
                self._occ.place(ref, comp.x, comp.y, comp.width, comp.height)

    def _save_state(self) -> Dict:
        """Save current component state (including occupancy grid)."""
        state = {ref: (c.x, c.y, c.rotation, c.width, c.height)
                 for ref, c in self.components.items()}
        if hasattr(self, '_occ') and self._occ:
            state['__occ__'] = self._occ.save_state()
        return state

    def _restore_state(self, state: Dict):
        """Restore component state (including occupancy grid)."""
        occ_state = state.get('__occ__', None)
        for ref, val in state.items():
            if ref == '__occ__':
                continue
            if ref in self.components and isinstance(val, tuple):
                x, y, rot, w, h = val
                self.components[ref].x = x
                self.components[ref].y = y
                self.components[ref].rotation = rot
                self.components[ref].width = w
                self.components[ref].height = h
        if occ_state and hasattr(self, '_occ') and self._occ:
            self._occ.restore_state(occ_state)

    def _create_result(self, algorithm: str, iterations: int, converged: bool) -> PlacementResult:
        """Create placement result from current state"""
        return PlacementResult(
            positions={ref: (c.x, c.y) for ref, c in self.components.items()},
            rotations={ref: c.rotation for ref, c in self.components.items()},
            cost=self._calculate_cost(),
            algorithm_used=algorithm,
            iterations=iterations,
            converged=converged,
            wirelength=self._calculate_wirelength(),
            overlap_area=self._calculate_overlap_area(),
            board_width=self.config.board_width,
            board_height=self.config.board_height,
        )

    def _post_unfuse_jitter(self):
        """Jitter unfused passives to reduce WL while staying near their owner IC.

        After unfuse, passives are on the IC perimeter — correct side but not
        necessarily WL-optimal position. This does a quick greedy pass: for each
        unfused passive, try 8 small shifts and keep the one that reduces cost
        without moving more than 2mm from the IC.
        """
        if not self._fusion or not self._fusion.fused_components:
            return

        # Collect all unfused passive refs and their owner positions
        unfused_refs = []
        owner_pos = {}
        leash = {}  # ref -> max distance from owner
        for fused in self._fusion.fused_components:
            owner_ref = fused.owner_ref
            if owner_ref not in self.components:
                continue
            oc = self.components[owner_ref]
            for ref in fused.fused_refs:
                if ref in self.components:
                    unfused_refs.append(ref)
                    owner_pos[ref] = (oc.x, oc.y)
                    # Leash = IC half-diagonal + 3mm
                    leash[ref] = math.sqrt(oc.width**2 + oc.height**2) / 2 + 3.0

        if not unfused_refs:
            return

        moves = 0
        step = 0.5  # mm jitter step
        directions = [(step, 0), (-step, 0), (0, step), (0, -step),
                      (step, step), (-step, step), (step, -step), (-step, -step)]

        for _ in range(3):  # 3 passes
            for ref in unfused_refs:
                comp = self.components[ref]
                old_x, old_y = comp.x, comp.y
                old_cost = self._calculate_wirelength()

                best_dx, best_dy = 0, 0
                best_cost = old_cost

                for dx, dy in directions:
                    nx = old_x + dx
                    ny = old_y + dy

                    # Check leash constraint
                    ox, oy = owner_pos[ref]
                    if math.sqrt((nx - ox)**2 + (ny - oy)**2) > leash[ref]:
                        continue

                    # Check board bounds
                    half_w, half_h = comp.width / 2, comp.height / 2
                    if (nx - half_w < self.config.origin_x or
                        nx + half_w > self.config.origin_x + self.config.board_width or
                        ny - half_h < self.config.origin_y or
                        ny + half_h > self.config.origin_y + self.config.board_height):
                        continue

                    # Check occupancy
                    if hasattr(self, '_occ') and self._occ:
                        self._occ.remove(ref, old_x, old_y, comp.width, comp.height)
                        can = self._occ.can_place(nx, ny, comp.width, comp.height)
                        if not can:
                            self._occ.place(ref, old_x, old_y, comp.width, comp.height)
                            continue
                        self._occ.place(ref, old_x, old_y, comp.width, comp.height)

                    # Evaluate WL
                    comp.x, comp.y = nx, ny
                    new_cost = self._calculate_wirelength()
                    comp.x, comp.y = old_x, old_y

                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_dx, best_dy = dx, dy

                if best_dx != 0 or best_dy != 0:
                    nx, ny = old_x + best_dx, old_y + best_dy
                    if hasattr(self, '_occ') and self._occ:
                        self._occ.remove(ref, old_x, old_y, comp.width, comp.height)
                        self._occ.place(ref, nx, ny, comp.width, comp.height)
                    comp.x, comp.y = nx, ny
                    moves += 1

        if moves > 0:
            print(f"  [FUSION] Post-unfuse jitter: {moves} moves")

    def _shrink_board_to_fit(self):
        """Shrink board to tightly fit placed components + routing margin.

        Called after placement when auto_board_size is True. Finds the actual
        bounding box of all components (including courtyards) and sets the
        board size to that + edge margin, rounded to 0.5mm.
        """
        margin = max(self.config.edge_margin, 1.0)
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for comp in self.components.values():
            half_w = comp.width / 2
            half_h = comp.height / 2
            min_x = min(min_x, comp.x - half_w)
            max_x = max(max_x, comp.x + half_w)
            min_y = min(min_y, comp.y - half_h)
            max_y = max(max_y, comp.y + half_h)

        if min_x == float('inf'):
            return  # No components

        # Board size = bounding box + margin on each side, rounded up to 0.5mm
        raw_w = (max_x - min_x) + 2 * margin
        raw_h = (max_y - min_y) + 2 * margin
        increment = 0.5
        new_w = math.ceil(raw_w / increment) * increment
        new_h = math.ceil(raw_h / increment) * increment

        # Shift all components so the board starts at origin
        shift_x = margin - min_x
        shift_y = margin - min_y
        for comp in self.components.values():
            comp.x += shift_x
            comp.y += shift_y

        self.config.board_width = new_w
        self.config.board_height = new_h
        self.config.origin_x = 0.0
        self.config.origin_y = 0.0
        print(f"  [PLACEMENT] Board SHRUNK to fit: {new_w}x{new_h}mm")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_placement(parts_db: Dict, graph: Dict,
                  algorithm: str = 'hybrid',
                  board_width: float = 50.0,
                  board_height: float = 40.0,
                  origin_x: float = 0.0,
                  origin_y: float = 0.0,
                  grid_size: float = 0.5,
                  seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Convenience function for running placement.

    Args:
        parts_db: Parts database
        graph: Connectivity graph
        algorithm: 'fd', 'sa', 'ga', 'analytical', 'human', 'hybrid', 'auto', 'parallel'
        board_width/height: Board dimensions in mm
        origin_x/y: Board origin in mm
        grid_size: Placement grid in mm
        seed: Random seed for reproducibility

    Returns:
        Dictionary of component positions {ref: (x, y)}
    """
    config = PlacementConfig(
        algorithm=algorithm,
        board_width=board_width,
        board_height=board_height,
        origin_x=origin_x,
        origin_y=origin_y,
        grid_size=grid_size,
        seed=seed
    )

    engine = PlacementEngine(config)
    result = engine.place(parts_db, graph)

    print(f"  Placement complete: {result.algorithm_used}")
    print(f"  Cost: {result.cost:.2f}, Wire length: {result.wirelength:.2f}")
    print(f"  Iterations: {result.iterations}, Converged: {result.converged}")

    return result.positions
