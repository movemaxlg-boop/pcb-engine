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
except ImportError:
    from common_types import get_pins


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
    power_net_weight: float = 2.0     # Weight for power nets (GND, VCC, etc.)
    critical_net_weight: float = 3.0  # Weight for critical signal nets


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

        # Initialize state from parts database
        self._init_from_parts(parts_db, graph)

        # Compute optimal distance for force-directed
        n = len(self.components)
        if n > 0:
            area = self.config.board_width * self.config.board_height
            self._optimal_dist = math.sqrt(area / n) * 0.8

        # Select algorithm
        algo = PlacementAlgorithm(self.config.algorithm)

        if algo == PlacementAlgorithm.FORCE_DIRECTED:
            return self._place_force_directed()
        elif algo == PlacementAlgorithm.SIMULATED_ANNEALING:
            return self._place_simulated_annealing()
        elif algo == PlacementAlgorithm.GENETIC:
            return self._place_genetic()
        elif algo == PlacementAlgorithm.ANALYTICAL:
            return self._place_analytical()
        elif algo == PlacementAlgorithm.HUMAN:
            return self._place_human_like()
        elif algo == PlacementAlgorithm.HYBRID:
            return self._place_hybrid()
        elif algo == PlacementAlgorithm.AUTO:
            return self._place_auto()
        elif algo == PlacementAlgorithm.PARALLEL:
            return self._place_parallel()
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

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
                # Fallback: Calculate courtyard from pad positions
                # This ensures placement accounts for pads that extend beyond body
                # (e.g., SOT-223 tab pad extends 3.25mm above center)
                size = part.get('size', (2.0, 2.0))
                base_width = size[0] if isinstance(size, (list, tuple)) else 2.0
                base_height = size[1] if isinstance(size, (list, tuple)) else 2.0

                # Calculate actual courtyard bounds from pad positions
                used_pins = get_pins(part)
                min_x, max_x, min_y, max_y = 0, 0, 0, 0
                courtyard_margin = 0.25  # IPC standard courtyard margin

                for pin in used_pins:
                    offset = pin.get('offset', (0, 0))
                    pad_size = pin.get('size', pin.get('pad_size', (1.0, 0.6)))
                    if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                        ox, oy = offset[0], offset[1]
                        pw = pad_size[0] / 2 if isinstance(pad_size, (list, tuple)) else 0.5
                        ph = pad_size[1] / 2 if isinstance(pad_size, (list, tuple)) and len(pad_size) > 1 else 0.3
                        # Track pad extents
                        min_x = min(min_x, ox - pw - courtyard_margin)
                        max_x = max(max_x, ox + pw + courtyard_margin)
                        min_y = min(min_y, oy - ph - courtyard_margin)
                        max_y = max(max_y, oy + ph + courtyard_margin)

                # Use the larger of body size or pad-based courtyard
                courtyard_width = max(base_width, max_x - min_x)
                courtyard_height = max(base_height, max_y - min_y)
                width = courtyard_width
                height = courtyard_height

            # Initial position: spread around center using golden angle
            idx = len(self.components)
            if ref == self.hub:
                x, y = cx, cy
            else:
                # Golden angle spiral for initial spread
                golden_angle = 137.508 * math.pi / 180
                angle = idx * golden_angle
                r = self.config.board_width * 0.3 * math.sqrt(idx + 1) / math.sqrt(len(parts) + 1)
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

        for iteration in range(self.config.fd_iterations):
            # Reset forces
            for comp in self.components.values():
                comp.fx = 0.0
                comp.fy = 0.0

            # Calculate repulsive forces between ALL component pairs
            refs = list(self.components.keys())
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

            # Apply boundary forces to keep components inside board
            for comp in self.components.values():
                self._apply_boundary_force(comp)

            # Update positions based on forces, limited by temperature
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

                    # Update position
                    comp.x += dx
                    comp.y += dy

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

        Force magnitude: f_r = k^2 / d
        Direction: Away from each other
        """
        ca = self.components[ref_a]
        cb = self.components[ref_b]

        dx = ca.x - cb.x
        dy = ca.y - cb.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # Avoid division by zero
        if dist < 0.01:
            dist = 0.01
            dx = random.uniform(-0.01, 0.01)
            dy = random.uniform(-0.01, 0.01)

        # Repulsive force (stronger when close)
        # Include component size in calculation
        min_dist = (ca.width + cb.width) / 2 + self.config.min_spacing
        effective_k = k * (1 + min_dist / k)

        force = (effective_k ** 2) / dist

        # Apply force in direction away from each other
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

        while temp > self.config.sa_final_temp:
            for _ in range(moves_per_temp):
                total_iterations += 1
                old_state = self._save_state()

                r = rng.random()
                if r < 0.5:
                    # Inline shift with worker's RNG
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if refs:
                        ref = rng.choice(refs)
                        comp = self.components[ref]
                        max_shift = (temp / initial_temp) * self.config.board_width * 0.3
                        max_shift = max(max_shift, self.config.grid_size)
                        comp.x += rng.uniform(-max_shift, max_shift)
                        comp.y += rng.uniform(-max_shift, max_shift)
                        comp.x, comp.y = self._clamp_to_board(comp)
                elif r < 0.75:
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if len(refs) >= 2:
                        a, b = rng.sample(refs, 2)
                        ca, cb = self.components[a], self.components[b]
                        ca.x, cb.x = cb.x, ca.x
                        ca.y, cb.y = cb.y, ca.y
                elif r < 0.90:
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if refs:
                        ref = rng.choice(refs)
                        comp = self.components[ref]
                        comp.rotation = (comp.rotation + 90) % 360
                        comp.width, comp.height = comp.height, comp.width
                        comp.x, comp.y = self._clamp_to_board(comp)
                else:
                    refs = [ref for ref, c in self.components.items() if not c.fixed]
                    if refs:
                        ref = rng.choice(refs)
                        comp = self.components[ref]
                        comp.rotation = (comp.rotation + 180) % 360
                        comp.x, comp.y = self._clamp_to_board(comp)

                new_cost = self._calculate_cost()
                delta = new_cost - current_cost

                pad_conflict = self._calculate_pad_conflict_penalty()
                if pad_conflict > 0.01:
                    self._restore_state(old_state)
                    rejections += 1
                    continue

                if delta < 0 or rng.random() < math.exp(-delta / temp):
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

        return best_cost, best_state, total_iterations, accepted_moves

    def _place_simulated_annealing(self) -> PlacementResult:
        """
        Multi-start simulated annealing placement.

        Runs multiple SA passes with different random seeds and initial temperatures,
        each exploring a different region of the solution space. The best result
        across all workers is selected.

        Reference: "Optimization by Simulated Annealing" (Kirkpatrick, 1983)
        Multi-start: (Ram et al., "Parallel Simulated Annealing Algorithms", 1996)
        """
        import time as _time

        num_workers = min(os.cpu_count() or 4, 8)  # Cap at 8
        starting_state = self._save_state()
        base_temp = self.config.sa_initial_temp

        # Reduce moves per worker so total time stays similar
        moves_per_temp = max(self.config.sa_moves_per_temp // num_workers, 20)

        print(f"  [SA] Running multi-start simulated annealing ({num_workers} starts)...")
        sa_start = _time.time()

        results = []
        for i in range(num_workers):
            seed = 42 + i * 1000
            temp_factor = 0.7 + (i / max(num_workers - 1, 1)) * 0.6  # 0.7x to 1.3x
            temp = base_temp * temp_factor

            self._restore_state(starting_state)
            cost, state, iters, acc = self._run_sa_single(seed, temp, moves_per_temp, starting_state)
            results.append((cost, state, iters, acc, temp))

        # Pick best
        results.sort(key=lambda r: r[0])
        best_cost, best_state, best_iters, best_acc, best_temp = results[0]

        self._restore_state(best_state)
        self._legalize()

        sa_elapsed = _time.time() - sa_start
        acceptance_rate = best_acc / best_iters if best_iters > 0 else 0
        print(f"  [SA] Final cost: {best_cost:.2f}, acceptance rate: {acceptance_rate:.2%}")
        print(f"  [SA] {num_workers} starts in {sa_elapsed:.1f}s, best temp={best_temp:.1f}")

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

        return PlacementResult(
            positions={ref: (c.x, c.y) for ref, c in self.components.items()},
            rotations={ref: c.rotation for ref, c in self.components.items()},
            cost=sa_result.cost,
            algorithm_used='hybrid (fd+sa)',
            iterations=fd_result.iterations + sa_result.iterations,
            converged=True,
            wirelength=self._calculate_wirelength(),
            overlap_area=self._calculate_overlap_area()
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

            return PlacementResult(
                positions={ref: (c.x, c.y) for ref, c in self.components.items()},
                rotations={ref: c.rotation for ref, c in self.components.items()},
                cost=best_result.cost,
                algorithm_used=f"auto ({best_result.algorithm_used})",
                iterations=sum(r.iterations for _, r, _ in results),
                converged=True
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

            return PlacementResult(
                positions=best_data['positions'],
                rotations=best_data['rotations'],
                cost=best_data['result'].cost,
                algorithm_used=f"parallel ({best_algo})",
                iterations=sum(d['result'].iterations for d in results.values()),
                converged=True
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

        Cost = Wire Length + Overlap Penalty + Pad Conflict Penalty + OOB Penalty

        Pad Conflict Penalty is CRITICAL - it prevents placing components such that
        pads from different nets overlap, which causes DRC failures and unroutable nets.
        """
        wirelength = self._calculate_wirelength()
        overlap = self._calculate_overlap_area()
        pad_conflict = self._calculate_pad_conflict_penalty()
        oob = self._calculate_oob_penalty()
        prox = self._calculate_proximity_cost()

        # Pad conflict has VERY HIGH weight because overlapping pads = guaranteed DRC fail
        return wirelength + overlap * 1000 + pad_conflict * 5000 + oob * 500 + prox * 50

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
        Calculate cost penalty for components that should be near each other
        but are placed too far apart (from placement_hints proximity_groups).
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
            max_dist = group.get('max_distance', 10.0)
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
                    if dist > max_dist:
                        total += (dist - max_dist) * priority

        return total

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

    def _legalize(self):
        """Legalize placement: snap to grid and resolve overlaps"""
        # Snap to grid
        for comp in self.components.values():
            comp.x = round(comp.x / self.config.grid_size) * self.config.grid_size
            comp.y = round(comp.y / self.config.grid_size) * self.config.grid_size
            comp.x, comp.y = self._clamp_to_board(comp)

        # Resolve overlaps
        placed = set()
        for ref in sorted(self.components.keys()):
            self._resolve_overlap_spiral(ref, placed)
            placed.add(ref)

    def _save_state(self) -> Dict:
        """Save current component state"""
        return {ref: (c.x, c.y, c.rotation, c.width, c.height)
                for ref, c in self.components.items()}

    def _restore_state(self, state: Dict):
        """Restore component state"""
        for ref, (x, y, rot, w, h) in state.items():
            if ref in self.components:
                self.components[ref].x = x
                self.components[ref].y = y
                self.components[ref].rotation = rot
                self.components[ref].width = w
                self.components[ref].height = h

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
            overlap_area=self._calculate_overlap_area()
        )


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
