"""
PCB Engine - Placement Piston (Sub-Engine)
============================================

A dedicated piston (sub-engine) for PCB component placement.

ALGORITHMS IMPLEMENTED (From Research):
========================================
1.  Force-Directed (Fruchterman-Reingold) - Physics-based graph drawing
    Reference: "Graph Drawing by Force-directed Placement" (1991)

2.  Simulated Annealing (SA) - Probabilistic optimization
    Reference: "Optimization by Simulated Annealing" (Kirkpatrick, 1983)
    OpenROAD SA-PCB: https://github.com/The-OpenROAD-Project/SA-PCB

3.  Genetic Algorithm (GA) - Evolutionary optimization
    Reference: "Self Organizing Genetic Algorithm (SOGA)" - Springer

4.  Quadratic/Analytical - Wirelength minimization via Laplacian
    Reference: "FastPlace: Efficient Analytical Placement" (2005)

5.  Min-Cut Partitioning - Recursive bisection (Breuer's algorithm)
    Reference: "A Min-Cut Placement Algorithm" (Breuer, 1977)
    Refinement: Fiduccia-Mattheyses (FM) algorithm

6.  Particle Swarm Optimization (PSO) - Swarm intelligence
    Reference: "APSO approach to PCB component placement" - Springer

7.  FastPlace Multilevel - Hierarchical quadratic placement
    Reference: "FastPlace 3.0: A Fast Multilevel Quadratic Placement Algorithm"

8.  ePlace (Electrostatic) - Density via Poisson's equation + FFT
    Reference: "Analytic VLSI Placement using Electrostatic Analogy"

The Main Engine (PCBEngine) is responsible for:
1. Deciding which algorithm to run
2. Evaluating the output quality
3. Deciding whether to accept or try another algorithm
"""

import math
import random
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from .common_types import get_pins
except ImportError:
    from common_types import get_pins


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class PlacementAlgorithm(Enum):
    """Available placement algorithms - ALL from research"""
    FORCE_DIRECTED = 'fd'           # Fruchterman-Reingold 1991
    SIMULATED_ANNEALING = 'sa'      # Kirkpatrick 1983
    GENETIC = 'ga'                  # SOGA Springer
    QUADRATIC = 'quadratic'         # FastPlace 2005
    MINCUT = 'mincut'               # Breuer 1977
    PSO = 'pso'                     # APSO Springer
    FASTPLACE = 'fastplace'         # FastPlace 3.0 multilevel
    EPLACE = 'eplace'               # Electrostatic placement
    AUTO = 'auto'                   # Try multiple, pick best
    PARALLEL = 'parallel'           # Run ALL concurrently


@dataclass
class PlacementConfig:
    """Configuration for the placement piston"""
    # Board parameters
    board_width: float = 50.0
    board_height: float = 40.0
    origin_x: float = 0.0  # Board starts at origin (100.0 caused out-of-bounds placement)
    origin_y: float = 0.0  # Board starts at origin
    grid_size: float = 0.5

    # Algorithm selection
    algorithm: str = 'sa'  # Default to Simulated Annealing (proven)

    # General parameters
    max_iterations: int = 500
    convergence_threshold: float = 0.001

    # Force-directed parameters (Fruchterman-Reingold 1991)
    fd_iterations: int = 200
    fd_initial_temp: float = 10.0
    fd_cooling_factor: float = 0.95
    fd_attraction_k: float = 1.0
    fd_repulsion_k: float = 10000.0
    fd_min_temp: float = 0.01
    fd_gravity: float = 0.1

    # Simulated annealing parameters (Kirkpatrick 1983)
    # P = exp(-deltaC/T) acceptance probability
    sa_initial_temp: float = 200.0
    sa_final_temp: float = 0.01
    sa_moves_per_temp: int = 100
    sa_cooling_rate: float = 0.97  # Alpha in T(k+1) = alpha * T(k)
    sa_reheat_threshold: int = 50
    sa_reheat_factor: float = 1.5

    # Genetic algorithm parameters (SOGA)
    ga_population_size: int = 100
    ga_generations: int = 150
    ga_elite_ratio: float = 0.1
    ga_crossover_rate: float = 0.8
    ga_mutation_rate: float = 0.15
    ga_tournament_size: int = 5
    ga_diversity_threshold: float = 0.1

    # Quadratic/Analytical placement (FastPlace)
    # Solves Lx = b using Conjugate Gradient
    qp_cg_iterations: int = 50
    qp_spreading_iterations: int = 20
    qp_spreading_factor: float = 0.5
    qp_anchor_weight: float = 0.1

    # Min-cut partitioning (Breuer 1977)
    mc_min_partition_size: int = 4
    mc_balance_factor: float = 0.45  # Allow 45-55% split
    mc_fm_passes: int = 3  # Fiduccia-Mattheyses refinement passes

    # Particle Swarm Optimization (APSO)
    pso_swarm_size: int = 50
    pso_iterations: int = 100
    pso_w: float = 0.729  # Inertia weight (Clerc constriction)
    pso_c1: float = 1.49445  # Cognitive (personal best)
    pso_c2: float = 1.49445  # Social (global best)
    pso_v_max: float = 5.0  # Maximum velocity

    # FastPlace Multilevel parameters
    fp_coarsening_levels: int = 3
    fp_cluster_size: int = 4
    fp_uncoarsening_iterations: int = 10

    # ePlace (Electrostatic) parameters
    ep_bin_size: float = 2.0
    ep_lambda: float = 0.01  # Penalty for density
    ep_nesterov_alpha: float = 0.01
    ep_iterations: int = 100

    # Component spacing and margins
    # min_spacing must account for:
    # - SMD pad extension beyond body (up to 0.5mm per side for 0805)
    # - Solder mask expansion (0.1mm per side)
    # - Manufacturing tolerances
    # 2.0mm ensures pads from adjacent components don't overlap solder masks
    min_spacing: float = 2.0
    edge_margin: float = 2.0
    routing_channel_width: float = 1.0

    # Random seed
    seed: Optional[int] = 42

    # Parallel execution
    parallel_workers: int = 8
    parallel_timeout: float = 120.0

    # Net weights
    power_net_weight: float = 2.0
    critical_net_weight: float = 3.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ComponentState:
    """State of a component during placement"""
    ref: str
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    fixed: bool = False
    layer: str = 'F.Cu'

    # Physics state (for force-directed)
    vx: float = 0.0
    vy: float = 0.0
    fx: float = 0.0
    fy: float = 0.0

    # PSO state
    px: float = 0.0  # Personal best x
    py: float = 0.0  # Personal best y
    p_cost: float = float('inf')  # Personal best cost

    # Component properties
    pin_count: int = 0
    is_decoupling_cap: bool = False
    is_power_component: bool = False
    category: str = ''
    cluster_id: int = -1

    def copy(self) -> 'ComponentState':
        """Create a deep copy"""
        return ComponentState(
            ref=self.ref, x=self.x, y=self.y,
            width=self.width, height=self.height,
            rotation=self.rotation, fixed=self.fixed, layer=self.layer,
            vx=self.vx, vy=self.vy, fx=self.fx, fy=self.fy,
            px=self.px, py=self.py, p_cost=self.p_cost,
            pin_count=self.pin_count,
            is_decoupling_cap=self.is_decoupling_cap,
            is_power_component=self.is_power_component,
            category=self.category, cluster_id=self.cluster_id
        )

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return bounding box (x1, y1, x2, y2)"""
        half_w = self.width / 2
        half_h = self.height / 2
        return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)


@dataclass
class PlacementResult:
    """Result from placement piston"""
    positions: Dict[str, Tuple[float, float]]
    rotations: Dict[str, float]
    layers: Dict[str, str]
    cost: float
    algorithm_used: str
    iterations: int
    converged: bool
    success: bool = True

    # Metrics
    wirelength: float = 0.0
    overlap_area: float = 0.0
    oob_penalty: float = 0.0
    routing_congestion: float = 0.0


# =============================================================================
# PLACEMENT PISTON
# =============================================================================

class PlacementPiston:
    """
    Placement Piston - Multi-Algorithm Component Placement

    This piston provides 8 research-backed placement algorithms that can be
    selected and controlled by the main PCB Engine.

    All algorithms are from peer-reviewed research:
    - Fruchterman-Reingold (1991)
    - Kirkpatrick Simulated Annealing (1983)
    - Genetic Algorithm / SOGA
    - FastPlace Quadratic (2005)
    - Breuer Min-Cut (1977)
    - APSO Particle Swarm
    - FastPlace 3.0 Multilevel
    - ePlace Electrostatic

    Usage:
        config = PlacementConfig(algorithm='sa')
        piston = PlacementPiston(config)
        result = piston.place(parts_db, graph)
    """

    def __init__(self, config: PlacementConfig = None):
        self.config = config or PlacementConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        # State
        self.components: Dict[str, ComponentState] = {}
        self.nets: Dict[str, Dict] = {}
        self.adjacency: Dict[str, Dict[str, int]] = {}
        self.hub: Optional[str] = None
        self.pin_offsets: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.net_weights: Dict[str, float] = {}

        # Derived values
        self._optimal_dist: float = 0.0
        self._board_area: float = 0.0

        # ePlace density grid
        self._density_grid: List[List[float]] = []
        self._potential_grid: List[List[float]] = []

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def place(self, parts_db: Dict, graph: Dict = None) -> PlacementResult:
        """
        Run placement with configured algorithm.

        Args:
            parts_db: Parts database with component info
            graph: Connectivity graph (optional, will build if not provided)

        Returns:
            PlacementResult with positions and metrics
        """
        # Initialize state
        self._init_from_parts(parts_db, graph or {})

        # Compute optimal distance
        n = len(self.components)
        if n > 0:
            self._board_area = self.config.board_width * self.config.board_height
            self._optimal_dist = math.sqrt(self._board_area / n) * 0.8

        # Select and run algorithm
        algo = self.config.algorithm.lower()

        algorithm_map = {
            'fd': self._place_force_directed,
            'force_directed': self._place_force_directed,
            'sa': self._place_simulated_annealing,
            'simulated_annealing': self._place_simulated_annealing,
            'ga': self._place_genetic,
            'genetic': self._place_genetic,
            'quadratic': self._place_quadratic,
            'analytical': self._place_quadratic,
            'mincut': self._place_mincut,
            'partition': self._place_mincut,
            'pso': self._place_pso,
            'particle_swarm': self._place_pso,
            'fastplace': self._place_fastplace,
            'multilevel': self._place_fastplace,
            'eplace': self._place_eplace,
            'electrostatic': self._place_eplace,
            'auto': self._place_auto,
            'parallel': self._place_parallel,
        }

        if algo in algorithm_map:
            return algorithm_map[algo]()
        else:
            print(f"  [PLACEMENT] Unknown algorithm '{algo}', using simulated annealing")
            return self._place_simulated_annealing()

    def _init_from_parts(self, parts_db: Dict, graph: Dict):
        """Initialize placement state from parts database"""
        parts = parts_db.get('parts', {})
        raw_nets = parts_db.get('nets', {})
        self.adjacency = graph.get('adjacency', {})

        # Process nets
        self._process_nets(raw_nets)

        # Build adjacency if not provided
        if not self.adjacency:
            self._build_adjacency_from_nets()

        # Find hub (most connected component)
        self.hub = self._find_hub()

        # Create component states
        cx = self.config.origin_x + self.config.board_width / 2
        cy = self.config.origin_y + self.config.board_height / 2

        for idx, (ref, part) in enumerate(sorted(parts.items())):
            size = part.get('size', part.get('body', {}).get('size', (2.0, 2.0)))
            if isinstance(size, (list, tuple)):
                width, height = size[0], size[1]
            else:
                width = height = float(size) if size else 2.0

            body = part.get('body', {})
            if 'width' in body:
                width = body['width']
            if 'height' in body:
                height = body['height']

            # Initial position using golden angle spiral
            if ref == self.hub:
                x, y = cx, cy
            else:
                golden_angle = 137.508 * math.pi / 180
                angle = idx * golden_angle
                r = self.config.board_width * 0.3 * math.sqrt(idx + 1) / math.sqrt(len(parts) + 1)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)

            # Use get_pins helper for consistent pin access
            used_pins = get_pins(part)
            pin_count = len(used_pins)

            # Calculate actual footprint extent from pins (pad offset + pad size)
            # This is more accurate than body size for placement overlap detection
            max_x = max_y = 0
            for pin in used_pins:
                offset = pin.get('offset', (0, 0))
                pad_size = pin.get('size', (0.5, 0.5))
                if isinstance(offset, (list, tuple)):
                    ox, oy = abs(offset[0]), abs(offset[1])
                else:
                    ox, oy = 0, 0
                if isinstance(pad_size, (list, tuple)):
                    pw, ph = pad_size[0] / 2, pad_size[1] / 2
                else:
                    pw = ph = 0.25
                max_x = max(max_x, ox + pw)
                max_y = max(max_y, oy + ph)

            # Use larger of body or pad extent
            actual_width = max(width, max_x * 2)
            actual_height = max(height, max_y * 2)

            is_decap = self._is_decoupling_cap(ref, used_pins)
            is_power = any(self._is_power_net(pin.get('net', '')) for pin in used_pins)

            self.components[ref] = ComponentState(
                ref=ref, x=x, y=y,
                width=max(actual_width, 0.5), height=max(actual_height, 0.5),
                pin_count=pin_count,
                is_decoupling_cap=is_decap,
                is_power_component=is_power,
                category=part.get('category', self._categorize_component(ref))
            )

            # Store pin offsets
            self.pin_offsets[ref] = {}
            for pin in used_pins:
                pin_num = str(pin.get('number', ''))
                offset = pin.get('offset', None)
                if not offset:
                    physical = pin.get('physical', {})
                    offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))
                if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                    self.pin_offsets[ref][pin_num] = (offset[0], offset[1])

    def _process_nets(self, raw_nets: Dict):
        """Process nets and assign weights"""
        self.nets = {}
        self.net_weights = {}

        for net_name, net_info in raw_nets.items():
            weight = 1.0
            if self._is_power_net(net_name):
                weight = self.config.power_net_weight
            elif self._is_critical_net(net_name):
                weight = self.config.critical_net_weight

            self.nets[net_name] = {
                'pins': net_info.get('pins', []),
                'weight': weight
            }
            self.net_weights[net_name] = weight

    def _build_adjacency_from_nets(self):
        """Build adjacency matrix from nets"""
        self.adjacency = {}

        for net_name, net_info in self.nets.items():
            pins = net_info.get('pins', [])
            # Pins can be strings like 'R1.1' or tuples like ('R1', '1')
            components_in_net = set()
            for pin_ref in pins:
                if isinstance(pin_ref, str):
                    # Parse 'R1.1' -> 'R1'
                    comp = pin_ref.split('.')[0] if '.' in pin_ref else pin_ref
                elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 1:
                    comp = str(pin_ref[0])
                else:
                    continue
                components_in_net.add(comp)

            for comp_a in components_in_net:
                if comp_a not in self.adjacency:
                    self.adjacency[comp_a] = {}
                for comp_b in components_in_net:
                    if comp_a != comp_b:
                        self.adjacency[comp_a][comp_b] = self.adjacency[comp_a].get(comp_b, 0) + 1

    def _find_hub(self) -> Optional[str]:
        """Find the most connected component (hub)"""
        max_connections = 0
        hub = None

        for ref, neighbors in self.adjacency.items():
            total = sum(neighbors.values())
            if total > max_connections:
                max_connections = total
                hub = ref

        return hub

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def _is_power_net(self, net_name: str) -> bool:
        power_names = ['GND', 'VCC', 'VDD', '3V3', '5V', 'VBUS', '12V', 'AVCC', 'AGND']
        return net_name.upper() in power_names or net_name.upper().startswith('V')

    def _is_critical_net(self, net_name: str) -> bool:
        keywords = ['CLK', 'CLOCK', 'RST', 'RESET', 'CS', 'EN', 'ENABLE']
        return any(kw in net_name.upper() for kw in keywords)

    def _is_decoupling_cap(self, ref: str, pins: List) -> bool:
        if not ref.startswith('C'):
            return False
        return any(self._is_power_net(pin.get('net', '')) for pin in pins)

    def _categorize_component(self, ref: str) -> str:
        prefix = ref.rstrip('0123456789').upper()
        categories = {
            'R': 'resistor', 'C': 'capacitor', 'L': 'inductor',
            'D': 'diode', 'Q': 'transistor', 'U': 'ic',
            'J': 'connector', 'P': 'connector', 'SW': 'switch',
            'LED': 'led', 'Y': 'crystal', 'F': 'fuse'
        }
        return categories.get(prefix, 'other')

    def _clamp_to_board(self, comp: ComponentState) -> Tuple[float, float]:
        """Clamp component to board boundaries"""
        margin = self.config.edge_margin
        half_w = comp.width / 2
        half_h = comp.height / 2

        x = max(self.config.origin_x + margin + half_w,
                min(self.config.origin_x + self.config.board_width - margin - half_w, comp.x))
        y = max(self.config.origin_y + margin + half_h,
                min(self.config.origin_y + self.config.board_height - margin - half_h, comp.y))

        return x, y

    def _snap_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinates to grid"""
        gs = self.config.grid_size
        return (round(x / gs) * gs, round(y / gs) * gs)

    # =========================================================================
    # COST CALCULATION
    # =========================================================================

    def _calculate_cost(self) -> float:
        """Calculate total placement cost"""
        wl = self._calculate_wirelength()
        overlap = self._calculate_overlap_area()
        oob = self._calculate_oob_penalty()

        return wl + overlap * 1000 + oob * 500

    def _calculate_wirelength(self) -> float:
        """Calculate HPWL (Half Perimeter Wire Length)"""
        total = 0.0

        for net_name, net_info in self.nets.items():
            pins = net_info.get('pins', [])
            weight = net_info.get('weight', 1.0)

            if len(pins) < 2:
                continue

            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')

            for pin_ref in pins:
                # Parse pin reference: 'R1.1' -> ('R1', '1') or tuple
                if isinstance(pin_ref, str):
                    parts = pin_ref.split('.')
                    comp_ref = parts[0] if parts else ''
                    pin_num = parts[1] if len(parts) > 1 else ''
                elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
                    comp_ref, pin_num = str(pin_ref[0]), str(pin_ref[1])
                else:
                    continue

                if comp_ref not in self.components:
                    continue
                comp = self.components[comp_ref]

                offset = self.pin_offsets.get(comp_ref, {}).get(str(pin_num), (0, 0))
                px = comp.x + offset[0]
                py = comp.y + offset[1]

                min_x, max_x = min(min_x, px), max(max_x, px)
                min_y, max_y = min(min_y, py), max(max_y, py)

            if min_x != float('inf'):
                total += ((max_x - min_x) + (max_y - min_y)) * weight

        return total

    def _calculate_overlap_area(self) -> float:
        """Calculate total overlap between components"""
        total = 0.0
        refs = list(self.components.keys())

        for i, ref_a in enumerate(refs):
            for ref_b in refs[i + 1:]:
                ca, cb = self.components[ref_a], self.components[ref_b]

                half_w_a = ca.width / 2 + self.config.min_spacing / 2
                half_h_a = ca.height / 2 + self.config.min_spacing / 2
                half_w_b = cb.width / 2 + self.config.min_spacing / 2
                half_h_b = cb.height / 2 + self.config.min_spacing / 2

                overlap_x = max(0, (half_w_a + half_w_b) - abs(ca.x - cb.x))
                overlap_y = max(0, (half_h_a + half_h_b) - abs(ca.y - cb.y))

                total += overlap_x * overlap_y

        return total

    def _calculate_oob_penalty(self) -> float:
        """Calculate out-of-bounds penalty"""
        total = 0.0
        margin = self.config.edge_margin

        for comp in self.components.values():
            half_w, half_h = comp.width / 2, comp.height / 2

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

    def _check_has_overlaps(self) -> bool:
        return self._calculate_overlap_area() > 0.01

    def _check_out_of_bounds(self) -> bool:
        return self._calculate_oob_penalty() > 0.01

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _save_state(self) -> Dict:
        """Save current component state"""
        return {ref: (c.x, c.y, c.rotation, c.width, c.height)
                for ref, c in self.components.items()}

    def _restore_state(self, state: Dict):
        """Restore component state"""
        for ref, (x, y, rot, w, h) in state.items():
            if ref in self.components:
                c = self.components[ref]
                c.x, c.y, c.rotation = x, y, rot
                c.width, c.height = w, h

    def _create_result(self, algorithm: str, iterations: int, converged: bool) -> PlacementResult:
        """Create placement result from current state"""
        return PlacementResult(
            positions={ref: (c.x, c.y) for ref, c in self.components.items()},
            rotations={ref: c.rotation for ref, c in self.components.items()},
            layers={ref: c.layer for ref, c in self.components.items()},
            cost=self._calculate_cost(),
            algorithm_used=algorithm,
            iterations=iterations,
            converged=converged,
            success=not self._check_has_overlaps() and not self._check_out_of_bounds(),
            wirelength=self._calculate_wirelength(),
            overlap_area=self._calculate_overlap_area(),
            oob_penalty=self._calculate_oob_penalty()
        )

    # =========================================================================
    # LEGALIZATION
    # =========================================================================

    def _legalize(self):
        """Legalize placement: snap to grid and resolve overlaps"""
        # Snap to grid
        for comp in self.components.values():
            comp.x, comp.y = self._snap_to_grid(comp.x, comp.y)
            comp.x, comp.y = self._clamp_to_board(comp)

        # Resolve overlaps
        placed = set()
        for ref in sorted(self.components.keys()):
            self._resolve_overlap_spiral(ref, placed)
            placed.add(ref)

    def _resolve_overlap_spiral(self, ref: str, placed: Set[str], max_attempts: int = 100):
        """Resolve overlap using spiral search"""
        comp = self.components[ref]
        original_x, original_y = comp.x, comp.y

        for attempt in range(max_attempts):
            if not self._has_overlap(ref, placed):
                return

            angle = (attempt * 30) % 360
            radius = self.config.grid_size * (1 + attempt // 12)

            comp.x = original_x + radius * math.cos(math.radians(angle))
            comp.y = original_y + radius * math.sin(math.radians(angle))
            comp.x, comp.y = self._clamp_to_board(comp)

    def _has_overlap(self, ref: str, placed: Set[str]) -> bool:
        """Check if component overlaps with placed components"""
        comp = self.components[ref]
        half_w = comp.width / 2 + self.config.min_spacing
        half_h = comp.height / 2 + self.config.min_spacing

        for other_ref in placed:
            other = self.components[other_ref]
            other_hw = other.width / 2 + self.config.min_spacing
            other_hh = other.height / 2 + self.config.min_spacing

            if (abs(comp.x - other.x) < half_w + other_hw and
                    abs(comp.y - other.y) < half_h + other_hh):
                return True
        return False

    # =========================================================================
    # ALGORITHM 1: FORCE-DIRECTED (Fruchterman-Reingold 1991)
    # =========================================================================

    def _place_force_directed(self) -> PlacementResult:
        """
        Force-directed placement using Fruchterman-Reingold algorithm.

        Reference: "Graph Drawing by Force-directed Placement" (1991)

        Physics simulation where:
        - Connected components attract (spring/Hooke's law)
        - All components repel (electrostatic/Coulomb's law)
        - Temperature controls movement and decreases over time

        Force equations:
            f_attract = d^2 / k
            f_repel = k^2 / d
        where k = sqrt(area / n)
        """
        print("  [FD] Running Fruchterman-Reingold force-directed placement...")

        n = len(self.components)
        if n == 0:
            return self._create_result('force_directed', 0, True)

        # Optimal distance (k in paper)
        k = self._optimal_dist if self._optimal_dist > 0 else 5.0

        # Temperature starts at k * sqrt(n)
        temp = self.config.fd_initial_temp * math.sqrt(n)

        iteration = 0
        converged = False

        for iteration in range(self.config.fd_iterations):
            # Reset forces
            for comp in self.components.values():
                comp.fx = comp.fy = 0.0

            # Repulsive forces between ALL pairs (Coulomb's law)
            refs = list(self.components.keys())
            for i, ref_a in enumerate(refs):
                for ref_b in refs[i + 1:]:
                    self._fd_apply_repulsive(ref_a, ref_b, k)

            # Attractive forces from nets (Hooke's law)
            for net_name, net_info in self.nets.items():
                pins = net_info.get('pins', [])
                weight = net_info.get('weight', 1.0)
                # Parse pin refs: 'R1.1' -> 'R1'
                comps_in_net = sorted(set(
                    (p.split('.')[0] if isinstance(p, str) else str(p[0]))
                    for p in pins if p
                ))

                for i, comp_a in enumerate(comps_in_net):
                    for comp_b in comps_in_net[i + 1:]:
                        if comp_a in self.components and comp_b in self.components:
                            self._fd_apply_attractive(comp_a, comp_b, k, weight)

            # Gravity towards center (additional stabilization)
            cx = self.config.origin_x + self.config.board_width / 2
            cy = self.config.origin_y + self.config.board_height / 2
            for comp in self.components.values():
                if not comp.fixed:
                    comp.fx += self.config.fd_gravity * (cx - comp.x)
                    comp.fy += self.config.fd_gravity * (cy - comp.y)

            # Boundary forces
            for comp in self.components.values():
                self._fd_apply_boundary_force(comp)

            # Update positions with temperature limiting displacement
            max_displacement = 0.0
            for comp in self.components.values():
                if comp.fixed:
                    continue

                force_mag = math.sqrt(comp.fx ** 2 + comp.fy ** 2)
                if force_mag > 0:
                    # Limit displacement by temperature
                    displacement = min(force_mag, temp)
                    dx = (comp.fx / force_mag) * displacement
                    dy = (comp.fy / force_mag) * displacement

                    comp.x += dx
                    comp.y += dy
                    max_displacement = max(max_displacement, abs(dx), abs(dy))

            # Clamp to board
            for comp in self.components.values():
                comp.x, comp.y = self._clamp_to_board(comp)

            # Cooling schedule: T(k+1) = T(k) * cooling_factor
            temp *= self.config.fd_cooling_factor

            # Check convergence
            if temp < self.config.fd_min_temp or max_displacement < self.config.convergence_threshold:
                converged = True
                break

        self._legalize()
        cost = self._calculate_cost()
        print(f"  [FD] Converged: {converged}, iterations: {iteration + 1}, cost: {cost:.2f}")

        return self._create_result('force_directed', iteration + 1, converged)

    def _fd_apply_repulsive(self, ref_a: str, ref_b: str, k: float):
        """Coulomb's law repulsion: f_repel = k^2 / d"""
        ca, cb = self.components[ref_a], self.components[ref_b]

        dx = ca.x - cb.x
        dy = ca.y - cb.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.01:
            dist = 0.01
            dx = random.uniform(-0.01, 0.01)
            dy = random.uniform(-0.01, 0.01)

        # Account for component sizes
        min_dist = (ca.width + cb.width) / 2 + self.config.min_spacing
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

    def _fd_apply_attractive(self, ref_a: str, ref_b: str, k: float, weight: float = 1.0):
        """Hooke's law attraction: f_attract = d^2 / k"""
        ca, cb = self.components[ref_a], self.components[ref_b]

        dx = cb.x - ca.x
        dy = cb.y - ca.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.01:
            return

        force = (dist ** 2) / k * weight * self.config.fd_attraction_k
        fx = force * dx / dist
        fy = force * dy / dist

        if not ca.fixed:
            ca.fx += fx
            ca.fy += fy
        if not cb.fixed:
            cb.fx -= fx
            cb.fy -= fy

    def _fd_apply_boundary_force(self, comp: ComponentState):
        """Apply force to keep within boundaries"""
        margin = self.config.edge_margin + max(comp.width, comp.height) / 2

        min_x = self.config.origin_x + margin
        max_x = self.config.origin_x + self.config.board_width - margin
        min_y = self.config.origin_y + margin
        max_y = self.config.origin_y + self.config.board_height - margin

        boundary_force = 100.0

        if comp.x < min_x:
            comp.fx += boundary_force * (min_x - comp.x)
        if comp.x > max_x:
            comp.fx -= boundary_force * (comp.x - max_x)
        if comp.y < min_y:
            comp.fy += boundary_force * (min_y - comp.y)
        if comp.y > max_y:
            comp.fy -= boundary_force * (comp.y - max_y)

    # =========================================================================
    # ALGORITHM 2: SIMULATED ANNEALING (Kirkpatrick 1983)
    # =========================================================================

    def _place_simulated_annealing(self) -> PlacementResult:
        """
        Simulated Annealing placement.

        Reference: "Optimization by Simulated Annealing" (Kirkpatrick, 1983)
        Also: OpenROAD SA-PCB project

        Metropolis criterion:
            P(accept) = 1             if deltaC < 0
            P(accept) = exp(-dC/T)    if deltaC >= 0

        Cooling schedule: T(k+1) = alpha * T(k)

        Move operators:
        - Shift: move component randomly
        - Swap: exchange two components
        - Rotate: rotate 90 degrees
        - Flip: rotate 180 degrees
        """
        print("  [SA] Running simulated annealing placement (Kirkpatrick 1983)...")

        current_cost = self._calculate_cost()
        best_cost = current_cost
        best_state = self._save_state()

        temp = self.config.sa_initial_temp
        rejections = 0
        total_iterations = 0
        accepted = 0

        while temp > self.config.sa_final_temp:
            for _ in range(self.config.sa_moves_per_temp):
                total_iterations += 1

                old_state = self._save_state()

                # Select move operator randomly (as in SA-PCB)
                r = random.random()
                if r < 0.5:
                    self._sa_move_shift(temp)
                elif r < 0.75:
                    self._sa_move_swap()
                elif r < 0.90:
                    self._sa_move_rotate()
                else:
                    self._sa_move_flip()

                new_cost = self._calculate_cost()
                delta = new_cost - current_cost

                # Metropolis criterion: P = exp(-delta/T)
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_cost = new_cost
                    accepted += 1
                    rejections = 0

                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_state = self._save_state()
                else:
                    self._restore_state(old_state)
                    rejections += 1

                    # Reheat if stuck (adaptive SA)
                    if rejections > self.config.sa_reheat_threshold:
                        temp *= self.config.sa_reheat_factor
                        rejections = 0

            # Cooling: T(k+1) = alpha * T(k)
            temp *= self.config.sa_cooling_rate

        self._restore_state(best_state)
        self._legalize()

        acceptance_rate = accepted / total_iterations if total_iterations > 0 else 0
        print(f"  [SA] Final cost: {best_cost:.2f}, acceptance: {acceptance_rate:.1%}")

        return self._create_result('simulated_annealing', total_iterations, True)

    def _sa_move_shift(self, temp: float):
        """Shift random component - distance proportional to temperature"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if not refs:
            return

        ref = random.choice(refs)
        comp = self.components[ref]

        # Move distance decreases with temperature
        max_shift = (temp / self.config.sa_initial_temp) * self.config.board_width * 0.3
        max_shift = max(max_shift, self.config.grid_size)

        comp.x += random.uniform(-max_shift, max_shift)
        comp.y += random.uniform(-max_shift, max_shift)
        comp.x, comp.y = self._clamp_to_board(comp)

    def _sa_move_swap(self):
        """Swap two random components"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if len(refs) < 2:
            return

        ref_a, ref_b = random.sample(refs, 2)
        ca, cb = self.components[ref_a], self.components[ref_b]
        ca.x, cb.x = cb.x, ca.x
        ca.y, cb.y = cb.y, ca.y

    def _sa_move_rotate(self):
        """Rotate random component 90 degrees"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if not refs:
            return

        ref = random.choice(refs)
        comp = self.components[ref]
        comp.rotation = (comp.rotation + 90) % 360
        comp.width, comp.height = comp.height, comp.width
        comp.x, comp.y = self._clamp_to_board(comp)

    def _sa_move_flip(self):
        """Flip component 180 degrees"""
        refs = [r for r, c in self.components.items() if not c.fixed]
        if not refs:
            return

        ref = random.choice(refs)
        self.components[ref].rotation = (self.components[ref].rotation + 180) % 360

    # =========================================================================
    # ALGORITHM 3: GENETIC ALGORITHM (SOGA - Self Organizing Genetic Algorithm)
    # =========================================================================

    def _place_genetic(self) -> PlacementResult:
        """
        Genetic Algorithm placement.

        Reference: "Self Organizing Genetic Algorithm (SOGA)" - Springer
        "Optimization of electronics component placement design on PCB"

        GA operators:
        - Selection: Tournament selection
        - Crossover: Blend crossover (BLX-alpha)
        - Mutation: Gaussian perturbation
        - Elitism: Keep best individuals

        Chromosome: {ref: (x, y, rotation)}
        Fitness: minimize placement cost
        """
        print("  [GA] Running genetic algorithm placement (SOGA)...")

        n = len(self.components)
        if n == 0:
            return self._create_result('genetic', 0, True)

        refs = sorted(self.components.keys())
        n_elite = max(1, int(self.config.ga_population_size * self.config.ga_elite_ratio))

        # Initialize population with diverse strategies
        population = self._ga_init_population(refs)

        best_individual = None
        best_fitness = float('inf')

        for gen in range(self.config.ga_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                self._ga_apply_individual(individual, refs)
                fitness = self._calculate_cost()
                fitness_scores.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)

            # Calculate diversity for adaptive mutation
            diversity = self._ga_calculate_diversity(population)

            # Adaptive mutation rate (SOGA feature)
            mutation_rate = self.config.ga_mutation_rate
            if diversity < self.config.ga_diversity_threshold:
                mutation_rate *= 2  # Increase diversity

            # Sort by fitness (lower is better)
            sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i])

            # Create new population
            new_population = []

            # Elitism: keep best individuals unchanged
            for i in range(n_elite):
                new_population.append(copy.deepcopy(population[sorted_indices[i]]))

            # Fill rest with offspring
            while len(new_population) < self.config.ga_population_size:
                # Tournament selection
                parent1 = self._ga_tournament_select(population, fitness_scores)
                parent2 = self._ga_tournament_select(population, fitness_scores)

                # Crossover (BLX-alpha)
                if random.random() < self.config.ga_crossover_rate:
                    child = self._ga_crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                # Mutation (Gaussian)
                if random.random() < mutation_rate:
                    self._ga_mutate(child)

                new_population.append(child)

            population = new_population

            if (gen + 1) % 25 == 0:
                print(f"  [GA] Gen {gen + 1}: best={best_fitness:.2f}, diversity={diversity:.3f}")

        # Apply best solution
        self._ga_apply_individual(best_individual, refs)
        self._legalize()

        print(f"  [GA] Final cost: {best_fitness:.2f}")
        return self._create_result('genetic', self.config.ga_generations, True)

    def _ga_init_population(self, refs: List[str]) -> List[Dict]:
        """Initialize diverse population using multiple strategies"""
        population = []

        for i in range(self.config.ga_population_size):
            individual = {}
            strategy = i % 4

            for ref in refs:
                comp = self.components[ref]
                margin = self.config.edge_margin + max(comp.width, comp.height) / 2

                if strategy == 0:  # Random uniform
                    x = self.config.origin_x + margin + random.random() * (self.config.board_width - 2 * margin)
                    y = self.config.origin_y + margin + random.random() * (self.config.board_height - 2 * margin)
                elif strategy == 1:  # Gaussian around center
                    cx = self.config.origin_x + self.config.board_width / 2
                    cy = self.config.origin_y + self.config.board_height / 2
                    x = cx + random.gauss(0, self.config.board_width / 4)
                    y = cy + random.gauss(0, self.config.board_height / 4)
                elif strategy == 2:  # Grid-based
                    n = len(refs)
                    cols = int(math.sqrt(n)) + 1
                    idx = refs.index(ref)
                    cell_w = (self.config.board_width - 2 * margin) / cols
                    cell_h = (self.config.board_height - 2 * margin) / max(1, (n + cols - 1) // cols)
                    x = self.config.origin_x + margin + (idx % cols + 0.5) * cell_w
                    y = self.config.origin_y + margin + (idx // cols + 0.5) * cell_h
                else:  # Golden angle spiral
                    idx = refs.index(ref)
                    angle = idx * 137.508 * math.pi / 180
                    r = self.config.board_width * 0.3 * math.sqrt(idx + 1) / math.sqrt(len(refs))
                    cx = self.config.origin_x + self.config.board_width / 2
                    cy = self.config.origin_y + self.config.board_height / 2
                    x = cx + r * math.cos(angle)
                    y = cy + r * math.sin(angle)

                x = max(self.config.origin_x + margin,
                        min(self.config.origin_x + self.config.board_width - margin, x))
                y = max(self.config.origin_y + margin,
                        min(self.config.origin_y + self.config.board_height - margin, y))

                individual[ref] = (x, y, 0)

            population.append(individual)

        return population

    def _ga_apply_individual(self, individual: Dict, refs: List[str]):
        """Apply individual's positions to components"""
        for ref, (x, y, rot) in individual.items():
            if ref in self.components:
                self.components[ref].x = x
                self.components[ref].y = y
                self.components[ref].rotation = rot

    def _ga_tournament_select(self, population: List, fitness: List) -> Dict:
        """Tournament selection"""
        indices = random.sample(range(len(population)), min(self.config.ga_tournament_size, len(population)))
        best_idx = min(indices, key=lambda i: fitness[i])
        return copy.deepcopy(population[best_idx])

    def _ga_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """BLX-alpha crossover"""
        child = {}
        refs = list(parent1.keys())
        split = random.randint(0, len(refs))

        for i, ref in enumerate(refs):
            if i < split:
                child[ref] = parent1[ref]
            else:
                x1, y1, r1 = parent1[ref]
                x2, y2, r2 = parent2[ref]
                # Blend crossover with alpha=0.5
                alpha = random.uniform(0.3, 0.7)
                child[ref] = (
                    alpha * x1 + (1 - alpha) * x2,
                    alpha * y1 + (1 - alpha) * y2,
                    r1 if random.random() < 0.5 else r2
                )

        return child

    def _ga_mutate(self, individual: Dict):
        """Gaussian mutation"""
        refs = list(individual.keys())
        n_mutations = max(1, len(refs) // 5)

        for _ in range(n_mutations):
            ref = random.choice(refs)
            if ref not in self.components:
                continue

            comp = self.components[ref]
            x, y, r = individual[ref]

            # Gaussian perturbation
            x += random.gauss(0, self.config.board_width * 0.1)
            y += random.gauss(0, self.config.board_height * 0.1)

            # Occasional rotation
            if random.random() < 0.1:
                r = (r + 90) % 360

            margin = self.config.edge_margin + max(comp.width, comp.height) / 2
            x = max(self.config.origin_x + margin,
                    min(self.config.origin_x + self.config.board_width - margin, x))
            y = max(self.config.origin_y + margin,
                    min(self.config.origin_y + self.config.board_height - margin, y))

            individual[ref] = (x, y, r)

    def _ga_calculate_diversity(self, population: List) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 1.0

        refs = list(population[0].keys())
        total_var = 0.0

        for ref in refs:
            x_vals = [ind[ref][0] for ind in population]
            y_vals = [ind[ref][1] for ind in population]

            if len(x_vals) > 1:
                x_mean = sum(x_vals) / len(x_vals)
                y_mean = sum(y_vals) / len(y_vals)
                x_var = sum((x - x_mean) ** 2 for x in x_vals) / len(x_vals)
                y_var = sum((y - y_mean) ** 2 for y in y_vals) / len(y_vals)
                total_var += x_var + y_var

        board_area = self.config.board_width * self.config.board_height
        return min(1.0, math.sqrt(total_var / (len(refs) * board_area + 0.001)))

    # =========================================================================
    # ALGORITHM 4: QUADRATIC/ANALYTICAL PLACEMENT (FastPlace 2005)
    # =========================================================================

    def _place_quadratic(self) -> PlacementResult:
        """
        Quadratic/Analytical placement.

        Reference: "FastPlace: Efficient Analytical Placement using Cell Shifting,
                   Iterative Local Refinement and a Hybrid Net Model"

        Minimizes: Sum_e(weight_e * sum_{(i,j) in e}((x_i - x_j)^2 + (y_i - y_j)^2))

        This leads to solving the linear system: Lx = b
        where L is the graph Laplacian matrix.

        Solved using Conjugate Gradient method.
        Followed by spreading to reduce overlap.
        """
        print("  [QP] Running quadratic/analytical placement (FastPlace)...")

        refs = sorted(self.components.keys())
        n = len(refs)
        if n == 0:
            return self._create_result('quadratic', 0, True)

        ref_to_idx = {ref: i for i, ref in enumerate(refs)}

        # Build Laplacian matrix and RHS vectors
        L = [[0.0] * n for _ in range(n)]
        bx = [0.0] * n
        by = [0.0] * n

        # Process nets using clique model
        for net_name, net_info in self.nets.items():
            pins = net_info.get('pins', [])
            weight = net_info.get('weight', 1.0)
            # Parse pin refs: 'R1.1' -> 'R1'
            comps_in_net = sorted(set(
                (p.split('.')[0] if isinstance(p, str) else str(p[0]))
                for p in pins if p and (p.split('.')[0] if isinstance(p, str) else str(p[0])) in ref_to_idx
            ))

            if len(comps_in_net) < 2:
                continue

            # Clique model: weight per edge = weight / (k-1)
            net_weight = weight / (len(comps_in_net) - 1)

            for i, comp_a in enumerate(comps_in_net):
                for comp_b in comps_in_net[i + 1:]:
                    idx_a, idx_b = ref_to_idx[comp_a], ref_to_idx[comp_b]

                    # Laplacian: L[i,i] += w, L[j,j] += w, L[i,j] -= w, L[j,i] -= w
                    L[idx_a][idx_a] += net_weight
                    L[idx_b][idx_b] += net_weight
                    L[idx_a][idx_b] -= net_weight
                    L[idx_b][idx_a] -= net_weight

        # Add anchor constraints (fixed positions + center gravity)
        cx = self.config.origin_x + self.config.board_width / 2
        cy = self.config.origin_y + self.config.board_height / 2

        for ref, comp in self.components.items():
            idx = ref_to_idx[ref]

            if comp.fixed:
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
                # Weak anchor to center to prevent drifting
                anchor_weight = self.config.qp_anchor_weight
                L[idx][idx] += anchor_weight
                bx[idx] += anchor_weight * cx
                by[idx] += anchor_weight * cy

        # Initial positions
        x = [comp.x for ref, comp in sorted(self.components.items())]
        y = [comp.y for ref, comp in sorted(self.components.items())]

        # Solve Lx = bx and Ly = by using Conjugate Gradient
        x = self._qp_conjugate_gradient(L, bx, x, self.config.qp_cg_iterations)
        y = self._qp_conjugate_gradient(L, by, y, self.config.qp_cg_iterations)

        # Apply solution
        for i, ref in enumerate(refs):
            self.components[ref].x = x[i]
            self.components[ref].y = y[i]

        # Spreading (cell shifting) to reduce overlap
        self._qp_spreading()

        # Legalization
        self._legalize()

        cost = self._calculate_cost()
        print(f"  [QP] Final cost: {cost:.2f}")
        return self._create_result('quadratic', self.config.qp_cg_iterations, True)

    def _qp_conjugate_gradient(self, A: List[List[float]], b: List[float],
                                x0: List[float], max_iter: int) -> List[float]:
        """
        Conjugate Gradient solver for Ax = b

        Standard CG algorithm from numerical analysis.
        """
        n = len(b)
        x = x0.copy()

        # r = b - Ax
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        p = r.copy()
        rs_old = sum(r[i] ** 2 for i in range(n))

        for _ in range(max_iter):
            # Ap
            Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]
            pAp = sum(p[i] * Ap[i] for i in range(n))

            if abs(pAp) < 1e-10:
                break

            alpha = rs_old / pAp

            # x = x + alpha * p
            # r = r - alpha * Ap
            for i in range(n):
                x[i] += alpha * p[i]
                r[i] -= alpha * Ap[i]

            rs_new = sum(r[i] ** 2 for i in range(n))

            if math.sqrt(rs_new) < 1e-6:
                break

            beta = rs_new / (rs_old + 1e-10)

            # p = r + beta * p
            for i in range(n):
                p[i] = r[i] + beta * p[i]

            rs_old = rs_new

        return x

    def _qp_spreading(self):
        """
        Cell shifting/spreading to reduce overlap.

        Iteratively push overlapping cells apart.
        """
        for _ in range(self.config.qp_spreading_iterations):
            overlaps_found = False

            refs = list(self.components.keys())
            for i, ref_a in enumerate(refs):
                for ref_b in refs[i + 1:]:
                    ca, cb = self.components[ref_a], self.components[ref_b]

                    dx = cb.x - ca.x
                    dy = cb.y - ca.y
                    dist = math.sqrt(dx ** 2 + dy ** 2) + 0.001

                    min_sep = (ca.width + cb.width) / 2 + self.config.min_spacing

                    if dist < min_sep:
                        overlaps_found = True
                        push = (min_sep - dist) * self.config.qp_spreading_factor
                        push_x = push * dx / dist
                        push_y = push * dy / dist

                        if not ca.fixed:
                            ca.x -= push_x
                            ca.y -= push_y
                        if not cb.fixed:
                            cb.x += push_x
                            cb.y += push_y

            for comp in self.components.values():
                comp.x, comp.y = self._clamp_to_board(comp)

            if not overlaps_found:
                break

    # =========================================================================
    # ALGORITHM 5: MIN-CUT PARTITIONING (Breuer 1977)
    # =========================================================================

    def _place_mincut(self) -> PlacementResult:
        """
        Min-cut partitioning placement.

        Reference: "A Min-Cut Placement Algorithm" (Breuer, 1977)
        Refinement: Fiduccia-Mattheyses (FM) algorithm

        Algorithm:
        1. Recursively bisect the placement region
        2. Partition components to minimize nets crossing the cut
        3. Refine using FM algorithm
        """
        print("  [MC] Running min-cut partitioning placement (Breuer 1977)...")

        refs = list(self.components.keys())
        if len(refs) == 0:
            return self._create_result('mincut', 0, True)

        # Define initial region (full board minus margin)
        region = (
            self.config.origin_x + self.config.edge_margin,
            self.config.origin_y + self.config.edge_margin,
            self.config.origin_x + self.config.board_width - self.config.edge_margin,
            self.config.origin_y + self.config.board_height - self.config.edge_margin
        )

        # Recursive partition
        iterations = self._mc_partition_recursive(refs, region, 0)

        # FM refinement passes
        for _ in range(self.config.mc_fm_passes):
            self._mc_fm_pass()

        self._legalize()

        cost = self._calculate_cost()
        print(f"  [MC] Final cost: {cost:.2f}")
        return self._create_result('mincut', iterations, True)

    def _mc_partition_recursive(self, refs: List[str], region: Tuple, depth: int) -> int:
        """Recursively partition components using min-cut"""
        if len(refs) <= self.config.mc_min_partition_size:
            # Base case: place in grid within region
            self._mc_place_in_region(refs, region)
            return 1

        x1, y1, x2, y2 = region
        width = x2 - x1
        height = y2 - y1

        # Choose cut direction based on aspect ratio
        if width > height:
            # Vertical cut
            mid_x = (x1 + x2) / 2
            left_refs, right_refs = self._mc_min_cut_partition(refs, 'vertical')
            left_region = (x1, y1, mid_x, y2)
            right_region = (mid_x, y1, x2, y2)
        else:
            # Horizontal cut
            mid_y = (y1 + y2) / 2
            left_refs, right_refs = self._mc_min_cut_partition(refs, 'horizontal')
            left_region = (x1, y1, x2, mid_y)
            right_region = (x1, mid_y, x2, y2)

        # Recurse
        iter1 = self._mc_partition_recursive(left_refs, left_region, depth + 1)
        iter2 = self._mc_partition_recursive(right_refs, right_region, depth + 1)

        return iter1 + iter2

    def _mc_min_cut_partition(self, refs: List[str], direction: str) -> Tuple[List[str], List[str]]:
        """
        Partition to minimize cut edges.

        Uses position-based heuristic followed by gain-based swaps.
        """
        if direction == 'vertical':
            sorted_refs = sorted(refs, key=lambda r: self.components[r].x)
        else:
            sorted_refs = sorted(refs, key=lambda r: self.components[r].y)

        # Initial partition: split in half with balance constraint
        mid = len(sorted_refs) // 2
        balance = int(len(sorted_refs) * self.config.mc_balance_factor)
        mid = max(balance, min(len(sorted_refs) - balance, mid))

        left = sorted_refs[:mid]
        right = sorted_refs[mid:]

        # Improve with gain-based swaps
        left_set = set(left)
        right_set = set(right)

        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None

            for l_ref in left_set:
                for r_ref in right_set:
                    # Calculate gain from swapping
                    gain = self._mc_calculate_swap_gain(l_ref, r_ref, left_set, right_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (l_ref, r_ref)

            if best_swap:
                l_ref, r_ref = best_swap
                left_set.remove(l_ref)
                left_set.add(r_ref)
                right_set.remove(r_ref)
                right_set.add(l_ref)
                improved = True

        return list(left_set), list(right_set)

    def _mc_calculate_swap_gain(self, l_ref: str, r_ref: str,
                                 left_set: Set[str], right_set: Set[str]) -> float:
        """Calculate gain from swapping two components between partitions"""
        gain = 0.0

        # Nets crossing the cut before swap
        for net_name, net_info in self.nets.items():
            comps = set(c for c, p in net_info.get('pins', []) if c in self.components)

            in_left = len(comps & left_set)
            in_right = len(comps & right_set)

            # Current cut
            if in_left > 0 and in_right > 0:
                old_cut = 1
            else:
                old_cut = 0

            # After swap
            new_left = left_set - {l_ref} | {r_ref}
            new_right = right_set - {r_ref} | {l_ref}

            new_in_left = len(comps & new_left)
            new_in_right = len(comps & new_right)

            if new_in_left > 0 and new_in_right > 0:
                new_cut = 1
            else:
                new_cut = 0

            gain += old_cut - new_cut

        return gain

    def _mc_place_in_region(self, refs: List[str], region: Tuple):
        """Place components in grid within region"""
        x1, y1, x2, y2 = region
        width = x2 - x1
        height = y2 - y1

        n = len(refs)
        if n == 0:
            return

        cols = max(1, int(math.sqrt(n * width / height)))
        rows = max(1, (n + cols - 1) // cols)

        cell_w = width / cols
        cell_h = height / rows

        for i, ref in enumerate(refs):
            col = i % cols
            row = i // cols
            comp = self.components[ref]

            comp.x = x1 + (col + 0.5) * cell_w
            comp.y = y1 + (row + 0.5) * cell_h
            comp.x, comp.y = self._clamp_to_board(comp)

    def _mc_fm_pass(self):
        """
        Fiduccia-Mattheyses (FM) refinement pass.

        Reference: Fiduccia & Mattheyses, "A Linear-Time Heuristic for Improving
                   Network Partitions" (1982)

        Proper implementation with:
        1. Gain = FS(v) - TE(v) where:
           - FS(v) = edges from v to vertices in the "From Set" (same partition)
           - TE(v) = edges from v to vertices in the "To Set" (other partition)
        2. LIFO gain bucket structure for O(1) access to max-gain vertex
        3. Single pass with locked vertices (no vertex moves twice)
        4. Track best partition seen and revert if needed
        """
        refs = list(self.components.keys())
        n = len(refs)
        if n < 4:
            return

        # Partition based on position (left/right of center)
        cx = self.config.origin_x + self.config.board_width / 2
        part_a = set(r for r in refs if self.components[r].x < cx)
        part_b = set(r for r in refs if self.components[r].x >= cx)

        # Balance constraint: each partition must have at least balance_factor * n
        min_size = int(n * self.config.mc_balance_factor)

        # Compute initial gains for all vertices
        # Gain = FS(v) - TE(v) = edges to same partition - edges to other partition
        def compute_gain(v: str, v_part: Set[str], other_part: Set[str]) -> int:
            fs = 0  # Edges to same partition (From Set)
            te = 0  # Edges to other partition (To Set)
            for neighbor, weight in self.adjacency.get(v, {}).items():
                if neighbor in v_part:
                    fs += weight
                elif neighbor in other_part:
                    te += weight
            return te - fs  # Gain from moving v (TE becomes FS, FS becomes TE)

        gains = {}  # vertex -> gain
        for v in refs:
            if v in part_a:
                gains[v] = compute_gain(v, part_a, part_b)
            else:
                gains[v] = compute_gain(v, part_b, part_a)

        # Build LIFO gain buckets
        # Key = gain value, Value = list of vertices (LIFO order)
        max_degree = max(sum(self.adjacency.get(v, {}).values()) for v in refs) if refs else 1
        gain_buckets_a = {}  # Gain buckets for vertices in partition A
        gain_buckets_b = {}  # Gain buckets for vertices in partition B

        for v in part_a:
            g = gains[v]
            if g not in gain_buckets_a:
                gain_buckets_a[g] = []
            gain_buckets_a[g].append(v)

        for v in part_b:
            g = gains[v]
            if g not in gain_buckets_b:
                gain_buckets_b[g] = []
            gain_buckets_b[g].append(v)

        # Track max gains for O(1) lookup
        max_gain_a = max(gain_buckets_a.keys()) if gain_buckets_a else float('-inf')
        max_gain_b = max(gain_buckets_b.keys()) if gain_buckets_b else float('-inf')

        # Single FM pass
        locked = set()
        move_sequence = []  # (vertex, from_part, to_part, cumulative_gain)
        cumulative_gain = 0
        best_gain = 0
        best_prefix = 0

        for step in range(n):
            # Find best unlocked vertex to move (considering balance)
            best_v = None
            best_v_gain = float('-inf')
            from_bucket = None

            # Try moving from A to B (if A has room to shrink)
            if len(part_a) > min_size and gain_buckets_a:
                while max_gain_a in gain_buckets_a and not gain_buckets_a[max_gain_a]:
                    del gain_buckets_a[max_gain_a]
                    max_gain_a = max(gain_buckets_a.keys()) if gain_buckets_a else float('-inf')
                if gain_buckets_a and max_gain_a > best_v_gain:
                    # LIFO: take from end
                    for v in reversed(gain_buckets_a.get(max_gain_a, [])):
                        if v not in locked:
                            best_v = v
                            best_v_gain = max_gain_a
                            from_bucket = 'a'
                            break

            # Try moving from B to A (if B has room to shrink)
            if len(part_b) > min_size and gain_buckets_b:
                while max_gain_b in gain_buckets_b and not gain_buckets_b[max_gain_b]:
                    del gain_buckets_b[max_gain_b]
                    max_gain_b = max(gain_buckets_b.keys()) if gain_buckets_b else float('-inf')
                if gain_buckets_b and max_gain_b > best_v_gain:
                    for v in reversed(gain_buckets_b.get(max_gain_b, [])):
                        if v not in locked:
                            best_v = v
                            best_v_gain = max_gain_b
                            from_bucket = 'b'
                            break

            if best_v is None:
                break  # No valid move

            # Lock the vertex and remove from bucket
            locked.add(best_v)
            if from_bucket == 'a':
                gain_buckets_a[best_v_gain].remove(best_v)
            else:
                gain_buckets_b[best_v_gain].remove(best_v)

            # Move the vertex
            cumulative_gain += best_v_gain
            if from_bucket == 'a':
                part_a.remove(best_v)
                part_b.add(best_v)
                move_sequence.append((best_v, 'a', 'b', cumulative_gain))
            else:
                part_b.remove(best_v)
                part_a.add(best_v)
                move_sequence.append((best_v, 'b', 'a', cumulative_gain))

            # Track best prefix
            if cumulative_gain > best_gain:
                best_gain = cumulative_gain
                best_prefix = len(move_sequence)

            # Update gains of neighbors (they're affected by the move)
            for neighbor, weight in self.adjacency.get(best_v, {}).items():
                if neighbor in locked or neighbor not in self.components:
                    continue

                old_gain = gains[neighbor]

                # Recompute neighbor's gain
                if neighbor in part_a:
                    new_gain = compute_gain(neighbor, part_a, part_b)
                    old_bucket = gain_buckets_a
                else:
                    new_gain = compute_gain(neighbor, part_b, part_a)
                    old_bucket = gain_buckets_b

                # Update bucket
                if old_gain in old_bucket and neighbor in old_bucket[old_gain]:
                    old_bucket[old_gain].remove(neighbor)
                if new_gain not in old_bucket:
                    old_bucket[new_gain] = []
                old_bucket[new_gain].append(neighbor)
                gains[neighbor] = new_gain

            # Update max gains
            if gain_buckets_a:
                max_gain_a = max(gain_buckets_a.keys())
            if gain_buckets_b:
                max_gain_b = max(gain_buckets_b.keys())

        # Revert moves after best_prefix
        for i in range(len(move_sequence) - 1, best_prefix - 1, -1):
            v, from_p, to_p, _ = move_sequence[i]
            if to_p == 'a':
                part_a.remove(v)
                part_b.add(v)
            else:
                part_b.remove(v)
                part_a.add(v)

        # Apply final partition positions
        # Spread part_a to left half, part_b to right half
        margin = self.config.edge_margin
        left_region = (
            self.config.origin_x + margin,
            self.config.origin_y + margin,
            cx - margin / 2,
            self.config.origin_y + self.config.board_height - margin
        )
        right_region = (
            cx + margin / 2,
            self.config.origin_y + margin,
            self.config.origin_x + self.config.board_width - margin,
            self.config.origin_y + self.config.board_height - margin
        )

        self._mc_place_in_region(list(part_a), left_region)
        self._mc_place_in_region(list(part_b), right_region)

    def _mc_fm_pass_simple(self):
        """
        Simple FM fallback for very small partitions.
        """
        refs = list(self.components.keys())

        for _ in range(len(refs)):
            ref_a = random.choice(refs)
            neighbors = list(self.adjacency.get(ref_a, {}).keys())

            if not neighbors:
                continue

            ref_b = random.choice(neighbors)
            if ref_b not in self.components:
                continue

            # Try swap
            ca, cb = self.components[ref_a], self.components[ref_b]
            old_cost = self._calculate_cost()

            ca.x, cb.x = cb.x, ca.x
            ca.y, cb.y = cb.y, ca.y

            new_cost = self._calculate_cost()

            if new_cost > old_cost:
                # Revert
                ca.x, cb.x = cb.x, ca.x
                ca.y, cb.y = cb.y, ca.y

    # =========================================================================
    # ALGORITHM 6: PARTICLE SWARM OPTIMIZATION (APSO)
    # =========================================================================

    def _place_pso(self) -> PlacementResult:
        """
        Adaptive Particle Swarm Optimization (APSO) placement.

        Reference: "Adaptive PSO approach to PCB component placement" - Springer
        "An Improved Adaptive PSO Algorithm for Printed Circuit Board Component"

        Key APSO Features:
        1. Adaptive Inertia Weight: w decreases linearly from w_max to w_min
           w(t) = w_max - (w_max - w_min) * t / T_max
        2. Local Search: Apply local refinement to personal best solutions
        3. Mutation Operator: Re-initialize stagnant particles for diversity
        4. Adaptive Acceleration: c1 decreases, c2 increases over time

        Velocity: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
        Position: x = x + v
        """
        print("  [APSO] Running adaptive particle swarm optimization...")

        refs = sorted(self.components.keys())
        n_components = len(refs)

        if n_components == 0:
            return self._create_result('pso', 0, True)

        # APSO Parameters (from research)
        w_max = 0.9    # Initial inertia weight
        w_min = 0.4    # Final inertia weight
        c1_init = 2.5  # Initial cognitive coefficient (personal influence)
        c1_final = 0.5 # Final cognitive coefficient
        c2_init = 0.5  # Initial social coefficient (swarm influence)
        c2_final = 2.5 # Final social coefficient
        mutation_prob = 0.1   # Probability of mutation
        stagnation_limit = 10 # Iterations without improvement before mutation
        local_search_prob = 0.2  # Probability of local search on pbest

        # Initialize swarm
        particles = []  # List of {ref: (x, y, vx, vy)}
        pbest = []      # Personal best positions
        pbest_cost = [] # Personal best costs
        stagnation_count = []  # Iterations since last improvement
        gbest = None    # Global best
        gbest_cost = float('inf')

        margin = self.config.edge_margin

        # Initialize particles with diverse positions
        for p_idx in range(self.config.pso_swarm_size):
            particle = {}

            for ref in refs:
                comp = self.components[ref]
                hw = comp.width / 2
                hh = comp.height / 2

                # Use Latin Hypercube Sampling for better coverage
                # Divide space into swarm_size regions
                region_x = (self.config.board_width - 2 * margin - comp.width) / self.config.pso_swarm_size
                region_y = (self.config.board_height - 2 * margin - comp.height) / self.config.pso_swarm_size

                base_x = self.config.origin_x + margin + hw
                base_y = self.config.origin_y + margin + hh

                # Randomize within region for diversity
                x = base_x + (p_idx % int(math.sqrt(self.config.pso_swarm_size))) * region_x + random.random() * region_x
                y = base_y + (p_idx // int(math.sqrt(self.config.pso_swarm_size))) * region_y + random.random() * region_y

                # Clamp to valid range
                max_x = self.config.origin_x + self.config.board_width - margin - hw
                max_y = self.config.origin_y + self.config.board_height - margin - hh
                x = min(max_x, max(base_x, x))
                y = min(max_y, max(base_y, y))

                # Small initial velocity
                vx = random.uniform(-self.config.pso_v_max * 0.1, self.config.pso_v_max * 0.1)
                vy = random.uniform(-self.config.pso_v_max * 0.1, self.config.pso_v_max * 0.1)

                particle[ref] = (x, y, vx, vy)

            particles.append(particle)
            stagnation_count.append(0)

            # Evaluate initial cost
            self._pso_apply_particle(particle)
            cost = self._calculate_cost()

            pbest.append(copy.deepcopy(particle))
            pbest_cost.append(cost)

            if cost < gbest_cost:
                gbest_cost = cost
                gbest = copy.deepcopy(particle)

        prev_gbest_cost = gbest_cost

        # Main APSO loop
        for iteration in range(self.config.pso_iterations):
            # Adaptive parameters
            progress = iteration / self.config.pso_iterations

            # Linearly decreasing inertia weight (exploration -> exploitation)
            w = w_max - (w_max - w_min) * progress

            # Adaptive acceleration coefficients
            # c1 decreases (less personal influence over time)
            # c2 increases (more swarm influence over time)
            c1 = c1_init - (c1_init - c1_final) * progress
            c2 = c2_init + (c2_final - c2_init) * progress

            for i, particle in enumerate(particles):
                # Check for stagnation and apply mutation
                if stagnation_count[i] > stagnation_limit:
                    if random.random() < mutation_prob:
                        # Re-initialize particle with random position
                        particle = self._pso_reinitialize_particle(refs)
                        stagnation_count[i] = 0

                # Update velocities and positions
                new_particle = {}

                for ref in refs:
                    x, y, vx, vy = particle[ref]
                    pbx, pby, _, _ = pbest[i][ref]
                    gbx, gby, _, _ = gbest[ref]

                    # APSO velocity update with adaptive parameters
                    r1, r2 = random.random(), random.random()

                    vx_new = (w * vx +
                              c1 * r1 * (pbx - x) +
                              c2 * r2 * (gbx - x))
                    vy_new = (w * vy +
                              c1 * r1 * (pby - y) +
                              c2 * r2 * (gby - y))

                    # Adaptive velocity clamping (shrinks over time)
                    v_max = self.config.pso_v_max * (1.0 - 0.5 * progress)
                    vx_new = max(-v_max, min(v_max, vx_new))
                    vy_new = max(-v_max, min(v_max, vy_new))

                    # Position update
                    x_new = x + vx_new
                    y_new = y + vy_new

                    # Boundary handling with damped reflection
                    comp = self.components[ref]
                    comp_margin = margin + max(comp.width, comp.height) / 2

                    min_x = self.config.origin_x + comp_margin
                    max_x = self.config.origin_x + self.config.board_width - comp_margin
                    min_y = self.config.origin_y + comp_margin
                    max_y = self.config.origin_y + self.config.board_height - comp_margin

                    if x_new < min_x:
                        x_new = min_x + random.random() * (max_x - min_x) * 0.1
                        vx_new = abs(vx_new) * 0.3
                    elif x_new > max_x:
                        x_new = max_x - random.random() * (max_x - min_x) * 0.1
                        vx_new = -abs(vx_new) * 0.3

                    if y_new < min_y:
                        y_new = min_y + random.random() * (max_y - min_y) * 0.1
                        vy_new = abs(vy_new) * 0.3
                    elif y_new > max_y:
                        y_new = max_y - random.random() * (max_y - min_y) * 0.1
                        vy_new = -abs(vy_new) * 0.3

                    new_particle[ref] = (x_new, y_new, vx_new, vy_new)

                particles[i] = new_particle

                # Evaluate
                self._pso_apply_particle(new_particle)
                cost = self._calculate_cost()

                # Update personal best
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest[i] = copy.deepcopy(new_particle)
                    stagnation_count[i] = 0

                    # Apply local search to improve personal best
                    if random.random() < local_search_prob:
                        improved = self._pso_local_search(pbest[i], refs)
                        self._pso_apply_particle(improved)
                        improved_cost = self._calculate_cost()
                        if improved_cost < pbest_cost[i]:
                            pbest_cost[i] = improved_cost
                            pbest[i] = improved

                    # Update global best
                    if pbest_cost[i] < gbest_cost:
                        gbest_cost = pbest_cost[i]
                        gbest = copy.deepcopy(pbest[i])
                else:
                    stagnation_count[i] += 1

            # Check for global stagnation
            if abs(prev_gbest_cost - gbest_cost) < 0.01:
                # Apply stronger diversification
                for i in range(len(particles)):
                    if random.random() < 0.3:
                        particles[i] = self._pso_reinitialize_particle(refs)
                        stagnation_count[i] = 0

            prev_gbest_cost = gbest_cost

            if (iteration + 1) % 20 == 0:
                print(f"  [APSO] Iter {iteration + 1}: best={gbest_cost:.2f}, w={w:.3f}, c1={c1:.2f}, c2={c2:.2f}")

        # Apply global best
        self._pso_apply_particle(gbest)
        self._legalize()

        print(f"  [APSO] Final cost: {gbest_cost:.2f}")
        return self._create_result('pso', self.config.pso_iterations, True)

    def _pso_reinitialize_particle(self, refs: List[str]) -> Dict:
        """Re-initialize a particle with random positions for diversity"""
        particle = {}
        margin = self.config.edge_margin

        for ref in refs:
            comp = self.components[ref]
            hw = comp.width / 2
            hh = comp.height / 2

            x = self.config.origin_x + margin + hw + random.random() * (
                self.config.board_width - 2 * margin - comp.width)
            y = self.config.origin_y + margin + hh + random.random() * (
                self.config.board_height - 2 * margin - comp.height)

            vx = random.uniform(-self.config.pso_v_max, self.config.pso_v_max)
            vy = random.uniform(-self.config.pso_v_max, self.config.pso_v_max)

            particle[ref] = (x, y, vx, vy)

        return particle

    def _pso_local_search(self, particle: Dict, refs: List[str]) -> Dict:
        """
        Local search to refine personal best.

        Uses simple 2-opt style neighborhood: try swapping pairs of components.
        Reference: "Local Search Operators for PSO" - various papers
        """
        improved = copy.deepcopy(particle)
        self._pso_apply_particle(improved)
        best_cost = self._calculate_cost()

        # Try a limited number of swap moves
        max_attempts = min(10, len(refs))

        for _ in range(max_attempts):
            # Select two random components
            if len(refs) < 2:
                break

            ref_a, ref_b = random.sample(refs, 2)
            xa, ya, vxa, vya = improved[ref_a]
            xb, yb, vxb, vyb = improved[ref_b]

            # Swap positions
            improved[ref_a] = (xb, yb, vxa, vya)
            improved[ref_b] = (xa, ya, vxb, vyb)

            self._pso_apply_particle(improved)
            new_cost = self._calculate_cost()

            if new_cost < best_cost:
                best_cost = new_cost
                # Keep the swap
            else:
                # Revert
                improved[ref_a] = (xa, ya, vxa, vya)
                improved[ref_b] = (xb, yb, vxb, vyb)

        return improved

    def _pso_apply_particle(self, particle: Dict):
        """Apply particle position to components"""
        for ref, (x, y, vx, vy) in particle.items():
            if ref in self.components:
                self.components[ref].x = x
                self.components[ref].y = y

    # =========================================================================
    # ALGORITHM 7: FASTPLACE MULTILEVEL (FastPlace 3.0)
    # =========================================================================

    def _place_fastplace(self) -> PlacementResult:
        """
        FastPlace multilevel quadratic placement.

        Reference: "FastPlace 3.0: A Fast Multilevel Quadratic Placement Algorithm"

        Algorithm:
        1. Coarsening: cluster components hierarchically
        2. Initial placement: quadratic on coarsest level
        3. Uncoarsening: refine at each level

        Key innovation: two-level clustering + iterative local refinement
        """
        print("  [FP] Running FastPlace multilevel placement (FastPlace 3.0)...")

        refs = list(self.components.keys())
        if len(refs) == 0:
            return self._create_result('fastplace', 0, True)

        # Phase 1: Coarsening - create hierarchy of clusters
        levels = self._fp_coarsen()
        print(f"  [FP] Created {len(levels)} coarsening levels")

        # Phase 2: Initial placement at coarsest level
        self._fp_initial_placement(levels[-1])

        # Phase 3: Uncoarsening with refinement at each level
        total_iterations = 0
        for level_idx in range(len(levels) - 2, -1, -1):
            iterations = self._fp_uncoarsen_level(levels, level_idx)
            total_iterations += iterations

        self._legalize()

        cost = self._calculate_cost()
        print(f"  [FP] Final cost: {cost:.2f}")
        return self._create_result('fastplace', total_iterations, True)

    def _fp_coarsen(self) -> List[List[Set[str]]]:
        """
        Create coarsening hierarchy.

        Each level contains clusters of components from the previous level.
        Fine-grain clustering based on connectivity.
        """
        levels = []

        # Level 0: each component is its own cluster
        current_level = [{ref} for ref in self.components.keys()]
        levels.append(current_level)

        # Create coarser levels
        for _ in range(self.config.fp_coarsening_levels):
            if len(current_level) <= self.config.fp_cluster_size:
                break

            next_level = []
            merged = set()

            # Sort clusters by connectivity
            cluster_connectivity = []
            for i, cluster_a in enumerate(current_level):
                conn = 0
                for ref_a in cluster_a:
                    for ref_b, weight in self.adjacency.get(ref_a, {}).items():
                        if ref_b not in cluster_a:
                            conn += weight
                cluster_connectivity.append((i, conn))

            cluster_connectivity.sort(key=lambda x: -x[1])

            for idx_a, _ in cluster_connectivity:
                if idx_a in merged:
                    continue

                cluster_a = current_level[idx_a]
                merged.add(idx_a)

                # Find best neighbor cluster to merge with
                best_idx = None
                best_conn = 0

                for idx_b, cluster_b in enumerate(current_level):
                    if idx_b in merged or idx_b == idx_a:
                        continue

                    # Calculate connectivity between clusters
                    conn = 0
                    for ref_a in cluster_a:
                        for ref_b in cluster_b:
                            conn += self.adjacency.get(ref_a, {}).get(ref_b, 0)

                    if conn > best_conn and len(cluster_a) + len(cluster_b) <= self.config.fp_cluster_size * 2:
                        best_conn = conn
                        best_idx = idx_b

                if best_idx is not None:
                    # Merge clusters
                    merged.add(best_idx)
                    new_cluster = cluster_a | current_level[best_idx]
                    next_level.append(new_cluster)
                else:
                    # Keep cluster as is
                    next_level.append(cluster_a)

            if len(next_level) >= len(current_level):
                break

            levels.append(next_level)
            current_level = next_level

        return levels

    def _fp_initial_placement(self, clusters: List[Set[str]]):
        """Place clusters at coarsest level using quadratic"""
        # Calculate cluster centers of mass
        cluster_positions = []

        for cluster in clusters:
            sum_x = sum(self.components[ref].x for ref in cluster)
            sum_y = sum(self.components[ref].y for ref in cluster)
            n = len(cluster)
            cluster_positions.append((sum_x / n, sum_y / n))

        # Use quadratic placement on cluster centroids
        # (simplified: just use force-directed for clusters)
        cx = self.config.origin_x + self.config.board_width / 2
        cy = self.config.origin_y + self.config.board_height / 2

        for _ in range(50):
            forces = [(0.0, 0.0) for _ in clusters]

            # Repulsion between clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dx = cluster_positions[i][0] - cluster_positions[j][0]
                    dy = cluster_positions[i][1] - cluster_positions[j][1]
                    dist = math.sqrt(dx**2 + dy**2) + 0.1

                    force = 100.0 / dist
                    fx = force * dx / dist
                    fy = force * dy / dist

                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                    forces[j] = (forces[j][0] - fx, forces[j][1] - fy)

            # Attraction between connected clusters
            for net_name, net_info in self.nets.items():
                comps_in_net = set(c for c, p in net_info.get('pins', []) if c in self.components)

                clusters_in_net = set()
                for i, cluster in enumerate(clusters):
                    if cluster & comps_in_net:
                        clusters_in_net.add(i)

                clusters_list = list(clusters_in_net)
                for i in range(len(clusters_list)):
                    for j in range(i + 1, len(clusters_list)):
                        ci, cj = clusters_list[i], clusters_list[j]
                        dx = cluster_positions[cj][0] - cluster_positions[ci][0]
                        dy = cluster_positions[cj][1] - cluster_positions[ci][1]
                        dist = math.sqrt(dx**2 + dy**2) + 0.1

                        force = dist * 0.1
                        fx = force * dx / dist
                        fy = force * dy / dist

                        forces[ci] = (forces[ci][0] + fx, forces[ci][1] + fy)
                        forces[cj] = (forces[cj][0] - fx, forces[cj][1] - fy)

            # Update positions
            margin = self.config.edge_margin
            for i in range(len(clusters)):
                x = cluster_positions[i][0] + forces[i][0] * 0.1
                y = cluster_positions[i][1] + forces[i][1] * 0.1

                x = max(self.config.origin_x + margin,
                        min(self.config.origin_x + self.config.board_width - margin, x))
                y = max(self.config.origin_y + margin,
                        min(self.config.origin_y + self.config.board_height - margin, y))

                cluster_positions[i] = (x, y)

        # Apply cluster positions to components
        for i, cluster in enumerate(clusters):
            cx, cy = cluster_positions[i]
            for ref in cluster:
                self.components[ref].x = cx + random.uniform(-1, 1)
                self.components[ref].y = cy + random.uniform(-1, 1)

    def _fp_uncoarsen_level(self, levels: List[List[Set[str]]], level_idx: int) -> int:
        """
        Uncoarsen one level with Iterative Local Refinement (ILR).

        Reference: "FastPlace 3.0: A Fast Multilevel Quadratic Placement Algorithm"

        Key innovation: Two types of ILR
        1. d-ILR (density ILR): Move cells from high-density to low-density bins
        2. r-ILR (refinement ILR): Move cells to reduce wirelength using 8-neighbor scoring

        8-Neighbor Scoring:
        - For each cell, compute score for moving to each of 8 neighboring bins
        - Score = wirelength_reduction - density_penalty
        - Move to best neighbor if score is positive
        """
        # Spread components within their parent cluster
        current_clusters = levels[level_idx]

        for cluster in current_clusters:
            refs = list(cluster)
            if len(refs) <= 1:
                continue

            # Get centroid
            sum_x = sum(self.components[ref].x for ref in refs)
            sum_y = sum(self.components[ref].y for ref in refs)
            cx, cy = sum_x / len(refs), sum_y / len(refs)

            # Spread components around centroid
            angle_step = 2 * math.pi / len(refs)
            radius = 2.0

            for i, ref in enumerate(refs):
                angle = i * angle_step
                self.components[ref].x = cx + radius * math.cos(angle)
                self.components[ref].y = cy + radius * math.sin(angle)

        # Initialize bin grid for ILR
        bin_size = max(self.config.ep_bin_size, 2.0)
        cols = int(math.ceil(self.config.board_width / bin_size))
        rows = int(math.ceil(self.config.board_height / bin_size))

        # Run ILR iterations
        total_moves = 0
        for iteration in range(self.config.fp_uncoarsening_iterations):
            # Phase 1: d-ILR (density-based)
            d_moves = self._fp_density_ilr(bin_size, rows, cols)

            # Phase 2: r-ILR (refinement with 8-neighbor scoring)
            r_moves = self._fp_refinement_ilr(bin_size, rows, cols)

            total_moves += d_moves + r_moves

            if d_moves == 0 and r_moves == 0:
                break  # Converged

        return total_moves

    def _fp_density_ilr(self, bin_size: float, rows: int, cols: int) -> int:
        """
        Density-based ILR (d-ILR).

        Move cells from high-density bins to low-density bins.
        Reference: FastPlace 3.0 paper
        """
        # Compute bin densities
        density = [[0.0] * cols for _ in range(rows)]
        bin_capacity = bin_size * bin_size

        ref_to_bin = {}
        for ref, comp in self.components.items():
            bx = int((comp.x - self.config.origin_x) / bin_size)
            by = int((comp.y - self.config.origin_y) / bin_size)
            bx = max(0, min(cols - 1, bx))
            by = max(0, min(rows - 1, by))
            ref_to_bin[ref] = (bx, by)
            density[by][bx] += comp.width * comp.height

        # Normalize
        for r in range(rows):
            for c in range(cols):
                density[r][c] /= bin_capacity

        # Find target density (average)
        total_density = sum(sum(row) for row in density)
        target_density = total_density / (rows * cols)

        # Move cells from overfilled bins
        moves = 0
        refs_sorted = sorted(self.components.keys(),
                            key=lambda r: density[ref_to_bin[r][1]][ref_to_bin[r][0]],
                            reverse=True)

        for ref in refs_sorted:
            bx, by = ref_to_bin[ref]
            comp = self.components[ref]

            if density[by][bx] <= target_density * 1.2:
                continue  # Bin not overfilled

            # Find best neighboring bin with lower density
            best_score = 0
            best_target = None

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = bx + dx, by + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    if density[ny][nx] < density[by][bx]:
                        score = density[by][bx] - density[ny][nx]
                        if score > best_score:
                            best_score = score
                            best_target = (nx, ny)

            if best_target:
                nx, ny = best_target
                # Move component
                new_x = self.config.origin_x + (nx + 0.5) * bin_size
                new_y = self.config.origin_y + (ny + 0.5) * bin_size

                # Update densities
                area = comp.width * comp.height
                density[by][bx] -= area / bin_capacity
                density[ny][nx] += area / bin_capacity

                comp.x = new_x
                comp.y = new_y
                ref_to_bin[ref] = (nx, ny)
                moves += 1

        return moves

    def _fp_refinement_ilr(self, bin_size: float, rows: int, cols: int) -> int:
        """
        Refinement ILR (r-ILR) with 8-neighbor scoring.

        For each cell, compute 8 scores for moving to neighboring bins:
        - Score = wirelength_reduction - alpha * density_penalty

        Reference: FastPlace 3.0, Section 4.2
        """
        # Compute bin densities
        density = [[0.0] * cols for _ in range(rows)]
        bin_capacity = bin_size * bin_size

        for comp in self.components.values():
            bx = int((comp.x - self.config.origin_x) / bin_size)
            by = int((comp.y - self.config.origin_y) / bin_size)
            bx = max(0, min(cols - 1, bx))
            by = max(0, min(rows - 1, by))
            density[by][bx] += comp.width * comp.height / bin_capacity

        # 8-neighbor directions
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        moves = 0
        alpha = 0.5  # Density penalty weight

        for ref, comp in self.components.items():
            if comp.fixed:
                continue

            bx = int((comp.x - self.config.origin_x) / bin_size)
            by = int((comp.y - self.config.origin_y) / bin_size)
            bx = max(0, min(cols - 1, bx))
            by = max(0, min(rows - 1, by))

            # Compute current wirelength contribution
            current_wl = self._fp_component_wirelength(ref)

            # Compute 8 neighbor scores
            best_score = 0
            best_move = None

            for dx, dy in neighbors:
                nx, ny = bx + dx, by + dy
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue

                # Tentatively move component
                old_x, old_y = comp.x, comp.y
                comp.x = self.config.origin_x + (nx + 0.5) * bin_size
                comp.y = self.config.origin_y + (ny + 0.5) * bin_size

                # Compute new wirelength
                new_wl = self._fp_component_wirelength(ref)
                wl_reduction = current_wl - new_wl

                # Density penalty
                density_penalty = max(0, density[ny][nx] - 1.0)

                # 8-neighbor score
                score = wl_reduction - alpha * density_penalty

                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)

                # Restore position
                comp.x, comp.y = old_x, old_y

            if best_move:
                nx, ny = best_move
                # Apply the move
                area = comp.width * comp.height / bin_capacity
                density[by][bx] -= area
                density[ny][nx] += area

                comp.x = self.config.origin_x + (nx + 0.5) * bin_size
                comp.y = self.config.origin_y + (ny + 0.5) * bin_size
                moves += 1

        return moves

    def _fp_component_wirelength(self, ref: str) -> float:
        """Compute HPWL contribution of a single component"""
        total_wl = 0.0
        comp = self.components[ref]

        for net_name, net_info in self.nets.items():
            pins = net_info.get('pins', [])
            comps_in_net = [c for c, p in pins if c in self.components]

            if ref not in comps_in_net or len(comps_in_net) < 2:
                continue

            xs = [self.components[c].x for c in comps_in_net]
            ys = [self.components[c].y for c in comps_in_net]

            hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            weight = net_info.get('weight', 1.0)
            total_wl += hpwl * weight

        return total_wl

    # =========================================================================
    # ALGORITHM 8: ePLACE (Electrostatic Placement)
    # =========================================================================

    def _place_eplace(self) -> PlacementResult:
        """
        ePlace: Electrostatic-based analytical placement.

        Reference: "Analytic VLSI Placement using Electrostatic Analogy"

        Key ideas:
        1. Model placement density as electric charge
        2. Solve Poisson's equation for potential field
        3. Use FFT for efficient spectral solution
        4. Nesterov's method for faster convergence

        Simplified implementation without full FFT.
        """
        print("  [EP] Running ePlace electrostatic placement...")

        refs = sorted(self.components.keys())
        n = len(refs)

        if n == 0:
            return self._create_result('eplace', 0, True)

        # Initialize density grid
        self._ep_init_grid()

        # Nesterov's accelerated gradient descent
        # y = x (initial)
        # For each iteration:
        #   gradient = wirelength_gradient + lambda * density_gradient
        #   x_new = y - alpha * gradient
        #   y = x_new + momentum * (x_new - x)

        x = {ref: (self.components[ref].x, self.components[ref].y) for ref in refs}
        y = copy.deepcopy(x)
        momentum = 0.9

        for iteration in range(self.config.ep_iterations):
            # Update density grid
            self._ep_update_density_grid()

            # Compute gradients
            for ref in refs:
                comp = self.components[ref]
                if comp.fixed:
                    continue

                # Wirelength gradient (HPWL approximation)
                wl_gx, wl_gy = self._ep_wirelength_gradient(ref)

                # Density gradient (from electric field)
                d_gx, d_gy = self._ep_density_gradient(ref)

                # Combined gradient
                gx = wl_gx + self.config.ep_lambda * d_gx
                gy = wl_gy + self.config.ep_lambda * d_gy

                # Nesterov update
                yx, yy = y[ref]
                new_x = yx - self.config.ep_nesterov_alpha * gx
                new_y = yy - self.config.ep_nesterov_alpha * gy

                # Clamp to board
                margin = self.config.edge_margin + max(comp.width, comp.height) / 2
                new_x = max(self.config.origin_x + margin,
                            min(self.config.origin_x + self.config.board_width - margin, new_x))
                new_y = max(self.config.origin_y + margin,
                            min(self.config.origin_y + self.config.board_height - margin, new_y))

                # Momentum
                old_x, old_y = x[ref]
                x[ref] = (new_x, new_y)
                y[ref] = (new_x + momentum * (new_x - old_x),
                          new_y + momentum * (new_y - old_y))

                # Apply to component
                comp.x, comp.y = new_x, new_y

            if (iteration + 1) % 20 == 0:
                cost = self._calculate_cost()
                print(f"  [EP] Iteration {iteration + 1}: cost = {cost:.2f}")

        self._legalize()

        cost = self._calculate_cost()
        print(f"  [EP] Final cost: {cost:.2f}")
        return self._create_result('eplace', self.config.ep_iterations, True)

    def _ep_init_grid(self):
        """Initialize density grid for ePlace"""
        bin_size = self.config.ep_bin_size

        cols = int(math.ceil(self.config.board_width / bin_size))
        rows = int(math.ceil(self.config.board_height / bin_size))

        self._density_grid = [[0.0] * cols for _ in range(rows)]
        self._potential_grid = [[0.0] * cols for _ in range(rows)]
        self._grid_cols = cols
        self._grid_rows = rows

    def _ep_update_density_grid(self):
        """
        Update density grid and solve Poisson's equation using FFT spectral method.

        Reference: "ePlace: Electrostatics-Based Placement Using Nesterov's Method"
                   "Efficient and Effective Placement for Very Large Circuits" (TCAD 2014)

        Poisson's equation: -∇²φ = ρ (charge density)

        FFT Spectral Method:
        1. Transform density to frequency domain: ρ̂ = FFT(ρ)
        2. Solve in frequency domain: φ̂(k) = ρ̂(k) / (kx² + ky²)
        3. Transform back: φ = IFFT(φ̂)

        This gives O(m log m) complexity vs O(m²) for direct solver.
        """
        bin_size = self.config.ep_bin_size
        rows = self._grid_rows
        cols = self._grid_cols

        # Reset density grid
        for r in range(rows):
            for c in range(cols):
                self._density_grid[r][c] = 0.0

        # Compute density with bell-shaped spreading (bilinear interpolation)
        bin_area = bin_size ** 2
        target_density = 1.0  # Uniform target

        for comp in self.components.values():
            # Find bin
            fx = (comp.x - self.config.origin_x) / bin_size
            fy = (comp.y - self.config.origin_y) / bin_size
            bx = int(fx)
            by = int(fy)

            # Bilinear weights
            wx1 = fx - bx
            wy1 = fy - by
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            area = comp.width * comp.height / bin_area

            # Distribute to 4 neighboring bins
            for dr, wy in [(0, wy0), (1, wy1)]:
                for dc, wx in [(0, wx0), (1, wx1)]:
                    r, c = by + dr, bx + dc
                    if 0 <= r < rows and 0 <= c < cols:
                        self._density_grid[r][c] += area * wx * wy

        # Compute overflow density (ρ - target)
        overflow = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                overflow[r][c] = self._density_grid[r][c] - target_density

        # =====================================================================
        # FFT-based Poisson Solver
        # Solve -∇²φ = ρ using spectral method
        #
        # In frequency domain: (kx² + ky²) φ̂ = ρ̂
        # Therefore: φ̂ = ρ̂ / (kx² + ky²)
        # =====================================================================

        # Step 1: Compute 2D DFT of overflow density
        # Using real DFT for efficiency (DCT-II for Neumann boundary conditions)
        rho_hat = self._dct2d(overflow)

        # Step 2: Solve in frequency domain
        phi_hat = [[0.0] * cols for _ in range(rows)]

        for ky in range(rows):
            for kx in range(cols):
                if kx == 0 and ky == 0:
                    # DC component (set to zero for zero-mean potential)
                    phi_hat[ky][kx] = 0.0
                else:
                    # Eigenvalues for DCT (Neumann BC)
                    # λ = 2(1 - cos(πk/N)) for each dimension
                    lambda_x = 2.0 * (1.0 - math.cos(math.pi * kx / cols)) if cols > 1 else 0.0
                    lambda_y = 2.0 * (1.0 - math.cos(math.pi * ky / rows)) if rows > 1 else 0.0

                    denom = lambda_x + lambda_y
                    if abs(denom) > 1e-10:
                        phi_hat[ky][kx] = rho_hat[ky][kx] / denom
                    else:
                        phi_hat[ky][kx] = 0.0

        # Step 3: Inverse DCT to get potential
        self._potential_grid = self._idct2d(phi_hat)

    def _dct2d(self, grid: List[List[float]]) -> List[List[float]]:
        """
        2D Discrete Cosine Transform (Type-II).

        DCT-II: X[k] = 2 * sum(x[n] * cos(π(2n+1)k / 2N))

        Used for Neumann boundary conditions (zero normal derivative at edges).
        This is the standard "DCT" used in JPEG, etc.
        """
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        if rows == 0 or cols == 0:
            return grid

        # First DCT along rows
        temp = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            for k in range(cols):
                total = 0.0
                for n in range(cols):
                    total += grid[r][n] * math.cos(math.pi * (2 * n + 1) * k / (2 * cols))
                temp[r][k] = 2.0 * total

        # Then DCT along columns
        result = [[0.0] * cols for _ in range(rows)]
        for c in range(cols):
            for k in range(rows):
                total = 0.0
                for n in range(rows):
                    total += temp[n][c] * math.cos(math.pi * (2 * n + 1) * k / (2 * rows))
                result[k][c] = 2.0 * total

        return result

    def _idct2d(self, grid: List[List[float]]) -> List[List[float]]:
        """
        2D Inverse Discrete Cosine Transform (Type-III).

        IDCT-III: x[n] = X[0]/2 + sum(X[k] * cos(π(2n+1)k / 2N))

        Inverse of DCT-II.
        """
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        if rows == 0 or cols == 0:
            return grid

        # First IDCT along columns
        temp = [[0.0] * cols for _ in range(rows)]
        for c in range(cols):
            for n in range(rows):
                total = grid[0][c] / 2.0
                for k in range(1, rows):
                    total += grid[k][c] * math.cos(math.pi * (2 * n + 1) * k / (2 * rows))
                temp[n][c] = total / rows

        # Then IDCT along rows
        result = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            for n in range(cols):
                total = temp[r][0] / 2.0
                for k in range(1, cols):
                    total += temp[r][k] * math.cos(math.pi * (2 * n + 1) * k / (2 * cols))
                result[r][n] = total / cols

        return result

    def _ep_wirelength_gradient(self, ref: str) -> Tuple[float, float]:
        """Compute wirelength gradient for a component"""
        gx, gy = 0.0, 0.0
        comp = self.components[ref]

        # Sum over all nets containing this component
        for net_name, net_info in self.nets.items():
            pins = net_info.get('pins', [])
            comps_in_net = [c for c, p in pins if c in self.components]

            if ref not in comps_in_net or len(comps_in_net) < 2:
                continue

            # HPWL gradient: derivative is 0 inside bounding box,
            # +weight or -weight at the boundary
            xs = [self.components[c].x for c in comps_in_net]
            ys = [self.components[c].y for c in comps_in_net]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            weight = net_info.get('weight', 1.0)

            if comp.x == min_x:
                gx -= weight
            if comp.x == max_x:
                gx += weight
            if comp.y == min_y:
                gy -= weight
            if comp.y == max_y:
                gy += weight

        return gx, gy

    def _ep_density_gradient(self, ref: str) -> Tuple[float, float]:
        """Compute density gradient (electric field) for a component"""
        comp = self.components[ref]
        bin_size = self.config.ep_bin_size

        bx = int((comp.x - self.config.origin_x) / bin_size)
        by = int((comp.y - self.config.origin_y) / bin_size)

        # Electric field is negative gradient of potential
        # E = -grad(phi)
        gx, gy = 0.0, 0.0

        if 0 < bx < self._grid_cols - 1 and 0 < by < self._grid_rows - 1:
            # Central difference
            gx = (self._potential_grid[by][bx + 1] - self._potential_grid[by][bx - 1]) / 2
            gy = (self._potential_grid[by + 1][bx] - self._potential_grid[by - 1][bx]) / 2

        # Scale by component area
        area = comp.width * comp.height
        return gx * area, gy * area

    # =========================================================================
    # AUTO AND PARALLEL MODES
    # =========================================================================

    def _place_auto(self) -> PlacementResult:
        """Try multiple algorithms and select the best result"""
        print("  [AUTO] Automatic algorithm selection...")

        results = []
        best_result = None
        best_score = float('inf')
        best_state = None

        initial_state = self._save_state()

        # Try algorithms in order of typical quality/speed tradeoff
        algorithms = [
            ('sa', self._place_simulated_annealing),
            ('quadratic', self._place_quadratic),
            ('fd', self._place_force_directed),
            ('pso', self._place_pso),
            ('mincut', self._place_mincut),
        ]

        for algo_name, algo_func in algorithms:
            print(f"\n  [AUTO] Trying {algo_name}...")

            self._restore_state(initial_state)
            for comp in self.components.values():
                comp.vx = comp.vy = comp.fx = comp.fy = 0.0

            try:
                result = algo_func()

                has_overlaps = self._check_has_overlaps()
                has_oob = self._check_out_of_bounds()
                score = result.cost + (10000 if has_overlaps else 0) + (5000 if has_oob else 0)

                quality = "GOOD" if not has_overlaps and not has_oob else "POOR"
                print(f"  [AUTO] {algo_name}: cost={result.cost:.2f}, score={score:.2f} -> {quality}")

                results.append((algo_name, result, score))

                if score < best_score:
                    best_score = score
                    best_result = result
                    best_state = self._save_state()

                    # Early termination if good enough
                    if not has_overlaps and not has_oob and result.cost < 100:
                        print(f"  [AUTO] Good solution found, stopping")
                        break

            except Exception as e:
                print(f"  [AUTO] {algo_name} failed: {e}")

        if best_state:
            self._restore_state(best_state)
            print(f"\n  [AUTO] Selected: {best_result.algorithm_used} (cost={best_result.cost:.2f})")

            return PlacementResult(
                positions={ref: (c.x, c.y) for ref, c in self.components.items()},
                rotations={ref: c.rotation for ref, c in self.components.items()},
                layers={ref: c.layer for ref, c in self.components.items()},
                cost=best_result.cost,
                algorithm_used=f"auto ({best_result.algorithm_used})",
                iterations=sum(r.iterations for _, r, _ in results),
                converged=True,
                success=best_result.success
            )

        return self._create_result('auto (fallback)', 0, False)

    def _place_parallel(self) -> PlacementResult:
        """Run ALL algorithms in parallel and pick the best"""
        print("  [PARALLEL] Running ALL algorithms in parallel...")

        algorithms = ['fd', 'sa', 'ga', 'quadratic', 'mincut', 'pso', 'fastplace', 'eplace']

        results = {}
        results_lock = threading.Lock()

        def run_algo(algo_name: str):
            try:
                # Create isolated piston
                isolated_config = copy.deepcopy(self.config)
                isolated_config.algorithm = algo_name
                if isolated_config.seed is not None:
                    isolated_config.seed = isolated_config.seed + hash(algo_name) % 10000

                isolated = PlacementPiston(isolated_config)
                isolated._copy_state_from(self)

                # Run algorithm
                algo_map = {
                    'fd': isolated._place_force_directed,
                    'sa': isolated._place_simulated_annealing,
                    'ga': isolated._place_genetic,
                    'quadratic': isolated._place_quadratic,
                    'mincut': isolated._place_mincut,
                    'pso': isolated._place_pso,
                    'fastplace': isolated._place_fastplace,
                    'eplace': isolated._place_eplace,
                }

                result = algo_map[algo_name]()

                has_overlaps = isolated._check_has_overlaps()
                has_oob = isolated._check_out_of_bounds()

                with results_lock:
                    results[algo_name] = {
                        'result': result,
                        'has_overlaps': has_overlaps,
                        'has_oob': has_oob,
                        'positions': {ref: (c.x, c.y) for ref, c in isolated.components.items()},
                        'rotations': {ref: c.rotation for ref, c in isolated.components.items()},
                        'layers': {ref: c.layer for ref, c in isolated.components.items()},
                    }

                quality = "GOOD" if not has_overlaps and not has_oob else "POOR"
                print(f"  [PARALLEL] {algo_name}: cost={result.cost:.2f} -> {quality}")

            except Exception as e:
                print(f"  [PARALLEL] {algo_name} failed: {e}")

        # Run in parallel
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {executor.submit(run_algo, algo): algo for algo in algorithms}
            for future in as_completed(futures, timeout=self.config.parallel_timeout):
                try:
                    future.result()
                except Exception as e:
                    print(f"  [PARALLEL] Exception: {e}")

        # Find best result
        best_algo = None
        best_score = float('inf')
        best_data = None

        for algo_name, data in results.items():
            score = data['result'].cost
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
                layers=best_data['layers'],
                cost=best_data['result'].cost,
                algorithm_used=f"parallel ({best_algo})",
                iterations=sum(d['result'].iterations for d in results.values()),
                converged=True,
                success=not best_data['has_overlaps'] and not best_data['has_oob']
            )

        return self._create_result('parallel (fallback)', 0, False)

    def _copy_state_from(self, other: 'PlacementPiston'):
        """Copy state from another piston"""
        self.nets = copy.deepcopy(other.nets)
        self.adjacency = copy.deepcopy(other.adjacency)
        self.hub = other.hub
        self.pin_offsets = copy.deepcopy(other.pin_offsets)
        self.net_weights = copy.deepcopy(other.net_weights)
        self._optimal_dist = other._optimal_dist
        self._board_area = other._board_area

        self.components = {}
        for ref, comp in other.components.items():
            self.components[ref] = comp.copy()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_placement(parts_db: Dict, graph: Dict = None,
                  algorithm: str = 'sa',
                  board_width: float = 50.0,
                  board_height: float = 40.0,
                  origin_x: float = 100.0,
                  origin_y: float = 100.0,
                  grid_size: float = 0.5,
                  seed: int = 42) -> PlacementResult:
    """
    Convenience function for running placement.

    Args:
        parts_db: Parts database
        graph: Connectivity graph (optional)
        algorithm: One of:
            'fd'        - Force-Directed (Fruchterman-Reingold 1991)
            'sa'        - Simulated Annealing (Kirkpatrick 1983)
            'ga'        - Genetic Algorithm (SOGA)
            'quadratic' - Quadratic/Analytical (FastPlace)
            'mincut'    - Min-Cut Partitioning (Breuer 1977)
            'pso'       - Particle Swarm Optimization (APSO)
            'fastplace' - FastPlace Multilevel (FastPlace 3.0)
            'eplace'    - Electrostatic (ePlace)
            'auto'      - Try multiple, pick best
            'parallel'  - Run ALL concurrently
        board_width/height: Board dimensions in mm
        origin_x/y: Board origin in mm
        grid_size: Placement grid in mm
        seed: Random seed

    Returns:
        PlacementResult with positions and metrics
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

    piston = PlacementPiston(config)
    result = piston.place(parts_db, graph or {})

    print(f"  Placement complete: {result.algorithm_used}")
    print(f"  Cost: {result.cost:.2f}, Wire length: {result.wirelength:.2f}")
    print(f"  Overlap: {result.overlap_area:.4f}, OOB: {result.oob_penalty:.4f}")
    print(f"  Success: {result.success}")

    return result
