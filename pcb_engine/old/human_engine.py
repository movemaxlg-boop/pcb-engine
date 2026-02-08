"""
PCB Engine - Human-Like Engine
==============================

This module provides a human-workflow-based PCB design engine.

PHILOSOPHY:
===========
Algorithmic engines try to optimize mathematically.
Human experts use intuition, pattern recognition, and visual thinking.

This engine mimics how a HUMAN EXPERT actually designs PCBs:
1. Look at the whole design first (analyze)
2. Place components logically (groups, chains, flow)
3. Escape pins perpendicular to component body
4. Route short nets first, use Manhattan paths

The result: CLEANER, MORE INTUITIVE layouts that pass DRC.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os

# Import base engine components
from .engine import BoardConfig, DesignRules, EngineState, EnginePhase, EngineStatus

# Import human-like modules
from .human_placement import human_like_placement, Position
from .human_escape import human_like_escapes, EscapeRoute
from .human_routing import human_like_routing, Route


@dataclass
class HumanEngineState:
    """State for human-like engine"""
    phase: str = "INIT"
    status: str = "ready"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Data
    parts_db: Dict = field(default_factory=dict)
    graph: Dict = field(default_factory=dict)
    hub: Optional[str] = None

    # Human-calculated outputs
    placement: Dict[str, Position] = field(default_factory=dict)
    escapes: Dict[str, Dict[str, EscapeRoute]] = field(default_factory=dict)
    routes: Dict[str, Route] = field(default_factory=dict)
    route_order: List[str] = field(default_factory=list)


class HumanPCBEngine:
    """
    PCB Engine that mimics human expert workflow.

    This engine replaces algorithmic optimization with human-like thinking:
    - Placement: Analyze design → place hub → place in signal flow order
    - Escapes: Simple perpendicular escape from component body
    - Routing: Short nets first, Manhattan paths, via when blocked

    Usage:
        engine = HumanPCBEngine(board_config, design_rules)
        engine.load_parts_from_dict(parts)
        engine.run()
        engine.generate_kicad_script('output.py')
    """

    def __init__(self, board: BoardConfig = None, rules: DesignRules = None):
        self.board = board or BoardConfig()
        self.rules = rules or DesignRules()
        self.state = HumanEngineState()

        # Parts collector from base engine
        self._parts_collector = None

    # =========================================================================
    # LOADING
    # =========================================================================

    def load_parts_from_dict(self, parts_dict: Dict) -> bool:
        """Load parts from dictionary"""
        from .parts import PartsCollector

        self._parts_collector = PartsCollector()

        for ref, part_data in parts_dict.items():
            part_data['ref_designator'] = ref
            if not self._parts_collector.add_part_manual(part_data):
                self.state.errors.append(f"Failed to add part: {ref}")
                return False
        return True

    # =========================================================================
    # RUN ALL PHASES (Human Workflow)
    # =========================================================================

    def run(self, placement_algorithm: str = 'human') -> bool:
        """
        Run the complete human-like workflow.

        Args:
            placement_algorithm: Algorithm for component placement
                - 'human': Rule-based human-like placement (default, fast)
                - 'fd': Force-directed placement
                - 'sa': Simulated annealing
                - 'ga': Genetic algorithm (slowest but thorough)
                - 'analytical': Quadratic placement
                - 'hybrid': Force-directed + simulated annealing (best quality)
                - 'auto': Try multiple algorithms, pick best

        Human workflow:
        1. UNDERSTAND: Build the connectivity graph
        2. PLAN: Identify hub and signal chains
        3. PLACE: Position components in logical order
        4. ESCAPE: Calculate simple perpendicular escapes
        5. ROUTE: Connect nets using Manhattan routing
        """
        self.state.status = "running"

        # Phase 1: Validate and export parts
        if not self._phase_1_parts():
            return False

        # Phase 2: Build connectivity graph
        if not self._phase_2_graph():
            return False

        # Phase 3: Placement (using selected algorithm)
        if not self._phase_3_placement(placement_algorithm):
            return False

        # Phase 4: Human-like escape calculation
        if not self._phase_4_escapes():
            return False

        # Phase 5: Human-like routing
        if not self._phase_5_routing():
            return False

        self.state.phase = "COMPLETE"
        self.state.status = "complete"
        return True

    def _phase_1_parts(self) -> bool:
        """Phase 1: Validate parts database and calculate optimal grid size"""
        self.state.phase = "PARTS"

        if self._parts_collector is None:
            self.state.errors.append("No parts loaded")
            return False

        if not self._parts_collector.validate_all():
            self.state.errors.extend(self._parts_collector.errors)
            return False

        self.state.parts_db = self._parts_collector.export()
        print(f"  Parts validated: {len(self.state.parts_db.get('parts', {}))} components")

        # DYNAMIC GRID: Calculate optimal grid size based on component requirements
        self._apply_dynamic_grid()

        return True

    def _apply_dynamic_grid(self):
        """
        Calculate and apply dynamic grid size based on component pad pitches.

        This ensures the routing grid is fine enough to:
        1. Accurately represent pad positions (prevent routes through pads)
        2. Check clearances precisely
        3. Not be so fine that routing becomes slow

        The grid size is calculated AFTER parts are loaded so we know
        the minimum pad pitch from all components.
        """
        from .grid_calculator import calculate_optimal_grid_size, estimate_grid_memory

        # Calculate optimal grid
        # Note: min_grid_size of 0.1mm balances accuracy with performance
        # 0.05mm creates very large grids (slow), 0.1mm is usually sufficient
        # For SOT-23-5 (0.95mm pitch): 0.1mm gives ~10 cells per pitch
        optimal_grid, analysis = calculate_optimal_grid_size(
            self.state.parts_db,
            min_clearance=self.rules.min_clearance,
            min_trace_width=self.rules.min_trace_width,
            max_grid_size=0.25,  # Performance limit
            min_grid_size=0.1    # Minimum practical grid (0.05 is too slow)
        )

        # Store original for comparison
        original_grid = self.board.grid_size

        # Apply the new grid size
        self.board.grid_size = optimal_grid

        # Log the change
        print(f"\n  [DYNAMIC GRID] Calculated optimal grid size:")
        print(f"    Min pad pitch detected: {analysis['min_pad_pitch']:.3f} mm")
        print(f"    Grid size: {original_grid:.3f} mm -> {optimal_grid:.3f} mm")
        print(f"    Limiting factor: {analysis['limiting_factor']}")

        # Memory estimate
        mem = estimate_grid_memory(
            self.board.width, self.board.height, optimal_grid, 2
        )
        print(f"    Grid dimensions: {mem['cols']} x {mem['rows']} cells")
        print(f"    Estimated memory: {mem['estimated_memory_mb']:.1f} MB")

    def _phase_2_graph(self) -> bool:
        """Phase 2: Build connectivity graph"""
        self.state.phase = "GRAPH"

        from .graph import GraphBuilder

        builder = GraphBuilder()
        self.state.graph = builder.build(self.state.parts_db)
        self.state.hub = builder.identify_hub(self.state.graph)

        nets = self.state.parts_db.get('nets', {})
        print(f"  Graph built: {len(nets)} nets, hub={self.state.hub}")
        return True

    def _phase_3_placement(self, algorithm: str = 'human') -> bool:
        """
        Phase 3: Component placement

        Args:
            algorithm: Placement algorithm to use
                - 'human': Rule-based human-like placement (default, fast)
                - 'fd': Force-directed placement
                - 'sa': Simulated annealing
                - 'ga': Genetic algorithm
                - 'analytical': Quadratic placement
                - 'hybrid': Force-directed + simulated annealing (best quality)
                - 'auto': Try multiple algorithms, pick best
        """
        self.state.phase = "PLACEMENT"

        # Use new placement engine for advanced algorithms
        if algorithm in ['fd', 'sa', 'ga', 'analytical', 'hybrid', 'auto']:
            from .placement_engine import PlacementEngine, PlacementConfig
            from .human_placement import Position

            config = PlacementConfig(
                algorithm=algorithm,
                board_width=self.board.width,
                board_height=self.board.height,
                origin_x=self.board.origin_x,
                origin_y=self.board.origin_y,
                grid_size=self.board.grid_size,
                seed=42  # Deterministic
            )

            engine = PlacementEngine(config)
            result = engine.place(self.state.parts_db, self.state.graph)

            # Convert to Position objects
            positions = {}
            for ref, (x, y) in result.positions.items():
                rotation = result.rotations.get(ref, 0)
                positions[ref] = Position(x=x, y=y, rotation=rotation)

            self.state.placement = positions
            self.state.hub = engine.hub

            print(f"  Placed {len(positions)} components using {result.algorithm_used}")
            print(f"  Placement cost: {result.cost:.2f}")
            for ref, pos in positions.items():
                print(f"    {ref}: ({pos.x:.1f}, {pos.y:.1f})")

            return True

        # Default: Use original human-like placer
        from .human_placement import HumanLikePlacer

        placer = HumanLikePlacer(
            board_width=self.board.width,
            board_height=self.board.height,
            origin_x=self.board.origin_x,
            origin_y=self.board.origin_y,
            grid_size=self.board.grid_size,
        )

        # Analyze design (this detects signal chains and hub)
        placer.analyze_design(self.state.parts_db, self.state.graph)

        # Update hub from placer's detection (overrides graph builder's conservative detection)
        if placer.hub:
            self.state.hub = placer.hub

        # Show signal chains detected
        if placer.signal_chains:
            print(f"    Signal chains: {placer.signal_chains}")

        # Show placement order
        print(f"    Placement order: {placer.placement_order}")

        # Place all components
        positions = placer.place_all()

        # Store as Position objects
        self.state.placement = positions

        print(f"  Placed {len(positions)} components")
        for ref, pos in positions.items():
            print(f"    {ref}: ({pos.x:.1f}, {pos.y:.1f})")

        return True

    def _phase_4_escapes(self) -> bool:
        """Phase 4: Human-like escape calculation"""
        self.state.phase = "ESCAPES"

        self.state.escapes = human_like_escapes(
            self.state.parts_db,
            self.state.placement,
            grid_size=self.board.grid_size
        )

        escape_count = sum(len(e) for e in self.state.escapes.values())
        print(f"  Calculated {escape_count} escape routes")
        return True

    def _phase_5_routing(self) -> bool:
        """Phase 5: Human-like routing"""
        self.state.phase = "ROUTING"

        nets = self.state.parts_db.get('nets', {})

        # Build route order: signal nets first (short first), then power nets
        # Power nets (VCC, 3V3, 5V) are routed LAST so they don't block signal routes
        # GND is EXCLUDED - it will be handled by copper pour/zone on B.Cu
        power_nets = ['VCC', '3V3', '5V', 'VBUS']  # GND excluded - using zone
        signal_nets = [n for n in nets.keys() if n not in power_nets and n != 'NC' and n != 'GND']
        power_to_route = [n for n in power_nets if n in nets.keys()]

        # Route signals first, then power (GND handled by zone, not traces)
        self.state.route_order = signal_nets + power_to_route

        # Note: GND will be connected via copper pour/zone in _generate_gnd_zone_code()

        # Route using human-like router (pass placement and design rules for DRC compliance)
        self.state.routes = human_like_routing(
            self.board,
            self.state.parts_db,
            self.state.escapes,
            self.state.route_order,
            self.state.placement,  # Pass placement so router avoids component bodies
            self.rules  # Pass design rules for trace width and clearance
        )

        # Report results
        success_count = sum(1 for r in self.state.routes.values() if r.success)
        total = len(self.state.route_order)

        print(f"  Routed {success_count}/{total} nets")

        if success_count < total:
            failed = [n for n, r in self.state.routes.items() if not r.success]
            self.state.warnings.append(f"Failed to route: {failed}")

        return success_count > 0 or total == 0

    # =========================================================================
    # DRC FEEDBACK LOOP
    # =========================================================================

    def run_with_drc_loop(self, max_iterations: int = 5,
                          output_path: str = None,
                          placement_algorithm: str = 'auto',
                          try_all_placements: bool = True) -> Dict:
        """
        Run the engine with automatic DRC feedback loop.

        This uses the built-in ValidationGate to:
        1. Generate initial layout (trying multiple placement algorithms)
        2. Check for DRC violations
        3. Fix violations (reroute nets, adjust clearance)
        4. If still failing, try different placement algorithm
        5. Repeat until DRC passes or all options exhausted

        Args:
            max_iterations: Maximum correction attempts per placement (default 5)
            output_path: Optional path for KiCad script output
            placement_algorithm: Initial placement algorithm ('human', 'hybrid', 'auto', etc.)
            try_all_placements: If True, try all placement algorithms when one fails

        Returns:
            {
                'success': bool,
                'iterations': int,
                'final_violations': list,
                'routed_nets': int,
                'total_nets': int,
                'placement_algorithm': str,
            }
        """
        from .drc_feedback import DRCFeedback

        # Define placement algorithms to try (in order of preference)
        placement_algorithms = ['hybrid', 'human', 'analytical', 'fd', 'sa', 'ga']

        # If specific algorithm requested, try it first
        if placement_algorithm != 'auto' and placement_algorithm in placement_algorithms:
            placement_algorithms.remove(placement_algorithm)
            placement_algorithms.insert(0, placement_algorithm)

        best_result = None
        best_routed = -1
        best_algorithm = None

        # Try each placement algorithm
        algorithms_to_try = placement_algorithms if try_all_placements else [placement_algorithm]

        for algo_idx, algo in enumerate(algorithms_to_try):
            if algo_idx > 0:
                print(f"\n{'='*60}")
                print(f"TRYING DIFFERENT PLACEMENT: {algo.upper()}")
                print(f"{'='*60}\n")

            # Reset state for new attempt
            self.state.routes = {}
            self.state.escapes = {}
            self.state.placement = {}
            self.state.phase = "INIT"
            self.state.status = "init"

            # Run phases 1-2 (validate and graph)
            if not self._phase_1_parts():
                continue
            if not self._phase_2_graph():
                continue

            # Run placement with current algorithm
            print(f"\n[PLACEMENT] Using {algo} algorithm...")
            if not self._phase_3_placement(algo):
                print(f"  [PLACEMENT] {algo} failed, trying next...")
                continue

            # Run escapes and routing
            if not self._phase_4_escapes():
                continue
            if not self._phase_5_routing():
                continue

            self.state.phase = "COMPLETE"
            self.state.status = "complete"

            # Run DRC feedback loop
            feedback = DRCFeedback(self, max_iterations=max_iterations)
            result = feedback.run_feedback_loop(kicad_script_path=output_path)

            # Track results
            routed = sum(1 for r in self.state.routes.values() if r.success)
            total = len(self.state.route_order)
            result['routed_nets'] = routed
            result['total_nets'] = total
            result['placement_algorithm'] = algo

            # Check if this is the best result so far
            if result['success']:
                print(f"\n[SUCCESS] Placement '{algo}' passed DRC!")
                return result

            # Track best partial result
            if routed > best_routed or (routed == best_routed and
                                        len(result.get('final_violations', [])) <
                                        len(best_result.get('final_violations', []) if best_result else [])):
                best_routed = routed
                best_result = result
                best_algorithm = algo
                # Save state for best result
                self._best_state = {
                    'routes': self.state.routes.copy(),
                    'escapes': self.state.escapes.copy(),
                    'placement': self.state.placement.copy(),
                }

            print(f"\n[PLACEMENT] {algo}: Routed {routed}/{total} nets, "
                  f"{len(result.get('final_violations', []))} violations")

            # If we got close (e.g., all nets routed but minor DRC issues), stop
            if routed == total and len(result.get('final_violations', [])) <= 2:
                print(f"  [PLACEMENT] Good enough with {algo}, stopping search")
                break

        # Return best result found
        if best_result:
            print(f"\n{'='*60}")
            print(f"BEST PLACEMENT: {best_algorithm}")
            print(f"  Routed: {best_result['routed_nets']}/{best_result['total_nets']} nets")
            print(f"  Violations: {len(best_result.get('final_violations', []))}")
            print(f"{'='*60}")

            # Restore best state
            if hasattr(self, '_best_state'):
                self.state.routes = self._best_state['routes']
                self.state.escapes = self._best_state['escapes']
                self.state.placement = self._best_state['placement']

            return best_result

        # Fallback - no successful placement
        return {
            'success': False,
            'iterations': 0,
            'final_violations': [],
            'routed_nets': 0,
            'total_nets': len(self.state.parts_db.get('nets', {})),
            'placement_algorithm': 'none',
        }

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def generate_kicad_script(self, output_path: str, include_routes: bool = True) -> bool:
        """
        Generate KiCad Python script.

        Uses the custom generator for human-like output.
        """
        if self.state.phase != "COMPLETE":
            self.state.errors.append("Engine must complete before generating script")
            return False

        script = self._generate_script(include_routes)

        with open(output_path, 'w') as f:
            f.write(script)

        print(f"\n  KiCad script written to: {output_path}")
        return True

    def _generate_script(self, include_routes: bool) -> str:
        """Generate the KiCad Python script content"""
        from datetime import datetime

        parts = self.state.parts_db.get('parts', {})
        nets = self.state.parts_db.get('nets', {})

        lines = [
            '"""',
            '=' * 80,
            'KiCad PCB Script - Generated by Human-Like PCB Engine',
            '=' * 80,
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'Board Size: {self.board.width}mm x {self.board.height}mm',
            f'Components: {len(parts)}',
            f'Nets: {len(nets)}',
            '',
            'HUMAN-LIKE DESIGN PRINCIPLES:',
            '  - Components placed in signal flow order',
            '  - Escapes perpendicular to component body',
            '  - Manhattan routing (horizontal + vertical)',
            '  - Short nets routed first',
            '',
            'Usage in KiCad Python console:',
            '  1. Open Tools -> Scripting Console',
            '  2. Run: exec(open("path/to/this_file.py").read())',
            '=' * 80,
            '"""',
            '',
            '# ' + '=' * 77,
            '# IMPORTS',
            '# ' + '=' * 77,
            'import pcbnew',
            'from pcbnew import (VECTOR2I, FromMM, ToMM, F_Cu, B_Cu, Edge_Cuts,',
            '                    F_SilkS, B_SilkS, F_Mask, B_Mask, F_Paste, B_Paste,',
            '                    F_Fab, B_Fab)',
            '',
            '# Get current board',
            'try:',
            '    board = pcbnew.GetBoard()',
            '    if board is None:',
            '        print("ERROR: No board loaded. Open a board first.")',
            '        raise SystemExit',
            'except SystemExit:',
            '    raise',
            'except Exception as e:',
            '    print(f"ERROR: Could not get board: {e}")',
            '    raise SystemExit',
            '',
            'print("=" * 60)',
            'print("HUMAN-LIKE PCB ENGINE - SCRIPT EXECUTION")',
            'print("=" * 60)',
            '',
        ]

        # Configuration
        lines.extend([
            '# ' + '=' * 77,
            '# CONFIGURATION',
            '# ' + '=' * 77,
            f'BX, BY = {self.board.origin_x}, {self.board.origin_y}  # Board origin (mm)',
            f'BW, BH = {self.board.width}, {self.board.height}  # Board size (mm)',
            f'GRID = {self.board.grid_size}  # Routing grid (mm)',
            '',
            '# Design rules',
            f'MIN_TRACE = {self.rules.min_trace_width}  # mm',
            f'MIN_CLEARANCE = {self.rules.min_clearance}  # mm',
            f'MIN_VIA_DIA = {self.rules.min_via_diameter}  # mm',
            f'MIN_VIA_DRILL = {self.rules.min_via_drill}  # mm',
            '',
        ])

        # Helper functions
        lines.extend([
            '# ' + '=' * 77,
            '# HELPER FUNCTIONS',
            '# ' + '=' * 77,
            'def mm(val):',
            '    """Convert mm to KiCad internal units"""',
            '    return FromMM(val)',
            '',
            'def pt(x, y):',
            '    """Create a point from mm coordinates"""',
            '    return VECTOR2I(mm(x), mm(y))',
            '',
            'nets_cache = {}',
            'track_count = 0',
            'via_count = 0',
            '',
            'def get_or_create_net(name):',
            '    """Get or create a net by name"""',
            '    if name in nets_cache:',
            '        return nets_cache[name]',
            '    existing = board.FindNet(name)',
            '    if existing:',
            '        nets_cache[name] = existing',
            '        return existing',
            '    net = pcbnew.NETINFO_ITEM(board, name)',
            '    board.Add(net)',
            '    nets_cache[name] = net',
            '    return net',
            '',
            'def add_track(x1, y1, x2, y2, net_name, width=MIN_TRACE, layer=F_Cu):',
            '    """Add a track segment"""',
            '    global track_count',
            '    track = pcbnew.PCB_TRACK(board)',
            '    track.SetStart(pt(x1, y1))',
            '    track.SetEnd(pt(x2, y2))',
            '    track.SetWidth(mm(width))',
            '    track.SetLayer(layer)',
            '    track.SetNet(get_or_create_net(net_name))',
            '    board.Add(track)',
            '    track_count += 1',
            '    return track',
            '',
            'def add_via(x, y, net_name, dia=MIN_VIA_DIA, drill=MIN_VIA_DRILL):',
            '    """Add a through-hole via"""',
            '    global via_count',
            '    via = pcbnew.PCB_VIA(board)',
            '    via.SetPosition(pt(x, y))',
            '    via.SetWidth(mm(dia))',
            '    via.SetDrill(mm(drill))',
            '    via.SetNet(get_or_create_net(net_name))',
            '    try:',
            '        via.SetViaType(pcbnew.VIATYPE_THROUGH)',
            '    except AttributeError:',
            '        try:',
            '            via.SetViaType(pcbnew.VIA_THROUGH)',
            '        except:',
            '            pass',
            '    board.Add(via)',
            '    via_count += 1',
            '    return via',
            '',
            'def find_footprint(ref):',
            '    """Find a footprint by reference"""',
            '    for fp in board.GetFootprints():',
            '        if fp.GetReference() == ref:',
            '            return fp',
            '    return None',
            '',
            'def assign_net_to_pad(fp, pad_name, net_name):',
            '    """Assign a net to a pad on a footprint"""',
            '    if fp is None:',
            '        return False',
            '    for pad in fp.Pads():',
            '        if pad.GetName() == str(pad_name):',
            '            pad.SetNet(get_or_create_net(net_name))',
            '            return True',
            '    return False',
            '',
            '# Footprint library mapping - maps our footprint names to KiCad library footprints',
            'FOOTPRINT_MAP = {',
            '    # Resistors',
            '    "0402": ("Resistor_SMD", "R_0402_1005Metric"),',
            '    "0603": ("Resistor_SMD", "R_0603_1608Metric"),',
            '    "0805": ("Resistor_SMD", "R_0805_2012Metric"),',
            '    "1206": ("Resistor_SMD", "R_1206_3216Metric"),',
            '    # Capacitors',
            '    "C0402": ("Capacitor_SMD", "C_0402_1005Metric"),',
            '    "C0603": ("Capacitor_SMD", "C_0603_1608Metric"),',
            '    "C0805": ("Capacitor_SMD", "C_0805_2012Metric"),',
            '    # LEDs',
            '    "LED_0603": ("LED_SMD", "LED_0603_1608Metric"),',
            '    "LED_0805": ("LED_SMD", "LED_0805_2012Metric"),',
            '    # QFN packages',
            '    "QFN-16": ("Package_DFN_QFN", "QFN-16-1EP_3x3mm_P0.5mm_EP1.75x1.75mm"),',
            '    "QFN-20": ("Package_DFN_QFN", "QFN-20-1EP_4x4mm_P0.5mm_EP2.5x2.5mm"),',
            '    # SOT packages',
            '    "SOT-223": ("Package_TO_SOT_SMD", "SOT-223-3_TabPin2"),',
            '    "SOT-23": ("Package_TO_SOT_SMD", "SOT-23"),',
            '}',
            '',
            'def create_smd_pad(fp, name, x, y, width, height, layer=F_Cu):',
            '    """Create an SMD pad on a footprint"""',
            '    pad = pcbnew.PAD(fp)',
            '    pad.SetName(str(name))',
            '    pad.SetShape(pcbnew.PAD_SHAPE_RECT)',
            '    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)',
            '    # KiCad 9.0: Use SMDMask() for proper layer set',
            '    pad.SetLayerSet(pad.SMDMask())',
            '    pad.SetSize(VECTOR2I(mm(width), mm(height)))',
            '    pad.SetPosition(pt(x, y))',
            '    fp.Add(pad)',
            '    return pad',
            '',
            'def create_simple_2pin_footprint(ref, value, pad_pitch, pad_width=0.6, pad_height=0.5, footprint_name=""):',
            '    """Create a simple 2-pad SMD footprint (resistor, capacitor, LED)"""',
            '    fp = pcbnew.FOOTPRINT(board)',
            '    fp.SetReference(ref)',
            '    fp.SetValue(value)',
            '    ',
            '    # Set footprint ID with CUSTOM_ prefix',
            '    custom_name = f"CUSTOM_{footprint_name}" if footprint_name else f"CUSTOM_2PIN_{ref}"',
            '    try:',
            '        from pcbnew import LIB_ID',
            '        fp.SetFPID(LIB_ID("PCBEngine_Custom", custom_name))',
            '    except:',
            '        fp.SetDescription(f"Custom footprint: {custom_name}")',
            '    ',
            '    # Create pads',
            '    create_smd_pad(fp, "1", -pad_pitch/2, 0, pad_width, pad_height)',
            '    create_smd_pad(fp, "2", pad_pitch/2, 0, pad_width, pad_height)',
            '    ',
            '    # Add reference text',
            '    fp.Reference().SetPosition(pt(0, -1.5))',
            '    fp.Reference().SetTextSize(VECTOR2I(mm(0.8), mm(0.8)))',
            '    fp.Reference().SetLayer(F_SilkS)',
            '    ',
            '    return fp',
            '',
            'def create_qfn_footprint(ref, value, body_size, pin_count, pitch=0.5):',
            '    """Create a QFN-style footprint"""',
            '    fp = pcbnew.FOOTPRINT(board)',
            '    fp.SetReference(ref)',
            '    fp.SetValue(value)',
            '    ',
            '    # Set footprint ID with CUSTOM_ prefix',
            '    custom_name = f"CUSTOM_QFN-{pin_count}"',
            '    try:',
            '        from pcbnew import LIB_ID',
            '        fp.SetFPID(LIB_ID("PCBEngine_Custom", custom_name))',
            '    except:',
            '        fp.SetDescription(f"Custom footprint: {custom_name}")',
            '    ',
            '    pins_per_side = pin_count // 4',
            '    half_body = body_size / 2',
            '    pad_width = 0.25',
            '    pad_height = 0.8',
            '    ',
            '    pin_num = 1',
            '    # Bottom side (pins go left to right)',
            '    start_x = -((pins_per_side - 1) * pitch) / 2',
            '    for i in range(pins_per_side):',
            '        x = start_x + i * pitch',
            '        y = half_body',
            '        create_smd_pad(fp, str(pin_num), x, y, pad_width, pad_height)',
            '        pin_num += 1',
            '    ',
            '    # Right side (pins go bottom to top)',
            '    for i in range(pins_per_side):',
            '        x = half_body',
            '        y = start_x + (pins_per_side - 1 - i) * pitch',
            '        create_smd_pad(fp, str(pin_num), x, y, pad_height, pad_width)',
            '        pin_num += 1',
            '    ',
            '    # Top side (pins go right to left)',
            '    for i in range(pins_per_side):',
            '        x = start_x + (pins_per_side - 1 - i) * pitch',
            '        y = -half_body',
            '        create_smd_pad(fp, str(pin_num), x, y, pad_width, pad_height)',
            '        pin_num += 1',
            '    ',
            '    # Left side (pins go top to bottom)',
            '    for i in range(pins_per_side):',
            '        x = -half_body',
            '        y = start_x + i * pitch',
            '        create_smd_pad(fp, str(pin_num), x, y, pad_height, pad_width)',
            '        pin_num += 1',
            '    ',
            '    fp.Reference().SetPosition(pt(0, -half_body - 1.5))',
            '    fp.Reference().SetTextSize(VECTOR2I(mm(0.8), mm(0.8)))',
            '    fp.Reference().SetLayer(F_SilkS)',
            '    ',
            '    return fp',
            '',
            'def create_custom_footprint(ref, value, pins_data, body_size, footprint_name=""):',
            '    """Create a custom footprint from user-provided pin positions"""',
            '    fp = pcbnew.FOOTPRINT(board)',
            '    fp.SetReference(ref)',
            '    fp.SetValue(value)',
            '    ',
            '    # Set footprint ID with CUSTOM_ prefix to distinguish from KiCad library footprints',
            '    custom_fp_name = f"CUSTOM_{footprint_name}" if footprint_name else f"CUSTOM_{ref}"',
            '    try:',
            '        # KiCad 8+: Use LIB_ID',
            '        from pcbnew import LIB_ID',
            '        fp.SetFPID(LIB_ID("PCBEngine_Custom", custom_fp_name))',
            '    except:',
            '        # Fallback: just set the description',
            '        fp.SetDescription(f"Custom footprint: {custom_fp_name}")',
            '    ',
            '    # Create pads at specified positions',
            '    for pin in pins_data:',
            '        pin_num = pin.get("number", "1")',
            '        offset = pin.get("offset", (0, 0))',
            '        pad_size = pin.get("pad_size", (0.6, 0.6))',
            '        ',
            '        pad = pcbnew.PAD(fp)',
            '        pad.SetName(str(pin_num))',
            '        pad.SetShape(pcbnew.PAD_SHAPE_RECT)',
            '        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)',
            '        # KiCad 9.0: Use SMDMask() for proper layer set',
            '        pad.SetLayerSet(pad.SMDMask())',
            '        pad.SetSize(VECTOR2I(mm(pad_size[0]), mm(pad_size[1])))',
            '        # Position relative to footprint center - use SetFPRelativePosition for KiCad 9',
            '        try:',
            '            pad.SetFPRelativePosition(VECTOR2I(mm(offset[0]), mm(offset[1])))',
            '        except AttributeError:',
            '            # Fallback for older KiCad versions',
            '            pad.SetPosition(pt(offset[0], offset[1]))',
            '        fp.Add(pad)',
            '    ',
            '    # Add silkscreen outline',
            '    if body_size:',
            '        half_w = body_size[0] / 2',
            '        half_h = body_size[1] / 2',
            '        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]',
            '        for i in range(4):',
            '            line = pcbnew.PCB_SHAPE(fp)',
            '            line.SetShape(pcbnew.SHAPE_T_SEGMENT)',
            '            line.SetStart(VECTOR2I(mm(corners[i][0]), mm(corners[i][1])))',
            '            line.SetEnd(VECTOR2I(mm(corners[(i+1)%4][0]), mm(corners[(i+1)%4][1])))',
            '            line.SetLayer(F_SilkS)',
            '            line.SetWidth(mm(0.12))',
            '            fp.Add(line)',
            '    ',
            '    # Reference text - position OUTSIDE solder mask to avoid clipping',
            '    # Calculate safe offset: body + pad height + text margin',
            '    # Pads extend beyond body edge, so need extra clearance',
            '    max_pad_extent = max((abs(p.get("offset", (0,0))[1]) + p.get("pad_size", (0.6, 0.6))[1]/2) for p in pins_data) if pins_data else 0',
            '    body_half_h = body_size[1]/2 if body_size else 1.0',
            '    ref_offset = max(body_half_h, max_pad_extent) + 1.5  # 1.5mm margin from solder mask',
            '    fp.Reference().SetPosition(pt(0, -ref_offset))',
            '    fp.Reference().SetTextSize(VECTOR2I(mm(0.8), mm(0.8)))',
            '    fp.Reference().SetLayer(F_SilkS)',
            '    ',
            '    return fp',
            '',
            'def load_kicad_library_footprint(ref, kicad_footprint, value=""):',
            '    """',
            '    Load a footprint from KiCad library by full path (e.g., "Package_TO_SOT_SMD:SOT-23-5").',
            '    These footprints include 3D models for visualization.',
            '    """',
            '    import os',
            '    ',
            '    # Parse library:footprint format',
            '    if ":" in kicad_footprint:',
            '        lib_name, fp_name = kicad_footprint.split(":", 1)',
            '    else:',
            '        # Assume it is just the footprint name, try common libraries',
            '        fp_name = kicad_footprint',
            '        lib_name = None',
            '    ',
            '    # KiCad installation paths to search',
            '    kicad_paths = [',
            '        "C:/Program Files/KiCad/9.0/share/kicad/footprints",',
            '        "C:/Program Files/KiCad/8.0/share/kicad/footprints",',
            '        "C:/Program Files/KiCad/7.0/share/kicad/footprints",',
            '        "/usr/share/kicad/footprints",',
            '        "/usr/local/share/kicad/footprints",',
            '    ]',
            '    ',
            '    # Also check environment variable',
            '    kicad_env = os.environ.get("KICAD8_FOOTPRINT_DIR") or os.environ.get("KICAD_FOOTPRINT_DIR")',
            '    if kicad_env:',
            '        kicad_paths.insert(0, kicad_env)',
            '    ',
            '    # Try to load from each path',
            '    for kicad_path in kicad_paths:',
            '        if not os.path.exists(kicad_path):',
            '            continue',
            '        ',
            '        if lib_name:',
            '            # Direct library path',
            '            lib_path = os.path.join(kicad_path, lib_name + ".pretty")',
            '            if os.path.exists(lib_path):',
            '                try:',
            '                    fp = pcbnew.FootprintLoad(lib_path, fp_name)',
            '                    if fp:',
            '                        fp.SetReference(ref)',
            '                        if value:',
            '                            fp.SetValue(value)',
            '                        board.Add(fp)',
            '                        print(f"    Loaded from {lib_path}")',
            '                        return fp',
            '                except Exception as e:',
            '                    print(f"    Error loading from {lib_path}: {e}")',
            '        else:',
            '            # Search all .pretty folders',
            '            for lib_folder in os.listdir(kicad_path):',
            '                if lib_folder.endswith(".pretty"):',
            '                    lib_path = os.path.join(kicad_path, lib_folder)',
            '                    try:',
            '                        fp = pcbnew.FootprintLoad(lib_path, fp_name)',
            '                        if fp:',
            '                            fp.SetReference(ref)',
            '                            if value:',
            '                                fp.SetValue(value)',
            '                            board.Add(fp)',
            '                            print(f"    Loaded from {lib_path}")',
            '                            return fp',
            '                    except:',
            '                        pass',
            '    ',
            '    return None',
            '',
            'def load_or_create_footprint(ref, footprint_name, value="", pins_data=None, body_size=None, kicad_footprint=None):',
            '    """',
            '    Load footprint from KiCad library (with 3D model) or create custom.',
            '    ',
            '    Priority:',
            '      1. kicad_footprint specified -> Load from KiCad library (has 3D model)',
            '      2. pins_data provided -> Create custom footprint (no 3D)',
            '      3. footprint_name in FOOTPRINT_MAP -> Load from KiCad library',
            '      4. Fallback -> Create generic footprint',
            '    """',
            '    # IMPORTANT: Delete existing footprint to ensure fresh creation',
            '    existing = find_footprint(ref)',
            '    if existing:',
            '        print(f"  Removing existing {ref} for fresh creation")',
            '        board.Remove(existing)',
            '    ',
            '    # PRIORITY 1: If kicad_footprint specified, load from KiCad library (has 3D model)',
            '    if kicad_footprint:',
            '        print(f"  Loading KiCad library footprint for {ref}: {kicad_footprint}")',
            '        fp = load_kicad_library_footprint(ref, kicad_footprint, value)',
            '        if fp:',
            '            return fp',
            '        print(f"    WARNING: Could not load {kicad_footprint}, falling back...")',
            '    ',
            '    # PRIORITY 2: If user provided pin positions, create custom footprint',
            '    if pins_data and len(pins_data) > 0:',
            '        print(f"  Creating custom footprint for {ref}: CUSTOM_{footprint_name}")',
            '        fp = create_custom_footprint(ref, value, pins_data, body_size, footprint_name)',
            '        if fp:',
            '            board.Add(fp)',
            '        return fp',
            '    ',
            '    # PRIORITY 3: Try to load from KiCad library using FOOTPRINT_MAP',
            '    if footprint_name in FOOTPRINT_MAP:',
            '        lib_name, fp_name = FOOTPRINT_MAP[footprint_name]',
            '        try:',
            '            fp = pcbnew.FootprintLoad(pcbnew.SETTINGS_MANAGER.GetUserSettingsPath() + "/footprints/" + lib_name + ".pretty", fp_name)',
            '            if fp:',
            '                fp.SetReference(ref)',
            '                if value:',
            '                    fp.SetValue(value)',
            '                board.Add(fp)',
            '                return fp',
            '        except:',
            '            pass',
            '        # Try system library path',
            '        try:',
            '            import os',
            '            kicad_paths = [',
            '                "C:/Program Files/KiCad/8.0/share/kicad/footprints",',
            '                "C:/Program Files/KiCad/7.0/share/kicad/footprints",',
            '                "/usr/share/kicad/footprints",',
            '            ]',
            '            for kicad_path in kicad_paths:',
            '                lib_path = os.path.join(kicad_path, lib_name + ".pretty")',
            '                if os.path.exists(lib_path):',
            '                    fp = pcbnew.FootprintLoad(lib_path, fp_name)',
            '                    if fp:',
            '                        fp.SetReference(ref)',
            '                        if value:',
            '                            fp.SetValue(value)',
            '                        board.Add(fp)',
            '                        return fp',
            '        except:',
            '            pass',
            '    ',
            '    # Create simple footprint based on type',
            '    print(f"  Creating simple footprint for {ref} ({footprint_name})")',
            '    fp = None',
            '    ',
            '    # 2-pin SMD components (resistors, capacitors, LEDs)',
            '    if footprint_name in ["0402", "R_0402"]:',
            '        fp = create_simple_2pin_footprint(ref, value, 0.9, 0.5, 0.5)',
            '    elif footprint_name in ["0603", "R_0603", "LED_0603"]:',
            '        fp = create_simple_2pin_footprint(ref, value, 1.5, 0.8, 0.8)',
            '    elif footprint_name in ["0805", "R_0805", "C0805"]:',
            '        fp = create_simple_2pin_footprint(ref, value, 1.9, 1.0, 1.0)',
            '    elif footprint_name in ["1206", "R_1206"]:',
            '        fp = create_simple_2pin_footprint(ref, value, 3.0, 1.2, 1.2)',
            '    # QFN packages',
            '    elif "QFN-16" in footprint_name or footprint_name == "QFN-16":',
            '        fp = create_qfn_footprint(ref, value, 3.0, 16, 0.5)',
            '    elif "QFN-20" in footprint_name:',
            '        fp = create_qfn_footprint(ref, value, 4.0, 20, 0.5)',
            '    # Default: create a generic 2-pin 0603',
            '    else:',
            '        print(f"    Unknown footprint {footprint_name}, creating generic 0603")',
            '        fp = create_simple_2pin_footprint(ref, value, 1.5, 0.8, 0.8)',
            '    ',
            '    if fp:',
            '        board.Add(fp)',
            '    return fp',
            '',
        ])

        # Board outline - ROOT CAUSE FIX #2: Check if outline exists before creating
        lines.extend([
            '# ' + '=' * 77,
            '# BOARD OUTLINE (with duplicate prevention)',
            '# ' + '=' * 77,
            '',
            '# ROOT CAUSE FIX: Check if board outline already exists',
            'def has_board_outline():',
            '    """Check if board already has an outline on Edge.Cuts"""',
            '    for item in board.GetDrawings():',
            '        if item.GetLayer() == Edge_Cuts:',
            '            return True',
            '    return False',
            '',
            'if has_board_outline():',
            '    print("Board outline already exists - skipping to prevent duplicates")',
            'else:',
            '    print("Drawing board outline...")',
            '    outline_pts = [',
            '        (BX, BY),',
            '        (BX + BW, BY),',
            '        (BX + BW, BY + BH),',
            '        (BX, BY + BH),',
            '    ]',
            '    ',
            '    for i in range(len(outline_pts)):',
            '        start = outline_pts[i]',
            '        end = outline_pts[(i + 1) % len(outline_pts)]',
            '        line = pcbnew.PCB_SHAPE(board)',
            '        line.SetShape(pcbnew.SHAPE_T_SEGMENT)',
            '        line.SetStart(pt(start[0], start[1]))',
            '        line.SetEnd(pt(end[0], end[1]))',
            '        line.SetLayer(Edge_Cuts)',
            '        line.SetWidth(mm(0.15))',
            '        board.Add(line)',
            '    ',
            f'    print(f"  Board outline: {self.board.width}mm x {self.board.height}mm")',
            '',
        ])

        # Create nets
        lines.extend([
            '# ' + '=' * 77,
            '# NET CREATION',
            '# ' + '=' * 77,
            'print("Creating nets...")',
            '',
        ])

        for net_name in nets.keys():
            lines.append(f'get_or_create_net("{net_name}")')

        lines.append(f'print(f"  Created {{len(nets_cache)}} nets")')
        lines.append('')

        # Component placement - CREATE footprints if they don't exist
        lines.extend([
            '# ' + '=' * 77,
            '# COMPONENT PLACEMENT (creates footprints automatically)',
            '# ' + '=' * 77,
            'print("Creating and placing components...")',
            'placement_success = 0',
            'placement_failed = 0',
            '',
        ])

        for ref, pos in self.state.placement.items():
            part = parts.get(ref, {})
            footprint = part.get('footprint', 'Unknown')
            kicad_footprint = part.get('kicad_footprint', None)  # Full KiCad library path
            value = part.get('value', '')
            used_pins = part.get('used_pins', [])
            size = part.get('size', None)

            # Only build pins_data for custom footprints (when kicad_footprint is NOT specified)
            if kicad_footprint:
                # Use KiCad library footprint - has 3D model
                pins_data = []
                pins_data_str = '[]'
            else:
                # Custom footprint - build from pin physical data
                pins_data = []
                for pin in used_pins:
                    # Get pad_size from pin data (user-provided) or use default
                    pad_size = pin.get('pad_size', (0.6, 0.6))
                    pin_data = {
                        'number': pin.get('number', '1'),
                        'offset': pin.get('offset', (0, 0)),
                        'pad_size': pad_size,
                    }
                    pins_data.append(pin_data)
                pins_data_str = repr(pins_data)

            # Format size and kicad_footprint for the script
            size_str = repr(size) if size else 'None'
            kicad_fp_str = f'"{kicad_footprint}"' if kicad_footprint else 'None'

            lines.extend([
                f'# {ref}: {kicad_footprint or footprint}',
                f'pins_data = {pins_data_str}',
                f'body_size = {size_str}',
                f'kicad_fp = {kicad_fp_str}',
                f'fp = load_or_create_footprint("{ref}", "{footprint}", "{value}", pins_data, body_size, kicad_fp)',
                'if fp:',
                f'    fp.SetPosition(pt({pos.x}, {pos.y}))',
                f'    fp.SetOrientationDegrees({pos.rotation})',
                '    placement_success += 1',
            ])

            # Assign nets to pads
            for pin in used_pins:
                pin_num = pin.get('number', '')
                net = pin.get('net', '')
                if net:
                    lines.append(f'    assign_net_to_pad(fp, "{pin_num}", "{net}")')

            lines.extend([
                'else:',
                f'    print(f"  ERROR: Could not create footprint {ref}")',
                '    placement_failed += 1',
                '',
            ])

        lines.append('print(f"  Placed: {placement_success}, Failed: {placement_failed}")')
        lines.append('')

        # Routes (if included) - ONLY create if all footprints found
        if include_routes:
            lines.extend([
                '# ' + '=' * 77,
                '# ROUTING (only if all footprints present)',
                '# ' + '=' * 77,
                '',
                '# Check if all footprints were found before creating routes',
                f'required_footprints = {list(self.state.placement.keys())}',
                'all_found = all(find_footprint(ref) is not None for ref in required_footprints)',
                '',
                'if not all_found:',
                '    print("")',
                '    print("=" * 60)',
                '    print("ROUTES SKIPPED - FOOTPRINTS MISSING")',
                '    print("=" * 60)',
                '    print("To create routes, first add these footprints to your board:")',
                '    for ref in required_footprints:',
                '        if find_footprint(ref) is None:',
                '            print(f"  - {ref}")',
                '    print("")',
                '    print("Then re-run this script.")',
                '    print("=" * 60)',
                'else:',
                '    print("Creating escape routes...")',
            ])

            # Escape routes (indented inside the else block)
            # GND escapes are SKIPPED - GND uses copper pour/zone, not traces
            for ref, comp_escapes in self.state.escapes.items():
                for pin_num, escape in comp_escapes.items():
                    sx, sy = escape.start
                    ex, ey = escape.end
                    net = escape.net
                    if net and net != 'GND':  # Skip GND - uses zone instead
                        lines.append(f'    add_track({sx}, {sy}, {ex}, {ey}, "{net}", {self.rules.min_trace_width}, F_Cu)')

            lines.append('    print(f"  Created escape routes")')
            lines.append('')
            lines.append('    print("Creating signal routes...")')

            # Build escape endpoint lookup for connecting routes to escapes
            # This ensures routes physically connect to escape endpoints (no dangling ends)
            # GND is excluded - uses copper pour/zone, not trace connections
            escape_endpoints = {}  # net_name -> list of escape endpoints
            for ref, comp_escapes in self.state.escapes.items():
                for pin_num, escape in comp_escapes.items():
                    net = escape.net
                    if net and net != 'GND':  # Skip GND - uses zone
                        if net not in escape_endpoints:
                            escape_endpoints[net] = []
                        escape_endpoints[net].append(escape.end)

            # Signal routes (indented inside the else block)
            for net_name, route in self.state.routes.items():
                if not route.success:
                    lines.append(f'    # Net {net_name}: FAILED - {route.error}')
                    continue

                lines.append(f'    # Net: {net_name}')

                # Collect all route segment endpoints with their layers
                route_points = set()
                route_points_by_layer = {'F.Cu': set(), 'B.Cu': set()}
                for seg in route.segments:
                    route_points.add(seg.start)
                    route_points.add(seg.end)
                    route_points_by_layer[seg.layer].add(seg.start)
                    route_points_by_layer[seg.layer].add(seg.end)

                # Collect via positions (these connect both layers)
                via_positions = set()
                for via in route.vias:
                    via_positions.add((round(via.position[0], 2), round(via.position[1], 2)))

                # FIX: Ensure escape endpoints connect to routes
                # Escapes are on F.Cu, so if route is on B.Cu, need via at escape endpoint
                net_escapes = escape_endpoints.get(net_name, [])
                for esc_end in net_escapes:
                    esc_rounded = (round(esc_end[0], 2), round(esc_end[1], 2))

                    # Check if this escape endpoint is already in the route
                    esc_in_route = any(
                        abs(esc_end[0] - rp[0]) < 0.05 and abs(esc_end[1] - rp[1]) < 0.05
                        for rp in route_points
                    )

                    # Check if there's already a via at this position
                    has_via = any(
                        abs(esc_rounded[0] - vp[0]) < 0.05 and abs(esc_rounded[1] - vp[1]) < 0.05
                        for vp in via_positions
                    )

                    if esc_in_route:
                        # Escape endpoint is in route - check if it needs a via
                        # (escape is on F.Cu, if route at this point is B.Cu, need via)
                        in_fcu = any(
                            abs(esc_end[0] - rp[0]) < 0.05 and abs(esc_end[1] - rp[1]) < 0.05
                            for rp in route_points_by_layer['F.Cu']
                        )
                        in_bcu = any(
                            abs(esc_end[0] - rp[0]) < 0.05 and abs(esc_end[1] - rp[1]) < 0.05
                            for rp in route_points_by_layer['B.Cu']
                        )

                        # If route at this point is ONLY on B.Cu (not F.Cu), add via to connect escape
                        if in_bcu and not in_fcu and not has_via:
                            lines.append(f'    add_via({esc_end[0]}, {esc_end[1]}, "{net_name}")  # Connect F.Cu escape to B.Cu route')
                            via_positions.add(esc_rounded)

                    elif route_points:
                        # Escape not in route - find nearest point and connect
                        nearest = min(route_points, key=lambda rp: abs(esc_end[0] - rp[0]) + abs(esc_end[1] - rp[1]))
                        dist = abs(esc_end[0] - nearest[0]) + abs(esc_end[1] - nearest[1])

                        if dist < 5.0 and dist > 0.01:
                            # Check what layer the nearest point is on
                            nearest_on_bcu = any(
                                abs(nearest[0] - rp[0]) < 0.05 and abs(nearest[1] - rp[1]) < 0.05
                                for rp in route_points_by_layer['B.Cu']
                            )
                            nearest_on_fcu = any(
                                abs(nearest[0] - rp[0]) < 0.05 and abs(nearest[1] - rp[1]) < 0.05
                                for rp in route_points_by_layer['F.Cu']
                            )

                            if nearest_on_bcu and not nearest_on_fcu:
                                # Nearest is on B.Cu only - add via at escape, then B.Cu track
                                if not has_via:
                                    lines.append(f'    add_via({esc_end[0]}, {esc_end[1]}, "{net_name}")  # Connect F.Cu escape')
                                lines.append(f'    add_track({esc_end[0]}, {esc_end[1]}, {nearest[0]}, {nearest[1]}, "{net_name}", {self.rules.min_trace_width}, B_Cu)')
                            else:
                                # Nearest is on F.Cu - just add F.Cu track
                                lines.append(f'    add_track({esc_end[0]}, {esc_end[1]}, {nearest[0]}, {nearest[1]}, "{net_name}", {self.rules.min_trace_width}, F_Cu)')

                # Add track segments
                for seg in route.segments:
                    layer = 'F_Cu' if seg.layer == 'F.Cu' else 'B_Cu'
                    lines.append(f'    add_track({seg.start[0]}, {seg.start[1]}, {seg.end[0]}, {seg.end[1]}, "{net_name}", {seg.width}, {layer})')

                # Add vias
                for via in route.vias:
                    lines.append(f'    add_via({via.position[0]}, {via.position[1]}, "{net_name}")')

            lines.append('')
            lines.append('    print(f"  Created {track_count} tracks, {via_count} vias")')
            lines.append('')

        # GND Zone - Always create on B.Cu with vias at GND pad locations
        lines.extend(self._generate_gnd_zone_code())

        # Finalization
        lines.extend([
            '# ' + '=' * 77,
            '# FINALIZATION',
            '# ' + '=' * 77,
            'print("")',
            'print("=" * 60)',
            'print("HUMAN-LIKE PCB GENERATION COMPLETE")',
            'print("=" * 60)',
            '',
            f'print(f"Board size: {self.board.width}mm x {self.board.height}mm")',
            'print(f"Components: {placement_success} placed, {placement_failed} missing")',
            'print(f"Tracks: {track_count}")',
            'print(f"Vias: {via_count}")',
            '',
            'print("")',
            'print("NEXT STEPS:")',
            'print("  1. Press \'B\' to fill GND zones")',
            'print("  2. Run Design Rules Check (DRC)")',
            'print("  3. Review 3D view (Alt+3)")',
            'print("=" * 60)',
            '',
            'try:',
            '    pcbnew.Refresh()',
            'except:',
            '    pass',
        ])

        return '\n'.join(lines)

    def _generate_gnd_zone_code(self) -> List[str]:
        """
        Generate GND zone code with vias at all GND pad locations.

        GND as copper pour on B.Cu is the professional approach:
        - Better EMI shielding
        - Lower impedance ground return path
        - No routing congestion from GND traces
        - Thermal relief connections to pads

        Each GND SMD pad gets a via to connect F.Cu pad to B.Cu zone.
        """
        lines = [
            '# ' + '=' * 77,
            '# GND ZONE (Bottom Layer Copper Pour)',
            '# ' + '=' * 77,
            'print("")',
            'print("Creating GND zone on bottom layer...")',
            '',
        ]

        # Collect all GND pad positions for via placement
        gnd_pads = []
        parts = self.state.parts_db.get('parts', {})
        for ref, pos in self.state.placement.items():
            part = parts.get(ref, {})
            # Note: Exported parts use 'used_pins' not 'pins'
            for pin in part.get('used_pins', []):
                if pin.get('net', '') == 'GND':
                    # Get pad offset
                    offset = pin.get('offset', None)
                    if not offset or offset == (0, 0):
                        physical = pin.get('physical', {})
                        if physical:
                            offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))
                        else:
                            offset = (0, 0)

                    pad_x = pos.x + offset[0]
                    pad_y = pos.y + offset[1]
                    gnd_pads.append((pad_x, pad_y, ref, pin.get('number', '?')))

        if not gnd_pads:
            lines.extend([
                '# No GND pads found - skipping zone',
                'print("  No GND pads in design - skipping zone")',
                '',
            ])
            return lines

        # Add vias at each GND pad location
        lines.extend([
            '# Add vias at GND pad locations to connect F.Cu pads to B.Cu zone',
            f'gnd_via_positions = [',
        ])
        for pad_x, pad_y, ref, pin_num in gnd_pads:
            lines.append(f'    ({pad_x}, {pad_y}),  # {ref}.{pin_num}')
        lines.extend([
            ']',
            '',
            'gnd_via_count = 0',
            'for x, y in gnd_via_positions:',
            '    add_via(x, y, "GND")',
            '    gnd_via_count += 1',
            'print(f"  Added {gnd_via_count} GND vias")',
            '',
        ])

        # Create the zone
        margin = 1.0  # mm from board edge
        lines.extend([
            '# Create GND zone on bottom copper layer',
            'zone = pcbnew.ZONE(board)',
            'zone.SetNet(get_or_create_net("GND"))',
            'zone.SetLayer(B_Cu)',
            'zone.SetIsFilled(False)  # Will be filled when user presses B',
            '',
            '# Zone settings',
            'try:',
            '    # Try KiCad 7+ API',
            '    if hasattr(zone, "GetZoneSettings"):',
            '        zs = zone.GetZoneSettings()',
            f'        zs.m_ZoneClearance = mm({self.rules.min_clearance})',
            f'        zs.m_ZoneMinThickness = mm({self.rules.min_trace_width})',
            '        zs.m_PadConnection = pcbnew.ZONE_CONNECTION_THERMAL',
            f'        zs.m_ThermalReliefGap = mm({self.rules.min_clearance})',
            f'        zs.m_ThermalReliefSpokeWidth = mm({self.rules.min_trace_width})',
            '        zone.SetZoneSettings(zs)',
            '    else:',
            '        # Older KiCad fallback',
            f'        zone.SetLocalClearance(mm({self.rules.min_clearance}))',
            f'        zone.SetMinThickness(mm({self.rules.min_trace_width}))',
            '        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)',
            'except Exception as e:',
            '    print(f"  Zone settings note: {e}")',
            '',
            '# Zone outline (inset from board edge)',
            f'margin = {margin}',
            'try:',
            '    outline = zone.Outline()',
            '    outline.NewOutline()',
            '    outline.Append(mm(BX + margin), mm(BY + margin))',
            '    outline.Append(mm(BX + BW - margin), mm(BY + margin))',
            '    outline.Append(mm(BX + BW - margin), mm(BY + BH - margin))',
            '    outline.Append(mm(BX + margin), mm(BY + BH - margin))',
            '    board.Add(zone)',
            '    print("  GND zone created on B.Cu")',
            '    print("  NOTE: Press B to fill the zone after script completes")',
            'except Exception as e:',
            '    print(f"  WARNING: Could not create zone outline: {e}")',
            '',
        ])

        return lines

    def get_report(self) -> str:
        """Get status report"""
        parts = self.state.parts_db.get('parts', {})
        nets = self.state.parts_db.get('nets', {})

        routed = sum(1 for r in self.state.routes.values() if r.success)
        total_nets = len(self.state.route_order)

        lines = [
            "=" * 60,
            "HUMAN-LIKE PCB ENGINE STATUS REPORT",
            "=" * 60,
            f"Phase: {self.state.phase}",
            f"Status: {self.state.status}",
            "",
            f"Board: {self.board.width}mm x {self.board.height}mm",
            f"Grid: {self.board.grid_size}mm",
            "",
            f"Parts: {len(parts)}",
            f"Nets: {len(nets)}",
            f"Hub: {self.state.hub}",
            "",
            f"Components placed: {len(self.state.placement)}",
            f"Escapes calculated: {sum(len(e) for e in self.state.escapes.values())}",
            f"Nets routed: {routed}/{total_nets}",
        ]

        if self.state.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for w in self.state.warnings:
                lines.append(f"  - {w}")

        if self.state.errors:
            lines.append("")
            lines.append("ERRORS:")
            for e in self.state.errors:
                lines.append(f"  - {e}")

        lines.append("=" * 60)
        return "\n".join(lines)
