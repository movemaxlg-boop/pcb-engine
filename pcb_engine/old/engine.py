"""
PCB Engine - Main Controller
============================

This is the main orchestrator that runs all phases of the algorithm
and generates the final KiCad Python script.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class EnginePhase(Enum):
    """Algorithm phases"""
    PHASE_0_PARTS = 0
    PHASE_1_GRAPH = 1
    PHASE_2_HUB = 2
    PHASE_3_PLACEMENT = 3
    PHASE_4_ESCAPE = 4
    PHASE_5_CORRIDOR = 5
    PHASE_6_ORDER = 6
    PHASE_7_ROUTE = 7
    PHASE_8_VALIDATE = 8
    COMPLETE = 9


class EngineStatus(Enum):
    """Engine status"""
    READY = "ready"
    RUNNING = "running"
    PHASE_COMPLETE = "phase_complete"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class BoardConfig:
    """Board configuration"""
    origin_x: float = 50.0      # mm
    origin_y: float = 50.0      # mm
    width: float = 100.0        # mm
    height: float = 100.0       # mm
    layers: int = 2             # 2 or 4
    grid_size: float = 0.5      # mm


@dataclass
class DesignRules:
    """DRC design rules"""
    min_trace_width: float = 0.5        # mm (user requested minimum 0.5mm)
    min_clearance: float = 0.2          # mm
    min_via_diameter: float = 0.8       # mm
    min_via_drill: float = 0.4          # mm
    min_hole_clearance: float = 0.25    # mm
    min_annular_ring: float = 0.15      # mm
    min_silk_height: float = 0.8        # mm
    min_silk_width: float = 0.15        # mm


@dataclass
class RoutingPreferences:
    """
    User-configurable routing preferences.

    Allows the user to set their priority:
    - "fewer_layers": Minimize use of vias and back layer (default)
    - "smaller_board": Allow more layers/vias to achieve smaller board

    Also sets constraints:
    - max_board_width/height: Maximum board dimensions allowed (mm)
    - max_layers: 1 = F.Cu only, 2 = F.Cu + B.Cu (default)
    """
    # Priority: what matters more to the user?
    priority: str = "fewer_layers"  # "fewer_layers" or "smaller_board"

    # Board size constraints
    max_board_width: float = 100.0   # mm - maximum allowed board width
    max_board_height: float = 100.0  # mm - maximum allowed board height

    # Layer constraints
    max_layers: int = 2  # 1 = F.Cu only (no vias), 2 = F.Cu + B.Cu (allow vias)

    # Auto-expansion settings (for "fewer_layers" priority)
    allow_board_expansion: bool = True   # Allow engine to expand board if routing fails
    expansion_step: float = 5.0          # mm - how much to expand board each iteration

    def validate(self) -> List[str]:
        """Validate preferences, return list of errors"""
        errors = []

        if self.priority not in ["fewer_layers", "smaller_board"]:
            errors.append(f"Invalid priority '{self.priority}'. Must be 'fewer_layers' or 'smaller_board'")

        if self.max_board_width <= 0:
            errors.append(f"max_board_width must be positive, got {self.max_board_width}")

        if self.max_board_height <= 0:
            errors.append(f"max_board_height must be positive, got {self.max_board_height}")

        if self.max_layers not in [1, 2]:
            errors.append(f"max_layers must be 1 or 2, got {self.max_layers}")

        if self.expansion_step <= 0:
            errors.append(f"expansion_step must be positive, got {self.expansion_step}")

        return errors

    def allows_vias(self) -> bool:
        """Check if vias are allowed based on user preferences"""
        return self.max_layers >= 2

    def allows_back_layer(self) -> bool:
        """Check if B.Cu layer is allowed"""
        return self.max_layers >= 2

    def can_expand_board(self, current_width: float, current_height: float) -> bool:
        """Check if board can be expanded within constraints"""
        if not self.allow_board_expansion:
            return False
        return (current_width + self.expansion_step <= self.max_board_width or
                current_height + self.expansion_step <= self.max_board_height)


@dataclass
class EngineState:
    """Current engine state"""
    phase: EnginePhase = EnginePhase.PHASE_0_PARTS
    status: EngineStatus = EngineStatus.READY
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Phase outputs
    parts_db: Dict = field(default_factory=dict)
    graph: Dict = field(default_factory=dict)
    hub: Optional[str] = None
    placement: Dict = field(default_factory=dict)
    escapes: Dict = field(default_factory=dict)
    corridors: List = field(default_factory=list)
    route_order: List = field(default_factory=list)
    routes: Dict = field(default_factory=dict)
    validation: Dict = field(default_factory=dict)


class PCBEngine:
    """
    Main PCB Design Engine

    This engine:
    1. Collects parts and their complete specifications
    2. Builds connectivity graph
    3. Identifies the hub component
    4. Optimizes component placement
    5. Calculates escape vectors
    6. Validates corridor capacity
    7. Determines routing order
    8. Executes routes (simulation)
    9. Validates the design
    10. Generates KiCad Python script
    """

    def __init__(self, board: BoardConfig = None, rules: DesignRules = None):
        self.board = board or BoardConfig()
        self.rules = rules or DesignRules()
        self.state = EngineState()

        # Phase handlers (to be imported from modules)
        self._parts_collector = None
        self._graph_builder = None
        self._placement_optimizer = None
        self._escape_calculator = None
        self._corridor_validator = None
        self._route_optimizer = None
        self._router = None
        self._validator = None
        self._kicad_generator = None

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_board_size(self, width: float, height: float):
        """Set board dimensions"""
        self.board.width = width
        self.board.height = height

    def set_origin(self, x: float, y: float):
        """Set board origin"""
        self.board.origin_x = x
        self.board.origin_y = y

    def set_layers(self, count: int):
        """Set layer count (2 or 4)"""
        if count not in [2, 4]:
            raise ValueError("Layer count must be 2 or 4")
        self.board.layers = count

    def set_grid(self, size: float):
        """Set routing grid size"""
        self.board.grid_size = size

    def set_design_rules(self, rules: Dict):
        """Set design rules from dictionary"""
        for key, value in rules.items():
            if hasattr(self.rules, key):
                setattr(self.rules, key, value)

    # =========================================================================
    # PHASE 0: PARTS
    # =========================================================================

    def add_part(self, part_data: Dict) -> bool:
        """Add a part to the database"""
        if self._parts_collector is None:
            from .parts import PartsCollector
            self._parts_collector = PartsCollector()

        return self._parts_collector.add_part_manual(part_data)

    def load_parts_from_file(self, filepath: str) -> bool:
        """Load parts from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if 'parts' in data:
                for part in data['parts']:
                    self.add_part(part)
            return True
        except Exception as e:
            self.state.errors.append(f"Failed to load parts: {e}")
            return False

    def load_parts_from_dict(self, parts_dict: Dict) -> bool:
        """Load parts from dictionary"""
        for ref, part_data in parts_dict.items():
            part_data['ref_designator'] = ref
            if not self.add_part(part_data):
                return False
        return True

    # =========================================================================
    # RUN PHASES
    # =========================================================================

    def run_phase(self, phase: EnginePhase) -> bool:
        """Run a single phase"""
        self.state.status = EngineStatus.RUNNING

        handlers = {
            EnginePhase.PHASE_0_PARTS: self._run_phase_0,
            EnginePhase.PHASE_1_GRAPH: self._run_phase_1,
            EnginePhase.PHASE_2_HUB: self._run_phase_2,
            EnginePhase.PHASE_3_PLACEMENT: self._run_phase_3,
            EnginePhase.PHASE_4_ESCAPE: self._run_phase_4,
            EnginePhase.PHASE_5_CORRIDOR: self._run_phase_5,
            EnginePhase.PHASE_6_ORDER: self._run_phase_6,
            EnginePhase.PHASE_7_ROUTE: self._run_phase_7,
            EnginePhase.PHASE_8_VALIDATE: self._run_phase_8,
        }

        if phase not in handlers:
            self.state.errors.append(f"Unknown phase: {phase}")
            self.state.status = EngineStatus.ERROR
            return False

        try:
            result = handlers[phase]()
            if result:
                self.state.phase = EnginePhase(phase.value + 1)
                self.state.status = EngineStatus.PHASE_COMPLETE
            else:
                self.state.status = EngineStatus.ERROR
            return result
        except Exception as e:
            self.state.errors.append(f"Phase {phase.name} failed: {e}")
            self.state.status = EngineStatus.ERROR
            return False

    def run(self) -> bool:
        """Run all phases"""
        for phase in EnginePhase:
            if phase == EnginePhase.COMPLETE:
                break
            if not self.run_phase(phase):
                return False

        self.state.status = EngineStatus.COMPLETE
        return True

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    def _run_phase_0(self) -> bool:
        """Phase 0: Validate parts database"""
        if self._parts_collector is None:
            self.state.errors.append("No parts loaded")
            return False

        if not self._parts_collector.validate_all():
            self.state.errors.extend(self._parts_collector.errors)
            return False

        self.state.parts_db = self._parts_collector.export()
        return True

    def _run_phase_1(self) -> bool:
        """Phase 1: Build connectivity graph"""
        from .graph import GraphBuilder

        self._graph_builder = GraphBuilder()
        self.state.graph = self._graph_builder.build(self.state.parts_db)
        return True

    def _run_phase_2(self) -> bool:
        """Phase 2: Identify hub"""
        self.state.hub = self._graph_builder.identify_hub(self.state.graph)

        if self.state.hub is None:
            self.state.warnings.append("No clear hub identified")

        return True

    def _run_phase_3(self) -> bool:
        """Phase 3: Optimize placement"""
        from .placement import PlacementOptimizer

        self._placement_optimizer = PlacementOptimizer(self.board, self.rules)
        self.state.placement = self._placement_optimizer.optimize(
            self.state.parts_db,
            self.state.graph,
            self.state.hub
        )

        if not self._placement_optimizer.validate():
            self.state.errors.extend(self._placement_optimizer.errors)
            return False

        return True

    def _run_phase_4(self) -> bool:
        """Phase 4: Calculate escapes"""
        from .escape import EscapeCalculator

        self._escape_calculator = EscapeCalculator(self.board, self.rules)
        self.state.escapes = self._escape_calculator.calculate(
            self.state.parts_db,
            self.state.placement,
            self.state.graph
        )
        return True

    def _run_phase_5(self) -> bool:
        """Phase 5: Validate corridor capacity"""
        from .corridor import CorridorValidator

        self._corridor_validator = CorridorValidator(self.board, self.rules)
        result = self._corridor_validator.validate(
            self.state.placement,
            self.state.escapes,
            self.state.graph,
            parts_db=self.state.parts_db  # Pass parts_db for net analysis
        )

        self.state.corridors = result['corridors']

        if not result['valid']:
            self.state.errors.append("Corridor capacity exceeded - adjust placement")
            self.state.errors.extend(result['violations'])
            return False

        return True

    def _run_phase_6(self) -> bool:
        """Phase 6: Determine routing order"""
        from .routing import RouteOrderOptimizer

        self._route_optimizer = RouteOrderOptimizer()
        self.state.route_order = self._route_optimizer.calculate(
            self.state.parts_db['nets'],
            self.state.placement,
            self.state.escapes,
            self.state.corridors
        )
        return True

    def _run_phase_7(self) -> bool:
        """Phase 7: Execute routes (simulation)"""
        from .routing import Router

        self._router = Router(self.board, self.rules)
        result = self._router.route_all(
            self.state.route_order,
            self.state.parts_db,
            self.state.placement,
            self.state.escapes
        )

        self.state.routes = result['routes']

        if result['failed']:
            # Store failures as warnings, but continue if we have some routes
            success_rate = result.get('success_rate', 0)
            failed_count = len(result['failed'])
            total_count = len(self.state.route_order)

            self.state.warnings.append(
                f"Routing: {len(self.state.routes)}/{total_count} nets routed ({success_rate*100:.0f}% success)"
            )
            self.state.warnings.extend([f"Unrouted: {net}" for net in result['failed']])

            # Continue if at least some routes succeeded
            # (A real designer would manually complete the remaining routes)
            if len(self.state.routes) > 0:
                return True

            self.state.errors.append("Routing failed completely - no nets routed")
            return False

        return True

    def _run_phase_8(self, strict: bool = False) -> bool:
        """Phase 8: Validate design

        Args:
            strict: If True, any DRC error fails the phase.
                   If False, warnings are reported but phase passes.
        """
        from .validation import ValidationGate

        self._validator = ValidationGate(self.rules)
        result = self._validator.validate(
            self.state.routes,
            self.state.placement,
            self.state.parts_db,
            self.state.escapes  # Pass escapes for via connectivity check
        )

        self.state.validation = result

        # Report all violations
        if result.get('violations'):
            # Separate errors from warnings
            errors = [v for v in result['violations'] if '[ERROR]' in v]
            warnings = [v for v in result['violations'] if '[WARN]' in v]

            self.state.errors.extend(errors)
            self.state.warnings.extend(warnings)

            # In strict mode, any error fails validation
            # In non-strict mode, we report but continue
            if strict and errors:
                return False

        # For non-strict mode, always pass validation
        # (KiCad script can still be generated for review)
        return True

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def generate_kicad_script(self, output_path: str, include_routes: bool = False) -> bool:
        """
        Generate KiCad Python script.

        Args:
            output_path: Path to write the script
            include_routes: If True, include escape and signal routes in output.
                           If False (default), skip routes to avoid DRC warnings
                           when footprints aren't placed in KiCad.
                           Set to True when you have footprints placed in your KiCad project.
        """
        # Allow generation if we've reached Phase 8 (validation) or beyond
        if self.state.phase.value < EnginePhase.PHASE_8_VALIDATE.value:
            self.state.errors.append(
                f"Engine must reach Phase 8 before generating script (current: {self.state.phase.name})"
            )
            return False

        from .kicad import KiCadGenerator

        self._kicad_generator = KiCadGenerator(self.board, self.rules)

        script = self._kicad_generator.generate(
            parts=self.state.parts_db,
            placement=self.state.placement,
            routes=self.state.routes,
            escapes=self.state.escapes,
            include_routes=include_routes
        )

        with open(output_path, 'w') as f:
            f.write(script)

        return True

    def get_report(self) -> str:
        """Get status report"""
        lines = [
            "=" * 60,
            "PCB ENGINE STATUS REPORT",
            "=" * 60,
            f"Phase: {self.state.phase.name}",
            f"Status: {self.state.status.value}",
            "",
            f"Board: {self.board.width}mm x {self.board.height}mm",
            f"Layers: {self.board.layers}",
            f"Grid: {self.board.grid_size}mm",
            "",
        ]

        if self.state.parts_db:
            summary = self.state.parts_db.get('summary', {})
            lines.extend([
                f"Parts: {summary.get('total_parts', 0)}",
                f"Nets: {summary.get('total_nets', 0)}",
            ])

        if self.state.hub:
            lines.append(f"Hub: {self.state.hub}")

        if self.state.placement:
            lines.append(f"Components placed: {len(self.state.placement)}")

        if self.state.routes:
            lines.append(f"Nets routed: {len(self.state.routes)}")

        if self.state.errors:
            lines.extend([
                "",
                "ERRORS:",
            ])
            for e in self.state.errors:
                lines.append(f"  - {e}")

        if self.state.warnings:
            lines.extend([
                "",
                "WARNINGS:",
            ])
            for w in self.state.warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 60)
        return "\n".join(lines)
