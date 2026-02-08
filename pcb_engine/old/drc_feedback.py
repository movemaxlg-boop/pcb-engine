"""
PCB Engine - DRC Feedback Loop
==============================

This module implements closed-loop DRC correction using the BUILT-IN
ValidationGate system (not external KiCad DRC):

1. Generate PCB layout
2. Run INTERNAL DRC check (ValidationGate)
3. Parse violations
4. Feed violations back to engine for correction
5. Repeat until DRC passes or max iterations

This is how professional PCB tools achieve DRC-clean designs automatically.

The internal DRC checks include:
- Connectivity (all nets properly connected)
- Clearance (trace-to-trace, trace-to-pad, via-to-via)
- Trace width compliance
- Via specifications
- Dangling traces
- Track overlaps/shorts
- Component overlaps
- Manufacturing checks (acid traps)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import os

# Import the built-in validation system
from .validation import ValidationGate, ViolationType, ViolationSeverity, Violation


@dataclass
class DRCViolation:
    """Represents a single DRC violation"""
    type: str              # 'clearance', 'short', 'unconnected', 'edge', etc.
    severity: str          # 'error', 'warning'
    description: str       # Full violation text
    net1: Optional[str] = None  # First net involved
    net2: Optional[str] = None  # Second net (for shorts/clearance)
    position: Optional[Tuple[float, float]] = None  # Location of violation
    actual_value: Optional[float] = None    # e.g., actual clearance
    required_value: Optional[float] = None  # e.g., required clearance


class DRCParser:
    """
    Parses KiCad DRC reports to extract violations.

    KiCad DRC report format:
    ```
    ** Found 5 DRC violations **
    [clearance_violation]: Clearance violation (0.125mm actual vs 0.2mm required)
        Local override; clearance: 0.2mm
        @(123.5mm, 145.0mm): Track [NET1] on F.Cu
        @(123.5mm, 145.5mm): Track [NET2] on F.Cu
    ```
    """

    # Violation type patterns
    PATTERNS = {
        'clearance': re.compile(r'\[clearance[^\]]*\].*?(\d+\.?\d*)mm actual.*?(\d+\.?\d*)mm required', re.IGNORECASE),
        'short': re.compile(r'\[shorting_items\]|items shorting', re.IGNORECASE),
        'unconnected': re.compile(r'unconnected|ratsnest', re.IGNORECASE),
        'edge': re.compile(r'board.?edge|edge.?clearance', re.IGNORECASE),
        'track_dangling': re.compile(r'track.?dangling', re.IGNORECASE),
        'via_dangling': re.compile(r'via.?dangling', re.IGNORECASE),
        'duplicate': re.compile(r'duplicate|holes.?co.?located', re.IGNORECASE),
        'crossing': re.compile(r'crossing|tracks.?cross', re.IGNORECASE),
    }

    # Net extraction pattern
    NET_PATTERN = re.compile(r'\[([^\]]+)\]')

    # Position extraction pattern
    POS_PATTERN = re.compile(r'@?\(?(\d+\.?\d*)mm?,?\s*(\d+\.?\d*)mm?\)?')

    def parse_report(self, report_text: str) -> List[DRCViolation]:
        """
        Parse DRC report text and extract violations.

        Args:
            report_text: Full DRC report content

        Returns:
            List of DRCViolation objects
        """
        violations = []

        # Split into violation blocks
        # Each violation typically starts with a bracket [violation_type]
        blocks = re.split(r'\n(?=\s*\[)', report_text)

        for block in blocks:
            if not block.strip():
                continue

            violation = self._parse_violation_block(block)
            if violation:
                violations.append(violation)

        return violations

    def _parse_violation_block(self, block: str) -> Optional[DRCViolation]:
        """Parse a single violation block"""
        # Determine violation type
        vtype = 'unknown'
        for name, pattern in self.PATTERNS.items():
            if pattern.search(block):
                vtype = name
                break

        # Skip warnings about dangling tracks (usually escape endpoints)
        if vtype == 'track_dangling':
            return None

        # Determine severity
        severity = 'error'
        if 'warning' in block.lower():
            severity = 'warning'

        # Extract nets
        nets = self.NET_PATTERN.findall(block)
        net1 = nets[0] if len(nets) > 0 else None
        net2 = nets[1] if len(nets) > 1 else None

        # Extract position
        pos_match = self.POS_PATTERN.search(block)
        position = None
        if pos_match:
            try:
                position = (float(pos_match.group(1)), float(pos_match.group(2)))
            except:
                pass

        # Extract actual/required values for clearance violations
        actual = None
        required = None
        if vtype == 'clearance':
            match = self.PATTERNS['clearance'].search(block)
            if match:
                actual = float(match.group(1))
                required = float(match.group(2))

        return DRCViolation(
            type=vtype,
            severity=severity,
            description=block.strip()[:200],  # Truncate long descriptions
            net1=net1,
            net2=net2,
            position=position,
            actual_value=actual,
            required_value=required,
        )

    def parse_file(self, file_path: str) -> List[DRCViolation]:
        """Parse DRC report from file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return self.parse_report(f.read())


class DRCFeedback:
    """
    Closed-loop DRC correction system using BUILT-IN ValidationGate.

    Workflow:
    1. Generate initial layout
    2. Run INTERNAL DRC check (ValidationGate)
    3. Convert violations to fix instructions
    4. Apply fixes and regenerate affected routes
    5. Repeat until DRC passes or max iterations

    This uses the engine's own validation system, NOT external KiCad DRC.
    """

    def __init__(self, engine, max_iterations: int = 5):
        """
        Args:
            engine: HumanPCBEngine instance
            max_iterations: Maximum correction attempts
        """
        self.engine = engine
        self.max_iterations = max_iterations

        # Built-in validation gate
        self.validator = ValidationGate(engine.rules)

        # Also keep the parser for external DRC reports (optional)
        self.parser = DRCParser()

        # Track violation history
        self.history: List[List[DRCViolation]] = []

    def generate_fix_instructions(self, violations: List[DRCViolation]) -> Dict:
        """
        Analyze violations and generate fix instructions for the engine.

        Returns:
            {
                'reroute_nets': ['NET1', 'NET2'],  # Nets to reroute
                'increase_clearance': True/False,  # Need wider spacing
                'use_more_vias': True/False,       # Route on different layers
                'move_inward': True/False,         # Edge clearance issue
                'affected_positions': [(x,y), ...], # Where problems are
            }
        """
        instructions = {
            'reroute_nets': set(),
            'increase_clearance': False,
            'use_more_vias': False,
            'move_inward': False,
            'affected_positions': [],
        }

        for v in violations:
            # Skip warnings
            if v.severity == 'warning':
                continue

            # Clearance violations -> need wider spacing or different layer
            if v.type == 'clearance':
                if v.net1:
                    instructions['reroute_nets'].add(v.net1)
                if v.net2:
                    instructions['reroute_nets'].add(v.net2)
                instructions['increase_clearance'] = True
                instructions['use_more_vias'] = True  # Try other layer

            # Shorts -> definitely reroute both nets
            elif v.type == 'short':
                if v.net1:
                    instructions['reroute_nets'].add(v.net1)
                if v.net2:
                    instructions['reroute_nets'].add(v.net2)
                instructions['use_more_vias'] = True

            # Edge clearance -> move tracks inward
            elif v.type == 'edge':
                instructions['move_inward'] = True
                if v.net1:
                    instructions['reroute_nets'].add(v.net1)

            # Unconnected -> need to route this net
            elif v.type == 'unconnected':
                if v.net1:
                    instructions['reroute_nets'].add(v.net1)

            # Crossing tracks -> reroute one on different layer
            elif v.type == 'crossing':
                if v.net1:
                    instructions['reroute_nets'].add(v.net1)
                if v.net2:
                    instructions['reroute_nets'].add(v.net2)
                instructions['use_more_vias'] = True

            # Track position for targeted fixes
            if v.position:
                instructions['affected_positions'].append(v.position)

        # Convert set to list for JSON serialization
        # CRITICAL FIX: Filter out:
        # - NC pad markers (they're not routable nets)
        # - GND (uses copper pour/zone, not traces)
        instructions['reroute_nets'] = [
            n for n in instructions['reroute_nets']
            if not n.startswith('__NC__') and n != 'GND'
        ]

        return instructions

    def run_feedback_loop(self, kicad_script_path: str = None,
                          use_external_drc: bool = False,
                          drc_report_path: str = None) -> Dict:
        """
        Run the complete feedback loop using BUILT-IN ValidationGate.

        Args:
            kicad_script_path: Path to output KiCad script (optional)
            use_external_drc: If True, use external KiCad DRC report instead
            drc_report_path: Path to external DRC report (only if use_external_drc=True)

        Returns:
            {
                'success': bool,
                'iterations': int,
                'final_violations': List[DRCViolation],
                'history': List[List[DRCViolation]],
            }
        """
        iteration = 0
        current_violations = []
        instructions = {}

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"DRC FEEDBACK LOOP - Iteration {iteration}/{self.max_iterations}")
            print('='*60)

            # Step 1: Generate/regenerate layout
            if iteration == 1:
                # First iteration - generate initial layout
                print("\n[1] Generating initial layout...")
                if not self.engine.run():
                    print("  ERROR: Engine failed to generate layout")
                    print("  REASON: Board may be too small or components too close")

                    # Create pseudo-violations for the physical solutions report
                    failed_nets = [n for n, r in self.engine.state.routes.items() if not r.success]
                    pseudo_violations = [
                        DRCViolation(
                            type='unconnected',
                            severity='error',
                            description=f'Net {net} could not be routed - insufficient board space',
                            net1=net,
                        )
                        for net in failed_nets
                    ]

                    # Still generate physical solutions report
                    suggestions = self._generate_physical_solutions_report(pseudo_violations)

                    print("\n" + "=" * 70)
                    print("PHYSICAL SOLUTIONS - INITIAL LAYOUT FAILED:")
                    print("=" * 70)

                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"\n{i}. {suggestion['title']}")
                        print(f"   Why: {suggestion['reason']}")
                        print(f"   How: {suggestion['action']}")
                        if suggestion.get('impact'):
                            print(f"   Impact: {suggestion['impact']}")

                    print("\n" + "=" * 70)

                    return {
                        'success': False,
                        'iterations': iteration,
                        'final_violations': pseudo_violations,
                        'history': self.history,
                        'physical_solutions': suggestions
                    }
            else:
                # Subsequent iterations - apply fixes and regenerate
                print(f"\n[1] Applying fixes and regenerating...")
                self._apply_fixes(instructions)

            # Step 2: Export KiCad script (if path provided)
            if kicad_script_path:
                print("\n[2] Exporting KiCad script...")
                self.engine.generate_kicad_script(kicad_script_path)
            else:
                print("\n[2] (Skipping KiCad export)")

            # Step 3: Run DRC check using BUILT-IN ValidationGate
            print("\n[3] Running INTERNAL DRC check (ValidationGate)...")
            if use_external_drc and drc_report_path and os.path.exists(drc_report_path):
                print("  (Using external KiCad DRC report)")
                current_violations = self.parser.parse_file(drc_report_path)
            else:
                print("  Checking: connectivity, clearance, trace width, vias, overlaps...")
                current_violations = self._internal_drc_check()

            self.history.append(current_violations)

            # Step 4: Check if we're done
            errors = [v for v in current_violations if v.severity == 'error']
            warnings = [v for v in current_violations if v.severity == 'warning']
            print(f"\n[4] DRC Result: {len(errors)} errors, {len(warnings)} warnings")

            # Show error summary
            if errors:
                error_types = {}
                for e in errors:
                    error_types[e.type] = error_types.get(e.type, 0) + 1
                print(f"    Error breakdown: {error_types}")

            if len(errors) == 0:
                print("\n[OK] DRC PASSED - No errors!")
                return {'success': True, 'iterations': iteration,
                        'final_violations': current_violations, 'history': self.history}

            # Step 5: Generate fix instructions
            print("\n[5] Analyzing violations and generating fix instructions...")
            instructions = self.generate_fix_instructions(errors)
            print(f"    Nets to reroute: {instructions['reroute_nets']}")
            print(f"    Increase clearance: {instructions['increase_clearance']}")
            print(f"    Use more vias: {instructions['use_more_vias']}")
            print(f"    Move inward: {instructions['move_inward']}")

            # Check if we're making progress
            if iteration > 1:
                prev_errors = len([v for v in self.history[-2] if v.severity == 'error'])
                if len(errors) >= prev_errors:
                    print(f"\n    WARNING: No improvement ({prev_errors} -> {len(errors)} errors)")

                    # If no improvement after 2 iterations, try aggressive fixes
                    if iteration > 2:
                        prev_prev_errors = len([v for v in self.history[-3] if v.severity == 'error'])
                        if len(errors) >= prev_prev_errors:
                            print("    Trying more aggressive fixes...")
                            instructions['use_more_vias'] = True
                            instructions['full_reroute'] = True

                    # If STILL no improvement after 4 iterations, the design may need
                    # board size increase or manual intervention
                    if iteration >= 4:
                        unconnected = [v for v in errors if v.type == 'unconnected']
                        if unconnected:
                            print(f"\n    CRITICAL: {len(unconnected)} nets cannot be routed")
                            print("    Possible causes:")
                            print("      - Board too small")
                            print("      - Components too close")
                            print("      - Need more routing layers")

        print(f"\n[FAILED] DRC FAILED after {iteration} iterations")

        # =====================================================================
        # PHYSICAL SOLUTIONS REPORT
        # When all software solutions fail, suggest hardware/physical changes
        # =====================================================================
        suggestions = self._generate_physical_solutions_report(current_violations)

        print("\n" + "=" * 70)
        print("PHYSICAL SOLUTIONS - SOFTWARE EXHAUSTED, CONSIDER THESE CHANGES:")
        print("=" * 70)

        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['title']}")
            print(f"   Why: {suggestion['reason']}")
            print(f"   How: {suggestion['action']}")
            if suggestion.get('impact'):
                print(f"   Impact: {suggestion['impact']}")

        print("\n" + "=" * 70)

        return {
            'success': False,
            'iterations': iteration,
            'final_violations': current_violations,
            'history': self.history,
            'physical_solutions': suggestions  # Include in result for programmatic access
        }

    def _apply_fixes(self, instructions: Dict):
        """
        Apply fix instructions to the engine.

        SMART STRATEGIES (in order of aggressiveness):
        1. Iteration 1: Selective reroute - remove only failed routes
        2. Iteration 2: Full reroute - clear all, try different net order
        3. Iteration 3: Force layer change - route conflicting nets on B.Cu
        4. Iteration 4: Targeted layer assignment - assign specific nets to layers
        5. Iteration 5: Adjust placement slightly (if supported)
        """
        iteration = len(self.history)

        # Count error types to decide strategy
        current_errors = self.history[-1] if self.history else []
        dangling_count = sum(1 for v in current_errors if v.type == 'track_dangling')
        clearance_count = sum(1 for v in current_errors if v.type == 'clearance')
        unconnected_count = sum(1 for v in current_errors if v.type == 'unconnected')

        # Get nets involved in clearance violations
        clearance_nets = set()
        for v in current_errors:
            if v.type == 'clearance':
                if v.net1:
                    clearance_nets.add(v.net1)
                if v.net2:
                    clearance_nets.add(v.net2)

        # =====================================================================
        # ITERATION 1: Selective reroute
        # =====================================================================
        if iteration == 1:
            nets_to_reroute = instructions.get('reroute_nets', [])
            if nets_to_reroute:
                routes_to_remove = []
                for net in nets_to_reroute:
                    if net in self.engine.state.routes:
                        route = self.engine.state.routes[net]
                        if not route.success or net in clearance_nets:
                            routes_to_remove.append(net)

                if routes_to_remove:
                    print(f"    Removing failed/violated routes: {routes_to_remove}")
                    for net in routes_to_remove:
                        if net in self.engine.state.routes:
                            del self.engine.state.routes[net]

                self.engine._phase_5_routing()
            return

        # =====================================================================
        # ITERATION 2: Full reroute with different net ordering
        # =====================================================================
        if iteration == 2:
            print("    Strategy: Full reroute (clearing all routes)")
            self.engine.state.routes = {}
            self.engine._phase_5_routing()
            return

        # =====================================================================
        # ITERATION 3+: SMART - Force conflicting nets to use different layers
        # This is the key insight: if two nets have a clearance violation,
        # route ONE of them on B.Cu instead of F.Cu
        # =====================================================================
        if iteration >= 3 and clearance_nets:
            print(f"    Strategy: Force layer separation for conflicting nets")
            print(f"    Conflicting nets: {list(clearance_nets)}")

            # Pick one net to force to B.Cu (pick the shorter one)
            net_to_force = self._pick_net_for_layer_change(clearance_nets)
            if net_to_force:
                print(f"    Forcing '{net_to_force}' to route on B.Cu")

                # Store preference for this net
                if not hasattr(self.engine, '_layer_preferences'):
                    self.engine._layer_preferences = {}
                self.engine._layer_preferences[net_to_force] = 'B.Cu'

            # Clear routes and reroute with layer preferences
            self.engine.state.routes = {}
            self._reroute_with_layer_preferences()
            return

        # =====================================================================
        # ITERATION 4+: More aggressive layer assignment
        # =====================================================================
        if iteration >= 4:
            print("    Strategy: Aggressive layer assignment")

            # Assign alternating layers to all signal nets
            signal_nets = [n for n in self.engine.state.parts_db.get('nets', {}).keys()
                          if n not in ('GND', '3V3', 'VCC', '5V')]

            if not hasattr(self.engine, '_layer_preferences'):
                self.engine._layer_preferences = {}

            # Alternate: even index -> F.Cu, odd index -> B.Cu
            for i, net in enumerate(sorted(signal_nets)):
                if net in clearance_nets or i % 2 == 1:
                    self.engine._layer_preferences[net] = 'B.Cu'
                    print(f"      {net} -> B.Cu")
                else:
                    self.engine._layer_preferences[net] = 'F.Cu'

            self.engine.state.routes = {}
            self._reroute_with_layer_preferences()
            return

    def _pick_net_for_layer_change(self, nets: set) -> str:
        """Pick the best net to move to a different layer"""
        if not nets:
            return None

        # Prefer shorter nets (easier to reroute)
        # Also prefer nets that aren't power nets
        power_nets = {'GND', '3V3', 'VCC', '5V', 'VBUS'}

        candidates = [n for n in nets if n not in power_nets]
        if not candidates:
            candidates = list(nets)

        # Pick the first one (could be smarter - pick shortest)
        return candidates[0] if candidates else None

    def _generate_physical_solutions_report(self, violations: List[DRCViolation]) -> List[Dict]:
        """
        Generate physical/hardware solutions when software routing fails.

        Analyzes the types of violations and suggests actionable physical changes:
        - Board size increase
        - More PCB layers
        - Component changes (different packages)
        - Design rule relaxation
        - Component repositioning
        """
        suggestions = []

        # Categorize violations
        clearance_errors = [v for v in violations if v.type == 'clearance' and v.severity == 'error']
        unconnected_errors = [v for v in violations if v.type == 'unconnected' and v.severity == 'error']
        dangling_errors = [v for v in violations if v.type == 'track_dangling' and v.severity == 'error']
        short_errors = [v for v in violations if v.type == 'short' and v.severity == 'error']

        # Get current board info
        current_width = self.engine.board.width
        current_height = self.engine.board.height
        current_layers = self.engine.board.layers

        # Count failed nets
        failed_nets = [n for n, r in self.engine.state.routes.items() if not r.success]
        total_nets = len(self.engine.state.routes)

        # ==================================================================
        # SUGGESTION 1: INCREASE BOARD SIZE
        # ==================================================================
        if clearance_errors or unconnected_errors or failed_nets:
            # Calculate recommended size increase
            recommended_increase = 10.0  # Default 10mm
            if len(clearance_errors) > 3 or len(failed_nets) > 1:
                recommended_increase = 20.0  # More issues = bigger increase

            new_width = current_width + recommended_increase
            new_height = current_height + recommended_increase

            suggestions.append({
                'title': 'INCREASE BOARD SIZE',
                'reason': f'{len(clearance_errors)} clearance violations + {len(failed_nets)} unrouted nets suggest insufficient routing space',
                'action': f'Current: {current_width}x{current_height}mm -> Recommended: {new_width}x{new_height}mm',
                'impact': 'More routing channels, easier placement, better signal integrity',
                'priority': 1,
            })

        # ==================================================================
        # SUGGESTION 2: ADD MORE LAYERS
        # ==================================================================
        if current_layers < 4 and (len(failed_nets) > 0 or len(clearance_errors) > 2):
            layer_recommendation = 4 if current_layers == 2 else 6

            suggestions.append({
                'title': 'ADD PCB LAYERS',
                'reason': f'Current {current_layers}-layer board cannot route all {total_nets} nets without conflicts',
                'action': f'Upgrade from {current_layers}-layer to {layer_recommendation}-layer PCB',
                'impact': f'Inner layers allow separated routing - power on inner planes, signals on outer',
                'priority': 2,
            })

        # ==================================================================
        # SUGGESTION 3: USE SMALLER COMPONENT PACKAGES
        # ==================================================================
        if clearance_errors:
            # Find components involved in clearance issues
            involved_components = set()
            for v in clearance_errors:
                if v.net1:
                    # Extract component ref from net info if available
                    involved_components.add(v.net1.split('.')[0] if '.' in v.net1 else v.net1)

            suggestions.append({
                'title': 'USE SMALLER COMPONENT PACKAGES',
                'reason': 'Large component footprints reduce available routing space',
                'action': 'Consider smaller packages: 0805->0603, 0603->0402, SOT-223->SOT-23',
                'impact': 'More space between pads, easier routing, smaller board possible',
                'priority': 3,
            })

        # ==================================================================
        # SUGGESTION 4: RELAX DESIGN RULES (if possible)
        # ==================================================================
        if clearance_errors:
            current_clearance = getattr(self.engine.rules, 'min_clearance', 0.2)
            current_trace = getattr(self.engine.rules, 'min_trace_width', 0.5)

            suggestions.append({
                'title': 'RELAX DESIGN RULES (check manufacturer capability)',
                'reason': f'Current clearance: {current_clearance}mm, trace: {current_trace}mm may be too restrictive',
                'action': f'If your manufacturer supports it, try clearance: {current_clearance - 0.05}mm, trace: {max(0.15, current_trace - 0.1)}mm',
                'impact': 'More routing flexibility, but verify with PCB manufacturer first',
                'priority': 4,
            })

        # ==================================================================
        # SUGGESTION 5: REPOSITION COMPONENTS
        # ==================================================================
        if failed_nets or unconnected_errors:
            suggestions.append({
                'title': 'REPOSITION COMPONENTS',
                'reason': f'{len(failed_nets)} nets failed routing - components may be blocking paths',
                'action': 'Move highly-connected components (hub) to center, spread peripheral components',
                'impact': 'Better routing flow, shorter traces, fewer crossings',
                'priority': 5,
            })

        # ==================================================================
        # SUGGESTION 6: SPLIT POWER/GROUND PLANES
        # ==================================================================
        gnd_issues = [v for v in violations if v.net1 == 'GND' or v.net2 == 'GND']
        if gnd_issues and current_layers >= 2:
            suggestions.append({
                'title': 'USE DEDICATED POWER/GROUND PLANES',
                'reason': f'{len(gnd_issues)} issues involve GND - routing GND as traces is problematic',
                'action': 'In 4-layer design: use entire inner layers as GND and VCC planes',
                'impact': 'GND doesn\'t need routing, better EMI performance, cleaner design',
                'priority': 2 if len(gnd_issues) > 2 else 6,
            })

        # ==================================================================
        # SUGGESTION 7: REDUCE NUMBER OF COMPONENTS
        # ==================================================================
        parts_count = len(self.engine.state.parts_db.get('parts', {}))
        if parts_count > 10 and (len(failed_nets) > 2 or len(clearance_errors) > 5):
            suggestions.append({
                'title': 'SIMPLIFY DESIGN - REDUCE COMPONENTS',
                'reason': f'{parts_count} components may be too dense for current board size',
                'action': 'Consider integrated modules, multi-function ICs, or eliminating non-essential parts',
                'impact': 'Fewer connections = easier routing, smaller board possible',
                'priority': 7,
            })

        # Sort by priority
        suggestions.sort(key=lambda x: x.get('priority', 99))

        return suggestions

    def _reroute_with_layer_preferences(self):
        """
        Reroute with layer preferences applied.

        This modifies the routing to prefer specific layers for specific nets.
        """
        # Get the routing function
        from .human_routing import human_like_routing

        parts_db = self.engine.state.parts_db
        escapes = self.engine.state.escapes
        placement = self.engine.state.placement

        # Build route order (excluding GND which may use a pour)
        nets = parts_db.get('nets', {})
        route_order = [n for n in nets.keys() if n != 'GND']

        # Get layer preferences
        layer_prefs = getattr(self.engine, '_layer_preferences', {})

        # Custom routing that respects layer preferences
        routes = self._route_with_preferences(route_order, parts_db, escapes,
                                               placement, layer_prefs)

        self.engine.state.routes = routes

        # Count results
        routed = sum(1 for r in routes.values() if r.success)
        total = len(route_order)
        print(f"  Routed {routed}/{total} nets (with layer preferences)")

    def _route_with_preferences(self, route_order, parts_db, escapes,
                                 placement, layer_prefs):
        """
        Route nets with layer preferences.

        For nets with a preference, start routing on that layer first.
        Uses the standard routing function but with layer hints.
        """
        from .human_routing import HumanLikeRouter, human_like_routing

        nets = parts_db.get('nets', {})

        # Build net_pins
        net_pins = {}
        for net_name, net_info in nets.items():
            pins = net_info.get('pins', [])
            net_pins[net_name] = pins

        # Get design rules
        trace_width = getattr(self.engine.rules, 'min_trace_width', 0.5)
        clearance = getattr(self.engine.rules, 'min_clearance', 0.2)

        # Create router
        router = HumanLikeRouter(
            board_width=self.engine.board.width,
            board_height=self.engine.board.height,
            origin_x=self.engine.board.origin_x,
            origin_y=self.engine.board.origin_y,
            grid_size=self.engine.board.grid_size,
            trace_width=trace_width,
            clearance=clearance,
        )

        # Register obstacles
        router.register_components_in_grid(placement, parts_db)
        router.register_escapes_in_grid(escapes)

        results = {}

        # Route nets with B.Cu preference FIRST (they need vias)
        bcu_nets = [n for n in route_order if layer_prefs.get(n) == 'B.Cu']
        fcu_nets = [n for n in route_order if n not in bcu_nets]

        # Route B.Cu preferred nets first
        for net_name in bcu_nets:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            route = router._route_net_with_layer_preference(
                net_name, pins, escapes, 'B.Cu'
            )
            results[net_name] = route

            if not route.success:
                router.failed.append(net_name)

        # Then route F.Cu nets (normal routing)
        for net_name in fcu_nets:
            pins = net_pins.get(net_name, [])
            if len(pins) < 2:
                continue

            # Use standard routing (tries all strategies)
            route = router._route_net(net_name, pins, escapes)
            results[net_name] = route

            if not route.success:
                router.failed.append(net_name)

        return results

    def _internal_drc_check(self) -> List[DRCViolation]:
        """
        Internal DRC check using ValidationGate.

        This runs ALL built-in checks:
        - Connectivity (all nets connected)
        - Clearance (trace/pad spacing)
        - Trace width compliance
        - Via specifications
        - Dangling traces
        - Track overlaps/shorts
        - Component overlaps
        - Manufacturing checks
        """
        # Run ValidationGate
        result = self.validator.validate_detailed(
            routes=self.engine.state.routes,
            placement=self.engine.state.placement,
            parts_db=self.engine.state.parts_db,
            escapes=self.engine.state.escapes,
        )

        # Convert Violation objects to DRCViolation objects
        violations = []

        # Map ViolationType to DRC violation type strings
        type_map = {
            ViolationType.CONNECTIVITY: 'unconnected',
            ViolationType.CLEARANCE: 'clearance',
            ViolationType.TRACE_WIDTH: 'trace_width',
            ViolationType.VIA_DRILL: 'via',
            ViolationType.VIA_ANNULAR: 'via',
            ViolationType.DANGLING_TRACE: 'track_dangling',
            ViolationType.ZONE_VIOLATION: 'zone',
            ViolationType.SILKSCREEN: 'silkscreen',
            ViolationType.MANUFACTURING: 'manufacturing',
            ViolationType.COMPONENT_OVERLAP: 'overlap',
            ViolationType.COURTYARD: 'courtyard',
        }

        severity_map = {
            ViolationSeverity.ERROR: 'error',
            ViolationSeverity.WARNING: 'warning',
            ViolationSeverity.INFO: 'info',
        }

        # Convert errors
        for v in result.errors:
            violations.append(DRCViolation(
                type=type_map.get(v.type, 'unknown'),
                severity=severity_map.get(v.severity, 'error'),
                description=v.message,
                net1=v.items[0] if v.items else None,
                net2=v.items[1] if len(v.items) > 1 else None,
                position=v.location,
                actual_value=v.value,
                required_value=v.limit,
            ))

        # Convert warnings (optional - only include if we want to fix them too)
        for v in result.warnings:
            violations.append(DRCViolation(
                type=type_map.get(v.type, 'unknown'),
                severity=severity_map.get(v.severity, 'warning'),
                description=v.message,
                net1=v.items[0] if v.items else None,
                net2=v.items[1] if len(v.items) > 1 else None,
                position=v.location,
                actual_value=v.value,
                required_value=v.limit,
            ))

        return violations


def run_with_drc_feedback(board_config, design_rules, parts_dict: Dict,
                          output_path: str, max_iterations: int = 5) -> Dict:
    """
    Convenience function to run the full workflow with DRC feedback.

    Args:
        board_config: BoardConfig instance
        design_rules: DesignRules instance
        parts_dict: Parts dictionary
        output_path: Path for output KiCad script
        max_iterations: Maximum DRC correction iterations

    Returns:
        DRC feedback result dictionary
    """
    from .human_engine import HumanPCBEngine

    # Create engine
    engine = HumanPCBEngine(board_config, design_rules)
    engine.load_parts_from_dict(parts_dict)

    # Run with feedback loop
    feedback = DRCFeedback(engine, max_iterations=max_iterations)
    result = feedback.run_feedback_loop(output_path)

    return result
