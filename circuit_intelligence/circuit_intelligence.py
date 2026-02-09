"""
Circuit Intelligence Engine - Main Orchestrator
=================================================

This is the MAIN ENGINE that coordinates all circuit intelligence
modules. It follows the same modular architecture as PCB Engine.

Architecture:
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                    CIRCUIT INTELLIGENCE ENGINE                         ║
    ║                         (The Engineer)                                 ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  INPUT: parts_db, placement (optional)                                ║
    ║         ↓                                                             ║
    ║  ┌─────────────────────────────────────────────────────────────────┐  ║
    ║  │  ANALYZERS (understand the circuit)                             │  ║
    ║  │  ├── Component Analyzer    → What each part does                │  ║
    ║  │  ├── Net Analyzer          → What each net does                 │  ║
    ║  │  ├── Pattern Recognizer    → What circuit blocks exist          │  ║
    ║  │  ├── Current Flow Analyzer → How current flows                  │  ║
    ║  │  └── Thermal Analyzer      → Where heat is generated            │  ║
    ║  └─────────────────────────────────────────────────────────────────┘  ║
    ║         ↓                                                             ║
    ║  ┌─────────────────────────────────────────────────────────────────┐  ║
    ║  │  KNOWLEDGE BASE (what we know)                                  │  ║
    ║  │  ├── Pattern Library       → Circuit pattern rules              │  ║
    ║  │  ├── Component Database    → Part specifications                │  ║
    ║  │  ├── Electrical Calculator → Engineering formulas               │  ║
    ║  │  └── Design Rules Database → Manufacturing constraints          │  ║
    ║  └─────────────────────────────────────────────────────────────────┘  ║
    ║         ↓                                                             ║
    ║  ┌─────────────────────────────────────────────────────────────────┐  ║
    ║  │  REVIEWERS (find problems)                                      │  ║
    ║  │  ├── Design Review AI      → Expert-level issue detection       │  ║
    ║  │  ├── DFM Reviewer          → Manufacturing issues               │  ║
    ║  │  ├── EMI Reviewer          → EMC/EMI concerns                   │  ║
    ║  │  └── Safety Reviewer       → Safety/reliability issues          │  ║
    ║  └─────────────────────────────────────────────────────────────────┘  ║
    ║         ↓                                                             ║
    ║  ┌─────────────────────────────────────────────────────────────────┐  ║
    ║  │  LEARNING (improve over time)                                   │  ║
    ║  │  ├── ML Engine             → Predictions from data              │  ║
    ║  │  ├── Learning Database     → Past design outcomes               │  ║
    ║  │  └── Feedback Loop         → Learn from results                 │  ║
    ║  └─────────────────────────────────────────────────────────────────┘  ║
    ║         ↓                                                             ║
    ║  ┌─────────────────────────────────────────────────────────────────┐  ║
    ║  │  GENERATORS (produce output)                                    │  ║
    ║  │  ├── Constraint Generator  → Rules for PCB Engine               │  ║
    ║  │  ├── Report Generator      → Human-readable analysis            │  ║
    ║  │  └── Fix Suggester         → Actionable recommendations         │  ║
    ║  └─────────────────────────────────────────────────────────────────┘  ║
    ║         ↓                                                             ║
    ║  OUTPUT: CircuitAnalysis, DesignConstraints, DesignReport           ║
    ║                                                                        ║
    ╚═══════════════════════════════════════════════════════════════════════╝

Usage:
    from circuit_intelligence import CircuitIntelligence

    ci = CircuitIntelligence()

    # Full analysis
    analysis = ci.analyze(parts_db)

    # Generate constraints for PCB Engine
    constraints = ci.generate_constraints(analysis)

    # Get human-readable report
    report = ci.generate_report(analysis)

    # After PCB generation, record outcome for learning
    ci.record_outcome(analysis, drc_passed=True, errors=[])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

from .circuit_types import (
    CircuitFunction, NetFunction, ComponentFunction,
    Severity, ThermalRating,
    ComponentAnalysis, NetAnalysis, CircuitBlock, CurrentLoop,
    CircuitAnalysis, DesignIssue, PlacementConstraint, RoutingConstraint,
    DesignConstraints
)
from .pattern_library import PatternLibrary, CircuitPattern
from .component_database import ComponentDatabase
from .electrical_calculator import ElectricalCalculator
from .ml_engine import MLEngine


# =============================================================================
# ANALYZER MODULES
# =============================================================================

class ComponentAnalyzer:
    """Analyzes individual components to understand their function."""

    def __init__(self, component_db: ComponentDatabase):
        self.db = component_db

    def analyze(self, parts_db: Dict) -> Dict[str, ComponentAnalysis]:
        """Analyze all components in a design."""
        results = {}

        for ref, part in parts_db.get('parts', {}).items():
            analysis = self._analyze_component(ref, part)
            results[ref] = analysis

        return results

    def _analyze_component(self, ref: str, part: Dict) -> ComponentAnalysis:
        """Analyze a single component."""
        value = part.get('value', '')
        footprint = part.get('footprint', '')

        # Determine function from reference designator and value
        function = self._determine_function(ref, value, footprint)

        # Look up in component database
        db_info = self.db.lookup(value)

        # Estimate thermal dissipation
        thermal_dissipation = 0.0
        if db_info and db_info.thermal.theta_ja > 0:
            # Will be calculated when voltage/current known
            pass

        return ComponentAnalysis(
            ref=ref,
            value=value,
            footprint=footprint,
            function=function,
            thermal_dissipation=thermal_dissipation,
        )

    def _determine_function(self, ref: str, value: str, footprint: str) -> ComponentFunction:
        """Determine component function from its designator and value."""
        ref_prefix = ref.rstrip('0123456789')
        value_upper = value.upper()

        # Resistors
        if ref_prefix == 'R':
            if 'PULL' in value_upper:
                return ComponentFunction.RESISTOR_PULLUP
            if 'FB' in value_upper or 'FEEDBACK' in value_upper:
                return ComponentFunction.RESISTOR_FEEDBACK
            return ComponentFunction.RESISTOR_SERIES

        # Capacitors
        if ref_prefix == 'C':
            if '100N' in value_upper or '0.1U' in value_upper:
                return ComponentFunction.CAPACITOR_BYPASS
            if 'BULK' in value_upper or '100U' in value_upper or '10U' in value_upper:
                return ComponentFunction.CAPACITOR_BULK
            return ComponentFunction.CAPACITOR_FILTER

        # ICs
        if ref_prefix == 'U':
            if any(reg in value_upper for reg in ['LM2596', 'TPS54', 'MP1584', 'BUCK']):
                return ComponentFunction.VOLTAGE_REGULATOR
            if any(reg in value_upper for reg in ['AMS1117', 'LM1117', 'LDO', '7805']):
                return ComponentFunction.VOLTAGE_REGULATOR
            if any(mcu in value_upper for mcu in ['ESP32', 'STM32', 'ATMEGA', 'PIC']):
                return ComponentFunction.MICROCONTROLLER
            if 'MUX' in value_upper:
                return ComponentFunction.MUX
            if 'AMP' in value_upper or 'OPAMP' in value_upper:
                return ComponentFunction.AMPLIFIER
            return ComponentFunction.UNKNOWN

        # Diodes
        if ref_prefix == 'D':
            if 'LED' in value_upper:
                return ComponentFunction.LED_INDICATOR
            return ComponentFunction.POLARITY_PROTECTION

        # Connectors
        if ref_prefix == 'J':
            return ComponentFunction.CONNECTOR_SIGNAL

        # Inductors
        if ref_prefix == 'L':
            return ComponentFunction.INDUCTOR_POWER

        # Fuses
        if ref_prefix == 'F':
            return ComponentFunction.FUSE

        # Crystals
        if ref_prefix in ('Y', 'X'):
            return ComponentFunction.CRYSTAL

        return ComponentFunction.UNKNOWN


class NetAnalyzer:
    """Analyzes nets to understand their function."""

    def analyze(self, parts_db: Dict,
                component_analysis: Dict[str, ComponentAnalysis]) -> Dict[str, NetAnalysis]:
        """Analyze all nets in a design."""
        results = {}

        for net_name, net_info in parts_db.get('nets', {}).items():
            analysis = self._analyze_net(net_name, net_info, component_analysis)
            results[net_name] = analysis

        return results

    def _analyze_net(self, net_name: str, net_info: Dict,
                     component_analysis: Dict[str, ComponentAnalysis]) -> NetAnalysis:
        """Analyze a single net."""
        name_upper = net_name.upper()
        pins = net_info.get('pins', [])

        # Determine function from name
        function = self._determine_function(name_upper)

        # Determine routing requirements based on function
        min_width = 0.25
        max_length = None
        preferred_layer = 'F.Cu'

        if function == NetFunction.POWER_RAIL:
            min_width = 0.5
        elif function == NetFunction.GROUND:
            min_width = 0.5
            preferred_layer = 'B.Cu'
        elif function == NetFunction.SWITCH_NODE:
            min_width = 0.5
            max_length = 10.0
        elif function == NetFunction.HIGH_SPEED_DATA:
            max_length = 50.0

        return NetAnalysis(
            name=net_name,
            function=function,
            pins=pins,
            min_width=min_width,
            recommended_width=min_width,
            max_length=max_length,
            preferred_layer=preferred_layer,
        )

    def _determine_function(self, name_upper: str) -> NetFunction:
        """Determine net function from its name."""
        # Ground nets
        if any(g in name_upper for g in ['GND', 'VSS', 'AGND', 'DGND', 'PGND']):
            return NetFunction.GROUND

        # Power nets
        if any(p in name_upper for p in ['VCC', 'VDD', 'VIN', '+5V', '+3V3', '+12V', 'VBUS']):
            return NetFunction.POWER_RAIL

        # Clock nets
        if any(c in name_upper for c in ['CLK', 'CLOCK', 'SCK', 'SCLK']):
            return NetFunction.CLOCK

        # High speed
        if any(h in name_upper for h in ['USB', 'ETH', 'RMII', 'DDR', 'LVDS']):
            return NetFunction.HIGH_SPEED_DATA

        # Analog
        if any(a in name_upper for a in ['ADC', 'DAC', 'AIN', 'AOUT', 'VREF']):
            return NetFunction.ANALOG_SIGNAL

        # Switch nodes (power converter)
        if 'SW' in name_upper and 'USB' not in name_upper:
            return NetFunction.SWITCH_NODE

        return NetFunction.LOW_SPEED_DATA


class PatternRecognizer:
    """Recognizes circuit patterns in a design."""

    def __init__(self, pattern_library: PatternLibrary):
        self.library = pattern_library

    def recognize(self, parts_db: Dict,
                  component_analysis: Dict[str, ComponentAnalysis],
                  net_analysis: Dict[str, NetAnalysis]) -> List[Tuple[CircuitPattern, Dict]]:
        """Recognize patterns in the design."""
        return self.library.detect_patterns(parts_db)


class CurrentFlowAnalyzer:
    """Analyzes current flow paths in the circuit."""

    def analyze(self, parts_db: Dict,
                net_analysis: Dict[str, NetAnalysis],
                patterns: List[Tuple[CircuitPattern, Dict]]) -> List[CurrentLoop]:
        """Identify critical current loops."""
        loops = []

        # For each pattern with critical loops
        for pattern, mapping in patterns:
            for critical_loop in pattern.critical_loops:
                # Create CurrentLoop object
                loop = CurrentLoop(
                    name=f"{pattern.name}_{critical_loop.name}",
                    components=[mapping.get(role, '') for role in critical_loop.component_roles],
                    nets=[],  # Would be filled with actual nets
                    is_critical=True,
                    max_allowed_area=critical_loop.max_area
                )
                loops.append(loop)

        return loops


class ThermalAnalyzer:
    """Analyzes thermal characteristics of the design."""

    def __init__(self, component_db: ComponentDatabase, calculator: ElectricalCalculator):
        self.db = component_db
        self.calc = calculator

    def analyze(self, parts_db: Dict,
                component_analysis: Dict[str, ComponentAnalysis],
                ambient_temp: float = 25.0) -> Dict[str, ComponentAnalysis]:
        """Update component analysis with thermal data."""
        for ref, analysis in component_analysis.items():
            db_info = self.db.lookup(analysis.value)

            if db_info and db_info.thermal.theta_ja > 0:
                # Estimate power dissipation (simplified)
                power = analysis.thermal_dissipation

                # Calculate junction temperature
                tj = self.calc.thermal.junction_temperature(
                    power, db_info.thermal.theta_ja, ambient_temp
                )

                analysis.temperature_estimate = tj

                # Determine thermal rating
                if tj > 100:
                    analysis.thermal_rating = ThermalRating.CRITICAL
                    analysis.needs_copper_pour = True
                    analysis.needs_thermal_vias = self.calc.thermal.thermal_via_count(power)
                elif tj > 70:
                    analysis.thermal_rating = ThermalRating.HOT
                    analysis.needs_copper_pour = True
                elif tj > 40:
                    analysis.thermal_rating = ThermalRating.WARM
                else:
                    analysis.thermal_rating = ThermalRating.COLD

        return component_analysis


# =============================================================================
# REVIEWER MODULES
# =============================================================================

class DesignReviewer:
    """Reviews design for expert-level issues."""

    def __init__(self, pattern_library: PatternLibrary):
        self.library = pattern_library

    def review(self, parts_db: Dict, placement: Optional[Dict],
               component_analysis: Dict[str, ComponentAnalysis],
               net_analysis: Dict[str, NetAnalysis],
               patterns: List[Tuple[CircuitPattern, Dict]]) -> List[DesignIssue]:
        """Review design for issues."""
        issues = []

        # Check bypass capacitors
        issues.extend(self._check_bypass_caps(parts_db, placement, component_analysis))

        # Check pattern-specific rules
        issues.extend(self._check_pattern_rules(patterns, placement))

        # Check for missing components
        issues.extend(self._check_missing_components(parts_db, component_analysis))

        return issues

    def _check_bypass_caps(self, parts_db: Dict, placement: Optional[Dict],
                           component_analysis: Dict[str, ComponentAnalysis]) -> List[DesignIssue]:
        """Check that all ICs have bypass caps."""
        issues = []

        # Find ICs
        ics = [ref for ref, a in component_analysis.items()
               if a.function in (ComponentFunction.MICROCONTROLLER,
                                 ComponentFunction.VOLTAGE_REGULATOR,
                                 ComponentFunction.MUX,
                                 ComponentFunction.AMPLIFIER)]

        # Find bypass caps
        bypass_caps = [ref for ref, a in component_analysis.items()
                       if a.function == ComponentFunction.CAPACITOR_BYPASS]

        if not bypass_caps and ics:
            for ic in ics:
                issues.append(DesignIssue(
                    severity=Severity.ERROR,
                    category='POWER',
                    component=ic,
                    net=None,
                    message=f'{ic} has no bypass capacitor',
                    recommendation='Add 100nF ceramic capacitor close to VCC pin',
                    auto_fixable=False
                ))

        # If placement provided, check distances
        if placement:
            for ic in ics:
                if ic not in placement:
                    continue
                ic_pos = placement[ic]

                # Find nearest bypass cap
                min_dist = float('inf')
                nearest_cap = None
                for cap in bypass_caps:
                    if cap not in placement:
                        continue
                    cap_pos = placement[cap]
                    dist = ((ic_pos[0] - cap_pos[0])**2 + (ic_pos[1] - cap_pos[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cap = cap

                if min_dist > 5.0:
                    issues.append(DesignIssue(
                        severity=Severity.WARNING,
                        category='POWER',
                        component=ic,
                        net=None,
                        message=f'{ic} bypass cap is {min_dist:.1f}mm away (should be < 5mm)',
                        recommendation=f'Move {nearest_cap} closer to {ic}',
                        auto_fixable=True
                    ))

        return issues

    def _check_pattern_rules(self, patterns: List[Tuple[CircuitPattern, Dict]],
                              placement: Optional[Dict]) -> List[DesignIssue]:
        """Check pattern-specific rules."""
        issues = []

        for pattern, mapping in patterns:
            # Check placement rules
            if placement:
                for rule in pattern.placement_rules:
                    component_ref = mapping.get(rule.component_role)
                    target_ref = mapping.get(rule.target_role)

                    if component_ref and target_ref:
                        if component_ref in placement and target_ref in placement:
                            pos1 = placement[component_ref]
                            pos2 = placement[target_ref]
                            dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

                            if dist > rule.max_distance:
                                issues.append(DesignIssue(
                                    severity=rule.severity,
                                    category='PLACEMENT',
                                    component=component_ref,
                                    net=None,
                                    message=f'{pattern.name}: {rule.description}',
                                    recommendation=f'Move {component_ref} within {rule.max_distance}mm of {target_ref}',
                                    auto_fixable=True
                                ))

        return issues

    def _check_missing_components(self, parts_db: Dict,
                                   component_analysis: Dict[str, ComponentAnalysis]) -> List[DesignIssue]:
        """Check for commonly missing components."""
        issues = []

        # Check for ESD protection on connectors
        has_usb = any('USB' in p.get('value', '').upper() for p in parts_db.get('parts', {}).values())
        has_esd = any('ESD' in p.get('value', '').upper() or 'TVS' in p.get('value', '').upper()
                      for p in parts_db.get('parts', {}).values())

        if has_usb and not has_esd:
            issues.append(DesignIssue(
                severity=Severity.WARNING,
                category='PROTECTION',
                component=None,
                net=None,
                message='USB interface without ESD protection',
                recommendation='Add ESD protection IC (e.g., USBLC6-2) near USB connector',
                auto_fixable=False
            ))

        return issues


# =============================================================================
# GENERATOR MODULES
# =============================================================================

class ConstraintGenerator:
    """Generates constraints for PCB Engine."""

    def generate(self, analysis: CircuitAnalysis) -> DesignConstraints:
        """Generate constraints from analysis."""
        constraints = DesignConstraints()

        # Placement constraints from component analysis
        for ref, comp in analysis.components.items():
            for must_near in comp.must_be_near:
                constraints.placement.append(PlacementConstraint(
                    type='PROXIMITY',
                    component=ref,
                    target=must_near,
                    value=5.0,
                    priority=1
                ))

            for target, max_dist in comp.max_distance_from.items():
                constraints.placement.append(PlacementConstraint(
                    type='DISTANCE',
                    component=ref,
                    target=target,
                    value=max_dist,
                    priority=1
                ))

        # Routing constraints from net analysis
        for net_name, net in analysis.nets.items():
            if net.min_width > 0.25:
                constraints.routing.append(RoutingConstraint(
                    type='WIDTH',
                    net=net_name,
                    value=net.min_width,
                    priority=1
                ))

            if net.max_length:
                constraints.routing.append(RoutingConstraint(
                    type='LENGTH',
                    net=net_name,
                    value=net.max_length,
                    priority=1
                ))

            if net.max_vias:
                constraints.routing.append(RoutingConstraint(
                    type='VIA_LIMIT',
                    net=net_name,
                    value=net.max_vias,
                    priority=1
                ))

        return constraints


class ReportGenerator:
    """Generates human-readable reports."""

    def generate(self, analysis: CircuitAnalysis) -> str:
        """Generate a design report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CIRCUIT INTELLIGENCE ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Design Score: {analysis.score}/100")
        lines.append("")

        # Component summary
        lines.append("COMPONENTS")
        lines.append("-" * 40)
        lines.append(f"Total: {len(analysis.components)}")
        for ref, comp in sorted(analysis.components.items()):
            lines.append(f"  {ref}: {comp.value} ({comp.function.name})")
        lines.append("")

        # Net summary
        lines.append("NETS")
        lines.append("-" * 40)
        lines.append(f"Total: {len(analysis.nets)}")
        for name, net in sorted(analysis.nets.items()):
            lines.append(f"  {name}: {net.function.name} ({len(net.pins)} pins)")
        lines.append("")

        # Issues
        if analysis.issues:
            lines.append("ISSUES FOUND")
            lines.append("-" * 40)
            for issue in sorted(analysis.issues, key=lambda x: x.severity.value):
                severity_str = issue.severity.name
                lines.append(f"  [{severity_str}] {issue.message}")
                lines.append(f"    → {issue.recommendation}")
            lines.append("")

        # Critical items
        if analysis.critical_nets:
            lines.append("CRITICAL NETS (require special attention)")
            lines.append("-" * 40)
            for net in analysis.critical_nets:
                lines.append(f"  - {net}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class CircuitIntelligence:
    """
    Main Circuit Intelligence Engine.

    This is the orchestrator that coordinates all modules.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the engine with all modules."""
        self.config = config or {}

        # Knowledge base
        self.pattern_library = PatternLibrary()
        self.component_db = ComponentDatabase()
        self.calculator = ElectricalCalculator()

        # Analyzers
        self.component_analyzer = ComponentAnalyzer(self.component_db)
        self.net_analyzer = NetAnalyzer()
        self.pattern_recognizer = PatternRecognizer(self.pattern_library)
        self.current_flow_analyzer = CurrentFlowAnalyzer()
        self.thermal_analyzer = ThermalAnalyzer(self.component_db, self.calculator)

        # Reviewers
        self.design_reviewer = DesignReviewer(self.pattern_library)

        # Generators
        self.constraint_generator = ConstraintGenerator()
        self.report_generator = ReportGenerator()

        # Learning
        self.ml_engine = MLEngine()

    def analyze(self, parts_db: Dict,
                placement: Optional[Dict] = None,
                board_width: float = 50.0,
                board_height: float = 35.0) -> CircuitAnalysis:
        """
        Perform complete circuit analysis.

        Args:
            parts_db: Component and net definitions
            placement: Optional component placement (for placement-dependent checks)
            board_width: Board width in mm
            board_height: Board height in mm

        Returns:
            CircuitAnalysis with complete analysis results
        """
        # Step 1: Analyze components
        component_analysis = self.component_analyzer.analyze(parts_db)

        # Step 2: Analyze nets
        net_analysis = self.net_analyzer.analyze(parts_db, component_analysis)

        # Step 3: Recognize patterns
        patterns = self.pattern_recognizer.recognize(parts_db, component_analysis, net_analysis)

        # Step 4: Analyze current flow
        current_loops = self.current_flow_analyzer.analyze(parts_db, net_analysis, patterns)

        # Step 5: Thermal analysis
        component_analysis = self.thermal_analyzer.analyze(
            parts_db, component_analysis, ambient_temp=self.config.get('ambient_temp', 25.0)
        )

        # Step 6: Design review
        issues = self.design_reviewer.review(
            parts_db, placement, component_analysis, net_analysis, patterns
        )

        # Step 7: ML predictions (if placement provided)
        if placement:
            ml_predictions = self.ml_engine.predict_issues(
                parts_db, placement, board_width, board_height
            )
            for pred in ml_predictions:
                issues.append(DesignIssue(
                    severity=Severity.WARNING,
                    category='ML_PREDICTION',
                    component=None,
                    net=None,
                    message=pred.explanation,
                    recommendation=f'Confidence: {pred.confidence:.0%}',
                    auto_fixable=False
                ))

        # Build circuit blocks from patterns
        blocks = []
        for pattern, mapping in patterns:
            block = CircuitBlock(
                name=pattern.name,
                function=pattern.function,
                components=list(mapping.values()),
                nets=[],
                input_nets=[],
                output_nets=[]
            )
            blocks.append(block)

        # Identify critical nets
        critical_nets = [name for name, net in net_analysis.items()
                         if net.function in (NetFunction.SWITCH_NODE,
                                             NetFunction.HIGH_SPEED_DATA,
                                             NetFunction.CLOCK)]

        # Calculate total power
        total_power = sum(c.thermal_dissipation for c in component_analysis.values())

        # Create analysis result
        analysis = CircuitAnalysis(
            components=component_analysis,
            nets=net_analysis,
            blocks=blocks,
            current_loops=current_loops,
            issues=issues,
            total_power_dissipation=total_power,
            critical_nets=critical_nets,
            critical_loops=[loop.name for loop in current_loops if loop.is_critical]
        )

        return analysis

    def generate_constraints(self, analysis: CircuitAnalysis) -> DesignConstraints:
        """Generate constraints for PCB Engine."""
        return self.constraint_generator.generate(analysis)

    def generate_report(self, analysis: CircuitAnalysis) -> str:
        """Generate human-readable report."""
        return self.report_generator.generate(analysis)

    def record_outcome(self, analysis: CircuitAnalysis,
                       drc_passed: bool, errors: List[str],
                       routing_success_rate: float = 0.0):
        """Record design outcome for learning."""
        # This would feed into the ML engine
        pass

    def get_calculator(self) -> ElectricalCalculator:
        """Access the electrical calculator."""
        return self.calculator

    def get_component_info(self, part_number: str) -> Optional[Dict]:
        """Look up component information."""
        data = self.component_db.lookup(part_number)
        if data:
            return data.__dict__
        return None
