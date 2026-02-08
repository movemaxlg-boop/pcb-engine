"""
PISTON ORCHESTRATOR - Smart Piston Selection
==============================================

Determines which of the 18 pistons are needed based on:
1. Design requirements (sensors, MCU, power, etc.)
2. User preferences (thermal analysis, 3D viz, BOM optimization)
3. Board complexity (layers, size, component density)
4. Design constraints (cost, time, quality)

PISTON CATEGORIES:
==================

ALWAYS REQUIRED (Core Flow):
- Parts      : Always - converts requirements to components
- Order      : Always - determines processing sequence
- Placement  : Always - places components on board
- Routing    : Always - connects components with traces
- DRC        : Always - validates design rules
- Output     : Always - generates final files

CONDITIONAL (Based on Design):
- Escape     : Only for BGA/QFN/dense packages
- Optimize   : When routing completion > 80%
- Silkscreen : When generate_silkscreen=True (default True)
- Stackup    : For 4+ layer boards
- Netlist    : Always (generates netlists)

OPTIONAL (User Preference):
- Thermal        : High-power designs, motor drivers, power supplies
- PDN            : High-speed designs, sensitive analog
- Signal Int.    : High-speed signals (>100MHz), DDR, USB3
- Topological    : Complex routing, differential pairs
- 3D Visual.     : When user wants 3D preview
- BOM Optimizer  : When cost optimization matters
- Learning       : When ML improvement is enabled

AUTOMATIC DETECTION:
====================
The orchestrator analyzes the parts_db and requirements to detect:
- High-power components -> Thermal analysis
- High-speed interfaces -> Signal integrity
- Dense packages -> Escape routing
- 4+ layers -> Stackup optimization
- Differential pairs -> Topological routing
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any
import re


class PistonCategory(Enum):
    """Piston categories"""
    CORE = 'core'           # Always run
    CONDITIONAL = 'cond'    # Based on design
    OPTIONAL = 'optional'   # User preference
    ML = 'ml'               # Machine learning


class PistonPriority(Enum):
    """Execution priority"""
    CRITICAL = 1    # Must run, abort if fails
    IMPORTANT = 2   # Should run, warn if fails
    OPTIONAL = 3    # Nice to have, skip if fails
    BACKGROUND = 4  # Can run async


@dataclass
class PistonSpec:
    """Specification for a piston"""
    name: str
    category: PistonCategory
    priority: PistonPriority
    description: str

    # Dependencies
    requires: List[str] = field(default_factory=list)      # Must run after these
    conflicts: List[str] = field(default_factory=list)     # Cannot run with these
    enhances: List[str] = field(default_factory=list)      # Makes these better

    # Conditions for automatic enabling
    auto_enable_conditions: List[str] = field(default_factory=list)

    # Resource requirements
    estimated_time_ms: int = 1000
    memory_mb: int = 50
    cpu_intensive: bool = False


@dataclass
class PistonSelection:
    """Result of piston selection"""
    required_pistons: List[str]         # Must run
    optional_pistons: List[str]         # Can skip
    skipped_pistons: List[str]          # Not needed
    execution_order: List[str]          # Optimal order

    # Reasoning
    selection_reasons: Dict[str, str]   # Why each was selected/skipped
    warnings: List[str] = field(default_factory=list)

    # Estimated resources
    estimated_total_time_ms: int = 0
    estimated_memory_mb: int = 0


@dataclass
class DesignProfile:
    """Profile of the design for piston selection"""
    # Component analysis
    has_bga: bool = False
    has_qfn: bool = False
    has_qfp: bool = False
    has_fine_pitch: bool = False          # <0.5mm pitch
    component_count: int = 0

    # Power analysis
    total_power_w: float = 0.0
    has_high_power: bool = False          # >2W single component
    has_motor_driver: bool = False
    has_power_supply: bool = False
    has_battery_charger: bool = False

    # Signal analysis
    max_frequency_mhz: float = 0.0
    has_high_speed: bool = False          # >100MHz signals
    has_differential_pairs: bool = False
    has_ddr: bool = False
    has_usb3: bool = False
    has_pcie: bool = False
    has_ethernet: bool = False

    # Board analysis
    layer_count: int = 2
    board_area_mm2: float = 0.0
    component_density: float = 0.0        # components per cm^2

    # Communication
    has_wifi: bool = False
    has_bluetooth: bool = False
    has_lora: bool = False
    has_rf: bool = False

    # Analog
    has_adc: bool = False
    has_dac: bool = False
    has_sensitive_analog: bool = False

    # Pour analysis
    gnd_pin_count: int = 0
    power_pin_count: int = 0


# =============================================================================
# PISTON REGISTRY
# =============================================================================

PISTON_REGISTRY: Dict[str, PistonSpec] = {
    # === CORE PISTONS (Always Required) ===
    'parts': PistonSpec(
        name='parts',
        category=PistonCategory.CORE,
        priority=PistonPriority.CRITICAL,
        description='Converts requirements to component database',
        requires=[],
        estimated_time_ms=500,
        memory_mb=30
    ),

    'order': PistonSpec(
        name='order',
        category=PistonCategory.CORE,
        priority=PistonPriority.CRITICAL,
        description='Determines placement and routing order',
        requires=['parts'],
        estimated_time_ms=200,
        memory_mb=20
    ),

    'placement': PistonSpec(
        name='placement',
        category=PistonCategory.CORE,
        priority=PistonPriority.CRITICAL,
        description='Places components optimally on board',
        requires=['order'],
        estimated_time_ms=2000,
        memory_mb=100,
        cpu_intensive=True
    ),

    'routing': PistonSpec(
        name='routing',
        category=PistonCategory.CORE,
        priority=PistonPriority.CRITICAL,
        description='Routes traces between components',
        requires=['placement'],
        estimated_time_ms=5000,
        memory_mb=150,
        cpu_intensive=True
    ),

    'drc': PistonSpec(
        name='drc',
        category=PistonCategory.CORE,
        priority=PistonPriority.CRITICAL,
        description='Validates design rules',
        requires=['routing'],
        estimated_time_ms=1000,
        memory_mb=50
    ),

    'output': PistonSpec(
        name='output',
        category=PistonCategory.CORE,
        priority=PistonPriority.CRITICAL,
        description='Generates final output files',
        requires=['drc'],
        estimated_time_ms=2000,
        memory_mb=100
    ),

    # === CONDITIONAL PISTONS ===
    'escape': PistonSpec(
        name='escape',
        category=PistonCategory.CONDITIONAL,
        priority=PistonPriority.IMPORTANT,
        description='Routes escape patterns for BGA/QFN packages',
        requires=['placement'],
        enhances=['routing'],
        auto_enable_conditions=['has_bga', 'has_qfn', 'has_fine_pitch'],
        estimated_time_ms=1500,
        memory_mb=80
    ),

    'optimize': PistonSpec(
        name='optimize',
        category=PistonCategory.CONDITIONAL,
        priority=PistonPriority.IMPORTANT,
        description='Optimizes trace lengths and via count',
        requires=['routing'],
        auto_enable_conditions=['routing_complete_80+'],
        estimated_time_ms=3000,
        memory_mb=100,
        cpu_intensive=True
    ),

    'silkscreen': PistonSpec(
        name='silkscreen',
        category=PistonCategory.CONDITIONAL,
        priority=PistonPriority.OPTIONAL,
        description='Places reference designators and labels',
        requires=['routing'],
        estimated_time_ms=500,
        memory_mb=30
    ),

    'stackup': PistonSpec(
        name='stackup',
        category=PistonCategory.CONDITIONAL,
        priority=PistonPriority.IMPORTANT,
        description='Optimizes layer stackup for 4+ layer boards',
        requires=['parts'],
        enhances=['routing', 'pdn', 'signal_integrity'],
        auto_enable_conditions=['layer_count_4+'],
        estimated_time_ms=500,
        memory_mb=40
    ),

    'netlist': PistonSpec(
        name='netlist',
        category=PistonCategory.CONDITIONAL,
        priority=PistonPriority.IMPORTANT,
        description='Generates netlist for simulation/verification',
        requires=['parts'],
        estimated_time_ms=300,
        memory_mb=30
    ),

    # === OPTIONAL PISTONS (User Preference / Auto-detect) ===
    'thermal': PistonSpec(
        name='thermal',
        category=PistonCategory.OPTIONAL,
        priority=PistonPriority.OPTIONAL,
        description='Thermal analysis and heat distribution',
        requires=['placement', 'routing'],
        auto_enable_conditions=['has_high_power', 'has_motor_driver', 'total_power_5w+'],
        estimated_time_ms=3000,
        memory_mb=200,
        cpu_intensive=True
    ),

    'pdn': PistonSpec(
        name='pdn',
        category=PistonCategory.OPTIONAL,
        priority=PistonPriority.OPTIONAL,
        description='Power Delivery Network analysis',
        requires=['routing'],
        auto_enable_conditions=['has_high_speed', 'has_sensitive_analog', 'layer_count_4+'],
        estimated_time_ms=2000,
        memory_mb=150
    ),

    'signal_integrity': PistonSpec(
        name='signal_integrity',
        category=PistonCategory.OPTIONAL,
        priority=PistonPriority.OPTIONAL,
        description='Signal integrity analysis for high-speed traces',
        requires=['routing'],
        auto_enable_conditions=['has_high_speed', 'has_ddr', 'has_usb3', 'has_pcie'],
        estimated_time_ms=4000,
        memory_mb=250,
        cpu_intensive=True
    ),

    'topological_router': PistonSpec(
        name='topological_router',
        category=PistonCategory.OPTIONAL,
        priority=PistonPriority.OPTIONAL,
        description='Advanced topological routing with rubber-banding',
        requires=['placement'],
        conflicts=['routing'],  # Alternative to standard routing
        auto_enable_conditions=['has_differential_pairs', 'complex_routing'],
        estimated_time_ms=8000,
        memory_mb=300,
        cpu_intensive=True
    ),

    'visualization_3d': PistonSpec(
        name='visualization_3d',
        category=PistonCategory.OPTIONAL,
        priority=PistonPriority.BACKGROUND,
        description='3D visualization and model export',
        requires=['placement', 'routing'],
        estimated_time_ms=5000,
        memory_mb=400
    ),

    'bom_optimizer': PistonSpec(
        name='bom_optimizer',
        category=PistonCategory.OPTIONAL,
        priority=PistonPriority.OPTIONAL,
        description='Optimizes BOM for cost/availability',
        requires=['parts'],
        estimated_time_ms=2000,
        memory_mb=100
    ),

    # === POUR PISTON (GND/Power Planes) ===
    'pour': PistonSpec(
        name='pour',
        category=PistonCategory.CONDITIONAL,
        priority=PistonPriority.IMPORTANT,
        description='Generates copper pours for GND/power planes',
        requires=['routing'],
        enhances=['drc', 'output'],  # Eliminates GND trace crossings
        auto_enable_conditions=[
            'layer_count_2',      # 2-layer boards benefit most
            'has_gnd_crossings',  # When routing causes GND crossings
            'has_many_gnd_pins',  # Many GND pins to connect
        ],
        estimated_time_ms=500,
        memory_mb=50
    ),

    # === ML PISTON ===
    'learning': PistonSpec(
        name='learning',
        category=PistonCategory.ML,
        priority=PistonPriority.BACKGROUND,
        description='Machine learning for design improvement',
        requires=[],
        estimated_time_ms=1000,
        memory_mb=200
    ),
}


# =============================================================================
# DESIGN ANALYZER
# =============================================================================

class DesignAnalyzer:
    """Analyzes design to create a profile for piston selection"""

    # Component patterns for detection
    BGA_PATTERNS = ['bga', 'ball grid', 'fbga', 'tfbga', 'wlcsp']
    QFN_PATTERNS = ['qfn', 'dfn', 'mlf', 'son', 'wson']
    QFP_PATTERNS = ['qfp', 'tqfp', 'lqfp', 'pqfp']
    MOTOR_PATTERNS = ['drv8', 'tb6612', 'l298', 'l9110', 'a4988', 'tmc', 'bts7960']
    POWER_PATTERNS = ['lm2596', 'mp1584', 'tps', 'ldo', 'buck', 'boost', 'dcdc']
    HIGH_SPEED_PATTERNS = ['usb3', 'pcie', 'ddr', 'hdmi', 'dp', 'sgmii', 'lvds']

    def analyze(self, parts_db: Dict, requirements: Any = None) -> DesignProfile:
        """Analyze design and create profile"""
        profile = DesignProfile()

        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Analyze each component
        for ref, part in parts.items():
            self._analyze_component(ref, part, profile)

        # Analyze nets
        for net_name, net_data in nets.items():
            self._analyze_net(net_name, net_data, profile)

        # Calculate derived metrics
        profile.component_count = len(parts)
        if requirements:
            self._apply_requirements(requirements, profile)

        # Determine composite flags
        profile.has_high_power = profile.total_power_w > 2.0 or profile.has_motor_driver
        profile.has_high_speed = profile.max_frequency_mhz > 100 or profile.has_usb3 or profile.has_pcie

        return profile

    def _analyze_component(self, ref: str, part: Dict, profile: DesignProfile):
        """Analyze a single component"""
        footprint = part.get('footprint', '').lower()
        value = part.get('value', '').lower()
        description = part.get('description', '').lower()

        combined = f"{footprint} {value} {description}"

        # Package detection
        if any(p in combined for p in self.BGA_PATTERNS):
            profile.has_bga = True
            profile.has_fine_pitch = True

        if any(p in combined for p in self.QFN_PATTERNS):
            profile.has_qfn = True

        if any(p in combined for p in self.QFP_PATTERNS):
            profile.has_qfp = True

        # Check pin pitch
        pitch_match = re.search(r'p?(\d+\.?\d*)mm', footprint)
        if pitch_match:
            pitch = float(pitch_match.group(1))
            if pitch < 0.5:
                profile.has_fine_pitch = True

        # Power components
        if any(p in combined for p in self.MOTOR_PATTERNS):
            profile.has_motor_driver = True
            profile.total_power_w += 5.0  # Estimate

        if any(p in combined for p in self.POWER_PATTERNS):
            profile.has_power_supply = True

        if 'tp4056' in combined or 'mcp7383' in combined or 'bq24' in combined:
            profile.has_battery_charger = True

        # High-speed detection
        if any(p in combined for p in self.HIGH_SPEED_PATTERNS):
            profile.has_high_speed = True

        if 'usb3' in combined or 'superspeed' in combined:
            profile.has_usb3 = True
            profile.max_frequency_mhz = max(profile.max_frequency_mhz, 5000)

        if 'pcie' in combined:
            profile.has_pcie = True
            profile.max_frequency_mhz = max(profile.max_frequency_mhz, 8000)

        if 'ddr' in combined:
            profile.has_ddr = True
            profile.max_frequency_mhz = max(profile.max_frequency_mhz, 1600)

        # Communication
        if 'esp32' in combined or 'esp8266' in combined or 'wifi' in combined:
            profile.has_wifi = True
            profile.has_bluetooth = True
            profile.max_frequency_mhz = max(profile.max_frequency_mhz, 2400)

        if 'sx127' in combined or 'rfm9' in combined or 'lora' in combined:
            profile.has_lora = True
            profile.has_rf = True

        if 'w5500' in combined or 'enc28' in combined or 'ethernet' in combined:
            profile.has_ethernet = True

        # Analog
        if ref.startswith('U') and ('adc' in combined or 'ads' in combined):
            profile.has_adc = True
            profile.has_sensitive_analog = True

        if ref.startswith('U') and 'dac' in combined:
            profile.has_dac = True

    def _analyze_net(self, net_name: str, net_data: Dict, profile: DesignProfile):
        """Analyze a net for characteristics"""
        net_lower = net_name.lower()

        # Differential pair detection
        if net_lower.endswith('_p') or net_lower.endswith('_n'):
            profile.has_differential_pairs = True
        if 'diff' in net_lower or 'dp' in net_lower or 'dn' in net_lower:
            profile.has_differential_pairs = True

        # Count GND/power pins for pour analysis
        pins = net_data.get('pins', [])
        pin_count = len(pins)
        if net_lower == 'gnd' or net_lower == 'vss' or net_lower == 'ground':
            profile.gnd_pin_count += pin_count
        elif net_lower in ('5v', '3v3', '3.3v', 'vcc', 'vdd', 'vin', 'vbus'):
            profile.power_pin_count += pin_count

    def _apply_requirements(self, requirements: Any, profile: DesignProfile):
        """Apply explicit requirements to profile"""
        if hasattr(requirements, 'layers'):
            profile.layer_count = requirements.layers
        if hasattr(requirements, 'board_size_mm'):
            w, h = requirements.board_size_mm
            profile.board_area_mm2 = w * h


# =============================================================================
# PISTON ORCHESTRATOR
# =============================================================================

class PistonOrchestrator:
    """
    Intelligently selects and orders pistons based on design requirements.

    Usage:
        orchestrator = PistonOrchestrator()
        selection = orchestrator.select_pistons(
            parts_db=my_parts,
            requirements=my_requirements,
            user_preferences={
                'generate_3d': True,
                'optimize_bom': True,
                'run_thermal': False  # Override auto-detection
            }
        )

        for piston_name in selection.execution_order:
            engine.run_piston(piston_name)
    """

    def __init__(self):
        self.analyzer = DesignAnalyzer()
        self.registry = PISTON_REGISTRY

    def select_pistons(
        self,
        parts_db: Dict,
        requirements: Any = None,
        user_preferences: Dict = None,
        force_enable: List[str] = None,
        force_disable: List[str] = None
    ) -> PistonSelection:
        """
        Select which pistons to run.

        Args:
            parts_db: Component and net database
            requirements: Circuit requirements (from NLP parser)
            user_preferences: User preference overrides
            force_enable: List of pistons to always enable
            force_disable: List of pistons to always disable

        Returns:
            PistonSelection with ordered list of pistons to run
        """
        user_preferences = user_preferences or {}
        force_enable = force_enable or []
        force_disable = force_disable or []

        # Analyze design
        profile = self.analyzer.analyze(parts_db, requirements)

        # Select pistons
        required = []
        optional = []
        skipped = []
        reasons = {}

        for name, spec in self.registry.items():
            # Check force flags first
            if name in force_disable:
                skipped.append(name)
                reasons[name] = "Force disabled by user"
                continue

            if name in force_enable:
                required.append(name)
                reasons[name] = "Force enabled by user"
                continue

            # Core pistons are always required
            if spec.category == PistonCategory.CORE:
                required.append(name)
                reasons[name] = "Core piston - always required"
                continue

            # Check user preferences
            pref_key = self._get_preference_key(name)
            if pref_key in user_preferences:
                if user_preferences[pref_key]:
                    required.append(name)
                    reasons[name] = f"Enabled by user preference: {pref_key}"
                else:
                    skipped.append(name)
                    reasons[name] = f"Disabled by user preference: {pref_key}"
                continue

            # Check auto-enable conditions
            should_enable, condition = self._check_conditions(spec, profile)
            if should_enable:
                if spec.priority == PistonPriority.IMPORTANT:
                    required.append(name)
                else:
                    optional.append(name)
                reasons[name] = f"Auto-enabled: {condition}"
            else:
                if spec.category == PistonCategory.CONDITIONAL:
                    # Conditional pistons may still be useful
                    optional.append(name)
                    reasons[name] = "Available but not auto-triggered"
                else:
                    skipped.append(name)
                    reasons[name] = "No conditions met"

        # Determine execution order
        execution_order = self._determine_order(required + optional)

        # Calculate estimates
        total_time = sum(
            self.registry[p].estimated_time_ms
            for p in execution_order
        )
        max_memory = max(
            (self.registry[p].memory_mb for p in execution_order),
            default=0
        )

        return PistonSelection(
            required_pistons=required,
            optional_pistons=optional,
            skipped_pistons=skipped,
            execution_order=execution_order,
            selection_reasons=reasons,
            estimated_total_time_ms=total_time,
            estimated_memory_mb=max_memory
        )

    def _get_preference_key(self, piston_name: str) -> str:
        """Get the user preference key for a piston"""
        mapping = {
            'thermal': 'run_thermal_analysis',
            'pdn': 'run_pdn_analysis',
            'signal_integrity': 'run_si_analysis',
            'topological_router': 'use_topological_routing',
            'visualization_3d': 'generate_3d',
            'bom_optimizer': 'optimize_bom',
            'learning': 'enable_learning',
            'silkscreen': 'generate_silkscreen',
            'stackup': 'run_stackup_analysis',
            'pour': 'generate_gnd_pour',  # GND pour on bottom layer
        }
        return mapping.get(piston_name, f'run_{piston_name}')

    def _check_conditions(self, spec: PistonSpec, profile: DesignProfile) -> tuple:
        """Check if auto-enable conditions are met"""
        for condition in spec.auto_enable_conditions:
            # Parse condition
            if condition == 'has_bga':
                if profile.has_bga:
                    return True, "BGA package detected"
            elif condition == 'has_qfn':
                if profile.has_qfn:
                    return True, "QFN package detected"
            elif condition == 'has_fine_pitch':
                if profile.has_fine_pitch:
                    return True, "Fine-pitch components detected"
            elif condition == 'has_high_power':
                if profile.has_high_power:
                    return True, f"High power ({profile.total_power_w:.1f}W)"
            elif condition == 'has_motor_driver':
                if profile.has_motor_driver:
                    return True, "Motor driver detected"
            elif condition == 'total_power_5w+':
                if profile.total_power_w >= 5.0:
                    return True, f"Power > 5W ({profile.total_power_w:.1f}W)"
            elif condition == 'has_high_speed':
                if profile.has_high_speed:
                    return True, f"High-speed signals ({profile.max_frequency_mhz:.0f}MHz)"
            elif condition == 'has_ddr':
                if profile.has_ddr:
                    return True, "DDR memory detected"
            elif condition == 'has_usb3':
                if profile.has_usb3:
                    return True, "USB 3.0 detected"
            elif condition == 'has_pcie':
                if profile.has_pcie:
                    return True, "PCIe detected"
            elif condition == 'has_sensitive_analog':
                if profile.has_sensitive_analog:
                    return True, "Sensitive analog circuits detected"
            elif condition == 'has_differential_pairs':
                if profile.has_differential_pairs:
                    return True, "Differential pairs detected"
            elif condition == 'layer_count_4+':
                if profile.layer_count >= 4:
                    return True, f"{profile.layer_count} layer board"
            elif condition == 'routing_complete_80+':
                # This is checked at runtime
                return False, None
            elif condition == 'complex_routing':
                # Heuristic: many nets or high density
                if profile.component_count > 50:
                    return True, "Complex design (50+ components)"
            # === POUR PISTON CONDITIONS ===
            elif condition == 'layer_count_2':
                if profile.layer_count == 2:
                    return True, "2-layer board (GND pour recommended)"
            elif condition == 'has_gnd_crossings':
                # This requires routing analysis - checked at runtime
                return False, None
            elif condition == 'has_many_gnd_pins':
                # Checked during parts analysis
                if getattr(profile, 'gnd_pin_count', 0) >= 5:
                    return True, f"Many GND pins ({profile.gnd_pin_count})"

        return False, None

    def _determine_order(self, pistons: List[str]) -> List[str]:
        """Determine optimal execution order based on dependencies"""
        ordered = []
        remaining = set(pistons)

        # Keep adding pistons whose dependencies are satisfied
        max_iterations = len(pistons) * 2
        iteration = 0

        while remaining and iteration < max_iterations:
            iteration += 1
            added_this_round = False

            for piston in list(remaining):
                spec = self.registry.get(piston)
                if not spec:
                    remaining.discard(piston)
                    continue

                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    dep in ordered or dep not in remaining
                    for dep in spec.requires
                )

                # Check for conflicts
                has_conflict = any(
                    conflict in ordered
                    for conflict in spec.conflicts
                )

                if deps_satisfied and not has_conflict:
                    ordered.append(piston)
                    remaining.discard(piston)
                    added_this_round = True

            if not added_this_round and remaining:
                # Circular dependency or missing dependency
                # Just add remaining in registry order
                for piston in remaining:
                    ordered.append(piston)
                remaining.clear()

        return ordered

    def get_piston_info(self, piston_name: str) -> Optional[PistonSpec]:
        """Get information about a specific piston"""
        return self.registry.get(piston_name)

    def list_all_pistons(self) -> Dict[str, Dict]:
        """List all pistons with their info"""
        result = {}
        for name, spec in self.registry.items():
            result[name] = {
                'category': spec.category.value,
                'priority': spec.priority.value,
                'description': spec.description,
                'requires': spec.requires,
                'auto_conditions': spec.auto_enable_conditions,
                'estimated_time_ms': spec.estimated_time_ms,
            }
        return result

    def explain_selection(self, selection: PistonSelection) -> str:
        """Generate human-readable explanation of selection"""
        lines = []
        lines.append("=" * 60)
        lines.append("PISTON SELECTION SUMMARY")
        lines.append("=" * 60)

        lines.append(f"\nRequired Pistons ({len(selection.required_pistons)}):")
        for p in selection.required_pistons:
            reason = selection.selection_reasons.get(p, '')
            lines.append(f"  [*] {p:20} - {reason}")

        lines.append(f"\nOptional Pistons ({len(selection.optional_pistons)}):")
        for p in selection.optional_pistons:
            reason = selection.selection_reasons.get(p, '')
            lines.append(f"  [?] {p:20} - {reason}")

        lines.append(f"\nSkipped Pistons ({len(selection.skipped_pistons)}):")
        for p in selection.skipped_pistons:
            reason = selection.selection_reasons.get(p, '')
            lines.append(f"  [ ] {p:20} - {reason}")

        lines.append(f"\nExecution Order:")
        for i, p in enumerate(selection.execution_order, 1):
            lines.append(f"  {i:2}. {p}")

        lines.append(f"\nEstimates:")
        lines.append(f"  Time: ~{selection.estimated_total_time_ms / 1000:.1f}s")
        lines.append(f"  Memory: ~{selection.estimated_memory_mb}MB")

        return '\n'.join(lines)


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def select_pistons_for_design(
    parts_db: Dict,
    requirements: Any = None,
    **user_preferences
) -> PistonSelection:
    """
    Quick function to select pistons for a design.

    Example:
        selection = select_pistons_for_design(
            parts_db=my_parts,
            generate_3d=True,
            run_thermal_analysis=True
        )

        print(f"Will run: {selection.execution_order}")
    """
    orchestrator = PistonOrchestrator()
    return orchestrator.select_pistons(
        parts_db=parts_db,
        requirements=requirements,
        user_preferences=user_preferences
    )


def get_minimal_pistons() -> List[str]:
    """Get list of minimum required pistons"""
    return ['parts', 'order', 'placement', 'routing', 'drc', 'output']


def get_full_pistons() -> List[str]:
    """Get list of all pistons for maximum analysis"""
    return list(PISTON_REGISTRY.keys())


# =============================================================================
# SELF TEST
# =============================================================================

if __name__ == '__main__':
    # Test with a sample design
    sample_parts_db = {
        'parts': {
            'U1': {'value': 'ESP32-WROOM-32', 'footprint': 'ESP32-WROOM-32'},
            'U2': {'value': 'DRV8833', 'footprint': 'HTSSOP-16'},
            'U3': {'value': 'BME280', 'footprint': 'LGA-8'},
            'J1': {'value': 'USB-C', 'footprint': 'USB_C_Receptacle'},
        },
        'nets': {
            'VCC': {'pins': ['U1.VCC', 'U2.VCC', 'U3.VCC']},
            'GND': {'pins': ['U1.GND', 'U2.GND', 'U3.GND']},
            'SDA': {'pins': ['U1.SDA', 'U3.SDA']},
            'SCL': {'pins': ['U1.SCL', 'U3.SCL']},
            'MOTOR_A_P': {'pins': ['U2.OUT1']},
            'MOTOR_A_N': {'pins': ['U2.OUT2']},
        }
    }

    print("Testing Piston Orchestrator")
    print("=" * 60)

    orchestrator = PistonOrchestrator()

    # Test 1: Basic selection
    print("\n[Test 1] Basic selection (no preferences)")
    selection = orchestrator.select_pistons(sample_parts_db)
    print(orchestrator.explain_selection(selection))

    # Test 2: With user preferences
    print("\n[Test 2] With user preferences (3D + thermal)")
    selection = orchestrator.select_pistons(
        sample_parts_db,
        user_preferences={
            'generate_3d': True,
            'run_thermal_analysis': True,
        }
    )
    print(orchestrator.explain_selection(selection))

    # Test 3: Force disable
    print("\n[Test 3] Force disable optimize and silkscreen")
    selection = orchestrator.select_pistons(
        sample_parts_db,
        force_disable=['optimize', 'silkscreen']
    )
    print(f"Execution order: {selection.execution_order}")

    print("\n[PASS] All tests completed!")
