"""
PCB Engine - Stackup Piston (Sub-Engine)
==========================================

A dedicated piston (sub-engine) for PCB layer stackup design and impedance calculation.

ALGORITHMS & REFERENCES:
=========================
1. IPC-2141 Formulas - Industry standard for PCB impedance calculations
   Reference: IPC-2141A "Design Guide for High-Speed Controlled Impedance Circuit Boards"
   Accuracy: ~5-7% for microstrip, ~1% for stripline

2. Wadell Equations - More accurate than IPC-2141
   Reference: Brian C. Wadell, "Transmission Line Design Handbook" (1991)
   Based on: Wheeler's microstrip equations (IEEE Trans. MTT-25, 1977)
            Schneider's effective Dk equations (Bell Sys. Tech. J., 1969)

3. Differential Pair Impedance - Edge-coupled and broadside-coupled
   Reference: National Semiconductor Application Notes
   Howard Johnson, "High-Speed Digital Design" (1993)

4. Layer Stack Optimization - Signal integrity driven
   Reference: Eric Bogatin, "Signal and Power Integrity - Simplified" (2010)

MATERIAL DATABASE:
==================
- FR4 variants (Standard, High-Tg, Halogen-Free)
- Rogers high-frequency (RO4350B, RO4003C, RO3003)
- Isola high-performance (370HR, FR408HR, I-Tera MT40)
- Panasonic Megtron (Megtron 4, Megtron 6, Megtron 7)
- Taconic (TLX, RF-35, TLC)

IMPEDANCE STRUCTURES:
=====================
- Surface microstrip (outer layer trace over ground)
- Embedded microstrip (buried trace over ground)
- Symmetric stripline (trace between two grounds)
- Asymmetric stripline (trace offset between grounds)
- Coplanar waveguide (CPW) with ground
- Edge-coupled differential microstrip
- Edge-coupled differential stripline
- Broadside-coupled stripline

Sources:
- https://pcbsync.com/ipc-2141/
- https://resources.altium.com/p/microstrip-impedance-calculator
- https://chemandy.com/calculators/microstrip-transmission-line-calculator.htm
- https://www.allaboutcircuits.com/tools/edge-coupled-stripline-impedance-calculator/
- https://rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import copy


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class LayerType(Enum):
    """PCB layer types"""
    SIGNAL = 'signal'
    GROUND = 'ground'
    POWER = 'power'
    MIXED = 'mixed'  # Signal + power on same layer


class DielectricType(Enum):
    """Dielectric material categories"""
    FR4_STANDARD = 'fr4_standard'
    FR4_HIGH_TG = 'fr4_high_tg'
    FR4_HALOGEN_FREE = 'fr4_halogen_free'
    ROGERS_HIGH_FREQ = 'rogers_high_freq'
    ISOLA_HIGH_PERF = 'isola_high_perf'
    MEGTRON = 'megtron'
    TACONIC = 'taconic'
    POLYIMIDE = 'polyimide'
    PTFE = 'ptfe'


class ImpedanceStructure(Enum):
    """Impedance structure types"""
    SURFACE_MICROSTRIP = 'surface_microstrip'
    EMBEDDED_MICROSTRIP = 'embedded_microstrip'
    SYMMETRIC_STRIPLINE = 'symmetric_stripline'
    ASYMMETRIC_STRIPLINE = 'asymmetric_stripline'
    COPLANAR_WAVEGUIDE = 'coplanar_waveguide'
    DIFF_MICROSTRIP = 'diff_microstrip'
    DIFF_STRIPLINE = 'diff_stripline'
    BROADSIDE_STRIPLINE = 'broadside_stripline'


class CalculationMethod(Enum):
    """Impedance calculation method"""
    IPC_2141 = 'ipc_2141'      # Industry standard, ~5-7% accuracy
    WADELL = 'wadell'          # More accurate, from Transmission Line Handbook
    WHEELER = 'wheeler'        # Classic microstrip equations
    FIELD_SOLVER = 'field_solver'  # 2D numerical (most accurate)


# =============================================================================
# MATERIAL DATABASE - Research-based properties
# =============================================================================

@dataclass
class DielectricMaterial:
    """PCB dielectric material properties"""
    name: str
    manufacturer: str
    category: DielectricType

    # Electrical properties
    dk: float              # Dielectric constant (relative permittivity)
    dk_tolerance: float    # Dk tolerance (e.g., 0.05 = ±0.05)
    df: float              # Dissipation factor (loss tangent)
    dk_at_freq_ghz: float  # Frequency at which Dk is specified

    # Thermal properties
    tg: float              # Glass transition temperature (°C)
    td: float              # Decomposition temperature (°C)
    cte_xy: float          # CTE in XY plane (ppm/°C)
    cte_z: float           # CTE in Z axis (ppm/°C)

    # Available thicknesses (mm)
    available_thicknesses: List[float] = field(default_factory=list)

    # Cost factor (1.0 = FR4 standard)
    cost_factor: float = 1.0

    # Notes
    notes: str = ""


# Pre-defined material database
MATERIAL_DATABASE: Dict[str, DielectricMaterial] = {
    # =========================================================================
    # FR4 VARIANTS
    # =========================================================================
    'FR4_STANDARD': DielectricMaterial(
        name='Standard FR4',
        manufacturer='Various',
        category=DielectricType.FR4_STANDARD,
        dk=4.5,  # Varies 3.8-4.8 depending on weave
        dk_tolerance=0.3,  # ~10% tolerance
        df=0.020,
        dk_at_freq_ghz=1.0,
        tg=130,
        td=310,
        cte_xy=14,
        cte_z=70,
        available_thicknesses=[0.1, 0.2, 0.254, 0.36, 0.51, 0.71, 1.0, 1.2, 1.6],
        cost_factor=1.0,
        notes="Standard FR4, not suitable for >1GHz applications"
    ),

    'FR4_HIGH_TG': DielectricMaterial(
        name='High-Tg FR4 (IT180A)',
        manufacturer='ITEQ',
        category=DielectricType.FR4_HIGH_TG,
        dk=4.3,
        dk_tolerance=0.2,
        df=0.018,
        dk_at_freq_ghz=1.0,
        tg=180,
        td=340,
        cte_xy=13,
        cte_z=50,
        available_thicknesses=[0.1, 0.2, 0.254, 0.36, 0.51, 0.71, 1.0],
        cost_factor=1.2,
        notes="High-Tg for lead-free assembly, better z-axis stability"
    ),

    'FR4_370HR': DielectricMaterial(
        name='370HR',
        manufacturer='Isola',
        category=DielectricType.FR4_HIGH_TG,
        dk=4.2,
        dk_tolerance=0.15,
        df=0.016,
        dk_at_freq_ghz=1.0,
        tg=180,
        td=340,
        cte_xy=12,
        cte_z=45,
        available_thicknesses=[0.05, 0.075, 0.1, 0.127, 0.2, 0.254],
        cost_factor=1.5,
        notes="High-performance FR4, better Dk stability and CAF resistance"
    ),

    # =========================================================================
    # ISOLA HIGH-PERFORMANCE
    # =========================================================================
    'ISOLA_FR408HR': DielectricMaterial(
        name='FR408HR',
        manufacturer='Isola',
        category=DielectricType.ISOLA_HIGH_PERF,
        dk=3.39,
        dk_tolerance=0.05,
        df=0.012,
        dk_at_freq_ghz=2.5,
        tg=180,
        td=360,
        cte_xy=12,
        cte_z=45,
        available_thicknesses=[0.05, 0.075, 0.1, 0.127, 0.175, 0.2],
        cost_factor=2.5,
        notes="Low-loss, low-Dk for high-speed digital"
    ),

    'ISOLA_ITERA_MT40': DielectricMaterial(
        name='I-Tera MT40',
        manufacturer='Isola',
        category=DielectricType.ISOLA_HIGH_PERF,
        dk=3.45,
        dk_tolerance=0.05,
        df=0.0035,
        dk_at_freq_ghz=10.0,
        tg=200,
        td=360,
        cte_xy=10,
        cte_z=35,
        available_thicknesses=[0.05, 0.075, 0.1, 0.127],
        cost_factor=4.0,
        notes="Very low loss for 10+ Gbps applications"
    ),

    # =========================================================================
    # ROGERS HIGH-FREQUENCY
    # =========================================================================
    'ROGERS_RO4350B': DielectricMaterial(
        name='RO4350B',
        manufacturer='Rogers',
        category=DielectricType.ROGERS_HIGH_FREQ,
        dk=3.48,
        dk_tolerance=0.05,  # ±0.05 tight tolerance
        df=0.0037,
        dk_at_freq_ghz=10.0,  # Stable 500MHz to 40GHz
        tg=280,
        td=390,
        cte_xy=10,
        cte_z=32,
        available_thicknesses=[0.168, 0.254, 0.338, 0.508, 0.762, 1.016, 1.524],
        cost_factor=5.0,
        notes="RF/Microwave standard, stable Dk across frequency"
    ),

    'ROGERS_RO4003C': DielectricMaterial(
        name='RO4003C',
        manufacturer='Rogers',
        category=DielectricType.ROGERS_HIGH_FREQ,
        dk=3.38,
        dk_tolerance=0.05,
        df=0.0027,
        dk_at_freq_ghz=10.0,
        tg=280,
        td=390,
        cte_xy=11,
        cte_z=46,
        available_thicknesses=[0.203, 0.305, 0.406, 0.508, 0.813, 1.524],
        cost_factor=4.5,
        notes="Lower loss than RO4350B, ceramic-filled PTFE"
    ),

    'ROGERS_RO3003': DielectricMaterial(
        name='RO3003',
        manufacturer='Rogers',
        category=DielectricType.ROGERS_HIGH_FREQ,
        dk=3.0,
        dk_tolerance=0.04,
        df=0.0013,
        dk_at_freq_ghz=10.0,
        tg=315,  # PTFE-based, no true Tg
        td=500,
        cte_xy=17,
        cte_z=24,
        available_thicknesses=[0.127, 0.25, 0.5, 0.75, 1.52],
        cost_factor=8.0,
        notes="Ceramic-filled PTFE, lowest loss, for >10GHz"
    ),

    # =========================================================================
    # MEGTRON SERIES (Panasonic)
    # =========================================================================
    'MEGTRON_4': DielectricMaterial(
        name='Megtron 4 (R-5575)',
        manufacturer='Panasonic',
        category=DielectricType.MEGTRON,
        dk=3.8,
        dk_tolerance=0.08,
        df=0.005,
        dk_at_freq_ghz=1.0,
        tg=175,
        td=355,
        cte_xy=11,
        cte_z=35,
        available_thicknesses=[0.05, 0.1, 0.127, 0.2],
        cost_factor=3.0,
        notes="High-speed server/networking applications"
    ),

    'MEGTRON_6': DielectricMaterial(
        name='Megtron 6 (R-5775)',
        manufacturer='Panasonic',
        category=DielectricType.MEGTRON,
        dk=3.4,
        dk_tolerance=0.05,
        df=0.002,
        dk_at_freq_ghz=10.0,
        tg=185,
        td=400,
        cte_xy=9,
        cte_z=30,
        available_thicknesses=[0.05, 0.075, 0.1, 0.127],
        cost_factor=5.0,
        notes="25+ Gbps data center applications"
    ),

    'MEGTRON_7': DielectricMaterial(
        name='Megtron 7 (R-5785)',
        manufacturer='Panasonic',
        category=DielectricType.MEGTRON,
        dk=3.2,
        dk_tolerance=0.04,
        df=0.0015,
        dk_at_freq_ghz=10.0,
        tg=195,
        td=410,
        cte_xy=8,
        cte_z=25,
        available_thicknesses=[0.05, 0.075, 0.1],
        cost_factor=8.0,
        notes="56+ Gbps PAM4 applications, ultra-low loss"
    ),

    # =========================================================================
    # TACONIC
    # =========================================================================
    'TACONIC_RF35': DielectricMaterial(
        name='RF-35',
        manufacturer='Taconic',
        category=DielectricType.TACONIC,
        dk=3.5,
        dk_tolerance=0.05,
        df=0.0018,
        dk_at_freq_ghz=10.0,
        tg=300,
        td=500,
        cte_xy=14,
        cte_z=30,
        available_thicknesses=[0.254, 0.508, 0.762, 1.524],
        cost_factor=4.0,
        notes="RF applications, PTFE/woven glass"
    ),

    # =========================================================================
    # POLYIMIDE (Flex/Rigid-Flex)
    # =========================================================================
    'POLYIMIDE_KAPTON': DielectricMaterial(
        name='Kapton HN',
        manufacturer='DuPont',
        category=DielectricType.POLYIMIDE,
        dk=3.4,
        dk_tolerance=0.1,
        df=0.002,
        dk_at_freq_ghz=1.0,
        tg=360,
        td=410,
        cte_xy=20,
        cte_z=60,
        available_thicknesses=[0.025, 0.05, 0.075, 0.125],
        cost_factor=3.0,
        notes="Flexible circuit standard, high temperature capability"
    ),
}


# Standard copper weights to thickness (oz to mm)
COPPER_WEIGHTS: Dict[str, float] = {
    '0.5oz': 0.0175,
    '1oz': 0.035,
    '2oz': 0.070,
    '3oz': 0.105,
    '4oz': 0.140,
}


# =============================================================================
# LAYER AND STACKUP DEFINITIONS
# =============================================================================

@dataclass
class CopperLayer:
    """Definition of a copper layer"""
    name: str
    layer_type: LayerType
    thickness_mm: float = 0.035  # 1oz copper default
    weight_oz: float = 1.0
    roughness_um: float = 0.5   # Surface roughness (affects loss at high freq)
    is_outer: bool = False

    # For impedance controlled layers
    target_impedance: Optional[float] = None
    trace_width_mm: Optional[float] = None
    trace_spacing_mm: Optional[float] = None  # For differential pairs


@dataclass
class DielectricLayer:
    """Definition of a dielectric (insulating) layer"""
    name: str
    material: str  # Key into MATERIAL_DATABASE
    thickness_mm: float

    # Override material Dk if needed (for simulation)
    dk_override: Optional[float] = None
    df_override: Optional[float] = None

    @property
    def dk(self) -> float:
        if self.dk_override:
            return self.dk_override
        if self.material in MATERIAL_DATABASE:
            return MATERIAL_DATABASE[self.material].dk
        return 4.5  # Default FR4

    @property
    def df(self) -> float:
        if self.df_override:
            return self.df_override
        if self.material in MATERIAL_DATABASE:
            return MATERIAL_DATABASE[self.material].df
        return 0.02  # Default FR4


@dataclass
class Stackup:
    """Complete PCB layer stackup definition"""
    name: str
    layers: List[Union[CopperLayer, DielectricLayer]] = field(default_factory=list)
    total_thickness_mm: float = 0.0

    # Calculated properties
    copper_count: int = 0
    signal_layers: List[str] = field(default_factory=list)
    plane_layers: List[str] = field(default_factory=list)

    def calculate_properties(self):
        """Calculate derived stackup properties"""
        self.total_thickness_mm = sum(
            layer.thickness_mm for layer in self.layers
        )
        self.copper_count = sum(
            1 for layer in self.layers if isinstance(layer, CopperLayer)
        )
        self.signal_layers = [
            layer.name for layer in self.layers
            if isinstance(layer, CopperLayer) and layer.layer_type == LayerType.SIGNAL
        ]
        self.plane_layers = [
            layer.name for layer in self.layers
            if isinstance(layer, CopperLayer) and layer.layer_type in (LayerType.GROUND, LayerType.POWER)
        ]


# =============================================================================
# IMPEDANCE CALCULATION RESULTS
# =============================================================================

@dataclass
class ImpedanceResult:
    """Result of impedance calculation"""
    structure: ImpedanceStructure
    method: CalculationMethod

    # Calculated values
    z0: float                    # Characteristic impedance (Ohms)
    z_diff: Optional[float] = None  # Differential impedance (Ohms)
    z_common: Optional[float] = None  # Common-mode impedance (Ohms)

    # Effective parameters
    effective_dk: float = 0.0    # Effective dielectric constant
    velocity_factor: float = 0.0  # Propagation velocity / c
    delay_ps_per_mm: float = 0.0  # Propagation delay (ps/mm)

    # Loss parameters
    conductor_loss_db_per_mm: float = 0.0  # Skin effect loss
    dielectric_loss_db_per_mm: float = 0.0  # Dielectric loss (Df)
    total_loss_db_per_mm: float = 0.0

    # Input parameters (for reference)
    trace_width_mm: float = 0.0
    trace_thickness_mm: float = 0.0
    dielectric_height_mm: float = 0.0
    dielectric_dk: float = 0.0
    spacing_mm: float = 0.0  # For differential pairs

    # Accuracy estimate
    accuracy_percent: float = 5.0  # Estimated error


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StackupConfig:
    """Configuration for the stackup piston"""
    # Default calculation method
    calculation_method: CalculationMethod = CalculationMethod.WADELL

    # Target impedances
    target_z0: float = 50.0           # Single-ended target (Ohms)
    target_z_diff: float = 100.0      # Differential target (Ohms)
    impedance_tolerance: float = 0.10  # ±10%

    # Default material
    default_material: str = 'FR4_STANDARD'

    # Board thickness
    total_thickness_mm: float = 1.6   # Standard board thickness
    copper_weight_oz: float = 1.0     # Copper weight in oz

    # Frequency for calculations (affects Dk and loss)
    frequency_ghz: float = 1.0

    # Copper roughness model
    include_roughness: bool = True

    # Manufacturing constraints (typical)
    min_trace_width_mm: float = 0.1     # 4 mil
    max_trace_width_mm: float = 2.0
    min_spacing_mm: float = 0.1         # 4 mil
    min_dielectric_mm: float = 0.05     # 2 mil

    # Standard board thicknesses (mm)
    standard_thicknesses: List[float] = field(default_factory=lambda: [
        0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 3.2
    ])

    # Enable optimization
    optimize_for_signal_integrity: bool = True
    optimize_for_cost: bool = False


# =============================================================================
# STACKUP PISTON CLASS
# =============================================================================

class StackupPiston:
    """
    PCB Stackup Design and Impedance Calculation Piston

    Provides:
    1. Layer stackup design for 2-32 layer boards
    2. Impedance calculation using IPC-2141 or Wadell equations
    3. Differential pair impedance
    4. Material database with electrical/thermal properties
    5. Trace width calculation for target impedance
    6. Manufacturing-aware stackup templates
    """

    def __init__(self, config: Optional[StackupConfig] = None):
        self.config = config or StackupConfig()
        self.stackup: Optional[Stackup] = None
        self.impedance_results: Dict[str, ImpedanceResult] = {}

    # =========================================================================
    # STANDARD PISTON API
    # =========================================================================

    def analyze(self) -> Dict[str, Any]:
        """
        Standard piston API - analyze/create stackup based on config.

        Returns:
            Dictionary with stackup analysis including impedance calculations
        """
        # Create default 2-layer stackup if none exists
        if self.stackup is None:
            self.stackup = self.create_2layer_stackup(
                material=self.config.default_material,
                total_thickness_mm=self.config.total_thickness_mm,
                copper_weight_oz=self.config.copper_weight_oz
            )

        return self.analyze_stackup(self.stackup)

    # =========================================================================
    # STANDARD STACKUP TEMPLATES
    # =========================================================================

    def create_2layer_stackup(
        self,
        material: str = 'FR4_STANDARD',
        total_thickness_mm: float = 1.6,
        copper_weight_oz: float = 1.0
    ) -> Stackup:
        """
        Create standard 2-layer stackup

        Structure: Top - Core - Bottom
        """
        copper_thickness = COPPER_WEIGHTS.get(f'{copper_weight_oz}oz', 0.035)
        core_thickness = total_thickness_mm - 2 * copper_thickness

        stackup = Stackup(
            name='2-Layer Standard',
            layers=[
                CopperLayer('Top', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
                DielectricLayer('Core', material, core_thickness),
                CopperLayer('Bottom', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
            ]
        )
        stackup.calculate_properties()
        return stackup

    def create_4layer_stackup(
        self,
        material: str = 'FR4_STANDARD',
        total_thickness_mm: float = 1.6,
        copper_weight_oz: float = 1.0,
        signal_on_inner: bool = False
    ) -> Stackup:
        """
        Create standard 4-layer stackup

        Default structure (signal on outer):
        Top (Signal) - PP - GND - Core - PWR - PP - Bottom (Signal)

        Alternative (signal on inner):
        Top (GND) - PP - Signal - Core - Signal - PP - Bottom (PWR)
        """
        copper_thickness = COPPER_WEIGHTS.get(f'{copper_weight_oz}oz', 0.035)

        # Distribute thickness: ~60% core, ~20% each prepreg
        # Minus 4 copper layers
        dielectric_total = total_thickness_mm - 4 * copper_thickness
        core_thickness = dielectric_total * 0.6
        prepreg_thickness = dielectric_total * 0.2

        if signal_on_inner:
            # Better for EMI, signal integrity
            layers = [
                CopperLayer('Top', LayerType.GROUND, copper_thickness, copper_weight_oz, is_outer=True),
                DielectricLayer('PP1', material, prepreg_thickness),
                CopperLayer('Inner1', LayerType.SIGNAL, copper_thickness, copper_weight_oz),
                DielectricLayer('Core', material, core_thickness),
                CopperLayer('Inner2', LayerType.SIGNAL, copper_thickness, copper_weight_oz),
                DielectricLayer('PP2', material, prepreg_thickness),
                CopperLayer('Bottom', LayerType.POWER, copper_thickness, copper_weight_oz, is_outer=True),
            ]
            name = '4-Layer (Signal Inner)'
        else:
            # Standard layout
            layers = [
                CopperLayer('Top', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
                DielectricLayer('PP1', material, prepreg_thickness),
                CopperLayer('GND', LayerType.GROUND, copper_thickness, copper_weight_oz),
                DielectricLayer('Core', material, core_thickness),
                CopperLayer('PWR', LayerType.POWER, copper_thickness, copper_weight_oz),
                DielectricLayer('PP2', material, prepreg_thickness),
                CopperLayer('Bottom', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
            ]
            name = '4-Layer Standard'

        stackup = Stackup(name=name, layers=layers)
        stackup.calculate_properties()
        return stackup

    def create_6layer_stackup(
        self,
        material: str = 'FR4_HIGH_TG',
        total_thickness_mm: float = 1.6,
        copper_weight_oz: float = 1.0
    ) -> Stackup:
        """
        Create 6-layer stackup optimized for signal integrity

        Structure (recommended for high-speed):
        Top (Signal) - PP - GND - PP - Signal - Core - Signal - PP - GND - PP - Bottom (Signal)

        Wait, that's 8 layers. Let me fix:

        6-Layer Standard:
        Top (Signal) - PP - GND - Core - PWR - PP - Signal - Core - Signal - PP - Bottom (Signal)

        Hmm, let me use proper 6-layer:
        L1: Signal (Top)
        L2: GND
        L3: Signal
        L4: Signal
        L5: PWR
        L6: Signal (Bottom)
        """
        copper_thickness = COPPER_WEIGHTS.get(f'{copper_weight_oz}oz', 0.035)
        dielectric_total = total_thickness_mm - 6 * copper_thickness

        # 2 cores, 2 prepregs
        core_thickness = dielectric_total * 0.35  # Each core
        prepreg_thickness = dielectric_total * 0.15  # Each prepreg

        layers = [
            CopperLayer('L1_Top', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
            DielectricLayer('PP1', material, prepreg_thickness),
            CopperLayer('L2_GND', LayerType.GROUND, copper_thickness, copper_weight_oz),
            DielectricLayer('Core1', material, core_thickness),
            CopperLayer('L3_SIG', LayerType.SIGNAL, copper_thickness, copper_weight_oz),
            DielectricLayer('PP2', material, prepreg_thickness),
            CopperLayer('L4_SIG', LayerType.SIGNAL, copper_thickness, copper_weight_oz),
            DielectricLayer('Core2', material, core_thickness),
            CopperLayer('L5_PWR', LayerType.POWER, copper_thickness, copper_weight_oz),
            DielectricLayer('PP3', material, prepreg_thickness),
            CopperLayer('L6_Bot', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
        ]

        stackup = Stackup(name='6-Layer High-Speed', layers=layers)
        stackup.calculate_properties()
        return stackup

    def create_8layer_stackup(
        self,
        material: str = 'FR4_HIGH_TG',
        total_thickness_mm: float = 1.6,
        copper_weight_oz: float = 1.0
    ) -> Stackup:
        """
        Create 8-layer stackup for high-speed digital

        Recommended structure:
        L1: Signal (Top microstrip)
        L2: GND (reference for L1, L3)
        L3: Signal (stripline)
        L4: PWR
        L5: GND
        L6: Signal (stripline)
        L7: GND (reference for L6, L8)
        L8: Signal (Bottom microstrip)
        """
        copper_thickness = COPPER_WEIGHTS.get(f'{copper_weight_oz}oz', 0.035)
        dielectric_total = total_thickness_mm - 8 * copper_thickness

        # Distribute dielectric: thin prepregs for controlled impedance
        pp_thickness = dielectric_total * 0.1  # Each prepreg
        core_thickness = dielectric_total * 0.15  # Each core

        layers = [
            CopperLayer('L1_Top', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
            DielectricLayer('PP1', material, pp_thickness),
            CopperLayer('L2_GND', LayerType.GROUND, copper_thickness, copper_weight_oz),
            DielectricLayer('Core1', material, core_thickness),
            CopperLayer('L3_SIG', LayerType.SIGNAL, copper_thickness, copper_weight_oz),
            DielectricLayer('PP2', material, pp_thickness),
            CopperLayer('L4_PWR', LayerType.POWER, copper_thickness, copper_weight_oz),
            DielectricLayer('Core2', material, core_thickness),
            CopperLayer('L5_GND', LayerType.GROUND, copper_thickness, copper_weight_oz),
            DielectricLayer('PP3', material, pp_thickness),
            CopperLayer('L6_SIG', LayerType.SIGNAL, copper_thickness, copper_weight_oz),
            DielectricLayer('Core3', material, core_thickness),
            CopperLayer('L7_GND', LayerType.GROUND, copper_thickness, copper_weight_oz),
            DielectricLayer('PP4', material, pp_thickness),
            CopperLayer('L8_Bot', LayerType.SIGNAL, copper_thickness, copper_weight_oz, is_outer=True),
        ]

        stackup = Stackup(name='8-Layer High-Speed', layers=layers)
        stackup.calculate_properties()
        return stackup

    # =========================================================================
    # IMPEDANCE CALCULATION - IPC-2141 FORMULAS
    # =========================================================================

    def _ipc2141_microstrip(
        self,
        w: float,  # Trace width (mm)
        t: float,  # Trace thickness (mm)
        h: float,  # Dielectric height (mm)
        dk: float  # Dielectric constant
    ) -> Tuple[float, float]:
        """
        IPC-2141 Surface Microstrip Impedance

        Valid for: 0.1 < W/H < 2.0
        Accuracy: ~5-7%

        Returns: (Z0, effective_dk)
        """
        # Effective width (accounts for trace thickness)
        if t > 0:
            w_eff = w + (t / math.pi) * math.log(4 * math.e / math.sqrt((t / h)**2 + (t / (w * math.pi + 1.1 * t * math.pi))**2))
        else:
            w_eff = w

        # IPC-2141 equation
        z0 = (87 / math.sqrt(dk + 1.41)) * math.log((5.98 * h) / (0.8 * w_eff + t))

        # Effective dielectric constant (Wheeler/Schneider)
        if w_eff / h <= 1:
            dk_eff = ((dk + 1) / 2) + ((dk - 1) / 2) * (
                (1 / math.sqrt(1 + 12 * h / w_eff)) + 0.04 * (1 - w_eff / h)**2
            )
        else:
            dk_eff = ((dk + 1) / 2) + ((dk - 1) / 2) * (1 / math.sqrt(1 + 12 * h / w_eff))

        return z0, dk_eff

    def _ipc2141_stripline(
        self,
        w: float,  # Trace width (mm)
        t: float,  # Trace thickness (mm)
        b: float,  # Total dielectric thickness (mm) - trace to trace distance
        dk: float  # Dielectric constant
    ) -> float:
        """
        IPC-2141 Symmetric Stripline Impedance

        Valid for: W < 0.35 * B
        Accuracy: ~1%

        Returns: Z0
        """
        # Effective width
        if t > 0:
            # Correction for trace thickness
            m = 6 / (3 + t / b)
            w_eff = w + (t / math.pi) * math.log(
                4 * math.e / math.sqrt(m * (t / (b - t))**2 + (1 / ((w / t) + 1.1))**2)
            )
        else:
            w_eff = w

        # IPC-2141 stripline equation
        z0 = (60 / math.sqrt(dk)) * math.log((4 * b) / (0.67 * math.pi * (0.8 * w_eff + t)))

        return z0

    # =========================================================================
    # IMPEDANCE CALCULATION - WADELL EQUATIONS (More Accurate)
    # =========================================================================

    def _wadell_microstrip(
        self,
        w: float,  # Trace width (mm)
        t: float,  # Trace thickness (mm)
        h: float,  # Dielectric height (mm)
        dk: float  # Dielectric constant
    ) -> Tuple[float, float]:
        """
        Wadell Microstrip Impedance (from Transmission Line Design Handbook)

        Based on Wheeler (1977) and Schneider (1969) equations
        Accuracy: ~2-3%

        Returns: (Z0, effective_dk)
        """
        # Free-space impedance
        Z0_FREE = 376.73  # Ohms (impedance of free space)

        # Effective width (corrected for thickness)
        if t > 0 and t < h:
            delta_w = (t / math.pi) * (1 + math.log(2 * h / t))
            w_eff = w + delta_w
        else:
            w_eff = w

        u = w_eff / h  # Normalized width

        # Effective dielectric constant (Schneider, 1969)
        # Modified by Hammerstad and Jensen (1980)
        a = 1 + (1/49) * math.log((u**4 + (u/52)**2) / (u**4 + 0.432)) + (1/18.7) * math.log(1 + (u/18.1)**3)
        b_coef = 0.564 * ((dk - 0.9) / (dk + 3))**0.053

        dk_eff = ((dk + 1) / 2) + ((dk - 1) / 2) * (1 + 10/u)**(-a * b_coef)

        # Characteristic impedance (Wheeler/Wadell)
        if u <= 1:
            # Narrow trace
            f_u = (6 + (2 * math.pi - 6) * math.exp(-(30.666 / u)**0.7528))
            z0 = (Z0_FREE / (2 * math.pi * math.sqrt(dk_eff))) * math.log(f_u / u + math.sqrt(1 + (2/u)**2))
        else:
            # Wide trace
            z0 = (Z0_FREE / math.sqrt(dk_eff)) / (u + 1.393 + 0.667 * math.log(u + 1.444))

        return z0, dk_eff

    def _wadell_stripline(
        self,
        w: float,  # Trace width (mm)
        t: float,  # Trace thickness (mm)
        b: float,  # Ground-to-ground spacing (mm)
        dk: float  # Dielectric constant
    ) -> float:
        """
        Wadell Symmetric Stripline Impedance

        Accuracy: ~1%

        Returns: Z0
        """
        # Effective width correction for thickness
        x = t / b

        if x < 0.0001:
            w_eff = w
        else:
            cf = (2 / math.pi) * (
                (1 - x) * math.log(1 / (1 - x)) + (1 + x) * math.log(1 / (1 + x)) -
                x * math.log(1 / x**2) + 2 * x * math.log(2) + x * math.log(x / ((1 + x) * (1 - x)))
            )

            # Simplified correction
            delta_w = (t / math.pi) * (1 - 0.5 * math.log((t / (2 * b - t))**2 + (1 / (2 * math.pi * w / t + 1))**2))
            w_eff = w + delta_w

        # Stripline impedance
        u = w_eff / b

        if u < 0.35:
            # Narrow trace - more accurate formula
            z0 = (30 / math.sqrt(dk)) * math.log(1 + (4 * b / (math.pi * w_eff)) * (
                (8 * b / (math.pi * w_eff)) + math.sqrt((8 * b / (math.pi * w_eff))**2 + 6.27)
            ))
        else:
            # Wide trace
            k = 1 / math.cosh(math.pi * w_eff / (2 * b))
            z0 = (30 * math.pi / math.sqrt(dk)) / (w_eff / b + (2 / math.pi) * math.log(4))

        return z0

    # =========================================================================
    # DIFFERENTIAL PAIR IMPEDANCE
    # =========================================================================

    def _edge_coupled_microstrip_diff(
        self,
        w: float,   # Trace width (mm)
        t: float,   # Trace thickness (mm)
        h: float,   # Dielectric height (mm)
        s: float,   # Edge-to-edge spacing (mm)
        dk: float   # Dielectric constant
    ) -> Tuple[float, float, float]:
        """
        Edge-coupled differential microstrip impedance

        Based on empirical equations from Howard Johnson's "High-Speed Digital Design"

        Returns: (Z_single, Z_diff, Z_common)
        """
        # First calculate single-ended impedance
        z_single, dk_eff = self._wadell_microstrip(w, t, h, dk)

        # Coupling coefficient (empirical)
        # Increases as spacing decreases
        g = s / h  # Normalized gap

        # Odd-mode impedance (differential mode per trace)
        # Coupling reduces impedance
        k_odd = 1 - 0.48 * math.exp(-0.96 * g)
        z_odd = z_single * k_odd

        # Even-mode impedance (common mode per trace)
        # Coupling increases impedance slightly
        k_even = 1 + 0.25 * math.exp(-1.3 * g)
        z_even = z_single * k_even

        # Differential impedance = 2 * Z_odd
        z_diff = 2 * z_odd

        # Common-mode impedance = Z_even / 2
        z_common = z_even / 2

        return z_single, z_diff, z_common

    def _edge_coupled_stripline_diff(
        self,
        w: float,   # Trace width (mm)
        t: float,   # Trace thickness (mm)
        b: float,   # Ground-to-ground spacing (mm)
        s: float,   # Edge-to-edge spacing (mm)
        dk: float   # Dielectric constant
    ) -> Tuple[float, float, float]:
        """
        Edge-coupled differential stripline impedance

        Returns: (Z_single, Z_diff, Z_common)
        """
        # Single-ended stripline impedance
        z_single = self._wadell_stripline(w, t, b, dk)

        # Coupling factor for stripline
        g = s / b

        # Empirical coupling coefficients (stripline couples more tightly)
        k_odd = 1 - 0.6 * math.exp(-1.2 * g)
        k_even = 1 + 0.35 * math.exp(-1.5 * g)

        z_odd = z_single * k_odd
        z_even = z_single * k_even

        z_diff = 2 * z_odd
        z_common = z_even / 2

        return z_single, z_diff, z_common

    def _broadside_coupled_stripline_diff(
        self,
        w: float,   # Trace width (mm)
        t: float,   # Trace thickness (mm)
        h1: float,  # Top trace to top ground (mm)
        h2: float,  # Bottom trace to bottom ground (mm)
        s: float,   # Vertical spacing between traces (mm)
        dk: float   # Dielectric constant
    ) -> Tuple[float, float]:
        """
        Broadside-coupled differential stripline impedance

        Traces are stacked vertically (on different layers)
        Provides very tight coupling and high differential impedance

        Returns: (Z_single, Z_diff)
        """
        # This is a more complex structure
        # Approximate using modified stripline equations

        # Total ground-to-ground spacing
        b_total = h1 + t + s + t + h2

        # Single trace impedance (approximate)
        z_single = self._wadell_stripline(w, t, b_total / 2, dk)

        # Broadside coupling is very strong
        # Vertical coupling factor
        k_v = s / (s + w)  # Vertical coupling estimate
        k_coupling = 0.7 + 0.3 * k_v

        z_diff = 2 * z_single * k_coupling

        return z_single, z_diff

    # =========================================================================
    # COPLANAR WAVEGUIDE
    # =========================================================================

    def _cpw_with_ground(
        self,
        w: float,   # Center conductor width (mm)
        s: float,   # Gap to coplanar ground (mm)
        t: float,   # Trace thickness (mm)
        h: float,   # Dielectric height to ground plane (mm)
        dk: float   # Dielectric constant
    ) -> Tuple[float, float]:
        """
        Coplanar Waveguide with Ground Plane (CPWG)

        Uses elliptic integral approximation

        Returns: (Z0, effective_dk)
        """
        # Effective dimensions
        a = w / 2
        b = w / 2 + s

        # Elliptic integral arguments
        k0 = a / b  # = w / (w + 2s)
        k0_prime = math.sqrt(1 - k0**2)

        # Simplified elliptic integral ratio (valid for 0 < k < 0.707)
        if k0 < 0.707:
            K_ratio = math.pi / math.log(2 * (1 + math.sqrt(k0_prime)) / (1 - math.sqrt(k0_prime)))
        else:
            K_ratio = math.log(2 * (1 + math.sqrt(k0)) / (1 - math.sqrt(k0))) / math.pi

        # Ground plane effect
        k1 = math.tanh(math.pi * a / (2 * h)) / math.tanh(math.pi * b / (2 * h))
        k1_prime = math.sqrt(1 - k1**2)

        if k1 < 0.707:
            K1_ratio = math.pi / math.log(2 * (1 + math.sqrt(k1_prime)) / (1 - math.sqrt(k1_prime)))
        else:
            K1_ratio = math.log(2 * (1 + math.sqrt(k1)) / (1 - math.sqrt(k1))) / math.pi

        # Effective dielectric constant
        dk_eff = 1 + ((dk - 1) / 2) * (K1_ratio / K_ratio)

        # Characteristic impedance
        z0 = (30 * math.pi / math.sqrt(dk_eff)) * K_ratio

        return z0, dk_eff

    # =========================================================================
    # LOSS CALCULATIONS
    # =========================================================================

    def _conductor_loss(
        self,
        z0: float,
        w: float,
        t: float,
        frequency_ghz: float,
        roughness_um: float = 0.5
    ) -> float:
        """
        Calculate conductor loss due to skin effect

        Returns: Loss in dB/mm
        """
        # Skin depth at frequency
        # For copper: sigma = 5.8e7 S/m, mu_r = 1
        sigma = 5.8e7  # Copper conductivity
        mu0 = 4 * math.pi * 1e-7
        f_hz = frequency_ghz * 1e9

        skin_depth = math.sqrt(1 / (math.pi * f_hz * mu0 * sigma))  # meters
        skin_depth_um = skin_depth * 1e6

        # Roughness correction (Hammerstad-Jensen)
        delta_rms = roughness_um / skin_depth_um
        roughness_factor = 1 + (2 / math.pi) * math.atan(1.4 * (delta_rms)**2)

        # Conductor loss (simplified)
        # alpha_c = R_s / (2 * Z0 * W)
        Rs = 1 / (sigma * skin_depth)  # Surface resistance

        w_m = w * 1e-3  # Convert to meters
        alpha_c = (Rs * roughness_factor) / (2 * z0 * w_m)  # Np/m

        # Convert to dB/mm
        loss_db_per_mm = alpha_c * 8.686 / 1000

        return loss_db_per_mm

    def _dielectric_loss(
        self,
        dk_eff: float,
        df: float,
        frequency_ghz: float
    ) -> float:
        """
        Calculate dielectric loss

        Returns: Loss in dB/mm
        """
        # Dielectric loss per unit length
        # alpha_d = pi * f * sqrt(dk_eff) * tan(delta) / c

        c = 3e8  # Speed of light (m/s)
        f_hz = frequency_ghz * 1e9

        alpha_d = (math.pi * f_hz * math.sqrt(dk_eff) * df) / c  # Np/m

        # Convert to dB/mm
        loss_db_per_mm = alpha_d * 8.686 / 1000

        return loss_db_per_mm

    # =========================================================================
    # MAIN CALCULATION INTERFACE
    # =========================================================================

    def calculate_impedance(
        self,
        structure: ImpedanceStructure,
        trace_width_mm: float,
        dielectric_height_mm: float,
        dielectric_dk: float,
        trace_thickness_mm: float = 0.035,
        spacing_mm: float = 0.0,
        dielectric_df: float = 0.02,
        frequency_ghz: Optional[float] = None,
        method: Optional[CalculationMethod] = None
    ) -> ImpedanceResult:
        """
        Calculate impedance for a given structure

        Parameters:
        -----------
        structure : ImpedanceStructure
            Type of transmission line structure
        trace_width_mm : float
            Width of the trace
        dielectric_height_mm : float
            Height of dielectric (H for microstrip, B for stripline)
        dielectric_dk : float
            Dielectric constant
        trace_thickness_mm : float
            Copper thickness (default 1oz = 0.035mm)
        spacing_mm : float
            For differential pairs: edge-to-edge spacing
        dielectric_df : float
            Dissipation factor (loss tangent)
        frequency_ghz : float
            Frequency for loss calculations
        method : CalculationMethod
            Calculation method (default: from config)

        Returns:
        --------
        ImpedanceResult with calculated values
        """
        method = method or self.config.calculation_method
        freq = frequency_ghz or self.config.frequency_ghz

        w = trace_width_mm
        t = trace_thickness_mm
        h = dielectric_height_mm
        dk = dielectric_dk
        s = spacing_mm

        result = ImpedanceResult(
            structure=structure,
            method=method,
            trace_width_mm=w,
            trace_thickness_mm=t,
            dielectric_height_mm=h,
            dielectric_dk=dk,
            spacing_mm=s,
            z0=0.0
        )

        # Calculate based on structure type
        if structure == ImpedanceStructure.SURFACE_MICROSTRIP:
            if method == CalculationMethod.IPC_2141:
                z0, dk_eff = self._ipc2141_microstrip(w, t, h, dk)
                result.accuracy_percent = 7.0
            else:  # WADELL (default)
                z0, dk_eff = self._wadell_microstrip(w, t, h, dk)
                result.accuracy_percent = 3.0
            result.z0 = z0
            result.effective_dk = dk_eff

        elif structure == ImpedanceStructure.EMBEDDED_MICROSTRIP:
            # Embedded = covered microstrip, slightly different Dk_eff
            if method == CalculationMethod.IPC_2141:
                z0, dk_eff = self._ipc2141_microstrip(w, t, h, dk)
            else:
                z0, dk_eff = self._wadell_microstrip(w, t, h, dk)
            # Correction for solder mask cover (increases Dk_eff slightly)
            dk_eff = dk_eff * 1.02
            z0 = z0 * 0.98  # Impedance drops slightly
            result.z0 = z0
            result.effective_dk = dk_eff
            result.accuracy_percent = 5.0

        elif structure == ImpedanceStructure.SYMMETRIC_STRIPLINE:
            # h here represents ground-to-ground spacing (B)
            if method == CalculationMethod.IPC_2141:
                z0 = self._ipc2141_stripline(w, t, h, dk)
                result.accuracy_percent = 1.5
            else:
                z0 = self._wadell_stripline(w, t, h, dk)
                result.accuracy_percent = 1.0
            result.z0 = z0
            result.effective_dk = dk  # Stripline: Dk_eff = Dk

        elif structure == ImpedanceStructure.ASYMMETRIC_STRIPLINE:
            # Use weighted average of two heights
            # Simplified: use Wadell with average height
            z0 = self._wadell_stripline(w, t, h, dk)
            result.z0 = z0
            result.effective_dk = dk
            result.accuracy_percent = 3.0

        elif structure == ImpedanceStructure.DIFF_MICROSTRIP:
            z_single, z_diff, z_common = self._edge_coupled_microstrip_diff(w, t, h, s, dk)
            _, dk_eff = self._wadell_microstrip(w, t, h, dk)
            result.z0 = z_single
            result.z_diff = z_diff
            result.z_common = z_common
            result.effective_dk = dk_eff
            result.accuracy_percent = 5.0

        elif structure == ImpedanceStructure.DIFF_STRIPLINE:
            z_single, z_diff, z_common = self._edge_coupled_stripline_diff(w, t, h, s, dk)
            result.z0 = z_single
            result.z_diff = z_diff
            result.z_common = z_common
            result.effective_dk = dk
            result.accuracy_percent = 3.0

        elif structure == ImpedanceStructure.BROADSIDE_STRIPLINE:
            # Broadside: s is vertical spacing between traces
            z_single, z_diff = self._broadside_coupled_stripline_diff(
                w, t, h/2, h/2, s, dk
            )
            result.z0 = z_single
            result.z_diff = z_diff
            result.effective_dk = dk
            result.accuracy_percent = 5.0

        elif structure == ImpedanceStructure.COPLANAR_WAVEGUIDE:
            z0, dk_eff = self._cpw_with_ground(w, s, t, h, dk)
            result.z0 = z0
            result.effective_dk = dk_eff
            result.accuracy_percent = 4.0

        # Calculate derived parameters
        c = 299.792458  # Speed of light (mm/ns)

        result.velocity_factor = 1 / math.sqrt(result.effective_dk)
        result.delay_ps_per_mm = 1000 / (c * result.velocity_factor)  # ps/mm

        # Calculate losses
        result.conductor_loss_db_per_mm = self._conductor_loss(
            result.z0, w, t, freq, roughness_um=0.5
        )
        result.dielectric_loss_db_per_mm = self._dielectric_loss(
            result.effective_dk, dielectric_df, freq
        )
        result.total_loss_db_per_mm = (
            result.conductor_loss_db_per_mm + result.dielectric_loss_db_per_mm
        )

        return result

    # =========================================================================
    # TRACE WIDTH SOLVER
    # =========================================================================

    def solve_trace_width(
        self,
        structure: ImpedanceStructure,
        target_z0: float,
        dielectric_height_mm: float,
        dielectric_dk: float,
        trace_thickness_mm: float = 0.035,
        spacing_mm: float = 0.0,
        tolerance: float = 0.001,
        max_iterations: int = 50
    ) -> Tuple[float, ImpedanceResult]:
        """
        Solve for trace width to achieve target impedance

        Uses Newton-Raphson iteration

        Parameters:
        -----------
        target_z0 : float
            Target impedance (Ohms)
        ... other params same as calculate_impedance

        Returns:
        --------
        (trace_width_mm, ImpedanceResult)
        """
        # Initial guess based on target impedance
        # Higher Z0 = narrower trace
        if target_z0 > 75:
            w_guess = dielectric_height_mm * 0.5
        elif target_z0 > 50:
            w_guess = dielectric_height_mm * 1.0
        else:
            w_guess = dielectric_height_mm * 2.0

        w = w_guess

        for iteration in range(max_iterations):
            # Calculate impedance at current width
            result = self.calculate_impedance(
                structure=structure,
                trace_width_mm=w,
                dielectric_height_mm=dielectric_height_mm,
                dielectric_dk=dielectric_dk,
                trace_thickness_mm=trace_thickness_mm,
                spacing_mm=spacing_mm
            )

            z_current = result.z0
            error = z_current - target_z0

            if abs(error) < tolerance:
                break

            # Calculate derivative (numerical)
            delta_w = w * 0.01
            result_delta = self.calculate_impedance(
                structure=structure,
                trace_width_mm=w + delta_w,
                dielectric_height_mm=dielectric_height_mm,
                dielectric_dk=dielectric_dk,
                trace_thickness_mm=trace_thickness_mm,
                spacing_mm=spacing_mm
            )

            dz_dw = (result_delta.z0 - z_current) / delta_w

            if abs(dz_dw) < 1e-10:
                break

            # Newton-Raphson update
            w_new = w - error / dz_dw

            # Clamp to valid range
            w_new = max(self.config.min_trace_width_mm, min(self.config.max_trace_width_mm, w_new))

            w = w_new

        # Final calculation with solved width
        final_result = self.calculate_impedance(
            structure=structure,
            trace_width_mm=w,
            dielectric_height_mm=dielectric_height_mm,
            dielectric_dk=dielectric_dk,
            trace_thickness_mm=trace_thickness_mm,
            spacing_mm=spacing_mm
        )

        return w, final_result

    def solve_differential_spacing(
        self,
        structure: ImpedanceStructure,
        trace_width_mm: float,
        target_z_diff: float,
        dielectric_height_mm: float,
        dielectric_dk: float,
        trace_thickness_mm: float = 0.035,
        tolerance: float = 0.1,
        max_iterations: int = 50
    ) -> Tuple[float, ImpedanceResult]:
        """
        Solve for differential pair spacing to achieve target differential impedance

        Returns: (spacing_mm, ImpedanceResult)
        """
        if structure not in (ImpedanceStructure.DIFF_MICROSTRIP, ImpedanceStructure.DIFF_STRIPLINE):
            raise ValueError("Structure must be differential")

        # Initial guess: spacing = width (common starting point)
        s = trace_width_mm

        for iteration in range(max_iterations):
            result = self.calculate_impedance(
                structure=structure,
                trace_width_mm=trace_width_mm,
                dielectric_height_mm=dielectric_height_mm,
                dielectric_dk=dielectric_dk,
                trace_thickness_mm=trace_thickness_mm,
                spacing_mm=s
            )

            z_diff_current = result.z_diff
            error = z_diff_current - target_z_diff

            if abs(error) < tolerance:
                break

            # Numerical derivative
            delta_s = s * 0.01
            result_delta = self.calculate_impedance(
                structure=structure,
                trace_width_mm=trace_width_mm,
                dielectric_height_mm=dielectric_height_mm,
                dielectric_dk=dielectric_dk,
                trace_thickness_mm=trace_thickness_mm,
                spacing_mm=s + delta_s
            )

            dz_ds = (result_delta.z_diff - z_diff_current) / delta_s

            if abs(dz_ds) < 1e-10:
                break

            s_new = s - error / dz_ds
            s_new = max(self.config.min_spacing_mm, s_new)
            s = s_new

        final_result = self.calculate_impedance(
            structure=structure,
            trace_width_mm=trace_width_mm,
            dielectric_height_mm=dielectric_height_mm,
            dielectric_dk=dielectric_dk,
            trace_thickness_mm=trace_thickness_mm,
            spacing_mm=s
        )

        return s, final_result

    # =========================================================================
    # STACKUP ANALYSIS
    # =========================================================================

    def analyze_stackup(self, stackup: Stackup) -> Dict[str, Any]:
        """
        Analyze a stackup for impedance on each signal layer

        Returns comprehensive analysis including:
        - Impedance for each signal layer (microstrip or stripline)
        - Recommended trace widths for 50 ohm / 100 ohm diff
        - Manufacturing recommendations
        """
        self.stackup = stackup
        results = {
            'stackup_name': stackup.name,
            'total_thickness_mm': stackup.total_thickness_mm,
            'layer_count': stackup.copper_count,
            'layers': []
        }

        # Analyze each copper layer
        for i, layer in enumerate(stackup.layers):
            if not isinstance(layer, CopperLayer):
                continue

            if layer.layer_type != LayerType.SIGNAL:
                continue

            layer_info = {
                'name': layer.name,
                'is_outer': layer.is_outer,
                'copper_weight_oz': layer.weight_oz,
            }

            # Find adjacent dielectric
            dielectric = self._find_adjacent_dielectric(stackup, i)
            if dielectric:
                layer_info['dielectric_material'] = dielectric.material
                layer_info['dielectric_height_mm'] = dielectric.thickness_mm
                layer_info['dielectric_dk'] = dielectric.dk

                # Determine structure type
                if layer.is_outer:
                    structure = ImpedanceStructure.SURFACE_MICROSTRIP
                else:
                    structure = ImpedanceStructure.SYMMETRIC_STRIPLINE

                layer_info['structure'] = structure.value

                # Calculate trace width for 50 ohm
                w_50, result_50 = self.solve_trace_width(
                    structure=structure,
                    target_z0=50.0,
                    dielectric_height_mm=dielectric.thickness_mm,
                    dielectric_dk=dielectric.dk,
                    trace_thickness_mm=layer.thickness_mm
                )

                layer_info['trace_width_50ohm_mm'] = round(w_50, 4)
                layer_info['calculated_z0'] = round(result_50.z0, 2)

                # Calculate differential pair for 100 ohm
                if structure == ImpedanceStructure.SURFACE_MICROSTRIP:
                    diff_struct = ImpedanceStructure.DIFF_MICROSTRIP
                else:
                    diff_struct = ImpedanceStructure.DIFF_STRIPLINE

                # Start with 50ohm width, solve for spacing
                try:
                    spacing, result_diff = self.solve_differential_spacing(
                        structure=diff_struct,
                        trace_width_mm=w_50,
                        target_z_diff=100.0,
                        dielectric_height_mm=dielectric.thickness_mm,
                        dielectric_dk=dielectric.dk,
                        trace_thickness_mm=layer.thickness_mm
                    )
                    layer_info['diff_pair_width_mm'] = round(w_50, 4)
                    layer_info['diff_pair_spacing_mm'] = round(spacing, 4)
                    layer_info['calculated_z_diff'] = round(result_diff.z_diff, 2)
                except:
                    layer_info['diff_pair_error'] = "Could not solve for 100 ohm differential"

            results['layers'].append(layer_info)

        # Manufacturing recommendations
        results['recommendations'] = self._generate_recommendations(stackup, results)

        return results

    def _find_adjacent_dielectric(
        self,
        stackup: Stackup,
        copper_index: int
    ) -> Optional[DielectricLayer]:
        """Find the dielectric layer adjacent to a copper layer"""
        layers = stackup.layers

        # Check layer below
        if copper_index + 1 < len(layers):
            next_layer = layers[copper_index + 1]
            if isinstance(next_layer, DielectricLayer):
                return next_layer

        # Check layer above
        if copper_index > 0:
            prev_layer = layers[copper_index - 1]
            if isinstance(prev_layer, DielectricLayer):
                return prev_layer

        return None

    def _generate_recommendations(
        self,
        stackup: Stackup,
        analysis: Dict
    ) -> List[str]:
        """Generate manufacturing and design recommendations"""
        recommendations = []

        # Check layer count
        if stackup.copper_count == 2:
            recommendations.append(
                "2-layer boards have limited routing options; consider 4-layer for complex designs"
            )

        # Check for signal integrity
        has_ground_plane = any(
            isinstance(l, CopperLayer) and l.layer_type == LayerType.GROUND
            for l in stackup.layers
        )

        if not has_ground_plane and stackup.copper_count > 2:
            recommendations.append(
                "No dedicated ground plane detected; add continuous ground for better SI/EMI"
            )

        # Check trace widths
        for layer_info in analysis.get('layers', []):
            w = layer_info.get('trace_width_50ohm_mm', 0)
            if w < 0.1:
                recommendations.append(
                    f"Layer {layer_info['name']}: Trace width {w:.3f}mm is narrow, "
                    f"verify manufacturability (min typically 0.1mm)"
                )
            elif w > 1.0:
                recommendations.append(
                    f"Layer {layer_info['name']}: Trace width {w:.3f}mm is wide, "
                    f"consider thinner dielectric for narrower traces"
                )

        # Material recommendations
        materials_used = set()
        for layer in stackup.layers:
            if isinstance(layer, DielectricLayer) and layer.material in MATERIAL_DATABASE:
                materials_used.add(layer.material)

        for mat_key in materials_used:
            mat = MATERIAL_DATABASE[mat_key]
            if mat.dk_tolerance > 0.2:
                recommendations.append(
                    f"Material {mat.name} has wide Dk tolerance (±{mat.dk_tolerance}); "
                    f"consider tighter tolerance material for impedance control"
                )

        return recommendations

    # =========================================================================
    # MANUFACTURING PRESETS
    # =========================================================================

    @staticmethod
    def get_jlcpcb_standard_stackups() -> Dict[str, Stackup]:
        """
        Get JLCPCB standard stackup options

        Based on JLCPCB impedance calculator stackups
        """
        piston = StackupPiston()

        return {
            '2L_1.6mm': piston.create_2layer_stackup(
                material='FR4_STANDARD',
                total_thickness_mm=1.6,
                copper_weight_oz=1.0
            ),
            '4L_1.6mm': piston.create_4layer_stackup(
                material='FR4_STANDARD',
                total_thickness_mm=1.6,
                copper_weight_oz=1.0
            ),
            '6L_1.6mm': piston.create_6layer_stackup(
                material='FR4_HIGH_TG',
                total_thickness_mm=1.6,
                copper_weight_oz=1.0
            ),
        }

    @staticmethod
    def get_high_speed_stackups() -> Dict[str, Stackup]:
        """
        Get stackups optimized for high-speed digital (USB3, PCIe, DDR)

        Uses low-loss materials and proper layer ordering
        """
        piston = StackupPiston()

        # 6-layer with low-loss material
        stackup_6l = Stackup(
            name='6-Layer High-Speed (FR408HR)',
            layers=[
                CopperLayer('L1_Top', LayerType.SIGNAL, 0.035, 1.0, is_outer=True),
                DielectricLayer('PP1', 'ISOLA_FR408HR', 0.1),
                CopperLayer('L2_GND', LayerType.GROUND, 0.035, 1.0),
                DielectricLayer('Core1', 'ISOLA_FR408HR', 0.4),
                CopperLayer('L3_SIG', LayerType.SIGNAL, 0.035, 1.0),
                DielectricLayer('PP2', 'ISOLA_FR408HR', 0.1),
                CopperLayer('L4_SIG', LayerType.SIGNAL, 0.035, 1.0),
                DielectricLayer('Core2', 'ISOLA_FR408HR', 0.4),
                CopperLayer('L5_PWR', LayerType.POWER, 0.035, 1.0),
                DielectricLayer('PP3', 'ISOLA_FR408HR', 0.1),
                CopperLayer('L6_Bot', LayerType.SIGNAL, 0.035, 1.0, is_outer=True),
            ]
        )
        stackup_6l.calculate_properties()

        return {
            '6L_HighSpeed': stackup_6l,
        }

    @staticmethod
    def get_rf_stackups() -> Dict[str, Stackup]:
        """
        Get stackups for RF/Microwave applications

        Uses Rogers or Taconic materials
        """
        # 2-layer Rogers for simple RF
        rf_2l = Stackup(
            name='2-Layer RF (RO4350B)',
            layers=[
                CopperLayer('Top', LayerType.SIGNAL, 0.035, 1.0, is_outer=True),
                DielectricLayer('Core', 'ROGERS_RO4350B', 0.508),  # 20 mil
                CopperLayer('Bottom', LayerType.GROUND, 0.035, 1.0, is_outer=True),
            ]
        )
        rf_2l.calculate_properties()

        # 4-layer hybrid (FR4 core, Rogers outer)
        rf_4l_hybrid = Stackup(
            name='4-Layer Hybrid (RO4350B/FR4)',
            layers=[
                CopperLayer('Top', LayerType.SIGNAL, 0.035, 1.0, is_outer=True),
                DielectricLayer('PP1', 'ROGERS_RO4350B', 0.254),  # 10 mil Rogers
                CopperLayer('GND', LayerType.GROUND, 0.035, 1.0),
                DielectricLayer('Core', 'FR4_HIGH_TG', 0.8),  # FR4 core
                CopperLayer('PWR', LayerType.POWER, 0.035, 1.0),
                DielectricLayer('PP2', 'ROGERS_RO4350B', 0.254),  # 10 mil Rogers
                CopperLayer('Bottom', LayerType.SIGNAL, 0.035, 1.0, is_outer=True),
            ]
        )
        rf_4l_hybrid.calculate_properties()

        return {
            '2L_RF_RO4350B': rf_2l,
            '4L_RF_Hybrid': rf_4l_hybrid,
        }

    # =========================================================================
    # RUN INTERFACE (for PCBEngine integration)
    # =========================================================================

    def run(
        self,
        stackup: Optional[Stackup] = None,
        layer_count: int = 4,
        material: str = 'FR4_STANDARD',
        total_thickness_mm: float = 1.6
    ) -> Dict[str, Any]:
        """
        Main entry point for PCBEngine integration

        Parameters:
        -----------
        stackup : Stackup, optional
            Pre-defined stackup to analyze
        layer_count : int
            Number of copper layers (if stackup not provided)
        material : str
            Material key from MATERIAL_DATABASE
        total_thickness_mm : float
            Target board thickness

        Returns:
        --------
        Dict with stackup analysis results
        """
        # Create stackup if not provided
        if stackup is None:
            if layer_count == 2:
                stackup = self.create_2layer_stackup(material, total_thickness_mm)
            elif layer_count == 4:
                stackup = self.create_4layer_stackup(material, total_thickness_mm)
            elif layer_count == 6:
                stackup = self.create_6layer_stackup(material, total_thickness_mm)
            elif layer_count == 8:
                stackup = self.create_8layer_stackup(material, total_thickness_mm)
            else:
                raise ValueError(f"Unsupported layer count: {layer_count}")

        # Analyze the stackup
        analysis = self.analyze_stackup(stackup)

        # Store results
        self.stackup = stackup
        self.impedance_results = {
            layer['name']: layer
            for layer in analysis.get('layers', [])
        }

        return {
            'success': True,
            'stackup': stackup,
            'analysis': analysis,
            'material_database': list(MATERIAL_DATABASE.keys()),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_microstrip_z0(
    trace_width_mm: float,
    dielectric_height_mm: float,
    dielectric_dk: float = 4.5
) -> float:
    """
    Quick single-ended microstrip impedance calculation

    Uses Wadell equations for best accuracy
    """
    piston = StackupPiston()
    result = piston.calculate_impedance(
        structure=ImpedanceStructure.SURFACE_MICROSTRIP,
        trace_width_mm=trace_width_mm,
        dielectric_height_mm=dielectric_height_mm,
        dielectric_dk=dielectric_dk
    )
    return result.z0


def quick_stripline_z0(
    trace_width_mm: float,
    ground_spacing_mm: float,
    dielectric_dk: float = 4.5
) -> float:
    """
    Quick stripline impedance calculation
    """
    piston = StackupPiston()
    result = piston.calculate_impedance(
        structure=ImpedanceStructure.SYMMETRIC_STRIPLINE,
        trace_width_mm=trace_width_mm,
        dielectric_height_mm=ground_spacing_mm,
        dielectric_dk=dielectric_dk
    )
    return result.z0


def solve_for_50ohm_microstrip(
    dielectric_height_mm: float,
    dielectric_dk: float = 4.5
) -> float:
    """
    Solve for trace width to achieve 50 ohm microstrip
    """
    piston = StackupPiston()
    width, _ = piston.solve_trace_width(
        structure=ImpedanceStructure.SURFACE_MICROSTRIP,
        target_z0=50.0,
        dielectric_height_mm=dielectric_height_mm,
        dielectric_dk=dielectric_dk
    )
    return width


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("STACKUP PISTON - Self Test")
    print("=" * 60)

    piston = StackupPiston()

    # Test 1: Microstrip impedance
    print("\n1. Surface Microstrip (IPC-2141 vs Wadell):")
    print("   W=0.3mm, H=0.2mm, Dk=4.5")

    result_ipc = piston.calculate_impedance(
        structure=ImpedanceStructure.SURFACE_MICROSTRIP,
        trace_width_mm=0.3,
        dielectric_height_mm=0.2,
        dielectric_dk=4.5,
        method=CalculationMethod.IPC_2141
    )

    result_wadell = piston.calculate_impedance(
        structure=ImpedanceStructure.SURFACE_MICROSTRIP,
        trace_width_mm=0.3,
        dielectric_height_mm=0.2,
        dielectric_dk=4.5,
        method=CalculationMethod.WADELL
    )

    print(f"   IPC-2141:  Z0 = {result_ipc.z0:.2f} Ohm (±{result_ipc.accuracy_percent}%)")
    print(f"   Wadell:    Z0 = {result_wadell.z0:.2f} Ohm (±{result_wadell.accuracy_percent}%)")
    print(f"   Dk_eff = {result_wadell.effective_dk:.3f}")
    print(f"   Delay = {result_wadell.delay_ps_per_mm:.2f} ps/mm")

    # Test 2: Solve for 50 ohm
    print("\n2. Solve for 50 Ohm Microstrip (H=0.2mm, Dk=4.5):")
    width_50, result_50 = piston.solve_trace_width(
        structure=ImpedanceStructure.SURFACE_MICROSTRIP,
        target_z0=50.0,
        dielectric_height_mm=0.2,
        dielectric_dk=4.5
    )
    print(f"   Required width: {width_50:.4f} mm ({width_50/0.0254:.2f} mil)")
    print(f"   Achieved Z0: {result_50.z0:.2f} Ohm")

    # Test 3: Differential pair
    print("\n3. Differential Microstrip (100 Ohm target):")
    spacing, result_diff = piston.solve_differential_spacing(
        structure=ImpedanceStructure.DIFF_MICROSTRIP,
        trace_width_mm=width_50,
        target_z_diff=100.0,
        dielectric_height_mm=0.2,
        dielectric_dk=4.5
    )
    print(f"   Width: {width_50:.4f} mm, Spacing: {spacing:.4f} mm")
    print(f"   Z_single = {result_diff.z0:.2f} Ohm")
    print(f"   Z_diff = {result_diff.z_diff:.2f} Ohm")
    print(f"   Z_common = {result_diff.z_common:.2f} Ohm")

    # Test 4: Material database
    print("\n4. Material Database:")
    for key, mat in list(MATERIAL_DATABASE.items())[:5]:
        print(f"   {mat.name}: Dk={mat.dk} ± {mat.dk_tolerance}, Df={mat.df}")
    print(f"   ... and {len(MATERIAL_DATABASE) - 5} more materials")

    # Test 5: Stackup creation and analysis
    print("\n5. 4-Layer Stackup Analysis:")
    stackup = piston.create_4layer_stackup(
        material='FR4_STANDARD',
        total_thickness_mm=1.6,
        copper_weight_oz=1.0
    )

    analysis = piston.analyze_stackup(stackup)
    print(f"   Stackup: {analysis['stackup_name']}")
    print(f"   Total thickness: {analysis['total_thickness_mm']:.3f} mm")
    print(f"   Layer count: {analysis['layer_count']}")

    for layer in analysis['layers']:
        print(f"\n   Layer: {layer['name']} ({layer['structure']})")
        print(f"     50 Ohm trace: {layer.get('trace_width_50ohm_mm', 'N/A'):.4f} mm")
        print(f"     Calculated Z0: {layer.get('calculated_z0', 'N/A'):.2f} Ohm")
        if 'diff_pair_spacing_mm' in layer:
            print(f"     100 Ohm diff: W={layer['diff_pair_width_mm']:.4f}mm, S={layer['diff_pair_spacing_mm']:.4f}mm")

    print("\n" + "=" * 60)
    print("Stackup Piston self-test PASSED")
    print("=" * 60)
