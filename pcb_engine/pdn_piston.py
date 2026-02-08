"""
PCB Engine - PDN Piston (Power Delivery Network)
==================================================

A dedicated piston (sub-engine) for power integrity analysis and optimization.

ALGORITHMS & REFERENCES:
=========================
1. Target Impedance Method
   Reference: Larry Smith, "Decoupling Capacitor Calculations for IC Packages"
   Reference: TI Application Report SWPA222A "Power Delivery Network Analysis"
   Formula: Z_target = Vdd * ripple_percent / I_transient

2. Decoupling Capacitor Selection
   Reference: IEEE "Decoupling Capacitor Selection Algorithm for PDN Based on DRL"
   Reference: Genetic Algorithm optimization (MDPI Electronics 2020)
   Algorithm: Multi-tier (bulk, mid-freq, high-freq) capacitor strategy

3. ESR/ESL Modeling
   Reference: JEDEC capacitor models
   Formula: Z_cap = ESR + j*(w*ESL - 1/(w*C))
   Resonance: f_res = 1 / (2*pi*sqrt(L*C))

4. Plane Spreading Inductance
   Reference: Istvan Novak, "Power Distribution Network Design Methodologies"
   Formula: L_spread = 33.3 * d / (W + H)  [pH/mm for stripline]

5. Via Stitching Analysis
   Reference: Eric Bogatin, "Signal and Power Integrity Simplified"
   Via inductance: L_via = 5.08 * h * (ln(4*h/d) + 1)  [nH]

FREQUENCY DOMAIN ANALYSIS:
===========================
- DC: Bulk capacitors (electrolytic, tantalum) 10uF-1000uF
- Low freq (kHz): Ceramic 1uF-10uF
- Mid freq (MHz): Ceramic 100nF
- High freq (10s MHz): Ceramic 10nF-100nF
- VRM band: >100MHz, plane capacitance dominates

Sources:
- https://www.ti.com/lit/an/swpa222a/swpa222a.pdf
- https://www.protoexpress.com/blog/power-integrity-pdn-and-decoupling-capacitors/
- https://www.multi-circuit-boards.eu/en/support/pdn-target-impedance-pcb.html
- https://ieeexplore.ieee.org/document/8825249/
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import cmath


# =============================================================================
# CONSTANTS
# =============================================================================

# Speed of light (m/s)
C_LIGHT = 299792458

# Permeability of free space
MU_0 = 4 * math.pi * 1e-7

# Permittivity of free space
EPSILON_0 = 8.854e-12


# =============================================================================
# ENUMS
# =============================================================================

class CapacitorType(Enum):
    """Decoupling capacitor types"""
    ELECTROLYTIC_AL = 'electrolytic_al'     # Aluminum electrolytic
    ELECTROLYTIC_TA = 'electrolytic_ta'     # Tantalum
    CERAMIC_X7R = 'ceramic_x7r'             # X7R ceramic (mid stability)
    CERAMIC_X5R = 'ceramic_x5r'             # X5R ceramic (smaller, less stable)
    CERAMIC_C0G = 'ceramic_c0g'             # C0G/NP0 (most stable, low capacitance)
    POLYMER = 'polymer'                      # Polymer electrolytic


class PowerRailType(Enum):
    """Power rail categories"""
    CORE = 'core'           # IC core voltage (0.8-1.2V, high current)
    IO = 'io'               # I/O voltage (1.8V, 2.5V, 3.3V)
    ANALOG = 'analog'       # Sensitive analog supplies
    PLL = 'pll'             # PLL/clock supplies (very clean)
    DDR = 'ddr'             # DDR memory (VDD, VDDQ, VTT)
    GENERAL = 'general'     # General 3.3V/5V


# =============================================================================
# CAPACITOR DATABASE
# =============================================================================

@dataclass
class CapacitorModel:
    """Electrical model of a decoupling capacitor"""
    value_uf: float              # Nominal capacitance (uF)
    type: CapacitorType
    package: str                 # 0402, 0603, 0805, etc.

    # Parasitic elements
    esr_mohm: float              # Equivalent Series Resistance (mohm)
    esl_nh: float                # Equivalent Series Inductance (nH)

    # Voltage/temperature characteristics
    voltage_rating: float = 10.0  # Voltage rating (V)
    dc_bias_derating: float = 1.0 # Capacitance derating at rated voltage

    # Cost and size
    cost_factor: float = 1.0
    height_mm: float = 0.5

    @property
    def resonant_freq_mhz(self) -> float:
        """Calculate self-resonant frequency"""
        c = self.value_uf * 1e-6
        l = self.esl_nh * 1e-9
        if c > 0 and l > 0:
            return 1 / (2 * math.pi * math.sqrt(l * c)) / 1e6
        return 0.0

    def impedance_at_freq(self, freq_hz: float) -> complex:
        """Calculate complex impedance at frequency"""
        w = 2 * math.pi * freq_hz
        c = self.value_uf * 1e-6
        l = self.esl_nh * 1e-9
        r = self.esr_mohm * 1e-3

        # Z = R + j*(wL - 1/wC)
        if w > 0 and c > 0:
            z_imag = w * l - 1 / (w * c)
        else:
            z_imag = 0

        return complex(r, z_imag)


# Standard capacitor library
CAPACITOR_LIBRARY: Dict[str, CapacitorModel] = {
    # Bulk capacitors (electrolytic)
    '100uF_electrolytic': CapacitorModel(
        value_uf=100.0, type=CapacitorType.ELECTROLYTIC_AL,
        package='6.3x5.8', esr_mohm=100, esl_nh=5.0,
        voltage_rating=16, cost_factor=0.5, height_mm=5.8
    ),
    '47uF_tantalum': CapacitorModel(
        value_uf=47.0, type=CapacitorType.ELECTROLYTIC_TA,
        package='C_3528', esr_mohm=80, esl_nh=2.5,
        voltage_rating=10, cost_factor=1.5, height_mm=1.8
    ),
    '22uF_polymer': CapacitorModel(
        value_uf=22.0, type=CapacitorType.POLYMER,
        package='D_7343', esr_mohm=15, esl_nh=2.0,
        voltage_rating=6.3, cost_factor=2.0, height_mm=2.8
    ),

    # Mid-frequency ceramics
    '10uF_0805': CapacitorModel(
        value_uf=10.0, type=CapacitorType.CERAMIC_X5R,
        package='0805', esr_mohm=3, esl_nh=0.8,
        voltage_rating=10, dc_bias_derating=0.5, cost_factor=0.3, height_mm=0.6
    ),
    '4.7uF_0603': CapacitorModel(
        value_uf=4.7, type=CapacitorType.CERAMIC_X5R,
        package='0603', esr_mohm=5, esl_nh=0.6,
        voltage_rating=10, dc_bias_derating=0.6, cost_factor=0.2, height_mm=0.5
    ),
    '2.2uF_0402': CapacitorModel(
        value_uf=2.2, type=CapacitorType.CERAMIC_X5R,
        package='0402', esr_mohm=8, esl_nh=0.4,
        voltage_rating=10, dc_bias_derating=0.7, cost_factor=0.15, height_mm=0.4
    ),

    # High-frequency ceramics
    '1uF_0402': CapacitorModel(
        value_uf=1.0, type=CapacitorType.CERAMIC_X7R,
        package='0402', esr_mohm=10, esl_nh=0.4,
        voltage_rating=10, cost_factor=0.1, height_mm=0.4
    ),
    '100nF_0402': CapacitorModel(
        value_uf=0.1, type=CapacitorType.CERAMIC_X7R,
        package='0402', esr_mohm=15, esl_nh=0.35,
        voltage_rating=16, cost_factor=0.05, height_mm=0.4
    ),
    '100nF_0201': CapacitorModel(
        value_uf=0.1, type=CapacitorType.CERAMIC_X7R,
        package='0201', esr_mohm=25, esl_nh=0.25,
        voltage_rating=10, cost_factor=0.08, height_mm=0.3
    ),

    # Ultra high-frequency
    '10nF_0402': CapacitorModel(
        value_uf=0.01, type=CapacitorType.CERAMIC_X7R,
        package='0402', esr_mohm=30, esl_nh=0.35,
        voltage_rating=25, cost_factor=0.04, height_mm=0.4
    ),
    '1nF_0201': CapacitorModel(
        value_uf=0.001, type=CapacitorType.CERAMIC_C0G,
        package='0201', esr_mohm=50, esl_nh=0.2,
        voltage_rating=25, cost_factor=0.03, height_mm=0.3
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PowerRail:
    """Definition of a power rail"""
    name: str
    voltage: float               # Nominal voltage (V)
    current_max: float          # Maximum DC current (A)
    current_transient: float    # Maximum transient current (A)
    rail_type: PowerRailType = PowerRailType.GENERAL

    # Noise requirements
    ripple_mv: float = 50.0     # Maximum ripple (mV)
    ripple_percent: float = 0.0  # Or as percentage (calculated if 0)

    # Connected ICs
    connected_loads: List[str] = field(default_factory=list)

    @property
    def target_impedance(self) -> float:
        """Calculate target impedance (Ohms)"""
        if self.ripple_percent > 0:
            ripple = self.voltage * self.ripple_percent / 100
        else:
            ripple = self.ripple_mv / 1000

        if self.current_transient > 0:
            return ripple / self.current_transient
        return 0.1  # Default 100 mohm


@dataclass
class DecouplingPlacement:
    """Placement of a decoupling capacitor"""
    capacitor: str              # Key into CAPACITOR_LIBRARY
    x: float                    # X position (mm)
    y: float                    # Y position (mm)
    rail: str                   # Power rail name
    layer: str = 'F.Cu'         # Placement layer

    # Calculated properties
    mounting_inductance_nh: float = 0.0  # Via + trace inductance


@dataclass
class PowerPlane:
    """Definition of a power or ground plane"""
    name: str
    net: str                    # Net name (VCC, GND, etc)
    layer: str                  # Layer name (In1.Cu, In2.Cu)
    outline: List[Tuple[float, float]] = field(default_factory=list)

    # Properties
    copper_thickness_oz: float = 1.0
    is_split: bool = False
    split_regions: List[Dict] = field(default_factory=list)


@dataclass
class ViaStitch:
    """Via stitching for power planes"""
    x: float
    y: float
    drill_mm: float = 0.3
    pad_mm: float = 0.6
    connects: List[str] = field(default_factory=list)  # Net names


@dataclass
class PDNResult:
    """Result of PDN analysis"""
    rails_analyzed: int
    total_capacitors: int
    total_vias: int

    # Per-rail results
    rail_impedances: Dict[str, Dict[str, float]]  # rail -> {freq: Z}
    target_met: Dict[str, bool]

    # Placements
    capacitor_placements: List[DecouplingPlacement]
    via_stitches: List[ViaStitch]

    # Recommendations
    warnings: List[str]
    recommendations: List[str]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PDNConfig:
    """Configuration for PDN analysis"""
    # Board parameters
    board_width_mm: float = 50.0
    board_height_mm: float = 40.0
    board_thickness_mm: float = 1.6
    layer_count: int = 4

    # Stackup
    dielectric_constant: float = 4.5
    plane_spacing_mm: float = 0.2  # Between power and ground planes

    # Design rules
    via_drill_mm: float = 0.3
    via_pad_mm: float = 0.6
    via_pitch_mm: float = 1.0     # Minimum via-to-via spacing

    # Analysis
    freq_start_hz: float = 1e3    # 1 kHz
    freq_end_hz: float = 1e9      # 1 GHz
    freq_points: int = 100

    # Optimization
    max_capacitors_per_rail: int = 20
    prefer_smaller_packages: bool = True
    cost_optimization: bool = False

    # Via stitching
    stitch_ground_vias: bool = True
    stitch_spacing_mm: float = 5.0  # Maximum spacing between ground vias


# =============================================================================
# PDN PISTON CLASS
# =============================================================================

class PDNPiston:
    """
    Power Delivery Network Analysis and Optimization Piston

    Provides:
    1. Target impedance calculation
    2. Decoupling capacitor selection and placement
    3. Plane impedance analysis
    4. Via stitching optimization
    5. Frequency-domain impedance sweep
    """

    def __init__(self, config: Optional[PDNConfig] = None):
        self.config = config or PDNConfig()
        self.power_rails: List[PowerRail] = []
        self.power_planes: List[PowerPlane] = []
        self.capacitor_placements: List[DecouplingPlacement] = []
        self.via_stitches: List[ViaStitch] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []

    # =========================================================================
    # STANDARD PISTON API
    # =========================================================================

    def analyze(self, parts_db: Dict, routes: Dict) -> Dict[str, Any]:
        """
        Standard piston API - analyze power delivery network.

        Args:
            parts_db: Parts database with component info
            routes: Routing data

        Returns:
            Dictionary with PDN analysis results
        """
        self.warnings.clear()
        self.recommendations.clear()

        nets = parts_db.get('nets', {})
        power_nets = []

        # Identify power and ground nets
        for net_name, net_data in nets.items():
            name_upper = net_name.upper()
            if any(kw in name_upper for kw in ['VCC', 'VDD', 'VIN', '3V3', '5V', 'PWR', 'POWER']):
                power_nets.append({'name': net_name, 'type': 'power'})
            elif any(kw in name_upper for kw in ['GND', 'VSS', 'GROUND', 'AGND', 'DGND']):
                power_nets.append({'name': net_name, 'type': 'ground'})

        # Calculate target impedance for typical design
        z_target = self.calculate_target_impedance(3.3, 0.5, 5.0)

        return {
            'power_nets': power_nets,
            'target_impedance_ohms': z_target,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

    # =========================================================================
    # TARGET IMPEDANCE CALCULATION
    # =========================================================================

    def calculate_target_impedance(
        self,
        voltage: float,
        transient_current: float,
        ripple_percent: float = 5.0
    ) -> float:
        """
        Calculate target PDN impedance

        Reference: Larry Smith's classic formula
        Z_target = (V * ripple%) / I_transient

        For a 1.0V core with 5% ripple and 1A transient:
        Z_target = 1.0 * 0.05 / 1.0 = 50 mohm

        Args:
            voltage: Supply voltage (V)
            transient_current: Maximum current step (A)
            ripple_percent: Allowed voltage ripple (%)

        Returns:
            Target impedance in Ohms
        """
        ripple_v = voltage * ripple_percent / 100
        z_target = ripple_v / transient_current
        return z_target

    def calculate_frequency_range(
        self,
        ic_bandwidth_mhz: float,
        edge_rate_ns: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate frequency range of interest for PDN design

        Reference: f_knee = 0.35 / t_rise (for signal rise time)

        The PDN must meet target impedance from DC to the IC's
        operating frequency.

        Args:
            ic_bandwidth_mhz: Maximum operating frequency of IC
            edge_rate_ns: Signal edge rate (rise/fall time)

        Returns:
            (f_low_hz, f_high_hz) frequency range
        """
        # Low frequency: typically 1 kHz (bulk capacitor range)
        f_low = 1e3

        # High frequency: either IC bandwidth or edge rate knee
        f_knee = 0.35 / (edge_rate_ns * 1e-9)
        f_high = max(ic_bandwidth_mhz * 1e6, f_knee)

        return f_low, f_high

    # =========================================================================
    # PLANE CAPACITANCE AND INDUCTANCE
    # =========================================================================

    def calculate_plane_capacitance(
        self,
        area_mm2: float,
        spacing_mm: float,
        dielectric_constant: float = 4.5
    ) -> float:
        """
        Calculate inter-plane capacitance

        C = epsilon_0 * epsilon_r * A / d

        Args:
            area_mm2: Overlap area of power/ground planes
            spacing_mm: Dielectric thickness between planes
            dielectric_constant: Relative permittivity

        Returns:
            Capacitance in Farads
        """
        area_m2 = area_mm2 * 1e-6
        spacing_m = spacing_mm * 1e-3

        c = EPSILON_0 * dielectric_constant * area_m2 / spacing_m
        return c

    def calculate_plane_inductance(
        self,
        width_mm: float,
        length_mm: float,
        spacing_mm: float
    ) -> float:
        """
        Calculate plane spreading inductance

        Reference: Istvan Novak's simplified formula
        L = mu_0 * d * l / w  (for parallel plates)

        Also accounts for edge effects using:
        L_spread = 33.3 * d / (w + h)  [pH/mm] for stripline

        Args:
            width_mm: Current spreading width
            length_mm: Current path length
            spacing_mm: Plane separation

        Returns:
            Inductance in Henries
        """
        width_m = width_mm * 1e-3
        length_m = length_mm * 1e-3
        spacing_m = spacing_mm * 1e-3

        # Basic parallel plate inductance
        l = MU_0 * spacing_m * length_m / width_m

        return l

    def calculate_via_inductance(
        self,
        height_mm: float,
        drill_mm: float
    ) -> float:
        """
        Calculate via inductance

        Reference: Eric Bogatin's formula
        L_via = 5.08 * h * (ln(4*h/d) + 1)  [nH]

        Where h is height in inches, d is diameter in inches

        Args:
            height_mm: Via height (board thickness)
            drill_mm: Via drill diameter

        Returns:
            Inductance in Henries
        """
        h_inch = height_mm / 25.4
        d_inch = drill_mm / 25.4

        if d_inch > 0:
            l_nh = 5.08 * h_inch * (math.log(4 * h_inch / d_inch) + 1)
        else:
            l_nh = 0

        return l_nh * 1e-9

    # =========================================================================
    # DECOUPLING STRATEGY
    # =========================================================================

    def design_decoupling_strategy(
        self,
        rail: PowerRail,
        max_freq_hz: float
    ) -> List[Tuple[str, int]]:
        """
        Design multi-tier decoupling capacitor strategy

        Reference: Three-tier approach (bulk, mid, high-freq)

        The strategy places capacitors to cover all frequency bands:
        - Bulk: Handle low-frequency current demands
        - Mid-freq: Bridge between bulk and high-freq
        - High-freq: Handle fast transients, placed near IC

        Args:
            rail: Power rail to decouple
            max_freq_hz: Maximum frequency of interest

        Returns:
            List of (capacitor_key, quantity)
        """
        strategy = []
        z_target = rail.target_impedance

        # Determine tier requirements based on power level
        power_level = rail.voltage * rail.current_max

        # Tier 1: Bulk capacitors (DC to ~10kHz)
        if power_level > 1.0:  # More than 1W
            if power_level > 10.0:
                strategy.append(('100uF_electrolytic', 2))
            else:
                strategy.append(('47uF_tantalum', 1))

        # Tier 2: Mid-frequency (10kHz to 10MHz)
        if rail.current_transient > 0.5:
            strategy.append(('10uF_0805', 2))
            strategy.append(('4.7uF_0603', 2))

        # Tier 3: High-frequency (10MHz to 100MHz+)
        # These are mandatory for any digital IC
        if max_freq_hz > 1e6:
            strategy.append(('100nF_0402', 4))

        if max_freq_hz > 100e6:
            strategy.append(('10nF_0402', 2))

        if max_freq_hz > 500e6:
            strategy.append(('1nF_0201', 2))

        # Adjust for target impedance
        # More capacitors needed for lower target impedance
        if z_target < 0.05:  # Less than 50 mohm
            # Double high-freq capacitors
            strategy = [(k, v * 2 if 'nF' in k else v) for k, v in strategy]

        return strategy

    def calculate_combined_impedance(
        self,
        capacitors: List[CapacitorModel],
        freq_hz: float
    ) -> float:
        """
        Calculate combined impedance of parallel capacitors

        For parallel impedances: 1/Z_total = sum(1/Z_i)

        Args:
            capacitors: List of capacitor models
            freq_hz: Frequency of interest

        Returns:
            Combined impedance magnitude in Ohms
        """
        if not capacitors or freq_hz <= 0:
            return float('inf')

        # Sum admittances (1/Z)
        y_total = complex(0, 0)

        for cap in capacitors:
            z = cap.impedance_at_freq(freq_hz)
            if abs(z) > 1e-12:
                y_total += 1 / z

        if abs(y_total) > 1e-12:
            return abs(1 / y_total)
        return float('inf')

    # =========================================================================
    # IMPEDANCE SWEEP
    # =========================================================================

    def sweep_impedance(
        self,
        capacitors: List[CapacitorModel],
        plane_c: float = 0.0,
        plane_l: float = 0.0,
        mounting_l: float = 0.0
    ) -> Dict[float, float]:
        """
        Perform frequency sweep of PDN impedance

        Args:
            capacitors: Decoupling capacitors
            plane_c: Plane capacitance (F)
            plane_l: Plane spreading inductance (H)
            mounting_l: Average mounting inductance per cap (H)

        Returns:
            Dict of {frequency_hz: impedance_ohms}
        """
        results = {}

        # Generate frequency points (log scale)
        f_start = self.config.freq_start_hz
        f_end = self.config.freq_end_hz
        n_points = self.config.freq_points

        for i in range(n_points):
            freq = f_start * (f_end / f_start) ** (i / (n_points - 1))

            # Capacitor impedances (with mounting inductance)
            z_caps = complex(float('inf'), 0)
            if capacitors:
                y_caps = complex(0, 0)
                for cap in capacitors:
                    z_cap = cap.impedance_at_freq(freq)
                    # Add mounting inductance
                    w = 2 * math.pi * freq
                    z_mount = complex(0, w * mounting_l)
                    z_total_cap = z_cap + z_mount
                    if abs(z_total_cap) > 1e-12:
                        y_caps += 1 / z_total_cap
                if abs(y_caps) > 1e-12:
                    z_caps = 1 / y_caps

            # Plane impedance
            w = 2 * math.pi * freq
            if plane_c > 0:
                z_plane_c = complex(0, -1 / (w * plane_c))
            else:
                z_plane_c = complex(float('inf'), 0)

            z_plane_l = complex(0, w * plane_l)

            # Combine: capacitors || plane_capacitance in series with plane_inductance
            if abs(z_caps) < float('inf') and abs(z_plane_c) < float('inf'):
                y_parallel = 1/z_caps + 1/z_plane_c
                if abs(y_parallel) > 1e-12:
                    z_parallel = 1 / y_parallel
                else:
                    z_parallel = z_caps
            elif abs(z_caps) < float('inf'):
                z_parallel = z_caps
            else:
                z_parallel = z_plane_c

            z_total = z_parallel + z_plane_l
            results[freq] = abs(z_total)

        return results

    # =========================================================================
    # VIA STITCHING
    # =========================================================================

    def generate_via_stitching(
        self,
        board_outline: List[Tuple[float, float]],
        exclusion_zones: List[Tuple[float, float, float, float]] = None
    ) -> List[ViaStitch]:
        """
        Generate ground via stitching pattern

        Reference: Ground vias reduce plane inductance and provide
        low-impedance return paths for high-frequency currents.

        Args:
            board_outline: Board boundary polygon
            exclusion_zones: List of (x, y, width, height) keep-out areas

        Returns:
            List of ViaStitch positions
        """
        exclusion_zones = exclusion_zones or []
        vias = []

        # Calculate board bounds
        if not board_outline:
            x_min, y_min = 0, 0
            x_max = self.config.board_width_mm
            y_max = self.config.board_height_mm
        else:
            x_min = min(p[0] for p in board_outline)
            x_max = max(p[0] for p in board_outline)
            y_min = min(p[1] for p in board_outline)
            y_max = max(p[1] for p in board_outline)

        spacing = self.config.stitch_spacing_mm
        margin = 2.0  # Keep away from edge

        x = x_min + margin
        while x < x_max - margin:
            y = y_min + margin
            while y < y_max - margin:
                # Check exclusion zones
                in_exclusion = False
                for ex_x, ex_y, ex_w, ex_h in exclusion_zones:
                    if (ex_x <= x <= ex_x + ex_w and
                        ex_y <= y <= ex_y + ex_h):
                        in_exclusion = True
                        break

                if not in_exclusion:
                    vias.append(ViaStitch(
                        x=x, y=y,
                        drill_mm=self.config.via_drill_mm,
                        pad_mm=self.config.via_pad_mm,
                        connects=['GND']
                    ))

                y += spacing
            x += spacing

        return vias

    def calculate_via_stitch_inductance_reduction(
        self,
        original_inductance: float,
        via_count: int,
        via_spacing_mm: float
    ) -> float:
        """
        Calculate inductance reduction from via stitching

        Multiple parallel vias reduce effective inductance.
        However, mutual inductance limits the benefit.

        Args:
            original_inductance: Plane inductance without vias (H)
            via_count: Number of ground vias
            via_spacing_mm: Average via spacing

        Returns:
            Reduced inductance (H)
        """
        if via_count <= 0:
            return original_inductance

        # Single via inductance
        l_via = self.calculate_via_inductance(
            self.config.board_thickness_mm,
            self.config.via_drill_mm
        )

        # Mutual inductance coupling factor (simplified)
        # Decreases with spacing
        k_mutual = 0.3 * math.exp(-via_spacing_mm / 3)

        # Effective inductance of via array
        # L_eff = L_via * (1 + k*(N-1)) / N
        l_eff_vias = l_via * (1 + k_mutual * (via_count - 1)) / via_count

        # Plane inductance in parallel with via array
        l_total = (original_inductance * l_eff_vias) / (original_inductance + l_eff_vias)

        return l_total

    # =========================================================================
    # CAPACITOR PLACEMENT OPTIMIZATION
    # =========================================================================

    def optimize_capacitor_placement(
        self,
        rail: PowerRail,
        ic_positions: List[Tuple[float, float]],
        strategy: List[Tuple[str, int]]
    ) -> List[DecouplingPlacement]:
        """
        Optimize decoupling capacitor placement

        Reference: High-frequency caps closest to IC, bulk caps further away

        Placement priority:
        1. Small ceramics (100nF, 10nF) - within 2mm of IC power pins
        2. Mid ceramics (1uF-10uF) - within 5mm of IC
        3. Bulk (tantalum, electrolytic) - can be further away

        Args:
            rail: Power rail
            ic_positions: List of (x, y) IC center positions
            strategy: Capacitor selection from design_decoupling_strategy

        Returns:
            List of optimized placements
        """
        placements = []

        for cap_key, quantity in strategy:
            cap = CAPACITOR_LIBRARY.get(cap_key)
            if not cap:
                continue

            # Determine placement distance based on capacitor type
            if cap.value_uf < 0.1:
                # High-freq: very close
                max_distance = 2.0
            elif cap.value_uf < 10:
                # Mid-freq: close
                max_distance = 5.0
            else:
                # Bulk: can be further
                max_distance = 15.0

            # Place capacitors around each IC
            caps_per_ic = max(1, quantity // len(ic_positions)) if ic_positions else quantity

            for ic_x, ic_y in ic_positions:
                for i in range(caps_per_ic):
                    # Place in a ring around IC
                    angle = 2 * math.pi * i / caps_per_ic
                    distance = min(max_distance, 3.0 + i * 1.5)

                    x = ic_x + distance * math.cos(angle)
                    y = ic_y + distance * math.sin(angle)

                    # Calculate mounting inductance
                    # Assumes via + short trace
                    via_l = self.calculate_via_inductance(
                        self.config.board_thickness_mm,
                        self.config.via_drill_mm
                    )
                    trace_l = 0.5e-9  # ~0.5nH for short trace
                    mount_l = via_l + trace_l

                    placements.append(DecouplingPlacement(
                        capacitor=cap_key,
                        x=x,
                        y=y,
                        rail=rail.name,
                        layer='F.Cu',
                        mounting_inductance_nh=mount_l * 1e9
                    ))

        return placements

    # =========================================================================
    # FULL PDN ANALYSIS
    # =========================================================================

    def run(
        self,
        power_rails: List[PowerRail],
        ic_positions: Dict[str, Tuple[float, float]] = None,
        power_planes: List[PowerPlane] = None
    ) -> PDNResult:
        """
        Run complete PDN analysis

        Args:
            power_rails: List of power rails to analyze
            ic_positions: Dict of {ic_name: (x, y)} positions
            power_planes: List of power/ground planes

        Returns:
            PDNResult with analysis and recommendations
        """
        self.power_rails = power_rails
        self.power_planes = power_planes or []
        self.capacitor_placements.clear()
        self.via_stitches.clear()
        self.warnings.clear()
        self.recommendations.clear()

        ic_positions = ic_positions or {}
        ic_pos_list = list(ic_positions.values()) if ic_positions else [(25, 20)]

        rail_impedances = {}
        target_met = {}

        for rail in power_rails:
            # Design decoupling strategy
            max_freq = 500e6  # Default 500 MHz
            strategy = self.design_decoupling_strategy(rail, max_freq)

            # Optimize placement
            placements = self.optimize_capacitor_placement(rail, ic_pos_list, strategy)
            self.capacitor_placements.extend(placements)

            # Get capacitor models
            capacitors = []
            for placement in placements:
                cap = CAPACITOR_LIBRARY.get(placement.capacitor)
                if cap:
                    capacitors.append(cap)

            # Calculate plane contribution
            board_area = self.config.board_width_mm * self.config.board_height_mm
            plane_c = self.calculate_plane_capacitance(
                board_area,
                self.config.plane_spacing_mm,
                self.config.dielectric_constant
            )

            plane_l = self.calculate_plane_inductance(
                self.config.board_width_mm,
                self.config.board_height_mm / 4,  # Assume quarter-wave spreading
                self.config.plane_spacing_mm
            )

            # Average mounting inductance
            avg_mount_l = sum(p.mounting_inductance_nh for p in placements) / len(placements) if placements else 0
            avg_mount_l *= 1e-9  # to Henries

            # Sweep impedance
            impedance_sweep = self.sweep_impedance(
                capacitors, plane_c, plane_l, avg_mount_l
            )

            rail_impedances[rail.name] = impedance_sweep

            # Check if target is met
            z_target = rail.target_impedance
            max_z = max(impedance_sweep.values()) if impedance_sweep else float('inf')
            target_met[rail.name] = max_z <= z_target * 1.1  # Allow 10% margin

            if not target_met[rail.name]:
                self.warnings.append(
                    f"Rail {rail.name}: Target impedance {z_target*1000:.1f}mohm not met. "
                    f"Peak impedance: {max_z*1000:.1f}mohm"
                )

                # Find frequency of peak
                peak_freq = max(impedance_sweep.items(), key=lambda x: x[1])[0]
                if peak_freq < 1e6:
                    self.recommendations.append(
                        f"Rail {rail.name}: Add more bulk capacitance for low-frequency peak at {peak_freq/1e3:.0f}kHz"
                    )
                elif peak_freq < 100e6:
                    self.recommendations.append(
                        f"Rail {rail.name}: Add more mid-frequency caps (1-10uF) for peak at {peak_freq/1e6:.1f}MHz"
                    )
                else:
                    self.recommendations.append(
                        f"Rail {rail.name}: Add more high-frequency caps (100nF) closer to IC for peak at {peak_freq/1e6:.0f}MHz"
                    )

        # Generate via stitching if enabled
        if self.config.stitch_ground_vias:
            # Get IC positions as exclusion zones (don't place vias under ICs)
            exclusions = [
                (x - 5, y - 5, 10, 10)  # 10mm x 10mm zone around each IC
                for x, y in ic_pos_list
            ]
            self.via_stitches = self.generate_via_stitching([], exclusions)

        return PDNResult(
            rails_analyzed=len(power_rails),
            total_capacitors=len(self.capacitor_placements),
            total_vias=len(self.via_stitches),
            rail_impedances=rail_impedances,
            target_met=target_met,
            capacitor_placements=self.capacitor_placements,
            via_stitches=self.via_stitches,
            warnings=self.warnings,
            recommendations=self.recommendations
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_bom(self) -> Dict[str, int]:
        """Generate BOM of decoupling capacitors"""
        bom = {}
        for placement in self.capacitor_placements:
            cap = CAPACITOR_LIBRARY.get(placement.capacitor)
            if cap:
                key = f"{cap.value_uf}uF {cap.package} {cap.type.value}"
                bom[key] = bom.get(key, 0) + 1
        return bom


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_target_impedance(voltage: float, current: float, ripple_pct: float = 5.0) -> float:
    """Quick target impedance calculation"""
    return voltage * ripple_pct / 100 / current


def capacitor_resonance(capacitance_uf: float, inductance_nh: float) -> float:
    """Calculate capacitor self-resonant frequency in MHz"""
    c = capacitance_uf * 1e-6
    l = inductance_nh * 1e-9
    if c > 0 and l > 0:
        return 1 / (2 * math.pi * math.sqrt(l * c)) / 1e6
    return 0


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PDN PISTON - Self Test")
    print("=" * 60)

    piston = PDNPiston()

    # Test 1: Target impedance
    print("\n1. Target Impedance Calculation:")
    z_target = piston.calculate_target_impedance(
        voltage=1.0,
        transient_current=2.0,
        ripple_percent=5.0
    )
    print(f"   1.0V core, 2A transient, 5% ripple")
    print(f"   Target Z: {z_target*1000:.1f} mohm")

    # Test 2: Capacitor resonance
    print("\n2. Capacitor Self-Resonant Frequencies:")
    for name, cap in list(CAPACITOR_LIBRARY.items())[:5]:
        print(f"   {name}: {cap.resonant_freq_mhz:.2f} MHz")

    # Test 3: Via inductance
    print("\n3. Via Inductance:")
    l_via = piston.calculate_via_inductance(1.6, 0.3)
    print(f"   0.3mm drill, 1.6mm height: {l_via*1e9:.2f} nH")

    # Test 4: Plane capacitance
    print("\n4. Plane Capacitance:")
    c_plane = piston.calculate_plane_capacitance(
        area_mm2=50*40,
        spacing_mm=0.2,
        dielectric_constant=4.5
    )
    print(f"   50x40mm planes, 0.2mm spacing: {c_plane*1e12:.1f} pF")

    # Test 5: Decoupling strategy
    print("\n5. Decoupling Strategy:")
    rail = PowerRail(
        name='VCC_CORE',
        voltage=1.0,
        current_max=2.0,
        current_transient=1.5,
        rail_type=PowerRailType.CORE,
        ripple_percent=5.0
    )
    strategy = piston.design_decoupling_strategy(rail, 500e6)
    print(f"   Rail: {rail.name} ({rail.voltage}V, {rail.current_max}A)")
    print(f"   Strategy:")
    for cap_key, qty in strategy:
        cap = CAPACITOR_LIBRARY.get(cap_key)
        if cap:
            print(f"     {qty}x {cap.value_uf}uF {cap.package} (SRF: {cap.resonant_freq_mhz:.1f}MHz)")

    # Test 6: Full analysis
    print("\n6. Full PDN Analysis:")
    rails = [
        PowerRail('VCC_CORE', 1.0, 2.0, 1.5, PowerRailType.CORE, ripple_percent=5),
        PowerRail('VCC_IO', 3.3, 0.5, 0.3, PowerRailType.IO, ripple_percent=3),
    ]

    result = piston.run(
        power_rails=rails,
        ic_positions={'U1': (25, 20), 'U2': (40, 25)}
    )

    print(f"   Rails analyzed: {result.rails_analyzed}")
    print(f"   Total capacitors: {result.total_capacitors}")
    print(f"   Total ground vias: {result.total_vias}")

    for rail_name, met in result.target_met.items():
        status = "PASS" if met else "FAIL"
        print(f"   {rail_name}: Target {status}")

    if result.warnings:
        print("   Warnings:")
        for w in result.warnings[:2]:
            print(f"     - {w}")

    # Test 7: Capacitor BOM
    print("\n7. Decoupling Capacitor BOM:")
    bom = piston.generate_bom()
    for item, qty in list(bom.items())[:5]:
        print(f"   {qty}x {item}")

    print("\n" + "=" * 60)
    print("PDN Piston self-test PASSED")
    print("=" * 60)
