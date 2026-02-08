"""
PCB Engine - Signal Integrity Piston (Sub-Engine)
===================================================

A dedicated piston (sub-engine) for signal integrity analysis.

ALGORITHMS & REFERENCES:
=========================
1. Reflection Coefficient
   Reference: Pozar, "Microwave Engineering"
   Formula: rho = (Z_L - Z_0) / (Z_L + Z_0)
   Voltage reflection: V_reflected = rho * V_incident

2. Transmission Line Delay
   Reference: Johnson & Graham, "High-Speed Digital Design"
   Formula: t_pd = sqrt(Dk_eff) / c * length

3. Crosstalk (Coupling)
   Reference: Hall, "Advanced Signal Integrity for High-Speed Digital Designs"
   Near-end: NEXT = K_b * length / t_rise (saturates at length = t_rise * v_p / 2)
   Far-end: FEXT = K_f * length / t_rise (for microstrip)

4. Eye Diagram Estimation
   Reference: IEEE Fundamentals of Signal Integrity
   Eye height = V_swing - 2*V_noise - ISI
   Eye width = UI - jitter

5. Rise Time / Bandwidth
   Reference: f_knee = 0.35 / t_rise (for 10-90% rise)
   Reference: f_3dB = 0.34 / t_rise

6. Lossy Line Analysis
   Reference: Skin effect: R_s = sqrt(pi * f * mu / sigma)
   Attenuation: alpha = R/(2*Z0) + G*Z0/2

CROSSTALK MECHANISMS:
======================
- Inductive coupling: mutual inductance between traces
- Capacitive coupling: electric field coupling
- Near-end crosstalk (NEXT): victim sees aggressor at near end
- Far-end crosstalk (FEXT): victim sees aggressor at far end

SIGNAL INTEGRITY RULES:
========================
- Critical length: L_crit = t_rise * v_p / 6
- Termination needed when: trace_length > L_crit
- 3W rule: spacing >= 3 * trace_width for <70% coupling
- 10H rule: ground plane distance from edge >= 10 * dielectric height

Sources:
- https://resources.altium.com/p/transmission-line-reflection-coefficient
- https://www.allaboutcircuits.com/tools/microstrip-crosstalk-calculator/
- https://en.wikipedia.org/wiki/Reflection_coefficient
- https://resources.pcb.cadence.com/blog/2021-how-to-use-a-transmission-line-reflection-coefficient-correctly
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

# Copper conductivity (S/m)
SIGMA_COPPER = 5.8e7

# Permeability of free space
MU_0 = 4 * math.pi * 1e-7


# =============================================================================
# ENUMS
# =============================================================================

class TerminationType(Enum):
    """Termination strategies"""
    NONE = 'none'
    SERIES = 'series'               # Series resistor at source
    PARALLEL = 'parallel'           # Parallel resistor at load
    THEVENIN = 'thevenin'           # Voltage divider at load
    AC = 'ac'                       # Capacitor + resistor at load
    DIODE = 'diode'                 # Schottky diode clamp


class SignalType(Enum):
    """Signal categories for analysis"""
    CLOCK = 'clock'
    DATA_SLOW = 'data_slow'         # <100 MHz
    DATA_FAST = 'data_fast'         # 100-500 MHz
    DATA_HIGHSPEED = 'data_highspeed'  # >500 MHz (DDR, SERDES)
    DIFFERENTIAL = 'differential'
    ANALOG = 'analog'
    POWER = 'power'


class CouplingType(Enum):
    """Trace coupling configuration"""
    EDGE_COUPLED = 'edge_coupled'     # Side by side on same layer
    BROADSIDE = 'broadside'           # Stacked vertically
    NONE = 'none'                     # No intentional coupling


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TraceGeometry:
    """Physical trace parameters"""
    width_mm: float
    thickness_mm: float = 0.035      # 1oz copper
    length_mm: float = 50.0

    # Dielectric
    height_mm: float = 0.2           # Height above reference plane
    dielectric_constant: float = 4.5

    # Layer
    layer: str = 'F.Cu'
    is_microstrip: bool = True       # vs stripline

    @property
    def effective_dk(self) -> float:
        """Effective dielectric constant for microstrip"""
        if self.is_microstrip:
            dk = self.dielectric_constant
            w_h = self.width_mm / self.height_mm
            # Schneider formula
            if w_h <= 1:
                dk_eff = (dk + 1) / 2 + (dk - 1) / 2 * (
                    1 / math.sqrt(1 + 12 / w_h) + 0.04 * (1 - w_h) ** 2
                )
            else:
                dk_eff = (dk + 1) / 2 + (dk - 1) / 2 / math.sqrt(1 + 12 / w_h)
            return dk_eff
        else:
            return self.dielectric_constant

    @property
    def propagation_velocity(self) -> float:
        """Propagation velocity (m/s)"""
        return C_LIGHT / math.sqrt(self.effective_dk)

    @property
    def delay_ps_per_mm(self) -> float:
        """Propagation delay in ps/mm"""
        return 1000 / (self.propagation_velocity / 1e6)  # ps/mm


@dataclass
class Signal:
    """Signal characteristics"""
    name: str
    signal_type: SignalType
    net_name: str

    # Electrical characteristics
    voltage_swing: float = 3.3       # Peak-to-peak voltage
    rise_time_ns: float = 1.0        # 10-90% rise time
    fall_time_ns: float = 1.0
    frequency_mhz: float = 100.0     # Fundamental frequency

    # Driver/receiver
    driver_impedance: float = 25.0   # Source impedance (ohms)
    receiver_impedance: float = 1e6  # Load impedance (ohms, high-Z)
    driver_current_ma: float = 8.0   # Drive current

    # Trace
    trace: Optional[TraceGeometry] = None

    @property
    def bandwidth_mhz(self) -> float:
        """Signal bandwidth (knee frequency)"""
        return 350 / self.rise_time_ns

    @property
    def wavelength_mm(self) -> float:
        """Wavelength at fundamental frequency"""
        if self.trace and self.frequency_mhz > 0:
            v = self.trace.propagation_velocity
            return v / (self.frequency_mhz * 1e6) * 1000
        return 0


@dataclass
class CrosstalkPair:
    """Aggressor-victim trace pair for crosstalk analysis"""
    aggressor: Signal
    victim: Signal

    spacing_mm: float              # Edge-to-edge spacing
    coupling_length_mm: float      # Parallel run length
    coupling_type: CouplingType = CouplingType.EDGE_COUPLED


@dataclass
class ReflectionResult:
    """Result of reflection analysis"""
    trace_name: str
    z0: float                      # Characteristic impedance
    z_source: float                # Source impedance
    z_load: float                  # Load impedance

    rho_source: float              # Reflection coefficient at source
    rho_load: float                # Reflection coefficient at load

    v_incident: float              # Incident voltage
    v_reflected_load: float        # First reflection at load
    v_reflected_source: float      # Reflection back from source

    # Settling
    settling_time_ns: float
    overshoot_percent: float
    undershoot_percent: float

    # Recommendations
    termination_needed: bool
    recommended_termination: TerminationType
    termination_value: float


@dataclass
class CrosstalkResult:
    """Result of crosstalk analysis"""
    aggressor: str
    victim: str

    next_mv: float                 # Near-end crosstalk voltage
    next_percent: float            # As percentage of aggressor
    fext_mv: float                 # Far-end crosstalk voltage
    fext_percent: float

    # Timing
    next_risetime_ns: float
    fext_risetime_ns: float

    # Recommendations
    acceptable: bool
    recommendations: List[str]


@dataclass
class EyeDiagramResult:
    """Estimated eye diagram parameters"""
    signal_name: str

    # Eye dimensions
    eye_height_mv: float
    eye_width_ps: float
    eye_opening_ratio: float       # 0-1, 1 = perfect

    # Jitter components
    deterministic_jitter_ps: float
    random_jitter_rms_ps: float
    total_jitter_ps: float

    # Margin
    voltage_margin_percent: float
    timing_margin_percent: float

    # Pass/fail
    meets_spec: bool


@dataclass
class SIResult:
    """Complete signal integrity analysis result"""
    signals_analyzed: int
    critical_signals: int

    reflection_results: List[ReflectionResult]
    crosstalk_results: List[CrosstalkResult]
    eye_results: List[EyeDiagramResult]

    warnings: List[str]
    recommendations: List[str]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SIConfig:
    """Configuration for signal integrity analysis"""
    # Default trace parameters
    default_z0: float = 50.0
    default_trace_width_mm: float = 0.2
    default_dielectric_height_mm: float = 0.2
    default_dk: float = 4.5

    # Analysis thresholds
    max_reflection_percent: float = 10.0    # Max acceptable reflection
    max_crosstalk_percent: float = 5.0      # Max acceptable crosstalk
    max_overshoot_percent: float = 10.0     # Max overshoot
    min_eye_opening: float = 0.7            # Minimum eye opening ratio

    # Termination
    auto_terminate: bool = True

    # Crosstalk rules
    min_spacing_3w: bool = True             # Enforce 3W rule
    guard_trace_threshold_mhz: float = 200  # Add guards above this freq


# =============================================================================
# SIGNAL INTEGRITY PISTON CLASS
# =============================================================================

class SignalIntegrityPiston:
    """
    Signal Integrity Analysis Piston

    Provides:
    1. Reflection coefficient and waveform analysis
    2. Crosstalk (NEXT/FEXT) calculation
    3. Eye diagram estimation
    4. Termination strategy recommendation
    5. Guard trace insertion
    """

    def __init__(self, config: Optional[SIConfig] = None):
        self.config = config or SIConfig()
        self.signals: List[Signal] = []
        self.crosstalk_pairs: List[CrosstalkPair] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []

    # =========================================================================
    # STANDARD PISTON API
    # =========================================================================

    def analyze(self, routes: Dict) -> Dict[str, Any]:
        """
        Standard piston API - analyze signal integrity of routes.

        Args:
            routes: Routing data (Dict of Route objects)

        Returns:
            Dictionary with SI analysis results
        """
        self.warnings.clear()
        self.recommendations.clear()
        self.signals.clear()

        # Analyze each route
        for net_name, route in routes.items():
            if hasattr(route, 'segments'):
                total_length = sum(
                    math.sqrt(
                        (s.end[0] - s.start[0])**2 + (s.end[1] - s.start[1])**2
                    ) if hasattr(s, 'start') else 0
                    for s in route.segments
                )

                # Check for SI concerns
                if total_length > 50:  # mm
                    self.warnings.append(f"Net {net_name} is {total_length:.1f}mm - may need impedance control")

        return {
            'signals_analyzed': len(routes),
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

    # =========================================================================
    # IMPEDANCE CALCULATION
    # =========================================================================

    def calculate_microstrip_z0(
        self,
        width_mm: float,
        height_mm: float,
        thickness_mm: float = 0.035,
        dk: float = 4.5
    ) -> float:
        """
        Calculate microstrip characteristic impedance

        Uses Wadell equations (same as stackup_piston)

        Args:
            width_mm: Trace width
            height_mm: Dielectric height
            thickness_mm: Copper thickness
            dk: Dielectric constant

        Returns:
            Characteristic impedance in ohms
        """
        w = width_mm
        h = height_mm
        t = thickness_mm

        # Effective width
        w_eff = w + (t / math.pi) * (1 + math.log(2 * h / t)) if t > 0 else w

        u = w_eff / h

        # Effective dielectric constant
        if u <= 1:
            dk_eff = (dk + 1) / 2 + (dk - 1) / 2 * (
                1 / math.sqrt(1 + 12 / u) + 0.04 * (1 - u) ** 2
            )
        else:
            dk_eff = (dk + 1) / 2 + (dk - 1) / 2 / math.sqrt(1 + 12 / u)

        # Characteristic impedance
        if u <= 1:
            z0 = (60 / math.sqrt(dk_eff)) * math.log(8 / u + u / 4)
        else:
            z0 = (120 * math.pi / math.sqrt(dk_eff)) / (u + 1.393 + 0.667 * math.log(u + 1.444))

        return z0

    # =========================================================================
    # REFLECTION ANALYSIS
    # =========================================================================

    def calculate_reflection_coefficient(
        self,
        z0: float,
        z_load: float
    ) -> complex:
        """
        Calculate voltage reflection coefficient

        rho = (Z_L - Z_0) / (Z_L + Z_0)

        Args:
            z0: Characteristic impedance
            z_load: Load impedance

        Returns:
            Complex reflection coefficient
        """
        if z0 + z_load == 0:
            return complex(0, 0)
        return (z_load - z0) / (z_load + z0)

    def analyze_reflections(self, signal: Signal) -> ReflectionResult:
        """
        Analyze signal reflections on a transmission line

        Uses lattice diagram approach to track multiple reflections.

        Args:
            signal: Signal to analyze

        Returns:
            ReflectionResult with reflection analysis
        """
        if not signal.trace:
            # Create default trace
            signal.trace = TraceGeometry(
                width_mm=self.config.default_trace_width_mm,
                height_mm=self.config.default_dielectric_height_mm,
                length_mm=50.0,
                dielectric_constant=self.config.default_dk
            )

        trace = signal.trace

        # Calculate Z0
        z0 = self.calculate_microstrip_z0(
            trace.width_mm,
            trace.height_mm,
            trace.thickness_mm,
            trace.dielectric_constant
        )

        z_source = signal.driver_impedance
        z_load = signal.receiver_impedance

        # Reflection coefficients
        rho_source = self.calculate_reflection_coefficient(z0, z_source)
        rho_load = self.calculate_reflection_coefficient(z0, z_load)

        # Initial voltage (voltage divider at source)
        v_initial = signal.voltage_swing * z0 / (z_source + z0)

        # First reflection at load
        v_reflected_load = v_initial * rho_load.real

        # Reflection back at source
        v_reflected_source = v_reflected_load * rho_source.real

        # Calculate settling
        propagation_delay_ns = trace.length_mm * trace.delay_ps_per_mm / 1000

        # Settling time (when reflections < 5% of signal)
        # Each round trip reduces by rho_source * rho_load
        attenuation_per_trip = abs(rho_source * rho_load)
        if attenuation_per_trip < 1:
            trips_to_settle = math.log(0.05) / math.log(attenuation_per_trip) if attenuation_per_trip > 0 else 1
        else:
            trips_to_settle = 10  # Won't settle

        settling_time = 2 * propagation_delay_ns * trips_to_settle

        # Overshoot/undershoot
        if rho_load.real > 0:
            overshoot = rho_load.real * 100
            undershoot = 0
        else:
            overshoot = 0
            undershoot = abs(rho_load.real) * 100

        # Termination recommendation
        critical_length = signal.rise_time_ns * trace.propagation_velocity / 1e6 / 6
        needs_termination = trace.length_mm > critical_length

        if needs_termination:
            if z_source < z0:
                # Source impedance low, use series termination
                term_type = TerminationType.SERIES
                term_value = z0 - z_source
            else:
                # Use parallel termination at receiver
                term_type = TerminationType.PARALLEL
                term_value = z0
        else:
            term_type = TerminationType.NONE
            term_value = 0

        return ReflectionResult(
            trace_name=signal.name,
            z0=z0,
            z_source=z_source,
            z_load=z_load,
            rho_source=rho_source.real,
            rho_load=rho_load.real,
            v_incident=v_initial,
            v_reflected_load=v_reflected_load,
            v_reflected_source=v_reflected_source,
            settling_time_ns=settling_time,
            overshoot_percent=overshoot,
            undershoot_percent=undershoot,
            termination_needed=needs_termination,
            recommended_termination=term_type,
            termination_value=term_value
        )

    # =========================================================================
    # CROSSTALK ANALYSIS
    # =========================================================================

    def calculate_crosstalk(self, pair: CrosstalkPair) -> CrosstalkResult:
        """
        Calculate near-end and far-end crosstalk

        Reference: Johnson & Graham, "High-Speed Digital Design"

        NEXT (backward crosstalk):
        - Rises while aggressor edge traverses coupled length
        - Saturates at length = t_rise * v_p / 2

        FEXT (forward crosstalk):
        - Appears at far end of victim
        - Proportional to (coupled_length / t_rise)
        - Microstrip: FEXT exists due to unequal L and C coupling
        - Stripline: FEXT cancels (homogeneous medium)

        Args:
            pair: Aggressor-victim pair

        Returns:
            CrosstalkResult
        """
        aggressor = pair.aggressor
        victim = pair.victim
        s = pair.spacing_mm
        L = pair.coupling_length_mm

        # Get trace geometry
        if aggressor.trace:
            w = aggressor.trace.width_mm
            h = aggressor.trace.height_mm
            dk = aggressor.trace.dielectric_constant
            is_microstrip = aggressor.trace.is_microstrip
        else:
            w = self.config.default_trace_width_mm
            h = self.config.default_dielectric_height_mm
            dk = self.config.default_dk
            is_microstrip = True

        # Propagation velocity
        if is_microstrip:
            dk_eff = (dk + 1) / 2 + (dk - 1) / 2 / math.sqrt(1 + 12 * h / w)
        else:
            dk_eff = dk
        v_p = C_LIGHT / math.sqrt(dk_eff)  # m/s

        t_rise = aggressor.rise_time_ns * 1e-9  # seconds
        L_m = L * 1e-3  # meters

        # Coupling coefficients (simplified model)
        # Based on edge-coupled microstrip
        ratio = s / h

        # Backward (NEXT) coupling coefficient
        # Decreases with spacing, approximately as 1/(1 + s/h)^2
        Kb = 0.25 / (1 + ratio) ** 2

        # Forward (FEXT) coupling coefficient
        # Only significant for microstrip (inhomogeneous medium)
        if is_microstrip:
            # FEXT coefficient depends on velocity difference between even/odd modes
            Kf = 0.1 / (1 + ratio) ** 2
        else:
            Kf = 0  # Stripline has no FEXT

        # Saturation length for NEXT
        L_sat = t_rise * v_p / 2  # meters

        # NEXT calculation
        if L_m <= L_sat:
            # Linear region
            next_coefficient = Kb * L_m / L_sat
        else:
            # Saturated
            next_coefficient = Kb

        next_mv = next_coefficient * aggressor.voltage_swing * 1000

        # FEXT calculation
        # FEXT = Kf * (L / (t_rise * v_p))
        fext_coefficient = Kf * L_m / (t_rise * v_p)
        fext_mv = fext_coefficient * aggressor.voltage_swing * 1000

        # As percentage
        next_percent = next_mv / (aggressor.voltage_swing * 1000) * 100
        fext_percent = fext_mv / (aggressor.voltage_swing * 1000) * 100

        # Rise times of crosstalk pulses
        # NEXT has same rise time as aggressor
        next_risetime = aggressor.rise_time_ns

        # FEXT is proportional to derivative, so faster
        fext_risetime = aggressor.rise_time_ns * 0.35

        # Recommendations
        recommendations = []
        acceptable = True

        if next_percent > self.config.max_crosstalk_percent:
            acceptable = False
            recommendations.append(
                f"Increase spacing to {3 * w:.2f}mm (3W rule) to reduce NEXT"
            )

        if fext_percent > self.config.max_crosstalk_percent:
            acceptable = False
            recommendations.append(
                f"Reduce parallel run length or add guard trace to reduce FEXT"
            )

        if L > 50 and aggressor.frequency_mhz > 100:
            recommendations.append(
                "Consider adding ground guard trace between aggressor and victim"
            )

        return CrosstalkResult(
            aggressor=aggressor.name,
            victim=victim.name,
            next_mv=next_mv,
            next_percent=next_percent,
            fext_mv=fext_mv,
            fext_percent=fext_percent,
            next_risetime_ns=next_risetime,
            fext_risetime_ns=fext_risetime,
            acceptable=acceptable,
            recommendations=recommendations
        )

    # =========================================================================
    # EYE DIAGRAM ESTIMATION
    # =========================================================================

    def estimate_eye_diagram(
        self,
        signal: Signal,
        reflections: ReflectionResult,
        crosstalk_noise_mv: float = 0
    ) -> EyeDiagramResult:
        """
        Estimate eye diagram parameters

        Eye opening is reduced by:
        - Inter-symbol interference (ISI) from reflections
        - Crosstalk noise
        - Timing jitter

        Args:
            signal: Signal being analyzed
            reflections: Reflection analysis result
            crosstalk_noise_mv: Crosstalk contribution to noise

        Returns:
            EyeDiagramResult
        """
        v_swing = signal.voltage_swing * 1000  # mV

        # ISI from reflections (simplified)
        # Each reflection contributes to ISI
        isi_mv = abs(reflections.v_reflected_load) * 1000

        # Total noise
        noise_mv = isi_mv + crosstalk_noise_mv

        # Eye height
        eye_height = v_swing - 2 * noise_mv
        if eye_height < 0:
            eye_height = 0

        # Timing
        ui_ps = 1e6 / signal.frequency_mhz  # Unit interval in ps

        # Deterministic jitter (from reflections)
        dj_ps = reflections.settling_time_ns * 1000 * 0.1  # 10% of settling

        # Random jitter (assumed)
        rj_rms_ps = signal.rise_time_ns * 1000 * 0.05  # 5% of rise time

        # Total jitter (simplified: DJ + 14*RJ for BER=1e-12)
        tj_ps = dj_ps + 14 * rj_rms_ps

        # Eye width
        eye_width = ui_ps - tj_ps
        if eye_width < 0:
            eye_width = 0

        # Opening ratio
        height_ratio = eye_height / v_swing if v_swing > 0 else 0
        width_ratio = eye_width / ui_ps if ui_ps > 0 else 0
        opening_ratio = height_ratio * width_ratio

        # Margins
        voltage_margin = height_ratio * 100
        timing_margin = width_ratio * 100

        meets_spec = opening_ratio >= self.config.min_eye_opening

        return EyeDiagramResult(
            signal_name=signal.name,
            eye_height_mv=eye_height,
            eye_width_ps=eye_width,
            eye_opening_ratio=opening_ratio,
            deterministic_jitter_ps=dj_ps,
            random_jitter_rms_ps=rj_rms_ps,
            total_jitter_ps=tj_ps,
            voltage_margin_percent=voltage_margin,
            timing_margin_percent=timing_margin,
            meets_spec=meets_spec
        )

    # =========================================================================
    # TERMINATION DESIGN
    # =========================================================================

    def design_termination(
        self,
        signal: Signal,
        z0: float,
        term_type: TerminationType
    ) -> Dict[str, Any]:
        """
        Design termination network

        Args:
            signal: Signal to terminate
            z0: Characteristic impedance
            term_type: Termination strategy

        Returns:
            Dict with component values
        """
        result = {'type': term_type.value}

        if term_type == TerminationType.SERIES:
            # Series resistor at source
            r_series = z0 - signal.driver_impedance
            if r_series < 0:
                r_series = 0
            result['R_series'] = r_series
            result['location'] = 'source'

        elif term_type == TerminationType.PARALLEL:
            # Parallel resistor at load
            result['R_parallel'] = z0
            result['location'] = 'load'
            result['power_mw'] = (signal.voltage_swing ** 2 / z0) * 1000

        elif term_type == TerminationType.THEVENIN:
            # Voltage divider termination
            # Two resistors to VCC and GND
            r_term = 2 * z0  # Each resistor
            result['R_high'] = r_term
            result['R_low'] = r_term
            result['v_bias'] = signal.voltage_swing / 2
            result['location'] = 'load'

        elif term_type == TerminationType.AC:
            # AC termination (RC)
            # C blocks DC, R provides termination at high freq
            r_ac = z0
            # C should pass signal frequency with low impedance
            # Xc = 1/(2*pi*f*C) << R at signal frequency
            c_ac = 10 / (2 * math.pi * signal.frequency_mhz * 1e6 * z0) * 1e9  # nF
            result['R_ac'] = r_ac
            result['C_ac_nf'] = c_ac
            result['location'] = 'load'

        return result

    # =========================================================================
    # GUARD TRACE DESIGN
    # =========================================================================

    def design_guard_trace(
        self,
        aggressor_trace: TraceGeometry,
        spacing_mm: float
    ) -> Dict[str, Any]:
        """
        Design guard trace for crosstalk reduction

        Guard traces should be:
        - Grounded at both ends
        - Grounded at intervals < lambda/20 for high frequencies
        - Same layer as protected traces

        Args:
            aggressor_trace: Trace to guard against
            spacing_mm: Spacing to protected trace

        Returns:
            Guard trace specifications
        """
        # Guard trace width (typically same as signal trace)
        guard_width = aggressor_trace.width_mm

        # Via spacing for grounding
        # At frequency f, lambda = v_p / f
        # Via spacing should be < lambda/20
        # For typical 500MHz signal on FR4:
        # lambda = 0.5 * c / sqrt(4.5) = 0.5 * 3e8 / 2.12 = 70mm
        # Spacing should be < 3.5mm

        via_spacing = min(10.0, spacing_mm * 3)

        return {
            'width_mm': guard_width,
            'spacing_from_victim_mm': spacing_mm / 2,
            'spacing_from_aggressor_mm': spacing_mm / 2,
            'ground_via_spacing_mm': via_spacing,
            'layer': aggressor_trace.layer,
            'effectiveness_db': 6 + 20 * math.log10(spacing_mm / guard_width) if guard_width > 0 else 0
        }

    # =========================================================================
    # CRITICAL LENGTH CALCULATION
    # =========================================================================

    def calculate_critical_length(
        self,
        rise_time_ns: float,
        dk_eff: float = 3.5
    ) -> float:
        """
        Calculate critical trace length for transmission line effects

        A trace becomes a transmission line when:
        length > t_rise * v_p / 6

        Below this length, lumped analysis is valid.

        Args:
            rise_time_ns: Signal rise time
            dk_eff: Effective dielectric constant

        Returns:
            Critical length in mm
        """
        v_p = C_LIGHT / math.sqrt(dk_eff)  # m/s
        t_rise = rise_time_ns * 1e-9  # seconds

        # Critical length in meters, divided by 6 for safety
        l_crit = t_rise * v_p / 6

        return l_crit * 1000  # mm

    # =========================================================================
    # LOSSY LINE ANALYSIS
    # =========================================================================

    def calculate_skin_effect_resistance(
        self,
        trace: TraceGeometry,
        frequency_hz: float
    ) -> float:
        """
        Calculate AC resistance due to skin effect

        At high frequencies, current crowds near the surface,
        increasing resistance.

        R_ac = R_dc * (1 + F(f/f_0))

        Where f_0 is the frequency where skin depth = trace thickness

        Args:
            trace: Trace geometry
            frequency_hz: Frequency of interest

        Returns:
            AC resistance per unit length (ohm/m)
        """
        w = trace.width_mm * 1e-3  # meters
        t = trace.thickness_mm * 1e-3  # meters

        # DC resistance per unit length
        rho = 1 / SIGMA_COPPER  # resistivity
        r_dc = rho / (w * t)  # ohm/m

        # Skin depth
        delta = math.sqrt(rho / (math.pi * frequency_hz * MU_0))

        if delta < t / 2:
            # Skin effect significant
            # Effective thickness becomes 2*delta (top and bottom surfaces)
            r_ac = rho / (w * 2 * delta)
        else:
            r_ac = r_dc

        return r_ac

    def calculate_insertion_loss(
        self,
        trace: TraceGeometry,
        frequency_hz: float,
        z0: float
    ) -> float:
        """
        Calculate trace insertion loss in dB

        Loss = conductor_loss + dielectric_loss

        Args:
            trace: Trace geometry
            frequency_hz: Frequency
            z0: Characteristic impedance

        Returns:
            Insertion loss in dB
        """
        length_m = trace.length_mm * 1e-3

        # Conductor loss
        r_ac = self.calculate_skin_effect_resistance(trace, frequency_hz)
        alpha_c = r_ac / (2 * z0)  # Np/m

        # Dielectric loss
        # alpha_d = pi * f * sqrt(dk_eff) * tan_delta / c
        tan_delta = 0.02  # FR4 loss tangent
        dk_eff = trace.effective_dk
        alpha_d = math.pi * frequency_hz * math.sqrt(dk_eff) * tan_delta / C_LIGHT  # Np/m

        # Total loss
        alpha_total = alpha_c + alpha_d
        loss_db = 20 * math.log10(math.e) * alpha_total * length_m

        return loss_db

    # =========================================================================
    # FULL ANALYSIS
    # =========================================================================

    def run(
        self,
        signals: List[Signal],
        crosstalk_pairs: Optional[List[CrosstalkPair]] = None
    ) -> SIResult:
        """
        Run complete signal integrity analysis

        Args:
            signals: List of signals to analyze
            crosstalk_pairs: Optional list of crosstalk pairs

        Returns:
            SIResult with complete analysis
        """
        self.signals = signals
        self.crosstalk_pairs = crosstalk_pairs or []
        self.warnings.clear()
        self.recommendations.clear()

        reflection_results = []
        crosstalk_results = []
        eye_results = []
        critical_count = 0

        # Analyze each signal
        for signal in signals:
            # Reflection analysis
            ref_result = self.analyze_reflections(signal)
            reflection_results.append(ref_result)

            if ref_result.termination_needed:
                self.warnings.append(
                    f"{signal.name}: Trace length exceeds critical length, "
                    f"termination recommended"
                )
                critical_count += 1

            if ref_result.overshoot_percent > self.config.max_overshoot_percent:
                self.warnings.append(
                    f"{signal.name}: Overshoot {ref_result.overshoot_percent:.1f}% "
                    f"exceeds {self.config.max_overshoot_percent}% limit"
                )

        # Crosstalk analysis
        for pair in self.crosstalk_pairs:
            xt_result = self.calculate_crosstalk(pair)
            crosstalk_results.append(xt_result)

            if not xt_result.acceptable:
                self.warnings.extend([
                    f"Crosstalk {xt_result.aggressor} -> {xt_result.victim}: " + r
                    for r in xt_result.recommendations
                ])

        # Eye diagram analysis for high-speed signals
        for signal in signals:
            if signal.signal_type in (SignalType.DATA_FAST, SignalType.DATA_HIGHSPEED):
                ref_result = next(
                    (r for r in reflection_results if r.trace_name == signal.name),
                    None
                )

                if ref_result:
                    # Get crosstalk contribution
                    xt_noise = sum(
                        xt.next_mv + xt.fext_mv
                        for xt in crosstalk_results
                        if xt.victim == signal.name
                    )

                    eye_result = self.estimate_eye_diagram(signal, ref_result, xt_noise)
                    eye_results.append(eye_result)

                    if not eye_result.meets_spec:
                        self.warnings.append(
                            f"{signal.name}: Eye opening {eye_result.eye_opening_ratio:.2f} "
                            f"below minimum {self.config.min_eye_opening}"
                        )

        # Generate recommendations
        self._generate_recommendations(reflection_results, crosstalk_results)

        return SIResult(
            signals_analyzed=len(signals),
            critical_signals=critical_count,
            reflection_results=reflection_results,
            crosstalk_results=crosstalk_results,
            eye_results=eye_results,
            warnings=self.warnings,
            recommendations=self.recommendations
        )

    def _generate_recommendations(
        self,
        reflections: List[ReflectionResult],
        crosstalk: List[CrosstalkResult]
    ):
        """Generate signal integrity recommendations"""

        # Termination recommendations
        for ref in reflections:
            if ref.termination_needed:
                term = self.design_termination(
                    next(s for s in self.signals if s.name == ref.trace_name),
                    ref.z0,
                    ref.recommended_termination
                )
                if ref.recommended_termination == TerminationType.SERIES:
                    self.recommendations.append(
                        f"{ref.trace_name}: Add {term['R_series']:.0f} ohm series "
                        f"resistor at source"
                    )
                elif ref.recommended_termination == TerminationType.PARALLEL:
                    self.recommendations.append(
                        f"{ref.trace_name}: Add {term['R_parallel']:.0f} ohm parallel "
                        f"resistor at load (power: {term['power_mw']:.1f}mW)"
                    )

        # Crosstalk recommendations
        for xt in crosstalk:
            if not xt.acceptable:
                self.recommendations.extend(xt.recommendations)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_reflection(z0: float, z_load: float) -> float:
    """Quick reflection coefficient calculation"""
    return abs((z_load - z0) / (z_load + z0))


def critical_trace_length(rise_time_ns: float, dk: float = 4.5) -> float:
    """Quick critical length calculation in mm"""
    v_p = C_LIGHT / math.sqrt(dk)
    return rise_time_ns * 1e-9 * v_p / 6 * 1000


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SIGNAL INTEGRITY PISTON - Self Test")
    print("=" * 60)

    piston = SignalIntegrityPiston()

    # Test 1: Impedance calculation
    print("\n1. Microstrip Impedance:")
    z0 = piston.calculate_microstrip_z0(
        width_mm=0.3,
        height_mm=0.2,
        thickness_mm=0.035,
        dk=4.5
    )
    print(f"   0.3mm trace, 0.2mm height, Dk=4.5")
    print(f"   Z0 = {z0:.1f} ohm")

    # Test 2: Reflection coefficient
    print("\n2. Reflection Coefficient:")
    rho = piston.calculate_reflection_coefficient(50, 1e6)
    print(f"   Z0=50 ohm, ZL=1M ohm (open)")
    print(f"   rho = {rho.real:.3f} (voltage nearly doubles)")

    rho = piston.calculate_reflection_coefficient(50, 25)
    print(f"   Z0=50 ohm, ZL=25 ohm")
    print(f"   rho = {rho.real:.3f} (negative reflection)")

    # Test 3: Critical length
    print("\n3. Critical Trace Length:")
    for tr in [0.5, 1.0, 2.0, 5.0]:
        l_crit = piston.calculate_critical_length(tr)
        print(f"   Rise time {tr}ns: critical length = {l_crit:.1f}mm")

    # Test 4: Full reflection analysis
    print("\n4. Reflection Analysis:")
    signal = Signal(
        name='CLK_100M',
        signal_type=SignalType.CLOCK,
        net_name='CLK',
        voltage_swing=3.3,
        rise_time_ns=1.0,
        frequency_mhz=100,
        driver_impedance=25,
        receiver_impedance=1e6,
        trace=TraceGeometry(
            width_mm=0.2,
            height_mm=0.2,
            length_mm=80,
            dielectric_constant=4.5
        )
    )

    ref_result = piston.analyze_reflections(signal)
    print(f"   Signal: {signal.name}")
    print(f"   Z0: {ref_result.z0:.1f} ohm")
    print(f"   Rho_load: {ref_result.rho_load:.3f}")
    print(f"   Overshoot: {ref_result.overshoot_percent:.1f}%")
    print(f"   Settling: {ref_result.settling_time_ns:.1f}ns")
    print(f"   Needs termination: {ref_result.termination_needed}")
    if ref_result.termination_needed:
        print(f"   Recommended: {ref_result.recommended_termination.value} "
              f"({ref_result.termination_value:.0f} ohm)")

    # Test 5: Crosstalk
    print("\n5. Crosstalk Analysis:")
    aggressor = Signal('DATA0', SignalType.DATA_FAST, 'D0', voltage_swing=3.3, rise_time_ns=1.0)
    victim = Signal('DATA1', SignalType.DATA_FAST, 'D1', voltage_swing=3.3, rise_time_ns=1.0)
    aggressor.trace = TraceGeometry(width_mm=0.2, height_mm=0.2, length_mm=50)
    victim.trace = TraceGeometry(width_mm=0.2, height_mm=0.2, length_mm=50)

    pair = CrosstalkPair(
        aggressor=aggressor,
        victim=victim,
        spacing_mm=0.3,
        coupling_length_mm=30
    )

    xt_result = piston.calculate_crosstalk(pair)
    print(f"   Spacing: {pair.spacing_mm}mm, Coupling: {pair.coupling_length_mm}mm")
    print(f"   NEXT: {xt_result.next_mv:.1f}mV ({xt_result.next_percent:.2f}%)")
    print(f"   FEXT: {xt_result.fext_mv:.1f}mV ({xt_result.fext_percent:.2f}%)")
    print(f"   Acceptable: {xt_result.acceptable}")

    # Test 6: Eye diagram
    print("\n6. Eye Diagram Estimation:")
    eye = piston.estimate_eye_diagram(signal, ref_result, xt_result.next_mv)
    print(f"   Eye height: {eye.eye_height_mv:.0f}mV")
    print(f"   Eye width: {eye.eye_width_ps:.0f}ps")
    print(f"   Opening ratio: {eye.eye_opening_ratio:.2f}")
    print(f"   Total jitter: {eye.total_jitter_ps:.1f}ps")
    print(f"   Meets spec: {eye.meets_spec}")

    # Test 7: Insertion loss
    print("\n7. Insertion Loss (100mm trace at 1GHz):")
    trace = TraceGeometry(width_mm=0.2, height_mm=0.2, length_mm=100)
    loss = piston.calculate_insertion_loss(trace, 1e9, 50)
    print(f"   Loss: {loss:.2f} dB")

    print("\n" + "=" * 60)
    print("Signal Integrity Piston self-test PASSED")
    print("=" * 60)
