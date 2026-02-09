"""
Electrical Engineering Calculator
==================================

A comprehensive calculator for all electrical engineering formulas
that a PCB designer needs. This is essential for the Circuit
Intelligence Engine to make informed decisions.

Categories:
1. Power calculations
2. Impedance calculations
3. Signal integrity
4. Thermal calculations
5. Component selection
6. EMI/EMC calculations
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


# =============================================================================
# CONSTANTS
# =============================================================================

class Constants:
    """Physical constants."""
    EPSILON_0 = 8.854e-12  # Permittivity of free space (F/m)
    MU_0 = 4 * math.pi * 1e-7  # Permeability of free space (H/m)
    C = 299792458  # Speed of light (m/s)
    COPPER_RESISTIVITY = 1.68e-8  # Ohm·m at 20°C
    COPPER_TEMP_COEFF = 0.00393  # Per °C


# =============================================================================
# POWER CALCULATIONS
# =============================================================================

class PowerCalculator:
    """Power-related calculations."""

    @staticmethod
    def power_from_vi(voltage: float, current: float) -> float:
        """P = V × I"""
        return voltage * current

    @staticmethod
    def power_from_ir(current: float, resistance: float) -> float:
        """P = I² × R"""
        return current ** 2 * resistance

    @staticmethod
    def power_from_vr(voltage: float, resistance: float) -> float:
        """P = V² / R"""
        return voltage ** 2 / resistance if resistance > 0 else 0

    @staticmethod
    def ldo_power_dissipation(vin: float, vout: float, current: float) -> float:
        """
        Power dissipation in LDO regulator.
        P = (Vin - Vout) × Iload
        """
        dropout = vin - vout
        return max(0, dropout * current)

    @staticmethod
    def switching_regulator_efficiency(
        vin: float, vout: float, iout: float,
        inductor_dcr: float = 0.1,
        switch_rds: float = 0.05,
        duty_cycle: float = None
    ) -> Tuple[float, float]:
        """
        Estimate switching regulator efficiency and losses.

        Returns: (efficiency, power_loss)
        """
        if duty_cycle is None:
            # Buck converter duty cycle
            duty_cycle = vout / vin if vin > 0 else 0

        # Conduction losses
        switch_loss = iout ** 2 * switch_rds * duty_cycle
        inductor_loss = iout ** 2 * inductor_dcr

        # Switching losses (simplified estimate)
        switching_loss = 0.05 * vout * iout  # ~5% for switching

        total_loss = switch_loss + inductor_loss + switching_loss
        output_power = vout * iout
        input_power = output_power + total_loss

        efficiency = output_power / input_power if input_power > 0 else 0

        return (efficiency, total_loss)

    @staticmethod
    def led_resistor(supply_voltage: float, led_vf: float,
                     led_current_ma: float) -> Tuple[float, float]:
        """
        Calculate LED current limiting resistor.

        Returns: (resistance_ohms, power_watts)
        """
        voltage_drop = supply_voltage - led_vf
        current_a = led_current_ma / 1000

        if voltage_drop <= 0 or current_a <= 0:
            return (0, 0)

        resistance = voltage_drop / current_a
        power = voltage_drop * current_a

        return (resistance, power)

    @staticmethod
    def pullup_resistor(vcc: float, vol_max: float, ioh_max: float,
                        bus_capacitance_pf: float,
                        rise_time_ns: float) -> Tuple[float, float]:
        """
        Calculate I2C/SPI pullup resistor.

        Returns: (min_resistance, max_resistance)

        Constraints:
        - Must sink enough current: R > (VCC - VOL) / IOH
        - Must charge bus capacitance: R < rise_time / (0.8 * C)
        """
        # Minimum R (current sinking limit)
        r_min = (vcc - vol_max) / ioh_max if ioh_max > 0 else 1000

        # Maximum R (rise time limit)
        # Rise time ≈ 0.8 × R × C (for 10% to 70%)
        c_farads = bus_capacitance_pf * 1e-12
        t_rise_seconds = rise_time_ns * 1e-9
        r_max = t_rise_seconds / (0.8 * c_farads) if c_farads > 0 else 100000

        return (r_min, r_max)


# =============================================================================
# TRACE CALCULATIONS
# =============================================================================

class TraceCalculator:
    """PCB trace calculations."""

    @staticmethod
    def trace_resistance(length_mm: float, width_mm: float,
                         thickness_oz: float = 1.0) -> float:
        """
        Calculate trace DC resistance.

        Args:
            length_mm: Trace length in mm
            width_mm: Trace width in mm
            thickness_oz: Copper weight (1oz = 35μm)

        Returns: Resistance in ohms
        """
        # Convert to meters
        length_m = length_mm / 1000
        width_m = width_mm / 1000
        thickness_m = thickness_oz * 35e-6  # 1oz = 35 microns

        # R = ρ × L / A
        cross_section = width_m * thickness_m
        resistance = Constants.COPPER_RESISTIVITY * length_m / cross_section

        return resistance

    @staticmethod
    def trace_width_for_current(current_a: float, temp_rise_c: float = 10,
                                thickness_oz: float = 1.0,
                                is_external: bool = True) -> float:
        """
        Calculate minimum trace width for given current.

        Based on IPC-2221 formula.

        Args:
            current_a: Current in amps
            temp_rise_c: Allowable temperature rise
            thickness_oz: Copper weight
            is_external: External (True) or internal (False) layer

        Returns: Minimum trace width in mm
        """
        # IPC-2221 constants
        if is_external:
            k = 0.048  # External
        else:
            k = 0.024  # Internal (half the current capacity)

        # Area in mils² = (I / (k × ΔT^0.44))^(1/0.725)
        area_mils_sq = (current_a / (k * (temp_rise_c ** 0.44))) ** (1 / 0.725)

        # Convert to mm
        thickness_mils = thickness_oz * 1.378  # 1oz ≈ 1.378 mils
        width_mils = area_mils_sq / thickness_mils
        width_mm = width_mils * 0.0254

        return width_mm

    @staticmethod
    def trace_inductance(length_mm: float, width_mm: float,
                         height_mm: float = 1.6) -> float:
        """
        Estimate trace inductance (nH).

        Simplified formula for microstrip.
        """
        # L ≈ 5.08 × L × (ln(2L/(W+T)) + 0.25) nH (empirical)
        length_inch = length_mm / 25.4
        width_inch = width_mm / 25.4

        l_nh = 5.08 * length_inch * (math.log(2 * length_mm / (width_mm + 0.035)) + 0.25)

        return max(0, l_nh)

    @staticmethod
    def microstrip_impedance(trace_width_mm: float, substrate_height_mm: float,
                              trace_thickness_mm: float = 0.035,
                              dielectric_constant: float = 4.5) -> float:
        """
        Calculate microstrip characteristic impedance.

        Args:
            trace_width_mm: Trace width
            substrate_height_mm: Dielectric thickness (distance to ground plane)
            trace_thickness_mm: Copper thickness (default 1oz = 0.035mm)
            dielectric_constant: Er (FR4 ≈ 4.5)

        Returns: Impedance in ohms
        """
        w = trace_width_mm
        h = substrate_height_mm
        t = trace_thickness_mm
        er = dielectric_constant

        # Effective width (accounting for thickness)
        w_eff = w + (t / math.pi) * (1 + math.log(2 * h / t))

        # Effective dielectric constant
        if w_eff / h <= 1:
            er_eff = (er + 1) / 2 + ((er - 1) / 2) * (
                (1 + 12 * h / w_eff) ** -0.5 + 0.04 * (1 - w_eff / h) ** 2
            )
        else:
            er_eff = (er + 1) / 2 + ((er - 1) / 2) * (1 + 12 * h / w_eff) ** -0.5

        # Characteristic impedance
        if w_eff / h <= 1:
            z0 = (60 / math.sqrt(er_eff)) * math.log(8 * h / w_eff + 0.25 * w_eff / h)
        else:
            z0 = (120 * math.pi) / (math.sqrt(er_eff) * (w_eff / h + 1.393 + 0.667 * math.log(w_eff / h + 1.444)))

        return z0

    @staticmethod
    def differential_impedance(trace_width_mm: float, trace_spacing_mm: float,
                                substrate_height_mm: float,
                                dielectric_constant: float = 4.5) -> float:
        """
        Calculate edge-coupled microstrip differential impedance.

        Returns: Differential impedance in ohms (target is typically 90Ω for USB)
        """
        z0 = TraceCalculator.microstrip_impedance(
            trace_width_mm, substrate_height_mm,
            dielectric_constant=dielectric_constant
        )

        w = trace_width_mm
        s = trace_spacing_mm
        h = substrate_height_mm

        # Coupling factor
        k = math.exp(-1.58 * s / h)

        # Odd-mode impedance
        z_odd = z0 * math.sqrt((1 - k) / (1 + k))

        # Differential impedance = 2 × Zodd
        z_diff = 2 * z_odd

        return z_diff


# =============================================================================
# THERMAL CALCULATIONS
# =============================================================================

class ThermalCalculator:
    """Thermal calculations."""

    @staticmethod
    def junction_temperature(power_w: float, theta_ja: float,
                              ambient_temp: float = 25) -> float:
        """
        Calculate junction temperature.

        Tj = Ta + (P × θja)
        """
        return ambient_temp + power_w * theta_ja

    @staticmethod
    def max_power_for_temperature(max_tj: float, theta_ja: float,
                                   ambient_temp: float = 25) -> float:
        """
        Calculate maximum allowable power dissipation.

        P = (Tj_max - Ta) / θja
        """
        return (max_tj - ambient_temp) / theta_ja if theta_ja > 0 else 0

    @staticmethod
    def thermal_via_count(power_w: float, theta_via: float = 30,
                          target_theta: float = 20) -> int:
        """
        Estimate number of thermal vias needed.

        Args:
            power_w: Power to dissipate
            theta_via: Thermal resistance per via (°C/W)
            target_theta: Target thermal resistance

        Returns: Number of vias needed
        """
        if theta_via <= 0 or target_theta <= 0:
            return 0

        # Vias in parallel: θ_total = θ_via / N
        # N = θ_via / θ_target
        n_vias = math.ceil(theta_via / target_theta)

        return n_vias

    @staticmethod
    def copper_pour_area(power_w: float, max_temp_rise: float = 40,
                          ambient_temp: float = 25) -> float:
        """
        Estimate copper pour area for heat dissipation.

        Simplified model: ~100 cm²/W for 40°C rise (still air)

        Returns: Area in mm²
        """
        area_cm2_per_watt = 100 / max_temp_rise * 40  # Scale for different temp rise
        area_cm2 = power_w * area_cm2_per_watt
        area_mm2 = area_cm2 * 100

        return area_mm2


# =============================================================================
# SIGNAL INTEGRITY CALCULATIONS
# =============================================================================

class SignalIntegrityCalculator:
    """Signal integrity calculations."""

    @staticmethod
    def propagation_delay(length_mm: float, dielectric_constant: float = 4.5) -> float:
        """
        Calculate signal propagation delay.

        Returns: Delay in nanoseconds
        """
        # v = c / √εr
        velocity = Constants.C / math.sqrt(dielectric_constant)

        # Convert length to meters
        length_m = length_mm / 1000

        # Time = distance / velocity
        delay_s = length_m / velocity
        delay_ns = delay_s * 1e9

        return delay_ns

    @staticmethod
    def max_trace_length_for_frequency(frequency_hz: float,
                                        dielectric_constant: float = 4.5,
                                        fraction: float = 0.1) -> float:
        """
        Calculate maximum trace length before transmission line effects matter.

        Rule of thumb: Treat as transmission line if length > λ/10

        Returns: Maximum length in mm
        """
        # Wavelength in medium
        velocity = Constants.C / math.sqrt(dielectric_constant)
        wavelength_m = velocity / frequency_hz

        # Maximum length
        max_length_m = wavelength_m * fraction
        max_length_mm = max_length_m * 1000

        return max_length_mm

    @staticmethod
    def rise_time_to_bandwidth(rise_time_ns: float) -> float:
        """
        Convert rise time to equivalent bandwidth.

        BW ≈ 0.35 / Tr (for 10%-90% rise time)

        Returns: Bandwidth in Hz
        """
        rise_time_s = rise_time_ns * 1e-9
        bandwidth_hz = 0.35 / rise_time_s

        return bandwidth_hz

    @staticmethod
    def length_matching_requirement(frequency_hz: float,
                                     max_skew_percent: float = 10,
                                     dielectric_constant: float = 4.5) -> float:
        """
        Calculate length matching requirement for parallel signals.

        Returns: Maximum length difference in mm
        """
        # Period
        period_s = 1 / frequency_hz

        # Maximum skew
        max_skew_s = period_s * (max_skew_percent / 100)

        # Convert to length
        velocity = Constants.C / math.sqrt(dielectric_constant)
        max_length_diff_m = velocity * max_skew_s
        max_length_diff_mm = max_length_diff_m * 1000

        return max_length_diff_mm


# =============================================================================
# EMI/EMC CALCULATIONS
# =============================================================================

class EMICalculator:
    """EMI/EMC calculations."""

    @staticmethod
    def loop_antenna_radiation(loop_area_mm2: float, current_a: float,
                                frequency_hz: float) -> float:
        """
        Estimate radiated emission from a current loop.

        E ≈ 1.316 × 10^-14 × A × I × f² / r (V/m at 3m)

        Returns: Electric field strength at 3m in V/m
        """
        area_m2 = loop_area_mm2 * 1e-6
        r = 3.0  # Standard measurement distance

        e_field = 1.316e-14 * area_m2 * current_a * (frequency_hz ** 2) / r

        return e_field

    @staticmethod
    def loop_area_for_emission_limit(current_a: float, frequency_hz: float,
                                      limit_dbuvpm: float = 40) -> float:
        """
        Calculate maximum allowable loop area for emission limit.

        Args:
            current_a: Loop current
            frequency_hz: Frequency
            limit_dbuvpm: Emission limit in dBμV/m (FCC Class B ≈ 40 dBμV/m)

        Returns: Maximum loop area in mm²
        """
        # Convert limit to V/m
        limit_uvpm = 10 ** (limit_dbuvpm / 20)
        limit_vpm = limit_uvpm * 1e-6

        # Calculate area
        # E = 1.316 × 10^-14 × A × I × f² / r
        # A = E × r / (1.316 × 10^-14 × I × f²)
        r = 3.0
        area_m2 = limit_vpm * r / (1.316e-14 * current_a * (frequency_hz ** 2))
        area_mm2 = area_m2 * 1e6

        return area_mm2

    @staticmethod
    def decoupling_cap_resonant_frequency(capacitance_f: float,
                                           esl_nh: float = 1.0) -> float:
        """
        Calculate self-resonant frequency of a decoupling capacitor.

        f = 1 / (2π√(LC))

        Returns: Resonant frequency in Hz
        """
        inductance_h = esl_nh * 1e-9

        if capacitance_f <= 0 or inductance_h <= 0:
            return 0

        f_res = 1 / (2 * math.pi * math.sqrt(inductance_h * capacitance_f))

        return f_res


# =============================================================================
# UNIFIED CALCULATOR INTERFACE
# =============================================================================

class ElectricalCalculator:
    """
    Unified interface to all calculators.

    Usage:
        calc = ElectricalCalculator()
        width = calc.trace.trace_width_for_current(3.0)  # 3A trace
        temp = calc.thermal.junction_temperature(2.0, 45)  # 2W, 45°C/W
    """

    def __init__(self):
        self.power = PowerCalculator()
        self.trace = TraceCalculator()
        self.thermal = ThermalCalculator()
        self.signal = SignalIntegrityCalculator()
        self.emi = EMICalculator()

    def design_buck_converter(self, vin: float, vout: float, iout: float,
                               switching_freq_hz: float = 500000) -> dict:
        """
        Complete buck converter design calculations.

        Returns dict with component values and recommendations.
        """
        # Duty cycle
        duty_cycle = vout / vin

        # Inductor value: L = (Vin - Vout) × D / (f × ΔIL)
        # Target ΔIL = 30% of Iout
        delta_il = 0.3 * iout
        inductance_h = (vin - vout) * duty_cycle / (switching_freq_hz * delta_il)
        inductance_uh = inductance_h * 1e6

        # Output capacitor: C = ΔIL / (8 × f × ΔVout)
        # Target ΔVout = 1% of Vout
        delta_vout = 0.01 * vout
        output_cap_f = delta_il / (8 * switching_freq_hz * delta_vout)
        output_cap_uf = output_cap_f * 1e6

        # Input capacitor: Handle RMS current
        # Iin_rms ≈ Iout × √(D × (1-D))
        iin_rms = iout * math.sqrt(duty_cycle * (1 - duty_cycle))
        # Recommend cap that can handle this RMS current

        # Diode current rating
        diode_current = iout * (1 - duty_cycle) * 1.5  # 50% margin

        # Efficiency estimate
        efficiency, losses = self.power.switching_regulator_efficiency(
            vin, vout, iout, duty_cycle=duty_cycle
        )

        return {
            'duty_cycle': duty_cycle,
            'inductor_uh': round(inductance_uh, 1),
            'output_cap_uf': round(output_cap_uf, 0),
            'input_rms_current': round(iin_rms, 2),
            'diode_current_rating': round(diode_current, 1),
            'estimated_efficiency': round(efficiency * 100, 1),
            'estimated_losses_w': round(losses, 2),
            'recommendations': [
                f'Use {round(inductance_uh * 1.2, 0)}µH inductor (20% margin)',
                f'Input cap: ≥{round(iin_rms * 3, 0)}µF with low ESR',
                f'Output cap: ≥{round(output_cap_uf * 1.5, 0)}µF',
                f'Diode: ≥{round(diode_current, 0)}A Schottky',
            ]
        }
