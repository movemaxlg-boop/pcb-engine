"""
Rules API for Circuit Intelligence Engine
=========================================

High-level API that exposes 631 design rules as callable functions.
Each function produces a RuleReport for AI agent review.

Categories:
- Electrical (spacing, current, clearance)
- Impedance (Z0 calculations)
- Placement (component distances, sequences)
- Routing (trace routing, length matching)
- High-Speed (DDR, PCIe, USB, HDMI, Ethernet)
- Thermal (junction temp, thermal vias)
- EMI (emissions, loop area)
- Fabrication (manufacturing limits)
- Stackup (layer configurations)
- BGA/HDI (escape routing, microvias)
- Assembly (spacing, test points)

Usage:
    from circuit_intelligence.rules_api import RulesAPI

    api = RulesAPI()

    # Simple value lookups
    spacing = api.get_conductor_spacing(voltage=12.0)
    width = api.get_trace_width(current_a=3.0)

    # Validation with RuleReport
    report = api.validate_usb_layout(d_plus_mm=45.0, d_minus_mm=46.5, impedance_ohm=90.0)
    print(report.to_json())  # AI-readable JSON

    # Full design review
    review = api.review_design(design_context)
    print(review.to_summary())  # Human-readable summary
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

from .verified_design_rules import get_verified_rules, VerifiedDesignRulesEngine
from .rule_types import (
    RuleReport, RuleStatus, RuleCategory, RuleSeverity,
    ValidationResult, DesignReviewReport,
    create_pass_report, create_fail_report, create_warning_report
)


class RulesAPI:
    """
    High-level API for accessing 631 design rules as callable functions.

    Every validation function returns a RuleReport that can be:
    - Reviewed by an AI agent
    - Serialized to JSON for external consumption
    - Accepted, rejected, or corrected via feedback commands
    """

    def __init__(self):
        self.rules = get_verified_rules()
        self._report_history: List[RuleReport] = []

    def _record_report(self, report: RuleReport) -> RuleReport:
        """Record a report for history and return it."""
        self._report_history.append(report)
        return report

    def get_report_history(self) -> List[RuleReport]:
        """Get all reports generated in this session."""
        return self._report_history.copy()

    def clear_history(self):
        """Clear report history."""
        self._report_history.clear()

    # =========================================================================
    # ELECTRICAL RULES
    # =========================================================================

    def get_conductor_spacing(
        self,
        voltage: float,
        layer_type: str = "external_coated"
    ) -> float:
        """
        Get minimum conductor spacing for voltage level.

        Args:
            voltage: Operating voltage in volts
            layer_type: "internal", "external_uncoated", or "external_coated"

        Returns:
            Minimum spacing in mm (IPC-2221B Table 6-1)
        """
        return self.rules.conductor_spacing.get_spacing(voltage, layer_type)

    def get_trace_width(
        self,
        current_a: float,
        temp_rise_c: float = 10.0,
        copper_oz: float = 1.0,
        is_external: bool = True
    ) -> float:
        """
        Get minimum trace width for current.

        Args:
            current_a: Required current in Amps
            temp_rise_c: Allowable temperature rise (default 10C)
            copper_oz: Copper weight in oz
            is_external: True for external layer

        Returns:
            Minimum trace width in mm (IPC-2152)
        """
        return max(
            self.rules.fabrication.MIN_TRACE_WIDTH_MM,
            self.rules.current_capacity.trace_width_for_current(
                current_a, temp_rise_c, copper_oz, is_external
            )
        )

    def get_current_capacity(
        self,
        width_mm: float,
        temp_rise_c: float = 10.0,
        copper_oz: float = 1.0,
        is_external: bool = True
    ) -> float:
        """
        Get current capacity for trace width.

        Returns:
            Maximum current in Amps (IPC-2152)
        """
        return self.rules.current_capacity.current_capacity(
            width_mm, temp_rise_c, copper_oz, is_external
        )

    def get_clearance(self, voltage: float) -> float:
        """
        Get minimum clearance for voltage level.

        Returns:
            Minimum clearance in mm
        """
        return self.rules.get_clearance(voltage)

    def validate_trace_current(
        self,
        trace_width_mm: float,
        current_a: float,
        copper_oz: float = 1.0,
        temp_rise_c: float = 10.0
    ) -> RuleReport:
        """
        Validate trace width for current capacity.

        Returns:
            RuleReport with PASS/FAIL status
        """
        capacity = self.get_current_capacity(trace_width_mm, temp_rise_c, copper_oz)
        required_width = self.get_trace_width(current_a, temp_rise_c, copper_oz)

        inputs = {
            "trace_width_mm": trace_width_mm,
            "current_a": current_a,
            "copper_oz": copper_oz,
            "temp_rise_c": temp_rise_c
        }

        metrics = {
            "current_capacity_a": round(capacity, 2),
            "required_width_mm": round(required_width, 2),
            "margin_pct": round((capacity - current_a) / current_a * 100, 1) if current_a > 0 else 100
        }

        if capacity >= current_a:
            report = create_pass_report(
                rule_id="TRACE_CURRENT_CAPACITY",
                category=RuleCategory.ELECTRICAL,
                source="IPC-2152 Section 5.1",
                inputs=inputs,
                rule_applied=f"current_capacity >= {current_a}A",
                threshold=current_a,
                actual_value=round(capacity, 2),
                metrics=metrics
            )
        else:
            report = create_fail_report(
                rule_id="TRACE_CURRENT_CAPACITY",
                category=RuleCategory.ELECTRICAL,
                source="IPC-2152 Section 5.1",
                inputs=inputs,
                rule_applied=f"current_capacity >= {current_a}A",
                threshold=current_a,
                actual_value=round(capacity, 2),
                violation=f"Trace {trace_width_mm}mm can carry {capacity:.2f}A, need {current_a}A. Widen to {required_width:.2f}mm",
                alternatives=[
                    f"Increase trace width to {required_width:.2f}mm",
                    f"Use 2oz copper (doubles capacity)",
                    f"Use parallel traces"
                ],
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # IMPEDANCE RULES
    # =========================================================================

    def calculate_microstrip_impedance(
        self,
        width_mm: float,
        height_mm: float,
        thickness_mm: float = 0.035,
        dk: float = 4.2
    ) -> float:
        """
        Calculate microstrip impedance.

        Args:
            width_mm: Trace width in mm
            height_mm: Dielectric height to reference plane
            thickness_mm: Copper thickness (default 1oz = 0.035mm)
            dk: Dielectric constant (default FR-4 = 4.2)

        Returns:
            Characteristic impedance in ohms (IPC-2141)
        """
        return self.rules.impedance.microstrip_z0(width_mm, height_mm, thickness_mm, dk)

    def calculate_stripline_impedance(
        self,
        width_mm: float,
        height_mm: float,
        thickness_mm: float = 0.035,
        dk: float = 4.2
    ) -> float:
        """Calculate stripline impedance (IPC-2141)."""
        return self.rules.impedance.stripline_z0(width_mm, height_mm, thickness_mm, dk)

    def calculate_differential_impedance(
        self,
        width_mm: float,
        spacing_mm: float,
        height_mm: float,
        thickness_mm: float = 0.035,
        dk: float = 4.2
    ) -> float:
        """Calculate differential microstrip impedance."""
        return self.rules.impedance.differential_microstrip_z0(
            width_mm, spacing_mm, height_mm, thickness_mm, dk
        )

    def get_width_for_impedance(
        self,
        target_z0: float,
        height_mm: float,
        trace_type: str = "microstrip",
        dk: float = 4.2
    ) -> float:
        """Get trace width for target impedance."""
        return self.rules.impedance.get_trace_width_for_impedance(
            target_z0, height_mm, trace_type, dk
        )

    # =========================================================================
    # PLACEMENT RULES
    # =========================================================================

    def get_decoupling_distance(self) -> float:
        """Get max distance from decoupling cap to VCC pin (mm)."""
        return self.rules.decoupling.MAX_DISTANCE_TO_VCC_PIN_MM

    def get_crystal_distance(self) -> float:
        """Get max distance from crystal to MCU oscillator pins (mm)."""
        return self.rules.crystal.MAX_DISTANCE_TO_MCU_MM

    def get_regulator_loop_length(self) -> float:
        """Get max critical loop length for switching regulators (mm)."""
        return self.rules.switching_regulator.MAX_CRITICAL_LOOP_LENGTH_MM

    def get_analog_separation(self) -> float:
        """Get min separation from analog to digital circuits (mm)."""
        return self.rules.analog.MIN_SEPARATION_FROM_DIGITAL_MM

    def get_placement_sequence(self) -> List[Dict]:
        """Get component placement sequence (priority order)."""
        return self.rules.placement_sequence.PLACEMENT_SEQUENCE

    def get_placement_zones(self) -> Dict:
        """Get placement zone definitions."""
        return self.rules.placement_sequence.PLACEMENT_ZONES

    def validate_decoupling_placement(
        self,
        cap_x: float,
        cap_y: float,
        ic_vcc_x: float,
        ic_vcc_y: float
    ) -> RuleReport:
        """
        Validate decoupling capacitor placement distance.

        Returns:
            RuleReport with PASS/FAIL/WARNING status
        """
        distance = math.sqrt((cap_x - ic_vcc_x)**2 + (cap_y - ic_vcc_y)**2)
        max_distance = self.rules.decoupling.MAX_DISTANCE_TO_VCC_PIN_MM

        inputs = {
            "cap_position": {"x": cap_x, "y": cap_y},
            "ic_vcc_position": {"x": ic_vcc_x, "y": ic_vcc_y},
            "distance_mm": round(distance, 2)
        }

        metrics = {
            "distance_mm": round(distance, 2),
            "max_distance_mm": max_distance,
            "margin_pct": round((max_distance - distance) / max_distance * 100, 1)
        }

        if distance <= max_distance:
            # Check if close to limit (warning)
            if distance > max_distance * 0.8:
                report = create_warning_report(
                    rule_id="DECOUPLING_DISTANCE",
                    category=RuleCategory.PLACEMENT,
                    source="Murata Application Notes, ADI AN-1142",
                    inputs=inputs,
                    rule_applied=f"distance <= {max_distance}mm",
                    threshold=max_distance,
                    actual_value=round(distance, 2),
                    warning=f"Cap is {distance:.1f}mm from VCC, limit is {max_distance}mm ({distance/max_distance*100:.0f}% of limit)",
                    metrics=metrics
                )
            else:
                report = create_pass_report(
                    rule_id="DECOUPLING_DISTANCE",
                    category=RuleCategory.PLACEMENT,
                    source="Murata Application Notes, ADI AN-1142",
                    inputs=inputs,
                    rule_applied=f"distance <= {max_distance}mm",
                    threshold=max_distance,
                    actual_value=round(distance, 2),
                    metrics=metrics
                )
        else:
            report = create_fail_report(
                rule_id="DECOUPLING_DISTANCE",
                category=RuleCategory.PLACEMENT,
                source="Murata Application Notes, ADI AN-1142",
                inputs=inputs,
                rule_applied=f"distance <= {max_distance}mm",
                threshold=max_distance,
                actual_value=round(distance, 2),
                violation=f"Decoupling cap is {distance:.1f}mm from VCC pin, max {max_distance}mm",
                severity=RuleSeverity.ERROR,
                alternatives=[
                    f"Move capacitor closer to VCC pin",
                    f"Reposition IC if possible"
                ],
                metrics=metrics
            )

        return self._record_report(report)

    def validate_crystal_placement(
        self,
        crystal_x: float,
        crystal_y: float,
        mcu_osc_x: float,
        mcu_osc_y: float
    ) -> RuleReport:
        """Validate crystal placement distance to MCU."""
        distance = math.sqrt((crystal_x - mcu_osc_x)**2 + (crystal_y - mcu_osc_y)**2)
        max_distance = self.rules.crystal.MAX_DISTANCE_TO_MCU_MM

        inputs = {
            "crystal_position": {"x": crystal_x, "y": crystal_y},
            "mcu_osc_position": {"x": mcu_osc_x, "y": mcu_osc_y},
            "distance_mm": round(distance, 2)
        }

        metrics = {
            "distance_mm": round(distance, 2),
            "max_distance_mm": max_distance,
            "margin_pct": round((max_distance - distance) / max_distance * 100, 1)
        }

        if distance <= max_distance:
            report = create_pass_report(
                rule_id="CRYSTAL_DISTANCE",
                category=RuleCategory.PLACEMENT,
                source="Microchip AN826, ST AN2867",
                inputs=inputs,
                rule_applied=f"distance <= {max_distance}mm",
                threshold=max_distance,
                actual_value=round(distance, 2),
                metrics=metrics
            )
        else:
            report = create_fail_report(
                rule_id="CRYSTAL_DISTANCE",
                category=RuleCategory.PLACEMENT,
                source="Microchip AN826, ST AN2867",
                inputs=inputs,
                rule_applied=f"distance <= {max_distance}mm",
                threshold=max_distance,
                actual_value=round(distance, 2),
                violation=f"Crystal is {distance:.1f}mm from MCU OSC pins, max {max_distance}mm",
                severity=RuleSeverity.ERROR,
                alternatives=["Move crystal closer to MCU oscillator pins"],
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # ROUTING RULES
    # =========================================================================

    def get_routing_sequence(self) -> List[Dict]:
        """Get signal routing sequence (priority order)."""
        return self.rules.routing_sequence.ROUTING_SEQUENCE

    def get_length_matching_rules(self, protocol: str) -> Dict:
        """
        Get length matching rules for a protocol.

        Args:
            protocol: "USB_2.0_HS", "USB_2.0_FS", "DDR3", "SPI_50MHz", "I2C"

        Returns:
            Dict with matching requirements
        """
        return self.rules.routing_sequence.LENGTH_MATCHING.get(protocol, {})

    def get_via_stitching_spacing(self, freq_ghz: float) -> float:
        """Get maximum via stitching spacing for frequency (mm)."""
        return self.rules.emi_design.max_via_stitch_spacing_mm(freq_ghz)

    def validate_usb_layout(
        self,
        d_plus_mm: float,
        d_minus_mm: float,
        impedance_ohm: float
    ) -> RuleReport:
        """
        Validate USB 2.0 High-Speed layout.

        Returns:
            RuleReport with detailed compliance check
        """
        mismatch = abs(d_plus_mm - d_minus_mm)
        max_mismatch = self.rules.usb2.MAX_LENGTH_MISMATCH_MM
        max_length = self.rules.usb2.MAX_TRACE_LENGTH_MM
        target_z = self.rules.usb2.DIFFERENTIAL_IMPEDANCE_OHM
        tolerance = self.rules.usb2.DIFFERENTIAL_TOLERANCE_PERCENT

        violations = []
        warnings = []

        # Check length matching
        if mismatch > max_mismatch:
            violations.append(
                f"D+/D- length mismatch {mismatch:.2f}mm exceeds {max_mismatch}mm limit"
            )
        elif mismatch > max_mismatch * 0.8:
            warnings.append(
                f"D+/D- mismatch {mismatch:.2f}mm is {mismatch/max_mismatch*100:.0f}% of limit"
            )

        # Check max length
        max_len = max(d_plus_mm, d_minus_mm)
        if max_len > max_length:
            violations.append(
                f"USB trace length {max_len:.1f}mm exceeds {max_length}mm limit"
            )

        # Check impedance
        z_error = abs(impedance_ohm - target_z) / target_z * 100
        if z_error > tolerance:
            violations.append(
                f"Impedance {impedance_ohm}ohm outside {target_z}ohm +/-{tolerance}%"
            )
        elif z_error > tolerance * 0.7:
            warnings.append(
                f"Impedance {impedance_ohm}ohm is {z_error:.1f}% from target"
            )

        inputs = {
            "d_plus_length_mm": d_plus_mm,
            "d_minus_length_mm": d_minus_mm,
            "differential_impedance_ohm": impedance_ohm
        }

        metrics = {
            "mismatch_mm": round(mismatch, 2),
            "mismatch_limit_mm": max_mismatch,
            "max_length_mm": round(max_len, 2),
            "impedance_ohm": impedance_ohm,
            "impedance_target_ohm": target_z,
            "impedance_error_pct": round(z_error, 1)
        }

        if violations:
            report = create_fail_report(
                rule_id="USB2_LENGTH_MATCHING",
                category=RuleCategory.HIGH_SPEED,
                source="USB 2.0 Spec Chapter 7, Microchip AN2972",
                inputs=inputs,
                rule_applied=f"mismatch <= {max_mismatch}mm, impedance = {target_z}ohm +/-{tolerance}%",
                threshold=max_mismatch,
                actual_value=round(mismatch, 2),
                violation=violations[0],
                severity=RuleSeverity.ERROR,
                alternatives=[
                    f"Add {mismatch:.1f}mm serpentine to shorter trace",
                    "Re-route to reduce length difference",
                    "Adjust trace width for impedance"
                ],
                metrics=metrics
            )
            report.violations = violations
            report.warnings = warnings
        elif warnings:
            report = create_warning_report(
                rule_id="USB2_LENGTH_MATCHING",
                category=RuleCategory.HIGH_SPEED,
                source="USB 2.0 Spec Chapter 7, Microchip AN2972",
                inputs=inputs,
                rule_applied=f"mismatch <= {max_mismatch}mm",
                threshold=max_mismatch,
                actual_value=round(mismatch, 2),
                warning=warnings[0],
                metrics=metrics
            )
            report.warnings = warnings
        else:
            report = create_pass_report(
                rule_id="USB2_LENGTH_MATCHING",
                category=RuleCategory.HIGH_SPEED,
                source="USB 2.0 Spec Chapter 7, Microchip AN2972",
                inputs=inputs,
                rule_applied=f"mismatch <= {max_mismatch}mm",
                threshold=max_mismatch,
                actual_value=round(mismatch, 2),
                metrics=metrics
            )

        return self._record_report(report)

    def validate_differential_pair(
        self,
        protocol: str,
        length_p_mm: float,
        length_n_mm: float,
        impedance_ohm: float
    ) -> RuleReport:
        """
        Validate differential pair for protocol.

        Args:
            protocol: "USB_2.0_HS", "USB_3.0", "SATA_3", "LVDS", etc.
        """
        proto_rules = self.rules.differential_pairs.PROTOCOLS.get(protocol, {})
        if not proto_rules:
            return self._record_report(RuleReport(
                rule_id=f"DIFF_PAIR_{protocol}",
                rule_category=RuleCategory.HIGH_SPEED,
                rule_source="Unknown",
                status=RuleStatus.SKIPPED,
                passed=True,
                warnings=[f"No rules defined for protocol {protocol}"]
            ))

        mismatch = abs(length_p_mm - length_n_mm)
        max_mismatch = proto_rules.get("max_skew_mm", 999)
        target_z = proto_rules.get("diff_impedance_ohm", 100)
        tolerance = proto_rules.get("tolerance_pct", 10)

        violations = []
        if mismatch > max_mismatch:
            violations.append(f"P/N mismatch {mismatch:.2f}mm > {max_mismatch}mm limit")

        z_error = abs(impedance_ohm - target_z) / target_z * 100
        if z_error > tolerance:
            violations.append(f"Impedance {impedance_ohm}ohm outside {target_z}ohm +/-{tolerance}%")

        inputs = {
            "protocol": protocol,
            "length_p_mm": length_p_mm,
            "length_n_mm": length_n_mm,
            "impedance_ohm": impedance_ohm
        }

        metrics = {
            "mismatch_mm": round(mismatch, 2),
            "max_mismatch_mm": max_mismatch,
            "impedance_error_pct": round(z_error, 1)
        }

        if violations:
            report = create_fail_report(
                rule_id=f"DIFF_PAIR_{protocol}",
                category=RuleCategory.HIGH_SPEED,
                source=f"{protocol} Specification",
                inputs=inputs,
                rule_applied=f"mismatch <= {max_mismatch}mm, Z = {target_z}ohm",
                threshold=max_mismatch,
                actual_value=round(mismatch, 2),
                violation=violations[0],
                metrics=metrics
            )
            report.violations = violations
        else:
            report = create_pass_report(
                rule_id=f"DIFF_PAIR_{protocol}",
                category=RuleCategory.HIGH_SPEED,
                source=f"{protocol} Specification",
                inputs=inputs,
                rule_applied=f"mismatch <= {max_mismatch}mm, Z = {target_z}ohm",
                threshold=max_mismatch,
                actual_value=round(mismatch, 2),
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # HIGH-SPEED INTERFACE RULES
    # =========================================================================

    def get_ddr3_rules(self) -> Dict:
        """Get DDR3 memory layout rules (JEDEC JESD79-3F)."""
        return self.rules.ddr_memory.DDR3

    def get_ddr4_rules(self) -> Dict:
        """Get DDR4 memory layout rules (JEDEC JESD79-4B)."""
        return self.rules.ddr_memory.DDR4

    def get_pcie_rules(self, gen: str = "Gen3") -> Dict:
        """Get PCIe layout rules for generation."""
        return self.rules.pcie.PCIE_SPECS.get(gen, {})

    def get_hdmi_rules(self, version: str = "HDMI_2.0") -> Dict:
        """Get HDMI layout rules for version."""
        return self.rules.hdmi.HDMI_SPECS.get(version, {})

    def get_ethernet_rules(self, speed: str = "1000BASE-T") -> Dict:
        """Get Ethernet layout rules for speed."""
        return self.rules.ethernet.ETHERNET_SPECS.get(speed, {})

    def validate_ddr_layout(
        self,
        version: str,
        dqs_to_dq_mm: float,
        dq_to_dq_mm: float,
        data_impedance_ohm: float
    ) -> RuleReport:
        """Validate DDR memory layout."""
        if version == "DDR3":
            rules = self.rules.ddr_memory.DDR3
        elif version == "DDR4":
            rules = self.rules.ddr_memory.DDR4
        else:
            return self._record_report(RuleReport(
                rule_id=f"DDR_{version}_LAYOUT",
                rule_category=RuleCategory.HIGH_SPEED,
                rule_source="Unknown",
                status=RuleStatus.SKIPPED,
                passed=True,
                warnings=[f"No rules for {version}"]
            ))

        violations = []
        max_dqs_dq = rules["dqs_to_dq_max_mm"]
        max_dq_dq = rules["dq_to_dq_max_mm"]
        target_z = rules["data_impedance_ohm"]
        tolerance = rules["impedance_tolerance_pct"]

        if dqs_to_dq_mm > max_dqs_dq:
            violations.append(f"DQS-DQ mismatch {dqs_to_dq_mm}mm > {max_dqs_dq}mm")
        if dq_to_dq_mm > max_dq_dq:
            violations.append(f"DQ-DQ mismatch {dq_to_dq_mm}mm > {max_dq_dq}mm")

        z_error = abs(data_impedance_ohm - target_z) / target_z * 100
        if z_error > tolerance:
            violations.append(f"Data impedance {data_impedance_ohm}ohm outside {target_z}ohm +/-{tolerance}%")

        inputs = {
            "version": version,
            "dqs_to_dq_mm": dqs_to_dq_mm,
            "dq_to_dq_mm": dq_to_dq_mm,
            "data_impedance_ohm": data_impedance_ohm
        }

        metrics = {
            "dqs_dq_mismatch_mm": dqs_to_dq_mm,
            "dq_dq_mismatch_mm": dq_to_dq_mm,
            "impedance_error_pct": round(z_error, 1)
        }

        if violations:
            report = create_fail_report(
                rule_id=f"DDR_{version}_LAYOUT",
                category=RuleCategory.HIGH_SPEED,
                source=f"JEDEC JESD79-{'3F' if version == 'DDR3' else '4B'}",
                inputs=inputs,
                rule_applied=f"DQS-DQ <= {max_dqs_dq}mm, DQ-DQ <= {max_dq_dq}mm",
                threshold=max_dqs_dq,
                actual_value=dqs_to_dq_mm,
                violation=violations[0],
                severity=RuleSeverity.CRITICAL,
                metrics=metrics
            )
            report.violations = violations
        else:
            report = create_pass_report(
                rule_id=f"DDR_{version}_LAYOUT",
                category=RuleCategory.HIGH_SPEED,
                source=f"JEDEC JESD79-{'3F' if version == 'DDR3' else '4B'}",
                inputs=inputs,
                rule_applied=f"DQS-DQ <= {max_dqs_dq}mm",
                threshold=max_dqs_dq,
                actual_value=dqs_to_dq_mm,
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # THERMAL RULES
    # =========================================================================

    def calculate_junction_temp(
        self,
        power_w: float,
        theta_ja_c_w: float,
        ambient_c: float = 25.0
    ) -> float:
        """Calculate junction temperature (JEDEC JESD51)."""
        return self.rules.junction_temp.junction_temp(power_w, theta_ja_c_w, ambient_c)

    def calculate_max_power(
        self,
        max_tj_c: float,
        theta_ja_c_w: float,
        ambient_c: float = 25.0
    ) -> float:
        """Calculate max power for target junction temp."""
        return self.rules.junction_temp.max_power_for_temp(max_tj_c, theta_ja_c_w, ambient_c)

    def calculate_via_thermal_resistance(
        self,
        num_vias: int,
        drill_mm: float = 0.3,
        board_thickness_mm: float = 1.6,
        copper_oz: float = 1.0
    ) -> float:
        """Calculate thermal resistance of via array (C/W)."""
        return self.rules.thermal_resistance.via_array_thermal_resistance(
            num_vias, drill_mm, board_thickness_mm, copper_oz
        )

    def get_thermal_pad_rules(self) -> Dict:
        """Get thermal pad design rules."""
        return {
            "via_grid_pitch_mm": self.rules.thermal_pad.VIA_GRID_PITCH_MM,
            "via_drill_mm": self.rules.thermal_pad.VIA_DRILL_MM,
            "min_vias": self.rules.thermal_pad.MIN_VIAS_PER_THERMAL_PAD,
            "solder_paste_pct": self.rules.thermal_pad.SOLDER_PASTE_COVERAGE_PERCENT
        }

    def validate_thermal_design(
        self,
        power_w: float,
        theta_ja_c_w: float,
        max_tj_c: float = 125.0,
        ambient_c: float = 25.0,
        num_thermal_vias: int = 0
    ) -> RuleReport:
        """Validate thermal design for a component."""
        tj = self.calculate_junction_temp(power_w, theta_ja_c_w, ambient_c)

        inputs = {
            "power_w": power_w,
            "theta_ja_c_w": theta_ja_c_w,
            "max_tj_c": max_tj_c,
            "ambient_c": ambient_c,
            "num_thermal_vias": num_thermal_vias
        }

        # Calculate with thermal vias if present
        if num_thermal_vias > 0:
            via_theta = self.calculate_via_thermal_resistance(num_thermal_vias)
            # Simplified: vias reduce effective theta_ja
            effective_theta = theta_ja_c_w * via_theta / (theta_ja_c_w + via_theta)
            tj_with_vias = self.calculate_junction_temp(power_w, effective_theta, ambient_c)
            metrics = {
                "tj_calculated_c": round(tj, 1),
                "tj_with_vias_c": round(tj_with_vias, 1),
                "via_thermal_resistance_c_w": round(via_theta, 1),
                "margin_c": round(max_tj_c - tj_with_vias, 1)
            }
            tj = tj_with_vias
        else:
            metrics = {
                "tj_calculated_c": round(tj, 1),
                "margin_c": round(max_tj_c - tj, 1)
            }

        if tj > max_tj_c:
            report = create_fail_report(
                rule_id="THERMAL_JUNCTION_TEMP",
                category=RuleCategory.THERMAL,
                source="JEDEC JESD51-1",
                inputs=inputs,
                rule_applied=f"Tj <= {max_tj_c}C",
                threshold=max_tj_c,
                actual_value=round(tj, 1),
                violation=f"Junction temp {tj:.1f}C exceeds max {max_tj_c}C",
                severity=RuleSeverity.CRITICAL,
                alternatives=[
                    f"Add thermal vias (min {self.rules.thermal_pad.MIN_VIAS_PER_THERMAL_PAD})",
                    "Increase copper pour area",
                    "Add external heatsink",
                    "Reduce power consumption"
                ],
                metrics=metrics
            )
        elif tj > max_tj_c * 0.85:
            report = create_warning_report(
                rule_id="THERMAL_JUNCTION_TEMP",
                category=RuleCategory.THERMAL,
                source="JEDEC JESD51-1",
                inputs=inputs,
                rule_applied=f"Tj <= {max_tj_c}C",
                threshold=max_tj_c,
                actual_value=round(tj, 1),
                warning=f"Junction temp {tj:.1f}C is {tj/max_tj_c*100:.0f}% of max",
                metrics=metrics
            )
        else:
            report = create_pass_report(
                rule_id="THERMAL_JUNCTION_TEMP",
                category=RuleCategory.THERMAL,
                source="JEDEC JESD51-1",
                inputs=inputs,
                rule_applied=f"Tj <= {max_tj_c}C",
                threshold=max_tj_c,
                actual_value=round(tj, 1),
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # EMI/EMC RULES
    # =========================================================================

    def get_emi_limits(self, standard: str = "FCC_CLASS_B") -> Dict:
        """Get EMI emissions limits for standard."""
        if standard == "FCC_CLASS_B":
            return {
                "conducted": dict(self.rules.emi_limits.FCC_CLASS_B_CONDUCTED),
                "radiated_3m": dict(self.rules.emi_limits.FCC_CLASS_B_RADIATED_3M)
            }
        elif standard == "CISPR32_CLASS_B":
            return {
                "radiated_10m": dict(self.rules.emi_limits.CISPR32_CLASS_B_RADIATED_10M)
            }
        return {}

    def calculate_loop_radiation(
        self,
        current_ma: float,
        freq_mhz: float,
        area_mm2: float,
        distance_m: float = 3.0
    ) -> float:
        """Calculate radiated field from current loop (uV/m)."""
        return self.rules.loop_calculator.estimate_radiated_field_uV_m(
            current_ma, freq_mhz, area_mm2, distance_m
        )

    def get_max_loop_area(
        self,
        current_ma: float,
        freq_mhz: float,
        limit_dBuV_m: float = 40.0,
        margin_dB: float = 6.0
    ) -> float:
        """Get max loop area to meet EMI limit (mm2)."""
        return self.rules.loop_calculator.max_loop_area_for_limit_mm2(
            current_ma, freq_mhz, limit_dBuV_m, 3.0, margin_dB
        )

    def validate_emi_loop_area(
        self,
        current_ma: float,
        freq_mhz: float,
        area_mm2: float,
        limit_dBuV_m: float = 40.0
    ) -> RuleReport:
        """Validate EMI loop area against FCC limits."""
        max_area = self.get_max_loop_area(current_ma, freq_mhz, limit_dBuV_m)
        field = self.calculate_loop_radiation(current_ma, freq_mhz, area_mm2)
        field_dBuV = 20 * math.log10(max(field, 0.001))

        inputs = {
            "current_ma": current_ma,
            "freq_mhz": freq_mhz,
            "area_mm2": area_mm2,
            "limit_dBuV_m": limit_dBuV_m
        }

        metrics = {
            "loop_area_mm2": area_mm2,
            "max_area_mm2": round(max_area, 1),
            "radiated_field_uV_m": round(field, 2),
            "radiated_field_dBuV_m": round(field_dBuV, 1),
            "margin_dB": round(limit_dBuV_m - field_dBuV, 1)
        }

        if area_mm2 > max_area:
            report = create_fail_report(
                rule_id="EMI_LOOP_AREA",
                category=RuleCategory.EMI,
                source="FCC Part 15, Henry Ott EMC Engineering",
                inputs=inputs,
                rule_applied=f"loop_area <= {max_area:.0f}mm2",
                threshold=round(max_area, 0),
                actual_value=area_mm2,
                violation=f"Loop area {area_mm2}mm2 exceeds {max_area:.0f}mm2 for {freq_mhz}MHz",
                alternatives=[
                    "Reduce loop area by placing bypass caps closer",
                    "Use ground plane to reduce loop",
                    "Shield the loop with ground pour"
                ],
                metrics=metrics
            )
        else:
            report = create_pass_report(
                rule_id="EMI_LOOP_AREA",
                category=RuleCategory.EMI,
                source="FCC Part 15, Henry Ott EMC Engineering",
                inputs=inputs,
                rule_applied=f"loop_area <= {max_area:.0f}mm2",
                threshold=round(max_area, 0),
                actual_value=area_mm2,
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # FABRICATION RULES
    # =========================================================================

    def get_min_trace_width(self, capability: str = "standard") -> float:
        """Get minimum trace width for fab capability."""
        if capability == "advanced":
            return self.rules.fabrication.MIN_TRACE_WIDTH_ADVANCED_MM
        elif capability == "hdi":
            return self.rules.fabrication.MIN_TRACE_WIDTH_HDI_MM
        return self.rules.fabrication.MIN_TRACE_WIDTH_MM

    def get_min_spacing(self, capability: str = "standard") -> float:
        """Get minimum spacing for fab capability."""
        if capability == "advanced":
            return self.rules.fabrication.MIN_SPACING_ADVANCED_MM
        return self.rules.fabrication.MIN_SPACING_MM

    def get_min_via_drill(self, via_type: str = "standard") -> float:
        """Get minimum via drill for type."""
        if via_type == "advanced":
            return self.rules.fabrication.MIN_VIA_DRILL_ADVANCED_MM
        elif via_type == "laser":
            return self.rules.fabrication.MIN_VIA_DRILL_LASER_MM
        return self.rules.fabrication.MIN_VIA_DRILL_MM

    def get_annular_ring(self, ipc_class: int = 2) -> float:
        """Get minimum annular ring for IPC class."""
        ring_map = {
            1: self.rules.via_design.ANNULAR_RING.get("class_1_mm", 0.05),
            2: self.rules.via_design.ANNULAR_RING.get("class_2_mm", 0.125),
            3: self.rules.via_design.ANNULAR_RING.get("class_3_mm", 0.15)
        }
        return ring_map.get(ipc_class, 0.125)

    def validate_fabrication(
        self,
        trace_width_mm: float,
        spacing_mm: float,
        via_drill_mm: float,
        capability: str = "standard"
    ) -> RuleReport:
        """Validate design against fabrication limits."""
        min_trace = self.get_min_trace_width(capability)
        min_space = self.get_min_spacing(capability)
        min_via = self.get_min_via_drill(capability)

        violations = []
        if trace_width_mm < min_trace:
            violations.append(f"Trace {trace_width_mm}mm < min {min_trace}mm")
        if spacing_mm < min_space:
            violations.append(f"Spacing {spacing_mm}mm < min {min_space}mm")
        if via_drill_mm < min_via:
            violations.append(f"Via drill {via_drill_mm}mm < min {min_via}mm")

        inputs = {
            "trace_width_mm": trace_width_mm,
            "spacing_mm": spacing_mm,
            "via_drill_mm": via_drill_mm,
            "capability": capability
        }

        metrics = {
            "trace_margin_mm": round(trace_width_mm - min_trace, 3),
            "spacing_margin_mm": round(spacing_mm - min_space, 3),
            "via_margin_mm": round(via_drill_mm - min_via, 3)
        }

        if violations:
            report = create_fail_report(
                rule_id="FABRICATION_LIMITS",
                category=RuleCategory.FABRICATION,
                source="IPC-2221B, Standard PCB Fab Capabilities",
                inputs=inputs,
                rule_applied=f"trace >= {min_trace}mm, space >= {min_space}mm, via >= {min_via}mm",
                threshold=min_trace,
                actual_value=trace_width_mm,
                violation=violations[0],
                severity=RuleSeverity.CRITICAL,
                alternatives=[
                    f"Increase to standard minimums",
                    f"Use advanced fab capability (tighter limits)"
                ],
                metrics=metrics
            )
            report.violations = violations
        else:
            report = create_pass_report(
                rule_id="FABRICATION_LIMITS",
                category=RuleCategory.FABRICATION,
                source="IPC-2221B, Standard PCB Fab Capabilities",
                inputs=inputs,
                rule_applied=f"trace >= {min_trace}mm, space >= {min_space}mm",
                threshold=min_trace,
                actual_value=trace_width_mm,
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # STACKUP RULES
    # =========================================================================

    def recommend_layer_count(
        self,
        max_freq_mhz: float,
        signal_count: int,
        has_usb: bool = False,
        has_ddr: bool = False,
        has_rf: bool = False,
        power_domains: int = 1
    ) -> Dict:
        """Recommend layer count for design requirements."""
        return self.rules.stackup_selector.recommend_layer_count(
            max_freq_mhz, signal_count, has_usb, has_ddr, has_rf, power_domains
        )

    def get_stackup(self, layers: int) -> Dict:
        """Get stackup configuration for layer count."""
        if layers == 2:
            return {
                "structure": self.rules.stackup_2layer.STRUCTURE,
                "thickness_mm": self.rules.stackup_2layer.TOTAL_THICKNESS_MM,
                "best_for": self.rules.stackup_2layer.BEST_FOR
            }
        elif layers == 4:
            return {
                "recommended": self.rules.stackup_4layer.RECOMMENDED_STRUCTURE,
                "high_speed": self.rules.stackup_4layer.HIGH_SPEED_STRUCTURE,
                "thickness_mm": self.rules.stackup_4layer.TOTAL_THICKNESS_MM,
                "best_for": self.rules.stackup_4layer.BEST_FOR
            }
        elif layers == 6:
            return {
                "structure": self.rules.stackup_6layer.RECOMMENDED_STRUCTURE,
                "thickness_mm": self.rules.stackup_6layer.TOTAL_THICKNESS_MM,
                "best_for": self.rules.stackup_6layer.BEST_FOR
            }
        return {}

    def get_prepreg_specs(self, prepreg_type: str = "2116") -> Dict:
        """Get prepreg specifications."""
        prepreg_map = {
            "1080": self.rules.prepregs.PREPREG_1080,
            "2116": self.rules.prepregs.PREPREG_2116,
            "7628": self.rules.prepregs.PREPREG_7628
        }
        return {
            "thickness_mm": prepreg_map.get(prepreg_type, 0.115),
            "dk": self.rules.prepregs.FR4_DK,
            "df": self.rules.prepregs.FR4_DF
        }

    # =========================================================================
    # BGA/HDI RULES
    # =========================================================================

    def get_bga_escape_rules(self, pitch_mm: float) -> Dict:
        """Get BGA escape routing rules for pitch."""
        pitch_key = f"{pitch_mm}mm"
        return self.rules.bga_escape.PITCH_CAPABILITIES.get(pitch_key, {})

    def get_microvia_rules(self) -> Dict:
        """Get microvia design rules (IPC-2226)."""
        return {
            "max_drill_mm": self.rules.hdi_design.MICROVIA_SPECS["max_drill_mm"],
            "typical_drill_mm": self.rules.hdi_design.MICROVIA_SPECS["typical_drill_mm"],
            "aspect_ratio_max": self.rules.hdi_design.MICROVIA_SPECS["aspect_ratio_max"],
            "capture_pad_mm": self.rules.hdi_design.MICROVIA_SPECS["capture_pad_mm"]
        }

    def validate_bga_escape(
        self,
        pitch_mm: float,
        trace_width_mm: float,
        via_drill_mm: float
    ) -> RuleReport:
        """Validate BGA escape routing capability."""
        rules = self.get_bga_escape_rules(pitch_mm)

        if not rules:
            return self._record_report(RuleReport(
                rule_id=f"BGA_ESCAPE_{pitch_mm}mm",
                rule_category=RuleCategory.BGA_HDI,
                rule_source="IPC-7093",
                status=RuleStatus.SKIPPED,
                passed=True,
                warnings=[f"No rules for {pitch_mm}mm pitch"]
            ))

        required_trace = rules.get("trace_width_mm", 0.1)
        required_via = rules.get("via_drill_mm", 0.2)
        via_type = rules.get("via_type", "Through-hole")

        violations = []
        if trace_width_mm > required_trace * 1.5:
            violations.append(f"Trace {trace_width_mm}mm too wide for {pitch_mm}mm pitch")
        if via_drill_mm > required_via * 1.5:
            violations.append(f"Via drill {via_drill_mm}mm too large for {pitch_mm}mm pitch")

        inputs = {
            "pitch_mm": pitch_mm,
            "trace_width_mm": trace_width_mm,
            "via_drill_mm": via_drill_mm
        }

        metrics = {
            "required_trace_mm": required_trace,
            "required_via_mm": required_via,
            "via_type": via_type,
            "layers_needed": rules.get("layers_needed", 2)
        }

        if violations:
            report = create_fail_report(
                rule_id=f"BGA_ESCAPE_{pitch_mm}mm",
                category=RuleCategory.BGA_HDI,
                source="IPC-7093",
                inputs=inputs,
                rule_applied=f"trace <= {required_trace*1.5}mm, via <= {required_via*1.5}mm",
                threshold=required_trace,
                actual_value=trace_width_mm,
                violation=violations[0],
                alternatives=[
                    f"Use {via_type} vias",
                    f"Reduce trace width to {required_trace}mm",
                    f"Use {rules.get('layers_needed', 4)} layer board"
                ],
                metrics=metrics
            )
            report.violations = violations
        else:
            report = create_pass_report(
                rule_id=f"BGA_ESCAPE_{pitch_mm}mm",
                category=RuleCategory.BGA_HDI,
                source="IPC-7093",
                inputs=inputs,
                rule_applied=f"trace and via compatible with {pitch_mm}mm pitch",
                threshold=required_trace,
                actual_value=trace_width_mm,
                metrics=metrics
            )

        return self._record_report(report)

    # =========================================================================
    # ASSEMBLY RULES
    # =========================================================================

    def get_component_spacing(self, comp_a: str, comp_b: str) -> float:
        """Get minimum spacing between component types (mm)."""
        spacing_map = self.rules.component_spacing.COMPONENT_SPACING
        key = f"{comp_a}_to_{comp_b}_min_mm"
        alt_key = f"{comp_b}_to_{comp_a}_min_mm"
        return spacing_map.get(key, spacing_map.get(alt_key, 0.5))

    def get_edge_spacing(self, comp_type: str) -> float:
        """Get minimum spacing from board edge (mm)."""
        edge_map = self.rules.component_spacing.EDGE_SPACING
        key = f"{comp_type}_to_edge_mm"
        return edge_map.get(key, 2.0)

    def get_test_point_rules(self) -> Dict:
        """Get test point design rules (IPC-9252)."""
        return {
            "min_diameter_mm": self.rules.test_points.TEST_POINT_SIZE["min_diameter_mm"],
            "preferred_diameter_mm": self.rules.test_points.TEST_POINT_SIZE["preferred_diameter_mm"],
            "min_spacing_mm": self.rules.test_points.SPACING["min_center_to_center_mm"],
            "to_edge_mm": self.rules.test_points.SPACING["to_board_edge_mm"]
        }

    # =========================================================================
    # DESIGN REVIEW (BATCH)
    # =========================================================================

    def create_design_review(self, design_name: str) -> DesignReviewReport:
        """Create a new design review report."""
        return DesignReviewReport(design_name=design_name)

    def finalize_design_review(self, review: DesignReviewReport) -> DesignReviewReport:
        """Finalize and calculate statistics for design review."""
        review.finalize()
        return review


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_rules_api() -> RulesAPI:
    """Get the Rules API instance."""
    return RulesAPI()
