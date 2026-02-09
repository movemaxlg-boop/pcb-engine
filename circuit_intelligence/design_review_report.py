"""
Design Review Report Generator for Circuit Intelligence Engine.

This module provides utilities for generating comprehensive design review reports
that can be consumed by external AI agents for validation and feedback.

Author: Circuit Intelligence Engine
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .rule_types import (
    RuleStatus,
    RuleCategory,
    RuleSeverity,
    RuleReport,
    DesignReviewReport,
    create_pass_report,
    create_fail_report,
    create_warning_report,
)
from .rules_api import RulesAPI
from .feedback import AIFeedbackProcessor


class DesignReviewGenerator:
    """
    Generates comprehensive design review reports.

    This class orchestrates the RulesAPI to validate a complete PCB design
    and produces a DesignReviewReport suitable for AI agent review.

    Usage:
        generator = DesignReviewGenerator()
        report = generator.review_design(design_data)
        print(report.to_summary())

        # Export for AI consumption
        json_report = report.to_json()
    """

    def __init__(self, rules_api: Optional[RulesAPI] = None):
        """
        Initialize the design review generator.

        Args:
            rules_api: Optional RulesAPI instance. Created if not provided.
        """
        self.api = rules_api or RulesAPI()
        self.current_report: Optional[DesignReviewReport] = None

    def review_design(self, design: Dict[str, Any]) -> DesignReviewReport:
        """
        Perform a comprehensive design review.

        Args:
            design: Dictionary containing design data with keys:
                - name: Design name
                - usb_layout: USB layout data (optional)
                - differential_pairs: Differential pair data (optional)
                - thermal_components: Thermal component data (optional)
                - fabrication: Fabrication specs (optional)
                - placements: Component placements (optional)
                - stackup: Layer stackup info (optional)
                - ddr_layout: DDR memory layout (optional)
                - pcie_layout: PCIe layout (optional)
                - emi_loops: EMI loop data (optional)

        Returns:
            DesignReviewReport with all validation results
        """
        design_name = design.get('name', 'Unnamed Design')
        report = DesignReviewReport(design_name=design_name)

        # Electrical validation
        if 'traces' in design:
            self._validate_electrical(design['traces'], report)

        # USB layout validation
        if 'usb_layout' in design:
            self._validate_usb(design['usb_layout'], report)

        # Differential pair validation
        if 'differential_pairs' in design:
            self._validate_differential_pairs(design['differential_pairs'], report)

        # Thermal validation
        if 'thermal_components' in design:
            self._validate_thermal(design['thermal_components'], report)

        # EMI validation
        if 'emi_loops' in design:
            self._validate_emi(design['emi_loops'], report)

        # Fabrication validation
        if 'fabrication' in design:
            self._validate_fabrication(design['fabrication'], report)

        # Placement validation
        if 'placements' in design:
            self._validate_placements(design['placements'], report)

        # DDR validation
        if 'ddr_layout' in design:
            self._validate_ddr(design['ddr_layout'], report)

        # PCIe validation
        if 'pcie_layout' in design:
            self._validate_pcie(design['pcie_layout'], report)

        # HDMI validation
        if 'hdmi_layout' in design:
            self._validate_hdmi(design['hdmi_layout'], report)

        # Ethernet validation
        if 'ethernet_layout' in design:
            self._validate_ethernet(design['ethernet_layout'], report)

        # Stackup validation
        if 'stackup' in design:
            self._validate_stackup(design['stackup'], report)

        # BGA validation
        if 'bga_components' in design:
            self._validate_bga(design['bga_components'], report)

        # Finalize the report
        report.finalize()
        self.current_report = report

        return report

    def _validate_electrical(self, traces: List[Dict], report: DesignReviewReport) -> None:
        """Validate electrical rules for traces."""
        for trace in traces:
            width = trace.get('width_mm', 0)
            current = trace.get('current_a', 0)
            voltage = trace.get('voltage', 0)

            if current > 0:
                rule_report = self.api.validate_trace_current(width, current)
                report.add_report(rule_report)

            if voltage > 0:
                spacing = trace.get('spacing_mm', 0)
                min_spacing = self.api.get_conductor_spacing(voltage)
                if spacing > 0:
                    if spacing >= min_spacing:
                        rule_report = create_pass_report(
                            rule_id=f"CONDUCTOR_SPACING_{int(voltage)}V",
                            rule_category=RuleCategory.ELECTRICAL,
                            inputs={'voltage': voltage, 'spacing_mm': spacing},
                            rule_applied=f"spacing >= {min_spacing}mm for {voltage}V",
                            threshold=min_spacing,
                            actual_value=spacing,
                        )
                    else:
                        rule_report = create_fail_report(
                            rule_id=f"CONDUCTOR_SPACING_{int(voltage)}V",
                            rule_category=RuleCategory.ELECTRICAL,
                            inputs={'voltage': voltage, 'spacing_mm': spacing},
                            rule_applied=f"spacing >= {min_spacing}mm for {voltage}V",
                            threshold=min_spacing,
                            actual_value=spacing,
                            violations=[f"Spacing {spacing}mm < required {min_spacing}mm for {voltage}V"],
                        )
                    report.add_report(rule_report)

    def _validate_usb(self, usb_layout: Dict, report: DesignReviewReport) -> None:
        """Validate USB layout rules."""
        d_plus = usb_layout.get('d_plus_length_mm', 0)
        d_minus = usb_layout.get('d_minus_length_mm', 0)
        impedance = usb_layout.get('impedance_ohm', 90)

        if d_plus > 0 and d_minus > 0:
            rule_report = self.api.validate_usb_layout(d_plus, d_minus, impedance)
            report.add_report(rule_report)

    def _validate_differential_pairs(self, pairs: List[Dict], report: DesignReviewReport) -> None:
        """Validate differential pair rules."""
        for pair in pairs:
            protocol = pair.get('protocol', 'generic')
            length_p = pair.get('positive_length_mm', 0)
            length_n = pair.get('negative_length_mm', 0)
            impedance = pair.get('impedance_ohm', 100)

            if length_p > 0 and length_n > 0:
                rule_report = self.api.validate_differential_pair(
                    protocol, length_p, length_n, impedance
                )
                report.add_report(rule_report)

    def _validate_thermal(self, components: List[Dict], report: DesignReviewReport) -> None:
        """Validate thermal design rules."""
        for comp in components:
            component_id = comp.get('id', 'unknown')
            power_w = comp.get('power_w', 0)
            theta_ja = comp.get('theta_ja', 50)
            via_count = comp.get('thermal_via_count', 0)
            via_diameter = comp.get('via_diameter_mm', 0.3)

            if power_w > 0:
                rule_report = self.api.validate_thermal_design(
                    component_id, power_w, theta_ja, via_count, via_diameter
                )
                report.add_report(rule_report)

    def _validate_emi(self, loops: List[Dict], report: DesignReviewReport) -> None:
        """Validate EMI loop area rules."""
        for loop in loops:
            area_mm2 = loop.get('area_mm2', 0)
            freq_mhz = loop.get('frequency_mhz', 100)
            current_ma = loop.get('current_ma', 10)

            if area_mm2 > 0:
                rule_report = self.api.validate_emi_loop_area(area_mm2, freq_mhz, current_ma)
                report.add_report(rule_report)

    def _validate_fabrication(self, fab_specs: Dict, report: DesignReviewReport) -> None:
        """Validate fabrication rules."""
        rule_report = self.api.validate_fabrication(fab_specs)
        report.add_report(rule_report)

    def _validate_placements(self, placements: Dict, report: DesignReviewReport) -> None:
        """Validate component placement rules."""
        # Decoupling capacitors
        if 'decoupling_caps' in placements:
            for cap in placements['decoupling_caps']:
                cap_pos = cap.get('position', (0, 0))
                ic_pos = cap.get('ic_position', (0, 0))
                ic_ref = cap.get('ic_ref', 'U1')

                rule_report = self.api.validate_decoupling_placement(cap_pos, ic_pos, ic_ref)
                report.add_report(rule_report)

        # Crystals
        if 'crystals' in placements:
            for crystal in placements['crystals']:
                crystal_pos = crystal.get('position', (0, 0))
                mcu_pos = crystal.get('mcu_position', (0, 0))
                mcu_ref = crystal.get('mcu_ref', 'U1')

                rule_report = self.api.validate_crystal_placement(crystal_pos, mcu_pos, mcu_ref)
                report.add_report(rule_report)

    def _validate_ddr(self, ddr_layout: Dict, report: DesignReviewReport) -> None:
        """Validate DDR memory layout rules."""
        version = ddr_layout.get('version', 'DDR3')
        rule_report = self.api.validate_ddr_layout(version, ddr_layout)
        report.add_report(rule_report)

    def _validate_pcie(self, pcie_layout: Dict, report: DesignReviewReport) -> None:
        """Validate PCIe layout rules."""
        gen = pcie_layout.get('generation', 'gen3')
        lanes = pcie_layout.get('lanes', [])

        for lane in lanes:
            lane_id = lane.get('id', 0)
            tx_p = lane.get('tx_p_length_mm', 0)
            tx_n = lane.get('tx_n_length_mm', 0)
            rx_p = lane.get('rx_p_length_mm', 0)
            rx_n = lane.get('rx_n_length_mm', 0)
            impedance = lane.get('impedance_ohm', 85)

            # Get PCIe rules
            pcie_rules = self.api.get_pcie_rules(gen)
            max_skew = pcie_rules.get('max_intra_pair_skew_mm', 0.127)

            # Check TX pair
            tx_skew = abs(tx_p - tx_n)
            if tx_skew <= max_skew:
                tx_report = create_pass_report(
                    rule_id=f"PCIE_{gen.upper()}_TX_LANE{lane_id}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'tx_p': tx_p, 'tx_n': tx_n},
                    rule_applied=f"TX pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=tx_skew,
                )
            else:
                tx_report = create_fail_report(
                    rule_id=f"PCIE_{gen.upper()}_TX_LANE{lane_id}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'tx_p': tx_p, 'tx_n': tx_n},
                    rule_applied=f"TX pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=tx_skew,
                    violations=[f"TX skew {tx_skew:.3f}mm exceeds {max_skew}mm limit"],
                )
            report.add_report(tx_report)

            # Check RX pair
            rx_skew = abs(rx_p - rx_n)
            if rx_skew <= max_skew:
                rx_report = create_pass_report(
                    rule_id=f"PCIE_{gen.upper()}_RX_LANE{lane_id}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'rx_p': rx_p, 'rx_n': rx_n},
                    rule_applied=f"RX pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=rx_skew,
                )
            else:
                rx_report = create_fail_report(
                    rule_id=f"PCIE_{gen.upper()}_RX_LANE{lane_id}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'rx_p': rx_p, 'rx_n': rx_n},
                    rule_applied=f"RX pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=rx_skew,
                    violations=[f"RX skew {rx_skew:.3f}mm exceeds {max_skew}mm limit"],
                )
            report.add_report(rx_report)

    def _validate_hdmi(self, hdmi_layout: Dict, report: DesignReviewReport) -> None:
        """Validate HDMI layout rules."""
        version = hdmi_layout.get('version', '2.0')
        hdmi_rules = self.api.get_hdmi_rules(version)

        max_skew = hdmi_rules.get('max_intra_pair_skew_mm', 0.15)
        target_impedance = hdmi_rules.get('differential_impedance_ohm', 100)

        pairs = hdmi_layout.get('pairs', [])
        for pair in pairs:
            pair_name = pair.get('name', 'TMDS')
            length_p = pair.get('positive_length_mm', 0)
            length_n = pair.get('negative_length_mm', 0)
            impedance = pair.get('impedance_ohm', 100)

            skew = abs(length_p - length_n)
            if skew <= max_skew:
                rule_report = create_pass_report(
                    rule_id=f"HDMI_{version.replace('.', '_')}_{pair_name}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'positive': length_p, 'negative': length_n},
                    rule_applied=f"Pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=skew,
                )
            else:
                rule_report = create_fail_report(
                    rule_id=f"HDMI_{version.replace('.', '_')}_{pair_name}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'positive': length_p, 'negative': length_n},
                    rule_applied=f"Pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=skew,
                    violations=[f"{pair_name} skew {skew:.3f}mm exceeds {max_skew}mm limit"],
                )
            report.add_report(rule_report)

    def _validate_ethernet(self, eth_layout: Dict, report: DesignReviewReport) -> None:
        """Validate Ethernet layout rules."""
        speed = eth_layout.get('speed', '1000BASE-T')
        eth_rules = self.api.get_ethernet_rules(speed)

        target_impedance = eth_rules.get('differential_impedance_ohm', 100)
        max_skew = eth_rules.get('max_intra_pair_skew_mm', 1.27)

        pairs = eth_layout.get('pairs', [])
        for pair in pairs:
            pair_name = pair.get('name', 'MDI')
            length_p = pair.get('positive_length_mm', 0)
            length_n = pair.get('negative_length_mm', 0)
            impedance = pair.get('impedance_ohm', 100)

            skew = abs(length_p - length_n)
            if skew <= max_skew:
                rule_report = create_pass_report(
                    rule_id=f"ETH_{speed.replace('-', '_')}_{pair_name}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'positive': length_p, 'negative': length_n},
                    rule_applied=f"Pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=skew,
                )
            else:
                rule_report = create_fail_report(
                    rule_id=f"ETH_{speed.replace('-', '_')}_{pair_name}_SKEW",
                    rule_category=RuleCategory.HIGH_SPEED,
                    inputs={'positive': length_p, 'negative': length_n},
                    rule_applied=f"Pair skew <= {max_skew}mm",
                    threshold=max_skew,
                    actual_value=skew,
                    violations=[f"{pair_name} skew {skew:.3f}mm exceeds {max_skew}mm limit"],
                )
            report.add_report(rule_report)

    def _validate_stackup(self, stackup: Dict, report: DesignReviewReport) -> None:
        """Validate layer stackup rules."""
        layers = stackup.get('layer_count', 2)
        freq_mhz = stackup.get('max_frequency_mhz', 100)
        signals = stackup.get('signal_count', 50)
        has_usb = stackup.get('has_usb', False)
        has_ddr = stackup.get('has_ddr', False)

        recommendation = self.api.recommend_layer_count(freq_mhz, signals, has_usb, has_ddr)
        min_layers = recommendation.get('min_layers', 2)

        if layers >= min_layers:
            rule_report = create_pass_report(
                rule_id="STACKUP_LAYER_COUNT",
                rule_category=RuleCategory.STACKUP,
                inputs={'layers': layers, 'freq_mhz': freq_mhz, 'has_usb': has_usb, 'has_ddr': has_ddr},
                rule_applied=f"Layer count >= {min_layers} for design requirements",
                threshold=min_layers,
                actual_value=layers,
            )
        else:
            rule_report = create_fail_report(
                rule_id="STACKUP_LAYER_COUNT",
                rule_category=RuleCategory.STACKUP,
                inputs={'layers': layers, 'freq_mhz': freq_mhz, 'has_usb': has_usb, 'has_ddr': has_ddr},
                rule_applied=f"Layer count >= {min_layers} for design requirements",
                threshold=min_layers,
                actual_value=layers,
                violations=[f"Design has {layers} layers but requires minimum {min_layers}"],
            )
            rule_report.recommendations = recommendation.get('recommendations', [])
        report.add_report(rule_report)

    def _validate_bga(self, bga_components: List[Dict], report: DesignReviewReport) -> None:
        """Validate BGA component rules."""
        for bga in bga_components:
            component_id = bga.get('id', 'U1')
            pitch_mm = bga.get('pitch_mm', 1.0)
            escape_via_size = bga.get('escape_via_size_mm', 0.3)
            trace_width = bga.get('escape_trace_width_mm', 0.15)

            rule_report = self.api.validate_bga_escape(
                pitch_mm, escape_via_size, trace_width
            )
            # Update rule_id to include component reference
            rule_report.rule_id = f"BGA_{component_id}_ESCAPE"
            report.add_report(rule_report)

    def get_ai_review_package(self) -> Dict[str, Any]:
        """
        Generate a complete AI review package.

        Returns:
            Dictionary with report and feedback processor ready for AI review
        """
        if not self.current_report:
            return {'error': 'No design review has been performed. Call review_design() first.'}

        # Create feedback processor and add all reports
        processor = AIFeedbackProcessor()
        for report in self.current_report.all_reports:
            processor.add_report(report)

        return {
            'design_name': self.current_report.design_name,
            'summary': {
                'total_rules': self.current_report.total_rules_checked,
                'passed': self.current_report.passed,
                'failed': self.current_report.failed,
                'warnings': self.current_report.warnings,
                'compliance_score': self.current_report.compliance_score,
                'design_status': self.current_report.design_status,
            },
            'blocking_violations': self.current_report.blocking_violations,
            'report_json': self.current_report.to_json(),
            'report_summary': self.current_report.to_summary(),
            'ai_review_prompt': processor.generate_ai_review_prompt(),
            'available_commands': [
                'ACCEPT <rule_id>',
                'REJECT <rule_id> reason="..."',
                'CORRECT <rule_id> action="..." value=...',
                'OVERRIDE <rule_id> new_threshold=... reason="..."',
                'QUERY <rule_id> question="..."',
                'EXPLAIN <rule_id>',
                'BATCH_ACCEPT status=PASS',
                'BATCH_REVIEW status=FAIL',
            ],
        }


def generate_quick_report(design: Dict[str, Any]) -> str:
    """
    Generate a quick design review report.

    Args:
        design: Design data dictionary

    Returns:
        String summary of the design review
    """
    generator = DesignReviewGenerator()
    report = generator.review_design(design)
    return report.to_summary()


def validate_design_for_ai(design: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a design and prepare it for AI review.

    Args:
        design: Design data dictionary

    Returns:
        Complete AI review package
    """
    generator = DesignReviewGenerator()
    generator.review_design(design)
    return generator.get_ai_review_package()
