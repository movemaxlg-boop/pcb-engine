"""
CIRCUIT INTELLIGENCE ENGINE - RULES API DEMONSTRATION
=====================================================

This script demonstrates the complete Rules API system:
- 631 verified design rules
- 100 callable functions
- AI-readable RuleReports
- Feedback commands for AI agents
"""

from circuit_intelligence.rules_api import RulesAPI
from circuit_intelligence.rule_types import RuleStatus, RuleCategory, DesignReviewReport
from circuit_intelligence.feedback import AIFeedbackProcessor
from circuit_intelligence.design_pipeline import DesignPipeline, DesignContext, BoardSpec, ComponentSpec, NetSpec
import json

def main():
    print("=" * 75)
    print("CIRCUIT INTELLIGENCE ENGINE - RULES API LIVE DEMO")
    print("631 Verified Design Rules | 100 Callable Functions | AI-Reviewable")
    print("=" * 75)

    # ========================================================================
    # PART 1: Direct RulesAPI Usage
    # ========================================================================
    print("\n" + "=" * 75)
    print("PART 1: DIRECT RULES API USAGE")
    print("=" * 75)

    api = RulesAPI()

    # 1. Electrical Rules
    print("\n--- ELECTRICAL RULES ---")
    print(f"Conductor spacing for 12V:  {api.get_conductor_spacing(12.0):.3f} mm")
    print(f"Conductor spacing for 50V:  {api.get_conductor_spacing(50.0):.3f} mm")
    print(f"Conductor spacing for 100V: {api.get_conductor_spacing(100.0):.3f} mm")
    print(f"Trace width for 1A current: {api.get_trace_width(1.0):.3f} mm")
    print(f"Trace width for 3A current: {api.get_trace_width(3.0):.3f} mm")
    print(f"Current capacity of 0.5mm trace: {api.get_current_capacity(0.5):.2f} A")

    # 2. Placement Rules
    print("\n--- PLACEMENT RULES ---")
    print(f"Decoupling cap max distance: {api.get_decoupling_distance()} mm")
    print(f"Crystal max distance to MCU: {api.get_crystal_distance()} mm")
    print(f"Regulator loop max length:   {api.get_regulator_loop_length()} mm")
    print(f"Analog-digital separation:   {api.get_analog_separation()} mm")

    # 3. High-Speed Interface Rules
    print("\n--- HIGH-SPEED INTERFACE RULES ---")
    ddr3 = api.get_ddr3_rules()
    print(f"DDR3 Data Impedance: {ddr3['data_impedance_ohm']} ohm")
    print(f"DDR3 Max DQS-DQ Mismatch: {ddr3['dqs_to_dq_max_mm']} mm")

    pcie = api.get_pcie_rules("Gen3")
    print(f"PCIe Gen3 Data Rate: {pcie['data_rate_GT_s']} GT/s")
    print(f"PCIe Gen3 Differential Z: {pcie['diff_impedance_ohm']} ohm")
    print(f"PCIe Gen3 Max Loss: {pcie['max_loss_dB']} dB")

    # 4. Impedance Calculations
    print("\n--- IMPEDANCE CALCULATIONS ---")
    z0_microstrip = api.calculate_microstrip_impedance(width_mm=0.2, height_mm=0.2, dk=4.2)
    print(f"Microstrip Z0 (0.2mm wide, 0.2mm height): {z0_microstrip:.1f} ohm")

    z0_diff = api.calculate_differential_impedance(width_mm=0.15, spacing_mm=0.15, height_mm=0.2)
    print(f"Differential Z0 (0.15mm wide, 0.15mm gap): {z0_diff:.1f} ohm")

    width_for_50 = api.get_width_for_impedance(target_z0=50.0, height_mm=0.2)
    print(f"Width for 50 ohm (0.2mm height): {width_for_50:.3f} mm")

    # 5. Thermal Calculations
    print("\n--- THERMAL CALCULATIONS ---")
    tj = api.calculate_junction_temp(power_w=1.5, theta_ja_c_w=50.0, ambient_c=25.0)
    print(f"Junction temp (1.5W, theta_ja=50): {tj:.1f} C")
    max_power = api.calculate_max_power(max_tj_c=125.0, theta_ja_c_w=50.0, ambient_c=40.0)
    print(f"Max power for Tj=125C, Ta=40C: {max_power:.2f} W")

    # ========================================================================
    # PART 2: Validation with RuleReports
    # ========================================================================
    print("\n" + "=" * 75)
    print("PART 2: VALIDATION WITH AI-READABLE REPORTS")
    print("=" * 75)

    # USB Layout Validation (FAILING CASE)
    print("\n--- USB 2.0 LAYOUT VALIDATION (FAILING) ---")
    usb_report = api.validate_usb_layout(
        d_plus_mm=45.0,
        d_minus_mm=46.8,   # 1.8mm mismatch > 1.25mm limit
        impedance_ohm=88.0  # Should be 90 ohm
    )
    print(f"Rule ID:     {usb_report.rule_id}")
    print(f"Status:      {usb_report.status.value}")
    print(f"Threshold:   {usb_report.threshold} mm")
    print(f"Actual:      {usb_report.actual_value} mm")
    print(f"Violations:  {usb_report.violations}")

    # Decoupling Placement (PASSING CASE)
    print("\n--- DECOUPLING PLACEMENT VALIDATION (PASSING) ---")
    cap_report = api.validate_decoupling_placement(
        cap_x=10.0, cap_y=10.0,
        ic_vcc_x=11.5, ic_vcc_y=10.5
    )
    print(f"Rule ID:    {cap_report.rule_id}")
    print(f"Status:     {cap_report.status.value}")
    print(f"Distance:   {cap_report.metrics['distance_mm']:.2f} mm")
    print(f"Max Allowed:{cap_report.threshold} mm")
    print(f"Margin:     {cap_report.metrics['margin_pct']:.1f}%")

    # Thermal Design Validation
    print("\n--- THERMAL DESIGN VALIDATION ---")
    thermal_report = api.validate_thermal_design(
        power_w=2.0,
        theta_ja_c_w=50.0,
        max_tj_c=125.0,
        ambient_c=40.0,
        num_thermal_vias=3
    )
    print(f"Rule ID:   {thermal_report.rule_id}")
    print(f"Status:    {thermal_report.status.value}")
    print(f"Tj:        {thermal_report.metrics['tj_calculated_c']:.1f} C")
    print(f"Margin:    {thermal_report.metrics['margin_c']:.1f} C")

    # ========================================================================
    # PART 3: AI Feedback System
    # ========================================================================
    print("\n" + "=" * 75)
    print("PART 3: AI FEEDBACK SYSTEM")
    print("=" * 75)

    processor = AIFeedbackProcessor()
    processor.add_report(usb_report)
    processor.add_report(cap_report)
    processor.add_report(thermal_report)

    print("\n--- REVIEW SUMMARY ---")
    summary = processor.get_review_summary()
    print(f"Total rules:    {summary['total']}")
    print(f"Pending review: {summary['pending_review']}")
    print(f"By status:      {summary['by_status']}")

    print("\n--- PROCESSING AI COMMANDS ---")

    # Accept the passing rule
    result1 = processor.process_command("ACCEPT DECOUPLING_DISTANCE")
    print(f"Command: ACCEPT DECOUPLING_DISTANCE")
    print(f"Result:  {result1.action_taken}")
    print(f"Status:  {result1.new_status.value}")

    # Override the USB rule
    result2 = processor.process_command('OVERRIDE USB2_LENGTH_MATCHING new_threshold=5.0 reason="Using USB Full-Speed mode"')
    print(f"\nCommand: OVERRIDE USB2_LENGTH_MATCHING new_threshold=5.0")
    print(f"Result:  {result2.action_taken}")

    # Query a rule
    result3 = processor.process_command("EXPLAIN USB2_LENGTH_MATCHING")
    print(f"\nCommand: EXPLAIN USB2_LENGTH_MATCHING")
    print(f"Result:  {result3.action_taken}")

    # ========================================================================
    # PART 4: Full Design Pipeline with AI Review
    # ========================================================================
    print("\n" + "=" * 75)
    print("PART 4: FULL DESIGN PIPELINE WITH AI REVIEW")
    print("=" * 75)

    # Create a sample design context
    context = DesignContext(
        design_name="ESP32_USB_Sensor",
        board=BoardSpec(width_mm=50, height_mm=40, layer_count=2),
        components=[
            ComponentSpec(
                ref_des="U1",
                part_number="ESP32-WROOM-32",
                category="MCU",
                subcategory="WIFI",
                voltage_rating=3.6,
                power_dissipation=0.5,
                theta_ja=45.0,
            ),
            ComponentSpec(
                ref_des="U2",
                part_number="AMS1117-3.3",
                category="REGULATOR",
                subcategory="LDO",
                voltage_rating=15.0,
                current_rating=1.0,
                power_dissipation=1.7,
                theta_ja=90.0,
            ),
            ComponentSpec(
                ref_des="C1",
                part_number="100nF",
                category="CAPACITOR",
                subcategory="CERAMIC",
                value=100e-9,
            ),
        ],
        nets=[
            NetSpec(name="GND", net_type="GND"),
            NetSpec(name="3V3", net_type="POWER", voltage=3.3),
        ],
    )

    # Run the pipeline with AI review enabled
    pipeline = DesignPipeline(enable_ai_review=True)
    plan = pipeline.create_design_plan(context)

    print("\n--- DESIGN PLAN SUMMARY ---")
    print(f"Placement decisions: {len(plan.placement_decisions)}")
    print(f"Routing decisions:   {len(plan.routing_decisions)}")
    print(f"Design rules:        {len(plan.design_rules)}")
    print(f"Warnings:            {len(plan.warnings)}")

    print("\n--- AI DESIGN REVIEW ---")
    if plan.design_review:
        review = plan.design_review
        print(f"Design Status:     {review.design_status}")
        print(f"Compliance Score:  {review.compliance_score:.1%}")
        print(f"Total Checked:     {review.total_rules_checked}")
        print(f"Passed:            {review.passed}")
        print(f"Failed:            {review.failed}")
        print(f"Warnings:          {review.warnings}")

        if review.blocking_violations:
            print(f"\nBlocking Violations:")
            for v in review.blocking_violations[:3]:
                print(f"  - {v}")

    print("\n--- GENERATED RULE REPORTS ---")
    for report in plan.rule_reports[:5]:
        print(f"  [{report.status.value:7}] {report.rule_id}")

    # ========================================================================
    # PART 5: JSON Export for External AI
    # ========================================================================
    print("\n" + "=" * 75)
    print("PART 5: JSON EXPORT FOR EXTERNAL AI")
    print("=" * 75)

    print("\n--- USB REPORT AS JSON ---")
    usb_json = usb_report.to_json()
    parsed = json.loads(usb_json)
    print(json.dumps({
        "rule_id": parsed["rule_id"],
        "status": parsed["status"],
        "threshold": parsed["threshold"],
        "actual_value": parsed["actual_value"],
        "violations": parsed["violations"],
        "feedback_commands": parsed["feedback_commands"]
    }, indent=2))

    print("\n" + "=" * 75)
    print("DEMONSTRATION COMPLETE - SYSTEM WORKS!")
    print("=" * 75)
    print("")
    print("The Rules API provides:")
    print("  * 631 verified design rules from IPC/JEDEC/USB/PCIe standards")
    print("  * 100 callable functions organized by category")
    print("  * AI-readable RuleReports with JSON export")
    print("  * Feedback commands: ACCEPT, REJECT, CORRECT, OVERRIDE, QUERY, EXPLAIN")
    print("  * Full integration with DesignPipeline for automated review")
    print("")


if __name__ == "__main__":
    main()
