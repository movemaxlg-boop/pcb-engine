"""
CONSTITUTIONAL LAYOUT SYSTEM - END-TO-END DEMONSTRATION
========================================================

This demo shows the complete workflow:
1. User provides natural language design description
2. AI parses intent and asks clarifying questions
3. AI generates Constitutional Layout with:
   - Component selection + inference
   - Rule hierarchy (Inviolable/Recommended/Optional)
   - Placement and routing hints
   - Overrides with justification
4. Validator checks c_layout before BBL
5. BBL adapter runs the design (simulated)
6. Results classified by rule priority

Run: python -m circuit_intelligence.demo_clayout
"""

import json
from pprint import pprint

# Import the Constitutional Layout system
from .circuit_ai import CircuitAI, create_clayout_from_text, get_clarifying_questions
from .clayout_types import (
    ConstitutionalLayout,
    RulePriority,
    ValidationStatus,
)
from .clayout_validator import validate_clayout, CLayoutValidator
from .clayout_bbl_adapter import CLayoutBBLAdapter, run_clayout_through_bbl
from .rule_hierarchy import RuleHierarchyEngine, get_default_hierarchy
from .parts_inference import PartsInferenceEngine


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def main():
    """Run the Constitutional Layout demonstration."""

    print_section("CONSTITUTIONAL LAYOUT SYSTEM - LIVE DEMO")
    print("Converting natural language to machine-readable PCB specs")
    print("with AI-powered component inference and rule hierarchy")

    # =========================================================================
    # PART 1: User provides natural language input
    # =========================================================================
    print_section("PART 1: USER INPUT (Natural Language)")

    user_input = """
    I need a sensor board with:
    - ESP32-WROOM-32 module for WiFi connectivity
    - USB-C connector for power and programming
    - 3 analog inputs for temperature sensors (0-3.3V)
    - Board size around 50x40mm
    - 2-layer PCB is fine
    """

    print(user_input)

    # =========================================================================
    # PART 2: AI parses intent
    # =========================================================================
    print_section("PART 2: AI INTENT PARSING")

    ai = CircuitAI()
    intent = ai.parse_user_intent(user_input)

    print(f"Design Name:     {intent.design_name}")
    print(f"Components:      {intent.required_components}")
    print(f"Features:        {intent.required_features}")
    print(f"Size Constraint: {intent.board_size_constraint}")
    print(f"Layer Constraint:{intent.layer_constraint}")

    # Show clarifying questions
    questions = ai.ask_clarifying_questions(intent)
    if questions:
        print_subsection("AI would ask these clarifying questions")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")

        # Simulate user answers
        print_subsection("Simulated user answers")
        answers = {
            "What USB speed is required?": "USB 2.0 Full-Speed is fine",
            "What is the power input?": "USB 5V power only",
        }
        for q, a in answers.items():
            if any(kw in q.lower() for kw in ["usb", "power"]):
                print(f"  Q: {q[:50]}...")
                print(f"  A: {a}")

        intent = ai.process_user_answers(intent, answers)

    # =========================================================================
    # PART 3: Generate Constitutional Layout
    # =========================================================================
    print_section("PART 3: CONSTITUTIONAL LAYOUT GENERATION")

    clayout = ai.generate_clayout(intent)

    print_subsection("Design Summary")
    summary = clayout.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print_subsection("Board Constraints")
    board = clayout.board
    print(f"  Size:       {board.width_mm} x {board.height_mm} mm")
    print(f"  Layers:     {board.layer_count}")
    print(f"  Min Trace:  {board.min_trace_mm} mm")
    print(f"  Min Space:  {board.min_space_mm} mm")

    print_subsection("User-Specified Components")
    user_comps = clayout.get_user_components()
    for comp in user_comps:
        print(f"  {comp.ref_des}: {comp.part_number} ({comp.footprint})")

    print_subsection("AI-Inferred Components (The Smart Part!)")
    inferred_comps = clayout.get_inferred_components()
    if inferred_comps:
        for comp in inferred_comps:
            print(f"  + {comp.ref_des}: {comp.value or comp.part_number}")
            print(f"      Reason: {comp.inference_reason}")
            print(f"      For: {comp.inferred_for}")
    else:
        print("  (No components inferred in this demo)")

    print_subsection("Nets Defined")
    for net in clayout.nets[:5]:  # Show first 5
        print(f"  {net.name}: {net.net_type.value}")
        if net.impedance_ohm:
            print(f"      Impedance: {net.impedance_ohm} ohm")
        if net.matched_with:
            print(f"      Matched with: {net.matched_with}")

    # =========================================================================
    # PART 4: Rule Hierarchy (THE KEY INNOVATION)
    # =========================================================================
    print_section("PART 4: RULE HIERARCHY (The Key Innovation)")

    rules = clayout.rules
    counts = rules.count()

    print(f"\nTotal Rules Classified: {counts['total']}")
    print(f"  INVIOLABLE:  {counts['inviolable']:3d} - Must pass or ABORT")
    print(f"  RECOMMENDED: {counts['recommended']:3d} - Should pass, warn if not")
    print(f"  OPTIONAL:    {counts['optional']:3d} - Nice to have")

    print_subsection("Sample INVIOLABLE Rules (Safety-Critical)")
    for rule in rules.inviolable[:5]:
        print(f"  [{rule.rule_id}]")
        print(f"      Reason: {rule.reason}")

    print_subsection("Sample RECOMMENDED Rules")
    for rule in rules.recommended[:3]:
        print(f"  [{rule.rule_id}]")
        print(f"      Reason: {rule.reason}")

    print_subsection("Sample OPTIONAL Rules")
    for rule in rules.optional[:3]:
        print(f"  [{rule.rule_id}]")
        print(f"      Reason: {rule.reason}")

    # =========================================================================
    # PART 5: Overrides (AI Engineering Authority)
    # =========================================================================
    print_section("PART 5: AI OVERRIDES (Engineering Authority)")

    if clayout.overrides:
        for override in clayout.overrides:
            print(f"Rule: {override.rule_id}")
            print(f"  Original: {override.original_value}")
            print(f"  New:      {override.new_value}")
            print(f"  Reason:   {override.justification}")
            print(f"  Evidence: {override.evidence}")
            print(f"  Approved: {override.approved_by}")
    else:
        print("  No overrides needed for this design")

    # =========================================================================
    # PART 6: Placement & Routing Hints
    # =========================================================================
    print_section("PART 6: PLACEMENT & ROUTING HINTS")

    print_subsection("Placement Hints")
    hints = clayout.placement_hints

    print(f"  Edge Components: {hints.edge_components}")

    if hints.proximity_groups:
        print("  Proximity Groups:")
        for group in hints.proximity_groups[:3]:
            print(f"    {group.components} - max {group.max_distance_mm}mm")
            print(f"      Reason: {group.reason}")

    if hints.keep_apart:
        print("  Keep Apart:")
        for ka in hints.keep_apart[:2]:
            print(f"    {ka.component_a} <--{ka.min_distance_mm}mm--> {ka.component_b}")
            print(f"      Reason: {ka.reason}")

    print_subsection("Routing Hints")
    routing = clayout.routing_hints

    print(f"  Priority Nets: {routing.priority_nets}")
    print(f"  Deprioritized: {routing.deprioritized_nets}")

    if routing.diff_pairs:
        print("  Differential Pairs:")
        for dp in routing.diff_pairs:
            print(f"    {dp.positive_net} / {dp.negative_net}")
            print(f"      Z={dp.impedance_ohm}ohm, max mismatch={dp.max_mismatch_mm}mm")

    # =========================================================================
    # PART 7: Validation Gate
    # =========================================================================
    print_section("PART 7: VALIDATION GATE")

    validator = CLayoutValidator()
    result = validator.validate(clayout)

    print(f"Valid:              {result.valid}")
    print(f"Routability Est:    {result.routability_estimate*100:.0f}%")

    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  ! {e}")

    if result.warnings:
        print(f"Warnings ({len(result.warnings)}):")
        for w in result.warnings[:3]:
            print(f"  * {w}")

    if result.suggestions:
        print(f"Suggestions ({len(result.suggestions)}):")
        for s in result.suggestions[:3]:
            print(f"  > {s}")

    # =========================================================================
    # PART 8: BBL Execution (Simulated)
    # =========================================================================
    print_section("PART 8: BBL EXECUTION (Simulated)")

    adapter = CLayoutBBLAdapter()  # No real BBL engine, will simulate
    bbl_result = adapter.run_with_clayout(clayout)

    print(f"Success:            {bbl_result.success}")
    print(f"Routing Completion: {bbl_result.routing_completion_pct:.0f}%")
    print(f"Routed Nets:        {bbl_result.routed_nets}/{bbl_result.total_nets}")
    print(f"DRC Passed:         {bbl_result.drc_passed}")
    print(f"KiCad DRC Passed:   {bbl_result.kicad_drc_passed}")
    print(f"Execution Time:     {bbl_result.total_time*1000:.1f}ms")

    print_subsection("Violations by Priority")
    print(f"  INVIOLABLE:  {len(bbl_result.inviolable_violations)}")
    print(f"  RECOMMENDED: {len(bbl_result.recommended_violations)}")
    print(f"  OPTIONAL:    {len(bbl_result.optional_violations)}")

    if bbl_result.recommended_violations:
        print("\n  Recommended violations (warnings):")
        for v in bbl_result.recommended_violations[:3]:
            print(f"    * {v}")

    if bbl_result.escalation_needed:
        print_subsection("ESCALATION REQUIRED")
        esc = bbl_result.escalation_report
        print(f"  Type:   {esc.failure_type}")
        print(f"  Phase:  {esc.phase}")
        print(f"  Suggestions:")
        for s in esc.suggestions[:3]:
            print(f"    - {s}")

    # =========================================================================
    # PART 9: AI Assumptions & Metadata
    # =========================================================================
    print_section("PART 9: AI ASSUMPTIONS & METADATA")

    print("User Requirements (original statements):")
    for req in clayout.user_requirements:
        print(f"  \"{req[:70]}...\"" if len(req) > 70 else f"  \"{req}\"")

    print("\nAI Assumptions:")
    for assumption in clayout.ai_assumptions:
        print(f"  - {assumption}")

    # =========================================================================
    # PART 10: JSON Export
    # =========================================================================
    print_section("PART 10: JSON EXPORT (Machine-Readable)")

    print("Exporting c_layout as JSON...")
    json_str = clayout.to_json()
    json_data = json.loads(json_str)

    # Show truncated JSON
    print(f"\nJSON size: {len(json_str)} bytes")
    print("\nJSON structure (keys at top level):")
    for key in json_data.keys():
        value = json_data[key]
        if isinstance(value, list):
            print(f"  {key}: [{len(value)} items]")
        elif isinstance(value, dict):
            print(f"  {key}: {{{len(value)} keys}}")
        else:
            print(f"  {key}: {value}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("DEMONSTRATION COMPLETE")

    print("""
The Constitutional Layout System provides:

1. NATURAL LANGUAGE PARSING
   - User describes circuit in plain English
   - AI extracts components, features, constraints

2. COMPONENT INFERENCE
   - AI adds required supporting components
   - Decoupling caps, ESD protection, etc.

3. RULE HIERARCHY (Key Innovation)
   - INVIOLABLE: Safety-critical, abort if violated
   - RECOMMENDED: Best practices, warn if violated
   - OPTIONAL: Quality improvements, log if violated

4. AI OVERRIDE AUTHORITY
   - AI can modify rules with justification
   - All overrides documented with evidence

5. VALIDATION GATE
   - Catches errors before BBL wastes cycles
   - Routability estimation

6. BBL INTEGRATION
   - Converts c_layout to parts_db format
   - Classifies violations by priority
   - Generates escalation reports

7. ESCALATION LOOP
   - Failed designs return to AI for adjustment
   - AI can iterate until success or escalate to user
""")


if __name__ == "__main__":
    main()
