"""
Test CPU Lab with the 18-component ESP32 board.

This validates that the CPU Lab correctly:
1. Detects GND needs pour (19 pins)
2. Sets layer directions (H/V)
3. Plans global routes
4. Estimates congestion
5. Orders nets by priority
6. Detects component groups
7. Factory inspector works

Run: python -u test_cpu_lab.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_real_20_component_board import create_20_component_parts_db
from pcb_engine.cpu_lab import CPULab, FactoryInspector


def test_cpu_lab():
    """Test CPU Lab with the ESP32 sensor board."""

    print("=" * 60)
    print("CPU LAB TEST - ESP32 Sensor Board")
    print("=" * 60)

    parts_db = create_20_component_parts_db()

    board_config = {
        'board_width': 50.0,
        'board_height': 40.0,
        'layers': 2,
    }

    # Run CPU Lab
    cpu_lab = CPULab()
    result = cpu_lab.enhance(parts_db, board_config)

    # Validate results
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    errors = []

    # 1. GND should use pour (19 pins!)
    if result.power_grid.gnd_strategy.value != 'pour':
        errors.append(f"FAIL: GND strategy should be 'pour', got '{result.power_grid.gnd_strategy.value}'")
    else:
        print("  [OK] GND strategy: pour (correct for 19 GND pins)")

    # 2. GND should be removed from routing
    if 'GND' not in result.power_grid.nets_removed_from_routing:
        errors.append("FAIL: GND should be removed from routing queue")
    else:
        print("  [OK] GND removed from routing queue")

    # 3. Layer directions should be set
    if len(result.layer_assignments) != 2:
        errors.append(f"FAIL: Expected 2 layer assignments, got {len(result.layer_assignments)}")
    else:
        print(f"  [OK] Layer assignments: {result.layer_assignments[0].layer_name}="
              f"{result.layer_assignments[0].preferred_direction.value}, "
              f"{result.layer_assignments[1].layer_name}="
              f"{result.layer_assignments[1].preferred_direction.value}")

    # 4. Global routing should have planned routes
    if len(result.global_routing.net_routes) == 0:
        errors.append("FAIL: No global routes planned")
    else:
        print(f"  [OK] Global routes: {len(result.global_routing.net_routes)} nets planned")

    # 5. Net priorities should exist and be ordered
    routable = [p for p in result.net_priorities if p.priority < 100]
    if len(routable) == 0:
        errors.append("FAIL: No routable nets in priority list")
    else:
        print(f"  [OK] Net priorities: {len(routable)} routable nets")

    # 6. Differential pairs should be highest priority
    diff_nets = [p for p in result.net_priorities if p.net_type == 'differential']
    if diff_nets:
        top_priority = min(p.priority for p in result.net_priorities if p.priority < 100)
        if diff_nets[0].priority <= top_priority + 5:
            print(f"  [OK] Differential pairs are top priority (P{diff_nets[0].priority})")
        else:
            errors.append(f"FAIL: Diff pairs not top priority (P{diff_nets[0].priority})")
    else:
        print("  [--] No differential pairs detected (check net classification)")

    # 7. Component groups should be detected
    if len(result.component_groups) > 0:
        print(f"  [OK] Component groups: {len(result.component_groups)} detected")
        for g in result.component_groups:
            print(f"       - {g.name}: {g.components}")
    else:
        print("  [--] No component groups detected")

    # 8. Enhanced parts_db should have cpu_lab section
    if 'cpu_lab' not in result.enhanced_parts_db:
        errors.append("FAIL: Enhanced parts_db missing 'cpu_lab' section")
    else:
        print("  [OK] Enhanced parts_db has cpu_lab decisions")

    # 9. Routing order should exclude GND
    routing_order = result.enhanced_parts_db.get('cpu_lab', {}).get('routing_order', [])
    gnd_in_order = any(r['net'] == 'GND' for r in routing_order)
    if gnd_in_order:
        errors.append("FAIL: GND should NOT be in routing order (handled by pour)")
    else:
        print("  [OK] GND excluded from routing order")

    # 10. Power nets should have wider traces
    power_widths = result.power_grid.power_trace_widths
    if power_widths:
        print(f"  [OK] Power trace widths: {power_widths}")

    # 11. 3V3 should be handled by pour (11 pins on 2-layer = pour)
    if '3V3' in result.power_grid.nets_removed_from_routing:
        print("  [OK] 3V3 removed from routing (handled by pour on F.Cu)")
    else:
        errors.append("FAIL: 3V3 (11 pins) should be removed from routing via pour")

    # 12. 3V3 should have pour config
    if '3V3' in result.power_grid.power_pour_configs:
        cfg = result.power_grid.power_pour_configs['3V3']
        print(f"  [OK] 3V3 pour config: layer={cfg['layer']}")
    else:
        errors.append("FAIL: 3V3 should have pour config in power_pour_configs")

    # 13. Routing order should exclude both GND and 3V3
    routing_order = result.enhanced_parts_db.get('cpu_lab', {}).get('routing_order', [])
    v33_in_order = any(r['net'] == '3V3' for r in routing_order)
    if v33_in_order:
        errors.append("FAIL: 3V3 should NOT be in routing order (handled by pour)")
    else:
        print("  [OK] 3V3 excluded from routing order")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"RESULT: {len(errors)} FAILURES")
        for e in errors:
            print(f"  {e}")
    else:
        print("RESULT: ALL CHECKS PASSED")

    print(f"\nProcessing time: {result.processing_time_ms:.0f}ms")
    print("=" * 60)

    # Test Factory Inspector
    print("\n\nFACTORY INSPECTOR TEST")
    print("-" * 40)

    inspector = cpu_lab.inspector

    # Simulate a placement
    dummy_placement = cpu_lab._create_dummy_placement(parts_db, 50.0, 40.0)
    report = inspector.inspect_placement(
        dummy_placement, parts_db, 50.0, 40.0, result.component_groups
    )
    print(f"  Placement inspection: {'PASS' if report.passed else 'FAIL'}")
    print(f"  Critical: {report.critical_count}, Errors: {report.error_count}, "
          f"Warnings: {report.warning_count}")
    if report.findings:
        for f in report.findings[:5]:
            print(f"    [{f.severity.value}] {f.message}")

    # Simulate routing results
    report2 = inspector.inspect_post_routing({}, 23, 15)
    print(f"\n  Routing inspection (15/23): {'PASS' if report2.passed else 'FAIL'}")
    for f in report2.findings:
        print(f"    [{f.severity.value}] {f.message}")

    cumulative = inspector.get_cumulative_report()
    print(f"\n  Cumulative: {cumulative['total_findings']} findings, "
          f"stop_line={cumulative['should_stop']}")


if __name__ == '__main__':
    test_cpu_lab()
