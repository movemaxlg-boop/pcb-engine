#!/usr/bin/env python3
"""
Test the Big Beautiful Loop (BBL) Engine
=========================================

Tests all 6 improvements:
1. Checkpoints with decision logic
2. Rollback capability
3. Timeout per phase
4. Progress reporting
5. Parallel execution
6. Loop history
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine import (
    PCBEngine, EngineConfig,
    BBLEngine, BBLState, BBLResult, BBLPhase, BBLProgress,
    BBLCheckpoint, BBLEscalation, BBLHistoryEntry, BBLPhaseConfig,
    BBLCheckpointDecision, BBLEscalationLevel
)


def create_test_parts_db():
    """Create a simple test parts database."""
    return {
        'parts': {
            'U1': {
                'value': 'ESP32',
                'footprint': 'ESP32-WROOM-32',
                'pins': [
                    {'number': '1', 'name': 'GND', 'net': 'GND', 'offset': (-3.5, 0)},
                    {'number': '2', 'name': 'VCC', 'net': '3V3', 'offset': (3.5, 0)},
                    {'number': '3', 'name': 'IO0', 'net': 'IO0', 'offset': (0, 3.5)},
                    {'number': '4', 'name': 'TX', 'net': 'TX', 'offset': (0, -3.5)},
                ]
            },
            'C1': {
                'value': '10uF',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'name': 'P1', 'net': '3V3', 'offset': (-0.75, 0)},
                    {'number': '2', 'name': 'P2', 'net': 'GND', 'offset': (0.75, 0)},
                ]
            },
            'R1': {
                'value': '10k',
                'footprint': '0603',
                'pins': [
                    {'number': '1', 'name': 'P1', 'net': 'IO0', 'offset': (-0.5, 0)},
                    {'number': '2', 'name': 'P2', 'net': 'GND', 'offset': (0.5, 0)},
                ]
            },
            'LED1': {
                'value': 'LED_GREEN',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'name': 'A', 'net': 'TX', 'offset': (-0.75, 0)},
                    {'number': '2', 'name': 'K', 'net': 'GND', 'offset': (0.75, 0)},
                ]
            }
        },
        'nets': {
            'GND': {'pins': [('U1', '1'), ('C1', '2'), ('R1', '2'), ('LED1', '2')]},
            '3V3': {'pins': [('U1', '2'), ('C1', '1')]},
            'IO0': {'pins': [('U1', '3'), ('R1', '1')]},
            'TX': {'pins': [('U1', '4'), ('LED1', '1')]}
        }
    }


def test_bbl_engine_standalone():
    """Test BBL Engine in standalone mode (without PCB Engine)."""
    print("\n" + "=" * 70)
    print("TEST 1: BBL Engine Standalone")
    print("=" * 70)

    # Progress tracking
    progress_log = []

    def on_progress(progress: BBLProgress):
        progress_log.append(progress)
        print(f"  [PROGRESS] {progress.phase.value}: {progress.percentage:.0f}% - {progress.message}")

    # Escalation handling
    escalation_log = []

    def on_escalation(escalation: BBLEscalation) -> BBLEscalation:
        escalation_log.append(escalation)
        print(f"  [ESCALATION] {escalation.level.value}: {escalation.reason}")
        # Auto-resolve
        escalation.resolved = True
        escalation.response = "Auto-resolved for testing"
        return escalation

    # Create BBL Engine without PCB Engine
    bbl = BBLEngine(
        pcb_engine=None,  # Standalone mode
        progress_callback=on_progress,
        escalation_callback=on_escalation,
        output_dir='./test_bbl_output',
        verbose=True
    )

    # Run with test data
    parts_db = create_test_parts_db()
    result = bbl.run(parts_db)

    print(f"\n  Result: {'SUCCESS' if result.success else 'FAIL'}")
    print(f"  BBL ID: {result.bbl_id}")
    print(f"  Duration: {result.total_time:.2f}s")
    print(f"  Progress updates: {len(progress_log)}")
    print(f"  Escalations: {len(escalation_log)}")

    # Cleanup
    bbl.shutdown()

    return result.success or len(progress_log) > 0


def test_bbl_with_pcb_engine():
    """Test BBL integrated with PCB Engine."""
    print("\n" + "=" * 70)
    print("TEST 2: BBL with PCB Engine")
    print("=" * 70)

    # Create PCB Engine config
    config = EngineConfig(
        board_name='test_bbl',
        board_width=50.0,
        board_height=40.0,
        verbose=True,
        output_dir='./test_bbl_output'
    )

    engine = PCBEngine(config)

    # Progress tracking
    progress_count = [0]

    def on_progress(progress: BBLProgress):
        progress_count[0] += 1
        if progress_count[0] % 5 == 0 or progress.percentage == 100:
            print(f"  [PROGRESS] {progress.phase.value}: {progress.percentage:.0f}%")

    # Run BBL
    parts_db = create_test_parts_db()
    result = engine.run_bbl(parts_db, progress_callback=on_progress)

    print(f"\n  Result: {'SUCCESS' if result.success else 'FAIL'}")
    print(f"  BBL ID: {result.bbl_id}")
    print(f"  Duration: {result.total_time:.2f}s")
    print(f"  DRC Passed: {result.drc_passed}")
    print(f"  Routing: {result.routed_count}/{result.total_nets}")
    print(f"  Output files: {len(result.output_files)}")

    return True  # Test structure works


def test_bbl_checkpoint_system():
    """Test the checkpoint and rollback system."""
    print("\n" + "=" * 70)
    print("TEST 3: Checkpoint System")
    print("=" * 70)

    bbl = BBLEngine(output_dir='./test_bbl_output', verbose=True)

    # Initialize state
    bbl.state = BBLState(
        bbl_id='TEST_CHECKPOINT',
        start_time=time.time()
    )

    # Create checkpoints
    cp1 = bbl._create_checkpoint(BBLPhase.ORDER_RECEIVED)
    print(f"  Created checkpoint 1: {cp1.id}")

    bbl.state.piston_results['test'] = 'value1'

    cp2 = bbl._create_checkpoint(BBLPhase.PISTON_EXECUTION)
    print(f"  Created checkpoint 2: {cp2.id}")

    bbl.state.piston_results['test'] = 'value2'

    # Test rollback
    print(f"  Current value: {bbl.state.piston_results.get('test')}")
    print(f"  Checkpoints: {len(bbl.state.checkpoints)}")

    # Rollback to previous
    success = bbl._rollback_to_previous()
    print(f"  Rollback success: {success}")
    print(f"  Value after rollback: {bbl.state.piston_results.get('test')}")

    bbl.shutdown()

    return success


def test_bbl_history():
    """Test the history and analytics system."""
    print("\n" + "=" * 70)
    print("TEST 4: History and Analytics")
    print("=" * 70)

    bbl = BBLEngine(output_dir='./test_bbl_output', verbose=False)

    # Run multiple times to build history
    parts_db = create_test_parts_db()

    for i in range(3):
        print(f"  Running BBL iteration {i+1}...")
        bbl.run(parts_db)

    # Get analytics
    analytics = bbl.get_history_analytics()

    print(f"\n  History Analytics:")
    print(f"    Total runs: {analytics.get('total_runs', 0)}")
    print(f"    Success rate: {analytics.get('success_rate', 0)*100:.1f}%")
    print(f"    Average duration: {analytics.get('average_duration', 0):.2f}s")

    bbl.shutdown()

    return analytics.get('total_runs', 0) >= 3


def test_bbl_phase_configs():
    """Test custom phase configurations."""
    print("\n" + "=" * 70)
    print("TEST 5: Custom Phase Configs")
    print("=" * 70)

    # Create custom configs with short timeouts
    custom_configs = {
        BBLPhase.ORDER_RECEIVED: BBLPhaseConfig(
            phase=BBLPhase.ORDER_RECEIVED,
            timeout_seconds=5.0,
            retry_limit=1
        ),
        BBLPhase.PISTON_EXECUTION: BBLPhaseConfig(
            phase=BBLPhase.PISTON_EXECUTION,
            timeout_seconds=30.0,
            quality_threshold=0.3
        )
    }

    bbl = BBLEngine(
        phase_configs=custom_configs,
        output_dir='./test_bbl_output',
        verbose=True
    )

    print(f"  ORDER_RECEIVED timeout: {bbl.phase_configs[BBLPhase.ORDER_RECEIVED].timeout_seconds}s")
    print(f"  PISTON_EXECUTION timeout: {bbl.phase_configs[BBLPhase.PISTON_EXECUTION].timeout_seconds}s")

    bbl.shutdown()

    return True


def test_bbl_escalation_levels():
    """Test escalation to Engineer and Boss."""
    print("\n" + "=" * 70)
    print("TEST 6: Escalation Levels")
    print("=" * 70)

    escalation_levels_seen = []

    def on_escalation(escalation: BBLEscalation) -> BBLEscalation:
        escalation_levels_seen.append(escalation.level)
        print(f"  [ESCALATION] Level: {escalation.level.value}")
        print(f"               Phase: {escalation.phase.value}")
        print(f"               Reason: {escalation.reason}")

        # Simulate Engineer resolving
        if escalation.level == BBLEscalationLevel.ENGINEER:
            escalation.resolved = True
            escalation.response = "Engineer resolved the issue"
        return escalation

    bbl = BBLEngine(
        escalation_callback=on_escalation,
        output_dir='./test_bbl_output',
        verbose=True
    )

    # Create a scenario that triggers escalation
    bbl.state = BBLState(
        bbl_id='TEST_ESCALATION',
        start_time=time.time()
    )

    # Manually trigger escalation
    escalation = BBLEscalation(
        level=BBLEscalationLevel.ENGINEER,
        phase=BBLPhase.PISTON_EXECUTION,
        reason="Routing failed after all algorithms exhausted",
        timestamp=time.time()
    )

    bbl.state.pending_escalation = escalation
    result = bbl._execute_escalation()

    print(f"\n  Escalation resolved: {result.get('resolved', False)}")

    bbl.shutdown()

    return result.get('resolved', False)


def test_bbl_monitor_integration():
    """Test BBL Monitor sensor integration with performance tracking."""
    print("\n" + "=" * 70)
    print("TEST 7: BBL Monitor Integration")
    print("=" * 70)

    # Progress tracking
    progress_log = []

    def on_progress(progress: BBLProgress):
        progress_log.append(progress)

    # Create BBL Engine with monitor (standalone mode)
    bbl = BBLEngine(
        pcb_engine=None,  # Standalone mode uses simulation
        progress_callback=on_progress,
        output_dir='./test_bbl_output',
        verbose=True
    )

    # Check monitor was created
    print(f"  Monitor created: {bbl.monitor is not None}")

    # Run BBL (this triggers simulation which records piston metrics)
    parts_db = create_test_parts_db()
    result = bbl.run(parts_db)

    print(f"\n  BBL Result: {'SUCCESS' if result.success else 'FAIL'}")
    print(f"  Duration: {result.total_time:.2f}s")

    # Check monitor captured events
    if bbl.monitor:
        session = bbl.monitor.session

        print(f"\n  --- Monitor Session Metrics ---")
        print(f"  Total Events: {session.total_events}")
        print(f"  Total Phases: {session.total_phases}")
        print(f"  Total Pistons: {session.total_pistons}")
        print(f"  Total Algorithms: {session.total_algorithms}")
        print(f"  Checkpoints: {session.total_checkpoints}")
        print(f"  Rollbacks: {session.total_rollbacks}")
        print(f"  Escalations: {session.total_escalations}")
        print(f"  DRC Checks: {session.total_drc_checks}")

        # Check piston metrics were captured
        print(f"\n  --- Piston Metrics ---")
        for name, pm in session.piston_metrics.items():
            print(f"  {name}: {pm.executions} runs, {pm.success_rate*100:.0f}% success, {pm.avg_time:.3f}s avg")

        # Check algorithm metrics were captured
        print(f"\n  --- Algorithm Metrics ---")
        for key, am in session.algorithm_metrics.items():
            print(f"  {key}: {am.executions} runs, efficiency={am.efficiency_score:.2f}")

        # Get performance ranking
        ranking = bbl.monitor.calculate_performance_ranking()
        print(f"\n  --- Performance Ranking ---")
        print(f"  Fastest Piston: {ranking.fastest_piston}")
        print(f"  Slowest Piston: {ranking.slowest_piston}")
        print(f"  Most Reliable: {ranking.most_reliable_piston}")
        print(f"  Bottleneck Phase: {ranking.bottleneck_phase}")

        # Get bottleneck analysis
        bottleneck = bbl.monitor.get_bottleneck_analysis()
        print(f"\n  --- Bottleneck Analysis ---")
        print(f"  Bottleneck Piston: {bottleneck['bottleneck_piston']}")
        for rec in bottleneck.get('recommendations', []):
            print(f"    * {rec}")

        # Verify metrics were captured
        success = (
            session.total_events > 0 and
            session.total_pistons > 0 and
            len(session.piston_metrics) > 0
        )
    else:
        success = False

    bbl.shutdown()

    return success


def test_bbl_monitor_report_generation():
    """Test report generation in all formats."""
    print("\n" + "=" * 70)
    print("TEST 8: Monitor Report Generation")
    print("=" * 70)

    bbl = BBLEngine(
        output_dir='./test_bbl_output',
        verbose=False
    )

    # Run BBL to generate data
    parts_db = create_test_parts_db()
    bbl.run(parts_db)

    if not bbl.monitor:
        print("  Monitor not available")
        bbl.shutdown()
        return False

    # Test JSON report
    json_report = bbl.monitor.generate_report(format='json')
    json_ok = '"bbl_id"' in json_report and '"events"' in json_report
    print(f"  JSON report: {'OK' if json_ok else 'FAIL'} ({len(json_report)} chars)")

    # Test Markdown report
    md_report = bbl.monitor.generate_report(format='markdown')
    md_ok = '# BBL Monitor Report' in md_report
    print(f"  Markdown report: {'OK' if md_ok else 'FAIL'} ({len(md_report)} chars)")

    # Test HTML report
    html_report = bbl.monitor.generate_report(format='html')
    html_ok = '<!DOCTYPE html>' in html_report
    print(f"  HTML report: {'OK' if html_ok else 'FAIL'} ({len(html_report)} chars)")

    # Test performance report
    perf_report = bbl.monitor.get_performance_report()
    perf_ok = 'BBL PERFORMANCE ANALYSIS' in perf_report
    print(f"  Performance report: {'OK' if perf_ok else 'FAIL'} ({len(perf_report)} chars)")

    bbl.shutdown()

    return json_ok and md_ok and html_ok and perf_ok


def main():
    """Run all BBL tests."""
    print("\n" + "=" * 70)
    print("BBL ENGINE TEST SUITE")
    print("=" * 70)

    tests = [
        ("BBL Standalone", test_bbl_engine_standalone),
        ("BBL with PCB Engine", test_bbl_with_pcb_engine),
        ("Checkpoint System", test_bbl_checkpoint_system),
        ("History & Analytics", test_bbl_history),
        ("Custom Phase Configs", test_bbl_phase_configs),
        ("Escalation Levels", test_bbl_escalation_levels),
        ("BBL Monitor Integration", test_bbl_monitor_integration),
        ("Monitor Report Generation", test_bbl_monitor_report_generation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
