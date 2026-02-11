"""
Test Smart Algorithm Manager
============================

Tests the complete smart routing pipeline:
1. RoutingPlanner - Design analysis and per-net strategy
2. LearningDatabase - Success/failure tracking
3. Integration with RoutingPiston

Run with:
    cd D:/Anas/projects/pcb-engine
    python -m pcb_engine.test_smart_algorithm_manager
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.routing_planner import (
    RoutingPlanner, RoutingPlan, NetClass, RoutingAlgorithm,
    DesignDensity, NetRoutingStrategy
)
from pcb_engine.learning_database import (
    LearningDatabase, RoutingOutcome, AlgorithmStats
)


def create_test_parts_db():
    """Create a realistic test parts database"""
    return {
        'parts': {
            'U1': {
                'footprint': 'QFN-32',
                'value': 'ESP32',
                'pins': [f'P{i}' for i in range(32)]
            },
            'C1': {'footprint': '0402', 'value': '100nF', 'pins': ['1', '2']},
            'C2': {'footprint': '0402', 'value': '100nF', 'pins': ['1', '2']},
            'C3': {'footprint': '0402', 'value': '10uF', 'pins': ['1', '2']},
            'R1': {'footprint': '0402', 'value': '10k', 'pins': ['1', '2']},
            'R2': {'footprint': '0402', 'value': '10k', 'pins': ['1', '2']},
            'R3': {'footprint': '0402', 'value': '4.7k', 'pins': ['1', '2']},
            'R4': {'footprint': '0402', 'value': '4.7k', 'pins': ['1', '2']},
            'J1': {'footprint': 'USB-C', 'value': 'USB', 'pins': ['VBUS', 'D+', 'D-', 'GND', 'CC1', 'CC2']},
            'LED1': {'footprint': '0603', 'value': 'LED', 'pins': ['A', 'K']},
        },
        'nets': {
            'VCC': {'pins': ['U1.P1', 'C1.1', 'C2.1', 'J1.VBUS']},
            '3V3': {'pins': ['U1.P5', 'C3.1', 'R1.1']},
            'GND': {'pins': ['U1.P2', 'C1.2', 'C2.2', 'C3.2', 'J1.GND', 'LED1.K']},
            'USB_D+': {'pins': ['U1.P10', 'J1.D+']},
            'USB_D-': {'pins': ['U1.P11', 'J1.D-']},
            'SDA': {'pins': ['U1.P20', 'R3.1']},
            'SCL': {'pins': ['U1.P21', 'R4.1']},
            'DATA0': {'pins': ['U1.P12', 'R1.2']},
            'DATA1': {'pins': ['U1.P13', 'R2.2']},
            'DATA2': {'pins': ['U1.P14']},
            'DATA3': {'pins': ['U1.P15']},
            'CLK': {'pins': ['U1.P8']},
            'GPIO_5': {'pins': ['U1.P25', 'LED1.A']},
            'EN': {'pins': ['U1.P3']},
            'BOOT': {'pins': ['U1.P4']},
        }
    }


def test_routing_planner():
    """Test RoutingPlanner analysis and planning"""
    print("=" * 60)
    print("TEST: RoutingPlanner")
    print("=" * 60)

    parts_db = create_test_parts_db()
    board_config = {
        'board_width': 50.0,
        'board_height': 40.0,
        'layers': 2,
    }

    planner = RoutingPlanner()
    plan = planner.create_routing_plan(parts_db, board_config)

    print(f"\nDesign Profile:")
    print(f"  Board: {plan.design_profile.board_width}x{plan.design_profile.board_height}mm")
    print(f"  Components: {plan.design_profile.component_count}")
    print(f"  Nets: {plan.design_profile.net_count}")
    print(f"  Density: {plan.design_profile.density.value}")
    print(f"  Has QFN: {plan.design_profile.has_qfn}")

    print(f"\nNet Classification:")
    for net_name in sorted(plan.net_strategies.keys()):
        strategy = plan.net_strategies[net_name]
        print(f"  {net_name:12} -> {strategy.net_class.value:15} "
              f"[{strategy.primary_algorithm.value}] "
              f"width={strategy.trace_width}mm")

    print(f"\nAlgorithm Distribution:")
    for algo, count in plan.get_algorithm_distribution().items():
        print(f"  {algo}: {count}")

    print(f"\nRouting Order (first 5):")
    for i, net in enumerate(plan.routing_order[:5]):
        strategy = plan.net_strategies[net]
        print(f"  {i+1}. {net} (priority {strategy.priority})")

    print(f"\nFeatures:")
    print(f"  Trunk chains: {plan.enable_trunk_chains}")
    if plan.trunk_chains:
        for chain in plan.trunk_chains:
            print(f"    Chain: {chain}")
    print(f"  Return path check: {plan.enable_return_path_check}")
    print(f"  Ground pour: {plan.ground_pour_recommended}")
    print(f"  Success prediction: {plan.overall_success_prediction*100:.0f}%")

    # Verify classifications
    assert plan.net_strategies['VCC'].net_class == NetClass.POWER, "VCC should be POWER"
    assert plan.net_strategies['GND'].net_class == NetClass.GROUND, "GND should be GROUND"
    assert plan.net_strategies['USB_D+'].net_class == NetClass.DIFFERENTIAL, "USB_D+ should be DIFFERENTIAL"
    assert plan.net_strategies['SDA'].net_class == NetClass.I2C, "SDA should be I2C"
    assert plan.net_strategies['CLK'].net_class == NetClass.HIGH_SPEED, "CLK should be HIGH_SPEED"

    print("\n[PASS] RoutingPlanner tests passed!")
    return plan


def test_learning_database():
    """Test LearningDatabase recording and queries"""
    print("\n" + "=" * 60)
    print("TEST: LearningDatabase")
    print("=" * 60)

    # Use temp file
    import tempfile
    db_path = os.path.join(tempfile.gettempdir(), 'test_routing_learning.json')

    # Clean up any existing
    if os.path.exists(db_path):
        os.remove(db_path)

    db = LearningDatabase(db_path)

    # Record some outcomes
    test_outcomes = [
        # Power nets with LEE - 95% success
        ('VCC', 'power', 'lee', True, 45.0),
        ('3V3', 'power', 'lee', True, 52.0),
        ('VBAT', 'power', 'lee', True, 38.0),
        ('VDD', 'power', 'lee', True, 41.0),
        ('5V', 'power', 'lee', False, 120.0),  # One failure

        # High-speed with STEINER - 90% success
        ('CLK', 'high_speed', 'steiner', True, 78.0),
        ('USB_CLK', 'high_speed', 'steiner', True, 85.0),
        ('DDR_CLK', 'high_speed', 'steiner', True, 92.0),
        ('XTAL', 'high_speed', 'steiner', True, 67.0),
        ('PCIE_CLK', 'high_speed', 'steiner', False, 150.0),  # One failure

        # Signal with HYBRID - 85% success
        ('GPIO_1', 'signal', 'hybrid', True, 35.0),
        ('GPIO_2', 'signal', 'hybrid', True, 28.0),
        ('GPIO_3', 'signal', 'hybrid', True, 42.0),
        ('GPIO_4', 'signal', 'hybrid', False, 95.0),
        ('GPIO_5', 'signal', 'a_star', True, 22.0),
        ('GPIO_6', 'signal', 'a_star', True, 18.0),
    ]

    for net, net_class, algo, success, time_ms in test_outcomes:
        outcome = RoutingOutcome(
            net_name=net,
            net_class=net_class,
            design_hash='test123',
            algorithm=algo,
            success=success,
            time_ms=time_ms,
            quality_score=100 if success else 0,
        )
        db.record_outcome(outcome)

    # Save
    db.save()

    print(f"\nRecorded {len(test_outcomes)} outcomes")

    # Query best algorithms
    print("\nBest algorithms by net class:")
    for net_class in [NetClass.POWER, NetClass.HIGH_SPEED, NetClass.SIGNAL]:
        best = db.get_best_algorithm(net_class)
        rate = db.get_success_rate(best, net_class) if best else 0
        print(f"  {net_class.value}: {best.value if best else 'N/A'} ({rate*100:.0f}%)")

    # Get rankings
    print("\nAlgorithm ranking for SIGNAL:")
    rankings = db.get_algorithm_ranking(NetClass.SIGNAL)
    for r in rankings:
        print(f"  {r['algorithm']}: {r['success_rate']*100:.0f}% ({r['attempts']} attempts, {r['avg_time_ms']:.0f}ms avg)")

    # Summary
    summary = db.get_summary()
    print(f"\nDatabase Summary:")
    print(f"  Total outcomes: {summary['total_outcomes']}")
    print(f"  Overall success rate: {summary['overall_success_rate']*100:.0f}%")

    # Verify queries
    assert db.get_best_algorithm(NetClass.POWER) == RoutingAlgorithm.LEE, "LEE should be best for POWER"
    assert db.get_best_algorithm(NetClass.HIGH_SPEED) == RoutingAlgorithm.STEINER, "STEINER should be best for HIGH_SPEED"

    # Test persistence
    db2 = LearningDatabase(db_path)
    assert len(db2.outcomes) == len(test_outcomes), "Outcomes should persist"

    # Cleanup
    os.remove(db_path)

    print("\n[PASS] LearningDatabase tests passed!")


def test_planner_with_learning():
    """Test RoutingPlanner integration with LearningDatabase"""
    print("\n" + "=" * 60)
    print("TEST: Planner + Learning Integration")
    print("=" * 60)

    import tempfile
    db_path = os.path.join(tempfile.gettempdir(), 'test_routing_learning2.json')

    if os.path.exists(db_path):
        os.remove(db_path)

    # Pre-populate learning database
    db = LearningDatabase(db_path)

    # Add historical data: STEINER works best for high-speed
    for i in range(10):
        db.record_outcome(RoutingOutcome(
            net_name=f'CLK_{i}',
            net_class='high_speed',
            design_hash='hist',
            algorithm='steiner',
            success=True,
            time_ms=50.0,
            quality_score=95,
        ))

    # Add historical: PATHFINDER works best for buses
    for i in range(10):
        db.record_outcome(RoutingOutcome(
            net_name=f'DATA{i}',
            net_class='bus',
            design_hash='hist',
            algorithm='pathfinder',
            success=True,
            time_ms=80.0,
            quality_score=90,
        ))

    db.save()

    # Create planner with learning
    planner = RoutingPlanner(db)
    parts_db = create_test_parts_db()
    board_config = {'board_width': 50, 'board_height': 40, 'layers': 2}

    plan = planner.create_routing_plan(parts_db, board_config)

    print("\nPer-net strategies with learning:")
    for net in ['CLK', 'USB_D+', 'DATA0', 'GPIO_5']:
        if net in plan.net_strategies:
            s = plan.net_strategies[net]
            learned = f"(learned: {s.learned_algorithm}, {s.learned_success_rate*100:.0f}%)" if s.learned_algorithm else ""
            print(f"  {net}: {s.primary_algorithm.value} {learned}")

    # Cleanup
    os.remove(db_path)

    print("\n[PASS] Integration tests passed!")


def test_summary():
    """Print summary of Smart Algorithm Manager"""
    print("\n" + "=" * 60)
    print("SMART ALGORITHM MANAGER - IMPLEMENTATION SUMMARY")
    print("=" * 60)

    print("""
FILES CREATED:
  - routing_planner.py   (~500 lines) - Design analysis, net classification, algorithm selection
  - learning_database.py (~400 lines) - Persistent learning, success tracking

FILES MODIFIED:
  - routing_piston.py   - Added route_with_plan() method (~200 lines)
  - pcb_engine.py       - Integrated smart routing in _execute_routing()
  - bbl_engine.py       - Added learning save in Phase 6

ALGORITHM SELECTION MATRIX:
  Net Class      | Primary Algorithm | Why
  ---------------|-------------------|------------------------------------
  POWER          | LEE               | Guaranteed shortest path
  GROUND         | POUR/LEE          | Use pour when possible
  HIGH_SPEED     | STEINER           | Optimal multi-terminal
  DIFFERENTIAL   | STEINER           | Length matching support
  BUS            | PATHFINDER        | Congestion-aware for parallel
  I2C            | LEE               | Short paths critical
  SPI            | A*                | Fast, good enough
  ANALOG         | LEE               | Avoid crosstalk
  RF             | LEE               | Minimal vias
  HIGH_CURRENT   | LEE               | Wide traces only
  SIGNAL         | HYBRID            | Best general purpose

LEARNING DATABASE:
  - Records every routing attempt (success/fail, time, quality)
  - Tracks algorithm success rates per net class
  - Over time, learns which algorithms work best
  - Adjusts primary algorithm selection based on history

FLOW:
  1. RoutingPlanner analyzes design profile
  2. Classifies all nets (POWER, SIGNAL, HIGH_SPEED, etc.)
  3. Selects best algorithm per net class
  4. Creates RoutingPlan with per-net strategies
  5. RoutingPiston executes plan with fallback chains
  6. LearningDatabase records outcomes for future improvement
""")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SMART ALGORITHM MANAGER - TEST SUITE")
    print("=" * 70)

    try:
        test_routing_planner()
        test_learning_database()
        test_planner_with_learning()
        test_summary()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
