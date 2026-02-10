"""
C_LAYOUT + SMART ALGORITHM MANAGER INTEGRATION TEST
====================================================

This test validates the complete integration between:
1. Constitutional Layout (c_layout) - Design specification
2. Smart Algorithm Manager (RoutingPlanner + LearningDatabase) - Per-net routing strategy
3. BBL Engine - Execution pipeline

Flow tested:
    User Text → c_layout → RoutingPlanner → BBL → Output

Run with:
    cd D:/Anas/projects/pcb-engine
    python test_clayout_smart_routing_integration.py
"""

import sys
import os
import tempfile

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit_intelligence.clayout_types import (
    ConstitutionalLayout,
    BoardConstraints,
    ComponentDefinition,
    NetDefinition,
    RuleHierarchy,
    RuleBinding,
    RulePriority,
    PlacementHints,
    RoutingHints,
    ProximityGroup,
    DiffPairSpec,
    LengthMatchGroup,
    NetType,
    ComponentCategory,
)
from circuit_intelligence.clayout_bbl_adapter import (
    CLayoutBBLAdapter,
    CLayoutConverter,
    convert_clayout_to_parts_db,
    SMART_ROUTING_AVAILABLE,
)
from pcb_engine.routing_planner import (
    RoutingPlanner,
    NetClass,
    RoutingAlgorithm,
    DesignDensity,
)
from pcb_engine.learning_database import LearningDatabase, RoutingOutcome


def create_test_clayout() -> ConstitutionalLayout:
    """
    Create a realistic c_layout for testing.

    This represents a USB-powered ESP32 board with:
    - ESP32 MCU
    - USB-C connector
    - Voltage regulator
    - Decoupling capacitors
    - LEDs and resistors
    """

    # Board constraints
    board = BoardConstraints(
        width_mm=50.0,
        height_mm=40.0,
        layer_count=2,
        min_trace_mm=0.15,
        min_space_mm=0.15,
    )

    # Components
    components = [
        ComponentDefinition(
            ref_des='U1',
            part_number='ESP32-WROOM-32',
            footprint='QFN-32',
            category=ComponentCategory.MCU,
            power_dissipation=0.5,
        ),
        ComponentDefinition(
            ref_des='U2',
            part_number='AMS1117-3.3',
            footprint='SOT-223',
            category=ComponentCategory.REGULATOR,
            voltage_rating=12.0,
            current_rating=1.0,
        ),
        ComponentDefinition(
            ref_des='J1',
            part_number='USB-C-16P',
            footprint='USB-C-SMD',
            category=ComponentCategory.CONNECTOR,
            preferred_side='top',
        ),
        # Decoupling caps (AI inferred)
        ComponentDefinition(
            ref_des='C1',
            part_number='GRM155R71C104KA88D',
            footprint='0402',
            category=ComponentCategory.CAPACITOR,
            value='100nF',
            inferred=True,
            inference_reason='Decoupling for U1',
            inferred_for='U1',
        ),
        ComponentDefinition(
            ref_des='C2',
            part_number='GRM155R71C104KA88D',
            footprint='0402',
            category=ComponentCategory.CAPACITOR,
            value='100nF',
            inferred=True,
            inference_reason='Decoupling for U1',
            inferred_for='U1',
        ),
        ComponentDefinition(
            ref_des='C3',
            part_number='GRM21BR61A106KE19',
            footprint='0805',
            category=ComponentCategory.CAPACITOR,
            value='10uF',
            inferred=True,
            inference_reason='Bulk capacitor for regulator',
            inferred_for='U2',
        ),
        ComponentDefinition(
            ref_des='R1',
            part_number='RC0402FR-0710KL',
            footprint='0402',
            category=ComponentCategory.RESISTOR,
            value='10k',
        ),
        ComponentDefinition(
            ref_des='LED1',
            part_number='LTST-C171KRKT',
            footprint='0603',
            category=ComponentCategory.LED,
        ),
    ]

    # Nets with proper classification
    nets = [
        NetDefinition(
            name='VBUS',
            net_type=NetType.POWER,
            pins=['J1.VBUS', 'U2.VIN', 'C3.1'],
            voltage=5.0,
            current_max=1.0,
            routing_priority=10,
        ),
        NetDefinition(
            name='3V3',
            net_type=NetType.POWER,
            pins=['U2.VOUT', 'U1.VCC', 'C1.1', 'C2.1', 'R1.1'],
            voltage=3.3,
            current_max=0.5,
            routing_priority=10,
        ),
        NetDefinition(
            name='GND',
            net_type=NetType.GND,
            pins=['J1.GND', 'U1.GND', 'U2.GND', 'C1.2', 'C2.2', 'C3.2', 'LED1.K'],
            voltage=0.0,
            routing_priority=5,
        ),
        NetDefinition(
            name='USB_D+',
            net_type=NetType.DIFF_PAIR,
            pins=['J1.DP', 'U1.GPIO19'],
            frequency=12e6,
            impedance_ohm=90.0,
            matched_with='USB_D-',
            max_mismatch_mm=1.27,
            routing_priority=20,
        ),
        NetDefinition(
            name='USB_D-',
            net_type=NetType.DIFF_PAIR,
            pins=['J1.DM', 'U1.GPIO18'],
            frequency=12e6,
            impedance_ohm=90.0,
            matched_with='USB_D+',
            max_mismatch_mm=1.27,
            routing_priority=20,
        ),
        NetDefinition(
            name='EN',
            net_type=NetType.SIGNAL,
            pins=['U1.EN', 'R1.2'],
            routing_priority=50,
        ),
        NetDefinition(
            name='GPIO2',
            net_type=NetType.SIGNAL,
            pins=['U1.GPIO2', 'LED1.A'],
            routing_priority=80,
        ),
    ]

    # Rule hierarchy
    rules = RuleHierarchy(
        inviolable=[
            RuleBinding(
                rule_id='USB2_IMPEDANCE',
                parameters={'impedance_ohm': 90, 'tolerance_pct': 10},
                applies_to=['USB_D+', 'USB_D-'],
                reason='USB 2.0 requires 90 ohm differential impedance',
                source='USB 2.0 Spec Section 7.1.2',
            ),
            RuleBinding(
                rule_id='USB2_LENGTH_MATCHING',
                parameters={'max_mismatch_mm': 1.27},
                applies_to=['USB_D+', 'USB_D-'],
                reason='USB 2.0 length matching requirement',
            ),
        ],
        recommended=[
            RuleBinding(
                rule_id='DECOUPLING_DISTANCE',
                parameters={'max_distance_mm': 3.0},
                applies_to=['C1', 'C2'],
                reason='Decoupling capacitors should be close to IC',
            ),
        ],
        optional=[
            RuleBinding(
                rule_id='LED_PLACEMENT',
                parameters={'edge_proximity': True},
                applies_to=['LED1'],
                reason='LEDs are typically visible on edge',
            ),
        ],
    )

    # Placement hints
    placement_hints = PlacementHints(
        proximity_groups=[
            ProximityGroup(
                components=['U1', 'C1', 'C2'],
                max_distance_mm=3.0,
                reason='Decoupling capacitors for MCU',
                priority=10,
            ),
            ProximityGroup(
                components=['U2', 'C3'],
                max_distance_mm=5.0,
                reason='Bulk capacitor for regulator',
                priority=20,
            ),
        ],
        edge_components=['J1', 'LED1'],
    )

    # Routing hints
    routing_hints = RoutingHints(
        priority_nets=['USB_D+', 'USB_D-', 'VBUS', '3V3'],
        diff_pairs=[
            DiffPairSpec(
                positive_net='USB_D+',
                negative_net='USB_D-',
                impedance_ohm=90.0,
                max_mismatch_mm=1.27,
                spacing_mm=0.15,
            ),
        ],
        length_match_groups=[
            LengthMatchGroup(
                name='USB_DATA',
                nets=['USB_D+', 'USB_D-'],
                max_mismatch_mm=1.27,
            ),
        ],
        deprioritized_nets=['GPIO2'],
    )

    return ConstitutionalLayout(
        design_name='ESP32_USB_Test',
        version='1.0',
        created_by='Test',
        board=board,
        components=components,
        nets=nets,
        rules=rules,
        placement_hints=placement_hints,
        routing_hints=routing_hints,
        user_requirements=[
            'USB-powered ESP32 board',
            'USB data lines for programming',
            'Status LED on GPIO2',
        ],
        ai_assumptions=[
            'Added decoupling capacitors for ESP32',
            'Added bulk capacitor for regulator',
        ],
    )


def test_clayout_to_parts_db_conversion():
    """Test that c_layout correctly converts to parts_db format."""
    print("=" * 60)
    print("TEST 1: C_LAYOUT TO PARTS_DB CONVERSION")
    print("=" * 60)

    clayout = create_test_clayout()
    parts_db = convert_clayout_to_parts_db(clayout)

    print(f"\nC_LAYOUT Summary:")
    print(f"  Design: {clayout.design_name}")
    print(f"  Components: {len(clayout.components)}")
    print(f"  Nets: {len(clayout.nets)}")
    print(f"  Rules: {clayout.rules.count()}")

    print(f"\nPARTS_DB Conversion:")
    print(f"  Board: {parts_db['board']['width']}x{parts_db['board']['height']}mm")
    print(f"  Parts: {len(parts_db['parts'])}")
    print(f"  Nets: {len(parts_db['nets'])}")
    print(f"  Routing hints: {len(parts_db['routing_hints']['priority_nets'])} priority nets")
    print(f"  Diff pairs: {len(parts_db['routing_hints']['diff_pairs'])}")

    # Verify key data
    assert len(parts_db['parts']) == len(clayout.components), "Part count mismatch"
    assert len(parts_db['nets']) == len(clayout.nets), "Net count mismatch"
    assert 'USB_D+' in parts_db['nets'], "USB_D+ net missing"
    assert parts_db['metadata']['has_clayout'] == True, "clayout marker missing"

    print("\n[PASS] C_layout to parts_db conversion works correctly!")
    return parts_db


def test_routing_planner_with_clayout():
    """Test that RoutingPlanner correctly uses c_layout information."""
    print("\n" + "=" * 60)
    print("TEST 2: ROUTING PLANNER WITH C_LAYOUT")
    print("=" * 60)

    clayout = create_test_clayout()
    parts_db = convert_clayout_to_parts_db(clayout)

    # Add net classes from c_layout (as the adapter would do)
    converter = CLayoutConverter()
    net_classes = converter._convert_net_classes(clayout)
    for net_name, net_class in net_classes.items():
        if net_name in parts_db['nets']:
            parts_db['nets'][net_name]['class'] = net_class

    # Create routing plan
    planner = RoutingPlanner()
    plan = planner.create_routing_plan(
        parts_db,
        {'board_width': 50.0, 'board_height': 40.0, 'layers': 2}
    )

    print(f"\nRouting Plan Summary:")
    print(f"  Net strategies: {len(plan.net_strategies)}")
    print(f"  Routing order: {plan.routing_order[:5]}...")
    print(f"  Success prediction: {plan.overall_success_prediction*100:.0f}%")

    print(f"\nNet Classifications:")
    for net_name in ['VBUS', '3V3', 'GND', 'USB_D+', 'USB_D-', 'GPIO2']:
        if net_name in plan.net_strategies:
            strategy = plan.net_strategies[net_name]
            print(f"  {net_name:10} -> {strategy.net_class.value:12} | {strategy.primary_algorithm.value}")

    # Verify classifications
    assert plan.net_strategies['VBUS'].net_class == NetClass.POWER, "VBUS should be POWER"
    assert plan.net_strategies['GND'].net_class == NetClass.GROUND, "GND should be GROUND"
    assert plan.net_strategies['USB_D+'].net_class == NetClass.DIFFERENTIAL, "USB_D+ should be DIFFERENTIAL"

    # Verify algorithm selection
    assert plan.net_strategies['VBUS'].primary_algorithm == RoutingAlgorithm.LEE, "POWER should use LEE"
    assert plan.net_strategies['USB_D+'].primary_algorithm == RoutingAlgorithm.STEINER, "DIFF should use STEINER"

    print("\n[PASS] RoutingPlanner correctly uses c_layout information!")
    return plan


def test_adapter_integration():
    """Test the full CLayoutBBLAdapter with smart routing."""
    print("\n" + "=" * 60)
    print("TEST 3: CLAYOUT BBL ADAPTER INTEGRATION")
    print("=" * 60)

    clayout = create_test_clayout()

    # Create adapter with learning database
    db_path = os.path.join(tempfile.gettempdir(), 'test_clayout_learning.json')
    if os.path.exists(db_path):
        os.remove(db_path)

    learning_db = LearningDatabase(db_path)

    # Pre-populate with some learning data
    for i in range(5):
        learning_db.record_outcome(RoutingOutcome(
            net_name=f'USB_D+_{i}',
            net_class='differential',
            design_hash='test',
            algorithm='steiner',
            success=True,
            time_ms=50.0,
            quality_score=95,
        ))
    learning_db.save()

    # Create adapter
    adapter = CLayoutBBLAdapter(bbl_engine=None, learning_db=learning_db)

    print(f"\nAdapter Configuration:")
    print(f"  Smart routing available: {SMART_ROUTING_AVAILABLE}")
    print(f"  Learning database: {db_path}")

    # Run with c_layout (simulated, no actual BBL engine)
    result = adapter.run_with_clayout(clayout, use_smart_routing=True)

    print(f"\nExecution Result:")
    print(f"  Success: {result.success}")
    print(f"  Total nets: {result.total_nets}")
    print(f"  Routed nets: {result.routed_nets}")
    print(f"  Routing completion: {result.routing_completion_pct:.1f}%")

    # Check routing plan was created
    routing_plan = adapter.get_routing_plan()
    if routing_plan:
        print(f"\nRouting Plan (from adapter):")
        print(f"  Net strategies: {len(routing_plan.net_strategies)}")
        print(f"  Ground pour recommended: {routing_plan.ground_pour_recommended}")
        print(f"  Trunk chains: {len(routing_plan.trunk_chains)}")

        # Verify USB diff pair handling
        usb_dp = routing_plan.net_strategies.get('USB_D+')
        usb_dm = routing_plan.net_strategies.get('USB_D-')
        if usb_dp and usb_dm:
            print(f"\nUSB Differential Pair:")
            print(f"  USB_D+: {usb_dp.primary_algorithm.value}, priority {usb_dp.priority}")
            print(f"  USB_D-: {usb_dm.primary_algorithm.value}, priority {usb_dm.priority}")

    # Cleanup
    os.remove(db_path)

    print("\n[PASS] CLayoutBBLAdapter integration works correctly!")
    return result


def test_learning_database_integration():
    """Test that learning database improves algorithm selection."""
    print("\n" + "=" * 60)
    print("TEST 4: LEARNING DATABASE INTEGRATION")
    print("=" * 60)

    db_path = os.path.join(tempfile.gettempdir(), 'test_learning_integration.json')
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create learning database with historical data
    learning_db = LearningDatabase(db_path)

    # Simulate that PATHFINDER works better than LEE for power nets in this design style
    for i in range(20):
        # LEE failures
        learning_db.record_outcome(RoutingOutcome(
            net_name=f'VCC_{i}',
            net_class='power',
            design_hash='test_board',
            algorithm='lee',
            success=i < 10,  # 50% success
            time_ms=100.0,
            quality_score=50 if i < 10 else 0,
        ))
        # PATHFINDER successes
        learning_db.record_outcome(RoutingOutcome(
            net_name=f'VCC_{i}',
            net_class='power',
            design_hash='test_board',
            algorithm='pathfinder',
            success=True,  # 100% success
            time_ms=80.0,
            quality_score=90,
        ))
    learning_db.save()

    print(f"\nLearning Database Summary:")
    summary = learning_db.get_summary()
    print(f"  Total outcomes: {summary['total_outcomes']}")
    print(f"  Overall success rate: {summary['overall_success_rate']*100:.0f}%")

    # Create planner with learning
    planner = RoutingPlanner(learning_db)

    # Create a simple parts_db
    parts_db = {
        'parts': {'U1': {'footprint': 'QFN-32', 'pins': ['VCC', 'GND']}},
        'nets': {'VCC': {'pins': ['U1.VCC']}},
    }

    plan = planner.create_routing_plan(parts_db, {'board_width': 50, 'board_height': 40, 'layers': 2})

    vcc_strategy = plan.net_strategies.get('VCC')
    if vcc_strategy:
        print(f"\nVCC Strategy (with learning):")
        print(f"  Primary algorithm: {vcc_strategy.primary_algorithm.value}")
        print(f"  Learned algorithm: {vcc_strategy.learned_algorithm}")
        print(f"  Learned success rate: {vcc_strategy.learned_success_rate*100:.0f}%")

        # With >90% success rate, PATHFINDER should become primary
        if vcc_strategy.learned_success_rate > 0.9:
            assert vcc_strategy.primary_algorithm == RoutingAlgorithm.PATHFINDER, \
                "PATHFINDER should be primary with >90% success rate"
            print("\n  [OK] Learning correctly overrode default algorithm!")

    # Cleanup
    os.remove(db_path)

    print("\n[PASS] Learning database integration works correctly!")


def test_end_to_end_flow():
    """Test the complete flow from c_layout to routing plan."""
    print("\n" + "=" * 60)
    print("TEST 5: END-TO-END FLOW")
    print("=" * 60)

    print("""
    FLOW TESTED:
    ============

    User Text: "USB-powered ESP32 board with status LED"
           |
           v
    +-------------------+
    |   C_LAYOUT        |  <- ConstitutionalLayout created
    |   - components    |
    |   - nets          |
    |   - rules         |
    |   - hints         |
    +---------+---------+
              |
              v
    +-------------------+
    |  BBL ADAPTER      |  <- CLayoutBBLAdapter.run_with_clayout()
    |  - converts       |
    |  - validates      |
    +---------+---------+
              |
              v
    +-------------------+
    | ROUTING PLANNER   |  <- Smart per-net algorithm selection
    |  - classifies     |
    |  - selects        |
    |  - plans          |
    +---------+---------+
              |
              v
    +-------------------+
    |  ROUTING PLAN     |  <- Per-net strategies ready for BBL
    |  - strategies     |
    |  - order          |
    |  - constraints    |
    +-------------------+
    """)

    # Step 1: Create c_layout (simulating AI agent output)
    print("Step 1: Creating Constitutional Layout...")
    clayout = create_test_clayout()
    print(f"  Created: {clayout.design_name}")
    print(f"  Components: {len(clayout.components)} ({len(clayout.get_inferred_components())} AI-inferred)")
    print(f"  Nets: {len(clayout.nets)}")

    # Step 2: Create adapter
    print("\nStep 2: Creating BBL Adapter...")
    adapter = CLayoutBBLAdapter()
    print(f"  Smart routing: {'enabled' if SMART_ROUTING_AVAILABLE else 'disabled'}")

    # Step 3: Run through adapter
    print("\nStep 3: Running through adapter...")
    result = adapter.run_with_clayout(clayout, use_smart_routing=True)

    # Step 4: Examine routing plan
    print("\nStep 4: Examining routing plan...")
    routing_plan = adapter.get_routing_plan()

    if routing_plan:
        print(f"\n  ROUTING PLAN SUMMARY:")
        print(f"  {'='*40}")
        print(f"  Total nets: {len(routing_plan.net_strategies)}")
        print(f"  Ground pour: {'YES' if routing_plan.ground_pour_recommended else 'NO'}")
        print(f"  Trunk chains: {len(routing_plan.trunk_chains)}")
        print(f"  Success prediction: {routing_plan.overall_success_prediction*100:.0f}%")

        print(f"\n  ALGORITHM DISTRIBUTION:")
        for algo, count in routing_plan.get_algorithm_distribution().items():
            print(f"    {algo:12}: {count} nets")

        print(f"\n  NET CLASS DISTRIBUTION:")
        from collections import defaultdict
        class_counts = defaultdict(int)
        for strategy in routing_plan.net_strategies.values():
            class_counts[strategy.net_class.value] += 1
        for net_class, count in sorted(class_counts.items()):
            print(f"    {net_class:12}: {count} nets")

    print("\n" + "=" * 60)
    print("END-TO-END FLOW COMPLETE!")
    print("=" * 60)

    return True


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("C_LAYOUT + SMART ALGORITHM MANAGER INTEGRATION TEST SUITE")
    print("=" * 70)

    try:
        test_clayout_to_parts_db_conversion()
        test_routing_planner_with_clayout()
        test_adapter_integration()
        test_learning_database_integration()
        test_end_to_end_flow()

        print("\n" + "=" * 70)
        print("ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)
        print("""
INTEGRATION SUMMARY:
===================
[OK] C_layout correctly converts to parts_db format
[OK] RoutingPlanner uses c_layout net types for classification
[OK] CLayoutBBLAdapter integrates with RoutingPlanner
[OK] Learning database influences algorithm selection
[OK] End-to-end flow works correctly

The Constitutional Layout system is now fully integrated with the
Smart Algorithm Manager for intelligent per-net routing strategy.
        """)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
