"""
Full PCB Engine Workflow Test
==============================

Tests all 18 pistons and reports which ones work vs which have issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple


def create_test_parts_db() -> Dict:
    """Create a realistic test parts database"""
    return {
        'parts': {
            'U1': {
                'value': 'AP2112K-3.3',
                'footprint': 'SOT-23-5',
                'description': 'LDO Voltage Regulator 3.3V',
                'pins': [
                    {'number': '1', 'name': 'VIN', 'net': 'VIN', 'offset': (-0.95, -0.8)},
                    {'number': '2', 'name': 'GND', 'net': 'GND', 'offset': (-0.95, 0.8)},
                    {'number': '3', 'name': 'EN', 'net': 'VIN', 'offset': (0.95, 0.8)},
                    {'number': '4', 'name': 'NC', 'net': '', 'offset': (0.95, 0)},
                    {'number': '5', 'name': 'VOUT', 'net': '3V3', 'offset': (0.95, -0.8)},
                ]
            },
            'C1': {
                'value': '10uF',
                'footprint': 'C_0805_2012Metric',
                'description': 'Input capacitor',
                'pins': [
                    {'number': '1', 'name': 'P1', 'net': 'VIN', 'offset': (-0.95, 0)},
                    {'number': '2', 'name': 'P2', 'net': 'GND', 'offset': (0.95, 0)},
                ]
            },
            'C2': {
                'value': '10uF',
                'footprint': 'C_0805_2012Metric',
                'description': 'Output capacitor',
                'pins': [
                    {'number': '1', 'name': 'P1', 'net': '3V3', 'offset': (-0.95, 0)},
                    {'number': '2', 'name': 'P2', 'net': 'GND', 'offset': (0.95, 0)},
                ]
            },
            'R1': {
                'value': '10k',
                'footprint': 'R_0603_1608Metric',
                'description': 'Pull-up resistor',
                'pins': [
                    {'number': '1', 'name': 'P1', 'net': '3V3', 'offset': (-0.8, 0)},
                    {'number': '2', 'name': 'P2', 'net': 'SDA', 'offset': (0.8, 0)},
                ]
            },
            'R2': {
                'value': '10k',
                'footprint': 'R_0603_1608Metric',
                'description': 'Pull-up resistor',
                'pins': [
                    {'number': '1', 'name': 'P1', 'net': '3V3', 'offset': (-0.8, 0)},
                    {'number': '2', 'name': 'P2', 'net': 'SCL', 'offset': (0.8, 0)},
                ]
            },
            'LED1': {
                'value': 'RED',
                'footprint': 'LED_0805_2012Metric',
                'description': 'Power indicator',
                'pins': [
                    {'number': '1', 'name': 'A', 'net': '3V3', 'offset': (-0.95, 0)},
                    {'number': '2', 'name': 'K', 'net': 'LED_GND', 'offset': (0.95, 0)},
                ]
            },
            'R3': {
                'value': '330',
                'footprint': 'R_0603_1608Metric',
                'description': 'LED current limiting',
                'pins': [
                    {'number': '1', 'name': 'P1', 'net': 'LED_GND', 'offset': (-0.8, 0)},
                    {'number': '2', 'name': 'P2', 'net': 'GND', 'offset': (0.8, 0)},
                ]
            },
        },
        'nets': {
            'VIN': {'pins': [('U1', '1'), ('U1', '3'), ('C1', '1')]},
            'GND': {'pins': [('U1', '2'), ('C1', '2'), ('C2', '2'), ('R3', '2')]},
            '3V3': {'pins': [('U1', '5'), ('C2', '1'), ('R1', '1'), ('R2', '1'), ('LED1', '1')]},
            'SDA': {'pins': [('R1', '2')]},
            'SCL': {'pins': [('R2', '2')]},
            'LED_GND': {'pins': [('LED1', '2'), ('R3', '1')]},
        }
    }


def test_piston(name: str, test_func) -> Tuple[bool, str]:
    """Test a single piston and return result"""
    try:
        result = test_func()
        if result:
            return True, "OK"
        else:
            return False, "Returned None or False"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def run_full_workflow():
    """Run and report on all pistons"""
    print("\n" + "=" * 70)
    print("PCB ENGINE - FULL WORKFLOW TEST")
    print("=" * 70)

    parts_db = create_test_parts_db()
    results = []

    # =========================================================================
    # CORE PISTONS (9)
    # =========================================================================
    print("\n--- CORE PISTONS ---")

    # 1. Parts Piston
    def test_parts():
        from pcb_engine import PartsPiston, PartsConfig
        piston = PartsPiston(PartsConfig())
        return piston.analyze(parts_db)

    ok, msg = test_piston("PartsPiston", test_parts)
    results.append(("PartsPiston", ok, msg))
    print(f"  1. PartsPiston:       {'PASS' if ok else 'FAIL'} - {msg}")

    # 2. Order Piston
    def test_order():
        from pcb_engine import OrderPiston, OrderConfig
        piston = OrderPiston(OrderConfig())
        return piston.order(parts_db)

    ok, msg = test_piston("OrderPiston", test_order)
    results.append(("OrderPiston", ok, msg))
    print(f"  2. OrderPiston:       {'PASS' if ok else 'FAIL'} - {msg}")

    # 3. Placement Piston
    placement = {}
    def test_placement():
        from pcb_engine import PlacementPiston
        from pcb_engine.placement_piston import PlacementConfig
        config = PlacementConfig(board_width=50.0, board_height=40.0)
        piston = PlacementPiston(config)
        result = piston.place(parts_db, {})
        if result and hasattr(result, 'positions'):
            nonlocal placement
            placement = result.positions
            return result
        return None

    ok, msg = test_piston("PlacementPiston", test_placement)
    results.append(("PlacementPiston", ok, msg))
    print(f"  3. PlacementPiston:   {'PASS' if ok else 'FAIL'} - {msg}")

    # 4. Escape Piston
    escapes = {}
    def test_escape():
        from pcb_engine import EscapePiston, EscapeConfig
        config = EscapeConfig()  # Use defaults - no board_width param
        piston = EscapePiston(config)
        result = piston.plan(parts_db, placement)
        if result and hasattr(result, 'escapes'):
            nonlocal escapes
            escapes = result.escapes
            return result
        return result

    ok, msg = test_piston("EscapePiston", test_escape)
    results.append(("EscapePiston", ok, msg))
    print(f"  4. EscapePiston:      {'PASS' if ok else 'FAIL'} - {msg}")

    # 5. Routing Piston
    routes = {}
    vias = []
    def test_routing():
        from pcb_engine import RoutingPiston, RoutingConfig
        config = RoutingConfig(board_width=50.0, board_height=40.0, grid_size=0.15)
        piston = RoutingPiston(config)
        net_order = list(parts_db.get('nets', {}).keys())
        result = piston.route(parts_db, escapes, placement, net_order)
        if result and hasattr(result, 'routes'):
            nonlocal routes, vias
            routes = result.routes
            # Collect vias from routes
            for net, route in routes.items():
                if hasattr(route, 'vias'):
                    vias.extend(route.vias)
            return result
        return result

    ok, msg = test_piston("RoutingPiston", test_routing)
    results.append(("RoutingPiston", ok, msg))
    print(f"  5. RoutingPiston:     {'PASS' if ok else 'FAIL'} - {msg}")

    # 6. Optimization Piston
    def test_optimization():
        from pcb_engine import OptimizationPiston, OptimizationConfig
        config = OptimizationConfig()
        piston = OptimizationPiston(config)
        return piston.optimize(routes, vias)

    ok, msg = test_piston("OptimizationPiston", test_optimization)
    results.append(("OptimizationPiston", ok, msg))
    print(f"  6. OptimizationPiston: {'PASS' if ok else 'FAIL'} - {msg}")

    # 7. Silkscreen Piston
    silkscreen = None
    def test_silkscreen():
        from pcb_engine import SilkscreenPiston, SilkscreenConfig
        config = SilkscreenConfig()
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement, routes)
        nonlocal silkscreen
        silkscreen = result
        return result

    ok, msg = test_piston("SilkscreenPiston", test_silkscreen)
    results.append(("SilkscreenPiston", ok, msg))
    print(f"  7. SilkscreenPiston:  {'PASS' if ok else 'FAIL'} - {msg}")

    # 8. DRC Piston
    def test_drc():
        from pcb_engine import DRCPiston, DRCConfig, DRCRules
        rules = DRCRules(min_track_width=0.15, min_clearance=0.15)
        config = DRCConfig(rules=rules, board_width=50.0, board_height=40.0)
        piston = DRCPiston(config)
        return piston.check(parts_db, placement, routes, vias)

    ok, msg = test_piston("DRCPiston", test_drc)
    results.append(("DRCPiston", ok, msg))
    print(f"  8. DRCPiston:         {'PASS' if ok else 'FAIL'} - {msg}")

    # 9. Output Piston
    def test_output():
        from pcb_engine import OutputPiston, OutputConfig
        import tempfile
        config = OutputConfig(output_dir=tempfile.mkdtemp(), generate_gerbers=False)
        piston = OutputPiston(config)
        return piston.generate(parts_db, placement, routes, vias, silkscreen)

    ok, msg = test_piston("OutputPiston", test_output)
    results.append(("OutputPiston", ok, msg))
    print(f"  9. OutputPiston:      {'PASS' if ok else 'FAIL'} - {msg}")

    # =========================================================================
    # ANALYSIS PISTONS (5)
    # =========================================================================
    print("\n--- ANALYSIS PISTONS ---")

    # 10. Stackup Piston
    def test_stackup():
        from pcb_engine import StackupPiston, StackupConfig
        config = StackupConfig()
        piston = StackupPiston(config)
        return piston.analyze()

    ok, msg = test_piston("StackupPiston", test_stackup)
    results.append(("StackupPiston", ok, msg))
    print(f" 10. StackupPiston:     {'PASS' if ok else 'FAIL'} - {msg}")

    # 11. Thermal Piston
    def test_thermal():
        from pcb_engine import ThermalPiston, ThermalConfig
        config = ThermalConfig()
        piston = ThermalPiston(config)
        return piston.analyze(parts_db, placement)

    ok, msg = test_piston("ThermalPiston", test_thermal)
    results.append(("ThermalPiston", ok, msg))
    print(f" 11. ThermalPiston:     {'PASS' if ok else 'FAIL'} - {msg}")

    # 12. PDN Piston
    def test_pdn():
        from pcb_engine import PDNPiston, PDNConfig
        config = PDNConfig()
        piston = PDNPiston(config)
        return piston.analyze(parts_db, routes)

    ok, msg = test_piston("PDNPiston", test_pdn)
    results.append(("PDNPiston", ok, msg))
    print(f" 12. PDNPiston:         {'PASS' if ok else 'FAIL'} - {msg}")

    # 13. Signal Integrity Piston
    def test_signal_integrity():
        from pcb_engine import SignalIntegrityPiston, SIConfig
        config = SIConfig()
        piston = SignalIntegrityPiston(config)
        return piston.analyze(routes)

    ok, msg = test_piston("SignalIntegrityPiston", test_signal_integrity)
    results.append(("SignalIntegrityPiston", ok, msg))
    print(f" 13. SignalIntegrityPiston: {'PASS' if ok else 'FAIL'} - {msg}")

    # 14. Netlist Piston
    def test_netlist():
        from pcb_engine import NetlistPiston
        piston = NetlistPiston()
        return piston.extract(parts_db)

    ok, msg = test_piston("NetlistPiston", test_netlist)
    results.append(("NetlistPiston", ok, msg))
    print(f" 14. NetlistPiston:     {'PASS' if ok else 'FAIL'} - {msg}")

    # =========================================================================
    # ADVANCED PISTONS (3)
    # =========================================================================
    print("\n--- ADVANCED PISTONS ---")

    # 15. Topological Router Piston
    def test_topological():
        from pcb_engine import TopologicalRouterPiston
        piston = TopologicalRouterPiston()
        return piston.route(parts_db, placement)

    ok, msg = test_piston("TopologicalRouterPiston", test_topological)
    results.append(("TopologicalRouterPiston", ok, msg))
    print(f" 15. TopologicalRouterPiston: {'PASS' if ok else 'FAIL'} - {msg}")

    # 16. 3D Visualization Piston
    def test_3d():
        from pcb_engine import Visualization3DPiston
        piston = Visualization3DPiston()
        return piston.generate(parts_db, placement, routes)

    ok, msg = test_piston("Visualization3DPiston", test_3d)
    results.append(("Visualization3DPiston", ok, msg))
    print(f" 16. Visualization3DPiston: {'PASS' if ok else 'FAIL'} - {msg}")

    # 17. BOM Optimizer Piston
    def test_bom():
        from pcb_engine import BOMOptimizerPiston
        piston = BOMOptimizerPiston()
        return piston.optimize(parts_db)

    ok, msg = test_piston("BOMOptimizerPiston", test_bom)
    results.append(("BOMOptimizerPiston", ok, msg))
    print(f" 17. BOMOptimizerPiston: {'PASS' if ok else 'FAIL'} - {msg}")

    # =========================================================================
    # LEARNING PISTON (1)
    # =========================================================================
    print("\n--- LEARNING PISTON ---")

    # 18. Learning Piston
    def test_learning():
        from pcb_engine import LearningPiston, LearningMode
        piston = LearningPiston()  # No mode in __init__, use observe() method
        return piston.observe({'parts_db': parts_db, 'placement': placement, 'routes': routes})

    ok, msg = test_piston("LearningPiston", test_learning)
    results.append(("LearningPiston", ok, msg))
    print(f" 18. LearningPiston:    {'PASS' if ok else 'FAIL'} - {msg}")

    # =========================================================================
    # ORCHESTRATION
    # =========================================================================
    print("\n--- ORCHESTRATION ---")

    # Piston Orchestrator
    def test_orchestrator():
        from pcb_engine import PistonOrchestrator
        orchestrator = PistonOrchestrator()
        return orchestrator.select_pistons(parts_db)

    ok, msg = test_piston("PistonOrchestrator", test_orchestrator)
    results.append(("PistonOrchestrator", ok, msg))
    print(f" 19. PistonOrchestrator: {'PASS' if ok else 'FAIL'} - {msg}")

    # Workflow Reporter
    def test_reporter():
        from pcb_engine import WorkflowReporter
        reporter = WorkflowReporter()
        return reporter.report({'parts_db': parts_db, 'placement': placement})

    ok, msg = test_piston("WorkflowReporter", test_reporter)
    results.append(("WorkflowReporter", ok, msg))
    print(f" 20. WorkflowReporter:  {'PASS' if ok else 'FAIL'} - {msg}")

    # Circuit AI
    def test_circuit_ai():
        from pcb_engine import CircuitAI
        ai = CircuitAI()
        return ai.suggest(parts_db)

    ok, msg = test_piston("CircuitAI", test_circuit_ai)
    results.append(("CircuitAI", ok, msg))
    print(f" 21. CircuitAI:         {'PASS' if ok else 'FAIL'} - {msg}")

    # =========================================================================
    # GRID CALCULATOR
    # =========================================================================
    print("\n--- UTILITIES ---")

    def test_grid_calculator():
        from pcb_engine import (
            calculate_optimal_grid_size, print_grid_analysis,
            PerformanceTier, suggest_performance_tier
        )
        grid, analysis = calculate_optimal_grid_size(parts_db)
        tier = suggest_performance_tier(parts_db)
        return grid > 0 and tier is not None

    ok, msg = test_piston("GridCalculator", test_grid_calculator)
    results.append(("GridCalculator", ok, msg))
    print(f" 22. GridCalculator:    {'PASS' if ok else 'FAIL'} - {msg}")

    # =========================================================================
    # MAIN ENGINE
    # =========================================================================
    print("\n--- MAIN ENGINE ---")

    def test_pcb_engine():
        from pcb_engine import PCBEngine, EngineConfig
        import tempfile
        config = EngineConfig(
            board_width=50.0,
            board_height=40.0,
            verbose=False,
            output_dir=tempfile.mkdtemp()
        )
        engine = PCBEngine(config)
        # Just test initialization, not full run (that's tested above)
        return engine is not None

    ok, msg = test_piston("PCBEngine", test_pcb_engine)
    results.append(("PCBEngine", ok, msg))
    print(f" 23. PCBEngine:         {'PASS' if ok else 'FAIL'} - {msg}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\nTotal: {len(results)} components tested")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if passed:
        print(f"\nWORKING ({len(passed)}):")
        for name, _, _ in passed:
            print(f"  + {name}")

    if failed:
        print(f"\nNOT WORKING ({len(failed)}):")
        for name, _, msg in failed:
            print(f"  - {name}: {msg}")

    print("\n" + "=" * 70)

    return len(failed) == 0


if __name__ == '__main__':
    success = run_full_workflow()
    sys.exit(0 if success else 1)
