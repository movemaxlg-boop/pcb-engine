"""
Test script to verify all bug fixes work correctly.
Run with: python test_bug_fixes.py
"""

import sys
sys.path.insert(0, '..')

def test_bug01_origin_defaults():
    """BUG-01: Origin defaults should be 0.0, not 100.0"""
    from pcb_engine.placement_piston import PlacementConfig
    config = PlacementConfig()
    assert config.origin_x == 0.0, f"Expected 0.0, got {config.origin_x}"
    assert config.origin_y == 0.0, f"Expected 0.0, got {config.origin_y}"
    print("BUG-01 FIX VERIFIED: Origin defaults are 0.0")


def test_bug03_position_type():
    """BUG-03: Position should support both tuple and attribute access"""
    from pcb_engine.common_types import Position, normalize_position, get_xy

    # Test Position class
    pos = Position(10.5, 20.3)
    assert pos.x == 10.5, "Attribute access failed"
    assert pos[0] == 10.5, "Index access failed"
    x, y = pos  # Unpacking
    assert x == 10.5 and y == 20.3, "Unpacking failed"

    # Test normalize_position
    pos2 = normalize_position((30.0, 40.0))
    assert pos2.x == 30.0, "Tuple conversion failed"

    pos3 = normalize_position({'x': 50.0, 'y': 60.0})
    assert pos3.x == 50.0, "Dict conversion failed"

    # Test get_xy
    x, y = get_xy((10, 20))
    assert x == 10 and y == 20, "get_xy tuple failed"

    x, y = get_xy(Position(30, 40))
    assert x == 30 and y == 40, "get_xy Position failed"

    print("BUG-03 FIX VERIFIED: Position type supports all access patterns")


def test_bug04_routing_types_unified():
    """BUG-04: Via should have from_layer and to_layer in both routing modules"""
    from pcb_engine.routing_types import Via, TrackSegment, Route

    via = Via(position=(10.0, 20.0), net='GND')
    assert hasattr(via, 'from_layer'), "Via missing from_layer"
    assert hasattr(via, 'to_layer'), "Via missing to_layer"
    assert via.from_layer == 'F.Cu', f"Expected 'F.Cu', got {via.from_layer}"
    assert via.to_layer == 'B.Cu', f"Expected 'B.Cu', got {via.to_layer}"

    # Test Route has bend_count
    route = Route(net='test')
    assert hasattr(route, 'bend_count'), "Route missing bend_count property"

    print("BUG-04 FIX VERIFIED: Unified routing types work correctly")


def test_bug06_pins_naming():
    """BUG-06: get_pins should handle all pin formats"""
    from pcb_engine.common_types import get_pins, get_pin_net

    # Test list format
    part1 = {'pins': [{'number': '1', 'net': 'VCC'}, {'number': '2', 'net': 'GND'}]}
    pins = get_pins(part1)
    assert len(pins) == 2, "List format failed"
    assert get_pin_net(part1, '1') == 'VCC', "get_pin_net list format failed"

    # Test used_pins format
    part2 = {'used_pins': [{'number': '1', 'net': '3V3'}]}
    pins = get_pins(part2)
    assert len(pins) == 1, "used_pins format failed"
    assert get_pin_net(part2, '1') == '3V3', "get_pin_net used_pins failed"

    # Test dict format (legacy)
    part3 = {'pins': {'1': {'net': 'PWR'}, '2': {'net': 'GND'}}}
    pins = get_pins(part3)
    assert len(pins) == 2, "Dict format failed"

    print("BUG-06 FIX VERIFIED: All pin formats handled correctly")


def test_bug13_grid_initialization():
    """BUG-13: Routing grids should be initialized, not None"""
    from pcb_engine.routing_types import RoutingConfig
    from pcb_engine.routing_piston import RoutingPiston

    config = RoutingConfig(board_width=50.0, board_height=40.0, grid_size=0.5)
    piston = RoutingPiston(config)

    assert piston.fcu_grid is not None, "fcu_grid is None"
    assert piston.bcu_grid is not None, "bcu_grid is None"
    assert len(piston.fcu_grid) > 0, "fcu_grid is empty"

    print("BUG-13 FIX VERIFIED: Grids are initialized in __init__")


def test_imports_work():
    """Verify all main imports work"""
    from pcb_engine import (
        PCBEngine, EngineConfig,
        Position, get_xy, get_pins, get_pin_net,
        TrackSegment, Via, Route, RoutingConfig
    )
    print("All imports work correctly")


def test_drc_violation_detection():
    """Test that DRC correctly catches violations"""
    from pcb_engine import DRCPiston, DRCConfig, DRCRules
    from pcb_engine.routing_types import TrackSegment, Route, Via

    # Test 1: Track too narrow
    parts_db = {'parts': {}, 'nets': {}}
    placement = {}
    routes = {
        'TEST': Route(
            net='TEST',
            segments=[
                TrackSegment(
                    start=(10.0, 10.0),
                    end=(20.0, 10.0),
                    layer='F.Cu',
                    width=0.05,  # Too narrow! (min is 0.15)
                    net='TEST'
                )
            ],
            success=True
        )
    }

    rules = DRCRules(min_track_width=0.15)
    config = DRCConfig(rules=rules, check_track_widths=True)
    drc = DRCPiston(config)
    result = drc.check(parts_db, placement, routes, [])
    assert result.error_count > 0, "Track width violation not detected"

    # Test 2: Via drill too small
    vias = [Via(position=(15.0, 15.0), net='TEST', diameter=0.6, drill=0.2)]
    rules = DRCRules(min_via_drill=0.3)
    config = DRCConfig(rules=rules, check_vias=True, check_holes=True)
    drc = DRCPiston(config)
    result = drc.check({'parts': {}, 'nets': {}}, {}, {}, vias)
    assert result.error_count > 0, "Via drill violation not detected"

    # Test 3: Valid design should pass
    parts_db = {
        'parts': {
            'R1': {
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            }
        },
        'nets': {'VCC': {'pins': [('R1', '1')]}, 'GND': {'pins': [('R1', '2')]}}
    }
    placement = {'R1': (25.0, 20.0)}
    config = DRCConfig(rules=DRCRules(), board_width=50.0, board_height=40.0)
    drc = DRCPiston(config)
    result = drc.check(parts_db, placement, {}, [])
    assert result.passed, "Valid design should pass DRC"

    print("DRC VIOLATION DETECTION VERIFIED: All tests passed")


def run_all_tests():
    """Run all bug fix verification tests"""
    print("=" * 60)
    print("RUNNING BUG FIX VERIFICATION TESTS")
    print("=" * 60)

    tests = [
        test_bug01_origin_defaults,
        test_bug03_position_type,
        test_bug04_routing_types_unified,
        test_bug06_pins_naming,
        test_bug13_grid_initialization,
        test_imports_work,
        test_drc_violation_detection,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
