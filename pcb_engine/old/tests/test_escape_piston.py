#!/usr/bin/env python3
"""
Unit Tests for EscapePiston
===========================

Tests all functionality of the EscapePiston:
- Pin array creation and classification
- Escape strategies (dog-bone, ring-based, MMCF, etc.)
- Via placement and layer assignment
- Escape path generation
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.escape_piston import (
    EscapePiston, EscapeConfig, EscapeResult,
    PinArray, Pin, Via, EscapeTrace, EscapePath,
    PackageType, EscapeStrategy, ViaType, EscapeDirection, PinLocation,
    escape_bga, escape_qfn, QFNThermalEscape, PeripheralFanout
)


class TestEscapePistonInit(unittest.TestCase):
    """Test EscapePiston initialization."""

    def test_default_init(self):
        """Test default initialization."""
        piston = EscapePiston()
        self.assertIsNotNone(piston)
        self.assertIsNotNone(piston.config)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = EscapeConfig(
            via_drill=0.2,
            via_pad=0.4,
            trace_width=0.1
        )
        piston = EscapePiston(config)
        self.assertEqual(piston.config.via_drill, 0.2)
        self.assertEqual(piston.config.via_pad, 0.4)
        self.assertEqual(piston.config.trace_width, 0.1)


class TestEscapeConfig(unittest.TestCase):
    """Test EscapeConfig defaults and customization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EscapeConfig()
        self.assertEqual(config.via_drill, 0.3)
        self.assertEqual(config.via_pad, 0.6)
        self.assertEqual(config.trace_width, 0.15)
        self.assertEqual(config.available_layers, 4)
        self.assertEqual(config.strategy, EscapeStrategy.DOG_BONE)

    def test_strategy_options(self):
        """Test different strategy configurations."""
        for strategy in EscapeStrategy:
            config = EscapeConfig(strategy=strategy)
            self.assertEqual(config.strategy, strategy)


class TestPinArray(unittest.TestCase):
    """Test PinArray creation and manipulation."""

    def test_create_pin_array(self):
        """Test creating a pin array."""
        pins = [
            Pin(id="A1", row=0, col=0, x=0, y=0, net="VCC"),
            Pin(id="A2", row=0, col=1, x=0.8, y=0, net="GND"),
        ]
        array = PinArray(
            package_type=PackageType.BGA,
            rows=2,
            cols=2,
            pitch=0.8,
            pins=pins
        )
        self.assertEqual(array.package_type, PackageType.BGA)
        self.assertEqual(array.rows, 2)
        self.assertEqual(array.cols, 2)
        self.assertEqual(len(array.pins), 2)

    def test_staggered_array(self):
        """Test staggered pin array."""
        array = PinArray(
            package_type=PackageType.BGA,
            rows=4,
            cols=4,
            pitch=0.5,
            is_staggered=True
        )
        self.assertTrue(array.is_staggered)


class TestPinClassification(unittest.TestCase):
    """Test pin location classification."""

    def setUp(self):
        """Create a 4x4 BGA array for testing."""
        self.pins = []
        for r in range(4):
            for c in range(4):
                pin = Pin(
                    id=f"{chr(ord('A')+r)}{c+1}",
                    row=r,
                    col=c,
                    x=c * 0.8,
                    y=r * 0.8,
                    net=f"NET_{r}_{c}"
                )
                self.pins.append(pin)

        self.array = PinArray(
            package_type=PackageType.BGA,
            rows=4,
            cols=4,
            pitch=0.8,
            pins=self.pins
        )

    def test_corner_pin_detection(self):
        """Test that corner pins are classified correctly."""
        piston = EscapePiston()
        piston.pin_array = self.array
        piston._classify_pin_locations()

        # Check corners
        corner_pins = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for pin in self.array.pins:
            if (pin.row, pin.col) in corner_pins:
                self.assertEqual(pin.location, PinLocation.CORNER,
                               f"Pin at ({pin.row}, {pin.col}) should be CORNER")

    def test_edge_pin_detection(self):
        """Test that edge pins are classified correctly."""
        piston = EscapePiston()
        piston.pin_array = self.array
        piston._classify_pin_locations()

        # Check edges (excluding corners)
        edge_positions = [(0, 1), (0, 2), (3, 1), (3, 2),
                         (1, 0), (2, 0), (1, 3), (2, 3)]
        for pin in self.array.pins:
            if (pin.row, pin.col) in edge_positions:
                self.assertEqual(pin.location, PinLocation.EDGE,
                               f"Pin at ({pin.row}, {pin.col}) should be EDGE")


class TestEscapeStrategies(unittest.TestCase):
    """Test different escape routing strategies."""

    def setUp(self):
        """Create test pin array."""
        self.pins = []
        for r in range(4):
            for c in range(4):
                pin = Pin(
                    id=f"{chr(ord('A')+r)}{c+1}",
                    row=r,
                    col=c,
                    x=c * 0.8,
                    y=r * 0.8,
                    net=f"NET_{r}_{c}"
                )
                self.pins.append(pin)

        self.array = PinArray(
            package_type=PackageType.BGA,
            rows=4,
            cols=4,
            pitch=0.8,
            pins=self.pins
        )

    def test_dog_bone_strategy(self):
        """Test dog-bone escape strategy."""
        config = EscapeConfig(strategy=EscapeStrategy.DOG_BONE)
        piston = EscapePiston(config)
        result = piston.escape(self.array)

        self.assertIsInstance(result, EscapeResult)
        self.assertEqual(result.strategy_used, EscapeStrategy.DOG_BONE)
        self.assertGreater(len(result.paths), 0)

    def test_ring_based_strategy(self):
        """Test ring-based escape strategy."""
        config = EscapeConfig(strategy=EscapeStrategy.RING_BASED)
        piston = EscapePiston(config)
        result = piston.escape(self.array)

        self.assertIsInstance(result, EscapeResult)
        self.assertEqual(result.strategy_used, EscapeStrategy.RING_BASED)

    def test_mmcf_strategy(self):
        """Test MMCF escape strategy."""
        config = EscapeConfig(strategy=EscapeStrategy.ORDERED_MMCF)
        piston = EscapePiston(config)
        result = piston.escape(self.array)

        self.assertIsInstance(result, EscapeResult)
        self.assertEqual(result.strategy_used, EscapeStrategy.ORDERED_MMCF)


class TestEscapeResult(unittest.TestCase):
    """Test EscapeResult structure."""

    def test_result_structure(self):
        """Test that result has all expected fields."""
        result = EscapeResult()
        self.assertIsInstance(result.paths, list)
        self.assertIsInstance(result.success_rate, float)
        self.assertIsInstance(result.total_vias, int)
        self.assertIsInstance(result.layers_used, set)
        self.assertIsInstance(result.total_wire_length, float)

    def test_success_rate_calculation(self):
        """Test success rate is calculated correctly."""
        pins = [
            Pin(id="A1", row=0, col=0, x=0, y=0, net="VCC"),
            Pin(id="A2", row=0, col=1, x=0.8, y=0, net="GND"),
        ]
        array = PinArray(
            package_type=PackageType.BGA,
            rows=2,
            cols=2,
            pitch=0.8,
            pins=pins
        )

        piston = EscapePiston()
        result = piston.escape(array)

        # Success rate should be between 0 and 1
        self.assertGreaterEqual(result.success_rate, 0.0)
        self.assertLessEqual(result.success_rate, 1.0)


class TestViaPlacement(unittest.TestCase):
    """Test via placement in escape routing."""

    def test_via_creation(self):
        """Test via object creation."""
        via = Via(
            x=1.0,
            y=2.0,
            drill_diameter=0.3,
            pad_diameter=0.6,
            via_type=ViaType.THROUGH_HOLE,
            start_layer=1,
            end_layer=4,
            net="GND"
        )
        self.assertEqual(via.x, 1.0)
        self.assertEqual(via.y, 2.0)
        self.assertEqual(via.via_type, ViaType.THROUGH_HOLE)

    def test_via_type_selection(self):
        """Test via type is selected based on layer transition."""
        config = EscapeConfig(
            via_types_available=[ViaType.THROUGH_HOLE, ViaType.MICRO_VIA]
        )
        piston = EscapePiston(config)

        # Test via selection for layer 1 to 2
        via_type = piston._select_via_type(1, 2)
        # Should prefer micro-via for single layer span
        self.assertEqual(via_type, ViaType.MICRO_VIA)


class TestEscapePath(unittest.TestCase):
    """Test escape path generation."""

    def test_path_creation(self):
        """Test escape path object creation."""
        pin = Pin(id="A1", row=0, col=0, x=0, y=0, net="VCC")
        path = EscapePath(pin=pin)

        self.assertEqual(path.pin.id, "A1")
        self.assertEqual(len(path.traces), 0)
        self.assertEqual(len(path.vias), 0)
        self.assertEqual(path.total_length, 0.0)
        self.assertFalse(path.escaped)

    def test_path_with_trace(self):
        """Test escape path with trace."""
        pin = Pin(id="A1", row=0, col=0, x=0, y=0, net="VCC")
        trace = EscapeTrace(
            start_x=0, start_y=0,
            end_x=0.5, end_y=0.5,
            width=0.15,
            layer=1,
            net="VCC"
        )
        path = EscapePath(
            pin=pin,
            traces=[trace],
            total_length=0.707,
            escaped=True
        )

        self.assertEqual(len(path.traces), 1)
        self.assertTrue(path.escaped)


class TestPlanAPI(unittest.TestCase):
    """Test the plan() API for integration with engine."""

    def test_plan_with_parts_db(self):
        """Test plan method with parts database."""
        parts_db = {
            'parts': {
                'U1': {
                    'value': 'ESP32',
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (0, 0)},
                        {'number': '2', 'net': 'GND', 'offset': (0, 1)},
                    ]
                }
            }
        }

        class MockPos:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        placement = {'U1': MockPos(10, 10)}

        piston = EscapePiston()
        result = piston.plan(parts_db, placement)

        self.assertIsInstance(result, EscapeResult)

    def test_plan_empty_parts(self):
        """Test plan with empty parts database."""
        parts_db = {'parts': {}}
        placement = {}

        piston = EscapePiston()
        result = piston.plan(parts_db, placement)

        self.assertIsInstance(result, EscapeResult)
        self.assertEqual(len(result.paths), 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for BGA and QFN escape."""

    def test_escape_bga(self):
        """Test escape_bga convenience function."""
        nets = {
            "A1": "VCC",
            "A2": "GND",
            "B1": "SIG1",
            "B2": "SIG2",
        }
        result = escape_bga(2, 2, 0.8, nets)

        self.assertIsInstance(result, EscapeResult)
        self.assertGreater(result.success_rate, 0.0)

    def test_escape_qfn(self):
        """Test escape_qfn convenience function."""
        nets = {i: f"NET_{i}" for i in range(1, 13)}  # 12-pin QFN

        result, thermal_vias = escape_qfn(
            pins_per_side=3,
            pitch=0.5,
            thermal_pad_size=(2.0, 2.0),
            net_assignments=nets
        )

        self.assertIsInstance(result, EscapeResult)
        self.assertIsInstance(thermal_vias, list)


class TestQFNThermalEscape(unittest.TestCase):
    """Test QFN thermal pad escape routing."""

    def test_thermal_via_pattern(self):
        """Test thermal via pattern generation."""
        thermal = QFNThermalEscape()
        vias = thermal.create_thermal_via_pattern(
            pad_width=3.0,
            pad_height=3.0,
            via_drill=0.3,
            via_pitch=1.0
        )

        self.assertGreater(len(vias), 0)
        for via in vias:
            self.assertEqual(via.net, "GND")
            self.assertEqual(via.drill_diameter, 0.3)

    def test_thermal_resistance_calculation(self):
        """Test thermal resistance calculation."""
        thermal = QFNThermalEscape()
        r_th = thermal.calculate_thermal_resistance(
            via_count=9,
            via_drill=0.3
        )

        self.assertIsInstance(r_th, float)
        self.assertGreater(r_th, 0)


class TestPeripheralFanout(unittest.TestCase):
    """Test peripheral package fanout routing."""

    def test_create_fanout(self):
        """Test fanout trace creation."""
        pins = [
            Pin(id="1", row=0, col=0, x=0, y=0, net="SIG1"),
            Pin(id="2", row=0, col=1, x=0.5, y=0, net="SIG2"),
        ]

        fanout = PeripheralFanout()
        traces = fanout.create_fanout(
            pins=pins,
            fanout_direction=EscapeDirection.SOUTH,
            fanout_length=1.0
        )

        self.assertEqual(len(traces), 2)
        for trace in traces:
            self.assertIsInstance(trace, EscapeTrace)


class TestStrategyRecommendation(unittest.TestCase):
    """Test automatic strategy recommendation."""

    def test_recommend_for_fine_pitch(self):
        """Test strategy recommendation for fine pitch BGA."""
        piston = EscapePiston()

        array = PinArray(
            package_type=PackageType.FBGA,
            rows=10,
            cols=10,
            pitch=0.4,
            pins=[]
        )

        strategy = piston.recommend_strategy(array)
        # Fine pitch should recommend advanced strategies
        self.assertIn(strategy, [EscapeStrategy.ORDERED_MMCF,
                                 EscapeStrategy.SAT_MULTI_LAYER])

    def test_recommend_for_standard_pitch(self):
        """Test strategy recommendation for standard pitch BGA."""
        piston = EscapePiston()

        array = PinArray(
            package_type=PackageType.BGA,
            rows=6,
            cols=6,
            pitch=1.0,
            pins=[]
        )

        strategy = piston.recommend_strategy(array)
        # Standard pitch should allow simpler strategies
        self.assertIn(strategy, [EscapeStrategy.DOG_BONE,
                                 EscapeStrategy.RING_BASED])


if __name__ == '__main__':
    unittest.main()
