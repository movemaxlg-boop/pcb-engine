#!/usr/bin/env python3
"""
Unit Tests for PartsPiston
==========================

Tests all functionality of the PartsPiston:
- Component parsing from text descriptions
- Parts database building
- Trace width calculations
- Clearance calculations
- Net classification
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.parts_piston import (
    PartsPiston, PartsConfig, PartsResult,
    Component, MountType, PinInfo, NetClass,
    parse_si_value
)


class TestPartsPistonInit(unittest.TestCase):
    """Test PartsPiston initialization."""

    def test_default_init(self):
        """Test default initialization."""
        piston = PartsPiston()
        self.assertIsNotNone(piston)
        self.assertIsNotNone(piston.config)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = PartsConfig(
            default_trace_width=0.3,
            default_clearance=0.2
        )
        piston = PartsPiston(config)
        self.assertEqual(piston.config.default_trace_width, 0.3)


class TestParseComponent(unittest.TestCase):
    """Test component parsing from text descriptions."""

    def setUp(self):
        self.piston = PartsPiston()

    def test_parse_resistor(self):
        """Test parsing a resistor description."""
        component = self.piston.parse_component("10k resistor 0603")
        self.assertIsNotNone(component)
        self.assertEqual(component.value, '10k')
        # Ref should start with R for resistor
        self.assertTrue(component.ref.startswith('R'))

    def test_parse_capacitor(self):
        """Test parsing a capacitor description."""
        component = self.piston.parse_component("100nF capacitor 0805")
        self.assertIsNotNone(component)
        # Ref should start with C for capacitor
        self.assertTrue(component.ref.startswith('C'))

    def test_parse_led(self):
        """Test parsing an LED description."""
        component = self.piston.parse_component("red LED 0805")
        self.assertIsNotNone(component)
        # Ref should start with D for LED/Diode
        self.assertTrue(component.ref.startswith('D') or component.ref.startswith('LED'))

    def test_parse_mcu(self):
        """Test parsing an MCU description."""
        component = self.piston.parse_component("ESP32-WROOM-32")
        self.assertIsNotNone(component)
        # MCU should start with U
        self.assertTrue(component.ref.startswith('U'))

    def test_parse_connector(self):
        """Test parsing a connector description."""
        component = self.piston.parse_component("USB-C connector")
        self.assertIsNotNone(component)


class TestBuildFromDict(unittest.TestCase):
    """Test building parts from dictionary input."""

    def setUp(self):
        self.piston = PartsPiston()
        self.sample_parts_db = {
            'parts': {
                'R1': {
                    'value': '10k',
                    'footprint': '0603',
                    'pins': [
                        {'number': '1', 'name': 'A', 'net': 'VCC', 'offset': (0, 0)},
                        {'number': '2', 'name': 'B', 'net': 'GPIO1', 'offset': (1.6, 0)}
                    ]
                },
                'C1': {
                    'value': '100nF',
                    'footprint': '0805',
                    'pins': [
                        {'number': '1', 'name': 'P1', 'net': 'VCC', 'offset': (0, 0)},
                        {'number': '2', 'name': 'P2', 'net': 'GND', 'offset': (2.0, 0)}
                    ]
                }
            },
            'nets': {
                'VCC': {'pins': [('R1', '1'), ('C1', '1')]},
                'GND': {'pins': [('C1', '2')]},
                'GPIO1': {'pins': [('R1', '2')]}
            }
        }

    def test_build_returns_result(self):
        """Test that build_from_dict returns a PartsResult."""
        result = self.piston.build_from_dict(self.sample_parts_db)
        self.assertIsInstance(result, PartsResult)

    def test_build_has_components(self):
        """Test that result has components."""
        result = self.piston.build_from_dict(self.sample_parts_db)
        self.assertIsNotNone(result.components)
        self.assertEqual(len(result.components), 2)

    def test_build_has_nets(self):
        """Test that result has nets."""
        result = self.piston.build_from_dict(self.sample_parts_db)
        self.assertIsNotNone(result.nets)

    def test_build_empty_db(self):
        """Test handling empty parts database."""
        result = self.piston.build_from_dict({'parts': {}, 'nets': {}})
        self.assertIsNotNone(result)

    def test_build_missing_parts_key(self):
        """Test handling missing 'parts' key."""
        result = self.piston.build_from_dict({'nets': {}})
        self.assertIsNotNone(result)


class TestTraceWidthCalculation(unittest.TestCase):
    """Test trace width calculations."""

    def setUp(self):
        self.piston = PartsPiston()

    def test_low_current_trace(self):
        """Test trace width for low current (100mA)."""
        width = self.piston.calculate_trace_width(current=0.1, temp_rise=10.0)
        self.assertIsInstance(width, float)
        self.assertGreater(width, 0)
        self.assertLess(width, 1.0)  # Should be less than 1mm for 100mA

    def test_high_current_trace(self):
        """Test trace width for high current (2A)."""
        width = self.piston.calculate_trace_width(current=2.0, temp_rise=10.0)
        self.assertIsInstance(width, float)
        self.assertGreater(width, 0.3)  # 2A needs substantial trace

    def test_higher_temp_rise_smaller_trace(self):
        """Test that higher temp rise allows smaller trace."""
        width_10c = self.piston.calculate_trace_width(current=1.0, temp_rise=10.0)
        width_20c = self.piston.calculate_trace_width(current=1.0, temp_rise=20.0)
        self.assertLess(width_20c, width_10c)

    def test_internal_layer_wider_trace(self):
        """Test that internal layers need wider traces."""
        width_external = self.piston.calculate_trace_width(current=1.0, temp_rise=10.0, is_internal=False)
        width_internal = self.piston.calculate_trace_width(current=1.0, temp_rise=10.0, is_internal=True)
        self.assertGreater(width_internal, width_external)


class TestClearanceCalculation(unittest.TestCase):
    """Test voltage clearance calculations."""

    def setUp(self):
        self.piston = PartsPiston()

    def test_low_voltage_clearance(self):
        """Test clearance for low voltage (5V)."""
        clearance = self.piston.calculate_clearance(voltage=5.0)
        self.assertIsInstance(clearance, float)
        self.assertGreater(clearance, 0)

    def test_voltage_affects_clearance(self):
        """Test that higher voltage requires more or equal clearance."""
        clearance_5v = self.piston.calculate_clearance(voltage=5.0)
        clearance_50v = self.piston.calculate_clearance(voltage=50.0)
        # Higher voltage should require >= clearance
        self.assertGreaterEqual(clearance_50v, clearance_5v)

    def test_internal_layer_clearance(self):
        """Test clearance for internal layers."""
        clearance = self.piston.calculate_clearance(voltage=50.0, is_internal=True)
        self.assertIsInstance(clearance, float)


class TestNetClassification(unittest.TestCase):
    """Test net classification after build."""

    def setUp(self):
        self.piston = PartsPiston()
        # Build first so net_classes gets populated
        self.sample_parts_db = {
            'parts': {
                'R1': {
                    'value': '10k',
                    'footprint': '0603',
                    'pins': [
                        {'number': '1', 'name': 'A', 'net': 'VCC'},
                        {'number': '2', 'name': 'B', 'net': 'GND'}
                    ]
                }
            },
            'nets': {
                'VCC': {'pins': [('R1', '1')]},
                'GND': {'pins': [('R1', '2')]},
                'GPIO1': {'pins': []}
            }
        }
        self.piston.build_from_dict(self.sample_parts_db)

    def test_power_net_classification(self):
        """Test power net is classified correctly."""
        net_class = self.piston.get_net_class('VCC')
        self.assertEqual(net_class, NetClass.POWER)

    def test_ground_net_classification(self):
        """Test ground net is classified correctly."""
        net_class = self.piston.get_net_class('GND')
        self.assertEqual(net_class, NetClass.GROUND)

    def test_signal_net_classification(self):
        """Test signal net is classified correctly."""
        net_class = self.piston.get_net_class('GPIO1')
        self.assertEqual(net_class, NetClass.SIGNAL)


class TestSIValueParsing(unittest.TestCase):
    """Test SI prefix value parsing."""

    def test_parse_kilo(self):
        """Test parsing kilo prefix (k)."""
        self.assertAlmostEqual(parse_si_value('10k'), 10000.0, places=1)

    def test_parse_mega(self):
        """Test parsing mega prefix (M)."""
        self.assertAlmostEqual(parse_si_value('1M'), 1000000.0, places=1)

    def test_parse_milli(self):
        """Test parsing milli prefix (m)."""
        self.assertAlmostEqual(parse_si_value('100m'), 0.1, places=3)

    def test_parse_micro(self):
        """Test parsing micro prefix (u)."""
        self.assertAlmostEqual(parse_si_value('10u'), 0.00001, places=7)

    def test_parse_nano(self):
        """Test parsing nano prefix (n)."""
        self.assertAlmostEqual(parse_si_value('100n'), 0.0000001, places=9)

    def test_parse_pico(self):
        """Test parsing pico prefix (p)."""
        self.assertAlmostEqual(parse_si_value('10p'), 0.00000000001, places=13)

    def test_parse_no_prefix(self):
        """Test parsing value without prefix."""
        self.assertAlmostEqual(parse_si_value('100'), 100.0, places=1)

    def test_parse_decimal(self):
        """Test parsing decimal value."""
        self.assertAlmostEqual(parse_si_value('4.7k'), 4700.0, places=1)


class TestRoutingRequirements(unittest.TestCase):
    """Test routing requirements extraction."""

    def setUp(self):
        self.piston = PartsPiston()
        # Build with sample data first
        self.sample_parts_db = {
            'parts': {
                'U1': {
                    'value': 'ESP32',
                    'footprint': 'QFN-48',
                    'pins': [
                        {'number': '1', 'name': 'VCC', 'net': 'VCC'},
                        {'number': '2', 'name': 'GND', 'net': 'GND'}
                    ]
                }
            },
            'nets': {
                'VCC': {'pins': [('U1', '1')]},
                'GND': {'pins': [('U1', '2')]}
            }
        }
        self.piston.build_from_dict(self.sample_parts_db)

    def test_get_routing_requirements(self):
        """Test getting routing requirements for a net."""
        req = self.piston.get_routing_requirements('VCC')
        self.assertIsInstance(req, dict)
        # Should have min_trace_width and min_clearance
        self.assertIn('min_trace_width', req)
        self.assertIn('min_clearance', req)

    def test_routing_requirements_values(self):
        """Test that routing requirements have valid values."""
        vcc_req = self.piston.get_routing_requirements('VCC')
        # Should have positive trace width
        self.assertGreater(vcc_req.get('min_trace_width', 0), 0)


class TestPlacementRequirements(unittest.TestCase):
    """Test placement requirements extraction."""

    def setUp(self):
        self.piston = PartsPiston()
        self.sample_parts_db = {
            'parts': {
                'U1': {
                    'value': 'ESP32',
                    'footprint': 'QFN-48',
                    'pins': []
                },
                'C1': {
                    'value': '100nF',
                    'footprint': '0603',
                    'pins': []
                }
            },
            'nets': {}
        }
        self.piston.build_from_dict(self.sample_parts_db)

    def test_get_placement_requirements(self):
        """Test getting placement requirements."""
        req = self.piston.get_placement_requirements('U1')
        self.assertIsInstance(req, dict)

    def test_placement_has_size_info(self):
        """Test that placement requirements include size info."""
        req = self.piston.get_placement_requirements('U1')
        # Should have some dimensional information
        self.assertTrue(len(req) > 0)


if __name__ == '__main__':
    unittest.main()
