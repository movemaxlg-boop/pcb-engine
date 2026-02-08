#!/usr/bin/env python3
"""
Unit Tests for OrderPiston
==========================

Tests all functionality of the OrderPiston:
- Net ordering for routing priority
- Component ordering for placement
- Critical path identification
- Fanout analysis
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.order_piston import OrderPiston, OrderConfig, OrderResult


class TestOrderPistonInit(unittest.TestCase):
    """Test OrderPiston initialization."""

    def test_default_init(self):
        """Test default initialization."""
        config = OrderConfig()
        piston = OrderPiston(config)
        self.assertIsNotNone(piston)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = OrderConfig(
            board_width=100.0,
            board_height=80.0
        )
        piston = OrderPiston(config)
        self.assertIsNotNone(piston)


class TestNetOrdering(unittest.TestCase):
    """Test net ordering functionality."""

    def setUp(self):
        self.piston = OrderPiston(OrderConfig())
        self.sample_parts_db = {
            'parts': {
                'U1': {
                    'value': 'ESP32',
                    'pins': [
                        {'number': '1', 'name': 'VCC', 'net': 'VCC', 'offset': (0, 0)},
                        {'number': '2', 'name': 'GND', 'net': 'GND', 'offset': (0, 1)},
                        {'number': '3', 'name': 'IO0', 'net': 'SIG1', 'offset': (1, 0)},
                        {'number': '4', 'name': 'IO1', 'net': 'SIG2', 'offset': (1, 1)},
                    ]
                },
                'C1': {
                    'value': '100nF',
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (0, 0)},
                        {'number': '2', 'net': 'GND', 'offset': (0, 1)},
                    ]
                },
                'R1': {
                    'value': '10k',
                    'pins': [
                        {'number': '1', 'net': 'SIG1', 'offset': (0, 0)},
                        {'number': '2', 'net': 'SIG3', 'offset': (0, 1)},
                    ]
                }
            },
            'nets': {
                'VCC': {'pins': [('U1', '1'), ('C1', '1')]},
                'GND': {'pins': [('U1', '2'), ('C1', '2')]},
                'SIG1': {'pins': [('U1', '3'), ('R1', '1')]},
                'SIG2': {'pins': [('U1', '4')]},
                'SIG3': {'pins': [('R1', '2')]}
            }
        }

    def test_order_returns_result(self):
        """Test that order returns OrderResult."""
        result = self.piston.order(self.sample_parts_db)
        self.assertIsInstance(result, OrderResult)

    def test_order_returns_net_order(self):
        """Test that result contains net order."""
        result = self.piston.order(self.sample_parts_db)
        self.assertIsNotNone(result.net_order)
        self.assertIsInstance(result.net_order, list)

    def test_routable_nets_in_order(self):
        """Test that routable nets (2+ pins) are included in order."""
        result = self.piston.order(self.sample_parts_db)
        # At minimum, nets with 2+ pins should be in order
        for net_name, net_data in self.sample_parts_db['nets'].items():
            if len(net_data.get('pins', [])) >= 2:
                self.assertIn(net_name, result.net_order)

    def test_power_nets_in_order(self):
        """Test that power nets are in the order."""
        result = self.piston.order(self.sample_parts_db)
        # VCC should be in the order
        self.assertIn('VCC', result.net_order)
        self.assertIn('GND', result.net_order)


class TestPlacementOrdering(unittest.TestCase):
    """Test component ordering for placement."""

    def setUp(self):
        self.piston = OrderPiston(OrderConfig())
        self.sample_parts_db = {
            'parts': {
                'U1': {'value': 'ESP32', 'pins': []},
                'U2': {'value': 'USB', 'pins': []},
                'C1': {'value': '100nF', 'pins': []},
                'C2': {'value': '10uF', 'pins': []},
                'R1': {'value': '10k', 'pins': []},
            },
            'nets': {}
        }

    def test_placement_order_exists(self):
        """Test that placement order is generated."""
        result = self.piston.order(self.sample_parts_db)
        self.assertIsNotNone(result.placement_order)

    def test_all_components_in_placement_order(self):
        """Test that all components are in the placement order."""
        result = self.piston.order(self.sample_parts_db)
        component_names = set(self.sample_parts_db['parts'].keys())
        ordered_components = set(result.placement_order)
        self.assertEqual(component_names, ordered_components)


class TestFanoutAnalysis(unittest.TestCase):
    """Test net fanout analysis."""

    def setUp(self):
        self.piston = OrderPiston(OrderConfig())
        # Create nets with different fanouts
        self.sample_parts_db = {
            'parts': {
                'U1': {'pins': [
                    {'number': '1', 'net': 'HIGH_FANOUT', 'offset': (0, 0)},
                    {'number': '2', 'net': 'LOW_FANOUT', 'offset': (0, 1)},
                ]},
                'C1': {'pins': [{'number': '1', 'net': 'HIGH_FANOUT', 'offset': (0, 0)}]},
                'C2': {'pins': [{'number': '1', 'net': 'HIGH_FANOUT', 'offset': (0, 0)}]},
                'C3': {'pins': [{'number': '1', 'net': 'HIGH_FANOUT', 'offset': (0, 0)}]},
                'R1': {'pins': [{'number': '1', 'net': 'LOW_FANOUT', 'offset': (0, 0)}]},
            },
            'nets': {
                'HIGH_FANOUT': {'pins': [('U1', '1'), ('C1', '1'), ('C2', '1'), ('C3', '1')]},
                'LOW_FANOUT': {'pins': [('U1', '2'), ('R1', '1')]}
            }
        }

    def test_fanout_order_exists(self):
        """Test that nets are ordered."""
        result = self.piston.order(self.sample_parts_db)
        self.assertIsNotNone(result.net_order)

    def test_all_nets_in_order(self):
        """Test that all nets are in the order."""
        result = self.piston.order(self.sample_parts_db)
        self.assertIn('HIGH_FANOUT', result.net_order)
        self.assertIn('LOW_FANOUT', result.net_order)


class TestCriticalPathOrdering(unittest.TestCase):
    """Test critical path identification and ordering."""

    def setUp(self):
        config = OrderConfig()
        self.piston = OrderPiston(config)
        self.sample_parts_db = {
            'parts': {
                'U1': {'pins': [
                    {'number': '1', 'net': 'CLK', 'offset': (0, 0)},
                    {'number': '2', 'net': 'DATA', 'offset': (0, 1)},
                ]},
                'U2': {'pins': [
                    {'number': '1', 'net': 'CLK', 'offset': (0, 0)},
                    {'number': '2', 'net': 'DATA', 'offset': (0, 1)},
                ]},
            },
            'nets': {
                'CLK': {'pins': [('U1', '1'), ('U2', '1')]},
                'DATA': {'pins': [('U1', '2'), ('U2', '2')]}
            }
        }

    def test_clock_nets_in_order(self):
        """Test that clock nets are in the order."""
        result = self.piston.order(self.sample_parts_db)
        self.assertIn('CLK', result.net_order)


class TestEmptyInput(unittest.TestCase):
    """Test handling of empty input."""

    def setUp(self):
        self.piston = OrderPiston(OrderConfig())

    def test_empty_parts_db(self):
        """Test handling of empty parts database."""
        result = self.piston.order({'parts': {}, 'nets': {}})
        self.assertIsNotNone(result)
        self.assertEqual(len(result.net_order), 0)
        self.assertEqual(len(result.placement_order), 0)

    def test_missing_nets(self):
        """Test handling when nets key is missing."""
        result = self.piston.order({'parts': {'R1': {'pins': []}}})
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
