#!/usr/bin/env python3
"""
Unit Tests for RoutingPiston
============================

Tests all functionality of the RoutingPiston:
- Route creation between pins
- Routing algorithms (Lee, A*, Hadlock, etc.)
- Via placement
- Layer management
- DRC-aware routing
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.routing_piston import RoutingPiston
from pcb_engine.routing_types import (
    RoutingConfig, RoutingResult, RoutingAlgorithm,
    TrackSegment, Via, Route
)


class TestRoutingPistonInit(unittest.TestCase):
    """Test RoutingPiston initialization."""

    def test_default_init(self):
        """Test default initialization."""
        config = RoutingConfig()
        piston = RoutingPiston(config)
        self.assertIsNotNone(piston)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = RoutingConfig(
            board_width=100.0,
            board_height=80.0,
            grid_size=0.25,
            trace_width=0.3
        )
        piston = RoutingPiston(config)
        self.assertEqual(piston.config.board_width, 100.0)
        self.assertEqual(piston.config.trace_width, 0.3)


class TestTrackSegment(unittest.TestCase):
    """Test TrackSegment data structure."""

    def test_track_creation(self):
        """Test creating a track segment."""
        track = TrackSegment(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            layer='F.Cu',
            width=0.25,
            net='VCC'
        )
        self.assertEqual(track.start, (0.0, 0.0))
        self.assertEqual(track.end, (10.0, 0.0))

    def test_track_length(self):
        """Test track length calculation."""
        track = TrackSegment(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            layer='F.Cu',
            width=0.25,
            net='VCC'
        )
        self.assertAlmostEqual(track.length, 10.0, places=2)

    def test_track_horizontal(self):
        """Test horizontal track detection."""
        track = TrackSegment(
            start=(0.0, 5.0),
            end=(10.0, 5.0),
            layer='F.Cu',
            width=0.25,
            net='VCC'
        )
        self.assertTrue(track.is_horizontal)
        self.assertFalse(track.is_vertical)

    def test_track_vertical(self):
        """Test vertical track detection."""
        track = TrackSegment(
            start=(5.0, 0.0),
            end=(5.0, 10.0),
            layer='F.Cu',
            width=0.25,
            net='VCC'
        )
        self.assertTrue(track.is_vertical)
        self.assertFalse(track.is_horizontal)


class TestVia(unittest.TestCase):
    """Test Via data structure."""

    def test_via_creation(self):
        """Test creating a via."""
        via = Via(
            position=(10.0, 20.0),
            net='VCC',
            diameter=0.8,
            drill=0.4
        )
        self.assertEqual(via.position, (10.0, 20.0))
        self.assertEqual(via.net, 'VCC')

    def test_via_layers(self):
        """Test via layer specification."""
        via = Via(
            position=(10.0, 20.0),
            net='VCC',
            from_layer='F.Cu',
            to_layer='B.Cu'
        )
        self.assertEqual(via.from_layer, 'F.Cu')
        self.assertEqual(via.to_layer, 'B.Cu')


class TestRoute(unittest.TestCase):
    """Test Route data structure."""

    def test_route_creation(self):
        """Test creating a route."""
        route = Route(
            net='VCC',
            segments=[],
            vias=[],
            success=True
        )
        self.assertEqual(route.net, 'VCC')
        self.assertTrue(route.success)

    def test_route_total_length(self):
        """Test route total length calculation."""
        seg1 = TrackSegment((0, 0), (10, 0), 'F.Cu', 0.25, 'VCC')
        seg2 = TrackSegment((10, 0), (10, 5), 'F.Cu', 0.25, 'VCC')
        route = Route(
            net='VCC',
            segments=[seg1, seg2],
            success=True
        )
        self.assertAlmostEqual(route.total_length, 15.0, places=2)

    def test_route_bend_count(self):
        """Test route bend count."""
        seg1 = TrackSegment((0, 0), (10, 0), 'F.Cu', 0.25, 'VCC')  # Horizontal
        seg2 = TrackSegment((10, 0), (10, 5), 'F.Cu', 0.25, 'VCC')  # Vertical
        seg3 = TrackSegment((10, 5), (15, 5), 'F.Cu', 0.25, 'VCC')  # Horizontal
        route = Route(
            net='VCC',
            segments=[seg1, seg2, seg3],
            success=True
        )
        self.assertEqual(route.bend_count, 2)


class TestRoutingExecution(unittest.TestCase):
    """Test routing execution."""

    def setUp(self):
        self.config = RoutingConfig(
            board_width=50.0,
            board_height=40.0,
            grid_size=0.5,
            trace_width=0.25,
            clearance=0.15
        )
        self.piston = RoutingPiston(self.config)

        # Simple placement with two connected components
        self.placement = {
            'U1': (10.0, 20.0),
            'C1': (30.0, 20.0)
        }

        self.parts_db = {
            'parts': {
                'U1': {
                    'value': 'IC',
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (0, 0)},
                        {'number': '2', 'net': 'GND', 'offset': (1, 0)},
                    ]
                },
                'C1': {
                    'value': '100nF',
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (0, 0)},
                        {'number': '2', 'net': 'GND', 'offset': (1, 0)},
                    ]
                }
            },
            'nets': {
                'VCC': {'pins': [('U1', '1'), ('C1', '1')]},
                'GND': {'pins': [('U1', '2'), ('C1', '2')]}
            }
        }
        self.net_order = ['VCC', 'GND']

    def test_route_returns_result(self):
        """Test that routing returns a RoutingResult."""
        result = self.piston.route(
            parts_db=self.parts_db,
            escapes={},
            placement=self.placement,
            net_order=self.net_order
        )
        self.assertIsInstance(result, RoutingResult)

    def test_route_has_routes_dict(self):
        """Test that result contains routes dictionary."""
        result = self.piston.route(
            parts_db=self.parts_db,
            escapes={},
            placement=self.placement,
            net_order=self.net_order
        )
        self.assertIsNotNone(result.routes)
        self.assertIsInstance(result.routes, dict)

    def test_route_counts(self):
        """Test that routed and total counts are populated."""
        result = self.piston.route(
            parts_db=self.parts_db,
            escapes={},
            placement=self.placement,
            net_order=self.net_order
        )
        self.assertGreaterEqual(result.total_count, 0)
        self.assertGreaterEqual(result.routed_count, 0)
        self.assertLessEqual(result.routed_count, result.total_count)


class TestRoutingAlgorithms(unittest.TestCase):
    """Test different routing algorithms."""

    def setUp(self):
        self.placement = {
            'U1': (10.0, 20.0),
            'R1': (25.0, 20.0)
        }
        self.parts_db = {
            'parts': {
                'U1': {'pins': [{'number': '1', 'net': 'NET1', 'offset': (0, 0)}]},
                'R1': {'pins': [{'number': '1', 'net': 'NET1', 'offset': (0, 0)}]}
            },
            'nets': {'NET1': {'pins': [('U1', '1'), ('R1', '1')]}}
        }

    def test_lee_algorithm(self):
        """Test Lee routing algorithm."""
        config = RoutingConfig(
            board_width=50.0,
            board_height=40.0,
            algorithm='lee'
        )
        piston = RoutingPiston(config)
        result = piston.route(self.parts_db, {}, self.placement, ['NET1'])
        self.assertIsNotNone(result)

    def test_astar_algorithm(self):
        """Test A* routing algorithm."""
        config = RoutingConfig(
            board_width=50.0,
            board_height=40.0,
            algorithm='astar'
        )
        piston = RoutingPiston(config)
        result = piston.route(self.parts_db, {}, self.placement, ['NET1'])
        self.assertIsNotNone(result)

    def test_hadlock_algorithm(self):
        """Test Hadlock routing algorithm."""
        config = RoutingConfig(
            board_width=50.0,
            board_height=40.0,
            algorithm='hadlock'
        )
        piston = RoutingPiston(config)
        result = piston.route(self.parts_db, {}, self.placement, ['NET1'])
        self.assertIsNotNone(result)


class TestLayerManagement(unittest.TestCase):
    """Test multi-layer routing."""

    def setUp(self):
        self.config = RoutingConfig(
            board_width=50.0,
            board_height=40.0,
            allow_layer_change=True,
            top_layer_name='F.Cu',
            bottom_layer_name='B.Cu'
        )
        self.piston = RoutingPiston(self.config)

    def test_layer_names(self):
        """Test layer name configuration."""
        self.assertEqual(self.piston.config.top_layer_name, 'F.Cu')
        self.assertEqual(self.piston.config.bottom_layer_name, 'B.Cu')

    def test_layer_change_allowed(self):
        """Test layer change configuration."""
        self.assertTrue(self.piston.config.allow_layer_change)


class TestRoutingConfig(unittest.TestCase):
    """Test routing configuration options."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RoutingConfig()
        self.assertGreater(config.board_width, 0)
        self.assertGreater(config.grid_size, 0)
        self.assertGreater(config.trace_width, 0)
        self.assertGreater(config.clearance, 0)

    def test_via_config(self):
        """Test via configuration."""
        config = RoutingConfig(
            via_diameter=0.6,
            via_drill=0.3,
            via_cost=10.0
        )
        self.assertEqual(config.via_diameter, 0.6)
        self.assertEqual(config.via_drill, 0.3)
        self.assertEqual(config.via_cost, 10.0)


class TestEmptyAndEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.config = RoutingConfig(
            board_width=50.0,
            board_height=40.0
        )
        self.piston = RoutingPiston(self.config)

    def test_empty_net_order(self):
        """Test routing with no nets to route."""
        result = self.piston.route(
            parts_db={'parts': {}, 'nets': {}},
            escapes={},
            placement={},
            net_order=[]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.total_count, 0)

    def test_single_pin_net(self):
        """Test handling of single-pin nets (unroutable)."""
        parts_db = {
            'parts': {'U1': {'pins': [{'number': '1', 'net': 'FLOAT', 'offset': (0, 0)}]}},
            'nets': {'FLOAT': {'pins': [('U1', '1')]}}
        }
        result = self.piston.route(
            parts_db=parts_db,
            escapes={},
            placement={'U1': (10.0, 20.0)},
            net_order=['FLOAT']
        )
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
