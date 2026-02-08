#!/usr/bin/env python3
"""
Unit Tests for PlacementPiston
==============================

Tests all functionality of the PlacementPiston:
- Component placement on board
- Placement algorithms (Force-Directed, Simulated Annealing, etc.)
- Boundary constraints
- Overlap detection
- Placement quality metrics
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.placement_piston import PlacementPiston, PlacementConfig, PlacementResult


class TestPlacementPistonInit(unittest.TestCase):
    """Test PlacementPiston initialization."""

    def test_default_init(self):
        """Test default initialization."""
        config = PlacementConfig()
        piston = PlacementPiston(config)
        self.assertIsNotNone(piston)

    def test_custom_board_size(self):
        """Test initialization with custom board size."""
        config = PlacementConfig(
            board_width=100.0,
            board_height=80.0
        )
        piston = PlacementPiston(config)
        self.assertEqual(piston.config.board_width, 100.0)
        self.assertEqual(piston.config.board_height, 80.0)


class TestPlacementExecution(unittest.TestCase):
    """Test placement execution."""

    def setUp(self):
        self.config = PlacementConfig(
            board_width=50.0,
            board_height=40.0,
            origin_x=0.0,
            origin_y=0.0
        )
        self.piston = PlacementPiston(self.config)
        self.sample_parts_db = {
            'parts': {
                'U1': {
                    'value': 'ESP32',
                    'footprint': 'QFN-48',
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (-3.5, -3.5)},
                        {'number': '2', 'net': 'GND', 'offset': (3.5, -3.5)},
                    ]
                },
                'C1': {
                    'value': '100nF',
                    'footprint': '0603',
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (-0.75, 0)},
                        {'number': '2', 'net': 'GND', 'offset': (0.75, 0)},
                    ]
                },
                'R1': {
                    'value': '10k',
                    'footprint': '0603',
                    'pins': [
                        {'number': '1', 'net': 'SIG', 'offset': (-0.75, 0)},
                        {'number': '2', 'net': 'GND', 'offset': (0.75, 0)},
                    ]
                }
            },
            'nets': {
                'VCC': {'pins': [('U1', '1'), ('C1', '1')]},
                'GND': {'pins': [('U1', '2'), ('C1', '2'), ('R1', '2')]},
                'SIG': {'pins': [('R1', '1')]}
            }
        }

    def test_placement_returns_result(self):
        """Test that placement returns a result."""
        result = self.piston.place(self.sample_parts_db)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, PlacementResult)

    def test_placement_has_positions(self):
        """Test that placement result has positions."""
        result = self.piston.place(self.sample_parts_db)
        self.assertIsNotNone(result.positions)

    def test_all_components_placed(self):
        """Test that all components are placed."""
        result = self.piston.place(self.sample_parts_db)
        placed_refs = set(result.positions.keys())
        expected_refs = set(self.sample_parts_db['parts'].keys())
        self.assertEqual(placed_refs, expected_refs)

    def test_positions_within_board(self):
        """Test that all positions are within board boundaries."""
        result = self.piston.place(self.sample_parts_db)
        for ref, pos in result.positions.items():
            # Get x, y from position (handle both tuple and object)
            if hasattr(pos, 'x'):
                x, y = pos.x, pos.y
            else:
                x, y = pos[0], pos[1]

            self.assertGreaterEqual(x, 0, f"{ref} x position below origin")
            self.assertGreaterEqual(y, 0, f"{ref} y position below origin")
            self.assertLessEqual(x, self.config.board_width, f"{ref} x exceeds board width")
            self.assertLessEqual(y, self.config.board_height, f"{ref} y exceeds board height")


class TestPlacementAlgorithms(unittest.TestCase):
    """Test different placement algorithms."""

    def setUp(self):
        self.sample_parts_db = {
            'parts': {
                'U1': {'value': 'IC', 'pins': []},
                'R1': {'value': '10k', 'pins': []},
                'C1': {'value': '100nF', 'pins': []},
            },
            'nets': {}
        }

    def test_force_directed_algorithm(self):
        """Test Force-Directed placement algorithm."""
        config = PlacementConfig(
            board_width=50.0,
            board_height=40.0,
            algorithm='fd'
        )
        piston = PlacementPiston(config)
        result = piston.place(self.sample_parts_db)
        self.assertIsNotNone(result)

    def test_simulated_annealing_algorithm(self):
        """Test Simulated Annealing placement algorithm."""
        config = PlacementConfig(
            board_width=50.0,
            board_height=40.0,
            algorithm='sa'
        )
        piston = PlacementPiston(config)
        result = piston.place(self.sample_parts_db)
        self.assertIsNotNone(result)

    def test_hybrid_algorithm(self):
        """Test Hybrid placement algorithm."""
        config = PlacementConfig(
            board_width=50.0,
            board_height=40.0,
            algorithm='hybrid'
        )
        piston = PlacementPiston(config)
        result = piston.place(self.sample_parts_db)
        self.assertIsNotNone(result)


class TestPlacementQuality(unittest.TestCase):
    """Test placement quality metrics."""

    def setUp(self):
        self.config = PlacementConfig(
            board_width=50.0,
            board_height=40.0
        )
        self.piston = PlacementPiston(self.config)
        # Create parts with connected nets for wirelength calculation
        self.sample_parts_db = {
            'parts': {
                'U1': {
                    'value': 'IC',
                    'pins': [
                        {'number': '1', 'net': 'NET1', 'offset': (0, 0)},
                        {'number': '2', 'net': 'NET2', 'offset': (1, 0)},
                    ]
                },
                'R1': {
                    'value': '10k',
                    'pins': [
                        {'number': '1', 'net': 'NET1', 'offset': (0, 0)},
                        {'number': '2', 'net': 'NET3', 'offset': (1, 0)},
                    ]
                },
                'C1': {
                    'value': '100nF',
                    'pins': [
                        {'number': '1', 'net': 'NET2', 'offset': (0, 0)},
                        {'number': '2', 'net': 'NET3', 'offset': (1, 0)},
                    ]
                },
            },
            'nets': {
                'NET1': {'pins': [('U1', '1'), ('R1', '1')]},
                'NET2': {'pins': [('U1', '2'), ('C1', '1')]},
                'NET3': {'pins': [('R1', '2'), ('C1', '2')]},
            }
        }

    def test_placement_has_wirelength(self):
        """Test that placement result includes wirelength."""
        result = self.piston.place(self.sample_parts_db)
        self.assertIsNotNone(result)
        # Check for wirelength attribute
        self.assertTrue(hasattr(result, 'wirelength'))

    def test_placement_success_flag(self):
        """Test that placement result has success flag."""
        result = self.piston.place(self.sample_parts_db)
        self.assertTrue(result.success)


class TestBoundaryConstraints(unittest.TestCase):
    """Test boundary and constraint handling."""

    def test_small_board_still_places(self):
        """Test placement on a small board."""
        config = PlacementConfig(
            board_width=20.0,
            board_height=15.0
        )
        piston = PlacementPiston(config)
        parts_db = {
            'parts': {
                'R1': {'value': '10k', 'pins': []},
                'C1': {'value': '100nF', 'pins': []},
            },
            'nets': {}
        }
        result = piston.place(parts_db)
        self.assertIsNotNone(result)

    def test_origin_offset(self):
        """Test placement with origin offset."""
        config = PlacementConfig(
            board_width=50.0,
            board_height=40.0,
            origin_x=10.0,
            origin_y=10.0
        )
        piston = PlacementPiston(config)
        parts_db = {
            'parts': {'R1': {'value': '10k', 'pins': []}},
            'nets': {}
        }
        result = piston.place(parts_db)
        self.assertIsNotNone(result)


class TestEmptyAndEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.config = PlacementConfig(
            board_width=50.0,
            board_height=40.0
        )
        self.piston = PlacementPiston(self.config)

    def test_empty_parts_db(self):
        """Test placement with no parts."""
        result = self.piston.place({'parts': {}, 'nets': {}})
        self.assertIsNotNone(result)
        self.assertEqual(len(result.positions), 0)

    def test_single_component(self):
        """Test placement with single component."""
        parts_db = {
            'parts': {'U1': {'value': 'IC', 'pins': []}},
            'nets': {}
        }
        result = self.piston.place(parts_db)
        self.assertEqual(len(result.positions), 1)
        self.assertIn('U1', result.positions)


if __name__ == '__main__':
    unittest.main()
