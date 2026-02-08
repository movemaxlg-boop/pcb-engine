#!/usr/bin/env python3
"""
Unit Tests for SilkscreenPiston
===============================

Tests all functionality of the SilkscreenPiston:
- Text placement and optimization
- Force-directed label positioning
- Collision detection
- Polarity and assembly markers
- IPC compliance
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.silkscreen_piston import (
    SilkscreenPiston, SilkscreenConfig, SilkscreenResult,
    SilkscreenText, SilkscreenLine, SilkscreenArc, SilkscreenPolygon,
    PolarityMarker, AssemblyMarker, Fiducial,
    TextAnchor, TextOrientation, PlacementPosition, ComponentCategory,
    ManufacturingStandard, IPC_TEXT_SIZES, MANUFACTURING_RULES
)


class MockPosition:
    """Mock position object for testing."""
    def __init__(self, x, y, rotation=0, layer='F.Cu'):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.layer = layer


class TestSilkscreenPistonInit(unittest.TestCase):
    """Test SilkscreenPiston initialization."""

    def test_default_init(self):
        """Test default initialization."""
        piston = SilkscreenPiston()
        self.assertIsNotNone(piston)
        self.assertIsNotNone(piston.config)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = SilkscreenConfig(
            ref_text_size=1.2,
            show_values=True,
            show_polarity_markers=True
        )
        piston = SilkscreenPiston(config)
        self.assertEqual(piston.config.ref_text_size, 1.2)
        self.assertTrue(piston.config.show_values)


class TestSilkscreenConfig(unittest.TestCase):
    """Test SilkscreenConfig defaults and customization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SilkscreenConfig()
        self.assertEqual(config.ref_text_size, 1.0)
        self.assertTrue(config.show_references)
        self.assertFalse(config.show_values)  # Usually disabled
        self.assertEqual(config.manufacturing_standard, ManufacturingStandard.IPC_CLASS_2)

    def test_manufacturing_standards(self):
        """Test different manufacturing standard configurations."""
        for standard in ManufacturingStandard:
            config = SilkscreenConfig(manufacturing_standard=standard)
            self.assertEqual(config.manufacturing_standard, standard)


class TestSilkscreenText(unittest.TestCase):
    """Test SilkscreenText element."""

    def test_text_creation(self):
        """Test creating silkscreen text."""
        text = SilkscreenText(
            text="R1",
            x=10.0,
            y=20.0,
            size=1.0,
            thickness=0.15
        )
        self.assertEqual(text.text, "R1")
        self.assertEqual(text.x, 10.0)
        self.assertEqual(text.y, 20.0)

    def test_bounding_box(self):
        """Test bounding box calculation."""
        text = SilkscreenText(
            text="U1",
            x=0,
            y=0,
            size=1.0,
            rotation=0
        )
        bbox = text.get_bounding_box()

        self.assertEqual(len(bbox), 4)  # min_x, min_y, max_x, max_y
        self.assertLess(bbox[0], bbox[2])  # min_x < max_x
        self.assertLess(bbox[1], bbox[3])  # min_y < max_y

    def test_rotated_bounding_box(self):
        """Test bounding box for rotated text."""
        text_h = SilkscreenText(text="TEST", x=0, y=0, size=1.0, rotation=0)
        text_v = SilkscreenText(text="TEST", x=0, y=0, size=1.0, rotation=90)

        bbox_h = text_h.get_bounding_box()
        bbox_v = text_v.get_bounding_box()

        # Rotated text should have different aspect ratio
        width_h = bbox_h[2] - bbox_h[0]
        height_h = bbox_h[3] - bbox_h[1]
        width_v = bbox_v[2] - bbox_v[0]
        height_v = bbox_v[3] - bbox_v[1]

        # The aspect ratios should be approximately inverted
        self.assertNotEqual(width_h / height_h, width_v / height_v)


class TestTextPlacement(unittest.TestCase):
    """Test text placement algorithms."""

    def setUp(self):
        """Set up test data."""
        self.parts_db = {
            'parts': {
                'R1': {
                    'reference': 'R1',
                    'value': '10k',
                    'category': 'resistor',
                    'body': {'width': 1.6, 'height': 0.8},
                    'used_pins': [
                        {'number': '1', 'offset': (-0.8, 0), 'pad_size': (0.8, 0.9)},
                        {'number': '2', 'offset': (0.8, 0), 'pad_size': (0.8, 0.9)},
                    ]
                },
                'U1': {
                    'reference': 'U1',
                    'value': 'ATmega328P',
                    'category': 'ic',
                    'body': {'width': 8.0, 'height': 8.0},
                    'used_pins': [
                        {'number': '1', 'offset': (-3.5, -3.5), 'pad_size': (0.6, 1.5)},
                    ]
                },
            }
        }
        self.placement = {
            'R1': MockPosition(10, 10, 0),
            'U1': MockPosition(25, 20, 0),
        }

    def test_generate_produces_result(self):
        """Test that generate produces a result."""
        piston = SilkscreenPiston()
        result = piston.generate(self.parts_db, self.placement)

        self.assertIsInstance(result, SilkscreenResult)
        self.assertIsInstance(result.texts, list)

    def test_reference_designators_created(self):
        """Test that reference designators are created."""
        config = SilkscreenConfig(show_references=True)
        piston = SilkscreenPiston(config)
        result = piston.generate(self.parts_db, self.placement)

        self.assertGreater(result.ref_count, 0)
        ref_texts = [t.text for t in result.texts if t.text_type == 'reference']
        self.assertIn('R1', ref_texts)
        self.assertIn('U1', ref_texts)

    def test_values_created_when_enabled(self):
        """Test that value texts are created when enabled."""
        config = SilkscreenConfig(show_values=True)
        piston = SilkscreenPiston(config)
        result = piston.generate(self.parts_db, self.placement)

        self.assertGreater(result.value_count, 0)


class TestCollisionDetection(unittest.TestCase):
    """Test collision detection for text placement."""

    def setUp(self):
        """Set up test data with potential collisions."""
        self.piston = SilkscreenPiston()

    def test_rects_overlap(self):
        """Test rectangle overlap detection."""
        # Overlapping rectangles
        r1 = (0, 0, 10, 10)
        r2 = (5, 5, 15, 15)
        self.assertTrue(self.piston._rects_overlap(r1, r2))

        # Non-overlapping rectangles
        r3 = (0, 0, 5, 5)
        r4 = (10, 10, 15, 15)
        self.assertFalse(self.piston._rects_overlap(r3, r4))

    def test_no_pad_overlap(self):
        """Test that text doesn't overlap pads."""
        parts_db = {
            'parts': {
                'R1': {
                    'reference': 'R1',
                    'value': '10k',
                    'body': {'width': 1.6, 'height': 0.8},
                    'used_pins': [
                        {'number': '1', 'offset': (-0.8, 0), 'pad_size': (0.8, 0.9)},
                        {'number': '2', 'offset': (0.8, 0), 'pad_size': (0.8, 0.9)},
                    ]
                },
            }
        }
        placement = {'R1': MockPosition(10, 10, 0)}

        piston = SilkscreenPiston()
        result = piston.generate(parts_db, placement)

        # Should have no or minimal collisions
        self.assertLessEqual(result.collision_count, 1)


class TestOptimization(unittest.TestCase):
    """Test text placement optimization algorithms."""

    def setUp(self):
        """Set up test data."""
        self.parts_db = {
            'parts': {
                'R1': {'reference': 'R1', 'value': '10k', 'body': {'width': 1.6, 'height': 0.8}},
                'R2': {'reference': 'R2', 'value': '4.7k', 'body': {'width': 1.6, 'height': 0.8}},
            }
        }
        self.placement = {
            'R1': MockPosition(10, 10, 0),
            'R2': MockPosition(12, 10, 0),  # Close to R1
        }

    def test_force_directed_optimization(self):
        """Test force-directed optimization method."""
        config = SilkscreenConfig(optimization_method='force_directed')
        piston = SilkscreenPiston(config)
        result = piston.generate(self.parts_db, self.placement)

        self.assertIsInstance(result, SilkscreenResult)
        self.assertGreater(result.optimization_iterations, 0)

    def test_simulated_annealing_optimization(self):
        """Test simulated annealing optimization."""
        config = SilkscreenConfig(optimization_method='simulated_annealing')
        piston = SilkscreenPiston(config)
        result = piston.generate(self.parts_db, self.placement)

        self.assertIsInstance(result, SilkscreenResult)

    def test_hybrid_optimization(self):
        """Test hybrid optimization (SA + force-directed)."""
        config = SilkscreenConfig(optimization_method='hybrid')
        piston = SilkscreenPiston(config)
        result = piston.generate(self.parts_db, self.placement)

        self.assertIsInstance(result, SilkscreenResult)

    def test_no_optimization(self):
        """Test with optimization disabled."""
        config = SilkscreenConfig(optimization_method='none')
        piston = SilkscreenPiston(config)
        result = piston.generate(self.parts_db, self.placement)

        self.assertEqual(result.optimization_iterations, 0)


class TestPolarityMarkers(unittest.TestCase):
    """Test polarity and assembly marker generation."""

    def test_diode_cathode_marker(self):
        """Test cathode marker for diodes."""
        parts_db = {
            'parts': {
                'D1': {
                    'reference': 'D1',
                    'category': 'diode',
                    'body': {'width': 2.5, 'height': 1.0},
                }
            }
        }
        placement = {'D1': MockPosition(10, 10, 0)}

        config = SilkscreenConfig(show_polarity_markers=True)
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)

        self.assertGreater(len(result.polarity_markers), 0)

    def test_led_polarity_marker(self):
        """Test polarity marker for LEDs."""
        parts_db = {
            'parts': {
                'LED1': {
                    'reference': 'LED1',
                    'category': 'led',
                    'body': {'width': 1.6, 'height': 1.0},
                }
            }
        }
        placement = {'LED1': MockPosition(10, 10, 0)}

        config = SilkscreenConfig(show_polarity_markers=True)
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)

        self.assertGreater(len(result.polarity_markers), 0)

    def test_ic_pin1_marker(self):
        """Test pin 1 marker for ICs."""
        parts_db = {
            'parts': {
                'U1': {
                    'reference': 'U1',
                    'category': 'ic',
                    'body': {'width': 8.0, 'height': 8.0},
                }
            }
        }
        placement = {'U1': MockPosition(25, 20, 0)}

        config = SilkscreenConfig(show_pin1_markers=True)
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)

        self.assertGreater(len(result.assembly_markers), 0)


class TestComponentClassification(unittest.TestCase):
    """Test component category classification."""

    def test_classify_resistor(self):
        """Test resistor classification."""
        piston = SilkscreenPiston()
        category = piston._classify_component({'reference': 'R1'})
        self.assertEqual(category, ComponentCategory.RESISTOR)

    def test_classify_capacitor(self):
        """Test capacitor classification."""
        piston = SilkscreenPiston()
        category = piston._classify_component({'reference': 'C1'})
        self.assertEqual(category, ComponentCategory.CAPACITOR)

    def test_classify_ic(self):
        """Test IC classification."""
        piston = SilkscreenPiston()
        category = piston._classify_component({'reference': 'U1'})
        self.assertEqual(category, ComponentCategory.IC)

    def test_classify_by_category_field(self):
        """Test classification by explicit category field."""
        piston = SilkscreenPiston()
        category = piston._classify_component({
            'reference': 'X1',
            'category': 'crystal'
        })
        self.assertEqual(category, ComponentCategory.CRYSTAL)


class TestFiducials(unittest.TestCase):
    """Test fiducial marker generation."""

    def test_generate_fiducials(self):
        """Test fiducial generation."""
        parts_db = {'board_width': 100, 'board_height': 80, 'parts': {}}
        placement = {}

        config = SilkscreenConfig(add_fiducials=True)
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)

        self.assertGreater(len(result.fiducials), 0)
        # Default is 3 fiducials in corners
        self.assertEqual(len(result.fiducials), 3)

    def test_custom_fiducial_positions(self):
        """Test custom fiducial positions."""
        parts_db = {'board_width': 100, 'board_height': 80, 'parts': {}}
        placement = {}

        config = SilkscreenConfig(
            add_fiducials=True,
            fiducial_positions=[(5, 5), (95, 5), (5, 75), (95, 75)]
        )
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)

        self.assertEqual(len(result.fiducials), 4)


class TestComponentOutlines(unittest.TestCase):
    """Test component outline generation."""

    def test_ic_outline(self):
        """Test IC outline generation."""
        parts_db = {
            'parts': {
                'U1': {
                    'reference': 'U1',
                    'category': 'ic',
                    'body': {'width': 8.0, 'height': 8.0},
                }
            }
        }
        placement = {'U1': MockPosition(25, 20, 0)}

        config = SilkscreenConfig(generate_outlines=True)
        piston = SilkscreenPiston(config)
        result = piston.generate(parts_db, placement)

        # IC should have outline lines and an arc (notch)
        self.assertGreater(len(result.lines), 0)


class TestKiCadExport(unittest.TestCase):
    """Test KiCad S-expression export."""

    def test_to_kicad_sexpr(self):
        """Test export to KiCad format."""
        parts_db = {
            'parts': {
                'R1': {
                    'reference': 'R1',
                    'value': '10k',
                    'body': {'width': 1.6, 'height': 0.8},
                }
            }
        }
        placement = {'R1': MockPosition(10, 10, 0)}

        piston = SilkscreenPiston()
        result = piston.generate(parts_db, placement)
        sexpr = piston.to_kicad_sexpr()

        self.assertIsInstance(sexpr, str)
        self.assertIn('gr_text', sexpr)
        self.assertIn('R1', sexpr)


class TestStatistics(unittest.TestCase):
    """Test statistics generation."""

    def test_get_statistics(self):
        """Test getting detailed statistics."""
        parts_db = {
            'parts': {
                'R1': {'reference': 'R1', 'body': {'width': 1.6, 'height': 0.8}},
            }
        }
        placement = {'R1': MockPosition(10, 10, 0)}

        piston = SilkscreenPiston()
        piston.generate(parts_db, placement)
        stats = piston.get_statistics()

        self.assertIn('total_texts', stats)
        self.assertIn('reference_count', stats)
        self.assertIn('total_lines', stats)
        self.assertIn('manufacturing_standard', stats)


class TestManufacturingRules(unittest.TestCase):
    """Test manufacturing rule application."""

    def test_jlcpcb_rules(self):
        """Test JLCPCB manufacturing rules."""
        config = SilkscreenConfig(
            manufacturing_standard=ManufacturingStandard.JLCPCB,
            ref_text_size=0.5  # Below minimum
        )
        piston = SilkscreenPiston(config)

        # Config should be adjusted to meet minimum
        self.assertGreaterEqual(piston.config.ref_text_size,
                               MANUFACTURING_RULES[ManufacturingStandard.JLCPCB]['min_text_height'])


class TestEmptyInput(unittest.TestCase):
    """Test handling of empty input."""

    def test_empty_parts_db(self):
        """Test with empty parts database."""
        piston = SilkscreenPiston()
        result = piston.generate({'parts': {}}, {})

        self.assertIsInstance(result, SilkscreenResult)
        self.assertEqual(result.ref_count, 0)

    def test_no_placement(self):
        """Test with parts but no placement."""
        parts_db = {
            'parts': {
                'R1': {'reference': 'R1', 'body': {'width': 1.6, 'height': 0.8}}
            }
        }

        piston = SilkscreenPiston()
        result = piston.generate(parts_db, {})

        self.assertIsInstance(result, SilkscreenResult)
        self.assertEqual(result.ref_count, 0)


if __name__ == '__main__':
    unittest.main()
