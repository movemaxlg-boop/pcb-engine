#!/usr/bin/env python3
"""
Unit Tests for PourPiston
=========================

Tests all functionality of the PourPiston:
- Copper pour zone generation
- Thermal relief configuration
- Keepout area creation
- KiCad zone export
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.pour_piston import (
    PourPiston, PourConfig, PourResult, PourZone,
    PourType, ThermalReliefStyle,
    create_ground_pour, create_power_pour
)


class TestPourPistonInit(unittest.TestCase):
    """Test PourPiston initialization."""

    def test_default_init(self):
        """Test default initialization."""
        piston = PourPiston()
        self.assertIsNotNone(piston)
        self.assertIsNotNone(piston.config)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = PourConfig(
            net="VCC",
            layer="F.Cu",
            clearance=0.4
        )
        piston = PourPiston(config)
        self.assertEqual(piston.config.net, "VCC")
        self.assertEqual(piston.config.layer, "F.Cu")
        self.assertEqual(piston.config.clearance, 0.4)


class TestPourConfig(unittest.TestCase):
    """Test PourConfig defaults and customization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PourConfig()
        self.assertEqual(config.net, "GND")
        self.assertEqual(config.layer, "B.Cu")
        self.assertEqual(config.pour_type, PourType.SOLID)
        self.assertEqual(config.clearance, 0.3)
        self.assertEqual(config.thermal_relief, ThermalReliefStyle.THERMAL)

    def test_pour_types(self):
        """Test different pour type configurations."""
        for pour_type in PourType:
            config = PourConfig(pour_type=pour_type)
            self.assertEqual(config.pour_type, pour_type)

    def test_thermal_relief_styles(self):
        """Test different thermal relief configurations."""
        for style in ThermalReliefStyle:
            config = PourConfig(thermal_relief=style)
            self.assertEqual(config.thermal_relief, style)


class TestPourGeneration(unittest.TestCase):
    """Test copper pour generation."""

    def setUp(self):
        """Set up test data."""
        self.parts_db = {
            'parts': {
                'R1': {
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (-0.8, 0), 'size': (0.8, 0.9)},
                        {'number': '2', 'net': 'GND', 'offset': (0.8, 0), 'size': (0.8, 0.9)},
                    ]
                },
                'C1': {
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (-0.8, 0), 'size': (0.8, 0.9)},
                        {'number': '2', 'net': 'GND', 'offset': (0.8, 0), 'size': (0.8, 0.9)},
                    ]
                },
            }
        }
        self.placement = {
            'R1': (10, 10),
            'C1': (20, 10),
        }

    def test_generate_produces_result(self):
        """Test that generate produces a PourResult."""
        piston = PourPiston()
        result = piston.generate(
            self.parts_db,
            self.placement,
            board_width=50,
            board_height=40
        )

        self.assertIsInstance(result, PourResult)
        self.assertTrue(result.success)
        self.assertGreater(len(result.zones), 0)

    def test_zone_has_correct_net(self):
        """Test that generated zone has correct net."""
        config = PourConfig(net="GND")
        piston = PourPiston(config)
        result = piston.generate(
            self.parts_db,
            self.placement,
            board_width=50,
            board_height=40
        )

        for zone in result.zones:
            self.assertEqual(zone.net, "GND")

    def test_zone_has_correct_layer(self):
        """Test that generated zone is on correct layer."""
        config = PourConfig(layer="B.Cu")
        piston = PourPiston(config)
        result = piston.generate(
            self.parts_db,
            self.placement,
            board_width=50,
            board_height=40
        )

        for zone in result.zones:
            self.assertEqual(zone.layer, "B.Cu")


class TestBoardOutline(unittest.TestCase):
    """Test board outline creation for pour zones."""

    def test_outline_with_edge_clearance(self):
        """Test outline respects edge clearance."""
        config = PourConfig(edge_clearance=0.5)
        piston = PourPiston(config)

        outline = piston._create_board_outline(50, 40)

        # First point should have clearance from edge
        self.assertGreaterEqual(outline[0][0], 0.5)
        self.assertGreaterEqual(outline[0][1], 0.5)

    def test_outline_closed_polygon(self):
        """Test that outline is a closed polygon."""
        piston = PourPiston()
        outline = piston._create_board_outline(50, 40)

        # First and last point should be the same
        self.assertEqual(outline[0], outline[-1])


class TestConnectedPads(unittest.TestCase):
    """Test finding pads connected to pour net."""

    def test_find_gnd_pads(self):
        """Test finding GND pads."""
        parts_db = {
            'parts': {
                'R1': {
                    'pins': [
                        {'number': '1', 'net': 'VCC'},
                        {'number': '2', 'net': 'GND'},
                    ]
                },
                'C1': {
                    'pins': [
                        {'number': '1', 'net': 'VCC'},
                        {'number': '2', 'net': 'GND'},
                    ]
                },
            }
        }
        placement = {'R1': (10, 10), 'C1': (20, 10)}

        config = PourConfig(net="GND")
        piston = PourPiston(config)
        connected = piston._find_connected_pads(parts_db, placement)

        # Should find 2 GND pads
        self.assertEqual(len(connected), 2)
        self.assertIn(('R1', '2'), connected)
        self.assertIn(('C1', '2'), connected)


class TestKeepoutCreation(unittest.TestCase):
    """Test keepout area creation."""

    def test_keepout_for_other_nets(self):
        """Test keepout creation for pads on other nets."""
        parts_db = {
            'parts': {
                'R1': {
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (0, 0), 'size': (1.0, 1.0)},
                    ]
                },
            }
        }
        placement = {'R1': (10, 10)}

        config = PourConfig(net="GND", clearance=0.3)
        piston = PourPiston(config)
        keepouts = piston._create_keepouts(parts_db, placement, None, None)

        # Should have keepout for VCC pad
        self.assertGreater(len(keepouts), 0)


class TestThermalRelief(unittest.TestCase):
    """Test thermal relief configuration."""

    def test_thermal_relief_settings(self):
        """Test thermal relief is configured in zone."""
        config = PourConfig(
            thermal_relief=ThermalReliefStyle.THERMAL,
            thermal_spoke_width=0.5,
            thermal_gap=0.5
        )
        piston = PourPiston(config)
        result = piston.generate({'parts': {}}, {}, 50, 40)

        for zone in result.zones:
            self.assertEqual(zone.thermal_relief, ThermalReliefStyle.THERMAL)
            self.assertEqual(zone.thermal_spoke_width, 0.5)
            self.assertEqual(zone.thermal_gap, 0.5)

    def test_solid_connection(self):
        """Test solid connection (no thermal relief)."""
        config = PourConfig(thermal_relief=ThermalReliefStyle.SOLID)
        piston = PourPiston(config)
        result = piston.generate({'parts': {}}, {}, 50, 40)

        for zone in result.zones:
            self.assertEqual(zone.thermal_relief, ThermalReliefStyle.SOLID)


class TestKiCadExport(unittest.TestCase):
    """Test KiCad zone export."""

    def test_to_kicad_zone(self):
        """Test export to KiCad zone format."""
        zone = PourZone(
            net="GND",
            layer="B.Cu",
            outline=[(0, 0), (50, 0), (50, 40), (0, 40), (0, 0)],
            clearance=0.3,
            priority=0,
            thermal_relief=ThermalReliefStyle.THERMAL,
            thermal_spoke_width=0.5,
            thermal_gap=0.5
        )

        piston = PourPiston()
        kicad_zone = piston.to_kicad_zone(zone, net_number=0)

        self.assertIsInstance(kicad_zone, str)
        self.assertIn('zone', kicad_zone)
        self.assertIn('GND', kicad_zone)
        self.assertIn('B.Cu', kicad_zone)
        self.assertIn('polygon', kicad_zone)

    def test_thermal_in_export(self):
        """Test thermal settings in export."""
        zone = PourZone(
            net="GND",
            layer="B.Cu",
            outline=[(0, 0), (50, 0), (50, 40), (0, 40), (0, 0)],
            clearance=0.3,
            priority=0,
            thermal_relief=ThermalReliefStyle.THERMAL,
            thermal_spoke_width=0.6,
            thermal_gap=0.4
        )

        piston = PourPiston()
        kicad_zone = piston.to_kicad_zone(zone)

        self.assertIn('thermal_gap', kicad_zone)
        self.assertIn('thermal_bridge_width', kicad_zone)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for creating pours."""

    def test_create_ground_pour(self):
        """Test create_ground_pour convenience function."""
        config = create_ground_pour(50, 40)

        self.assertEqual(config.net, "GND")
        self.assertEqual(config.layer, "B.Cu")
        self.assertEqual(config.thermal_relief, ThermalReliefStyle.THERMAL)

    def test_create_power_pour(self):
        """Test create_power_pour convenience function."""
        config = create_power_pour("VCC", 50, 40)

        self.assertEqual(config.net, "VCC")
        self.assertEqual(config.thermal_relief, ThermalReliefStyle.SOLID)
        self.assertGreater(config.priority, 0)

    def test_power_pour_layer(self):
        """Test power pour layer configuration."""
        config = create_power_pour("3V3", 50, 40, layer="F.Cu")
        self.assertEqual(config.layer, "F.Cu")


class TestPourZone(unittest.TestCase):
    """Test PourZone data structure."""

    def test_zone_creation(self):
        """Test creating a pour zone."""
        zone = PourZone(
            net="GND",
            layer="B.Cu",
            outline=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            clearance=0.3,
            priority=0,
            thermal_relief=ThermalReliefStyle.THERMAL,
            thermal_spoke_width=0.5,
            thermal_gap=0.5
        )

        self.assertEqual(zone.net, "GND")
        self.assertEqual(len(zone.outline), 5)
        self.assertFalse(zone.filled)  # Initially not filled

    def test_zone_fill_polygons(self):
        """Test zone fill polygon assignment."""
        zone = PourZone(
            net="GND",
            layer="B.Cu",
            outline=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            clearance=0.3,
            priority=0,
            thermal_relief=ThermalReliefStyle.THERMAL,
            thermal_spoke_width=0.5,
            thermal_gap=0.5
        )

        zone.fill_polygons = [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]
        zone.filled = True

        self.assertTrue(zone.filled)
        self.assertEqual(len(zone.fill_polygons), 1)


class TestPourResult(unittest.TestCase):
    """Test PourResult structure."""

    def test_result_structure(self):
        """Test that result has expected fields."""
        result = PourResult(
            success=True,
            zones=[],
            connected_pads=[]
        )

        self.assertTrue(result.success)
        self.assertIsInstance(result.zones, list)
        self.assertIsInstance(result.connected_pads, list)
        self.assertIsInstance(result.warnings, list)
        self.assertIsInstance(result.errors, list)


class TestTrackKeepout(unittest.TestCase):
    """Test track-to-keepout conversion."""

    def test_track_to_keepout(self):
        """Test converting track segment to keepout polygon."""
        piston = PourPiston()
        keepout = piston._track_to_keepout(
            start=(0, 0),
            end=(10, 0),
            width=0.5
        )

        self.assertIsNotNone(keepout)
        self.assertEqual(len(keepout), 5)  # 4 corners + close

    def test_zero_length_track(self):
        """Test handling zero-length track."""
        piston = PourPiston()
        keepout = piston._track_to_keepout(
            start=(0, 0),
            end=(0, 0),
            width=0.5
        )

        self.assertIsNone(keepout)

    def test_diagonal_track(self):
        """Test diagonal track keepout."""
        piston = PourPiston()
        keepout = piston._track_to_keepout(
            start=(0, 0),
            end=(10, 10),
            width=0.5
        )

        self.assertIsNotNone(keepout)
        self.assertEqual(len(keepout), 5)


class TestEmptyInput(unittest.TestCase):
    """Test handling of empty input."""

    def test_empty_parts_db(self):
        """Test with empty parts database."""
        piston = PourPiston()
        result = piston.generate({'parts': {}}, {}, 50, 40)

        self.assertIsInstance(result, PourResult)
        self.assertTrue(result.success)
        self.assertGreater(len(result.zones), 0)  # Should still create zone

    def test_no_connected_pads(self):
        """Test when no pads connect to pour net."""
        parts_db = {
            'parts': {
                'R1': {
                    'pins': [
                        {'number': '1', 'net': 'VCC'},
                        {'number': '2', 'net': 'VCC'},
                    ]
                },
            }
        }

        config = PourConfig(net="GND")  # No GND pads
        piston = PourPiston(config)
        result = piston.generate(parts_db, {'R1': (10, 10)}, 50, 40)

        self.assertEqual(len(result.connected_pads), 0)


class TestPourPriority(unittest.TestCase):
    """Test pour zone priority handling."""

    def test_priority_setting(self):
        """Test that priority is set correctly."""
        config = PourConfig(priority=5)
        piston = PourPiston(config)
        result = piston.generate({'parts': {}}, {}, 50, 40)

        for zone in result.zones:
            self.assertEqual(zone.priority, 5)

    def test_power_higher_priority(self):
        """Test that power pour has higher priority than ground."""
        gnd_config = create_ground_pour(50, 40)
        vcc_config = create_power_pour("VCC", 50, 40)

        self.assertGreater(vcc_config.priority, gnd_config.priority)


class TestHatchedPour(unittest.TestCase):
    """Test hatched copper pour configuration."""

    def test_hatched_config(self):
        """Test hatched pour configuration."""
        config = PourConfig(
            pour_type=PourType.HATCHED,
            hatch_width=0.5,
            hatch_gap=0.5,
            hatch_orientation=45.0
        )

        self.assertEqual(config.pour_type, PourType.HATCHED)
        self.assertEqual(config.hatch_width, 0.5)
        self.assertEqual(config.hatch_gap, 0.5)
        self.assertEqual(config.hatch_orientation, 45.0)


if __name__ == '__main__':
    unittest.main()
