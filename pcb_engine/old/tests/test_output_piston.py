#!/usr/bin/env python3
"""
Unit Tests for OutputPiston
===========================

Tests all functionality of the OutputPiston:
- KiCad PCB file generation
- Component placement output
- Track output
- Via output
- Silkscreen output
- Board outline generation
"""

import sys
import os
import unittest
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.output_piston import OutputPiston, OutputConfig, OutputResult
from pcb_engine.routing_types import Via


def make_segment(start, end, layer='F.Cu', width=0.25, net=''):
    """Create a segment dict for testing."""
    return {'start': start, 'end': end, 'layer': layer, 'width': width, 'net': net}


class TestOutputPistonInit(unittest.TestCase):
    """Test OutputPiston initialization."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_init(self):
        """Test default initialization."""
        config = OutputConfig(output_dir=self.test_dir)
        piston = OutputPiston(config)
        self.assertIsNotNone(piston)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = OutputConfig(
            output_dir=self.test_dir,
            board_name='test_board',
            board_width=100.0,
            board_height=80.0
        )
        piston = OutputPiston(config)
        self.assertEqual(piston.config.board_name, 'test_board')


class TestKiCadGeneration(unittest.TestCase):
    """Test KiCad PCB file generation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = OutputConfig(
            output_dir=self.test_dir,
            board_name='test_pcb',
            board_width=50.0,
            board_height=40.0
        )
        self.piston = OutputPiston(config)

        self.parts_db = {
            'parts': {
                'R1': {
                    'value': '10k',
                    'footprint': '0603',
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'offset': (-0.75, 0)},
                        {'number': '2', 'net': 'GND', 'offset': (0.75, 0)}
                    ]
                }
            },
            'nets': {
                'VCC': {'pins': [('R1', '1')]},
                'GND': {'pins': [('R1', '2')]}
            }
        }
        self.placement = {'R1': (25.0, 20.0)}
        self.routes = {}
        self.vias = []

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_generate_returns_result(self):
        """Test that generate returns OutputResult."""
        result = self.piston.generate(
            parts_db=self.parts_db,
            placement=self.placement,
            routes=self.routes,
            vias=self.vias
        )
        self.assertIsInstance(result, OutputResult)

    def test_generate_creates_file(self):
        """Test that generate creates a file."""
        result = self.piston.generate(
            parts_db=self.parts_db,
            placement=self.placement,
            routes=self.routes,
            vias=self.vias
        )
        self.assertGreater(len(result.files_generated), 0)
        # Check at least one file exists
        for filepath in result.files_generated:
            if filepath.endswith('.kicad_pcb'):
                self.assertTrue(os.path.exists(filepath))

    def test_kicad_file_has_header(self):
        """Test that generated file has KiCad header."""
        result = self.piston.generate(
            parts_db=self.parts_db,
            placement=self.placement,
            routes=self.routes,
            vias=self.vias
        )
        for filepath in result.files_generated:
            if filepath.endswith('.kicad_pcb'):
                with open(filepath, 'r') as f:
                    content = f.read()
                    self.assertIn('kicad_pcb', content)


class TestComponentOutput(unittest.TestCase):
    """Test component output in KiCad file."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = OutputConfig(
            output_dir=self.test_dir,
            board_name='test_components'
        )
        self.piston = OutputPiston(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_component_in_output(self):
        """Test that components appear in output."""
        parts_db = {
            'parts': {
                'U1': {'value': 'ESP32', 'footprint': 'QFN-48', 'pins': []},
                'C1': {'value': '100nF', 'footprint': '0603', 'pins': []}
            },
            'nets': {}
        }
        placement = {
            'U1': (25.0, 30.0),
            'C1': (10.0, 20.0)
        }
        result = self.piston.generate(
            parts_db=parts_db,
            placement=placement,
            routes={},
            vias=[]
        )

        for filepath in result.files_generated:
            if filepath.endswith('.kicad_pcb'):
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Component references should appear
                    self.assertIn('U1', content)
                    self.assertIn('C1', content)


class TestTrackOutput(unittest.TestCase):
    """Test track output in KiCad file."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = OutputConfig(
            output_dir=self.test_dir,
            board_name='test_tracks'
        )
        self.piston = OutputPiston(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tracks_in_output(self):
        """Test that tracks appear in output."""
        routes = {
            'VCC': {
                'segments': [
                    make_segment((10, 20), (30, 20), 'F.Cu', 0.25, 'VCC')
                ],
                'vias': [],
                'success': True
            }
        }
        result = self.piston.generate(
            parts_db={'parts': {}, 'nets': {}},
            placement={},
            routes=routes,
            vias=[]
        )

        for filepath in result.files_generated:
            if filepath.endswith('.kicad_pcb'):
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Track segment should appear
                    self.assertIn('segment', content)


class TestViaOutput(unittest.TestCase):
    """Test via output in KiCad file."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = OutputConfig(
            output_dir=self.test_dir,
            board_name='test_vias'
        )
        self.piston = OutputPiston(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_vias_in_output(self):
        """Test that vias appear in output."""
        vias = [
            Via((15.0, 25.0), 'VCC', diameter=0.8, drill=0.4)
        ]
        result = self.piston.generate(
            parts_db={'parts': {}, 'nets': {}},
            placement={},
            routes={},
            vias=vias
        )

        for filepath in result.files_generated:
            if filepath.endswith('.kicad_pcb'):
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Via should appear
                    self.assertIn('via', content)


class TestBoardOutline(unittest.TestCase):
    """Test board outline generation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_board_outline_in_output(self):
        """Test that board outline appears in output."""
        config = OutputConfig(
            output_dir=self.test_dir,
            board_name='test_outline',
            board_width=50.0,
            board_height=40.0
        )
        piston = OutputPiston(config)
        result = piston.generate(
            parts_db={'parts': {}, 'nets': {}},
            placement={},
            routes={},
            vias=[]
        )

        for filepath in result.files_generated:
            if filepath.endswith('.kicad_pcb'):
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Board outline should be on Edge.Cuts layer
                    self.assertIn('Edge.Cuts', content)


class TestOutputResult(unittest.TestCase):
    """Test output result structure."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = OutputConfig(output_dir=self.test_dir)
        self.piston = OutputPiston(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_result_has_files_generated(self):
        """Test that result has files_generated list."""
        result = self.piston.generate(
            parts_db={'parts': {}, 'nets': {}},
            placement={},
            routes={},
            vias=[]
        )
        self.assertIsInstance(result.files_generated, list)

    def test_result_success_flag(self):
        """Test that result has success flag."""
        result = self.piston.generate(
            parts_db={'parts': {}, 'nets': {}},
            placement={},
            routes={},
            vias=[]
        )
        self.assertTrue(result.success)


class TestSilkscreenOutput(unittest.TestCase):
    """Test silkscreen output."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = OutputConfig(
            output_dir=self.test_dir,
            board_name='test_silk'
        )
        self.piston = OutputPiston(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_silkscreen_labels(self):
        """Test that silkscreen labels appear."""
        parts_db = {
            'parts': {
                'R1': {'value': '10k', 'footprint': '0603', 'pins': []}
            },
            'nets': {}
        }
        placement = {'R1': (25.0, 20.0)}
        silkscreen = {
            'R1': {'position': (25.0, 18.0), 'text': 'R1', 'size': 1.0}
        }
        result = self.piston.generate(
            parts_db=parts_db,
            placement=placement,
            routes={},
            vias=[],
            silkscreen=silkscreen
        )
        # Should complete without error
        self.assertTrue(result.success)


if __name__ == '__main__':
    unittest.main()
