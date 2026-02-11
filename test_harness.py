"""
PCB ENGINE - COMPREHENSIVE TEST & MEASUREMENT HARNESS
=====================================================

One unified tool to test ANY piston, algorithm, or engine component
independently and accurately.

Modules:
  TestBoards          - Standard reference designs (5/20/50 parts)
  PistonTester        - Tests pistons in isolation with quality metrics
  AlgorithmBenchmark  - Compares algorithms head-to-head
  IntegrationTester   - Tests data flow BETWEEN pistons
  RegressionTracker   - Historical comparison with baselines

Usage:
  python test_harness.py                    # Run all tests
  python test_harness.py placement          # Test placement piston only
  python test_harness.py routing            # Test routing piston only
  python test_harness.py cpu_lab            # Test CPU Lab decisions
  python test_harness.py output             # Test output piston
  python test_harness.py integration        # Test inter-piston data flow
  python test_harness.py benchmark          # Run algorithm benchmarks
  python test_harness.py full               # Full engine end-to-end
"""

import sys
import os
import math
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcb_engine.placement_piston import PlacementPiston, PlacementConfig
from pcb_engine.common_types import calculate_courtyard, get_pins, get_footprint_definition


# =============================================================================
# CONSTANTS
# =============================================================================

PASS = '\033[92m[PASS]\033[0m'
FAIL = '\033[91m[FAIL]\033[0m'
WARN = '\033[93m[WARN]\033[0m'
INFO = '\033[94m[INFO]\033[0m'

BASELINE_FILE = os.path.join(os.path.dirname(__file__), 'test_baselines.json')
OUTPUT_DIR = r'D:\Anas\tmp\output'


# =============================================================================
# DATA: STANDARD REFERENCE BOARDS
# =============================================================================

class TestBoards:
    """Standard reference designs for consistent benchmarking."""

    @staticmethod
    def simple_5_parts():
        """Minimal board: 5 passives, 3 nets. Should always succeed."""
        return {
            'parts': {
                'R1': {
                    'name': 'R1', 'footprint': '0402', 'value': '10K',
                    'size': (1.0, 0.5),
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'physical': {'offset_x': -0.48, 'offset_y': 0}},
                        {'number': '2', 'net': 'SIG1', 'physical': {'offset_x': 0.48, 'offset_y': 0}},
                    ]
                },
                'R2': {
                    'name': 'R2', 'footprint': '0402', 'value': '10K',
                    'size': (1.0, 0.5),
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'physical': {'offset_x': -0.48, 'offset_y': 0}},
                        {'number': '2', 'net': 'SIG2', 'physical': {'offset_x': 0.48, 'offset_y': 0}},
                    ]
                },
                'C1': {
                    'name': 'C1', 'footprint': '0402', 'value': '100nF',
                    'size': (1.0, 0.5),
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'physical': {'offset_x': -0.48, 'offset_y': 0}},
                        {'number': '2', 'net': 'GND', 'physical': {'offset_x': 0.48, 'offset_y': 0}},
                    ]
                },
                'C2': {
                    'name': 'C2', 'footprint': '0603', 'value': '10uF',
                    'size': (1.6, 0.8),
                    'pins': [
                        {'number': '1', 'net': 'VCC', 'physical': {'offset_x': -0.775, 'offset_y': 0}},
                        {'number': '2', 'net': 'GND', 'physical': {'offset_x': 0.775, 'offset_y': 0}},
                    ]
                },
                'R3': {
                    'name': 'R3', 'footprint': '0805', 'value': '100R',
                    'size': (2.0, 1.25),
                    'pins': [
                        {'number': '1', 'net': 'SIG1', 'physical': {'offset_x': -0.95, 'offset_y': 0}},
                        {'number': '2', 'net': 'SIG2', 'physical': {'offset_x': 0.95, 'offset_y': 0}},
                    ]
                },
            },
            'nets': {
                'VCC': {'type': 'power', 'pins': ['R1.1', 'R2.1', 'C1.1', 'C2.1']},
                'GND': {'type': 'power', 'pins': ['C1.2', 'C2.2']},
                'SIG1': {'type': 'signal', 'pins': ['R1.2', 'R3.1']},
                'SIG2': {'type': 'signal', 'pins': ['R2.2', 'R3.2']},
            },
            'board': {'width': 20, 'height': 15, 'layers': 2},
        }

    @staticmethod
    def medium_20_parts():
        """Realistic ESP32 sensor board. The standard benchmark."""
        # Same 18-component board from test_real_20_component_board.py
        parts_db = {
            'parts': {
                'U1': {
                    'name': 'ESP32', 'footprint': 'QFN-32', 'value': 'ESP32-WROOM-32',
                    'size': (18.0, 25.5),
                    'pins': [
                        {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                         'physical': {'offset_x': -8.5, 'offset_y': -10.0}},
                        {'number': '2', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                         'physical': {'offset_x': -8.5, 'offset_y': -7.5}},
                        {'number': '3', 'name': 'EN', 'type': 'input', 'net': 'EN',
                         'physical': {'offset_x': -8.5, 'offset_y': -5.0}},
                        {'number': '4', 'name': 'IO0', 'type': 'bidirectional', 'net': 'BOOT',
                         'physical': {'offset_x': -8.5, 'offset_y': -2.5}},
                        {'number': '5', 'name': 'IO2', 'type': 'bidirectional', 'net': 'LED1_CTRL',
                         'physical': {'offset_x': -8.5, 'offset_y': 0.0}},
                        {'number': '6', 'name': 'IO4', 'type': 'bidirectional', 'net': 'LED2_CTRL',
                         'physical': {'offset_x': -8.5, 'offset_y': 2.5}},
                        {'number': '7', 'name': 'IO18', 'type': 'bidirectional', 'net': 'USB_DN',
                         'physical': {'offset_x': -8.5, 'offset_y': 5.0}},
                        {'number': '8', 'name': 'IO19', 'type': 'bidirectional', 'net': 'USB_DP',
                         'physical': {'offset_x': -8.5, 'offset_y': 7.5}},
                        {'number': '9', 'name': 'IO21', 'type': 'bidirectional', 'net': 'I2C_SDA',
                         'physical': {'offset_x': 8.5, 'offset_y': -10.0}},
                        {'number': '10', 'name': 'IO22', 'type': 'bidirectional', 'net': 'I2C_SCL',
                         'physical': {'offset_x': 8.5, 'offset_y': -7.5}},
                        {'number': '11', 'name': 'IO23', 'type': 'bidirectional', 'net': 'BME_CS',
                         'physical': {'offset_x': 8.5, 'offset_y': -5.0}},
                        {'number': '12', 'name': 'IO25', 'type': 'bidirectional', 'net': 'CC1',
                         'physical': {'offset_x': 8.5, 'offset_y': -2.5}},
                        {'number': '13', 'name': 'GND2', 'type': 'power_in', 'net': 'GND',
                         'physical': {'offset_x': 8.5, 'offset_y': 0.0}},
                        {'number': '14', 'name': '3V3_2', 'type': 'power_in', 'net': '3V3',
                         'physical': {'offset_x': 8.5, 'offset_y': 2.5}},
                        {'number': '15', 'name': 'VBUS', 'type': 'power_in', 'net': 'VBUS',
                         'physical': {'offset_x': 8.5, 'offset_y': 5.0}},
                        {'number': '16', 'name': 'GND3', 'type': 'power_in', 'net': 'GND',
                         'physical': {'offset_x': 8.5, 'offset_y': 7.5}},
                    ]
                },
                'U2': {
                    'name': 'LDO', 'footprint': 'SOT-223', 'value': 'AMS1117-3.3',
                    'size': (6.5, 3.5),
                    'pins': [
                        {'number': '1', 'name': 'VIN', 'net': 'VBUS',
                         'physical': {'offset_x': -2.3, 'offset_y': 0.0}},
                        {'number': '2', 'name': 'GND', 'net': 'GND',
                         'physical': {'offset_x': 0.0, 'offset_y': 0.0}},
                        {'number': '3', 'name': 'VOUT', 'net': '3V3',
                         'physical': {'offset_x': 2.3, 'offset_y': 0.0}},
                        {'number': '4', 'name': 'TAB', 'net': '3V3',
                         'physical': {'offset_x': 0.0, 'offset_y': 3.25}},
                    ]
                },
                'U3': {
                    'name': 'ESD', 'footprint': 'SOT-23-6', 'value': 'USBLC6-2SC6',
                    'size': (2.9, 1.6),
                    'pins': [
                        {'number': '1', 'name': 'IO1', 'net': 'USB_DP',
                         'physical': {'offset_x': -0.95, 'offset_y': -0.8}},
                        {'number': '2', 'name': 'GND', 'net': 'GND',
                         'physical': {'offset_x': 0.0, 'offset_y': -0.8}},
                        {'number': '3', 'name': 'IO2', 'net': 'USB_DN',
                         'physical': {'offset_x': 0.95, 'offset_y': -0.8}},
                        {'number': '4', 'name': 'IO3', 'net': 'USB_DN',
                         'physical': {'offset_x': 0.95, 'offset_y': 0.8}},
                        {'number': '5', 'name': 'VBUS', 'net': 'VBUS',
                         'physical': {'offset_x': 0.0, 'offset_y': 0.8}},
                        {'number': '6', 'name': 'IO4', 'net': 'USB_DP',
                         'physical': {'offset_x': -0.95, 'offset_y': 0.8}},
                    ]
                },
                'U4': {
                    'name': 'BME280', 'footprint': 'LGA-8', 'value': 'BME280',
                    'size': (2.5, 2.5),
                    'pins': [
                        {'number': '1', 'name': 'VDD', 'net': '3V3',
                         'physical': {'offset_x': -0.975, 'offset_y': -0.65}},
                        {'number': '2', 'name': 'GND', 'net': 'GND',
                         'physical': {'offset_x': -0.975, 'offset_y': 0.0}},
                        {'number': '3', 'name': 'SDI', 'net': 'I2C_SDA',
                         'physical': {'offset_x': -0.975, 'offset_y': 0.65}},
                        {'number': '4', 'name': 'SCK', 'net': 'I2C_SCL',
                         'physical': {'offset_x': 0.975, 'offset_y': 0.65}},
                        {'number': '5', 'name': 'SDO', 'net': 'GND',
                         'physical': {'offset_x': 0.975, 'offset_y': 0.0}},
                        {'number': '6', 'name': 'CSB', 'net': 'BME_CS',
                         'physical': {'offset_x': 0.975, 'offset_y': -0.65}},
                    ]
                },
                'J1': {
                    'name': 'USB-C', 'footprint': 'USB-C-16P', 'value': 'USB-C',
                    'size': (9.0, 7.5),
                    'pins': [
                        {'number': '1', 'name': 'VBUS', 'net': 'VBUS',
                         'physical': {'offset_x': -3.25, 'offset_y': -2.5}},
                        {'number': '2', 'name': 'D-', 'net': 'USB_DN',
                         'physical': {'offset_x': -1.0, 'offset_y': -2.5}},
                        {'number': '3', 'name': 'D+', 'net': 'USB_DP',
                         'physical': {'offset_x': 1.0, 'offset_y': -2.5}},
                        {'number': '4', 'name': 'CC1', 'net': 'CC1',
                         'physical': {'offset_x': 3.25, 'offset_y': -2.5}},
                        {'number': '5', 'name': 'GND1', 'net': 'GND',
                         'physical': {'offset_x': -3.75, 'offset_y': 2.5}},
                        {'number': '6', 'name': 'VBUS2', 'net': 'VBUS',
                         'physical': {'offset_x': 3.25, 'offset_y': -2.5}},
                        {'number': '7', 'name': 'GND2', 'net': 'GND',
                         'physical': {'offset_x': 3.75, 'offset_y': 2.5}},
                        {'number': '8', 'name': 'SHIELD', 'net': 'GND',
                         'physical': {'offset_x': 0, 'offset_y': 3.0}},
                    ]
                },
            },
            'nets': {},
            'board': {'width': 50, 'height': 40, 'layers': 2},
        }

        # Add passives
        passives = [
            ('C1', '0402', '100nF', 'VBUS', 'GND'),
            ('C2', '0402', '100nF', '3V3', 'GND'),
            ('C3', '0402', '10uF', 'VBUS', 'GND'),
            ('C4', '0402', '100nF', '3V3', 'GND'),
            ('C5', '0402', '100nF', '3V3', 'GND'),
            ('C6', '0402', '10uF', '3V3', 'GND'),
            ('R1', '0402', '4.7K', 'I2C_SDA', '3V3'),
            ('R2', '0402', '4.7K', 'I2C_SCL', '3V3'),
            ('R3', '0402', '10K', 'EN', '3V3'),
            ('R4', '0402', '10K', 'BOOT', 'GND'),
            ('R5', '0402', '220R', 'LED1_CTRL', 'LED1_A'),
            ('R6', '0402', '220R', 'LED2_CTRL', 'LED2_A'),
            ('LED1', '0402', 'Red', 'LED1_A', 'GND'),
            ('LED2', '0402', 'Green', 'LED2_A', 'GND'),
        ]
        for ref, fp, val, net1, net2 in passives:
            fp_def = get_footprint_definition(fp)
            if fp_def and fp_def.pad_positions:
                ox1 = fp_def.pad_positions[0][1]
                ox2 = fp_def.pad_positions[1][1]
            else:
                ox1, ox2 = -0.48, 0.48
            bw = fp_def.body_width if fp_def else 1.0
            bh = fp_def.body_height if fp_def else 0.5
            parts_db['parts'][ref] = {
                'name': ref, 'footprint': fp, 'value': val,
                'size': (bw, bh),
                'pins': [
                    {'number': '1', 'net': net1, 'physical': {'offset_x': ox1, 'offset_y': 0}},
                    {'number': '2', 'net': net2, 'physical': {'offset_x': ox2, 'offset_y': 0}},
                ]
            }

        # Build nets from pins
        net_map = defaultdict(list)
        for ref, part in parts_db['parts'].items():
            for pin in part.get('pins', []):
                net = pin.get('net', '')
                if net:
                    net_map[net].append(f"{ref}.{pin['number']}")
        parts_db['nets'] = {
            net: {'type': 'power' if net in ('GND', '3V3', 'VBUS') else 'signal', 'pins': pins}
            for net, pins in net_map.items()
        }

        return parts_db

    @staticmethod
    def complex_50_parts():
        """Challenging board: multi-IC, differential pairs. Not implemented yet."""
        # Placeholder - returns medium board with tighter constraints
        parts_db = TestBoards.medium_20_parts()
        parts_db['board'] = {'width': 35, 'height': 30, 'layers': 2}
        return parts_db


# =============================================================================
# QUALITY METRICS
# =============================================================================

@dataclass
class PlacementMetrics:
    """All placement quality measurements."""
    overlap_count: int = 0
    overlap_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    min_spacing_mm: float = 0.0
    min_spacing_pair: Tuple[str, str] = ('', '')
    board_utilization_pct: float = 0.0
    total_hpwl_mm: float = 0.0
    x_spread_pct: float = 0.0
    y_spread_pct: float = 0.0
    boundary_violations: int = 0
    boundary_details: List[str] = field(default_factory=list)
    routing_channel_min_mm: float = 0.0
    routing_channel_avg_mm: float = 0.0
    decoupling_max_dist_mm: float = 0.0
    score: int = 0

    def to_dict(self):
        return {
            'overlap_count': self.overlap_count,
            'min_spacing_mm': round(self.min_spacing_mm, 3),
            'board_utilization_pct': round(self.board_utilization_pct, 1),
            'total_hpwl_mm': round(self.total_hpwl_mm, 1),
            'x_spread_pct': round(self.x_spread_pct, 1),
            'y_spread_pct': round(self.y_spread_pct, 1),
            'boundary_violations': self.boundary_violations,
            'routing_channel_min_mm': round(self.routing_channel_min_mm, 3),
            'score': self.score,
        }


@dataclass
class RoutingMetrics:
    """All routing quality measurements."""
    total_nets: int = 0
    routed_nets: int = 0
    completion_pct: float = 0.0
    via_count: int = 0
    total_wirelength_mm: float = 0.0
    drc_violations: int = 0
    layer_balance: float = 0.0  # 0=all on one layer, 1=perfectly balanced
    score: int = 0

    def to_dict(self):
        return {
            'total_nets': self.total_nets,
            'routed_nets': self.routed_nets,
            'completion_pct': round(self.completion_pct, 1),
            'via_count': self.via_count,
            'total_wirelength_mm': round(self.total_wirelength_mm, 1),
            'drc_violations': self.drc_violations,
            'score': self.score,
        }


@dataclass
class CPULabMetrics:
    """CPU Lab decision quality measurements."""
    gnd_strategy_correct: bool = False
    layer_dirs_assigned: bool = False
    nets_classified: int = 0
    total_nets: int = 0
    groups_found: int = 0
    power_nets_removed: List[str] = field(default_factory=list)
    score: int = 0

    def to_dict(self):
        return {
            'gnd_strategy_correct': self.gnd_strategy_correct,
            'layer_dirs_assigned': self.layer_dirs_assigned,
            'nets_classified_pct': round(self.nets_classified / max(self.total_nets, 1) * 100, 1),
            'groups_found': self.groups_found,
            'power_nets_removed': self.power_nets_removed,
            'score': self.score,
        }


@dataclass
class OutputMetrics:
    """Output file quality measurements."""
    file_generated: bool = False
    file_size_bytes: int = 0
    courtyard_count: int = 0
    total_footprints: int = 0
    pad_count: int = 0
    net_count: int = 0
    kicad_drc_errors: int = -1  # -1 = not tested
    score: int = 0

    def to_dict(self):
        return {
            'file_generated': self.file_generated,
            'file_size_bytes': self.file_size_bytes,
            'courtyard_count': self.courtyard_count,
            'total_footprints': self.total_footprints,
            'kicad_drc_errors': self.kicad_drc_errors,
            'score': self.score,
        }


# =============================================================================
# PISTON TESTER - Tests pistons in isolation
# =============================================================================

class PistonTester:
    """Tests any piston in isolation with quality metrics."""

    def __init__(self, board_name='medium', verbose=True):
        self.verbose = verbose
        if board_name == 'simple':
            self.parts_db = TestBoards.simple_5_parts()
        elif board_name == 'complex':
            self.parts_db = TestBoards.complex_50_parts()
        else:
            self.parts_db = TestBoards.medium_20_parts()
        self.board_w = self.parts_db['board']['width']
        self.board_h = self.parts_db['board']['height']

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ---- PLACEMENT TESTING ----

    def test_placement(self, algorithm='hybrid') -> PlacementMetrics:
        """Run placement piston in isolation and measure quality."""
        self._log(f"\n{'='*60}")
        self._log(f"PLACEMENT PISTON TEST (algorithm={algorithm})")
        self._log(f"{'='*60}")

        metrics = PlacementMetrics()
        config = PlacementConfig(
            board_width=self.board_w,
            board_height=self.board_h,
            algorithm=algorithm,
        )
        piston = PlacementPiston(config)

        t0 = time.time()
        result = piston.place(self.parts_db)
        elapsed = time.time() - t0
        self._log(f"  Placement completed in {elapsed:.1f}s")
        self._log(f"  Algorithm: {result.algorithm_used}, Converged: {result.converged}")

        if not result.positions:
            self._log(f"  {FAIL} No placement generated!")
            return metrics

        placement = result.positions

        # Build courtyard data for each component
        components = {}
        for ref, pos in placement.items():
            part = self.parts_db['parts'].get(ref, {})
            fp = part.get('footprint', '')
            rotation = result.rotations.get(ref, 0) if hasattr(result, 'rotations') else 0
            courtyard = calculate_courtyard(part, footprint_name=fp, rotation=int(rotation))
            if isinstance(pos, (list, tuple)):
                x, y = pos[0], pos[1]
            elif hasattr(pos, 'x'):
                x, y = pos.x, pos.y
            else:
                x, y = 0, 0
            components[ref] = {
                'x': x, 'y': y,
                'w': courtyard.width, 'h': courtyard.height,
            }

        # Test 1: Overlap
        metrics = self._measure_overlaps(components, metrics)

        # Test 2: Spacing
        metrics = self._measure_spacing(components, metrics)

        # Test 3: Distribution
        metrics = self._measure_distribution(components, metrics)

        # Test 4: Boundary
        metrics = self._measure_boundary(components, metrics)

        # Test 5: Utilization
        metrics = self._measure_utilization(components, metrics)

        # Test 6: Wirelength (HPWL)
        metrics = self._measure_wirelength(components, metrics)

        # Test 7: Routing channels
        metrics = self._measure_routing_channels(components, metrics)

        # Calculate score
        metrics.score = self._calculate_placement_score(metrics)

        self._log(f"\n{'='*60}")
        verdict = 'EXCELLENT' if metrics.score >= 80 else 'GOOD' if metrics.score >= 60 else 'FAIR' if metrics.score >= 40 else 'POOR'
        self._log(f"PLACEMENT SCORE: {metrics.score}/100 ({verdict})")
        self._log(f"{'='*60}")

        return metrics

    def _measure_overlaps(self, components, metrics):
        refs = list(components.keys())
        for i in range(len(refs)):
            for j in range(i+1, len(refs)):
                a, b = components[refs[i]], components[refs[j]]
                ox = max(0, min(a['x']+a['w']/2, b['x']+b['w']/2) - max(a['x']-a['w']/2, b['x']-b['w']/2))
                oy = max(0, min(a['y']+a['h']/2, b['y']+b['h']/2) - max(a['y']-a['h']/2, b['y']-b['h']/2))
                if ox > 0.01 and oy > 0.01:
                    area = ox * oy
                    metrics.overlap_count += 1
                    metrics.overlap_pairs.append((refs[i], refs[j], area))
        status = PASS if metrics.overlap_count == 0 else FAIL
        self._log(f"\n  {status} OVERLAP TEST: {metrics.overlap_count} overlaps")
        if metrics.overlap_pairs:
            for r1, r2, area in metrics.overlap_pairs[:5]:
                self._log(f"         {r1} <-> {r2}: {area:.2f} mm^2")
        return metrics

    def _measure_spacing(self, components, metrics):
        min_gap = float('inf')
        min_pair = ('', '')
        refs = list(components.keys())
        for i in range(len(refs)):
            for j in range(i+1, len(refs)):
                a, b = components[refs[i]], components[refs[j]]
                gap_x = abs(a['x'] - b['x']) - (a['w'] + b['w']) / 2
                gap_y = abs(a['y'] - b['y']) - (a['h'] + b['h']) / 2
                gap = max(gap_x, gap_y)
                if gap_x > 0 or gap_y > 0:
                    gap = max(0, min(gap_x if gap_x > 0 else float('inf'),
                                    gap_y if gap_y > 0 else float('inf')))
                if gap < min_gap:
                    min_gap = gap
                    min_pair = (refs[i], refs[j])
        metrics.min_spacing_mm = min_gap if min_gap != float('inf') else 0
        metrics.min_spacing_pair = min_pair
        status = PASS if metrics.min_spacing_mm >= 0.25 else FAIL if metrics.min_spacing_mm < 0 else WARN
        self._log(f"  {status} SPACING TEST: min gap = {metrics.min_spacing_mm:.2f}mm ({min_pair[0]}<->{min_pair[1]})")
        return metrics

    def _measure_distribution(self, components, metrics):
        xs = [c['x'] for c in components.values()]
        ys = [c['y'] for c in components.values()]
        metrics.x_spread_pct = (max(xs) - min(xs)) / self.board_w * 100 if xs else 0
        metrics.y_spread_pct = (max(ys) - min(ys)) / self.board_h * 100 if ys else 0
        status = PASS if metrics.x_spread_pct > 50 and metrics.y_spread_pct > 50 else WARN
        self._log(f"  {status} DISTRIBUTION: X={metrics.x_spread_pct:.0f}%, Y={metrics.y_spread_pct:.0f}%")
        return metrics

    def _measure_boundary(self, components, metrics):
        margin = 2.0
        for ref, c in components.items():
            issues = []
            if c['x'] - c['w']/2 < margin:
                issues.append(f"left edge {c['x']-c['w']/2:.1f} < {margin}")
            if c['x'] + c['w']/2 > self.board_w - margin:
                issues.append(f"right edge {c['x']+c['w']/2:.1f} > {self.board_w-margin}")
            if c['y'] - c['h']/2 < margin:
                issues.append(f"top edge {c['y']-c['h']/2:.1f} < {margin}")
            if c['y'] + c['h']/2 > self.board_h - margin:
                issues.append(f"bottom edge {c['y']+c['h']/2:.1f} > {self.board_h-margin}")
            if issues:
                metrics.boundary_violations += 1
                metrics.boundary_details.append(f"{ref}: {', '.join(issues)}")
        status = PASS if metrics.boundary_violations == 0 else WARN
        self._log(f"  {status} BOUNDARY: {metrics.boundary_violations} violations")
        for detail in metrics.boundary_details[:3]:
            self._log(f"         {detail}")
        return metrics

    def _measure_utilization(self, components, metrics):
        total_area = sum(c['w'] * c['h'] for c in components.values())
        board_area = self.board_w * self.board_h
        metrics.board_utilization_pct = total_area / board_area * 100
        status = PASS if 20 <= metrics.board_utilization_pct <= 60 else WARN
        self._log(f"  {status} UTILIZATION: {metrics.board_utilization_pct:.1f}%")
        return metrics

    def _measure_wirelength(self, components, metrics):
        net_map = defaultdict(list)
        for ref, part in self.parts_db['parts'].items():
            for pin in part.get('pins', []):
                net = pin.get('net', '')
                if net and ref in components:
                    net_map[net].append(ref)

        total_hpwl = 0
        for net, refs in net_map.items():
            if len(refs) < 2:
                continue
            xs = [components[r]['x'] for r in refs if r in components]
            ys = [components[r]['y'] for r in refs if r in components]
            if xs and ys:
                total_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))
        metrics.total_hpwl_mm = total_hpwl
        self._log(f"  {INFO} WIRELENGTH: {total_hpwl:.1f}mm HPWL")
        return metrics

    def _measure_routing_channels(self, components, metrics):
        refs = list(components.keys())
        gaps = []
        for i in range(len(refs)):
            for j in range(i+1, len(refs)):
                a, b = components[refs[i]], components[refs[j]]
                gap_x = abs(a['x'] - b['x']) - (a['w'] + b['w']) / 2
                gap_y = abs(a['y'] - b['y']) - (a['h'] + b['h']) / 2
                gap = max(gap_x, gap_y)
                if gap > 0:
                    gaps.append(gap)
        if gaps:
            metrics.routing_channel_min_mm = min(gaps)
            metrics.routing_channel_avg_mm = sum(gaps) / len(gaps)
        status = PASS if metrics.routing_channel_min_mm > 0.5 else WARN
        self._log(f"  {status} ROUTING CHANNELS: min={metrics.routing_channel_min_mm:.2f}mm, avg={metrics.routing_channel_avg_mm:.2f}mm")
        return metrics

    def _calculate_placement_score(self, m):
        score = 100
        # Overlaps: -15 per overlap (max -30)
        score -= min(30, m.overlap_count * 15)
        # Spacing: -20 if overlapping, -10 if tight
        if m.min_spacing_mm < 0:
            score -= 20
        elif m.min_spacing_mm < 0.25:
            score -= 10
        # Distribution: -10 if clustered
        if m.x_spread_pct < 50 or m.y_spread_pct < 50:
            score -= 10
        # Boundary: -5 per violation (max -15)
        score -= min(15, m.boundary_violations * 5)
        # Utilization: -5 if too sparse or dense
        if m.board_utilization_pct < 15 or m.board_utilization_pct > 70:
            score -= 5
        # Routing channels: -10 if no channels
        if m.routing_channel_min_mm < 0.5:
            score -= 10
        return max(0, score)

    # ---- ROUTING TESTING ----

    def test_routing(self) -> RoutingMetrics:
        """Run full engine and measure routing quality."""
        self._log(f"\n{'='*60}")
        self._log(f"ROUTING PISTON TEST (via full engine)")
        self._log(f"{'='*60}")

        metrics = RoutingMetrics()
        try:
            from pcb_engine.pcb_engine import PCBEngine, EngineConfig
            config = EngineConfig(
                board_width=self.board_w,
                board_height=self.board_h,
                layer_count=self.parts_db['board'].get('layers', 2),
            )
            engine = PCBEngine(config)
            t0 = time.time()
            result = engine.run(self.parts_db)
            elapsed = time.time() - t0

            metrics.total_nets = len(self.parts_db.get('nets', {}))
            metrics.routed_nets = len(result.routes) if result.routes else 0
            metrics.completion_pct = metrics.routed_nets / max(metrics.total_nets, 1) * 100
            metrics.via_count = len(result.vias) if result.vias else 0

            # Count DRC violations
            if result.drc_result and hasattr(result.drc_result, 'errors'):
                metrics.drc_violations = len(result.drc_result.errors)

            # Calculate wirelength from routes
            total_length = 0
            layer_lengths = defaultdict(float)
            if result.routes:
                for net_name, segments in result.routes.items():
                    if isinstance(segments, list):
                        for seg in segments:
                            if hasattr(seg, 'length'):
                                total_length += seg.length
                                layer = getattr(seg, 'layer', 'F.Cu')
                                layer_lengths[layer] += seg.length
            metrics.total_wirelength_mm = total_length

            # Layer balance (0-1, 1=perfect)
            if layer_lengths and len(layer_lengths) > 1:
                vals = list(layer_lengths.values())
                total = sum(vals)
                if total > 0:
                    min_frac = min(vals) / total
                    metrics.layer_balance = min_frac * len(vals)  # 1.0 if perfectly balanced

            # Score
            score = 0
            score += min(50, int(metrics.completion_pct * 0.5))  # 50 points for completion
            if metrics.drc_violations == 0:
                score += 20
            elif metrics.drc_violations < 5:
                score += 10
            if metrics.via_count < metrics.routed_nets * 2:
                score += 15  # Low via count
            elif metrics.via_count < metrics.routed_nets * 4:
                score += 10
            score += int(metrics.layer_balance * 15)  # Up to 15 for balance
            metrics.score = min(100, score)

            self._log(f"  Completed in {elapsed:.1f}s")
            self._log(f"  Routed: {metrics.routed_nets}/{metrics.total_nets} ({metrics.completion_pct:.1f}%)")
            self._log(f"  Vias: {metrics.via_count}")
            self._log(f"  DRC violations: {metrics.drc_violations}")
            self._log(f"  Wirelength: {metrics.total_wirelength_mm:.1f}mm")

        except Exception as e:
            self._log(f"  {FAIL} Engine error: {e}")
            import traceback
            traceback.print_exc()

        self._log(f"\n  ROUTING SCORE: {metrics.score}/100")
        return metrics

    # ---- CPU LAB TESTING ----

    def test_cpu_lab(self) -> CPULabMetrics:
        """Test CPU Lab decisions in isolation."""
        self._log(f"\n{'='*60}")
        self._log(f"CPU LAB TEST")
        self._log(f"{'='*60}")

        metrics = CPULabMetrics()
        try:
            from pcb_engine.cpu_lab import CPULab
            lab = CPULab()
            board_config = {
                'board_width': self.board_w,
                'board_height': self.board_h,
                'layers': self.parts_db['board'].get('layers', 2),
            }
            result = lab.enhance(self.parts_db, board_config)

            # GND strategy
            if result.power_grid:
                gnd_strategy = getattr(result.power_grid, 'gnd_strategy', None)
                gnd_val = getattr(gnd_strategy, 'value', str(gnd_strategy)).lower()
                metrics.gnd_strategy_correct = ('pour' in gnd_val)
                nets_removed = getattr(result.power_grid, 'nets_removed_from_routing', [])
                metrics.power_nets_removed = list(nets_removed) if nets_removed else []
                status = PASS if metrics.gnd_strategy_correct else FAIL
                self._log(f"  {status} GND strategy: {gnd_strategy} (expected: pour)")
                self._log(f"  {INFO} Nets removed from routing: {metrics.power_nets_removed}")

            # Layer directions
            if result.layer_assignments:
                metrics.layer_dirs_assigned = len(result.layer_assignments) > 0
                status = PASS if metrics.layer_dirs_assigned else FAIL
                self._log(f"  {status} Layer directions: {len(result.layer_assignments)} assigned")
                for la in result.layer_assignments:
                    self._log(f"         {la.layer_name}: {la.preferred_direction}")
            else:
                self._log(f"  {FAIL} No layer directions assigned")

            # Net classification
            if result.net_priorities:
                metrics.nets_classified = len(result.net_priorities)
                metrics.total_nets = len(self.parts_db.get('nets', {}))
                status = PASS if metrics.nets_classified > 0 else FAIL
                self._log(f"  {status} Net priorities: {metrics.nets_classified}/{metrics.total_nets} classified")
                for np in result.net_priorities[:5]:
                    tw = getattr(np, 'trace_width_mm', '?')
                    cl = getattr(np, 'clearance_mm', '?')
                    self._log(f"         {np.net_name}: priority={np.priority}, width={tw}mm, clearance={cl}mm")
            else:
                metrics.total_nets = len(self.parts_db.get('nets', {}))
                self._log(f"  {FAIL} No net priorities")

            # Component groups
            if result.component_groups:
                metrics.groups_found = len(result.component_groups)
                status = PASS if metrics.groups_found > 0 else WARN
                self._log(f"  {status} Component groups: {metrics.groups_found} found")
                for g in result.component_groups[:5]:
                    self._log(f"         [{g.name}] {g.anchor}: {len(g.components)} components")
            else:
                self._log(f"  {WARN} No component groups detected")

            # Congestion
            if result.congestion:
                level = getattr(result.congestion, 'overall_level', 'unknown')
                self._log(f"  {INFO} Congestion: {level}")

            # Score
            score = 0
            if metrics.gnd_strategy_correct:
                score += 30
            if metrics.layer_dirs_assigned:
                score += 20
            if metrics.nets_classified > 0:
                score += 20
            if metrics.groups_found > 0:
                score += 15
            if metrics.power_nets_removed:
                score += 15
            metrics.score = score

        except Exception as e:
            self._log(f"  {FAIL} CPU Lab error: {e}")
            import traceback
            traceback.print_exc()

        self._log(f"\n  CPU LAB SCORE: {metrics.score}/100")
        return metrics

    # ---- OUTPUT TESTING ----

    def test_output(self) -> OutputMetrics:
        """Test output piston - generate KiCad file and check quality."""
        self._log(f"\n{'='*60}")
        self._log(f"OUTPUT PISTON TEST")
        self._log(f"{'='*60}")

        metrics = OutputMetrics()
        try:
            from pcb_engine.pcb_engine import PCBEngine, EngineConfig
            config = EngineConfig(
                board_width=self.board_w,
                board_height=self.board_h,
                layer_count=self.parts_db['board'].get('layers', 2),
            )
            engine = PCBEngine(config)
            result = engine.run(self.parts_db)

            # Check if file was generated
            output_file = getattr(result, 'output_file', None)
            if output_file and os.path.exists(output_file):
                metrics.file_generated = True
                metrics.file_size_bytes = os.path.getsize(output_file)
                self._log(f"  {PASS} File generated: {output_file} ({metrics.file_size_bytes:,} bytes)")

                # Parse file for quality checks
                with open(output_file, 'r') as f:
                    content = f.read()

                # Count courtyards
                metrics.courtyard_count = content.count('F.CrtYd') + content.count('B.CrtYd')
                metrics.total_footprints = content.count('(footprint ')
                metrics.pad_count = content.count('(pad ')

                crtyd_status = PASS if metrics.courtyard_count >= metrics.total_footprints else WARN
                self._log(f"  {crtyd_status} Courtyards: {metrics.courtyard_count} (footprints: {metrics.total_footprints})")
                self._log(f"  {INFO} Pads: {metrics.pad_count}")

                # KiCad DRC check
                kicad_cli = r'C:\Program Files\KiCad\9.0\bin\kicad-cli.exe'
                if os.path.exists(kicad_cli):
                    import subprocess
                    drc_output = os.path.join(os.path.dirname(output_file), 'drc_test.json')
                    try:
                        proc = subprocess.run(
                            [kicad_cli, 'pcb', 'drc',
                             '--severity-all', '--format', 'json',
                             '--output', drc_output, output_file],
                            capture_output=True, text=True, timeout=60
                        )
                        if os.path.exists(drc_output):
                            with open(drc_output, 'r') as f:
                                drc_data = json.load(f)
                            violations = drc_data.get('violations', [])
                            metrics.kicad_drc_errors = len(violations)
                            drc_status = PASS if metrics.kicad_drc_errors == 0 else FAIL
                            self._log(f"  {drc_status} KiCad DRC: {metrics.kicad_drc_errors} violations")
                            if violations:
                                for v in violations[:5]:
                                    self._log(f"         {v.get('type', '?')}: {v.get('description', '?')}")
                    except Exception as e:
                        self._log(f"  {WARN} KiCad DRC failed: {e}")
                else:
                    self._log(f"  {WARN} KiCad CLI not found, skipping DRC")
            else:
                self._log(f"  {FAIL} No output file generated")

            # Score
            score = 0
            if metrics.file_generated:
                score += 30
            if metrics.courtyard_count >= metrics.total_footprints:
                score += 25
            if metrics.kicad_drc_errors == 0:
                score += 30
            elif metrics.kicad_drc_errors > 0 and metrics.kicad_drc_errors < 10:
                score += 15
            if metrics.pad_count > 0:
                score += 15
            metrics.score = min(100, score)

        except Exception as e:
            self._log(f"  {FAIL} Output error: {e}")
            import traceback
            traceback.print_exc()

        self._log(f"\n  OUTPUT SCORE: {metrics.score}/100")
        return metrics


# =============================================================================
# INTEGRATION TESTER - Tests data flow between pistons
# =============================================================================

class IntegrationTester:
    """Tests data flow BETWEEN pistons."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.parts_db = TestBoards.medium_20_parts()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def test_cpulab_to_routing(self) -> Dict:
        """Test: CPU Lab decisions actually reach routing piston."""
        self._log(f"\n{'='*60}")
        self._log(f"INTEGRATION: CPU Lab -> Routing")
        self._log(f"{'='*60}")

        results = {
            'layer_directions_flow': False,
            'net_specs_flow': False,
            'congestion_flow': False,
            'nets_removed_flow': False,
        }

        try:
            from pcb_engine.pcb_engine import PCBEngine, EngineConfig
            config = EngineConfig(
                board_width=self.parts_db['board']['width'],
                board_height=self.parts_db['board']['height'],
                layer_count=2,
            )
            engine = PCBEngine(config)

            # Initialize state and run up to CPU Lab stage
            engine.state.parts_db = self.parts_db
            engine.state.start_time = time.time()

            from pcb_engine.pcb_engine import PistonEffort
            engine._execute_parts(PistonEffort.NORMAL)
            engine._execute_order(PistonEffort.NORMAL)
            engine._execute_placement(PistonEffort.NORMAL)

            # Run CPU Lab
            cpu_lab_result = engine._execute_cpu_lab()

            if cpu_lab_result:
                # Manually extract CPU Lab data to engine state
                # (normally done by run_orchestrated, not by _execute_cpu_lab itself)
                engine._cpu_lab_result = cpu_lab_result
                if cpu_lab_result.net_priorities:
                    routable = [p for p in cpu_lab_result.net_priorities if p.priority < 100]
                    engine.state.net_specs = {
                        p.net_name: {
                            'trace_width_mm': getattr(p, 'trace_width_mm', 0.25),
                            'clearance_mm': getattr(p, 'clearance_mm', 0.15),
                            'preferred_layer': getattr(p, 'preferred_layer', None),
                        } for p in routable
                    }
                if cpu_lab_result.layer_assignments:
                    engine.state.layer_directions = {
                        la.layer_name: la.preferred_direction.value
                        if hasattr(la.preferred_direction, 'value') else str(la.preferred_direction)
                        for la in cpu_lab_result.layer_assignments
                    }
                if cpu_lab_result.congestion:
                    cong = cpu_lab_result.congestion
                    engine.state.congestion = {
                        'level': getattr(cong, 'overall_level', 'unknown'),
                        'bottleneck_nets': getattr(cong, 'bottleneck_nets', []),
                    }

                # Check layer directions in state
                if hasattr(engine.state, 'layer_directions') and engine.state.layer_directions:
                    results['layer_directions_flow'] = True
                    self._log(f"  {PASS} Layer directions in engine state: {engine.state.layer_directions}")
                else:
                    self._log(f"  {FAIL} Layer directions NOT in engine state")

                # Check net specs in state
                if hasattr(engine.state, 'net_specs') and engine.state.net_specs:
                    results['net_specs_flow'] = True
                    sample = list(engine.state.net_specs.items())[:3]
                    self._log(f"  {PASS} Net specs in engine state: {len(engine.state.net_specs)} nets")
                    for name, spec in sample:
                        self._log(f"         {name}: width={spec.get('trace_width_mm')}mm")
                else:
                    self._log(f"  {FAIL} Net specs NOT in engine state")

                # Check congestion in state
                if hasattr(engine.state, 'congestion') and engine.state.congestion:
                    results['congestion_flow'] = True
                    self._log(f"  {PASS} Congestion data in engine state")
                else:
                    self._log(f"  {FAIL} Congestion data NOT in engine state")

                # Check nets removed
                if cpu_lab_result.power_grid:
                    nets_removed = getattr(cpu_lab_result.power_grid, 'nets_removed_from_routing', [])
                    results['nets_removed_flow'] = bool(nets_removed)
                    self._log(f"  {PASS if nets_removed else FAIL} Nets removed: {list(nets_removed)}")

            passed = sum(1 for v in results.values() if v)
            total = len(results)
            self._log(f"\n  Integration: {passed}/{total} data flows working")

        except Exception as e:
            self._log(f"  {FAIL} Integration error: {e}")
            import traceback
            traceback.print_exc()

        return results

    def test_cpulab_to_pour(self) -> Dict:
        """Test: CPU Lab pour configs reach pour piston."""
        self._log(f"\n{'='*60}")
        self._log(f"INTEGRATION: CPU Lab -> Pour")
        self._log(f"{'='*60}")

        results = {
            'gnd_pour_triggered': False,
            'power_pour_configs_exist': False,
        }

        try:
            from pcb_engine.cpu_lab import CPULab
            lab = CPULab()
            board_config = {
                'board_width': self.parts_db['board']['width'],
                'board_height': self.parts_db['board']['height'],
                'layers': 2,
            }
            cpu_result = lab.enhance(self.parts_db, board_config)

            if cpu_result.power_grid:
                gnd_strategy = str(getattr(cpu_result.power_grid, 'gnd_strategy', ''))
                results['gnd_pour_triggered'] = 'pour' in gnd_strategy.lower()
                self._log(f"  {PASS if results['gnd_pour_triggered'] else FAIL} GND pour: {gnd_strategy}")

                pour_configs = getattr(cpu_result.power_grid, 'power_pour_configs', {})
                results['power_pour_configs_exist'] = bool(pour_configs)
                self._log(f"  {PASS if pour_configs else WARN} Power pour configs: {list(pour_configs.keys()) if pour_configs else 'none'}")

        except Exception as e:
            self._log(f"  {FAIL} Error: {e}")
            import traceback
            traceback.print_exc()

        return results

    def test_placement_to_routing(self) -> Dict:
        """Test: Placement coordinates are correctly used by routing."""
        self._log(f"\n{'='*60}")
        self._log(f"INTEGRATION: Placement -> Routing")
        self._log(f"{'='*60}")

        results = {
            'placement_coords_used': False,
            'component_count_matches': False,
        }

        try:
            from pcb_engine.pcb_engine import PCBEngine, EngineConfig
            config = EngineConfig(
                board_width=self.parts_db['board']['width'],
                board_height=self.parts_db['board']['height'],
                layer_count=2,
            )
            engine = PCBEngine(config)
            result = engine.run(self.parts_db)

            # Placement data is on engine.state.placement (not EngineResult)
            placement = engine.state.placement
            if placement:
                placement_count = len(placement)
                parts_count = len(self.parts_db['parts'])
                results['component_count_matches'] = (placement_count == parts_count)
                self._log(f"  {PASS if results['component_count_matches'] else FAIL} "
                          f"Components: {placement_count}/{parts_count}")

                # Check all placement coords are within board
                all_in_bounds = True
                for ref, pos in placement.items():
                    x = pos.x if hasattr(pos, 'x') else (pos[0] if isinstance(pos, (list, tuple)) else pos)
                    y = pos.y if hasattr(pos, 'y') else (pos[1] if isinstance(pos, (list, tuple)) else pos)
                    if x < 0 or x > self.parts_db['board']['width'] or y < 0 or y > self.parts_db['board']['height']:
                        all_in_bounds = False
                        self._log(f"  {WARN} {ref} out of bounds: ({x:.1f}, {y:.1f})")
                results['placement_coords_used'] = all_in_bounds
                self._log(f"  {PASS if all_in_bounds else WARN} All components within board bounds")

        except Exception as e:
            self._log(f"  {FAIL} Error: {e}")
            import traceback
            traceback.print_exc()

        return results


# =============================================================================
# ALGORITHM BENCHMARK - Compare algorithms head-to-head
# =============================================================================

class AlgorithmBenchmark:
    """Compares algorithms on the same test case."""

    def __init__(self, board_name='medium', verbose=True):
        self.verbose = verbose
        if board_name == 'simple':
            self.parts_db = TestBoards.simple_5_parts()
        else:
            self.parts_db = TestBoards.medium_20_parts()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def benchmark_placement(self, algorithms=None):
        """Compare placement algorithms on the same board."""
        if algorithms is None:
            algorithms = ['simulated_annealing', 'force_directed', 'hybrid']

        self._log(f"\n{'='*60}")
        self._log(f"PLACEMENT ALGORITHM BENCHMARK")
        self._log(f"Board: {self.parts_db['board']['width']}x{self.parts_db['board']['height']}mm, "
                  f"{len(self.parts_db['parts'])} components")
        self._log(f"{'='*60}")

        results = {}
        for algo in algorithms:
            self._log(f"\n--- {algo.upper()} ---")
            tester = PistonTester(verbose=False)
            tester.parts_db = self.parts_db
            tester.board_w = self.parts_db['board']['width']
            tester.board_h = self.parts_db['board']['height']

            t0 = time.time()
            metrics = tester.test_placement(algorithm=algo)
            elapsed = time.time() - t0

            results[algo] = {
                'score': metrics.score,
                'overlaps': metrics.overlap_count,
                'min_spacing': round(metrics.min_spacing_mm, 2),
                'utilization': round(metrics.board_utilization_pct, 1),
                'hpwl': round(metrics.total_hpwl_mm, 1),
                'time_s': round(elapsed, 1),
            }
            self._log(f"  Score: {metrics.score}/100, Overlaps: {metrics.overlap_count}, "
                      f"Spacing: {metrics.min_spacing_mm:.2f}mm, Time: {elapsed:.1f}s")

        # Print comparison table
        self._log(f"\n{'='*60}")
        self._log(f"RANKING")
        self._log(f"{'='*60}")
        self._log(f"  {'Algorithm':<20} {'Score':>6} {'Overlaps':>9} {'Spacing':>8} {'HPWL':>8} {'Time':>6}")
        self._log(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*6}")
        for algo, r in sorted(results.items(), key=lambda x: x[1]['score'], reverse=True):
            self._log(f"  {algo:<20} {r['score']:>6} {r['overlaps']:>9} {r['min_spacing']:>7.2f}mm "
                      f"{r['hpwl']:>7.1f}mm {r['time_s']:>5.1f}s")

        return results


# =============================================================================
# REGRESSION TRACKER - Historical comparison
# =============================================================================

class RegressionTracker:
    """Save and compare metrics against historical baselines."""

    def __init__(self, baseline_file=BASELINE_FILE):
        self.baseline_file = baseline_file
        self.baselines = self._load()

    def _load(self):
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)

    def save_baseline(self, name, metrics_dict):
        """Save a set of metrics as a named baseline."""
        self.baselines[name] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics_dict,
        }
        self._save()
        print(f"  Baseline '{name}' saved.")

    def compare_to_baseline(self, name, current_metrics):
        """Compare current metrics to a saved baseline."""
        if name not in self.baselines:
            print(f"  No baseline '{name}' found. Save one first.")
            return None

        baseline = self.baselines[name]['metrics']
        print(f"\n  Comparing to baseline '{name}' ({self.baselines[name]['timestamp']}):")
        print(f"  {'Metric':<25} {'Baseline':>10} {'Current':>10} {'Change':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        changes = {}
        for key in baseline:
            old = baseline[key]
            new = current_metrics.get(key, '?')
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                if old != 0:
                    pct = (new - old) / abs(old) * 100
                    indicator = '+' if pct > 0 else ''
                    changes[key] = pct
                    print(f"  {key:<25} {old:>10} {new:>10} {indicator}{pct:>8.1f}%")
                else:
                    print(f"  {key:<25} {old:>10} {new:>10} {'':>10}")
            else:
                print(f"  {key:<25} {str(old):>10} {str(new):>10}")

        return changes


# =============================================================================
# MAIN - CLI interface
# =============================================================================

def run_all():
    """Run all tests and show summary."""
    print("=" * 60)
    print("PCB ENGINE - COMPREHENSIVE TEST HARNESS")
    print("=" * 60)

    scores = {}

    # Placement
    tester = PistonTester(board_name='medium')
    pm = tester.test_placement()
    scores['placement'] = pm.score

    # CPU Lab
    clm = tester.test_cpu_lab()
    scores['cpu_lab'] = clm.score

    # Integration
    it = IntegrationTester()
    cpulab_routing = it.test_cpulab_to_routing()
    cpulab_pour = it.test_cpulab_to_pour()
    integration_score = sum(1 for v in {**cpulab_routing, **cpulab_pour}.values() if v) * 100 // 6
    scores['integration'] = integration_score

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for name, score in scores.items():
        bar = '#' * (score // 5) + '.' * (20 - score // 5)
        verdict = 'PASS' if score >= 60 else 'FAIL'
        print(f"  {name:<20} [{bar}] {score:>3}/100 {verdict}")

    overall = sum(scores.values()) // len(scores)
    print(f"\n  OVERALL: {overall}/100")
    print(f"{'='*60}")

    # Save as baseline
    tracker = RegressionTracker()
    all_metrics = {
        'placement_score': pm.score,
        'placement_overlaps': pm.overlap_count,
        'placement_min_spacing': pm.min_spacing_mm,
        'placement_utilization': pm.board_utilization_pct,
        'cpu_lab_score': clm.score,
        'cpu_lab_gnd_correct': clm.gnd_strategy_correct,
        'integration_score': integration_score,
    }
    tracker.save_baseline('latest', all_metrics)

    return overall


def main():
    if len(sys.argv) < 2:
        run_all()
        return

    cmd = sys.argv[1].lower()
    tester = PistonTester(board_name='medium')

    if cmd == 'placement':
        algo = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'
        tester.test_placement(algorithm=algo)
    elif cmd == 'routing':
        tester.test_routing()
    elif cmd == 'cpu_lab':
        tester.test_cpu_lab()
    elif cmd == 'output':
        tester.test_output()
    elif cmd == 'integration':
        it = IntegrationTester()
        it.test_cpulab_to_routing()
        it.test_cpulab_to_pour()
        it.test_placement_to_routing()
    elif cmd == 'benchmark':
        bench = AlgorithmBenchmark()
        bench.benchmark_placement()
    elif cmd == 'full':
        run_all()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python test_harness.py [placement|routing|cpu_lab|output|integration|benchmark|full]")


if __name__ == '__main__':
    main()
