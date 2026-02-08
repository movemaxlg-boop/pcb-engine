#!/usr/bin/env python3
"""
Unit Tests for MonitorOptimizerBridge
=====================================

Tests all functionality of the MonitorOptimizerBridge:
- Session management
- Algorithm execution recording
- Cascade optimization
- Synchronization between monitor and optimizer
- Combined reporting
"""

import sys
import os
import unittest
import tempfile
import shutil
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.monitor_optimizer_bridge import (
    MonitorOptimizerBridge, BridgeConfig, BridgeStatistics,
    create_bridge, quick_optimize
)
from pcb_engine.cascade_optimizer import CascadeOptimizer, DesignProfile
from pcb_engine.bbl_monitor import BBLMonitor


class TestBridgeInit(unittest.TestCase):
    """Test MonitorOptimizerBridge initialization."""

    def test_default_init(self):
        """Test default initialization."""
        bridge = MonitorOptimizerBridge()
        self.assertIsNotNone(bridge)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = BridgeConfig(
            output_dir='./test_output',
            auto_sync=False,
            verbose=False
        )
        bridge = MonitorOptimizerBridge(config)
        self.assertEqual(bridge.config.output_dir, './test_output')
        self.assertFalse(bridge.config.auto_sync)

    def test_lazy_loading(self):
        """Test that monitor and optimizer are lazy-loaded."""
        bridge = MonitorOptimizerBridge()
        # Internal attributes should be None initially
        self.assertIsNone(bridge._monitor)
        self.assertIsNone(bridge._optimizer)


class TestSessionManagement(unittest.TestCase):
    """Test session management."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = BridgeConfig(output_dir=self.test_dir, verbose=False)
        self.bridge = MonitorOptimizerBridge(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_start_session(self):
        """Test starting a session."""
        self.bridge.start_session('TEST_001')
        self.assertTrue(self.bridge._session_active)
        self.assertEqual(self.bridge._session_id, 'TEST_001')

    def test_end_session(self):
        """Test ending a session."""
        self.bridge.start_session('TEST_002')
        self.bridge.end_session(success=True)
        self.assertFalse(self.bridge._session_active)

    def test_auto_generated_session_id(self):
        """Test auto-generated session ID."""
        self.bridge.start_session()
        self.assertTrue(self.bridge._session_id.startswith('BBL_'))

    def test_session_statistics_updated(self):
        """Test that session statistics are updated."""
        initial_count = self.bridge.stats.sessions_monitored
        self.bridge.start_session('TEST_003')
        self.assertEqual(self.bridge.stats.sessions_monitored, initial_count + 1)


class TestAlgorithmRecording(unittest.TestCase):
    """Test algorithm execution recording."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = BridgeConfig(output_dir=self.test_dir, verbose=False)
        self.bridge = MonitorOptimizerBridge(config)
        self.bridge.start_session('ALGO_TEST')

    def tearDown(self):
        self.bridge.end_session()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_algorithm_start(self):
        """Test recording algorithm start."""
        self.bridge.record_algorithm_start('routing', 'astar')
        # Should not raise any errors

    def test_record_algorithm_end(self):
        """Test recording algorithm end."""
        self.bridge.record_algorithm_start('routing', 'astar')
        self.bridge.record_algorithm_end(
            piston='routing',
            algorithm='astar',
            success=True,
            time_ms=500.0,
            quality_score=0.85
        )
        self.assertEqual(self.bridge.stats.algorithms_executed, 1)

    def test_record_cascade(self):
        """Test recording algorithm cascade."""
        self.bridge.record_cascade('routing', 'astar', 'lee')
        self.assertEqual(self.bridge.stats.cascades_triggered, 1)


class TestCascadeOptimization(unittest.TestCase):
    """Test cascade optimization."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = BridgeConfig(output_dir=self.test_dir, verbose=False)
        self.bridge = MonitorOptimizerBridge(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_optimized_cascade(self):
        """Test getting optimized cascade."""
        parts_db = {
            'parts': {'R1': {'footprint': '0603'}},
            'nets': {'VCC': {}, 'GND': {}}
        }
        cascade = self.bridge.get_optimized_cascade('routing', parts_db)

        self.assertIsInstance(cascade, list)
        self.assertGreater(len(cascade), 0)

    def test_get_recommendations(self):
        """Test getting recommendations."""
        parts_db = {
            'parts': {'R1': {}},
            'nets': {}
        }
        recs = self.bridge.get_recommendations(parts_db)

        # May be empty if no learning yet, but should be a dict
        self.assertIsInstance(recs, dict)

    def test_optimization_count_updated(self):
        """Test that optimization count is updated."""
        initial_count = self.bridge.stats.optimizations_applied
        parts_db = {'parts': {}, 'nets': {}}
        self.bridge.get_optimized_cascade('routing', parts_db)
        self.assertEqual(self.bridge.stats.optimizations_applied, initial_count + 1)


class TestSynchronization(unittest.TestCase):
    """Test synchronization between monitor and optimizer."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = BridgeConfig(
            output_dir=self.test_dir,
            verbose=False,
            auto_sync=False  # Manual sync for testing
        )
        self.bridge = MonitorOptimizerBridge(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sync_to_optimizer(self):
        """Test syncing to optimizer."""
        self.bridge.start_session('SYNC_TEST')
        self.bridge.record_algorithm_end('routing', 'astar', True, 500)
        self.bridge.record_algorithm_end('routing', 'lee', False, 1000)
        self.bridge.end_session(success=True)

        count = self.bridge.sync_to_optimizer()
        # Count may vary based on implementation
        self.assertIsInstance(count, int)
        self.assertEqual(self.bridge.stats.sync_operations, 1)

    def test_sync_from_optimizer(self):
        """Test syncing from optimizer."""
        stats = self.bridge.sync_from_optimizer()
        self.assertIsInstance(stats, dict)


class TestCombinedReporting(unittest.TestCase):
    """Test combined reporting."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = BridgeConfig(output_dir=self.test_dir, verbose=False)
        self.bridge = MonitorOptimizerBridge(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_combined_statistics(self):
        """Test getting combined statistics."""
        self.bridge.start_session('STATS_TEST')
        self.bridge.record_algorithm_end('routing', 'hybrid', True, 300)
        stats = self.bridge.get_combined_statistics()

        self.assertIn('bridge', stats)
        self.assertIn('session_active', stats)
        self.assertTrue(stats['session_active'])

        self.bridge.end_session()

    def test_get_performance_report(self):
        """Test getting performance report."""
        self.bridge.start_session('REPORT_TEST')
        self.bridge.record_algorithm_end('placement', 'sa', True, 400)
        self.bridge.end_session(success=True)

        report = self.bridge.get_performance_report()
        self.assertIsInstance(report, str)
        self.assertIn('BRIDGE', report.upper())

    def test_generate_dashboard_data(self):
        """Test generating dashboard data."""
        self.bridge.start_session('DASHBOARD_TEST')
        self.bridge.record_algorithm_end('routing', 'astar', True, 200)
        data = self.bridge.generate_dashboard_data()

        self.assertIn('timestamp', data)
        self.assertIn('bridge_stats', data)
        self.assertIn('session', data)

        self.bridge.end_session()


class TestUserPreferences(unittest.TestCase):
    """Test user preferences."""

    def setUp(self):
        self.bridge = MonitorOptimizerBridge(BridgeConfig(verbose=False))

    def test_set_user_preference(self):
        """Test setting user preferences."""
        self.bridge.set_user_preference('routing', ['astar', 'lee', 'hybrid'])
        # Should not raise any errors

    def test_clear_user_preference(self):
        """Test clearing user preferences."""
        self.bridge.set_user_preference('routing', ['astar', 'lee'])
        self.bridge.clear_user_preference('routing')
        # Should not raise any errors


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_create_bridge(self):
        """Test create_bridge function."""
        bridge = create_bridge(verbose=False)
        self.assertIsInstance(bridge, MonitorOptimizerBridge)

    def test_quick_optimize(self):
        """Test quick_optimize function."""
        parts_db = {
            'parts': {'R1': {'footprint': '0603'}},
            'nets': {'VCC': {}}
        }
        cascade = quick_optimize('routing', parts_db)

        self.assertIsInstance(cascade, list)


class TestBridgeStatistics(unittest.TestCase):
    """Test BridgeStatistics dataclass."""

    def test_to_dict(self):
        """Test statistics to_dict method."""
        stats = BridgeStatistics(
            sessions_monitored=5,
            algorithms_executed=100,
            cascades_triggered=10
        )
        data = stats.to_dict()

        self.assertEqual(data['sessions_monitored'], 5)
        self.assertEqual(data['algorithms_executed'], 100)
        self.assertEqual(data['cascades_triggered'], 10)


class TestIntegrationScenario(unittest.TestCase):
    """Test a full integration scenario."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config = BridgeConfig(
            output_dir=self.test_dir,
            verbose=False,
            persist_on_sync=False
        )
        self.bridge = MonitorOptimizerBridge(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_workflow(self):
        """Test a complete BBL-like workflow."""
        parts_db = {
            'parts': {
                'R1': {'footprint': '0603'},
                'R2': {'footprint': '0603'},
                'U1': {'footprint': 'SOIC-8'}
            },
            'nets': {
                'VCC': {},
                'GND': {},
                'NET1': {}
            }
        }

        # Start session
        self.bridge.start_session('FULL_WORKFLOW')

        # Get optimized cascade for placement
        placement_cascade = self.bridge.get_optimized_cascade('placement', parts_db)
        self.assertGreater(len(placement_cascade), 0)

        # Simulate placement execution
        self.bridge.record_algorithm_start('placement', 'hybrid')
        time.sleep(0.01)  # Simulate work
        self.bridge.record_algorithm_end('placement', 'hybrid', True, 150, 0.9)

        # Get optimized cascade for routing
        routing_cascade = self.bridge.get_optimized_cascade('routing', parts_db)
        self.assertGreater(len(routing_cascade), 0)

        # Simulate routing with cascade
        self.bridge.record_algorithm_start('routing', 'astar')
        time.sleep(0.01)
        self.bridge.record_algorithm_end('routing', 'astar', False, 500, 0.0)

        # Cascade to next algorithm
        self.bridge.record_cascade('routing', 'astar', 'lee')

        self.bridge.record_algorithm_start('routing', 'lee')
        time.sleep(0.01)
        self.bridge.record_algorithm_end('routing', 'lee', True, 800, 0.85)

        # End session
        routing_stats = {
            'completion': 1.0,
            'routed': 3,
            'total': 3
        }
        self.bridge.end_session(success=True, routing_stats=routing_stats)

        # Verify statistics
        stats = self.bridge.stats
        self.assertEqual(stats.sessions_monitored, 1)
        self.assertEqual(stats.algorithms_executed, 3)  # 1 placement + 2 routing
        self.assertEqual(stats.cascades_triggered, 1)
        self.assertGreaterEqual(stats.optimizations_applied, 2)

        # Get recommendations for future runs
        recs = self.bridge.get_recommendations(parts_db)
        # Routing should now prefer 'lee' since it succeeded
        # (depends on learning implementation)
        self.assertIsInstance(recs, dict)


if __name__ == '__main__':
    unittest.main()
