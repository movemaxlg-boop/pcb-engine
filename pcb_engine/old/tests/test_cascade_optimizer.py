#!/usr/bin/env python3
"""
Unit Tests for CascadeOptimizer
===============================

Tests all functionality of the CascadeOptimizer:
- Design profile classification
- Algorithm statistics tracking
- Dynamic cascade ordering
- Learning from success/failure
- Persistence (save/load)
- User preferences
"""

import sys
import os
import unittest
import tempfile
import shutil
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pcb_engine.cascade_optimizer import (
    CascadeOptimizer, CascadeOptimizerConfig, DesignProfile,
    DesignComplexity, DesignDensity, DesignType, LayerCount,
    AlgorithmStats, DEFAULT_CASCADES, create_optimizer, get_optimized_cascades
)


class TestDesignProfile(unittest.TestCase):
    """Test DesignProfile classification."""

    def test_simple_design_classification(self):
        """Test that simple designs are classified correctly."""
        parts_db = {
            'parts': {f'R{i}': {'footprint': '0603'} for i in range(10)},
            'nets': {f'NET{i}': {} for i in range(20)}
        }
        profile = DesignProfile.from_parts_db(parts_db, 50.0, 40.0)

        self.assertEqual(profile.complexity, DesignComplexity.SIMPLE)
        self.assertEqual(profile.component_count, 10)
        self.assertEqual(profile.net_count, 20)

    def test_complex_design_classification(self):
        """Test that complex designs are classified correctly."""
        parts_db = {
            'parts': {f'U{i}': {'footprint': 'QFN-48'} for i in range(200)},
            'nets': {f'NET{i}': {} for i in range(500)}
        }
        profile = DesignProfile.from_parts_db(parts_db, 100.0, 80.0)

        self.assertEqual(profile.complexity, DesignComplexity.COMPLEX)

    def test_bga_detection(self):
        """Test that BGA packages are detected."""
        parts_db = {
            'parts': {
                'U1': {'footprint': 'BGA-256'},
                'U2': {'footprint': 'QFP-100'},
                'C1': {'footprint': '0603'}
            },
            'nets': {}
        }
        profile = DesignProfile.from_parts_db(parts_db)

        self.assertEqual(profile.bga_count, 1)

    def test_qfn_detection(self):
        """Test that QFN packages are detected."""
        parts_db = {
            'parts': {
                'U1': {'footprint': 'QFN-48'},
                'U2': {'footprint': 'DFN-8'},
                'C1': {'footprint': '0603'}
            },
            'nets': {}
        }
        profile = DesignProfile.from_parts_db(parts_db)

        self.assertEqual(profile.qfn_count, 2)

    def test_power_net_detection(self):
        """Test that power nets are detected."""
        parts_db = {
            'parts': {},
            'nets': {
                'VCC': {},
                'GND': {},
                '3V3': {},
                'NET1': {},
                'NET2': {}
            }
        }
        profile = DesignProfile.from_parts_db(parts_db)

        self.assertEqual(profile.power_net_count, 3)

    def test_density_calculation(self):
        """Test density classification."""
        # Small board with many components = high density
        parts_db = {
            'parts': {f'R{i}': {} for i in range(100)},
            'nets': {}
        }
        profile = DesignProfile.from_parts_db(parts_db, 20.0, 20.0)

        self.assertIn(profile.density, [DesignDensity.HIGH, DesignDensity.ULTRA_HIGH])

    def test_profile_to_key(self):
        """Test that profile generates a consistent key."""
        profile = DesignProfile(
            complexity=DesignComplexity.MODERATE,
            density=DesignDensity.MEDIUM,
            design_type=DesignType.DIGITAL,
            layer_count=LayerCount.TWO_LAYER
        )
        key = profile.to_key()

        self.assertIn('moderate', key)
        self.assertIn('medium', key)
        self.assertIn('digital', key)


class TestAlgorithmStats(unittest.TestCase):
    """Test AlgorithmStats tracking."""

    def test_initial_success_rate(self):
        """Test that initial success rate is neutral."""
        stats = AlgorithmStats(algorithm='test')
        self.assertEqual(stats.success_rate, 0.5)  # Neutral prior

    def test_record_success(self):
        """Test recording a successful attempt."""
        stats = AlgorithmStats(algorithm='test')
        stats.record_attempt('profile_key', success=True, time_ms=100)

        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.failure_count, 0)
        self.assertEqual(stats.success_rate, 1.0)

    def test_record_failure(self):
        """Test recording a failed attempt."""
        stats = AlgorithmStats(algorithm='test')
        stats.record_attempt('profile_key', success=False, time_ms=200)

        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 1)
        self.assertEqual(stats.success_rate, 0.0)

    def test_profile_specific_success_rate(self):
        """Test profile-specific success tracking."""
        stats = AlgorithmStats(algorithm='test')
        stats.record_attempt('profile_A', success=True, time_ms=100)
        stats.record_attempt('profile_A', success=True, time_ms=100)
        stats.record_attempt('profile_B', success=False, time_ms=100)

        self.assertEqual(stats.get_profile_success_rate('profile_A'), 1.0)
        self.assertEqual(stats.get_profile_success_rate('profile_B'), 0.0)

    def test_serialization(self):
        """Test stats serialization and deserialization."""
        stats = AlgorithmStats(algorithm='hybrid')
        stats.record_attempt('profile_A', success=True, time_ms=500)
        stats.record_attempt('profile_A', success=False, time_ms=1000)

        # Serialize
        data = stats.to_dict()
        self.assertEqual(data['algorithm'], 'hybrid')
        self.assertEqual(data['success_count'], 1)
        self.assertEqual(data['failure_count'], 1)

        # Deserialize
        restored = AlgorithmStats.from_dict(data)
        self.assertEqual(restored.algorithm, 'hybrid')
        self.assertEqual(restored.success_rate, 0.5)


class TestCascadeOptimizerInit(unittest.TestCase):
    """Test CascadeOptimizer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        optimizer = CascadeOptimizer()
        self.assertIsNotNone(optimizer)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = CascadeOptimizerConfig(
            learning_rate=0.2,
            exploration_rate=0.15
        )
        optimizer = CascadeOptimizer(config)
        self.assertEqual(optimizer.config.learning_rate, 0.2)
        self.assertEqual(optimizer.config.exploration_rate, 0.15)


class TestCascadeOptimizerGetCascade(unittest.TestCase):
    """Test getting optimized cascades."""

    def setUp(self):
        self.optimizer = CascadeOptimizer()
        self.profile = DesignProfile.from_parts_db(
            {'parts': {'R1': {}}, 'nets': {}},
            50.0, 40.0
        )

    def test_get_routing_cascade(self):
        """Test getting routing cascade."""
        cascade = self.optimizer.get_cascade('routing', self.profile)

        self.assertIsInstance(cascade, list)
        self.assertGreater(len(cascade), 0)
        # Each item should be (algo_id, description)
        algo, desc = cascade[0]
        self.assertIsInstance(algo, str)
        self.assertIsInstance(desc, str)

    def test_get_placement_cascade(self):
        """Test getting placement cascade."""
        cascade = self.optimizer.get_cascade('placement', self.profile)

        self.assertIsInstance(cascade, list)
        self.assertGreater(len(cascade), 0)

    def test_cascade_has_all_algorithms(self):
        """Test that cascade includes all default algorithms."""
        cascade = self.optimizer.get_cascade('routing', self.profile)
        algo_ids = [algo for algo, _ in cascade]

        # Check that standard algorithms are present
        self.assertIn('hybrid', algo_ids)
        self.assertIn('astar', algo_ids)
        self.assertIn('lee', algo_ids)

    def test_unknown_piston_returns_empty(self):
        """Test that unknown piston returns empty cascade."""
        cascade = self.optimizer.get_cascade('unknown_piston', self.profile)
        self.assertEqual(cascade, [])


class TestCascadeOptimizerLearning(unittest.TestCase):
    """Test cascade learning from results."""

    def setUp(self):
        self.optimizer = CascadeOptimizer()
        self.profile = DesignProfile.from_parts_db(
            {'parts': {'R1': {}, 'R2': {}}, 'nets': {'NET1': {}}},
            50.0, 40.0
        )

    def test_record_result_updates_stats(self):
        """Test that recording results updates statistics."""
        self.optimizer.record_result(
            'routing', 'astar', self.profile,
            success=True, time_ms=500, quality_score=0.8
        )

        stats = self.optimizer.stats['routing']['astar']
        self.assertEqual(stats.success_count, 1)
        self.assertGreater(stats.avg_time_ms, 0)

    def test_successful_algorithm_promoted(self):
        """Test that consistently successful algorithms are promoted."""
        # Record multiple successes for 'astar'
        for _ in range(5):
            self.optimizer.record_result(
                'routing', 'astar', self.profile,
                success=True, time_ms=300
            )

        # Record failures for 'hybrid'
        for _ in range(5):
            self.optimizer.record_result(
                'routing', 'hybrid', self.profile,
                success=False, time_ms=1000
            )

        # Get cascade - astar should be higher than hybrid
        cascade = self.optimizer.get_cascade('routing', self.profile)
        algo_ids = [algo for algo, _ in cascade]

        astar_idx = algo_ids.index('astar')
        hybrid_idx = algo_ids.index('hybrid')
        self.assertLess(astar_idx, hybrid_idx)


class TestCascadeOptimizerPersistence(unittest.TestCase):
    """Test saving and loading cascade history."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.test_dir, 'cascade_history.json')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load(self):
        """Test saving and loading history."""
        # Create optimizer and record some results
        config = CascadeOptimizerConfig(
            history_file=self.history_file,
            auto_save=False
        )
        optimizer = CascadeOptimizer(config)
        profile = DesignProfile.from_parts_db({'parts': {'R1': {}}, 'nets': {}})

        optimizer.record_result('routing', 'astar', profile, success=True, time_ms=400)
        optimizer.record_result('routing', 'lee', profile, success=False, time_ms=1000)
        optimizer.save()

        # Verify file exists
        self.assertTrue(os.path.exists(self.history_file))

        # Load in new optimizer
        optimizer2 = CascadeOptimizer(config)
        loaded = optimizer2.load()
        self.assertTrue(loaded)

        # Verify stats were loaded
        self.assertEqual(optimizer2.stats['routing']['astar'].success_count, 1)
        self.assertEqual(optimizer2.stats['routing']['lee'].failure_count, 1)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        optimizer = CascadeOptimizer()
        loaded = optimizer.load('/nonexistent/path.json')
        self.assertFalse(loaded)


class TestUserPreferences(unittest.TestCase):
    """Test user algorithm preferences."""

    def setUp(self):
        self.optimizer = CascadeOptimizer()
        self.profile = DesignProfile()

    def test_set_user_preference(self):
        """Test setting user preferences."""
        self.optimizer.set_user_preference('routing', ['astar', 'lee', 'hybrid'])

        cascade = self.optimizer.get_cascade('routing', self.profile)
        algo_ids = [algo for algo, _ in cascade]

        # User preferences should be respected exactly
        self.assertEqual(algo_ids[0], 'astar')
        self.assertEqual(algo_ids[1], 'lee')
        self.assertEqual(algo_ids[2], 'hybrid')

    def test_clear_user_preference(self):
        """Test clearing user preferences."""
        self.optimizer.set_user_preference('routing', ['lee', 'astar'])
        self.optimizer.clear_user_preference('routing')

        # Should use default/learned order now
        cascade = self.optimizer.get_cascade('routing', self.profile)
        algo_ids = [algo for algo, _ in cascade]

        # Default has hybrid first
        self.assertEqual(algo_ids[0], 'hybrid')


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_create_optimizer(self):
        """Test create_optimizer function."""
        optimizer = create_optimizer()
        self.assertIsInstance(optimizer, CascadeOptimizer)

    def test_get_optimized_cascades(self):
        """Test get_optimized_cascades function."""
        parts_db = {
            'parts': {'R1': {'footprint': '0603'}},
            'nets': {'VCC': {}, 'GND': {}}
        }
        cascades = get_optimized_cascades(parts_db, 50.0, 40.0)

        self.assertIn('routing', cascades)
        self.assertIn('placement', cascades)
        self.assertIsInstance(cascades['routing'], list)


class TestStatistics(unittest.TestCase):
    """Test statistics retrieval."""

    def setUp(self):
        self.optimizer = CascadeOptimizer()
        self.profile = DesignProfile()

    def test_get_statistics(self):
        """Test getting statistics."""
        self.optimizer.record_result('routing', 'hybrid', self.profile, success=True)
        self.optimizer.record_result('routing', 'hybrid', self.profile, success=False)

        stats = self.optimizer.get_statistics('routing')

        self.assertIn('hybrid', stats)
        self.assertEqual(stats['hybrid']['success_rate'], 0.5)
        self.assertEqual(stats['hybrid']['total_attempts'], 2)

    def test_get_all_statistics(self):
        """Test getting all statistics."""
        self.optimizer.record_result('routing', 'hybrid', self.profile, success=True)
        self.optimizer.record_result('placement', 'sa', self.profile, success=True)

        stats = self.optimizer.get_statistics()

        self.assertIn('routing', stats)
        self.assertIn('placement', stats)


class TestRecommendations(unittest.TestCase):
    """Test algorithm recommendations."""

    def setUp(self):
        self.optimizer = CascadeOptimizer()

    def test_get_recommendations(self):
        """Test getting recommendations for a profile."""
        profile = DesignProfile.from_parts_db(
            {'parts': {'U1': {'footprint': 'BGA-256'}}, 'nets': {}},
            100.0, 80.0
        )
        recommendations = self.optimizer.get_recommendations(profile)

        self.assertIn('routing', recommendations)
        self.assertIn('placement', recommendations)
        # Each should have top 3 recommendations
        self.assertLessEqual(len(recommendations['routing']), 3)


if __name__ == '__main__':
    unittest.main()
