"""
Machine Learning Engine for Circuit Intelligence
=================================================

This module provides ML capabilities for:
1. Learning from past designs (what worked, what failed)
2. Predicting issues before they happen
3. Improving recommendations over time
4. Pattern recognition beyond rule-based systems

Architecture:
    TRAINING DATA
    ├── Successful designs (DRC pass)
    ├── Failed designs (DRC fail + root cause)
    ├── Real-world feedback (post-production issues)
    └── Expert annotations

    MODELS
    ├── Issue Predictor (classification)
    ├── Placement Quality Scorer (regression)
    ├── Routing Difficulty Estimator (regression)
    ├── Component Recommender (collaborative filtering)
    └── Design Pattern Recognizer (clustering/classification)

Note: This is a framework. Actual training requires data collection.
For now, we use rule-based fallbacks with hooks for ML integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import json
import os
from pathlib import Path
import random
import math


# =============================================================================
# DATA STRUCTURES FOR LEARNING
# =============================================================================

@dataclass
class DesignSample:
    """A design sample for training."""
    design_id: str
    parts_db: Dict
    placement: Dict[str, Tuple[float, float]]
    routes: Optional[Dict] = None

    # Outcomes
    drc_passed: bool = False
    drc_errors: List[str] = field(default_factory=list)
    routing_success_rate: float = 0.0
    total_wirelength: float = 0.0

    # Expert annotations
    issues_found: List[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100

    # Real-world feedback (post-production)
    production_issues: List[str] = field(default_factory=list)
    thermal_issues: bool = False
    emi_issues: bool = False


@dataclass
class FeatureVector:
    """Features extracted from a design for ML."""
    # Board features
    board_area: float = 0.0
    component_count: int = 0
    net_count: int = 0
    component_density: float = 0.0  # components per cm²

    # Placement features
    avg_component_spacing: float = 0.0
    min_component_spacing: float = 0.0
    power_to_ic_distances: List[float] = field(default_factory=list)
    bypass_cap_distances: List[float] = field(default_factory=list)

    # Connectivity features
    avg_net_pin_count: float = 0.0
    max_net_pin_count: int = 0
    power_net_count: int = 0
    ground_net_count: int = 0

    # Pattern features
    has_switching_regulator: bool = False
    has_high_speed_signals: bool = False
    has_analog_section: bool = False

    def to_list(self) -> List[float]:
        """Convert to numeric list for ML models."""
        return [
            self.board_area,
            float(self.component_count),
            float(self.net_count),
            self.component_density,
            self.avg_component_spacing,
            self.min_component_spacing,
            self.avg_net_pin_count,
            float(self.max_net_pin_count),
            float(self.power_net_count),
            float(self.ground_net_count),
            float(self.has_switching_regulator),
            float(self.has_high_speed_signals),
            float(self.has_analog_section),
        ]


@dataclass
class Prediction:
    """A prediction from an ML model."""
    prediction: Any
    confidence: float
    explanation: str
    model_name: str


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Extracts features from designs for ML models."""

    def extract(self, parts_db: Dict, placement: Dict[str, Tuple[float, float]],
                board_width: float, board_height: float) -> FeatureVector:
        """Extract feature vector from a design."""
        features = FeatureVector()

        # Board features
        features.board_area = board_width * board_height
        features.component_count = len(parts_db.get('parts', {}))
        features.net_count = len(parts_db.get('nets', {}))
        features.component_density = features.component_count / (features.board_area / 100)  # per cm²

        # Placement features
        if placement:
            spacings = self._calculate_spacings(placement)
            features.avg_component_spacing = sum(spacings) / len(spacings) if spacings else 0
            features.min_component_spacing = min(spacings) if spacings else 0

            features.bypass_cap_distances = self._calculate_bypass_distances(parts_db, placement)

        # Connectivity features
        nets = parts_db.get('nets', {})
        pin_counts = [len(net.get('pins', [])) for net in nets.values()]
        features.avg_net_pin_count = sum(pin_counts) / len(pin_counts) if pin_counts else 0
        features.max_net_pin_count = max(pin_counts) if pin_counts else 0

        # Count power/ground nets
        for net_name in nets.keys():
            name_upper = net_name.upper()
            if any(p in name_upper for p in ('VCC', 'VDD', 'VIN', 'PWR', '+5', '+3')):
                features.power_net_count += 1
            if any(g in name_upper for g in ('GND', 'VSS', 'AGND', 'DGND')):
                features.ground_net_count += 1

        # Pattern features
        for ref, part in parts_db.get('parts', {}).items():
            value = part.get('value', '').upper()
            if any(sw in value for sw in ('LM2596', 'TPS54', 'MP1584', 'BUCK', 'BOOST')):
                features.has_switching_regulator = True
            if any(hs in value for hs in ('USB', 'ETH', 'DDR', 'LVDS')):
                features.has_high_speed_signals = True

        # Check for analog nets
        for net_name in nets.keys():
            if any(a in net_name.upper() for a in ('ADC', 'DAC', 'AIN', 'VREF')):
                features.has_analog_section = True
                break

        return features

    def _calculate_spacings(self, placement: Dict) -> List[float]:
        """Calculate pairwise component spacings."""
        refs = list(placement.keys())
        spacings = []

        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                pos1 = placement[refs[i]]
                pos2 = placement[refs[j]]
                dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                spacings.append(dist)

        return spacings

    def _calculate_bypass_distances(self, parts_db: Dict,
                                     placement: Dict) -> List[float]:
        """Calculate distances from bypass caps to ICs."""
        distances = []
        parts = parts_db.get('parts', {})

        # Find ICs (U*)
        ics = [ref for ref in parts.keys() if ref.startswith('U')]

        # Find bypass caps (100nF)
        caps = []
        for ref, part in parts.items():
            if ref.startswith('C'):
                value = part.get('value', '').lower()
                if '100n' in value or '0.1u' in value:
                    caps.append(ref)

        # Calculate distances
        for ic in ics:
            if ic not in placement:
                continue
            ic_pos = placement[ic]

            min_dist = float('inf')
            for cap in caps:
                if cap not in placement:
                    continue
                cap_pos = placement[cap]
                dist = math.sqrt((ic_pos[0] - cap_pos[0])**2 +
                                (ic_pos[1] - cap_pos[1])**2)
                min_dist = min(min_dist, dist)

            if min_dist < float('inf'):
                distances.append(min_dist)

        return distances


# =============================================================================
# MODELS (Placeholder implementations with rule-based fallbacks)
# =============================================================================

class IssuePredictor:
    """
    Predicts potential issues in a design.

    This is where ML would shine - learning from thousands of designs
    what combinations of features lead to problems.

    Current implementation: Rule-based with ML hooks.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.feature_extractor = FeatureExtractor()

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load trained model (placeholder)."""
        # In real implementation:
        # self.model = joblib.load(path)
        # or
        # self.model = tf.keras.models.load_model(path)
        pass

    def predict(self, parts_db: Dict, placement: Dict,
                board_width: float, board_height: float) -> List[Prediction]:
        """Predict potential issues."""
        features = self.feature_extractor.extract(parts_db, placement,
                                                   board_width, board_height)

        predictions = []

        # If ML model is available, use it
        if self.model:
            # ml_pred = self.model.predict(features.to_list())
            pass

        # Rule-based predictions (always available)
        predictions.extend(self._rule_based_predictions(features, parts_db, placement))

        return predictions

    def _rule_based_predictions(self, features: FeatureVector,
                                 parts_db: Dict, placement: Dict) -> List[Prediction]:
        """Rule-based issue prediction."""
        predictions = []

        # High density warning
        if features.component_density > 2.0:  # > 2 components per cm²
            predictions.append(Prediction(
                prediction='ROUTING_DIFFICULTY',
                confidence=0.7,
                explanation=f'High component density ({features.component_density:.1f}/cm²) may cause routing issues',
                model_name='rule_based'
            ))

        # Bypass cap distance warning
        if features.bypass_cap_distances:
            max_dist = max(features.bypass_cap_distances)
            if max_dist > 5.0:
                predictions.append(Prediction(
                    prediction='BYPASS_CAP_TOO_FAR',
                    confidence=0.9,
                    explanation=f'Bypass cap {max_dist:.1f}mm from IC (should be < 5mm)',
                    model_name='rule_based'
                ))

        # Switching regulator EMI warning
        if features.has_switching_regulator:
            predictions.append(Prediction(
                prediction='EMI_RISK',
                confidence=0.6,
                explanation='Switching regulator detected - check loop areas',
                model_name='rule_based'
            ))

        # Mixed signal warning
        if features.has_analog_section and features.has_switching_regulator:
            predictions.append(Prediction(
                prediction='NOISE_COUPLING_RISK',
                confidence=0.8,
                explanation='Mixed analog/switching - ensure proper separation',
                model_name='rule_based'
            ))

        return predictions


class PlacementScorer:
    """
    Scores placement quality.

    ML can learn what "good" placement looks like from successful designs.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def score(self, parts_db: Dict, placement: Dict,
              board_width: float, board_height: float) -> Tuple[float, List[str]]:
        """
        Score placement quality.

        Returns: (score 0-100, list of issues)
        """
        features = self.feature_extractor.extract(parts_db, placement,
                                                   board_width, board_height)

        score = 100.0
        issues = []

        # Penalize high density
        if features.component_density > 2.5:
            penalty = (features.component_density - 2.5) * 10
            score -= penalty
            issues.append(f'High density: -{penalty:.0f} points')

        # Penalize far bypass caps
        if features.bypass_cap_distances:
            far_caps = [d for d in features.bypass_cap_distances if d > 5.0]
            if far_caps:
                penalty = len(far_caps) * 5
                score -= penalty
                issues.append(f'{len(far_caps)} bypass cap(s) > 5mm: -{penalty:.0f} points')

        # Penalize tight spacing
        if features.min_component_spacing < 1.0:
            penalty = (1.0 - features.min_component_spacing) * 20
            score -= penalty
            issues.append(f'Tight spacing ({features.min_component_spacing:.1f}mm): -{penalty:.0f} points')

        # Bonus for good spacing
        if features.avg_component_spacing > 5.0:
            bonus = min(10, (features.avg_component_spacing - 5.0) * 2)
            score += bonus
            issues.append(f'Good spacing: +{bonus:.0f} points')

        return (max(0, min(100, score)), issues)


class RoutingDifficultyEstimator:
    """
    Estimates how difficult a design will be to route.

    ML can learn correlations between design features and routing outcomes.
    """

    def estimate(self, parts_db: Dict, placement: Dict,
                 board_width: float, board_height: float) -> Tuple[float, str]:
        """
        Estimate routing difficulty.

        Returns: (difficulty 0-1, explanation)
        """
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract(parts_db, placement,
                                              board_width, board_height)

        # Base difficulty
        difficulty = 0.0

        # Density factor
        density_factor = min(1.0, features.component_density / 5.0) * 0.3
        difficulty += density_factor

        # Net complexity factor
        complexity_factor = min(1.0, features.max_net_pin_count / 10) * 0.2
        difficulty += complexity_factor

        # Component count factor
        count_factor = min(1.0, features.component_count / 50) * 0.2
        difficulty += count_factor

        # Net count factor
        net_factor = min(1.0, features.net_count / 30) * 0.2
        difficulty += net_factor

        # Pattern factors
        if features.has_switching_regulator:
            difficulty += 0.05
        if features.has_high_speed_signals:
            difficulty += 0.05

        difficulty = min(1.0, difficulty)

        # Generate explanation
        if difficulty < 0.3:
            explanation = "Easy - should route without issues"
        elif difficulty < 0.5:
            explanation = "Moderate - may need some manual adjustment"
        elif difficulty < 0.7:
            explanation = "Challenging - expect some routing failures"
        else:
            explanation = "Difficult - likely needs board size increase or layer count"

        return (difficulty, explanation)


# =============================================================================
# LEARNING DATABASE
# =============================================================================

class LearningDatabase:
    """
    Database for storing and retrieving design samples for learning.

    In production, this would be backed by a proper database.
    For now, we use JSON files.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / 'learning_data.json')
        self.db_path = db_path
        self.samples: List[DesignSample] = []
        self._load()

    def _load(self):
        """Load samples from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    # Convert dicts back to DesignSample objects
                    # (simplified - real implementation would be more robust)
                    self.samples = data.get('samples', [])
            except Exception:
                self.samples = []

    def _save(self):
        """Save samples to disk."""
        with open(self.db_path, 'w') as f:
            json.dump({'samples': self.samples}, f, indent=2, default=str)

    def add_sample(self, sample: DesignSample):
        """Add a design sample to the database."""
        self.samples.append(sample.__dict__)
        self._save()

    def get_similar_designs(self, features: FeatureVector, n: int = 5) -> List[Dict]:
        """Find similar designs in the database."""
        # Simplified similarity - real implementation would use proper metrics
        return self.samples[:n]

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if not self.samples:
            return {'total': 0, 'passed': 0, 'failed': 0}

        passed = sum(1 for s in self.samples if s.get('drc_passed', False))
        return {
            'total': len(self.samples),
            'passed': passed,
            'failed': len(self.samples) - passed,
            'pass_rate': passed / len(self.samples) if self.samples else 0
        }


# =============================================================================
# ML ENGINE INTERFACE
# =============================================================================

class MLEngine:
    """
    Main interface for ML capabilities.

    Usage:
        ml = MLEngine()

        # Predict issues
        issues = ml.predict_issues(parts_db, placement, 50, 35)

        # Score placement
        score, feedback = ml.score_placement(parts_db, placement, 50, 35)

        # Estimate difficulty
        difficulty, explanation = ml.estimate_difficulty(parts_db, placement, 50, 35)

        # Record outcome for learning
        ml.record_outcome(parts_db, placement, drc_passed=True, errors=[])
    """

    def __init__(self, db_path: Optional[str] = None):
        self.issue_predictor = IssuePredictor()
        self.placement_scorer = PlacementScorer()
        self.difficulty_estimator = RoutingDifficultyEstimator()
        self.learning_db = LearningDatabase(db_path)
        self.feature_extractor = FeatureExtractor()

    def predict_issues(self, parts_db: Dict, placement: Dict,
                       board_width: float, board_height: float) -> List[Prediction]:
        """Predict potential issues in a design."""
        return self.issue_predictor.predict(parts_db, placement,
                                            board_width, board_height)

    def score_placement(self, parts_db: Dict, placement: Dict,
                        board_width: float, board_height: float) -> Tuple[float, List[str]]:
        """Score placement quality."""
        return self.placement_scorer.score(parts_db, placement,
                                           board_width, board_height)

    def estimate_difficulty(self, parts_db: Dict, placement: Dict,
                            board_width: float, board_height: float) -> Tuple[float, str]:
        """Estimate routing difficulty."""
        return self.difficulty_estimator.estimate(parts_db, placement,
                                                   board_width, board_height)

    def record_outcome(self, parts_db: Dict, placement: Dict,
                       drc_passed: bool, errors: List[str],
                       routing_success_rate: float = 0.0,
                       quality_score: float = 0.0):
        """Record design outcome for learning."""
        sample = DesignSample(
            design_id=f"design_{len(self.learning_db.samples) + 1}",
            parts_db=parts_db,
            placement=placement,
            drc_passed=drc_passed,
            drc_errors=errors,
            routing_success_rate=routing_success_rate,
            quality_score=quality_score
        )
        self.learning_db.add_sample(sample)

    def get_learning_stats(self) -> Dict:
        """Get learning database statistics."""
        return self.learning_db.get_statistics()
