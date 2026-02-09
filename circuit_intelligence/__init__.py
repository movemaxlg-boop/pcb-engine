"""
Circuit Intelligence Engine
============================

A separate engine that understands circuits like a human expert.
This engine provides circuit-level intelligence that complements
the geometry-focused PCB Engine.

Architecture:
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║              CIRCUIT INTELLIGENCE ENGINE (The Engineer's Brain)        ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  KNOWLEDGE BASE                    ANALYZERS                           ║
    ║  ├── PartsLibrary (indexed)        ├── CurrentFlowAnalyzer             ║
    ║  ├── ComponentDatabase             ├── ThermalAnalyzer                 ║
    ║  ├── PatternLibrary                └── PowerIntegrityAnalyzer          ║
    ║  └── ElectricalCalculator                                              ║
    ║                                                                        ║
    ║  INTELLIGENCE                      MACHINE LEARNING                    ║
    ║  ├── CircuitIntelligence (main)    ├── FeatureExtractor                ║
    ║  ├── DesignReviewAI                ├── IssuePredictor                  ║
    ║  └── ConstraintGenerator           ├── PlacementScorer                 ║
    ║                                    └── LearningDatabase                ║
    ║                                                                        ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                          OUTPUT TO PCB ENGINE                          ║
    ║  • PlacementConstraints (keep-together, critical placement)            ║
    ║  • RoutingConstraints (width, clearance, impedance)                    ║
    ║  • DesignIssues (warnings, recommendations)                            ║
    ╚═══════════════════════════════════════════════════════════════════════╝

The Circuit Intelligence Engine ADVISES.
The PCB Engine EXECUTES.

Usage:
    from circuit_intelligence import CircuitIntelligence, PartsLibrary

    # Initialize with parts library
    parts_lib = PartsLibrary()
    parts_lib.load_defaults()

    ci = CircuitIntelligence()
    analysis = ci.analyze(parts_db)
    constraints = ci.generate_constraints(analysis)

    # Feed constraints to PCB Engine
    pcb_engine.apply_constraints(constraints)
    pcb_engine.generate(parts_db)
"""

# Core types
from .circuit_types import (
    CircuitFunction, NetFunction, ComponentFunction,
    ComponentAnalysis, NetAnalysis, CircuitBlock,
    CurrentLoop, DesignIssue, CircuitAnalysis, DesignConstraints
)

# Knowledge base
from .pattern_library import PatternLibrary, CircuitPattern
from .component_database import ComponentDatabase, ComponentData
from .parts_library import PartsLibrary, Part, PartsIndex
from .electrical_calculator import ElectricalCalculator

# Analyzers
from .current_analyzer import CurrentFlowAnalyzer, PowerIntegrityAnalyzer
from .thermal_analyzer import ThermalAnalyzer

# Intelligence
from .circuit_intelligence import CircuitIntelligence
from .design_review import DesignReviewAI
from .constraint_generator import ConstraintGenerator

# Machine Learning
from .ml_engine import (
    MLEngine, FeatureExtractor, IssuePredictor,
    PlacementScorer, RoutingDifficultyEstimator, LearningDatabase
)

# Rules API (NEW in v2.0) - AI-reviewable design validation
from .rule_types import (
    RuleStatus, RuleCategory, RuleSeverity,
    RuleReport, ValidationResult, FeedbackResult, DesignReviewReport,
    create_pass_report, create_fail_report, create_warning_report,
)
from .rules_api import RulesAPI
from .feedback import AIFeedbackProcessor, FeedbackCommand, CommandParseError
from .design_review_report import (
    DesignReviewGenerator,
    generate_quick_report,
    validate_design_for_ai,
)

# Design Pipeline (updated in v2.0)
from .design_pipeline import (
    DesignPipeline, DesignContext, DesignPlan,
    BoardSpec, ComponentSpec, NetSpec,
    PlacementDecision, RoutingDecision, PlacementZone,
    create_pipeline, analyze_design,
)

__version__ = '2.0.0'
__all__ = [
    # Main entry point
    'CircuitIntelligence',

    # Core types
    'CircuitFunction', 'NetFunction', 'ComponentFunction',
    'ComponentAnalysis', 'NetAnalysis', 'CircuitBlock',
    'CurrentLoop', 'DesignIssue', 'CircuitAnalysis', 'DesignConstraints',

    # Knowledge base
    'PatternLibrary', 'CircuitPattern',
    'ComponentDatabase', 'ComponentData',
    'PartsLibrary', 'Part', 'PartsIndex',
    'ElectricalCalculator',

    # Analyzers
    'CurrentFlowAnalyzer', 'PowerIntegrityAnalyzer',
    'ThermalAnalyzer',

    # Intelligence
    'DesignReviewAI',
    'ConstraintGenerator',

    # Machine Learning
    'MLEngine', 'FeatureExtractor', 'IssuePredictor',
    'PlacementScorer', 'RoutingDifficultyEstimator', 'LearningDatabase',

    # Rules API (NEW in v2.0)
    'RuleStatus', 'RuleCategory', 'RuleSeverity',
    'RuleReport', 'ValidationResult', 'FeedbackResult', 'DesignReviewReport',
    'create_pass_report', 'create_fail_report', 'create_warning_report',
    'RulesAPI',
    'AIFeedbackProcessor', 'FeedbackCommand', 'CommandParseError',
    'DesignReviewGenerator', 'generate_quick_report', 'validate_design_for_ai',

    # Design Pipeline (updated in v2.0)
    'DesignPipeline', 'DesignContext', 'DesignPlan',
    'BoardSpec', 'ComponentSpec', 'NetSpec',
    'PlacementDecision', 'RoutingDecision', 'PlacementZone',
    'create_pipeline', 'analyze_design',
]
