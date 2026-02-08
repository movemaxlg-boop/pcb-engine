"""
PCB Engine - Supervisor Module
==============================

The Supervisor is the "expert driver" that controls the engine.
It provides:
1. Step-by-step execution with approval gates
2. Learning from outputs, feedback, and errors
3. Real-time monitoring and intervention
4. Knowledge persistence across sessions

SUPERVISION PHILOSOPHY:
=======================
"The engine is the car, you are the expert driver."

The Supervisor:
- Controls when each phase executes
- Evaluates phase outputs before proceeding
- Intervenes when issues are detected
- Learns from every execution
- Suggests improvements based on history

LEARNING SYSTEM:
================
The engine learns from:
1. DRC violations → Placement/routing adjustments
2. User corrections → Updated preferences
3. Successful designs → Best practice patterns
4. Failed designs → What NOT to do

Knowledge is persisted in JSON for future sessions.
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import os


# =============================================================================
# LEARNING SYSTEM
# =============================================================================

@dataclass
class Lesson:
    """A single learned lesson"""
    category: str           # placement, routing, escape, etc.
    situation: str          # What was the context
    problem: str            # What went wrong (or right)
    solution: str           # What fixed it (or worked)
    confidence: float       # 0.0 to 1.0
    timestamp: str
    source: str             # 'drc', 'user', 'self', 'engine'

    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'situation': self.situation,
            'problem': self.problem,
            'solution': self.solution,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Lesson':
        return cls(**data)


@dataclass
class DesignPattern:
    """A successful design pattern to remember"""
    name: str
    description: str
    conditions: List[str]   # When to apply
    actions: List[str]      # What to do
    success_rate: float     # Historical success
    use_count: int
    last_used: str

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'conditions': self.conditions,
            'actions': self.actions,
            'success_rate': self.success_rate,
            'use_count': self.use_count,
            'last_used': self.last_used,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DesignPattern':
        return cls(**data)


class KnowledgeBase:
    """
    Persistent knowledge base for the engine.

    Stores lessons learned and patterns discovered.
    """

    def __init__(self, knowledge_path: str = None):
        self.knowledge_path = knowledge_path or 'pcb_engine_knowledge.json'
        self.lessons: List[Lesson] = []
        self.patterns: List[DesignPattern] = []
        self.statistics: Dict = {
            'total_designs': 0,
            'successful_designs': 0,
            'drc_violations_fixed': 0,
            'user_corrections': 0,
        }

        # Built-in lessons from our development history
        self._init_built_in_lessons()

        # Load persisted knowledge
        self.load()

    def _init_built_in_lessons(self):
        """Initialize with lessons learned during engine development"""
        built_in = [
            Lesson(
                category='escape',
                situation='Multi-pin hub component with destinations to the south',
                problem='Escape direction was EAST (+1, 0) while destinations were SOUTH',
                solution='Escape direction MUST point toward destination centroid, not natural pin side',
                confidence=1.0,
                timestamp='2025-01-01',
                source='engine',
            ),
            Lesson(
                category='corridor',
                situation='16 traces competing for same U-turn corridor',
                problem='Corridor capacity exceeded causing routing failure',
                solution='Validate corridor capacity BEFORE routing, not after',
                confidence=1.0,
                timestamp='2025-01-01',
                source='engine',
            ),
            Lesson(
                category='placement',
                situation='Hub component with many connections',
                problem='Hub rotation not optimized for destination alignment',
                solution='Rotate hub so most pins face their destinations directly',
                confidence=0.95,
                timestamp='2025-01-01',
                source='engine',
            ),
            Lesson(
                category='routing',
                situation='A* pathfinding produces step-by-step grid paths',
                problem='Raw paths have hundreds of tiny segments, looks unprofessional',
                solution='Apply path simplification: remove collinear points, line-of-sight optimization',
                confidence=1.0,
                timestamp='2025-01-01',
                source='engine',
            ),
            Lesson(
                category='routing',
                situation='Net routed late in sequence',
                problem='No path available because other nets blocked it',
                solution='Route most constrained nets first, not last',
                confidence=0.9,
                timestamp='2025-01-01',
                source='engine',
            ),
            Lesson(
                category='placement',
                situation='Force-directed placement with weak forces',
                problem='Components not reaching optimal positions',
                solution='Use simulated annealing with force-directed: high temp exploration, low temp refinement',
                confidence=0.85,
                timestamp='2025-01-01',
                source='engine',
            ),
        ]

        self.lessons.extend(built_in)

    def add_lesson(self, lesson: Lesson):
        """Add a new lesson"""
        # Check for duplicates (similar situation/problem)
        for existing in self.lessons:
            if (existing.category == lesson.category and
                existing.situation == lesson.situation and
                existing.problem == lesson.problem):
                # Update confidence if same lesson learned again
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.timestamp = lesson.timestamp
                return

        self.lessons.append(lesson)
        self.save()

    def add_pattern(self, pattern: DesignPattern):
        """Add or update a design pattern"""
        for existing in self.patterns:
            if existing.name == pattern.name:
                # Update existing pattern
                existing.use_count += 1
                existing.last_used = pattern.last_used
                # Update success rate with moving average
                existing.success_rate = (
                    existing.success_rate * 0.8 + pattern.success_rate * 0.2
                )
                return

        self.patterns.append(pattern)
        self.save()

    def get_relevant_lessons(self, category: str = None,
                              keywords: List[str] = None) -> List[Lesson]:
        """Get lessons relevant to current situation"""
        relevant = []

        for lesson in self.lessons:
            if category and lesson.category != category:
                continue

            if keywords:
                text = f"{lesson.situation} {lesson.problem} {lesson.solution}".lower()
                if not any(kw.lower() in text for kw in keywords):
                    continue

            relevant.append(lesson)

        # Sort by confidence (highest first)
        return sorted(relevant, key=lambda l: -l.confidence)

    def get_patterns_for_situation(self, conditions: List[str]) -> List[DesignPattern]:
        """Get patterns that match given conditions"""
        matching = []

        for pattern in self.patterns:
            # Check if any pattern conditions match
            matches = sum(
                1 for cond in pattern.conditions
                if any(c.lower() in cond.lower() for c in conditions)
            )
            if matches > 0:
                matching.append((pattern, matches))

        # Sort by match count and success rate
        matching.sort(key=lambda x: (-x[1], -x[0].success_rate))
        return [p for p, _ in matching]

    def record_drc_violation(self, violation_type: str, context: str,
                              fix_applied: str):
        """Record a DRC violation and its fix"""
        self.statistics['drc_violations_fixed'] += 1

        lesson = Lesson(
            category='drc',
            situation=context,
            problem=violation_type,
            solution=fix_applied,
            confidence=0.7,  # Start lower, increases with repetition
            timestamp=datetime.now().isoformat(),
            source='drc',
        )
        self.add_lesson(lesson)

    def record_user_correction(self, what_was_wrong: str, user_fix: str,
                                category: str = 'general'):
        """Record a user correction"""
        self.statistics['user_corrections'] += 1

        lesson = Lesson(
            category=category,
            situation='User reviewed output',
            problem=what_was_wrong,
            solution=user_fix,
            confidence=0.9,  # User corrections are high confidence
            timestamp=datetime.now().isoformat(),
            source='user',
        )
        self.add_lesson(lesson)

    def record_design_outcome(self, success: bool, drc_clean: bool,
                               details: Dict = None):
        """Record the outcome of a design run"""
        self.statistics['total_designs'] += 1
        if success and drc_clean:
            self.statistics['successful_designs'] += 1

        self.save()

    def save(self):
        """Save knowledge to file"""
        data = {
            'lessons': [l.to_dict() for l in self.lessons if l.source != 'engine'],
            'patterns': [p.to_dict() for p in self.patterns],
            'statistics': self.statistics,
        }

        try:
            with open(self.knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save knowledge: {e}")

    def load(self):
        """Load knowledge from file"""
        if not os.path.exists(self.knowledge_path):
            return

        try:
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for lesson_data in data.get('lessons', []):
                self.lessons.append(Lesson.from_dict(lesson_data))

            for pattern_data in data.get('patterns', []):
                self.patterns.append(DesignPattern.from_dict(pattern_data))

            self.statistics.update(data.get('statistics', {}))

        except Exception as e:
            print(f"Warning: Could not load knowledge: {e}")

    def get_summary(self) -> str:
        """Get knowledge base summary"""
        return (
            f"Knowledge Base Summary:\n"
            f"  Lessons: {len(self.lessons)}\n"
            f"  Patterns: {len(self.patterns)}\n"
            f"  Total designs: {self.statistics['total_designs']}\n"
            f"  Successful: {self.statistics['successful_designs']}\n"
            f"  DRC fixes: {self.statistics['drc_violations_fixed']}\n"
            f"  User corrections: {self.statistics['user_corrections']}"
        )


# =============================================================================
# APPROVAL GATES
# =============================================================================

class ApprovalStatus(Enum):
    """Status of an approval gate"""
    PENDING = 'pending'
    APPROVED = 'approved'
    REJECTED = 'rejected'
    AUTO_APPROVED = 'auto_approved'
    NEEDS_REVIEW = 'needs_review'


@dataclass
class ApprovalGate:
    """An approval checkpoint in the pipeline"""
    phase: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    auto_approve: bool = False
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    reviewer_notes: str = ''
    timestamp: str = ''

    def approve(self, notes: str = ''):
        """Approve this gate"""
        self.status = ApprovalStatus.APPROVED
        self.reviewer_notes = notes
        self.timestamp = datetime.now().isoformat()

    def reject(self, reason: str):
        """Reject this gate"""
        self.status = ApprovalStatus.REJECTED
        self.reviewer_notes = reason
        self.timestamp = datetime.now().isoformat()


# =============================================================================
# PHASE EVALUATORS
# =============================================================================

class PhaseEvaluator:
    """Evaluates phase outputs for quality"""

    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge

    def evaluate_placement(self, placement: Dict, parts_db: Dict,
                            hub: str) -> Tuple[bool, List[str], List[str]]:
        """
        Evaluate placement quality.

        Returns: (acceptable, issues, warnings)
        """
        issues = []
        warnings = []

        if not placement:
            issues.append("No components placed")
            return False, issues, warnings

        # Check hub is placed
        if hub and hub not in placement:
            issues.append(f"Hub component {hub} not placed")

        # Check for overlaps
        placed_refs = list(placement.keys())
        for i, ref1 in enumerate(placed_refs):
            pos1 = placement[ref1]
            part1 = parts_db['parts'].get(ref1, {})
            body1 = part1.get('physical', {}).get('body', (5, 5))

            for ref2 in placed_refs[i+1:]:
                pos2 = placement[ref2]
                part2 = parts_db['parts'].get(ref2, {})
                body2 = part2.get('physical', {}).get('body', (5, 5))

                # Check overlap
                dist = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)**0.5
                min_dist = (body1[0] + body2[0]) / 2

                if dist < min_dist:
                    issues.append(f"Overlap: {ref1} and {ref2}")

        # Apply lessons
        for lesson in self.knowledge.get_relevant_lessons('placement'):
            if 'rotation' in lesson.solution.lower():
                # Check if we're following hub rotation lessons
                warnings.append(f"Reminder: {lesson.solution}")

        acceptable = len(issues) == 0
        return acceptable, issues, warnings

    def evaluate_escapes(self, escapes: Dict, placement: Dict,
                          parts_db: Dict) -> Tuple[bool, List[str], List[str]]:
        """Evaluate escape calculation quality"""
        issues = []
        warnings = []

        if not escapes:
            warnings.append("No escape routes calculated")
            return True, issues, warnings

        for ref, pin_escapes in escapes.items():
            for pin_num, escape in pin_escapes.items():
                # Check escape endpoint is on grid
                endpoint = escape.endpoint
                grid_aligned = (
                    endpoint[0] % 0.5 < 0.01 and
                    endpoint[1] % 0.5 < 0.01
                )
                if not grid_aligned:
                    warnings.append(f"{ref}.{pin_num}: Escape not grid-aligned")

                # Check escape direction toward destination
                dest = getattr(escape, 'dest_centroid', None)
                if dest:
                    direction = escape.direction
                    to_dest = (dest[0] - escape.start[0], dest[1] - escape.start[1])
                    # Check if generally same direction
                    dot = direction[0] * to_dest[0] + direction[1] * to_dest[1]
                    if dot < 0:
                        warnings.append(
                            f"{ref}.{pin_num}: Escape direction AWAY from destination!"
                        )

        acceptable = len(issues) == 0
        return acceptable, issues, warnings

    def evaluate_routing(self, routes: Dict, failed: List[str],
                          quality_scores: Dict) -> Tuple[bool, List[str], List[str]]:
        """Evaluate routing quality"""
        issues = []
        warnings = []

        if failed:
            for net in failed:
                issues.append(f"Net '{net}' routing failed")

        # Check quality scores
        if quality_scores:
            poor_routes = [
                net for net, score in quality_scores.items()
                if score < 50
            ]
            if poor_routes:
                warnings.append(
                    f"Poor quality routes: {', '.join(poor_routes)}"
                )

        # Success rate
        total = len(routes) + len(failed)
        if total > 0:
            success_rate = len(routes) / total
            if success_rate < 0.9:
                warnings.append(f"Low success rate: {success_rate:.0%}")

        acceptable = len(issues) == 0
        return acceptable, issues, warnings

    def evaluate_validation(self, validation_result: Dict) -> Tuple[bool, List[str], List[str]]:
        """Evaluate DRC validation results"""
        issues = []
        warnings = []

        if not validation_result.get('valid', False):
            issues.append("Design failed DRC validation")

        violations = validation_result.get('violations', [])
        errors = validation_result.get('errors', 0)
        warns = validation_result.get('warnings', 0)

        if errors > 0:
            issues.append(f"{errors} DRC errors found")
            for v in violations[:5]:  # Show first 5
                issues.append(f"  - {v}")

        if warns > 0:
            warnings.append(f"{warns} DRC warnings found")

        acceptable = errors == 0
        return acceptable, issues, warnings


# =============================================================================
# SUPERVISOR
# =============================================================================

class Supervisor:
    """
    The expert driver that controls the PCB Engine.

    Provides:
    - Step-by-step execution with approval gates
    - Learning from outputs and feedback
    - Real-time monitoring
    - Intervention capabilities
    """

    def __init__(self, engine, knowledge_path: str = None):
        self.engine = engine
        self.knowledge = KnowledgeBase(knowledge_path)
        self.evaluator = PhaseEvaluator(self.knowledge)

        # Gates for each phase
        self.gates: Dict[str, ApprovalGate] = {}

        # Execution mode
        self.auto_approve_all = False
        self.stop_on_warning = False

        # Callbacks
        self.on_phase_complete: Optional[Callable] = None
        self.on_approval_needed: Optional[Callable] = None
        self.on_issue_detected: Optional[Callable] = None

    def configure(self, auto_approve: bool = False,
                   stop_on_warning: bool = False,
                   phases_to_pause: List[str] = None):
        """Configure supervision settings"""
        self.auto_approve_all = auto_approve
        self.stop_on_warning = stop_on_warning

        # Set which phases need manual approval
        phases_needing_approval = phases_to_pause or ['placement', 'routing', 'validation']

        for phase in ['parts', 'graph', 'hub', 'placement', 'escape',
                      'corridor', 'order', 'routing', 'validation']:
            self.gates[phase] = ApprovalGate(
                phase=phase,
                auto_approve=phase not in phases_needing_approval
            )

    def run_supervised(self, callback: Callable = None) -> Dict:
        """
        Run the engine with supervision.

        The callback is called after each phase with:
        - phase_name
        - gate (ApprovalGate)
        - engine_state

        Returns final result.
        """
        results = {
            'success': False,
            'phases_completed': [],
            'phases_failed': [],
            'gates': {},
            'lessons_learned': [],
        }

        # Apply lessons before starting
        self._apply_pre_run_lessons()

        phases = [
            ('parts', self._run_parts),
            ('graph', self._run_graph),
            ('hub', self._run_hub),
            ('placement', self._run_placement),
            ('escape', self._run_escape),
            ('corridor', self._run_corridor),
            ('order', self._run_order),
            ('routing', self._run_routing),
            ('validation', self._run_validation),
        ]

        for phase_name, phase_func in phases:
            print(f"\n{'='*60}")
            print(f"PHASE: {phase_name.upper()}")
            print('='*60)

            # Run phase
            try:
                phase_result = phase_func()
            except Exception as e:
                self.gates[phase_name].status = ApprovalStatus.REJECTED
                self.gates[phase_name].issues.append(f"Exception: {str(e)}")
                results['phases_failed'].append(phase_name)

                # Learn from failure
                self._learn_from_failure(phase_name, str(e))
                break

            # Evaluate phase output
            gate = self._evaluate_phase(phase_name, phase_result)
            results['gates'][phase_name] = gate

            # Call callback
            if callback:
                callback(phase_name, gate, self.engine.state)

            # Check if we should continue
            if gate.status == ApprovalStatus.REJECTED:
                results['phases_failed'].append(phase_name)
                break

            if gate.status == ApprovalStatus.NEEDS_REVIEW:
                if self.on_approval_needed:
                    self.on_approval_needed(phase_name, gate)
                # Wait for manual approval (in interactive mode)

            results['phases_completed'].append(phase_name)

        # Final assessment
        if 'validation' in results['phases_completed']:
            results['success'] = self.engine.state.validation.get('valid', False)

            # Record outcome
            self.knowledge.record_design_outcome(
                success=results['success'],
                drc_clean=results['success'],
                details={'phases': results['phases_completed']}
            )

        return results

    def _apply_pre_run_lessons(self):
        """Apply relevant lessons before running"""
        print("\nApplying learned lessons...")

        for lesson in self.knowledge.get_relevant_lessons():
            print(f"  [{lesson.category}] {lesson.solution[:60]}...")

    def _run_parts(self) -> Dict:
        """Run parts phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_0_PARTS)
        return {'parts_count': len(self.engine.state.parts_db.get('parts', {}))}

    def _run_graph(self) -> Dict:
        """Run graph phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_1_GRAPH)
        return {'edges': len(self.engine.state.graph.get('edges', []))}

    def _run_hub(self) -> Dict:
        """Run hub identification phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_2_HUB)
        return {'hub': self.engine.state.hub}

    def _run_placement(self) -> Dict:
        """Run placement phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_3_PLACEMENT)
        return {
            'placement': self.engine.state.placement,
            'parts_db': self.engine.state.parts_db,
            'hub': self.engine.state.hub,
        }

    def _run_escape(self) -> Dict:
        """Run escape phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_4_ESCAPE)
        return {
            'escapes': self.engine.state.escapes,
            'placement': self.engine.state.placement,
            'parts_db': self.engine.state.parts_db,
        }

    def _run_corridor(self) -> Dict:
        """Run corridor validation phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_5_CORRIDOR)
        return {'corridors': self.engine.state.corridors}

    def _run_order(self) -> Dict:
        """Run routing order phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_6_ORDER)
        return {'order': self.engine.state.route_order}

    def _run_routing(self) -> Dict:
        """Run routing phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_7_ROUTE)
        return {
            'routes': self.engine.state.routes,
            'failed': self.engine._router.failed if hasattr(self.engine, '_router') else [],
            'quality_scores': self.engine._router.quality_scores if hasattr(self.engine, '_router') else {},
        }

    def _run_validation(self) -> Dict:
        """Run validation phase"""
        from .engine import EnginePhase
        self.engine.run_phase(EnginePhase.PHASE_8_VALIDATE)
        return {'validation': self.engine.state.validation}

    def _evaluate_phase(self, phase_name: str, phase_result: Dict) -> ApprovalGate:
        """Evaluate a phase's output"""
        gate = self.gates[phase_name]

        acceptable = True
        issues = []
        warnings = []

        if phase_name == 'placement':
            acceptable, issues, warnings = self.evaluator.evaluate_placement(
                phase_result.get('placement', {}),
                phase_result.get('parts_db', {}),
                phase_result.get('hub'),
            )
        elif phase_name == 'escape':
            acceptable, issues, warnings = self.evaluator.evaluate_escapes(
                phase_result.get('escapes', {}),
                phase_result.get('placement', {}),
                phase_result.get('parts_db', {}),
            )
        elif phase_name == 'routing':
            acceptable, issues, warnings = self.evaluator.evaluate_routing(
                phase_result.get('routes', {}),
                phase_result.get('failed', []),
                phase_result.get('quality_scores', {}),
            )
        elif phase_name == 'validation':
            acceptable, issues, warnings = self.evaluator.evaluate_validation(
                phase_result.get('validation', {}),
            )

        gate.issues = issues
        gate.warnings = warnings
        gate.metrics = phase_result

        # Determine gate status
        if not acceptable:
            gate.status = ApprovalStatus.REJECTED
        elif issues:
            gate.status = ApprovalStatus.NEEDS_REVIEW
        elif warnings and self.stop_on_warning:
            gate.status = ApprovalStatus.NEEDS_REVIEW
        elif gate.auto_approve or self.auto_approve_all:
            gate.status = ApprovalStatus.AUTO_APPROVED
        else:
            gate.status = ApprovalStatus.NEEDS_REVIEW

        gate.timestamp = datetime.now().isoformat()

        # Print summary
        print(f"\nPhase {phase_name}: {gate.status.value}")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  ❌ {issue}")
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")

        return gate

    def _learn_from_failure(self, phase: str, error: str):
        """Learn from a phase failure"""
        lesson = Lesson(
            category=phase,
            situation=f"Phase {phase} execution",
            problem=f"Phase failed with: {error}",
            solution="Need investigation",  # Will be filled by user/manual analysis
            confidence=0.3,
            timestamp=datetime.now().isoformat(),
            source='self',
        )
        self.knowledge.add_lesson(lesson)

    def provide_feedback(self, phase: str, what_was_wrong: str,
                          how_to_fix: str):
        """
        Record user feedback about a phase.

        This is how the engine learns from the user.
        """
        self.knowledge.record_user_correction(
            what_was_wrong=what_was_wrong,
            user_fix=how_to_fix,
            category=phase,
        )
        print(f"✓ Feedback recorded for phase '{phase}'")

    def suggest_improvements(self, current_issues: List[str]) -> List[str]:
        """
        Suggest improvements based on knowledge base.

        Returns list of suggestions.
        """
        suggestions = []

        # Search for relevant lessons
        for issue in current_issues:
            keywords = issue.lower().split()[:3]  # First 3 words as keywords
            lessons = self.knowledge.get_relevant_lessons(keywords=keywords)

            for lesson in lessons[:2]:  # Top 2 most relevant
                if lesson.confidence > 0.5:
                    suggestions.append(f"[{lesson.category}] {lesson.solution}")

        return list(set(suggestions))  # Remove duplicates

    def get_session_report(self) -> str:
        """Generate a report for this session"""
        lines = [
            "=" * 60,
            "SUPERVISION SESSION REPORT",
            "=" * 60,
            "",
            "GATE STATUS:",
        ]

        for phase, gate in self.gates.items():
            status_icon = {
                ApprovalStatus.APPROVED: '✓',
                ApprovalStatus.AUTO_APPROVED: '✓',
                ApprovalStatus.REJECTED: '✗',
                ApprovalStatus.NEEDS_REVIEW: '?',
                ApprovalStatus.PENDING: '-',
            }.get(gate.status, '?')

            lines.append(f"  {status_icon} {phase}: {gate.status.value}")

            if gate.issues:
                for issue in gate.issues:
                    lines.append(f"      Issue: {issue}")

        lines.extend([
            "",
            self.knowledge.get_summary(),
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

class InteractiveSupervisor(Supervisor):
    """
    Interactive version of the Supervisor for CLI use.

    Prompts the user at each approval gate.
    """

    def request_approval(self, phase: str, gate: ApprovalGate) -> bool:
        """
        Request user approval for a gate.

        Returns True if approved, False if rejected.
        """
        print(f"\n{'='*60}")
        print(f"APPROVAL GATE: {phase.upper()}")
        print('='*60)

        print(f"\nStatus: {gate.status.value}")

        if gate.issues:
            print("\nIssues found:")
            for i, issue in enumerate(gate.issues, 1):
                print(f"  {i}. {issue}")

        if gate.warnings:
            print("\nWarnings:")
            for i, warning in enumerate(gate.warnings, 1):
                print(f"  {i}. {warning}")

        # Get suggestions
        all_issues = gate.issues + gate.warnings
        suggestions = self.suggest_improvements(all_issues)
        if suggestions:
            print("\nSuggestions from knowledge base:")
            for s in suggestions:
                print(f"  💡 {s}")

        # Prompt for approval
        print("\nOptions:")
        print("  [a] Approve and continue")
        print("  [r] Reject and stop")
        print("  [f] Provide feedback")
        print("  [s] Skip (auto-approve remaining)")

        while True:
            choice = input("\nYour choice: ").strip().lower()

            if choice == 'a':
                gate.approve("User approved")
                return True
            elif choice == 'r':
                reason = input("Reason for rejection: ").strip()
                gate.reject(reason)
                return False
            elif choice == 'f':
                what = input("What was wrong: ").strip()
                fix = input("How should it be fixed: ").strip()
                self.provide_feedback(phase, what, fix)
            elif choice == 's':
                self.auto_approve_all = True
                gate.approve("User skipped - auto-approving")
                return True
            else:
                print("Invalid choice. Please enter a, r, f, or s.")
