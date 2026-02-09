"""
AI Feedback Processor for Circuit Intelligence Engine.

This module provides the AIFeedbackProcessor class that allows external AI agents
to review, validate, correct, or reject rule outcomes through feedback commands.

Commands supported:
- ACCEPT <rule_id>                              - Validate the rule outcome
- REJECT <rule_id> reason="..."                 - Reject with reasoning
- CORRECT <rule_id> action="..." value=...      - Provide a correction
- OVERRIDE <rule_id> new_threshold=... reason="..." - Override the threshold
- QUERY <rule_id> question="..."                - Ask for clarification
- EXPLAIN <rule_id>                             - Get detailed explanation
- BATCH_ACCEPT status=<status>                  - Accept all rules with status
- BATCH_REVIEW category=<category>              - Review rules by category

Author: Circuit Intelligence Engine
Version: 1.0.0
"""

import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .rule_types import (
    RuleStatus,
    RuleCategory,
    RuleReport,
    FeedbackResult,
    DesignReviewReport,
)


@dataclass
class FeedbackCommand:
    """Parsed feedback command structure."""

    command_type: str           # ACCEPT, REJECT, CORRECT, OVERRIDE, QUERY, EXPLAIN, BATCH_ACCEPT, BATCH_REVIEW
    rule_id: Optional[str]      # Target rule ID (None for batch commands)
    parameters: Dict[str, Any]  # Command parameters
    raw_command: str            # Original command string
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RuleContext:
    """Context for a rule that has been executed."""

    rule_id: str
    report: RuleReport
    original_threshold: Any
    current_threshold: Any
    feedback_history: List[FeedbackResult] = field(default_factory=list)
    status: RuleStatus = RuleStatus.PENDING


class CommandParseError(Exception):
    """Raised when a feedback command cannot be parsed."""
    pass


class RuleNotFoundError(Exception):
    """Raised when a rule ID is not found in the context."""
    pass


class AIFeedbackProcessor:
    """
    Process feedback commands from external AI agents.

    This processor allows AI agents to:
    - ACCEPT: Validate that a rule outcome is correct
    - REJECT: Reject an outcome with reasoning
    - CORRECT: Provide corrections to fix violations
    - OVERRIDE: Override rule thresholds with justification
    - QUERY: Ask questions about rules
    - EXPLAIN: Get detailed rule explanations

    Usage:
        processor = AIFeedbackProcessor()

        # Add rule reports for review
        processor.add_report(usb_report)
        processor.add_report(thermal_report)

        # Process AI feedback
        result = processor.process_command("ACCEPT USB2_LENGTH_MATCHING")
        result = processor.process_command('REJECT THERMAL_VIA_COUNT reason="Design uses external heatsink"')
        result = processor.process_command('CORRECT USB2_LENGTH_MATCHING action="extend_d_plus" value=1.5')
        result = processor.process_command('OVERRIDE USB2_LENGTH_MATCHING new_threshold=5.0 reason="Using Full-Speed mode"')
    """

    # Command patterns for parsing
    COMMAND_PATTERNS = {
        'ACCEPT': r'^ACCEPT\s+(\S+)$',
        'REJECT': r'^REJECT\s+(\S+)\s+reason="([^"]+)"$',
        'CORRECT': r'^CORRECT\s+(\S+)\s+action="([^"]+)"\s+value=(\S+)$',
        'OVERRIDE': r'^OVERRIDE\s+(\S+)\s+new_threshold=(\S+)\s+reason="([^"]+)"$',
        'QUERY': r'^QUERY\s+(\S+)\s+question="([^"]+)"$',
        'EXPLAIN': r'^EXPLAIN\s+(\S+)$',
        'BATCH_ACCEPT': r'^BATCH_ACCEPT\s+status=(\S+)$',
        'BATCH_REVIEW': r'^BATCH_REVIEW\s+(?:status|category)=(\S+)$',
    }

    # Rule explanations (source references)
    RULE_EXPLANATIONS = {
        'USB2_LENGTH_MATCHING': {
            'description': 'USB 2.0 High-Speed differential pair length matching requirement',
            'source': 'USB 2.0 Specification Chapter 7, Microchip AN2972',
            'reasoning': 'Length mismatch causes timing skew between D+ and D-, degrading signal integrity at 480 Mbps',
            'threshold_basis': '1.25mm corresponds to ~8.5ps skew at 170ps/mm propagation delay, within USB 2.0 HS timing budget',
            'alternatives': [
                'Full-Speed mode allows 5mm mismatch (12 Mbps is more tolerant)',
                'Use serpentine meanders on shorter trace to match lengths',
            ]
        },
        'DECOUPLING_DISTANCE': {
            'description': 'Maximum distance from decoupling capacitor to IC VCC pin',
            'source': 'Murata Design Guide, TI SLVA242A, IPC-2141',
            'reasoning': 'Longer traces add inductance, reducing decoupling effectiveness at high frequencies',
            'threshold_basis': '3mm limit ensures effective decoupling up to ~100 MHz',
            'alternatives': [
                'Use multiple smaller capacitors closer to pins',
                'Add ferrite bead for additional filtering',
            ]
        },
        'THERMAL_VIA_COUNT': {
            'description': 'Minimum thermal vias under thermal pads',
            'source': 'IPC-7093, JEDEC JESD51-5, TI Thermal Design Guide',
            'reasoning': 'Thermal vias conduct heat to inner/bottom copper layers for dissipation',
            'threshold_basis': '5 vias minimum for typical QFN/DFN with 2-3W dissipation',
            'alternatives': [
                'Use external heatsink if thermal vias insufficient',
                'Increase via diameter for better thermal conductivity',
            ]
        },
        'CRYSTAL_DISTANCE': {
            'description': 'Maximum distance from crystal to MCU oscillator pins',
            'source': 'STM32 AN2867, Microchip AN826, Murata Crystal Application Notes',
            'reasoning': 'Long traces add parasitic capacitance and are susceptible to EMI',
            'threshold_basis': '5mm limit ensures stable oscillation and reduces EMI coupling',
            'alternatives': [
                'Use internal RC oscillator if crystal placement is impossible',
                'Use clock buffer for longer distances',
            ]
        },
        'EMI_LOOP_AREA': {
            'description': 'Maximum loop area for power/signal return paths',
            'source': 'Henry Ott EMC Engineering, FCC Part 15, CISPR 32',
            'reasoning': 'Loop area directly affects radiated emissions (E = k * I * f^2 * A)',
            'threshold_basis': '100mm^2 limit at 100 MHz keeps emissions below regulatory limits',
            'alternatives': [
                'Route return path directly under signal trace',
                'Use ground plane on adjacent layer',
            ]
        },
        'DDR_SKEW': {
            'description': 'Maximum skew between DDR data group signals',
            'source': 'JEDEC JESD79-4B (DDR4), JEDEC JESD79-3F (DDR3)',
            'reasoning': 'Data signals must arrive within setup/hold timing window',
            'threshold_basis': 'DDR3: 10ps max skew, DDR4: 5ps max skew per JEDEC spec',
            'alternatives': [
                'Use serpentine length matching',
                'Group DQ signals in same byte lane',
            ]
        },
        'IMPEDANCE_TOLERANCE': {
            'description': 'Characteristic impedance tolerance for controlled-impedance traces',
            'source': 'IPC-2141, IPC-2251, SI Analysis Best Practices',
            'reasoning': 'Impedance mismatch causes reflections, degrading signal quality',
            'threshold_basis': '+/-10% is standard; +/-5% for high-speed (>1 Gbps)',
            'alternatives': [
                'Work with PCB fab to achieve tighter tolerance',
                'Use field solver for accurate impedance calculation',
            ]
        },
        'CLEARANCE_VOLTAGE': {
            'description': 'Minimum clearance based on voltage differential',
            'source': 'IPC-2221B, UL 60950-1, EN 60664-1',
            'reasoning': 'Prevents arcing and ensures electrical safety',
            'threshold_basis': 'Based on pollution degree 2, material group IIIb per IPC-2221B',
            'alternatives': [
                'Use conformal coating for reduced clearances',
                'Use slots/cutouts for additional isolation',
            ]
        },
        'VIA_ANNULAR_RING': {
            'description': 'Minimum annular ring for plated through holes',
            'source': 'IPC-6012D, IPC-2221B',
            'reasoning': 'Ensures reliable via connection despite drilling tolerances',
            'threshold_basis': 'Class 2: 0.125mm, Class 3: 0.150mm per IPC-6012D',
            'alternatives': [
                'Use larger pads if fab capability is limited',
                'Consider microvias for tighter layouts',
            ]
        },
        'TRACK_WIDTH_CURRENT': {
            'description': 'Minimum track width for given current capacity',
            'source': 'IPC-2152, Douglas Brooks PCB Trace Calculator',
            'reasoning': 'Prevents overheating due to I^2*R losses in copper',
            'threshold_basis': 'Based on 10C temperature rise, external layer, 1oz copper',
            'alternatives': [
                'Use polygon pour instead of trace for high current',
                'Use multiple vias for layer transitions',
            ]
        },
    }

    def __init__(self):
        """Initialize the feedback processor."""
        self.rule_contexts: Dict[str, RuleContext] = {}
        self.feedback_log: List[FeedbackResult] = []
        self.pending_queries: List[Dict[str, Any]] = []

    def add_report(self, report: RuleReport) -> None:
        """
        Add a rule report for AI review.

        Args:
            report: The RuleReport to add
        """
        context = RuleContext(
            rule_id=report.rule_id,
            report=report,
            original_threshold=report.threshold,
            current_threshold=report.threshold,
            status=report.status,
        )
        self.rule_contexts[report.rule_id] = context

    def add_reports(self, reports: List[RuleReport]) -> None:
        """
        Add multiple rule reports for AI review.

        Args:
            reports: List of RuleReports to add
        """
        for report in reports:
            self.add_report(report)

    def get_pending_reviews(self) -> List[RuleReport]:
        """Get all rules pending AI review."""
        return [
            ctx.report for ctx in self.rule_contexts.values()
            if ctx.status == RuleStatus.PENDING
        ]

    def get_failed_rules(self) -> List[RuleReport]:
        """Get all rules that failed validation."""
        return [
            ctx.report for ctx in self.rule_contexts.values()
            if ctx.status == RuleStatus.FAIL
        ]

    def get_review_summary(self) -> Dict[str, Any]:
        """Get summary of rules by status."""
        summary = {status.value: 0 for status in RuleStatus}
        for ctx in self.rule_contexts.values():
            summary[ctx.status.value] += 1
        return {
            'total': len(self.rule_contexts),
            'by_status': summary,
            'pending_review': summary.get('PENDING', 0) + summary.get('FAIL', 0),
        }

    def parse_command(self, command: str) -> FeedbackCommand:
        """
        Parse a feedback command string.

        Args:
            command: The command string to parse

        Returns:
            FeedbackCommand with parsed components

        Raises:
            CommandParseError: If command cannot be parsed
        """
        command = command.strip()

        for cmd_type, pattern in self.COMMAND_PATTERNS.items():
            match = re.match(pattern, command)
            if match:
                groups = match.groups()

                if cmd_type == 'ACCEPT':
                    return FeedbackCommand(
                        command_type='ACCEPT',
                        rule_id=groups[0],
                        parameters={},
                        raw_command=command,
                    )
                elif cmd_type == 'REJECT':
                    return FeedbackCommand(
                        command_type='REJECT',
                        rule_id=groups[0],
                        parameters={'reason': groups[1]},
                        raw_command=command,
                    )
                elif cmd_type == 'CORRECT':
                    return FeedbackCommand(
                        command_type='CORRECT',
                        rule_id=groups[0],
                        parameters={'action': groups[1], 'value': self._parse_value(groups[2])},
                        raw_command=command,
                    )
                elif cmd_type == 'OVERRIDE':
                    return FeedbackCommand(
                        command_type='OVERRIDE',
                        rule_id=groups[0],
                        parameters={'new_threshold': self._parse_value(groups[1]), 'reason': groups[2]},
                        raw_command=command,
                    )
                elif cmd_type == 'QUERY':
                    return FeedbackCommand(
                        command_type='QUERY',
                        rule_id=groups[0],
                        parameters={'question': groups[1]},
                        raw_command=command,
                    )
                elif cmd_type == 'EXPLAIN':
                    return FeedbackCommand(
                        command_type='EXPLAIN',
                        rule_id=groups[0],
                        parameters={},
                        raw_command=command,
                    )
                elif cmd_type == 'BATCH_ACCEPT':
                    return FeedbackCommand(
                        command_type='BATCH_ACCEPT',
                        rule_id=None,
                        parameters={'status': groups[0]},
                        raw_command=command,
                    )
                elif cmd_type == 'BATCH_REVIEW':
                    return FeedbackCommand(
                        command_type='BATCH_REVIEW',
                        rule_id=None,
                        parameters={'filter': groups[0]},
                        raw_command=command,
                    )

        raise CommandParseError(f"Unable to parse command: {command}")

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string to appropriate type."""
        # Try float
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Try boolean
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'

        # Return as string
        return value_str

    def process_command(self, command: str) -> FeedbackResult:
        """
        Parse and execute an AI feedback command.

        Args:
            command: The command string to process

        Returns:
            FeedbackResult with action taken
        """
        try:
            parsed = self.parse_command(command)
        except CommandParseError as e:
            return FeedbackResult(
                command=command,
                rule_id='',
                action_taken=f'PARSE_ERROR: {str(e)}',
                new_status=RuleStatus.PENDING,
                applied_correction=None,
                requires_revalidation=False,
                next_steps=['Fix command syntax and retry'],
            )

        # Dispatch to appropriate handler
        handlers = {
            'ACCEPT': self.accept,
            'REJECT': self.reject,
            'CORRECT': self.correct,
            'OVERRIDE': self.override,
            'QUERY': self.query,
            'EXPLAIN': self._handle_explain,
            'BATCH_ACCEPT': self._handle_batch_accept,
            'BATCH_REVIEW': self._handle_batch_review,
        }

        handler = handlers.get(parsed.command_type)
        if handler:
            if parsed.command_type in ('BATCH_ACCEPT', 'BATCH_REVIEW'):
                result = handler(parsed.parameters)
            elif parsed.command_type == 'ACCEPT':
                result = handler(parsed.rule_id)
            elif parsed.command_type == 'REJECT':
                result = handler(parsed.rule_id, parsed.parameters.get('reason', ''))
            elif parsed.command_type == 'CORRECT':
                result = handler(
                    parsed.rule_id,
                    parsed.parameters.get('action', ''),
                    parsed.parameters.get('value')
                )
            elif parsed.command_type == 'OVERRIDE':
                result = handler(
                    parsed.rule_id,
                    parsed.parameters.get('new_threshold'),
                    parsed.parameters.get('reason', '')
                )
            elif parsed.command_type == 'QUERY':
                result = handler(parsed.rule_id, parsed.parameters.get('question', ''))
            elif parsed.command_type == 'EXPLAIN':
                result = handler(parsed.rule_id)
            else:
                result = handler(parsed.rule_id)

            # Log the feedback
            self.feedback_log.append(result)
            return result

        return FeedbackResult(
            command=command,
            rule_id=parsed.rule_id or '',
            action_taken='UNKNOWN_COMMAND',
            new_status=RuleStatus.PENDING,
            applied_correction=None,
            requires_revalidation=False,
            next_steps=['Use a valid command: ACCEPT, REJECT, CORRECT, OVERRIDE, QUERY, EXPLAIN'],
        )

    def accept(self, rule_id: str) -> FeedbackResult:
        """
        AI validates the rule outcome.

        Args:
            rule_id: The rule to accept

        Returns:
            FeedbackResult confirming acceptance
        """
        if rule_id not in self.rule_contexts:
            return FeedbackResult(
                command=f'ACCEPT {rule_id}',
                rule_id=rule_id,
                action_taken='RULE_NOT_FOUND',
                new_status=RuleStatus.PENDING,
                applied_correction=None,
                requires_revalidation=False,
                next_steps=[f'Rule {rule_id} not found. Check rule_id spelling.'],
            )

        context = self.rule_contexts[rule_id]
        context.status = RuleStatus.ACCEPTED

        return FeedbackResult(
            command=f'ACCEPT {rule_id}',
            rule_id=rule_id,
            action_taken=f'Rule {rule_id} accepted and validated by AI',
            new_status=RuleStatus.ACCEPTED,
            applied_correction=None,
            requires_revalidation=False,
            next_steps=['Rule outcome confirmed. No further action needed.'],
        )

    def reject(self, rule_id: str, reason: str) -> FeedbackResult:
        """
        AI rejects the rule outcome with reasoning.

        Args:
            rule_id: The rule to reject
            reason: The reason for rejection

        Returns:
            FeedbackResult with rejection details
        """
        if rule_id not in self.rule_contexts:
            return FeedbackResult(
                command=f'REJECT {rule_id} reason="{reason}"',
                rule_id=rule_id,
                action_taken='RULE_NOT_FOUND',
                new_status=RuleStatus.PENDING,
                applied_correction=None,
                requires_revalidation=False,
                next_steps=[f'Rule {rule_id} not found.'],
            )

        context = self.rule_contexts[rule_id]
        context.status = RuleStatus.REJECTED

        return FeedbackResult(
            command=f'REJECT {rule_id} reason="{reason}"',
            rule_id=rule_id,
            action_taken=f'Rule {rule_id} rejected: {reason}',
            new_status=RuleStatus.REJECTED,
            applied_correction={'rejection_reason': reason},
            requires_revalidation=True,
            next_steps=[
                'Review the rejection reason',
                'Consider alternative design approach',
                'Re-run validation after design changes',
            ],
        )

    def correct(self, rule_id: str, action: str, value: Any) -> FeedbackResult:
        """
        AI provides a correction to apply.

        Args:
            rule_id: The rule to correct
            action: The corrective action to take
            value: The value for the correction

        Returns:
            FeedbackResult with correction details
        """
        if rule_id not in self.rule_contexts:
            return FeedbackResult(
                command=f'CORRECT {rule_id} action="{action}" value={value}',
                rule_id=rule_id,
                action_taken='RULE_NOT_FOUND',
                new_status=RuleStatus.PENDING,
                applied_correction=None,
                requires_revalidation=False,
                next_steps=[f'Rule {rule_id} not found.'],
            )

        context = self.rule_contexts[rule_id]
        context.status = RuleStatus.CORRECTED

        correction = {
            'action': action,
            'value': value,
            'applied_at': datetime.now().isoformat(),
        }

        return FeedbackResult(
            command=f'CORRECT {rule_id} action="{action}" value={value}',
            rule_id=rule_id,
            action_taken=f'Correction applied: {action} = {value}',
            new_status=RuleStatus.CORRECTED,
            applied_correction=correction,
            requires_revalidation=True,
            next_steps=[
                f'Apply correction: {action} with value {value}',
                'Re-run DRC after applying correction',
                'Verify correction resolves the violation',
            ],
        )

    def override(self, rule_id: str, new_threshold: Any, reason: str) -> FeedbackResult:
        """
        AI overrides the rule threshold with justification.

        Args:
            rule_id: The rule to override
            new_threshold: The new threshold value
            reason: Justification for the override

        Returns:
            FeedbackResult with override details
        """
        if rule_id not in self.rule_contexts:
            return FeedbackResult(
                command=f'OVERRIDE {rule_id} new_threshold={new_threshold} reason="{reason}"',
                rule_id=rule_id,
                action_taken='RULE_NOT_FOUND',
                new_status=RuleStatus.PENDING,
                applied_correction=None,
                requires_revalidation=False,
                next_steps=[f'Rule {rule_id} not found.'],
            )

        context = self.rule_contexts[rule_id]
        old_threshold = context.current_threshold
        context.current_threshold = new_threshold
        context.status = RuleStatus.ACCEPTED  # Override implies acceptance

        override_details = {
            'old_threshold': old_threshold,
            'new_threshold': new_threshold,
            'reason': reason,
            'overridden_at': datetime.now().isoformat(),
        }

        return FeedbackResult(
            command=f'OVERRIDE {rule_id} new_threshold={new_threshold} reason="{reason}"',
            rule_id=rule_id,
            action_taken=f'Threshold overridden: {old_threshold} -> {new_threshold}',
            new_status=RuleStatus.ACCEPTED,
            applied_correction=override_details,
            requires_revalidation=True,
            next_steps=[
                f'Threshold changed from {old_threshold} to {new_threshold}',
                'Re-validate with new threshold',
                'Document override reason in design notes',
            ],
        )

    def query(self, rule_id: str, question: str) -> FeedbackResult:
        """
        AI asks for clarification about a rule.

        Args:
            rule_id: The rule to query
            question: The question to ask

        Returns:
            FeedbackResult with query response
        """
        # Store the query for later response
        query_entry = {
            'rule_id': rule_id,
            'question': question,
            'asked_at': datetime.now().isoformat(),
            'answered': False,
        }
        self.pending_queries.append(query_entry)

        # Try to provide an immediate answer from explanations
        if rule_id in self.RULE_EXPLANATIONS:
            explanation = self.RULE_EXPLANATIONS[rule_id]
            answer = self._generate_query_answer(question, explanation)
            query_entry['answered'] = True
            query_entry['answer'] = answer

            return FeedbackResult(
                command=f'QUERY {rule_id} question="{question}"',
                rule_id=rule_id,
                action_taken=f'Query answered: {answer[:100]}...',
                new_status=RuleStatus.PENDING,
                applied_correction={'query': question, 'answer': answer},
                requires_revalidation=False,
                next_steps=['Review answer and proceed with appropriate action'],
            )

        return FeedbackResult(
            command=f'QUERY {rule_id} question="{question}"',
            rule_id=rule_id,
            action_taken='Query logged for response',
            new_status=RuleStatus.PENDING,
            applied_correction={'query': question, 'answer': 'Pending expert response'},
            requires_revalidation=False,
            next_steps=[
                'Query has been logged',
                'Use EXPLAIN command for rule details',
                'Check rule source documentation',
            ],
        )

    def _generate_query_answer(self, question: str, explanation: Dict) -> str:
        """Generate an answer to a query based on rule explanation."""
        question_lower = question.lower()

        if 'why' in question_lower or 'reason' in question_lower:
            return explanation.get('reasoning', 'See source documentation.')
        elif 'source' in question_lower or 'reference' in question_lower:
            return explanation.get('source', 'See IPC/JEDEC standards.')
        elif 'alternative' in question_lower or 'workaround' in question_lower:
            alts = explanation.get('alternatives', [])
            return '; '.join(alts) if alts else 'No documented alternatives.'
        elif 'threshold' in question_lower or 'limit' in question_lower:
            return explanation.get('threshold_basis', 'Based on industry standards.')
        else:
            return explanation.get('description', 'See rule documentation.')

    def _handle_explain(self, rule_id: str) -> FeedbackResult:
        """
        Return detailed explanation of why a rule exists.

        Args:
            rule_id: The rule to explain

        Returns:
            FeedbackResult with explanation
        """
        if rule_id in self.RULE_EXPLANATIONS:
            explanation = self.RULE_EXPLANATIONS[rule_id]
            explanation_text = json.dumps(explanation, indent=2)

            return FeedbackResult(
                command=f'EXPLAIN {rule_id}',
                rule_id=rule_id,
                action_taken='Explanation provided',
                new_status=RuleStatus.PENDING,
                applied_correction={'explanation': explanation},
                requires_revalidation=False,
                next_steps=['Review explanation and proceed with ACCEPT, REJECT, or OVERRIDE'],
            )

        # Try to get explanation from rule context
        if rule_id in self.rule_contexts:
            context = self.rule_contexts[rule_id]
            report = context.report

            basic_explanation = {
                'rule_id': rule_id,
                'category': report.rule_category.value if report.rule_category else 'unknown',
                'source': report.rule_source,
                'rule_applied': report.rule_applied,
                'threshold': report.threshold,
            }

            return FeedbackResult(
                command=f'EXPLAIN {rule_id}',
                rule_id=rule_id,
                action_taken='Basic explanation from rule report',
                new_status=RuleStatus.PENDING,
                applied_correction={'explanation': basic_explanation},
                requires_revalidation=False,
                next_steps=['For detailed explanation, consult rule source documentation'],
            )

        return FeedbackResult(
            command=f'EXPLAIN {rule_id}',
            rule_id=rule_id,
            action_taken='Rule not found in explanations database',
            new_status=RuleStatus.PENDING,
            applied_correction=None,
            requires_revalidation=False,
            next_steps=[f'Rule {rule_id} has no documented explanation. Check source standards.'],
        )

    def _handle_batch_accept(self, parameters: Dict[str, Any]) -> FeedbackResult:
        """
        Accept all rules with a specific status.

        Args:
            parameters: Contains 'status' to filter by

        Returns:
            FeedbackResult with batch action details
        """
        status_filter = parameters.get('status', 'PASS').upper()

        try:
            target_status = RuleStatus[status_filter]
        except KeyError:
            return FeedbackResult(
                command=f'BATCH_ACCEPT status={status_filter}',
                rule_id='BATCH',
                action_taken='Invalid status filter',
                new_status=RuleStatus.PENDING,
                applied_correction=None,
                requires_revalidation=False,
                next_steps=[f'Valid statuses: {[s.value for s in RuleStatus]}'],
            )

        accepted_count = 0
        accepted_rules = []

        for rule_id, context in self.rule_contexts.items():
            if context.status == target_status:
                context.status = RuleStatus.ACCEPTED
                accepted_count += 1
                accepted_rules.append(rule_id)

        return FeedbackResult(
            command=f'BATCH_ACCEPT status={status_filter}',
            rule_id='BATCH',
            action_taken=f'Accepted {accepted_count} rules with status {status_filter}',
            new_status=RuleStatus.ACCEPTED,
            applied_correction={'accepted_rules': accepted_rules},
            requires_revalidation=False,
            next_steps=[
                f'{accepted_count} rules accepted',
                'Review remaining rules with BATCH_REVIEW status=FAIL',
            ],
        )

    def _handle_batch_review(self, parameters: Dict[str, Any]) -> FeedbackResult:
        """
        Get list of rules to review by status or category.

        Args:
            parameters: Contains 'filter' for status or category

        Returns:
            FeedbackResult with rules to review
        """
        filter_value = parameters.get('filter', 'FAIL').upper()

        # Try as status first
        rules_to_review = []
        try:
            target_status = RuleStatus[filter_value]
            for rule_id, context in self.rule_contexts.items():
                if context.status == target_status:
                    rules_to_review.append({
                        'rule_id': rule_id,
                        'status': context.status.value,
                        'threshold': context.current_threshold,
                        'violations': context.report.violations[:3] if context.report.violations else [],
                    })
        except KeyError:
            # Try as category
            try:
                target_category = RuleCategory[filter_value]
                for rule_id, context in self.rule_contexts.items():
                    if context.report.rule_category == target_category:
                        rules_to_review.append({
                            'rule_id': rule_id,
                            'status': context.status.value,
                            'category': target_category.value,
                        })
            except KeyError:
                return FeedbackResult(
                    command=f'BATCH_REVIEW filter={filter_value}',
                    rule_id='BATCH',
                    action_taken='Invalid filter',
                    new_status=RuleStatus.PENDING,
                    applied_correction=None,
                    requires_revalidation=False,
                    next_steps=[
                        f'Valid statuses: {[s.value for s in RuleStatus]}',
                        f'Valid categories: {[c.value for c in RuleCategory]}',
                    ],
                )

        return FeedbackResult(
            command=f'BATCH_REVIEW filter={filter_value}',
            rule_id='BATCH',
            action_taken=f'Found {len(rules_to_review)} rules for review',
            new_status=RuleStatus.PENDING,
            applied_correction={'rules_to_review': rules_to_review},
            requires_revalidation=False,
            next_steps=[
                f'Review {len(rules_to_review)} rules',
                'Use ACCEPT, REJECT, CORRECT, or OVERRIDE for each rule',
            ],
        )

    def explain(self, rule_id: str) -> str:
        """
        Return detailed explanation of why a rule exists.

        Args:
            rule_id: The rule to explain

        Returns:
            String explanation
        """
        if rule_id in self.RULE_EXPLANATIONS:
            explanation = self.RULE_EXPLANATIONS[rule_id]
            parts = [
                f"Rule: {rule_id}",
                f"Description: {explanation.get('description', 'N/A')}",
                f"Source: {explanation.get('source', 'N/A')}",
                f"Reasoning: {explanation.get('reasoning', 'N/A')}",
                f"Threshold Basis: {explanation.get('threshold_basis', 'N/A')}",
            ]
            if explanation.get('alternatives'):
                parts.append(f"Alternatives: {'; '.join(explanation['alternatives'])}")
            return '\n'.join(parts)

        return f"No detailed explanation available for {rule_id}. Check source documentation."

    def get_feedback_log(self) -> List[Dict[str, Any]]:
        """Get the complete feedback log as JSON-serializable dicts."""
        return [
            {
                'command': fb.command,
                'rule_id': fb.rule_id,
                'action_taken': fb.action_taken,
                'new_status': fb.new_status.value,
                'applied_correction': fb.applied_correction,
                'requires_revalidation': fb.requires_revalidation,
                'next_steps': fb.next_steps,
            }
            for fb in self.feedback_log
        ]

    def export_session(self) -> Dict[str, Any]:
        """Export the complete feedback session for persistence."""
        return {
            'timestamp': datetime.now().isoformat(),
            'rule_contexts': {
                rule_id: {
                    'rule_id': ctx.rule_id,
                    'status': ctx.status.value,
                    'original_threshold': ctx.original_threshold,
                    'current_threshold': ctx.current_threshold,
                    'feedback_count': len(ctx.feedback_history),
                }
                for rule_id, ctx in self.rule_contexts.items()
            },
            'feedback_log': self.get_feedback_log(),
            'pending_queries': self.pending_queries,
            'summary': self.get_review_summary(),
        }

    def generate_ai_review_prompt(self) -> str:
        """
        Generate a prompt for an external AI to review the rules.

        Returns:
            String prompt with all rules needing review
        """
        failed_rules = self.get_failed_rules()
        pending_rules = self.get_pending_reviews()

        lines = [
            "=" * 60,
            "PCB DESIGN RULE REVIEW REQUEST",
            "=" * 60,
            "",
            f"Total rules to review: {len(failed_rules) + len(pending_rules)}",
            f"  - Failed: {len(failed_rules)}",
            f"  - Pending: {len(pending_rules)}",
            "",
            "FAILED RULES (require action):",
            "-" * 40,
        ]

        for report in failed_rules:
            lines.append(f"\nRule ID: {report.rule_id}")
            lines.append(f"Category: {report.rule_category.value if report.rule_category else 'unknown'}")
            lines.append(f"Applied: {report.rule_applied}")
            lines.append(f"Threshold: {report.threshold}")
            lines.append(f"Actual: {report.actual_value}")
            lines.append(f"Violations:")
            for v in report.violations[:3]:
                lines.append(f"  - {v}")
            lines.append(f"Commands:")
            commands = report.get_feedback_commands()
            lines.append(f"  ACCEPT: {commands['accept']}")
            lines.append(f"  REJECT: {commands['reject']}")
            lines.append(f"  EXPLAIN: {commands['explain']}")

        lines.extend([
            "",
            "=" * 60,
            "Please respond with feedback commands for each rule.",
            "=" * 60,
        ])

        return '\n'.join(lines)
