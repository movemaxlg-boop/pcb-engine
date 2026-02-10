"""
CONSTITUTIONAL LAYOUT VALIDATOR
================================

Validates a c_layout before passing it to the BBL Engine.
This is the "gate" that catches errors early, before wasting
compute cycles on an impossible design.

Validation checks:
1. All parts exist in parts_db
2. All nets have valid endpoints
3. No conflicting rules
4. Board can physically fit all parts (rough check)
5. Overrides have valid justifications
6. Rule hierarchy is consistent
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import math

from .clayout_types import (
    ConstitutionalLayout,
    CLayoutValidationResult,
    ValidationStatus,
    ComponentDefinition,
    NetDefinition,
    RuleHierarchy,
    RuleOverride,
    PlacementHints,
    RoutingHints,
    RulePriority,
)


# =============================================================================
# COMPONENT SIZE ESTIMATES (for area calculations)
# =============================================================================

# Approximate component footprint sizes in mm^2
FOOTPRINT_SIZES = {
    # Passives
    "0201": (0.6, 0.3),
    "0402": (1.0, 0.5),
    "0603": (1.6, 0.8),
    "0805": (2.0, 1.25),
    "1206": (3.2, 1.6),
    "1210": (3.2, 2.5),
    "2010": (5.0, 2.5),
    "2512": (6.3, 3.2),

    # SOT packages
    "SOT-23": (3.0, 1.4),
    "SOT-23-3": (3.0, 1.4),
    "SOT-23-5": (3.0, 1.75),
    "SOT-23-6": (3.0, 1.75),
    "SOT-223": (6.7, 3.7),
    "SOT-89": (4.6, 2.6),

    # SOIC
    "SOIC-8": (5.0, 4.0),
    "SOIC-14": (8.75, 4.0),
    "SOIC-16": (10.0, 4.0),

    # QFP
    "LQFP-32": (7.0, 7.0),
    "LQFP-48": (9.0, 9.0),
    "LQFP-64": (12.0, 12.0),
    "LQFP-100": (14.0, 14.0),
    "TQFP-32": (7.0, 7.0),
    "TQFP-44": (10.0, 10.0),

    # QFN
    "QFN-16": (4.0, 4.0),
    "QFN-20": (4.0, 4.0),
    "QFN-24": (4.0, 4.0),
    "QFN-32": (5.0, 5.0),
    "QFN-48": (7.0, 7.0),

    # Modules
    "ESP32-WROOM": (25.5, 18.0),
    "ESP-WROOM-32": (25.5, 18.0),
    "ESP32-WROVER": (31.4, 18.0),

    # Crystals
    "HC49": (11.0, 5.0),
    "HC49-SMD": (11.5, 4.7),
    "3215": (3.2, 1.5),
    "2520": (2.5, 2.0),

    # Connectors
    "USB-C": (9.0, 7.5),
    "USB-A": (14.0, 13.0),
    "MICRO-USB": (8.0, 5.5),
    "USB-MICRO": (8.0, 5.5),
    "RJ45": (16.0, 13.5),

    # Through-hole
    "DO-41": (5.0, 2.5),
    "SMA": (5.0, 2.5),
    "SMB": (4.5, 2.0),

    # Electrolytic
    "CAP-5MM": (6.5, 6.5),
    "CAP-6.3MM": (8.0, 8.0),
    "CAP-8MM": (10.0, 10.0),
    "CAP-10MM": (12.0, 12.0),

    # Default
    "DEFAULT": (5.0, 5.0),
}


# =============================================================================
# VALIDATOR CLASS
# =============================================================================

class CLayoutValidator:
    """
    Validates a c_layout before passing to BBL.

    This is the "gate" that ensures we don't waste time on impossible designs.
    """

    def __init__(self, parts_db: Optional[Dict] = None):
        """
        Initialize validator.

        Args:
            parts_db: Optional parts database for validation
        """
        self.parts_db = parts_db or {}
        self.footprint_sizes = FOOTPRINT_SIZES

    def validate(self, clayout: ConstitutionalLayout) -> CLayoutValidationResult:
        """
        Run all validation checks on a c_layout.

        Args:
            clayout: The constitutional layout to validate

        Returns:
            CLayoutValidationResult with errors, warnings, and suggestions
        """

        errors = []
        warnings = []
        suggestions = []

        # Run all validation checks
        errors.extend(self.check_parts_exist(clayout))
        errors.extend(self.check_nets_valid(clayout))
        errors.extend(self.check_no_rule_conflicts(clayout))
        errors.extend(self.check_overrides_valid(clayout))
        errors.extend(self.check_hierarchy_consistent(clayout))

        # Warnings (non-blocking)
        warnings.extend(self.check_board_fits(clayout))
        warnings.extend(self.check_placement_hints_valid(clayout))
        warnings.extend(self.check_routing_hints_valid(clayout))

        # Suggestions
        suggestions.extend(self.generate_suggestions(clayout))

        # Calculate routability estimate
        routability = self.estimate_routability(clayout)

        # Determine validation status
        valid = len(errors) == 0
        if valid:
            if warnings:
                status = ValidationStatus.WARNINGS
            else:
                status = ValidationStatus.VALID
        else:
            status = ValidationStatus.INVALID

        # Update clayout status
        clayout.validation_status = status
        clayout.validation_errors = errors
        clayout.validation_warnings = warnings

        return CLayoutValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            routability_estimate=routability,
        )

    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================

    def check_parts_exist(self, clayout: ConstitutionalLayout) -> List[str]:
        """Verify all parts exist in parts_db (if provided)."""

        errors = []

        if not self.parts_db:
            # No parts_db, skip this check
            return errors

        for comp in clayout.components:
            if comp.part_number not in self.parts_db:
                errors.append(
                    f"Component {comp.ref_des}: Part number '{comp.part_number}' "
                    f"not found in parts database"
                )

        return errors

    def check_nets_valid(self, clayout: ConstitutionalLayout) -> List[str]:
        """Verify all net endpoints exist."""

        errors = []

        # Build set of valid pins
        valid_pins = set()
        for comp in clayout.components:
            # Add common pin patterns
            for pin in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                valid_pins.add(f"{comp.ref_des}.{pin}")
            # Add common named pins
            for pin in ["VCC", "GND", "VDD", "VSS", "IN", "OUT", "EN", "NC"]:
                valid_pins.add(f"{comp.ref_des}.{pin}")

        # Check each net
        for net in clayout.nets:
            for pin in net.pins:
                # Extract component reference
                if "." in pin:
                    ref_des = pin.split(".")[0]
                    comp = clayout.get_component(ref_des)
                    if not comp:
                        errors.append(
                            f"Net '{net.name}': Pin '{pin}' references unknown "
                            f"component '{ref_des}'"
                        )

        return errors

    def check_no_rule_conflicts(self, clayout: ConstitutionalLayout) -> List[str]:
        """Check for conflicting rules."""

        errors = []

        # Check for obvious conflicts
        board = clayout.board

        # Min trace width vs min spacing
        if board.min_trace_mm > 0 and board.min_space_mm > 0:
            if board.min_trace_mm + board.min_space_mm > 1.0:
                # Not necessarily an error, but check board size
                pass

        # Via drill vs annular ring
        if board.min_via_drill_mm > 0 and board.min_via_annular_mm > 0:
            min_via_outer = board.min_via_drill_mm + 2 * board.min_via_annular_mm
            if min_via_outer > 1.5:  # Very large via
                errors.append(
                    f"Via outer diameter ({min_via_outer:.2f}mm) seems too large. "
                    f"Check min_via_drill ({board.min_via_drill_mm}mm) and "
                    f"min_via_annular ({board.min_via_annular_mm}mm)"
                )

        # Check rule hierarchy for conflicts
        rules = clayout.rules
        all_rule_ids = set()

        for binding in rules.inviolable + rules.recommended + rules.optional:
            if binding.rule_id in all_rule_ids:
                errors.append(
                    f"Rule '{binding.rule_id}' appears in multiple categories"
                )
            all_rule_ids.add(binding.rule_id)

        return errors

    def check_board_fits(self, clayout: ConstitutionalLayout) -> List[str]:
        """Rough check: can all components fit on board?"""

        warnings = []

        board = clayout.board
        board_area = board.width_mm * board.height_mm

        # Calculate total component area (with courtyard margin)
        total_component_area = 0.0
        for comp in clayout.components:
            size = self._get_footprint_size(comp.footprint)
            # Add courtyard margin (typically 0.25mm on each side)
            courtyard_area = (size[0] + 0.5) * (size[1] + 0.5)
            total_component_area += courtyard_area

        # Calculate fill ratio
        fill_ratio = total_component_area / board_area if board_area > 0 else float('inf')

        if fill_ratio > 0.7:
            warnings.append(
                f"Component area ({total_component_area:.1f}mm²) is {fill_ratio*100:.0f}% "
                f"of board area ({board_area:.1f}mm²). Design may be too dense. "
                f"Consider larger board or smaller components."
            )
        elif fill_ratio > 0.5:
            warnings.append(
                f"Component fill ratio is {fill_ratio*100:.0f}%. Routing may be challenging."
            )

        return warnings

    def check_overrides_valid(self, clayout: ConstitutionalLayout) -> List[str]:
        """Verify all overrides have justifications."""

        errors = []

        for override in clayout.overrides:
            if not override.justification:
                errors.append(
                    f"Override for rule '{override.rule_id}' has no justification"
                )

            if not override.evidence:
                errors.append(
                    f"Override for rule '{override.rule_id}' has no evidence/source"
                )

            # Check that the rule being overridden exists in hierarchy
            if not clayout.rules.get_rule(override.rule_id):
                errors.append(
                    f"Override references unknown rule '{override.rule_id}'"
                )

        return errors

    def check_hierarchy_consistent(self, clayout: ConstitutionalLayout) -> List[str]:
        """Check that rule hierarchy is internally consistent."""

        errors = []
        rules = clayout.rules

        # No rule should be in multiple categories
        seen = set()
        for binding in rules.inviolable:
            if binding.rule_id in seen:
                errors.append(f"Rule '{binding.rule_id}' duplicated in hierarchy")
            seen.add(binding.rule_id)

        for binding in rules.recommended:
            if binding.rule_id in seen:
                errors.append(f"Rule '{binding.rule_id}' duplicated in hierarchy")
            seen.add(binding.rule_id)

        for binding in rules.optional:
            if binding.rule_id in seen:
                errors.append(f"Rule '{binding.rule_id}' duplicated in hierarchy")
            seen.add(binding.rule_id)

        return errors

    def check_placement_hints_valid(self, clayout: ConstitutionalLayout) -> List[str]:
        """Check that placement hints reference valid components."""

        warnings = []
        hints = clayout.placement_hints
        valid_refs = {c.ref_des for c in clayout.components}

        # Check proximity groups
        for group in hints.proximity_groups:
            for ref in group.components:
                if ref not in valid_refs:
                    warnings.append(
                        f"Proximity group references unknown component '{ref}'"
                    )

        # Check edge components
        for ref in hints.edge_components:
            if ref not in valid_refs:
                warnings.append(
                    f"Edge component '{ref}' not found in design"
                )

        # Check keep-apart
        for ka in hints.keep_apart:
            if ka.component_a not in valid_refs:
                warnings.append(f"Keep-apart references unknown '{ka.component_a}'")
            if ka.component_b not in valid_refs:
                warnings.append(f"Keep-apart references unknown '{ka.component_b}'")

        # Check fixed positions
        for ref in hints.fixed_positions:
            if ref not in valid_refs:
                warnings.append(f"Fixed position for unknown component '{ref}'")

        return warnings

    def check_routing_hints_valid(self, clayout: ConstitutionalLayout) -> List[str]:
        """Check that routing hints reference valid nets."""

        warnings = []
        hints = clayout.routing_hints
        valid_nets = {n.name for n in clayout.nets}

        # Check priority nets
        for net in hints.priority_nets:
            if net not in valid_nets:
                warnings.append(f"Priority net '{net}' not found in design")

        # Check diff pairs
        for dp in hints.diff_pairs:
            if dp.positive_net not in valid_nets:
                warnings.append(f"Diff pair positive net '{dp.positive_net}' not found")
            if dp.negative_net not in valid_nets:
                warnings.append(f"Diff pair negative net '{dp.negative_net}' not found")

        # Check length match groups
        for group in hints.length_match_groups:
            for net in group.nets:
                if net not in valid_nets:
                    warnings.append(
                        f"Length match group '{group.name}' references unknown net '{net}'"
                    )

        return warnings

    # =========================================================================
    # ROUTABILITY ESTIMATE
    # =========================================================================

    def estimate_routability(self, clayout: ConstitutionalLayout) -> float:
        """
        Estimate probability of successful routing.

        Returns:
            Float 0.0-1.0 representing estimated routability
        """

        score = 1.0

        board = clayout.board
        board_area = board.width_mm * board.height_mm

        # Factor 1: Component density
        total_comp_area = sum(
            self._get_footprint_size(c.footprint)[0] *
            self._get_footprint_size(c.footprint)[1]
            for c in clayout.components
        )
        fill_ratio = total_comp_area / board_area if board_area > 0 else 1.0

        if fill_ratio > 0.5:
            score *= 0.5  # High density is hard
        elif fill_ratio > 0.3:
            score *= 0.8

        # Factor 2: Layer count vs net count
        net_count = len(clayout.nets)
        layers = board.layer_count

        nets_per_layer = net_count / layers if layers > 0 else float('inf')
        if nets_per_layer > 50:
            score *= 0.6
        elif nets_per_layer > 30:
            score *= 0.8

        # Factor 3: Differential pairs (more complex routing)
        diff_pair_count = len(clayout.routing_hints.diff_pairs)
        if diff_pair_count > 5:
            score *= 0.8
        elif diff_pair_count > 2:
            score *= 0.9

        # Factor 4: Length matching (more constraints)
        length_match_count = len(clayout.routing_hints.length_match_groups)
        if length_match_count > 3:
            score *= 0.7
        elif length_match_count > 1:
            score *= 0.85

        # Factor 5: Board aspect ratio (square is easier)
        aspect = max(board.width_mm, board.height_mm) / min(board.width_mm, board.height_mm)
        if aspect > 3:
            score *= 0.8
        elif aspect > 2:
            score *= 0.9

        return max(0.0, min(1.0, score))

    # =========================================================================
    # SUGGESTIONS
    # =========================================================================

    def generate_suggestions(self, clayout: ConstitutionalLayout) -> List[str]:
        """Generate improvement suggestions."""

        suggestions = []

        board = clayout.board

        # Layer count suggestion
        net_count = len(clayout.nets)
        if net_count > 30 and board.layer_count == 2:
            suggestions.append(
                f"Design has {net_count} nets on 2 layers. "
                f"Consider 4 layers for easier routing."
            )

        # Component count suggestion
        comp_count = len(clayout.components)
        if comp_count > 50:
            suggestions.append(
                f"Design has {comp_count} components. "
                f"Consider hierarchical design or modular approach."
            )

        # GND pour suggestion
        has_gnd_hints = any(
            n.name.upper() in ["GND", "GROUND", "VSS"]
            for n in clayout.nets
        )
        if board.layer_count >= 2 and has_gnd_hints:
            suggestions.append(
                "Consider using a GND pour on the bottom layer "
                "for improved EMI and simplified GND routing."
            )

        # Decoupling suggestion
        mcus = [c for c in clayout.components if "MCU" in str(c.category)]
        caps = [c for c in clayout.components if "CAPACITOR" in str(c.category)]
        if mcus and len(caps) < len(mcus) * 4:
            suggestions.append(
                f"MCU found but only {len(caps)} capacitors. "
                f"Ensure adequate decoupling (typically 4-6 per MCU)."
            )

        return suggestions

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_footprint_size(self, footprint: str) -> Tuple[float, float]:
        """Get the size of a footprint in mm (width, height)."""

        footprint_upper = footprint.upper()

        # Check exact match first
        if footprint_upper in self.footprint_sizes:
            return self.footprint_sizes[footprint_upper]

        # Check partial matches
        for key, size in self.footprint_sizes.items():
            if key in footprint_upper:
                return size

        # Default size
        return self.footprint_sizes["DEFAULT"]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_clayout(
    clayout: ConstitutionalLayout,
    parts_db: Optional[Dict] = None
) -> CLayoutValidationResult:
    """
    Convenience function to validate a c_layout.

    Args:
        clayout: The constitutional layout to validate
        parts_db: Optional parts database

    Returns:
        CLayoutValidationResult
    """
    validator = CLayoutValidator(parts_db)
    return validator.validate(clayout)


def quick_validate(clayout: ConstitutionalLayout) -> bool:
    """
    Quick validation - just returns True/False.

    Args:
        clayout: The constitutional layout to validate

    Returns:
        True if valid, False if errors
    """
    validator = CLayoutValidator()
    result = validator.validate(clayout)
    return result.valid
