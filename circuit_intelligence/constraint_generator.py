"""
Constraint Generator
=====================

Generates constraints for the PCB Engine based on circuit analysis.
This is the bridge between Circuit Intelligence and PCB Engine.
"""

from typing import Dict, List
from .circuit_types import (
    CircuitAnalysis, DesignConstraints,
    PlacementConstraint, RoutingConstraint,
    NetFunction, ComponentFunction
)


class ConstraintGenerator:
    """
    Generates PCB constraints from circuit analysis.

    This translates circuit-level understanding into
    geometric constraints that the PCB Engine can use.
    """

    def generate(self, analysis: CircuitAnalysis,
                 board_width: float = 50.0,
                 board_height: float = 35.0) -> DesignConstraints:
        """Generate all constraints from analysis."""
        constraints = DesignConstraints()

        # Generate placement constraints
        constraints.placement = self._generate_placement_constraints(analysis)

        # Generate routing constraints
        constraints.routing = self._generate_routing_constraints(analysis)

        # Generate zones
        constraints.zones = self._generate_zones(analysis, board_width, board_height)

        return constraints

    def _generate_placement_constraints(self, analysis: CircuitAnalysis) -> List[PlacementConstraint]:
        """Generate placement constraints."""
        constraints = []

        # Bypass caps near ICs
        bypass_caps = [ref for ref, comp in analysis.components.items()
                       if comp.function == ComponentFunction.CAPACITOR_BYPASS]
        ics = [ref for ref, comp in analysis.components.items()
               if comp.function in (ComponentFunction.MICROCONTROLLER,
                                    ComponentFunction.VOLTAGE_REGULATOR,
                                    ComponentFunction.MUX,
                                    ComponentFunction.AMPLIFIER)]

        for ic in ics:
            # Each IC should have a nearby bypass cap
            if bypass_caps:
                constraints.append(PlacementConstraint(
                    type='PROXIMITY',
                    component=bypass_caps[0] if bypass_caps else '',
                    target=ic,
                    value=5.0,  # mm
                    priority=1
                ))

        # Crystal near MCU
        crystals = [ref for ref, comp in analysis.components.items()
                    if comp.function == ComponentFunction.CRYSTAL]
        mcus = [ref for ref, comp in analysis.components.items()
                if comp.function == ComponentFunction.MICROCONTROLLER]

        for crystal in crystals:
            for mcu in mcus:
                constraints.append(PlacementConstraint(
                    type='PROXIMITY',
                    component=crystal,
                    target=mcu,
                    value=5.0,
                    priority=1
                ))

        # Thermal constraints
        for ref, comp in analysis.components.items():
            if comp.needs_copper_pour:
                constraints.append(PlacementConstraint(
                    type='THERMAL',
                    component=ref,
                    target=None,
                    value=comp.needs_thermal_vias,
                    priority=2
                ))

        return constraints

    def _generate_routing_constraints(self, analysis: CircuitAnalysis) -> List[RoutingConstraint]:
        """Generate routing constraints."""
        constraints = []

        for net_name, net in analysis.nets.items():
            # Width constraint
            if net.min_width > 0.25:
                constraints.append(RoutingConstraint(
                    type='WIDTH',
                    net=net_name,
                    value=net.min_width,
                    priority=1
                ))

            # Length constraint
            if net.max_length:
                constraints.append(RoutingConstraint(
                    type='MAX_LENGTH',
                    net=net_name,
                    value=net.max_length,
                    priority=1
                ))

            # Via limit
            if net.max_vias:
                constraints.append(RoutingConstraint(
                    type='MAX_VIAS',
                    net=net_name,
                    value=net.max_vias,
                    priority=1
                ))

            # Layer preference
            if net.preferred_layer:
                constraints.append(RoutingConstraint(
                    type='PREFERRED_LAYER',
                    net=net_name,
                    value=net.preferred_layer,
                    priority=2
                ))

            # Clearance for sensitive nets
            if net.function in (NetFunction.ANALOG_SIGNAL, NetFunction.SWITCH_NODE):
                constraints.append(RoutingConstraint(
                    type='CLEARANCE',
                    net=net_name,
                    value=0.3,  # Extra clearance
                    priority=1
                ))

        return constraints

    def _generate_zones(self, analysis: CircuitAnalysis,
                        board_width: float, board_height: float) -> Dict:
        """Generate placement zones."""
        zones = {}

        # Check if we have distinct circuit blocks
        has_power = any(b.function.name.startswith('POWER') for b in analysis.blocks)
        has_analog = any(b.function.name.startswith('ANALOG') for b in analysis.blocks)

        if has_power and has_analog:
            # Separate power and analog zones
            zones['POWER'] = (0, 0, board_width * 0.4, board_height)
            zones['ANALOG'] = (board_width * 0.6, 0, board_width, board_height)
            zones['DIGITAL'] = (board_width * 0.4, 0, board_width * 0.6, board_height)
        elif has_power:
            # Power on left
            zones['POWER'] = (0, 0, board_width * 0.3, board_height)

        return zones


def export_constraints_to_json(constraints: DesignConstraints) -> str:
    """Export constraints to JSON for PCB Engine."""
    import json

    data = {
        'placement': [
            {
                'type': c.type,
                'component': c.component,
                'target': c.target,
                'value': c.value,
                'priority': c.priority
            }
            for c in constraints.placement
        ],
        'routing': [
            {
                'type': c.type,
                'net': c.net,
                'value': c.value,
                'priority': c.priority
            }
            for c in constraints.routing
        ],
        'zones': constraints.zones,
        'keep_outs': constraints.keep_outs
    }

    return json.dumps(data, indent=2)
