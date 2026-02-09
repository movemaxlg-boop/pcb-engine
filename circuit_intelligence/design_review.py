"""
Design Review AI
=================

Expert-level design review that catches issues DRC misses.
This is the "experienced engineer looking over your shoulder".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .circuit_types import Severity, DesignIssue


class DesignReviewAI:
    """
    AI-powered design review.
    Catches issues that require engineering judgment.
    """

    def __init__(self):
        # Review categories
        self.reviewers = [
            self._review_power_integrity,
            self._review_signal_integrity,
            self._review_thermal,
            self._review_emi,
            self._review_manufacturing,
            self._review_reliability,
            self._review_component_selection,
        ]

    def review(self, parts_db: Dict, placement: Optional[Dict] = None,
               board_width: float = 50.0, board_height: float = 35.0) -> List[DesignIssue]:
        """Perform complete design review."""
        all_issues = []

        for reviewer in self.reviewers:
            issues = reviewer(parts_db, placement, board_width, board_height)
            all_issues.extend(issues)

        # Sort by severity
        all_issues.sort(key=lambda x: x.severity.value)

        return all_issues

    def _review_power_integrity(self, parts_db: Dict, placement: Optional[Dict],
                                  board_width: float, board_height: float) -> List[DesignIssue]:
        """Review power integrity."""
        issues = []
        parts = parts_db.get('parts', {})
        nets = parts_db.get('nets', {})

        # Check 1: Every IC needs bypass cap
        ics = [ref for ref in parts if ref.startswith('U')]
        bypass_caps = [ref for ref, p in parts.items()
                       if ref.startswith('C') and
                       ('100n' in p.get('value', '').lower() or '0.1u' in p.get('value', '').lower())]

        for ic in ics:
            # Check if IC has nearby bypass cap (if placement available)
            if placement and ic in placement:
                ic_pos = placement[ic]
                has_nearby_cap = False

                for cap in bypass_caps:
                    if cap in placement:
                        cap_pos = placement[cap]
                        dist = ((ic_pos[0] - cap_pos[0])**2 + (ic_pos[1] - cap_pos[1])**2)**0.5
                        if dist < 5.0:
                            has_nearby_cap = True
                            break

                if not has_nearby_cap:
                    issues.append(DesignIssue(
                        severity=Severity.ERROR,
                        category='POWER_INTEGRITY',
                        component=ic,
                        net=None,
                        message=f'{ic} has no bypass cap within 5mm',
                        recommendation='Add 100nF ceramic cap close to VCC/GND pins',
                    ))

        # Check 2: Bulk capacitor present
        bulk_caps = [ref for ref, p in parts.items()
                     if ref.startswith('C') and
                     any(v in p.get('value', '').lower() for v in ['10u', '22u', '47u', '100u'])]

        if ics and not bulk_caps:
            issues.append(DesignIssue(
                severity=Severity.WARNING,
                category='POWER_INTEGRITY',
                component=None,
                net=None,
                message='No bulk capacitor found',
                recommendation='Add 10-100uF bulk cap near power input',
            ))

        # Check 3: Power net width (if we had routing info)
        power_nets = [n for n in nets if any(p in n.upper() for p in ['VCC', 'VIN', 'VDD', '+5', '+3'])]
        for net in power_nets:
            pin_count = len(nets[net].get('pins', []))
            if pin_count > 5:
                issues.append(DesignIssue(
                    severity=Severity.INFO,
                    category='POWER_INTEGRITY',
                    component=None,
                    net=net,
                    message=f'Power net {net} has {pin_count} connections',
                    recommendation='Consider wider traces (0.5mm+) or power plane',
                ))

        return issues

    def _review_signal_integrity(self, parts_db: Dict, placement: Optional[Dict],
                                   board_width: float, board_height: float) -> List[DesignIssue]:
        """Review signal integrity."""
        issues = []
        nets = parts_db.get('nets', {})

        # Check for high-speed signals
        high_speed_nets = []
        for net_name in nets:
            name_upper = net_name.upper()
            if any(hs in name_upper for hs in ['USB', 'ETH', 'CLK', 'SCLK', 'MISO', 'MOSI']):
                high_speed_nets.append(net_name)

        if high_speed_nets:
            issues.append(DesignIssue(
                severity=Severity.INFO,
                category='SIGNAL_INTEGRITY',
                component=None,
                net=None,
                message=f'High-speed signals detected: {", ".join(high_speed_nets[:3])}...',
                recommendation='Keep traces short, use ground plane, avoid vias',
            ))

        # Check for USB without ESD
        has_usb = any('USB' in n.upper() for n in nets)
        has_esd = any('ESD' in p.get('value', '').upper() or 'TVS' in p.get('value', '').upper()
                      for p in parts_db.get('parts', {}).values())

        if has_usb and not has_esd:
            issues.append(DesignIssue(
                severity=Severity.WARNING,
                category='SIGNAL_INTEGRITY',
                component=None,
                net=None,
                message='USB signals without ESD protection',
                recommendation='Add ESD protection IC near USB connector',
            ))

        return issues

    def _review_thermal(self, parts_db: Dict, placement: Optional[Dict],
                         board_width: float, board_height: float) -> List[DesignIssue]:
        """Review thermal considerations."""
        issues = []
        parts = parts_db.get('parts', {})

        # Identify potentially hot components
        hot_parts = []
        for ref, part in parts.items():
            value = part.get('value', '').upper()
            if any(reg in value for reg in ['LM2596', 'TPS54', 'AMS1117', 'LM1117', '7805']):
                hot_parts.append(ref)

        if hot_parts:
            issues.append(DesignIssue(
                severity=Severity.INFO,
                category='THERMAL',
                component=hot_parts[0] if hot_parts else None,
                net=None,
                message=f'Voltage regulator(s) detected: {", ".join(hot_parts)}',
                recommendation='Ensure adequate copper pour and/or thermal vias',
            ))

        # Check board size vs component count (thermal density)
        board_area = board_width * board_height
        component_count = len(parts)
        density = component_count / (board_area / 100)  # per cm²

        if density > 2.0:
            issues.append(DesignIssue(
                severity=Severity.WARNING,
                category='THERMAL',
                component=None,
                net=None,
                message=f'High component density ({density:.1f}/cm²)',
                recommendation='May have thermal issues - consider larger board or forced cooling',
            ))

        return issues

    def _review_emi(self, parts_db: Dict, placement: Optional[Dict],
                     board_width: float, board_height: float) -> List[DesignIssue]:
        """Review EMI/EMC considerations."""
        issues = []
        parts = parts_db.get('parts', {})

        # Check for switching regulators
        has_switcher = any(
            any(sw in p.get('value', '').upper() for sw in ['LM2596', 'TPS54', 'MP1584', 'BUCK', 'BOOST'])
            for p in parts.values()
        )

        if has_switcher:
            issues.append(DesignIssue(
                severity=Severity.WARNING,
                category='EMI',
                component=None,
                net=None,
                message='Switching regulator detected - EMI concern',
                recommendation='Keep switch node trace short, minimize loop area, use input ceramic cap',
            ))

        # Check for crystals
        has_crystal = any(ref.startswith(('Y', 'X')) for ref in parts)
        if has_crystal:
            issues.append(DesignIssue(
                severity=Severity.INFO,
                category='EMI',
                component=None,
                net=None,
                message='Crystal oscillator detected',
                recommendation='Keep crystal close to MCU, route ground plane underneath',
            ))

        return issues

    def _review_manufacturing(self, parts_db: Dict, placement: Optional[Dict],
                                board_width: float, board_height: float) -> List[DesignIssue]:
        """Review manufacturing considerations."""
        issues = []
        parts = parts_db.get('parts', {})

        # Check for mixed SMD/THT
        smd_parts = []
        tht_parts = []

        for ref, part in parts.items():
            footprint = part.get('footprint', '').upper()
            if any(t in footprint for t in ['0402', '0603', '0805', '1206', 'SOT', 'SOIC', 'QFN', 'QFP']):
                smd_parts.append(ref)
            elif any(t in footprint for t in ['DIP', 'TO-220', 'HEADER', 'CONN']):
                tht_parts.append(ref)

        if smd_parts and tht_parts:
            issues.append(DesignIssue(
                severity=Severity.INFO,
                category='MANUFACTURING',
                component=None,
                net=None,
                message=f'Mixed SMD ({len(smd_parts)}) and THT ({len(tht_parts)}) components',
                recommendation='Consider all-SMD for easier automated assembly',
            ))

        # Check for very small components
        small_parts = [ref for ref, p in parts.items()
                       if '0402' in p.get('footprint', '').upper()]
        if small_parts:
            issues.append(DesignIssue(
                severity=Severity.INFO,
                category='MANUFACTURING',
                component=None,
                net=None,
                message=f'{len(small_parts)} 0402 components detected',
                recommendation='0402 is difficult for hand assembly - consider 0603 if prototyping',
            ))

        return issues

    def _review_reliability(self, parts_db: Dict, placement: Optional[Dict],
                             board_width: float, board_height: float) -> List[DesignIssue]:
        """Review reliability considerations."""
        issues = []
        parts = parts_db.get('parts', {})

        # Check for protection devices
        has_fuse = any(ref.startswith('F') for ref in parts)
        has_tvs = any('TVS' in p.get('value', '').upper() for p in parts.values())
        has_polarity_protection = any(
            any(d in p.get('value', '').upper() for d in ['SS34', 'B340', 'SCHOTTKY'])
            for ref, p in parts.items() if ref.startswith('D')
        )

        # Power input connector present?
        has_power_connector = any(
            ref.startswith('J') and
            any('VIN' in pin.get('net', '').upper() or 'VCC' in pin.get('net', '').upper()
                for pin in p.get('pins', []))
            for ref, p in parts.items()
        )

        if has_power_connector:
            if not has_fuse:
                issues.append(DesignIssue(
                    severity=Severity.WARNING,
                    category='RELIABILITY',
                    component=None,
                    net=None,
                    message='No fuse protection on power input',
                    recommendation='Add resettable fuse (PTC) for overcurrent protection',
                ))

            if not has_polarity_protection:
                issues.append(DesignIssue(
                    severity=Severity.INFO,
                    category='RELIABILITY',
                    component=None,
                    net=None,
                    message='No reverse polarity protection',
                    recommendation='Consider Schottky diode or P-FET for polarity protection',
                ))

        return issues

    def _review_component_selection(self, parts_db: Dict, placement: Optional[Dict],
                                      board_width: float, board_height: float) -> List[DesignIssue]:
        """Review component selection."""
        issues = []
        parts = parts_db.get('parts', {})

        # Check for obsolete/uncommon parts
        for ref, part in parts.items():
            value = part.get('value', '').upper()

            # LM7805 is old, suggest modern LDO
            if '7805' in value:
                issues.append(DesignIssue(
                    severity=Severity.INFO,
                    category='COMPONENT',
                    component=ref,
                    net=None,
                    message=f'{ref}: LM7805 has high dropout (2V)',
                    recommendation='Consider AMS1117 or AP2112 for lower dropout',
                ))

        return issues
