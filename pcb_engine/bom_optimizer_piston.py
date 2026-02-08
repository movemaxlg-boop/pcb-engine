"""
BOM Optimizer Piston - PCB Engine Worker

Optimizes Bill of Materials for cost, availability, and sourcing:
- Cost optimization across suppliers
- Part consolidation (reduce unique part numbers)
- Alternate part suggestions
- Stock availability checking
- Lead time optimization
- Supplier preference handling

HIERARCHY:
    USER (Boss) → Circuit AI (Engineer) → PCB Engine (Foreman) → This Piston (Worker)

This piston receives work orders from the Foreman (PCB Engine) and reports back
with results, warnings, and any issues encountered.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from enum import Enum, auto
import json
import math


class PartCategory(Enum):
    """Component categories for grouping"""
    RESISTOR = auto()
    CAPACITOR = auto()
    INDUCTOR = auto()
    DIODE = auto()
    TRANSISTOR = auto()
    IC_ANALOG = auto()
    IC_DIGITAL = auto()
    IC_MIXED = auto()
    IC_POWER = auto()
    MCU = auto()
    CONNECTOR = auto()
    CRYSTAL = auto()
    LED = auto()
    SWITCH = auto()
    RELAY = auto()
    TRANSFORMER = auto()
    FUSE = auto()
    SENSOR = auto()
    MODULE = auto()
    MECHANICAL = auto()
    OTHER = auto()


class SupplierTier(Enum):
    """Supplier reliability tiers"""
    AUTHORIZED = auto()     # Official authorized distributor
    FRANCHISE = auto()      # Franchise distributor
    INDEPENDENT = auto()    # Independent distributor
    BROKER = auto()         # Broker (use with caution)
    DIRECT = auto()         # Direct from manufacturer


class OptimizationGoal(Enum):
    """BOM optimization priorities"""
    LOWEST_COST = auto()
    FASTEST_DELIVERY = auto()
    SINGLE_SUPPLIER = auto()
    HIGHEST_QUALITY = auto()
    BALANCED = auto()


@dataclass
class Supplier:
    """Supplier information"""
    name: str
    tier: SupplierTier
    lead_time_days: int = 3
    minimum_order: float = 0.0
    shipping_cost: float = 0.0
    reliability_score: float = 0.95  # 0-1
    api_available: bool = False

    def total_cost(self, parts_cost: float) -> float:
        """Calculate total cost including shipping"""
        return max(parts_cost, self.minimum_order) + self.shipping_cost


@dataclass
class PartOffer:
    """A specific offer for a part from a supplier"""
    supplier: Supplier
    part_number: str          # Supplier's part number
    manufacturer: str
    mpn: str                  # Manufacturer part number
    unit_price: float
    quantity_available: int
    lead_time_days: int
    minimum_order_qty: int = 1
    price_breaks: Dict[int, float] = field(default_factory=dict)
    packaging: str = "Cut Tape"
    is_authorized: bool = True

    def price_for_quantity(self, qty: int) -> float:
        """Get price for specific quantity using price breaks"""
        if not self.price_breaks:
            return self.unit_price

        # Find best applicable price break
        best_price = self.unit_price
        for break_qty, price in sorted(self.price_breaks.items()):
            if qty >= break_qty:
                best_price = price

        return best_price

    def total_for_quantity(self, qty: int) -> float:
        """Calculate total cost for quantity"""
        order_qty = max(qty, self.minimum_order_qty)
        return order_qty * self.price_for_quantity(order_qty)


@dataclass
class BOMLine:
    """A line item in the BOM"""
    designators: List[str]      # List of reference designators (R1, R2, etc.)
    quantity: int
    value: str                  # Value (10k, 100nF, etc.)
    footprint: str              # Package/footprint
    description: str
    manufacturer: str = ""
    mpn: str = ""               # Manufacturer part number
    category: PartCategory = PartCategory.OTHER
    dnp: bool = False           # Do Not Populate
    critical: bool = False      # Critical for function
    alternatives: List[str] = field(default_factory=list)  # Alternative MPNs

    @property
    def line_id(self) -> str:
        """Generate unique ID for this line"""
        return f"{self.value}_{self.footprint}_{self.mpn or 'generic'}"


@dataclass
class BOMLineResult:
    """Optimization result for a BOM line"""
    bom_line: BOMLine
    selected_offer: Optional[PartOffer]
    alternative_offers: List[PartOffer]
    total_cost: float
    lead_time_days: int
    warnings: List[str] = field(default_factory=list)
    consolidation_note: str = ""


@dataclass
class BOMOptimizationResult:
    """Complete BOM optimization result"""
    line_results: List[BOMLineResult]
    total_parts_cost: float
    total_shipping_cost: float
    grand_total: float
    suppliers_used: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    consolidation_savings: float = 0.0
    max_lead_time_days: int = 0


class BOMOptimizerPiston:
    """
    BOM Optimizer Piston - Optimizes Bill of Materials

    Capabilities:
    - Multi-supplier cost optimization
    - Part consolidation to reduce unique parts
    - Alternative part suggestions
    - Stock and lead time optimization
    - Price break analysis
    - Supplier preference handling
    """

    def __init__(self):
        self.name = "BOMOptimizer"
        self.version = "1.0.0"

        # Default supplier database
        self.suppliers: Dict[str, Supplier] = {
            "DigiKey": Supplier(
                name="DigiKey",
                tier=SupplierTier.AUTHORIZED,
                lead_time_days=2,
                minimum_order=0,
                shipping_cost=6.99,
                reliability_score=0.99,
                api_available=True
            ),
            "Mouser": Supplier(
                name="Mouser",
                tier=SupplierTier.AUTHORIZED,
                lead_time_days=2,
                minimum_order=0,
                shipping_cost=7.99,
                reliability_score=0.99,
                api_available=True
            ),
            "LCSC": Supplier(
                name="LCSC",
                tier=SupplierTier.FRANCHISE,
                lead_time_days=7,
                minimum_order=0,
                shipping_cost=15.00,
                reliability_score=0.90,
                api_available=True
            ),
            "Arrow": Supplier(
                name="Arrow",
                tier=SupplierTier.AUTHORIZED,
                lead_time_days=3,
                minimum_order=0,
                shipping_cost=0,  # Free shipping
                reliability_score=0.95,
                api_available=True
            ),
            "Newark": Supplier(
                name="Newark",
                tier=SupplierTier.AUTHORIZED,
                lead_time_days=3,
                minimum_order=0,
                shipping_cost=8.50,
                reliability_score=0.95,
                api_available=True
            ),
            "TME": Supplier(
                name="TME",
                tier=SupplierTier.FRANCHISE,
                lead_time_days=5,
                minimum_order=0,
                shipping_cost=12.00,
                reliability_score=0.90,
                api_available=False
            ),
        }

        # Part category detection patterns
        self.category_patterns = {
            PartCategory.RESISTOR: ['resistor', 'res', 'ohm', 'Ω'],
            PartCategory.CAPACITOR: ['capacitor', 'cap', 'farad', 'pf', 'nf', 'uf', 'μf'],
            PartCategory.INDUCTOR: ['inductor', 'ind', 'henry', 'nh', 'uh', 'mh', 'μh'],
            PartCategory.DIODE: ['diode', 'zener', 'schottky', 'tvs', 'esd'],
            PartCategory.TRANSISTOR: ['transistor', 'mosfet', 'bjt', 'fet', 'jfet'],
            PartCategory.LED: ['led', 'light emit'],
            PartCategory.IC_ANALOG: ['opamp', 'op-amp', 'comparator', 'amplifier', 'adc', 'dac'],
            PartCategory.IC_DIGITAL: ['logic', 'gate', 'flip-flop', 'counter', 'shift'],
            PartCategory.IC_POWER: ['regulator', 'ldo', 'buck', 'boost', 'power'],
            PartCategory.MCU: ['mcu', 'microcontroller', 'cpu', 'processor', 'esp32', 'stm32', 'atmega'],
            PartCategory.CONNECTOR: ['connector', 'header', 'socket', 'jack', 'plug', 'usb', 'hdmi'],
            PartCategory.CRYSTAL: ['crystal', 'oscillator', 'xtal', 'resonator', 'mhz', 'khz'],
            PartCategory.SWITCH: ['switch', 'button', 'tactile'],
            PartCategory.FUSE: ['fuse', 'ptc', 'polyfuse'],
            PartCategory.SENSOR: ['sensor', 'thermistor', 'ntc', 'ptc', 'accelerometer', 'gyro'],
        }

        # Value consolidation rules
        self.consolidation_rules = {
            PartCategory.RESISTOR: {
                'tolerance_allowed': ['1%', '5%'],  # Can substitute 1% for 5%
                'power_upgrade': True,  # Can use higher power rating
            },
            PartCategory.CAPACITOR: {
                'voltage_upgrade': True,  # Can use higher voltage rating
                'tolerance_upgrade': True,
            },
        }

    def optimize(self,
                 parts_db_or_bom: Any,
                 goal: OptimizationGoal = OptimizationGoal.BALANCED,
                 preferred_suppliers: Optional[List[str]] = None,
                 max_lead_time_days: Optional[int] = None,
                 production_quantity: int = 1,
                 offer_lookup: Optional[Callable[['BOMLine'], List[PartOffer]]] = None) -> 'BOMOptimizationResult':
        """
        Standard piston API - optimize BOM.

        Accepts either:
        - parts_db dict (converted to BOM)
        - List[BOMLine] (original format)
        """
        # Convert parts_db to BOM if needed
        if isinstance(parts_db_or_bom, dict) and 'parts' in parts_db_or_bom:
            bom = self._parts_db_to_bom(parts_db_or_bom)
        elif isinstance(parts_db_or_bom, list):
            bom = parts_db_or_bom
        else:
            # Assume it's a parts_db without 'parts' wrapper
            bom = []

        return self._optimize_internal(
            bom, goal, preferred_suppliers,
            max_lead_time_days, production_quantity, offer_lookup
        )

    def _parts_db_to_bom(self, parts_db: Dict) -> List['BOMLine']:
        """Convert parts_db to BOM lines"""
        bom = []
        parts = parts_db.get('parts', {})

        for ref, part in parts.items():
            if isinstance(part, dict):
                # Create BOMLine first without category
                bom_line = BOMLine(
                    designators=[ref],
                    quantity=1,
                    value=part.get('value', ''),
                    footprint=part.get('footprint', ''),
                    description=part.get('description', ''),
                    manufacturer=part.get('manufacturer', ''),
                    mpn=part.get('mpn', ''),
                    dnp=part.get('dnp', False)
                )
                # Then detect category using the BOMLine
                bom_line.category = self._detect_category(bom_line)
                bom.append(bom_line)

        return bom

    def _optimize_internal(self,
                 bom: List['BOMLine'],
                 goal: OptimizationGoal = OptimizationGoal.BALANCED,
                 preferred_suppliers: Optional[List[str]] = None,
                 max_lead_time_days: Optional[int] = None,
                 production_quantity: int = 1,
                 offer_lookup: Optional[Callable[['BOMLine'], List[PartOffer]]] = None) -> 'BOMOptimizationResult':
        """
        Main entry point - optimize BOM

        Args:
            bom: List of BOM lines
            goal: Optimization goal
            preferred_suppliers: List of preferred supplier names
            max_lead_time_days: Maximum acceptable lead time
            production_quantity: Number of boards to build
            offer_lookup: Optional function to look up offers (for real API integration)

        Returns:
            BOMOptimizationResult with optimized selections
        """
        warnings = []
        line_results = []

        # Filter suppliers based on preferences
        active_suppliers = self._filter_suppliers(preferred_suppliers)

        # First pass: Identify consolidation opportunities
        consolidation_groups = self._find_consolidation_opportunities(bom)

        # Process each BOM line
        for bom_line in bom:
            if bom_line.dnp:
                # Do Not Populate - skip but include in results
                line_results.append(BOMLineResult(
                    bom_line=bom_line,
                    selected_offer=None,
                    alternative_offers=[],
                    total_cost=0,
                    lead_time_days=0,
                    warnings=["DNP - Not ordering"]
                ))
                continue

            # Get offers for this part
            if offer_lookup:
                offers = offer_lookup(bom_line)
            else:
                offers = self._generate_mock_offers(bom_line, active_suppliers)

            # Filter by lead time if specified
            if max_lead_time_days:
                offers = [o for o in offers if o.lead_time_days <= max_lead_time_days]

            # Select best offer based on goal
            qty_needed = bom_line.quantity * production_quantity
            selected, alternatives = self._select_best_offer(offers, qty_needed, goal)

            # Build result
            line_warnings = []
            if not selected:
                line_warnings.append(f"No offers found for {bom_line.mpn or bom_line.value}")
            elif selected.quantity_available < qty_needed:
                line_warnings.append(f"Insufficient stock: need {qty_needed}, available {selected.quantity_available}")

            # Check if this line was consolidated
            consolidation_note = ""
            line_id = bom_line.line_id
            for group_name, group_lines in consolidation_groups.items():
                if line_id in group_lines and len(group_lines) > 1:
                    consolidation_note = f"Consolidated with {len(group_lines)-1} other lines"
                    break

            line_results.append(BOMLineResult(
                bom_line=bom_line,
                selected_offer=selected,
                alternative_offers=alternatives[:5],  # Top 5 alternatives
                total_cost=selected.total_for_quantity(qty_needed) if selected else 0,
                lead_time_days=selected.lead_time_days if selected else 0,
                warnings=line_warnings,
                consolidation_note=consolidation_note
            ))

            warnings.extend(line_warnings)

        # Calculate totals
        total_parts = sum(r.total_cost for r in line_results)

        # Calculate shipping (one order per supplier)
        suppliers_used = set()
        for r in line_results:
            if r.selected_offer:
                suppliers_used.add(r.selected_offer.supplier.name)

        total_shipping = sum(
            self.suppliers[s].shipping_cost
            for s in suppliers_used
            if s in self.suppliers
        )

        # Calculate max lead time
        max_lead = max((r.lead_time_days for r in line_results), default=0)

        # Calculate consolidation savings (estimated)
        consolidation_savings = self._estimate_consolidation_savings(consolidation_groups, bom)

        # Build statistics
        stats = self._calculate_statistics(bom, line_results, consolidation_groups)

        return BOMOptimizationResult(
            line_results=line_results,
            total_parts_cost=total_parts,
            total_shipping_cost=total_shipping,
            grand_total=total_parts + total_shipping,
            suppliers_used=list(suppliers_used),
            warnings=warnings,
            statistics=stats,
            consolidation_savings=consolidation_savings,
            max_lead_time_days=max_lead
        )

    def _filter_suppliers(self, preferred: Optional[List[str]]) -> Dict[str, Supplier]:
        """Filter suppliers based on preferences"""
        if not preferred:
            return self.suppliers

        return {
            name: supplier
            for name, supplier in self.suppliers.items()
            if name in preferred
        }

    def _find_consolidation_opportunities(self, bom: List[BOMLine]) -> Dict[str, List[str]]:
        """
        Find parts that can be consolidated

        Returns dict mapping consolidated part to list of original line IDs
        """
        groups: Dict[str, List[str]] = {}

        # Group by category and footprint
        for line in bom:
            if line.dnp:
                continue

            category = self._detect_category(line)

            # Create consolidation key
            key = f"{category.name}_{line.footprint}"

            if key not in groups:
                groups[key] = []

            groups[key].append(line.line_id)

        # Filter to groups with multiple items
        return {k: v for k, v in groups.items() if len(v) > 1}

    def _detect_category(self, line: BOMLine) -> PartCategory:
        """Detect part category from description and value"""
        text = f"{line.description} {line.value} {line.designators[0] if line.designators else ''}".lower()

        # Check designator prefix
        if line.designators:
            prefix = line.designators[0][0].upper()
            prefix_map = {
                'R': PartCategory.RESISTOR,
                'C': PartCategory.CAPACITOR,
                'L': PartCategory.INDUCTOR,
                'D': PartCategory.DIODE,
                'Q': PartCategory.TRANSISTOR,
                'U': PartCategory.IC_DIGITAL,
                'J': PartCategory.CONNECTOR,
                'Y': PartCategory.CRYSTAL,
                'F': PartCategory.FUSE,
                'S': PartCategory.SWITCH,
            }
            if prefix in prefix_map:
                return prefix_map[prefix]

        # Pattern matching
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return category

        return line.category if line.category != PartCategory.OTHER else PartCategory.OTHER

    def _generate_mock_offers(self,
                              line: BOMLine,
                              suppliers: Dict[str, Supplier]) -> List[PartOffer]:
        """
        Generate mock offers for testing

        In production, this would call actual supplier APIs
        """
        offers = []

        # Base price estimation
        category = self._detect_category(line)
        base_price = self._estimate_base_price(line, category)

        for name, supplier in suppliers.items():
            # Vary price by supplier
            price_factor = {
                "DigiKey": 1.0,
                "Mouser": 1.02,
                "LCSC": 0.6,
                "Arrow": 0.95,
                "Newark": 1.05,
                "TME": 0.8,
            }.get(name, 1.0)

            unit_price = base_price * price_factor

            # Generate price breaks
            price_breaks = {
                10: unit_price * 0.95,
                25: unit_price * 0.90,
                100: unit_price * 0.80,
                500: unit_price * 0.70,
                1000: unit_price * 0.60,
            }

            # Simulate stock availability
            stock = 1000 if supplier.tier == SupplierTier.AUTHORIZED else 500

            offers.append(PartOffer(
                supplier=supplier,
                part_number=f"{name[:2].upper()}-{line.mpn or line.value}",
                manufacturer=line.manufacturer or "Generic",
                mpn=line.mpn or f"GEN-{line.value}",
                unit_price=unit_price,
                quantity_available=stock,
                lead_time_days=supplier.lead_time_days,
                minimum_order_qty=1,
                price_breaks=price_breaks,
                is_authorized=supplier.tier == SupplierTier.AUTHORIZED
            ))

        return offers

    def _estimate_base_price(self, line: BOMLine, category: PartCategory) -> float:
        """Estimate base unit price for a part"""
        # Very rough estimates for demonstration
        base_prices = {
            PartCategory.RESISTOR: 0.01,
            PartCategory.CAPACITOR: 0.02,
            PartCategory.INDUCTOR: 0.15,
            PartCategory.DIODE: 0.05,
            PartCategory.TRANSISTOR: 0.10,
            PartCategory.LED: 0.05,
            PartCategory.IC_ANALOG: 0.50,
            PartCategory.IC_DIGITAL: 0.30,
            PartCategory.IC_POWER: 0.80,
            PartCategory.MCU: 3.00,
            PartCategory.CONNECTOR: 0.50,
            PartCategory.CRYSTAL: 0.30,
            PartCategory.SWITCH: 0.15,
            PartCategory.FUSE: 0.20,
            PartCategory.SENSOR: 2.00,
            PartCategory.OTHER: 0.50,
        }

        base = base_prices.get(category, 0.50)

        # Adjust for footprint size (larger = slightly more expensive)
        footprint = line.footprint.upper()
        if '0201' in footprint:
            base *= 1.2  # Tiny parts cost more
        elif '1206' in footprint or '1210' in footprint:
            base *= 1.1
        elif '2512' in footprint:
            base *= 1.3

        return base

    def _select_best_offer(self,
                           offers: List[PartOffer],
                           quantity: int,
                           goal: OptimizationGoal) -> Tuple[Optional[PartOffer], List[PartOffer]]:
        """Select best offer based on optimization goal"""
        if not offers:
            return None, []

        def score_offer(offer: PartOffer) -> float:
            """Score an offer (lower is better)"""
            total_cost = offer.total_for_quantity(quantity)
            lead_time = offer.lead_time_days
            reliability = offer.supplier.reliability_score

            if goal == OptimizationGoal.LOWEST_COST:
                return total_cost

            elif goal == OptimizationGoal.FASTEST_DELIVERY:
                return lead_time * 1000 + total_cost

            elif goal == OptimizationGoal.SINGLE_SUPPLIER:
                # Prefer already-used suppliers (handled at higher level)
                return total_cost

            elif goal == OptimizationGoal.HIGHEST_QUALITY:
                # Prefer authorized distributors
                tier_penalty = {
                    SupplierTier.AUTHORIZED: 0,
                    SupplierTier.FRANCHISE: 100,
                    SupplierTier.INDEPENDENT: 500,
                    SupplierTier.BROKER: 1000,
                    SupplierTier.DIRECT: 50,
                }.get(offer.supplier.tier, 200)
                return tier_penalty + total_cost

            else:  # BALANCED
                # Weighted combination
                cost_score = total_cost
                time_score = lead_time * 10
                quality_score = (1 - reliability) * 500
                return cost_score + time_score + quality_score

        # Sort by score
        sorted_offers = sorted(offers, key=score_offer)

        return sorted_offers[0], sorted_offers[1:]

    def _estimate_consolidation_savings(self,
                                        groups: Dict[str, List[str]],
                                        bom: List[BOMLine]) -> float:
        """Estimate savings from part consolidation"""
        # Rough estimate: each unique part reduction saves setup/handling costs
        unique_before = len([l for l in bom if not l.dnp])
        unique_after = unique_before - sum(len(g) - 1 for g in groups.values())

        parts_consolidated = unique_before - unique_after
        estimated_savings_per_part = 0.50  # Setup/handling cost estimate

        return parts_consolidated * estimated_savings_per_part

    def _calculate_statistics(self,
                              bom: List[BOMLine],
                              results: List[BOMLineResult],
                              consolidation_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate BOM statistics"""
        total_parts = sum(line.quantity for line in bom if not line.dnp)
        unique_parts = len([line for line in bom if not line.dnp])

        # Count by category
        category_counts: Dict[str, int] = {}
        for line in bom:
            if line.dnp:
                continue
            cat = self._detect_category(line)
            category_counts[cat.name] = category_counts.get(cat.name, 0) + line.quantity

        # Cost breakdown by category
        category_costs: Dict[str, float] = {}
        for result in results:
            if result.selected_offer:
                cat = self._detect_category(result.bom_line)
                category_costs[cat.name] = category_costs.get(cat.name, 0) + result.total_cost

        # Find most expensive parts
        expensive_parts = sorted(
            [(r.bom_line.designators[0] if r.bom_line.designators else "?",
              r.bom_line.value,
              r.total_cost)
             for r in results if r.total_cost > 0],
            key=lambda x: x[2],
            reverse=True
        )[:5]

        return {
            'total_part_count': total_parts,
            'unique_part_count': unique_parts,
            'category_counts': category_counts,
            'category_costs': category_costs,
            'consolidation_groups': len(consolidation_groups),
            'parts_consolidated': sum(len(g) - 1 for g in consolidation_groups.values()),
            'most_expensive_parts': expensive_parts,
            'dnp_count': len([l for l in bom if l.dnp]),
            'critical_parts': len([l for l in bom if l.critical])
        }

    def suggest_alternatives(self, line: BOMLine) -> List[Dict[str, Any]]:
        """
        Suggest alternative parts for a BOM line

        Returns list of alternatives with compatibility info
        """
        alternatives = []
        category = self._detect_category(line)

        if category == PartCategory.RESISTOR:
            # For resistors, suggest different tolerances/power ratings
            alternatives.append({
                'type': 'tolerance_upgrade',
                'description': f"Use 1% tolerance instead of 5%",
                'benefit': "Better precision, minor cost increase",
                'compatibility': 'full'
            })

        elif category == PartCategory.CAPACITOR:
            # Suggest higher voltage rating
            alternatives.append({
                'type': 'voltage_upgrade',
                'description': "Use higher voltage rating for reliability margin",
                'benefit': "Better reliability, same cost in most cases",
                'compatibility': 'full'
            })

            # Suggest ceramic vs electrolytic
            if 'elec' in line.description.lower() or 'μf' in line.value.lower():
                alternatives.append({
                    'type': 'technology_change',
                    'description': "Consider MLCC ceramic instead of electrolytic",
                    'benefit': "Smaller size, longer life, no polarity",
                    'compatibility': 'check_value'
                })

        elif category == PartCategory.IC_POWER:
            # Suggest pin-compatible alternatives
            alternatives.append({
                'type': 'pin_compatible',
                'description': "Check for pin-compatible regulators from other manufacturers",
                'benefit': "Potential cost savings or better availability",
                'compatibility': 'check_pinout'
            })

        return alternatives

    def generate_order_report(self, result: BOMOptimizationResult) -> str:
        """Generate human-readable order report"""
        lines = [
            "=" * 60,
            "BOM OPTIMIZATION REPORT",
            "=" * 60,
            "",
            f"Total Parts Cost:     ${result.total_parts_cost:,.2f}",
            f"Total Shipping:       ${result.total_shipping_cost:,.2f}",
            f"Grand Total:          ${result.grand_total:,.2f}",
            f"Consolidation Savings: ${result.consolidation_savings:,.2f}",
            "",
            f"Suppliers Used: {', '.join(result.suppliers_used)}",
            f"Max Lead Time:  {result.max_lead_time_days} days",
            "",
            "-" * 60,
            "ORDER DETAILS BY SUPPLIER",
            "-" * 60,
        ]

        # Group by supplier
        by_supplier: Dict[str, List[BOMLineResult]] = {}
        for lr in result.line_results:
            if lr.selected_offer:
                supplier = lr.selected_offer.supplier.name
                if supplier not in by_supplier:
                    by_supplier[supplier] = []
                by_supplier[supplier].append(lr)

        for supplier, items in by_supplier.items():
            supplier_total = sum(item.total_cost for item in items)
            lines.append(f"\n{supplier} (${supplier_total:,.2f}):")
            lines.append("-" * 40)

            for item in items:
                ref = ', '.join(item.bom_line.designators[:3])
                if len(item.bom_line.designators) > 3:
                    ref += f"... (+{len(item.bom_line.designators)-3})"

                lines.append(
                    f"  {ref:20} {item.bom_line.value:15} "
                    f"Qty:{item.bom_line.quantity:4}  ${item.total_cost:,.2f}"
                )

        # Warnings section
        if result.warnings:
            lines.append("\n" + "-" * 60)
            lines.append("WARNINGS")
            lines.append("-" * 60)
            for w in result.warnings:
                lines.append(f"  ⚠ {w}")

        # Statistics
        lines.append("\n" + "-" * 60)
        lines.append("STATISTICS")
        lines.append("-" * 60)
        stats = result.statistics
        lines.append(f"  Total Parts:        {stats.get('total_part_count', 0)}")
        lines.append(f"  Unique Parts:       {stats.get('unique_part_count', 0)}")
        lines.append(f"  Parts Consolidated: {stats.get('parts_consolidated', 0)}")
        lines.append(f"  DNP Components:     {stats.get('dnp_count', 0)}")
        lines.append(f"  Critical Parts:     {stats.get('critical_parts', 0)}")

        if stats.get('most_expensive_parts'):
            lines.append("\n  Most Expensive Parts:")
            for ref, value, cost in stats['most_expensive_parts']:
                lines.append(f"    {ref}: {value} - ${cost:,.2f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def export_to_csv(self, result: BOMOptimizationResult) -> str:
        """Export optimization result to CSV format"""
        lines = [
            "Reference,Value,Footprint,Quantity,Supplier,Part Number,Unit Price,Total,Lead Time"
        ]

        for lr in result.line_results:
            refs = ';'.join(lr.bom_line.designators)
            value = lr.bom_line.value.replace(',', ';')
            footprint = lr.bom_line.footprint

            if lr.selected_offer:
                offer = lr.selected_offer
                lines.append(
                    f'"{refs}","{value}","{footprint}",'
                    f'{lr.bom_line.quantity},{offer.supplier.name},'
                    f'"{offer.part_number}",{offer.unit_price:.4f},'
                    f'{lr.total_cost:.2f},{offer.lead_time_days}'
                )
            else:
                lines.append(
                    f'"{refs}","{value}","{footprint}",'
                    f'{lr.bom_line.quantity},NO OFFER,,,,,'
                )

        return "\n".join(lines)


# Convenience function for direct use
def optimize_bom(bom_data: List[Dict[str, Any]],
                 goal: str = "balanced",
                 preferred_suppliers: Optional[List[str]] = None,
                 quantity: int = 1) -> Dict[str, Any]:
    """
    Optimize a BOM from raw data

    Args:
        bom_data: List of dicts with keys: designators, quantity, value, footprint, etc.
        goal: "lowest_cost", "fastest_delivery", "single_supplier", "highest_quality", "balanced"
        preferred_suppliers: List of preferred supplier names
        quantity: Production quantity

    Returns:
        Optimization result as dict
    """
    piston = BOMOptimizerPiston()

    # Convert dict data to BOMLine objects
    bom_lines = []
    for item in bom_data:
        bom_lines.append(BOMLine(
            designators=item.get('designators', [item.get('reference', 'X?')]),
            quantity=item.get('quantity', 1),
            value=item.get('value', ''),
            footprint=item.get('footprint', ''),
            description=item.get('description', ''),
            manufacturer=item.get('manufacturer', ''),
            mpn=item.get('mpn', ''),
            dnp=item.get('dnp', False),
            critical=item.get('critical', False),
            alternatives=item.get('alternatives', [])
        ))

    # Map goal string to enum
    goal_map = {
        "lowest_cost": OptimizationGoal.LOWEST_COST,
        "fastest_delivery": OptimizationGoal.FASTEST_DELIVERY,
        "single_supplier": OptimizationGoal.SINGLE_SUPPLIER,
        "highest_quality": OptimizationGoal.HIGHEST_QUALITY,
        "balanced": OptimizationGoal.BALANCED,
    }
    opt_goal = goal_map.get(goal.lower(), OptimizationGoal.BALANCED)

    result = piston.optimize(
        bom=bom_lines,
        goal=opt_goal,
        preferred_suppliers=preferred_suppliers,
        production_quantity=quantity
    )

    # Generate report
    report = piston.generate_order_report(result)
    csv_export = piston.export_to_csv(result)

    return {
        'success': True,
        'total_cost': result.grand_total,
        'parts_cost': result.total_parts_cost,
        'shipping_cost': result.total_shipping_cost,
        'suppliers': result.suppliers_used,
        'max_lead_time': result.max_lead_time_days,
        'warnings': result.warnings,
        'statistics': result.statistics,
        'report': report,
        'csv': csv_export
    }
