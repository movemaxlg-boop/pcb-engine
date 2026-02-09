"""
Thermal Analyzer
=================

Analyzes thermal characteristics of PCB designs.
Predicts hot spots and recommends thermal management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

from .component_database import ComponentDatabase


@dataclass
class ThermalHotspot:
    """A thermal hotspot on the board."""
    component: str
    power_watts: float
    estimated_temp_c: float
    area_mm2: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ThermalAnalysisResult:
    """Complete thermal analysis result."""
    total_power_watts: float
    hotspots: List[ThermalHotspot]
    board_temp_rise: float  # Average board temperature rise
    needs_forced_cooling: bool
    recommendations: List[str]


class ThermalAnalyzer:
    """
    Thermal analysis for PCB designs.
    """

    def __init__(self, component_db: Optional[ComponentDatabase] = None):
        self.db = component_db or ComponentDatabase()

        # Thermal constants
        self.COPPER_THERMAL_CONDUCTIVITY = 385  # W/(m·K)
        self.FR4_THERMAL_CONDUCTIVITY = 0.3     # W/(m·K)
        self.STILL_AIR_H = 10  # W/(m²·K) convection coefficient

    def analyze(self, parts_db: Dict, placement: Optional[Dict] = None,
                board_width: float = 50.0, board_height: float = 35.0,
                ambient_temp: float = 25.0) -> ThermalAnalysisResult:
        """
        Complete thermal analysis.
        """
        hotspots = []
        total_power = 0.0
        recommendations = []

        for ref, part in parts_db.get('parts', {}).items():
            value = part.get('value', '')

            # Look up thermal data
            db_info = self.db.lookup(value)

            # Estimate power dissipation
            power = self._estimate_power(ref, part, db_info)

            if power > 0:
                total_power += power

                # Calculate temperature
                theta_ja = db_info.thermal.theta_ja if db_info else 100.0
                temp = ambient_temp + power * theta_ja

                hotspot = ThermalHotspot(
                    component=ref,
                    power_watts=power,
                    estimated_temp_c=temp,
                    area_mm2=0.0,  # Would need footprint data
                    recommendations=[]
                )

                # Add recommendations for hot components
                if temp > 85:
                    hotspot.recommendations.append(f'Add copper pour under {ref}')
                if temp > 100:
                    hotspot.recommendations.append(f'Add thermal vias under {ref}')
                if temp > 120:
                    hotspot.recommendations.append(f'Consider heatsink for {ref}')

                if power > 0.1:  # Only track significant heat sources
                    hotspots.append(hotspot)

        # Calculate board temperature rise
        board_area_m2 = (board_width * board_height) * 1e-6
        # Simple model: ΔT = P / (h × A × 2) (both sides)
        board_temp_rise = total_power / (self.STILL_AIR_H * board_area_m2 * 2) if board_area_m2 > 0 else 0

        # Determine if forced cooling needed
        needs_forced_cooling = board_temp_rise > 40 or any(h.estimated_temp_c > 100 for h in hotspots)

        # Overall recommendations
        if total_power > 5:
            recommendations.append('High power design - consider thermal simulation')
        if needs_forced_cooling:
            recommendations.append('Forced air cooling recommended')
        if total_power > 2 and board_area_m2 < 2500e-6:  # < 50x50mm
            recommendations.append('Board may be too small for thermal dissipation')

        # Sort hotspots by temperature
        hotspots.sort(key=lambda h: h.estimated_temp_c, reverse=True)

        return ThermalAnalysisResult(
            total_power_watts=total_power,
            hotspots=hotspots,
            board_temp_rise=board_temp_rise,
            needs_forced_cooling=needs_forced_cooling,
            recommendations=recommendations
        )

    def _estimate_power(self, ref: str, part: Dict, db_info) -> float:
        """Estimate power dissipation for a component."""
        value = part.get('value', '').upper()

        # Voltage regulators
        if any(reg in value for reg in ['LM2596', 'TPS54', 'MP1584']):
            # Switching: assume 90% efficiency, 1A output, 12V->5V
            return 0.5  # Rough estimate

        if any(reg in value for reg in ['AMS1117', 'LM1117', '7805', 'LDO']):
            # LDO: P = (Vin - Vout) × I, assume 7V->3.3V, 200mA
            return (7 - 3.3) * 0.2

        # LEDs
        if 'LED' in value:
            return 0.02  # 20mA × 2V drop

        # Resistors (would need circuit analysis)
        if ref.startswith('R'):
            return 0.01  # Assume small signal

        # ICs - use database if available
        if db_info and db_info.electrical.supply_current_typ > 0:
            # P = Vcc × Icc
            return 3.3 * (db_info.electrical.supply_current_typ / 1000)

        return 0.0

    def calculate_copper_pour_area(self, power_watts: float,
                                     max_temp_rise: float = 40) -> float:
        """
        Calculate copper pour area needed for heat dissipation.

        Returns area in mm².
        """
        # Simplified model: 1 cm² copper per 0.5W (for 40°C rise in still air)
        area_cm2 = power_watts / 0.5 * (40 / max_temp_rise)
        return area_cm2 * 100  # Convert to mm²

    def calculate_thermal_vias(self, power_watts: float,
                                 theta_via: float = 30) -> int:
        """
        Calculate number of thermal vias needed.

        Each via has ~30°C/W thermal resistance.
        Vias in parallel: θ_total = θ_via / N

        Returns number of vias.
        """
        # Target: reduce θja by 50%
        if power_watts < 0.1:
            return 0

        # For 1W dissipation, we want θ < 30°C/W
        # N = power × current_theta / target_temp_rise
        target_theta = 30 / power_watts if power_watts > 0 else 100
        n_vias = max(1, int(math.ceil(theta_via / target_theta)))

        return min(n_vias, 16)  # Cap at 16 vias
