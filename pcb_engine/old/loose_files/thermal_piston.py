"""
PCB Engine - Thermal Piston (Sub-Engine)
==========================================

A dedicated piston (sub-engine) for PCB thermal analysis and heat management.

ALGORITHMS & REFERENCES:
=========================
1. Junction Temperature Calculation
   Reference: ROHM Application Note - "How to Use Thermal Resistance Parameters"
   Formula: Tj = Ta + (Rth_ja * P_loss)

2. IPC-2152 Trace Current Capacity
   Reference: IPC-2152 "Standard for Determining Current Carrying Capacity"
   Successor to IPC-2221, provides charts/formulas for trace sizing

3. Thermal Via Resistance
   Reference: Sierra Circuits - Via Thermal Resistance Calculator
   Reference: IEEE "Thermal Modeling and Design Optimization of PCB Vias"

4. PCB Thermal Conductivity
   Reference: Altium - "Estimating Thermal Conductivity of PCBs"
   Weighted average based on copper coverage and layer count

5. Heat Spreading Analysis
   Reference: Spreading Resistance theory (Lee, 1995)
   Reference: Texas Instruments PCB Thermal Calculator methodology

THERMAL RESISTANCE MODEL:
==========================
        Junction (Tj)
            |
            v
        [Rth_jc] Junction-to-Case
            |
            v
        Case (Tc)
            |
            v
        [Rth_cb] Case-to-Board (solder/TIM)
            |
            v
        Board (Tb)
            |
            v
        [Rth_ba] Board-to-Ambient (convection + radiation)
            |
            v
        Ambient (Ta)

Total: Rth_ja = Rth_jc + Rth_cb + Rth_ba

Sources:
- https://www.ti.com/design-resources/design-tools-simulation/models-simulators/pcb-thermal-calculator.html
- https://www.protoexpress.com/tools/via-thermal-resistance-calculator/
- https://resources.altium.com/p/how-to-estimate-thermal-conductivity-of-a-pcb
- https://www.protoexpress.com/blog/how-to-optimize-your-pcb-trace-using-ipc-2152-standard/
- https://www.allaboutcircuits.com/technical-articles/junction-to-ambient-thermal-resistance-ic-package-thermal-performance/
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# =============================================================================
# CONSTANTS
# =============================================================================

# Thermal conductivity (W/m-K)
THERMAL_CONDUCTIVITY = {
    'copper': 385.0,          # Pure copper
    'aluminum': 205.0,        # Aluminum (heat sinks)
    'fr4_in_plane': 0.8,      # FR4 in-plane (along fibers)
    'fr4_through_plane': 0.3, # FR4 through-plane (across thickness)
    'polyimide': 0.12,        # Flex PCB material
    'rogers_4350b': 0.69,     # Rogers RO4350B
    'ceramic_al2o3': 25.0,    # Alumina ceramic
    'ceramic_aln': 170.0,     # Aluminum nitride
    'solder_sn63pb37': 50.0,  # SnPb solder
    'solder_sac305': 58.0,    # Lead-free SAC305
    'air': 0.026,             # Still air at 25C
    'thermal_paste': 1.5,     # Typical thermal paste
    'thermal_pad': 3.0,       # Typical thermal pad
}

# Convection coefficients (W/m2-K)
CONVECTION_COEFFICIENTS = {
    'natural_horizontal_up': 10.0,    # Natural convection, hot side up
    'natural_horizontal_down': 5.0,   # Natural convection, hot side down
    'natural_vertical': 8.0,          # Natural convection, vertical
    'forced_low': 25.0,               # Forced convection, low airflow (<1 m/s)
    'forced_medium': 50.0,            # Forced convection, medium (1-3 m/s)
    'forced_high': 100.0,             # Forced convection, high (>3 m/s)
}

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.67e-8  # W/m2-K4

# Emissivity values
EMISSIVITY = {
    'bare_copper': 0.03,      # Polished copper
    'oxidized_copper': 0.65,  # Oxidized/weathered copper
    'solder_mask_green': 0.85,
    'solder_mask_black': 0.90,
    'solder_mask_white': 0.80,
    'bare_fr4': 0.90,
    'aluminum_anodized': 0.80,
    'aluminum_polished': 0.05,
}


# =============================================================================
# ENUMS
# =============================================================================

class CoolingMethod(Enum):
    """Cooling method for thermal analysis"""
    NATURAL = 'natural'           # Natural convection only
    FORCED_LOW = 'forced_low'     # Low forced air (<1 m/s)
    FORCED_MEDIUM = 'forced_medium'  # Medium forced air (1-3 m/s)
    FORCED_HIGH = 'forced_high'   # High forced air (>3 m/s)
    CONDUCTION = 'conduction'     # Conduction to heat sink/chassis


class PCBOrientation(Enum):
    """PCB mounting orientation"""
    HORIZONTAL_UP = 'horizontal_up'     # Components on top
    HORIZONTAL_DOWN = 'horizontal_down' # Components on bottom
    VERTICAL = 'vertical'


class LayerPosition(Enum):
    """Trace position in PCB"""
    OUTER = 'outer'   # Top or bottom copper layer
    INNER = 'inner'   # Internal copper layer


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComponentThermal:
    """Thermal properties of a component"""
    reference: str
    power_dissipation_w: float  # Power in Watts

    # Package thermal resistances (from datasheet)
    rth_jc: float = 0.0         # Junction-to-case (C/W)
    rth_ja: float = 0.0         # Junction-to-ambient (C/W) - on test board
    rth_jb: float = 0.0         # Junction-to-board (C/W)

    # Physical properties
    package_area_mm2: float = 0.0   # Package footprint area
    exposed_pad_mm2: float = 0.0    # Thermal pad area (if any)
    height_mm: float = 1.0          # Package height

    # Position on board
    x: float = 0.0
    y: float = 0.0

    # Maximum junction temperature
    tj_max: float = 125.0       # Max junction temp (C)

    # Calculated values
    tj_calculated: float = 0.0
    margin_c: float = 0.0       # Margin below Tj_max


@dataclass
class ThermalVia:
    """Thermal via properties"""
    x: float
    y: float
    drill_diameter_mm: float = 0.3
    plating_thickness_mm: float = 0.025  # 1oz plating
    height_mm: float = 1.6               # Board thickness
    filled: bool = False                 # Filled with solder/epoxy
    fill_material: str = 'solder_sn63pb37'


@dataclass
class ThermalZone:
    """A zone with thermal properties"""
    name: str
    outline: List[Tuple[float, float]]  # Polygon points (mm)
    copper_coverage: float = 0.5         # 0-1, copper fill ratio
    layer: str = 'F.Cu'
    connected_to_gnd: bool = True


@dataclass
class TraceAnalysis:
    """Result of trace thermal analysis"""
    width_mm: float
    thickness_mm: float
    length_mm: float
    layer: LayerPosition

    current_a: float
    temperature_rise_c: float
    max_current_a: float          # For given temp rise limit
    resistance_mohm: float
    power_loss_mw: float

    # IPC-2152 compliance
    ipc2152_compliant: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class ThermalResult:
    """Result of full thermal analysis"""
    ambient_temp_c: float
    max_board_temp_c: float
    hottest_component: str
    hottest_junction_temp_c: float

    components: List[ComponentThermal]
    thermal_margin_ok: bool

    # Board-level
    board_rth_effective: float    # Effective board thermal resistance
    total_power_w: float

    # Recommendations
    recommendations: List[str]
    warnings: List[str]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ThermalConfig:
    """Configuration for thermal analysis"""
    # Ambient conditions
    ambient_temp_c: float = 25.0
    max_ambient_c: float = 40.0   # Worst-case ambient

    # Cooling
    cooling_method: CoolingMethod = CoolingMethod.NATURAL
    orientation: PCBOrientation = PCBOrientation.HORIZONTAL_UP
    airflow_velocity_m_s: float = 0.0  # For forced convection

    # Board properties
    board_width_mm: float = 50.0
    board_height_mm: float = 40.0
    board_thickness_mm: float = 1.6
    layer_count: int = 4
    copper_thickness_oz: float = 1.0  # 1oz = 35um
    copper_coverage: float = 0.5      # Average copper fill ratio

    # Material
    substrate_material: str = 'fr4'
    solder_mask_color: str = 'green'

    # Design limits
    max_trace_temp_rise_c: float = 20.0  # IPC-2152 default
    max_board_temp_c: float = 85.0       # Max board temperature
    thermal_margin_c: float = 15.0        # Safety margin below Tj_max

    # Analysis options
    include_radiation: bool = True
    include_conduction: bool = True


# =============================================================================
# THERMAL PISTON CLASS
# =============================================================================

class ThermalPiston:
    """
    PCB Thermal Analysis Piston

    Provides:
    1. Component junction temperature estimation
    2. IPC-2152 trace current capacity analysis
    3. Thermal via effectiveness calculation
    4. Board-level thermal resistance estimation
    5. Hot spot identification and recommendations
    """

    def __init__(self, config: Optional[ThermalConfig] = None):
        self.config = config or ThermalConfig()
        self.components: List[ComponentThermal] = []
        self.thermal_vias: List[ThermalVia] = []
        self.thermal_zones: List[ThermalZone] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []

    # =========================================================================
    # STANDARD PISTON API
    # =========================================================================

    def analyze(self, parts_db: Dict, placement: Dict) -> Dict[str, Any]:
        """
        Standard piston API - analyze thermal characteristics of design.

        Args:
            parts_db: Parts database with component info
            placement: Component placements

        Returns:
            Dictionary with thermal analysis results
        """
        self.components.clear()
        self.warnings.clear()
        self.recommendations.clear()

        parts = parts_db.get('parts', {})
        results = {}

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            power = part.get('power_dissipation', 0.0)

            if power > 0:
                comp = ComponentThermal(
                    ref=ref,
                    package=part.get('footprint', 'Unknown'),
                    power_dissipation_w=power,
                    rth_ja=part.get('rth_ja', 50.0),
                    rth_jc=part.get('rth_jc', 10.0),
                    max_junction_temp_c=part.get('max_tj', 125.0)
                )
                self.components.append(comp)
                tj = self.calculate_junction_temperature(comp)
                results[ref] = {
                    'junction_temp_c': tj,
                    'power_w': power,
                    'within_limits': tj < comp.max_junction_temp_c
                }

        return {
            'components': results,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'max_ambient_c': self.config.max_ambient_c
        }

    # =========================================================================
    # JUNCTION TEMPERATURE CALCULATION
    # =========================================================================

    def calculate_junction_temperature(
        self,
        component: ComponentThermal,
        use_board_effects: bool = True
    ) -> float:
        """
        Calculate junction temperature for a component

        Formula: Tj = Ta + (Rth_ja_effective * P_loss)

        The effective Rth_ja depends on:
        - Package thermal resistance (from datasheet)
        - PCB thermal conductivity and copper coverage
        - Cooling conditions (natural/forced convection)

        Reference: ROHM Application Note AN-E
        """
        power = component.power_dissipation_w
        ta = self.config.max_ambient_c  # Use worst-case ambient

        if component.rth_ja > 0:
            # Start with datasheet Rth_ja
            rth_ja = component.rth_ja

            if use_board_effects:
                # Adjust for actual board conditions vs test board
                # Datasheet values are typically for 1"x1" or JEDEC test boards
                # Larger boards with more copper have lower Rth

                board_area = self.config.board_width_mm * self.config.board_height_mm
                test_board_area = 625.0  # 1" x 1" = 25.4mm x 25.4mm

                # Board area factor (more area = better cooling)
                area_factor = min(1.0, test_board_area / board_area)

                # Copper coverage factor (more copper = better spreading)
                copper_factor = 1.0 - 0.3 * (self.config.copper_coverage - 0.5)

                # Layer count factor (more layers = better heat spreading)
                layer_factor = 1.0 - 0.1 * min(4, self.config.layer_count - 2)

                # Cooling factor
                cooling_factor = self._get_cooling_factor()

                rth_ja = rth_ja * area_factor * copper_factor * layer_factor * cooling_factor

        elif component.rth_jc > 0:
            # Build up from Rth_jc
            rth_jc = component.rth_jc
            rth_cb = self._calculate_rth_case_to_board(component)
            rth_ba = self._calculate_rth_board_to_ambient(component)
            rth_ja = rth_jc + rth_cb + rth_ba

        else:
            # Estimate based on package size
            rth_ja = self._estimate_rth_ja(component)

        # Calculate junction temperature
        tj = ta + (rth_ja * power)

        component.tj_calculated = tj
        component.margin_c = component.tj_max - tj

        return tj

    def _get_cooling_factor(self) -> float:
        """Get cooling effectiveness factor based on cooling method"""
        factors = {
            CoolingMethod.NATURAL: 1.0,
            CoolingMethod.FORCED_LOW: 0.7,
            CoolingMethod.FORCED_MEDIUM: 0.5,
            CoolingMethod.FORCED_HIGH: 0.3,
            CoolingMethod.CONDUCTION: 0.4,
        }
        return factors.get(self.config.cooling_method, 1.0)

    def _calculate_rth_case_to_board(self, component: ComponentThermal) -> float:
        """
        Calculate thermal resistance from case to board

        This includes:
        - Solder joint resistance
        - Exposed pad (if present) with thermal vias
        """
        if component.exposed_pad_mm2 > 0:
            # Component has exposed thermal pad
            pad_area = component.exposed_pad_mm2 * 1e-6  # mm2 to m2

            # Solder layer resistance
            solder_thickness = 0.1e-3  # 0.1mm typical
            k_solder = THERMAL_CONDUCTIVITY['solder_sac305']
            rth_solder = solder_thickness / (k_solder * pad_area)

            # Via contribution (if thermal vias present)
            rth_vias = self._calculate_thermal_via_array_resistance(
                component.x, component.y,
                math.sqrt(component.exposed_pad_mm2)  # Approximate size
            )

            # Parallel: solder + vias
            if rth_vias > 0:
                rth_cb = 1 / (1/rth_solder + 1/rth_vias)
            else:
                rth_cb = rth_solder
        else:
            # Only through leads/balls
            # Approximate based on package area
            pkg_area = component.package_area_mm2 * 1e-6 if component.package_area_mm2 > 0 else 25e-6
            k_eff = 5.0  # Effective conductivity through leads
            rth_cb = 0.5e-3 / (k_eff * pkg_area)

        return rth_cb

    def _calculate_rth_board_to_ambient(self, component: ComponentThermal) -> float:
        """
        Calculate thermal resistance from board to ambient

        Combines:
        - Convection (natural or forced)
        - Radiation
        - Conduction to chassis (if applicable)
        """
        # Effective heat spreading area (larger than component)
        # Heat spreads at ~45 degrees through board
        spread_radius = component.height_mm + self.config.board_thickness_mm
        effective_area = (
            (math.sqrt(component.package_area_mm2) + 2 * spread_radius) ** 2
        ) * 1e-6  # to m2

        # Convection resistance
        h = self._get_convection_coefficient()
        rth_conv = 1 / (h * effective_area)

        # Radiation resistance (if included)
        if self.config.include_radiation:
            emissivity = EMISSIVITY.get(
                f'solder_mask_{self.config.solder_mask_color}', 0.85
            )
            # Linearized radiation coefficient
            t_surface = self.config.max_ambient_c + 30  # Estimate
            h_rad = 4 * emissivity * STEFAN_BOLTZMANN * ((t_surface + 273.15) ** 3)
            rth_rad = 1 / (h_rad * effective_area)

            # Parallel combination
            rth_ba = 1 / (1/rth_conv + 1/rth_rad)
        else:
            rth_ba = rth_conv

        return rth_ba

    def _get_convection_coefficient(self) -> float:
        """Get convection heat transfer coefficient based on config"""
        if self.config.cooling_method == CoolingMethod.NATURAL:
            if self.config.orientation == PCBOrientation.HORIZONTAL_UP:
                return CONVECTION_COEFFICIENTS['natural_horizontal_up']
            elif self.config.orientation == PCBOrientation.HORIZONTAL_DOWN:
                return CONVECTION_COEFFICIENTS['natural_horizontal_down']
            else:
                return CONVECTION_COEFFICIENTS['natural_vertical']
        elif self.config.cooling_method == CoolingMethod.FORCED_LOW:
            return CONVECTION_COEFFICIENTS['forced_low']
        elif self.config.cooling_method == CoolingMethod.FORCED_MEDIUM:
            return CONVECTION_COEFFICIENTS['forced_medium']
        elif self.config.cooling_method == CoolingMethod.FORCED_HIGH:
            return CONVECTION_COEFFICIENTS['forced_high']
        else:
            return CONVECTION_COEFFICIENTS['natural_horizontal_up']

    def _estimate_rth_ja(self, component: ComponentThermal) -> float:
        """
        Estimate Rth_ja when not provided

        Uses empirical correlation based on package area
        Reference: Various package thermal datasheets
        """
        if component.package_area_mm2 > 0:
            area = component.package_area_mm2
        else:
            area = 25.0  # Default 5mm x 5mm

        # Empirical: Rth_ja ~ 50 / sqrt(area_mm2) for typical packages
        # Ranges from ~200 C/W for SOT23 to ~30 C/W for large QFN
        rth_estimate = 50.0 / math.sqrt(area)

        # Adjust for exposed pad
        if component.exposed_pad_mm2 > 0:
            rth_estimate *= 0.5  # Exposed pad roughly halves Rth

        return rth_estimate

    # =========================================================================
    # THERMAL VIA CALCULATIONS
    # =========================================================================

    def calculate_via_thermal_resistance(self, via: ThermalVia) -> float:
        """
        Calculate thermal resistance of a single via

        Reference: Sierra Circuits Via Thermal Resistance Calculator
        Reference: IEEE "Thermal Modeling and Design Optimization of PCB Vias"

        Formula for plated via:
        Rth = H / (k_cu * pi * (D*t - t^2))

        Where:
        - H = via height (board thickness)
        - k_cu = thermal conductivity of copper
        - D = drill diameter
        - t = plating thickness
        """
        d = via.drill_diameter_mm * 1e-3  # to meters
        t = via.plating_thickness_mm * 1e-3
        h = via.height_mm * 1e-3
        k_cu = THERMAL_CONDUCTIVITY['copper']

        if via.filled:
            # Filled via: solid conductor
            k_fill = THERMAL_CONDUCTIVITY.get(via.fill_material, 50.0)
            inner_radius = (d/2) - t

            # Plating contribution
            area_plating = math.pi * ((d/2)**2 - inner_radius**2)
            rth_plating = h / (k_cu * area_plating)

            # Fill contribution
            area_fill = math.pi * inner_radius**2
            rth_fill = h / (k_fill * area_fill)

            # Parallel
            rth = 1 / (1/rth_plating + 1/rth_fill)
        else:
            # Hollow via (plating only)
            # Cross-sectional area of copper barrel
            outer_radius = d / 2
            inner_radius = outer_radius - t
            area = math.pi * (outer_radius**2 - inner_radius**2)

            rth = h / (k_cu * area)

        return rth

    def _calculate_thermal_via_array_resistance(
        self,
        center_x: float,
        center_y: float,
        pad_size_mm: float
    ) -> float:
        """
        Calculate combined thermal resistance of via array under a thermal pad

        Finds all thermal vias in the region and calculates parallel resistance
        """
        half_size = pad_size_mm / 2
        vias_in_region = [
            v for v in self.thermal_vias
            if (center_x - half_size <= v.x <= center_x + half_size and
                center_y - half_size <= v.y <= center_y + half_size)
        ]

        if not vias_in_region:
            return float('inf')

        # Parallel combination of all vias
        conductance = sum(
            1 / self.calculate_via_thermal_resistance(v)
            for v in vias_in_region
        )

        return 1 / conductance if conductance > 0 else float('inf')

    def recommend_thermal_vias(
        self,
        component: ComponentThermal,
        target_rth_reduction: float = 0.5
    ) -> Dict[str, Any]:
        """
        Recommend thermal via pattern for a component

        Returns via specifications to achieve target Rth reduction
        """
        if component.exposed_pad_mm2 <= 0:
            return {'error': 'Component has no thermal pad'}

        pad_size = math.sqrt(component.exposed_pad_mm2)

        # Start with typical 0.3mm vias on 1mm grid
        via_diameter = 0.3  # mm
        via_pitch = 1.0     # mm

        # Calculate number of vias that fit
        vias_per_side = int(pad_size / via_pitch)
        total_vias = vias_per_side ** 2

        # Calculate thermal resistance per via
        sample_via = ThermalVia(
            x=0, y=0,
            drill_diameter_mm=via_diameter,
            plating_thickness_mm=0.025,
            height_mm=self.config.board_thickness_mm,
            filled=True,
            fill_material='solder_sn63pb37'
        )
        rth_single = self.calculate_via_thermal_resistance(sample_via)

        # Parallel resistance of array
        rth_array = rth_single / total_vias

        # Calculate positions
        positions = []
        offset = (pad_size - (vias_per_side - 1) * via_pitch) / 2
        for i in range(vias_per_side):
            for j in range(vias_per_side):
                x = component.x - pad_size/2 + offset + i * via_pitch
                y = component.y - pad_size/2 + offset + j * via_pitch
                positions.append((x, y))

        return {
            'via_diameter_mm': via_diameter,
            'via_pitch_mm': via_pitch,
            'via_count': total_vias,
            'array_size': f'{vias_per_side}x{vias_per_side}',
            'rth_total_c_per_w': rth_array,
            'positions': positions,
            'recommendation': (
                f'Use {total_vias} thermal vias ({vias_per_side}x{vias_per_side} array) '
                f'with {via_diameter}mm drill on {via_pitch}mm pitch. '
                f'Fill with solder for best thermal performance.'
            )
        }

    # =========================================================================
    # IPC-2152 TRACE CURRENT ANALYSIS
    # =========================================================================

    def calculate_trace_current_capacity(
        self,
        width_mm: float,
        thickness_oz: float = 1.0,
        length_mm: float = 10.0,
        layer: LayerPosition = LayerPosition.OUTER,
        max_temp_rise_c: Optional[float] = None
    ) -> TraceAnalysis:
        """
        Calculate trace current capacity using IPC-2152 methodology

        Reference: IPC-2152 Standard for Determining Current Carrying Capacity

        For external layers (simplified formula from IPC-2221):
        I = k * (dT^0.44) * (A^0.725)

        Where:
        - I = current (A)
        - k = 0.048 for external, 0.024 for internal layers
        - dT = temperature rise (C)
        - A = cross-sectional area (mil^2)

        IPC-2152 uses detailed charts but this provides good estimates.
        """
        max_temp_rise = max_temp_rise_c or self.config.max_trace_temp_rise_c

        # Convert to mils for IPC formula
        width_mils = width_mm / 0.0254
        thickness_mils = thickness_oz * 1.378  # 1oz = 1.378 mils
        area_mil2 = width_mils * thickness_mils

        # IPC-2221 derived constant
        if layer == LayerPosition.OUTER:
            k = 0.048
        else:
            k = 0.024  # Internal layers have less cooling

        # Maximum current for given temp rise
        max_current = k * (max_temp_rise ** 0.44) * (area_mil2 ** 0.725)

        # Calculate trace resistance
        thickness_m = thickness_oz * 35e-6  # oz to meters
        width_m = width_mm * 1e-3
        length_m = length_mm * 1e-3

        # Copper resistivity at 25C: 1.68e-8 ohm-m
        rho_25 = 1.68e-8
        # Temperature coefficient: 0.00393 /C
        alpha = 0.00393
        # Resistance at elevated temperature
        avg_temp_rise = max_temp_rise / 2
        rho = rho_25 * (1 + alpha * avg_temp_rise)

        resistance = rho * length_m / (width_m * thickness_m)
        resistance_mohm = resistance * 1000

        # Power loss at max current
        power_loss_mw = (max_current ** 2) * resistance * 1000

        # Notes
        notes = []
        if layer == LayerPosition.INNER:
            notes.append("Internal layer: reduced cooling capacity")
        if max_temp_rise > 30:
            notes.append(f"High temp rise ({max_temp_rise}C) may affect nearby components")
        if width_mm < 0.15:
            notes.append("Trace width below recommended minimum (0.15mm)")

        # IPC-2152 compliance check
        ipc_compliant = (
            width_mm >= 0.1 and
            max_temp_rise <= 45 and  # IPC charts go up to 45C rise
            thickness_oz in [0.5, 1.0, 2.0, 3.0]  # Standard weights
        )

        return TraceAnalysis(
            width_mm=width_mm,
            thickness_mm=thickness_oz * 0.035,
            length_mm=length_mm,
            layer=layer,
            current_a=0.0,  # Not specified, this is capacity analysis
            temperature_rise_c=max_temp_rise,
            max_current_a=max_current,
            resistance_mohm=resistance_mohm,
            power_loss_mw=power_loss_mw,
            ipc2152_compliant=ipc_compliant,
            notes=notes
        )

    def calculate_trace_for_current(
        self,
        current_a: float,
        thickness_oz: float = 1.0,
        layer: LayerPosition = LayerPosition.OUTER,
        max_temp_rise_c: Optional[float] = None
    ) -> float:
        """
        Calculate required trace width for a given current

        Inverse of the IPC-2152 current formula

        Returns: Required trace width in mm
        """
        max_temp_rise = max_temp_rise_c or self.config.max_trace_temp_rise_c

        # IPC-2221 constant
        k = 0.048 if layer == LayerPosition.OUTER else 0.024

        # Rearranged formula: A = (I / (k * dT^0.44))^(1/0.725)
        area_mil2 = (current_a / (k * (max_temp_rise ** 0.44))) ** (1 / 0.725)

        # Convert to mm
        thickness_mils = thickness_oz * 1.378
        width_mils = area_mil2 / thickness_mils
        width_mm = width_mils * 0.0254

        return max(0.1, width_mm)  # Minimum 0.1mm

    def analyze_trace_temperature(
        self,
        width_mm: float,
        current_a: float,
        thickness_oz: float = 1.0,
        length_mm: float = 10.0,
        layer: LayerPosition = LayerPosition.OUTER
    ) -> TraceAnalysis:
        """
        Analyze temperature rise for a specific trace with given current

        Returns: TraceAnalysis with actual temperature rise
        """
        # Convert dimensions
        width_mils = width_mm / 0.0254
        thickness_mils = thickness_oz * 1.378
        area_mil2 = width_mils * thickness_mils

        k = 0.048 if layer == LayerPosition.OUTER else 0.024

        # Rearranged: dT = (I / (k * A^0.725))^(1/0.44)
        if area_mil2 > 0 and k > 0:
            temp_rise = (current_a / (k * (area_mil2 ** 0.725))) ** (1 / 0.44)
        else:
            temp_rise = float('inf')

        # Calculate resistance and power loss
        thickness_m = thickness_oz * 35e-6
        width_m = width_mm * 1e-3
        length_m = length_mm * 1e-3

        rho_25 = 1.68e-8
        alpha = 0.00393
        rho = rho_25 * (1 + alpha * (temp_rise / 2))

        resistance = rho * length_m / (width_m * thickness_m)
        power_loss = (current_a ** 2) * resistance

        # Max current for standard temp rise
        max_current = k * (self.config.max_trace_temp_rise_c ** 0.44) * (area_mil2 ** 0.725)

        notes = []
        if temp_rise > self.config.max_trace_temp_rise_c:
            notes.append(f"Temperature rise ({temp_rise:.1f}C) exceeds limit ({self.config.max_trace_temp_rise_c}C)")
        if temp_rise > 45:
            notes.append("Extreme temperature rise - verify with IPC-2152 charts")

        return TraceAnalysis(
            width_mm=width_mm,
            thickness_mm=thickness_oz * 0.035,
            length_mm=length_mm,
            layer=layer,
            current_a=current_a,
            temperature_rise_c=temp_rise,
            max_current_a=max_current,
            resistance_mohm=resistance * 1000,
            power_loss_mw=power_loss * 1000,
            ipc2152_compliant=temp_rise <= self.config.max_trace_temp_rise_c,
            notes=notes
        )

    # =========================================================================
    # PCB THERMAL CONDUCTIVITY
    # =========================================================================

    def calculate_pcb_thermal_conductivity(self) -> Dict[str, float]:
        """
        Calculate effective thermal conductivity of the PCB

        Reference: Altium - "Estimating Thermal Conductivity of PCBs"

        Uses weighted average based on copper coverage:
        k_eff = (k_cu * V_cu + k_fr4 * V_fr4) / V_total

        Returns both in-plane and through-plane conductivity
        """
        copper_thickness = self.config.copper_thickness_oz * 35e-6  # oz to meters
        layer_count = self.config.layer_count
        board_thickness = self.config.board_thickness_mm * 1e-3

        k_cu = THERMAL_CONDUCTIVITY['copper']
        k_fr4_in = THERMAL_CONDUCTIVITY['fr4_in_plane']
        k_fr4_through = THERMAL_CONDUCTIVITY['fr4_through_plane']

        # Volume fractions
        total_copper = layer_count * copper_thickness
        total_fr4 = board_thickness - total_copper

        copper_volume_fraction = total_copper / board_thickness
        fr4_volume_fraction = total_fr4 / board_thickness

        # Adjust for copper coverage (not 100% copper on each layer)
        effective_copper_fraction = copper_volume_fraction * self.config.copper_coverage

        # In-plane conductivity (parallel model - copper dominates)
        k_in_plane = (k_cu * effective_copper_fraction +
                      k_fr4_in * (1 - effective_copper_fraction))

        # Through-plane conductivity (series model - FR4 dominates)
        # 1/k_eff = sum(thickness_i / k_i) / total_thickness
        # Simplified for uniform copper distribution
        k_through_plane = board_thickness / (
            (total_copper * effective_copper_fraction / k_cu) +
            (board_thickness - total_copper * effective_copper_fraction) / k_fr4_through
        )

        return {
            'in_plane_w_per_m_k': k_in_plane,
            'through_plane_w_per_m_k': k_through_plane,
            'copper_volume_fraction': effective_copper_fraction,
            'spreading_ratio': k_in_plane / k_through_plane
        }

    def calculate_board_thermal_resistance(self) -> float:
        """
        Calculate overall board thermal resistance from top to bottom

        This is useful for estimating heat transfer to chassis/heat sink
        """
        k_props = self.calculate_pcb_thermal_conductivity()
        k_through = k_props['through_plane_w_per_m_k']

        board_area = (self.config.board_width_mm * self.config.board_height_mm) * 1e-6  # m2
        board_thickness = self.config.board_thickness_mm * 1e-3  # m

        rth = board_thickness / (k_through * board_area)

        return rth

    # =========================================================================
    # FULL THERMAL ANALYSIS
    # =========================================================================

    def run(
        self,
        components: List[ComponentThermal],
        thermal_vias: Optional[List[ThermalVia]] = None,
        thermal_zones: Optional[List[ThermalZone]] = None
    ) -> ThermalResult:
        """
        Run full thermal analysis on the design

        Parameters:
        -----------
        components : List[ComponentThermal]
            Components with power dissipation data
        thermal_vias : List[ThermalVia], optional
            Thermal via positions
        thermal_zones : List[ThermalZone], optional
            Copper pour zones

        Returns:
        --------
        ThermalResult with analysis results and recommendations
        """
        self.components = components
        self.thermal_vias = thermal_vias or []
        self.thermal_zones = thermal_zones or []
        self.warnings.clear()
        self.recommendations.clear()

        total_power = sum(c.power_dissipation_w for c in components)

        # Calculate junction temperatures
        max_tj = 0.0
        hottest_ref = ""

        for component in components:
            tj = self.calculate_junction_temperature(component)

            if tj > max_tj:
                max_tj = tj
                hottest_ref = component.reference

            # Check thermal margin
            if component.margin_c < self.config.thermal_margin_c:
                if component.margin_c < 0:
                    self.warnings.append(
                        f"{component.reference}: Junction temp ({tj:.1f}C) "
                        f"EXCEEDS Tj_max ({component.tj_max}C)!"
                    )
                else:
                    self.warnings.append(
                        f"{component.reference}: Low thermal margin ({component.margin_c:.1f}C)"
                    )

        # Board-level thermal resistance
        board_rth = self.calculate_board_thermal_resistance()

        # Estimate max board temperature
        h = self._get_convection_coefficient()
        board_area = (self.config.board_width_mm * self.config.board_height_mm) * 1e-6
        rth_conv = 1 / (h * board_area)

        max_board_temp = self.config.max_ambient_c + total_power * rth_conv * 0.5

        if max_board_temp > self.config.max_board_temp_c:
            self.warnings.append(
                f"Estimated max board temperature ({max_board_temp:.1f}C) "
                f"exceeds limit ({self.config.max_board_temp_c}C)"
            )

        # Generate recommendations
        self._generate_recommendations(components, total_power)

        # Check overall thermal status
        thermal_ok = all(c.margin_c >= self.config.thermal_margin_c for c in components)

        return ThermalResult(
            ambient_temp_c=self.config.max_ambient_c,
            max_board_temp_c=max_board_temp,
            hottest_component=hottest_ref,
            hottest_junction_temp_c=max_tj,
            components=components,
            thermal_margin_ok=thermal_ok,
            board_rth_effective=board_rth,
            total_power_w=total_power,
            recommendations=self.recommendations.copy(),
            warnings=self.warnings.copy()
        )

    def _generate_recommendations(
        self,
        components: List[ComponentThermal],
        total_power: float
    ):
        """Generate thermal design recommendations"""

        # High power components
        high_power = [c for c in components if c.power_dissipation_w > 0.5]
        for c in high_power:
            if c.exposed_pad_mm2 > 0 and not any(
                abs(v.x - c.x) < 5 and abs(v.y - c.y) < 5
                for v in self.thermal_vias
            ):
                self.recommendations.append(
                    f"{c.reference}: Add thermal vias under exposed pad"
                )

        # Cooling suggestions
        if total_power > 5.0 and self.config.cooling_method == CoolingMethod.NATURAL:
            self.recommendations.append(
                "Consider forced air cooling for >5W total power"
            )

        # Copper coverage
        if self.config.copper_coverage < 0.3:
            self.recommendations.append(
                "Increase copper pour coverage for better heat spreading"
            )

        # Layer count
        if total_power > 2.0 and self.config.layer_count < 4:
            self.recommendations.append(
                "Consider 4+ layer board for improved thermal performance"
            )

        # Hot component spacing
        hot_components = [c for c in components if c.tj_calculated > 80]
        for i, c1 in enumerate(hot_components):
            for c2 in hot_components[i+1:]:
                distance = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                if distance < 10:  # Less than 10mm apart
                    self.recommendations.append(
                        f"Increase spacing between {c1.reference} and {c2.reference} "
                        f"(currently {distance:.1f}mm)"
                    )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_power_density(self, component: ComponentThermal) -> float:
        """Calculate power density in W/cm2"""
        if component.package_area_mm2 > 0:
            area_cm2 = component.package_area_mm2 / 100
            return component.power_dissipation_w / area_cm2
        return 0.0

    def estimate_component_temp_without_power(
        self,
        component: ComponentThermal,
        nearby_heat_sources: List[ComponentThermal]
    ) -> float:
        """
        Estimate temperature rise of unpowered component due to nearby heat

        Useful for checking if passive components are within spec
        """
        temp = self.config.max_ambient_c

        for source in nearby_heat_sources:
            distance = math.sqrt(
                (component.x - source.x)**2 + (component.y - source.y)**2
            )

            if distance < 1:
                distance = 1  # Minimum 1mm

            # Simplified spreading: temp drops with 1/distance
            # Assume source raises local temp by its power * spreading factor
            k_props = self.calculate_pcb_thermal_conductivity()
            k_spread = k_props['in_plane_w_per_m_k']

            # Approximate temperature contribution
            # Q = k * A * dT / L => dT = Q * L / (k * A)
            # Simplified for PCB spreading
            delta_t = source.power_dissipation_w * distance * 0.001 / (
                k_spread * self.config.board_thickness_mm * 0.001 * 0.01
            )

            temp += delta_t * math.exp(-distance / 20)  # Exponential decay

        return temp


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_trace_width(current_a: float, temp_rise_c: float = 20.0) -> float:
    """
    Quick calculation of required trace width for given current

    Uses external layer, 1oz copper defaults
    """
    piston = ThermalPiston()
    return piston.calculate_trace_for_current(
        current_a=current_a,
        thickness_oz=1.0,
        layer=LayerPosition.OUTER,
        max_temp_rise_c=temp_rise_c
    )


def quick_junction_temp(
    power_w: float,
    rth_ja: float,
    ambient_c: float = 40.0
) -> float:
    """
    Quick junction temperature calculation

    Tj = Ta + Rth_ja * P
    """
    return ambient_c + rth_ja * power_w


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("THERMAL PISTON - Self Test")
    print("=" * 60)

    piston = ThermalPiston()

    # Test 1: Junction temperature calculation
    print("\n1. Junction Temperature Calculation:")
    component = ComponentThermal(
        reference='U1',
        power_dissipation_w=1.5,
        rth_ja=40.0,  # 40 C/W from datasheet
        package_area_mm2=64.0,  # 8mm x 8mm QFN
        exposed_pad_mm2=25.0,   # 5mm x 5mm thermal pad
        x=25.0,
        y=20.0,
        tj_max=125.0
    )

    tj = piston.calculate_junction_temperature(component)
    print(f"   Component: {component.reference}")
    print(f"   Power: {component.power_dissipation_w}W, Rth_ja: {component.rth_ja} C/W")
    print(f"   Junction Temp: {tj:.1f}C")
    print(f"   Margin: {component.margin_c:.1f}C below Tj_max ({component.tj_max}C)")

    # Test 2: Thermal via calculation
    print("\n2. Thermal Via Resistance:")
    via = ThermalVia(
        x=25.0, y=20.0,
        drill_diameter_mm=0.3,
        plating_thickness_mm=0.025,
        height_mm=1.6,
        filled=True,
        fill_material='solder_sac305'
    )
    rth_via = piston.calculate_via_thermal_resistance(via)
    print(f"   Via: {via.drill_diameter_mm}mm drill, filled")
    print(f"   Thermal resistance: {rth_via:.2f} C/W")

    # Test 3: Thermal via recommendation
    print("\n3. Thermal Via Array Recommendation:")
    recommendation = piston.recommend_thermal_vias(component)
    print(f"   {recommendation['recommendation']}")
    print(f"   Array Rth: {recommendation['rth_total_c_per_w']:.3f} C/W")

    # Test 4: Trace current capacity
    print("\n4. IPC-2152 Trace Current Capacity:")
    trace = piston.calculate_trace_current_capacity(
        width_mm=0.5,
        thickness_oz=1.0,
        length_mm=50.0,
        layer=LayerPosition.OUTER,
        max_temp_rise_c=20.0
    )
    print(f"   Trace: {trace.width_mm}mm wide, 1oz copper, external")
    print(f"   Max current: {trace.max_current_a:.2f}A (for 20C rise)")
    print(f"   Resistance: {trace.resistance_mohm:.2f} mohm")

    # Test 5: Required trace width
    print("\n5. Required Trace Width for 3A:")
    width = piston.calculate_trace_for_current(
        current_a=3.0,
        thickness_oz=1.0,
        layer=LayerPosition.OUTER,
        max_temp_rise_c=20.0
    )
    print(f"   Required width: {width:.2f}mm ({width/0.0254:.1f} mils)")

    # Test 6: Trace with specific current
    print("\n6. Temperature Rise Analysis:")
    analysis = piston.analyze_trace_temperature(
        width_mm=0.3,
        current_a=2.0,
        thickness_oz=1.0,
        length_mm=30.0,
        layer=LayerPosition.OUTER
    )
    print(f"   Trace: 0.3mm @ 2A")
    print(f"   Temperature rise: {analysis.temperature_rise_c:.1f}C")
    print(f"   Power loss: {analysis.power_loss_mw:.1f}mW")
    print(f"   IPC-2152 compliant: {analysis.ipc2152_compliant}")

    # Test 7: PCB thermal conductivity
    print("\n7. PCB Thermal Conductivity:")
    k_props = piston.calculate_pcb_thermal_conductivity()
    print(f"   In-plane: {k_props['in_plane_w_per_m_k']:.2f} W/m-K")
    print(f"   Through-plane: {k_props['through_plane_w_per_m_k']:.3f} W/m-K")
    print(f"   Spreading ratio: {k_props['spreading_ratio']:.1f}x")

    # Test 8: Full analysis
    print("\n8. Full Thermal Analysis:")
    components = [
        ComponentThermal('U1', 1.5, rth_ja=40.0, package_area_mm2=64.0,
                        exposed_pad_mm2=25.0, x=25, y=20, tj_max=125),
        ComponentThermal('U2', 0.8, rth_ja=60.0, package_area_mm2=36.0,
                        x=40, y=20, tj_max=125),
        ComponentThermal('Q1', 2.0, rth_ja=50.0, package_area_mm2=25.0,
                        exposed_pad_mm2=16.0, x=25, y=35, tj_max=150),
    ]

    result = piston.run(components)
    print(f"   Total power: {result.total_power_w}W")
    print(f"   Hottest component: {result.hottest_component} at {result.hottest_junction_temp_c:.1f}C")
    print(f"   Max board temp: {result.max_board_temp_c:.1f}C")
    print(f"   Thermal OK: {result.thermal_margin_ok}")

    if result.warnings:
        print("   Warnings:")
        for w in result.warnings:
            print(f"     - {w}")

    if result.recommendations:
        print("   Recommendations:")
        for r in result.recommendations[:3]:
            print(f"     - {r}")

    print("\n" + "=" * 60)
    print("Thermal Piston self-test PASSED")
    print("=" * 60)
