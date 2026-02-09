"""
Current Flow Analyzer
======================

Analyzes how current flows through a circuit.
This is a key piece of expert knowledge that machines typically lack.

Current flow understanding is critical for:
1. Minimizing EMI (loop areas)
2. Proper grounding (star vs mesh)
3. Power integrity (voltage drops)
4. Thermal management (high current paths)

Key Insight:
    Current ALWAYS returns to its source.
    - DC: Through lowest RESISTANCE path
    - AC/High-freq: Through lowest INDUCTANCE path (directly under the trace)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import math

from .circuit_types import NetFunction, ComponentFunction


@dataclass
class CurrentPath:
    """A path that current flows through."""
    name: str
    source: str  # Component ref (power source)
    load: str    # Component ref (power consumer)
    components: List[str]  # Components in path order
    nets: List[str]  # Nets in path order
    estimated_current: float = 0.0  # Amps
    path_resistance: float = 0.0  # Ohms
    voltage_drop: float = 0.0  # Volts
    is_return_path: bool = False


@dataclass
class GroundNetwork:
    """Analysis of the ground network."""
    topology: str  # 'STAR', 'MESH', 'MIXED', 'BUS'
    star_point: Optional[str] = None  # Component ref if star topology
    ground_pins: List[Tuple[str, str]] = field(default_factory=list)  # (ref, pin)
    issues: List[str] = field(default_factory=list)


@dataclass
class LoopAnalysis:
    """Analysis of a current loop."""
    name: str
    forward_path: CurrentPath
    return_path: CurrentPath
    estimated_area: float = 0.0  # mm² (after placement)
    frequency: float = 0.0  # Hz (if switching)
    emi_risk: str = 'LOW'  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


class CurrentFlowAnalyzer:
    """
    Analyzes current flow in a circuit.

    This module answers questions like:
    - Where does current flow from/to?
    - What is the return path?
    - What is the loop area?
    - Where are the high-current paths?
    """

    def __init__(self):
        pass

    def analyze(self, parts_db: Dict,
                placement: Optional[Dict] = None) -> Dict[str, any]:
        """
        Complete current flow analysis.

        Returns dict with:
        - power_paths: List of power current paths
        - signal_paths: List of signal current paths
        - ground_network: Ground network analysis
        - loops: List of critical loops
        """
        # Build connectivity graph
        connectivity = self._build_connectivity(parts_db)

        # Identify power sources and loads
        sources, loads = self._identify_power_components(parts_db)

        # Trace power paths
        power_paths = self._trace_power_paths(parts_db, connectivity, sources, loads)

        # Analyze ground network
        ground_network = self._analyze_ground_network(parts_db, connectivity)

        # Identify signal paths
        signal_paths = self._trace_signal_paths(parts_db, connectivity)

        # Identify current loops
        loops = self._identify_loops(power_paths, ground_network, parts_db, placement)

        return {
            'power_paths': power_paths,
            'signal_paths': signal_paths,
            'ground_network': ground_network,
            'loops': loops,
        }

    def _build_connectivity(self, parts_db: Dict) -> Dict[str, Set[str]]:
        """Build component connectivity graph."""
        # Component -> set of connected components
        connectivity = defaultdict(set)

        for net_name, net_info in parts_db.get('nets', {}).items():
            pins = net_info.get('pins', [])
            components = list(set(ref for ref, pin in pins))

            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    connectivity[comp1].add(comp2)
                    connectivity[comp2].add(comp1)

        return connectivity

    def _identify_power_components(self, parts_db: Dict) -> Tuple[List[str], List[str]]:
        """Identify power sources and loads."""
        sources = []  # Components that provide power
        loads = []    # Components that consume power

        for ref, part in parts_db.get('parts', {}).items():
            value = part.get('value', '').upper()

            # Power input connectors are sources
            if ref.startswith('J'):
                # Check if connected to power net
                for pin in part.get('pins', []):
                    net = pin.get('net', '').upper()
                    if any(p in net for p in ['VIN', 'VCC', 'VDD', '+5', '+12', 'PWR']):
                        sources.append(ref)
                        break

            # Regulators are both loads (input) and sources (output)
            if any(reg in value for reg in ['LM2596', 'AMS1117', 'LM1117', '7805', 'LDO', 'BUCK']):
                loads.append(ref)  # Load on input side
                sources.append(ref)  # Source on output side

            # MCUs, ICs are loads
            if ref.startswith('U') and ref not in sources:
                loads.append(ref)

            # LEDs are loads
            if 'LED' in value:
                loads.append(ref)

        return (list(set(sources)), list(set(loads)))

    def _trace_power_paths(self, parts_db: Dict, connectivity: Dict,
                           sources: List[str], loads: List[str]) -> List[CurrentPath]:
        """Trace power current paths from sources to loads."""
        paths = []

        for source in sources:
            for load in loads:
                if source == load:
                    continue

                # Find path from source to load
                path = self._find_path(source, load, connectivity, parts_db)

                if path:
                    # Find which nets connect the components
                    path_nets = self._get_path_nets(path, parts_db)

                    paths.append(CurrentPath(
                        name=f"{source}_to_{load}",
                        source=source,
                        load=load,
                        components=path,
                        nets=path_nets,
                    ))

        return paths

    def _find_path(self, start: str, end: str,
                   connectivity: Dict, parts_db: Dict) -> Optional[List[str]]:
        """Find path between two components using BFS."""
        from collections import deque

        if start not in connectivity or end not in connectivity:
            return None

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if current == end:
                return path

            for neighbor in connectivity[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _get_path_nets(self, path: List[str], parts_db: Dict) -> List[str]:
        """Get nets that connect components in a path."""
        nets = []

        for i in range(len(path) - 1):
            comp1, comp2 = path[i], path[i + 1]

            # Find net that connects these two components
            for net_name, net_info in parts_db.get('nets', {}).items():
                pins = net_info.get('pins', [])
                refs = [ref for ref, pin in pins]

                if comp1 in refs and comp2 in refs:
                    nets.append(net_name)
                    break

        return nets

    def _analyze_ground_network(self, parts_db: Dict,
                                 connectivity: Dict) -> GroundNetwork:
        """Analyze the ground network topology."""
        # Find all GND pins
        gnd_pins = []
        gnd_components = set()

        for net_name, net_info in parts_db.get('nets', {}).items():
            if 'GND' in net_name.upper() or 'VSS' in net_name.upper():
                for ref, pin in net_info.get('pins', []):
                    gnd_pins.append((ref, pin))
                    gnd_components.add(ref)

        # Determine topology
        # Star: All grounds connect to a single point
        # Mesh: Grounds form a web
        # Bus: Grounds connect in a line

        issues = []

        # Simple heuristic for topology
        if len(gnd_components) <= 3:
            topology = 'STAR'
        elif len(gnd_components) > 10:
            topology = 'MESH'
        else:
            topology = 'BUS'

        # Check for potential issues
        # Issue: Power and signal grounds mixed
        power_grounds = []
        signal_grounds = []

        for ref, pin in gnd_pins:
            if ref.startswith(('J', 'U')) and 'REG' not in ref.upper():
                signal_grounds.append(ref)
            else:
                power_grounds.append(ref)

        if power_grounds and signal_grounds:
            # Check if they share a star point (good) or are daisy-chained (bad)
            pass  # Would need placement to fully analyze

        return GroundNetwork(
            topology=topology,
            ground_pins=gnd_pins,
            issues=issues
        )

    def _trace_signal_paths(self, parts_db: Dict,
                            connectivity: Dict) -> List[CurrentPath]:
        """Trace signal current paths."""
        paths = []

        for net_name, net_info in parts_db.get('nets', {}).items():
            name_upper = net_name.upper()

            # Skip power and ground
            if any(p in name_upper for p in ['VCC', 'VDD', 'GND', 'VSS', 'VIN', '+5', '+3']):
                continue

            pins = net_info.get('pins', [])
            if len(pins) < 2:
                continue

            # For signals, trace from driver to receiver
            # Simplified: first component is driver, rest are receivers
            refs = [ref for ref, pin in pins]

            path = CurrentPath(
                name=f"signal_{net_name}",
                source=refs[0],
                load=refs[-1] if len(refs) > 1 else refs[0],
                components=refs,
                nets=[net_name],
            )
            paths.append(path)

        return paths

    def _identify_loops(self, power_paths: List[CurrentPath],
                        ground_network: GroundNetwork,
                        parts_db: Dict,
                        placement: Optional[Dict]) -> List[LoopAnalysis]:
        """Identify critical current loops."""
        loops = []

        for path in power_paths:
            # Every power path has a return path through ground
            return_path = CurrentPath(
                name=f"return_{path.name}",
                source=path.load,
                load=path.source,
                components=list(reversed(path.components)),
                nets=['GND'],  # Simplified
                is_return_path=True
            )

            # Calculate loop area if placement available
            area = 0.0
            if placement:
                area = self._estimate_loop_area(path, return_path, placement)

            # Determine EMI risk
            if area > 100:
                emi_risk = 'CRITICAL'
            elif area > 50:
                emi_risk = 'HIGH'
            elif area > 20:
                emi_risk = 'MEDIUM'
            else:
                emi_risk = 'LOW'

            loops.append(LoopAnalysis(
                name=f"loop_{path.name}",
                forward_path=path,
                return_path=return_path,
                estimated_area=area,
                emi_risk=emi_risk
            ))

        return loops

    def _estimate_loop_area(self, forward: CurrentPath, return_path: CurrentPath,
                            placement: Dict) -> float:
        """
        Estimate loop area from placement.

        Simplified calculation: Uses convex hull of all component positions.
        Real implementation would trace actual paths.
        """
        positions = []

        for comp in forward.components + return_path.components:
            if comp in placement:
                pos = placement[comp]
                if isinstance(pos, (tuple, list)):
                    positions.append((pos[0], pos[1]))
                elif hasattr(pos, 'x') and hasattr(pos, 'y'):
                    positions.append((pos.x, pos.y))

        if len(positions) < 3:
            return 0.0

        # Simple bounding box area (conservative estimate)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # Loop area is roughly half the bounding box (assuming straight paths)
        return width * height * 0.5


class PowerIntegrityAnalyzer:
    """
    Analyzes power integrity: voltage drops, decoupling, etc.
    """

    def analyze_voltage_drop(self, current_a: float, trace_length_mm: float,
                              trace_width_mm: float, copper_oz: float = 1.0) -> float:
        """
        Calculate voltage drop along a trace.

        V = I × R
        R = ρ × L / A
        """
        # Copper resistivity at 20°C: 1.68 × 10^-8 Ω·m
        resistivity = 1.68e-8

        # Convert to meters
        length_m = trace_length_mm / 1000
        width_m = trace_width_mm / 1000
        thickness_m = copper_oz * 35e-6  # 1 oz = 35 μm

        # Cross-sectional area
        area_m2 = width_m * thickness_m

        # Resistance
        resistance = resistivity * length_m / area_m2

        # Voltage drop
        voltage_drop = current_a * resistance

        return voltage_drop

    def recommend_trace_width(self, current_a: float, max_voltage_drop: float,
                               trace_length_mm: float, copper_oz: float = 1.0) -> float:
        """
        Recommend trace width for given current and max voltage drop.

        Returns width in mm.
        """
        resistivity = 1.68e-8
        length_m = trace_length_mm / 1000
        thickness_m = copper_oz * 35e-6

        # R = V / I
        max_resistance = max_voltage_drop / current_a if current_a > 0 else float('inf')

        # A = ρ × L / R
        min_area = resistivity * length_m / max_resistance if max_resistance > 0 else 0

        # Width = A / thickness
        min_width_m = min_area / thickness_m if thickness_m > 0 else 0
        min_width_mm = min_width_m * 1000

        return min_width_mm

    def analyze_decoupling(self, ic_current_ma: float, frequency_hz: float,
                            max_ripple_mv: float) -> Dict:
        """
        Analyze decoupling capacitor requirements.

        Returns dict with capacitor recommendations.
        """
        # C = I × dt / dV
        # For switching transients, dt ≈ 1/(2f)

        current_a = ic_current_ma / 1000
        ripple_v = max_ripple_mv / 1000
        dt = 1 / (2 * frequency_hz) if frequency_hz > 0 else 1e-6

        # Minimum capacitance
        min_cap_f = current_a * dt / ripple_v if ripple_v > 0 else 1e-6
        min_cap_uf = min_cap_f * 1e6

        # Capacitor self-resonant frequency
        # For effective decoupling, SRF > 2 × frequency
        # Typical 100nF 0805: SRF ≈ 30 MHz
        # Typical 10uF 0805: SRF ≈ 3 MHz

        recommendations = {
            'min_capacitance_uf': round(min_cap_uf, 3),
            'recommendations': []
        }

        if frequency_hz < 1e6:  # < 1 MHz
            recommendations['recommendations'].append('10uF ceramic for bulk decoupling')

        if frequency_hz < 100e6:  # < 100 MHz
            recommendations['recommendations'].append('100nF ceramic for high-frequency decoupling')

        if frequency_hz > 100e6:  # > 100 MHz
            recommendations['recommendations'].append('10nF or 1nF ceramic for very high frequency')
            recommendations['recommendations'].append('Consider multiple caps in parallel')

        recommendations['recommendations'].append('Place caps as close as possible to IC power pins')

        return recommendations
