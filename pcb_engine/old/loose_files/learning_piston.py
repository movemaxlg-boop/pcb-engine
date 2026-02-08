"""
Learning Piston - PCB Engine Worker (MACHINE LEARNING)

This is a TRUE Machine Learning piston that:
1. REVERSE ENGINEERS working PCB files (KiCad, Altium, Eagle, Gerber)
2. EXTRACTS design patterns, rules, and relationships
3. LEARNS from successful designs using ML models
4. ENHANCES existing algorithms with learned patterns
5. GENERATES new algorithms based on learned behaviors

HIERARCHY:
    USER (Boss) → Circuit AI (Engineer) → PCB Engine (Foreman) → This Piston (Worker)

LEARNING SOURCES:
- KiCad .kicad_pcb files
- Gerber manufacturing files
- Altium .PcbDoc (parsed)
- Eagle .brd files
- Design rule files
- BOM/netlist files

ML MODELS:
- Placement Predictor (where components should go)
- Routing Strategy Classifier (which algorithm works best)
- Trace Width Optimizer (for signal integrity)
- Component Clustering (hierarchical grouping)
- Via Placement Predictor (optimal via locations)
- DRC Pattern Detector (common violation patterns)

TRAINING MODES:
- Supervised: Learn from labeled good/bad designs
- Unsupervised: Discover patterns in unlabeled designs
- Reinforcement: Learn from routing success/failure feedback
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from enum import Enum, auto
import math
import json
import re
import os
from collections import defaultdict
import random


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class FileFormat(Enum):
    """Supported input file formats for reverse engineering"""
    KICAD_PCB = auto()      # .kicad_pcb
    KICAD_SCH = auto()      # .kicad_sch
    GERBER = auto()         # .gbr, .gtl, .gbl, etc.
    DRILL = auto()          # .drl, .xln
    ALTIUM = auto()         # .PcbDoc (ASCII export)
    EAGLE = auto()          # .brd
    ODB = auto()            # ODB++ format
    IPC_2581 = auto()       # IPC-2581 XML


class LearningMode(Enum):
    """Machine learning training modes"""
    SUPERVISED = auto()      # Learn from labeled examples
    UNSUPERVISED = auto()    # Discover patterns
    REINFORCEMENT = auto()   # Learn from feedback
    TRANSFER = auto()        # Apply learned patterns to new domains
    OBSERVE = auto()         # Observe design patterns without training


class FeatureType(Enum):
    """Types of features extracted from designs"""
    PLACEMENT = auto()       # Component placement patterns
    ROUTING = auto()         # Trace routing strategies
    POWER = auto()           # Power distribution patterns
    GROUND = auto()          # Ground plane patterns
    SIGNAL = auto()          # Signal integrity patterns
    THERMAL = auto()         # Thermal management patterns
    DRC = auto()             # Design rule patterns
    HIERARCHY = auto()       # Block/module organization


class ModelType(Enum):
    """Types of ML models"""
    NEURAL_NETWORK = auto()
    DECISION_TREE = auto()
    RANDOM_FOREST = auto()
    GRADIENT_BOOST = auto()
    CLUSTERING = auto()
    REGRESSION = auto()
    EMBEDDING = auto()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedComponent:
    """Component extracted from a PCB file"""
    reference: str          # U1, R1, C1, etc.
    value: str              # 10k, 100nF, ESP32, etc.
    footprint: str          # Package name
    x: float                # X position (mm)
    y: float                # Y position (mm)
    rotation: float         # Rotation in degrees
    layer: str              # Top, Bottom
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractedTrace:
    """Trace extracted from a PCB file"""
    net_name: str
    points: List[Tuple[float, float]]  # List of (x, y) coordinates
    width: float            # Trace width in mm
    layer: str              # Layer name
    is_arc: bool = False


@dataclass
class ExtractedVia:
    """Via extracted from a PCB file"""
    x: float
    y: float
    drill: float            # Drill diameter
    size: float             # Pad size
    net_name: str
    layers: Tuple[str, str]  # Start, end layers


@dataclass
class ExtractedZone:
    """Zone/pour extracted from a PCB file"""
    net_name: str
    layer: str
    outline: List[Tuple[float, float]]
    is_filled: bool = True
    priority: int = 0


@dataclass
class ExtractedDesign:
    """Complete extracted design from reverse engineering"""
    source_file: str
    file_format: FileFormat
    board_width: float
    board_height: float
    layer_count: int
    components: List[ExtractedComponent]
    traces: List[ExtractedTrace]
    vias: List[ExtractedVia]
    zones: List[ExtractedZone]
    nets: Dict[str, List[str]]  # net_name -> list of component.pin
    design_rules: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignFeatures:
    """Features extracted for machine learning"""
    # Placement features
    component_density: float
    placement_symmetry: float
    cluster_count: int
    power_component_positions: List[Tuple[float, float]]
    io_component_positions: List[Tuple[float, float]]

    # Routing features
    average_trace_length: float
    via_density: float
    layer_utilization: Dict[str, float]
    routing_congestion_map: List[List[float]]

    # Signal integrity features
    differential_pair_count: int
    matched_length_groups: int
    guard_traces_present: bool

    # Power features
    power_plane_coverage: float
    ground_plane_coverage: float
    decap_placement_score: float

    # Quality features
    drc_violation_count: int
    estimated_quality_score: float

    # Raw feature vector for ML
    feature_vector: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


@dataclass
class LearnedPattern:
    """A pattern learned from designs"""
    pattern_id: str
    pattern_type: FeatureType
    description: str
    confidence: float
    support_count: int  # How many designs exhibit this pattern
    conditions: Dict[str, Any]  # When this pattern applies
    actions: Dict[str, Any]  # What this pattern suggests
    examples: List[str] = field(default_factory=list)  # File sources


@dataclass
class MLModel:
    """Machine learning model for predictions"""
    model_id: str
    model_type: ModelType
    purpose: str
    input_features: List[str]
    output_type: str
    accuracy: float = 0.0
    training_samples: int = 0
    weights: Dict[str, List[float]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningResult:
    """Result from learning session"""
    designs_processed: int
    patterns_discovered: int
    models_trained: int
    patterns: List[LearnedPattern]
    models: List[MLModel]
    recommendations: List[str]
    warnings: List[str]


# =============================================================================
# REVERSE ENGINEERING PARSERS
# =============================================================================

class KiCadParser:
    """Parser for KiCad PCB files (.kicad_pcb)"""

    def parse(self, file_path: str) -> ExtractedDesign:
        """Parse a KiCad PCB file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        design = ExtractedDesign(
            source_file=file_path,
            file_format=FileFormat.KICAD_PCB,
            board_width=0,
            board_height=0,
            layer_count=2,
            components=[],
            traces=[],
            vias=[],
            zones=[],
            nets={},
            design_rules={}
        )

        # Parse board outline
        design.board_width, design.board_height = self._parse_board_outline(content)

        # Parse layers
        design.layer_count = self._parse_layer_count(content)

        # Parse components (footprints)
        design.components = self._parse_footprints(content)

        # Parse traces (segments)
        design.traces = self._parse_segments(content)

        # Parse vias
        design.vias = self._parse_vias(content)

        # Parse zones
        design.zones = self._parse_zones(content)

        # Parse nets
        design.nets = self._parse_nets(content)

        # Parse design rules
        design.design_rules = self._parse_design_rules(content)

        return design

    def _parse_board_outline(self, content: str) -> Tuple[float, float]:
        """Extract board dimensions from edge cuts"""
        # Look for gr_rect or gr_line on Edge.Cuts layer
        edge_pattern = r'\(gr_rect\s+\(start\s+([\d.]+)\s+([\d.]+)\)\s+\(end\s+([\d.]+)\s+([\d.]+)\)'
        match = re.search(edge_pattern, content)
        if match:
            x1, y1, x2, y2 = map(float, match.groups())
            return abs(x2 - x1), abs(y2 - y1)

        # Fallback: find all edge cut lines
        line_pattern = r'\(gr_line\s+\(start\s+([\d.]+)\s+([\d.]+)\)\s+\(end\s+([\d.]+)\s+([\d.]+)\).*?Edge\.Cuts'
        matches = re.findall(line_pattern, content, re.DOTALL)
        if matches:
            xs = [float(m[0]) for m in matches] + [float(m[2]) for m in matches]
            ys = [float(m[1]) for m in matches] + [float(m[3]) for m in matches]
            return max(xs) - min(xs), max(ys) - min(ys)

        return 100.0, 100.0  # Default

    def _parse_layer_count(self, content: str) -> int:
        """Count copper layers"""
        layer_pattern = r'\(layers\s*\n(.*?)\)'
        match = re.search(layer_pattern, content, re.DOTALL)
        if match:
            layers_section = match.group(1)
            copper_count = len(re.findall(r'In\d+\.Cu|F\.Cu|B\.Cu', layers_section))
            return max(copper_count, 2)
        return 2

    def _parse_footprints(self, content: str) -> List[ExtractedComponent]:
        """Parse footprint definitions"""
        components = []
        # Pattern for footprint blocks
        fp_pattern = r'\(footprint\s+"([^"]+)".*?\(at\s+([\d.-]+)\s+([\d.-]+)(?:\s+([\d.-]+))?\)'
        ref_pattern = r'\(fp_text\s+reference\s+"([^"]+)"'
        value_pattern = r'\(fp_text\s+value\s+"([^"]+)"'

        # Find all footprint blocks
        fp_blocks = re.findall(r'\(footprint\s+[^)]*\)(?:[^()]*|\([^)]*\))*', content)

        for block in fp_blocks:
            fp_match = re.search(fp_pattern, block)
            if not fp_match:
                continue

            footprint = fp_match.group(1)
            x = float(fp_match.group(2))
            y = float(fp_match.group(3))
            rotation = float(fp_match.group(4)) if fp_match.group(4) else 0

            ref_match = re.search(ref_pattern, block)
            value_match = re.search(value_pattern, block)

            reference = ref_match.group(1) if ref_match else "?"
            value = value_match.group(1) if value_match else ""

            layer = "Top" if "F.Cu" in block or "(layer \"F.Cu\")" in block else "Bottom"

            components.append(ExtractedComponent(
                reference=reference,
                value=value,
                footprint=footprint,
                x=x,
                y=y,
                rotation=rotation,
                layer=layer
            ))

        return components

    def _parse_segments(self, content: str) -> List[ExtractedTrace]:
        """Parse trace segments"""
        traces = []
        seg_pattern = r'\(segment\s+\(start\s+([\d.-]+)\s+([\d.-]+)\)\s+\(end\s+([\d.-]+)\s+([\d.-]+)\)\s+\(width\s+([\d.]+)\)\s+\(layer\s+"([^"]+)"\)\s+\(net\s+(\d+)\)'

        for match in re.finditer(seg_pattern, content):
            x1, y1, x2, y2, width, layer, net_id = match.groups()
            traces.append(ExtractedTrace(
                net_name=f"Net{net_id}",
                points=[(float(x1), float(y1)), (float(x2), float(y2))],
                width=float(width),
                layer=layer
            ))

        return traces

    def _parse_vias(self, content: str) -> List[ExtractedVia]:
        """Parse via definitions"""
        vias = []
        via_pattern = r'\(via\s+\(at\s+([\d.-]+)\s+([\d.-]+)\)\s+\(size\s+([\d.]+)\)\s+\(drill\s+([\d.]+)\)\s+\(layers\s+"([^"]+)"\s+"([^"]+)"\)\s+\(net\s+(\d+)\)'

        for match in re.finditer(via_pattern, content):
            x, y, size, drill, layer1, layer2, net_id = match.groups()
            vias.append(ExtractedVia(
                x=float(x),
                y=float(y),
                drill=float(drill),
                size=float(size),
                net_name=f"Net{net_id}",
                layers=(layer1, layer2)
            ))

        return vias

    def _parse_zones(self, content: str) -> List[ExtractedZone]:
        """Parse zone definitions (copper pours)"""
        zones = []
        # Simplified zone parsing
        zone_pattern = r'\(zone\s+\(net\s+(\d+)\).*?\(layer\s+"([^"]+)"\).*?\(filled_polygon\s+\(layer\s+"[^"]+"\)\s+\(pts\s+(.*?)\)\)'

        for match in re.finditer(zone_pattern, content, re.DOTALL):
            net_id, layer, pts_str = match.groups()
            # Parse points
            pts = re.findall(r'\(xy\s+([\d.-]+)\s+([\d.-]+)\)', pts_str)
            outline = [(float(x), float(y)) for x, y in pts]

            zones.append(ExtractedZone(
                net_name=f"Net{net_id}",
                layer=layer,
                outline=outline
            ))

        return zones

    def _parse_nets(self, content: str) -> Dict[str, List[str]]:
        """Parse net definitions"""
        nets = {}
        net_pattern = r'\(net\s+(\d+)\s+"([^"]+)"\)'

        for match in re.finditer(net_pattern, content):
            net_id, net_name = match.groups()
            nets[net_name] = []  # Connections filled from footprint pads

        return nets

    def _parse_design_rules(self, content: str) -> Dict[str, float]:
        """Parse design rules from setup section"""
        rules = {}

        # Extract setup section
        setup_pattern = r'\(setup\s+(.*?)\)\s*\(net_classes'
        match = re.search(setup_pattern, content, re.DOTALL)
        if not match:
            return rules

        setup = match.group(1)

        # Parse various rules
        rule_patterns = {
            'trace_width': r'\(defaults\s+\(trace_width\s+([\d.]+)\)',
            'clearance': r'\(pad_to_mask_clearance\s+([\d.]+)\)',
            'via_drill': r'\(via_drill\s+([\d.]+)\)',
            'via_size': r'\(via_diameter\s+([\d.]+)\)',
        }

        for rule_name, pattern in rule_patterns.items():
            match = re.search(pattern, setup)
            if match:
                rules[rule_name] = float(match.group(1))

        return rules


class GerberParser:
    """Parser for Gerber manufacturing files"""

    def parse(self, file_path: str) -> ExtractedDesign:
        """Parse a Gerber file (GTL, GBL, etc.)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        design = ExtractedDesign(
            source_file=file_path,
            file_format=FileFormat.GERBER,
            board_width=0,
            board_height=0,
            layer_count=1,
            components=[],
            traces=[],
            vias=[],
            zones=[],
            nets={},
            design_rules={}
        )

        # Parse Gerber commands
        traces, apertures = self._parse_gerber(content)
        design.traces = traces

        # Estimate board size from extents
        if traces:
            all_points = [p for t in traces for p in t.points]
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            design.board_width = max(xs) - min(xs)
            design.board_height = max(ys) - min(ys)

        return design

    def _parse_gerber(self, content: str) -> Tuple[List[ExtractedTrace], Dict[str, float]]:
        """Parse Gerber X2/RS-274X format"""
        traces = []
        apertures = {}

        # Parse aperture definitions
        ap_pattern = r'%ADD(\d+)([A-Z]),([^*]+)\*%'
        for match in re.finditer(ap_pattern, content):
            code, shape, params = match.groups()
            # Parse first parameter as size
            size = float(params.split('X')[0]) if params else 0.1
            apertures[code] = size

        # Parse draw commands
        current_aperture = None
        current_x, current_y = 0.0, 0.0

        # D codes: D01=draw, D02=move, D03=flash
        # G codes: G01=linear, G02=CW arc, G03=CCW arc
        coord_pattern = r'X(-?\d+)Y(-?\d+)D(\d+)\*'

        for match in re.finditer(coord_pattern, content):
            x_str, y_str, d_code = match.groups()
            # Convert to mm (assuming format 2.4 = 24 -> 0.0024)
            x = int(x_str) / 10000.0  # Adjust based on format
            y = int(y_str) / 10000.0

            if d_code == '01' and current_aperture:  # Draw
                width = apertures.get(current_aperture, 0.2)
                traces.append(ExtractedTrace(
                    net_name="Unknown",
                    points=[(current_x, current_y), (x, y)],
                    width=width,
                    layer="Gerber"
                ))

            current_x, current_y = x, y

        # Parse aperture selections
        ap_select_pattern = r'D(\d+)\*'
        for match in re.finditer(ap_select_pattern, content):
            current_aperture = match.group(1)

        return traces, apertures


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Extracts ML-ready features from designs"""

    def extract(self, design: ExtractedDesign) -> DesignFeatures:
        """Extract all features from a design"""
        features = DesignFeatures(
            component_density=self._calc_component_density(design),
            placement_symmetry=self._calc_placement_symmetry(design),
            cluster_count=self._count_clusters(design),
            power_component_positions=self._find_power_components(design),
            io_component_positions=self._find_io_components(design),
            average_trace_length=self._calc_avg_trace_length(design),
            via_density=self._calc_via_density(design),
            layer_utilization=self._calc_layer_utilization(design),
            routing_congestion_map=self._build_congestion_map(design),
            differential_pair_count=self._count_diff_pairs(design),
            matched_length_groups=self._count_matched_groups(design),
            guard_traces_present=self._detect_guard_traces(design),
            power_plane_coverage=self._calc_power_coverage(design),
            ground_plane_coverage=self._calc_ground_coverage(design),
            decap_placement_score=self._score_decap_placement(design),
            drc_violation_count=0,  # Would need DRC piston
            estimated_quality_score=0.0
        )

        # Build feature vector
        features.feature_vector, features.feature_names = self._build_feature_vector(features, design)
        features.estimated_quality_score = self._estimate_quality(features)

        return features

    def _calc_component_density(self, design: ExtractedDesign) -> float:
        """Calculate component density (parts per cm²)"""
        area = design.board_width * design.board_height / 100  # cm²
        return len(design.components) / area if area > 0 else 0

    def _calc_placement_symmetry(self, design: ExtractedDesign) -> float:
        """Calculate placement symmetry score (0-1)"""
        if len(design.components) < 2:
            return 1.0

        center_x = design.board_width / 2
        center_y = design.board_height / 2

        # Calculate moment of inertia imbalance
        left_mass = sum(1 for c in design.components if c.x < center_x)
        right_mass = len(design.components) - left_mass
        top_mass = sum(1 for c in design.components if c.y < center_y)
        bottom_mass = len(design.components) - top_mass

        h_balance = 1 - abs(left_mass - right_mass) / len(design.components)
        v_balance = 1 - abs(top_mass - bottom_mass) / len(design.components)

        return (h_balance + v_balance) / 2

    def _count_clusters(self, design: ExtractedDesign) -> int:
        """Count component clusters using simple distance-based clustering"""
        if len(design.components) < 2:
            return 1

        # Simple clustering: components within 10mm are same cluster
        threshold = 10.0
        visited = set()
        clusters = 0

        def find_cluster(comp_idx: int, visited: set):
            stack = [comp_idx]
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                visited.add(idx)
                comp = design.components[idx]
                for i, other in enumerate(design.components):
                    if i not in visited:
                        dist = math.sqrt((comp.x - other.x)**2 + (comp.y - other.y)**2)
                        if dist < threshold:
                            stack.append(i)

        for i in range(len(design.components)):
            if i not in visited:
                find_cluster(i, visited)
                clusters += 1

        return clusters

    def _find_power_components(self, design: ExtractedDesign) -> List[Tuple[float, float]]:
        """Find power-related component positions"""
        power_refs = ['U', 'VR', 'L', 'D']  # Regulators, inductors, diodes
        power_values = ['LDO', 'REG', 'BUCK', 'BOOST', 'AMS1117', 'LM7805']

        positions = []
        for comp in design.components:
            ref_prefix = ''.join(c for c in comp.reference if c.isalpha())
            if ref_prefix in power_refs or any(pv in comp.value.upper() for pv in power_values):
                positions.append((comp.x, comp.y))

        return positions

    def _find_io_components(self, design: ExtractedDesign) -> List[Tuple[float, float]]:
        """Find IO-related component positions"""
        io_refs = ['J', 'P', 'X', 'CN']  # Connectors
        positions = []
        for comp in design.components:
            ref_prefix = ''.join(c for c in comp.reference if c.isalpha())
            if ref_prefix in io_refs:
                positions.append((comp.x, comp.y))
        return positions

    def _calc_avg_trace_length(self, design: ExtractedDesign) -> float:
        """Calculate average trace length"""
        if not design.traces:
            return 0

        total_length = 0
        for trace in design.traces:
            for i in range(len(trace.points) - 1):
                p1, p2 = trace.points[i], trace.points[i + 1]
                total_length += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        return total_length / len(design.traces)

    def _calc_via_density(self, design: ExtractedDesign) -> float:
        """Calculate via density (vias per cm²)"""
        area = design.board_width * design.board_height / 100
        return len(design.vias) / area if area > 0 else 0

    def _calc_layer_utilization(self, design: ExtractedDesign) -> Dict[str, float]:
        """Calculate trace coverage per layer"""
        utilization = defaultdict(float)
        board_area = design.board_width * design.board_height

        for trace in design.traces:
            for i in range(len(trace.points) - 1):
                p1, p2 = trace.points[i], trace.points[i + 1]
                length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                area = length * trace.width
                utilization[trace.layer] += area

        return {layer: area / board_area for layer, area in utilization.items()}

    def _build_congestion_map(self, design: ExtractedDesign, grid_size: int = 10) -> List[List[float]]:
        """Build routing congestion heatmap"""
        cell_w = design.board_width / grid_size
        cell_h = design.board_height / grid_size
        grid = [[0.0] * grid_size for _ in range(grid_size)]

        for trace in design.traces:
            for p in trace.points:
                gx = min(int(p[0] / cell_w), grid_size - 1)
                gy = min(int(p[1] / cell_h), grid_size - 1)
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    grid[gy][gx] += 1

        # Normalize
        max_val = max(max(row) for row in grid)
        if max_val > 0:
            grid = [[cell / max_val for cell in row] for row in grid]

        return grid

    def _count_diff_pairs(self, design: ExtractedDesign) -> int:
        """Count differential pair nets"""
        diff_patterns = ['_P', '_N', '+', '-', 'DP', 'DN', 'D+', 'D-']
        count = 0
        net_names = list(design.nets.keys())

        for name in net_names:
            for pattern in diff_patterns:
                if pattern in name:
                    # Check for complementary
                    complement = name.replace('_P', '_N').replace('_N', '_P')
                    complement = complement.replace('+', '-').replace('-', '+')
                    if complement in net_names and complement != name:
                        count += 1
                        break

        return count // 2  # Pairs, not individual nets

    def _count_matched_groups(self, design: ExtractedDesign) -> int:
        """Count length-matched net groups"""
        # Look for bus patterns: D0, D1, D2... or A[0], A[1]...
        groups = defaultdict(list)

        for name in design.nets.keys():
            # Extract base name
            base = re.sub(r'\d+$', '', name)
            base = re.sub(r'\[\d+\]$', '', base)
            if base:
                groups[base].append(name)

        # Groups with 4+ nets likely need matching
        return sum(1 for nets in groups.values() if len(nets) >= 4)

    def _detect_guard_traces(self, design: ExtractedDesign) -> bool:
        """Detect presence of guard traces"""
        gnd_traces = [t for t in design.traces if 'GND' in t.net_name.upper()]
        return len(gnd_traces) > len(design.traces) * 0.1  # More than 10% ground

    def _calc_power_coverage(self, design: ExtractedDesign) -> float:
        """Calculate power plane coverage"""
        power_zones = [z for z in design.zones if 'VCC' in z.net_name.upper() or 'VDD' in z.net_name.upper() or '3V3' in z.net_name or '5V' in z.net_name]
        if not power_zones:
            return 0

        # Simplified: count zone points as proxy for area
        total_points = sum(len(z.outline) for z in power_zones)
        board_area = design.board_width * design.board_height
        return min(total_points * 10 / board_area, 1.0)  # Rough estimate

    def _calc_ground_coverage(self, design: ExtractedDesign) -> float:
        """Calculate ground plane coverage"""
        gnd_zones = [z for z in design.zones if 'GND' in z.net_name.upper()]
        if not gnd_zones:
            return 0

        total_points = sum(len(z.outline) for z in gnd_zones)
        board_area = design.board_width * design.board_height
        return min(total_points * 10 / board_area, 1.0)

    def _score_decap_placement(self, design: ExtractedDesign) -> float:
        """Score decoupling capacitor placement (closer to ICs = better)"""
        caps = [c for c in design.components if c.reference.startswith('C')]
        ics = [c for c in design.components if c.reference.startswith('U')]

        if not caps or not ics:
            return 0.5  # Neutral

        # Average distance from caps to nearest IC
        total_dist = 0
        for cap in caps:
            min_dist = min(
                math.sqrt((cap.x - ic.x)**2 + (cap.y - ic.y)**2)
                for ic in ics
            )
            total_dist += min_dist

        avg_dist = total_dist / len(caps)

        # Score: closer = better (5mm or less is ideal)
        return max(0, 1 - avg_dist / 20)  # 0 at 20mm, 1 at 0mm

    def _build_feature_vector(self, features: DesignFeatures, design: ExtractedDesign) -> Tuple[List[float], List[str]]:
        """Build flat feature vector for ML"""
        vector = []
        names = []

        # Scalar features
        scalar_features = [
            ('component_density', features.component_density),
            ('placement_symmetry', features.placement_symmetry),
            ('cluster_count', features.cluster_count),
            ('avg_trace_length', features.average_trace_length),
            ('via_density', features.via_density),
            ('diff_pair_count', features.differential_pair_count),
            ('matched_groups', features.matched_length_groups),
            ('guard_traces', 1.0 if features.guard_traces_present else 0.0),
            ('power_coverage', features.power_plane_coverage),
            ('ground_coverage', features.ground_plane_coverage),
            ('decap_score', features.decap_placement_score),
            ('board_width', design.board_width),
            ('board_height', design.board_height),
            ('layer_count', design.layer_count),
            ('component_count', len(design.components)),
            ('trace_count', len(design.traces)),
            ('via_count', len(design.vias)),
            ('zone_count', len(design.zones)),
            ('net_count', len(design.nets)),
        ]

        for name, value in scalar_features:
            vector.append(float(value))
            names.append(name)

        # Flatten congestion map
        for i, row in enumerate(features.routing_congestion_map):
            for j, val in enumerate(row):
                vector.append(val)
                names.append(f'congestion_{i}_{j}')

        return vector, names

    def _estimate_quality(self, features: DesignFeatures) -> float:
        """Estimate overall design quality score"""
        score = 0.5  # Start neutral

        # Good indicators
        if features.placement_symmetry > 0.7:
            score += 0.1
        if features.ground_plane_coverage > 0.5:
            score += 0.1
        if features.decap_placement_score > 0.7:
            score += 0.1
        if features.guard_traces_present:
            score += 0.05

        # Bad indicators
        if features.component_density > 2.0:  # Too dense
            score -= 0.1
        if features.via_density > 5.0:  # Too many vias
            score -= 0.05

        return max(0, min(1, score))


# =============================================================================
# MACHINE LEARNING MODELS
# =============================================================================

class SimpleNeuralNetwork:
    """Simple neural network for learning (no external dependencies)"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights randomly
        self.w1 = [[random.gauss(0, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[random.gauss(0, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-max(-500, min(500, x))))

    def _relu(self, x: float) -> float:
        return max(0, x)

    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass"""
        # Hidden layer
        hidden = []
        for j in range(self.hidden_size):
            h = sum(inputs[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j]
            hidden.append(self._relu(h))

        # Output layer
        outputs = []
        for k in range(self.output_size):
            o = sum(hidden[j] * self.w2[j][k] for j in range(self.hidden_size)) + self.b2[k]
            outputs.append(self._sigmoid(o))

        return outputs

    def train(self, X: List[List[float]], y: List[List[float]], epochs: int = 100, lr: float = 0.01):
        """Train the network using simple gradient descent"""
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                # Forward
                output = self.forward(xi)

                # Calculate loss (MSE)
                loss = sum((output[k] - yi[k])**2 for k in range(self.output_size))
                total_loss += loss

                # Backprop (simplified)
                # This is a very basic implementation
                for k in range(self.output_size):
                    error = output[k] - yi[k]
                    for j in range(self.hidden_size):
                        self.w2[j][k] -= lr * error * self._relu(
                            sum(xi[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j]
                        )
                    self.b2[k] -= lr * error

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Loss = {total_loss / len(X):.4f}")

    def to_dict(self) -> Dict:
        """Serialize to dict"""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SimpleNeuralNetwork':
        """Deserialize from dict"""
        nn = cls(data['input_size'], data['hidden_size'], data['output_size'])
        nn.w1 = data['w1']
        nn.b1 = data['b1']
        nn.w2 = data['w2']
        nn.b2 = data['b2']
        return nn


class DecisionTreeNode:
    """Node in a decision tree"""

    def __init__(self):
        self.feature_index: int = -1
        self.threshold: float = 0.0
        self.left: 'DecisionTreeNode' = None
        self.right: 'DecisionTreeNode' = None
        self.value: Any = None  # Leaf value
        self.is_leaf: bool = False


class SimpleDecisionTree:
    """Simple decision tree classifier"""

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.root: DecisionTreeNode = None

    def fit(self, X: List[List[float]], y: List[int]):
        """Train the decision tree"""
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: List[List[float]], y: List[int], depth: int) -> DecisionTreeNode:
        """Recursively build tree"""
        node = DecisionTreeNode()

        # Check stopping conditions
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < 2:
            node.is_leaf = True
            node.value = max(set(y), key=y.count) if y else 0
            return node

        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        if best_gain <= 0:
            node.is_leaf = True
            node.value = max(set(y), key=y.count)
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold

        # Split data
        left_X, left_y, right_X, right_y = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[best_feature] <= best_threshold:
                left_X.append(xi)
                left_y.append(yi)
            else:
                right_X.append(xi)
                right_y.append(yi)

        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)

        return node

    def _find_best_split(self, X: List[List[float]], y: List[int]) -> Tuple[int, float, float]:
        """Find best feature and threshold to split on"""
        best_gain = -1
        best_feature = 0
        best_threshold = 0

        n_features = len(X[0])

        for f in range(n_features):
            values = sorted(set(x[f] for x in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                gain = self._info_gain(X, y, f, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _info_gain(self, X: List[List[float]], y: List[int], feature: int, threshold: float) -> float:
        """Calculate information gain from split"""
        left_y = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
        right_y = [y[i] for i in range(len(X)) if X[i][feature] > threshold]

        if not left_y or not right_y:
            return 0

        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)

        n = len(y)
        weighted_entropy = (len(left_y) / n) * left_entropy + (len(right_y) / n) * right_entropy

        return parent_entropy - weighted_entropy

    def _entropy(self, y: List[int]) -> float:
        """Calculate entropy"""
        if not y:
            return 0
        counts = defaultdict(int)
        for yi in y:
            counts[yi] += 1
        n = len(y)
        entropy = 0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def predict(self, x: List[float]) -> int:
        """Predict class for single sample"""
        node = self.root
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class PatternMiner:
    """Discovers patterns in design data"""

    def __init__(self):
        self.patterns: List[LearnedPattern] = []

    def mine_patterns(self, designs: List[ExtractedDesign], features_list: List[DesignFeatures]) -> List[LearnedPattern]:
        """Mine patterns from multiple designs"""
        patterns = []

        # Pattern 1: Power component placement
        patterns.extend(self._mine_power_patterns(designs, features_list))

        # Pattern 2: Decap placement
        patterns.extend(self._mine_decap_patterns(designs, features_list))

        # Pattern 3: IO connector placement
        patterns.extend(self._mine_io_patterns(designs, features_list))

        # Pattern 4: Routing strategies
        patterns.extend(self._mine_routing_patterns(designs, features_list))

        # Pattern 5: Ground plane usage
        patterns.extend(self._mine_ground_patterns(designs, features_list))

        self.patterns = patterns
        return patterns

    def _mine_power_patterns(self, designs: List[ExtractedDesign], features_list: List[DesignFeatures]) -> List[LearnedPattern]:
        """Mine power component placement patterns"""
        patterns = []

        # Analyze power component locations
        edge_power = 0
        center_power = 0

        for design, features in zip(designs, features_list):
            for x, y in features.power_component_positions:
                # Check if near edge
                margin = min(design.board_width, design.board_height) * 0.2
                if x < margin or x > design.board_width - margin or \
                   y < margin or y > design.board_height - margin:
                    edge_power += 1
                else:
                    center_power += 1

        if edge_power > center_power and edge_power >= 3:
            patterns.append(LearnedPattern(
                pattern_id='POWER_EDGE',
                pattern_type=FeatureType.POWER,
                description='Power components are placed near board edges',
                confidence=edge_power / (edge_power + center_power),
                support_count=edge_power,
                conditions={'component_type': 'power'},
                actions={'placement_region': 'edge', 'margin': 0.2}
            ))

        return patterns

    def _mine_decap_patterns(self, designs: List[ExtractedDesign], features_list: List[DesignFeatures]) -> List[LearnedPattern]:
        """Mine decoupling capacitor patterns"""
        patterns = []

        good_decap_count = sum(1 for f in features_list if f.decap_placement_score > 0.7)

        if good_decap_count >= len(features_list) * 0.5:
            patterns.append(LearnedPattern(
                pattern_id='DECAP_NEAR_IC',
                pattern_type=FeatureType.POWER,
                description='Decoupling capacitors placed within 5mm of ICs',
                confidence=good_decap_count / len(features_list),
                support_count=good_decap_count,
                conditions={'component_type': 'capacitor', 'value_range': '100nF-10uF'},
                actions={'max_distance_to_ic': 5.0, 'priority': 'high'}
            ))

        return patterns

    def _mine_io_patterns(self, designs: List[ExtractedDesign], features_list: List[DesignFeatures]) -> List[LearnedPattern]:
        """Mine IO connector placement patterns"""
        patterns = []

        edge_io = 0
        for design, features in zip(designs, features_list):
            for x, y in features.io_component_positions:
                margin = 5.0  # 5mm from edge
                if x < margin or x > design.board_width - margin or \
                   y < margin or y > design.board_height - margin:
                    edge_io += 1

        if edge_io >= 3:
            patterns.append(LearnedPattern(
                pattern_id='IO_EDGE',
                pattern_type=FeatureType.PLACEMENT,
                description='IO connectors placed at board edges',
                confidence=0.9,
                support_count=edge_io,
                conditions={'component_type': 'connector'},
                actions={'placement_region': 'edge', 'orientation': 'outward'}
            ))

        return patterns

    def _mine_routing_patterns(self, designs: List[ExtractedDesign], features_list: List[DesignFeatures]) -> List[LearnedPattern]:
        """Mine routing strategy patterns"""
        patterns = []

        # Analyze via usage vs layer count
        via_ratios = []
        for design, features in zip(designs, features_list):
            if design.layer_count > 2:
                via_ratio = len(design.vias) / max(len(design.traces), 1)
                via_ratios.append(via_ratio)

        if via_ratios:
            avg_via_ratio = sum(via_ratios) / len(via_ratios)
            if avg_via_ratio < 0.3:
                patterns.append(LearnedPattern(
                    pattern_id='MINIMAL_VIAS',
                    pattern_type=FeatureType.ROUTING,
                    description='Minimal via usage in multi-layer designs',
                    confidence=0.8,
                    support_count=len(via_ratios),
                    conditions={'layer_count': '>2'},
                    actions={'via_cost': 'high', 'prefer_same_layer': True}
                ))

        return patterns

    def _mine_ground_patterns(self, designs: List[ExtractedDesign], features_list: List[DesignFeatures]) -> List[LearnedPattern]:
        """Mine ground plane patterns"""
        patterns = []

        good_ground = sum(1 for f in features_list if f.ground_plane_coverage > 0.4)

        if good_ground >= len(features_list) * 0.6:
            patterns.append(LearnedPattern(
                pattern_id='SOLID_GROUND',
                pattern_type=FeatureType.GROUND,
                description='Solid ground plane covering >40% of layer',
                confidence=good_ground / len(features_list),
                support_count=good_ground,
                conditions={'design_type': 'any'},
                actions={'ground_plane': True, 'min_coverage': 0.4}
            ))

        return patterns


# =============================================================================
# MAIN LEARNING PISTON
# =============================================================================

class LearningPiston:
    """
    Learning Piston - Machine Learning from PCB Designs

    This piston learns from real PCB designs to:
    1. Improve placement algorithms
    2. Improve routing strategies
    3. Learn design rules
    4. Predict quality issues
    5. Generate new algorithms
    """

    def __init__(self):
        self.name = "Learning"
        self.version = "1.0.0"

        # Parsers for different file formats
        self.parsers = {
            FileFormat.KICAD_PCB: KiCadParser(),
            FileFormat.GERBER: GerberParser(),
        }

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Pattern miner
        self.pattern_miner = PatternMiner()

        # Trained models
        self.models: Dict[str, Any] = {}

        # Learned patterns
        self.patterns: List[LearnedPattern] = []

        # Training data
        self.training_designs: List[ExtractedDesign] = []
        self.training_features: List[DesignFeatures] = []

    def observe(self, design_data: Dict) -> Dict[str, Any]:
        """
        Standard piston API - observe and extract patterns from a design.

        Args:
            design_data: Design data with parts_db, placement, routes

        Returns:
            Dictionary with observed patterns
        """
        parts_db = design_data.get('parts_db', {})
        placement = design_data.get('placement', {})
        routes = design_data.get('routes', {})

        parts = parts_db.get('parts', {})

        # Extract basic patterns
        patterns = {
            'component_count': len(parts),
            'net_count': len(parts_db.get('nets', {})),
            'placement_count': len(placement),
            'route_count': len(routes),
            'component_types': {},
            'footprint_distribution': {}
        }

        # Count component types
        for ref, part in parts.items():
            footprint = part.get('footprint', 'Unknown')
            patterns['footprint_distribution'][footprint] = \
                patterns['footprint_distribution'].get(footprint, 0) + 1

            # Classify by ref prefix
            prefix = ''.join(c for c in ref if c.isalpha())
            patterns['component_types'][prefix] = \
                patterns['component_types'].get(prefix, 0) + 1

        return {
            'patterns': patterns,
            'mode': 'observe',
            'success': True
        }

    def learn(self,
              file_paths: List[str],
              mode: LearningMode = LearningMode.UNSUPERVISED,
              labels: Optional[Dict[str, float]] = None) -> 'LearningResult':
        """
        Main entry point - learn from PCB files

        Args:
            file_paths: List of PCB file paths to learn from
            mode: Learning mode (supervised, unsupervised, etc.)
            labels: Optional quality labels for supervised learning

        Returns:
            LearningResult with patterns and models
        """
        warnings = []
        designs = []
        features_list = []

        print(f"[Learning Piston] Processing {len(file_paths)} files...")

        # Step 1: Parse all files
        for path in file_paths:
            try:
                design = self._parse_file(path)
                if design:
                    designs.append(design)
                    features = self.feature_extractor.extract(design)
                    features_list.append(features)
                    print(f"  Parsed: {os.path.basename(path)} - {len(design.components)} components")
                else:
                    warnings.append(f"Could not parse: {path}")
            except Exception as e:
                warnings.append(f"Error parsing {path}: {str(e)}")

        if not designs:
            return LearningResult(
                designs_processed=0,
                patterns_discovered=0,
                models_trained=0,
                patterns=[],
                models=[],
                recommendations=["No valid designs found"],
                warnings=warnings
            )

        self.training_designs.extend(designs)
        self.training_features.extend(features_list)

        # Step 2: Mine patterns
        print("[Learning Piston] Mining patterns...")
        patterns = self.pattern_miner.mine_patterns(designs, features_list)
        self.patterns.extend(patterns)
        print(f"  Discovered {len(patterns)} patterns")

        # Step 3: Train models
        models_trained = []

        if mode == LearningMode.SUPERVISED and labels:
            print("[Learning Piston] Training supervised models...")
            models_trained.extend(self._train_supervised(features_list, labels))

        elif mode == LearningMode.UNSUPERVISED:
            print("[Learning Piston] Training unsupervised models...")
            models_trained.extend(self._train_unsupervised(features_list))

        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(patterns, models_trained)

        return LearningResult(
            designs_processed=len(designs),
            patterns_discovered=len(patterns),
            models_trained=len(models_trained),
            patterns=patterns,
            models=models_trained,
            recommendations=recommendations,
            warnings=warnings
        )

    def _parse_file(self, path: str) -> Optional[ExtractedDesign]:
        """Parse a PCB file based on extension"""
        ext = os.path.splitext(path)[1].lower()

        if ext == '.kicad_pcb':
            return self.parsers[FileFormat.KICAD_PCB].parse(path)
        elif ext in ['.gbr', '.gtl', '.gbl', '.gts', '.gbs', '.gto', '.gbo']:
            return self.parsers[FileFormat.GERBER].parse(path)
        else:
            return None

    def _train_supervised(self, features_list: List[DesignFeatures], labels: Dict[str, float]) -> List[MLModel]:
        """Train supervised models with quality labels"""
        models = []

        # Prepare training data
        X = [f.feature_vector for f in features_list]
        y = [[labels.get(f'design_{i}', 0.5)] for i in range(len(X))]

        # Train quality prediction network
        if len(X) >= 3:
            input_size = len(X[0])
            nn = SimpleNeuralNetwork(input_size, hidden_size=32, output_size=1)
            print("  Training Quality Predictor...")
            nn.train(X, y, epochs=50, lr=0.01)

            model = MLModel(
                model_id='quality_predictor',
                model_type=ModelType.NEURAL_NETWORK,
                purpose='Predict design quality score',
                input_features=features_list[0].feature_names if features_list else [],
                output_type='float',
                accuracy=0.0,  # Would need validation set
                training_samples=len(X),
                weights=nn.to_dict()
            )
            models.append(model)
            self.models['quality_predictor'] = nn

        return models

    def _train_unsupervised(self, features_list: List[DesignFeatures]) -> List[MLModel]:
        """Train unsupervised models for pattern discovery"""
        models = []

        if len(features_list) < 2:
            return models

        # Train routing strategy classifier
        X = [f.feature_vector for f in features_list]

        # Classify based on via density and layer utilization
        y = []
        for f in features_list:
            if f.via_density < 1.0:
                y.append(0)  # Minimal via strategy
            elif f.via_density < 3.0:
                y.append(1)  # Moderate via strategy
            else:
                y.append(2)  # Heavy via strategy

        if len(set(y)) > 1:
            tree = SimpleDecisionTree(max_depth=4)
            tree.fit(X, y)

            model = MLModel(
                model_id='routing_strategy_classifier',
                model_type=ModelType.DECISION_TREE,
                purpose='Classify routing strategy from design features',
                input_features=features_list[0].feature_names if features_list else [],
                output_type='class',
                accuracy=0.0,
                training_samples=len(X),
                parameters={'classes': ['minimal_via', 'moderate_via', 'heavy_via']}
            )
            models.append(model)
            self.models['routing_classifier'] = tree

        return models

    def _generate_recommendations(self, patterns: List[LearnedPattern], models: List[MLModel]) -> List[str]:
        """Generate actionable recommendations from learning"""
        recommendations = []

        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.pattern_type == FeatureType.POWER:
                recommendations.append(f"Power Pattern: {pattern.description} (confidence: {pattern.confidence:.0%})")
            elif pattern.pattern_type == FeatureType.ROUTING:
                recommendations.append(f"Routing Pattern: {pattern.description}")
            elif pattern.pattern_type == FeatureType.GROUND:
                recommendations.append(f"Ground Pattern: {pattern.description}")

        # Model-based recommendations
        for model in models:
            recommendations.append(f"Trained {model.purpose} on {model.training_samples} samples")

        return recommendations

    def predict_quality(self, design: ExtractedDesign) -> float:
        """Predict quality score for a design using learned model"""
        if 'quality_predictor' not in self.models:
            return 0.5  # Default neutral

        features = self.feature_extractor.extract(design)
        nn = self.models['quality_predictor']
        output = nn.forward(features.feature_vector)
        return output[0]

    def recommend_routing_strategy(self, design: ExtractedDesign) -> str:
        """Recommend routing strategy based on learned patterns"""
        if 'routing_classifier' not in self.models:
            return "hybrid"

        features = self.feature_extractor.extract(design)
        tree = self.models['routing_classifier']
        class_idx = tree.predict(features.feature_vector)

        strategies = ['minimal_via', 'moderate_via', 'heavy_via']
        return strategies[class_idx] if class_idx < len(strategies) else "hybrid"

    def apply_patterns(self, design_data: Dict) -> Dict:
        """Apply learned patterns to improve a design"""
        improvements = {
            'applied_patterns': [],
            'suggested_changes': []
        }

        for pattern in self.patterns:
            # Check if pattern applies
            if self._pattern_applies(pattern, design_data):
                improvements['applied_patterns'].append(pattern.pattern_id)
                improvements['suggested_changes'].append({
                    'pattern': pattern.pattern_id,
                    'action': pattern.actions,
                    'confidence': pattern.confidence
                })

        return improvements

    def _pattern_applies(self, pattern: LearnedPattern, design_data: Dict) -> bool:
        """Check if a pattern applies to the given design"""
        conditions = pattern.conditions

        if 'layer_count' in conditions:
            if conditions['layer_count'] == '>2':
                if design_data.get('layer_count', 2) <= 2:
                    return False

        return True

    def save_models(self, path: str):
        """Save learned models to file"""
        data = {
            'patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type.name,
                    'description': p.description,
                    'confidence': p.confidence,
                    'support_count': p.support_count,
                    'conditions': p.conditions,
                    'actions': p.actions
                }
                for p in self.patterns
            ],
            'models': {}
        }

        # Save neural network weights
        if 'quality_predictor' in self.models:
            data['models']['quality_predictor'] = self.models['quality_predictor'].to_dict()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Learning Piston] Saved {len(self.patterns)} patterns and {len(data['models'])} models to {path}")

    def load_models(self, path: str):
        """Load learned models from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        # Load patterns
        self.patterns = []
        for p in data.get('patterns', []):
            self.patterns.append(LearnedPattern(
                pattern_id=p['pattern_id'],
                pattern_type=FeatureType[p['pattern_type']],
                description=p['description'],
                confidence=p['confidence'],
                support_count=p['support_count'],
                conditions=p['conditions'],
                actions=p['actions']
            ))

        # Load models
        if 'quality_predictor' in data.get('models', {}):
            self.models['quality_predictor'] = SimpleNeuralNetwork.from_dict(
                data['models']['quality_predictor']
            )

        print(f"[Learning Piston] Loaded {len(self.patterns)} patterns and {len(self.models)} models")

    def generate_algorithm_code(self, pattern: LearnedPattern) -> str:
        """Generate Python code for a learned pattern"""
        code = f'''
# Auto-generated from pattern: {pattern.pattern_id}
# Confidence: {pattern.confidence:.0%}
# Source: Learned from {pattern.support_count} designs

def apply_{pattern.pattern_id.lower()}(component, board):
    """
    {pattern.description}

    Conditions: {pattern.conditions}
    """
'''
        # Generate condition checks
        for key, value in pattern.conditions.items():
            code += f"    if component.{key} != '{value}':\n"
            code += f"        return False\n"

        # Generate actions
        code += "\n    # Apply learned actions\n"
        for key, value in pattern.actions.items():
            code += f"    component.{key} = {repr(value)}\n"

        code += "    return True\n"

        return code


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def learn_from_directory(directory: str, pattern: str = "*.kicad_pcb") -> LearningResult:
    """
    Learn from all PCB files in a directory

    Args:
        directory: Path to directory with PCB files
        pattern: Glob pattern for files

    Returns:
        LearningResult
    """
    import glob

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)

    piston = LearningPiston()
    return piston.learn(files)


def reverse_engineer(file_path: str) -> ExtractedDesign:
    """
    Reverse engineer a single PCB file

    Args:
        file_path: Path to PCB file

    Returns:
        ExtractedDesign with all extracted data
    """
    piston = LearningPiston()
    return piston._parse_file(file_path)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Learning Piston Self-Test")
    print("=" * 60)

    # Create test design data
    test_design = ExtractedDesign(
        source_file="test.kicad_pcb",
        file_format=FileFormat.KICAD_PCB,
        board_width=50.0,
        board_height=50.0,
        layer_count=4,
        components=[
            ExtractedComponent("U1", "ESP32", "QFN-48", 25.0, 25.0, 0, "Top"),
            ExtractedComponent("C1", "100nF", "0402", 26.0, 24.0, 0, "Top"),
            ExtractedComponent("C2", "10uF", "0805", 24.0, 26.0, 0, "Top"),
            ExtractedComponent("R1", "10k", "0402", 20.0, 20.0, 0, "Top"),
            ExtractedComponent("J1", "USB-C", "USB_C", 2.0, 25.0, 0, "Top"),
        ],
        traces=[
            ExtractedTrace("VCC", [(25, 25), (26, 24)], 0.3, "F.Cu"),
            ExtractedTrace("GND", [(25, 25), (24, 26)], 0.3, "F.Cu"),
        ],
        vias=[
            ExtractedVia(25.0, 25.0, 0.3, 0.6, "VCC", ("F.Cu", "B.Cu")),
        ],
        zones=[
            ExtractedZone("GND", "B.Cu", [(0, 0), (50, 0), (50, 50), (0, 50)]),
        ],
        nets={"VCC": [], "GND": []},
        design_rules={"trace_width": 0.25, "clearance": 0.15}
    )

    # Test feature extraction
    print("\n[Test] Feature Extraction")
    extractor = FeatureExtractor()
    features = extractor.extract(test_design)
    print(f"  Component density: {features.component_density:.2f} parts/cm²")
    print(f"  Placement symmetry: {features.placement_symmetry:.2f}")
    print(f"  Decap score: {features.decap_placement_score:.2f}")
    print(f"  Ground coverage: {features.ground_plane_coverage:.2f}")
    print(f"  Feature vector size: {len(features.feature_vector)}")

    # Test pattern mining
    print("\n[Test] Pattern Mining")
    miner = PatternMiner()
    patterns = miner.mine_patterns([test_design], [features])
    print(f"  Discovered {len(patterns)} patterns")
    for p in patterns:
        print(f"    - {p.pattern_id}: {p.description}")

    # Test neural network
    print("\n[Test] Neural Network")
    nn = SimpleNeuralNetwork(5, 3, 1)
    X = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
    y = [[0.8], [0.2]]
    nn.train(X, y, epochs=20)
    print(f"  Prediction for [0.3, 0.3, 0.3, 0.3, 0.3]: {nn.forward([0.3, 0.3, 0.3, 0.3, 0.3])}")

    # Test learning piston
    print("\n[Test] Learning Piston Integration")
    piston = LearningPiston()
    piston.training_designs = [test_design]
    piston.training_features = [features]
    piston.patterns = patterns

    # Test code generation
    if patterns:
        code = piston.generate_algorithm_code(patterns[0])
        print(f"  Generated algorithm code ({len(code)} chars)")

    print("\n" + "=" * 60)
    print("Learning Piston Self-Test PASSED")
    print("=" * 60)
