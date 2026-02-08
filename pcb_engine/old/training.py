"""
PCB Engine - Training Module
============================

Enables the engine to learn from existing PCB designs.

TRAINING PHILOSOPHY:
====================
"Show me working designs, and I'll learn the patterns."

This module:
1. Parses existing KiCad PCB files
2. Extracts placement, routing, and design patterns
3. Analyzes relationships between components
4. Generates lessons for the knowledge base
5. Validates patterns across multiple designs

WHAT THE ENGINE LEARNS:
=======================
1. PLACEMENT PATTERNS
   - Component groupings (decoupling caps near ICs)
   - Relative positions (hub orientation, sensor layout)
   - Spacing and clearances used

2. ROUTING PATTERNS
   - Escape directions relative to destinations
   - Trace width by net type
   - Via placement strategies
   - Multi-pin net topologies

3. DESIGN CONVENTIONS
   - Layer usage (signals on top, GND pour bottom)
   - Zone configurations
   - Silkscreen placement

TRAINING DATA FORMAT:
=====================
You provide:
1. .kicad_pcb file - The working design
2. Optional: design_notes.json - Your annotations about decisions

The engine extracts patterns and updates its knowledge base.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import math
import json
import os
from datetime import datetime


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedComponent:
    """Component extracted from PCB"""
    reference: str
    footprint: str
    position: Tuple[float, float]  # mm
    rotation: float  # degrees
    layer: str
    pads: List[Dict] = field(default_factory=list)
    value: str = ''

    @property
    def is_smd(self) -> bool:
        return self.layer in ['F.Cu', 'B.Cu']


@dataclass
class ExtractedNet:
    """Net extracted from PCB"""
    name: str
    net_id: int
    pads: List[Tuple[str, str]] = field(default_factory=list)  # (ref, pad)
    tracks: List[Dict] = field(default_factory=list)
    vias: List[Dict] = field(default_factory=list)

    @property
    def total_length(self) -> float:
        """Calculate total track length"""
        total = 0
        for track in self.tracks:
            start = track.get('start', (0, 0))
            end = track.get('end', (0, 0))
            total += math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        return total


@dataclass
class ExtractedZone:
    """Zone extracted from PCB"""
    net_name: str
    layer: str
    outline: List[Tuple[float, float]] = field(default_factory=list)
    clearance: float = 0.0
    min_thickness: float = 0.0


@dataclass
class DesignAnalysis:
    """Complete analysis of a PCB design"""
    filename: str
    board_size: Tuple[float, float]  # width, height in mm
    layer_count: int
    component_count: int
    net_count: int
    track_count: int
    via_count: int
    zone_count: int

    # Extracted data
    components: Dict[str, ExtractedComponent] = field(default_factory=dict)
    nets: Dict[str, ExtractedNet] = field(default_factory=dict)
    zones: List[ExtractedZone] = field(default_factory=list)

    # Analyzed patterns
    placement_patterns: List[Dict] = field(default_factory=list)
    routing_patterns: List[Dict] = field(default_factory=list)
    design_rules: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'filename': self.filename,
            'board_size': self.board_size,
            'layer_count': self.layer_count,
            'component_count': self.component_count,
            'net_count': self.net_count,
            'track_count': self.track_count,
            'via_count': self.via_count,
            'zone_count': self.zone_count,
            'placement_patterns': self.placement_patterns,
            'routing_patterns': self.routing_patterns,
            'design_rules': self.design_rules,
        }


# =============================================================================
# KICAD PCB PARSER
# =============================================================================

class KiCadPCBParser:
    """
    Parses KiCad PCB files (.kicad_pcb) to extract design information.

    Supports KiCad 6+ file format.
    """

    def __init__(self):
        self.components = {}
        self.nets = {}
        self.tracks = []
        self.vias = []
        self.zones = []
        self.board_outline = []

    def parse(self, filepath: str) -> DesignAnalysis:
        """Parse a KiCad PCB file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Reset state
        self.components = {}
        self.nets = {}
        self.tracks = []
        self.vias = []
        self.zones = []
        self.board_outline = []

        # Parse sections
        self._parse_nets(content)
        self._parse_footprints(content)
        self._parse_tracks(content)
        self._parse_vias(content)
        self._parse_zones(content)
        self._parse_board_outline(content)

        # Calculate board size
        board_size = self._calculate_board_size()

        # Create analysis
        analysis = DesignAnalysis(
            filename=os.path.basename(filepath),
            board_size=board_size,
            layer_count=self._detect_layer_count(content),
            component_count=len(self.components),
            net_count=len(self.nets),
            track_count=len(self.tracks),
            via_count=len(self.vias),
            zone_count=len(self.zones),
            components=self.components,
            nets=self.nets,
            zones=self.zones,
        )

        return analysis

    def _parse_nets(self, content: str):
        """Parse net definitions"""
        # Match (net N "name") or (net N name)
        net_pattern = r'\(net\s+(\d+)\s+"?([^")\s]+)"?\)'
        matches = re.findall(net_pattern, content)

        for net_id, net_name in matches:
            self.nets[net_name] = ExtractedNet(
                name=net_name,
                net_id=int(net_id),
            )

    def _parse_footprints(self, content: str):
        """Parse footprint placements"""
        # Find footprint blocks
        fp_pattern = r'\(footprint\s+"([^"]+)"[^)]*\(layer\s+"([^"]+)"\)[^)]*\(at\s+([\d.-]+)\s+([\d.-]+)(?:\s+([\d.-]+))?\)'
        # This is simplified - real parsing needs nested parentheses handling

        # Alternative: find component blocks more carefully
        footprint_starts = [m.start() for m in re.finditer(r'\(footprint\s+"', content)]

        for start in footprint_starts:
            # Extract the footprint block (find matching parentheses)
            block = self._extract_sexp_block(content, start)
            if block:
                component = self._parse_footprint_block(block)
                if component:
                    self.components[component.reference] = component

    def _parse_footprint_block(self, block: str) -> Optional[ExtractedComponent]:
        """Parse a single footprint block"""
        try:
            # Extract footprint name
            fp_match = re.search(r'\(footprint\s+"([^"]+)"', block)
            footprint = fp_match.group(1) if fp_match else ''

            # Extract layer
            layer_match = re.search(r'\(layer\s+"([^"]+)"\)', block)
            layer = layer_match.group(1) if layer_match else 'F.Cu'

            # Extract position
            at_match = re.search(r'\(at\s+([\d.-]+)\s+([\d.-]+)(?:\s+([\d.-]+))?\)', block)
            if at_match:
                x = float(at_match.group(1))
                y = float(at_match.group(2))
                rotation = float(at_match.group(3)) if at_match.group(3) else 0
            else:
                return None

            # Extract reference
            ref_match = re.search(r'\(fp_text\s+reference\s+"([^"]+)"', block)
            reference = ref_match.group(1) if ref_match else ''

            if not reference:
                # Try alternate format
                ref_match = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', block)
                reference = ref_match.group(1) if ref_match else ''

            if not reference:
                return None

            # Extract value
            val_match = re.search(r'\(fp_text\s+value\s+"([^"]+)"', block)
            value = val_match.group(1) if val_match else ''

            # Extract pads
            pads = []
            pad_pattern = r'\(pad\s+"?(\d+)"?\s+(\w+)\s+(\w+)[^)]*\(at\s+([\d.-]+)\s+([\d.-]+)[^)]*\)[^)]*(?:\(net\s+(\d+)\s+"([^"]+)"\))?'
            for pad_match in re.finditer(pad_pattern, block):
                pads.append({
                    'number': pad_match.group(1),
                    'type': pad_match.group(2),
                    'shape': pad_match.group(3),
                    'offset': (float(pad_match.group(4)), float(pad_match.group(5))),
                    'net_id': int(pad_match.group(6)) if pad_match.group(6) else 0,
                    'net_name': pad_match.group(7) if pad_match.group(7) else '',
                })

                # Record pad in net
                if pad_match.group(7) and pad_match.group(7) in self.nets:
                    self.nets[pad_match.group(7)].pads.append((reference, pad_match.group(1)))

            return ExtractedComponent(
                reference=reference,
                footprint=footprint,
                position=(x, y),
                rotation=rotation,
                layer=layer,
                pads=pads,
                value=value,
            )

        except Exception as e:
            return None

    def _parse_tracks(self, content: str):
        """Parse track segments"""
        # Match segment (track) definitions
        track_pattern = r'\(segment\s+\(start\s+([\d.-]+)\s+([\d.-]+)\)\s+\(end\s+([\d.-]+)\s+([\d.-]+)\)\s+\(width\s+([\d.-]+)\)\s+\(layer\s+"([^"]+)"\)\s+\(net\s+(\d+)\)'

        for match in re.finditer(track_pattern, content):
            track = {
                'start': (float(match.group(1)), float(match.group(2))),
                'end': (float(match.group(3)), float(match.group(4))),
                'width': float(match.group(5)),
                'layer': match.group(6),
                'net_id': int(match.group(7)),
            }
            self.tracks.append(track)

            # Find net name and add track
            for net_name, net in self.nets.items():
                if net.net_id == track['net_id']:
                    net.tracks.append(track)
                    break

    def _parse_vias(self, content: str):
        """Parse via definitions"""
        via_pattern = r'\(via\s+\(at\s+([\d.-]+)\s+([\d.-]+)\)\s+\(size\s+([\d.-]+)\)\s+\(drill\s+([\d.-]+)\)[^)]*\(net\s+(\d+)\)'

        for match in re.finditer(via_pattern, content):
            via = {
                'position': (float(match.group(1)), float(match.group(2))),
                'size': float(match.group(3)),
                'drill': float(match.group(4)),
                'net_id': int(match.group(5)),
            }
            self.vias.append(via)

            # Add to net
            for net_name, net in self.nets.items():
                if net.net_id == via['net_id']:
                    net.vias.append(via)
                    break

    def _parse_zones(self, content: str):
        """Parse zone definitions"""
        # Simplified zone parsing
        zone_pattern = r'\(zone\s+\(net\s+\d+\)\s+\(net_name\s+"([^"]+)"\)[^)]*\(layer\s+"([^"]+)"\)'

        for match in re.finditer(zone_pattern, content):
            zone = ExtractedZone(
                net_name=match.group(1),
                layer=match.group(2),
            )
            self.zones.append(zone)

    def _parse_board_outline(self, content: str):
        """Parse board outline from Edge.Cuts layer"""
        # Match line segments on Edge.Cuts
        edge_pattern = r'\(gr_line\s+\(start\s+([\d.-]+)\s+([\d.-]+)\)\s+\(end\s+([\d.-]+)\s+([\d.-]+)\)[^)]*\(layer\s+"Edge.Cuts"\)'

        for match in re.finditer(edge_pattern, content):
            self.board_outline.append({
                'start': (float(match.group(1)), float(match.group(2))),
                'end': (float(match.group(3)), float(match.group(4))),
            })

    def _calculate_board_size(self) -> Tuple[float, float]:
        """Calculate board size from outline"""
        if not self.board_outline:
            # Estimate from component positions
            if self.components:
                xs = [c.position[0] for c in self.components.values()]
                ys = [c.position[1] for c in self.components.values()]
                return (max(xs) - min(xs) + 20, max(ys) - min(ys) + 20)
            return (100, 100)

        all_x = []
        all_y = []
        for edge in self.board_outline:
            all_x.extend([edge['start'][0], edge['end'][0]])
            all_y.extend([edge['start'][1], edge['end'][1]])

        return (max(all_x) - min(all_x), max(all_y) - min(all_y))

    def _detect_layer_count(self, content: str) -> int:
        """Detect number of copper layers"""
        # Count copper layer definitions
        copper_layers = re.findall(r'\(\d+\s+"[FB]\.Cu"\s+signal\)', content)
        inner_layers = re.findall(r'\(\d+\s+"In\d+\.Cu"\s+signal\)', content)
        return len(copper_layers) + len(inner_layers)

    def _extract_sexp_block(self, content: str, start: int) -> str:
        """Extract a complete S-expression block starting at position"""
        depth = 0
        i = start
        while i < len(content):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
                if depth == 0:
                    return content[start:i+1]
            i += 1
        return ''


# =============================================================================
# PATTERN ANALYZER
# =============================================================================

class PatternAnalyzer:
    """
    Analyzes extracted PCB data to discover patterns.
    """

    def __init__(self):
        self.patterns = []

    def analyze(self, design: DesignAnalysis) -> DesignAnalysis:
        """Analyze a design and extract patterns"""

        # Analyze placement patterns
        design.placement_patterns = self._analyze_placement(design)

        # Analyze routing patterns
        design.routing_patterns = self._analyze_routing(design)

        # Extract design rules
        design.design_rules = self._extract_design_rules(design)

        return design

    def _analyze_placement(self, design: DesignAnalysis) -> List[Dict]:
        """Analyze component placement patterns"""
        patterns = []

        components = list(design.components.values())

        # Pattern 1: Find component clusters
        clusters = self._find_clusters(components)
        for cluster in clusters:
            patterns.append({
                'type': 'cluster',
                'components': [c.reference for c in cluster],
                'centroid': self._calculate_centroid(cluster),
                'radius': self._calculate_radius(cluster),
            })

        # Pattern 2: Identify hub component (most connected)
        hub = self._identify_hub(design)
        if hub:
            patterns.append({
                'type': 'hub',
                'component': hub.reference,
                'connection_count': self._count_connections(hub, design),
                'position': hub.position,
            })

        # Pattern 3: Analyze relative positions
        if hub:
            for comp in components:
                if comp.reference != hub.reference:
                    relative = self._analyze_relative_position(hub, comp, design)
                    if relative:
                        patterns.append(relative)

        # Pattern 4: Decoupling capacitor placement
        decoupling = self._find_decoupling_patterns(design)
        patterns.extend(decoupling)

        return patterns

    def _analyze_routing(self, design: DesignAnalysis) -> List[Dict]:
        """Analyze routing patterns"""
        patterns = []

        for net_name, net in design.nets.items():
            if not net.tracks:
                continue

            # Pattern 1: Track width by net type
            widths = [t['width'] for t in net.tracks]
            avg_width = sum(widths) / len(widths) if widths else 0

            net_type = self._classify_net(net_name)
            patterns.append({
                'type': 'track_width',
                'net_type': net_type,
                'avg_width': avg_width,
                'min_width': min(widths) if widths else 0,
                'max_width': max(widths) if widths else 0,
            })

            # Pattern 2: Via usage
            if net.vias:
                patterns.append({
                    'type': 'via_usage',
                    'net': net_name,
                    'via_count': len(net.vias),
                    'track_length': net.total_length,
                    'vias_per_cm': len(net.vias) / max(0.1, net.total_length / 10),
                })

            # Pattern 3: Routing topology
            if len(net.pads) >= 2:
                topology = self._analyze_topology(net)
                patterns.append({
                    'type': 'topology',
                    'net': net_name,
                    'pin_count': len(net.pads),
                    'topology': topology,
                })

        # Pattern 4: Layer usage
        layer_usage = self._analyze_layer_usage(design)
        patterns.append({
            'type': 'layer_usage',
            **layer_usage,
        })

        return patterns

    def _extract_design_rules(self, design: DesignAnalysis) -> Dict:
        """Extract implicit design rules from the design"""
        rules = {}

        # Track width range
        all_widths = []
        for net in design.nets.values():
            all_widths.extend([t['width'] for t in net.tracks])

        if all_widths:
            rules['min_track_width'] = min(all_widths)
            rules['max_track_width'] = max(all_widths)
            rules['common_track_width'] = self._most_common(all_widths)

        # Via sizes
        if design.nets:
            all_vias = []
            for net in design.nets.values():
                all_vias.extend(net.vias)

            if all_vias:
                rules['via_diameter'] = all_vias[0].get('size', 0.8)
                rules['via_drill'] = all_vias[0].get('drill', 0.4)

        # Clearances (estimated from track spacing)
        # This is complex - would need spatial analysis

        return rules

    def _find_clusters(self, components: List[ExtractedComponent],
                        max_distance: float = 15.0) -> List[List[ExtractedComponent]]:
        """Find component clusters using simple distance-based clustering"""
        if not components:
            return []

        clusters = []
        used = set()

        for comp in components:
            if comp.reference in used:
                continue

            # Start new cluster
            cluster = [comp]
            used.add(comp.reference)

            # Find nearby components
            for other in components:
                if other.reference in used:
                    continue

                dist = math.sqrt(
                    (comp.position[0] - other.position[0])**2 +
                    (comp.position[1] - other.position[1])**2
                )

                if dist < max_distance:
                    cluster.append(other)
                    used.add(other.reference)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _identify_hub(self, design: DesignAnalysis) -> Optional[ExtractedComponent]:
        """Identify the hub component (most connections)"""
        connection_counts = {}

        for comp_ref, comp in design.components.items():
            count = 0
            for pad in comp.pads:
                if pad.get('net_name') and pad['net_name'] != 'GND':
                    count += 1
            connection_counts[comp_ref] = count

        if connection_counts:
            hub_ref = max(connection_counts, key=connection_counts.get)
            return design.components.get(hub_ref)

        return None

    def _count_connections(self, comp: ExtractedComponent,
                            design: DesignAnalysis) -> int:
        """Count unique net connections for a component"""
        nets = set()
        for pad in comp.pads:
            if pad.get('net_name') and pad['net_name'] != 'GND':
                nets.add(pad['net_name'])
        return len(nets)

    def _analyze_relative_position(self, hub: ExtractedComponent,
                                     other: ExtractedComponent,
                                     design: DesignAnalysis) -> Optional[Dict]:
        """Analyze position of component relative to hub"""
        dx = other.position[0] - hub.position[0]
        dy = other.position[1] - hub.position[1]
        distance = math.sqrt(dx*dx + dy*dy)

        # Determine direction
        angle = math.degrees(math.atan2(dy, dx))
        direction = self._angle_to_direction(angle)

        # Check if connected
        connected_nets = set()
        for pad in other.pads:
            net_name = pad.get('net_name', '')
            if net_name and net_name != 'GND':
                # Check if hub has same net
                for hub_pad in hub.pads:
                    if hub_pad.get('net_name') == net_name:
                        connected_nets.add(net_name)

        if connected_nets:
            return {
                'type': 'relative_position',
                'component': other.reference,
                'relative_to': hub.reference,
                'direction': direction,
                'distance': distance,
                'connected_via': list(connected_nets),
                'lesson': f"Component {other.reference} placed {direction} of hub at {distance:.1f}mm"
            }

        return None

    def _find_decoupling_patterns(self, design: DesignAnalysis) -> List[Dict]:
        """Find decoupling capacitor placement patterns"""
        patterns = []

        # Find capacitors (reference starts with C)
        caps = [c for c in design.components.values()
                if c.reference.startswith('C')]

        # Find ICs (reference starts with U)
        ics = [c for c in design.components.values()
               if c.reference.startswith('U')]

        for cap in caps:
            # Check if it's a decoupling cap (connected to power and GND)
            has_power = False
            has_gnd = False
            for pad in cap.pads:
                net = pad.get('net_name', '')
                if 'VCC' in net.upper() or 'VDD' in net.upper() or '3V3' in net or '5V' in net:
                    has_power = True
                if net == 'GND':
                    has_gnd = True

            if has_power and has_gnd:
                # Find nearest IC
                nearest_ic = None
                min_dist = float('inf')

                for ic in ics:
                    dist = math.sqrt(
                        (cap.position[0] - ic.position[0])**2 +
                        (cap.position[1] - ic.position[1])**2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ic = ic

                if nearest_ic and min_dist < 10:  # Within 10mm
                    patterns.append({
                        'type': 'decoupling_placement',
                        'capacitor': cap.reference,
                        'ic': nearest_ic.reference,
                        'distance': min_dist,
                        'lesson': f"Decoupling cap {cap.reference} placed {min_dist:.1f}mm from IC {nearest_ic.reference}"
                    })

        return patterns

    def _classify_net(self, net_name: str) -> str:
        """Classify a net by its name"""
        name_upper = net_name.upper()

        if 'GND' in name_upper:
            return 'ground'
        elif any(p in name_upper for p in ['VCC', 'VDD', '3V3', '5V', '12V', 'PWR', 'POWER']):
            return 'power'
        elif any(s in name_upper for s in ['CLK', 'CLOCK', 'SCK', 'SCLK']):
            return 'clock'
        elif any(s in name_upper for s in ['SDA', 'SCL', 'MOSI', 'MISO', 'TX', 'RX']):
            return 'signal_critical'
        else:
            return 'signal'

    def _analyze_topology(self, net: ExtractedNet) -> str:
        """Analyze the routing topology of a net"""
        if len(net.pads) == 2:
            return 'point_to_point'
        elif len(net.pads) <= 4:
            # Check if star or daisy-chain
            # Simplified: if there's a central point, it's star
            return 'star'  # Simplified
        else:
            return 'complex'

    def _analyze_layer_usage(self, design: DesignAnalysis) -> Dict:
        """Analyze which layers are used for what"""
        layer_tracks = {}
        layer_zones = {}

        for net in design.nets.values():
            for track in net.tracks:
                layer = track['layer']
                if layer not in layer_tracks:
                    layer_tracks[layer] = 0
                layer_tracks[layer] += 1

        for zone in design.zones:
            if zone.layer not in layer_zones:
                layer_zones[zone.layer] = []
            layer_zones[zone.layer].append(zone.net_name)

        return {
            'track_counts': layer_tracks,
            'zones': layer_zones,
        }

    def _calculate_centroid(self, components: List[ExtractedComponent]) -> Tuple[float, float]:
        """Calculate centroid of component positions"""
        if not components:
            return (0, 0)
        x = sum(c.position[0] for c in components) / len(components)
        y = sum(c.position[1] for c in components) / len(components)
        return (x, y)

    def _calculate_radius(self, components: List[ExtractedComponent]) -> float:
        """Calculate radius of component cluster"""
        if len(components) < 2:
            return 0
        centroid = self._calculate_centroid(components)
        max_dist = 0
        for c in components:
            dist = math.sqrt(
                (c.position[0] - centroid[0])**2 +
                (c.position[1] - centroid[1])**2
            )
            max_dist = max(max_dist, dist)
        return max_dist

    def _angle_to_direction(self, angle: float) -> str:
        """Convert angle to cardinal direction"""
        # Normalize angle to 0-360
        angle = angle % 360
        if angle < 0:
            angle += 360

        if angle < 22.5 or angle >= 337.5:
            return 'E'
        elif angle < 67.5:
            return 'NE'
        elif angle < 112.5:
            return 'N'
        elif angle < 157.5:
            return 'NW'
        elif angle < 202.5:
            return 'W'
        elif angle < 247.5:
            return 'SW'
        elif angle < 292.5:
            return 'S'
        else:
            return 'SE'

    def _most_common(self, values: List[float], tolerance: float = 0.01) -> float:
        """Find most common value with tolerance"""
        if not values:
            return 0

        # Group by rounded value
        groups = {}
        for v in values:
            rounded = round(v / tolerance) * tolerance
            if rounded not in groups:
                groups[rounded] = 0
            groups[rounded] += 1

        return max(groups, key=groups.get)


# =============================================================================
# TRAINING MODULE
# =============================================================================

class TrainingModule:
    """
    Main training module for the PCB Engine.

    Coordinates parsing, analysis, and knowledge extraction.
    """

    def __init__(self, knowledge_base):
        """
        Initialize with reference to knowledge base.

        Args:
            knowledge_base: The engine's KnowledgeBase instance
        """
        self.knowledge = knowledge_base
        self.parser = KiCadPCBParser()
        self.analyzer = PatternAnalyzer()
        self.trained_designs = []

    def train_from_pcb(self, pcb_path: str,
                        notes_path: str = None) -> DesignAnalysis:
        """
        Train from a KiCad PCB file.

        Args:
            pcb_path: Path to .kicad_pcb file
            notes_path: Optional path to design notes JSON

        Returns:
            DesignAnalysis with extracted patterns
        """
        print(f"\n{'='*60}")
        print(f"TRAINING FROM: {os.path.basename(pcb_path)}")
        print('='*60)

        # Parse the PCB
        print("Parsing PCB file...")
        design = self.parser.parse(pcb_path)

        print(f"  Components: {design.component_count}")
        print(f"  Nets: {design.net_count}")
        print(f"  Tracks: {design.track_count}")
        print(f"  Vias: {design.via_count}")

        # Analyze patterns
        print("\nAnalyzing patterns...")
        design = self.analyzer.analyze(design)

        print(f"  Placement patterns: {len(design.placement_patterns)}")
        print(f"  Routing patterns: {len(design.routing_patterns)}")

        # Load designer notes if provided
        if notes_path and os.path.exists(notes_path):
            with open(notes_path, 'r') as f:
                notes = json.load(f)
                self._apply_designer_notes(design, notes)

        # Convert patterns to lessons
        print("\nExtracting lessons...")
        lessons = self._patterns_to_lessons(design)

        for lesson in lessons:
            self.knowledge.add_lesson(lesson)
            print(f"  + {lesson.category}: {lesson.solution[:50]}...")

        self.trained_designs.append(design)

        print(f"\n✓ Training complete. {len(lessons)} lessons learned.")

        return design

    def train_from_directory(self, directory: str) -> List[DesignAnalysis]:
        """Train from all PCB files in a directory"""
        designs = []

        for filename in os.listdir(directory):
            if filename.endswith('.kicad_pcb'):
                pcb_path = os.path.join(directory, filename)
                notes_path = os.path.join(directory, filename.replace('.kicad_pcb', '_notes.json'))

                try:
                    design = self.train_from_pcb(pcb_path, notes_path)
                    designs.append(design)
                except Exception as e:
                    print(f"Error training from {filename}: {e}")

        # Cross-validate patterns across designs
        if len(designs) > 1:
            self._cross_validate_patterns(designs)

        return designs

    def _patterns_to_lessons(self, design: DesignAnalysis) -> List:
        """Convert extracted patterns to lessons"""
        from .supervisor import Lesson

        lessons = []
        timestamp = datetime.now().isoformat()

        # Placement patterns
        for pattern in design.placement_patterns:
            if pattern['type'] == 'hub':
                lessons.append(Lesson(
                    category='placement',
                    situation=f"Design with hub component {pattern['component']}",
                    problem='Hub identification',
                    solution=f"Component with {pattern['connection_count']} connections is the hub",
                    confidence=0.8,
                    timestamp=timestamp,
                    source='training',
                ))

            elif pattern['type'] == 'relative_position':
                lessons.append(Lesson(
                    category='placement',
                    situation=f"Positioning {pattern['component']} relative to hub",
                    problem='Component placement direction',
                    solution=pattern.get('lesson', f"Place {pattern['direction']} at {pattern['distance']:.1f}mm"),
                    confidence=0.7,
                    timestamp=timestamp,
                    source='training',
                ))

            elif pattern['type'] == 'decoupling_placement':
                lessons.append(Lesson(
                    category='placement',
                    situation='Decoupling capacitor placement',
                    problem='Where to place decoupling caps',
                    solution=pattern.get('lesson', f"Place within {pattern['distance']:.1f}mm of IC"),
                    confidence=0.9,
                    timestamp=timestamp,
                    source='training',
                ))

        # Routing patterns
        for pattern in design.routing_patterns:
            if pattern['type'] == 'track_width':
                lessons.append(Lesson(
                    category='routing',
                    situation=f"Routing {pattern['net_type']} nets",
                    problem='Track width selection',
                    solution=f"Use {pattern['avg_width']:.2f}mm width for {pattern['net_type']} nets",
                    confidence=0.75,
                    timestamp=timestamp,
                    source='training',
                ))

            elif pattern['type'] == 'layer_usage':
                track_counts = pattern.get('track_counts', {})
                zones = pattern.get('zones', {})

                if 'B.Cu' in zones and 'GND' in zones.get('B.Cu', []):
                    lessons.append(Lesson(
                        category='routing',
                        situation='Layer usage for GND',
                        problem='Which layer for GND zone',
                        solution='Use bottom copper layer (B.Cu) for GND pour',
                        confidence=0.85,
                        timestamp=timestamp,
                        source='training',
                    ))

        # Design rules
        rules = design.design_rules
        if rules.get('min_track_width'):
            lessons.append(Lesson(
                category='design_rules',
                situation='Track width minimum',
                problem='What minimum track width to use',
                solution=f"Minimum track width: {rules['min_track_width']:.2f}mm",
                confidence=0.9,
                timestamp=timestamp,
                source='training',
            ))

        return lessons

    def _apply_designer_notes(self, design: DesignAnalysis, notes: Dict):
        """Apply designer's annotations to improve learning"""
        if 'lessons' in notes:
            for lesson_data in notes['lessons']:
                from .supervisor import Lesson
                lesson = Lesson(
                    category=lesson_data.get('category', 'general'),
                    situation=lesson_data.get('situation', 'Designer note'),
                    problem=lesson_data.get('problem', ''),
                    solution=lesson_data.get('solution', ''),
                    confidence=0.95,  # High confidence for explicit notes
                    timestamp=datetime.now().isoformat(),
                    source='designer',
                )
                self.knowledge.add_lesson(lesson)

    def _cross_validate_patterns(self, designs: List[DesignAnalysis]):
        """Validate patterns across multiple designs to increase confidence"""
        print("\nCross-validating patterns across designs...")

        # Find common patterns
        all_lessons = list(self.knowledge.lessons)

        for lesson in all_lessons:
            # Count how many designs support this lesson
            support_count = 0
            for design in designs:
                if self._design_supports_lesson(design, lesson):
                    support_count += 1

            # Increase confidence based on support
            if support_count > 1:
                old_confidence = lesson.confidence
                lesson.confidence = min(1.0, lesson.confidence + 0.1 * (support_count - 1))
                if lesson.confidence > old_confidence:
                    print(f"  ↑ Confidence: {lesson.solution[:40]}... ({old_confidence:.2f} → {lesson.confidence:.2f})")

        self.knowledge.save()

    def _design_supports_lesson(self, design: DesignAnalysis, lesson) -> bool:
        """Check if a design supports a particular lesson"""
        # Simple text matching for now
        lesson_text = f"{lesson.situation} {lesson.problem} {lesson.solution}".lower()

        for pattern in design.placement_patterns + design.routing_patterns:
            pattern_text = str(pattern).lower()
            if any(word in pattern_text for word in lesson_text.split()[:3]):
                return True

        return False

    def get_training_summary(self) -> str:
        """Get summary of training"""
        lines = [
            "=" * 60,
            "TRAINING SUMMARY",
            "=" * 60,
            f"Designs trained: {len(self.trained_designs)}",
            "",
        ]

        for design in self.trained_designs:
            lines.append(f"  • {design.filename}")
            lines.append(f"    Components: {design.component_count}, Nets: {design.net_count}")
            lines.append(f"    Patterns: {len(design.placement_patterns)} placement, {len(design.routing_patterns)} routing")

        lines.extend([
            "",
            self.knowledge.get_summary(),
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def train_engine(engine, pcb_files: List[str], notes_files: List[str] = None):
    """
    Convenience function to train an engine from PCB files.

    Args:
        engine: PCBEngine instance
        pcb_files: List of .kicad_pcb file paths
        notes_files: Optional list of corresponding notes JSON files

    Returns:
        TrainingModule with trained data
    """
    from .supervisor import KnowledgeBase

    # Get or create knowledge base
    if hasattr(engine, '_supervisor') and engine._supervisor:
        knowledge = engine._supervisor.knowledge
    else:
        knowledge = KnowledgeBase()

    trainer = TrainingModule(knowledge)

    notes_files = notes_files or [None] * len(pcb_files)

    for pcb_path, notes_path in zip(pcb_files, notes_files):
        try:
            trainer.train_from_pcb(pcb_path, notes_path)
        except Exception as e:
            print(f"Error training from {pcb_path}: {e}")

    print(trainer.get_training_summary())

    return trainer
