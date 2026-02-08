#!/usr/bin/env python3
"""
PourPiston - Copper Pour / Ground Plane Generator

Generates copper fills (pours) for power and ground planes.
This eliminates the need to route GND traces and prevents crossing issues.

Features:
- Ground pour on bottom layer (default)
- Power pour support
- Thermal relief for pads
- Clearance from other nets
- Zone priority handling

Research:
- KiCad zone fill algorithm
- Polygon clipping (Clipper library concepts)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import math


class PourType(Enum):
    """Type of copper pour"""
    SOLID = "solid"           # Solid fill
    HATCHED = "hatched"       # Cross-hatch pattern (reduces warping)
    THERMAL = "thermal"       # Thermal spokes only (for high-current)


class ThermalReliefStyle(Enum):
    """Thermal relief connection style"""
    SOLID = "solid"           # Direct connection (high current)
    THERMAL = "thermal"       # Thermal spokes (soldering ease)
    NONE = "none"             # No connection (isolation)


@dataclass
class StitchingVia:
    """Represents a stitching via connecting pour to top layer"""
    x: float
    y: float
    drill: float = 0.3
    size: float = 0.6
    net: str = "GND"


@dataclass
class PourConfig:
    """Configuration for copper pour generation"""
    # Target net for pour
    net: str = "GND"

    # Layer(s) for pour
    layer: str = "B.Cu"  # Bottom copper by default

    # Pour type
    pour_type: PourType = PourType.SOLID

    # Clearance from other nets (mm)
    clearance: float = 0.3

    # Minimum width of copper (mm)
    min_width: float = 0.2

    # Thermal relief settings
    thermal_relief: ThermalReliefStyle = ThermalReliefStyle.THERMAL
    thermal_spoke_width: float = 0.5
    thermal_gap: float = 0.5

    # Zone priority (higher = fills first)
    priority: int = 0

    # Edge clearance from board outline (mm)
    edge_clearance: float = 0.3

    # Hatch settings (if hatched)
    hatch_width: float = 0.5
    hatch_gap: float = 0.5
    hatch_orientation: float = 45.0  # degrees

    # Connect pads with thermal relief
    connect_pads: bool = True

    # Island removal - remove isolated copper areas smaller than this (mm²)
    min_island_area: float = 1.0

    # GND stitching vias - connect pour to top layer
    add_stitching_vias: bool = True
    stitching_via_spacing: float = 10.0  # mm between vias (grid spacing)
    stitching_via_drill: float = 0.3  # via drill size
    stitching_via_size: float = 0.6  # via pad size
    stitching_via_clearance: float = 1.0  # min distance from pads/tracks


@dataclass
class PourZone:
    """Represents a copper pour zone"""
    net: str
    layer: str
    outline: List[Tuple[float, float]]  # Polygon vertices
    clearance: float
    priority: int
    thermal_relief: ThermalReliefStyle
    thermal_spoke_width: float
    thermal_gap: float
    hatch_edge: float = 0.0  # 0 = solid, >0 = hatched
    hatch_gap: float = 0.0
    min_thickness: float = 0.0
    filled: bool = False
    fill_polygons: List[List[Tuple[float, float]]] = field(default_factory=list)


@dataclass
class PourResult:
    """Result from pour generation"""
    success: bool
    zones: List[PourZone]
    connected_pads: List[Tuple[str, str]]  # (ref, pin) pairs connected via pour
    stitching_vias: List[StitchingVia] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PourPiston:
    """
    Generates copper pours (ground/power planes) for PCB designs.

    Using ground pours eliminates the need to route GND traces,
    which prevents track crossing issues on single/double layer boards.
    """

    def __init__(self, config: Optional[PourConfig] = None):
        self.config = config or PourConfig()
        self.zones: List[PourZone] = []
        self.stitching_vias: List[StitchingVia] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def generate(self,
                 parts_db: Dict,
                 placement: Dict,
                 board_width: float,
                 board_height: float,
                 routes: Optional[Dict] = None,
                 vias: Optional[List] = None,
                 existing_pours: Optional[List[PourZone]] = None) -> PourResult:
        """
        Generate copper pour zones.

        Args:
            parts_db: Component database with nets
            placement: Component placement positions
            board_width: Board width in mm
            board_height: Board height in mm
            routes: Existing routes (to avoid)
            vias: Existing vias (to connect or avoid)
            existing_pours: Higher priority pours already generated

        Returns:
            PourResult with generated zones
        """
        self.zones = []
        self.stitching_vias = []
        self.warnings = []
        self.errors = []

        # Step 1: Create board outline zone
        outline = self._create_board_outline(board_width, board_height)

        # Step 2: Find all pads that connect to pour net
        connected_pads = self._find_connected_pads(parts_db, placement)

        # Step 3: Create keepout areas for other nets
        keepouts = self._create_keepouts(parts_db, placement, routes, vias)

        # Step 4: Generate the pour zone
        zone = PourZone(
            net=self.config.net,
            layer=self.config.layer,
            outline=outline,
            clearance=self.config.clearance,
            priority=self.config.priority,
            thermal_relief=self.config.thermal_relief,
            thermal_spoke_width=self.config.thermal_spoke_width,
            thermal_gap=self.config.thermal_gap,
            min_thickness=self.config.min_width
        )

        # Step 5: Calculate fill polygons (simplified - full board minus keepouts)
        zone.fill_polygons = self._calculate_fill(outline, keepouts)
        zone.filled = len(zone.fill_polygons) > 0

        self.zones.append(zone)

        # Step 6: Generate stitching vias to connect pour to top layer
        if self.config.add_stitching_vias:
            self.stitching_vias = self._generate_stitching_vias(
                board_width, board_height, parts_db, placement, routes, vias, keepouts
            )

        return PourResult(
            success=len(self.errors) == 0,
            zones=self.zones,
            connected_pads=connected_pads,
            stitching_vias=self.stitching_vias,
            warnings=self.warnings,
            errors=self.errors
        )

    def _create_board_outline(self, width: float, height: float) -> List[Tuple[float, float]]:
        """Create board outline polygon with edge clearance"""
        ec = self.config.edge_clearance
        return [
            (ec, ec),
            (width - ec, ec),
            (width - ec, height - ec),
            (ec, height - ec),
            (ec, ec)  # Close polygon
        ]

    def _find_connected_pads(self, parts_db: Dict, placement: Dict) -> List[Tuple[str, str]]:
        """Find all pads that should connect to the pour net"""
        connected = []
        target_net = self.config.net

        parts = parts_db.get('parts', {})
        for ref, part in parts.items():
            if ref not in placement:
                continue
            for pin in part.get('pins', []):
                if pin.get('net') == target_net:
                    connected.append((ref, pin.get('number', '?')))

        return connected

    def _create_keepouts(self,
                         parts_db: Dict,
                         placement: Dict,
                         routes: Optional[Dict],
                         vias: Optional[List]) -> List[List[Tuple[float, float]]]:
        """Create keepout polygons for elements not on pour net"""
        keepouts = []
        target_net = self.config.net
        clearance = self.config.clearance

        parts = parts_db.get('parts', {})

        # Keepouts for pads on other nets
        for ref, part in parts.items():
            if ref not in placement:
                continue
            cx, cy = placement[ref][:2] if isinstance(placement[ref], (list, tuple)) else (placement[ref], placement[ref])
            if isinstance(placement[ref], (list, tuple)) and len(placement[ref]) >= 2:
                cx, cy = placement[ref][0], placement[ref][1]
            else:
                continue

            for pin in part.get('pins', []):
                net = pin.get('net', '')
                if net == target_net or net == 'NC':
                    continue  # Don't keep out our own net or NC

                # Get pad position and size
                offset = pin.get('offset', (0, 0))
                size = pin.get('size', (1.0, 1.0))

                px = cx + offset[0]
                py = cy + offset[1]
                hw = size[0] / 2 + clearance
                hh = size[1] / 2 + clearance

                # Create rectangular keepout
                keepouts.append([
                    (px - hw, py - hh),
                    (px + hw, py - hh),
                    (px + hw, py + hh),
                    (px - hw, py + hh),
                    (px - hw, py - hh)
                ])

        # Keepouts for tracks on other nets (on pour layer)
        if routes:
            for net_name, route in routes.items():
                if net_name == target_net:
                    continue
                segments = getattr(route, 'segments', [])
                for seg in segments:
                    layer = getattr(seg, 'layer', 'F.Cu')
                    if layer != self.config.layer:
                        continue  # Different layer, no keepout needed

                    start = getattr(seg, 'start', (0, 0))
                    end = getattr(seg, 'end', (0, 0))
                    width = getattr(seg, 'width', 0.25)

                    # Create track keepout (expanded rectangle along track)
                    keepout = self._track_to_keepout(start, end, width + clearance * 2)
                    if keepout:
                        keepouts.append(keepout)

        return keepouts

    def _track_to_keepout(self,
                          start: Tuple[float, float],
                          end: Tuple[float, float],
                          width: float) -> Optional[List[Tuple[float, float]]]:
        """Convert a track segment to a keepout polygon"""
        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)

        if length < 0.001:
            return None

        # Perpendicular unit vector
        px = -dy / length
        py = dx / length

        hw = width / 2

        # Four corners of the track rectangle
        return [
            (x1 + px * hw, y1 + py * hw),
            (x2 + px * hw, y2 + py * hw),
            (x2 - px * hw, y2 - py * hw),
            (x1 - px * hw, y1 - py * hw),
            (x1 + px * hw, y1 + py * hw)
        ]

    def _calculate_fill(self,
                        outline: List[Tuple[float, float]],
                        keepouts: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """
        Calculate fill polygons by subtracting keepouts from outline.

        This is a simplified implementation. A full implementation would use
        polygon clipping algorithms (Clipper, CGAL, etc.)
        """
        # For now, return the outline as a single fill polygon
        # The keepouts will be handled by KiCad's zone fill algorithm
        # when we export - we just define the zone boundary and let KiCad fill
        return [outline]

    def _generate_stitching_vias(self,
                                  board_width: float,
                                  board_height: float,
                                  parts_db: Dict,
                                  placement: Dict,
                                  routes: Optional[Dict],
                                  vias: Optional[List],
                                  keepouts: List[List[Tuple[float, float]]]) -> List[StitchingVia]:
        """
        Generate stitching vias to connect pour to top layer GND net.

        This eliminates "Isolated copper fill" warnings by ensuring the bottom
        layer pour is connected to the top layer GND traces.

        Placement rules (DRC-safe):
        1. Grid-based placement at regular intervals
        2. Clear of all pads (same net or different)
        3. Clear of all tracks
        4. Clear of existing vias
        5. Clear of board edges
        6. Near GND pads/traces for effective connection

        Args:
            board_width: Board width in mm
            board_height: Board height in mm
            parts_db: Component database
            placement: Component placements
            routes: Existing routes
            vias: Existing vias
            keepouts: Keepout polygons

        Returns:
            List of StitchingVia objects
        """
        stitching_vias = []
        target_net = self.config.net
        spacing = self.config.stitching_via_spacing
        via_clearance = self.config.stitching_via_clearance
        via_size = self.config.stitching_via_size
        edge_clearance = self.config.edge_clearance + via_size / 2

        # Collect all obstacles (pads, tracks, vias) with their clearance zones
        obstacles = []  # List of (x, y, radius) tuples

        # Add pad obstacles
        parts = parts_db.get('parts', {})
        for ref, part in parts.items():
            if ref not in placement:
                continue

            pos = placement[ref]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                cx, cy = pos[0], pos[1]
            else:
                continue

            for pin in part.get('pins', []):
                offset = pin.get('offset', (0, 0))
                size = pin.get('size', (1.0, 1.0))

                px = cx + offset[0]
                py = cy + offset[1]
                # Clearance radius = half diagonal + clearance
                pad_radius = math.sqrt((size[0]/2)**2 + (size[1]/2)**2) + via_clearance
                obstacles.append((px, py, pad_radius))

        # Add track obstacles
        if routes:
            for net_name, route in routes.items():
                segments = getattr(route, 'segments', [])
                for seg in segments:
                    start = getattr(seg, 'start', (0, 0))
                    end = getattr(seg, 'end', (0, 0))
                    width = getattr(seg, 'width', 0.25)

                    # Sample points along track
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = math.sqrt(dx*dx + dy*dy)

                    if length < 0.001:
                        obstacles.append((start[0], start[1], width/2 + via_clearance))
                    else:
                        # Add points along track
                        num_points = max(2, int(length / 1.0))  # Every 1mm
                        for i in range(num_points + 1):
                            t = i / num_points
                            tx = start[0] + dx * t
                            ty = start[1] + dy * t
                            obstacles.append((tx, ty, width/2 + via_clearance))

        # Add existing via obstacles
        if vias:
            for via in vias:
                vx = getattr(via, 'x', 0)
                vy = getattr(via, 'y', 0)
                vsize = getattr(via, 'size', 0.6)
                obstacles.append((vx, vy, vsize/2 + via_clearance))

        # Find GND pad locations (prefer placing vias near them)
        gnd_locations = []
        for ref, part in parts.items():
            if ref not in placement:
                continue

            pos = placement[ref]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                cx, cy = pos[0], pos[1]
            else:
                continue

            for pin in part.get('pins', []):
                if pin.get('net') == target_net:
                    offset = pin.get('offset', (0, 0))
                    gnd_locations.append((cx + offset[0], cy + offset[1]))

        # Generate grid of candidate positions
        x_start = edge_clearance
        y_start = edge_clearance
        x_end = board_width - edge_clearance
        y_end = board_height - edge_clearance

        # Generate candidates at grid positions
        candidates = []
        x = x_start
        while x <= x_end:
            y = y_start
            while y <= y_end:
                candidates.append((x, y))
                y += spacing
            x += spacing

        # Also add candidates near GND pads (offset to avoid pads)
        for gx, gy in gnd_locations:
            offsets = [
                (via_clearance + via_size, 0),
                (-via_clearance - via_size, 0),
                (0, via_clearance + via_size),
                (0, -via_clearance - via_size),
            ]
            for ox, oy in offsets:
                nx, ny = gx + ox, gy + oy
                if edge_clearance <= nx <= board_width - edge_clearance:
                    if edge_clearance <= ny <= board_height - edge_clearance:
                        candidates.append((nx, ny))

        # Filter candidates that don't violate any obstacles
        valid_vias = []
        for cx, cy in candidates:
            is_clear = True

            # Check against all obstacles
            for ox, oy, radius in obstacles:
                dist = math.sqrt((cx - ox)**2 + (cy - oy)**2)
                if dist < radius + via_size / 2:
                    is_clear = False
                    break

            if is_clear:
                # Check not too close to board edge
                if cx < edge_clearance or cx > board_width - edge_clearance:
                    is_clear = False
                if cy < edge_clearance or cy > board_height - edge_clearance:
                    is_clear = False

            if is_clear:
                # Check not inside any keepout polygon
                for keepout in keepouts:
                    if self._point_in_polygon(cx, cy, keepout):
                        is_clear = False
                        break

            if is_clear:
                valid_vias.append((cx, cy))

        # Remove duplicates (within via_size distance)
        final_vias = []
        for vx, vy in valid_vias:
            is_duplicate = False
            for fx, fy in final_vias:
                if math.sqrt((vx - fx)**2 + (vy - fy)**2) < via_size:
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_vias.append((vx, vy))

        # Create StitchingVia objects
        for vx, vy in final_vias:
            stitching_vias.append(StitchingVia(
                x=vx,
                y=vy,
                drill=self.config.stitching_via_drill,
                size=self.config.stitching_via_size,
                net=self.config.net
            ))

        if stitching_vias:
            self.warnings.append(f"Added {len(stitching_vias)} GND stitching vias")
        else:
            self.warnings.append("No valid positions found for stitching vias")

        return stitching_vias

    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 0.0001) + xi):
                inside = not inside

            j = i

        return inside

    def to_kicad_zone(self, zone: PourZone, net_number: int = 0) -> str:
        """
        Convert a PourZone to KiCad zone format.

        Args:
            zone: The zone to convert
            net_number: Net number in KiCad netlist

        Returns:
            KiCad zone S-expression string
        """
        # Build outline points
        pts = "\n      ".join([f"(xy {x:.4f} {y:.4f})" for x, y in zone.outline[:-1]])

        # Thermal relief settings
        if zone.thermal_relief == ThermalReliefStyle.SOLID:
            connect_pads = "(connect_pads yes)"
        elif zone.thermal_relief == ThermalReliefStyle.NONE:
            connect_pads = "(connect_pads no)"
        else:
            connect_pads = f"(connect_pads thru_hole_only (clearance {zone.thermal_gap:.4f}))"

        # Hatch mode
        if zone.hatch_edge > 0:
            hatch_mode = f"(hatch edge {zone.hatch_edge:.4f})"
        else:
            hatch_mode = "(hatch edge 0.5)"  # Default edge hatch for visibility

        kicad_zone = f"""  (zone
    (net {net_number})
    (net_name "{zone.net}")
    (layer "{zone.layer}")
    (uuid "{self._generate_uuid()}")
    {hatch_mode}
    (priority {zone.priority})
    {connect_pads}
    (min_thickness {zone.min_thickness:.4f})
    (filled_areas_thickness no)
    (fill yes
      (thermal_gap {zone.thermal_gap:.4f})
      (thermal_bridge_width {zone.thermal_spoke_width:.4f})
    )
    (polygon
      (pts
      {pts}
      )
    )
  )"""
        return kicad_zone

    def _generate_uuid(self) -> str:
        """Generate a UUID for KiCad"""
        import uuid
        return str(uuid.uuid4())

    def to_kicad_stitching_vias(self, stitching_vias: List[StitchingVia], net_number: int = 0) -> List[str]:
        """
        Convert stitching vias to KiCad via format.

        Args:
            stitching_vias: List of StitchingVia objects
            net_number: Net number in KiCad netlist

        Returns:
            List of KiCad via S-expression strings
        """
        kicad_vias = []
        for via in stitching_vias:
            kicad_via = f"""  (via
    (at {via.x:.4f} {via.y:.4f})
    (size {via.size:.4f})
    (drill {via.drill:.4f})
    (layers "F.Cu" "B.Cu")
    (net {net_number})
    (uuid "{self._generate_uuid()}")
  )"""
            kicad_vias.append(kicad_via)
        return kicad_vias


# Convenience function for quick ground pour
def create_ground_pour(board_width: float,
                       board_height: float,
                       layer: str = "B.Cu",
                       clearance: float = 0.3) -> PourConfig:
    """Create a standard ground pour configuration"""
    return PourConfig(
        net="GND",
        layer=layer,
        clearance=clearance,
        thermal_relief=ThermalReliefStyle.THERMAL,
        thermal_spoke_width=0.5,
        thermal_gap=0.5,
        priority=0
    )


def create_power_pour(net: str,
                      board_width: float,
                      board_height: float,
                      layer: str = "F.Cu",
                      clearance: float = 0.3) -> PourConfig:
    """Create a power pour configuration"""
    return PourConfig(
        net=net,
        layer=layer,
        clearance=clearance,
        thermal_relief=ThermalReliefStyle.SOLID,  # Direct connection for power
        thermal_spoke_width=0.8,
        thermal_gap=0.3,
        priority=1  # Higher priority than GND
    )
