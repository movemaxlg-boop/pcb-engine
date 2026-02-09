# Advanced Routing Features Implementation Plan

**Date:** 2026-02-09
**Author:** Claude (PCB Engine Development)
**Status:** PLANNING

---

## Overview

Four features to implement, ordered by priority and dependency:

| # | Feature | Priority | Effort | Dependencies |
|---|---------|----------|--------|--------------|
| 1 | Trunk Chain Detection | MEDIUM | Easy | None |
| 2 | Net Class Constraints | MEDIUM | Medium | None |
| 3 | Push-and-Shove | HIGH | Hard | Net Class Constraints |
| 4 | Return Path Awareness | LOW | Hard | Push-and-Shove |

**Implementation Order:** 1 → 2 → 3 → 4 (build foundation first)

---

## Phase 1: Trunk Chain Detection (Week 1)

### Goal
Detect logical flow paths in power nets (connector → fuse → diode → regulator) and route them in that order, not MST order.

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pcb_engine/graph_analyzer.py` | CREATE | Graph analysis utilities |
| `pcb_engine/intelligent_router.py` | MODIFY | Use trunk chain for power routing |
| `pcb_engine/routing_piston.py` | MODIFY | Integrate trunk detection |

### Implementation Steps

#### Step 1.1: Create Graph Analyzer Module
```python
# pcb_engine/graph_analyzer.py

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

@dataclass
class PowerFlowGraph:
    """Represents power flow through a circuit."""
    nodes: Set[str]  # Component references
    edges: Dict[str, Set[str]]  # Adjacency list
    source: str  # Power input (e.g., J1)
    sink: str  # Power consumer (e.g., U1)
    trunk_path: List[str]  # Ordered path from source to sink

def build_connectivity_graph(parts_db: Dict) -> Dict[str, Set[str]]:
    """
    Build part-level undirected graph.
    Parts are nodes, edges exist if they share a net.
    """
    net_to_parts = defaultdict(set)
    for net_name, net_info in parts_db['nets'].items():
        for ref, pin in net_info['pins']:
            net_to_parts[net_name].add(ref)

    graph = defaultdict(set)
    for parts in net_to_parts.values():
        parts_list = list(parts)
        for i in range(len(parts_list)):
            for j in range(i + 1, len(parts_list)):
                a, b = parts_list[i], parts_list[j]
                graph[a].add(b)
                graph[b].add(a)
    return graph

def find_trunk_chain(parts_db: Dict, start_ref: str, end_ref: str) -> List[str]:
    """
    Find power trunk path using BFS shortest path.
    Example: J1 -> F1 -> D1 -> U1
    """
    graph = build_connectivity_graph(parts_db)

    queue = deque([start_ref])
    prev = {start_ref: None}

    while queue:
        current = queue.popleft()
        if current == end_ref:
            break
        for neighbor in graph[current]:
            if neighbor not in prev:
                prev[neighbor] = current
                queue.append(neighbor)

    if end_ref not in prev:
        return [start_ref, end_ref]  # Fallback: direct connection

    # Reconstruct path
    path = []
    current = end_ref
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path

def detect_power_source(parts_db: Dict) -> str:
    """Auto-detect power input connector."""
    for ref, part in parts_db['parts'].items():
        # Connectors typically start with J and have VIN/VCC on pin 1
        if ref.startswith('J'):
            for pin in part['pins']:
                if pin['net'] in ('VIN', 'VCC', '+5V', '+3V3', 'PWR'):
                    return ref
    return None

def detect_power_sink(parts_db: Dict) -> str:
    """Auto-detect main power consumer (regulator, MCU)."""
    for ref, part in parts_db['parts'].items():
        # Regulators and ICs typically start with U
        if ref.startswith('U'):
            return ref
    return None

def analyze_power_flow(parts_db: Dict) -> PowerFlowGraph:
    """
    Complete power flow analysis.
    Returns graph with detected trunk path.
    """
    source = detect_power_source(parts_db)
    sink = detect_power_sink(parts_db)

    if not source or not sink:
        return None

    graph = build_connectivity_graph(parts_db)
    trunk = find_trunk_chain(parts_db, source, sink)

    return PowerFlowGraph(
        nodes=set(parts_db['parts'].keys()),
        edges=graph,
        source=source,
        sink=sink,
        trunk_path=trunk
    )
```

#### Step 1.2: Integrate into Intelligent Router
```python
# In intelligent_router.py, modify route_all():

def route_all(self, parts_db: Dict, placement: Dict) -> Dict[str, Route]:
    # NEW: Analyze power flow
    from .graph_analyzer import analyze_power_flow
    power_flow = analyze_power_flow(parts_db)

    # Route power nets along trunk chain
    if power_flow and power_flow.trunk_path:
        self._route_power_trunk(parts_db, placement, power_flow)

    # Then route remaining nets...
```

#### Step 1.3: Unit Tests
```python
# tests/test_graph_analyzer.py

def test_trunk_detection_simple():
    """J1 -> C1 -> U1 should find path."""
    parts_db = {
        'parts': {
            'J1': {...},  # Connector
            'C1': {...},  # Cap
            'U1': {...},  # Regulator
        },
        'nets': {
            'VIN': {'pins': [('J1', '1'), ('C1', '1'), ('U1', '1')]},
        }
    }
    path = find_trunk_chain(parts_db, 'J1', 'U1')
    assert path == ['J1', 'C1', 'U1'] or path == ['J1', 'U1']

def test_trunk_detection_with_fuse():
    """J1 -> F1 -> D1 -> U1"""
    # ...
```

### Deliverables
- [ ] `graph_analyzer.py` created
- [ ] `intelligent_router.py` modified to use trunk detection
- [ ] Unit tests passing
- [ ] Test on LED driver board (generate_intelligent.py)

---

## Phase 2: Net Class Constraints (Week 2)

### Goal
Define per-net-class rules: trace width, clearance, via size, allowed layers, max length.

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pcb_engine/net_classes.py` | CREATE | Net class definitions |
| `pcb_engine/routing_types.py` | MODIFY | Add net class to Route |
| `pcb_engine/routing_piston.py` | MODIFY | Use net class constraints |
| `pcb_engine/intelligent_router.py` | MODIFY | Apply constraints |

### Implementation Steps

#### Step 2.1: Create Net Classes Module
```python
# pcb_engine/net_classes.py

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum, auto

class NetClassType(Enum):
    POWER = auto()      # VCC, VIN, +5V, +3V3
    GROUND = auto()     # GND, VSS, AGND, DGND
    SIGNAL = auto()     # Generic signals
    HIGH_SPEED = auto() # CLK, USB, SPI_CLK
    ANALOG = auto()     # ADC, DAC, sensor signals
    DIFFERENTIAL = auto()  # USB_D+/D-, LVDS

@dataclass
class NetClassConstraints:
    """Constraints for a net class."""
    name: str
    net_type: NetClassType

    # Trace parameters
    trace_width: float = 0.25  # mm
    min_trace_width: float = 0.15
    max_trace_width: float = 1.0

    # Clearance
    clearance: float = 0.2  # mm
    clearance_to_power: float = 0.3
    clearance_to_gnd: float = 0.2

    # Via parameters
    via_diameter: float = 0.8
    via_drill: float = 0.4
    max_vias: Optional[int] = None  # None = unlimited

    # Layer restrictions
    allowed_layers: List[str] = field(default_factory=lambda: ['F.Cu', 'B.Cu'])
    preferred_layer: str = 'F.Cu'

    # Length constraints
    max_length: Optional[float] = None  # mm, None = unlimited
    min_length: Optional[float] = None
    length_tolerance: float = 0.1  # 10% tolerance for matching

    # Special rules
    no_stubs: bool = False  # No T-junctions
    require_matched_length: bool = False
    shield_required: bool = False

# Default net classes
DEFAULT_NET_CLASSES = {
    'POWER': NetClassConstraints(
        name='POWER',
        net_type=NetClassType.POWER,
        trace_width=0.5,
        min_trace_width=0.3,
        clearance=0.25,
        via_diameter=1.0,
        via_drill=0.5,
        preferred_layer='F.Cu',
    ),
    'GROUND': NetClassConstraints(
        name='GROUND',
        net_type=NetClassType.GROUND,
        trace_width=0.5,
        min_trace_width=0.3,
        clearance=0.2,
        via_diameter=0.8,
        via_drill=0.4,
        preferred_layer='B.Cu',  # GND on bottom
    ),
    'SIGNAL': NetClassConstraints(
        name='SIGNAL',
        net_type=NetClassType.SIGNAL,
        trace_width=0.25,
        clearance=0.2,
        preferred_layer='F.Cu',
    ),
    'HIGH_SPEED': NetClassConstraints(
        name='HIGH_SPEED',
        net_type=NetClassType.HIGH_SPEED,
        trace_width=0.2,
        clearance=0.25,
        max_vias=2,  # Minimize vias for high-speed
        max_length=50.0,  # Keep short
        no_stubs=True,
    ),
    'ANALOG': NetClassConstraints(
        name='ANALOG',
        net_type=NetClassType.ANALOG,
        trace_width=0.3,
        clearance=0.3,  # Extra clearance from digital
        clearance_to_power=0.5,
        shield_required=True,
    ),
}

def classify_net(net_name: str) -> NetClassConstraints:
    """Auto-classify net by name patterns."""
    name_upper = net_name.upper()

    # Ground nets
    if any(g in name_upper for g in ('GND', 'VSS', 'AGND', 'DGND', 'PGND')):
        return DEFAULT_NET_CLASSES['GROUND']

    # Power nets
    if any(p in name_upper for p in ('VCC', 'VDD', 'VIN', '+5V', '+3V3', '+12V', 'PWR', 'VBUS')):
        return DEFAULT_NET_CLASSES['POWER']

    # High-speed nets
    if any(h in name_upper for h in ('CLK', 'SCK', 'SCLK', 'USB', 'ETH', 'RMII', 'MDIO')):
        return DEFAULT_NET_CLASSES['HIGH_SPEED']

    # Analog nets
    if any(a in name_upper for a in ('ADC', 'DAC', 'AIN', 'AOUT', 'VREF', 'SENSE')):
        return DEFAULT_NET_CLASSES['ANALOG']

    # Default to signal
    return DEFAULT_NET_CLASSES['SIGNAL']

def get_net_class_for_design(parts_db: Dict) -> Dict[str, NetClassConstraints]:
    """
    Classify all nets in a design.
    Returns dict: net_name -> NetClassConstraints
    """
    net_classes = {}
    for net_name in parts_db['nets'].keys():
        net_classes[net_name] = classify_net(net_name)
    return net_classes
```

#### Step 2.2: Modify Routing Types
```python
# In routing_types.py, add to TrackSegment:

@dataclass
class TrackSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    layer: str
    width: float
    net: str
    net_class: Optional[str] = None  # NEW: Net class reference
```

#### Step 2.3: Integrate into Routers
```python
# In routing_piston.py, when creating segments:

def _create_segment(self, start, end, net_name, layer):
    from .net_classes import classify_net
    nc = classify_net(net_name)

    return TrackSegment(
        start=start,
        end=end,
        layer=layer,
        width=nc.trace_width,  # Use net class width
        net=net_name,
        net_class=nc.name
    )
```

### Deliverables
- [ ] `net_classes.py` created
- [ ] `routing_types.py` modified
- [ ] `routing_piston.py` uses net class constraints
- [ ] `intelligent_router.py` uses net class constraints
- [ ] Unit tests for net classification

---

## Phase 3: Push-and-Shove (Week 3-4)

### Goal
When routing a new trace, nudge existing traces aside instead of failing or ripping up entirely.

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pcb_engine/push_shove_router.py` | CREATE | Push-and-shove algorithm |
| `pcb_engine/spatial_index.py` | CREATE | Fast collision detection |
| `pcb_engine/routing_piston.py` | MODIFY | Integrate push-shove |

### Algorithm Design

```
PUSH-AND-SHOVE ALGORITHM:

1. Attempt to route new trace using A*/Lee
2. If blocked by existing trace:
   a. Calculate how much space is needed
   b. Check if existing trace can be PUSHED (moved parallel)
   c. If pushable:
      - Calculate new position for existing trace
      - Check clearance at new position
      - Move existing trace
      - Continue routing new trace
   d. If not pushable:
      - Try SHOVE (reroute small segment)
      - If shove works, continue
      - If not, fall back to rip-up
3. Validate all moved traces still meet DRC
```

### Implementation Steps

#### Step 3.1: Create Spatial Index
```python
# pcb_engine/spatial_index.py

from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
import math

@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def intersects(self, other: 'BoundingBox') -> bool:
        return not (self.x_max < other.x_min or
                    self.x_min > other.x_max or
                    self.y_max < other.y_min or
                    self.y_min > other.y_max)

    def expand(self, margin: float) -> 'BoundingBox':
        return BoundingBox(
            self.x_min - margin,
            self.y_min - margin,
            self.x_max + margin,
            self.y_max + margin
        )

class SpatialIndex:
    """
    Grid-based spatial index for fast collision queries.
    """
    def __init__(self, width: float, height: float, cell_size: float = 1.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(math.ceil(width / cell_size))
        self.rows = int(math.ceil(height / cell_size))
        self.grid = [[set() for _ in range(self.cols)] for _ in range(self.rows)]
        self.segments = {}  # id -> segment data

    def _get_cells(self, bbox: BoundingBox) -> List[Tuple[int, int]]:
        """Get all grid cells that a bounding box touches."""
        x1 = max(0, int(bbox.x_min / self.cell_size))
        y1 = max(0, int(bbox.y_min / self.cell_size))
        x2 = min(self.cols - 1, int(bbox.x_max / self.cell_size))
        y2 = min(self.rows - 1, int(bbox.y_max / self.cell_size))

        cells = []
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                cells.append((x, y))
        return cells

    def insert(self, segment_id: str, segment_data: dict, bbox: BoundingBox):
        """Insert a segment into the index."""
        self.segments[segment_id] = (segment_data, bbox)
        for x, y in self._get_cells(bbox):
            self.grid[y][x].add(segment_id)

    def remove(self, segment_id: str):
        """Remove a segment from the index."""
        if segment_id not in self.segments:
            return
        _, bbox = self.segments[segment_id]
        for x, y in self._get_cells(bbox):
            self.grid[y][x].discard(segment_id)
        del self.segments[segment_id]

    def query(self, bbox: BoundingBox) -> Set[str]:
        """Find all segments that might intersect the bounding box."""
        candidates = set()
        for x, y in self._get_cells(bbox):
            candidates.update(self.grid[y][x])
        return candidates

    def find_collisions(self, bbox: BoundingBox, exclude_net: str = None) -> List[str]:
        """Find segments that actually collide with bbox."""
        candidates = self.query(bbox)
        collisions = []
        for seg_id in candidates:
            seg_data, seg_bbox = self.segments[seg_id]
            if exclude_net and seg_data.get('net') == exclude_net:
                continue
            if bbox.intersects(seg_bbox):
                collisions.append(seg_id)
        return collisions
```

#### Step 3.2: Create Push-and-Shove Router
```python
# pcb_engine/push_shove_router.py

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .spatial_index import SpatialIndex, BoundingBox
from .net_classes import classify_net, NetClassConstraints
from .routing_types import TrackSegment, Route

@dataclass
class PushResult:
    success: bool
    pushed_segments: List[Tuple[str, TrackSegment, TrackSegment]]  # (id, old, new)
    message: str = ""

class PushShoveRouter:
    """
    Push-and-shove routing algorithm.

    When a new trace would collide with existing traces:
    1. Try to PUSH (translate parallel) the blocking trace
    2. If push fails, try to SHOVE (local reroute)
    3. If shove fails, report failure (caller can try rip-up)
    """

    def __init__(self, board_width: float, board_height: float,
                 min_clearance: float = 0.2, max_push_distance: float = 2.0):
        self.board_width = board_width
        self.board_height = board_height
        self.min_clearance = min_clearance
        self.max_push_distance = max_push_distance

        self.spatial_index = SpatialIndex(board_width, board_height, cell_size=1.0)
        self.routes: Dict[str, Route] = {}
        self.segment_to_net: Dict[str, str] = {}

    def add_existing_route(self, net_name: str, route: Route):
        """Register an existing route in the spatial index."""
        self.routes[net_name] = route
        for i, seg in enumerate(route.segments):
            seg_id = f"{net_name}_seg_{i}"
            bbox = self._segment_bbox(seg)
            self.spatial_index.insert(seg_id, {'net': net_name, 'index': i}, bbox)
            self.segment_to_net[seg_id] = net_name

    def _segment_bbox(self, seg: TrackSegment) -> BoundingBox:
        """Calculate bounding box for a segment including clearance."""
        half_width = seg.width / 2 + self.min_clearance
        x1, y1 = seg.start
        x2, y2 = seg.end
        return BoundingBox(
            min(x1, x2) - half_width,
            min(y1, y2) - half_width,
            max(x1, x2) + half_width,
            max(y1, y2) + half_width
        )

    def try_push(self, new_segment: TrackSegment, blocking_seg_id: str) -> PushResult:
        """
        Try to push a blocking segment out of the way.

        Strategy:
        1. Determine push direction (perpendicular to new segment)
        2. Calculate required push distance
        3. Check if pushed position is valid (within board, clearance OK)
        4. Return push result
        """
        if blocking_seg_id not in self.segments:
            return PushResult(False, [], "Segment not found")

        blocking_data, blocking_bbox = self.spatial_index.segments[blocking_seg_id]
        blocking_net = blocking_data['net']
        blocking_route = self.routes[blocking_net]
        blocking_seg = blocking_route.segments[blocking_data['index']]

        # Calculate push direction and distance
        push_dir = self._calculate_push_direction(new_segment, blocking_seg)
        push_dist = self._calculate_push_distance(new_segment, blocking_seg)

        if push_dist > self.max_push_distance:
            return PushResult(False, [], f"Push distance {push_dist:.2f} exceeds max {self.max_push_distance}")

        # Create pushed segment
        dx, dy = push_dir[0] * push_dist, push_dir[1] * push_dist
        pushed_seg = TrackSegment(
            start=(blocking_seg.start[0] + dx, blocking_seg.start[1] + dy),
            end=(blocking_seg.end[0] + dx, blocking_seg.end[1] + dy),
            layer=blocking_seg.layer,
            width=blocking_seg.width,
            net=blocking_seg.net
        )

        # Validate pushed position
        if not self._is_valid_position(pushed_seg, exclude_net=blocking_net):
            return PushResult(False, [], "Pushed position invalid")

        return PushResult(
            success=True,
            pushed_segments=[(blocking_seg_id, blocking_seg, pushed_seg)],
            message=f"Pushed {push_dist:.2f}mm"
        )

    def _calculate_push_direction(self, new_seg: TrackSegment,
                                   blocking_seg: TrackSegment) -> Tuple[float, float]:
        """Calculate perpendicular push direction."""
        # New segment direction
        dx = new_seg.end[0] - new_seg.start[0]
        dy = new_seg.end[1] - new_seg.start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.001:
            return (0, 1)  # Default up

        # Perpendicular (rotate 90 degrees)
        perp_x = -dy / length
        perp_y = dx / length

        # Choose direction that moves blocking segment away
        blocking_center = (
            (blocking_seg.start[0] + blocking_seg.end[0]) / 2,
            (blocking_seg.start[1] + blocking_seg.end[1]) / 2
        )
        new_center = (
            (new_seg.start[0] + new_seg.end[0]) / 2,
            (new_seg.start[1] + new_seg.end[1]) / 2
        )

        # Dot product to determine which side
        to_blocking = (blocking_center[0] - new_center[0],
                       blocking_center[1] - new_center[1])
        dot = perp_x * to_blocking[0] + perp_y * to_blocking[1]

        if dot < 0:
            perp_x, perp_y = -perp_x, -perp_y

        return (perp_x, perp_y)

    def _calculate_push_distance(self, new_seg: TrackSegment,
                                  blocking_seg: TrackSegment) -> float:
        """Calculate minimum push distance for clearance."""
        # Combined half-widths plus clearance
        required_gap = (new_seg.width / 2 + blocking_seg.width / 2 +
                       self.min_clearance * 2)

        # Current distance between segment centerlines
        current_dist = self._segment_distance(new_seg, blocking_seg)

        return max(0, required_gap - current_dist + 0.1)  # +0.1 safety margin

    def _segment_distance(self, seg1: TrackSegment, seg2: TrackSegment) -> float:
        """Calculate minimum distance between two segments."""
        # Simplified: use center-to-center distance
        c1 = ((seg1.start[0] + seg1.end[0]) / 2, (seg1.start[1] + seg1.end[1]) / 2)
        c2 = ((seg2.start[0] + seg2.end[0]) / 2, (seg2.start[1] + seg2.end[1]) / 2)
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def _is_valid_position(self, seg: TrackSegment, exclude_net: str = None) -> bool:
        """Check if segment position is valid (within board, no collisions)."""
        bbox = self._segment_bbox(seg)

        # Check board boundaries
        if (bbox.x_min < 0 or bbox.y_min < 0 or
            bbox.x_max > self.board_width or bbox.y_max > self.board_height):
            return False

        # Check collisions with other segments
        collisions = self.spatial_index.find_collisions(bbox, exclude_net)
        return len(collisions) == 0

    def apply_pushes(self, push_result: PushResult):
        """Apply pushed segments to the routes and spatial index."""
        for seg_id, old_seg, new_seg in push_result.pushed_segments:
            net_name = self.segment_to_net[seg_id]
            seg_index = int(seg_id.split('_')[-1])

            # Update route
            self.routes[net_name].segments[seg_index] = new_seg

            # Update spatial index
            self.spatial_index.remove(seg_id)
            bbox = self._segment_bbox(new_seg)
            self.spatial_index.insert(seg_id, {'net': net_name, 'index': seg_index}, bbox)

import math
```

#### Step 3.3: Integrate into Routing Piston
```python
# In routing_piston.py, add push-shove as a routing strategy:

def _route_with_push_shove(self, net_order, net_pins, placement, layer):
    """Route using push-and-shove strategy."""
    from .push_shove_router import PushShoveRouter

    ps_router = PushShoveRouter(
        self.board_width,
        self.board_height,
        self.min_clearance
    )

    # Add existing routes to spatial index
    for net_name, route in self.completed_routes.items():
        ps_router.add_existing_route(net_name, route)

    # Route each net
    for net_name in net_order:
        # Try normal A* first
        route = self._route_astar(net_name, net_pins[net_name], layer)

        if not route.success:
            # Check what's blocking
            # Try push-and-shove
            # If successful, apply pushes and retry
            pass
```

### Deliverables
- [ ] `spatial_index.py` created and tested
- [ ] `push_shove_router.py` created
- [ ] Integration with `routing_piston.py`
- [ ] Unit tests for push operations
- [ ] Test on sensor array board (should improve routing success)

---

## Phase 4: Return Path Awareness (Week 5-6)

### Goal
Ensure signals have continuous return paths on adjacent layers; avoid crossing plane splits; auto-stitch ground vias.

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pcb_engine/return_path_analyzer.py` | CREATE | Return path analysis |
| `pcb_engine/ground_stitching.py` | CREATE | Auto via stitching |
| `pcb_engine/drc_piston.py` | MODIFY | Add return path checks |

### Implementation Steps

#### Step 4.1: Create Return Path Analyzer
```python
# pcb_engine/return_path_analyzer.py

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

class ReturnPathQuality(Enum):
    EXCELLENT = "excellent"  # Solid plane directly below
    GOOD = "good"            # Plane with minor gaps
    MARGINAL = "marginal"    # Partial coverage
    POOR = "poor"            # No return path
    CRITICAL = "critical"    # Crosses plane split

@dataclass
class ReturnPathAnalysis:
    net_name: str
    quality: ReturnPathQuality
    coverage_percent: float
    gaps: List[Tuple[float, float, float, float]]  # (x1, y1, x2, y2) uncovered regions
    plane_splits_crossed: int
    recommendations: List[str]

class ReturnPathAnalyzer:
    """
    Analyzes signal return paths.

    For high-frequency signals, the return current flows directly
    beneath the signal trace on the reference plane. Gaps or splits
    in this plane cause EMI and signal integrity issues.
    """

    def __init__(self, board_width: float, board_height: float):
        self.board_width = board_width
        self.board_height = board_height
        self.ground_zones = []  # List of ground pour polygons
        self.power_zones = []   # List of power pour polygons
        self.plane_splits = []  # List of split boundaries

    def add_ground_zone(self, layer: str, polygon: List[Tuple[float, float]]):
        """Register a ground zone/pour."""
        self.ground_zones.append({'layer': layer, 'polygon': polygon})

    def add_plane_split(self, layer: str, line: Tuple[Tuple[float, float], Tuple[float, float]]):
        """Register a plane split (gap between zones)."""
        self.plane_splits.append({'layer': layer, 'line': line})

    def analyze_route(self, route: 'Route', signal_layer: str,
                      reference_layer: str) -> ReturnPathAnalysis:
        """
        Analyze return path quality for a routed signal.

        Args:
            route: The signal route to analyze
            signal_layer: Layer the signal is on (e.g., 'F.Cu')
            reference_layer: Expected return path layer (e.g., 'B.Cu' for 2-layer)
        """
        total_length = 0
        covered_length = 0
        gaps = []
        splits_crossed = 0

        for seg in route.segments:
            if seg.layer != signal_layer:
                continue

            seg_length = self._segment_length(seg)
            total_length += seg_length

            # Check coverage by reference plane
            coverage = self._check_plane_coverage(seg, reference_layer)
            covered_length += seg_length * coverage

            if coverage < 1.0:
                gaps.append((seg.start[0], seg.start[1], seg.end[0], seg.end[1]))

            # Check plane split crossings
            if self._crosses_plane_split(seg, reference_layer):
                splits_crossed += 1

        coverage_percent = (covered_length / total_length * 100) if total_length > 0 else 100

        # Determine quality
        if splits_crossed > 0:
            quality = ReturnPathQuality.CRITICAL
        elif coverage_percent >= 95:
            quality = ReturnPathQuality.EXCELLENT
        elif coverage_percent >= 80:
            quality = ReturnPathQuality.GOOD
        elif coverage_percent >= 50:
            quality = ReturnPathQuality.MARGINAL
        else:
            quality = ReturnPathQuality.POOR

        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality, gaps, splits_crossed, route.net
        )

        return ReturnPathAnalysis(
            net_name=route.net,
            quality=quality,
            coverage_percent=coverage_percent,
            gaps=gaps,
            plane_splits_crossed=splits_crossed,
            recommendations=recommendations
        )

    def _segment_length(self, seg) -> float:
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        return math.sqrt(dx*dx + dy*dy)

    def _check_plane_coverage(self, seg, reference_layer: str) -> float:
        """Check what fraction of segment is covered by reference plane."""
        # Simplified: check if segment endpoints are within any ground zone
        # Full implementation would do line-polygon intersection
        for zone in self.ground_zones:
            if zone['layer'] == reference_layer:
                if self._point_in_polygon(seg.start, zone['polygon']):
                    if self._point_in_polygon(seg.end, zone['polygon']):
                        return 1.0
        return 0.0  # Not covered

    def _point_in_polygon(self, point, polygon) -> bool:
        """Ray casting algorithm for point-in-polygon."""
        x, y = point
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def _crosses_plane_split(self, seg, reference_layer: str) -> bool:
        """Check if segment crosses a plane split."""
        for split in self.plane_splits:
            if split['layer'] == reference_layer:
                if self._segments_intersect(
                    seg.start, seg.end,
                    split['line'][0], split['line'][1]
                ):
                    return True
        return False

    def _segments_intersect(self, p1, p2, p3, p4) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
                ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def _generate_recommendations(self, quality, gaps, splits, net_name) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if quality == ReturnPathQuality.CRITICAL:
            recs.append(f"CRITICAL: {net_name} crosses {splits} plane split(s). Reroute to avoid.")

        if quality in (ReturnPathQuality.POOR, ReturnPathQuality.MARGINAL):
            recs.append(f"Add ground pour under {net_name} trace")

        if len(gaps) > 0:
            recs.append(f"Add {len(gaps)} stitching vias near gaps")

        return recs

import math
```

#### Step 4.2: Create Ground Stitching Module
```python
# pcb_engine/ground_stitching.py

from typing import List, Tuple, Dict
from dataclasses import dataclass
from .routing_types import Via

@dataclass
class StitchingVia:
    position: Tuple[float, float]
    reason: str  # "return_path", "plane_connection", "shielding"

class GroundStitcher:
    """
    Automatically places ground stitching vias.

    Stitching vias:
    1. Connect ground planes on different layers
    2. Provide return path continuity near signals
    3. Shield sensitive signals
    """

    def __init__(self, via_diameter: float = 0.8, via_drill: float = 0.4,
                 stitch_spacing: float = 5.0):
        self.via_diameter = via_diameter
        self.via_drill = via_drill
        self.stitch_spacing = stitch_spacing

    def generate_stitching_vias(self,
                                 routes: Dict[str, 'Route'],
                                 return_path_analyses: List['ReturnPathAnalysis'],
                                 board_width: float,
                                 board_height: float) -> List[Via]:
        """
        Generate stitching vias based on return path analysis.
        """
        vias = []

        # 1. Vias for return path gaps
        for analysis in return_path_analyses:
            if analysis.quality in ('poor', 'marginal', 'critical'):
                gap_vias = self._vias_for_gaps(analysis.gaps)
                vias.extend(gap_vias)

        # 2. Perimeter stitching (board edges)
        perimeter_vias = self._perimeter_stitching(board_width, board_height)
        vias.extend(perimeter_vias)

        # 3. Signal shielding vias (for high-speed nets)
        for net_name, route in routes.items():
            if self._is_high_speed_net(net_name):
                shield_vias = self._shield_signal(route)
                vias.extend(shield_vias)

        return vias

    def _vias_for_gaps(self, gaps: List[Tuple[float, float, float, float]]) -> List[Via]:
        """Place vias at return path gaps."""
        vias = []
        for x1, y1, x2, y2 in gaps:
            # Place via at center of gap
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            vias.append(Via(
                position=(cx, cy),
                net='GND',
                diameter=self.via_diameter,
                drill=self.via_drill,
                from_layer='F.Cu',
                to_layer='B.Cu'
            ))
        return vias

    def _perimeter_stitching(self, width: float, height: float) -> List[Via]:
        """Place stitching vias around board perimeter."""
        vias = []
        margin = 2.0  # mm from edge

        # Top and bottom edges
        for x in self._spacing_range(margin, width - margin, self.stitch_spacing):
            vias.append(self._make_gnd_via(x, margin))
            vias.append(self._make_gnd_via(x, height - margin))

        # Left and right edges
        for y in self._spacing_range(margin, height - margin, self.stitch_spacing):
            vias.append(self._make_gnd_via(margin, y))
            vias.append(self._make_gnd_via(width - margin, y))

        return vias

    def _shield_signal(self, route: 'Route') -> List[Via]:
        """Place shielding vias along a high-speed signal."""
        vias = []
        shield_offset = 1.5  # mm from trace

        for seg in route.segments:
            # Place vias on both sides of segment
            dx = seg.end[0] - seg.start[0]
            dy = seg.end[1] - seg.start[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.1:
                continue

            # Perpendicular unit vector
            px, py = -dy/length, dx/length

            # Via at segment midpoint, offset to each side
            mx = (seg.start[0] + seg.end[0]) / 2
            my = (seg.start[1] + seg.end[1]) / 2

            vias.append(self._make_gnd_via(mx + px * shield_offset,
                                           my + py * shield_offset))
            vias.append(self._make_gnd_via(mx - px * shield_offset,
                                           my - py * shield_offset))

        return vias

    def _make_gnd_via(self, x: float, y: float) -> Via:
        return Via(
            position=(x, y),
            net='GND',
            diameter=self.via_diameter,
            drill=self.via_drill,
            from_layer='F.Cu',
            to_layer='B.Cu'
        )

    def _spacing_range(self, start: float, end: float, step: float) -> List[float]:
        result = []
        x = start
        while x <= end:
            result.append(x)
            x += step
        return result

    def _is_high_speed_net(self, net_name: str) -> bool:
        name_upper = net_name.upper()
        return any(h in name_upper for h in ('CLK', 'SCK', 'USB', 'ETH', 'SPI'))

import math
```

### Deliverables
- [ ] `return_path_analyzer.py` created
- [ ] `ground_stitching.py` created
- [ ] Integration with DRC piston
- [ ] Unit tests
- [ ] Documentation

---

## Testing Strategy

### Unit Tests (Per Phase)
Each phase has dedicated unit tests in `tests/test_<module>.py`.

### Integration Tests
After each phase, run:
1. `generate_intelligent.py` (5-component LED driver)
2. `generate_laser_sensor_array.py` (multi-sensor module)
3. Full BBL test suite

### Success Criteria

| Phase | Success Metric |
|-------|----------------|
| 1 | Power trunk detected correctly in test cases |
| 2 | Net classes applied, different widths visible in KiCad |
| 3 | Routing success rate improves by >20% on sensor array |
| 4 | Return path analysis reports generated, stitching vias placed |

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Phase 1 | Trunk Chain Detection |
| 2 | Phase 2 | Net Class Constraints |
| 3-4 | Phase 3 | Push-and-Shove Router |
| 5-6 | Phase 4 | Return Path Awareness |

**Total: 6 weeks for full implementation**

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Push-and-shove complexity | Start with simple parallel push only |
| Return path calculation expensive | Use spatial indexing, cache results |
| Integration breaks existing code | Incremental integration, feature flags |
| DRC validation overhead | Run return path analysis only on request |

---

## Notes

- Each phase builds on the previous one
- Feature flags allow gradual rollout
- Unit tests prevent regressions
- Real-world testing on sensor array board validates improvements
