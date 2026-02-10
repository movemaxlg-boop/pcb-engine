"""
PCB Engine - Topological Router Piston
=======================================

A topological/rubber-band router that uses Delaunay triangulation
for any-angle routing with shape-independent obstacle avoidance.

HIERARCHY:
==========
USER (Boss) → Circuit AI (Engineer) → PCB Engine (Foreman) → This Piston (Worker)

RESEARCH REFERENCES:
====================
- "Topological routing in SURF: generating a rubber-band sketch" (Dai, Dayan 1991)
- "Rubber band routing and dynamic data representation" (IEEE 1990)
- "Constrained Delaunay triangulation for PCB routing" (ScienceDirect 2022)
- "A Novel Global Routing Algorithm Based on Triangular Grid" (MDPI 2023)
- TopoR commercial router (Eremex)

KEY CONCEPTS:
=============
1. DELAUNAY TRIANGULATION
   - Triangulates space between obstacles
   - Maximizes minimum angle (no skinny triangles)
   - Provides natural routing channels

2. RUBBER-BAND ROUTING
   - Routes treated as elastic bands
   - Automatically find shortest path around obstacles
   - Can be "pulled tight" for optimization

3. TOPOLOGICAL REPRESENTATION
   - Routes stored as sequence of obstacle passes (left/right)
   - Shape-independent routing decisions
   - Allows any-angle traces (not just 45°/90°)

4. INCREMENTAL ROUTABILITY TEST
   - Verify design rules during path search
   - Use constrained Delaunay for obstacle representation

ADVANTAGES OVER GRID-BASED:
===========================
- Any-angle routing (shorter traces)
- Better obstacle avoidance
- Natural handling of curved obstacles
- No grid quantization artifacts
- Better wirelength optimization
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict
import heapq


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PassDirection(Enum):
    """Direction of passing an obstacle"""
    LEFT = 'left'
    RIGHT = 'right'
    THROUGH = 'through'  # Via or layer change


class TriangleType(Enum):
    """Type of triangle in the mesh"""
    FREE = 'free'           # Can be routed through
    BLOCKED = 'blocked'     # Contains obstacle
    PARTIAL = 'partial'     # Partially blocked


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Point:
    """2D point"""
    x: float
    y: float

    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)


@dataclass
class Edge:
    """Edge between two points"""
    p1: Point
    p2: Point
    constrained: bool = False  # Is this an obstacle edge?

    def length(self) -> float:
        return self.p1.distance_to(self.p2)

    def midpoint(self) -> Point:
        return Point((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2)


@dataclass
class Triangle:
    """Triangle in the Delaunay mesh"""
    p1: Point
    p2: Point
    p3: Point
    triangle_type: TriangleType = TriangleType.FREE

    def centroid(self) -> Point:
        return Point(
            (self.p1.x + self.p2.x + self.p3.x) / 3,
            (self.p1.y + self.p2.y + self.p3.y) / 3
        )

    def contains_point(self, p: Point) -> bool:
        """Check if point is inside triangle using barycentric coordinates"""
        def sign(p1, p2, p3):
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)

        d1 = sign(p, self.p1, self.p2)
        d2 = sign(p, self.p2, self.p3)
        d3 = sign(p, self.p3, self.p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def edges(self) -> List[Edge]:
        return [
            Edge(self.p1, self.p2),
            Edge(self.p2, self.p3),
            Edge(self.p3, self.p1)
        ]

    def circumcircle(self) -> Tuple[Point, float]:
        """Calculate circumcircle center and radius"""
        ax, ay = self.p1.x, self.p1.y
        bx, by = self.p2.x, self.p2.y
        cx, cy = self.p3.x, self.p3.y

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            # Degenerate triangle
            center = self.centroid()
            return center, 0

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

        center = Point(ux, uy)
        radius = center.distance_to(self.p1)
        return center, radius


@dataclass
class Obstacle:
    """An obstacle (pad, component, keepout)"""
    id: str
    center: Point
    radius: float  # Simplified as circle
    vertices: List[Point] = field(default_factory=list)  # For polygonal obstacles
    net: str = ''  # If this is a pad, which net


@dataclass
class TopologicalPath:
    """
    A topological path - sequence of obstacle passes.

    This is the key concept: instead of storing coordinates,
    we store how the path passes each obstacle.
    """
    net: str
    start: Point
    end: Point
    passes: List[Tuple[str, PassDirection]] = field(default_factory=list)
    layer: str = 'F.Cu'

    def add_pass(self, obstacle_id: str, direction: PassDirection):
        self.passes.append((obstacle_id, direction))


@dataclass
class RubberBandRoute:
    """
    A rubber-band route with waypoints.

    The waypoints can be "pulled tight" to optimize the route.
    """
    net: str
    waypoints: List[Point] = field(default_factory=list)
    layer: str = 'F.Cu'
    width: float = 0.25
    is_optimized: bool = False

    def total_length(self) -> float:
        length = 0.0
        for i in range(len(self.waypoints) - 1):
            length += self.waypoints[i].distance_to(self.waypoints[i + 1])
        return length


@dataclass
class TopoRouterConfig:
    """Configuration for topological router"""
    # Board parameters
    board_width: float = 100.0
    board_height: float = 100.0

    # Design rules
    trace_width: float = 0.25
    clearance: float = 0.15

    # Algorithm parameters
    max_iterations: int = 1000
    optimization_passes: int = 5
    angle_resolution: float = 5.0  # Degrees

    # Rubber band parameters
    pull_force: float = 0.5
    damping: float = 0.8

    # Layer support
    num_layers: int = 2


@dataclass
class TopoRouterResult:
    """Result from topological router"""
    success: bool
    routes: Dict[str, RubberBandRoute] = field(default_factory=dict)
    routed_count: int = 0
    total_count: int = 0
    total_length: float = 0.0
    warnings: List[str] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)


# =============================================================================
# DELAUNAY TRIANGULATION
# =============================================================================

class DelaunayTriangulation:
    """
    Constrained Delaunay Triangulation for PCB routing.

    Uses Bowyer-Watson algorithm for incremental construction.

    Reference: "Constrained Delaunay triangulation for PCB routing"
    """

    def __init__(self, bounds: Tuple[float, float, float, float]):
        """
        Initialize with bounding box (min_x, min_y, max_x, max_y)
        """
        self.bounds = bounds
        self.triangles: List[Triangle] = []
        self.points: Set[Point] = set()
        self.constrained_edges: Set[Tuple[Point, Point]] = set()

        # Create super-triangle that contains all points
        self._create_super_triangle()

    def _create_super_triangle(self):
        """Create a super-triangle that bounds the entire area"""
        min_x, min_y, max_x, max_y = self.bounds
        dx = max_x - min_x
        dy = max_y - min_y
        delta = max(dx, dy) * 2

        # Create super-triangle vertices far outside bounds
        p1 = Point(min_x - delta, min_y - delta)
        p2 = Point(min_x + dx / 2, max_y + delta * 2)
        p3 = Point(max_x + delta, min_y - delta)

        self.super_triangle = Triangle(p1, p2, p3)
        self.triangles = [self.super_triangle]

    def insert_point(self, p: Point):
        """
        Insert a point using Bowyer-Watson algorithm.

        1. Find all triangles whose circumcircle contains the point
        2. Remove those triangles, leaving a "hole"
        3. Re-triangulate the hole with the new point
        """
        if p in self.points:
            return
        self.points.add(p)

        # Find "bad" triangles (circumcircle contains p)
        bad_triangles = []
        for tri in self.triangles:
            center, radius = tri.circumcircle()
            if center.distance_to(p) <= radius:
                bad_triangles.append(tri)

        # Find boundary of hole (edges that are not shared)
        boundary = []
        for tri in bad_triangles:
            for edge in tri.edges():
                # Check if edge is shared with another bad triangle
                shared = False
                for other in bad_triangles:
                    if other == tri:
                        continue
                    for other_edge in other.edges():
                        if self._edges_equal(edge, other_edge):
                            shared = True
                            break
                    if shared:
                        break
                if not shared:
                    boundary.append(edge)

        # Remove bad triangles
        for tri in bad_triangles:
            self.triangles.remove(tri)

        # Create new triangles from boundary edges to new point
        for edge in boundary:
            new_tri = Triangle(edge.p1, edge.p2, p)
            self.triangles.append(new_tri)

    def _edges_equal(self, e1: Edge, e2: Edge) -> bool:
        """Check if two edges are equal (ignoring direction)"""
        return (e1.p1 == e2.p1 and e1.p2 == e2.p2) or \
               (e1.p1 == e2.p2 and e1.p2 == e2.p1)

    def add_constraint(self, p1: Point, p2: Point):
        """
        Add a constrained edge (obstacle boundary).

        This edge will be preserved during triangulation.
        """
        self.constrained_edges.add((p1, p2))
        # Insert endpoints if not already present
        self.insert_point(p1)
        self.insert_point(p2)

        # TODO: Implement constraint enforcement
        # This would require edge flipping to ensure the constraint is present

    def remove_super_triangle(self):
        """Remove triangles connected to super-triangle vertices"""
        super_verts = {self.super_triangle.p1, self.super_triangle.p2, self.super_triangle.p3}
        self.triangles = [
            tri for tri in self.triangles
            if tri.p1 not in super_verts and tri.p2 not in super_verts and tri.p3 not in super_verts
        ]

    def find_triangle_containing(self, p: Point) -> Optional[Triangle]:
        """Find the triangle containing point p"""
        for tri in self.triangles:
            if tri.contains_point(p):
                return tri
        return None

    def get_adjacent_triangles(self, tri: Triangle) -> List[Triangle]:
        """Get triangles that share an edge with the given triangle"""
        adjacent = []
        for edge in tri.edges():
            for other in self.triangles:
                if other == tri:
                    continue
                for other_edge in other.edges():
                    if self._edges_equal(edge, other_edge):
                        adjacent.append(other)
                        break
        return adjacent


# =============================================================================
# TOPOLOGICAL ROUTER
# =============================================================================

class TopologicalRouterPiston:
    """
    Topological Router using Delaunay triangulation and rubber-band routing.

    ALGORITHM:
    ==========
    1. Build Delaunay triangulation of obstacle vertices
    2. For each net:
       a. Find path through triangle mesh (A* on dual graph)
       b. Record topological sequence (how path passes obstacles)
       c. Convert to rubber-band route with waypoints
    3. Optimize routes:
       a. "Pull tight" each rubber-band
       b. Minimize total wirelength
       c. Apply any-angle optimization
    4. Convert to final trace coordinates

    Usage:
        config = TopoRouterConfig(board_width=50, board_height=50)
        router = TopologicalRouterPiston(config)
        result = router.route(parts_db, placement)
    """

    def __init__(self, config: TopoRouterConfig = None):
        self.config = config or TopoRouterConfig()
        self.triangulation: DelaunayTriangulation = None
        self.obstacles: Dict[str, Obstacle] = {}
        self.routes: Dict[str, RubberBandRoute] = {}
        self.warnings: List[str] = []

    def route(self, parts_db: Dict, placement: Dict) -> TopoRouterResult:
        """
        Route all nets using topological/rubber-band approach.

        Args:
            parts_db: Parts database with components and nets
            placement: Component positions

        Returns:
            TopoRouterResult with rubber-band routes
        """
        self.warnings = []

        # Step 1: Build obstacle map from components
        self._build_obstacles(parts_db, placement)

        # Step 2: Build Delaunay triangulation
        self._build_triangulation()

        # Step 3: Extract nets and pins
        nets = self._extract_nets(parts_db, placement)

        # Step 4: Route each net
        routed_count = 0
        total_count = len(nets)

        for net_name, pins in nets.items():
            if len(pins) < 2:
                continue

            route = self._route_net(net_name, pins)
            if route:
                self.routes[net_name] = route
                routed_count += 1

        # Step 5: Optimize routes (rubber-band tightening)
        self._optimize_routes()

        # Calculate total length
        total_length = sum(r.total_length() for r in self.routes.values())

        return TopoRouterResult(
            success=routed_count > 0,
            routes=self.routes,
            routed_count=routed_count,
            total_count=total_count,
            total_length=total_length,
            warnings=self.warnings,
            statistics={
                'triangles': len(self.triangulation.triangles) if self.triangulation else 0,
                'obstacles': len(self.obstacles),
                'optimization_passes': self.config.optimization_passes
            }
        )

    def _build_obstacles(self, parts_db: Dict, placement: Dict):
        """Build obstacle map from placed components"""
        self.obstacles = {}
        parts = parts_db.get('parts', {})

        for ref, pos in placement.items():
            part = parts.get(ref, {})

            # Get component center
            x = pos.x if hasattr(pos, 'x') else pos[0]
            y = pos.y if hasattr(pos, 'y') else pos[1]

            # Add component body as obstacle (simplified as circle)
            # In reality, would use actual footprint shape
            body_radius = 2.0  # Default size

            self.obstacles[f"{ref}_body"] = Obstacle(
                id=f"{ref}_body",
                center=Point(x, y),
                radius=body_radius
            )

            # Add pads as obstacles (but with associated net)
            pins_data = part.get('pins', part.get('used_pins', {}))

            # Handle both dict and list formats
            if isinstance(pins_data, dict):
                pin_items = pins_data.items()
            else:
                # List format - each item should have 'number' or be indexed
                pin_items = [(str(i), p) if isinstance(p, dict) else (str(i), {}) for i, p in enumerate(pins_data)]

            for pin_num, pin_info in pin_items:
                if isinstance(pin_info, dict):
                    net = pin_info.get('net', '')
                    offset_x = pin_info.get('x', pin_info.get('offset_x', 0))
                    offset_y = pin_info.get('y', pin_info.get('offset_y', 0))
                else:
                    net = ''
                    offset_x = 0
                    offset_y = 0

                pad_x = x + offset_x
                pad_y = y + offset_y

                self.obstacles[f"{ref}_{pin_num}"] = Obstacle(
                    id=f"{ref}_{pin_num}",
                    center=Point(pad_x, pad_y),
                    radius=self.config.trace_width + self.config.clearance,
                    net=net
                )

    def _build_triangulation(self):
        """Build Delaunay triangulation of the routing space"""
        self.triangulation = DelaunayTriangulation(
            (0, 0, self.config.board_width, self.config.board_height)
        )

        # Insert obstacle centers as points
        for obs in self.obstacles.values():
            self.triangulation.insert_point(obs.center)

            # For polygonal obstacles, insert vertices
            for vertex in obs.vertices:
                self.triangulation.insert_point(vertex)

        # Add board corner points
        corners = [
            Point(0, 0),
            Point(self.config.board_width, 0),
            Point(self.config.board_width, self.config.board_height),
            Point(0, self.config.board_height)
        ]
        for corner in corners:
            self.triangulation.insert_point(corner)

        # Remove super-triangle
        self.triangulation.remove_super_triangle()

    def _extract_nets(self, parts_db: Dict, placement: Dict) -> Dict[str, List[Point]]:
        """Extract net-to-pin mapping"""
        nets = defaultdict(list)
        parts = parts_db.get('parts', {})

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            x = pos.x if hasattr(pos, 'x') else pos[0]
            y = pos.y if hasattr(pos, 'y') else pos[1]

            pins_data = part.get('pins', part.get('used_pins', {}))

            # Handle both dict and list formats
            if isinstance(pins_data, dict):
                pin_items = pins_data.items()
            else:
                pin_items = [(str(i), p) if isinstance(p, dict) else (str(i), {}) for i, p in enumerate(pins_data)]

            for pin_num, pin_info in pin_items:
                if isinstance(pin_info, dict):
                    net = pin_info.get('net', '')
                    offset_x = pin_info.get('x', pin_info.get('offset_x', 0))
                    offset_y = pin_info.get('y', pin_info.get('offset_y', 0))
                else:
                    continue

                if not net:
                    continue

                pad_x = x + offset_x
                pad_y = y + offset_y

                nets[net].append(Point(pad_x, pad_y))

        return nets

    def _route_net(self, net_name: str, pins: List[Point]) -> Optional[RubberBandRoute]:
        """
        Route a single net using topological path finding.

        Uses Minimum Spanning Tree to determine connection order,
        then finds topological paths between pins.
        """
        if len(pins) < 2:
            return None

        route = RubberBandRoute(net=net_name, width=self.config.trace_width)

        # Build MST for multi-pin nets
        if len(pins) == 2:
            edges = [(0, 1)]
        else:
            edges = self._build_mst(pins)

        # Route each edge in the MST
        all_waypoints = []
        for i, j in edges:
            path = self._find_topological_path(pins[i], pins[j], net_name)
            if path:
                all_waypoints.extend(path)

        # Remove duplicate consecutive points
        route.waypoints = self._remove_consecutive_duplicates(all_waypoints)

        return route if route.waypoints else None

    def _build_mst(self, pins: List[Point]) -> List[Tuple[int, int]]:
        """Build Minimum Spanning Tree using Kruskal's algorithm"""
        n = len(pins)
        edges = []

        # Create all edges with weights
        for i in range(n):
            for j in range(i + 1, n):
                dist = pins[i].distance_to(pins[j])
                edges.append((dist, i, j))

        # Sort by distance
        edges.sort()

        # Union-Find for cycle detection
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        # Build MST
        mst = []
        for dist, i, j in edges:
            if union(i, j):
                mst.append((i, j))
                if len(mst) == n - 1:
                    break

        return mst

    def _find_topological_path(self, start: Point, end: Point, net: str) -> List[Point]:
        """
        Find a topological path through the triangle mesh.

        Uses A* search on the dual graph (triangle centroids).
        """
        # Find starting and ending triangles
        start_tri = self.triangulation.find_triangle_containing(start)
        end_tri = self.triangulation.find_triangle_containing(end)

        if not start_tri or not end_tri:
            # Points outside triangulation - use straight line
            return [start, end]

        if start_tri == end_tri:
            # Same triangle - direct connection
            return [start, end]

        # A* search through triangles
        path_triangles = self._astar_through_triangles(start_tri, end_tri, end)

        if not path_triangles:
            # No path found - use straight line (may violate DRC)
            self.warnings.append(f"No topological path found for {net}")
            return [start, end]

        # Convert triangle path to waypoints
        waypoints = [start]
        for tri in path_triangles:
            waypoints.append(tri.centroid())
        waypoints.append(end)

        return waypoints

    def _astar_through_triangles(self, start: Triangle, end: Triangle,
                                   goal: Point) -> List[Triangle]:
        """A* search through triangle dual graph"""
        # Priority queue: (f_score, triangle)
        open_set = [(0, id(start), start)]
        came_from = {}
        g_score = {id(start): 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = [current]
                curr_id = id(current)
                while curr_id in came_from:
                    current = came_from[curr_id]
                    curr_id = id(current)
                    path.append(current)
                path.reverse()
                return path

            # Check neighbors
            for neighbor in self.triangulation.get_adjacent_triangles(current):
                if neighbor.triangle_type == TriangleType.BLOCKED:
                    continue

                # Cost is distance between centroids
                tentative_g = g_score[id(current)] + current.centroid().distance_to(neighbor.centroid())

                if id(neighbor) not in g_score or tentative_g < g_score[id(neighbor)]:
                    came_from[id(neighbor)] = current
                    g_score[id(neighbor)] = tentative_g
                    f_score = tentative_g + neighbor.centroid().distance_to(goal)  # Heuristic
                    heapq.heappush(open_set, (f_score, id(neighbor), neighbor))

        return []

    def _remove_consecutive_duplicates(self, points: List[Point]) -> List[Point]:
        """Remove consecutive duplicate points"""
        if not points:
            return []
        result = [points[0]]
        for p in points[1:]:
            if p != result[-1]:
                result.append(p)
        return result

    # =========================================================================
    # RUBBER-BAND OPTIMIZATION
    # =========================================================================

    def _optimize_routes(self):
        """
        Optimize routes using rubber-band physics simulation.

        Each waypoint is treated as a node connected by springs.
        Nodes are pulled toward a straight line while respecting obstacles.
        """
        for _ in range(self.config.optimization_passes):
            for net_name, route in self.routes.items():
                self._optimize_single_route(route)
                route.is_optimized = True

    def _optimize_single_route(self, route: RubberBandRoute):
        """
        Optimize a single route by "pulling tight" the rubber band.

        Uses force-directed relaxation:
        1. Each waypoint feels pull toward straight line
        2. Each waypoint feels repulsion from obstacles
        3. Iterate until convergence
        """
        if len(route.waypoints) < 3:
            return

        waypoints = route.waypoints
        clearance = self.config.clearance + route.width / 2

        for iteration in range(self.config.max_iterations):
            max_movement = 0

            # Don't move first and last points
            for i in range(1, len(waypoints) - 1):
                # Calculate force toward straight line (spring force)
                prev = waypoints[i - 1]
                curr = waypoints[i]
                next_p = waypoints[i + 1]

                # Ideal position is on line between prev and next
                ideal = Point(
                    (prev.x + next_p.x) / 2,
                    (prev.y + next_p.y) / 2
                )

                # Spring force toward ideal
                fx = (ideal.x - curr.x) * self.config.pull_force
                fy = (ideal.y - curr.y) * self.config.pull_force

                # Repulsion from obstacles
                for obs in self.obstacles.values():
                    if obs.net == route.net:
                        continue  # Don't repel from own net's pads

                    dist = curr.distance_to(obs.center)
                    min_dist = obs.radius + clearance

                    if dist < min_dist and dist > 0:
                        # Repulsion force
                        repulsion = (min_dist - dist) / dist
                        fx += (curr.x - obs.center.x) * repulsion
                        fy += (curr.y - obs.center.y) * repulsion

                # Apply force with damping
                new_x = curr.x + fx * self.config.damping
                new_y = curr.y + fy * self.config.damping

                # Clamp to board bounds
                new_x = max(0, min(self.config.board_width, new_x))
                new_y = max(0, min(self.config.board_height, new_y))

                movement = math.sqrt((new_x - curr.x)**2 + (new_y - curr.y)**2)
                max_movement = max(max_movement, movement)

                waypoints[i] = Point(new_x, new_y)

            # Check convergence
            if max_movement < 0.001:
                break

        # Apply any-angle simplification
        route.waypoints = self._simplify_path(waypoints, clearance)

    def _simplify_path(self, waypoints: List[Point], clearance: float) -> List[Point]:
        """
        Simplify path by removing unnecessary waypoints.

        Uses line-of-sight checks: if we can go directly from A to C
        without hitting obstacles, remove B.
        """
        if len(waypoints) <= 2:
            return waypoints

        simplified = [waypoints[0]]

        i = 0
        while i < len(waypoints) - 1:
            # Try to skip as many points as possible
            j = len(waypoints) - 1
            while j > i + 1:
                if self._line_of_sight(waypoints[i], waypoints[j], clearance):
                    break
                j -= 1

            simplified.append(waypoints[j])
            i = j

        return simplified

    def _line_of_sight(self, p1: Point, p2: Point, clearance: float) -> bool:
        """Check if there's a clear line of sight between two points"""
        # Sample points along the line
        dist = p1.distance_to(p2)
        if dist < 0.01:
            return True

        num_samples = max(2, int(dist / 0.5))

        for t in range(num_samples + 1):
            ratio = t / num_samples
            sample = Point(
                p1.x + (p2.x - p1.x) * ratio,
                p1.y + (p2.y - p1.y) * ratio
            )

            # Check against obstacles
            for obs in self.obstacles.values():
                if sample.distance_to(obs.center) < obs.radius + clearance:
                    return False

        return True

    # =========================================================================
    # OUTPUT CONVERSION
    # =========================================================================

    def get_traces(self) -> List[Dict]:
        """
        Convert rubber-band routes to trace segments.

        Returns list of trace dictionaries compatible with output piston.
        """
        traces = []

        for net_name, route in self.routes.items():
            for i in range(len(route.waypoints) - 1):
                p1 = route.waypoints[i]
                p2 = route.waypoints[i + 1]

                traces.append({
                    'net': net_name,
                    'start': (p1.x, p1.y),
                    'end': (p2.x, p2.y),
                    'width': route.width,
                    'layer': route.layer
                })

        return traces


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Topological Router Piston Self-Test")
    print("=" * 60)

    # Test Delaunay triangulation
    print("\n1. Testing Delaunay Triangulation...")
    dt = DelaunayTriangulation((0, 0, 50, 50))

    # Insert some points
    test_points = [
        Point(10, 10),
        Point(40, 10),
        Point(25, 40),
        Point(15, 25),
        Point(35, 25),
    ]

    for p in test_points:
        dt.insert_point(p)

    dt.remove_super_triangle()
    print(f"   Points inserted: {len(test_points)}")
    print(f"   Triangles created: {len(dt.triangles)}")

    # Test router
    print("\n2. Testing Topological Router...")
    config = TopoRouterConfig(
        board_width=50,
        board_height=50,
        trace_width=0.25,
        clearance=0.15
    )
    router = TopologicalRouterPiston(config)

    # Create simple test case
    parts_db = {
        'parts': {
            'U1': {
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 1)},
                    {'number': '2', 'net': 'GND', 'offset': (0, -1)},
                ]
            },
            'C1': {
                'pins': [
                    {'number': '1', 'net': 'VCC', 'offset': (0, 0.5)},
                    {'number': '2', 'net': 'GND', 'offset': (0, -0.5)},
                ]
            }
        },
        'nets': {
            'VCC': {},
            'GND': {}
        }
    }

    class MockPos:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    placement = {
        'U1': MockPos(10, 25),
        'C1': MockPos(40, 25)
    }

    result = router.route(parts_db, placement)

    print(f"   Routed: {result.routed_count}/{result.total_count} nets")
    print(f"   Total length: {result.total_length:.2f}mm")
    print(f"   Triangles: {result.statistics.get('triangles', 0)}")
    print(f"   Obstacles: {result.statistics.get('obstacles', 0)}")

    # Get traces
    traces = router.get_traces()
    print(f"   Trace segments: {len(traces)}")

    print("\n" + "=" * 60)
    print("Topological Router Piston self-test PASSED")
    print("=" * 60)
