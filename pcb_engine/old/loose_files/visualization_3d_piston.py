"""
3D Visualization Piston - PCB Engine Worker

Generates 3D visualizations of PCB designs including:
- STEP file export for mechanical CAD integration
- STL mesh export for 3D printing enclosures
- Interactive 3D preview data
- Component height maps
- Clearance visualization

HIERARCHY:
    USER (Boss) → Circuit AI (Engineer) → PCB Engine (Foreman) → This Piston (Worker)

This piston receives work orders from the Foreman (PCB Engine) and reports back
with results, warnings, and any issues encountered.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum, auto
import math
import json


class OutputFormat(Enum):
    """Supported 3D output formats"""
    STEP = auto()      # STEP file for mechanical CAD
    STL = auto()       # STL mesh for 3D printing
    OBJ = auto()       # Wavefront OBJ for visualization
    GLTF = auto()      # glTF for web viewers
    JSON_3D = auto()   # JSON for custom viewers


class ComponentShape(Enum):
    """Standard component 3D shapes"""
    BOX = auto()           # Rectangular box (resistors, caps)
    CYLINDER = auto()      # Cylindrical (electrolytic caps)
    IC_PACKAGE = auto()    # IC with pins
    CONNECTOR = auto()     # Connector housing
    THROUGH_HOLE = auto()  # Through-hole component
    BGA = auto()           # Ball grid array
    QFP = auto()           # Quad flat package
    SOT = auto()           # Small outline transistor
    CUSTOM = auto()        # Custom 3D model


@dataclass
class Point3D:
    """3D point with x, y, z coordinates"""
    x: float
    y: float
    z: float

    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


@dataclass
class Triangle3D:
    """3D triangle for mesh representation"""
    v1: Point3D
    v2: Point3D
    v3: Point3D
    normal: Optional[Point3D] = None

    def __post_init__(self):
        if self.normal is None:
            self.normal = self._calculate_normal()

    def _calculate_normal(self) -> Point3D:
        """Calculate surface normal using cross product"""
        u = self.v2 - self.v1
        v = self.v3 - self.v1

        nx = u.y * v.z - u.z * v.y
        ny = u.z * v.x - u.x * v.z
        nz = u.x * v.y - u.y * v.x

        # Normalize
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 0:
            nx /= length
            ny /= length
            nz /= length

        return Point3D(nx, ny, nz)


@dataclass
class Mesh3D:
    """3D mesh composed of triangles"""
    triangles: List[Triangle3D] = field(default_factory=list)
    name: str = ""
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # RGB 0-1

    def add_triangle(self, v1: Point3D, v2: Point3D, v3: Point3D):
        self.triangles.append(Triangle3D(v1, v2, v3))

    def vertex_count(self) -> int:
        return len(self.triangles) * 3

    def triangle_count(self) -> int:
        return len(self.triangles)


@dataclass
class Component3D:
    """3D representation of a component"""
    name: str
    designator: str
    position: Point3D
    rotation: float  # Degrees around Z axis
    shape: ComponentShape
    dimensions: Tuple[float, float, float]  # width, depth, height in mm
    mesh: Optional[Mesh3D] = None
    pins: List[Point3D] = field(default_factory=list)
    color: Tuple[float, float, float] = (0.2, 0.2, 0.2)

    def generate_mesh(self) -> Mesh3D:
        """Generate mesh based on component shape"""
        if self.shape == ComponentShape.BOX:
            return self._generate_box_mesh()
        elif self.shape == ComponentShape.CYLINDER:
            return self._generate_cylinder_mesh()
        elif self.shape == ComponentShape.IC_PACKAGE:
            return self._generate_ic_mesh()
        else:
            return self._generate_box_mesh()  # Default to box

    def _generate_box_mesh(self) -> Mesh3D:
        """Generate box mesh for component"""
        mesh = Mesh3D(name=self.designator, color=self.color)

        w, d, h = self.dimensions
        hw, hd = w / 2, d / 2

        # Apply rotation
        cos_r = math.cos(math.radians(self.rotation))
        sin_r = math.sin(math.radians(self.rotation))

        def rotate_point(x: float, y: float) -> Tuple[float, float]:
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            return rx, ry

        # 8 corners of the box
        corners_local = [
            (-hw, -hd, 0), (hw, -hd, 0), (hw, hd, 0), (-hw, hd, 0),  # Bottom
            (-hw, -hd, h), (hw, -hd, h), (hw, hd, h), (-hw, hd, h),  # Top
        ]

        corners = []
        for x, y, z in corners_local:
            rx, ry = rotate_point(x, y)
            corners.append(Point3D(
                self.position.x + rx,
                self.position.y + ry,
                self.position.z + z
            ))

        # 12 triangles for 6 faces (2 triangles per face)
        faces = [
            (0, 1, 2), (0, 2, 3),  # Bottom
            (4, 6, 5), (4, 7, 6),  # Top
            (0, 4, 1), (1, 4, 5),  # Front
            (2, 6, 3), (3, 6, 7),  # Back
            (0, 3, 4), (3, 7, 4),  # Left
            (1, 5, 2), (2, 5, 6),  # Right
        ]

        for i1, i2, i3 in faces:
            mesh.add_triangle(corners[i1], corners[i2], corners[i3])

        return mesh

    def _generate_cylinder_mesh(self, segments: int = 16) -> Mesh3D:
        """Generate cylinder mesh for component"""
        mesh = Mesh3D(name=self.designator, color=self.color)

        radius = min(self.dimensions[0], self.dimensions[1]) / 2
        height = self.dimensions[2]

        # Generate circle points
        bottom_points = []
        top_points = []

        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = self.position.x + radius * math.cos(angle)
            y = self.position.y + radius * math.sin(angle)
            bottom_points.append(Point3D(x, y, self.position.z))
            top_points.append(Point3D(x, y, self.position.z + height))

        center_bottom = Point3D(self.position.x, self.position.y, self.position.z)
        center_top = Point3D(self.position.x, self.position.y, self.position.z + height)

        # Bottom cap
        for i in range(segments):
            mesh.add_triangle(center_bottom, bottom_points[(i+1) % segments], bottom_points[i])

        # Top cap
        for i in range(segments):
            mesh.add_triangle(center_top, top_points[i], top_points[(i+1) % segments])

        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            mesh.add_triangle(bottom_points[i], bottom_points[next_i], top_points[i])
            mesh.add_triangle(bottom_points[next_i], top_points[next_i], top_points[i])

        return mesh

    def _generate_ic_mesh(self) -> Mesh3D:
        """Generate IC package mesh with body and pins"""
        mesh = Mesh3D(name=self.designator, color=self.color)

        # IC body (box with slightly raised edges)
        body_mesh = self._generate_box_mesh()
        mesh.triangles.extend(body_mesh.triangles)

        # Add pin representations if pins are defined
        for pin_pos in self.pins:
            pin_mesh = self._generate_pin_mesh(pin_pos)
            mesh.triangles.extend(pin_mesh.triangles)

        return mesh

    def _generate_pin_mesh(self, pin_pos: Point3D) -> Mesh3D:
        """Generate small box for pin"""
        pin_size = 0.3  # mm
        mesh = Mesh3D(color=(0.8, 0.8, 0.8))  # Silver color

        corners = [
            Point3D(pin_pos.x - pin_size/2, pin_pos.y - pin_size/2, pin_pos.z),
            Point3D(pin_pos.x + pin_size/2, pin_pos.y - pin_size/2, pin_pos.z),
            Point3D(pin_pos.x + pin_size/2, pin_pos.y + pin_size/2, pin_pos.z),
            Point3D(pin_pos.x - pin_size/2, pin_pos.y + pin_size/2, pin_pos.z),
            Point3D(pin_pos.x - pin_size/2, pin_pos.y - pin_size/2, pin_pos.z + 0.2),
            Point3D(pin_pos.x + pin_size/2, pin_pos.y - pin_size/2, pin_pos.z + 0.2),
            Point3D(pin_pos.x + pin_size/2, pin_pos.y + pin_size/2, pin_pos.z + 0.2),
            Point3D(pin_pos.x - pin_size/2, pin_pos.y + pin_size/2, pin_pos.z + 0.2),
        ]

        faces = [
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (2, 3, 7), (2, 7, 6),
            (0, 4, 7), (0, 7, 3),
            (1, 2, 6), (1, 6, 5),
        ]

        for i1, i2, i3 in faces:
            mesh.add_triangle(corners[i1], corners[i2], corners[i3])

        return mesh


@dataclass
class PCBBoard3D:
    """3D representation of the PCB board itself"""
    width: float  # mm
    height: float  # mm
    thickness: float = 1.6  # mm (standard PCB thickness)
    layer_count: int = 2
    origin: Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    color: Tuple[float, float, float] = (0.0, 0.4, 0.0)  # PCB green

    def generate_mesh(self) -> Mesh3D:
        """Generate PCB board mesh"""
        mesh = Mesh3D(name="PCB_Board", color=self.color)

        # Simple box for PCB
        corners = [
            Point3D(0, 0, -self.thickness),
            Point3D(self.width, 0, -self.thickness),
            Point3D(self.width, self.height, -self.thickness),
            Point3D(0, self.height, -self.thickness),
            Point3D(0, 0, 0),
            Point3D(self.width, 0, 0),
            Point3D(self.width, self.height, 0),
            Point3D(0, self.height, 0),
        ]

        faces = [
            (0, 1, 2), (0, 2, 3),  # Bottom
            (4, 6, 5), (4, 7, 6),  # Top
            (0, 4, 1), (1, 4, 5),  # Front
            (2, 6, 3), (3, 6, 7),  # Back
            (0, 3, 4), (3, 7, 4),  # Left
            (1, 5, 2), (2, 5, 6),  # Right
        ]

        for i1, i2, i3 in faces:
            mesh.add_triangle(corners[i1], corners[i2], corners[i3])

        return mesh


@dataclass
class Trace3D:
    """3D representation of a PCB trace"""
    points: List[Point3D]
    width: float
    layer: int
    net_name: str = ""

    def generate_mesh(self, layer_z: float) -> Mesh3D:
        """Generate mesh for trace (extruded path)"""
        mesh = Mesh3D(name=f"Trace_{self.net_name}", color=(0.8, 0.6, 0.2))  # Copper color

        trace_height = 0.035  # 1oz copper = 35 microns

        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]

            # Calculate perpendicular direction
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = math.sqrt(dx*dx + dy*dy)

            if length < 0.001:
                continue

            # Perpendicular vector (normalized)
            px = -dy / length * self.width / 2
            py = dx / length * self.width / 2

            # 4 corners of trace segment
            corners = [
                Point3D(p1.x + px, p1.y + py, layer_z),
                Point3D(p1.x - px, p1.y - py, layer_z),
                Point3D(p2.x - px, p2.y - py, layer_z),
                Point3D(p2.x + px, p2.y + py, layer_z),
                Point3D(p1.x + px, p1.y + py, layer_z + trace_height),
                Point3D(p1.x - px, p1.y - py, layer_z + trace_height),
                Point3D(p2.x - px, p2.y - py, layer_z + trace_height),
                Point3D(p2.x + px, p2.y + py, layer_z + trace_height),
            ]

            # Add faces
            faces = [
                (0, 1, 2), (0, 2, 3),  # Bottom
                (4, 6, 5), (4, 7, 6),  # Top
                (0, 4, 5), (0, 5, 1),  # Side 1
                (2, 6, 7), (2, 7, 3),  # Side 2
            ]

            for i1, i2, i3 in faces:
                mesh.add_triangle(corners[i1], corners[i2], corners[i3])

        return mesh


@dataclass
class Via3D:
    """3D representation of a via"""
    position: Point3D
    drill_diameter: float
    pad_diameter: float
    start_layer: int
    end_layer: int

    def generate_mesh(self, layer_heights: Dict[int, float], segments: int = 8) -> Mesh3D:
        """Generate via mesh (hollow cylinder)"""
        mesh = Mesh3D(name="Via", color=(0.8, 0.6, 0.2))  # Copper color

        # Get Z range
        z_start = layer_heights.get(self.start_layer, 0)
        z_end = layer_heights.get(self.end_layer, -1.6)

        outer_r = self.pad_diameter / 2
        inner_r = self.drill_diameter / 2

        # Generate points for outer and inner circles
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments

            # Outer surface
            o1_bottom = Point3D(
                self.position.x + outer_r * math.cos(angle1),
                self.position.y + outer_r * math.sin(angle1),
                z_end
            )
            o2_bottom = Point3D(
                self.position.x + outer_r * math.cos(angle2),
                self.position.y + outer_r * math.sin(angle2),
                z_end
            )
            o1_top = Point3D(o1_bottom.x, o1_bottom.y, z_start)
            o2_top = Point3D(o2_bottom.x, o2_bottom.y, z_start)

            mesh.add_triangle(o1_bottom, o2_bottom, o1_top)
            mesh.add_triangle(o2_bottom, o2_top, o1_top)

            # Inner surface (drill hole)
            i1_bottom = Point3D(
                self.position.x + inner_r * math.cos(angle1),
                self.position.y + inner_r * math.sin(angle1),
                z_end
            )
            i2_bottom = Point3D(
                self.position.x + inner_r * math.cos(angle2),
                self.position.y + inner_r * math.sin(angle2),
                z_end
            )
            i1_top = Point3D(i1_bottom.x, i1_bottom.y, z_start)
            i2_top = Point3D(i2_bottom.x, i2_bottom.y, z_start)

            mesh.add_triangle(i1_top, i2_bottom, i1_bottom)
            mesh.add_triangle(i1_top, i2_top, i2_bottom)

            # Top annular ring
            mesh.add_triangle(o1_top, o2_top, i1_top)
            mesh.add_triangle(o2_top, i2_top, i1_top)

            # Bottom annular ring
            mesh.add_triangle(o1_bottom, i1_bottom, o2_bottom)
            mesh.add_triangle(i1_bottom, i2_bottom, o2_bottom)

        return mesh


@dataclass
class Scene3D:
    """Complete 3D scene for PCB visualization"""
    board: PCBBoard3D
    components: List[Component3D] = field(default_factory=list)
    traces: List[Trace3D] = field(default_factory=list)
    vias: List[Via3D] = field(default_factory=list)

    def get_all_meshes(self) -> List[Mesh3D]:
        """Get all meshes in the scene"""
        meshes = [self.board.generate_mesh()]

        layer_heights = self._calculate_layer_heights()

        for comp in self.components:
            if comp.mesh is None:
                comp.mesh = comp.generate_mesh()
            meshes.append(comp.mesh)

        for trace in self.traces:
            layer_z = layer_heights.get(trace.layer, 0)
            meshes.append(trace.generate_mesh(layer_z))

        for via in self.vias:
            meshes.append(via.generate_mesh(layer_heights))

        return meshes

    def _calculate_layer_heights(self) -> Dict[int, float]:
        """Calculate Z height for each layer"""
        heights = {}
        layer_thickness = self.board.thickness / (self.board.layer_count - 1) if self.board.layer_count > 1 else self.board.thickness

        for i in range(self.board.layer_count):
            heights[i] = -i * layer_thickness

        return heights

    def get_bounding_box(self) -> Tuple[Point3D, Point3D]:
        """Get bounding box of entire scene"""
        min_pt = Point3D(0, 0, -self.board.thickness)
        max_pt = Point3D(self.board.width, self.board.height, 0)

        for comp in self.components:
            h = comp.dimensions[2] if comp.dimensions else 0
            max_pt = Point3D(
                max(max_pt.x, comp.position.x),
                max(max_pt.y, comp.position.y),
                max(max_pt.z, comp.position.z + h)
            )

        return min_pt, max_pt


class Visualization3DPiston:
    """
    3D Visualization Piston - Generates 3D models and previews

    Capabilities:
    - Generate STEP files for mechanical CAD
    - Generate STL files for 3D printing
    - Generate OBJ/glTF for web visualization
    - Component height analysis
    - Clearance checking
    - Enclosure fit validation
    """

    def __init__(self):
        self.name = "Visualization3D"
        self.version = "1.0.0"

        # Component library (shape/dimension mappings)
        self.component_shapes: Dict[str, Tuple[ComponentShape, Tuple[float, float, float]]] = {
            # Resistors
            "0402": (ComponentShape.BOX, (1.0, 0.5, 0.35)),
            "0603": (ComponentShape.BOX, (1.6, 0.8, 0.45)),
            "0805": (ComponentShape.BOX, (2.0, 1.25, 0.6)),
            "1206": (ComponentShape.BOX, (3.2, 1.6, 0.6)),

            # Capacitors
            "CAP_0402": (ComponentShape.BOX, (1.0, 0.5, 0.5)),
            "CAP_0603": (ComponentShape.BOX, (1.6, 0.8, 0.8)),
            "CAP_0805": (ComponentShape.BOX, (2.0, 1.25, 1.25)),
            "CAP_ELEC_6.3x5.4": (ComponentShape.CYLINDER, (6.3, 6.3, 5.4)),
            "CAP_ELEC_8x10": (ComponentShape.CYLINDER, (8.0, 8.0, 10.0)),

            # ICs
            "SOIC-8": (ComponentShape.IC_PACKAGE, (5.0, 4.0, 1.75)),
            "SOIC-14": (ComponentShape.IC_PACKAGE, (8.65, 4.0, 1.75)),
            "SOIC-16": (ComponentShape.IC_PACKAGE, (10.0, 4.0, 1.75)),
            "TQFP-32": (ComponentShape.QFP, (7.0, 7.0, 1.0)),
            "TQFP-44": (ComponentShape.QFP, (10.0, 10.0, 1.0)),
            "TQFP-64": (ComponentShape.QFP, (12.0, 12.0, 1.0)),
            "QFN-32": (ComponentShape.BOX, (5.0, 5.0, 0.85)),
            "QFN-48": (ComponentShape.BOX, (7.0, 7.0, 0.85)),
            "BGA-256": (ComponentShape.BGA, (17.0, 17.0, 1.2)),

            # Connectors
            "USB_C": (ComponentShape.CONNECTOR, (8.94, 7.3, 3.26)),
            "USB_MICRO": (ComponentShape.CONNECTOR, (7.5, 5.0, 2.7)),
            "HEADER_2x5": (ComponentShape.CONNECTOR, (12.7, 5.08, 8.5)),
            "HEADER_1x4": (ComponentShape.CONNECTOR, (10.16, 2.54, 8.5)),

            # Transistors
            "SOT-23": (ComponentShape.SOT, (2.9, 1.3, 1.1)),
            "SOT-223": (ComponentShape.SOT, (6.5, 3.5, 1.8)),
        }

    def generate(self, parts_db: Dict, placement: Dict, routes: Dict) -> Dict[str, Any]:
        """
        Standard piston API - generate 3D visualization data.

        Args:
            parts_db: Parts database
            placement: Component placements
            routes: Routing data

        Returns:
            Dictionary with 3D visualization data
        """
        # Build design data from inputs
        design_data = {
            'parts': parts_db.get('parts', {}),
            'placement': placement,
            'board_width': 50.0,
            'board_height': 40.0,
            'board_thickness': 1.6
        }

        return self.visualize(design_data)

    def visualize(self,
                  design_data: Dict[str, Any],
                  output_format: OutputFormat = OutputFormat.STL,
                  include_traces: bool = True,
                  include_vias: bool = True) -> Dict[str, Any]:
        """
        Main entry point - generate 3D visualization

        Args:
            design_data: PCB design data from engine
            output_format: Desired output format
            include_traces: Whether to include copper traces
            include_vias: Whether to include vias

        Returns:
            Dict with 'success', 'output', 'warnings', 'stats'
        """
        warnings = []

        # Build 3D scene from design data
        scene = self._build_scene(design_data, include_traces, include_vias, warnings)

        # Generate output in requested format
        if output_format == OutputFormat.STL:
            output = self._export_stl(scene)
        elif output_format == OutputFormat.OBJ:
            output = self._export_obj(scene)
        elif output_format == OutputFormat.GLTF:
            output = self._export_gltf(scene)
        elif output_format == OutputFormat.JSON_3D:
            output = self._export_json(scene)
        elif output_format == OutputFormat.STEP:
            output, step_warnings = self._export_step(scene)
            warnings.extend(step_warnings)
        else:
            output = self._export_stl(scene)  # Default

        # Calculate statistics
        stats = self._calculate_stats(scene)

        return {
            'success': True,
            'output': output,
            'format': output_format.name,
            'warnings': warnings,
            'stats': stats
        }

    def _build_scene(self,
                     design_data: Dict[str, Any],
                     include_traces: bool,
                     include_vias: bool,
                     warnings: List[str]) -> Scene3D:
        """Build 3D scene from design data"""

        # Extract board dimensions
        board_width = design_data.get('board_width', 100)
        board_height = design_data.get('board_height', 100)
        board_thickness = design_data.get('board_thickness', 1.6)
        layer_count = design_data.get('layer_count', 2)

        board = PCBBoard3D(
            width=board_width,
            height=board_height,
            thickness=board_thickness,
            layer_count=layer_count
        )

        scene = Scene3D(board=board)

        # Add components
        components_data = design_data.get('components', [])
        for comp_data in components_data:
            comp_3d = self._create_component_3d(comp_data, warnings)
            if comp_3d:
                scene.components.append(comp_3d)

        # Add traces
        if include_traces:
            traces_data = design_data.get('traces', [])
            for trace_data in traces_data:
                trace_3d = self._create_trace_3d(trace_data)
                if trace_3d:
                    scene.traces.append(trace_3d)

        # Add vias
        if include_vias:
            vias_data = design_data.get('vias', [])
            for via_data in vias_data:
                via_3d = self._create_via_3d(via_data)
                if via_3d:
                    scene.vias.append(via_3d)

        return scene

    def _create_component_3d(self,
                             comp_data: Dict[str, Any],
                             warnings: List[str]) -> Optional[Component3D]:
        """Create 3D component from data"""

        footprint = comp_data.get('footprint', '')
        designator = comp_data.get('designator', 'U?')
        name = comp_data.get('name', '')
        x = comp_data.get('x', 0)
        y = comp_data.get('y', 0)
        rotation = comp_data.get('rotation', 0)

        # Look up shape and dimensions
        shape, dimensions = self._get_component_shape(footprint, comp_data)

        if dimensions is None:
            warnings.append(f"Unknown footprint for {designator}: {footprint}, using default box")
            shape = ComponentShape.BOX
            dimensions = (2.0, 1.0, 1.0)

        return Component3D(
            name=name,
            designator=designator,
            position=Point3D(x, y, 0),
            rotation=rotation,
            shape=shape,
            dimensions=dimensions
        )

    def _get_component_shape(self,
                             footprint: str,
                             comp_data: Dict[str, Any]) -> Tuple[ComponentShape, Optional[Tuple[float, float, float]]]:
        """Get component shape and dimensions from footprint"""

        # Direct lookup
        if footprint in self.component_shapes:
            return self.component_shapes[footprint]

        # Try partial match
        for key, value in self.component_shapes.items():
            if key in footprint or footprint in key:
                return value

        # Check if dimensions are provided in data
        if 'dimensions' in comp_data:
            dims = comp_data['dimensions']
            if isinstance(dims, (list, tuple)) and len(dims) >= 3:
                return (ComponentShape.BOX, tuple(dims[:3]))

        # Infer from footprint name
        if 'QFP' in footprint or 'QFN' in footprint:
            return (ComponentShape.QFP, None)
        elif 'SOIC' in footprint or 'SOP' in footprint:
            return (ComponentShape.IC_PACKAGE, None)
        elif 'SOT' in footprint:
            return (ComponentShape.SOT, None)
        elif 'BGA' in footprint:
            return (ComponentShape.BGA, None)
        elif 'USB' in footprint or 'CONN' in footprint:
            return (ComponentShape.CONNECTOR, None)
        elif 'CAP' in footprint and 'ELEC' in footprint:
            return (ComponentShape.CYLINDER, None)

        return (ComponentShape.BOX, None)

    def _create_trace_3d(self, trace_data: Dict[str, Any]) -> Optional[Trace3D]:
        """Create 3D trace from data"""
        points_data = trace_data.get('points', [])
        if len(points_data) < 2:
            return None

        points = [Point3D(p[0], p[1], 0) for p in points_data]

        return Trace3D(
            points=points,
            width=trace_data.get('width', 0.2),
            layer=trace_data.get('layer', 0),
            net_name=trace_data.get('net', '')
        )

    def _create_via_3d(self, via_data: Dict[str, Any]) -> Optional[Via3D]:
        """Create 3D via from data"""
        return Via3D(
            position=Point3D(via_data.get('x', 0), via_data.get('y', 0), 0),
            drill_diameter=via_data.get('drill', 0.3),
            pad_diameter=via_data.get('pad', 0.6),
            start_layer=via_data.get('start_layer', 0),
            end_layer=via_data.get('end_layer', 1)
        )

    def _export_stl(self, scene: Scene3D) -> str:
        """Export scene as STL format"""
        lines = ["solid PCB_Model"]

        meshes = scene.get_all_meshes()

        for mesh in meshes:
            for tri in mesh.triangles:
                n = tri.normal
                lines.append(f"  facet normal {n.x:.6f} {n.y:.6f} {n.z:.6f}")
                lines.append("    outer loop")
                lines.append(f"      vertex {tri.v1.x:.6f} {tri.v1.y:.6f} {tri.v1.z:.6f}")
                lines.append(f"      vertex {tri.v2.x:.6f} {tri.v2.y:.6f} {tri.v2.z:.6f}")
                lines.append(f"      vertex {tri.v3.x:.6f} {tri.v3.y:.6f} {tri.v3.z:.6f}")
                lines.append("    endloop")
                lines.append("  endfacet")

        lines.append("endsolid PCB_Model")

        return "\n".join(lines)

    def _export_obj(self, scene: Scene3D) -> str:
        """Export scene as Wavefront OBJ format"""
        lines = ["# PCB 3D Model", "# Generated by PCB Engine Visualization3D Piston", ""]

        meshes = scene.get_all_meshes()
        vertex_offset = 0

        for mesh in meshes:
            lines.append(f"o {mesh.name or 'Object'}")

            # Collect unique vertices
            vertices = []
            for tri in mesh.triangles:
                vertices.extend([tri.v1, tri.v2, tri.v3])

            # Write vertices
            for v in vertices:
                lines.append(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}")

            # Write faces
            for i in range(0, len(vertices), 3):
                v1 = vertex_offset + i + 1
                v2 = vertex_offset + i + 2
                v3 = vertex_offset + i + 3
                lines.append(f"f {v1} {v2} {v3}")

            vertex_offset += len(vertices)
            lines.append("")

        return "\n".join(lines)

    def _export_gltf(self, scene: Scene3D) -> str:
        """Export scene as glTF JSON format"""
        meshes = scene.get_all_meshes()

        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "PCB Engine Visualization3D Piston"
            },
            "scene": 0,
            "scenes": [{"nodes": list(range(len(meshes)))}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": []
        }

        for i, mesh in enumerate(meshes):
            gltf["nodes"].append({
                "mesh": i,
                "name": mesh.name or f"Mesh_{i}"
            })

            gltf["meshes"].append({
                "name": mesh.name or f"Mesh_{i}",
                "primitives": [{
                    "mode": 4,  # TRIANGLES
                    "attributes": {"POSITION": i}
                }]
            })

        return json.dumps(gltf, indent=2)

    def _export_json(self, scene: Scene3D) -> str:
        """Export scene as JSON for custom viewers"""
        data = {
            "board": {
                "width": scene.board.width,
                "height": scene.board.height,
                "thickness": scene.board.thickness,
                "layers": scene.board.layer_count
            },
            "components": [],
            "traces": [],
            "vias": []
        }

        for comp in scene.components:
            data["components"].append({
                "name": comp.name,
                "designator": comp.designator,
                "position": [comp.position.x, comp.position.y, comp.position.z],
                "rotation": comp.rotation,
                "dimensions": list(comp.dimensions)
            })

        for trace in scene.traces:
            data["traces"].append({
                "points": [[p.x, p.y] for p in trace.points],
                "width": trace.width,
                "layer": trace.layer,
                "net": trace.net_name
            })

        for via in scene.vias:
            data["vias"].append({
                "position": [via.position.x, via.position.y],
                "drill": via.drill_diameter,
                "pad": via.pad_diameter,
                "layers": [via.start_layer, via.end_layer]
            })

        return json.dumps(data, indent=2)

    def _export_step(self, scene: Scene3D) -> Tuple[str, List[str]]:
        """
        Export scene as STEP format

        Note: Full STEP export requires specialized libraries (OCC/FreeCAD)
        This generates a simplified STEP-like structure
        """
        warnings = ["STEP export is simplified - for full STEP, use OpenCASCADE integration"]

        # Generate simplified STEP header
        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('PCB 3D Model'),'2;1');
FILE_NAME('pcb_model.step','2024-01-01',('PCB Engine'),(''),'',' ','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
/* Simplified STEP - Component positions and bounding boxes */
"""
        entity_id = 1

        # Add board
        step_content += f"#{entity_id}=PRODUCT('PCB_Board','PCB Board','',(#2));\n"
        entity_id += 1

        # Add components as simplified shapes
        for comp in scene.components:
            step_content += f"#{entity_id}=PRODUCT('{comp.designator}','{comp.name}','',(#2));\n"
            entity_id += 1
            step_content += f"/* Position: ({comp.position.x},{comp.position.y},{comp.position.z}) */\n"
            step_content += f"/* Dimensions: {comp.dimensions} */\n"

        step_content += "ENDSEC;\nEND-ISO-10303-21;\n"

        return step_content, warnings

    def _calculate_stats(self, scene: Scene3D) -> Dict[str, Any]:
        """Calculate statistics about the 3D scene"""
        meshes = scene.get_all_meshes()

        total_triangles = sum(mesh.triangle_count() for mesh in meshes)
        total_vertices = total_triangles * 3

        min_pt, max_pt = scene.get_bounding_box()

        # Find tallest component
        max_height = 0
        tallest_component = None
        for comp in scene.components:
            h = comp.dimensions[2] if comp.dimensions else 0
            if h > max_height:
                max_height = h
                tallest_component = comp.designator

        return {
            'total_meshes': len(meshes),
            'total_triangles': total_triangles,
            'total_vertices': total_vertices,
            'component_count': len(scene.components),
            'trace_count': len(scene.traces),
            'via_count': len(scene.vias),
            'bounding_box': {
                'min': min_pt.to_tuple(),
                'max': max_pt.to_tuple(),
                'size': (max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z)
            },
            'max_component_height': max_height,
            'tallest_component': tallest_component
        }

    def check_enclosure_fit(self,
                           scene: Scene3D,
                           enclosure_dims: Tuple[float, float, float],
                           clearance: float = 1.0) -> Dict[str, Any]:
        """
        Check if PCB fits in enclosure with clearance

        Args:
            scene: 3D scene
            enclosure_dims: (width, depth, height) in mm
            clearance: Required clearance in mm

        Returns:
            Fit analysis results
        """
        min_pt, max_pt = scene.get_bounding_box()

        pcb_width = max_pt.x - min_pt.x
        pcb_depth = max_pt.y - min_pt.y
        pcb_height = max_pt.z - min_pt.z

        enc_width, enc_depth, enc_height = enclosure_dims

        # Check fit with clearance
        width_ok = pcb_width + 2 * clearance <= enc_width
        depth_ok = pcb_depth + 2 * clearance <= enc_depth
        height_ok = pcb_height + clearance <= enc_height  # Only top clearance needed

        fits = width_ok and depth_ok and height_ok

        return {
            'fits': fits,
            'pcb_size': (pcb_width, pcb_depth, pcb_height),
            'enclosure_size': enclosure_dims,
            'clearance_required': clearance,
            'width_margin': enc_width - pcb_width - 2 * clearance,
            'depth_margin': enc_depth - pcb_depth - 2 * clearance,
            'height_margin': enc_height - pcb_height - clearance,
            'issues': [
                issue for issue, ok in [
                    (f"Width too large by {pcb_width + 2*clearance - enc_width:.2f}mm", not width_ok),
                    (f"Depth too large by {pcb_depth + 2*clearance - enc_depth:.2f}mm", not depth_ok),
                    (f"Height too large by {pcb_height + clearance - enc_height:.2f}mm", not height_ok),
                ] if not ok
            ]
        }

    def get_height_map(self, scene: Scene3D, resolution: float = 1.0) -> Dict[str, Any]:
        """
        Generate height map of component placement

        Args:
            scene: 3D scene
            resolution: Grid resolution in mm

        Returns:
            Height map data
        """
        grid_width = int(scene.board.width / resolution) + 1
        grid_height = int(scene.board.height / resolution) + 1

        height_map = [[0.0] * grid_width for _ in range(grid_height)]

        for comp in scene.components:
            if not comp.dimensions:
                continue

            w, d, h = comp.dimensions
            cx, cy = comp.position.x, comp.position.y

            # Calculate grid cells covered by component
            x_start = max(0, int((cx - w/2) / resolution))
            x_end = min(grid_width, int((cx + w/2) / resolution) + 1)
            y_start = max(0, int((cy - d/2) / resolution))
            y_end = min(grid_height, int((cy + d/2) / resolution) + 1)

            for gy in range(y_start, y_end):
                for gx in range(x_start, x_end):
                    height_map[gy][gx] = max(height_map[gy][gx], h)

        return {
            'grid': height_map,
            'resolution': resolution,
            'width': grid_width,
            'height': grid_height,
            'max_height': max(max(row) for row in height_map)
        }


# Convenience function for direct use
def create_3d_visualization(design_data: Dict[str, Any],
                           output_format: str = "STL") -> Dict[str, Any]:
    """
    Create 3D visualization from design data

    Args:
        design_data: PCB design data
        output_format: "STL", "OBJ", "GLTF", "JSON", or "STEP"

    Returns:
        Visualization result with output data
    """
    piston = Visualization3DPiston()

    format_map = {
        "STL": OutputFormat.STL,
        "OBJ": OutputFormat.OBJ,
        "GLTF": OutputFormat.GLTF,
        "JSON": OutputFormat.JSON_3D,
        "STEP": OutputFormat.STEP,
    }

    fmt = format_map.get(output_format.upper(), OutputFormat.STL)

    return piston.visualize(design_data, fmt)
