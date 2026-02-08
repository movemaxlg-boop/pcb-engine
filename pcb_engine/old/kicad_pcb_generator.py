"""
KiCad PCB File Generator
========================

Generates complete .kicad_pcb files with embedded footprints.
No external libraries required - footprints are defined inline.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import math


@dataclass
class Pad:
    """SMD or THT pad definition"""
    number: str
    x: float  # relative to footprint center
    y: float
    width: float
    height: float
    pad_type: str = "smd"  # smd, thru_hole
    shape: str = "rect"  # rect, circle, oval, roundrect
    drill: float = 0.0  # for THT
    layers: str = "F.Cu F.Paste F.Mask"  # default for SMD


@dataclass
class Footprint:
    """Complete footprint definition"""
    name: str
    pads: List[Pad]
    courtyard_width: float
    courtyard_height: float
    silk_lines: List[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2


# =============================================================================
# FOOTPRINT LIBRARY - Common SMD packages
# =============================================================================

FOOTPRINTS = {
    # Passive components
    "0402": Footprint(
        name="Capacitor_SMD:C_0402_1005Metric",
        pads=[
            Pad("1", -0.48, 0, 0.56, 0.62),
            Pad("2", 0.48, 0, 0.56, 0.62),
        ],
        courtyard_width=1.8,
        courtyard_height=1.0,
        silk_lines=[(-0.1, -0.4, -0.1, 0.4)],
    ),
    "0603": Footprint(
        name="Capacitor_SMD:C_0603_1608Metric",
        pads=[
            Pad("1", -0.775, 0, 0.75, 0.9),
            Pad("2", 0.775, 0, 0.75, 0.9),
        ],
        courtyard_width=2.3,
        courtyard_height=1.3,
    ),
    "0805": Footprint(
        name="Capacitor_SMD:C_0805_2012Metric",
        pads=[
            Pad("1", -0.95, 0, 0.9, 1.25),
            Pad("2", 0.95, 0, 0.9, 1.25),
        ],
        courtyard_width=2.7,
        courtyard_height=1.7,
    ),
    "1206": Footprint(
        name="Capacitor_SMD:C_1206_3216Metric",
        pads=[
            Pad("1", -1.475, 0, 1.05, 1.75),
            Pad("2", 1.475, 0, 1.05, 1.75),
        ],
        courtyard_width=3.8,
        courtyard_height=2.2,
    ),

    # LED
    "LED_0805": Footprint(
        name="LED_SMD:LED_0805_2012Metric",
        pads=[
            Pad("1", -1.05, 0, 0.9, 1.2),  # Cathode
            Pad("2", 1.05, 0, 0.9, 1.2),   # Anode
        ],
        courtyard_width=2.9,
        courtyard_height=1.6,
    ),

    # USB Type-C (simplified 16-pin)
    "USB_C_16": Footprint(
        name="Connector_USB:USB_C_Receptacle_GCT_USB4105",
        pads=[
            # Top row (A side)
            Pad("A1", -2.75, -2.65, 0.3, 1.0),   # GND
            Pad("A4", -1.75, -2.65, 0.3, 1.0),   # VBUS
            Pad("A5", -1.00, -2.65, 0.3, 1.0),   # CC1
            Pad("A6", -0.25, -2.65, 0.3, 1.0),   # D+
            Pad("A7", 0.25, -2.65, 0.3, 1.0),    # D-
            Pad("A8", 1.00, -2.65, 0.3, 1.0),    # SBU1
            Pad("A9", 1.75, -2.65, 0.3, 1.0),    # VBUS
            Pad("A12", 2.75, -2.65, 0.3, 1.0),   # GND
            # Bottom row (B side)
            Pad("B1", -2.75, 2.65, 0.3, 1.0),    # GND
            Pad("B4", -1.75, 2.65, 0.3, 1.0),    # VBUS
            Pad("B5", -1.00, 2.65, 0.3, 1.0),    # CC2
            Pad("B6", -0.25, 2.65, 0.3, 1.0),    # D+
            Pad("B7", 0.25, 2.65, 0.3, 1.0),     # D-
            Pad("B8", 1.00, 2.65, 0.3, 1.0),     # SBU2
            Pad("B9", 1.75, 2.65, 0.3, 1.0),     # VBUS
            Pad("B12", 2.75, 2.65, 0.3, 1.0),    # GND
            # Shield
            Pad("S1", -4.32, -1.3, 1.0, 1.8, "thru_hole", "oval", 0.6, "*.Cu *.Mask"),
            Pad("S2", -4.32, 1.3, 1.0, 1.8, "thru_hole", "oval", 0.6, "*.Cu *.Mask"),
            Pad("S3", 4.32, -1.3, 1.0, 1.8, "thru_hole", "oval", 0.6, "*.Cu *.Mask"),
            Pad("S4", 4.32, 1.3, 1.0, 1.8, "thru_hole", "oval", 0.6, "*.Cu *.Mask"),
        ],
        courtyard_width=10.0,
        courtyard_height=8.0,
    ),

    # QFN packages
    "QFN-32-5x5": Footprint(
        name="Package_DFN_QFN:QFN-32-1EP_5x5mm_P0.5mm_EP3.1x3.1mm",
        pads=_generate_qfn_pads(32, 5.0, 0.5, 0.3, 0.85, 3.1),
        courtyard_width=5.5,
        courtyard_height=5.5,
    ),

    # LGA sensor packages
    "LGA-8_2x2.5": Footprint(
        name="Package_LGA:Bosch_LGA-8_2x2.5mm_P0.65mm",
        pads=[
            Pad("1", -0.975, -0.75, 0.45, 0.35),
            Pad("2", -0.325, -0.75, 0.45, 0.35),
            Pad("3", 0.325, -0.75, 0.45, 0.35),
            Pad("4", 0.975, -0.75, 0.45, 0.35),
            Pad("5", 0.975, 0.75, 0.45, 0.35),
            Pad("6", 0.325, 0.75, 0.45, 0.35),
            Pad("7", -0.325, 0.75, 0.45, 0.35),
            Pad("8", -0.975, 0.75, 0.45, 0.35),
        ],
        courtyard_width=2.5,
        courtyard_height=3.0,
    ),

    # SOT-223 (voltage regulators)
    "SOT-223": Footprint(
        name="Package_TO_SOT_SMD:SOT-223-3_TabPin2",
        pads=[
            Pad("1", -2.3, 3.15, 1.0, 1.5),    # Input
            Pad("2", 0, 3.15, 1.0, 1.5),       # GND (tab)
            Pad("3", 2.3, 3.15, 1.0, 1.5),     # Output
            Pad("2", 0, -3.15, 3.5, 1.5),      # Tab (GND)
        ],
        courtyard_width=7.0,
        courtyard_height=7.5,
    ),

    # SOT-23-5 (small ICs)
    "SOT-23-5": Footprint(
        name="Package_TO_SOT_SMD:SOT-23-5",
        pads=[
            Pad("1", -0.95, 1.1, 0.6, 0.7),
            Pad("2", 0, 1.1, 0.6, 0.7),
            Pad("3", 0.95, 1.1, 0.6, 0.7),
            Pad("4", 0.95, -1.1, 0.6, 0.7),
            Pad("5", -0.95, -1.1, 0.6, 0.7),
        ],
        courtyard_width=2.2,
        courtyard_height=2.8,
    ),
}


def _generate_qfn_pads(pin_count: int, body_size: float, pitch: float,
                       pad_width: float, pad_length: float, ep_size: float) -> List[Pad]:
    """Generate QFN pad array"""
    pads = []
    pins_per_side = pin_count // 4

    # Calculate starting position
    start_offset = (pins_per_side - 1) * pitch / 2
    edge_center = body_size / 2 - pad_length / 2 + 0.3  # Slightly outside body

    pin = 1
    # Bottom side (left to right)
    for i in range(pins_per_side):
        x = -start_offset + i * pitch
        y = edge_center
        pads.append(Pad(str(pin), x, y, pad_width, pad_length))
        pin += 1

    # Right side (bottom to top)
    for i in range(pins_per_side):
        x = edge_center
        y = start_offset - i * pitch
        pads.append(Pad(str(pin), x, y, pad_length, pad_width))
        pin += 1

    # Top side (right to left)
    for i in range(pins_per_side):
        x = start_offset - i * pitch
        y = -edge_center
        pads.append(Pad(str(pin), x, y, pad_width, pad_length))
        pin += 1

    # Left side (top to bottom)
    for i in range(pins_per_side):
        x = -edge_center
        y = -start_offset + i * pitch
        pads.append(Pad(str(pin), x, y, pad_length, pad_width))
        pin += 1

    # Exposed pad
    pads.append(Pad(str(pin), 0, 0, ep_size, ep_size))

    return pads


class KiCadPCBFileGenerator:
    """
    Generates complete .kicad_pcb files with embedded footprints.
    """

    def __init__(self, board_config, design_rules):
        self.board = board_config
        self.rules = design_rules
        self.uuid_counter = 0

    def _uuid(self) -> str:
        """Generate a unique UUID"""
        self.uuid_counter += 1
        return str(uuid.uuid4())

    def generate(self, parts: Dict, placement: Dict, routes: Dict,
                 output_path: str) -> bool:
        """Generate complete .kicad_pcb file"""

        lines = []

        # Build net index for proper track/via net association
        nets = parts.get('nets', {})
        net_index = {"": 0}
        for i, net_name in enumerate(sorted(nets.keys()), start=1):
            net_index[net_name] = i

        # Header
        lines.extend(self._header())

        # Board setup
        lines.extend(self._board_setup())

        # Nets
        lines.extend(self._nets(parts))

        # Footprints
        lines.extend(self._footprints(parts, placement))

        # Tracks (with net IDs)
        lines.extend(self._tracks(routes, net_index))

        # Vias (with net IDs)
        lines.extend(self._vias(routes, net_index))

        # Board outline
        lines.extend(self._board_outline())

        # GND Zone
        lines.extend(self._gnd_zone())

        # Close
        lines.append(")")

        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        return True

    def _header(self) -> List[str]:
        """Generate KiCad PCB header"""
        return [
            f'(kicad_pcb (version 20231014) (generator "pcb_engine")',
            '',
            '  (general',
            f'    (thickness 1.6)',
            '  )',
            '',
        ]

    def _board_setup(self) -> List[str]:
        """Generate board setup section"""
        return [
            '  (paper "A4")',
            '',
            '  (layers',
            '    (0 "F.Cu" signal)',
            '    (31 "B.Cu" signal)',
            '    (32 "B.Adhes" user "B.Adhesive")',
            '    (33 "F.Adhes" user "F.Adhesive")',
            '    (34 "B.Paste" user)',
            '    (35 "F.Paste" user)',
            '    (36 "B.SilkS" user "B.Silkscreen")',
            '    (37 "F.SilkS" user "F.Silkscreen")',
            '    (38 "B.Mask" user)',
            '    (39 "F.Mask" user)',
            '    (40 "Dwgs.User" user "User.Drawings")',
            '    (41 "Cmts.User" user "User.Comments")',
            '    (42 "Eco1.User" user "User.Eco1")',
            '    (43 "Eco2.User" user "User.Eco2")',
            '    (44 "Edge.Cuts" user)',
            '    (45 "Margin" user)',
            '    (46 "B.CrtYd" user "B.Courtyard")',
            '    (47 "F.CrtYd" user "F.Courtyard")',
            '    (48 "B.Fab" user)',
            '    (49 "F.Fab" user)',
            '  )',
            '',
            '  (setup',
            f'    (pad_to_mask_clearance {self.rules.min_clearance})',
            '    (aux_axis_origin 0 0)',
            '    (pcbplotparams',
            '      (layerselection 0x00010fc_ffffffff)',
            '      (plot_on_all_layers_selection 0x0000000_00000000)',
            '    )',
            '  )',
            '',
        ]

    def _nets(self, parts: Dict) -> List[str]:
        """Generate net definitions"""
        lines = ['  ; Nets']
        nets = parts.get('nets', {})

        # Net 0 is always unconnected
        lines.append('  (net 0 "")')

        for i, net_name in enumerate(sorted(nets.keys()), start=1):
            lines.append(f'  (net {i} "{net_name}")')

        lines.append('')
        return lines

    def _footprints(self, parts: Dict, placement: Dict) -> List[str]:
        """Generate footprint definitions"""
        lines = ['  ; Footprints']

        parts_db = parts.get('parts', {})
        nets = parts.get('nets', {})

        # Build net name to index map
        net_index = {"": 0}
        for i, net_name in enumerate(sorted(nets.keys()), start=1):
            net_index[net_name] = i

        for ref, pos in placement.items():
            part = parts_db.get(ref, {})
            package = part.get('footprint', '0402')

            # Get footprint template
            fp_def = FOOTPRINTS.get(package, FOOTPRINTS['0402'])

            x = pos.get('x', 100)
            y = pos.get('y', 100)
            rotation = pos.get('rotation', 0)

            # Generate footprint
            lines.extend(self._single_footprint(ref, part, fp_def, x, y, rotation, net_index))

        lines.append('')
        return lines

    def _single_footprint(self, ref: str, part: Dict, fp_def: Footprint,
                          x: float, y: float, rotation: float,
                          net_index: Dict) -> List[str]:
        """Generate a single footprint"""

        lines = [
            f'  (footprint "{fp_def.name}"',
            f'    (layer "F.Cu")',
            f'    (uuid "{self._uuid()}")',
            f'    (at {x} {y} {rotation})',
        ]

        # Reference designator
        lines.extend([
            f'    (fp_text reference "{ref}"',
            f'      (at 0 -{fp_def.courtyard_height/2 + 1})',
            '      (layer "F.SilkS")',
            f'      (uuid "{self._uuid()}")',
            '      (effects (font (size 0.8 0.8) (thickness 0.15)))',
            '    )',
        ])

        # Value
        value = part.get('value', ref)
        lines.extend([
            f'    (fp_text value "{value}"',
            f'      (at 0 {fp_def.courtyard_height/2 + 1})',
            '      (layer "F.Fab")',
            f'      (uuid "{self._uuid()}")',
            '      (effects (font (size 0.8 0.8) (thickness 0.15)))',
            '    )',
        ])

        # Courtyard
        cw, ch = fp_def.courtyard_width / 2, fp_def.courtyard_height / 2
        lines.extend([
            f'    (fp_rect (start {-cw} {-ch}) (end {cw} {ch})',
            '      (stroke (width 0.05) (type solid))',
            '      (fill none)',
            '      (layer "F.CrtYd")',
            f'      (uuid "{self._uuid()}")',
            '    )',
        ])

        # Pads
        pins = part.get('pins', {})
        for pad in fp_def.pads:
            # Find net for this pad
            net_name = ""
            for pin_num, pin_info in pins.items():
                if str(pin_num) == str(pad.number):
                    net_name = pin_info.get('net', '')
                    break

            net_id = net_index.get(net_name, 0)

            # Pad type
            if pad.pad_type == "thru_hole":
                pad_type = "thru_hole"
                drill = f' (drill {pad.drill})'
            else:
                pad_type = "smd"
                drill = ""

            lines.append(
                f'    (pad "{pad.number}" {pad_type} {pad.shape} '
                f'(at {pad.x} {pad.y}) (size {pad.width} {pad.height}){drill} '
                f'(layers "{pad.layers}") (net {net_id} "{net_name}") '
                f'(uuid "{self._uuid()}"))'
            )

        lines.append('  )')
        return lines

    def _tracks(self, routes: Dict, net_index: Dict) -> List[str]:
        """
        Generate track segments with proper net associations.

        DRC FIX: Tracks must have correct net ID to connect to pads.
        """
        lines = ['  ; Tracks']

        for net_name, route_data in routes.items():
            segments = route_data.get('segments', [])
            net_id = net_index.get(net_name, 0)

            for seg in segments:
                x1, y1 = seg.get('start', (0, 0))
                x2, y2 = seg.get('end', (0, 0))
                width = seg.get('width', self.rules.min_trace_width)
                layer = "F.Cu" if seg.get('layer', 'F.Cu') == 'F.Cu' else "B.Cu"

                lines.append(
                    f'  (segment (start {x1} {y1}) (end {x2} {y2}) '
                    f'(width {width}) (layer "{layer}") (net {net_id}) (uuid "{self._uuid()}"))'
                )

        lines.append('')
        return lines

    def _vias(self, routes: Dict, net_index: Dict) -> List[str]:
        """
        Generate vias with proper net associations.

        DRC FIX: Vias must have correct net ID to connect layers properly.
        """
        lines = ['  ; Vias']

        for net_name, route_data in routes.items():
            vias = route_data.get('vias', [])
            net_id = net_index.get(net_name, 0)

            for via in vias:
                x, y = via.get('x', 0), via.get('y', 0)
                size = via.get('size', self.rules.min_via_diameter)
                drill = via.get('drill', self.rules.min_via_drill)

                lines.append(
                    f'  (via (at {x} {y}) (size {size}) (drill {drill}) '
                    f'(layers "F.Cu" "B.Cu") (net {net_id}) (uuid "{self._uuid()}"))'
                )

        lines.append('')
        return lines

    def _board_outline(self) -> List[str]:
        """Generate board outline"""
        ox, oy = self.board.origin_x, self.board.origin_y
        w, h = self.board.width, self.board.height

        return [
            '  ; Board outline',
            f'  (gr_rect (start {ox} {oy}) (end {ox + w} {oy + h})',
            '    (stroke (width 0.15) (type solid))',
            '    (fill none)',
            '    (layer "Edge.Cuts")',
            f'    (uuid "{self._uuid()}")',
            '  )',
            '',
        ]

    def _gnd_zone(self) -> List[str]:
        """Generate GND zone on bottom layer"""
        ox, oy = self.board.origin_x, self.board.origin_y
        w, h = self.board.width, self.board.height
        margin = 1.0

        return [
            '  ; GND Zone',
            '  (zone',
            f'    (net 0) (net_name "GND")',
            '    (layer "B.Cu")',
            f'    (uuid "{self._uuid()}")',
            '    (hatch edge 0.5)',
            '    (connect_pads (clearance 0.2))',
            f'    (min_thickness {self.rules.min_trace_width})',
            '    (filled_areas_thickness no)',
            '    (fill yes (thermal_gap 0.3) (thermal_bridge_width 0.3))',
            '    (polygon',
            '      (pts',
            f'        (xy {ox + margin} {oy + margin})',
            f'        (xy {ox + w - margin} {oy + margin})',
            f'        (xy {ox + w - margin} {oy + h - margin})',
            f'        (xy {ox + margin} {oy + h - margin})',
            '      )',
            '    )',
            '  )',
            '',
        ]


def generate_kicad_pcb(engine, output_path: str) -> bool:
    """
    Generate a complete .kicad_pcb file from engine state.

    Args:
        engine: PCBEngine instance with completed run
        output_path: Path for output .kicad_pcb file

    Returns:
        True if successful
    """
    generator = KiCadPCBFileGenerator(engine.board, engine.rules)

    return generator.generate(
        parts=engine.state.parts_db,
        placement=engine.state.placement,
        routes=engine.state.routes,
        output_path=output_path
    )
