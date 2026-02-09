"""
PCB Engine - Output Piston
===========================

A dedicated piston (sub-engine) for generating manufacturing outputs.

This piston handles:
1. KiCad PCB File Generation - .kicad_pcb format
2. Gerber Generation - Manufacturing layers
3. Drill File Generation - Excellon format
4. BOM Export - Bill of Materials (CSV, HTML)
5. Pick and Place - Assembly data
6. 3D Model Export - STEP/VRML for visualization

Output generation is the final step in the PCB design flow.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime

# Import common types for consistent pin handling
from .common_types import (
    get_pins, get_pin_net as common_get_pin_net, normalize_position, get_xy,
    get_footprint_definition, FootprintDefinition, FOOTPRINT_LIBRARY, is_smd_footprint
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class OutputFormat(Enum):
    """Available output formats"""
    KICAD_PCB = 'kicad_pcb'   # KiCad PCB format
    GERBER = 'gerber'         # Gerber RS-274X
    DRILL = 'drill'           # Excellon drill
    BOM_CSV = 'bom_csv'       # CSV BOM
    BOM_HTML = 'bom_html'     # HTML BOM
    PNP = 'pnp'               # Pick and Place
    STEP = 'step'             # 3D model
    SVG = 'svg'               # Vector preview


@dataclass
class GerberLayer:
    """A Gerber layer definition"""
    name: str
    extension: str
    kicad_layer: str
    description: str


# Import from single source of truth for paths
from .paths import OUTPUT_BASE, get_output_dir as get_board_output_dir, get_old_archive

# Default output base directory (imported from paths.py for consistency)
DEFAULT_OUTPUT_BASE = str(OUTPUT_BASE)


@dataclass
class OutputConfig:
    """Configuration for the output piston"""
    # Output directory - each design gets its own subfolder
    output_dir: str = ''  # Will be set to DEFAULT_OUTPUT_BASE/<board_name> if empty

    # Board info
    board_name: str = 'pcb'
    revision: str = '1.0'
    author: str = ''
    company: str = ''

    # Board dimensions
    board_width: float = 100.0
    board_height: float = 100.0
    board_origin_x: float = 0.0
    board_origin_y: float = 0.0
    board_thickness: float = 1.6

    # Design rules (for KiCad file)
    trace_width: float = 0.25
    clearance: float = 0.15
    via_diameter: float = 0.8
    via_drill: float = 0.4

    # Gerber settings
    gerber_precision: int = 6
    gerber_units: str = 'mm'

    # What to generate
    generate_kicad: bool = True
    generate_gerbers: bool = True
    generate_drill: bool = True
    generate_bom: bool = True
    generate_pnp: bool = True
    generate_gnd_pour: bool = True  # Add GND pour on bottom layer


@dataclass
class OutputResult:
    """Result from the output piston"""
    success: bool
    files_generated: List[str]
    errors: List[str] = field(default_factory=list)


# =============================================================================
# STANDARD GERBER LAYERS
# =============================================================================

STANDARD_GERBER_LAYERS = [
    GerberLayer('F.Cu', '.gtl', 'F.Cu', 'Top Copper'),
    GerberLayer('B.Cu', '.gbl', 'B.Cu', 'Bottom Copper'),
    GerberLayer('F.SilkS', '.gto', 'F.SilkS', 'Top Silkscreen'),
    GerberLayer('B.SilkS', '.gbo', 'B.SilkS', 'Bottom Silkscreen'),
    GerberLayer('F.Mask', '.gts', 'F.Mask', 'Top Solder Mask'),
    GerberLayer('B.Mask', '.gbs', 'B.Mask', 'Bottom Solder Mask'),
    GerberLayer('F.Paste', '.gtp', 'F.Paste', 'Top Paste'),
    GerberLayer('B.Paste', '.gbp', 'B.Paste', 'Bottom Paste'),
    GerberLayer('Edge.Cuts', '.gm1', 'Edge.Cuts', 'Board Outline'),
]


# =============================================================================
# OUTPUT PISTON
# =============================================================================

class OutputPiston:
    """
    Output Piston

    Generates all manufacturing files from the completed PCB design.

    Usage:
        config = OutputConfig(output_dir='./gerbers')
        piston = OutputPiston(config)
        result = piston.generate(parts_db, placement, routes, silkscreen)
    """

    def __init__(self, config: OutputConfig = None):
        self.config = config or OutputConfig()
        self.files_generated: List[str] = []
        self.errors: List[str] = []

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def generate(self, parts_db: Dict, placement: Dict, routes: Dict,
                 vias: List = None, silkscreen=None,
                 run_drc: bool = True, fail_on_drc_error: bool = False) -> OutputResult:
        """
        Generate all output files.

        Args:
            parts_db: Parts database
            placement: Component placements
            routes: Routing data
            vias: Via positions (optional)
            silkscreen: Silkscreen data (optional)
            run_drc: Run DRC validation before generating (default: True)
            fail_on_drc_error: If True, abort generation on DRC errors (default: False)

        Returns:
            OutputResult with list of generated files
        """
        self.files_generated.clear()
        self.errors.clear()

        vias = vias or []

        # Auto-generate placement if not provided
        if not placement and parts_db.get('parts'):
            placement = self._auto_generate_placement(parts_db)

        # ===================================================================
        # PRE-GENERATION DRC VALIDATION
        # Bug #14 fix: Run internal DRC to catch issues like track crossings
        # BEFORE generating files, so users know about problems early
        # ===================================================================
        if run_drc and routes:
            try:
                from .drc_piston import DRCPiston, DRCConfig
                drc_config = DRCConfig()
                drc = DRCPiston(drc_config)
                drc_result = drc.check(parts_db, placement, routes, vias, silkscreen)

                # Store DRC error count for folder naming
                self._drc_errors = drc_result.error_count

                if drc_result.error_count > 0:
                    error_msg = f"DRC found {drc_result.error_count} error(s):"
                    for v in drc_result.violations[:5]:  # Show first 5
                        if v.severity.value == 'error':
                            msg = getattr(v, 'message', '') or getattr(v, 'description', '')
                            error_msg += f"\n  - {v.violation_type.value}: {msg}"

                    if fail_on_drc_error:
                        self.errors.append(error_msg)
                        return OutputResult(
                            success=False,
                            files_generated=[],
                            errors=self.errors.copy()
                        )
                    else:
                        # Continue but warn
                        print(f"  WARNING: {error_msg}")
            except ImportError:
                pass  # DRC piston not available, continue without validation

        # ===================================================================
        # OUTPUT DIRECTORY STRUCTURE
        # Each design gets its own subfolder under DEFAULT_OUTPUT_BASE
        # Folder name format: <board_name>_<YYYYMMDD_HHMM>_<status>
        # ===================================================================
        if not self.config.output_dir:
            # Generate informative folder name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            # Determine DRC status for folder name
            drc_status = 'PASS' if not hasattr(self, '_drc_errors') or self._drc_errors == 0 else 'FAIL'
            folder_name = f"{self.config.board_name}_{timestamp}_{drc_status}"
            self.config.output_dir = os.path.join(DEFAULT_OUTPUT_BASE, folder_name)

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Clear previous LAST_PCB markers and create new one
        self._update_last_pcb_marker()

        # Generate outputs
        if self.config.generate_kicad:
            self._generate_kicad_pcb(parts_db, placement, routes, vias, silkscreen)

        if self.config.generate_bom:
            self._generate_bom(parts_db)

        if self.config.generate_pnp:
            self._generate_pnp(parts_db, placement)

        # ===================================================================
        # THE BIG BEAUTIFUL LOOP - KiCad DRC is THE AUTHORITY
        # Our internal DRC is the student, KiCad is the teacher
        # KiCad's verdict is FINAL - it determines TRUE pass/fail
        # ===================================================================
        kicad_passed = True
        kicad_errors = 0

        if self.config.generate_kicad:
            try:
                from .kicad_drc_teacher import KiCadDRCTeacher

                # Find the generated PCB file
                pcb_file = None
                for f in self.files_generated:
                    if f.endswith('.kicad_pcb'):
                        pcb_file = f
                        break

                if pcb_file:
                    print("\n" + "=" * 60)
                    print("[TEACHER] KiCad DRC - THE AUTHORITY")
                    print("=" * 60)

                    teacher = KiCadDRCTeacher()
                    internal_errors = getattr(self, '_drc_errors', 0)

                    result = teacher.teach(
                        pcb_file,
                        internal_drc_errors=internal_errors
                    )

                    print(result.message)

                    kicad_passed = result.passed
                    kicad_errors = result.kicad_errors

                    # Learning report
                    if result.learning_records:
                        print(f"\n[LEARNING] Recorded {len(result.learning_records)} learning opportunities")

                    # KiCad's verdict OVERRIDES our internal DRC
                    self._kicad_passed = kicad_passed
                    self._kicad_errors = kicad_errors

                    # Update folder status based on KiCad's verdict
                    if not kicad_passed:
                        self._drc_errors = kicad_errors
                        # Update the folder icon to reflect KiCad's verdict
                        self._update_folder_status_from_kicad()

                    print("=" * 60)

            except ImportError as e:
                print(f"   Note: KiCad DRC Teacher not available: {e}")
            except Exception as e:
                print(f"   Note: KiCad DRC Teacher error: {e}")

        # Success is determined by KiCad's verdict (if available)
        success = len(self.errors) == 0 and kicad_passed

        return OutputResult(
            success=success,
            files_generated=self.files_generated.copy(),
            errors=self.errors.copy()
        )

    def _update_folder_status_from_kicad(self):
        """Update folder icon based on KiCad's verdict (the authority)"""
        import subprocess

        kicad_errors = getattr(self, '_kicad_errors', 0)

        # Update _info.txt with KiCad's verdict
        info_file = os.path.join(self.config.output_dir, '_info.txt')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                content = f.read()

            # Add KiCad verdict
            if 'KiCad DRC:' not in content:
                kicad_status = 'PASS' if kicad_errors == 0 else f'FAIL ({kicad_errors} errors)'
                content = content.replace(
                    '=====================================\n',
                    f'=====================================\nKiCad DRC: {kicad_status} (THE AUTHORITY)\n',
                    1  # Only first occurrence
                )
                with open(info_file, 'w') as f:
                    f.write(content)

    def _update_last_pcb_marker(self):
        r"""
        NEW FOLDER STRATEGY - Maximum 3 folders in output root:

        Root (D:\Anas\tmp\output\):
        ├── <board>_<date>_PASS/   ← LAST_DESIGN if passed (star icon)
        │   OR
        ├── <board>_<date>_FAIL/   ← LAST_DESIGN if failed (warning icon)
        ├── LAST_WORKING/          ← Last design that PASSED (green check) - only if latest failed
        └── old/                   ← All previous designs moved here
            ├── design1_PASS/      (green check)
            ├── design2_FAIL/      (warning)
            └── ...

        Rules:
        1. Only 2-3 folders max in root
        2. LAST_DESIGN: most recent (star if pass, warning if fail)
        3. LAST_WORKING: only shown if LAST_DESIGN failed (to preserve last good one)
        4. old/: archive for all previous designs

        Icons:
        - imageres 101 (green check): PASS folders
        - shell32 109 (warning): FAIL folders
        - imageres 43 (star): LAST_DESIGN that PASSED
        """
        import subprocess
        import shutil

        base_dir = os.path.dirname(self.config.output_dir)
        old_dir = os.path.join(base_dir, 'old')
        drc_errors = getattr(self, '_drc_errors', 0)
        current_passed = drc_errors == 0

        # Ensure old/ directory exists
        os.makedirs(old_dir, exist_ok=True)

        # Set icon for old folder (folder icon)
        self._set_folder_icon(old_dir, 'shell32', 4)  # Standard folder icon

        # First, move any loose files to old/loose_files/
        loose_files_dir = os.path.join(old_dir, 'loose_files')
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                # Skip directories and the current output
                if os.path.isdir(item_path) or item_path == self.config.output_dir:
                    continue
                # Move loose files to old/loose_files/
                os.makedirs(loose_files_dir, exist_ok=True)
                try:
                    dest = os.path.join(loose_files_dir, item)
                    if os.path.exists(dest):
                        os.remove(dest)
                    shutil.move(item_path, dest)
                except Exception:
                    pass  # Ignore locked files

        # Find current LAST_WORKING and previous LAST_DESIGN folders
        # Also collect ALL unmarked folders to move to old/
        previous_last_design = None
        previous_last_working = None
        unmarked_folders = []

        if os.path.exists(base_dir):
            for folder in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder)

                # Skip old/, current output, and non-directories
                if folder == 'old' or folder_path == self.config.output_dir or not os.path.isdir(folder_path):
                    continue

                info_file = os.path.join(folder_path, '_info.txt')
                has_marker = False
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        content = f.read()

                    # Check for LAST_DESIGN marker (previously LAST_PCB)
                    if '<!-- LAST_DESIGN -->' in content or '<!-- LAST_PCB -->' in content:
                        previous_last_design = folder_path
                        has_marker = True

                    # Check for LAST_WORKING marker (previously LAST_PASS when separate)
                    if '<!-- LAST_WORKING -->' in content:
                        previous_last_working = folder_path
                        has_marker = True

                # Collect unmarked folders to move to old/
                if not has_marker:
                    unmarked_folders.append(folder_path)

        # Move all unmarked folders to old/ (enforce 3-folder max)
        for folder_path in unmarked_folders:
            self._move_to_old(folder_path, old_dir)

        # Move folders according to new strategy
        if previous_last_design and previous_last_design != self.config.output_dir:
            # Check if previous LAST_DESIGN passed
            info_file = os.path.join(previous_last_design, '_info.txt')
            prev_passed = False
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    prev_passed = 'PASS' in f.read() and 'FAIL' not in f.read()

            if current_passed:
                # Current PASSED: move previous LAST_DESIGN to old/
                # Also move previous LAST_WORKING to old/ (we have a new best)
                self._move_to_old(previous_last_design, old_dir)
                if previous_last_working:
                    self._move_to_old(previous_last_working, old_dir)
            else:
                # Current FAILED:
                if prev_passed:
                    # Previous was PASS - make it LAST_WORKING, move older LAST_WORKING to old
                    if previous_last_working and previous_last_working != previous_last_design:
                        self._move_to_old(previous_last_working, old_dir)
                    self._make_last_working(previous_last_design)
                else:
                    # Previous also FAILED - move it to old/, keep LAST_WORKING as is
                    self._move_to_old(previous_last_design, old_dir)

        # Create _info.txt for current folder
        info_file = os.path.join(self.config.output_dir, '_info.txt')
        drc_status = 'PASS' if current_passed else f'FAIL ({drc_errors} errors)'

        with open(info_file, 'w') as f:
            f.write(f"<!-- LAST_DESIGN -->\n")
            f.write(f"=====================================\n")
            f.write(f"PCB Design: {self.config.board_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Board Size: {self.config.board_width}mm x {self.config.board_height}mm\n")
            f.write(f"DRC Status: {drc_status}\n")
            f.write(f"=====================================\n")

        # Set icon for current folder
        if current_passed:
            self._set_folder_icon(self.config.output_dir, 'imageres', 43)  # Star (passed + latest)
        else:
            self._set_folder_icon(self.config.output_dir, 'shell32', 109)  # Warning (failed)

    def _move_to_old(self, folder_path: str, old_dir: str):
        """Move a folder to the old/ archive directory"""
        import shutil

        if not os.path.exists(folder_path):
            return

        folder_name = os.path.basename(folder_path)
        dest_path = os.path.join(old_dir, folder_name)

        # Handle name collision
        if os.path.exists(dest_path):
            i = 1
            while os.path.exists(f"{dest_path}_{i}"):
                i += 1
            dest_path = f"{dest_path}_{i}"

        try:
            # Update info file to remove LAST_DESIGN/LAST_WORKING markers
            info_file = os.path.join(folder_path, '_info.txt')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    content = f.read()
                content = content.replace('<!-- LAST_DESIGN -->\n', '')
                content = content.replace('<!-- LAST_DESIGN -->', '')
                content = content.replace('<!-- LAST_WORKING -->\n', '')
                content = content.replace('<!-- LAST_WORKING -->', '')
                content = content.replace('<!-- LAST_PCB -->\n', '')
                content = content.replace('<!-- LAST_PCB -->', '')
                content = content.replace('<!-- LAST_PASS -->\n', '')
                content = content.replace('<!-- LAST_PASS -->', '')
                with open(info_file, 'w') as f:
                    f.write(content)

            # Move the folder
            shutil.move(folder_path, dest_path)

            # Set appropriate icon in old/ based on pass/fail
            if '_PASS' in folder_name or ('PASS' in content if 'content' in dir() else False):
                self._set_folder_icon(dest_path, 'imageres', 101)  # Green check
            else:
                self._set_folder_icon(dest_path, 'shell32', 109)  # Warning

        except Exception as e:
            print(f"   Note: Could not move folder to old/: {e}")

    def _make_last_working(self, folder_path: str):
        """Update a folder to become LAST_WORKING (last successful design)"""
        info_file = os.path.join(folder_path, '_info.txt')

        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                content = f.read()

            # Remove old markers, add LAST_WORKING
            content = content.replace('<!-- LAST_DESIGN -->\n', '')
            content = content.replace('<!-- LAST_DESIGN -->', '')
            content = content.replace('<!-- LAST_PCB -->\n', '')
            content = content.replace('<!-- LAST_PCB -->', '')

            if '<!-- LAST_WORKING -->' not in content:
                content = '<!-- LAST_WORKING -->\n' + content

            with open(info_file, 'w') as f:
                f.write(content)

        # Set green check icon
        self._set_folder_icon(folder_path, 'imageres', 101)

    def _set_folder_icon(self, folder_path: str, icon_dll: str, icon_index: int):
        """Set a custom folder icon using desktop.ini"""
        import subprocess
        desktop_ini = os.path.join(folder_path, 'desktop.ini')
        try:
            # Remove existing attributes first
            if os.path.exists(desktop_ini):
                subprocess.run(['attrib', '-s', '-h', desktop_ini], capture_output=True)

            with open(desktop_ini, 'w') as f:
                f.write('[.ShellClassInfo]\n')
                f.write(f'IconResource=%SystemRoot%\\System32\\{icon_dll}.dll,{icon_index}\n')

            subprocess.run(['attrib', '+s', '+h', desktop_ini], capture_output=True)
            subprocess.run(['attrib', '+r', folder_path], capture_output=True)
        except Exception as e:
            print(f"   Note: Could not set folder icon: {e}")

    def _remove_folder_icon(self, folder_path: str):
        """Remove custom folder icon"""
        import subprocess
        desktop_ini = os.path.join(folder_path, 'desktop.ini')
        if os.path.exists(desktop_ini):
            try:
                subprocess.run(['attrib', '-s', '-h', desktop_ini], capture_output=True)
                os.remove(desktop_ini)
                subprocess.run(['attrib', '-r', folder_path], capture_output=True)
            except:
                pass

    def _auto_generate_placement(self, parts_db: Dict) -> Dict:
        """
        Auto-generate smart placement when none is provided.
        Places components based on their actual sizes to avoid overlaps.
        """
        from dataclasses import dataclass

        @dataclass
        class Position:
            x: float
            y: float
            rotation: int = 0
            layer: str = 'F.Cu'

        placement = {}
        parts = parts_db.get('parts', {})

        if not parts:
            return placement

        # Get component sizes based on footprint
        def get_component_size(footprint: str) -> tuple:
            """Returns (width, height) in mm based on footprint"""
            fp_lower = footprint.lower()

            # ESP32 modules - very large
            if 'esp32-wroom' in fp_lower or 'esp32-wrover' in fp_lower:
                return (25.5, 18.0)
            if 'esp32' in fp_lower:
                return (20.0, 15.0)

            # USB connectors
            if 'usb_c' in fp_lower or 'usb-c' in fp_lower:
                return (9.0, 7.5)
            if 'usb' in fp_lower:
                return (8.0, 6.0)

            # IC packages
            if 'qfp' in fp_lower or 'tqfp' in fp_lower:
                return (12.0, 12.0)
            if 'qfn' in fp_lower:
                return (6.0, 6.0)
            if 'soic' in fp_lower or 'sop' in fp_lower:
                return (10.0, 5.0)
            if 'sot-23-5' in fp_lower or 'sot23-5' in fp_lower:
                return (3.0, 3.0)
            if 'sot-23' in fp_lower or 'sot23' in fp_lower:
                return (3.0, 2.5)
            if 'sot' in fp_lower:
                return (3.5, 3.0)

            # Capacitors and resistors
            if '0201' in fp_lower:
                return (1.0, 0.6)
            if '0402' in fp_lower:
                return (1.5, 1.0)
            if '0603' in fp_lower:
                return (2.0, 1.2)
            if '0805' in fp_lower:
                return (2.5, 1.5)
            if '1206' in fp_lower:
                return (3.5, 2.0)
            if '1210' in fp_lower:
                return (4.0, 2.8)
            if '2512' in fp_lower:
                return (7.0, 3.5)

            # LED
            if 'led' in fp_lower:
                return (3.0, 1.5)

            # Default small component
            return (4.0, 3.0)

        # Sort components by size (largest first for better placement)
        sized_parts = []
        for ref, part in parts.items():
            fp = part.get('footprint', '')
            w, h = get_component_size(fp)
            sized_parts.append((ref, part, w, h, w * h))

        sized_parts.sort(key=lambda x: x[4], reverse=True)

        # Use board dimensions with margin
        margin = 3.0
        board_left = self.config.board_origin_x + margin
        board_top = self.config.board_origin_y + margin
        board_right = self.config.board_origin_x + self.config.board_width - margin
        board_bottom = self.config.board_origin_y + self.config.board_height - margin

        # Placement using simple row-based algorithm
        current_x = board_left
        current_y = board_top
        row_height = 0
        component_gap = 3.0  # Gap between components

        for ref, part, comp_width, comp_height, _ in sized_parts:
            # Check if component fits in current row
            if current_x + comp_width > board_right:
                # Move to next row
                current_x = board_left
                current_y += row_height + component_gap
                row_height = 0

            # Check if we've exceeded board height
            if current_y + comp_height > board_bottom:
                # Board is full - place anyway with warning
                pass

            # Place component (center of component)
            x = current_x + comp_width / 2
            y = current_y + comp_height / 2

            placement[ref] = Position(x=x, y=y, rotation=0, layer='F.Cu')

            # Update position for next component
            current_x += comp_width + component_gap
            row_height = max(row_height, comp_height)

        return placement

    # =========================================================================
    # KICAD PCB GENERATION
    # =========================================================================

    def _generate_kicad_pcb(self, parts_db: Dict, placement: Dict, routes: Dict,
                            vias: List, silkscreen) -> str:
        """Generate KiCad PCB file"""
        try:
            filename = f"{self.config.board_name}.kicad_pcb"
            filepath = os.path.join(self.config.output_dir, filename)

            content = self._build_kicad_content(parts_db, placement, routes, vias, silkscreen)

            with open(filepath, 'w') as f:
                f.write(content)

            self.files_generated.append(filepath)
            return filepath

        except Exception as e:
            self.errors.append(f"Failed to generate KiCad PCB: {e}")
            return ''

    def _build_kicad_content(self, parts_db: Dict, placement: Dict, routes: Dict,
                              vias: List, silkscreen) -> str:
        """Build KiCad PCB file content"""
        lines = []

        # Header - KiCad 9 format
        lines.append('(kicad_pcb')
        lines.append('  (version 20240108)')
        lines.append('  (generator "pcb_engine")')
        lines.append('  (generator_version "1.0")')
        lines.append('')

        # General section
        lines.append('  (general')
        lines.append(f'    (thickness {self.config.board_thickness})')
        lines.append('    (legacy_teardrops no)')
        lines.append('  )')
        lines.append('')

        # Paper size
        lines.append('  (paper "A4")')
        lines.append('')

        # Layers
        lines.append('  (layers')
        lines.append('    (0 "F.Cu" signal)')
        lines.append('    (31 "B.Cu" signal)')
        lines.append('    (32 "B.Adhes" user "B.Adhesive")')
        lines.append('    (33 "F.Adhes" user "F.Adhesive")')
        lines.append('    (34 "B.Paste" user)')
        lines.append('    (35 "F.Paste" user)')
        lines.append('    (36 "B.SilkS" user "B.Silkscreen")')
        lines.append('    (37 "F.SilkS" user "F.Silkscreen")')
        lines.append('    (38 "B.Mask" user)')
        lines.append('    (39 "F.Mask" user)')
        lines.append('    (40 "Dwgs.User" user "User.Drawings")')
        lines.append('    (41 "Cmts.User" user "User.Comments")')
        lines.append('    (42 "Eco1.User" user "User.Eco1")')
        lines.append('    (43 "Eco2.User" user "User.Eco2")')
        lines.append('    (44 "Edge.Cuts" user)')
        lines.append('    (45 "Margin" user)')
        lines.append('    (46 "B.CrtYd" user "B.Courtyard")')
        lines.append('    (47 "F.CrtYd" user "F.Courtyard")')
        lines.append('    (48 "B.Fab" user)')
        lines.append('    (49 "F.Fab" user)')
        lines.append('  )')
        lines.append('')

        # Setup section
        lines.append('  (setup')
        lines.append('    (stackup')
        lines.append('      (layer "F.SilkS" (type "Top Silk Screen"))')
        lines.append('      (layer "F.Paste" (type "Top Solder Paste"))')
        lines.append('      (layer "F.Mask" (type "Top Solder Mask") (thickness 0.01))')
        lines.append(f'      (layer "F.Cu" (type "copper") (thickness 0.035))')
        lines.append(f'      (layer "dielectric 1" (type "core") (thickness {self.config.board_thickness - 0.07}) (material "FR4"))')
        lines.append(f'      (layer "B.Cu" (type "copper") (thickness 0.035))')
        lines.append('      (layer "B.Mask" (type "Bottom Solder Mask") (thickness 0.01))')
        lines.append('      (layer "B.Paste" (type "Bottom Solder Paste"))')
        lines.append('      (layer "B.SilkS" (type "Bottom Silk Screen"))')
        lines.append('    )')
        lines.append(f'    (pad_to_mask_clearance {self.config.clearance})')
        lines.append('  )')
        lines.append('')

        # Net definitions
        nets = parts_db.get('nets', {})
        lines.append('  (net 0 "")')
        for i, net_name in enumerate(nets.keys(), 1):
            lines.append(f'  (net {i} "{net_name}")')
        lines.append('')

        # Net classes
        lines.append('  (net_class "Default" ""')
        lines.append(f'    (clearance {self.config.clearance})')
        lines.append(f'    (trace_width {self.config.trace_width})')
        lines.append(f'    (via_dia {self.config.via_diameter})')
        lines.append(f'    (via_drill {self.config.via_drill})')
        lines.append('    (uvia_dia 0.3)')
        lines.append('    (uvia_drill 0.1)')
        lines.append('  )')
        lines.append('')

        # Footprints (components)
        parts = parts_db.get('parts', {})
        net_ids = {name: i+1 for i, name in enumerate(nets.keys())}

        # Build pin_to_net lookup: 'R1.1' -> 'VCC'
        pin_to_net = {}
        for net_name, net_data in nets.items():
            for pin_ref in net_data.get('pins', []):
                if isinstance(pin_ref, str):
                    pin_to_net[pin_ref] = net_name
                elif isinstance(pin_ref, (list, tuple)) and len(pin_ref) >= 2:
                    pin_to_net[f"{pin_ref[0]}.{pin_ref[1]}"] = net_name

        for ref, pos in placement.items():
            part = parts.get(ref, {})
            fp_content = self._generate_footprint(ref, pos, part, net_ids, pin_to_net)
            lines.append(fp_content)

        # Tracks - skip GND traces if GND pour is enabled (GND connects via pour)
        # Collect actual GND PAD positions (not segment endpoints) for via generation
        gnd_pad_positions = []
        if self.config.generate_gnd_pour:
            # Get GND pins from parts database
            for ref, pos in placement.items():
                part = parts_db.get('parts', {}).get(ref, {})
                pins = part.get('pins', [])
                if isinstance(pins, list):
                    for pin in pins:
                        if pin.get('net') == 'GND':
                            # Calculate absolute pin position
                            offset = pin.get('offset', (0, 0))
                            if isinstance(offset, (list, tuple)):
                                ox, oy = offset[0], offset[1]
                            else:
                                ox, oy = 0, 0
                            if isinstance(pos, (list, tuple)):
                                px, py = pos[0] + ox, pos[1] + oy
                            elif hasattr(pos, 'x'):
                                px, py = pos.x + ox, pos.y + oy
                            else:
                                continue
                            gnd_pad_positions.append((px, py))

        for net_name, route in routes.items():
            # BUG FIX: When GND pour is enabled, we STILL need ALL segments.
            # The pour fills empty space but does NOT satisfy via_dangling checks.
            # KiCad requires a track segment to terminate at each via on BOTH layers.
            #
            # Previous bug: Filtered to only F.Cu segments, leaving B.Cu vias dangling.
            # The B.Cu segments connect vias to each other and must be kept.
            #
            # The pour provides a solid ground plane for EMI/return paths, but
            # the routing traces are still needed for electrical connectivity.

            net_id = net_ids.get(net_name, 0)
            tracks_content = self._generate_tracks(route, net_id)
            lines.append(tracks_content)

        # Vias - collect from routes AND from external parameter
        # Extract vias stored in Route objects by routing piston
        all_vias = list(vias)  # Start with externally provided vias
        via_positions_seen = set()  # Track (x, y, net) to deduplicate

        for net_name, route in routes.items():
            route_vias = getattr(route, 'vias', [])
            for via in route_vias:
                # Get via position for deduplication
                pos = getattr(via, 'position', None)
                if pos:
                    if isinstance(pos, (list, tuple)):
                        via_key = (round(pos[0], 3), round(pos[1], 3), net_name)
                    elif hasattr(pos, 'x'):
                        via_key = (round(pos.x, 3), round(pos.y, 3), net_name)
                    else:
                        via_key = None

                    if via_key and via_key not in via_positions_seen:
                        via_positions_seen.add(via_key)
                        all_vias.append(via)
                else:
                    all_vias.append(via)  # No position, add anyway

        # Generate via content - skip invalid vias (empty strings returned)
        for via in all_vias:
            via_content = self._generate_via(via, net_ids)
            if via_content:  # Only append valid vias
                lines.append(via_content)

        # GND SMD pads connectivity to B.Cu ground pour:
        # SMD pads on F.Cu need ROUTING to connect to the B.Cu GND pour.
        # The correct approach is:
        # 1. Routing piston routes GND net like any other net (includes layer-change vias)
        # 2. OR: Use via-in-pad for direct connection (advanced feature)
        # For now, GND must be in routeable_nets for proper connectivity.
        # The GND pour provides solid ground plane, but SMD pads still need routed connections.

        # Board outline
        outline = self._generate_board_outline()
        lines.append(outline)

        # Silkscreen - skip refs that are already in footprints
        if silkscreen:
            refs_in_footprints = set(placement.keys())
            silk_content = self._generate_silkscreen(silkscreen, skip_refs=refs_in_footprints)
            if silk_content.strip():  # Only add if non-empty
                lines.append(silk_content)

        # GND Pour Zone on bottom layer (if enabled)
        if self.config.generate_gnd_pour:
            gnd_net_id = net_ids.get('GND', 0)
            pour_content = self._generate_gnd_pour_zone(gnd_net_id)
            lines.append(pour_content)

            # Generate stitching vias to connect bottom GND pour to top layer
            # This eliminates "Isolated copper fill" warnings from KiCad DRC
            stitching_vias = self._generate_stitching_vias(
                parts_db, placement, routes, vias, gnd_net_id
            )
            for via_content in stitching_vias:
                lines.append(via_content)

        # Close
        lines.append(')')

        return '\n'.join(lines)

    def _generate_footprint(self, ref: str, pos, part: Dict, net_ids: Dict, pin_to_net: Dict = None) -> str:
        """Generate footprint section for a component with embedded pads"""
        pin_to_net = pin_to_net or {}
        import uuid

        lines = []

        # Handle both tuple (x, y) and object with .x/.y attributes
        if isinstance(pos, (list, tuple)):
            x, y = pos[0], pos[1]
            rotation = 0
            layer = 'F.Cu'
        elif hasattr(pos, 'x'):
            x = pos.x
            y = pos.y
            rotation = getattr(pos, 'rotation', 0)
            layer = getattr(pos, 'layer', 'F.Cu')
        else:
            x, y = 0, 0
            rotation = 0
            layer = 'F.Cu'

        footprint_name = part.get('footprint', 'Unknown:Unknown')
        value = part.get('value', '')

        lines.append(f'  (footprint "{footprint_name}"')
        lines.append(f'    (layer "{layer}")')
        lines.append(f'    (uuid "{uuid.uuid4()}")')
        lines.append(f'    (at {x:.4f} {y:.4f} {rotation})')

        # Properties
        lines.append(f'    (property "Reference" "{ref}"')
        lines.append(f'      (at 0 -2.5 0)')
        lines.append(f'      (layer "F.SilkS")')
        lines.append(f'      (uuid "{uuid.uuid4()}")')
        lines.append(f'      (effects (font (size 1 1) (thickness 0.15)))')
        lines.append(f'    )')

        lines.append(f'    (property "Value" "{value}"')
        lines.append(f'      (at 0 2.5 0)')
        lines.append(f'      (layer "F.Fab")')
        lines.append(f'      (uuid "{uuid.uuid4()}")')
        lines.append(f'      (effects (font (size 1 1) (thickness 0.15)))')
        lines.append(f'    )')

        lines.append(f'    (property "Footprint" "{footprint_name}"')
        lines.append(f'      (at 0 0 0)')
        lines.append(f'      (unlocked yes)')
        lines.append(f'      (layer "F.Fab")')
        lines.append(f'      (hide yes)')
        lines.append(f'      (uuid "{uuid.uuid4()}")')
        lines.append(f'      (effects (font (size 1.27 1.27)))')
        lines.append(f'    )')

        # Get pads based on footprint type
        pads = self._get_footprint_pads(footprint_name, part, net_ids, ref, pin_to_net)
        for pad in pads:
            lines.append(pad)

        # Add silkscreen outline
        silk = self._get_footprint_silk(footprint_name)
        for line_str in silk:
            lines.append(f'    {line_str}')

        lines.append('  )')

        return '\n'.join(lines)

    def _get_footprint_pads(self, footprint_name: str, part: Dict, net_ids: Dict,
                            ref: str = '', pin_to_net: Dict = None) -> List[str]:
        """Generate pads based on footprint type.

        FIX: Uses unified FOOTPRINT_LIBRARY from common_types.py as the single
        source of truth. This ensures routing and output use the same pad positions.
        """
        import uuid as uuid_module
        pads = []
        pin_to_net = pin_to_net or {}

        # Get net for a pin, checking both the part data and the nets lookup
        def get_pin_net(part_data, pin_num):
            # First try the part's own pin net assignment
            net = common_get_pin_net(part_data, pin_num)
            if net:
                return net
            # Fall back to looking up in pin_to_net (from parts_db['nets'])
            pin_ref = f"{ref}.{pin_num}"
            return pin_to_net.get(pin_ref, '')

        # PRIORITY: Use parts_db['pins'] as single source of truth for pad positions
        # Fall back to common_types.py FootprintDefinition ONLY if not in parts_db
        part_pins = part.get('pins', [])

        # If parts_db has pin definitions with offsets, use them (single source of truth)
        if part_pins and any('offset' in p for p in part_pins):
            for pin_data in part_pins:
                pin_num = str(pin_data.get('number', ''))
                offset = pin_data.get('offset', (0, 0))
                size = pin_data.get('size', (0.5, 0.5))
                pad_x = float(offset[0]) if isinstance(offset, (list, tuple)) else 0
                pad_y = float(offset[1]) if isinstance(offset, (list, tuple)) else 0
                pad_w = float(size[0]) if isinstance(size, (list, tuple)) else 0.5
                pad_h = float(size[1]) if isinstance(size, (list, tuple)) else 0.5
                net = get_pin_net(part, pin_num)
                net_id = net_ids.get(net, 0)
                pads.append(f'    (pad "{pin_num}" smd roundrect (at {pad_x:.4f} {pad_y:.4f}) (size {pad_w:.4f} {pad_h:.4f}) (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')

        # FALLBACK: Use unified footprint library if no pins with offsets in parts_db
        else:
            fp_def = get_footprint_definition(footprint_name)

            # Common SMD 2-pin footprints - ONLY if parts_db has no pin offsets
            if any(x in footprint_name.lower() for x in ['0402', '0603', '0805', '1206']) or \
               footprint_name in ['0402', '0603', '0805', '1206']:
                # Use pad positions from unified library
                for pin_num, pad_x, pad_y, pad_w, pad_h in fp_def.pad_positions:
                    net = get_pin_net(part, pin_num)
                    net_id = net_ids.get(net, 0)
                    pads.append(f'    (pad "{pin_num}" smd roundrect (at {pad_x:.4f} {pad_y:.4f}) (size {pad_w:.4f} {pad_h:.4f}) (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')

            # SOT-23-5 (voltage regulators like AP2112K)
            # FIX: Use unified footprint library
            elif 'SOT-23-5' in footprint_name or 'sot-23-5' in footprint_name.lower():
                sot_def = get_footprint_definition('SOT-23-5')
                for pin_num, pad_x, pad_y, pad_w, pad_h in sot_def.pad_positions:
                    net = get_pin_net(part, pin_num)
                    net_id = net_ids.get(net, 0)
                    pads.append(f'    (pad "{pin_num}" smd roundrect (at {pad_x:.4f} {pad_y:.4f}) (size {pad_w:.4f} {pad_h:.4f}) (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')

            # SOT-223 (LDO regulators like LM1117, AMS1117)
            # FIX: Use unified footprint library - pad positions must match routing!
            elif 'SOT-223' in footprint_name or 'sot-223' in footprint_name.lower():
                sot223_def = get_footprint_definition('SOT-223')
                for pin_num, pad_x, pad_y, pad_w, pad_h in sot223_def.pad_positions:
                    net = get_pin_net(part, pin_num)
                    net_id = net_ids.get(net, 0)
                    pads.append(f'    (pad "{pin_num}" smd roundrect (at {pad_x:.4f} {pad_y:.4f}) (size {pad_w:.4f} {pad_h:.4f}) (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')

            # USB-C receptacle
            elif 'USB_C' in footprint_name:
                # Simplified USB-C with key pins
                usb_pins = [
                    ('A1', -2.75, -3.0), ('A4', -1.75, -3.0), ('A5', -1.0, -3.0), ('A6', -0.25, -3.0),
                    ('A7', 0.25, -3.0), ('A8', 1.0, -3.0), ('A9', 1.75, -3.0), ('A12', 2.75, -3.0),
                    ('B1', -2.75, 3.0), ('B4', -1.75, 3.0), ('B5', -1.0, 3.0), ('B6', -0.25, 3.0),
                    ('B7', 0.25, 3.0), ('B8', 1.0, 3.0), ('B9', 1.75, 3.0), ('B12', 2.75, 3.0),
                ]
                for pin_num, px, py in usb_pins:
                    net = get_pin_net(part, pin_num)  # Pass part, not pins
                    net_id = net_ids.get(net, 0)
                    pads.append(f'    (pad "{pin_num}" smd roundrect (at {px:.4f} {py:.4f}) (size 0.3 1.0) (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')
                # Shield pins (THT)
                for i, (px, py) in enumerate([(-4.32, -1.5), (-4.32, 1.5), (4.32, -1.5), (4.32, 1.5)], 1):
                    pads.append(f'    (pad "S{i}" thru_hole oval (at {px:.4f} {py:.4f}) (size 1.0 1.8) (drill 0.6) (layers "*.Cu" "*.Mask") (uuid "{uuid_module.uuid4()}"))')

            # ESP32-WROOM-32 module (simplified)
            elif 'ESP32-WROOM' in footprint_name:
                # Bottom row pads (1-19)
                for i in range(1, 20):
                    px = -8.89 + (i-1) * 1.27
                    net = get_pin_net(part, str(i))  # Pass part, not pins
                    net_id = net_ids.get(net, 0)
                    pads.append(f'    (pad "{i}" smd rect (at {px:.4f} 8.5) (size 0.9 2.0) (layers "F.Cu" "F.Paste" "F.Mask") (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')
                # Top row pads (20-38)
                for i in range(20, 39):
                    px = -8.89 + (i-20) * 1.27
                    net = get_pin_net(part, str(i))  # Pass part, not pins
                    net_id = net_ids.get(net, 0)
                    pads.append(f'    (pad "{i}" smd rect (at {px:.4f} -8.5) (size 0.9 2.0) (layers "F.Cu" "F.Paste" "F.Mask") (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')
                # Center GND pad
                net = get_pin_net(part, '39') or 'GND'  # Pass part, not pins
                net_id = net_ids.get(net, 0)
                pads.append(f'    (pad "39" smd rect (at 0 0) (size 6.0 6.0) (layers "F.Cu" "F.Paste" "F.Mask") (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')

            # Pin Headers (through-hole)
            elif 'PinHeader' in footprint_name or 'HEADER' in footprint_name.upper():
                # Parse pin count from footprint name (e.g., PinHeader_1x03, PinHeader_1x04)
                import re
                match = re.search(r'1x0?(\d+)', footprint_name)
                pin_count = int(match.group(1)) if match else 3  # Default to 3 pins

                # Standard 2.54mm pitch (0.1 inch)
                pitch = 2.54
                start_y = -(pin_count - 1) * pitch / 2

                for i in range(1, pin_count + 1):
                    py = start_y + (i - 1) * pitch
                    net = get_pin_net(part, str(i))
                    net_id = net_ids.get(net, 0)
                    pads.append(f'    (pad "{i}" thru_hole circle (at 0 {py:.4f}) (size 1.7 1.7) (drill 1.0) (layers "*.Cu" "*.Mask") (net {net_id} "{net}") (uuid "{uuid_module.uuid4()}"))')

            # Generic fallback - 2 pads
            else:
                # Use common_get_pin_net which handles all pin formats (list, dict, etc.)
                # Pass the full part dict, not just pins
                net1 = get_pin_net(part, '1')
                net2 = get_pin_net(part, '2')
                net1_id = net_ids.get(net1, 0)
                net2_id = net_ids.get(net2, 0)
                pads.append(f'    (pad "1" smd roundrect (at -1.0 0) (size 1.0 1.5) (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net1_id} "{net1}") (uuid "{uuid_module.uuid4()}"))')
                pads.append(f'    (pad "2" smd roundrect (at 1.0 0) (size 1.0 1.5) (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net2_id} "{net2}") (uuid "{uuid_module.uuid4()}"))')

        return pads

    def _get_footprint_silk(self, footprint_name: str, pins: List = None) -> List[str]:
        """
        Generate silkscreen lines that AVOID pad areas per IPC-7351B.

        This generates silkscreen using NOTCHED RECTANGLE pattern:
        - Top and bottom edges have gaps where pads are located
        - Left and right edges are complete (no pads on sides for SMD)
        - Pin 1 marker is placed OUTSIDE the component body

        The key insight: SMD pads are on the LEFT and RIGHT edges of
        most 2-terminal components (0402, 0603, 0805, etc) so we only
        draw TOP and BOTTOM edges with gaps at corners.
        """
        import uuid as uuid_module
        lines = []

        # Component body dimensions and pad info
        # Format: (body_width, body_height, pad_x_from_center, pad_y_from_center)
        # pad_x/pad_y is the distance from component center to the OUTER edge of the pad
        footprint_specs = {
            # 0402: pad at ±0.48, size 0.56x0.62 → outer edge at 0.48+0.28=0.76
            '0402': (0.6, 0.3, 0.76, 0.0),
            'C_0402': (0.6, 0.3, 0.76, 0.0),
            'R_0402': (0.6, 0.3, 0.76, 0.0),
            # 0603: pad at ±0.775, size 0.75x0.9 → outer edge at 0.775+0.375=1.15
            '0603': (0.9, 0.45, 1.15, 0.0),
            'C_0603': (0.9, 0.45, 1.15, 0.0),
            'R_0603': (0.9, 0.45, 1.15, 0.0),
            # 0805: pad at ±0.95, size 0.9x1.25 → outer edge at 0.95+0.45=1.4
            '0805': (1.2, 0.6, 1.4, 0.0),
            'C_0805': (1.2, 0.6, 1.4, 0.0),
            'R_0805': (1.2, 0.6, 1.4, 0.0),
            'LED_0805': (1.2, 0.6, 1.4, 0.0),
            # 1206: pad at ±1.475, size 1.05x1.75 → outer edge at 1.475+0.525=2.0
            '1206': (1.8, 1.0, 2.0, 0.0),
            'C_1206': (1.8, 1.0, 2.0, 0.0),
            'R_1206': (1.8, 1.0, 2.0, 0.0),
            # SOT-23-5: pads on multiple edges
            'SOT-23-5': (1.6, 2.9, 1.5, 1.5),
            # Complex - skip silkscreen
            'USB_C': (0.0, 0.0, 0.0, 0.0),
            'ESP32-WROOM': (0.0, 0.0, 0.0, 0.0),
            'PinHeader': (0.0, 0.0, 0.0, 0.0),
        }

        # Find matching spec - check exact match first, then partial
        body_w, body_h, pad_x, pad_y = None, None, None, None

        # Exact match first
        if footprint_name in footprint_specs:
            body_w, body_h, pad_x, pad_y = footprint_specs[footprint_name]
        else:
            # Partial match
            for key, spec in footprint_specs.items():
                if key in footprint_name:
                    body_w, body_h, pad_x, pad_y = spec
                    break

        # Default for unknown footprints - skip silkscreen to be safe
        if body_w is None:
            return lines

        # Skip silkscreen for complex components (USB-C, large ICs)
        if body_w == 0 and body_h == 0:
            return lines

        # For 2-terminal SMD (resistors, caps), pads are on left/right
        # We only draw partial silkscreen on TOP and BOTTOM edges
        # The silkscreen should be OUTSIDE the solder mask opening
        silk_margin = 0.25  # Distance outside pad edge

        if pad_y == 0.0:  # Pads only on X axis (standard 2-terminal SMD)
            # Only draw short line segments at top-center and bottom-center
            # These lines are BETWEEN the two pads where no copper exists
            gap_start = pad_x + silk_margin  # Start of line (inside from pad)

            # If the component is small, pad_x might be larger than body_w/2
            # In that case, there's no safe place for silkscreen body outline
            # Just add pin 1 marker OUTSIDE the pad area + solder mask expansion
            # Solder mask expands 0.1mm, so use 0.5mm clearance to be safe
            if gap_start >= body_w / 2:
                # Pin 1 marker - placed OUTSIDE the pad outer edge + solder mask
                marker_x = -(pad_x + 0.5)  # 0.5mm outside pad edge (clears solder mask)
                marker_y = -(body_h / 2 + 0.5)  # Above the component
                lines.append(f'(fp_circle (center {marker_x:.3f} {marker_y:.3f}) (end {marker_x + 0.1:.3f} {marker_y:.3f}) (stroke (width 0.1) (type solid)) (fill solid) (layer "F.SilkS") (uuid "{uuid_module.uuid4()}"))')
                return lines

            # Draw only the center portion of top edge
            silk_x = body_w / 2 - gap_start  # How much of the edge to draw
            if silk_x > 0.2:  # Only draw if segment > 0.2mm
                # Top center segment
                lines.append(f'(fp_line (start -{silk_x:.3f} -{body_h/2:.3f}) (end {silk_x:.3f} -{body_h/2:.3f}) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{uuid_module.uuid4()}"))')
                # Bottom center segment
                lines.append(f'(fp_line (start -{silk_x:.3f} {body_h/2:.3f}) (end {silk_x:.3f} {body_h/2:.3f}) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{uuid_module.uuid4()}"))')

            # Pin 1 marker - placed well outside the pad area + solder mask
            marker_x = -(pad_x + 0.5)  # 0.5mm clearance from pad (clears solder mask)
            marker_y = -(body_h / 2 + 0.4)  # 0.4mm above component
            lines.append(f'(fp_circle (center {marker_x:.3f} {marker_y:.3f}) (end {marker_x + 0.1:.3f} {marker_y:.3f}) (stroke (width 0.1) (type solid)) (fill solid) (layer "F.SilkS") (uuid "{uuid_module.uuid4()}"))')

        else:  # SOT-23 style with pads on multiple edges
            # For multi-edge pad components, use corner markers only
            # These go in the corners where there are no pads
            corner_size = 0.3

            # Top-left corner (usually pad-free)
            lines.append(f'(fp_line (start -{body_w/2:.3f} -{body_h/2:.3f}) (end -{body_w/2 - corner_size:.3f} -{body_h/2:.3f}) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{uuid_module.uuid4()}"))')
            lines.append(f'(fp_line (start -{body_w/2:.3f} -{body_h/2:.3f}) (end -{body_w/2:.3f} -{body_h/2 - corner_size:.3f}) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{uuid_module.uuid4()}"))')

            # Pin 1 marker
            lines.append(f'(fp_circle (center -{body_w/2 - 0.25:.3f} -{body_h/2 - 0.25:.3f}) (end -{body_w/2 - 0.15:.3f} -{body_h/2 - 0.25:.3f}) (stroke (width 0.1) (type solid)) (fill solid) (layer "F.SilkS") (uuid "{uuid_module.uuid4()}"))')

        return lines

    def _generate_pad(self, pin: Dict, net_ids: Dict) -> str:
        """Generate pad section (legacy, kept for compatibility)"""
        import uuid as uuid_module
        pin_num = pin.get('number', '1')
        net = pin.get('net', '')
        net_id = net_ids.get(net, 0)

        offset = pin.get('offset', (0, 0))
        if not offset or offset == (0, 0):
            physical = pin.get('physical', {})
            offset = (physical.get('offset_x', 0), physical.get('offset_y', 0))

        pad_size = pin.get('pad_size', pin.get('size', (1.0, 0.6)))
        if not isinstance(pad_size, (list, tuple)):
            pad_size = (1.0, 0.6)

        shape = pin.get('shape', 'rect')
        hole = pin.get('hole', 0)

        pad_type = 'thru_hole' if hole > 0 else 'smd'
        layers = '"*.Cu" "*.Mask"' if hole > 0 else '"F.Cu" "F.Paste" "F.Mask"'

        shape_kicad = 'roundrect' if shape == 'rect' else shape

        line = f'    (pad "{pin_num}" {pad_type} {shape_kicad}'
        line += f' (at {offset[0]:.4f} {offset[1]:.4f})'
        line += f' (size {pad_size[0]:.4f} {pad_size[1]:.4f})'
        if hole > 0:
            line += f' (drill {hole:.4f})'
        line += f' (layers {layers})'
        if net:
            line += f' (net {net_id} "{net}")'
        line += f' (uuid "{uuid_module.uuid4()}")'
        line += ')'

        return line

    def _generate_tracks(self, route, net_id: int) -> str:
        """Generate track segments"""
        lines = []

        if hasattr(route, 'segments'):
            segments = route.segments
        elif isinstance(route, dict):
            segments = route.get('segments', [])
        else:
            return ''

        for seg in segments:
            if hasattr(seg, 'start'):
                start = seg.start
                end = seg.end
                width = seg.width
                layer = seg.layer
            else:
                start = seg.get('start', (0, 0))
                end = seg.get('end', (0, 0))
                width = seg.get('width', 0.25)
                layer = seg.get('layer', 'F.Cu')

            lines.append(f'  (segment')
            lines.append(f'    (start {start[0]:.4f} {start[1]:.4f})')
            lines.append(f'    (end {end[0]:.4f} {end[1]:.4f})')
            lines.append(f'    (width {width:.4f})')
            lines.append(f'    (layer "{layer}")')
            lines.append(f'    (net {net_id})')
            lines.append(f'  )')

        # Generate arcs (if present)
        arcs = []
        if hasattr(route, 'arcs'):
            arcs = route.arcs
        elif isinstance(route, dict):
            arcs = route.get('arcs', [])

        for arc in arcs:
            if hasattr(arc, 'start'):
                start = arc.start
                mid = arc.mid
                end = arc.end
                width = arc.width
                layer = arc.layer
            else:
                start = arc.get('start', (0, 0))
                mid = arc.get('mid', (0, 0))
                end = arc.get('end', (0, 0))
                width = arc.get('width', 0.25)
                layer = arc.get('layer', 'F.Cu')

            # KiCad arc format uses start, mid, end points
            lines.append(f'  (arc')
            lines.append(f'    (start {start[0]:.4f} {start[1]:.4f})')
            lines.append(f'    (mid {mid[0]:.4f} {mid[1]:.4f})')
            lines.append(f'    (end {end[0]:.4f} {end[1]:.4f})')
            lines.append(f'    (width {width:.4f})')
            lines.append(f'    (layer "{layer}")')
            lines.append(f'    (net {net_id})')
            lines.append(f'  )')

        return '\n'.join(lines)

    def _generate_via(self, via, net_ids: Dict = None) -> str:
        """
        Generate via KiCad format string.

        Returns empty string for invalid vias (at origin, out of bounds, etc.)
        to prevent DRC violations.
        """
        net_ids = net_ids or {}
        x, y = None, None
        net_name = ''

        if isinstance(via, dict):
            pos = via.get('position', None)
            if pos is None:
                return ''  # No position, skip
            if hasattr(pos, 'x'):
                x, y = pos.x, pos.y
            elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
            diameter = via.get('diameter', self.config.via_diameter)
            drill = via.get('drill', self.config.via_drill)
            net_name = via.get('net', '')
        elif hasattr(via, 'position'):  # Via dataclass with position tuple
            pos = via.position
            if pos is None:
                return ''  # No position, skip
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
            diameter = getattr(via, 'diameter', self.config.via_diameter)
            drill = getattr(via, 'drill', self.config.via_drill)
            net_name = getattr(via, 'net', '')
        elif isinstance(via, (list, tuple)) and len(via) >= 2:
            x, y = via[0], via[1]
            diameter = self.config.via_diameter
            drill = self.config.via_drill
        else:
            return ''  # Unrecognized format, skip

        # Validate via position - skip invalid vias
        if x is None or y is None:
            return ''  # No valid position extracted

        # Skip vias at origin (0,0) - likely placeholder/invalid
        if abs(x) < 0.001 and abs(y) < 0.001:
            return ''  # Skip via at origin

        # Skip vias outside board bounds (with margin for edge clearance)
        margin = 0.5  # mm edge clearance
        if x < self.config.board_origin_x + margin or \
           y < self.config.board_origin_y + margin or \
           x > self.config.board_origin_x + self.config.board_width - margin or \
           y > self.config.board_origin_y + self.config.board_height - margin:
            return ''  # Skip via too close to edge or outside board

        net_id = net_ids.get(net_name, 0)

        return f'''  (via
    (at {x:.4f} {y:.4f})
    (size {diameter:.4f})
    (drill {drill:.4f})
    (layers "F.Cu" "B.Cu")
    (net {net_id})
  )'''

    def _generate_board_outline(self) -> str:
        """Generate board edge cuts"""
        x0 = self.config.board_origin_x
        y0 = self.config.board_origin_y
        x1 = x0 + self.config.board_width
        y1 = y0 + self.config.board_height

        lines = []
        edges = [
            (x0, y0, x1, y0),  # Bottom
            (x1, y0, x1, y1),  # Right
            (x1, y1, x0, y1),  # Top
            (x0, y1, x0, y0),  # Left
        ]

        for sx, sy, ex, ey in edges:
            lines.append(f'  (gr_line (start {sx:.4f} {sy:.4f}) (end {ex:.4f} {ey:.4f}) (layer "Edge.Cuts") (width 0.1))')

        return '\n'.join(lines)

    def _generate_gnd_pour_zone(self, gnd_net_id: int) -> str:
        """
        Generate a GND copper pour zone on the bottom layer.

        This eliminates the need to route GND traces on F.Cu, preventing
        track crossings. GND pads connect to the pour via thermal reliefs.
        """
        import uuid

        # Zone covers the entire board with edge clearance
        edge_clearance = 0.3
        x0 = self.config.board_origin_x + edge_clearance
        y0 = self.config.board_origin_y + edge_clearance
        x1 = self.config.board_origin_x + self.config.board_width - edge_clearance
        y1 = self.config.board_origin_y + self.config.board_height - edge_clearance

        lines = []
        lines.append(f'  (zone')
        lines.append(f'    (net {gnd_net_id})')
        lines.append(f'    (net_name "GND")')
        lines.append(f'    (layer "B.Cu")')
        lines.append(f'    (uuid "{uuid.uuid4()}")')
        lines.append(f'    (hatch edge 0.5)')
        lines.append(f'    (priority 0)')
        lines.append(f'    (connect_pads thru_hole_only (clearance 0.3))')
        lines.append(f'    (min_thickness 0.2)')
        lines.append(f'    (filled_areas_thickness no)')
        lines.append(f'    (fill yes')
        lines.append(f'      (thermal_gap 0.3)')
        lines.append(f'      (thermal_bridge_width 0.4)')
        lines.append(f'    )')
        lines.append(f'    (polygon')
        lines.append(f'      (pts')
        lines.append(f'        (xy {x0:.4f} {y0:.4f})')
        lines.append(f'        (xy {x1:.4f} {y0:.4f})')
        lines.append(f'        (xy {x1:.4f} {y1:.4f})')
        lines.append(f'        (xy {x0:.4f} {y1:.4f})')
        lines.append(f'      )')
        lines.append(f'    )')
        lines.append(f'  )')

        return '\n'.join(lines)

    def _generate_gnd_via(self, pos: tuple, gnd_net_id: int) -> str:
        """Generate a via to connect GND pad to bottom layer pour"""
        import uuid
        x, y = pos
        return f'  (via (at {x:.4f} {y:.4f}) (size {self.config.via_diameter}) (drill {self.config.via_drill}) (layers "F.Cu" "B.Cu") (net {gnd_net_id}) (uuid "{uuid.uuid4()}"))'

    def _generate_stitching_vias(self, parts_db: Dict, placement: Dict,
                                  routes: Dict, existing_vias: List,
                                  gnd_net_id: int) -> List[str]:
        """
        Generate stitching vias to connect bottom layer GND pour to top layer.

        This eliminates "Isolated copper fill" warnings from KiCad DRC by ensuring
        the bottom layer pour is connected to the top layer GND net.

        IMPORTANT: Vias must be placed NEAR GND elements on F.Cu (pads or tracks)
        to actually create a connection. Random grid vias would just be dangling.

        Placement is DRC-safe:
        - Clear of all pads (any net)
        - Clear of all tracks (but near GND tracks)
        - Clear of existing vias
        - Clear of board edges

        Args:
            parts_db: Parts database
            placement: Component placements
            routes: Routing data
            existing_vias: Already placed vias
            gnd_net_id: Net ID for GND

        Returns:
            List of KiCad via S-expression strings
        """
        import uuid
        import math

        via_strings = []

        # Configuration
        via_size = self.config.via_diameter
        via_drill = self.config.via_drill
        via_clearance = 0.5  # Min distance from obstacles (reduced for small boards)
        edge_clearance = 0.5 + via_size / 2

        board_x0 = self.config.board_origin_x
        board_y0 = self.config.board_origin_y
        board_x1 = board_x0 + self.config.board_width
        board_y1 = board_y0 + self.config.board_height

        # Collect GND pad positions on F.Cu (potential connection points)
        gnd_pads = []
        parts = parts_db.get('parts', {})
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                cx, cy = pos[0], pos[1]
            elif hasattr(pos, 'x'):
                cx, cy = pos.x, pos.y
            else:
                continue

            for pin in part.get('pins', []):
                if pin.get('net') == 'GND':
                    offset = pin.get('offset', (0, 0))
                    if isinstance(offset, (list, tuple)):
                        px = cx + offset[0]
                        py = cy + offset[1]
                    else:
                        px, py = cx, cy
                    gnd_pads.append((px, py))

        # Collect GND track ENDPOINT positions on F.Cu (potential connection points)
        # IMPORTANT: Only use track ENDPOINTS, not midpoints!
        # KiCad considers a via "dangling" if no track segment ends at the via position.
        # Placing a via on a track midpoint does NOT create a connection in KiCad's view.
        gnd_track_endpoints = []
        if routes and 'GND' in routes:
            gnd_route = routes['GND']
            segments = getattr(gnd_route, 'segments', [])
            for seg in segments:
                layer = getattr(seg, 'layer', 'F.Cu')
                if layer != 'F.Cu':
                    continue  # Only tracks on F.Cu can connect to pour vias
                start = getattr(seg, 'start', (0, 0))
                end = getattr(seg, 'end', (0, 0))
                # Only add endpoints - NOT midpoints!
                # Midpoint vias would be "dangling" because no track terminates there
                gnd_track_endpoints.append(start)
                gnd_track_endpoints.append(end)

        # Deduplicate endpoints (shared vertices between segments)
        gnd_track_points = []
        seen = set()
        for pt in gnd_track_endpoints:
            key = (round(pt[0], 3), round(pt[1], 3))
            if key not in seen:
                seen.add(key)
                gnd_track_points.append(pt)

        # If no GND on F.Cu, we can't connect pour - skip stitching vias
        if not gnd_pads and not gnd_track_points:
            print(f"  [POUR] No GND elements on F.Cu - skipping stitching vias")
            return []

        # Collect all obstacles: (x, y, radius)
        obstacles = []

        # Add ALL pad obstacles (need clearance from all pads)
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                cx, cy = pos[0], pos[1]
            elif hasattr(pos, 'x'):
                cx, cy = pos.x, pos.y
            else:
                continue

            for pin in part.get('pins', []):
                offset = pin.get('offset', (0, 0))
                size = pin.get('size', (1.0, 1.0))

                if isinstance(offset, (list, tuple)):
                    px = cx + offset[0]
                    py = cy + offset[1]
                else:
                    px, py = cx, cy

                if isinstance(size, (list, tuple)):
                    pad_radius = math.sqrt((size[0]/2)**2 + (size[1]/2)**2) + via_clearance
                else:
                    pad_radius = 1.0 + via_clearance

                obstacles.append((px, py, pad_radius))

        # Add track obstacles (SKIP GND tracks - we WANT vias to overlap GND tracks)
        if routes:
            for net_name, route in routes.items():
                if net_name == 'GND':
                    continue  # Skip GND tracks - we place vias ON them
                segments = getattr(route, 'segments', [])
                for seg in segments:
                    start = getattr(seg, 'start', (0, 0))
                    end = getattr(seg, 'end', (0, 0))
                    width = getattr(seg, 'width', 0.25)

                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = math.sqrt(dx*dx + dy*dy)

                    if length < 0.001:
                        obstacles.append((start[0], start[1], width/2 + via_clearance))
                    else:
                        num_points = max(2, int(length / 0.5))  # More points for better coverage
                        for i in range(num_points + 1):
                            t = i / num_points
                            tx = start[0] + dx * t
                            ty = start[1] + dy * t
                            obstacles.append((tx, ty, width/2 + via_clearance))

        # Add existing via obstacles
        for via in existing_vias:
            pos = getattr(via, 'position', None)
            if pos:
                if isinstance(pos, (list, tuple)):
                    vx, vy = pos[0], pos[1]
                elif hasattr(pos, 'x'):
                    vx, vy = pos.x, pos.y
                else:
                    continue
                vsize = getattr(via, 'size', 0.6)
                obstacles.append((vx, vy, vsize/2 + via_clearance))

        # Also add vias from routes
        if routes:
            for net_name, route in routes.items():
                route_vias = getattr(route, 'vias', [])
                for via in route_vias:
                    pos = getattr(via, 'position', None)
                    if pos:
                        if isinstance(pos, (list, tuple)):
                            vx, vy = pos[0], pos[1]
                        elif hasattr(pos, 'x'):
                            vx, vy = pos.x, pos.y
                        else:
                            continue
                        vsize = getattr(via, 'size', 0.6)
                        obstacles.append((vx, vy, vsize/2 + via_clearance))

        # Generate candidate positions AT GND track ENDPOINTS only
        # CRITICAL: Vias must be at track endpoints, not midpoints!
        # KiCad considers a via "dangling" if no track segment terminates at the via.
        # A via overlapping a track midpoint is NOT recognized as connected.
        candidates = []

        # STRATEGY: Place vias AT GND track endpoints
        # This ensures KiCad recognizes the connection (track ends at via)
        # The via connects F.Cu track to B.Cu pour, completing the circuit
        for tx, ty in gnd_track_points:
            # Check board bounds
            if board_x0 + edge_clearance <= tx <= board_x1 - edge_clearance:
                if board_y0 + edge_clearance <= ty <= board_y1 - edge_clearance:
                    candidates.append((tx, ty, 'track_direct'))

        # Filter candidates that don't violate pad/non-GND-track obstacles
        # (GND tracks are excluded from obstacles since vias will be placed ON them)
        valid_positions = []
        for cx, cy, source in candidates:
            is_clear = True

            for ox, oy, radius in obstacles:
                dist = math.sqrt((cx - ox)**2 + (cy - oy)**2)
                if dist < radius + via_size / 2:
                    is_clear = False
                    break

            if is_clear:
                # Check not too close to existing valid positions
                too_close = False
                for vx, vy in valid_positions:
                    if math.sqrt((cx - vx)**2 + (cy - vy)**2) < via_size * 2:
                        too_close = True
                        break
                if not too_close:
                    valid_positions.append((cx, cy))

        # Limit to 1-2 vias for small boards, more for larger
        board_area = self.config.board_width * self.config.board_height
        max_vias = max(1, min(len(valid_positions), int(board_area / 200) + 1))
        valid_positions = valid_positions[:max_vias]

        # Generate via strings AND tiny B.Cu anchor tracks for valid positions
        # The anchor tracks ensure KiCad recognizes the via as connected on B.Cu
        # (filled zones don't satisfy the via_dangling check, only tracks do)
        trace_width = getattr(self.config, 'trace_width', 0.25)
        anchor_length = 0.1  # Tiny 0.1mm anchor track

        for vx, vy in valid_positions:
            # Via
            via_str = f'  (via (at {vx:.4f} {vy:.4f}) (size {via_size}) (drill {via_drill}) (layers "F.Cu" "B.Cu") (net {gnd_net_id}) (uuid "{uuid.uuid4()}"))'
            via_strings.append(via_str)

            # B.Cu anchor track - a tiny segment that starts and ends at the via
            # This satisfies KiCad's via_dangling check by creating a track connection on B.Cu
            anchor_str = f'''  (segment
    (start {vx:.4f} {vy:.4f})
    (end {vx + anchor_length:.4f} {vy:.4f})
    (width {trace_width:.4f})
    (layer "B.Cu")
    (net {gnd_net_id})
  )'''
            via_strings.append(anchor_str)

        if valid_positions:
            print(f"  [POUR] Added {len(valid_positions)} GND stitching vias with B.Cu anchors")
        else:
            print(f"  [POUR] WARNING: No valid positions for stitching vias found")

        return via_strings

    def _generate_silkscreen(self, silkscreen, skip_refs: set = None) -> str:
        """
        Generate silkscreen elements.

        Skips reference designator texts since they're already in footprint properties.
        This prevents the "silkscreen overlap" DRC violation from KiCad.

        Args:
            silkscreen: SilkscreenResult with texts, lines, etc.
            skip_refs: Set of reference designators to skip (already in footprints)
        """
        lines = []
        skip_refs = skip_refs or set()

        if hasattr(silkscreen, 'texts'):
            for text in silkscreen.texts:
                # Skip reference designator texts - they're already in footprint properties
                # Reference designators are typically single uppercase letter followed by numbers
                # e.g., R1, C1, U1, J1, D1, L1, Q1, etc.
                text_str = str(text.text).strip()
                if text_str in skip_refs:
                    continue  # Skip - already in footprint
                # Also skip if text matches common ref designator pattern
                import re
                if re.match(r'^[A-Z]+[0-9]+$', text_str):
                    continue  # Skip reference designators

                lines.append(f'  (gr_text "{text.text}"')
                lines.append(f'    (at {text.x:.4f} {text.y:.4f} {text.rotation})')
                lines.append(f'    (layer "{text.layer}")')
                lines.append(f'    (effects (font (size {text.size} {text.size}) (thickness {text.thickness})))')
                lines.append('  )')

        return '\n'.join(lines)

    # =========================================================================
    # BOM GENERATION
    # =========================================================================

    def _generate_bom(self, parts_db: Dict) -> str:
        """Generate Bill of Materials"""
        try:
            # Group components by value and footprint
            groups = {}
            parts = parts_db.get('parts', {})

            for ref, part in parts.items():
                key = (part.get('value', ''), part.get('footprint', ''))
                if key not in groups:
                    groups[key] = []
                groups[key].append(ref)

            # CSV BOM
            csv_lines = ['Quantity,References,Value,Footprint,Description']
            for (value, footprint), refs in sorted(groups.items()):
                sorted_refs = sorted(refs)
                csv_lines.append(f'{len(refs)},"{",".join(sorted_refs)}","{value}","{footprint}",""')

            csv_path = os.path.join(self.config.output_dir, f'{self.config.board_name}_bom.csv')
            with open(csv_path, 'w') as f:
                f.write('\n'.join(csv_lines))

            self.files_generated.append(csv_path)

            # HTML BOM
            html_lines = ['<!DOCTYPE html><html><head><title>BOM</title>',
                          '<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style></head><body>',
                          f'<h1>BOM: {self.config.board_name}</h1>',
                          f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>',
                          '<table><tr><th>Qty</th><th>References</th><th>Value</th><th>Footprint</th></tr>']

            for (value, footprint), refs in sorted(groups.items()):
                html_lines.append(f'<tr><td>{len(refs)}</td><td>{", ".join(sorted(refs))}</td><td>{value}</td><td>{footprint}</td></tr>')

            html_lines.extend(['</table></body></html>'])

            html_path = os.path.join(self.config.output_dir, f'{self.config.board_name}_bom.html')
            with open(html_path, 'w') as f:
                f.write('\n'.join(html_lines))

            self.files_generated.append(html_path)

            return csv_path

        except Exception as e:
            self.errors.append(f"Failed to generate BOM: {e}")
            return ''

    # =========================================================================
    # PICK AND PLACE
    # =========================================================================

    def _generate_pnp(self, parts_db: Dict, placement: Dict) -> str:
        """Generate Pick and Place file"""
        try:
            lines = ['Ref,Val,Package,PosX,PosY,Rot,Side']

            parts = parts_db.get('parts', {})

            for ref, pos in sorted(placement.items()):
                part = parts.get(ref, {})
                value = part.get('value', '')
                footprint = part.get('footprint', '').split(':')[-1]  # Remove library prefix

                # Handle both tuple (x, y) and object with .x/.y attributes
                if isinstance(pos, (list, tuple)):
                    x, y = pos[0], pos[1]
                    rotation = 0
                    layer = 'F.Cu'
                elif hasattr(pos, 'x'):
                    x, y = pos.x, pos.y
                    rotation = getattr(pos, 'rotation', 0)
                    layer = getattr(pos, 'layer', 'F.Cu')
                else:
                    x, y = 0, 0
                    rotation = 0
                    layer = 'F.Cu'

                side = 'top' if layer == 'F.Cu' else 'bottom'

                lines.append(f'{ref},{value},{footprint},{x:.4f},{y:.4f},{rotation},{side}')

            filepath = os.path.join(self.config.output_dir, f'{self.config.board_name}_pnp.csv')
            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))

            self.files_generated.append(filepath)
            return filepath

        except Exception as e:
            self.errors.append(f"Failed to generate PnP: {e}")
            return ''

    # =========================================================================
    # GERBER GENERATION (RS-274X Format)
    # =========================================================================
    # Reference: https://www.ucamco.com/gerber (Gerber Format Specification)
    # RS-274X is the extended Gerber format with embedded apertures

    def generate_gerbers(self, parts_db: Dict, placement: Dict, routes: Dict,
                         vias: List = None, zones: List = None) -> List[str]:
        """
        Generate complete Gerber file set for manufacturing.

        Generates:
        - F.Cu (top copper) - .gtl
        - B.Cu (bottom copper) - .gbl
        - F.Mask (top solder mask) - .gts
        - B.Mask (bottom solder mask) - .gbs
        - F.SilkS (top silkscreen) - .gto
        - B.SilkS (bottom silkscreen) - .gbo
        - F.Paste (top paste) - .gtp
        - B.Paste (bottom paste) - .gbp
        - Edge.Cuts (board outline) - .gm1

        Args:
            parts_db: Parts database with components
            placement: Component placements
            routes: Routing data
            vias: Via positions
            zones: Copper zones (ground planes, etc.)

        Returns:
            List of generated file paths
        """
        vias = vias or []
        zones = zones or []
        files = []

        gerber_dir = os.path.join(self.config.output_dir, 'gerbers')
        os.makedirs(gerber_dir, exist_ok=True)

        # Generate each layer
        for layer in STANDARD_GERBER_LAYERS:
            try:
                filepath = self._generate_gerber_layer(
                    layer, parts_db, placement, routes, vias, zones, gerber_dir
                )
                if filepath:
                    files.append(filepath)
                    self.files_generated.append(filepath)
            except Exception as e:
                self.errors.append(f"Failed to generate {layer.name}: {e}")

        return files

    def _generate_gerber_layer(self, layer: GerberLayer, parts_db: Dict,
                                placement: Dict, routes: Dict, vias: List,
                                zones: List, gerber_dir: str) -> str:
        """Generate a single Gerber layer file"""
        filename = f"{self.config.board_name}{layer.extension}"
        filepath = os.path.join(gerber_dir, filename)

        lines = []

        # Header
        lines.extend(self._gerber_header(layer))

        # Aperture definitions
        apertures = self._generate_aperture_list(parts_db, routes)
        lines.extend(self._gerber_aperture_definitions(apertures))

        # Layer content based on type
        if layer.name in ('F.Cu', 'B.Cu'):
            # Copper layer: pads, tracks, vias, zones
            lines.extend(self._gerber_copper_layer(
                layer.name, parts_db, placement, routes, vias, zones, apertures
            ))
        elif layer.name in ('F.Mask', 'B.Mask'):
            # Solder mask: openings for pads
            lines.extend(self._gerber_mask_layer(
                layer.name, parts_db, placement, apertures
            ))
        elif layer.name in ('F.Paste', 'B.Paste'):
            # Paste: SMD pads only
            lines.extend(self._gerber_paste_layer(
                layer.name, parts_db, placement, apertures
            ))
        elif layer.name in ('F.SilkS', 'B.SilkS'):
            # Silkscreen: reference designators, outlines
            lines.extend(self._gerber_silk_layer(
                layer.name, parts_db, placement
            ))
        elif layer.name == 'Edge.Cuts':
            # Board outline
            lines.extend(self._gerber_edge_cuts())

        # Footer
        lines.append('M02*')  # End of file

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        return filepath

    def _gerber_header(self, layer: GerberLayer) -> List[str]:
        """Generate Gerber file header (RS-274X)"""
        return [
            '%TF.GenerationSoftware,PCBEngine,1.0*%',
            f'%TF.CreationDate,{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}*%',
            f'%TF.ProjectId,{self.config.board_name},{self.config.revision}*%',
            f'%TF.FileFunction,{layer.description}*%',
            '%FSLAX46Y46*%',  # Format: Leading zeros omitted, Abs coords, 4.6 format
            '%MOIN*%' if self.config.gerber_units == 'inch' else '%MOMM*%',
            '%LPD*%',  # Load polarity: Dark
        ]

    def _generate_aperture_list(self, parts_db: Dict, routes: Dict) -> Dict:
        """
        Generate aperture list from design data.

        Apertures in Gerber are like "stamps" - predefined shapes.
        Each unique pad size, trace width, etc. needs an aperture.
        """
        apertures = {}
        next_id = 10  # Aperture IDs start at D10

        # Standard trace widths
        trace_widths = {0.15, 0.2, 0.25, 0.3, 0.4, 0.5, self.config.trace_width}
        for width in sorted(trace_widths):
            apertures[f'trace_{width}'] = {
                'id': f'D{next_id}',
                'type': 'C',  # Circle
                'params': f'{width:.4f}',
                'width': width
            }
            next_id += 1

        # Via aperture
        apertures['via'] = {
            'id': f'D{next_id}',
            'type': 'C',
            'params': f'{self.config.via_diameter:.4f}',
            'width': self.config.via_diameter
        }
        next_id += 1

        # Pad apertures from parts
        parts = parts_db.get('parts', {})
        for ref, part in parts.items():
            for pin in part.get('pins', []):
                phys = pin.get('physical', {})
                size_x = phys.get('size_x', 1.0)
                size_y = phys.get('size_y', 0.6)
                key = f'pad_{size_x:.3f}x{size_y:.3f}'
                if key not in apertures:
                    if abs(size_x - size_y) < 0.01:
                        # Square/circular
                        apertures[key] = {
                            'id': f'D{next_id}',
                            'type': 'C',
                            'params': f'{size_x:.4f}',
                            'size': (size_x, size_y)
                        }
                    else:
                        # Rectangle
                        apertures[key] = {
                            'id': f'D{next_id}',
                            'type': 'R',
                            'params': f'{size_x:.4f}X{size_y:.4f}',
                            'size': (size_x, size_y)
                        }
                    next_id += 1

        return apertures

    def _gerber_aperture_definitions(self, apertures: Dict) -> List[str]:
        """Generate aperture definition block"""
        lines = []
        for name, ap in sorted(apertures.items(), key=lambda x: int(x[1]['id'][1:])):
            lines.append(f'%AD{ap["id"]}{ap["type"]},{ap["params"]}*%')
        return lines

    def _gerber_copper_layer(self, layer_name: str, parts_db: Dict,
                              placement: Dict, routes: Dict, vias: List,
                              zones: List, apertures: Dict) -> List[str]:
        """Generate copper layer content"""
        lines = []
        is_top = layer_name == 'F.Cu'

        # Pads
        parts = parts_db.get('parts', {})
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            part_layer = getattr(pos, 'layer', 'F.Cu')
            if (is_top and part_layer == 'F.Cu') or (not is_top and part_layer == 'B.Cu'):
                lines.extend(self._gerber_pads(ref, pos, part, apertures))

        # Tracks
        for net_name, route in routes.items():
            lines.extend(self._gerber_tracks(route, layer_name, apertures))

        # Vias (appear on both copper layers)
        via_ap = apertures.get('via', {}).get('id', 'D10')
        lines.append(f'{via_ap}*')  # Select via aperture
        for via in vias:
            if isinstance(via, dict):
                x, y = via.get('position', (0, 0))
            else:
                x, y = via[0], via[1] if len(via) > 1 else (0, 0)
            lines.append(f'X{self._gerber_coord(x)}Y{self._gerber_coord(y)}D03*')

        # Zones (copper pours)
        for zone in zones:
            if zone.get('layer') == layer_name:
                lines.extend(self._gerber_zone(zone))

        return lines

    def _gerber_pads(self, ref: str, pos, part: Dict, apertures: Dict) -> List[str]:
        """Generate pad flashes for a component"""
        lines = []
        for pin in part.get('pins', []):
            phys = pin.get('physical', {})
            offset_x = phys.get('offset_x', 0)
            offset_y = phys.get('offset_y', 0)
            size_x = phys.get('size_x', 1.0)
            size_y = phys.get('size_y', 0.6)

            # Find matching aperture
            key = f'pad_{size_x:.3f}x{size_y:.3f}'
            ap = apertures.get(key, {}).get('id', 'D10')

            # Calculate absolute position
            x = pos.x + offset_x
            y = pos.y + offset_y

            lines.append(f'{ap}*')
            lines.append(f'X{self._gerber_coord(x)}Y{self._gerber_coord(y)}D03*')

        return lines

    def _gerber_tracks(self, route, layer_name: str, apertures: Dict) -> List[str]:
        """Generate track draws"""
        lines = []

        if hasattr(route, 'segments'):
            segments = route.segments
        elif isinstance(route, dict):
            segments = route.get('segments', [])
        else:
            return lines

        for seg in segments:
            if hasattr(seg, 'layer'):
                seg_layer = seg.layer
                width = seg.width
                start = seg.start
                end = seg.end
            else:
                seg_layer = seg.get('layer', 'F.Cu')
                width = seg.get('width', 0.25)
                start = seg.get('start', (0, 0))
                end = seg.get('end', (0, 0))

            if seg_layer != layer_name:
                continue

            # Find trace aperture
            ap_key = f'trace_{width}'
            if ap_key not in apertures:
                ap_key = f'trace_{self.config.trace_width}'
            ap = apertures.get(ap_key, {}).get('id', 'D10')

            lines.append(f'{ap}*')
            lines.append(f'X{self._gerber_coord(start[0])}Y{self._gerber_coord(start[1])}D02*')  # Move
            lines.append(f'X{self._gerber_coord(end[0])}Y{self._gerber_coord(end[1])}D01*')  # Draw

        return lines

    def _gerber_zone(self, zone: Dict) -> List[str]:
        """Generate copper zone (polygon pour)"""
        lines = []
        outline = zone.get('outline', [])
        if len(outline) < 3:
            return lines

        # Start region
        lines.append('G36*')  # Begin region

        # Move to first point
        x0, y0 = outline[0]
        lines.append(f'X{self._gerber_coord(x0)}Y{self._gerber_coord(y0)}D02*')

        # Draw outline
        for x, y in outline[1:]:
            lines.append(f'X{self._gerber_coord(x)}Y{self._gerber_coord(y)}D01*')

        # Close
        lines.append(f'X{self._gerber_coord(x0)}Y{self._gerber_coord(y0)}D01*')
        lines.append('G37*')  # End region

        return lines

    def _gerber_mask_layer(self, layer_name: str, parts_db: Dict,
                           placement: Dict, apertures: Dict) -> List[str]:
        """Generate solder mask layer (openings for pads)"""
        lines = []
        is_top = layer_name == 'F.Mask'
        expansion = 0.05  # Mask expansion around pads

        parts = parts_db.get('parts', {})
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            part_layer = getattr(pos, 'layer', 'F.Cu')
            if (is_top and part_layer == 'F.Cu') or (not is_top and part_layer == 'B.Cu'):
                for pin in part.get('pins', []):
                    phys = pin.get('physical', {})
                    offset_x = phys.get('offset_x', 0)
                    offset_y = phys.get('offset_y', 0)
                    size_x = phys.get('size_x', 1.0) + expansion * 2
                    size_y = phys.get('size_y', 0.6) + expansion * 2

                    x = pos.x + offset_x
                    y = pos.y + offset_y

                    # Use rectangle aperture
                    key = f'pad_{size_x:.3f}x{size_y:.3f}'
                    ap = apertures.get(key, {}).get('id', 'D10')
                    lines.append(f'{ap}*')
                    lines.append(f'X{self._gerber_coord(x)}Y{self._gerber_coord(y)}D03*')

        return lines

    def _gerber_paste_layer(self, layer_name: str, parts_db: Dict,
                            placement: Dict, apertures: Dict) -> List[str]:
        """Generate paste layer (SMD pads only, no THT)"""
        lines = []
        is_top = layer_name == 'F.Paste'
        reduction = 0.05  # Paste reduction from pad

        parts = parts_db.get('parts', {})
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            part_layer = getattr(pos, 'layer', 'F.Cu')
            if (is_top and part_layer == 'F.Cu') or (not is_top and part_layer == 'B.Cu'):
                for pin in part.get('pins', []):
                    phys = pin.get('physical', {})
                    # Skip THT pads
                    if phys.get('type') == 'tht' or phys.get('drill', 0) > 0:
                        continue

                    offset_x = phys.get('offset_x', 0)
                    offset_y = phys.get('offset_y', 0)
                    size_x = max(0.1, phys.get('size_x', 1.0) - reduction * 2)
                    size_y = max(0.1, phys.get('size_y', 0.6) - reduction * 2)

                    x = pos.x + offset_x
                    y = pos.y + offset_y

                    key = f'pad_{size_x:.3f}x{size_y:.3f}'
                    ap = apertures.get(key, {}).get('id', 'D10')
                    lines.append(f'{ap}*')
                    lines.append(f'X{self._gerber_coord(x)}Y{self._gerber_coord(y)}D03*')

        return lines

    def _gerber_silk_layer(self, layer_name: str, parts_db: Dict,
                           placement: Dict) -> List[str]:
        """Generate silkscreen layer"""
        lines = []
        is_top = layer_name == 'F.SilkS'

        # Use thin line for silkscreen
        lines.append('%ADD99C,0.15*%')  # 0.15mm line
        lines.append('D99*')

        parts = parts_db.get('parts', {})
        for ref, pos in placement.items():
            part = parts.get(ref, {})
            part_layer = getattr(pos, 'layer', 'F.Cu')
            if (is_top and part_layer == 'F.Cu') or (not is_top and part_layer == 'B.Cu'):
                # Draw component outline (simple rectangle)
                size = part.get('size', (2.0, 1.0))
                if isinstance(size, (list, tuple)) and len(size) >= 2:
                    w, h = size[0] / 2, size[1] / 2
                else:
                    w, h = 1.0, 0.5

                x, y = pos.x, pos.y
                # Rectangle outline
                lines.append(f'X{self._gerber_coord(x-w)}Y{self._gerber_coord(y-h)}D02*')
                lines.append(f'X{self._gerber_coord(x+w)}Y{self._gerber_coord(y-h)}D01*')
                lines.append(f'X{self._gerber_coord(x+w)}Y{self._gerber_coord(y+h)}D01*')
                lines.append(f'X{self._gerber_coord(x-w)}Y{self._gerber_coord(y+h)}D01*')
                lines.append(f'X{self._gerber_coord(x-w)}Y{self._gerber_coord(y-h)}D01*')

        return lines

    def _gerber_edge_cuts(self) -> List[str]:
        """Generate board outline"""
        lines = []

        # Use 0.1mm line for edge
        lines.append('%ADD98C,0.1*%')
        lines.append('D98*')

        x0 = self.config.board_origin_x
        y0 = self.config.board_origin_y
        x1 = x0 + self.config.board_width
        y1 = y0 + self.config.board_height

        # Draw rectangle
        lines.append(f'X{self._gerber_coord(x0)}Y{self._gerber_coord(y0)}D02*')
        lines.append(f'X{self._gerber_coord(x1)}Y{self._gerber_coord(y0)}D01*')
        lines.append(f'X{self._gerber_coord(x1)}Y{self._gerber_coord(y1)}D01*')
        lines.append(f'X{self._gerber_coord(x0)}Y{self._gerber_coord(y1)}D01*')
        lines.append(f'X{self._gerber_coord(x0)}Y{self._gerber_coord(y0)}D01*')

        return lines

    def _gerber_coord(self, value: float) -> str:
        """Convert mm to Gerber coordinate (4.6 format = microns)"""
        # 4.6 format: 4 integer digits, 6 decimal digits
        # 1mm = 1000000 in 4.6 format
        return f'{int(value * 1000000):010d}'

    # =========================================================================
    # EXCELLON DRILL FILE GENERATION
    # =========================================================================
    # Reference: IPC-NC-349 (Excellon Format)

    def generate_drill(self, parts_db: Dict, placement: Dict,
                       vias: List = None) -> str:
        """
        Generate Excellon drill file.

        This file tells the CNC drill machine where to drill holes.

        Args:
            parts_db: Parts database
            placement: Component placements
            vias: Via positions

        Returns:
            Path to generated drill file
        """
        vias = vias or []

        try:
            filename = f"{self.config.board_name}.drl"
            filepath = os.path.join(self.config.output_dir, filename)

            lines = []

            # Collect all drill sizes and holes
            drills = {}  # size -> list of (x, y)

            # Via holes
            via_drill = self.config.via_drill
            if via_drill not in drills:
                drills[via_drill] = []
            for via in vias:
                if isinstance(via, dict):
                    x, y = via.get('position', (0, 0))
                else:
                    x, y = via[0], via[1] if len(via) > 1 else (0, 0)
                drills[via_drill].append((x, y))

            # THT component holes
            parts = parts_db.get('parts', {})
            for ref, pos in placement.items():
                part = parts.get(ref, {})
                for pin in part.get('pins', []):
                    phys = pin.get('physical', {})
                    drill_size = phys.get('drill', 0)
                    if drill_size > 0:
                        if drill_size not in drills:
                            drills[drill_size] = []
                        offset_x = phys.get('offset_x', 0)
                        offset_y = phys.get('offset_y', 0)
                        x = pos.x + offset_x
                        y = pos.y + offset_y
                        drills[drill_size].append((x, y))

            # Header
            lines.append('M48')  # Start of header
            lines.append(';EXCELLON DRILL FILE')
            lines.append(f';Generated by PCB Engine')
            lines.append(f';Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            lines.append('FMAT,2')  # Format 2
            lines.append('METRIC,TZ')  # Metric, trailing zeros

            # Tool definitions
            tool_num = 1
            tool_map = {}
            for size in sorted(drills.keys()):
                lines.append(f'T{tool_num:02d}C{size:.3f}')
                tool_map[size] = tool_num
                tool_num += 1

            lines.append('%')  # End of header

            # Drill commands
            for size in sorted(drills.keys()):
                tool = tool_map[size]
                lines.append(f'T{tool:02d}')
                for x, y in drills[size]:
                    lines.append(f'X{x:.3f}Y{y:.3f}')

            lines.append('M30')  # End of program

            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))

            self.files_generated.append(filepath)
            return filepath

        except Exception as e:
            self.errors.append(f"Failed to generate drill file: {e}")
            return ''

    # =========================================================================
    # KICAD PYTHON SCRIPT GENERATION
    # =========================================================================

    def generate_kicad_script(self, parts_db: Dict, placement: Dict,
                               routes: Dict, vias: List = None,
                               zones: List = None) -> str:
        """
        Generate a Python script that can be run in KiCad's scripting console.

        This is useful because it uses KiCad's native API to:
        - Load footprints from libraries (with 3D models)
        - Create proper net connections
        - Generate zone fills

        Usage in KiCad:
        1. Open PCB Editor
        2. Tools -> Scripting Console (F4)
        3. exec(open("path/to/script.py").read())
        """
        vias = vias or []
        zones = zones or []

        try:
            filename = f"{self.config.board_name}_script.py"
            filepath = os.path.join(self.config.output_dir, filename)

            lines = []

            # Header
            lines.append('"""')
            lines.append(f'KiCad PCB Script - Generated by PCB Engine')
            lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            lines.append(f'Board: {self.config.board_width}mm x {self.config.board_height}mm')
            lines.append('"""')
            lines.append('')
            lines.append('import pcbnew')
            lines.append('from pcbnew import VECTOR2I, FromMM, ToMM')
            lines.append('')
            lines.append('# Get current board')
            lines.append('board = pcbnew.GetBoard()')
            lines.append('if board is None:')
            lines.append('    print("ERROR: No board loaded")')
            lines.append('    raise SystemExit')
            lines.append('')

            # Design rules
            lines.append('# Design Rules')
            lines.append('settings = board.GetDesignSettings()')
            lines.append(f'settings.SetCopperLayerCount(2)')
            lines.append(f'settings.m_TrackMinWidth = FromMM({self.config.trace_width})')
            lines.append(f'settings.m_ViasMinSize = FromMM({self.config.via_diameter})')
            lines.append(f'settings.m_ViasMinDrill = FromMM({self.config.via_drill})')
            lines.append('')

            # Board outline
            lines.append('# Board Outline')
            x0, y0 = self.config.board_origin_x, self.config.board_origin_y
            x1 = x0 + self.config.board_width
            y1 = y0 + self.config.board_height
            lines.append(f'outline = [')
            lines.append(f'    ({x0}, {y0}), ({x1}, {y0}),')
            lines.append(f'    ({x1}, {y1}), ({x0}, {y1})')
            lines.append(f']')
            lines.append('for i in range(4):')
            lines.append('    seg = pcbnew.PCB_SHAPE(board)')
            lines.append('    seg.SetShape(pcbnew.SHAPE_T_SEGMENT)')
            lines.append('    seg.SetLayer(pcbnew.Edge_Cuts)')
            lines.append('    seg.SetStart(VECTOR2I(FromMM(outline[i][0]), FromMM(outline[i][1])))')
            lines.append('    seg.SetEnd(VECTOR2I(FromMM(outline[(i+1)%4][0]), FromMM(outline[(i+1)%4][1])))')
            lines.append('    board.Add(seg)')
            lines.append('')

            # Nets
            nets = parts_db.get('nets', {})
            lines.append('# Create Nets')
            lines.append('net_map = {}')
            for net_name in nets.keys():
                lines.append(f'net = pcbnew.NETINFO_ITEM(board, "{net_name}")')
                lines.append(f'board.Add(net)')
                lines.append(f'net_map["{net_name}"] = net')
            lines.append('')

            # Components
            parts = parts_db.get('parts', {})
            lines.append('# Load Footprints')
            for ref, pos in placement.items():
                part = parts.get(ref, {})
                fp_lib = part.get('kicad_footprint', part.get('footprint', 'Package_SO:SOIC-8'))
                value = part.get('value', '')
                rotation = getattr(pos, 'rotation', 0)

                lines.append(f'# {ref}')
                lines.append(f'fp = pcbnew.FootprintLoad("{fp_lib.split(":")[0]}", "{fp_lib.split(":")[-1]}")')
                lines.append(f'if fp:')
                lines.append(f'    fp.SetReference("{ref}")')
                lines.append(f'    fp.SetValue("{value}")')
                lines.append(f'    fp.SetPosition(VECTOR2I(FromMM({pos.x}), FromMM({pos.y})))')
                lines.append(f'    fp.SetOrientationDegrees({rotation})')
                lines.append(f'    board.Add(fp)')

                # Set pad nets
                for pin in part.get('pins', []):
                    net = pin.get('net', '')
                    if net:
                        pin_num = pin.get('number', '1')
                        lines.append(f'    pad = fp.FindPadByNumber("{pin_num}")')
                        lines.append(f'    if pad and "{net}" in net_map:')
                        lines.append(f'        pad.SetNet(net_map["{net}"])')
                lines.append('')

            # Tracks
            lines.append('# Create Tracks')
            for net_name, route in routes.items():
                if hasattr(route, 'segments'):
                    segments = route.segments
                elif isinstance(route, dict):
                    segments = route.get('segments', [])
                else:
                    continue

                for seg in segments:
                    if hasattr(seg, 'start'):
                        start, end = seg.start, seg.end
                        width = seg.width
                        layer = seg.layer
                    else:
                        start = seg.get('start', (0, 0))
                        end = seg.get('end', (0, 0))
                        width = seg.get('width', 0.25)
                        layer = seg.get('layer', 'F.Cu')

                    layer_id = 'pcbnew.F_Cu' if layer == 'F.Cu' else 'pcbnew.B_Cu'
                    lines.append(f'track = pcbnew.PCB_TRACK(board)')
                    lines.append(f'track.SetStart(VECTOR2I(FromMM({start[0]}), FromMM({start[1]})))')
                    lines.append(f'track.SetEnd(VECTOR2I(FromMM({end[0]}), FromMM({end[1]})))')
                    lines.append(f'track.SetWidth(FromMM({width}))')
                    lines.append(f'track.SetLayer({layer_id})')
                    lines.append(f'if "{net_name}" in net_map:')
                    lines.append(f'    track.SetNet(net_map["{net_name}"])')
                    lines.append(f'board.Add(track)')
            lines.append('')

            # Vias
            if vias:
                lines.append('# Create Vias')
                for via in vias:
                    if isinstance(via, dict):
                        x, y = via.get('position', (0, 0))
                        net = via.get('net', '')
                    else:
                        x, y = via[0], via[1] if len(via) > 1 else (0, 0)
                        net = ''

                    lines.append(f'via = pcbnew.PCB_VIA(board)')
                    lines.append(f'via.SetPosition(VECTOR2I(FromMM({x}), FromMM({y})))')
                    lines.append(f'via.SetDrill(FromMM({self.config.via_drill}))')
                    lines.append(f'via.SetWidth(FromMM({self.config.via_diameter}))')
                    lines.append(f'via.SetViaType(pcbnew.VIATYPE_THROUGH)')
                    if net:
                        lines.append(f'if "{net}" in net_map:')
                        lines.append(f'    via.SetNet(net_map["{net}"])')
                    lines.append(f'board.Add(via)')
                lines.append('')

            # Ground zone
            lines.append('# Create Ground Zone on B.Cu')
            lines.append('if "GND" in net_map:')
            lines.append(f'    zone = pcbnew.ZONE(board)')
            lines.append(f'    zone.SetLayer(pcbnew.B_Cu)')
            lines.append(f'    zone.SetNet(net_map["GND"])')
            lines.append(f'    zone.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_NEVER)')
            lines.append(f'    outline = zone.Outline()')
            lines.append(f'    outline.NewOutline()')
            lines.append(f'    outline.Append(FromMM({x0}), FromMM({y0}))')
            lines.append(f'    outline.Append(FromMM({x1}), FromMM({y0}))')
            lines.append(f'    outline.Append(FromMM({x1}), FromMM({y1}))')
            lines.append(f'    outline.Append(FromMM({x0}), FromMM({y1}))')
            lines.append(f'    board.Add(zone)')
            lines.append('')

            # Finish
            lines.append('# Fill zones and refresh')
            lines.append('filler = pcbnew.ZONE_FILLER(board)')
            lines.append('filler.Fill(board.Zones())')
            lines.append('pcbnew.Refresh()')
            lines.append('print("PCB generated successfully!")')

            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))

            self.files_generated.append(filepath)
            return filepath

        except Exception as e:
            self.errors.append(f"Failed to generate KiCad script: {e}")
            return ''

    # =========================================================================
    # FULL MANUFACTURING PACKAGE
    # =========================================================================

    def generate_manufacturing_package(self, parts_db: Dict, placement: Dict,
                                        routes: Dict, vias: List = None,
                                        zones: List = None) -> OutputResult:
        """
        Generate complete manufacturing package.

        Creates:
        - Gerber files (all layers)
        - Excellon drill file
        - BOM (CSV + HTML)
        - Pick and Place
        - KiCad script (for verification)

        This is what you send to a PCB manufacturer like JLCPCB, PCBWay, etc.
        """
        vias = vias or []
        zones = zones or []

        self.files_generated.clear()
        self.errors.clear()

        # Create manufacturing directory
        mfg_dir = os.path.join(self.config.output_dir, 'manufacturing')
        os.makedirs(mfg_dir, exist_ok=True)

        original_dir = self.config.output_dir
        self.config.output_dir = mfg_dir

        try:
            # Generate all files
            self.generate_gerbers(parts_db, placement, routes, vias, zones)
            self.generate_drill(parts_db, placement, vias)
            self._generate_bom(parts_db)
            self._generate_pnp(parts_db, placement)
            self.generate_kicad_script(parts_db, placement, routes, vias, zones)

            # Create README
            readme_path = os.path.join(mfg_dir, 'README.txt')
            with open(readme_path, 'w') as f:
                f.write(f'Manufacturing Package: {self.config.board_name}\n')
                f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'Board Size: {self.config.board_width}mm x {self.config.board_height}mm\n')
                f.write(f'\nFiles:\n')
                for file in self.files_generated:
                    f.write(f'  - {os.path.basename(file)}\n')
            self.files_generated.append(readme_path)

        finally:
            self.config.output_dir = original_dir

        return OutputResult(
            success=len(self.errors) == 0,
            files_generated=self.files_generated.copy(),
            errors=self.errors.copy()
        )
