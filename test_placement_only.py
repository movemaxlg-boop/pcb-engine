"""
Placement-only test — runs placement, then generates a BARE KiCad PCB
with ONLY the board outline and rectangular courtyards + pads.
No traces, no pours, no silkscreen.

Auto-generates a provenance_report.md alongside the .kicad_pcb showing
every step, algorithm, engine, and tool used to produce the output.
"""
import sys, os, time
from datetime import datetime
sys.path.insert(0, '.')

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.footprint_resolver import FootprintResolver
from test_harness import TestBoards

OUTPUT_DIR = r'D:\Anas\tmp\output'
OUTPUT_NAME = 'placement_only'


# =============================================================================
# PROVENANCE COLLECTOR
# =============================================================================

class ProvenanceCollector:
    """Collects execution data for the provenance report."""

    def __init__(self, board_name: str):
        self.board_name = board_name
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Input
        self.component_count = 0
        self.net_count = 0
        self.board_initial = ''
        self.board_final = ''
        self.auto_sized = False

        # Footprint resolution per component
        self.fp_resolutions = []   # list of (ref, name, fp, tier, body_w, body_h, pads)
        self.tier_counts = {}

        # Placement
        self.placement_algorithm = ''
        self.placement_time = 0.0
        self.placement_cost = 0.0
        self.placement_wirelength = 0.0
        self.placement_overlap_area = 0.0
        self.placement_iterations = 0
        self.placement_converged = False
        self.placement_config = {}

        # Output
        self.output_path = ''
        self.output_size = 0
        self.net_count_output = 0
        self.footprints_written = 0

        # Quality
        self.overlaps = []     # list of (ref1, ref2, dx, min_dx, dy, min_dy)
        self.boundary_violations = []  # list of (ref, x, y, cw, ch, bw, bh)

        # Timing
        self.timings = {}  # step_name -> seconds

    def record_timing(self, step: str, elapsed: float):
        self.timings[step] = elapsed

    def resolve_footprints(self, parts_db):
        """Resolve all footprints and record which tier each used."""
        resolver = FootprintResolver.get_instance()
        # Clear memory cache so we get fresh tier info per component
        resolver._memory_cache.clear()

        seen_fps = {}  # fp_name -> tier (first resolution)
        for ref in sorted(parts_db.get('parts', {}).keys()):
            part = parts_db['parts'][ref]
            fp_name = part.get('footprint', 'unknown')
            name = part.get('name', ref)

            # If we've seen this footprint before, it'll be memory_cache
            # Record the original tier, not the cache hit
            if fp_name in seen_fps:
                tier = seen_fps[fp_name]
            else:
                fp_def = resolver.resolve(fp_name)
                tier = resolver._last_tier
                seen_fps[fp_name] = tier

            # Resolve again (will be cached) to get the definition
            fp_def = resolver.resolve(fp_name)
            pad_count = len(fp_def.pad_positions)

            self.fp_resolutions.append((
                ref, name, fp_name, tier,
                fp_def.body_width, fp_def.body_height, pad_count
            ))
            self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1

    def generate_report(self) -> str:
        """Generate the provenance markdown report."""
        total_time = time.time() - self.start_time
        overlap_count = len(self.overlaps)
        boundary_ok = len(self.boundary_violations) == 0
        status = 'PASS' if overlap_count == 0 and boundary_ok else 'PARTIAL'

        lines = []

        # === HEADER ===
        lines.append(f'# Provenance Report: {self.board_name}')
        lines.append(f'')
        lines.append(f'**Generated:** {self.timestamp}')
        lines.append(f'**Engine:** PCB Engine (Placement-Only Mode)')
        lines.append(f'**Status:** {status}')
        lines.append(f'**Total Time:** {total_time:.1f}s')
        lines.append(f'')

        # === INPUT ===
        lines.append(f'## Input')
        lines.append(f'')
        lines.append(f'| Metric | Value |')
        lines.append(f'|--------|-------|')
        lines.append(f'| Components | {self.component_count} |')
        lines.append(f'| Nets | {self.net_count} |')
        lines.append(f'| Board (initial) | {self.board_initial} |')
        lines.append(f'| Board (final) | {self.board_final} |')
        lines.append(f'| Auto-sized | {"Yes" if self.auto_sized else "No"} |')
        lines.append(f'')

        # === COMPONENTS TABLE ===
        lines.append(f'### Components')
        lines.append(f'')
        lines.append(f'| Ref | Name | Footprint | Resolver Tier | Body (mm) | Pads |')
        lines.append(f'|-----|------|-----------|---------------|-----------|------|')
        for ref, name, fp, tier, bw, bh, pads in self.fp_resolutions:
            lines.append(f'| {ref} | {name} | {fp} | {tier} | {bw:.1f} x {bh:.1f} | {pads} |')
        lines.append(f'')

        # === FOOTPRINT RESOLUTION STATS ===
        lines.append(f'### Footprint Resolution Stats')
        lines.append(f'')
        tier_order = ['memory_cache', 'disk_cache', 'kicad_exact', 'hardcoded',
                      'kicad_search', 'name_inference', 'default']
        tier_labels = {
            'memory_cache': 'Memory cache (Tier 1)',
            'disk_cache': 'Disk cache (Tier 2)',
            'kicad_exact': 'KiCad exact path (Tier 3a)',
            'hardcoded': 'Hardcoded library (Tier 4)',
            'kicad_search': 'KiCad fuzzy search (Tier 3b)',
            'name_inference': 'Name inference (Tier 5)',
            'default': 'Default fallback (Tier 6)',
        }
        lines.append(f'| Tier | Hits |')
        lines.append(f'|------|------|')
        for tier in tier_order:
            count = self.tier_counts.get(tier, 0)
            label = tier_labels.get(tier, tier)
            if count > 0:
                lines.append(f'| {label} | {count} |')
        # Show any tiers not in the predefined list
        for tier, count in self.tier_counts.items():
            if tier not in tier_order:
                lines.append(f'| {tier} | {count} |')
        lines.append(f'')

        # === PLACEMENT ===
        lines.append(f'## Placement')
        lines.append(f'')
        lines.append(f'| Parameter | Value |')
        lines.append(f'|-----------|-------|')
        lines.append(f'| Algorithm | {self.placement_algorithm} |')
        lines.append(f'| Time | {self.placement_time:.1f}s |')
        lines.append(f'| Iterations | {self.placement_iterations} |')
        lines.append(f'| Converged | {"Yes" if self.placement_converged else "No"} |')
        lines.append(f'| Final cost | {self.placement_cost:.1f} |')
        lines.append(f'| Total wirelength | {self.placement_wirelength:.1f} mm |')
        lines.append(f'| Overlap area | {self.placement_overlap_area:.2f} mm2 |')
        lines.append(f'')
        if self.placement_config:
            lines.append(f'### Placement Config')
            lines.append(f'')
            lines.append(f'| Setting | Value |')
            lines.append(f'|---------|-------|')
            for k, v in sorted(self.placement_config.items()):
                lines.append(f'| {k} | {v} |')
            lines.append(f'')

        # === OUTPUT GENERATION ===
        lines.append(f'## Output Generation')
        lines.append(f'')
        lines.append(f'| Item | Value |')
        lines.append(f'|------|-------|')
        lines.append(f'| File | {os.path.basename(self.output_path)} |')
        lines.append(f'| Size | {self.output_size:,} bytes |')
        lines.append(f'| Layers | F.Cu, B.Cu, Edge.Cuts, F.CrtYd, F.Fab, F.SilkS, F.Mask |')
        lines.append(f'| Nets declared | {self.net_count_output} |')
        lines.append(f'| Footprints written | {self.footprints_written} |')
        lines.append(f'')

        # === QUALITY CHECKS ===
        lines.append(f'## Quality Checks')
        lines.append(f'')

        # Overlaps
        lines.append(f'### Courtyard Overlaps: {overlap_count}')
        lines.append(f'')
        if self.overlaps:
            lines.append(f'| Component A | Component B | dx | min_dx | dy | min_dy |')
            lines.append(f'|-------------|-------------|-----|--------|-----|--------|')
            for ref1, ref2, dx, min_dx, dy, min_dy in self.overlaps:
                lines.append(f'| {ref1} | {ref2} | {dx:.2f} | {min_dx:.2f} | {dy:.2f} | {min_dy:.2f} |')
        else:
            lines.append(f'None - all courtyards clear.')
        lines.append(f'')

        # Boundary
        lines.append(f'### Boundary Check: {"ALL WITHIN BOUNDS" if boundary_ok else f"{len(self.boundary_violations)} VIOLATIONS"}')
        lines.append(f'')
        if self.boundary_violations:
            lines.append(f'| Ref | Position | Courtyard | Board |')
            lines.append(f'|-----|----------|-----------|-------|')
            for ref, x, y, cw, ch, bw, bh in self.boundary_violations:
                lines.append(f'| {ref} | ({x:.2f}, {y:.2f}) | {cw:.1f} x {ch:.1f} | {bw} x {bh} |')
            lines.append(f'')

        # === TIMING BREAKDOWN ===
        lines.append(f'## Timing')
        lines.append(f'')
        lines.append(f'| Step | Time |')
        lines.append(f'|------|------|')
        for step, elapsed in self.timings.items():
            lines.append(f'| {step} | {elapsed:.2f}s |')
        lines.append(f'| **Total** | **{total_time:.1f}s** |')
        lines.append(f'')

        # === TOOLS & ENGINES ===
        lines.append(f'## Tools & Engines Used')
        lines.append(f'')
        lines.append(f'| Tool | Purpose | Version/Details |')
        lines.append(f'|------|---------|-----------------|')
        lines.append(f'| PlacementEngine | Component placement | {self.placement_algorithm} algorithm |')
        lines.append(f'| FootprintResolver | Footprint data lookup | 6-tier chain, {len(FootprintResolver.get_instance()._disk_cache)} disk cache entries |')
        lines.append(f'| OccupancyGrid | Overlap prevention | 0.5mm cell size |')
        lines.append(f'| KiCad .kicad_mod parser | Pad/courtyard extraction | Regex-based S-expr parser |')
        lines.append(f'| KiCad PCB writer | Output generation | v20240108 format |')
        lines.append(f'')

        lines.append(f'---')
        lines.append(f'*Report auto-generated by PCB Engine Provenance System*')

        return '\n'.join(lines)


# =============================================================================
# ORIGINAL TEST FUNCTIONS (with provenance hooks)
# =============================================================================

def run_placement(parts_db, prov: ProvenanceCollector):
    """Run just the placement engine and return positions."""
    board = parts_db.get('board', {})
    bw = board.get('width', 0)
    bh = board.get('height', 0)
    auto = (bw == 0 or bh == 0)

    prov.auto_sized = auto
    prov.board_initial = 'AUTO' if auto else f'{bw} x {bh} mm'

    config = PlacementConfig(
        board_width=float(bw) if not auto else 80.0,
        board_height=float(bh) if not auto else 60.0,
        auto_board_size=auto,
        origin_x=0.0,
        origin_y=0.0,
        algorithm='hybrid',
    )

    prov.placement_config = {
        'board_width': config.board_width,
        'board_height': config.board_height,
        'auto_board_size': config.auto_board_size,
        'algorithm': config.algorithm,
        'origin': f'({config.origin_x}, {config.origin_y})',
    }

    engine = PlacementEngine(config)

    # Build adjacency graph from nets
    adjacency = {}
    nets = parts_db.get('nets', {})
    for net_name, net_info in nets.items():
        pins = net_info.get('pins', []) if isinstance(net_info, dict) else []
        refs = set()
        for p in pins:
            if isinstance(p, str) and '.' in p:
                refs.add(p.split('.')[0])
        refs = list(refs)
        for i, r1 in enumerate(refs):
            if r1 not in adjacency:
                adjacency[r1] = {}
            for r2 in refs[i+1:]:
                adjacency[r1][r2] = adjacency[r1].get(r2, 0) + 1
                if r2 not in adjacency:
                    adjacency[r2] = {}
                adjacency[r2][r1] = adjacency[r2].get(r1, 0) + 1

    graph = {'adjacency': adjacency}

    t0 = time.time()
    result = engine.place(parts_db, graph)
    elapsed = time.time() - t0

    # Record to provenance
    prov.placement_algorithm = result.algorithm_used
    prov.placement_time = elapsed
    prov.placement_cost = result.cost
    prov.placement_wirelength = result.wirelength
    prov.placement_overlap_area = result.overlap_area
    prov.placement_iterations = result.iterations
    prov.placement_converged = result.converged
    prov.record_timing('Placement (FD+SA)', elapsed)

    print(f"Placement: {len(result.positions)} components in {elapsed:.1f}s")
    print(f"  Algorithm: {result.algorithm_used}")
    print(f"  Board: {result.board_width}x{result.board_height}mm" +
          (" (auto-sized)" if auto else ""))
    print(f"  Cost: {result.cost:.1f}")
    print(f"  Wirelength: {result.wirelength:.1f}mm")
    print(f"  Overlaps: {result.overlap_area:.2f}")

    # Update parts_db board size for downstream
    if result.board_width > 0 and result.board_height > 0:
        parts_db.setdefault('board', {})['width'] = result.board_width
        parts_db.setdefault('board', {})['height'] = result.board_height
        prov.board_final = f'{result.board_width} x {result.board_height} mm'

    return result.positions


def generate_bare_kicad(parts_db, positions, output_path, prov: ProvenanceCollector):
    """Generate a .kicad_pcb file with ONLY board outline + component courtyards."""
    t0 = time.time()

    board = parts_db.get('board', {})
    bw = board.get('width', 50)
    bh = board.get('height', 40)

    lines = []
    lines.append('(kicad_pcb (version 20240108) (generator "pcb_engine_placement_test")')
    lines.append('  (general (thickness 1.6) (legacy_teardrops no))')
    lines.append('')

    # Layer definitions (minimal)
    lines.append('  (layers')
    lines.append('    (0 "F.Cu" signal)')
    lines.append('    (31 "B.Cu" signal)')
    lines.append('    (36 "B.SilkS" user "B.Silkscreen")')
    lines.append('    (37 "F.SilkS" user "F.Silkscreen")')
    lines.append('    (38 "B.Mask" user "B.Mask")')
    lines.append('    (39 "F.Mask" user "F.Mask")')
    lines.append('    (40 "Dwgs.User" user "User.Drawings")')
    lines.append('    (41 "Cmts.User" user "User.Comments")')
    lines.append('    (44 "Edge.Cuts" user)')
    lines.append('    (46 "B.CrtYd" user "B.Courtyard")')
    lines.append('    (47 "F.CrtYd" user "F.Courtyard")')
    lines.append('    (48 "B.Fab" user "B.Fab")')
    lines.append('    (49 "F.Fab" user "F.Fab")')
    lines.append('  )')
    lines.append('')

    # Setup (design rules)
    lines.append('  (setup')
    lines.append('    (pad_to_mask_clearance 0.05)')
    lines.append('    (allow_soldermask_bridges_in_footprints no)')
    lines.append('    (pcbplotparams (layerselection 0x00010fc_ffffffff) (plot_on_all_layers_selection 0x0000000_00000000))')
    lines.append('  )')
    lines.append('')

    # Net declarations
    lines.append('  (net 0 "")')
    net_ids = {'': 0}
    net_idx = 1
    for net_name in sorted(parts_db.get('nets', {}).keys()):
        net_ids[net_name] = net_idx
        lines.append(f'  (net {net_idx} "{net_name}")')
        net_idx += 1
    lines.append('')

    prov.net_count_output = net_idx - 1

    # Board outline (Edge.Cuts)
    lines.append(f'  (gr_rect (start 0 0) (end {bw} {bh}) (stroke (width 0.15) (type default)) (fill none) (layer "Edge.Cuts") (uuid "board-outline"))')
    lines.append('')

    # Footprints — use REAL pad positions/sizes from FootprintResolver
    resolver = FootprintResolver.get_instance()
    fp_count = 0
    for ref, pos in sorted(positions.items()):
        part = parts_db['parts'].get(ref)
        if not part:
            continue

        if isinstance(pos, (list, tuple)):
            cx, cy = pos[0], pos[1]
        elif isinstance(pos, dict):
            cx, cy = pos.get('x', 0), pos.get('y', 0)
        else:
            cx, cy = getattr(pos, 'x', 0), getattr(pos, 'y', 0)

        fp_name = part.get('footprint', 'unknown')
        name = part.get('name', ref)

        # Get REAL footprint definition from resolver
        fp_def = resolver.resolve(fp_name)
        body_w = fp_def.body_width
        body_h = fp_def.body_height
        is_smd = fp_def.is_smd

        # Courtyard from resolver
        court_w, court_h = fp_def.courtyard_size

        lines.append(f'  (footprint "{fp_name}" (layer "F.Cu")')
        lines.append(f'    (at {cx:.4f} {cy:.4f})')
        lines.append(f'    (property "Reference" "{ref}" (at 0 {-court_h/2 - 0.8:.2f}) (layer "F.SilkS") (uuid "ref-{ref}")')
        lines.append(f'      (effects (font (size 1 1) (thickness 0.15))))')
        lines.append(f'    (property "Value" "{name}" (at 0 {court_h/2 + 0.8:.2f}) (layer "F.Fab") (uuid "val-{ref}")')
        lines.append(f'      (effects (font (size 0.8 0.8) (thickness 0.12))))')

        # F.CrtYd rectangle (courtyard)
        lines.append(f'    (fp_rect (start {-court_w/2:.4f} {-court_h/2:.4f}) (end {court_w/2:.4f} {court_h/2:.4f})')
        lines.append(f'      (stroke (width 0.05) (type default)) (fill none) (layer "F.CrtYd") (uuid "crtyd-{ref}"))')

        # F.Fab rectangle (component body)
        lines.append(f'    (fp_rect (start {-body_w/2:.4f} {-body_h/2:.4f}) (end {body_w/2:.4f} {body_h/2:.4f})')
        lines.append(f'      (stroke (width 0.1) (type default)) (fill none) (layer "F.Fab") (uuid "fab-{ref}"))')

        # Build net mapping: pin_number -> net_name from parts_db
        pin_nets = {}
        for pin in part.get('pins', []):
            pnum = str(pin.get('number', ''))
            net = pin.get('net', '')
            if pnum and net:
                pin_nets[pnum] = net

        # Use REAL pad positions and sizes from FootprintResolver
        pad_type = 'smd' if is_smd else 'thru_hole'
        layers = '"F.Cu" "F.Mask"' if is_smd else '"*.Cu" "*.Mask"'
        for pad_num, pad_x, pad_y, pad_w, pad_h in fp_def.pad_positions:
            pnum = str(pad_num)
            net = pin_nets.get(pnum, '')
            net_id = net_ids.get(net, 0)

            lines.append(f'    (pad "{pnum}" {pad_type} roundrect (at {pad_x:.4f} {pad_y:.4f}) (size {pad_w:.4f} {pad_h:.4f}) (layers {layers}) (roundrect_rratio 0.25)')
            if net:
                lines.append(f'      (net {net_id} "{net}")')
            lines.append(f'    )')

        lines.append(f'  )')
        lines.append('')
        fp_count += 1

    lines.append(')')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path)

    prov.output_path = output_path
    prov.output_size = file_size
    prov.footprints_written = fp_count
    prov.record_timing('KiCad output', elapsed)

    print(f"\nKiCad output: {output_path}")
    print(f"  Size: {file_size:,} bytes")


def check_overlaps(parts_db, positions, prov: ProvenanceCollector):
    """Check courtyard overlaps using real footprint data from resolver."""
    t0 = time.time()
    resolver = FootprintResolver.get_instance()
    print()
    print("OVERLAP CHECK:")

    pos_list = list(positions.items())
    for i in range(len(pos_list)):
        ref1, p1 = pos_list[i]
        part1 = parts_db['parts'].get(ref1, {})
        fp1 = resolver.resolve(part1.get('footprint', 'unknown'))
        cw1, ch1 = fp1.courtyard_size
        x1, y1 = (p1[0], p1[1]) if isinstance(p1, (list, tuple)) else p1

        for j in range(i+1, len(pos_list)):
            ref2, p2 = pos_list[j]
            part2 = parts_db['parts'].get(ref2, {})
            fp2 = resolver.resolve(part2.get('footprint', 'unknown'))
            cw2, ch2 = fp2.courtyard_size
            x2, y2 = (p2[0], p2[1]) if isinstance(p2, (list, tuple)) else p2

            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            min_dx = (cw1 + cw2) / 2
            min_dy = (ch1 + ch2) / 2

            if dx < min_dx and dy < min_dy:
                prov.overlaps.append((ref1, ref2, dx, min_dx, dy, min_dy))
                print(f"  OVERLAP: {ref1} <-> {ref2} (dx={dx:.2f} < {min_dx:.2f}, dy={dy:.2f} < {min_dy:.2f})")

    if not prov.overlaps:
        print("  No overlaps — GOOD")
    else:
        print(f"  {len(prov.overlaps)} total overlaps")

    prov.record_timing('Overlap check', time.time() - t0)


def check_boundaries(parts_db, positions, prov: ProvenanceCollector):
    """Check boundary violations using real courtyard data from resolver."""
    t0 = time.time()
    resolver = FootprintResolver.get_instance()
    board = parts_db.get('board', {})
    bw, bh = board['width'], board['height']

    print()
    print("BOUNDARY CHECK:")

    for ref, pos in positions.items():
        part = parts_db['parts'].get(ref, {})
        fp = resolver.resolve(part.get('footprint', 'unknown'))
        cw, ch = fp.courtyard_size
        x, y = (pos[0], pos[1]) if isinstance(pos, (list, tuple)) else pos

        if x - cw/2 < 0 or x + cw/2 > bw or y - ch/2 < 0 or y + ch/2 > bh:
            prov.boundary_violations.append((ref, x, y, cw, ch, bw, bh))
            print(f"  OUT: {ref} at ({x:.2f},{y:.2f}), courtyard {cw:.1f}x{ch:.1f} exceeds {bw}x{bh} board")

    if not prov.boundary_violations:
        print("  All within bounds — GOOD")

    prov.record_timing('Boundary check', time.time() - t0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parts_db = TestBoards.medium_20_parts()
    board = parts_db.get('board', {})
    bw_initial = board.get('width', 0)
    bh_initial = board.get('height', 0)

    if bw_initial and bh_initial:
        board_str = f"{bw_initial}x{bh_initial}mm"
    else:
        board_str = "AUTO (will be calculated from components)"

    # Initialize provenance collector
    prov = ProvenanceCollector(OUTPUT_NAME)
    prov.component_count = len(parts_db['parts'])
    prov.net_count = len(parts_db.get('nets', {}))

    print("=" * 70)
    print(f"PLACEMENT-ONLY — {prov.component_count} components, {board_str}")
    print("  Board outline + component courtyards + pads ONLY")
    print("  No traces, no pours, no silkscreen")
    print("=" * 70)
    print()

    # Resolve footprints (for provenance tracking)
    t0 = time.time()
    prov.resolve_footprints(parts_db)
    prov.record_timing('Footprint resolution', time.time() - t0)

    # Run placement
    positions = run_placement(parts_db, prov)

    # Print component table (using real footprint data from resolver)
    resolver = FootprintResolver.get_instance()
    print()
    print(f"{'Ref':<8} {'Name':<15} {'FP':<12} {'X':>7} {'Y':>7} {'Body':>10} {'Court':>10} {'Pads':>5}")
    print("-" * 80)
    for ref, pos in sorted(positions.items()):
        part = parts_db['parts'].get(ref, {})
        fp_def = resolver.resolve(part.get('footprint', 'unknown'))
        court_w, court_h = fp_def.courtyard_size
        x, y = (pos[0], pos[1]) if isinstance(pos, (list, tuple)) else pos
        print(f"{ref:<8} {part.get('name','?'):<15} {part.get('footprint','?'):<12} "
              f"{x:7.2f} {y:7.2f} "
              f"{fp_def.body_width:.1f}x{fp_def.body_height:.1f}{'':>3} "
              f"{court_w:.1f}x{court_h:.1f}"
              f"{len(fp_def.pad_positions):>5}")

    # Generate KiCad file
    out_dir = os.path.join(OUTPUT_DIR, 'placement_only')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{OUTPUT_NAME}.kicad_pcb')

    generate_bare_kicad(parts_db, positions, out_path, prov)

    # Quality checks
    check_overlaps(parts_db, positions, prov)
    check_boundaries(parts_db, positions, prov)

    # Generate provenance report
    report_path = os.path.join(out_dir, 'provenance_report.md')
    report_content = prov.generate_report()
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\nProvenance report: {report_path}")

    print()
    print(f"Open in KiCad: {out_dir}")


if __name__ == '__main__':
    main()
