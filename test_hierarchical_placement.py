"""
Hierarchical Placement Test — Compares flat vs hierarchical engines
===================================================================

Runs both Engine A (flat PlacementEngine) and Engine B (HierarchicalPlacementEngine)
on the same test board, compares results, and generates KiCad PCB files for both.

Usage:
    python test_hierarchical_placement.py              # medium (19 parts)
    python test_hierarchical_placement.py complex      # complex (52 parts)
    python test_hierarchical_placement.py both         # run both boards
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcb_engine.placement_engine import PlacementConfig
from pcb_engine.hierarchical_placement import PlacementComparator
from pcb_engine.footprint_resolver import FootprintResolver
from test_harness import TestBoards
from collections import defaultdict

OUTPUT_DIR = r'D:\Anas\tmp\output\hierarchical_test'


def build_graph(parts_db):
    """Build adjacency graph from parts_db nets."""
    adjacency = defaultdict(lambda: defaultdict(int))
    for net_name, net_data in parts_db.get('nets', {}).items():
        pins = net_data.get('pins', [])
        refs = list(set(
            p.split('.')[0] if '.' in p else p
            for p in pins
        ))
        for i, r1 in enumerate(refs):
            for r2 in refs[i + 1:]:
                adjacency[r1][r2] += 1
                adjacency[r2][r1] += 1

    # Convert to regular dict
    return {'adjacency': {k: dict(v) for k, v in adjacency.items()}}


def generate_bare_kicad(parts_db, positions, output_path, board_name='test'):
    """Generate a bare KiCad PCB with courtyards + pads (no traces)."""
    board = parts_db.get('board', {})
    bw = board.get('width', 50)
    bh = board.get('height', 40)

    lines = []
    lines.append(f'(kicad_pcb (version 20240108) (generator "pcb_engine_{board_name}")')
    lines.append('  (general (thickness 1.6) (legacy_teardrops no))')
    lines.append('')

    # Layers
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

    # Setup
    lines.append('  (setup')
    lines.append('    (pad_to_mask_clearance 0.05)')
    lines.append('    (allow_soldermask_bridges_in_footprints no)')
    lines.append('    (pcbplotparams (layerselection 0x00010fc_ffffffff) '
                 '(plot_on_all_layers_selection 0x0000000_00000000))')
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

    # Board outline
    lines.append(f'  (gr_rect (start 0 0) (end {bw} {bh}) '
                 f'(stroke (width 0.15) (type default)) (fill none) '
                 f'(layer "Edge.Cuts") (uuid "board-outline"))')
    lines.append('')

    # Footprints
    resolver = FootprintResolver.get_instance()
    for ref, pos in sorted(positions.items()):
        part = parts_db['parts'].get(ref)
        if not part:
            continue

        if isinstance(pos, (list, tuple)):
            cx, cy = pos[0], pos[1]
        else:
            cx, cy = getattr(pos, 'x', 0), getattr(pos, 'y', 0)

        fp_name = part.get('footprint', 'unknown')
        name = part.get('name', ref)

        fp_def = resolver.resolve(fp_name)
        body_w = fp_def.body_width
        body_h = fp_def.body_height
        is_smd = fp_def.is_smd
        court_w, court_h = fp_def.courtyard_size

        lines.append(f'  (footprint "{fp_name}" (layer "F.Cu")')
        lines.append(f'    (at {cx:.4f} {cy:.4f})')
        lines.append(f'    (property "Reference" "{ref}" '
                     f'(at 0 {-court_h / 2 - 0.8:.2f}) (layer "F.SilkS") '
                     f'(uuid "ref-{ref}")')
        lines.append(f'      (effects (font (size 1 1) (thickness 0.15))))')
        lines.append(f'    (property "Value" "{name}" '
                     f'(at 0 {court_h / 2 + 0.8:.2f}) (layer "F.Fab") '
                     f'(uuid "val-{ref}")')
        lines.append(f'      (effects (font (size 0.8 0.8) (thickness 0.12))))')

        # Courtyard
        lines.append(f'    (fp_rect (start {-court_w / 2:.4f} {-court_h / 2:.4f}) '
                     f'(end {court_w / 2:.4f} {court_h / 2:.4f})')
        lines.append(f'      (stroke (width 0.05) (type default)) (fill none) '
                     f'(layer "F.CrtYd") (uuid "crtyd-{ref}"))')

        # Body
        lines.append(f'    (fp_rect (start {-body_w / 2:.4f} {-body_h / 2:.4f}) '
                     f'(end {body_w / 2:.4f} {body_h / 2:.4f})')
        lines.append(f'      (stroke (width 0.1) (type default)) (fill none) '
                     f'(layer "F.Fab") (uuid "fab-{ref}"))')

        # Pads
        pin_nets = {}
        for pin in part.get('pins', []):
            pnum = str(pin.get('number', ''))
            net = pin.get('net', '')
            if pnum and net:
                pin_nets[pnum] = net

        pad_type = 'smd' if is_smd else 'thru_hole'
        layers = '"F.Cu" "F.Mask"' if is_smd else '"*.Cu" "*.Mask"'
        for pad_num, pad_x, pad_y, pad_w, pad_h in fp_def.pad_positions:
            pnum = str(pad_num)
            net = pin_nets.get(pnum, '')

            if not net and abs(pad_x) < 0.1 and abs(pad_y) < 0.1 and pad_w > body_w * 0.4:
                net = 'GND'

            net_id = net_ids.get(net, 0)

            lines.append(f'    (pad "{pnum}" {pad_type} roundrect '
                         f'(at {pad_x:.4f} {pad_y:.4f}) '
                         f'(size {pad_w:.4f} {pad_h:.4f}) '
                         f'(layers {layers}) (roundrect_rratio 0.25)')
            if net:
                lines.append(f'      (net {net_id} "{net}")')
            lines.append(f'    )')

        lines.append(f'  )')
        lines.append('')

    lines.append(')')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    file_size = os.path.getsize(output_path)
    print(f"  KiCad output: {output_path} ({file_size:,} bytes)")


def print_positions(label, positions, parts_db):
    """Print component positions table."""
    resolver = FootprintResolver.get_instance()
    print(f"\n{label} Positions:")
    print(f"  {'Ref':<8} {'Name':<15} {'FP':<12} {'X':>7} {'Y':>7} {'Court':>10}")
    print("  " + "-" * 65)
    for ref, pos in sorted(positions.items()):
        part = parts_db['parts'].get(ref, {})
        fp_def = resolver.resolve(part.get('footprint', 'unknown'))
        court_w, court_h = fp_def.courtyard_size
        x, y = (pos[0], pos[1]) if isinstance(pos, (list, tuple)) else (0, 0)
        print(f"  {ref:<8} {part.get('name', '?'):<15} "
              f"{part.get('footprint', '?'):<12} "
              f"{x:7.2f} {y:7.2f} {court_w:.1f}x{court_h:.1f}")


def run_test(board_name, parts_db):
    """Run flat vs hierarchical comparison on a single board."""
    graph = build_graph(parts_db)

    board = parts_db.get('board', {})
    bw = board.get('width', 50)
    bh = board.get('height', 40)
    n_parts = len(parts_db['parts'])
    n_nets = len(parts_db.get('nets', {}))

    print("=" * 70)
    print(f"  HIERARCHICAL vs FLAT PLACEMENT TEST — {board_name.upper()}")
    print(f"  {n_parts} components, {n_nets} nets, {bw}x{bh}mm board")
    print("=" * 70)

    config = PlacementConfig(
        board_width=bw,
        board_height=bh,
        seed=42,
    )

    t0 = time.time()
    comparator = PlacementComparator(config)
    result = comparator.compare(parts_db, graph)
    elapsed = time.time() - t0

    # Print detailed positions for both
    print_positions("FLAT", result.flat_result.positions, parts_db)
    print_positions("HIERARCHICAL", result.hier_result.positions, parts_db)

    # Generate KiCad outputs for visual comparison
    board_dir = os.path.join(OUTPUT_DIR, board_name)
    os.makedirs(board_dir, exist_ok=True)

    print("\nGenerating KiCad outputs...")
    flat_path = os.path.join(board_dir, 'flat.kicad_pcb')
    hier_path = os.path.join(board_dir, 'hierarchical.kicad_pcb')

    generate_bare_kicad(parts_db, result.flat_result.positions,
                        flat_path, f'{board_name}_flat')
    generate_bare_kicad(parts_db, result.hier_result.positions,
                        hier_path, f'{board_name}_hier')

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\nOpen both in KiCad to visually compare:")
    print(f"  Flat:         {flat_path}")
    print(f"  Hierarchical: {hier_path}")

    return result


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'medium'

    if mode == 'both':
        boards = [('medium', TestBoards.medium_20_parts()),
                  ('complex', TestBoards.complex_50_parts())]
    elif mode == 'complex':
        boards = [('complex', TestBoards.complex_50_parts())]
    else:
        boards = [('medium', TestBoards.medium_20_parts())]

    results = {}
    for board_name, parts_db in boards:
        results[board_name] = run_test(board_name, parts_db)
        print("\n")

    # Summary if multiple boards
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("  MULTI-BOARD SUMMARY")
        print("=" * 70)
        for name, res in results.items():
            flat_wl = res.flat_result.wirelength
            hier_wl = res.hier_result.wirelength
            diff_pct = (hier_wl - flat_wl) / flat_wl * 100 if flat_wl > 0 else 0
            n_parts = len(res.flat_result.positions)
            print(f"  {name:<12} {n_parts:>3} parts  "
                  f"flat={flat_wl:7.1f}mm  hier={hier_wl:7.1f}mm  "
                  f"diff={diff_pct:+6.1f}%  winner={res.winner}")


if __name__ == '__main__':
    main()
