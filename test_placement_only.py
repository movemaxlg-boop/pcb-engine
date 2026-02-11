"""
Placement-only test — runs placement, then generates a BARE KiCad PCB
with ONLY the board outline and rectangular courtyards + pads.
No traces, no pours, no silkscreen.
"""
import sys, os, time
sys.path.insert(0, '.')

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from test_harness import TestBoards

OUTPUT_DIR = r'D:\Anas\tmp\output'
OUTPUT_NAME = 'placement_only'


def run_placement(parts_db):
    """Run just the placement engine and return positions."""
    board = parts_db.get('board', {})
    bw = board.get('width', 50)
    bh = board.get('height', 40)

    config = PlacementConfig(
        board_width=float(bw),
        board_height=float(bh),
        origin_x=0.0,
        origin_y=0.0,
        algorithm='hybrid',
    )

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

    print(f"Placement: {len(result.positions)} components in {elapsed:.1f}s")
    print(f"  Algorithm: {result.algorithm_used}")
    print(f"  Cost: {result.cost:.1f}")
    print(f"  Wirelength: {result.wirelength:.1f}mm")
    print(f"  Overlaps: {result.overlap_area:.2f}")

    return result.positions


def generate_bare_kicad(parts_db, positions, output_path):
    """Generate a .kicad_pcb file with ONLY board outline + component courtyards."""
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

    # Board outline (Edge.Cuts)
    lines.append(f'  (gr_rect (start 0 0) (end {bw} {bh}) (stroke (width 0.15) (type default)) (fill none) (layer "Edge.Cuts") (uuid "board-outline"))')
    lines.append('')

    # Footprints — each component as a rectangle courtyard + pads
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
        size = part.get('size', (1.0, 0.5))
        body_w, body_h = size[0], size[1]

        # Courtyard margin
        margin = 0.5 if max(body_w, body_h) > 3 else 0.25
        cw = body_w + 2 * margin
        ch = body_h + 2 * margin

        lines.append(f'  (footprint "{fp_name}" (layer "F.Cu")')
        lines.append(f'    (at {cx:.4f} {cy:.4f})')
        lines.append(f'    (property "Reference" "{ref}" (at 0 {-ch/2 - 0.8:.2f}) (layer "F.SilkS") (uuid "ref-{ref}")')
        lines.append(f'      (effects (font (size 1 1) (thickness 0.15))))')
        lines.append(f'    (property "Value" "{name}" (at 0 {ch/2 + 0.8:.2f}) (layer "F.Fab") (uuid "val-{ref}")')
        lines.append(f'      (effects (font (size 0.8 0.8) (thickness 0.12))))')

        # F.CrtYd rectangle (courtyard)
        lines.append(f'    (fp_rect (start {-cw/2:.4f} {-ch/2:.4f}) (end {cw/2:.4f} {ch/2:.4f})')
        lines.append(f'      (stroke (width 0.05) (type default)) (fill none) (layer "F.CrtYd") (uuid "crtyd-{ref}"))')

        # F.Fab rectangle (component body)
        lines.append(f'    (fp_rect (start {-body_w/2:.4f} {-body_h/2:.4f}) (end {body_w/2:.4f} {body_h/2:.4f})')
        lines.append(f'      (stroke (width 0.1) (type default)) (fill none) (layer "F.Fab") (uuid "fab-{ref}"))')

        # Pads (SMD)
        pins = part.get('pins', [])
        for pin in pins:
            pnum = pin.get('number', '?')
            net = pin.get('net', '')
            phys = pin.get('physical', {})
            ox = phys.get('offset_x', 0)
            oy = phys.get('offset_y', 0)
            net_id = net_ids.get(net, 0)

            # Pad size based on footprint
            if '0402' in fp_name:
                pw, ph = 0.6, 0.5
            elif '0603' in fp_name:
                pw, ph = 0.8, 0.75
            elif '0805' in fp_name:
                pw, ph = 1.0, 1.0
            elif 'SOT-23' in fp_name or 'SOT-223' in fp_name:
                pw, ph = 1.0, 0.7
            elif 'QFN' in fp_name or 'QFP' in fp_name or 'SOIC' in fp_name:
                pw, ph = 0.6, 0.3
            elif 'USB' in fp_name:
                pw, ph = 0.6, 1.2
            elif 'LGA' in fp_name:
                pw, ph = 0.5, 0.35
            else:
                pw, ph = 0.8, 0.8

            lines.append(f'    (pad "{pnum}" smd rect (at {ox:.4f} {oy:.4f}) (size {pw} {ph}) (layers "F.Cu" "F.Mask")')
            if net:
                lines.append(f'      (net {net_id} "{net}")')
            lines.append(f'    )')

        lines.append(f'  )')
        lines.append('')

    lines.append(')')

    # Write file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nKiCad output: {output_path}")
    file_size = os.path.getsize(output_path)
    print(f"  Size: {file_size:,} bytes")


def main():
    parts_db = TestBoards.medium_20_parts()
    board = parts_db.get('board', {})

    print("=" * 70)
    print(f"PLACEMENT-ONLY — {len(parts_db['parts'])} components, {board['width']}x{board['height']}mm")
    print("  Board outline + component courtyards + pads ONLY")
    print("  No traces, no pours, no silkscreen")
    print("=" * 70)
    print()

    positions = run_placement(parts_db)

    # Print component table
    print()
    print(f"{'Ref':<8} {'Name':<15} {'FP':<12} {'X':>7} {'Y':>7} {'Body':>10} {'Court':>10}")
    print("-" * 75)
    for ref, pos in sorted(positions.items()):
        part = parts_db['parts'].get(ref, {})
        size = part.get('size', (0, 0))
        margin = 0.5 if max(size) > 3 else 0.25
        if isinstance(pos, (list, tuple)):
            x, y = pos[0], pos[1]
        else:
            x, y = pos

        print(f"{ref:<8} {part.get('name','?'):<15} {part.get('footprint','?'):<12} "
              f"{x:7.2f} {y:7.2f} "
              f"{size[0]:.1f}x{size[1]:.1f}{'':>3} "
              f"{size[0]+2*margin:.1f}x{size[1]+2*margin:.1f}")

    # Generate KiCad file
    out_dir = os.path.join(OUTPUT_DIR, 'placement_only')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{OUTPUT_NAME}.kicad_pcb')

    generate_bare_kicad(parts_db, positions, out_path)

    # Check overlaps
    print()
    print("OVERLAP CHECK:")
    overlap_count = 0
    pos_list = list(positions.items())
    for i in range(len(pos_list)):
        ref1, p1 = pos_list[i]
        part1 = parts_db['parts'].get(ref1, {})
        s1 = part1.get('size', (0, 0))
        m1 = 0.5 if max(s1) > 3 else 0.25
        if isinstance(p1, (list, tuple)):
            x1, y1 = p1[0], p1[1]
        else:
            x1, y1 = p1

        for j in range(i+1, len(pos_list)):
            ref2, p2 = pos_list[j]
            part2 = parts_db['parts'].get(ref2, {})
            s2 = part2.get('size', (0, 0))
            m2 = 0.5 if max(s2) > 3 else 0.25
            if isinstance(p2, (list, tuple)):
                x2, y2 = p2[0], p2[1]
            else:
                x2, y2 = p2

            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            min_dx = (s1[0] + 2*m1 + s2[0] + 2*m2) / 2
            min_dy = (s1[1] + 2*m1 + s2[1] + 2*m2) / 2

            if dx < min_dx and dy < min_dy:
                overlap_count += 1
                print(f"  OVERLAP: {ref1} <-> {ref2} (dx={dx:.2f} < {min_dx:.2f}, dy={dy:.2f} < {min_dy:.2f})")

    if overlap_count == 0:
        print("  No overlaps — GOOD")
    else:
        print(f"  {overlap_count} total overlaps")

    # Boundary check
    print()
    print("BOUNDARY CHECK:")
    bw, bh = board['width'], board['height']
    violations = 0
    for ref, pos in positions.items():
        part = parts_db['parts'].get(ref, {})
        size = part.get('size', (0, 0))
        margin = 0.5 if max(size) > 3 else 0.25
        cw = size[0] + 2 * margin
        ch = size[1] + 2 * margin
        if isinstance(pos, (list, tuple)):
            x, y = pos[0], pos[1]
        else:
            x, y = pos

        if x - cw/2 < 0 or x + cw/2 > bw or y - ch/2 < 0 or y + ch/2 > bh:
            violations += 1
            print(f"  OUT: {ref} at ({x:.2f},{y:.2f}), courtyard {cw:.1f}x{ch:.1f} exceeds {bw}x{bh} board")

    if violations == 0:
        print("  All within bounds — GOOD")

    print()
    print(f"Open in KiCad: {out_dir}")


if __name__ == '__main__':
    main()
