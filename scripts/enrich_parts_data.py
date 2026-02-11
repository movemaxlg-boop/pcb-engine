"""
Enrich Parts Data with Mechanical Specs
=========================================

One-time script that fills in the empty mechanical fields in parts_data
JSON files by parsing the referenced KiCad .kicad_mod footprint files.

Fields updated:
- mechanical.package: Package name (e.g., "SOIC-8")
- mechanical.length: Body width from F.Fab outline (mm)
- mechanical.width: Body height from F.Fab outline (mm)
- mechanical.pin_pitch: Computed from pad positions (mm)

Usage:
    cd D:\\Anas\\projects\\pcb-engine
    python scripts/enrich_parts_data.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcb_engine.kicad_footprint_parser import KiCadFootprintParser
from pcb_engine.paths import KICAD_FOOTPRINT_BASE


def resolve_footprint_path(ref: str) -> Path | None:
    """Resolve a footprint reference to a .kicad_mod file path."""
    if ':' in ref:
        lib, fp_name = ref.split(':', 1)
        candidate = KICAD_FOOTPRINT_BASE / f"{lib}.pretty" / f"{fp_name}.kicad_mod"
        if candidate.exists():
            return candidate
    return None


def compute_pitch(pad_positions: list) -> float:
    """Compute minimum pitch from pad positions."""
    if len(pad_positions) < 2:
        return 0.0

    # Get sorted unique Y positions for left-column pads (negative X)
    left_ys = sorted(set(
        round(p[2], 3) for p in pad_positions
        if p[1] < 0  # left side
    ))

    if len(left_ys) >= 2:
        # Compute minimum Y gap
        gaps = [left_ys[i+1] - left_ys[i] for i in range(len(left_ys) - 1)]
        min_gap = min(gaps)
        if min_gap > 0.01:
            return round(min_gap, 3)

    # Try bottom side (positive Y, varying X)
    bottom_xs = sorted(set(
        round(p[1], 3) for p in pad_positions
        if p[2] > 0 and abs(p[2] - max(p2[2] for p2 in pad_positions)) < 0.5
    ))

    if len(bottom_xs) >= 2:
        gaps = [bottom_xs[i+1] - bottom_xs[i] for i in range(len(bottom_xs) - 1)]
        min_gap = min(gaps)
        if min_gap > 0.01:
            return round(min_gap, 3)

    # Fallback: min distance between any two adjacent pads
    positions = [(p[1], p[2]) for p in pad_positions]
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[j][0] - positions[i][0]
            dy = positions[j][1] - positions[i][1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist > 0.01 and dist < min_dist:
                min_dist = dist

    return round(min_dist, 3) if min_dist < float('inf') else 0.0


def extract_package_name(fp_name: str) -> str:
    """Extract clean package name from footprint name."""
    # Remove library prefix
    if ':' in fp_name:
        fp_name = fp_name.split(':', 1)[1]

    # Take first segment before underscore-delimited dimensions
    # "SOIC-8_3.9x4.9mm_P1.27mm" → "SOIC-8"
    parts = fp_name.split('_')
    if parts:
        return parts[0]
    return fp_name


def main():
    print("=" * 60)
    print("Enriching Parts Data with Mechanical Specs")
    print("=" * 60)

    parts_data_dir = project_root / 'circuit_intelligence' / 'parts_data'
    if not parts_data_dir.exists():
        print(f"ERROR: {parts_data_dir} not found")
        sys.exit(1)

    if not KICAD_FOOTPRINT_BASE.exists():
        print(f"ERROR: KiCad footprints not found at {KICAD_FOOTPRINT_BASE}")
        sys.exit(1)

    # Cache parsed footprints to avoid re-parsing the same .kicad_mod
    fp_cache = {}

    total_parts = 0
    enriched = 0
    skipped_no_fp = 0
    skipped_no_file = 0
    failed = 0

    start = time.time()

    json_files = sorted(parts_data_dir.glob('*.json'))
    print(f"\nProcessing {len(json_files)} JSON files...\n")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ERROR reading {json_file.name}: {e}")
            continue

        parts = data.get('parts', [])
        file_enriched = 0

        for part in parts:
            total_parts += 1
            fp_ref = part.get('footprint', '')

            if not fp_ref:
                skipped_no_fp += 1
                continue

            # Check cache
            if fp_ref in fp_cache:
                fp_def = fp_cache[fp_ref]
            else:
                filepath = resolve_footprint_path(fp_ref)
                if filepath is None:
                    skipped_no_file += 1
                    fp_cache[fp_ref] = None
                    continue

                try:
                    fp_def = KiCadFootprintParser.parse_file(str(filepath))
                    fp_cache[fp_ref] = fp_def
                except Exception:
                    failed += 1
                    fp_cache[fp_ref] = None
                    continue

            if fp_def is None:
                continue

            # Update mechanical fields
            mech = part.get('mechanical', {})
            if not mech.get('package') or mech.get('package') == '':
                mech['package'] = extract_package_name(fp_ref)
            if mech.get('length', 0.0) == 0.0:
                mech['length'] = round(fp_def.body_width, 3)
            if mech.get('width', 0.0) == 0.0:
                mech['width'] = round(fp_def.body_height, 3)
            if mech.get('pin_pitch', 0.0) == 0.0 and len(fp_def.pad_positions) >= 2:
                mech['pin_pitch'] = compute_pitch(fp_def.pad_positions)

            part['mechanical'] = mech
            enriched += 1
            file_enriched += 1

        # Write back
        if file_enriched > 0:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  {json_file.name}: {file_enriched}/{len(parts)} enriched")

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total parts:       {total_parts}")
    print(f"  Enriched:          {enriched}")
    print(f"  No footprint ref:  {skipped_no_fp}")
    print(f"  No .kicad_mod:     {skipped_no_file}")
    print(f"  Parse failures:    {failed}")
    print(f"  Unique footprints: {len(fp_cache)}")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
