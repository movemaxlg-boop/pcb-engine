"""
Pre-build Footprint Cache
==========================

Parses all footprints referenced by parts_data JSON files (plus common
passives) from KiCad .kicad_mod files and saves them to
pcb_engine/footprint_cache.json.

This cache is committed to the repo so the engine works without KiCad installed.

Usage:
    cd D:\\Anas\\projects\\pcb-engine
    python scripts/prebuild_footprint_cache.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcb_engine.kicad_footprint_parser import KiCadFootprintParser
from pcb_engine.paths import KICAD_FOOTPRINT_BASE, FOOTPRINT_CACHE_FILE


def collect_referenced_footprints() -> set:
    """Collect all unique footprint references from parts_data JSON files."""
    parts_data_dir = project_root / 'circuit_intelligence' / 'parts_data'
    refs = set()

    if not parts_data_dir.exists():
        print(f"  WARNING: {parts_data_dir} not found")
        return refs

    for json_file in sorted(parts_data_dir.glob('*.json')):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for part in data.get('parts', []):
                fp = part.get('footprint', '')
                if fp:
                    refs.add(fp)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Error reading {json_file.name}: {e}")

    return refs


def resolve_footprint_path(ref: str) -> Path | None:
    """Resolve a footprint reference to a .kicad_mod file path."""
    if ':' in ref:
        lib, fp_name = ref.split(':', 1)
        candidate = KICAD_FOOTPRINT_BASE / f"{lib}.pretty" / f"{fp_name}.kicad_mod"
        if candidate.exists():
            return candidate
    return None


def add_common_passives() -> set:
    """Add common passive footprints that are frequently used."""
    common = set()
    passive_libs = ['Capacitor_SMD', 'Resistor_SMD', 'Inductor_SMD']
    sizes = ['0201', '0402', '0603', '0805', '1206', '1210', '2010', '2512']

    for lib in passive_libs:
        lib_dir = KICAD_FOOTPRINT_BASE / f"{lib}.pretty"
        if not lib_dir.exists():
            continue
        prefix = lib.split('_')[0][0]  # C, R, or L
        for size in sizes:
            # Try common naming patterns
            for pattern in [f"{prefix}_{size}_*Metric.kicad_mod"]:
                for f in lib_dir.glob(pattern):
                    common.add(f"{lib}:{f.stem}")

    return common


def main():
    print("=" * 60)
    print("Pre-building Footprint Cache")
    print("=" * 60)

    if not KICAD_FOOTPRINT_BASE.exists():
        print(f"ERROR: KiCad footprints not found at {KICAD_FOOTPRINT_BASE}")
        sys.exit(1)

    # Collect footprint references
    print("\n1. Collecting footprint references from parts_data...")
    refs = collect_referenced_footprints()
    print(f"   Found {len(refs)} unique references")

    print("\n2. Adding common passives...")
    common = add_common_passives()
    refs.update(common)
    print(f"   Added {len(common)} common passives, total: {len(refs)}")

    # Parse each footprint
    print(f"\n3. Parsing .kicad_mod files...")
    start = time.time()
    cache_entries = {}
    parsed = 0
    skipped = 0
    failed = 0

    for ref in sorted(refs):
        filepath = resolve_footprint_path(ref)
        if filepath is None:
            skipped += 1
            continue

        try:
            fp_def = KiCadFootprintParser.parse_file(str(filepath))
            # Use the full reference as the key
            cache_entries[ref] = fp_def
            # Also cache by just the footprint name (without library prefix)
            if ':' in ref:
                _, fp_name = ref.split(':', 1)
                if fp_name not in cache_entries:
                    cache_entries[fp_name] = fp_def
            parsed += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"   FAILED: {ref} — {e}")

    elapsed = time.time() - start
    print(f"   Parsed: {parsed}, Skipped (no file): {skipped}, Failed: {failed}")
    print(f"   Time: {elapsed:.1f}s ({parsed / max(elapsed, 0.001):.0f} fps)")
    print(f"   Cache entries: {len(cache_entries)}")

    # Save cache
    print(f"\n4. Saving to {FOOTPRINT_CACHE_FILE}...")
    cache_data = {
        'version': '1.0',
        'entries': {}
    }

    for key, fp_def in cache_entries.items():
        cache_data['entries'][key] = {
            'name': fp_def.name,
            'body_width': fp_def.body_width,
            'body_height': fp_def.body_height,
            'is_smd': fp_def.is_smd,
            'pad_positions': [
                [p[0], p[1], p[2], p[3], p[4]]
                for p in fp_def.pad_positions
            ],
        }

    FOOTPRINT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FOOTPRINT_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f)  # No indent to keep file small

    file_size = FOOTPRINT_CACHE_FILE.stat().st_size
    print(f"   Saved: {file_size / 1024 / 1024:.1f} MB ({len(cache_data['entries'])} entries)")

    print(f"\n{'=' * 60}")
    print("Cache built successfully!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
