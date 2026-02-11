"""
Footprint Resolver — Multi-Tier Resolution with Caching
=========================================================

Resolves footprint names to FootprintDefinition objects using a 6-tier
resolution chain. This replaces the 7-entry FOOTPRINT_LIBRARY as the
primary lookup mechanism while preserving backward compatibility.

Resolution chain:
  Tier 1: In-memory dict cache (instant)
  Tier 2: Pre-parsed JSON cache file (footprint_cache.json)
  Tier 3: Live parse .kicad_mod from KiCad installation
  Tier 4: Hardcoded FOOTPRINT_LIBRARY (7 entries, backward compat)
  Tier 5: Infer from name string (parse dimensions from name)
  Tier 6: Default 2-pad fallback

Usage:
    from pcb_engine.footprint_resolver import FootprintResolver

    resolver = FootprintResolver.get_instance()
    fp_def = resolver.resolve("Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    # Returns FootprintDefinition with 8 real pads
"""

import json
import os
import re
import threading
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .common_types import FootprintDefinition


class FootprintResolver:
    """
    Multi-tier footprint resolver with caching.

    Singleton — one instance shared across all pistons.
    Thread-safe cache writes via Lock.
    """

    _instance: Optional['FootprintResolver'] = None
    _lock = threading.Lock()

    def __init__(
        self,
        kicad_fp_path: Optional[str] = None,
        cache_file: Optional[str] = None,
    ):
        # Tier 1: In-memory cache
        self._memory_cache: Dict[str, FootprintDefinition] = {}

        # Tier 2: Disk cache (loaded once)
        self._disk_cache: Dict[str, dict] = {}
        self._disk_cache_loaded = False

        # KiCad footprint base path
        try:
            from .paths import KICAD_FOOTPRINT_BASE, FOOTPRINT_CACHE_FILE
            self._kicad_fp_path = Path(kicad_fp_path) if kicad_fp_path else KICAD_FOOTPRINT_BASE
            self._cache_file = Path(cache_file) if cache_file else FOOTPRINT_CACHE_FILE
        except ImportError:
            self._kicad_fp_path = Path(kicad_fp_path or r'C:/Program Files/KiCad/9.0/share/kicad/footprints')
            self._cache_file = Path(cache_file or 'footprint_cache.json')

        # Index of .pretty directories (lazy-built)
        self._pretty_dirs: Optional[Dict[str, Path]] = None

        # Tier 4: Hardcoded library reference (imported lazily)
        self._hardcoded_lib: Optional[Dict[str, FootprintDefinition]] = None

        # Provenance tracking: which tier resolved the last call
        self._last_tier: str = ''

    @classmethod
    def get_instance(cls) -> 'FootprintResolver':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def resolve(self, footprint_name: str) -> FootprintDefinition:
        """
        Resolve a footprint name to a FootprintDefinition.

        Tries all 6 tiers in order, caches the result.

        Args:
            footprint_name: Any footprint name format:
                - Simple: "0805", "SOT-23", "SOIC-8"
                - Prefixed: "R_0805", "C_0603"
                - KiCad full: "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"

        Returns:
            FootprintDefinition (never None — Tier 6 always returns a default)
        """
        if not footprint_name:
            return self._default_footprint()

        # Normalize the name
        canonical, kicad_lib, kicad_fp = self._normalize_name(footprint_name)

        # Tier 1: In-memory cache
        if canonical in self._memory_cache:
            self._last_tier = 'memory_cache'
            return self._memory_cache[canonical]

        # Try resolution chain
        # Tier 4 (hardcoded) runs BEFORE Tier 3b (KiCad fuzzy search) because:
        # - Hardcoded values are manually verified against datasheets
        # - KiCad fuzzy search can return wrong matches (e.g., "0603" matching C_0201_0603Metric)
        # - But Tier 3a with EXACT library:footprint reference runs before hardcoded
        result = None
        for tier_name, tier_fn in [
            ('disk_cache',     lambda: self._try_disk_cache(canonical)),
            ('kicad_exact',    lambda: self._try_kicad_exact(kicad_lib, kicad_fp)),
            ('hardcoded',      lambda: self._try_hardcoded(footprint_name)),
            ('kicad_search',   lambda: self._try_kicad_search(kicad_fp or canonical)),
            ('name_inference', lambda: self._try_name_inference(kicad_fp or canonical)),
            ('default',        lambda: self._default_footprint()),
        ]:
            result = tier_fn()
            if result:
                self._last_tier = tier_name
                break

        # Cache the result (thread-safe)
        with self._lock:
            self._memory_cache[canonical] = result

        return result

    # =========================================================================
    # NAME NORMALIZATION
    # =========================================================================

    def _normalize_name(self, name: str) -> Tuple[str, str, str]:
        """
        Normalize footprint name into (canonical_key, kicad_lib, kicad_fp_name).

        Examples:
            "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
                → ("Package_SO:SOIC-8_3.9x4.9mm_P1.27mm", "Package_SO", "SOIC-8_3.9x4.9mm_P1.27mm")
            "R_0805"
                → ("0805", "", "0805")
            "0805"
                → ("0805", "", "0805")
            "SOIC-8"
                → ("SOIC-8", "", "SOIC-8")
        """
        name = name.strip()

        # KiCad full reference: "Library:Footprint"
        if ':' in name:
            lib, fp = name.split(':', 1)
            return (name, lib, fp)

        # Strip common prefixes: R_, C_, L_, D_, U_
        stripped = name
        prefix_match = re.match(r'^[RCLDUQJY]_(.+)$', name)
        if prefix_match:
            stripped = prefix_match.group(1)

        return (stripped, '', stripped)

    # =========================================================================
    # TIER 2: DISK CACHE
    # =========================================================================

    def _try_disk_cache(self, canonical: str) -> Optional[FootprintDefinition]:
        """Try loading from pre-parsed JSON cache file."""
        self._ensure_disk_cache()

        entry = self._disk_cache.get(canonical)
        if entry is None:
            return None

        return self._dict_to_definition(entry)

    def _ensure_disk_cache(self):
        """Load disk cache once."""
        if self._disk_cache_loaded:
            return

        self._disk_cache_loaded = True

        if not self._cache_file.exists():
            return

        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._disk_cache = data.get('entries', {})
        except (json.JSONDecodeError, OSError):
            self._disk_cache = {}

    def _dict_to_definition(self, d: dict) -> FootprintDefinition:
        """Convert a cache dict entry to FootprintDefinition."""
        pad_positions = [
            (str(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
            for p in d.get('pad_positions', [])
        ]
        return FootprintDefinition(
            name=d.get('name', 'cached'),
            body_width=float(d.get('body_width', 2.0)),
            body_height=float(d.get('body_height', 1.0)),
            pad_positions=pad_positions,
            is_smd=d.get('is_smd', True),
        )

    # =========================================================================
    # TIER 3: LIVE KICAD FILE PARSING
    # =========================================================================

    def _try_kicad_exact(
        self, kicad_lib: str, kicad_fp: str
    ) -> Optional[FootprintDefinition]:
        """Tier 3a: Try exact library:footprint path (no fuzzy search)."""
        if not kicad_lib or not kicad_fp:
            return None
        if not self._kicad_fp_path.exists():
            return None

        candidate = self._kicad_fp_path / f"{kicad_lib}.pretty" / f"{kicad_fp}.kicad_mod"
        if not candidate.exists():
            return None

        try:
            from .kicad_footprint_parser import KiCadFootprintParser
            return KiCadFootprintParser.parse_file(str(candidate))
        except (FileNotFoundError, ValueError, OSError):
            return None

    def _try_kicad_search(self, fp_name: str) -> Optional[FootprintDefinition]:
        """Tier 3b: Search .pretty directories for a matching footprint."""
        if not fp_name or not self._kicad_fp_path.exists():
            return None

        filepath = self._search_pretty_dirs(fp_name)
        if filepath is None:
            return None

        try:
            from .kicad_footprint_parser import KiCadFootprintParser
            return KiCadFootprintParser.parse_file(str(filepath))
        except (FileNotFoundError, ValueError, OSError):
            return None

    def _search_pretty_dirs(self, fp_name: str) -> Optional[Path]:
        """Search all .pretty directories for a matching footprint file."""
        self._ensure_pretty_index()

        if self._pretty_dirs is None:
            return None

        # Direct filename match
        target = f"{fp_name}.kicad_mod"
        for lib_name, lib_path in self._pretty_dirs.items():
            candidate = lib_path / target
            if candidate.exists():
                return candidate

        # Fuzzy: search for files containing the name
        fp_lower = fp_name.lower()
        best_match = None
        best_score = 0

        for lib_name, lib_path in self._pretty_dirs.items():
            if not lib_path.exists():
                continue
            try:
                for f in lib_path.iterdir():
                    if not f.suffix == '.kicad_mod':
                        continue
                    fname = f.stem.lower()
                    if fp_lower in fname:
                        # Score: prefer exact start match, shorter names
                        score = 100 if fname.startswith(fp_lower) else 50
                        score -= len(fname) - len(fp_lower)  # penalize extra chars
                        if score > best_score:
                            best_score = score
                            best_match = f
            except OSError:
                continue

        return best_match

    def _ensure_pretty_index(self):
        """Build index of .pretty directories (once)."""
        if self._pretty_dirs is not None:
            return

        self._pretty_dirs = {}
        if not self._kicad_fp_path.exists():
            return

        try:
            for item in self._kicad_fp_path.iterdir():
                if item.is_dir() and item.suffix == '.pretty':
                    self._pretty_dirs[item.stem] = item
        except OSError:
            pass

    # =========================================================================
    # TIER 4: HARDCODED FOOTPRINT_LIBRARY
    # =========================================================================

    def _try_hardcoded(self, original_name: str) -> Optional[FootprintDefinition]:
        """Try the hardcoded FOOTPRINT_LIBRARY (7 entries)."""
        if self._hardcoded_lib is None:
            from .common_types import FOOTPRINT_LIBRARY
            self._hardcoded_lib = FOOTPRINT_LIBRARY

        # Direct match
        if original_name in self._hardcoded_lib:
            return self._hardcoded_lib[original_name]

        # Fuzzy: extract size code from name
        name_lower = original_name.lower()
        for size in ['0402', '0603', '0805', '1206', 'sot-23-5', 'sot-23', 'sot-223']:
            if size in name_lower:
                key = size.upper() if 'sot' in size else size
                if key in self._hardcoded_lib:
                    return self._hardcoded_lib[key]

        return None

    # =========================================================================
    # TIER 5: NAME-BASED INFERENCE
    # =========================================================================

    def _try_name_inference(self, name: str) -> Optional[FootprintDefinition]:
        """
        Infer footprint from name string.

        Many KiCad names embed dimensions:
        "SOIC-8_3.9x4.9mm_P1.27mm" → body 3.9x4.9mm, 8 pins, pitch 1.27mm
        "TSSOP-16_4.4x5mm_P0.65mm" → body 4.4x5mm, 16 pins, pitch 0.65mm
        """
        if not name:
            return None

        # Extract pin count
        pin_match = re.search(r'[-_](\d+)(?:[-_EP]|$)', name)
        pin_count = int(pin_match.group(1)) if pin_match else 0

        # Extract body dimensions: WxHmm
        dim_match = re.search(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)mm', name)
        if not dim_match:
            return None

        body_w = float(dim_match.group(1))
        body_h = float(dim_match.group(2))

        # Extract pitch: P<value>mm
        pitch_match = re.search(r'P(\d+(?:\.\d+)?)mm', name)
        pitch = float(pitch_match.group(1)) if pitch_match else 0.0

        if pin_count < 2 or pitch <= 0:
            return None

        # Determine package type from name
        name_upper = name.upper()
        is_smd = True

        # Generate pad positions algorithmically
        pad_positions = []

        if any(pkg in name_upper for pkg in ['SOIC', 'SSOP', 'TSSOP', 'MSOP', 'SOP']):
            # Dual-row gull-wing: pins on left and right
            pad_positions = self._generate_dual_row_pads(
                pin_count, body_w, body_h, pitch
            )
        elif any(pkg in name_upper for pkg in ['QFP', 'LQFP', 'TQFP', 'PQFP']):
            # Quad-flat: pins on all 4 sides
            pad_positions = self._generate_quad_flat_pads(
                pin_count, body_w, body_h, pitch
            )
        elif any(pkg in name_upper for pkg in ['QFN', 'DFN', 'MLF']):
            # No-lead: pins on all 4 sides, closer to body
            pad_positions = self._generate_quad_flat_pads(
                pin_count, body_w, body_h, pitch, lead_extend=0.4
            )
        elif 'DIP' in name_upper:
            # Through-hole dual inline
            is_smd = False
            pad_positions = self._generate_dip_pads(pin_count, body_w, pitch)
        else:
            return None

        if not pad_positions:
            return None

        return FootprintDefinition(
            name=name,
            body_width=body_w,
            body_height=body_h,
            pad_positions=pad_positions,
            is_smd=is_smd,
        )

    def _generate_dual_row_pads(
        self, pin_count: int, body_w: float, body_h: float, pitch: float
    ) -> List[Tuple[str, float, float, float, float]]:
        """Generate pads for dual-row IC (SOIC, SSOP, etc.)."""
        half_pins = pin_count // 2
        if half_pins < 1:
            return []

        # Pad dimensions (IPC-7351B nominal)
        pad_w = min(1.95, body_w * 0.3)  # lead length ~30% of body width
        pad_h = max(0.25, pitch * 0.55)  # pad height ~55% of pitch

        # Pad center X offset from component center
        x_offset = body_w / 2 + pad_w / 2 - 0.5  # leads extend ~0.5mm past body

        # Generate pin 1..half on left, half+1..count on right
        span = (half_pins - 1) * pitch
        pads = []

        for i in range(half_pins):
            y = -span / 2 + i * pitch
            pads.append((str(i + 1), round(-x_offset, 4), round(y, 4),
                        round(pad_w, 4), round(pad_h, 4)))

        for i in range(half_pins):
            y = span / 2 - i * pitch
            pads.append((str(half_pins + i + 1), round(x_offset, 4), round(y, 4),
                        round(pad_w, 4), round(pad_h, 4)))

        return pads

    def _generate_quad_flat_pads(
        self, pin_count: int, body_w: float, body_h: float, pitch: float,
        lead_extend: float = 1.0
    ) -> List[Tuple[str, float, float, float, float]]:
        """Generate pads for quad-flat IC (QFP, QFN, etc.)."""
        pins_per_side = pin_count // 4
        if pins_per_side < 1:
            return []

        # Pad dimensions
        pad_long = max(0.25, min(lead_extend, body_w * 0.15))
        pad_short = max(0.2, pitch * 0.55)

        pads = []
        pin_num = 1
        span = (pins_per_side - 1) * pitch

        # Left side (top to bottom)
        x = -(body_w / 2 + pad_long / 2 - 0.2)
        for i in range(pins_per_side):
            y = -span / 2 + i * pitch
            pads.append((str(pin_num), round(x, 4), round(y, 4),
                        round(pad_long, 4), round(pad_short, 4)))
            pin_num += 1

        # Bottom side (left to right)
        y = body_h / 2 + pad_long / 2 - 0.2
        for i in range(pins_per_side):
            x = -span / 2 + i * pitch
            pads.append((str(pin_num), round(x, 4), round(y, 4),
                        round(pad_short, 4), round(pad_long, 4)))
            pin_num += 1

        # Right side (bottom to top)
        x = body_w / 2 + pad_long / 2 - 0.2
        for i in range(pins_per_side):
            y = span / 2 - i * pitch
            pads.append((str(pin_num), round(x, 4), round(y, 4),
                        round(pad_long, 4), round(pad_short, 4)))
            pin_num += 1

        # Top side (right to left)
        y = -(body_h / 2 + pad_long / 2 - 0.2)
        for i in range(pins_per_side):
            x = span / 2 - i * pitch
            pads.append((str(pin_num), round(x, 4), round(y, 4),
                        round(pad_short, 4), round(pad_long, 4)))
            pin_num += 1

        return pads

    def _generate_dip_pads(
        self, pin_count: int, row_spacing: float, pitch: float
    ) -> List[Tuple[str, float, float, float, float]]:
        """Generate pads for through-hole DIP."""
        half_pins = pin_count // 2
        if half_pins < 1:
            return []

        pad_size = 1.6  # standard DIP pad
        span = (half_pins - 1) * pitch
        pads = []

        # Left column (pins 1..half, top to bottom)
        for i in range(half_pins):
            y = i * pitch
            pads.append((str(i + 1), 0.0, round(y, 4),
                        round(pad_size, 4), round(pad_size, 4)))

        # Right column (pins half+1..count, bottom to top)
        for i in range(half_pins):
            y = span - i * pitch
            pads.append((str(half_pins + i + 1), round(row_spacing, 4), round(y, 4),
                        round(pad_size, 4), round(pad_size, 4)))

        return pads

    # =========================================================================
    # TIER 6: DEFAULT FALLBACK
    # =========================================================================

    def _default_footprint(self) -> FootprintDefinition:
        """Return default 2-pad SMD footprint (Tier 6 fallback)."""
        return FootprintDefinition(
            name='default',
            body_width=2.0,
            body_height=1.0,
            pad_positions=[
                ('1', -0.95, 0.0, 0.9, 1.25),
                ('2', 0.95, 0.0, 0.9, 1.25),
            ],
            is_smd=True,
        )

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def save_to_disk_cache(self, entries: Dict[str, FootprintDefinition]):
        """
        Save FootprintDefinitions to the disk cache file.
        Used by prebuild_footprint_cache.py script.
        """
        cache_data = {
            'version': '1.0',
            'entries': {}
        }

        for key, fp_def in entries.items():
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

        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)

    def get_cache_stats(self) -> Dict:
        """Return cache statistics."""
        self._ensure_disk_cache()
        return {
            'memory_cache_size': len(self._memory_cache),
            'disk_cache_size': len(self._disk_cache),
            'kicad_available': self._kicad_fp_path.exists(),
            'pretty_dirs': len(self._pretty_dirs) if self._pretty_dirs else 'not loaded',
        }
