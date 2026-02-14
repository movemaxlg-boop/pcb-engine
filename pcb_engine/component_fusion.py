"""
Component Fusion System — FUSE → PLACE → UNFUSE
=================================================

Treats functional passives (decoupling caps, pull-ups, etc.) as IC appendages
rather than independent components. Before placement, passives are FUSED into
their owner IC's courtyard. After placement, they are UNFUSED back to individual
components positioned on the IC perimeter.

This guarantees passives land within 1-2mm of their owner IC — structurally
impossible to achieve with penalty-based approaches alone.

Usage:
    fusion = ComponentFusion(priority_threshold=1.5, max_per_owner=6)
    fusion.fuse(engine)      # Before placement
    # ... run placement algorithm ...
    fusion.unfuse(engine)    # After placement
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import math


@dataclass
class FusedComponent:
    """Records one fusion operation for later undo."""
    owner_ref: str                          # IC reference (e.g., "U1")
    fused_refs: List[str]                   # Absorbed passive refs (e.g., ["C1", "C2"])
    original_owner_courtyard: Tuple[float, float]  # IC's pre-fusion (width, height)
    fused_courtyard: Tuple[float, float]    # Expanded (width, height)

    # Backup of each passive's engine data for restore
    passive_data: Dict[str, Dict] = field(default_factory=dict)
    # Maps: ref → {width, height, pin_offsets, pin_sizes, pin_nets,
    #              x, y, rotation, fixed, pin_count,
    #              is_decoupling_cap, is_power_component, vx, vy, fx, fy}

    # Synthetic pin mapping: fused_pin_id → (original_ref, original_pin_num)
    pin_remap: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # All synthetic pin offsets relative to fused IC center
    fused_pin_offsets: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fused_pin_sizes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fused_pin_nets: Dict[str, str] = field(default_factory=dict)

    # Perimeter positions: ref → (dx_from_ic_center, dy_from_ic_center, rotation)
    perimeter_positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


class ComponentFusion:
    """Fuses functional passives into their owner IC before placement."""

    def __init__(self, priority_threshold: float = 1.5, max_per_owner: int = 6):
        self.priority_threshold = priority_threshold
        self.max_per_owner = max_per_owner
        self.fused_components: List[FusedComponent] = []
        self._original_nets: Dict[str, Dict] = {}   # Backup of net pin lists
        self._original_adjacency: Dict = {}          # Backup of adjacency graph

    def fuse(self, engine) -> int:
        """
        Fuse functional passives into their owner ICs.

        Modifies engine state IN PLACE:
        - Removes passives from engine.components
        - Expands IC courtyards
        - Remaps passive pins onto IC as synthetic pins
        - Updates net pin references

        Returns number of passives fused.
        """
        roles = getattr(engine, '_functional_roles', {})
        if not roles:
            return 0

        # Group passives by owner IC, filter by priority threshold
        owner_groups: Dict[str, List[Tuple[str, Dict]]] = {}
        for ref, role in roles.items():
            if role.get('priority', 0) < self.priority_threshold:
                continue
            owner = role.get('owner', '')
            if not owner or owner not in engine.components:
                continue
            if ref not in engine.components:
                continue
            owner_groups.setdefault(owner, []).append((ref, role))

        if not owner_groups:
            return 0

        # Backup nets and adjacency for restore
        self._original_nets = {
            net_name: {'pins': list(info['pins']), 'weight': info['weight']}
            for net_name, info in engine.nets.items()
        }
        self._original_adjacency = {
            ref: dict(neighbors) for ref, neighbors in engine.adjacency.items()
        } if hasattr(engine, 'adjacency') else {}

        total_fused = 0

        # Prevent cascading fusion: if a component is BOTH a passive (has owner)
        # AND an owner (has its own passives), do NOT fuse it into its owner.
        # Example: U6 (ESD) is both:
        #   - a passive of J1 (ESD_PROTECTION, fuses into J1)
        #   - an owner of C9 (DECOUPLING, C9 fuses into U6)
        # If we fuse U6 into J1, then U6 is gone, and C9 can't fuse into U6.
        # Solution: skip fusing components that are themselves owners.
        owner_set = set(owner_groups.keys())
        for owner_ref in list(owner_groups.keys()):
            passive_list = owner_groups[owner_ref]
            # Remove passives that are themselves owners (prevent cascading)
            filtered = [(ref, role) for ref, role in passive_list
                        if ref not in owner_set]
            if filtered != passive_list:
                skipped = [ref for ref, _ in passive_list if ref in owner_set]
                owner_groups[owner_ref] = filtered

        for owner_ref, passive_list in sorted(owner_groups.items()):
            # Skip if owner was itself fused into another IC in a prior iteration
            if owner_ref not in engine.components:
                continue

            # Sort by priority desc, take top N
            passive_list.sort(key=lambda x: x[1].get('priority', 0), reverse=True)
            # Filter out passives already fused in a prior iteration
            to_fuse = [(ref, role) for ref, role in passive_list
                       if ref in engine.components][:self.max_per_owner]

            if not to_fuse:
                continue

            owner_comp = engine.components[owner_ref]
            original_w, original_h = owner_comp.width, owner_comp.height

            # Collect passive data
            passive_data = {}
            for ref, role in to_fuse:
                comp = engine.components[ref]
                passive_data[ref] = {
                    'width': comp.width,
                    'height': comp.height,
                    'x': comp.x, 'y': comp.y,
                    'rotation': comp.rotation,
                    'fixed': comp.fixed,
                    'pin_count': comp.pin_count,
                    'is_decoupling_cap': comp.is_decoupling_cap,
                    'is_power_component': comp.is_power_component,
                    'vx': getattr(comp, 'vx', 0), 'vy': getattr(comp, 'vy', 0),
                    'fx': getattr(comp, 'fx', 0), 'fy': getattr(comp, 'fy', 0),
                    'pin_offsets': dict(engine.pin_offsets.get(ref, {})),
                    'pin_sizes': dict(engine.pin_sizes.get(ref, {})),
                    'pin_nets': dict(engine.pin_nets.get(ref, {})),
                    'role': role,
                }

            # Calculate fused courtyard
            fused_w, fused_h = self._calculate_fused_courtyard(
                original_w, original_h, passive_data)

            # Assign passives to IC sides and build pin remap
            pin_remap, fused_pin_offsets, fused_pin_sizes, fused_pin_nets, perimeter_pos = \
                self._build_pin_remap(
                    owner_ref, engine, passive_data,
                    original_w, original_h)

            # Create fusion record
            fused = FusedComponent(
                owner_ref=owner_ref,
                fused_refs=[ref for ref, _ in to_fuse],
                original_owner_courtyard=(original_w, original_h),
                fused_courtyard=(fused_w, fused_h),
                passive_data=passive_data,
                pin_remap=pin_remap,
                fused_pin_offsets=fused_pin_offsets,
                fused_pin_sizes=fused_pin_sizes,
                fused_pin_nets=fused_pin_nets,
                perimeter_positions=perimeter_pos,
            )
            self.fused_components.append(fused)

            # --- Apply fusion to engine state ---

            # 1. Expand IC courtyard
            owner_comp.width = fused_w
            owner_comp.height = fused_h

            # 2. Add synthetic pins to IC
            for pin_id, offset in fused_pin_offsets.items():
                engine.pin_offsets.setdefault(owner_ref, {})[pin_id] = offset
            for pin_id, size in fused_pin_sizes.items():
                engine.pin_sizes.setdefault(owner_ref, {})[pin_id] = size
            for pin_id, net in fused_pin_nets.items():
                engine.pin_nets.setdefault(owner_ref, {})[pin_id] = net

            # 3. Update net pin references: "C1.1" → "U1._f1"
            for pin_id, (orig_ref, orig_pin) in pin_remap.items():
                old_pin_ref = f"{orig_ref}.{orig_pin}"
                new_pin_ref = f"{owner_ref}.{pin_id}"
                for net_name, net_info in engine.nets.items():
                    pins = net_info.get('pins', [])
                    for i, p in enumerate(pins):
                        if isinstance(p, str) and p == old_pin_ref:
                            pins[i] = new_pin_ref
                        elif isinstance(p, (list, tuple)) and len(p) >= 2:
                            if str(p[0]) == orig_ref and str(p[1]) == orig_pin:
                                pins[i] = new_pin_ref

            # 4. Merge adjacency: passive's neighbors → IC's neighbors
            if hasattr(engine, 'adjacency'):
                for ref, _ in to_fuse:
                    if ref in engine.adjacency:
                        for neighbor, weight in engine.adjacency[ref].items():
                            if neighbor == owner_ref:
                                continue  # Skip self-loop
                            # Add to IC's adjacency
                            engine.adjacency.setdefault(owner_ref, {})
                            existing = engine.adjacency[owner_ref].get(neighbor, 0)
                            engine.adjacency[owner_ref][neighbor] = existing + weight
                            # Update reverse direction
                            if neighbor in engine.adjacency:
                                engine.adjacency[neighbor].pop(ref, None)
                                existing_rev = engine.adjacency[neighbor].get(owner_ref, 0)
                                engine.adjacency[neighbor][owner_ref] = existing_rev + weight
                        # Remove passive from adjacency
                        del engine.adjacency[ref]
                    # Also remove passive from other entries
                    for _, neighbors in engine.adjacency.items():
                        neighbors.pop(ref, None)

            # 5. Remove passives from engine
            for ref, _ in to_fuse:
                # Remove from occupancy grid if it exists
                comp = engine.components.get(ref)
                if comp and hasattr(engine, '_occ') and engine._occ:
                    engine._occ.remove(ref, comp.x, comp.y, comp.width, comp.height)
                # Remove from components
                engine.components.pop(ref, None)
                engine.pin_offsets.pop(ref, None)
                engine.pin_sizes.pop(ref, None)
                engine.pin_nets.pop(ref, None)
                # Remove from _owner_of
                if hasattr(engine, '_owner_of'):
                    engine._owner_of.pop(ref, None)

            total_fused += len(to_fuse)

        if total_fused > 0:
            # Rebuild occupancy grid with fused state
            if hasattr(engine, '_occ') and engine._occ:
                engine._occ.clear()
                for ref, comp in engine.components.items():
                    engine._occ.place(ref, comp.x, comp.y, comp.width, comp.height)

            print(f"  [FUSION] Fused {total_fused} passives into "
                  f"{len(self.fused_components)} ICs "
                  f"(threshold={self.priority_threshold}, max={self.max_per_owner})")

        return total_fused

    def unfuse(self, engine) -> int:
        """
        Unfuse all components — restore passives as independent components
        positioned on their owner IC's perimeter.

        Returns number of passives restored.
        """
        if not self.fused_components:
            return 0

        total_restored = 0

        for fused in self.fused_components:
            owner_ref = fused.owner_ref
            if owner_ref not in engine.components:
                continue

            owner_comp = engine.components[owner_ref]
            owner_x, owner_y = owner_comp.x, owner_comp.y
            owner_rotation = owner_comp.rotation

            # 1. Restore IC's original courtyard and update occupancy grid
            orig_w, orig_h = fused.original_owner_courtyard
            fused_w, fused_h = fused.fused_courtyard

            # CRITICAL: Update occupancy grid FIRST so perimeter cells become free
            if hasattr(engine, '_occ') and engine._occ:
                engine._occ.remove(owner_ref, owner_x, owner_y, fused_w, fused_h)
                # Re-place with original (smaller) courtyard — frees the perimeter ring
                engine._occ.place(owner_ref, owner_x, owner_y, orig_w, orig_h)

            owner_comp.width = orig_w
            owner_comp.height = orig_h

            # 2. Remove synthetic pins from IC
            for pin_id in fused.pin_remap:
                engine.pin_offsets.get(owner_ref, {}).pop(pin_id, None)
                engine.pin_sizes.get(owner_ref, {}).pop(pin_id, None)
                engine.pin_nets.get(owner_ref, {}).pop(pin_id, None)

            # 3. Restore each passive as independent component
            # Sort by priority DESC so highest-priority caps get the best perimeter spots
            sorted_refs = sorted(
                fused.fused_refs,
                key=lambda r: fused.passive_data.get(r, {}).get('role', {}).get('priority', 0),
                reverse=True)
            for ref in sorted_refs:
                pdata = fused.passive_data.get(ref)
                if not pdata:
                    continue

                # Compute target position on IC perimeter
                perimeter = fused.perimeter_positions.get(ref, (0, 0, 0))
                dx, dy, passive_rot = perimeter

                # Rotate perimeter offset by IC's actual rotation
                rdx, rdy = self._rotate_offset(dx, dy, owner_rotation)
                target_x = owner_x + rdx
                target_y = owner_y + rdy

                # Clamp to board bounds
                bw = engine.config.board_width
                bh = engine.config.board_height
                ox = engine.config.origin_x
                oy = engine.config.origin_y
                pw, ph = pdata['width'], pdata['height']
                half_w, half_h = pw / 2, ph / 2

                target_x = max(ox + half_w, min(ox + bw - half_w, target_x))
                target_y = max(oy + half_h, min(oy + bh - half_h, target_y))

                # Try to place — if blocked, spiral search
                from .placement_engine import ComponentState
                comp = ComponentState(
                    ref=ref,
                    x=target_x,
                    y=target_y,
                    width=pw,
                    height=ph,
                    rotation=(owner_rotation + passive_rot) % 360,
                    fixed=False,
                    pin_count=pdata.get('pin_count', 2),
                    is_decoupling_cap=pdata.get('is_decoupling_cap', False),
                    is_power_component=pdata.get('is_power_component', False),
                )

                # Place with spiral fallback
                placed = False
                if hasattr(engine, '_occ') and engine._occ:
                    if engine._occ.can_place(target_x, target_y, pw, ph):
                        engine._occ.place(ref, target_x, target_y, pw, ph)
                        placed = True
                    else:
                        # Spiral search around IC — stay close
                        max_r = orig_w + orig_h
                        px, py = self._spiral_search(
                            engine, ref, target_x, target_y, pw, ph,
                            owner_x, owner_y, max_radius=max_r)
                        if px is not None:
                            comp.x = px
                            comp.y = py
                            engine._occ.place(ref, px, py, pw, ph)
                            placed = True
                        else:
                            # Last resort: place at target anyway (may overlap)
                            comp.x = target_x
                            comp.y = target_y
                            placed = True
                else:
                    placed = True

                # Add component back
                engine.components[ref] = comp

                # Restore pin data
                engine.pin_offsets[ref] = pdata['pin_offsets']
                engine.pin_sizes[ref] = pdata['pin_sizes']
                engine.pin_nets[ref] = pdata['pin_nets']

                total_restored += 1

            # 4. Restore net pin references: "U1._f1" → "C1.1"
            for pin_id, (orig_ref, orig_pin) in fused.pin_remap.items():
                new_pin_ref = f"{owner_ref}.{pin_id}"
                old_pin_ref = f"{orig_ref}.{orig_pin}"
                for net_name, net_info in engine.nets.items():
                    pins = net_info.get('pins', [])
                    for i, p in enumerate(pins):
                        if isinstance(p, str) and p == new_pin_ref:
                            pins[i] = old_pin_ref

        # 5. Restore adjacency from original backup
        if self._original_adjacency and hasattr(engine, 'adjacency'):
            engine.adjacency = {
                ref: dict(neighbors)
                for ref, neighbors in self._original_adjacency.items()
            }

        # 6. Rebuild _owner_of from functional roles
        if hasattr(engine, '_owner_of') and hasattr(engine, '_functional_roles'):
            engine._owner_of = {}
            for ref, role in engine._functional_roles.items():
                owner = role.get('owner')
                if owner:
                    engine._owner_of[ref] = owner

        # Step 7 removed — IC occupancy grid update now happens in step 1
        # (before passive placement) to free the perimeter ring.

        if total_restored > 0:
            print(f"  [FUSION] Unfused {total_restored} passives from "
                  f"{len(self.fused_components)} ICs")

        return total_restored

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _calculate_fused_courtyard(
        self, ic_w: float, ic_h: float,
        passive_data: Dict[str, Dict]
    ) -> Tuple[float, float]:
        """
        Calculate expanded courtyard that encompasses IC + one ring of passives.

        The ring depth = max passive courtyard dimension + 0.3mm spacing.
        """
        if not passive_data:
            return (ic_w, ic_h)

        # Ring depth = max passive dimension + spacing
        # Use max(pw, ph) because occupancy grid is NOT rotation-aware
        max_dim = 0.0
        for ref, pdata in passive_data.items():
            pw, ph = pdata['width'], pdata['height']
            max_dim = max(max_dim, pw, ph)

        spacing = 0.3  # mm gap between IC courtyard edge and passive courtyard edge
        ring = max_dim + spacing

        fused_w = ic_w + 2 * ring
        fused_h = ic_h + 2 * ring

        return (round(fused_w, 2), round(fused_h, 2))

    def _build_pin_remap(
        self, owner_ref: str, engine,
        passive_data: Dict[str, Dict],
        ic_w: float, ic_h: float
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Assign passives to IC sides and create synthetic pin mappings.

        Strategy:
        - Find which IC pin shares a net with each passive → place passive on that side
        - Balance sides (max ceil(N/4) per side)
        - Position passive centers on IC perimeter

        Returns:
            (pin_remap, fused_pin_offsets, fused_pin_sizes, fused_pin_nets, perimeter_positions)
        """
        # Determine which side of the IC each passive should be placed on
        # Sides: 0=top, 1=right, 2=bottom, 3=left
        sides: Dict[str, int] = {}
        side_counts = [0, 0, 0, 0]

        ic_pin_offsets = engine.pin_offsets.get(owner_ref, {})
        ic_pin_nets = engine.pin_nets.get(owner_ref, {})

        max_per_side = max(1, math.ceil(len(passive_data) / 4))

        for ref, pdata in sorted(passive_data.items()):
            # Find which IC pin shares a net with this passive
            passive_nets = set(pdata['pin_nets'].values()) - {'', 'GND'}
            # Remove pure power nets too — they're on every side
            passive_signal_nets = {n for n in passive_nets
                                   if not self._is_power_net_name(n)}

            best_side = -1
            if passive_signal_nets and ic_pin_offsets:
                # Find IC pin that shares a signal net with passive
                for ic_pin, ic_net in ic_pin_nets.items():
                    if ic_net in passive_signal_nets and ic_pin in ic_pin_offsets:
                        px, py = ic_pin_offsets[ic_pin]
                        best_side = self._offset_to_side(px, py, ic_w, ic_h)
                        break

            if best_side < 0:
                # No signal net match — use least-populated side
                best_side = side_counts.index(min(side_counts))

            # Balance: if this side is full, use next least-populated
            if side_counts[best_side] >= max_per_side:
                candidates = sorted(range(4), key=lambda s: side_counts[s])
                for s in candidates:
                    if side_counts[s] < max_per_side:
                        best_side = s
                        break

            sides[ref] = best_side
            side_counts[best_side] += 1

        # Position passives on perimeter
        # Group by side
        side_refs: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
        for ref, side in sides.items():
            side_refs[side].append(ref)

        spacing = 0.3  # mm gap between IC courtyard edge and passive courtyard edge
        pin_remap: Dict[str, Tuple[str, str]] = {}
        fused_pin_offsets: Dict[str, Tuple[float, float]] = {}
        fused_pin_sizes: Dict[str, Tuple[float, float]] = {}
        fused_pin_nets: Dict[str, str] = {}
        perimeter_positions: Dict[str, Tuple[float, float, float]] = {}

        fused_pin_counter = 0

        for side, refs in side_refs.items():
            if not refs:
                continue

            n = len(refs)
            for idx, ref in enumerate(sorted(refs)):
                pdata = passive_data[ref]
                pw, ph = pdata['width'], pdata['height']

                # Perpendicular offset from IC center to passive center
                # Use ACTUAL courtyard half-dimension that faces the IC
                # (occupancy grid is NOT rotation-aware, always uses pw/ph as-is)
                if side in (0, 2):  # Top/bottom: perpendicular axis = Y
                    perp = ic_h / 2 + spacing + ph / 2
                    # Spread along width
                    total_span = ic_w * 0.8  # Use 80% of IC width
                    step = total_span / (n + 1)
                    along = -total_span / 2 + step * (idx + 1)
                else:  # Left/right: perpendicular axis = X
                    perp = ic_w / 2 + spacing + pw / 2
                    total_span = ic_h * 0.8
                    step = total_span / (n + 1)
                    along = -total_span / 2 + step * (idx + 1)

                # Calculate (dx, dy) from IC center and rotation
                if side == 0:    # Top
                    dx, dy = along, -perp
                    rot = 0
                elif side == 1:  # Right
                    dx, dy = perp, along
                    rot = 90
                elif side == 2:  # Bottom
                    dx, dy = along, perp
                    rot = 0
                else:            # Left
                    dx, dy = -perp, along
                    rot = 90

                perimeter_positions[ref] = (dx, dy, rot)

                # Remap each passive pin to a synthetic IC pin
                for pin_num, pin_offset in pdata['pin_offsets'].items():
                    fused_pin_counter += 1
                    syn_id = f"_f{fused_pin_counter}"

                    pin_remap[syn_id] = (ref, pin_num)

                    # Synthetic offset = passive center on perimeter + original pin offset
                    # Rotate pin offset if passive is rotated
                    rpx, rpy = self._rotate_offset(pin_offset[0], pin_offset[1], rot)
                    fused_pin_offsets[syn_id] = (dx + rpx, dy + rpy)

                    # Copy pin size and net
                    fused_pin_sizes[syn_id] = pdata['pin_sizes'].get(
                        pin_num, (1.0, 0.6))
                    fused_pin_nets[syn_id] = pdata['pin_nets'].get(pin_num, '')

        return (pin_remap, fused_pin_offsets, fused_pin_sizes,
                fused_pin_nets, perimeter_positions)

    def _offset_to_side(self, px: float, py: float,
                        ic_w: float, ic_h: float) -> int:
        """
        Determine which side of the IC a pin offset (px, py) is closest to.
        Returns: 0=top, 1=right, 2=bottom, 3=left.
        """
        half_w = ic_w / 2
        half_h = ic_h / 2

        # Normalized distances to each edge
        d_top = abs(py - (-half_h))    # py near -half_h → top
        d_bottom = abs(py - half_h)     # py near +half_h → bottom
        d_left = abs(px - (-half_w))    # px near -half_w → left
        d_right = abs(px - half_w)      # px near +half_w → right

        dists = [d_top, d_right, d_bottom, d_left]
        return dists.index(min(dists))

    def _spiral_search(
        self, engine, ref: str,
        start_x: float, start_y: float,
        pw: float, ph: float,
        anchor_x: float, anchor_y: float,
        max_radius: float = 10.0
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Spiral search for valid placement near (start_x, start_y),
        biased toward staying close to (anchor_x, anchor_y).
        """
        step = 0.5  # mm step
        best_pos = None
        best_dist = float('inf')

        for r_idx in range(1, int(max_radius / step) + 1):
            radius = r_idx * step
            # Check 8 directions per radius ring
            for angle_idx in range(8):
                angle = angle_idx * math.pi / 4
                tx = start_x + radius * math.cos(angle)
                ty = start_y + radius * math.sin(angle)

                # Clamp to board
                bw = engine.config.board_width
                bh = engine.config.board_height
                ox = engine.config.origin_x
                oy = engine.config.origin_y
                tx = max(ox + pw / 2, min(ox + bw - pw / 2, tx))
                ty = max(oy + ph / 2, min(oy + bh - ph / 2, ty))

                if engine._occ.can_place(tx, ty, pw, ph):
                    dist = math.sqrt((tx - anchor_x) ** 2 + (ty - anchor_y) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (tx, ty)

            # If found a valid spot in this ring, use it (greedy — closest ring wins)
            if best_pos is not None:
                return best_pos

        return (None, None)

    @staticmethod
    def _rotate_offset(dx: float, dy: float, rotation: float) -> Tuple[float, float]:
        """Rotate (dx, dy) by rotation degrees (0, 90, 180, 270)."""
        r = rotation % 360
        if r == 0:
            return (dx, dy)
        elif r == 90:
            return (-dy, dx)
        elif r == 180:
            return (-dx, -dy)
        elif r == 270:
            return (dy, -dx)
        else:
            # General rotation
            rad = math.radians(r)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            return (dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r)

    @staticmethod
    def _is_power_net_name(net_name: str) -> bool:
        """Check if a net name is a power/ground net."""
        upper = net_name.upper()
        power_names = {'GND', 'VCC', 'VDD', '3V3', '5V', 'VBUS', '12V',
                       'AVCC', 'AGND', 'DVDD', 'AVDD', 'VSS', 'AVSS',
                       'DGND', 'PGND'}
        return upper in power_names or upper.startswith('V')
