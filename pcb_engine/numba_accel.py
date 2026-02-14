"""
Numba JIT-accelerated placement math.

Standalone module — ALL functions operate on flat NumPy arrays, not engine objects.
Import this conditionally: if Numba is missing, everything gracefully degrades.

Usage:
    from .numba_accel import accel_available, JITCostEvaluator
    if accel_available:
        evaluator = JITCostEvaluator.from_engine(engine)
        cost = evaluator.total_cost(positions)
"""

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

    # Stub decorators so the module can be imported even without numba
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper

    def prange(*args):
        return range(*args)


accel_available = _NUMBA_OK


# ──────────────────────────────────────────────────────────────
#  JIT kernels — pure numeric, no Python objects
# ──────────────────────────────────────────────────────────────

@njit(cache=True)
def _hpwl(positions, net_pin_comp, net_pin_ox, net_pin_oy,
          net_offsets, net_weights):
    """
    HPWL wirelength for all nets.

    positions : (N, 2) float64 — component centers [x, y]
    net_pin_comp : (total_pins,) int32 — component index for each pin
    net_pin_ox   : (total_pins,) float64 — pin offset x
    net_pin_oy   : (total_pins,) float64 — pin offset y
    net_offsets  : (M+1,) int32 — net_pin_comp[net_offsets[i]:net_offsets[i+1]] = pins of net i
    net_weights  : (M,) float64 — weight per net
    """
    total = 0.0
    n_nets = len(net_weights)
    for i in range(n_nets):
        start = net_offsets[i]
        end = net_offsets[i + 1]
        if end - start < 2:
            continue
        min_x = 1e30
        max_x = -1e30
        min_y = 1e30
        max_y = -1e30
        for j in range(start, end):
            ci = net_pin_comp[j]
            px = positions[ci, 0] + net_pin_ox[j]
            py = positions[ci, 1] + net_pin_oy[j]
            if px < min_x:
                min_x = px
            if px > max_x:
                max_x = px
            if py < min_y:
                min_y = py
            if py > max_y:
                max_y = py
        total += ((max_x - min_x) + (max_y - min_y)) * net_weights[i]
    return total


@njit(cache=True)
def _overlap_area(positions, sizes, min_spacing, n):
    """
    Total overlap area (courtyard overlap) between all component pairs.

    positions : (N, 2)   — centers
    sizes     : (N, 2)   — (width, height) courtyard sizes
    min_spacing : float   — minimum spacing between components
    n : int               — number of components
    """
    total = 0.0
    hs = min_spacing / 2.0
    for i in range(n):
        hw_a = sizes[i, 0] / 2.0 + hs
        hh_a = sizes[i, 1] / 2.0 + hs
        for j in range(i + 1, n):
            hw_b = sizes[j, 0] / 2.0 + hs
            hh_b = sizes[j, 1] / 2.0 + hs
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])
            ox = (hw_a + hw_b) - dx
            oy = (hh_a + hh_b) - dy
            if ox > 0.0 and oy > 0.0:
                total += ox * oy
    return total


@njit(cache=True)
def _proximity_cost(positions, prox_pairs, prox_targets, prox_priorities):
    """
    Continuous proximity penalty for functional pairs.

    prox_pairs     : (P, 2) int32 — component index pairs
    prox_targets   : (P,) float64 — target distance per pair
    prox_priorities: (P,) float64 — priority weight per pair
    """
    total = 0.0
    n = len(prox_targets)
    for i in range(n):
        a = prox_pairs[i, 0]
        b = prox_pairs[i, 1]
        dx = positions[a, 0] - positions[b, 0]
        dy = positions[a, 1] - positions[b, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        tgt = prox_targets[i]
        pri = prox_priorities[i]
        if dist <= tgt:
            total += (dist / tgt) * pri
        else:
            excess = dist - tgt
            total += pri + pri * (excess / tgt) ** 2
    return total


@njit(cache=True)
def _edge_cost(positions, sizes, edge_indices,
               board_ox, board_oy, board_w, board_h):
    """
    Quadratic penalty for edge components far from board edges.

    edge_indices : (E,) int32 — indices of edge components
    """
    total = 0.0
    for k in range(len(edge_indices)):
        i = edge_indices[k]
        hw = sizes[i, 0] / 2.0
        hh = sizes[i, 1] / 2.0
        d_left = positions[i, 0] - hw - board_ox
        d_right = (board_ox + board_w) - (positions[i, 0] + hw)
        d_top = positions[i, 1] - hh - board_oy
        d_bot = (board_oy + board_h) - (positions[i, 1] + hh)
        nearest = min(d_left, d_right, d_top, d_bot)
        if nearest < 0.0:
            nearest = 0.0
        total += nearest * nearest * 5.0
    return total


@njit(cache=True)
def _keep_apart_cost(positions, apart_pairs, apart_min_dists):
    """
    Penalty when keep-apart pairs are too close.

    apart_pairs : (K, 2) int32
    apart_min_dists : (K,) float64
    """
    total = 0.0
    for i in range(len(apart_min_dists)):
        a = apart_pairs[i, 0]
        b = apart_pairs[i, 1]
        dx = positions[a, 0] - positions[b, 0]
        dy = positions[a, 1] - positions[b, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < apart_min_dists[i]:
            total += (apart_min_dists[i] - dist) * 2.0
    return total


@njit(cache=True)
def _oob_penalty(positions, sizes, board_ox, board_oy, board_w, board_h, n):
    """Out-of-bounds penalty."""
    total = 0.0
    for i in range(n):
        hw = sizes[i, 0] / 2.0
        hh = sizes[i, 1] / 2.0
        left = positions[i, 0] - hw
        right = positions[i, 0] + hw
        top = positions[i, 1] - hh
        bot = positions[i, 1] + hh
        if left < board_ox:
            total += (board_ox - left) ** 2
        if right > board_ox + board_w:
            total += (right - board_ox - board_w) ** 2
        if top < board_oy:
            total += (board_oy - top) ** 2
        if bot > board_oy + board_h:
            total += (bot - board_oy - board_h) ** 2
    return total


@njit(cache=True)
def _total_cost(positions, sizes, min_spacing, n,
                net_pin_comp, net_pin_ox, net_pin_oy,
                net_offsets, net_weights,
                prox_pairs, prox_targets, prox_priorities,
                edge_indices, board_ox, board_oy, board_w, board_h,
                apart_pairs, apart_min_dists):
    """
    Full cost function — single JIT call instead of 6+ Python method calls.
    """
    wl = _hpwl(positions, net_pin_comp, net_pin_ox, net_pin_oy,
               net_offsets, net_weights)
    ov = _overlap_area(positions, sizes, min_spacing, n)
    oob = _oob_penalty(positions, sizes, board_ox, board_oy, board_w, board_h, n)
    prox = _proximity_cost(positions, prox_pairs, prox_targets, prox_priorities)
    edge = _edge_cost(positions, sizes, edge_indices,
                      board_ox, board_oy, board_w, board_h)
    apart = _keep_apart_cost(positions, apart_pairs, apart_min_dists)

    return (wl +
            ov * 1000.0 +
            oob * 500.0 +
            prox * 150.0 +
            edge * 100.0 +
            apart * 200.0)


# ──────────────────────────────────────────────────────────────
#  FD force kernel
# ──────────────────────────────────────────────────────────────

@njit(cache=True)
def _fd_forces(positions, sizes, fixed, n, k, min_spacing,
               attraction_k,
               net_pin_comp, net_offsets, net_weights,
               prox_pairs, prox_targets, prox_priorities):
    """
    Compute all FD forces in one pass. Returns (N, 2) force array.
    """
    forces = np.zeros((n, 2), dtype=np.float64)

    # Repulsive forces (all pairs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]

            if abs(dx) < 0.01 and abs(dy) < 0.01:
                dx = 0.3
                dy = 0.3

            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                dist = 0.01

            min_sep_x = (sizes[i, 0] + sizes[j, 0]) / 2.0 + min_spacing
            min_sep_y = (sizes[i, 1] + sizes[j, 1]) / 2.0 + min_spacing

            overlap_x = min_sep_x - abs(dx)
            overlap_y = min_sep_y - abs(dy)

            if overlap_x > 0.0 and overlap_y > 0.0:
                force_scale = 50.0
                if overlap_x < overlap_y:
                    sign_x = 1.0 if dx >= 0.0 else -1.0
                    sign_y = 1.0 if dy >= 0.0 else -1.0
                    fx = force_scale * overlap_x * sign_x
                    fy = force_scale * overlap_y * 0.3 * sign_y
                else:
                    sign_x = 1.0 if dx >= 0.0 else -1.0
                    sign_y = 1.0 if dy >= 0.0 else -1.0
                    fx = force_scale * overlap_x * 0.3 * sign_x
                    fy = force_scale * overlap_y * sign_y
            else:
                min_dist = max(min_sep_x, min_sep_y)
                effective_k = k * (1.0 + min_dist / k)
                force = (effective_k ** 2) / dist
                fx = force * dx / dist
                fy = force * dy / dist

            if not fixed[i]:
                forces[i, 0] += fx
                forces[i, 1] += fy
            if not fixed[j]:
                forces[j, 0] -= fx
                forces[j, 1] -= fy

    # Attractive forces (net connections)
    n_nets = len(net_weights)
    for ni in range(n_nets):
        start = net_offsets[ni]
        end = net_offsets[ni + 1]
        w = net_weights[ni]
        # All pairs within this net
        for pi in range(start, end):
            ci = net_pin_comp[pi]
            for pj in range(pi + 1, end):
                cj = net_pin_comp[pj]
                if ci == cj:
                    continue
                dx = positions[cj, 0] - positions[ci, 0]
                dy = positions[cj, 1] - positions[ci, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 0.01:
                    continue
                force = (dist ** 2) / k * w * attraction_k
                fx = force * dx / dist
                fy = force * dy / dist
                if not fixed[ci]:
                    forces[ci, 0] += fx
                    forces[ci, 1] += fy
                if not fixed[cj]:
                    forces[cj, 0] -= fx
                    forces[cj, 1] -= fy

    # Extra attraction for proximity groups
    n_prox = len(prox_targets)
    for i in range(n_prox):
        a = prox_pairs[i, 0]
        b = prox_pairs[i, 1]
        pri = prox_priorities[i]
        dx = positions[b, 0] - positions[a, 0]
        dy = positions[b, 1] - positions[a, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            continue
        force = (dist ** 2) / k * pri * 2.0 * attraction_k
        fx = force * dx / dist
        fy = force * dy / dist
        if not fixed[a]:
            forces[a, 0] += fx
            forces[a, 1] += fy
        if not fixed[b]:
            forces[b, 0] -= fx
            forces[b, 1] -= fy

    return forces


# ──────────────────────────────────────────────────────────────
#  Array builder — bridges engine objects to flat arrays
# ──────────────────────────────────────────────────────────────

class JITCostEvaluator:
    """
    Snapshot of engine state as flat NumPy arrays for JIT evaluation.

    Create once before SA/FD starts, then call total_cost() or fd_forces()
    with a mutable positions array.
    """

    def __init__(self):
        self.n = 0
        self.ref_to_idx = {}
        self.positions = None  # (N, 2)
        self.sizes = None      # (N, 2)
        self.fixed = None      # (N,) bool

        # Net data
        self.net_pin_comp = None
        self.net_pin_ox = None
        self.net_pin_oy = None
        self.net_offsets = None
        self.net_weights = None

        # Proximity groups
        self.prox_pairs = None
        self.prox_targets = None
        self.prox_priorities = None

        # Edge components
        self.edge_indices = None

        # Keep-apart pairs
        self.apart_pairs = None
        self.apart_min_dists = None

        # Board params
        self.board_ox = 0.0
        self.board_oy = 0.0
        self.board_w = 80.0
        self.board_h = 60.0
        self.min_spacing = 0.25
        self.attraction_k = 1.0
        self.k = 5.0

    @classmethod
    def from_engine(cls, engine) -> 'JITCostEvaluator':
        """Build array snapshot from a PlacementEngine instance."""
        ev = cls()

        # Index mapping
        refs = sorted(engine.components.keys())
        ev.n = len(refs)
        ev.ref_to_idx = {r: i for i, r in enumerate(refs)}

        # Positions & sizes
        ev.positions = np.zeros((ev.n, 2), dtype=np.float64)
        ev.sizes = np.zeros((ev.n, 2), dtype=np.float64)
        ev.fixed = np.zeros(ev.n, dtype=np.bool_)

        for r, i in ev.ref_to_idx.items():
            c = engine.components[r]
            ev.positions[i] = [c.x, c.y]
            ev.sizes[i] = [c.width, c.height]
            ev.fixed[i] = c.fixed

        # Board params
        ev.board_ox = engine.config.origin_x
        ev.board_oy = engine.config.origin_y
        ev.board_w = engine.config.board_width
        ev.board_h = engine.config.board_height
        ev.min_spacing = engine.config.min_spacing
        ev.attraction_k = engine.config.fd_attraction_k
        ev.k = engine._optimal_dist if engine._optimal_dist > 0 else 5.0

        # Nets -> flat arrays
        pin_comp_list = []
        pin_ox_list = []
        pin_oy_list = []
        offsets = [0]
        weights = []

        for net_name, net_info in engine.nets.items():
            pins = net_info.get('pins', [])
            w = net_info.get('weight', 1.0)
            count = 0
            for pin_ref in pins:
                comp_ref, pin_num = engine._parse_pin_ref(pin_ref)
                if comp_ref not in ev.ref_to_idx:
                    continue
                idx = ev.ref_to_idx[comp_ref]
                offset = engine.pin_offsets.get(comp_ref, {}).get(pin_num, (0, 0))
                pin_comp_list.append(idx)
                pin_ox_list.append(offset[0])
                pin_oy_list.append(offset[1])
                count += 1
            offsets.append(offsets[-1] + count)
            weights.append(w)

        ev.net_pin_comp = np.array(pin_comp_list, dtype=np.int32)
        ev.net_pin_ox = np.array(pin_ox_list, dtype=np.float64)
        ev.net_pin_oy = np.array(pin_oy_list, dtype=np.float64)
        ev.net_offsets = np.array(offsets, dtype=np.int32)
        ev.net_weights = np.array(weights, dtype=np.float64)

        # Proximity groups -> pairs
        prox_a, prox_b, prox_t, prox_p = [], [], [], []
        hints = getattr(engine, '_placement_hints', None)
        if hints:
            for group in hints.get('proximity_groups', []):
                comps = group.get('components', [])
                tgt = group.get('max_distance', 10.0)
                pri = group.get('priority', 1.0)
                for i, ra in enumerate(comps):
                    if ra not in ev.ref_to_idx:
                        continue
                    for rb in comps[i + 1:]:
                        if rb not in ev.ref_to_idx:
                            continue
                        prox_a.append(ev.ref_to_idx[ra])
                        prox_b.append(ev.ref_to_idx[rb])
                        prox_t.append(tgt)
                        prox_p.append(pri)

        if prox_a:
            ev.prox_pairs = np.column_stack([
                np.array(prox_a, dtype=np.int32),
                np.array(prox_b, dtype=np.int32)])
            ev.prox_targets = np.array(prox_t, dtype=np.float64)
            ev.prox_priorities = np.array(prox_p, dtype=np.float64)
        else:
            ev.prox_pairs = np.zeros((0, 2), dtype=np.int32)
            ev.prox_targets = np.zeros(0, dtype=np.float64)
            ev.prox_priorities = np.zeros(0, dtype=np.float64)

        # Edge components
        edge_list = []
        if hints:
            for r in hints.get('edge_components', []):
                if r in ev.ref_to_idx:
                    edge_list.append(ev.ref_to_idx[r])
        ev.edge_indices = np.array(edge_list, dtype=np.int32)

        # Keep-apart pairs
        apart_a, apart_b, apart_d = [], [], []
        if hints:
            for pair in hints.get('keep_apart', []):
                a = pair.get('a')
                b = pair.get('b')
                if a in ev.ref_to_idx and b in ev.ref_to_idx:
                    apart_a.append(ev.ref_to_idx[a])
                    apart_b.append(ev.ref_to_idx[b])
                    apart_d.append(pair.get('min_distance', 10.0))
        if apart_a:
            ev.apart_pairs = np.column_stack([
                np.array(apart_a, dtype=np.int32),
                np.array(apart_b, dtype=np.int32)])
            ev.apart_min_dists = np.array(apart_d, dtype=np.float64)
        else:
            ev.apart_pairs = np.zeros((0, 2), dtype=np.int32)
            ev.apart_min_dists = np.zeros(0, dtype=np.float64)

        return ev

    def sync_from_engine(self, engine):
        """Update positions+sizes array from engine components."""
        for r, i in self.ref_to_idx.items():
            c = engine.components[r]
            self.positions[i, 0] = c.x
            self.positions[i, 1] = c.y
            self.sizes[i, 0] = c.width
            self.sizes[i, 1] = c.height

    def sync_positions_from_engine(self, engine):
        """Update only positions (not sizes) — faster for SA moves that don't rotate."""
        for r, i in self.ref_to_idx.items():
            c = engine.components[r]
            self.positions[i, 0] = c.x
            self.positions[i, 1] = c.y

    def sync_to_engine(self, engine):
        """Write positions back to engine components."""
        for r, i in self.ref_to_idx.items():
            c = engine.components[r]
            c.x = self.positions[i, 0]
            c.y = self.positions[i, 1]

    def update_one(self, ref, x, y, w=None, h=None):
        """Update a single component's position in the array (O(1))."""
        i = self.ref_to_idx.get(ref)
        if i is not None:
            self.positions[i, 0] = x
            self.positions[i, 1] = y
            if w is not None:
                self.sizes[i, 0] = w
                self.sizes[i, 1] = h

    def save_positions(self):
        """Snapshot positions for fast rollback. Returns a copy."""
        return self.positions.copy(), self.sizes.copy()

    def restore_positions(self, snapshot):
        """Restore positions from a snapshot (fast array copy)."""
        self.positions[:] = snapshot[0]
        self.sizes[:] = snapshot[1]

    def total_cost(self) -> float:
        """Compute full placement cost using JIT kernels."""
        return _total_cost(
            self.positions, self.sizes, self.min_spacing, self.n,
            self.net_pin_comp, self.net_pin_ox, self.net_pin_oy,
            self.net_offsets, self.net_weights,
            self.prox_pairs, self.prox_targets, self.prox_priorities,
            self.edge_indices,
            self.board_ox, self.board_oy, self.board_w, self.board_h,
            self.apart_pairs, self.apart_min_dists)

    def fd_forces(self) -> np.ndarray:
        """Compute FD forces using JIT kernel. Returns (N, 2) array."""
        return _fd_forces(
            self.positions, self.sizes, self.fixed, self.n,
            self.k, self.min_spacing, self.attraction_k,
            self.net_pin_comp, self.net_offsets, self.net_weights,
            self.prox_pairs, self.prox_targets, self.prox_priorities)
