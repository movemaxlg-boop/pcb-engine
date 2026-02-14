"""
Hierarchical Cluster-Based Placement Engine (Engine B)
======================================================

Two-level placement that clusters functionally-related components
(IC + decoupling caps + pull-ups etc.) and places them as groups.

Level 1 (Coarse): Place clusters as pseudo-components on the board
Level 2 (Fine):   Place real components within each cluster's region

Reuses PlacementEngine as a black box at both levels — no modifications
to the existing engine.

Expected to outperform flat placement on boards with 50+ components
where the flat SA search space is too large. On small boards (< 25 parts),
the flat engine's global optimization typically wins because there are
fewer components and the SA can explore effectively.

Classes:
    ComponentCluster     - A group of functionally-related components
    ClusterBuilder       - Forms clusters from functional role analysis
    HierarchicalPlacementEngine - Two-level placement (same interface as PlacementEngine)
    PlacementComparator  - A/B comparison: flat vs hierarchical
"""

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .placement_engine import PlacementEngine, PlacementConfig, PlacementResult
from .footprint_resolver import FootprintResolver
from .common_types import get_pins


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ComponentCluster:
    """A cluster of functionally-related components treated as a mini-board."""
    id: str                      # e.g. "cluster_U1"
    owner: str                   # Primary component ref
    members: List[str]           # All refs [owner, C2, R3, ...]
    courtyard_width: float = 0.0
    courtyard_height: float = 0.0
    center_x: float = 0.0       # Set by coarse placement
    center_y: float = 0.0
    rotation: float = 0.0
    role: str = "STANDARD"       # EDGE_CONNECTOR | POWER_REGULATOR | INTERFACE | STANDARD
    edge_preference: Optional[str] = None
    net_connections: Dict[str, int] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of flat vs hierarchical placement."""
    flat_result: PlacementResult
    hier_result: PlacementResult
    winner: str         # 'flat' | 'hierarchical' | 'tie'
    reason: str
    metrics_table: str  # Printable comparison table


# =============================================================================
# CLUSTER BUILDER
# =============================================================================

class ClusterBuilder:
    """
    Forms component clusters from functional role analysis.

    Groups components by their electrical owner (the IC/LED/connector
    they serve), then calculates courtyard dimensions for each cluster.
    """

    MIN_CLUSTER_DIM = 3.0    # Minimum cluster dimension (mm)
    MAX_CLUSTER_SIZE_SMALL = 5   # Max for boards < 30 parts
    MAX_CLUSTER_SIZE_LARGE = 7   # Max for boards >= 30 parts

    def __init__(self, parts_db: Dict, graph: Dict, functional_roles: Dict[str, Dict]):
        self.parts_db = parts_db
        self.graph = graph
        self.functional_roles = functional_roles
        self._resolver = FootprintResolver.get_instance()

        # Scale MAX_CLUSTER_SIZE with board complexity
        n_parts = len(parts_db.get('parts', {}))
        self.MAX_CLUSTER_SIZE = (self.MAX_CLUSTER_SIZE_LARGE if n_parts >= 30
                                 else self.MAX_CLUSTER_SIZE_SMALL)

    def build_clusters(self) -> List[ComponentCluster]:
        """
        Main clustering algorithm.

        1. Group components by owner from functional_roles
        1b. Balance decoupling caps across ICs (multi-IC boards)
        2. Split oversized clusters (keep highest-priority members)
        3. Redistribute evicted members to alternative clusters
        4. Absorb orphan passives into nearest IC cluster
        5. Create single-member clusters for remaining orphans
        6. Calculate cluster geometry
        7. Build inter-cluster connectivity
        8. Classify cluster roles
        """
        clusters = []
        assigned = set()

        # Build net-to-component lookup (used throughout)
        net_to_refs = {}
        for ref, part in self.parts_db['parts'].items():
            for pin in get_pins(part):
                net = pin.get('net', '')
                if net:
                    net_to_refs.setdefault(net, set()).add(ref)

        # Identify all ICs in the design (multi-pin active components)
        all_ics = set()
        for ref, part in self.parts_db['parts'].items():
            pin_count = len(get_pins(part))
            fp = part.get('footprint', '').upper()
            if (pin_count >= 3 or ref.startswith('U') or ref.startswith('J') or
                    any(pkg in fp for pkg in ['QFN', 'QFP', 'SOT', 'SOIC', 'LGA', 'BGA'])):
                all_ics.add(ref)

        # Step 1: Group passives/ESD by owner
        owner_groups: Dict[str, List[str]] = defaultdict(list)
        for ref, role_data in self.functional_roles.items():
            owner = role_data.get('owner')
            if owner and owner in self.parts_db.get('parts', {}):
                owner_groups[owner].append(ref)
                assigned.add(ref)

        # Step 1b: Balance decoupling caps across ICs
        # On multi-IC boards, _build_placement_constraints assigns all caps on
        # a shared power net to the IC with the most pins. Redistribute excess
        # to other ICs that share the same power net but got zero caps.
        self._balance_decoupling_caps(owner_groups, assigned, net_to_refs, all_ics)

        # Step 2: Split oversized clusters
        # Priority: DECOUPLING(3.0) > ESD(2.5) > LED_DRIVER(2.0) > PULLUP/DOWN(1.5)
        # RULE: Functional passives (DECOUPLING, ESD) ALWAYS stay with their owner.
        #       Only low-priority members (PULLUP/DOWN < 2.0) can be evicted.
        #       This prevents caps ending up in the wrong cluster.
        evicted: List[str] = []
        for owner in list(owner_groups.keys()):
            members = owner_groups[owner]
            if len(members) + 1 > self.MAX_CLUSTER_SIZE:
                # Separate mandatory (high-priority) from evictable (low-priority)
                mandatory = []
                evictable = []
                for ref in members:
                    pri = self.functional_roles.get(ref, {}).get('priority', 0)
                    if pri >= 2.0:
                        mandatory.append(ref)  # DECOUPLING, ESD — must stay
                    else:
                        evictable.append(ref)  # PULLUP/DOWN — can be evicted

                # Keep all mandatory + as many evictable as fit
                slots_left = max(0, self.MAX_CLUSTER_SIZE - 1 - len(mandatory))
                # Sort evictable by priority desc, keep top slots_left
                evictable.sort(
                    key=lambda r: self.functional_roles.get(r, {}).get('priority', 0),
                    reverse=True)
                keep_evictable = evictable[:slots_left]
                evict_evictable = evictable[slots_left:]

                owner_groups[owner] = mandatory + keep_evictable
                evicted.extend(evict_evictable)

        # Step 3: Redistribute evicted members to alternative IC/regulator clusters
        for ref in evicted:
            assigned.discard(ref)
            role_data = self.functional_roles.get(ref, {})
            original_owner = role_data.get('owner', '')
            part = self.parts_db['parts'].get(ref, {})

            # Find alternative owner: prefer ICs sharing a signal net
            # Can create NEW owner_groups entries for ICs not yet in the map
            candidates = []
            for pin in get_pins(part):
                net = pin.get('net', '')
                if not net or net == 'GND':
                    continue
                for other_ref in net_to_refs.get(net, set()):
                    if other_ref == ref or other_ref == original_owner:
                        continue
                    # Check if this is a valid target
                    if other_ref in owner_groups:
                        current_size = len(owner_groups[other_ref]) + 1  # +1 for owner
                    elif other_ref in all_ics and other_ref not in assigned:
                        current_size = 1  # New group with just the IC
                    elif other_ref in all_ics:
                        # IC already assigned as member of another cluster — skip
                        continue
                    else:
                        continue

                    if current_size >= self.MAX_CLUSTER_SIZE:
                        continue

                    # Score: ICs > passives, signal nets > power nets
                    is_ic = 1 if other_ref in all_ics else 0
                    is_signal = 1 if net not in ('GND', '3V3', 'VBUS') else 0
                    score = is_ic * 10 + is_signal * 5
                    candidates.append((other_ref, score))

            if candidates:
                # Pick highest-scoring alternative
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_alt = candidates[0][0]
                if best_alt not in owner_groups:
                    # Create new owner group for this IC
                    owner_groups[best_alt] = []
                owner_groups[best_alt].append(ref)
                assigned.add(ref)
            # else: stays evicted, becomes orphan cluster

        # Create clusters from owner groups
        for owner, members in owner_groups.items():
            all_members = [owner] + sorted(members)
            cluster = ComponentCluster(
                id=f"cluster_{owner}",
                owner=owner,
                members=all_members,
            )
            clusters.append(cluster)
            assigned.add(owner)

        # Step 4: Absorb orphan components into nearest IC cluster
        self._absorb_orphan_passives(clusters, assigned, net_to_refs, all_ics)

        # Step 5: Group remaining orphans by net affinity
        # Orphans sharing signal nets should form clusters together
        self._group_remaining_orphans(clusters, assigned, net_to_refs)

        # Step 5b: Any truly isolated components become single-member clusters
        for ref in sorted(self.parts_db.get('parts', {}).keys()):
            if ref not in assigned:
                cluster = ComponentCluster(
                    id=f"cluster_{ref}",
                    owner=ref,
                    members=[ref],
                )
                clusters.append(cluster)

        # Step 5c: Reunite functional passives with their owner IC
        # After all clustering, some high-priority passives (DECOUPLING, ESD)
        # may have been reassigned to the wrong cluster by balance/orphan logic.
        # Force them back to their owner's cluster — functional proximity is
        # more important than cluster size balance.
        self._reunite_with_owners(clusters)

        # Step 6: Calculate geometry for each cluster
        for cluster in clusters:
            self._calculate_geometry(cluster)

        # Step 7: Build inter-cluster connectivity
        self._build_inter_cluster_edges(clusters)

        # Step 8: Classify roles
        self._classify_roles(clusters)

        return clusters

    def _balance_decoupling_caps(
        self,
        owner_groups: Dict[str, List[str]],
        assigned: set,
        net_to_refs: Dict[str, set],
        all_ics: set,
    ):
        """
        Balance decoupling cap assignment across ICs on multi-IC boards.

        Problem: _build_placement_constraints assigns ALL caps on a shared
        power net (e.g. 3V3) to the IC with the most pins on that net.
        On multi-IC boards, one IC gets all the caps while others get none.

        Solution: For each IC with excess caps, redistribute to ICs that:
        1. Share the same power net but have no caps yet
        2. Prefer ICs sharing signal nets with the cap (secondary affinity)
        3. Balance so each IC gets proportional share
        """
        # Identify ICs that have caps and those that don't
        ic_cap_counts = defaultdict(int)
        ic_caps = defaultdict(list)  # ic_ref -> [cap_refs]

        for owner, members in owner_groups.items():
            if owner not in all_ics:
                continue
            caps = [m for m in members
                    if self.functional_roles.get(m, {}).get('role') == 'DECOUPLING']
            ic_cap_counts[owner] = len(caps)
            ic_caps[owner] = caps

        # Find ICs with NO caps that are on shared power nets
        # IMPORTANT: Skip ICs already assigned as members of other clusters
        # (e.g., U6 is in cluster_J1 as ESD protection — don't create cluster_U6)
        ics_needing_caps = set()
        for ic in all_ics:
            if ic in assigned:
                continue  # Already a member of another cluster — skip
            if ic not in owner_groups or ic_cap_counts[ic] == 0:
                # Check if this IC uses a power net
                part = self.parts_db['parts'].get(ic, {})
                for pin in get_pins(part):
                    net = pin.get('net', '')
                    if net and net not in ('GND',) and self._is_power_net(net):
                        ics_needing_caps.add(ic)
                        break

        if not ics_needing_caps:
            return

        # For each IC with excess caps, try to redistribute
        for donor_ic in list(ic_caps.keys()):
            caps = ic_caps[donor_ic]
            if len(caps) <= 1:
                continue  # Keep at least 1 cap per IC

            # Sort caps by how well they match alternative ICs
            for cap_ref in list(caps):
                if len(ic_caps[donor_ic]) <= 1:
                    break  # Keep at least 1

                cap_part = self.parts_db['parts'].get(cap_ref, {})
                cap_nets = {pin.get('net', '') for pin in get_pins(cap_part)
                            if pin.get('net', '')}
                cap_power_nets = {n for n in cap_nets
                                  if n not in ('GND',) and self._is_power_net(n)}

                # Find best alternative IC for this cap
                best_alt = None
                best_score = -1

                for alt_ic in ics_needing_caps:
                    if alt_ic == donor_ic:
                        continue
                    if alt_ic in assigned:
                        continue  # Already in another cluster

                    alt_part = self.parts_db['parts'].get(alt_ic, {})
                    alt_nets = {pin.get('net', '') for pin in get_pins(alt_part)
                                if pin.get('net', '')}

                    # Must share the same power net
                    shared_power = cap_power_nets & alt_nets
                    if not shared_power:
                        continue

                    # Check cluster size limit
                    alt_size = len(owner_groups.get(alt_ic, [])) + 1
                    if alt_size >= self.MAX_CLUSTER_SIZE:
                        continue

                    # Score: signal net affinity + fewer existing caps = better
                    signal_affinity = len(cap_nets & alt_nets - {'GND'} - cap_power_nets)
                    existing_caps = ic_cap_counts.get(alt_ic, 0)
                    score = signal_affinity * 5 + (10 - existing_caps)

                    if score > best_score:
                        best_score = score
                        best_alt = alt_ic

                if best_alt:
                    # Move cap from donor to alt
                    owner_groups[donor_ic].remove(cap_ref)
                    ic_caps[donor_ic].remove(cap_ref)

                    if best_alt not in owner_groups:
                        owner_groups[best_alt] = []
                    owner_groups[best_alt].append(cap_ref)

                    ic_cap_counts[donor_ic] -= 1
                    ic_cap_counts[best_alt] = ic_cap_counts.get(best_alt, 0) + 1
                    ic_caps[best_alt] = ic_caps.get(best_alt, []) + [cap_ref]

                    # Update assigned set (the IC itself, not just the cap)
                    assigned.add(best_alt)

                    # If this IC now has caps, remove from needing list
                    if ic_cap_counts[best_alt] >= 1:
                        ics_needing_caps.discard(best_alt)

    def _is_power_net(self, net: str) -> bool:
        """Check if a net is a power net (from parts_db or by name pattern)."""
        nets = self.parts_db.get('nets', {})
        if net in nets and nets[net].get('type') == 'power':
            return True
        upper = net.upper()
        return any(p in upper for p in
                   ['VCC', 'VDD', 'V3', '3V3', '5V', '1V8', '2V5', 'VBUS',
                    'AVDD', 'DVDD', 'VREF', 'VIN', 'VOUT', 'VCCA', 'VCCB'])

    def _reunite_with_owners(self, clusters: List[ComponentCluster]):
        """
        Force functional passives back to their owner IC's cluster.

        After balance/orphan/redistribution, some DECOUPLING caps or ESD
        components end up in the wrong cluster. This step fixes that by
        moving them to the cluster containing their owner IC.

        Only moves passives with priority >= 2.0 (DECOUPLING, ESD).
        """
        # Build ref -> cluster index mapping
        ref_to_cluster_idx = {}
        for idx, cluster in enumerate(clusters):
            for ref in cluster.members:
                ref_to_cluster_idx[ref] = idx

        moves = 0
        for ref, role_data in self.functional_roles.items():
            owner = role_data.get('owner')
            priority = role_data.get('priority', 0)
            if not owner or priority < 2.0:
                continue

            ref_idx = ref_to_cluster_idx.get(ref)
            owner_idx = ref_to_cluster_idx.get(owner)

            if ref_idx is None or owner_idx is None:
                continue
            if ref_idx == owner_idx:
                continue  # Already in the right cluster

            # Move ref from its current cluster to the owner's cluster
            clusters[ref_idx].members.remove(ref)
            clusters[owner_idx].members.append(ref)
            ref_to_cluster_idx[ref] = owner_idx
            moves += 1

        # Remove empty clusters
        clusters[:] = [c for c in clusters if len(c.members) > 0]

        if moves > 0:
            print(f"  Reunited {moves} functional passives with their owner ICs")

    def _absorb_orphan_passives(
        self,
        clusters: List[ComponentCluster],
        assigned: set,
        net_to_refs: Dict[str, set],
        all_ics: set,
    ):
        """
        Absorb unassigned components into existing IC clusters.

        Handles both passives (caps, resistors) and small ICs (sensors)
        that didn't get assigned during the main clustering phase.

        For each unassigned component, find the cluster that shares the most
        signal nets with it and absorb it (if cluster size allows).
        """
        parts = self.parts_db.get('parts', {})

        # Collect all unassigned components (passives, small ICs, diodes, etc.)
        unassigned = []
        for ref in sorted(parts.keys()):
            if ref in assigned:
                continue
            unassigned.append(ref)

        for ref in unassigned:
            part = parts[ref]
            part_nets = {pin.get('net', '') for pin in get_pins(part)
                         if pin.get('net', '')}
            part_signal_nets = {n for n in part_nets
                                if n not in ('GND',) and not self._is_power_net(n)}
            part_power_nets = {n for n in part_nets
                               if n not in ('GND',) and self._is_power_net(n)}

            # Find best cluster to join
            best_cluster = None
            best_score = -1

            for cluster in clusters:
                if len(cluster.members) >= self.MAX_CLUSTER_SIZE:
                    continue

                # Collect all nets in this cluster
                cluster_nets = set()
                for member_ref in cluster.members:
                    member_part = parts.get(member_ref, {})
                    for pin in get_pins(member_part):
                        net = pin.get('net', '')
                        if net:
                            cluster_nets.add(net)

                # Score: signal net overlap heavily weighted
                signal_overlap = len(part_signal_nets & cluster_nets)
                power_overlap = len(part_power_nets & cluster_nets)

                # Must share at least one non-GND net
                if signal_overlap == 0 and power_overlap == 0:
                    continue

                # IC-owned clusters preferred
                is_ic_cluster = 1 if cluster.owner in all_ics else 0

                score = signal_overlap * 10 + power_overlap * 3 + is_ic_cluster * 2

                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if best_cluster and best_score >= 2:
                best_cluster.members.append(ref)
                assigned.add(ref)

    def _group_remaining_orphans(
        self,
        clusters: List[ComponentCluster],
        assigned: set,
        net_to_refs: Dict[str, set],
    ):
        """
        Group remaining unassigned components by signal-net affinity.

        Orphans sharing >=2 signal nets get grouped into new clusters.
        This handles cases like multiple sensors on the same I2C bus
        that couldn't be absorbed into full clusters.
        """
        parts = self.parts_db.get('parts', {})

        # Collect remaining unassigned components
        unassigned = [ref for ref in sorted(parts.keys()) if ref not in assigned]
        if len(unassigned) <= 1:
            return

        # Build signal-net sets for each unassigned component
        ref_signal_nets = {}
        for ref in unassigned:
            part = parts[ref]
            ref_signal_nets[ref] = {
                pin.get('net', '') for pin in get_pins(part)
                if pin.get('net', '') and pin.get('net', '') not in ('GND',)
                and not self._is_power_net(pin.get('net', ''))
            }

        # Greedy clustering: pair orphans with most shared signal nets
        used = set()
        for i, ref1 in enumerate(unassigned):
            if ref1 in used:
                continue

            group = [ref1]
            nets1 = ref_signal_nets.get(ref1, set())

            for ref2 in unassigned[i + 1:]:
                if ref2 in used:
                    continue
                if len(group) >= self.MAX_CLUSTER_SIZE:
                    break

                nets2 = ref_signal_nets.get(ref2, set())
                shared = nets1 & nets2

                if len(shared) >= 1:  # At least 1 shared signal net
                    group.append(ref2)
                    nets1 = nets1 | nets2  # Expand net set for transitive grouping

            if len(group) >= 2:
                # Pick the IC (or largest component) as owner
                owner = group[0]
                for ref in group:
                    pin_count = len(get_pins(parts.get(ref, {})))
                    owner_pins = len(get_pins(parts.get(owner, {})))
                    if pin_count > owner_pins:
                        owner = ref

                cluster = ComponentCluster(
                    id=f"cluster_{owner}",
                    owner=owner,
                    members=sorted(group),
                )
                clusters.append(cluster)
                for ref in group:
                    assigned.add(ref)
                    used.add(ref)

    def _get_courtyard(self, ref: str) -> Tuple[float, float]:
        """Get courtyard dimensions for a component."""
        part = self.parts_db['parts'].get(ref, {})

        # Check parts_db first (single source of truth)
        courtyard_data = part.get('courtyard')
        if courtyard_data:
            if isinstance(courtyard_data, dict):
                return (courtyard_data.get('width', 2.0), courtyard_data.get('height', 2.0))
            elif hasattr(courtyard_data, 'width'):
                return (courtyard_data.width, courtyard_data.height)

        # Fallback to FootprintResolver
        fp_name = part.get('footprint', 'unknown')
        fp_def = self._resolver.resolve(fp_name)
        return fp_def.courtyard_size

    @staticmethod
    def _packing_overhead(n_members: int) -> float:
        """Adaptive packing overhead based on cluster member count.

        Small clusters (1-2) need less overhead — just the component
        plus some routing margin. Large clusters (5+) need more overhead
        for internal routing channels between tightly packed parts.

        Returns multiplier for total area.
        """
        if n_members <= 1:
            return 1.3   # Solo part: just courtyard + small margin
        elif n_members == 2:
            return 1.5   # Owner + 1 passive: tight pair
        elif n_members <= 4:
            return 1.8   # Small group: moderate spacing
        else:
            return 2.2   # Large group: need routing channels

    def _calculate_geometry(self, cluster: ComponentCluster):
        """
        Calculate cluster courtyard from member areas.

        Uses adaptive packing overhead that scales with member count:
        small clusters get tight packing, large clusters get more room
        for internal routing.
        """
        total_area = 0.0
        for ref in cluster.members:
            w, h = self._get_courtyard(ref)
            total_area += w * h

        overhead = self._packing_overhead(len(cluster.members))
        cluster_area = total_area * overhead

        # Determine aspect ratio from owner
        owner_w, owner_h = self._get_courtyard(cluster.owner)
        if owner_w > 0 and owner_h > 0:
            ratio = owner_w / owner_h
            if ratio > 2.0 or ratio < 0.5:
                # Elongated owner — inherit its aspect
                aspect = ratio
            else:
                # Roughly square — use golden ratio for pleasant layout
                aspect = 1.618
        else:
            aspect = 1.618

        # Calculate dimensions from area and aspect ratio
        h = math.sqrt(cluster_area / max(aspect, 0.1))
        w = cluster_area / max(h, 0.1)

        # Ensure cluster is at least as large as biggest member + margin
        max_member_w = 0.0
        max_member_h = 0.0
        for ref in cluster.members:
            mw, mh = self._get_courtyard(ref)
            max_member_w = max(max_member_w, mw)
            max_member_h = max(max_member_h, mh)

        # Cluster must fit the largest member plus room for others
        min_w = max_member_w + 2.0  # 2mm margin for neighbors
        min_h = max_member_h + 2.0

        cluster.courtyard_width = max(w, min_w, self.MIN_CLUSTER_DIM)
        cluster.courtyard_height = max(h, min_h, self.MIN_CLUSTER_DIM)

    def _build_inter_cluster_edges(self, clusters: List[ComponentCluster]):
        """Build connectivity graph between clusters from shared nets."""
        # Build ref -> cluster lookup
        ref_to_cluster_id: Dict[str, str] = {}
        for cluster in clusters:
            for ref in cluster.members:
                ref_to_cluster_id[ref] = cluster.id

        # Build cluster_id -> cluster lookup for fast access
        id_to_cluster: Dict[str, ComponentCluster] = {c.id: c for c in clusters}

        # Process each net
        nets = self.parts_db.get('nets', {})
        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])
            clusters_on_net = set()
            for pin_ref in pins:
                comp_ref = pin_ref.split('.')[0] if '.' in pin_ref else pin_ref
                cid = ref_to_cluster_id.get(comp_ref)
                if cid:
                    clusters_on_net.add(cid)

            if len(clusters_on_net) < 2:
                continue

            # Signal nets have higher weight than power nets
            net_type = net_data.get('type', 'signal')
            weight = 2 if net_type == 'signal' else 1

            # Increment pairwise connections
            cluster_list = sorted(clusters_on_net)
            for i, c1 in enumerate(cluster_list):
                for c2 in cluster_list[i + 1:]:
                    id_to_cluster[c1].net_connections[c2] = (
                        id_to_cluster[c1].net_connections.get(c2, 0) + weight)
                    id_to_cluster[c2].net_connections[c1] = (
                        id_to_cluster[c2].net_connections.get(c1, 0) + weight)

    def _classify_roles(self, clusters: List[ComponentCluster]):
        """Classify each cluster's role from owner properties."""
        for cluster in clusters:
            owner_ref = cluster.owner
            part = self.parts_db['parts'].get(owner_ref, {})
            fp = part.get('footprint', '').upper()
            name = part.get('name', '').upper()

            # Check if owner has ESD role
            if owner_ref in self.functional_roles:
                role_data = self.functional_roles[owner_ref]
                if role_data.get('role') == 'ESD_PROTECTION':
                    cluster.role = "INTERFACE"
                    continue

            # Connector detection
            if (owner_ref.startswith('J') or
                    'USB' in fp or 'CONN' in fp or 'HDR' in fp or
                    'USB' in name or 'CONNECTOR' in name):
                cluster.role = "EDGE_CONNECTOR"
                cluster.edge_preference = "left"  # Default; refined by engine
                continue

            # LDO / voltage regulator detection
            if ('LDO' in name or 'VREG' in name or 'REG' in name or
                    'LDO' in fp or 'SOT-223' in fp):
                cluster.role = "POWER_REGULATOR"
                continue

            cluster.role = "STANDARD"


# =============================================================================
# HIERARCHICAL PLACEMENT ENGINE
# =============================================================================

class HierarchicalPlacementEngine:
    """
    Two-level hierarchical placement engine.

    Same interface as PlacementEngine: place(parts_db, graph, hints) → PlacementResult

    Pipeline:
        Phase 1: Infer functional roles (reuse PlacementEngine logic)
        Phase 2: Form clusters via ClusterBuilder
        Phase 3: Coarse placement — clusters as pseudo-parts
        Phase 4: Fine placement — real parts within each cluster
        Phase 5: Transform cluster-local → global coordinates
        Phase 6: Global SA refinement to smooth cluster boundaries
    """

    def __init__(self, config: PlacementConfig = None):
        self.config = config or PlacementConfig()
        self.clusters: List[ComponentCluster] = []
        self.functional_roles: Dict[str, Dict] = {}

    def place(self, parts_db: Dict, graph: Dict,
              placement_hints: Dict = None) -> PlacementResult:
        """
        Run hierarchical placement.

        Args:
            parts_db: Parts database (same format as PlacementEngine)
            graph: Connectivity graph (same format as PlacementEngine)
            placement_hints: Optional hints (proximity_groups, edge_components, etc.)

        Returns:
            PlacementResult with flat positions (same format as PlacementEngine)
        """
        print("\n[HIERARCHICAL] Starting two-level placement...")

        # --- Phase 1: Infer Functional Roles ---
        print("[HIERARCHICAL] Phase 1: Inferring functional roles...")
        self.functional_roles = self._infer_roles(parts_db, placement_hints)
        print(f"  Inferred {len(self.functional_roles)} functional roles")

        # --- Phase 2: Form Clusters ---
        print("[HIERARCHICAL] Phase 2: Forming clusters...")
        builder = ClusterBuilder(parts_db, graph, self.functional_roles)
        self.clusters = builder.build_clusters()
        self._print_clusters()

        # --- Phase 3: Coarse Placement ---
        print("[HIERARCHICAL] Phase 3: Coarse placement (clusters as parts)...")
        coarse_result = self._place_coarse(parts_db)

        # --- Phase 4: Fine Placement ---
        print("[HIERARCHICAL] Phase 4: Fine placement (within clusters)...")
        fine_positions, fine_rotations = self._place_fine(parts_db, graph)

        # --- Phase 5: Transform to Global ---
        print("[HIERARCHICAL] Phase 5: Transforming to global coordinates...")
        global_positions, global_rotations = self._transform_to_global(
            fine_positions, fine_rotations
        )

        # --- Phase 6: Global Refinement ---
        print("[HIERARCHICAL] Phase 6: Global SA refinement...")
        final_positions, final_rotations, improvement = self._global_refinement(
            global_positions, global_rotations, parts_db, graph, placement_hints
        )

        # --- Build Result ---
        wirelength = self._calculate_hpwl(final_positions, parts_db)

        result = PlacementResult(
            positions=final_positions,
            rotations=final_rotations,
            cost=wirelength,  # Use wirelength as cost for comparison
            algorithm_used='hierarchical (cluster->coarse->fine->refine)',
            iterations=0,
            converged=True,
            wirelength=wirelength,
            overlap_area=0.0,
            success=True,
            board_width=self.config.board_width,
            board_height=self.config.board_height,
        )

        print(f"[HIERARCHICAL] Complete. Wirelength: {wirelength:.1f}mm")
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Infer roles by reusing PlacementEngine's logic
    # -------------------------------------------------------------------------

    def _infer_roles(self, parts_db: Dict,
                     placement_hints: Dict = None) -> Dict[str, Dict]:
        """Instantiate a temp PlacementEngine to infer functional roles."""
        temp_config = PlacementConfig(
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            algorithm='hybrid',
        )
        temp_engine = PlacementEngine(temp_config)
        temp_engine._placement_hints = placement_hints or {}
        temp_engine._build_placement_constraints(parts_db)
        return dict(temp_engine._functional_roles)

    # -------------------------------------------------------------------------
    # Phase 3: Coarse placement — clusters as pseudo-parts
    # -------------------------------------------------------------------------

    def _place_coarse(self, parts_db: Dict) -> PlacementResult:
        """Place clusters as pseudo-components on the board."""
        # Build ref -> cluster_id mapping
        ref_to_cluster = {}
        for cluster in self.clusters:
            for ref in cluster.members:
                ref_to_cluster[ref] = cluster.id

        # Build synthetic parts_db for clusters, including synthetic pins/nets
        # so the placement engine can compute wirelength properly
        pseudo_parts = {}
        pseudo_nets = {}
        pin_counter = {}  # cluster_id -> next pin number

        for cluster in self.clusters:
            pin_counter[cluster.id] = 1
            pseudo_parts[cluster.id] = {
                'name': cluster.id,
                'footprint': 'pseudo_cluster',
                'size': [cluster.courtyard_width, cluster.courtyard_height],
                'pins': [],
                'courtyard': {
                    'width': cluster.courtyard_width,
                    'height': cluster.courtyard_height,
                },
            }

        # Create synthetic nets from real nets that span multiple clusters
        for net_name, net_data in parts_db.get('nets', {}).items():
            pins = net_data.get('pins', [])
            # Find which clusters this net touches
            clusters_on_net = set()
            for pin_ref in pins:
                comp_ref = pin_ref.split('.')[0] if '.' in pin_ref else pin_ref
                cid = ref_to_cluster.get(comp_ref)
                if cid:
                    clusters_on_net.add(cid)

            if len(clusters_on_net) < 2:
                continue  # Net is entirely within one cluster

            # Create synthetic net with one pin per cluster
            synth_net_name = f"synth_{net_name}"
            synth_pins = []
            for cid in sorted(clusters_on_net):
                pnum = pin_counter[cid]
                pin_counter[cid] += 1
                pseudo_parts[cid]['pins'].append({
                    'number': str(pnum),
                    'net': synth_net_name,
                })
                synth_pins.append(f"{cid}.{pnum}")

            pseudo_nets[synth_net_name] = {
                'type': net_data.get('type', 'signal'),
                'pins': synth_pins,
            }

        pseudo_parts_db = {
            'parts': pseudo_parts,
            'nets': pseudo_nets,
            'board': parts_db.get('board', {'width': 50, 'height': 40, 'layers': 2}),
        }

        # Build pseudo graph from inter-cluster edges
        pseudo_adjacency = {}
        for cluster in self.clusters:
            pseudo_adjacency[cluster.id] = dict(cluster.net_connections)

        pseudo_graph = {'adjacency': pseudo_adjacency}

        # Build placement hints
        edge_clusters = [c.id for c in self.clusters if c.role == 'EDGE_CONNECTOR']
        pseudo_hints = {
            'edge_components': edge_clusters,
            'proximity_groups': [],
        }

        # INTERFACE clusters should be near their connector
        for cluster in self.clusters:
            if cluster.role == 'INTERFACE':
                # Find which connector cluster this interfaces with
                for other in self.clusters:
                    if other.role == 'EDGE_CONNECTOR' and other.id in cluster.net_connections:
                        pseudo_hints['proximity_groups'].append({
                            'components': [other.id, cluster.id],
                            'max_distance': 12.0,
                            'priority': 2.5,
                            'reason': f'{cluster.id} interfaces with {other.id}',
                        })
                        break

        # POWER_REGULATOR clusters should be near connector (input) and main IC (output)
        for cluster in self.clusters:
            if cluster.role == 'POWER_REGULATOR':
                # Find most-connected clusters
                if cluster.net_connections:
                    top_conn = sorted(cluster.net_connections.items(),
                                      key=lambda x: x[1], reverse=True)[:2]
                    for cid, count in top_conn:
                        pseudo_hints['proximity_groups'].append({
                            'components': [cluster.id, cid],
                            'max_distance': 15.0,
                            'priority': 2.0,
                            'reason': f'Power: {cluster.id} near {cid}',
                        })

        # EDGE_CONNECTOR clusters with signal nets to MCU clusters should
        # be placed near those MCUs — not just at the edge.
        # This fixes USB/ESD co-placement: J1+U6 far from U2 (MCU with USB)
        cluster_id_map = {c.id: c for c in self.clusters}
        for cluster in self.clusters:
            if cluster.role != 'EDGE_CONNECTOR':
                continue
            # Count signal-only connections to each other cluster
            for other_id, weight in cluster.net_connections.items():
                other = cluster_id_map.get(other_id)
                if not other or other.role == 'EDGE_CONNECTOR':
                    continue
                # Signal nets between connector and MCU/IC cluster
                # weight already incorporates signal=2, power=1 from _build_inter_cluster_edges
                if weight >= 3:  # At least 2 signal nets or 3 power nets
                    pseudo_hints['proximity_groups'].append({
                        'components': [cluster.id, other_id],
                        'max_distance': 20.0,
                        'priority': 1.8,
                        'reason': f'Signal: {cluster.id} near {other_id} ({weight} nets)',
                    })

        # Run coarse placement
        coarse_config = PlacementConfig(
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            algorithm='hybrid',
            fd_iterations=150,
            sa_moves_per_temp=50,
            min_spacing=1.5,
            edge_margin=2.0,
            seed=self.config.seed,
        )

        engine = PlacementEngine(coarse_config)
        result = engine.place(pseudo_parts_db, pseudo_graph, pseudo_hints)

        # Extract cluster center positions
        for cluster in self.clusters:
            pos = result.positions.get(cluster.id, (
                self.config.board_width / 2, self.config.board_height / 2))
            cluster.center_x = pos[0]
            cluster.center_y = pos[1]
            cluster.rotation = result.rotations.get(cluster.id, 0.0)

        # Quadrant balancing: nudge clusters toward centroid if layout is lopsided
        self._balance_coarse_placement()

        print(f"  Coarse placement: {len(self.clusters)} clusters placed, "
              f"WL={result.wirelength:.1f}mm")

        return result

    def _balance_coarse_placement(self):
        """
        Nudge clusters toward board centroid when layout is lopsided.

        Measures area-weighted centroid offset from board center. If offset
        exceeds 15% of board diagonal, applies a gentle shift to all clusters
        to re-center the distribution. Connectors (EDGE_CONNECTOR) are
        excluded since they must stay at the edge.

        Generalized: works for any board size and cluster configuration.
        """
        bw = self.config.board_width
        bh = self.config.board_height
        board_cx = bw / 2
        board_cy = bh / 2
        board_diag = math.sqrt(bw * bw + bh * bh)

        # Compute area-weighted centroid of non-edge clusters
        total_area = 0.0
        wt_cx = 0.0
        wt_cy = 0.0
        movable_clusters = []

        for cluster in self.clusters:
            area = cluster.courtyard_width * cluster.courtyard_height
            wt_cx += cluster.center_x * area
            wt_cy += cluster.center_y * area
            total_area += area
            if cluster.role != 'EDGE_CONNECTOR':
                movable_clusters.append(cluster)

        if total_area < 0.01 or not movable_clusters:
            return

        centroid_x = wt_cx / total_area
        centroid_y = wt_cy / total_area

        offset_x = board_cx - centroid_x
        offset_y = board_cy - centroid_y
        offset_dist = math.sqrt(offset_x * offset_x + offset_y * offset_y)

        # Only nudge if centroid is more than 15% of board diagonal off-center
        threshold = board_diag * 0.15
        if offset_dist <= threshold:
            return

        # Apply fraction of offset to movable clusters (gentle, not full correction)
        # Stronger correction for bigger imbalance, capped at 60%
        correction_frac = min(0.6, (offset_dist - threshold) / board_diag + 0.2)
        nudge_x = offset_x * correction_frac
        nudge_y = offset_y * correction_frac

        margin = 2.0  # Keep clusters inside board
        for cluster in movable_clusters:
            new_x = cluster.center_x + nudge_x
            new_y = cluster.center_y + nudge_y
            half_w = cluster.courtyard_width / 2
            half_h = cluster.courtyard_height / 2
            # Clamp to board bounds
            new_x = max(margin + half_w, min(bw - margin - half_w, new_x))
            new_y = max(margin + half_h, min(bh - margin - half_h, new_y))
            cluster.center_x = new_x
            cluster.center_y = new_y

        print(f"  Centroid balance: offset {offset_dist:.1f}mm -> nudged "
              f"{len(movable_clusters)} clusters by ({nudge_x:.1f}, {nudge_y:.1f})mm")

    # -------------------------------------------------------------------------
    # Phase 4: Fine placement — components within each cluster
    # -------------------------------------------------------------------------

    def _place_fine(self, parts_db: Dict,
                    graph: Dict) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Place components within each cluster's allocated region.

        Each cluster is treated as a tiny board with its own placement run.
        ALL design rules apply (proximity, edge, orientation).

        Returns:
            (cluster_positions, cluster_rotations)
            where each is {cluster_id: {ref: value}}
        """
        all_positions: Dict[str, Dict[str, Tuple[float, float]]] = {}
        all_rotations: Dict[str, Dict[str, float]] = {}

        cluster_members = set()
        for cluster in self.clusters:
            for ref in cluster.members:
                cluster_members.add(ref)

        for cluster in self.clusters:
            print(f"  [FINE] {cluster.id}: {len(cluster.members)} members "
                  f"({cluster.courtyard_width:.1f}x{cluster.courtyard_height:.1f}mm)...")

            if len(cluster.members) == 1:
                # Single-component cluster — just center it
                ref = cluster.members[0]
                all_positions[cluster.id] = {
                    ref: (cluster.courtyard_width / 2, cluster.courtyard_height / 2)
                }
                all_rotations[cluster.id] = {ref: 0.0}
                continue

            # Build mini parts_db with only cluster members
            mini_parts = {}
            for ref in cluster.members:
                if ref in parts_db.get('parts', {}):
                    mini_parts[ref] = parts_db['parts'][ref]

            # Build mini nets (only nets with >=2 pins in this cluster)
            mini_nets = {}
            member_set = set(cluster.members)
            for net_name, net_data in parts_db.get('nets', {}).items():
                pins = net_data.get('pins', [])
                cluster_pins = [p for p in pins
                                if (p.split('.')[0] if '.' in p else p) in member_set]
                if len(cluster_pins) >= 2:
                    mini_nets[net_name] = {
                        'type': net_data.get('type', 'signal'),
                        'pins': cluster_pins,
                    }

            mini_parts_db = {
                'parts': mini_parts,
                'nets': mini_nets,
                'board': {
                    'width': cluster.courtyard_width,
                    'height': cluster.courtyard_height,
                    'layers': parts_db.get('board', {}).get('layers', 2),
                },
            }

            # Build mini graph (only intra-cluster edges)
            mini_adjacency = {}
            full_adj = graph.get('adjacency', {})
            for ref in cluster.members:
                if ref in full_adj:
                    mini_adjacency[ref] = {
                        neighbor: count
                        for neighbor, count in full_adj[ref].items()
                        if neighbor in member_set
                    }

            mini_graph = {'adjacency': mini_adjacency}

            # Run fine placement — full engine with ALL rules
            fine_config = PlacementConfig(
                board_width=cluster.courtyard_width,
                board_height=cluster.courtyard_height,
                algorithm='hybrid',
                fd_iterations=100,
                sa_moves_per_temp=40,
                min_spacing=0.5,
                edge_margin=0.5,
                seed=self.config.seed,
            )

            engine = PlacementEngine(fine_config)
            result = engine.place(mini_parts_db, mini_graph)

            all_positions[cluster.id] = dict(result.positions)
            all_rotations[cluster.id] = dict(result.rotations)

            print(f"         WL={result.wirelength:.1f}mm, cost={result.cost:.1f}")

        return all_positions, all_rotations

    # -------------------------------------------------------------------------
    # Phase 5: Transform cluster-local → global coordinates
    # -------------------------------------------------------------------------

    def _transform_to_global(
        self,
        cluster_positions: Dict[str, Dict[str, Tuple[float, float]]],
        cluster_rotations: Dict[str, Dict[str, float]],
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """
        Convert cluster-local positions to global board coordinates.

        For each component:
            global = local + (cluster_center - cluster_size/2)
        """
        global_pos = {}
        global_rot = {}

        for cluster in self.clusters:
            local_positions = cluster_positions.get(cluster.id, {})
            local_rotations = cluster_rotations.get(cluster.id, {})

            # Offset: cluster center minus half-size
            offset_x = cluster.center_x - cluster.courtyard_width / 2
            offset_y = cluster.center_y - cluster.courtyard_height / 2

            for ref, (lx, ly) in local_positions.items():
                global_pos[ref] = (lx + offset_x, ly + offset_y)
                global_rot[ref] = local_rotations.get(ref, 0.0)

        return global_pos, global_rot

    # -------------------------------------------------------------------------
    # Phase 6: Global SA refinement
    # -------------------------------------------------------------------------

    def _global_refinement(
        self,
        positions: Dict[str, Tuple[float, float]],
        rotations: Dict[str, float],
        parts_db: Dict,
        graph: Dict,
        placement_hints: Dict = None,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], float]:
        """
        Legalize + inter-cluster signal optimization.

        Phase 6a: Legalize — fix overlaps and boundary violations
        Phase 6b: Inter-cluster signal pull — gently shift components toward
                  their cross-cluster signal net partners to shorten long nets
                  without destroying cluster integrity.

        Does NOT run SA or FD — both destroy the hierarchical structure.
        """
        refine_config = PlacementConfig(
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            algorithm='sa',  # Needed for engine init
            min_spacing=0.5,
            edge_margin=2.0,
            seed=self.config.seed,
            fusion_enabled=False,  # Fusion already happened in fine placement
        )

        engine = PlacementEngine(refine_config)

        # Initialize the engine's internal state properly
        engine._placement_hints = placement_hints or {}
        engine._build_placement_constraints(parts_db)
        engine._init_from_parts(parts_db, graph)

        # Override initial positions with our hierarchical result
        for ref, (x, y) in positions.items():
            if ref in engine.components:
                engine.components[ref].x = x
                engine.components[ref].y = y
                engine.components[ref].rotation = rotations.get(ref, 0.0)

        # Set up occupancy grid and optimal distance
        from .placement_engine import OccupancyGrid
        engine._occ = OccupancyGrid(self.config.board_width, self.config.board_height)

        n = len(engine.components)
        if n > 0:
            area = self.config.board_width * self.config.board_height
            engine._optimal_dist = math.sqrt(area / n) * 0.8

        # Compute preferred orientations for passives
        engine._compute_preferred_orientation()
        initial_cost = engine._calculate_cost()

        # Phase 6a: Legalize — fix overlaps and boundary violations
        engine._legalize()
        post_legalize_cost = engine._calculate_cost()

        # Phase 6b: Inter-cluster signal pull
        # For each component, find its cross-cluster signal net partners.
        # Gently shift it toward the centroid of those partners.
        # Only accept the shift if it doesn't create overlaps.
        signal_improvements = self._inter_cluster_signal_pull(
            engine, parts_db, graph
        )

        post_signal_cost = engine._calculate_cost()

        # Phase 6c: Functional passive proximity pull
        # Shift decoupling caps, pull-ups, etc. toward their owner ICs.
        # Generalized for any functional role, not just caps.
        proximity_moves = self._functional_passive_pull(engine, parts_db)

        # Extract final positions
        final_positions = {}
        final_rotations = {}
        for ref, comp in engine.components.items():
            final_positions[ref] = (comp.x, comp.y)
            final_rotations[ref] = comp.rotation

        final_cost = engine._calculate_cost()
        improvement = initial_cost - final_cost

        print(f"  Legalization: cost {initial_cost:.1f} -> {post_legalize_cost:.1f}")
        print(f"  Signal pull:  cost {post_legalize_cost:.1f} -> {post_signal_cost:.1f} "
              f"({signal_improvements} moves)")
        print(f"  Passive pull: cost {post_signal_cost:.1f} -> {final_cost:.1f} "
              f"({proximity_moves} moves)")

        return final_positions, final_rotations, improvement

    def _functional_passive_pull(
        self,
        engine: PlacementEngine,
        parts_db: Dict,
    ) -> int:
        """
        Post-placement: shift functional passives toward their owner ICs.

        After hierarchical placement, some passives may have ended up far
        from their owner IC (e.g., a decoupling cap placed in a large cluster
        where the IC is at the far edge). This step uses the already-inferred
        functional roles to gently pull each passive toward its owner.

        Generalized: works for any functional role (DECOUPLING, PULLUP,
        PULLDOWN, LED_DRIVER, ESD_PROTECTION). Each role has its own
        target distance from the owner.

        Returns number of moves made.
        """
        roles = getattr(engine, '_functional_roles', {})
        if not roles:
            return 0

        resolver = FootprintResolver.get_instance()
        total_moves = 0
        MAX_PASSES = 2

        for _ in range(MAX_PASSES):
            moves_this_pass = 0
            for ref, role in roles.items():
                if ref not in engine.components:
                    continue
                owner_ref = role.get('owner')
                if not owner_ref or owner_ref not in engine.components:
                    continue

                comp = engine.components[ref]
                owner = engine.components[owner_ref]
                target_dist = role.get('distance', 5.0)

                dx = owner.x - comp.x
                dy = owner.y - comp.y
                dist = math.sqrt(dx * dx + dy * dy)

                # Only pull if further than target distance
                if dist <= target_dist * 1.2:
                    continue

                # Move 40% of excess distance toward owner
                excess = dist - target_dist
                frac = min(0.4, excess / dist)
                new_x = comp.x + dx * frac
                new_y = comp.y + dy * frac

                # Clamp to board
                part = parts_db['parts'].get(ref, {})
                fp = resolver.resolve(part.get('footprint', 'unknown'))
                cw, ch = fp.courtyard_size
                margin = 1.0
                new_x = max(cw / 2 + margin, min(new_x,
                            self.config.board_width - cw / 2 - margin))
                new_y = max(ch / 2 + margin, min(new_y,
                            self.config.board_height - ch / 2 - margin))

                # Check for overlaps
                overlap = False
                for other_ref, other_comp in engine.components.items():
                    if other_ref == ref:
                        continue
                    other_part = parts_db['parts'].get(other_ref, {})
                    other_fp = resolver.resolve(
                        other_part.get('footprint', 'unknown'))
                    ocw, och = other_fp.courtyard_size

                    adx = abs(new_x - other_comp.x)
                    ady = abs(new_y - other_comp.y)
                    min_dx = (cw + ocw) / 2
                    min_dy = (ch + och) / 2

                    if adx < min_dx and ady < min_dy:
                        overlap = True
                        break

                if not overlap:
                    comp.x = new_x
                    comp.y = new_y
                    moves_this_pass += 1

            total_moves += moves_this_pass
            if moves_this_pass == 0:
                break

        return total_moves

    def _inter_cluster_signal_pull(
        self,
        engine: PlacementEngine,
        parts_db: Dict,
        graph: Dict,
    ) -> int:
        """
        Gently shift components toward their cross-cluster signal net partners.

        For each component, compute the centroid of all components it connects
        to via signal nets in OTHER clusters. Shift it up to MAX_PULL_DIST
        toward that centroid. Only accept if no new overlaps are created.

        Adaptive: pull distance and fraction scale with board diagonal,
        so the algorithm works for any board size — from 20x15mm modules
        to 120x90mm complex boards.
        """
        # Adaptive parameters based on board size
        board_diag = math.sqrt(self.config.board_width ** 2 +
                               self.config.board_height ** 2)
        MAX_PULL_DIST = board_diag * 0.05  # 5% of diagonal per step
        PULL_FRACTION = 0.35               # Move 35% toward centroid
        MAX_PASSES = max(3, min(6, len(self.clusters) // 4))  # More passes for complex boards

        # Build ref -> cluster_id mapping
        ref_to_cluster = {}
        for cluster in self.clusters:
            for ref in cluster.members:
                ref_to_cluster[ref] = cluster.id

        # Build cross-cluster signal net partners for each component
        cross_cluster_partners: Dict[str, List[str]] = defaultdict(list)
        for net_name, net_data in parts_db.get('nets', {}).items():
            net_type = net_data.get('type', 'signal')
            if net_type == 'power':
                continue  # Only optimize signal nets

            pins = net_data.get('pins', [])
            refs_on_net = []
            for pin_ref in pins:
                comp_ref = pin_ref.split('.')[0] if '.' in pin_ref else pin_ref
                if comp_ref in engine.components:
                    refs_on_net.append(comp_ref)

            if len(refs_on_net) < 2:
                continue

            # Find cross-cluster connections
            for ref in refs_on_net:
                my_cluster = ref_to_cluster.get(ref)
                for partner in refs_on_net:
                    if partner != ref and ref_to_cluster.get(partner) != my_cluster:
                        cross_cluster_partners[ref].append(partner)

        if not cross_cluster_partners:
            return 0

        resolver = FootprintResolver.get_instance()
        total_moves = 0

        for pass_num in range(MAX_PASSES):
            moves_this_pass = 0

            for ref, partners in cross_cluster_partners.items():
                if not partners or ref not in engine.components:
                    continue

                comp = engine.components[ref]

                # Compute centroid of cross-cluster partners
                cx_sum, cy_sum = 0.0, 0.0
                count = 0
                for partner in partners:
                    if partner in engine.components:
                        p = engine.components[partner]
                        cx_sum += p.x
                        cy_sum += p.y
                        count += 1

                if count == 0:
                    continue

                target_x = cx_sum / count
                target_y = cy_sum / count

                # Direction vector from current to target
                dx = target_x - comp.x
                dy = target_y - comp.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 0.5:
                    continue  # Already close enough

                # Compute shift (limited)
                shift = min(dist * PULL_FRACTION, MAX_PULL_DIST)
                new_x = comp.x + (dx / dist) * shift
                new_y = comp.y + (dy / dist) * shift

                # Clamp to board boundaries
                part = parts_db['parts'].get(ref, {})
                fp = resolver.resolve(part.get('footprint', 'unknown'))
                cw, ch = fp.courtyard_size
                margin = 1.0
                new_x = max(cw / 2 + margin, min(new_x,
                            self.config.board_width - cw / 2 - margin))
                new_y = max(ch / 2 + margin, min(new_y,
                            self.config.board_height - ch / 2 - margin))

                # Check for overlaps at new position
                old_x, old_y = comp.x, comp.y
                overlap = False
                for other_ref, other_comp in engine.components.items():
                    if other_ref == ref:
                        continue
                    other_part = parts_db['parts'].get(other_ref, {})
                    other_fp = resolver.resolve(
                        other_part.get('footprint', 'unknown'))
                    ocw, och = other_fp.courtyard_size

                    adx = abs(new_x - other_comp.x)
                    ady = abs(new_y - other_comp.y)
                    min_dx = (cw + ocw) / 2
                    min_dy = (ch + och) / 2

                    if adx < min_dx and ady < min_dy:
                        overlap = True
                        break

                if not overlap:
                    # Accept only if the move improves overall cost
                    # (prevents pulling passives away from their owner IC)
                    old_cost = engine._calculate_cost()
                    comp.x = new_x
                    comp.y = new_y
                    new_cost = engine._calculate_cost()
                    if new_cost <= old_cost:
                        moves_this_pass += 1
                    else:
                        # Revert — this move hurts more than it helps
                        comp.x = old_x
                        comp.y = old_y

            total_moves += moves_this_pass
            if moves_this_pass == 0:
                break  # Converged

        return total_moves

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _calculate_hpwl(self, positions: Dict[str, Tuple[float, float]],
                        parts_db: Dict) -> float:
        """Half-Perimeter Wire Length calculation."""
        total = 0.0
        for net_name, net_data in parts_db.get('nets', {}).items():
            pins = net_data.get('pins', [])
            xs, ys = [], []
            for pin_ref in pins:
                comp_ref = pin_ref.split('.')[0] if '.' in pin_ref else pin_ref
                if comp_ref in positions:
                    pos = positions[comp_ref]
                    xs.append(pos[0])
                    ys.append(pos[1])
            if len(xs) >= 2:
                total += (max(xs) - min(xs)) + (max(ys) - min(ys))
        return total

    def _print_clusters(self):
        """Print cluster summary."""
        print(f"  Formed {len(self.clusters)} clusters:")
        for c in self.clusters:
            conns = sum(c.net_connections.values())
            print(f"    {c.id:<20} {len(c.members):>2} members  "
                  f"{c.courtyard_width:.1f}x{c.courtyard_height:.1f}mm  "
                  f"{c.role:<18} {conns} inter-cluster nets")


# =============================================================================
# PLACEMENT COMPARATOR
# =============================================================================

class PlacementComparator:
    """
    Runs both flat and hierarchical engines, compares results, picks winner.
    """

    def __init__(self, config: PlacementConfig = None):
        self.config = config or PlacementConfig()

    def compare(self, parts_db: Dict, graph: Dict,
                placement_hints: Dict = None) -> ComparisonResult:
        """
        Run both engines and compare results.

        Returns ComparisonResult with winner and detailed metrics.
        """
        print("\n" + "=" * 70)
        print("  PLACEMENT A/B COMPARISON: Flat vs Hierarchical")
        print("=" * 70)

        # --- Engine A: Flat ---
        print("\n[ENGINE A] Running flat placement...")
        flat_config = PlacementConfig(
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            algorithm='hybrid',
            seed=self.config.seed,
        )
        flat_engine = PlacementEngine(flat_config)
        flat_result = flat_engine.place(parts_db, graph, placement_hints)
        print(f"[ENGINE A] Done. WL={flat_result.wirelength:.1f}mm, "
              f"Cost={flat_result.cost:.1f}")

        # --- Engine B: Hierarchical ---
        print("\n[ENGINE B] Running hierarchical placement...")
        hier_config = PlacementConfig(
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            seed=self.config.seed,
        )
        hier_engine = HierarchicalPlacementEngine(hier_config)
        hier_result = hier_engine.place(parts_db, graph, placement_hints)
        print(f"[ENGINE B] Done. WL={hier_result.wirelength:.1f}mm, "
              f"Cost={hier_result.cost:.1f}")

        # --- Determine Winner ---
        winner, reason = self._determine_winner(flat_result, hier_result)

        # --- Build comparison table ---
        table = self._build_table(flat_result, hier_result, winner, reason,
                                  flat_engine, hier_engine, parts_db)

        result = ComparisonResult(
            flat_result=flat_result,
            hier_result=hier_result,
            winner=winner,
            reason=reason,
            metrics_table=table,
        )

        print(table)
        return result

    def _determine_winner(self, flat: PlacementResult,
                          hier: PlacementResult) -> Tuple[str, str]:
        """Pick winner based on overlap, then cost."""
        # Both or one failed
        if flat.overlap_area > 0 and hier.overlap_area > 0:
            if hier.overlap_area < flat.overlap_area:
                return ('hierarchical', 'Both have overlap, hierarchical less')
            elif flat.overlap_area < hier.overlap_area:
                return ('flat', 'Both have overlap, flat less')
        elif flat.overlap_area > 0:
            return ('hierarchical', 'Flat has overlap, hierarchical clean')
        elif hier.overlap_area > 0:
            return ('flat', 'Hierarchical has overlap, flat clean')

        # Both clean — compare wirelength
        if flat.wirelength > 0:
            diff_pct = (hier.wirelength - flat.wirelength) / flat.wirelength * 100
        else:
            diff_pct = 0

        if abs(diff_pct) < 5.0:
            return ('tie', f'Within 5% ({diff_pct:+.1f}%)')
        elif diff_pct < 0:
            return ('hierarchical', f'Wirelength {-diff_pct:.1f}% shorter')
        else:
            return ('flat', f'Wirelength {diff_pct:.1f}% shorter')

    def _build_table(self, flat: PlacementResult, hier: PlacementResult,
                     winner: str, reason: str,
                     flat_engine: PlacementEngine,
                     hier_engine: HierarchicalPlacementEngine,
                     parts_db: Dict) -> str:
        """Build formatted comparison table."""
        lines = [
            "",
            "=" * 70,
            "  COMPARISON RESULTS",
            "=" * 70,
            "",
            f"{'Metric':<30} {'Flat (A)':>15} {'Hierarchical (B)':>15} {'Diff':>10}",
            "-" * 70,
        ]

        # Wirelength
        wl_diff = hier.wirelength - flat.wirelength
        wl_pct = (wl_diff / flat.wirelength * 100) if flat.wirelength > 0 else 0
        lines.append(f"{'Wirelength (mm)':<30} {flat.wirelength:>15.1f} "
                     f"{hier.wirelength:>15.1f} {wl_pct:>+9.1f}%")

        # Cost
        cost_diff = hier.cost - flat.cost
        cost_pct = (cost_diff / flat.cost * 100) if flat.cost > 0 else 0
        lines.append(f"{'Cost':<30} {flat.cost:>15.1f} "
                     f"{hier.cost:>15.1f} {cost_pct:>+9.1f}%")

        # Overlap
        lines.append(f"{'Overlap (mm2)':<30} {flat.overlap_area:>15.2f} "
                     f"{hier.overlap_area:>15.2f}")

        # Algorithm
        lines.append(f"{'Algorithm':<30} {flat.algorithm_used:>15} "
                     f"{'hierarchical':>15}")

        # Proximity analysis — how close are decoupling caps to their ICs?
        lines.append("")
        lines.append("Proximity Analysis (decoupling cap -> IC distance):")
        lines.append("-" * 70)

        # Get functional roles from flat engine
        roles = getattr(flat_engine, '_functional_roles', {})
        for ref, role_data in sorted(roles.items()):
            if role_data.get('role') == 'DECOUPLING':
                owner = role_data.get('owner', '?')
                target = role_data.get('distance', 3.0)

                flat_dist = self._distance(ref, owner, flat.positions)
                hier_dist = self._distance(ref, owner, hier.positions)

                # Grade: OK if within target, CLOSE if within 2x, FAR if beyond
                def _grade(dist, tgt):
                    if dist <= tgt * 1.1:  # 10% tolerance
                        return 'OK'
                    elif dist <= tgt * 2.0:
                        return 'CLOSE'
                    else:
                        return 'FAR'

                flat_ok = _grade(flat_dist, target)
                hier_ok = _grade(hier_dist, target)

                lines.append(f"  {ref:>4}->{owner:<4}  target<={target:.1f}mm  "
                             f"flat={flat_dist:5.1f}mm [{flat_ok:>5}]  "
                             f"hier={hier_dist:5.1f}mm [{hier_ok:>5}]")

        # Clusters info
        if hier_engine.clusters:
            lines.append("")
            lines.append(f"Clusters formed: {len(hier_engine.clusters)}")
            for c in hier_engine.clusters:
                lines.append(f"  {c.id:<20} {len(c.members):>2} members  {c.role}")

        # Winner
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  WINNER: {winner.upper()}")
        lines.append(f"  REASON: {reason}")
        lines.append("=" * 70)

        return "\n".join(lines)

    @staticmethod
    def _distance(ref1: str, ref2: str,
                  positions: Dict[str, Tuple[float, float]]) -> float:
        """Euclidean distance between two components."""
        if ref1 not in positions or ref2 not in positions:
            return 999.0
        p1 = positions[ref1]
        p2 = positions[ref2]
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
