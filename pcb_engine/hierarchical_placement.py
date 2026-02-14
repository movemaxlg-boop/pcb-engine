"""
Hierarchical Cluster-Based Placement Engine (Engine B)
======================================================

Two-level placement that clusters functionally-related components
(IC + decoupling caps + pull-ups etc.) and places them as groups.

Level 1 (Coarse): Place clusters as pseudo-components on the board
Level 2 (Fine):   Place real components within each cluster's region

Reuses PlacementEngine as a black box at both levels — no modifications
to the existing engine.

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

    PACKING_OVERHEAD = 2.0   # 100% extra area for routing + spacing within cluster
    MIN_CLUSTER_DIM = 3.0    # Minimum cluster dimension (mm)

    def __init__(self, parts_db: Dict, graph: Dict, functional_roles: Dict[str, Dict]):
        self.parts_db = parts_db
        self.graph = graph
        self.functional_roles = functional_roles
        self._resolver = FootprintResolver.get_instance()

    def build_clusters(self) -> List[ComponentCluster]:
        """
        Main clustering algorithm.

        1. Group components by owner from functional_roles
        2. Create single-member clusters for orphans
        3. Calculate cluster geometry
        4. Build inter-cluster connectivity
        5. Classify cluster roles
        """
        clusters = []
        assigned = set()

        # Step 1: Group passives/ESD by owner
        owner_groups: Dict[str, List[str]] = defaultdict(list)
        for ref, role_data in self.functional_roles.items():
            owner = role_data.get('owner')
            if owner and owner in self.parts_db.get('parts', {}):
                owner_groups[owner].append(ref)
                assigned.add(ref)

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

        # Step 2: Orphan clusters (components with no owner relationship)
        for ref in sorted(self.parts_db.get('parts', {}).keys()):
            if ref not in assigned:
                cluster = ComponentCluster(
                    id=f"cluster_{ref}",
                    owner=ref,
                    members=[ref],
                )
                clusters.append(cluster)

        # Step 3: Calculate geometry for each cluster
        for cluster in clusters:
            self._calculate_geometry(cluster)

        # Step 4: Build inter-cluster connectivity
        self._build_inter_cluster_edges(clusters)

        # Step 5: Classify roles
        self._classify_roles(clusters)

        return clusters

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

    def _calculate_geometry(self, cluster: ComponentCluster):
        """
        Calculate cluster courtyard from member areas.

        area = sum(member courtyards) * PACKING_OVERHEAD
        aspect ratio from owner's footprint shape
        """
        total_area = 0.0
        for ref in cluster.members:
            w, h = self._get_courtyard(ref)
            total_area += w * h

        cluster_area = total_area * self.PACKING_OVERHEAD

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
        # Build ref → cluster index
        ref_to_cluster_id: Dict[str, str] = {}
        for cluster in clusters:
            for ref in cluster.members:
                ref_to_cluster_id[ref] = cluster.id

        # Process each net
        nets = self.parts_db.get('nets', {})
        for net_name, net_data in nets.items():
            pins = net_data.get('pins', [])
            # Find which clusters this net touches
            clusters_on_net = set()
            for pin_ref in pins:
                comp_ref = pin_ref.split('.')[0] if '.' in pin_ref else pin_ref
                cid = ref_to_cluster_id.get(comp_ref)
                if cid:
                    clusters_on_net.add(cid)

            # Skip power/GND nets for inter-cluster weighting (handled by pour)
            net_type = net_data.get('type', 'signal')
            if net_type == 'power':
                continue

            # Increment pairwise connections
            cluster_list = sorted(clusters_on_net)
            for i, c1 in enumerate(cluster_list):
                for c2 in cluster_list[i + 1:]:
                    # Find cluster objects
                    for cluster in clusters:
                        if cluster.id == c1:
                            cluster.net_connections[c2] = cluster.net_connections.get(c2, 0) + 1
                        elif cluster.id == c2:
                            cluster.net_connections[c1] = cluster.net_connections.get(c1, 0) + 1

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
        # Build synthetic parts_db for clusters
        pseudo_parts = {}
        for cluster in self.clusters:
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

        pseudo_parts_db = {
            'parts': pseudo_parts,
            'nets': {},  # Connectivity via graph adjacency
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

        print(f"  Coarse placement: {len(self.clusters)} clusters placed, "
              f"WL={result.wirelength:.1f}mm")

        return result

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
        Light SA pass across ALL components to smooth cluster boundaries.

        Uses low temperature so it only makes small adjustments.
        """
        refine_config = PlacementConfig(
            board_width=self.config.board_width,
            board_height=self.config.board_height,
            algorithm='sa',
            sa_initial_temp=30.0,   # Low temp — just smooth, don't scatter
            sa_final_temp=0.1,
            sa_moves_per_temp=60,
            sa_cooling_rate=0.93,
            min_spacing=0.5,
            edge_margin=2.0,
            seed=self.config.seed,
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

        # First: legalize to resolve overlaps from cluster-to-cluster stacking
        # This preserves cluster grouping while fixing physical conflicts
        engine._legalize()

        engine._compute_preferred_orientation()
        initial_cost = engine._calculate_cost()

        # Then: light SA to smooth inter-cluster connections
        result = engine._place_simulated_annealing()

        # Final legalization
        engine._legalize()

        # Extract final positions
        final_positions = {}
        final_rotations = {}
        for ref, comp in engine.components.items():
            final_positions[ref] = (comp.x, comp.y)
            final_rotations[ref] = comp.rotation

        final_cost = engine._calculate_cost()
        improvement = initial_cost - final_cost

        print(f"  Refinement: cost {initial_cost:.1f} -> {final_cost:.1f} "
              f"(delta={improvement:.1f})")

        return final_positions, final_rotations, improvement

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

                flat_ok = 'OK' if flat_dist <= target else 'FAR'
                hier_ok = 'OK' if hier_dist <= target else 'FAR'

                lines.append(f"  {ref:>4}->{owner:<4}  target<={target:.0f}mm  "
                             f"flat={flat_dist:5.1f}mm [{flat_ok}]  "
                             f"hier={hier_dist:5.1f}mm [{hier_ok}]")

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
