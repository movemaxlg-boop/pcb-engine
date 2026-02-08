"""
PCB Engine - Graph Module
=========================

Builds connectivity graph from parts database.
Used in Phase 1 and Phase 2 (hub identification).
"""

from typing import Dict, List, Optional, Tuple


class GraphBuilder:
    """
    Builds a connectivity graph from the parts database.

    The graph represents:
    - Nodes: Components
    - Edges: Nets connecting components
    - Weights: Number of connections between components
    """

    def __init__(self):
        self.graph = {}
        self.node_scores = {}

    def build(self, parts_db: Dict) -> Dict:
        """
        Build connectivity graph from parts database.

        Returns:
            {
                'nodes': {ref: node_data, ...},
                'edges': [(ref_a, ref_b, weight, nets), ...],
                'adjacency': {ref: {neighbor: weight, ...}, ...},
            }
        """
        nodes = {}
        edges = []
        adjacency = {}

        # Build nodes from parts
        for ref, part in parts_db['parts'].items():
            nodes[ref] = {
                'ref': ref,
                'role': part.get('circuit_role', 'passive'),
                'pin_count': part.get('pin_count', {}).get('total', 0),
                'used_pins': part.get('pin_count', {}).get('used', 0),
            }
            adjacency[ref] = {}

        # Build edges from nets
        nets = parts_db.get('nets', {})
        for net_name, net_info in nets.items():
            pins = net_info.get('pins', [])

            # Each net creates edges between all components on it
            # CRITICAL: Sort to ensure deterministic order (set() has random order)
            components = sorted(set(comp for comp, pin in pins))

            for i, comp_a in enumerate(components):
                for comp_b in components[i+1:]:
                    # Count connections between these components on this net
                    count = sum(1 for c, p in pins if c == comp_a)
                    count *= sum(1 for c, p in pins if c == comp_b)

                    # Add edge
                    edges.append((comp_a, comp_b, count, net_name))

                    # Update adjacency
                    adjacency[comp_a][comp_b] = adjacency[comp_a].get(comp_b, 0) + count
                    adjacency[comp_b][comp_a] = adjacency[comp_b].get(comp_a, 0) + count

        self.graph = {
            'nodes': nodes,
            'edges': edges,
            'adjacency': adjacency,
        }

        # Calculate node scores for hub identification
        self._calculate_scores()

        return self.graph

    def identify_hub(self, graph: Dict = None) -> Optional[str]:
        """
        Identify the hub component (highest connectivity).

        The hub is the component with the highest connectivity score.
        Score = sum of all edge weights * number of unique neighbors
        """
        if graph is None:
            graph = self.graph

        adjacency = graph.get('adjacency', {})
        nodes = graph.get('nodes', {})

        scores = {}
        for ref, neighbors in adjacency.items():
            total_weight = sum(neighbors.values())
            neighbor_count = len(neighbors)
            scores[ref] = total_weight * neighbor_count

        if not scores:
            return None

        # Return highest scoring component
        hub = max(scores, key=lambda k: scores[k])

        # Verify it's actually a hub (not just a capacitor with 2 connections)
        if scores[hub] < 10:  # Arbitrary threshold
            return None

        return hub

    def get_component_connectivity(self, ref: str) -> Dict:
        """Get connectivity info for a component"""
        if ref not in self.graph.get('adjacency', {}):
            return {}

        neighbors = self.graph['adjacency'][ref]
        return {
            'neighbors': list(neighbors.keys()),
            'total_connections': sum(neighbors.values()),
            'neighbor_count': len(neighbors),
            'score': self.node_scores.get(ref, 0),
        }

    def get_destination_analysis(self, ref: str) -> Dict:
        """
        Analyze where a component's connections go.
        Used for escape direction calculation.
        """
        if ref not in self.graph.get('adjacency', {}):
            return {}

        neighbors = self.graph['adjacency'][ref]

        # Count by direction (to be calculated based on placement)
        # This is a placeholder - actual implementation needs placement data
        return {
            'neighbor_refs': list(neighbors.keys()),
            'weights': neighbors,
        }

    def _calculate_scores(self):
        """Calculate connectivity scores for all nodes"""
        adjacency = self.graph.get('adjacency', {})

        for ref, neighbors in adjacency.items():
            total_weight = sum(neighbors.values())
            neighbor_count = len(neighbors)
            self.node_scores[ref] = total_weight * neighbor_count
