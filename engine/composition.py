"""Scenario composition module - DAG-based scenario construction.

Allows building complex scenarios from modular components using a directed
acyclic graph (DAG) structure. Supports parallel/sequential execution,
conditional branching, and node-level hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable


class NodeType(Enum):
    """Types of DAG nodes."""

    SCENARIO = "scenario"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    OUTPUT = "output"


@dataclass
class DAGNode:
    """A node in the scenario DAG."""

    id: str
    type: NodeType
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    # Hook called after node execution
    on_complete: Optional[Callable] = None


@dataclass
class ScenarioDAG:
    """Directed acyclic graph for scenario composition."""

    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)  # (from_id, to_id)

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG."""
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge from one node to another."""
        self.edges.append((from_id, to_id))

    def execute(self, engine) -> Dict[str, Any]:
        """Execute the DAG using the scenario engine.

        Returns a dict of output results keyed by node ID.
        """
        # Topological sort
        order = self._topological_sort()
        results: Dict[str, Any] = {}

        for node_id in order:
            node = self.nodes[node_id]
            if node.type == NodeType.SCENARIO:
                result = engine.run_scenario(node.label, node.params)
                results[node_id] = result
            elif node.type == NodeType.CONDITION:
                # Evaluate condition based on previous results
                deps = [e[0] for e in self.edges if e[1] == node_id]
                dep_results = {d: results[d] for d in deps if d in results}
                result = engine.evaluate_condition(node.label, dep_results, node.params)
                results[node_id] = result
            elif node.type == NodeType.PARALLEL:
                # Execute all predecessors in parallel
                predecessors = [e[0] for e in self.edges if e[1] == node_id]
                dep_results = {d: results[d] for d in predecessors if d in results}
                result = engine.run_parallel(dep_results, node.params)
                results[node_id] = result
            elif node.type == NodeType.SEQUENCE:
                predecessors = [e[0] for e in self.edges if e[1] == node_id]
                prev_result = results.get(predecessors[0]) if predecessors else None
                result = engine.run_sequence(prev_result, node.params)
                results[node_id] = result

        return results

    def _topological_sort(self) -> List[str]:
        """Return node IDs in topological order."""
        # Kahn's algorithm
        in_degree = {node_id: 0 for node_id in self.nodes}
        for from_id, to_id in self.edges:
            in_degree[to_id] = in_degree.get(to_id, 0) + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for from_id, to_id in self.edges:
                if to_id == node_id:
                    continue
                if from_id == node_id:
                    in_degree[to_id] -= 1
                    if in_degree[to_id] == 0:
                        queue.append(to_id)

        return order
