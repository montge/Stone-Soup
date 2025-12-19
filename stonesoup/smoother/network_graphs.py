"""Road and Rail Network Graph Utilities

This module provides utilities for loading, creating, and converting road and
rail network data into graph formats suitable for the GraphViterbiSmoother.

Supported formats:
- NetworkX graphs
- Dictionary format with nodes/edges
- GeoJSON LineString features
- OSM (OpenStreetMap) data via overpass API or files

"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class NetworkGraphBuilder(ABC):
    """Abstract base class for building network graphs.

    Subclasses implement specific data source parsers that produce
    graphs compatible with GraphViterbiSmoother.

    """

    @abstractmethod
    def build_graph(self) -> dict[str, Any]:
        """Build and return the graph structure.

        Returns
        -------
        dict
            Dictionary with 'nodes' and 'edges' keys compatible with
            GraphViterbiSmoother.

        """
        raise NotImplementedError


class RoadNetworkBuilder(NetworkGraphBuilder):
    """Builder for road network graphs.

    This builder creates road network graphs from various sources.
    Roads are represented as undirected edges by default (vehicles
    can travel in both directions unless specified).

    Parameters
    ----------
    directed : bool, optional
        If True, create a directed graph (one-way roads). Default is False.
    simplify : bool, optional
        If True, simplify the graph by merging nodes with degree 2.
        Default is False.

    Examples
    --------
    >>> builder = RoadNetworkBuilder()
    >>> builder.add_road([(0, 0), (100, 0), (200, 50)])  # A curved road
    >>> builder.add_intersection((100, 0), "intersection_1")
    >>> graph = builder.build_graph()

    """

    def __init__(self, directed: bool = False, simplify: bool = False):
        self.directed = directed
        self.simplify = simplify
        self._nodes: dict[str, tuple[float, float]] = {}
        self._edges: list[tuple[str, str, dict]] = []
        self._node_counter = 0

    def _get_node_id(
        self, position: tuple[float, float], node_id: str | None = None, tolerance: float = 1e-6
    ) -> str:
        """Get or create a node ID for a position.

        If a node already exists at this position (within tolerance),
        return its ID. Otherwise, create a new node.

        """
        if node_id is not None:
            if node_id in self._nodes:
                return node_id
            self._nodes[node_id] = position
            return node_id

        # Check for existing node at this position
        for nid, pos in self._nodes.items():
            if abs(pos[0] - position[0]) < tolerance and abs(pos[1] - position[1]) < tolerance:
                return nid

        # Create new node
        new_id = f"node_{self._node_counter}"
        self._node_counter += 1
        self._nodes[new_id] = position
        return new_id

    def add_node(self, position: tuple[float, float], node_id: str | None = None) -> str:
        """Add a node to the network.

        Parameters
        ----------
        position : tuple of float
            (x, y) coordinates of the node.
        node_id : str, optional
            Custom ID for the node. If None, auto-generated.

        Returns
        -------
        str
            The node ID.

        """
        return self._get_node_id(position, node_id)

    def add_road(
        self,
        coordinates: list[tuple[float, float]],
        road_id: str | None = None,
        attributes: dict | None = None,
    ) -> list[str]:
        """Add a road segment to the network.

        A road is defined as a sequence of coordinates. Intermediate
        points become nodes in the graph, connected by edges.

        Parameters
        ----------
        coordinates : list of tuple
            List of (x, y) coordinate pairs defining the road.
        road_id : str, optional
            Identifier for this road segment.
        attributes : dict, optional
            Additional attributes (e.g., speed_limit, road_type).

        Returns
        -------
        list of str
            Node IDs along this road.

        """
        if len(coordinates) < 2:
            raise ValueError("Road must have at least 2 coordinates")

        attrs = attributes or {}
        if road_id:
            attrs["road_id"] = road_id

        node_ids = []
        for coord in coordinates:
            node_id = self._get_node_id(coord)
            node_ids.append(node_id)

        # Add edges between consecutive nodes
        for i in range(len(node_ids) - 1):
            edge_attrs = attrs.copy()
            # Calculate edge length
            p1 = self._nodes[node_ids[i]]
            p2 = self._nodes[node_ids[i + 1]]
            edge_attrs["length"] = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            self._edges.append((node_ids[i], node_ids[i + 1], edge_attrs))

            if not self.directed:
                # Add reverse edge for undirected graph
                self._edges.append((node_ids[i + 1], node_ids[i], edge_attrs))

        return node_ids

    def add_intersection(
        self, position: tuple[float, float], intersection_id: str | None = None
    ) -> str:
        """Add an intersection point.

        This is a convenience method - intersections are just nodes
        that connect multiple roads.

        Parameters
        ----------
        position : tuple of float
            (x, y) coordinates of the intersection.
        intersection_id : str, optional
            Custom ID for the intersection.

        Returns
        -------
        str
            The node ID.

        """
        return self.add_node(position, intersection_id)

    def connect_roads(self, node_id1: str, node_id2: str, attributes: dict | None = None):
        """Connect two existing nodes with a direct edge.

        Parameters
        ----------
        node_id1 : str
            First node ID.
        node_id2 : str
            Second node ID.
        attributes : dict, optional
            Edge attributes.

        """
        if node_id1 not in self._nodes or node_id2 not in self._nodes:
            raise ValueError("Both nodes must exist")

        attrs = attributes or {}
        p1 = self._nodes[node_id1]
        p2 = self._nodes[node_id2]
        attrs["length"] = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        self._edges.append((node_id1, node_id2, attrs))
        if not self.directed:
            self._edges.append((node_id2, node_id1, attrs))

    def build_graph(self) -> dict[str, Any]:
        """Build the road network graph.

        Returns
        -------
        dict
            Graph dictionary with 'nodes' and 'edges' keys.

        """
        # Convert edges to simple tuples (without duplicates for undirected)
        seen_edges = set()
        edges = []
        for n1, n2, _ in self._edges:
            if self.directed:
                edges.append((n1, n2))
            else:
                edge_key = tuple(sorted([n1, n2]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append((n1, n2))

        return {"nodes": dict(self._nodes), "edges": edges}

    def to_networkx(self):
        """Convert to a NetworkX graph.

        Returns
        -------
        networkx.Graph or networkx.DiGraph
            NetworkX graph with 'pos' attributes on nodes.

        Raises
        ------
        ImportError
            If NetworkX is not installed.

        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for this method")

        G = nx.DiGraph() if self.directed else nx.Graph()

        for node_id, pos in self._nodes.items():
            G.add_node(node_id, pos=pos)

        for n1, n2, attrs in self._edges:
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2, **attrs)

        return G


class RailNetworkBuilder(NetworkGraphBuilder):
    """Builder for rail network graphs.

    Rail networks have specific characteristics:
    - Typically follow fixed tracks
    - May have switches/junctions
    - Often have defined stations/stops

    Parameters
    ----------
    include_stations : bool, optional
        If True, include station nodes as special attributes.
        Default is True.

    Examples
    --------
    >>> builder = RailNetworkBuilder()
    >>> builder.add_track([(0, 0), (1000, 0), (2000, 0)])
    >>> builder.add_station((1000, 0), "Central Station")
    >>> graph = builder.build_graph()

    """

    def __init__(self, include_stations: bool = True):
        self.include_stations = include_stations
        self._nodes: dict[str, tuple[float, float]] = {}
        self._edges: list[tuple[str, str, dict]] = []
        self._stations: dict[str, str] = {}  # station_name -> node_id
        self._node_counter = 0

    def _get_node_id(
        self, position: tuple[float, float], node_id: str | None = None, tolerance: float = 1e-6
    ) -> str:
        """Get or create a node ID for a position."""
        if node_id is not None:
            if node_id in self._nodes:
                return node_id
            self._nodes[node_id] = position
            return node_id

        for nid, pos in self._nodes.items():
            if abs(pos[0] - position[0]) < tolerance and abs(pos[1] - position[1]) < tolerance:
                return nid

        new_id = f"rail_node_{self._node_counter}"
        self._node_counter += 1
        self._nodes[new_id] = position
        return new_id

    def add_track(
        self,
        coordinates: list[tuple[float, float]],
        track_id: str | None = None,
        bidirectional: bool = True,
        attributes: dict | None = None,
    ) -> list[str]:
        """Add a rail track segment.

        Parameters
        ----------
        coordinates : list of tuple
            List of (x, y) coordinate pairs defining the track.
        track_id : str, optional
            Identifier for this track segment.
        bidirectional : bool, optional
            If True, trains can travel in both directions. Default is True.
        attributes : dict, optional
            Additional attributes (e.g., track_type, max_speed).

        Returns
        -------
        list of str
            Node IDs along this track.

        """
        if len(coordinates) < 2:
            raise ValueError("Track must have at least 2 coordinates")

        attrs = attributes or {}
        attrs["type"] = "track"
        if track_id:
            attrs["track_id"] = track_id

        node_ids = []
        for coord in coordinates:
            node_id = self._get_node_id(coord)
            node_ids.append(node_id)

        for i in range(len(node_ids) - 1):
            edge_attrs = attrs.copy()
            p1 = self._nodes[node_ids[i]]
            p2 = self._nodes[node_ids[i + 1]]
            edge_attrs["length"] = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            self._edges.append((node_ids[i], node_ids[i + 1], edge_attrs))
            if bidirectional:
                self._edges.append((node_ids[i + 1], node_ids[i], edge_attrs))

        return node_ids

    def add_station(
        self, position: tuple[float, float], station_name: str, attributes: dict | None = None
    ) -> str:
        """Add a station at a position.

        Parameters
        ----------
        position : tuple of float
            (x, y) coordinates of the station.
        station_name : str
            Name of the station.
        attributes : dict, optional
            Additional station attributes.

        Returns
        -------
        str
            The node ID for this station.

        """
        node_id = self._get_node_id(position)
        self._stations[station_name] = node_id
        return node_id

    def add_junction(self, position: tuple[float, float], junction_id: str | None = None) -> str:
        """Add a rail junction (switch point).

        Parameters
        ----------
        position : tuple of float
            (x, y) coordinates of the junction.
        junction_id : str, optional
            Custom ID for the junction.

        Returns
        -------
        str
            The node ID.

        """
        return self._get_node_id(position, junction_id)

    def build_graph(self) -> dict[str, Any]:
        """Build the rail network graph.

        Returns
        -------
        dict
            Graph dictionary with 'nodes', 'edges', and optionally
            'stations' keys.

        """
        seen_edges = set()
        edges = []
        for n1, n2, _ in self._edges:
            edge_key = tuple(sorted([n1, n2]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append((n1, n2))

        result = {"nodes": dict(self._nodes), "edges": edges}

        if self.include_stations and self._stations:
            result["stations"] = dict(self._stations)

        return result

    def to_networkx(self):
        """Convert to a NetworkX graph."""
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for this method")

        G = nx.Graph()

        for node_id, pos in self._nodes.items():
            is_station = node_id in self._stations.values()
            G.add_node(node_id, pos=pos, is_station=is_station)

        for n1, n2, attrs in self._edges:
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2, **attrs)

        return G


def graph_from_geojson(geojson_data: dict, network_type: str = "road") -> dict[str, Any]:
    """Create a network graph from GeoJSON data.

    Parses GeoJSON LineString or MultiLineString features into a
    graph structure.

    Parameters
    ----------
    geojson_data : dict
        GeoJSON FeatureCollection or Feature.
    network_type : str, optional
        Type of network: 'road' or 'rail'. Default is 'road'.

    Returns
    -------
    dict
        Graph dictionary with 'nodes' and 'edges' keys.

    Examples
    --------
    >>> geojson = {
    ...     "type": "FeatureCollection",
    ...     "features": [
    ...         {
    ...             "type": "Feature",
    ...             "geometry": {
    ...                 "type": "LineString",
    ...                 "coordinates": [[0, 0], [100, 0], [200, 50]]
    ...             },
    ...             "properties": {"name": "Main Street"}
    ...         }
    ...     ]
    ... }
    >>> graph = graph_from_geojson(geojson)

    """
    builder = RoadNetworkBuilder() if network_type == "road" else RailNetworkBuilder()

    # Handle both FeatureCollection and single Feature
    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data.get("features", [])
    elif geojson_data.get("type") == "Feature":
        features = [geojson_data]
    else:
        features = []

    for feature in features:
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])

        if geom_type == "LineString":
            # Convert coordinates to tuples
            coord_tuples = [(c[0], c[1]) for c in coords]
            if network_type == "road":
                builder.add_road(coord_tuples, attributes=properties)
            else:
                builder.add_track(coord_tuples, attributes=properties)

        elif geom_type == "MultiLineString":
            for line_coords in coords:
                coord_tuples = [(c[0], c[1]) for c in line_coords]
                if network_type == "road":
                    builder.add_road(coord_tuples, attributes=properties)
                else:
                    builder.add_track(coord_tuples, attributes=properties)

    return builder.build_graph()


def simplify_graph(graph: dict[str, Any], tolerance: float = 0.0) -> dict[str, Any]:
    """Simplify a graph by removing intermediate nodes.

    Removes nodes with degree 2 that lie on a straight path,
    merging the connected edges.

    Parameters
    ----------
    graph : dict
        Input graph with 'nodes' and 'edges' keys.
    tolerance : float, optional
        Angular tolerance for considering nodes as intermediate.
        Default is 0.0 (only remove perfectly collinear nodes).

    Returns
    -------
    dict
        Simplified graph.

    """
    nodes = dict(graph["nodes"])
    edges = list(graph["edges"])

    # Build adjacency
    adjacency = {n: set() for n in nodes}
    for n1, n2 in edges:
        adjacency[n1].add(n2)
        adjacency[n2].add(n1)

    # Find nodes with degree 2
    to_remove = []
    for node_id, neighbors in adjacency.items():
        if len(neighbors) == 2:
            n1, n2 = list(neighbors)
            # Check if approximately collinear
            p0 = np.array(nodes[n1])
            p1 = np.array(nodes[node_id])
            p2 = np.array(nodes[n2])

            v1 = p1 - p0
            v2 = p2 - p1

            # Normalize
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)

            if len1 > 1e-10 and len2 > 1e-10:
                v1 = v1 / len1
                v2 = v2 / len2
                # Angle between vectors
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot)

                if angle < tolerance or abs(angle - np.pi) < tolerance:
                    to_remove.append((node_id, n1, n2))

    # Remove nodes and reconnect
    for node_id, n1, n2 in to_remove:
        del nodes[node_id]
        # Remove old edges
        edges = [e for e in edges if node_id not in e]
        # Add new direct edge if not already exists
        if (n1, n2) not in edges and (n2, n1) not in edges:
            edges.append((n1, n2))

    return {"nodes": nodes, "edges": edges}


def create_grid_network(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    spacing: float,
    network_type: str = "road",
) -> dict[str, Any]:
    """Create a regular grid network.

    Useful for testing or representing urban grid street patterns.

    Parameters
    ----------
    x_min, x_max : float
        X-axis bounds.
    y_min, y_max : float
        Y-axis bounds.
    spacing : float
        Distance between grid lines.
    network_type : str, optional
        Type of network: 'road' or 'rail'. Default is 'road'.

    Returns
    -------
    dict
        Grid graph with 'nodes' and 'edges' keys.

    Examples
    --------
    >>> # Create a 1km x 1km grid with 100m spacing
    >>> graph = create_grid_network(0, 1000, 0, 1000, 100)

    """
    nodes = {}
    edges = []

    x_coords = np.arange(x_min, x_max + spacing / 2, spacing)
    y_coords = np.arange(y_min, y_max + spacing / 2, spacing)

    # Create nodes
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            node_id = f"grid_{i}_{j}"
            nodes[node_id] = (float(x), float(y))

    # Create edges (horizontal and vertical connections)
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            current = f"grid_{i}_{j}"

            # Connect to right neighbor
            if i < len(x_coords) - 1:
                right = f"grid_{i+1}_{j}"
                edges.append((current, right))

            # Connect to top neighbor
            if j < len(y_coords) - 1:
                top = f"grid_{i}_{j+1}"
                edges.append((current, top))

    return {"nodes": nodes, "edges": edges}


def merge_graphs(*graphs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple graphs into one.

    Nodes at the same position are automatically merged.

    Parameters
    ----------
    *graphs : dict
        Graph dictionaries to merge.

    Returns
    -------
    dict
        Merged graph.

    """
    all_nodes = {}
    all_edges = []
    position_to_id = {}  # (x, y) -> node_id

    node_counter = 0

    for graph in graphs:
        # Map old node IDs to new ones
        id_mapping = {}

        for old_id, pos in graph["nodes"].items():
            pos_key = (round(pos[0], 6), round(pos[1], 6))

            if pos_key in position_to_id:
                # Reuse existing node
                id_mapping[old_id] = position_to_id[pos_key]
            else:
                # Create new node
                new_id = f"merged_{node_counter}"
                node_counter += 1
                all_nodes[new_id] = pos
                position_to_id[pos_key] = new_id
                id_mapping[old_id] = new_id

        # Add edges with new IDs
        for n1, n2 in graph["edges"]:
            new_n1 = id_mapping[n1]
            new_n2 = id_mapping[n2]
            edge = (new_n1, new_n2)
            reverse_edge = (new_n2, new_n1)
            if edge not in all_edges and reverse_edge not in all_edges:
                all_edges.append(edge)

    return {"nodes": all_nodes, "edges": all_edges}
