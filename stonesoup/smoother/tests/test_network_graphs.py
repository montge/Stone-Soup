"""Tests for road and rail network graph utilities."""

import pytest

from ..network_graphs import (
    RailNetworkBuilder,
    RoadNetworkBuilder,
    create_grid_network,
    graph_from_geojson,
    merge_graphs,
    simplify_graph,
)


def test_road_builder_creation():
    """Test RoadNetworkBuilder instantiation."""
    builder = RoadNetworkBuilder()
    assert builder.directed is False
    assert builder.simplify is False

    builder_directed = RoadNetworkBuilder(directed=True)
    assert builder_directed.directed is True


def test_road_builder_add_node():
    """Test adding nodes to road network."""
    builder = RoadNetworkBuilder()

    node_id = builder.add_node((100.0, 200.0))
    assert node_id is not None

    # Adding same position should return same node
    node_id2 = builder.add_node((100.0, 200.0))
    assert node_id == node_id2


def test_road_builder_add_node_custom_id():
    """Test adding nodes with custom IDs."""
    builder = RoadNetworkBuilder()

    node_id = builder.add_node((100.0, 200.0), node_id="my_node")
    assert node_id == "my_node"


def test_road_builder_add_road():
    """Test adding road segments."""
    builder = RoadNetworkBuilder()

    coords = [(0, 0), (100, 0), (200, 50)]
    node_ids = builder.add_road(coords)

    assert len(node_ids) == 3

    graph = builder.build_graph()
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2  # 2 edges connecting 3 nodes


def test_road_builder_add_road_minimum_coords():
    """Test road requires at least 2 coordinates."""
    builder = RoadNetworkBuilder()

    with pytest.raises(ValueError):
        builder.add_road([(0, 0)])


def test_road_builder_intersection():
    """Test adding intersections."""
    builder = RoadNetworkBuilder()

    # Add two roads that cross
    builder.add_road([(0, 0), (100, 0), (200, 0)])
    builder.add_road([(100, -50), (100, 0), (100, 50)])

    graph = builder.build_graph()

    # (100, 0) should be shared - only 5 unique nodes
    assert len(graph["nodes"]) == 5


def test_road_builder_explicit_intersection():
    """Test adding explicit intersection."""
    builder = RoadNetworkBuilder()

    int_id = builder.add_intersection((100, 0), "main_intersection")
    assert int_id == "main_intersection"


def test_road_builder_connect_roads():
    """Test connecting existing roads."""
    builder = RoadNetworkBuilder()

    builder.add_road([(0, 0), (100, 0)], road_id="road1")
    builder.add_road([(200, 0), (300, 0)], road_id="road2")

    # Get node IDs
    graph = builder.build_graph()
    list(graph["nodes"].keys())

    # Connect them
    builder = RoadNetworkBuilder()
    builder.add_node((100, 0), "end1")
    builder.add_node((200, 0), "start2")
    builder.connect_roads("end1", "start2")

    graph = builder.build_graph()
    assert len(graph["edges"]) == 1


def test_road_builder_connect_nonexistent():
    """Test connecting non-existent nodes raises error."""
    builder = RoadNetworkBuilder()
    builder.add_node((0, 0), "node1")

    with pytest.raises(ValueError):
        builder.connect_roads("node1", "nonexistent")


def test_road_builder_directed():
    """Test directed road network."""
    builder = RoadNetworkBuilder(directed=True)
    builder.add_road([(0, 0), (100, 0)])

    graph = builder.build_graph()

    # Directed graph should have edges in both directions added separately
    # But build_graph deduplicates for the simple dict format
    assert len(graph["edges"]) == 1


def test_rail_builder_creation():
    """Test RailNetworkBuilder instantiation."""
    builder = RailNetworkBuilder()
    assert builder.include_stations is True


def test_rail_builder_add_track():
    """Test adding rail tracks."""
    builder = RailNetworkBuilder()

    coords = [(0, 0), (1000, 0), (2000, 0)]
    node_ids = builder.add_track(coords)

    assert len(node_ids) == 3

    graph = builder.build_graph()
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2


def test_rail_builder_add_track_minimum():
    """Test track requires at least 2 coordinates."""
    builder = RailNetworkBuilder()

    with pytest.raises(ValueError):
        builder.add_track([(0, 0)])


def test_rail_builder_add_station():
    """Test adding stations."""
    builder = RailNetworkBuilder()

    builder.add_track([(0, 0), (1000, 0), (2000, 0)])
    builder.add_station((1000, 0), "Central Station")

    graph = builder.build_graph()

    assert "stations" in graph
    assert "Central Station" in graph["stations"]


def test_rail_builder_add_junction():
    """Test adding junctions."""
    builder = RailNetworkBuilder()

    junction_id = builder.add_junction((100, 100), "junction_1")
    assert junction_id == "junction_1"


def test_rail_builder_no_stations():
    """Test building without stations."""
    builder = RailNetworkBuilder(include_stations=False)
    builder.add_track([(0, 0), (1000, 0)])
    builder.add_station((500, 0), "Hidden Station")

    graph = builder.build_graph()
    assert "stations" not in graph


def test_graph_from_geojson_linestring():
    """Test creating graph from GeoJSON LineString."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0], [200, 50]]},
                "properties": {"name": "Main Street"},
            }
        ],
    }

    graph = graph_from_geojson(geojson)

    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2


def test_graph_from_geojson_multilinestring():
    """Test creating graph from GeoJSON MultiLineString."""
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[0, 0], [100, 0]], [[200, 0], [300, 0]]],
        },
        "properties": {},
    }

    graph = graph_from_geojson(geojson)

    assert len(graph["nodes"]) == 4
    assert len(graph["edges"]) == 2


def test_graph_from_geojson_rail():
    """Test creating rail graph from GeoJSON."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1000, 0]]},
                "properties": {"track_type": "main"},
            }
        ],
    }

    graph = graph_from_geojson(geojson, network_type="rail")

    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1


def test_create_grid_network():
    """Test creating grid network."""
    graph = create_grid_network(0, 200, 0, 200, 100)

    # 3x3 grid = 9 nodes
    assert len(graph["nodes"]) == 9

    # Horizontal: 2 per row * 3 rows = 6
    # Vertical: 3 per column * 2 columns = 6
    # But edges are undirected, so counted once
    assert len(graph["edges"]) == 12


def test_create_grid_network_small():
    """Test creating 2x2 grid."""
    graph = create_grid_network(0, 100, 0, 100, 100)

    # 2x2 grid = 4 nodes
    assert len(graph["nodes"]) == 4
    # 4 edges (forming a square)
    assert len(graph["edges"]) == 4


def test_merge_graphs():
    """Test merging multiple graphs."""
    graph1 = {"nodes": {"a": (0, 0), "b": (100, 0)}, "edges": [("a", "b")]}
    graph2 = {"nodes": {"c": (200, 0), "d": (300, 0)}, "edges": [("c", "d")]}

    merged = merge_graphs(graph1, graph2)

    assert len(merged["nodes"]) == 4
    assert len(merged["edges"]) == 2


def test_merge_graphs_shared_position():
    """Test merging graphs with shared node positions."""
    graph1 = {"nodes": {"a": (0, 0), "b": (100, 0)}, "edges": [("a", "b")]}
    graph2 = {
        "nodes": {"c": (100, 0), "d": (200, 0)},  # c is at same position as b
        "edges": [("c", "d")],
    }

    merged = merge_graphs(graph1, graph2)

    # Should have 3 nodes (shared position merged)
    assert len(merged["nodes"]) == 3
    assert len(merged["edges"]) == 2


def test_simplify_graph_collinear():
    """Test simplifying graph with collinear nodes."""
    # Three collinear points
    graph = {
        "nodes": {"a": (0, 0), "b": (100, 0), "c": (200, 0)},  # Intermediate, should be removed
        "edges": [("a", "b"), ("b", "c")],
    }

    # With very small tolerance, collinear points should be kept
    simplified = simplify_graph(graph, tolerance=0.01)

    # The collinear point should be removed
    assert len(simplified["nodes"]) == 2
    assert len(simplified["edges"]) == 1


def test_simplify_graph_not_collinear():
    """Test simplifying graph keeps non-collinear nodes."""
    graph = {
        "nodes": {"a": (0, 0), "b": (100, 50), "c": (200, 0)},  # Not collinear
        "edges": [("a", "b"), ("b", "c")],
    }

    simplified = simplify_graph(graph, tolerance=0.01)

    # Non-collinear point should be kept
    assert len(simplified["nodes"]) == 3
    assert len(simplified["edges"]) == 2


def test_simplify_graph_junction():
    """Test simplifying preserves junctions."""
    graph = {
        "nodes": {
            "a": (0, 0),
            "b": (100, 0),  # Junction (degree 3)
            "c": (200, 0),
            "d": (100, 100),
        },
        "edges": [("a", "b"), ("b", "c"), ("b", "d")],
    }

    simplified = simplify_graph(graph, tolerance=0.01)

    # Junction (degree 3) should be preserved
    assert len(simplified["nodes"]) == 4


def test_road_builder_attributes():
    """Test road attributes are stored."""
    builder = RoadNetworkBuilder()

    attrs = {"speed_limit": 50, "lanes": 2}
    builder.add_road([(0, 0), (100, 0)], road_id="test_road", attributes=attrs)

    # Build with networkx if available to check attributes
    try:
        G = builder.to_networkx()
        # Check edge has length attribute
        edges = list(G.edges(data=True))
        assert len(edges) == 1
        assert "length" in edges[0][2]
    except ImportError:
        pass  # NetworkX not available


def test_rail_builder_to_networkx():
    """Test converting rail network to NetworkX."""
    builder = RailNetworkBuilder()
    builder.add_track([(0, 0), (1000, 0)])

    try:
        G = builder.to_networkx()
        assert len(G.nodes) == 2
        assert len(G.edges) == 1
    except ImportError:
        pytest.skip("NetworkX not available")


def test_road_builder_to_networkx():
    """Test converting road network to NetworkX."""
    builder = RoadNetworkBuilder()
    builder.add_road([(0, 0), (100, 0), (200, 0)])

    try:
        G = builder.to_networkx()
        assert len(G.nodes) == 3
        assert len(G.edges) == 2

        # Check nodes have pos attribute
        for node in G.nodes:
            assert "pos" in G.nodes[node]
    except ImportError:
        pytest.skip("NetworkX not available")


def test_grid_network_positions():
    """Test grid network node positions are correct."""
    graph = create_grid_network(0, 100, 0, 100, 50)

    # Check corner positions exist
    positions = set(graph["nodes"].values())

    assert (0.0, 0.0) in positions
    assert (100.0, 0.0) in positions
    assert (0.0, 100.0) in positions
    assert (100.0, 100.0) in positions
    assert (50.0, 50.0) in positions  # Center


def test_geojson_empty():
    """Test handling empty GeoJSON."""
    geojson = {"type": "FeatureCollection", "features": []}

    graph = graph_from_geojson(geojson)
    assert len(graph["nodes"]) == 0
    assert len(graph["edges"]) == 0
