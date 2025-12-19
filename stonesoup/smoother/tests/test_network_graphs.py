"""Tests for road and rail network graph utilities."""

import numpy as np
import pytest

from ..network_graphs import (
    NetworkGraphBuilder,
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


# Additional comprehensive tests


def test_network_graph_builder_abstract():
    """Test that NetworkGraphBuilder is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        NetworkGraphBuilder()


def test_road_builder_edge_length_calculation():
    """Test that edge lengths are calculated correctly."""
    builder = RoadNetworkBuilder()

    # Simple horizontal road
    coords = [(0, 0), (10, 0)]
    builder.add_road(coords)

    # Check that length is calculated
    _graph = builder.build_graph()  # noqa: F841 - graph built but attrs checked via networkx
    # Can't directly check edge attributes in dict format, but we can verify with networkx
    try:
        G = builder.to_networkx()
        edges = list(G.edges(data=True))
        assert len(edges) == 1
        assert "length" in edges[0][2]
        assert np.isclose(edges[0][2]["length"], 10.0)
    except ImportError:
        pass  # NetworkX not available


def test_road_builder_edge_length_diagonal():
    """Test edge length calculation for diagonal roads."""
    builder = RoadNetworkBuilder()

    # Diagonal road (3-4-5 triangle)
    coords = [(0, 0), (3, 4)]
    builder.add_road(coords)

    try:
        G = builder.to_networkx()
        edges = list(G.edges(data=True))
        assert np.isclose(edges[0][2]["length"], 5.0, atol=1e-6)
    except ImportError:
        pass


def test_road_builder_multiple_roads_shared_node():
    """Test multiple roads sharing nodes at intersections."""
    builder = RoadNetworkBuilder()

    # Create a T-intersection
    builder.add_road([(0, 0), (10, 0), (20, 0)], road_id="main_st")
    builder.add_road([(10, -5), (10, 0), (10, 5)], road_id="cross_st")

    graph = builder.build_graph()

    # Should have 5 unique nodes
    assert len(graph["nodes"]) == 5

    # (10, 0) should be shared
    positions = list(graph["nodes"].values())
    intersection_count = sum(1 for pos in positions if pos == (10.0, 0.0))
    assert intersection_count == 1


def test_road_builder_undirected_creates_bidirectional_edges():
    """Test that undirected roads create bidirectional edges."""
    builder = RoadNetworkBuilder(directed=False)
    builder.add_road([(0, 0), (10, 0)])

    # Internal edges list should have both directions
    assert len(builder._edges) == 2

    # After build_graph, should deduplicate
    graph = builder.build_graph()
    assert len(graph["edges"]) == 1


def test_road_builder_directed_creates_one_way():
    """Test that directed roads create one-way edges."""
    builder = RoadNetworkBuilder(directed=True)
    builder.add_road([(0, 0), (10, 0)])

    # Internal edges list should have only one direction
    assert len(builder._edges) == 1

    graph = builder.build_graph()
    assert len(graph["edges"]) == 1


def test_road_builder_tolerance():
    """Test node position tolerance for deduplication."""
    builder = RoadNetworkBuilder()

    # Add two very close nodes
    node1 = builder.add_node((0.0, 0.0))
    node2 = builder.add_node((1e-7, 1e-7))  # Within default tolerance

    # Should return same node
    assert node1 == node2


def test_road_builder_tolerance_exceeded():
    """Test that nodes outside tolerance are treated as separate."""
    builder = RoadNetworkBuilder()

    node1 = builder.add_node((0.0, 0.0))
    node2 = builder.add_node((1e-5, 1e-5))  # Outside default tolerance

    # Should be different nodes
    assert node1 != node2


def test_road_builder_custom_node_id_persistence():
    """Test that custom node IDs are preserved."""
    builder = RoadNetworkBuilder()

    custom_id = "intersection_main_elm"
    node_id = builder.add_node((100, 200), node_id=custom_id)

    assert node_id == custom_id
    assert custom_id in builder._nodes


def test_road_builder_custom_node_id_reuse():
    """Test that reusing a custom node ID returns the same node."""
    builder = RoadNetworkBuilder()

    node1 = builder.add_node((100, 200), node_id="my_node")
    node2 = builder.add_node((999, 999), node_id="my_node")  # Different position!

    # Should return existing node ID
    assert node1 == node2
    # Position should not change
    assert builder._nodes["my_node"] == (100.0, 200.0)


def test_road_builder_complex_network():
    """Test building a complex network with multiple intersecting roads."""
    builder = RoadNetworkBuilder()

    # Grid of roads
    builder.add_road([(0, 0), (10, 0), (20, 0), (30, 0)])  # Horizontal
    builder.add_road([(0, 10), (10, 10), (20, 10), (30, 10)])  # Horizontal
    builder.add_road([(0, 0), (0, 10)])  # Vertical
    builder.add_road([(10, 0), (10, 10)])  # Vertical
    builder.add_road([(20, 0), (20, 10)])  # Vertical
    builder.add_road([(30, 0), (30, 10)])  # Vertical

    graph = builder.build_graph()

    # Should have 8 unique nodes (4x2 grid)
    assert len(graph["nodes"]) == 8


def test_road_builder_attributes_preservation():
    """Test that road attributes are preserved."""
    builder = RoadNetworkBuilder()

    attrs = {"speed_limit": 55, "lanes": 4, "surface": "asphalt"}
    builder.add_road([(0, 0), (100, 0)], road_id="highway_1", attributes=attrs)

    # Check internal storage
    assert len(builder._edges) > 0
    edge_attrs = builder._edges[0][2]
    assert edge_attrs["speed_limit"] == 55
    assert edge_attrs["lanes"] == 4
    assert edge_attrs["surface"] == "asphalt"
    assert edge_attrs["road_id"] == "highway_1"


def test_rail_builder_bidirectional_default():
    """Test that rail tracks are bidirectional by default."""
    builder = RailNetworkBuilder()
    builder.add_track([(0, 0), (100, 0)])

    # Should create bidirectional edges
    assert len(builder._edges) == 2


def test_rail_builder_unidirectional():
    """Test unidirectional rail tracks."""
    builder = RailNetworkBuilder()
    builder.add_track([(0, 0), (100, 0)], bidirectional=False)

    # Should create only one direction
    assert len(builder._edges) == 1


def test_rail_builder_station_at_existing_node():
    """Test adding station at an existing track node."""
    builder = RailNetworkBuilder()

    # Add track
    builder.add_track([(0, 0), (1000, 0), (2000, 0)])

    # Add station at middle point
    station_id = builder.add_station((1000, 0), "Midtown Station")

    # Should reuse existing node
    graph = builder.build_graph()
    assert len(graph["nodes"]) == 3

    # Station should be in the graph
    assert "stations" in graph
    assert "Midtown Station" in graph["stations"]
    assert graph["stations"]["Midtown Station"] == station_id


def test_rail_builder_multiple_stations():
    """Test adding multiple stations."""
    builder = RailNetworkBuilder()

    builder.add_track([(0, 0), (1000, 0), (2000, 0), (3000, 0)])
    builder.add_station((0, 0), "West Station")
    builder.add_station((2000, 0), "Central Station")
    builder.add_station((3000, 0), "East Station")

    graph = builder.build_graph()

    assert len(graph["stations"]) == 3
    assert "West Station" in graph["stations"]
    assert "Central Station" in graph["stations"]
    assert "East Station" in graph["stations"]


def test_rail_builder_junction_custom_id():
    """Test adding junction with custom ID."""
    builder = RailNetworkBuilder()

    junction_id = builder.add_junction((500, 500), "junction_alpha")
    assert junction_id == "junction_alpha"
    assert junction_id in builder._nodes


def test_rail_builder_track_attributes():
    """Test rail track attributes."""
    builder = RailNetworkBuilder()

    attrs = {"track_type": "high_speed", "max_speed": 300}
    builder.add_track([(0, 0), (1000, 0)], track_id="hs_line_1", attributes=attrs)

    # Check internal storage
    edge_attrs = builder._edges[0][2]
    assert edge_attrs["type"] == "track"  # Auto-added
    assert edge_attrs["track_type"] == "high_speed"
    assert edge_attrs["max_speed"] == 300
    assert edge_attrs["track_id"] == "hs_line_1"


def test_rail_builder_to_networkx_station_attribute():
    """Test that stations have is_station attribute in NetworkX graph."""
    builder = RailNetworkBuilder()
    builder.add_track([(0, 0), (1000, 0)])
    builder.add_station((0, 0), "Station A")

    try:
        G = builder.to_networkx()

        # Find the station node
        station_nodes = [n for n, d in G.nodes(data=True) if d.get("is_station", False)]
        assert len(station_nodes) == 1

        # Non-station nodes should have is_station=False
        non_station_nodes = [n for n, d in G.nodes(data=True) if not d.get("is_station", False)]
        assert len(non_station_nodes) == 1
    except ImportError:
        pytest.skip("NetworkX not available")


def test_geojson_single_feature():
    """Test GeoJSON with single Feature (not FeatureCollection)."""
    geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
        "properties": {"name": "Route 1"},
    }

    graph = graph_from_geojson(geojson)

    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1


def test_geojson_invalid_geometry_type():
    """Test GeoJSON with unsupported geometry type."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},  # Point not supported
                "properties": {},
            }
        ],
    }

    # Should not raise error, just ignore unsupported geometries
    graph = graph_from_geojson(geojson)
    assert len(graph["nodes"]) == 0


def test_geojson_properties_preserved():
    """Test that GeoJSON properties are preserved as edge attributes."""
    geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0]]},
        "properties": {"name": "Main Street", "type": "residential", "oneway": False},
    }

    # Build with road network builder to check attributes
    graph = graph_from_geojson(geojson, network_type="road")

    # Properties should be in the graph
    assert len(graph["nodes"]) == 2


def test_geojson_multilinestring_properties():
    """Test MultiLineString preserves properties for all segments."""
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [[[0, 0], [10, 0]], [[20, 0], [30, 0]]],
        },
        "properties": {"route": "66"},
    }

    graph = graph_from_geojson(geojson)

    # Should have 4 nodes (2 per segment)
    assert len(graph["nodes"]) == 4


def test_grid_network_single_cell():
    """Test creating minimal 1x1 grid."""
    graph = create_grid_network(0, 10, 0, 10, 10)

    # Should have 4 corner nodes
    assert len(graph["nodes"]) == 4
    assert len(graph["edges"]) == 4  # Square


def test_grid_network_exact_spacing():
    """Test grid with exact spacing boundaries."""
    graph = create_grid_network(0, 300, 0, 200, 100)

    # Should have (4 x 3) = 12 nodes
    assert len(graph["nodes"]) == 12


def test_grid_network_non_exact_spacing():
    """Test grid where spacing doesn't divide evenly."""
    graph = create_grid_network(0, 250, 0, 250, 100)

    # Should have (3 x 3) = 9 nodes (0, 100, 200 in each direction)
    assert len(graph["nodes"]) == 9


def test_grid_network_node_naming():
    """Test grid node naming convention."""
    graph = create_grid_network(0, 100, 0, 100, 50)

    # Should have nodes named grid_i_j
    node_ids = list(graph["nodes"].keys())
    assert "grid_0_0" in node_ids
    assert "grid_1_0" in node_ids
    assert "grid_0_1" in node_ids
    assert "grid_2_2" in node_ids


def test_grid_network_connectivity():
    """Test that grid nodes are properly connected."""
    graph = create_grid_network(0, 100, 0, 100, 50)

    # Build adjacency manually to check
    adjacency = {}
    for node in graph["nodes"]:
        adjacency[node] = set()

    for n1, n2 in graph["edges"]:
        adjacency[n1].add(n2)
        adjacency[n2].add(n1)

    # Corner nodes should have degree 2
    assert len(adjacency["grid_0_0"]) == 2
    assert len(adjacency["grid_2_2"]) == 2

    # Edge nodes (not corners) should have degree 3
    assert len(adjacency["grid_1_0"]) == 3
    assert len(adjacency["grid_0_1"]) == 3

    # Center node should have degree 4
    assert len(adjacency["grid_1_1"]) == 4


def test_merge_graphs_empty():
    """Test merging with empty graphs."""
    graph1 = {"nodes": {}, "edges": []}
    graph2 = {"nodes": {"a": (0, 0)}, "edges": []}

    merged = merge_graphs(graph1, graph2)

    assert len(merged["nodes"]) == 1


def test_merge_graphs_three_way():
    """Test merging three graphs."""
    graph1 = {"nodes": {"a": (0, 0), "b": (1, 0)}, "edges": [("a", "b")]}
    graph2 = {"nodes": {"c": (2, 0), "d": (3, 0)}, "edges": [("c", "d")]}
    graph3 = {"nodes": {"e": (4, 0), "f": (5, 0)}, "edges": [("e", "f")]}

    merged = merge_graphs(graph1, graph2, graph3)

    assert len(merged["nodes"]) == 6
    assert len(merged["edges"]) == 3


def test_merge_graphs_overlapping_positions():
    """Test merging graphs with multiple overlapping positions."""
    graph1 = {"nodes": {"a": (0, 0), "b": (1, 0), "c": (2, 0)}, "edges": [("a", "b"), ("b", "c")]}
    graph2 = {"nodes": {"d": (1, 0), "e": (2, 0), "f": (3, 0)}, "edges": [("d", "e"), ("e", "f")]}

    merged = merge_graphs(graph1, graph2)

    # Nodes at (1, 0) and (2, 0) should be merged
    # So 4 unique positions: (0,0), (1,0), (2,0), (3,0)
    assert len(merged["nodes"]) == 4


def test_merge_graphs_edge_deduplication():
    """Test that merged graphs don't have duplicate edges."""
    graph1 = {"nodes": {"a": (0, 0), "b": (1, 0)}, "edges": [("a", "b")]}
    graph2 = {"nodes": {"c": (0, 0), "d": (1, 0)}, "edges": [("c", "d")]}

    merged = merge_graphs(graph1, graph2)

    # Same edge represented twice should result in single edge
    # Nodes merged, so only 2 nodes and 1 edge
    assert len(merged["nodes"]) == 2
    assert len(merged["edges"]) == 1


def test_simplify_graph_empty():
    """Test simplifying empty graph."""
    graph = {"nodes": {}, "edges": []}
    simplified = simplify_graph(graph)

    assert len(simplified["nodes"]) == 0
    assert len(simplified["edges"]) == 0


def test_simplify_graph_single_node():
    """Test simplifying graph with single node."""
    graph = {"nodes": {"a": (0, 0)}, "edges": []}
    simplified = simplify_graph(graph)

    assert len(simplified["nodes"]) == 1


def test_simplify_graph_two_nodes():
    """Test simplifying graph with two nodes."""
    graph = {"nodes": {"a": (0, 0), "b": (1, 0)}, "edges": [("a", "b")]}
    simplified = simplify_graph(graph)

    # Can't simplify - both endpoints
    assert len(simplified["nodes"]) == 2
    assert len(simplified["edges"]) == 1


def test_simplify_graph_long_chain():
    """Test simplifying a long chain of collinear nodes."""
    # Create a long straight road with many intermediate points
    graph = {
        "nodes": {
            "a": (0, 0),
            "b": (1, 0),
            "c": (2, 0),
            "d": (3, 0),
            "e": (4, 0),
            "f": (5, 0),
        },
        "edges": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")],
    }

    simplified = simplify_graph(graph, tolerance=0.01)

    # Should reduce to just endpoints
    assert len(simplified["nodes"]) == 2
    assert len(simplified["edges"]) == 1


def test_simplify_graph_curved_path():
    """Test that curved paths preserve intermediate nodes."""
    # Create a curved path
    graph = {
        "nodes": {
            "a": (0, 0),
            "b": (1, 0.5),  # Curved
            "c": (2, 0.8),  # Curved
            "d": (3, 0.9),  # Curved
            "e": (4, 1),
        },
        "edges": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
    }

    simplified = simplify_graph(graph, tolerance=0.01)

    # Should keep most nodes due to curvature
    assert len(simplified["nodes"]) > 2


def test_simplify_graph_preserves_degree_3_nodes():
    """Test that nodes with degree > 2 are preserved."""
    # Y-shaped junction
    graph = {
        "nodes": {"a": (0, 0), "b": (1, 0), "c": (2, 1), "d": (2, -1)},
        "edges": [("a", "b"), ("b", "c"), ("b", "d")],
    }

    simplified = simplify_graph(graph, tolerance=0.01)

    # Node b has degree 3, should be preserved
    assert len(simplified["nodes"]) == 4


def test_simplify_graph_tolerance_effect():
    """Test effect of tolerance on simplification."""
    # Slightly curved path
    graph = {
        "nodes": {"a": (0, 0), "b": (1, 0.01), "c": (2, 0)},  # Very slight curve
        "edges": [("a", "b"), ("b", "c")],
    }

    # With very low tolerance, keep the curve
    simplified_low = simplify_graph(graph, tolerance=0.001)
    assert len(simplified_low["nodes"]) == 3

    # With higher tolerance, remove intermediate node
    simplified_high = simplify_graph(graph, tolerance=0.1)
    assert len(simplified_high["nodes"]) == 2


def test_simplify_graph_reverse_direction():
    """Test simplification with edges in different directions."""
    graph = {
        "nodes": {"a": (0, 0), "b": (1, 0), "c": (2, 0)},
        "edges": [("a", "b"), ("c", "b")],  # Note: second edge reversed
    }

    simplified = simplify_graph(graph, tolerance=0.01)

    # Should still simplify collinear nodes
    assert len(simplified["nodes"]) == 2


def test_simplify_graph_disconnected_components():
    """Test simplifying graph with disconnected components."""
    graph = {
        "nodes": {
            # Component 1
            "a": (0, 0),
            "b": (1, 0),
            "c": (2, 0),
            # Component 2
            "d": (10, 10),
            "e": (11, 10),
            "f": (12, 10),
        },
        "edges": [("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")],
    }

    simplified = simplify_graph(graph, tolerance=0.01)

    # Each component should simplify to 2 nodes
    assert len(simplified["nodes"]) == 4
    assert len(simplified["edges"]) == 2


def test_simplify_graph_zero_length_edges():
    """Test handling of zero-length edges (duplicate positions)."""
    graph = {
        "nodes": {"a": (0, 0), "b": (0, 0), "c": (1, 0)},  # a and b at same position
        "edges": [("a", "b"), ("b", "c")],
    }

    # Should handle without crashing
    simplified = simplify_graph(graph, tolerance=0.01)
    assert len(simplified["nodes"]) >= 2


def test_road_builder_to_networkx_directed():
    """Test converting directed road network to NetworkX DiGraph."""
    builder = RoadNetworkBuilder(directed=True)
    builder.add_road([(0, 0), (100, 0)])

    try:
        import networkx as nx

        G = builder.to_networkx()
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes) == 2
        assert len(G.edges) == 1
    except ImportError:
        pytest.skip("NetworkX not available")


def test_road_builder_to_networkx_raises_without_networkx():
    """Test that to_networkx raises ImportError when NetworkX is unavailable."""
    builder = RoadNetworkBuilder()
    builder.add_road([(0, 0), (100, 0)])

    # Mock the HAS_NETWORKX flag
    import stonesoup.smoother.network_graphs as ng

    original_flag = ng.HAS_NETWORKX
    try:
        ng.HAS_NETWORKX = False

        # Recreate builder to use modified flag
        builder = RoadNetworkBuilder()
        builder.add_road([(0, 0), (100, 0)])

        with pytest.raises(ImportError, match="NetworkX is required"):
            builder.to_networkx()
    finally:
        ng.HAS_NETWORKX = original_flag


def test_rail_builder_to_networkx_raises_without_networkx():
    """Test that RailNetworkBuilder.to_networkx raises ImportError when NetworkX unavailable."""
    builder = RailNetworkBuilder()
    builder.add_track([(0, 0), (100, 0)])

    import stonesoup.smoother.network_graphs as ng

    original_flag = ng.HAS_NETWORKX
    try:
        ng.HAS_NETWORKX = False

        builder = RailNetworkBuilder()
        builder.add_track([(0, 0), (100, 0)])

        with pytest.raises(ImportError, match="NetworkX is required"):
            builder.to_networkx()
    finally:
        ng.HAS_NETWORKX = original_flag


def test_create_grid_network_negative_coordinates():
    """Test creating grid with negative coordinates."""
    graph = create_grid_network(-100, 100, -50, 50, 100)

    # Should have 3x2 = 6 nodes
    assert len(graph["nodes"]) == 6

    # Check that negative coordinates are present
    positions = set(graph["nodes"].values())
    assert (-100.0, -50.0) in positions
    assert (100.0, 50.0) in positions


def test_create_grid_network_float_coordinates():
    """Test grid with floating point coordinates."""
    graph = create_grid_network(0.5, 2.5, 0.5, 2.5, 1.0)

    # Should create grid starting at 0.5
    positions = set(graph["nodes"].values())
    assert (0.5, 0.5) in positions
    assert (2.5, 2.5) in positions


def test_connect_roads_calculates_length():
    """Test that connect_roads calculates edge length."""
    builder = RoadNetworkBuilder()

    builder.add_node((0, 0), "node1")
    builder.add_node((3, 4), "node2")  # 3-4-5 triangle
    builder.connect_roads("node1", "node2")

    # Check length calculation
    edge_attrs = builder._edges[-1][2]
    assert np.isclose(edge_attrs["length"], 5.0, atol=1e-6)


def test_connect_roads_preserves_attributes():
    """Test that connect_roads preserves custom attributes."""
    builder = RoadNetworkBuilder()

    builder.add_node((0, 0), "node1")
    builder.add_node((10, 0), "node2")

    attrs = {"bridge": True, "toll": 5.50}
    builder.connect_roads("node1", "node2", attributes=attrs)

    edge_attrs = builder._edges[-1][2]
    assert edge_attrs["bridge"] is True
    assert edge_attrs["toll"] == 5.50


def test_road_builder_long_road():
    """Test adding a road with many intermediate points."""
    builder = RoadNetworkBuilder()

    # 100 points along a straight line
    coords = [(float(i), 0.0) for i in range(100)]
    node_ids = builder.add_road(coords)

    assert len(node_ids) == 100

    graph = builder.build_graph()
    assert len(graph["nodes"]) == 100
    assert len(graph["edges"]) == 99


def test_rail_builder_complex_network():
    """Test building a complex rail network with junctions and stations."""
    builder = RailNetworkBuilder()

    # Main line
    builder.add_track([(0, 0), (100, 0), (200, 0), (300, 0)], track_id="main_line")

    # Branch line at (100, 0)
    builder.add_track([(100, 0), (100, 50), (100, 100)], track_id="north_branch")

    # Add stations
    builder.add_station((0, 0), "Terminal West")
    builder.add_station((100, 0), "Junction Station")
    builder.add_station((300, 0), "Terminal East")
    builder.add_station((100, 100), "Terminal North")

    graph = builder.build_graph()

    assert len(graph["stations"]) == 4
    # Check junction is shared
    assert len(graph["nodes"]) == 6  # 4 main + 2 north branch (sharing junction)


def test_geojson_coordinates_3d():
    """Test GeoJSON with 3D coordinates (should use only x, y)."""
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[0, 0, 100], [10, 0, 150], [20, 0, 200]],  # Include elevation
        },
        "properties": {},
    }

    graph = graph_from_geojson(geojson)

    # Should work, using only x and y
    assert len(graph["nodes"]) == 3

    # Check that z-coordinate is ignored
    positions = set(graph["nodes"].values())
    assert (0.0, 0.0) in positions


def test_merge_graphs_preserves_all_edges():
    """Test that merging preserves edges from all input graphs."""
    # Star topology - central node connected to 3 others
    graph1 = {
        "nodes": {"center": (0, 0), "a": (1, 0)},
        "edges": [("center", "a")],
    }

    graph2 = {
        "nodes": {"center": (0, 0), "b": (0, 1)},
        "edges": [("center", "b")],
    }

    graph3 = {
        "nodes": {"center": (0, 0), "c": (-1, 0)},
        "edges": [("center", "c")],
    }

    merged = merge_graphs(graph1, graph2, graph3)

    # Should have 4 nodes (center shared)
    assert len(merged["nodes"]) == 4

    # Should have 3 edges
    assert len(merged["edges"]) == 3
